//! Qualify fused block publication against the existing three-stage producer.
use super::*;
use kernels_sm120::pooled_indexer::{
    position_blocks::CandidateEntry, position_candidates_cached as cached,
};

pub(super) const CUBIN: &str = "position_candidates_cached_sm120";
const COUNTS: [u32; 3] = [1, 31, 2048];
const WIDTHS: [u32; 6] = [1, 8, 512, 16384, 16385, 32768];

fn entry() -> CandidateEntry {
    CandidateEntry {
        name: FLASH.kernel_tag.into(),
        query: indexer::candidates(&FLASH, 8192),
    }
}
fn sources() -> Vec<Case> {
    cases().into_iter().filter(|c| c.layer == 20).collect()
}
fn memcheck_source(c: Case) -> bool {
    c.capture.is_some() || matches!(c.kind, 3 | 6 | 7 | 8 | 9 | 10 | 12)
}
fn timed_source(c: Case) -> bool {
    c.capture.is_some() || c.kind == 0 && matches!(c.rows, 2496 | 8192)
}

#[allow(clippy::too_many_arguments)]
fn pipeline(
    m: &mut Memory<'_>,
    c: Case,
    input: &Input,
    out: &Outputs,
    part: &Part,
    scratch: View,
    kept: u32,
    fused: bool,
) -> LabResult<Vec<Launch>> {
    let n = part.rows.end - part.rows.start;
    let arm = if n == 512 { Arm::Tile } else { Arm::Coarse };
    let mut result = calls(m, c, input, out, part, scratch, arm)?;
    result.truncate(4);
    let producer = backend(true)
        .indexer_candidate_block_ids_runtime(
            &indexer::candidates(&FLASH, n),
            kept,
            part.width,
            Need::rows(n),
        )
        .map_err(|e| e.to_string())?;
    let buffers = CandidateBlockIdsBuffers {
        scores: m.slice(
            out.scores,
            part.first_word * 4,
            u64::from(n) * u64::from(part.width) * 4,
        ),
        visible_positions: m.row(input.visible, 4, part.rows.clone()),
        row_tags: m.row(input.tags, 8, part.rows.clone()),
        source_status: m.row(out.status, 4, part.rows.clone()),
        live_rows: m.input(
            &c.live()
                .saturating_sub(part.rows.start)
                .min(n)
                .to_le_bytes(),
        )?,
        output: record(m, &out.candidates, part.rows.clone(), part.width),
    };
    let next = producer
        .launches(&buffers, scratch, Counts::rows(n))
        .map_err(|e| e.to_string())?;
    if fused {
        result.push(cached::remap(&entry(), &next).map_err(|e| e.to_string())?);
    } else {
        result.extend(next);
    }
    Ok(result)
}

fn compare_candidates(
    m: &Memory<'_>,
    probe: &Module,
    a: &CandidateBlockIds<View>,
    b: &CandidateBlockIds<View>,
    counters: View,
) -> LabResult<()> {
    for (x, y) in [
        (a.ids, b.ids),
        (a.row_tags, b.row_tags),
        (a.visible_positions, b.visible_positions),
        (a.status, b.status),
    ] {
        compare(m, probe, x, y, counters)?;
    }
    Ok(())
}

fn reject_poison(
    m: &Memory<'_>,
    probe: &Module,
    a: View,
    b: View,
    counters: View,
) -> LabResult<()> {
    // Touch only the first active row. Replaying a live prefix must not
    // repair deliberately poisoned inactive rows outside that prefix.
    m.fill(probe, View { bytes: 4, ..b }, 0xdeadbeef)?;
    m.fill(probe, counters, 0)?;
    let words = u32::try_from(a.bytes / 4).unwrap();
    m.compare(probe, a, b, counters, 1, 1, words, words, 0, 0)?;
    assert!(
        m.read(counters)?.iter().any(|x| *x != 0),
        "negative control was accepted"
    );
    Ok(())
}

#[test]
fn cpu_publication_inventory_and_admission() {
    let sources = sources();
    assert_eq!(sources.len(), 30);
    assert_eq!(sources.iter().filter(|c| c.capture.is_some()).count(), 2);
    assert_eq!(sources.iter().filter(|&&c| memcheck_source(c)).count(), 9);
    assert_eq!(sources.iter().filter(|&&c| timed_source(c)).count(), 10);
    for c in sources {
        let positions: Vec<_> = (c.context - c.rows..c.context).map(|p| p as i32).collect();
        let parts = parts(c, &positions);
        assert_eq!(
            parts.iter().map(|p| p.rows.end - p.rows.start).sum::<u32>(),
            c.rows
        );
        assert!(parts.iter().all(|p| p.width <= 32768));
    }
    assert_eq!(WIDTHS.len() * 3 * COUNTS.len() * 2, 108);
    assert_eq!(cached::spec(&entry()).unwrap().params.len(), 11);
}

#[test]
#[ignore = "requires sealed prefill operands and an idle GPU pair"]
fn gpu_publication_real_edges_and_timing() {
    let ran=kernel_harness::gpu_test_with(test_cubins(),16<<30,|session| {
        let shard:usize=std::env::var("CHOPINFER_L4_SHARD").unwrap().parse().unwrap();
        let memcheck=std::env::var_os("CHOPINFER_L4_MEMCHECK").is_some();
        let mut modules=BTreeMap::new();
        let probe=session.load_cubin(&bytes(&artifact().join("indexer_prefill_probe.cubin"))?)?;
        let mut completed=0;
        for (source,c) in sources().into_iter().enumerate() {
            if memcheck && !memcheck_source(c) {continue;}
            let mut m=Memory{session,buffers:Vec::new(),views:Vec::new()};
            let h=Host::new(c)?;
            let input=Input::upload(&mut m,c,&h)?;
            let parts=parts(c,&h.positions);
            let last=parts.last().unwrap();
            let words=last.first_word+u64::from(last.rows.end-last.rows.start)*u64::from(last.width);
            let a=Outputs::allocate(&mut m,c.rows,words*4,c.context)?;
            let b=Outputs::allocate(&mut m,c.rows,words*4,c.context)?;
            let scratch=[m.alloc(scratch_bytes(c,&parts) as usize)?,m.alloc(scratch_bytes(c,&parts) as usize)?];
            let counters=m.input(&[0;32])?;
            for (ki,kept) in COUNTS.into_iter().enumerate() {
                let id=source*3+ki;
                if id%2!=shard {continue;}
                let mut plans=[Vec::new(),Vec::new()];
                for (arm,(out,scratch)) in [(&a,scratch[0]),(&b,scratch[1])].into_iter().enumerate() {
                    for p in &parts {
                        plans[arm].push(pipeline(&mut m,c,&input,out,p,scratch,kept,arm==1)?);
                    }
                }
                for arm in &plans {for p in arm {load(&m,&mut modules,p)?;}}
                a.poison(&m,&probe)?;b.poison(&m,&probe)?;
                for arm in &plans {for p in arm {fire(&m,&modules,p)?;}}
                compare_all(&m,&probe,&a,&b,counters)?;
                reject_poison(&m,&probe,a.candidates.ids,b.candidates.ids,counters)?;
                reject_poison(&m,&probe,a.candidates.status,b.candidates.status,counters)?;
                for p in &plans[1] {fire(&m,&modules,p)?;}
                compare_candidates(&m,&probe,&a.candidates,&b.candidates,counters)?;
                let timed=timed_source(c)&&!memcheck;
                if timed {
                    for producer_only in [true,false] {
                        let stage=if producer_only {"publish"}else{"all"};
                        let timed_plans=if producer_only {plans.clone()}else{
                            let mut reused=[Vec::new(),Vec::new()];
                            for (arm,(out,scratch)) in [(&a,scratch[0]),(&b,scratch[1])].into_iter().enumerate(){
                                for p in &parts {reused[arm].push(pipeline(&mut m,c,&input,out,&p.reused(),scratch,kept,arm==1)?);}
                            }
                            reused
                        };
                        let graphs=timed_plans.iter().map(|arm|session.capture(|_| {
                            for p in arm {fire(&m,&modules,if producer_only {&p[4..]}else{p})?;}Ok(())
                        })).collect::<LabResult<Vec<_>>>()?;
                        let timing=kernel_harness::with_timing_label(&format!("candidate-cached-{id}-{stage}"),||
                            kernel_harness::time_ab(session,1,3,[EvictionMode::Write;2],|arm,stream|graphs[arm].launch(stream).map_err(|e|e.to_string())))?;
                        compare_all(&m,&probe,&a,&b,counters)?;
                        println!("CANDIDATE_CACHED_TIMING {}",serde_json::json!({"id":id,"stage":stage,"median_ms":timing.arms.map(|a|a.median_ms),"paired_ratio":timing.paired_ratio,"exclusive":true,"score_buffer_reused":!producer_only}));
                    }
                }
                println!("CANDIDATE_CACHED_CASE {}",serde_json::json!({"id":id,"direct":false,"source":source,"context":c.context,"rows":c.rows,"kind":c.kind,"capture":c.capture,"kept":kept,"comparisons":8,"controls":2,"mismatches":0,"timed":timed,"memcheck":memcheck}));
                session.receipt("candidate-cached",BTreeMap::from([("id".into(),serde_json::json!(id)),("timed".into(),serde_json::json!(timed)),("mismatches".into(),serde_json::json!(0))]))?;
                completed+=1;
            }
        }
        completed+=direct(session,&probe,&mut modules,shard,memcheck)?;
        println!("CANDIDATE_CACHED_COMPLETE shard={shard} cases={completed} memcheck={memcheck}");
        Ok(())
    }).unwrap();
    assert!(ran.is_some(), "GPU test skipped");
}

fn direct(
    session: &kernel_harness::Session,
    probe: &Module,
    modules: &mut BTreeMap<String, Module>,
    shard: usize,
    memcheck: bool,
) -> LabResult<usize> {
    let mut completed = 0;
    let mut id = 90;
    for width in WIDTHS {
        for pattern in 0..3 {
            for kept in COUNTS {
                for live in [4u32, 8] {
                    let current = id;
                    id += 1;
                    if current % 2 != shard
                        || memcheck
                            && !(matches!(width, 1 | 16385 | 32768) && pattern == 2 && live == 8)
                    {
                        continue;
                    }
                    let mut m = Memory {
                        session,
                        buffers: Vec::new(),
                        views: Vec::new(),
                    };
                    let mut values: Vec<f32> = (0..8 * width)
                        .map(|i| match pattern {
                            0 => {
                                if i % 2 == 0 {
                                    0.0
                                } else {
                                    -0.0
                                }
                            }
                            1 => ((i.wrapping_mul(17) % 31) as f32 - 15.0) / 8.0,
                            _ => f32::NEG_INFINITY,
                        })
                        .collect();
                    // Inactive tails are poisoned too. Only visible entries may fail.
                    values[3 * width as usize] = f32::NAN;
                    values[4 * width as usize] = f32::INFINITY;
                    if width > 1 {
                        values[width as usize + 1] = f32::INFINITY;
                    }
                    let score = m.input(
                        &values
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )?;
                    let visible = m.input(&ints([
                        0,
                        1,
                        width as i32,
                        width as i32,
                        width as i32,
                        width as i32,
                        -1,
                        width as i32 + 1,
                    ]))?;
                    let tags = m.input(
                        &(0..8u64)
                            .flat_map(|i| (0x1234000000000000 + i).to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let source = m.input(&ints([0, 0, 0, 0, 0, 1, 0, 0]))?;
                    let live_rows = m.input(&live.to_le_bytes())?;
                    let a = Outputs::allocate(&mut m, 8, 8 * u64::from(width) * 4, width)?;
                    let b = Outputs::allocate(&mut m, 8, 8 * u64::from(width) * 4, width)?;
                    let op = backend(true)
                        .indexer_candidate_block_ids_runtime(
                            &indexer::candidates(&FLASH, 8),
                            kept,
                            width,
                            Need::rows(8),
                        )
                        .map_err(|e| e.to_string())?;
                    let scratch = m.alloc(op.info().workspace as usize)?;
                    let mut plans = Vec::new();
                    for (arm, out) in [&a, &b].into_iter().enumerate() {
                        let inputs = CandidateBlockIdsBuffers {
                            scores: score,
                            visible_positions: visible,
                            row_tags: tags,
                            source_status: source,
                            live_rows,
                            output: out.candidates,
                        };
                        let calls = op
                            .launches(&inputs, scratch, Counts::rows(8))
                            .map_err(|e| e.to_string())?;
                        plans.push(if arm == 1 {
                            vec![cached::remap(&entry(), &calls).map_err(|e| e.to_string())?]
                        } else {
                            calls
                        });
                    }
                    for p in &plans {
                        load(&m, modules, p)?;
                    }
                    a.poison(&m, probe)?;
                    b.poison(&m, probe)?;
                    for p in &plans {
                        fire(&m, modules, p)?;
                    }
                    let counters = m.input(&[0; 32])?;
                    compare_candidates(&m, probe, &a.candidates, &b.candidates, counters)?;
                    reject_poison(&m, probe, a.candidates.ids, b.candidates.ids, counters)?;
                    reject_poison(
                        &m,
                        probe,
                        a.candidates.status,
                        b.candidates.status,
                        counters,
                    )?;
                    println!(
                        "CANDIDATE_CACHED_CASE {}",
                        serde_json::json!({"id":current,"direct":true,"width":width,"pattern":pattern,"kept":kept,"live":live,"comparisons":4,"controls":2,"mismatches":0,"timed":false,"memcheck":memcheck})
                    );
                    session.receipt(
                        "candidate-cached",
                        BTreeMap::from([
                            ("id".into(), serde_json::json!(current)),
                            ("timed".into(), serde_json::json!(false)),
                            ("mismatches".into(), serde_json::json!(0)),
                        ]),
                    )?;
                    completed += 1;
                }
            }
        }
    }
    assert_eq!(id, 198);
    Ok(completed)
}
