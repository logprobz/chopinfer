//! Compare bounded prefill scoring with the public and coarse guarded paths.
#[path = "indexer_prefill_tile_fixture.rs"]
mod fixture;
#[path = "indexer_prefill_publish.rs"]
mod publish;
use super::*;
use fixture::{Host, Input};
use kernel_core::position_blocks::PositionBlockSelectBuffers;
use kernels_sm120::pooled_indexer::position_mma_prefill_tile as tile;

#[derive(Clone, Copy, Debug)]
struct Case {
    layer: u32,
    context: u32,
    rows: u32,
    kind: u32,
    capture: Option<usize>,
}
impl Case {
    fn timed(self) -> bool {
        self.capture.is_some() || self.rows == 8192 && self.kind == 0
    }
    fn live(self) -> u32 {
        if self.kind == 12 {
            self.rows - 1
        } else {
            self.rows
        }
    }
    fn kept(self) -> u32 {
        if self.kind == 11 { 31 } else { 512 }
    }
}
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for context in [8192, 32768] {
        for layer in [2, 8, 14, 20, 24] {
            out.push(Case {
                layer,
                context,
                rows: 32,
                kind: 0,
                capture: Some(out.len()),
            });
        }
    }
    for layer in [2, 20, 24] {
        for context in [8192, 16384, 24576, 32768] {
            for rows in [8192, 8191, 2496, 65] {
                out.push(Case {
                    layer,
                    context,
                    rows,
                    kind: 0,
                    capture: None,
                });
            }
        }
        for kind in 1..=12 {
            out.push(Case {
                layer,
                context: 32768,
                rows: 65,
                kind,
                capture: None,
            });
        }
    }
    assert_eq!(out.len(), 94);
    out
}
fn artifact() -> PathBuf {
    std::env::var_os("CHOPINFER_INDEX_PREFILL_TILE_ARTIFACTS")
        .expect("sealed prefill tile artifacts")
        .into()
}
fn backend(coarse: bool) -> Sm120 {
    let b = super::super::backend(false);
    if coarse {
        b.with_coarse_guarded_position_scores()
    } else {
        b.with_guarded_position_scores()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Public,
    Coarse,
    Narrow,
    Wide,
    Tile,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Coarse => "coarse",
            Self::Narrow => "narrow",
            Self::Wide => "wide",
            Self::Tile => "tile",
        }
    }
    fn adapt(self, c: Case, call: &mut Launch) {
        match self {
            Self::Public | Self::Coarse => (),
            Self::Narrow => call.plan.grid[0] = 8,
            Self::Wide | Self::Tile => tile::remap(
                call,
                c.layer >= 24,
                if self == Self::Wide { 192 } else { 8 },
            )
            .unwrap(),
        }
    }
}
fn module(symbol: &str) -> LabResult<&'static str> {
    if symbol.starts_with("chop_candidate_block_ids_") && symbol.ends_with("_cached4_v1") {
        Ok(publish::CUBIN)
    } else if symbol.contains("_counts_v1") || symbol.contains("_mma_fp4_prefill_tile_v1") {
        Ok(tile::CUBIN)
    } else {
        Ok(bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol)
            .ok_or("missing indexer module")?
            .0)
    }
}
fn load(m: &Memory<'_>, modules: &mut BTreeMap<String, Module>, calls: &[Launch]) -> LabResult<()> {
    for call in calls {
        let name = module(&call.symbol)?;
        if let std::collections::btree_map::Entry::Vacant(slot) = modules.entry(name.into()) {
            let data = if name == tile::CUBIN {
                bytes(&artifact().join(format!("{name}.cubin")))?
            } else if let Some((_, data)) =
                bundle_deepseek41::sm120::fold_cubin_for_symbol(&call.symbol)
            {
                data.to_vec()
            } else {
                bytes(&artifact().join(format!("{name}.cubin")))?
            };
            slot.insert(m.session.load_cubin(&data)?);
        }
    }
    Ok(())
}
fn fire(m: &Memory<'_>, modules: &BTreeMap<String, Module>, calls: &[Launch]) -> LabResult<()> {
    for call in calls {
        let f = modules[module(&call.symbol)?]
            .function(&call.symbol)
            .map_err(|e| e.to_string())?;
        let args = call
            .arguments
            .0
            .iter()
            .map(|a| match a {
                Argument::DeviceBuffer(h) => m.argument(v(*h, 0)),
                Argument::U32(x) => KernelArgument::U32(*x),
                Argument::U64(x) => KernelArgument::U64(*x),
                Argument::I32(x) => KernelArgument::I32(*x),
                Argument::NullDevicePointer => KernelArgument::NullBuffer,
                _ => panic!("unexpected indexer argument"),
            })
            .collect::<Vec<_>>();
        kernel_harness::launch::launch_with(
            &f,
            &m.session.stream,
            kernel_harness::launch::config(call.plan)?,
            &args,
        )?;
    }
    Ok(())
}
#[derive(Clone, Debug)]
struct Part {
    rows: std::ops::Range<u32>,
    width: u32,
    first_word: u64,
}
impl Part {
    fn reused(&self) -> Self {
        Self {
            first_word: 0,
            ..self.clone()
        }
    }
}
fn parts(c: Case, positions: &[i32]) -> Vec<Part> {
    let batches = if c.layer >= 24 {
        (0..c.rows)
            .step_by(4096)
            .map(|start| Batch {
                rows: start..(start + 4096).min(c.rows),
                width: 1 << 20,
            })
            .collect()
    } else {
        plan(
            Shape {
                layer: c.layer,
                context: c.context,
                rows: c.rows,
                case: 0,
            },
            positions,
        )
    };
    let mut word = 0;
    batches
        .into_iter()
        .map(|b| {
            let part = Part {
                rows: b.rows.clone(),
                width: b.width,
                first_word: word,
            };
            word += u64::from(b.rows.end - b.rows.start)
                * u64::from(if c.layer >= 24 { 16384 } else { b.width });
            part
        })
        .collect()
}
fn record(
    m: &mut Memory<'_>,
    r: &CandidateBlockIds<View>,
    range: std::ops::Range<u32>,
    width: u32,
) -> CandidateBlockIds<View> {
    CandidateBlockIds {
        ids: m.row(r.ids, 8192, range.clone()),
        row_tags: m.row(r.row_tags, 8, range.clone()),
        visible_positions: m.row(r.visible_positions, 4, range.clone()),
        status: m.row(r.status, 4, range),
        position_capacity: width,
    }
}
#[allow(clippy::too_many_arguments)]
fn calls(
    m: &mut Memory<'_>,
    c: Case,
    input: &Input,
    out: &Outputs,
    p: &Part,
    scratch: View,
    arm: Arm,
) -> LabResult<Vec<Launch>> {
    let n = p.rows.end - p.rows.start;
    let backend = backend(arm != Arm::Public);
    let q = m.row(input.dense.q, 8192, p.rows.clone());
    let weights = m.row(input.dense.head_weights, 128, p.rows.clone());
    let positions = m.row(input.dense.positions, 4, p.rows.clone());
    let batches = m.row(input.dense.row_batch, 4, p.rows.clone());
    let live = m.input(&c.live().saturating_sub(p.rows.start).min(n).to_le_bytes())?;
    let stride = if c.layer >= 24 { 16384 } else { p.width };
    let scores = m.slice(
        out.scores,
        p.first_word * 4,
        u64::from(n) * u64::from(stride) * 4,
    );
    let status = m.row(out.status, 4, p.rows.clone());
    let output = m.row(out.ids, 2048, p.rows.clone());
    let mut calls = if c.layer >= 24 {
        let query = indexer::block_selection(&FLASH, c.layer, n).unwrap();
        let op = backend
            .position_block_indexer(&query, c.kept(), p.width, Need::rows(n))
            .unwrap();
        let tags = m.row(input.tags, 8, p.rows.clone());
        let candidates = record(m, &input.retained, p.rows.clone(), p.width);
        op.launches(
            &PositionBlockSelectBuffers {
                q,
                head_weights: weights,
                keys: input.dense.keys,
                positions,
                row_batch: batches,
                live_rows: live,
                row_tags: tags,
                candidates,
                scores,
                status,
                output,
            },
            scratch,
            Counts::rows(n),
        )
        .map_err(|e| e.to_string())?
    } else {
        let query = indexer::selection(&FLASH, c.layer, n).unwrap();
        let op = backend
            .position_indexer(&query, c.kept(), p.width, Need::rows(n))
            .unwrap();
        op.launches(
            &PositionSelectBuffers {
                q,
                head_weights: weights,
                keys: input.dense.keys,
                positions,
                row_batch: batches,
                live_rows: live,
                candidate_mask: None,
                position_capacity: p.width,
                scores,
                status,
                output,
            },
            scratch,
            Counts::rows(n),
        )
        .map_err(|e| e.to_string())?
    };
    assert_eq!(calls.len(), 4);
    arm.adapt(c, &mut calls[1]);
    if c.layer == 20 {
        let producer = backend
            .indexer_candidate_block_ids_runtime(
                &indexer::candidates(&FLASH, n),
                2048,
                p.width,
                Need::rows(n),
            )
            .unwrap();
        let visible = m.row(input.visible, 4, p.rows.clone());
        let tags = m.row(input.tags, 8, p.rows.clone());
        let output = record(m, &out.candidates, p.rows.clone(), p.width);
        calls.extend(
            producer
                .launches(
                    &CandidateBlockIdsBuffers {
                        scores,
                        visible_positions: visible,
                        row_tags: tags,
                        source_status: status,
                        live_rows: live,
                        output,
                    },
                    scratch,
                    Counts::rows(n),
                )
                .map_err(|e| e.to_string())?,
        );
    }
    Ok(calls)
}
fn scratch_bytes(c: Case, parts: &[Part]) -> u64 {
    let b = backend(false);
    parts
        .iter()
        .map(|p| {
            let n = p.rows.end - p.rows.start;
            let score = if c.layer >= 24 {
                b.position_block_indexer(
                    &indexer::block_selection(&FLASH, c.layer, n).unwrap(),
                    c.kept(),
                    p.width,
                    Need::rows(n),
                )
                .unwrap()
                .info()
                .workspace
            } else {
                b.position_indexer(
                    &indexer::selection(&FLASH, c.layer, n).unwrap(),
                    c.kept(),
                    p.width,
                    Need::rows(n),
                )
                .unwrap()
                .info()
                .workspace
            };
            if c.layer == 20 {
                score.max(
                    b.indexer_candidate_block_ids_runtime(
                        &indexer::candidates(&FLASH, n),
                        2048,
                        p.width,
                        Need::rows(n),
                    )
                    .unwrap()
                    .info()
                    .workspace,
                )
            } else {
                score
            }
        })
        .max()
        .unwrap()
}
fn compare(m: &Memory<'_>, probe: &Module, a: View, b: View, counters: View) -> LabResult<()> {
    assert_eq!(a.bytes, b.bytes);
    m.fill(probe, counters, 0)?;
    let words = u32::try_from(a.bytes / 4).unwrap();
    m.compare(probe, a, b, counters, 1, 1, words, words, 0, 0)?;
    assert!(
        m.read(counters)?.iter().all(|x| *x == 0),
        "bitwise comparison failed"
    );
    Ok(())
}
fn compare_all(
    m: &Memory<'_>,
    probe: &Module,
    a: &Outputs,
    b: &Outputs,
    counters: View,
) -> LabResult<()> {
    for (a, b) in [
        (a.scores, b.scores),
        (a.status, b.status),
        (a.ids, b.ids),
        (a.candidates.ids, b.candidates.ids),
        (a.candidates.row_tags, b.candidates.row_tags),
        (
            a.candidates.visible_positions,
            b.candidates.visible_positions,
        ),
        (a.candidates.status, b.candidates.status),
    ] {
        compare(m, probe, a, b, counters)?;
    }
    Ok(())
}
fn diagnose(
    m: &mut Memory<'_>,
    modules: &mut BTreeMap<String, Module>,
    plans: &[Vec<Launch>],
    scratch: View,
) -> LabResult<serde_json::Value> {
    let counters = m.input(&[0; 16])?;
    let mut pack_fallback_rows = 0;
    for plan in plans {
        fire(m, modules, &plan[..1])?;
        let Argument::U64(offset) = plan[0].arguments.0.last().unwrap() else {
            panic!("pack status offset")
        };
        let status = m.slice(scratch, *offset, u64::from(plan[1].rows) * 4);
        pack_fallback_rows += m
            .read(status)?
            .chunks_exact(4)
            .filter(|b| b.iter().any(|x| *x != 0))
            .count();
        let mut call = plan[1].clone();
        call.symbol
            .push_str(if call.symbol.contains("_mma_fp4_prefill_tile_v1") {
                "_counts_v1"
            } else {
                "_counts_v1_prefill_probe_v1"
            });
        call.arguments
            .0
            .push(Argument::DeviceBuffer(counters.handle));
        load(m, modules, std::slice::from_ref(&call))?;
        fire(m, modules, &[call])?;
    }
    let data = m.read(counters)?;
    let words: Vec<_> = data
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok(
        serde_json::json!({"admitted":words[0],"repairs":words[1],"coarse_safe":words[2],
        "fine_checked":words[3],"pack_fallback_rows":pack_fallback_rows,"timed":false}),
    )
}
#[test]
fn cpu_prefill_tiles_preserve_abi_fallback_and_chunk_offsets() {
    for c in cases().into_iter().filter(|c| c.capture.is_none()) {
        let fixture = Case {
            rows: c.rows.min(65),
            ..c
        };
        let h = Host::new(fixture).unwrap();
        assert_eq!(h.q.len(), fixture.rows as usize * 8192);
        if c.kind == 3 {
            assert_eq!(&h.q[8192..8194], &0x3f81_u16.to_le_bytes());
        }
        let positions: Vec<_> = (c.context - c.rows..c.context).map(|p| p as i32).collect();
        let p = parts(c, &positions);
        assert_eq!(p.first().unwrap().rows.start, 0);
        assert_eq!(p.last().unwrap().rows.end, c.rows);
        assert!(scratch_bytes(c, &p) > 0);
        for pair in p.windows(2) {
            assert_eq!(pair[0].rows.end, pair[1].rows.start);
        }
        for part in &p {
            let reused = part.reused();
            assert_eq!(reused.rows, part.rows);
            let stride = if c.layer >= 24 { 16384 } else { reused.width };
            assert!(
                u64::from(reused.rows.end - reused.rows.start) * u64::from(stride) * 4
                    <= SCORE_BYTES
            );
            assert_eq!(reused.first_word, 0);
        }
    }
    for e in crate::indexer_entries::block_entries(&FLASH) {
        let old = kernels_sm120::pooled_indexer::position_mma_guard::compact_specs(&e)
            .unwrap()
            .remove(1);
        let now = tile::compact_spec(&e).unwrap();
        assert_eq!(old.params, now.params);
        assert_eq!(
            old.body,
            now.body
                .replace("position_mma_prefill_tile::", "position_mma_guard::")
        );
        assert!(now.body.contains("position_valid::compact_scores"));
    }
    assert_eq!(cases().len(), 94);
}
#[test]
#[ignore = "requires sealed prefill operands and an idle GPU pair"]
fn gpu_prefill_tiles_real_edges_partitions_and_timing() {
    let ran=kernel_harness::gpu_test_with(test_cubins(),16<<30,|session| {
        let shard:usize=std::env::var("CHOPINFER_L4_SHARD").unwrap().parse().unwrap();
        let memcheck=std::env::var_os("CHOPINFER_L4_MEMCHECK").is_some();
        let mut modules=BTreeMap::new();
        let probe=session.load_cubin(&bytes(&artifact().join("indexer_prefill_probe.cubin"))?)?;
        let mut completed=0;
        for (id,c) in cases().into_iter().enumerate().filter(|(i,c)|i%2==shard && (!memcheck || *i<5 || c.kind==3 || c.kind>=8)) {
            let mut m=Memory{session,buffers:Vec::new(),views:Vec::new()};
            let h=Host::new(c)?;
            let input=Input::upload(&mut m,c,&h)?;
            let parts=parts(c,&h.positions);
            let last=parts.last().unwrap();
            let words=last.first_word+u64::from(last.rows.end-last.rows.start)*u64::from(if c.layer>=24 {16384}else{last.width});
            let a=Outputs::allocate(&mut m,c.rows,words*4,c.context)?;
            let b=Outputs::allocate(&mut m,c.rows,words*4,c.context)?;
            let scratch=[m.alloc(scratch_bytes(c,&parts) as usize)?,m.alloc(scratch_bytes(c,&parts) as usize)?];
            let counts=m.input(&[0;32])?;
            let schedules=|m:&mut Memory<'_>,out:&Outputs,scratch,arm|parts.iter().map(|p|calls(m,c,&input,out,p,scratch,arm)).collect::<LabResult<Vec<_>>>();
            let reused=|m:&mut Memory<'_>,out:&Outputs,scratch,arm|parts.iter().map(|p|calls(m,c,&input,out,&p.reused(),scratch,arm)).collect::<LabResult<Vec<_>>>();
            let public=schedules(&mut m,&a,scratch[0],Arm::Public)?;
            for p in &public {load(&m,&mut modules,p)?;}
            let mut raw=0;let mut pipelines=0;
            for arm in [Arm::Coarse,Arm::Narrow,Arm::Wide,Arm::Tile] {
                let candidate=schedules(&mut m,&b,scratch[1],arm)?;
                for p in &candidate {load(&m,&mut modules,p)?;}
                a.poison(&m,&probe)?;b.poison(&m,&probe)?;
                for p in &public {fire(&m,&modules,&p[..2])?;}
                for p in &candidate {fire(&m,&modules,&p[..2])?;}
                compare(&m,&probe,a.scores,b.scores,counts)?;raw+=1;
                for p in &public {fire(&m,&modules,p)?;}
                for p in &candidate {fire(&m,&modules,p)?;}
                compare_all(&m,&probe,&a,&b,counts)?;pipelines+=1;
            }
            if let Some(expected)=&h.selected {assert_eq!(&m.read(a.ids)?,expected);assert_eq!(m.read(a.status)?,h.expected_status.clone().unwrap());}
            let first_status=m.slice(a.status,0,4);
            if matches!(c.kind,3..=5) || c.layer>=24 && matches!(c.kind,8..=10) {
                assert_eq!(m.read(first_status)?,1_i32.to_le_bytes());
            }
            if c.kind==3 {
                let finite_fallback=m.slice(a.status,4,4);
                assert_eq!(m.read(finite_fallback)?,0_i32.to_le_bytes());
            }
            if c.kind==1 {
                let first_ids=m.slice(a.ids,0,512*4);
                assert_eq!(m.read(first_status)?,0_i32.to_le_bytes());
                let expected=if c.layer>=24 {
                    h.ids[..2048].iter().copied().filter(|&id|id>=0)
                        .flat_map(|id|(0..8).map(move |offset|id*8+offset)).take(512).collect::<Vec<_>>()
                }else {(0..512).collect()};
                assert_eq!(m.read(first_ids)?,ints(expected));
            }
            if c.kind==12 {
                let inactive=m.slice(a.status,u64::from(c.live())*4,4);
                assert_eq!(m.read(inactive)?,POISON.to_le_bytes());
            }
            // Split every original bounded batch without changing its score stride.
            b.poison(&m,&probe)?;
            for p in &parts {
                let mid=p.rows.start+(p.rows.end-p.rows.start)/2;
                for r in [p.rows.start..mid,mid..p.rows.end] {
                    if r.is_empty() {continue;}
                    let next=Part{first_word:p.first_word+u64::from(r.start-p.rows.start)*u64::from(if c.layer>=24 {16384}else{p.width}),rows:r,width:p.width};
                    let plan=calls(&mut m,c,&input,&b,&next,scratch[1],Arm::Tile)?;
                    fire(&m,&modules,&plan)?;
                }
            }
            compare_all(&m,&probe,&a,&b,counts)?;
            for (old,now) in [(a.scores,b.scores),(a.ids,b.ids)] {
                let first=m.slice(old,0,4);let target=m.slice(now,0,4);
                let bits=u32::from_le_bytes(m.read(first)?.try_into().unwrap());
                m.fill(&probe,target,bits^1)?;m.fill(&probe,counts,0)?;
                m.compare(&probe,first,target,counts,1,1,1,1,0,0)?;
                assert_eq!(m.read(counts)?[0],1,"negative control escaped");m.fill(&probe,target,bits)?;
            }
            // Serving overwrites one score buffer after each batch's ordered
            // consumer. Keep materialized planes only for the raw comparison.
            a.poison(&m,&probe)?;b.poison(&m,&probe)?;
            for (out,workspace,arm) in [(&a,scratch[0],Arm::Public),(&b,scratch[1],Arm::Tile)] {
                for plan in reused(&mut m,out,workspace,arm)? {fire(&m,&modules,&plan)?;}
            }
            compare_all(&m,&probe,&a,&b,counts)?;
            let timed=c.timed() && !memcheck;
            if timed {
                let fine=diagnose(&mut m,&mut modules,&public,scratch[0])?;
                let coarse=schedules(&mut m,&b,scratch[1],Arm::Coarse)?;
                let coarse=diagnose(&mut m,&mut modules,&coarse,scratch[1])?;
                assert_eq!(fine["admitted"],coarse["admitted"]);
                assert_eq!(fine["repairs"],coarse["repairs"]);
                let wide=if c.layer>=24 {
                    let plans=schedules(&mut m,&b,scratch[1],Arm::Tile)?;
                    let wide=diagnose(&mut m,&mut modules,&plans,scratch[1])?;
                    assert_eq!(wide,coarse);
                    Some(wide)
                }else{None};
                println!("PREFILL_TILE_GUARDS {}",serde_json::json!({"id":id,"public":fine,"coarse":coarse,"compact_tile":wide}));
                for (label,reference,alternative) in [("coarse",Arm::Public,Arm::Coarse),("narrow",Arm::Coarse,Arm::Narrow),("wide",Arm::Coarse,Arm::Wide),("tile",Arm::Coarse,Arm::Tile)] {
                    a.poison(&m,&probe)?;b.poison(&m,&probe)?;
                    let plans=[reused(&mut m,&a,scratch[0],reference)?,reused(&mut m,&b,scratch[1],alternative)?];
                    for arm in &plans {for p in arm {fire(&m,&modules,p)?;}}
                    for (stage,score_only) in [("pack_score",true),("all",false)] {
                        let graphs=plans.iter().map(|arm|session.capture(|_| {for p in arm {fire(&m,&modules,if score_only {&p[..2]}else{p})?;}Ok(())})).collect::<LabResult<Vec<_>>>()?;
                        let timing=kernel_harness::with_timing_label(&format!("prefill-tile-{id}-{label}-{stage}"),||kernel_harness::time_ab(session,1,3,[EvictionMode::Write;2],|arm,stream|graphs[arm].launch(stream).map_err(|e|e.to_string())))?;
                        compare_all(&m,&probe,&a,&b,counts)?;
                        println!("PREFILL_TILE_TIMING {}",serde_json::json!({"id":id,"stage":stage,"candidate":label,"reference":reference.name(),"batches":parts.len(),"median_ms":timing.arms.map(|a|a.median_ms),"paired_ratio":timing.paired_ratio,"exclusive":true,"score_buffer_reused":true}));
                    }
                }
            }
            println!("PREFILL_TILE_CASE {}",serde_json::json!({"id":id,"layer":c.layer,"context":c.context,"rows":c.rows,"kind":c.kind,"capture":c.capture,"live":c.live(),"kept":c.kept(),"batches":parts.len(),"raw":raw,"pipelines":pipelines,"reuse_pipelines":1,"partitions":1,"controls":2,"mismatches":0,"timed":timed,"memcheck":memcheck}));
            session.receipt("indexer-prefill-tile",BTreeMap::from([("id".into(),serde_json::json!(id)),("timed".into(),serde_json::json!(timed)),("mismatches".into(),serde_json::json!(0))]))?;
            completed+=1;
        }
        println!("PREFILL_TILE_COMPLETE shard={shard} cases={completed} memcheck={memcheck}");Ok(())
    }).unwrap();
    assert!(ran.is_some(), "GPU test skipped");
}
