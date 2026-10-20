//! Qualify early-stop prefill selection against the served selectors.
//!
//! Both arms run the served bounded scoring (`Arm::Tile`); the candidate arm
//! retargets only the local selector and its merge. Every case compares raw
//! scores, status, selected ids and candidate records bitwise, with reused
//! score buffers, split batches and crafted score planes (ties, equal
//! scores, holes, signed zeros, fewer high scores than k, a failed row).
use super::*;
use kernels_sm120::pooled_indexer::position_select_early as early;

pub(super) const CUBIN: &str = "position_select_early_deepseek41_sm120";
/// Crafted score patterns per crafted case.
const PATTERNS: u32 = 6;

/// The 65-row synthetic case of each layer at the widest context: both
/// dense selectors and the compact one.
fn crafted(c: Case) -> bool {
    c.capture.is_none() && c.kind == 0 && c.rows == 65 && c.context == 32768
}
fn memcheck_case(id: usize, c: Case) -> bool {
    id < 5 || c.kind == 3 || c.kind >= 8 || crafted(c)
}

/// The served schedule with only its selectors retargeted.
fn early_calls(mut calls: Vec<Launch>) -> LabResult<Vec<Launch>> {
    for call in &mut calls[2..4] {
        early::remap(call).map_err(|e| e.to_string())?;
    }
    Ok(calls)
}

/// Score values of one crafted pattern. Every value is finite except one
/// NaN, which fails its row: a non-finite admitted score is a refusal, so
/// holes use the lowest finite value instead of `-inf`.
fn pattern(kind: u32, row: u32, i: u32) -> f32 {
    let h = (u64::from(row) << 32 | u64::from(i)).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40;
    let unit = h as f32 / (1u64 << 24) as f32;
    match kind {
        // Distinct scores.
        0 => 1.0 + unit,
        // Sixteen score levels: positions break the ties.
        1 => (h % 16) as f32 * 0.25,
        // One score for the whole row.
        2 => 1.5,
        // Holes on odd positions and one failed row.
        3 if row == 1 && i == 7 => f32::NAN,
        3 if i % 2 == 1 => f32::MIN,
        3 => 0.5 + unit,
        // Signed zeros among values around zero.
        4 if h % 8 == 0 => {
            if h % 16 == 0 {
                -0.0
            } else {
                0.0
            }
        }
        4 => unit - 0.5,
        // Fewer distinct high scores than k above an equal floor.
        _ if i >= 300 => f32::MIN,
        _ => 2.0 + unit,
    }
}

/// Write crafted scores over the first rows of a score plane.
fn overwrite(m: &Memory<'_>, view: View, data: &[u8]) -> LabResult<()> {
    let (buffer, offset, len) = m.views[view.handle as usize - 1];
    assert!(data.len() <= len, "crafted scores fit the plane");
    m.buffers[buffer]
        .copy_from_bytes_at(offset, data, &m.session.stream)
        .map_err(|e| e.to_string())
}

#[test]
fn cpu_early_selection_inventory() {
    // Every served selector of the qualified schedules has an early entry
    // with the same parameters; nothing else in a schedule changes.
    let mut declared = BTreeMap::new();
    for e in crate::indexer_entries::entries(&FLASH) {
        let served = kernels_sm120::pooled_indexer::position_live::dense_specs(&e, false).unwrap();
        for (old, now) in served[1..].iter().zip(early::dense(&e).unwrap()) {
            assert_eq!(old.params, now.params);
            assert_eq!(
                old.body,
                now.body
                    .replace("position_select_early::", "position_live::")
            );
            declared.insert(now.symbol.clone(), ());
        }
    }
    for e in crate::indexer_entries::block_entries(&FLASH) {
        let served =
            kernels_sm120::pooled_indexer::position_valid::compact_specs(&e, false).unwrap();
        for (old, now) in served[1..].iter().zip(early::compact(&e).unwrap()) {
            assert_eq!(old.params, now.params);
            assert_eq!(
                old.body,
                now.body
                    .replace("position_select_early::", "position_valid::")
            );
            declared.insert(now.symbol.clone(), ());
        }
    }
    let mut remapped = BTreeMap::new();
    for c in cases() {
        let positions: Vec<_> = (c.context - c.rows..c.context).map(|p| p as i32).collect();
        for p in parts(c, &positions) {
            let n = p.rows.end - p.rows.start;
            let b = backend(true);
            let stride = if c.layer >= 24 { 16384 } else { p.width };
            let dense = PositionSelectBuffers {
                scores: v(12, u64::from(n) * u64::from(stride) * 4),
                ..cpu_buffers(n, FLASH.compress_ratio(c.layer), p.width)
            };
            let scratch = v(15, 1 << 32);
            let mut calls = if c.layer >= 24 {
                b.position_block_indexer(
                    &indexer::block_selection(&FLASH, c.layer, n).unwrap(),
                    c.kept(),
                    p.width,
                    Need::rows(n),
                )
                .unwrap()
                .launches(
                    &PositionBlockSelectBuffers {
                        q: dense.q,
                        head_weights: dense.head_weights,
                        keys: dense.keys,
                        positions: dense.positions,
                        row_batch: dense.row_batch,
                        live_rows: dense.live_rows,
                        row_tags: v(13, u64::from(n) * 8),
                        candidates: CandidateBlockIds {
                            ids: v(14, u64::from(n) * 8192),
                            row_tags: v(16, u64::from(n) * 8),
                            visible_positions: v(17, u64::from(n) * 4),
                            status: v(18, u64::from(n) * 4),
                            position_capacity: p.width,
                        },
                        scores: dense.scores,
                        status: dense.status,
                        output: dense.output,
                    },
                    scratch,
                    Counts::rows(n),
                )
                .unwrap()
            } else {
                b.position_indexer(
                    &indexer::selection(&FLASH, c.layer, n).unwrap(),
                    c.kept(),
                    p.width,
                    Need::rows(n),
                )
                .unwrap()
                .launches(&dense, scratch, Counts::rows(n))
                .unwrap()
            };
            assert_eq!(calls.len(), 4);
            Arm::Tile.adapt(c, &mut calls[1]);
            let served = calls.clone();
            let now = early_calls(calls).unwrap();
            for (i, (old, new)) in served.iter().zip(&now).enumerate() {
                assert_eq!(
                    (&old.arguments, old.plan, old.rows),
                    (&new.arguments, new.plan, new.rows)
                );
                assert_eq!(old.symbol == new.symbol, !(2..4).contains(&i));
            }
            for call in &now[2..4] {
                assert!(declared.contains_key(&call.symbol), "{}", call.symbol);
                assert_eq!(module(&call.symbol).unwrap(), CUBIN);
                *remapped.entry(call.symbol.clone()).or_insert(0) += 1;
            }
            // A second remap refuses: the early entries are not served names.
            assert!(early_calls(now).is_err());
        }
    }
    // Both dense ratios, the compact query, each with its merge.
    assert_eq!(remapped.len(), 6, "{remapped:?}");
    assert_eq!(declared.len(), 6);
    let ids: Vec<_> = cases().into_iter().enumerate().collect();
    assert_eq!(ids.iter().filter(|(_, c)| crafted(*c)).count(), 3);
    assert_eq!(
        ids.iter().filter(|(i, c)| memcheck_case(*i, *c)).count(),
        26
    );
    assert_eq!(ids.iter().filter(|(_, c)| c.timed()).count(), 22);
    let non_finite = (0..PATTERNS)
        .flat_map(|k| (0..65).flat_map(move |r| (0..1024).map(move |i| pattern(k, r, i))))
        .filter(|x| !x.is_finite())
        .count();
    assert!(non_finite == 1 && pattern(3, 1, 7).is_nan());
}

#[test]
#[ignore = "requires sealed prefill operands and an idle GPU pair"]
fn gpu_early_selection_real_edges_and_timing() {
    let ran=kernel_harness::gpu_test_with(test_cubins(),16<<30,|session| {
        let shard:usize=std::env::var("CHOPINFER_L4_SHARD").unwrap().parse().unwrap();
        let memcheck=std::env::var_os("CHOPINFER_L4_MEMCHECK").is_some();
        let mut modules=BTreeMap::new();
        let probe=session.load_cubin(&bytes(&artifact().join("indexer_prefill_probe.cubin"))?)?;
        let mut completed=0;
        for (id,c) in cases().into_iter().enumerate().filter(|(i,c)|i%2==shard && (!memcheck || memcheck_case(*i,*c))) {
            let mut m=Memory{session,buffers:Vec::new(),views:Vec::new()};
            let h=Host::new(c)?;
            let input=Input::upload(&mut m,c,&h)?;
            let parts=parts(c,&h.positions);
            let last=parts.last().unwrap();
            let stride=|p:&Part|u64::from(if c.layer>=24 {16384}else{p.width});
            let words=last.first_word+u64::from(last.rows.end-last.rows.start)*stride(last);
            let a=Outputs::allocate(&mut m,c.rows,words*4,c.context)?;
            let b=Outputs::allocate(&mut m,c.rows,words*4,c.context)?;
            let scratch=[m.alloc(scratch_bytes(c,&parts) as usize)?,m.alloc(scratch_bytes(c,&parts) as usize)?];
            let counts=m.input(&[0;32])?;
            let served=|m:&mut Memory<'_>,out:&Outputs,scratch,ps:&[Part]|ps.iter().map(|p|calls(m,c,&input,out,p,scratch,Arm::Tile)).collect::<LabResult<Vec<_>>>();
            let candidate=|m:&mut Memory<'_>,out:&Outputs,scratch,ps:&[Part]|ps.iter().map(|p|early_calls(calls(m,c,&input,out,p,scratch,Arm::Tile)?)).collect::<LabResult<Vec<_>>>();
            let reused:Vec<Part>=parts.iter().map(Part::reused).collect();
            let plans=[served(&mut m,&a,scratch[0],&parts)?,candidate(&mut m,&b,scratch[1],&parts)?];
            for arm in &plans {for p in arm {load(&m,&mut modules,p)?;}}
            let mut comparisons=0;
            // Complete pipelines from poisoned outputs.
            a.poison(&m,&probe)?;b.poison(&m,&probe)?;
            for arm in &plans {for p in arm {fire(&m,&modules,p)?;}}
            compare_all(&m,&probe,&a,&b,counts)?;comparisons+=1;
            if let Some(expected)=&h.selected {assert_eq!(&m.read(b.ids)?,expected);assert_eq!(m.read(b.status)?,h.expected_status.clone().unwrap());}
            // Selection alone again over the same scores is idempotent.
            for arm in &plans {for p in arm {fire(&m,&modules,&p[2..4])?;}}
            compare_all(&m,&probe,&a,&b,counts)?;comparisons+=1;
            // Split every original batch without changing its score stride.
            b.poison(&m,&probe)?;
            for p in &parts {
                let mid=p.rows.start+(p.rows.end-p.rows.start)/2;
                for r in [p.rows.start..mid,mid..p.rows.end] {
                    if r.is_empty() {continue;}
                    let next=Part{first_word:p.first_word+u64::from(r.start-p.rows.start)*stride(p),rows:r,width:p.width};
                    let plan=early_calls(calls(&mut m,c,&input,&b,&next,scratch[1],Arm::Tile)?)?;
                    fire(&m,&modules,&plan)?;
                }
            }
            compare_all(&m,&probe,&a,&b,counts)?;comparisons+=1;
            // Serving overwrites one score buffer after each batch's consumer.
            a.poison(&m,&probe)?;b.poison(&m,&probe)?;
            let reuse=[served(&mut m,&a,scratch[0],&reused)?,candidate(&mut m,&b,scratch[1],&reused)?];
            for arm in &reuse {for p in arm {fire(&m,&modules,p)?;}}
            compare_all(&m,&probe,&a,&b,counts)?;comparisons+=1;
            // Crafted score planes through the selectors alone, one reused
            // plane per batch as serving writes them.
            let mut patterns=0;
            if crafted(c) {
                for kind in 0..PATTERNS {
                    a.poison(&m,&probe)?;b.poison(&m,&probe)?;
                    for (k,p) in reused.iter().enumerate() {
                        let width=u32::try_from(stride(p)).unwrap();
                        let data:Vec<u8>=p.rows.clone().flat_map(|row|(0..width).map(move |i|pattern(kind,row,i))).flat_map(f32::to_le_bytes).collect();
                        for (out,arm) in [(&a,&reuse[0]),(&b,&reuse[1])] {
                            overwrite(&m,out.scores,&data)?;
                            fire(&m,&modules,&arm[k][2..4])?;
                        }
                    }
                    for (x,y) in [(a.scores,b.scores),(a.status,b.status),(a.ids,b.ids)] {compare(&m,&probe,x,y,counts)?;}
                    comparisons+=1;patterns+=1;
                }
            }
            // Negative controls: one flipped selected id and one flipped score.
            for (old,now) in [(a.ids,b.ids),(a.scores,b.scores)] {
                let first=m.slice(old,0,4);let target=m.slice(now,0,4);
                let bits=u32::from_le_bytes(m.read(first)?.try_into().unwrap());
                m.fill(&probe,target,bits^1)?;m.fill(&probe,counts,0)?;
                m.compare(&probe,first,target,counts,1,1,1,1,0,0)?;
                assert_eq!(m.read(counts)?[0],1,"negative control escaped");m.fill(&probe,target,bits)?;
            }
            let timed=c.timed() && !memcheck;
            if timed {
                a.poison(&m,&probe)?;b.poison(&m,&probe)?;
                let plans=[served(&mut m,&a,scratch[0],&reused)?,candidate(&mut m,&b,scratch[1],&reused)?];
                for arm in &plans {for p in arm {fire(&m,&modules,p)?;}}
                for (stage,select_only) in [("select",true),("all",false)] {
                    let graphs=plans.iter().map(|arm|session.capture(|_| {for p in arm {fire(&m,&modules,if select_only {&p[2..4]}else{p})?;}Ok(())})).collect::<LabResult<Vec<_>>>()?;
                    let timing=kernel_harness::with_timing_label(&format!("select-early-{id}-{stage}"),||kernel_harness::time_ab(session,1,3,[EvictionMode::Write;2],|arm,stream|graphs[arm].launch(stream).map_err(|e|e.to_string())))?;
                    compare_all(&m,&probe,&a,&b,counts)?;
                    println!("SELECT_EARLY_TIMING {}",serde_json::json!({"id":id,"stage":stage,"batches":parts.len(),"median_ms":timing.arms.map(|a|a.median_ms),"paired_ratio":timing.paired_ratio,"exclusive":true,"score_buffer_reused":true}));
                }
            }
            println!("SELECT_EARLY_CASE {}",serde_json::json!({"id":id,"layer":c.layer,"context":c.context,"rows":c.rows,"kind":c.kind,"capture":c.capture,"live":c.live(),"kept":c.kept(),"batches":parts.len(),"comparisons":comparisons,"patterns":patterns,"controls":2,"mismatches":0,"timed":timed,"memcheck":memcheck}));
            session.receipt("select-early",BTreeMap::from([("id".into(),serde_json::json!(id)),("timed".into(),serde_json::json!(timed)),("mismatches".into(),serde_json::json!(0))]))?;
            completed+=1;
        }
        println!("SELECT_EARLY_COMPLETE shard={shard} cases={completed} memcheck={memcheck}");Ok(())
    }).unwrap();
    assert!(ran.is_some(), "GPU test skipped");
}
