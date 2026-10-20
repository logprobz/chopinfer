//! Guarded serving entries, the unguarded negative control and sealed operands.
#[path = "indexer_mma_guard_edges.rs"]
mod edges;
use super::*;
use kernel_harness::{Cubins, eviction::EvictionMode};

pub(super) fn guarded(native: bool) -> Sm120 {
    let backend = super::backend(false);
    if native {
        backend.with_guarded_position_scores()
    } else {
        backend
    }
}
#[test]
fn cpu_guard_keeps_plans_workspace_and_selection() {
    for layer in [2, 20, 24] {
        let b = cpu_buffers(8192, FLASH.compress_ratio(layer), 16384);
        let query = indexer::selection(&FLASH, layer, 8192).unwrap();
        let old = super::backend(true)
            .position_indexer(&query, 512, 16384, Need::rows(8192))
            .unwrap();
        let new = guarded(true)
            .position_indexer(&query, 512, 16384, Need::rows(8192))
            .unwrap();
        assert_eq!(old.info().workspace, new.info().workspace);
        for rows in [1, 2, 4, 6, 8, 16, 32, 8192] {
            let scratch = v(99, new.info().workspace);
            let mut expected = old.launches(&b, scratch, Counts::rows(rows)).unwrap();
            for call in &mut expected {
                call.symbol = call.symbol.replace("_fp4_v1", "_fp4_guard_v1");
            }
            assert_eq!(
                expected,
                new.launches(&b, scratch, Counts::rows(rows)).unwrap()
            );
        }
    }
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_INDEX_CAPTURE_MANIFEST and idle GPUs"]
fn gpu_capture_failures_match_simt() {
    captured_comparisons(guarded);
}

fn load_calls(
    session: &Session,
    modules: &mut BTreeMap<String, runtime_cuda::Module>,
    calls: &[Launch],
) -> LabResult<()> {
    for call in calls {
        super::guard_fast::load(session, modules, &call.symbol)?;
    }
    Ok(())
}
fn mismatch(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(x, y)| x != y)
        .count()
}
#[allow(clippy::too_many_arguments)]
fn exercise<B>(
    session: &Session,
    arena: &mut Arena<'_>,
    b: &B,
    planes: [View; 3],
    ops: [Prepared<B>; 3],
    rows: u32,
    label: serde_json::Value,
    timing: bool,
    negative: bool,
    modules: &mut BTreeMap<String, runtime_cuda::Module>,
    probe: &runtime_cuda::Module,
    fast: bool,
    hierarchical: bool,
) -> LabResult<()> {
    if hierarchical {
        let [_, _, op] = ops;
        return super::hierarchical::exercise(session, arena, b, planes, op, rows, label, modules);
    }
    let workspace =
        arena.output(ops.iter().map(|op| op.info().workspace).max().unwrap() as usize)?;
    let mut calls: Vec<_> = ops
        .iter()
        .map(|op| {
            op.launches(b, workspace, Counts::rows(rows))
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    if fast {
        let mut candidate = calls[2].clone();
        super::guard_fast::remap(&mut candidate);
        calls.push(candidate);
    }
    for arm in &calls {
        load_calls(session, modules, arm)?;
    }
    let mut reference = None;
    let mut old_mismatch = 0;
    for (arm, call) in calls.iter().enumerate() {
        arena.poison(&[planes[0], planes[1], planes[2], workspace])?;
        fire(session, arena, modules, call)?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let out = [
            arena.read(planes[0])?,
            arena.read(planes[1])?,
            arena.read(planes[2])?,
        ];
        if arm == 0 {
            reference = Some(out);
        } else {
            let old = reference.as_ref().unwrap();
            let differences: Vec<_> = old.iter().zip(&out).map(|(a, b)| mismatch(a, b)).collect();
            if arm >= 2 {
                assert_eq!(differences, vec![0, 0, 0], "guard versus SIMT {label}");
            } else {
                old_mismatch = differences[0];
            }
        }
    }
    drop(reference);
    if negative {
        assert!(
            old_mismatch > 0,
            "unguarded arithmetic negative control did not differ"
        );
    }
    // Diagnostic launch uses the identical guarded body with counters enabled.
    // Pack first, then score without selection; counters are outside timing.
    let mut admitted = 0;
    let mut fallback = 0;
    let mut coarse = 0;
    let mut fine = 0;
    if rows <= 32 {
        fire(session, arena, modules, &calls[2][..1])?;
        let counts = arena.input(&[0; 8])?;
        let mut call = calls[2][1].clone();
        call.symbol.push_str("_counts_v1");
        call.arguments.0.push(Argument::DeviceBuffer(counts.handle));
        let function = probe.function(&call.symbol).map_err(|e| e.to_string())?;
        kernel_harness::launch::launch(
            &session.context,
            &function,
            &session.stream,
            call.plan,
            &call.arguments,
            &kernel_harness::launch::Bindings {
                buffers: arena
                    .buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i as u64 + 1, b))
                    .collect(),
            },
        )?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let stats = arena.read(counts)?;
        admitted = u32::from_le_bytes(stats[..4].try_into().unwrap());
        fallback = u32::from_le_bytes(stats[4..].try_into().unwrap());
        assert!(fallback <= admitted);
        if fast {
            let stats = super::guard_fast::diagnose(session, arena, modules, &calls[3])?;
            assert_eq!(
                stats[..2],
                [admitted, fallback],
                "repair counts must match the original guard"
            );
            coarse = stats[2];
            fine = stats[3];
            assert!(coarse <= admitted && fine <= admitted);
            match label["case"].as_u64() {
                Some(8) => {
                    assert_eq!((coarse, fine, fallback), (0, admitted, 0));
                }
                Some(9) => {
                    assert!(coarse > 0 && coarse < admitted && fallback > 0 && fine > fallback);
                }
                Some(10) => {
                    assert_eq!((coarse, fine, fallback), (0, admitted, admitted));
                }
                _ => (),
            }
        }
        if negative {
            assert!(fallback > 0);
        }
    }
    // Repacked rows are counted independently: the wrapper handles them before
    // the guarded body, so they are deliberately absent from its slot counters.
    let layout = Workspace::new(ops[0].info().workspace, rows).unwrap();
    let scratch = arena.read(workspace)?;
    let row_fallback = scratch[layout.status as usize..layout.status as usize + rows as usize * 4]
        .chunks_exact(4)
        .filter(|x| *x != [0, 0, 0, 0])
        .count();
    if negative {
        assert_eq!(row_fallback, 0, "witness must reach the scale guard");
    }
    println!(
        "GUARD_CASE {}",
        serde_json::json!({"shape":label,"admitted":admitted,"fallback_slots":fallback,
        "fallback_rows":row_fallback,"counts_measured":rows<=32,"guard_mismatches":0,"unguarded_score_mismatches":old_mismatch,
        "negative_control_rejected":negative,"fast":fast,"coarse_slots":coarse,"fine_slots":fine})
    );
    if timing {
        let score_calls: Vec<_> = calls
            .iter()
            .enumerate()
            .map(|(arm, c)| &c[..if arm == 0 { 1 } else { 2 }])
            .collect();
        let graphs = score_calls
            .iter()
            .map(|c| session.capture(|_| fire(session, arena, modules, c)))
            .collect::<LabResult<Vec<_>>>()?;
        // One stream, only the pack and score sequence enclosed by events.
        // This measures exclusive wall time, including any PDL overlap once.
        let pairs: &[[usize; 2]] = if fast {
            &[[2, 3], [1, 3], [0, 3]]
        } else {
            &[[0, 2], [1, 2]]
        };
        for &pair in pairs {
            let measured =
                kernel_harness::with_timing_label(&format!("guard-{label}-{pair:?}"), || {
                    kernel_harness::time_ab(
                        session,
                        2,
                        5,
                        [EvictionMode::Write; 2],
                        |arm, stream| graphs[pair[arm]].launch(stream).map_err(|e| e.to_string()),
                    )
                })?;
            println!(
                "GUARD_TIMING {}",
                serde_json::json!({"shape":label,"arms":pair,
                "median_ms":measured.arms.map(|a|a.median_ms),"paired_ratio":measured.paired_ratio,"exclusive":true})
            );
        }
    }
    session.receipt(
        "indexer-guard-cost",
        BTreeMap::from([
            ("shape".into(), label),
            ("fast".into(), serde_json::json!(fast)),
            ("mismatches".into(), serde_json::json!(0)),
        ]),
    )?;
    Ok(())
}
fn test_cubins() -> Cubins {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_INDEX_GUARD_ARTIFACTS").expect("staged guard artifacts"),
    );
    let data = Box::leak(
        bytes(&root.join("probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    Cubins::new(
        Box::leak(Box::new([("probe_sm120", &data[..])])),
        &[(
            "probe_sm120",
            &[
                "chop_probe_gate_v1",
                "chop_probe_flush_v1",
                "chop_probe_read_v1",
            ],
        )],
    )
}
fn probe(session: &Session) -> LabResult<runtime_cuda::Module> {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_INDEX_GUARD_ARTIFACTS").ok_or("staged artifacts required")?,
    );
    session.load_cubin(&bytes(
        &root.join("position_mma_guard_deepseek41_sm120_counts.cubin"),
    )?)
}
fn backends() -> [Sm120; 3] {
    [super::backend(false), super::backend(true), guarded(true)]
}

#[test]
#[ignore = "requires sealed captures and CHOPINFER_INDEX_GUARD_ARTIFACTS on idle GPUs"]
fn gpu_real_guard_frequency_and_cost() {
    real(false);
}
pub(super) fn real(fast: bool) {
    real_variant(fast, false);
}
pub(super) fn real_variant(fast: bool, hierarchical: bool) {
    let ran=kernel_harness::gpu_test_with(test_cubins(),3<<30,|session| {
        let manifest=PathBuf::from(std::env::var_os("CHOPINFER_INDEX_CAPTURE_MANIFEST").ok_or("capture manifest required")?);
        let data=read_json(&manifest)?;
        let shard:usize=std::env::var("CHOPINFER_L4_SHARD").unwrap().parse().unwrap();assert!(shard<2);
        let mut modules=BTreeMap::new();let probe=probe(session)?;
        for (i,capture) in data["captures"].as_array().unwrap().iter().enumerate().filter(|(i,_)|i%2==shard) {
            let dir=Path::new(capture["path"].as_str().unwrap());let meta=read_json(&dir.join("operands.json"))?;
            let width=meta["width"].as_u64().unwrap() as u32;
            let page=meta["page_size"].as_u64().unwrap() as u32;
            let layer=meta["layer"].as_u64().unwrap() as u32;
            let compact=meta["reindex"].as_bool().unwrap();
            let row_counts: &[u32]=if fast { &[1,2,4,6,8,16,32] } else { &[32] };
            for &rows in row_counts {
            let mut arena=Arena {session,buffers:vec![]};
            let b=PositionSelectBuffers {
                q:arena.input(&bytes(&dir.join("query.bf16"))?)?,head_weights:arena.input(&bytes(&dir.join("weights.f32"))?)?,
                keys:PositionKeyPages {codes:arena.input(&bytes(&dir.join("keys.e2m1"))?)?,scales:arena.input(&bytes(&dir.join("scales.ue8m0"))?)?,
                    pages:meta["physical_pages"].as_array().unwrap().len() as u32,code_page_stride:u64::from(page)*64,scale_page_stride:u64::from(page)*4,
                    page_tables:BlockTables {tables:arena.input(&ints(values(&meta,"page_table")))?,sequences:1,max_pages:width/page}},
                positions:arena.input(&ints(values(&meta,"positions")))?,row_batch:arena.input(&ints(vec![0;32]))?,live_rows:arena.input(&rows.to_le_bytes())?,
                candidate_mask:None,position_capacity:width,scores:arena.output(32*if compact{16384}else{width as usize}*4)?,status:arena.output(128)?,output:arena.output(32*512*4)?,
            };
            let label=serde_json::json!({"capture":i,"context":capture["context"],"layer":layer,"rows":rows,"width":width,"compact":compact});
            let planes=[b.scores,b.status,b.output];
            if compact {
                let b=PositionBlockSelectBuffers {q:b.q,head_weights:b.head_weights,keys:b.keys,positions:b.positions,row_batch:b.row_batch,live_rows:b.live_rows,
                    row_tags:arena.input(&bytes(&dir.join("row_tags.u64"))?)?,
                    candidates:CandidateBlockIds {ids:arena.input(&bytes(&dir.join("candidates.i32"))?)?,row_tags:arena.input(&bytes(&dir.join("candidate_tags.u64"))?)?,
                        visible_positions:arena.input(&bytes(&dir.join("candidate_visible.i32"))?)?,status:arena.input(&bytes(&dir.join("candidate_status.u32"))?)?,position_capacity:width},
                    scores:b.scores,status:b.status,output:b.output};
                let query=indexer::block_selection(&FLASH,layer,rows).unwrap();
                let ops=backends().map(|backend|backend.position_block_indexer(&query,512,width,Need::rows(rows)).unwrap());
                exercise(session,&mut arena,&b,planes,ops,rows,label,true,false,&mut modules,&probe,fast,hierarchical)?;
            } else {
                let query=indexer::selection(&FLASH,layer,rows).unwrap();
                let ops=backends().map(|backend|backend.position_indexer(&query,512,width,Need::rows(rows)).unwrap());
                exercise(session,&mut arena,&b,planes,ops,rows,label,true,false,&mut modules,&probe,fast,hierarchical)?;
            }
        }
        }
        println!("GUARD_REAL_COMPLETE shard={shard} captures={}",if fast {105}else{15});Ok(())
    }).unwrap();
    assert!(ran.is_some());
}

fn matrix_shapes() -> Vec<(u32, u32, u32, u32)> {
    let mut shapes = vec![];
    for layer in [2, 20, 24] {
        for rows in [1, 2, 4, 6, 8, 16, 32] {
            for offset in [8192, 32768, 131072] {
                shapes.push((layer, rows, offset, 0));
            }
        }
        for offset in [0, 32768, 131072] {
            shapes.push((layer, 8192, offset, 0));
        }
        for case in 1..8 {
            shapes.push((layer, 6, 1024, case));
        }
    }
    shapes
}

fn suite_shapes(fast: bool) -> Vec<(u32, u32, u32, u32)> {
    let mut shapes = matrix_shapes();
    if fast {
        for layer in [2, 20, 24] {
            for case in 8..=10 {
                shapes.push((layer, 6, 1024, case));
            }
        }
    }
    shapes
}

// CPU admission and GPU execution use the same metadata and allocations.
fn compact_matrix_buffers(
    b: &PositionSelectBuffers<View>,
    visible: Vec<i32>,
    mut input: impl FnMut(&[u8]) -> LabResult<View>,
) -> LabResult<PositionBlockSelectBuffers<View>> {
    let rows = visible.len();
    let mut ids = vec![-1; rows * 2048];
    for (row, count) in visible.iter().enumerate() {
        let blocks = (*count as usize).div_ceil(8);
        let kept = blocks.min(2048);
        for i in 0..kept {
            ids[row * 2048 + i] = if i + 1 == kept {
                (blocks - 1) as i32
            } else {
                i as i32
            };
        }
    }
    let tags = input(&vec![0; rows * 8])?;
    let retained_tags = input(&vec![0; rows * 8])?;
    Ok(PositionBlockSelectBuffers {
        q: b.q,
        head_weights: b.head_weights,
        keys: b.keys,
        positions: b.positions,
        row_batch: b.row_batch,
        live_rows: b.live_rows,
        row_tags: tags,
        candidates: CandidateBlockIds {
            ids: input(&ints(ids))?,
            row_tags: retained_tags,
            visible_positions: input(&ints(visible))?,
            status: input(&vec![0; rows * 4])?,
            position_capacity: b.position_capacity,
        },
        scores: b.scores,
        status: b.status,
        output: b.output,
    })
}

#[test]
fn cpu_compact_matrix_uses_distinct_tags() {
    compact_matrix_cpu(false);
}
pub(super) fn compact_matrix_cpu(fast: bool) {
    let mut cases = 0;
    for (layer, rows, offset, case) in suite_shapes(fast).into_iter().filter(|s| s.0 == 24) {
        let ratio = FLASH.compress_ratio(layer);
        let page = 64 / ratio;
        let width = ((offset + rows).div_ceil(64) * 64 / ratio).max(page);
        let mut dense = cpu_buffers(rows, ratio, width);
        dense.scores = v(9, u64::from(rows) * 16384 * 4);
        let visible = (0..rows)
            .map(|i| ((offset + i + 1) / ratio) as i32)
            .collect();
        let mut handle = 100;
        let b = compact_matrix_buffers(&dense, visible, |bytes| {
            handle += 1;
            Ok(v(handle, bytes.len() as u64))
        })
        .unwrap();
        let query = indexer::block_selection(&FLASH, layer, rows).unwrap();
        for backend in backends() {
            let op = backend
                .position_block_indexer(&query, 512, width, Need::rows(rows))
                .unwrap();
            let scratch = v(99, op.info().workspace);
            op.launches(&b, scratch, Counts::rows(rows))
                .unwrap_or_else(|error| {
                    panic!("compact rows={rows} offset={offset} case={case}: {error}");
                });
            let mut aliased = b;
            aliased.candidates.row_tags = aliased.row_tags;
            assert_eq!(
                op.launches(&aliased, scratch, Counts::rows(rows))
                    .unwrap_err()
                    .0,
                "current compact row tags must be distinct from retained tags"
            );
        }
        assert_ne!(b.row_tags.handle, b.candidates.row_tags.handle);
        cases += 1;
    }
    assert_eq!(cases, if fast { 34 } else { 31 });
}

fn bf16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
#[test]
#[ignore = "requires CHOPINFER_INDEX_GUARD_ARTIFACTS and idle GPUs"]
fn gpu_decode_prefill_and_exponent_edges() {
    matrix(false);
}
pub(super) fn matrix(fast: bool) {
    matrix_variant(fast, false);
}
pub(super) fn matrix_variant(fast: bool, hierarchical: bool) {
    let ran=kernel_harness::gpu_test_with(test_cubins(),12<<30,|session| {
        let shard:usize=std::env::var("CHOPINFER_L4_SHARD").unwrap().parse().unwrap();assert!(shard<2);
        let mut shapes=suite_shapes(fast);
        if hierarchical {
            for layer in [2,20,24] {
                for rows in [1,32] { shapes.push((layer,rows,1024,0)); }
                for case in 20..=32 { shapes.push((layer,6,131072,case)); }
            }
        }
        let total=shapes.len();let mut completed=0;
        let mut modules=BTreeMap::new();let probe=probe(session)?;
        for (shape,(layer,rows,offset,case)) in shapes.into_iter().enumerate().filter(|(i,_)|i%2==shard) {
            let ratio=FLASH.compress_ratio(layer);let page=64/ratio;
            let width=((offset+rows).div_ceil(64)*64/ratio).max(page);
            let compact=layer==24;let extent=if compact{16384}else{width};
            let mut arena=Arena {session,buffers:vec![]};
            let mut seed=0xa76204_u32;let mut rng=||{seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;seed};
            let mut q=Vec::with_capacity(rows as usize*4096);
            let magnitude=[0.0,0.5,1.,1.5,2.,3.,4.,6.];
            for _ in 0..rows {for _ in 0..32 {for group in 0..4 {for lane in 0..32 {
                let r=rng();let mut x=magnitude[(r&7) as usize]*if r&8==0 {1.}else{-1.};
                if lane==31 {x=6.;} // Make every group's fake quantization scale exact.
                let exponent=match case {2=>[0,24,0,-24][group],4=>-126,5=>120,8=>[0,12,0,12][group],_=>group as i32-2};
                q.push(x*2_f32.powi(exponent));
            }}}}
            let mut w=vec![0.03125_f32;rows as usize*32];
            let mut codes:Vec<_>=(0..width as usize*64).map(|_|rng() as u8).collect();
            let mut scales:Vec<_>=(0..width as usize*4).map(|i|125+(i%4) as u8).collect();
            if case==1 {
                q.fill(0.);w.fill(0.);codes.fill(0x22);scales.fill(127);
                for row in 0..rows as usize {q[row*4096]=2_f32.powi(25);q[row*4096+33]=1.;q[row*4096+64] = -2_f32.powi(25);w[row*32]=1.;}
            }
            if case==3 {for (i,s) in scales.iter_mut().enumerate(){*s=[0,1,253,255][i%4];}}
            if case==5 {scales.fill(247);}
            if case==6 {q[0]=f32::from_bits(0x3f810000);}
            if case==7 {w[0]=f32::INFINITY;w[32]=f32::NAN;}
            if case==8 {for (i,s) in scales.iter_mut().enumerate(){*s=[139,127,139,127][i%4];}}
            if case==9 {for slot in (0..width as usize).step_by(32) {scales[slot*4]=151;}}
            if case==10 {for row in 0..rows as usize {for g in 0..4 {for lane in 0..32 {
                q[row*4096+31*128+g*32+lane]=6.0*2_f32.powi([0,24,0,-24][g]);
            }}}}
            let positions:Vec<i32>=(0..rows).map(|i|if hierarchical && case==30 {-1} else if hierarchical && case==31 {0} else {(offset+i) as i32}).collect();
            let top_k=if hierarchical {super::hierarchical::top_k(case)}else{512};
            let b=PositionSelectBuffers {
                q:arena.input(&bf16_bytes(q))?,head_weights:arena.input(&w.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())?,
                keys:PositionKeyPages {codes:arena.input(&codes)?,scales:arena.input(&scales)?,pages:width/page,
                    code_page_stride:u64::from(page)*64,scale_page_stride:u64::from(page)*4,
                    page_tables:BlockTables {tables:arena.input(&ints((0..width/page).map(|i|if hierarchical && case==29 && i%7==0 {-1}else{i as i32})))?,sequences:1,max_pages:width/page}},
                positions:arena.input(&ints(positions.clone()))?,row_batch:arena.input(&ints(vec![0;rows as usize]))?,live_rows:arena.input(&rows.to_le_bytes())?,
                candidate_mask:None,position_capacity:width,scores:arena.output(rows as usize*extent as usize*4)?,status:arena.output(rows as usize*4)?,output:arena.output(rows as usize*512*4)?,
            };
            let planes=[b.scores,b.status,b.output];
            let label=serde_json::json!({"matrix":shape,"rows":rows,"offset":offset,"layer":layer,"ratio":ratio,"case":case,"width":width,"top_k":top_k});
            if compact {
                let visible:Vec<_>=positions.iter().map(|p|(p+1)/ratio as i32).collect();
                let b=compact_matrix_buffers(&b,visible,|bytes|arena.input(bytes))?;
                let query=indexer::block_selection(&FLASH,layer,rows).unwrap();
                let ops=backends().map(|backend|backend.position_block_indexer(&query,top_k,width,Need::rows(rows)).unwrap());
                exercise(session,&mut arena,&b,planes,ops,rows,label,case==0,case==1,&mut modules,&probe,fast,hierarchical)?;
            } else {
                let query=indexer::selection(&FLASH,layer,rows).unwrap();
                let ops=backends().map(|backend|backend.position_indexer(&query,top_k,width,Need::rows(rows)).unwrap());
                exercise(session,&mut arena,&b,planes,ops,rows,label,case==0,case==1,&mut modules,&probe,fast,hierarchical)?;
            }
            completed+=1;
        }
        println!("GUARD_MATRIX_COMPLETE shard={shard} shapes={completed} total={total}");Ok(())
    }).unwrap();
    assert!(ran.is_some());
}
