//! Exact-product domain edges and accumulator-width probes for a later pair.
use super::*;

#[derive(Clone, Copy, Debug)]
struct Edge {
    layer: u32,
    lo: i32,
    spread: i32,
    factor: u32,
    pattern: u32,
}
fn cases() -> Vec<Edge> {
    let mut result = Vec::new();
    for layer in [2, 20, 24] {
        for spread in [7, 8, 9, 10] {
            for lo in [129, 130, 131, 254, 368 - spread, 369 - spread, 370 - spread] {
                for factor in 0..3 {
                    for pattern in 0..3 {
                        result.push(Edge {
                            layer,
                            lo,
                            spread,
                            factor,
                            pattern,
                        });
                    }
                }
            }
        }
        for lo in [130, 131, 254, 368, 369] {
            for factor in 0..3 {
                result.push(Edge {
                    layer,
                    lo,
                    spread: 0,
                    factor,
                    pattern: 3,
                });
            }
        }
    }
    assert_eq!(result.len(), 801);
    result
}
fn scales(c: Edge) -> ([u8; 4], [u8; 4]) {
    let sums = [c.lo, c.lo + c.spread, c.lo + c.spread, c.lo + c.spread];
    let q = std::array::from_fn(|g| {
        let s = sums[g];
        (match c.factor {
            0 => s / 2,
            1 => s / 2 + if g == 0 { 6 } else { -6 },
            _ => {
                if c.lo + c.spread <= 254 {
                    2
                } else {
                    252
                }
            }
        }) as u8
    });
    let k = std::array::from_fn(|g| (sums[g] - i32::from(q[g])) as u8);
    assert!(q.iter().chain(&k).all(|&x| (2..=252).contains(&x)));
    (q, k)
}
fn decode(code: u8, scale: u8) -> f32 {
    let magnitude = [0., 1., 2., 3., 4., 6., 8., 12.][usize::from(code & 7)];
    (if code & 8 == 0 { magnitude } else { -magnitude }) * 2_f32.powi(i32::from(scale) - 128)
}
fn coefficients(c: Edge, row: usize, slot: usize) -> ([u8; 128], [u8; 128]) {
    let mut q = [7; 128];
    let mut k = [7; 128];
    match c.pattern {
        0 => {
            q[0] = 3;
            k[0] = 3;
        }
        1 => {
            k.fill(0);
            q[0] = 3;
            k[0] = 3;
            k[32..64].fill(7);
            k[64..96].fill(15);
        }
        2 => {
            q[0] = 1;
            k[0] = 1;
        }
        3 => {
            k.fill(0);
            q[0] = row as u8;
            k[0] = (slot % 16) as u8;
            k[64] = 7;
        }
        _ => unreachable!(),
    }
    (q, k)
}
fn expected(c: Edge, row: usize, slot: usize) -> (f32, bool) {
    let (qs, ks) = scales(c);
    let (q, k) = coefficients(c, row, slot);
    let mut value = 0_f64;
    let mut magnitude = 0_f64;
    for i in 0..128 {
        let product = f64::from(decode(q[i], qs[i / 32])) * f64::from(decode(k[i], ks[i / 32]));
        value += product;
        magnitude += product.abs();
    }
    let unit = 2_f64.powi(c.lo - 256);
    let exact = c.lo >= 130 && c.lo + c.spread <= 369 && magnitude / unit < (1_u64 << 24) as f64;
    let value = value.max(0.) as f32;
    (value, exact)
}
fn cpu_dot(c: Edge, reduced_precision: bool) -> f32 {
    let (qs, ks) = scales(c);
    let (q, k) = coefficients(c, 0, 0);
    if reduced_precision {
        let mut sum = 0_f32;
        for i in 0..128 {
            sum += decode(q[i], qs[i / 32]) * decode(k[i], ks[i / 32]);
            sum = f32::from_bits(sum.to_bits() & !1);
        }
        return sum.max(0.);
    }
    let mut lanes = [0_f32; 32];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        for g in 0..4 {
            let i = g * 32 + lane;
            *sum = decode(q[i], qs[g]).mul_add(decode(k[i], ks[g]), *sum);
        }
    }
    for offset in [16, 8, 4, 2, 1] {
        let old = lanes;
        for lane in 0..32 - offset {
            lanes[lane] += old[lane + offset];
        }
    }
    (0. + lanes[0]).max(0.)
}
fn accepted(c: Edge) -> bool {
    c.lo >= 130 && c.lo + c.spread <= 369 && c.spread <= 9
}
fn rows(c: Edge) -> u32 {
    if c.pattern == 3 { 16 } else { 6 }
}

#[test]
fn cpu_edges_preserve_quantizer_scales_and_expose_lost_low_bits() {
    let mut probes = 0;
    for c in cases() {
        let (qs, ks) = scales(c);
        for row in 0..rows(c) as usize {
            let (q, k) = coefficients(c, row, row % 16);
            for g in 0..4 {
                assert_eq!(q[g * 32 + 31], 7, "each group pins the exact repack scale");
                for i in g * 32..g * 32 + 32 {
                    for value in [decode(q[i], qs[g]), decode(k[i], ks[g])] {
                        assert!(value.is_finite());
                        assert_eq!(value.to_bits() & 0xffff, 0, "operands are exact BF16");
                    }
                }
            }
        }
        let (value, exact) = expected(c, 0, 0);
        if exact {
            assert_eq!(cpu_dot(c, false).to_bits(), value.to_bits());
        }
        if c.spread == 10 && c.pattern == 0 && exact {
            assert_ne!(value.to_bits() & 1, 0);
            assert_ne!(
                value.to_bits(),
                cpu_dot(c, true).to_bits(),
                "23-bit accumulation control must differ"
            );
            probes += 1;
        }
        if accepted(c) {
            assert!(exact);
        }
        let query = indexer::selection(&FLASH, c.layer, rows(c)).unwrap();
        assert_eq!(
            kernel_core::position_indexer_oracle::weight_scale(query.selection()).unwrap() * 64.,
            1.
        );
        let mut dense = cpu_buffers(rows(c), FLASH.compress_ratio(c.layer), 64);
        if c.layer == 24 {
            dense.scores = v(9, u64::from(rows(c)) * 16384 * 4);
            let mut next = 100;
            let b = compact_matrix_buffers(&dense, vec![64; rows(c) as usize], |bytes| {
                next += 1;
                Ok(v(next, bytes.len() as u64))
            })
            .unwrap();
            let query = indexer::block_selection(&FLASH, c.layer, rows(c)).unwrap();
            for backend in backends() {
                let op = backend
                    .position_block_indexer(&query, 512, 64, Need::rows(rows(c)))
                    .unwrap();
                op.launches(&b, v(99, op.info().workspace), Counts::rows(rows(c)))
                    .unwrap();
            }
        }
    }
    assert_eq!(probes, 45);
}

fn fixture(arena: &mut Arena<'_>, c: Edge) -> LabResult<PositionSelectBuffers<View>> {
    let n = rows(c);
    let page = 64 / FLASH.compress_ratio(c.layer);
    let (qs, ks) = scales(c);
    let mut q = Vec::new();
    let mut w = vec![0_f32; n as usize * 32];
    for row in 0..n as usize {
        let (values, _) = coefficients(c, row, 0);
        for _ in 0..32 {
            q.extend((0..128).map(|i| decode(values[i], qs[i / 32])));
        }
        w[row * 32] = 64.;
    }
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    for slot in 0..64 {
        let (_, values) = coefficients(c, 0, slot);
        codes.extend(values.chunks_exact(2).map(|pair| pair[0] | pair[1] << 4));
        scales.extend(ks);
    }
    Ok(PositionSelectBuffers {
        q: arena.input(&bf16_bytes(q))?,
        head_weights: arena.input(&w.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())?,
        keys: PositionKeyPages {
            codes: arena.input(&codes)?,
            scales: arena.input(&scales)?,
            pages: 64 / page,
            code_page_stride: u64::from(page) * 64,
            scale_page_stride: u64::from(page) * 4,
            page_tables: BlockTables {
                tables: arena.input(&ints((0..64 / page).map(|p| p as i32)))?,
                sequences: 1,
                max_pages: 64 / page,
            },
        },
        positions: arena.input(&ints(vec![
            (64 * FLASH.compress_ratio(c.layer) - 1) as i32;
            n as usize
        ]))?,
        row_batch: arena.input(&ints(vec![0; n as usize]))?,
        live_rows: arena.input(&n.to_le_bytes())?,
        candidate_mask: None,
        position_capacity: 64,
        scores: arena.output(n as usize * if c.layer == 24 { 16384 * 4 } else { 64 * 4 })?,
        status: arena.output(n as usize * 4)?,
        output: arena.output(n as usize * 512 * 4)?,
    })
}

fn exercise_edge<B>(
    session: &Session,
    arena: &mut Arena<'_>,
    c: Edge,
    b: &B,
    planes: [View; 3],
    ops: [Prepared<B>; 3],
    modules: &mut BTreeMap<String, runtime_cuda::Module>,
) -> LabResult<usize> {
    let n = rows(c);
    let scratch = arena.output(ops.iter().map(|op| op.info().workspace).max().unwrap() as usize)?;
    let mut calls = ops
        .iter()
        .map(|op| {
            op.launches(b, scratch, Counts::rows(n))
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let mut coarse = calls[2].clone();
    super::super::guard_fast::remap(&mut coarse);
    calls.push(coarse);
    for arm in &calls {
        load_calls(session, modules, arm)?;
    }
    let mut raw_reference = None;
    let mut final_reference = None;
    let mut native_checks = 0;
    for (arm, call) in calls.iter().enumerate() {
        arena.poison(&[planes[0], planes[1], planes[2], scratch])?;
        let score_end = if arm == 0 { 1 } else { 2 };
        fire(session, arena, modules, &call[..score_end])?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let raw = arena.read(planes[0])?;
        if arm == 0 {
            raw_reference = Some(raw.clone());
        }
        if arm >= 2 {
            assert_eq!(
                raw,
                raw_reference.as_ref().unwrap().as_slice(),
                "raw guard edge {c:?}"
            );
        }
        if arm == 1 {
            let stride = if c.layer == 24 { 16384 } else { 64 };
            for row in 0..n as usize {
                for slot in 0..64 {
                    let (oracle, exact) = expected(c, row, slot);
                    if exact {
                        let at = (row * stride + slot) * 4;
                        assert_eq!(
                            u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()),
                            oracle.to_bits(),
                            "native exact-product premise {c:?} row={row} slot={slot}"
                        );
                        native_checks += 1;
                    }
                }
            }
        }
        fire(session, arena, modules, &call[score_end..])?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let output = [
            arena.read(planes[0])?,
            arena.read(planes[1])?,
            arena.read(planes[2])?,
        ];
        if arm == 0 {
            final_reference = Some(output);
        } else if arm >= 2 {
            assert_eq!(
                output,
                final_reference.as_ref().unwrap().clone(),
                "guard selection edge {c:?}"
            );
        }
    }
    let counts = super::super::guard_fast::diagnose(session, arena, modules, &calls[3])?;
    assert_eq!(counts[0], n * 64);
    assert_eq!(
        counts[1],
        if accepted(c) { 0 } else { n * 64 },
        "guard boundary {c:?}"
    );
    if c.factor == 1 && accepted(c) {
        assert_eq!(counts[2], 0, "fine-only fixture {c:?}");
    }
    let layout = Workspace::new(ops[0].info().workspace, n).unwrap();
    let bytes = arena.read(scratch)?;
    assert!(
        bytes[layout.status as usize..layout.status as usize + n as usize * 4]
            .iter()
            .all(|&x| x == 0),
        "query row fallback hides the edge {c:?}"
    );
    Ok(native_checks)
}

#[test]
#[ignore = "requires CHOPINFER_INDEX_GUARD_ARTIFACTS and inclusion in a later idle-GPU pair"]
fn gpu_domain_edges_and_native_accumulation() {
    let ran=kernel_harness::gpu_test_with(test_cubins(),1<<30,|session| {
        let shard:usize=std::env::var("CHOPINFER_L4_SHARD").unwrap().parse().unwrap();assert!(shard<2);
        let mut modules=BTreeMap::new();let mut completed=0;let mut checked=0;
        for (id,c) in cases().into_iter().enumerate().filter(|(i,_)|i%2==shard) {
            let mut arena=Arena {session,buffers:vec![]};let b=fixture(&mut arena,c)?;let planes=[b.scores,b.status,b.output];
            let checks=if c.layer==24 {
                let b=compact_matrix_buffers(&b,vec![64;rows(c) as usize],|bytes|arena.input(bytes))?;
                let query=indexer::block_selection(&FLASH,c.layer,rows(c)).unwrap();
                let ops=backends().map(|backend|backend.position_block_indexer(&query,512,64,Need::rows(rows(c))).unwrap());
                exercise_edge(session,&mut arena,c,&b,planes,ops,&mut modules)?
            } else {
                let query=indexer::selection(&FLASH,c.layer,rows(c)).unwrap();
                let ops=backends().map(|backend|backend.position_indexer(&query,512,64,Need::rows(rows(c))).unwrap());
                exercise_edge(session,&mut arena,c,&b,planes,ops,&mut modules)?
            };
            println!("GUARD_EDGE_CASE {}",serde_json::json!({"id":id,"layer":c.layer,"lo":c.lo,"spread":c.spread,"factor":c.factor,"pattern":c.pattern,"accepted":accepted(c),"raw_comparisons":2,"selection_comparisons":2,"native_oracle_checks":checks,"mismatches":0}));
            completed+=1;checked+=checks;
        }
        assert_eq!(completed,if shard==0 {401}else{400});
        println!("GUARD_EDGES_COMPLETE shard={shard} cases={completed} total=801 native_oracle_checks={checked}");
        Ok(())
    }).unwrap();
    assert!(ran.is_some());
}
