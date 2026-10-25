//! The complete HC, input norm, WQ_a, query norm and WQ_b PDL chain.
//! Both HC producers use existing serving cubins. The chain producer is
//! OSR's admitted small-row form; the baseline keeps main's producer.
use super::*;
use kernels_sm120::gemm::{mxfp8_fold_cluster_fused, mxfp8_fold_local_wide};
const H: usize = 5120;
const GUARD: usize = 64;

pub(super) fn symbols() -> Vec<String> {
    let hc = sm120::hc_round_expand_mix::entry(&FLASH);
    hc.symbols()
        .into_iter()
        .chain([
            sm120::hc_round_chain::entry(&FLASH).symbols()[1].clone(),
            sm120::fold_fused::fused("attention.q-a", 2)
                .unwrap()
                .packed_spec()
                .unwrap()
                .symbol,
            sm120::fold_fused::wide(2)
                .unwrap()
                .packed_spec()
                .unwrap()
                .symbol,
        ])
        .collect()
}
pub(super) fn dynamic_shared(symbol: &str) -> u32 {
    let qa = sm120::fold_fused::fused("attention.q-a", 2).unwrap();
    let qb = sm120::fold_fused::wide(2).unwrap();
    if symbol == qa.packed_spec().unwrap().symbol {
        qa.plan(2).unwrap().dynamic_shared_bytes
    } else if symbol == qb.packed_spec().unwrap().symbol {
        qb.plan(2).unwrap().dynamic_shared_bytes
    } else {
        0
    }
}
fn random(n: usize, seed: u64, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let x = (i as u64)
                .wrapping_mul(747796405)
                .wrapping_add(seed)
                .rotate_left(11);
            ((x % 65521) as f32 / 32760. - 1.)
                .mul_add(scale, 0.)
                .to_ne_bytes()
        })
        .collect()
}
fn buffer(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let b = s.allocate(bytes.len())?;
    upload(&b, bytes, &s.stream)?;
    Ok(b)
}
struct Inputs {
    hc: Vec<DeviceBuffer>,
    input: Input,
    query: Input,
    qa_weights: DeviceBuffer,
    qa_scales: DeviceBuffer,
    qb_weights: DeviceBuffer,
    qb_scales: DeviceBuffer,
}
impl Inputs {
    fn new(s: &Session, rows: u32, rank: u64) -> LabResult<Self> {
        let f = random(rows as usize * 4 * H, 123 + rank, 4.);
        let streams = f
            .chunks_exact(4)
            .flat_map(|v| ((u32::from_ne_bytes(v.try_into().unwrap()) >> 16) as u16).to_ne_bytes())
            .collect::<Vec<_>>();
        let hc = [
            streams,
            random(rows as usize * H, 71 + rank, 2.),
            random(rows as usize * 4, 31 + rank, 1.),
            random(rows as usize * 16, 47 + rank, 0.5),
            random(24 * 4 * H, 853 + rank, 0.02),
            random(24, 61 + rank, 1.),
            [0.5f32, 0.25, 1.]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
            random(rows as usize * 4, 13 + rank, 1.),
        ]
        .iter()
        .map(|v| buffer(s, v))
        .collect::<LabResult<Vec<_>>>()?;
        let live = buffer(s, &rows.to_ne_bytes())?;
        let input = Input {
            values: s.allocate(4)?,
            weight: buffer(s, &data::weight_of(H, "finite", 19 + rank))?,
            live: live.clone(),
        };
        let query = Input {
            values: s.allocate(4)?,
            weight: buffer(s, &data::weight_of(1280, "finite", 37 + rank))?,
            live,
        };
        let weights = |n: usize, k: usize| {
            (0..n * k)
                .map(|i| {
                    (8 + ((i as u64 * 73 + rank * 17) % 80) as u8) | (((i >> 4) & 1) as u8 * 128)
                })
                .collect::<Vec<_>>()
        };
        let scales = |n: usize, k: usize| {
            (0..n * (k / 32))
                .map(|i| 119 + ((i as u64 + rank) % 5) as u8)
                .collect::<Vec<_>>()
        };
        Ok(Self {
            hc,
            input,
            query,
            qa_weights: buffer(s, &weights(1280, 5120))?,
            qa_scales: buffer(s, &scales(1280, 5120))?,
            qb_weights: buffer(s, &weights(8192, 1280))?,
            qb_scales: buffer(s, &scales(8192, 1280))?,
        })
    }
}
struct Frame {
    hc: Vec<DeviceBuffer>,
    scratch: DeviceBuffer,
    input: Output,
    query: Output,
    qa: DeviceBuffer,
    qb: DeviceBuffer,
    work: DeviceBuffer,
}
impl Frame {
    fn new(s: &Session, rows: u32, qa: &mxfp8_fold_cluster_fused::Entry) -> LabResult<Self> {
        let hc = [H * 2, 4 * H * 2, H * 2, 4 * 4, 4 * 4, 16 * 4]
            .into_iter()
            .map(|n| s.allocate(n * rows as usize + GUARD))
            .collect::<LabResult<_>>()?;
        let input = Output::new(s, &cases()[1], rows)?;
        let mut query = Output::new(s, &cases()[2], rows)?;
        query.buffers[4] = input.buffers[4].clone();
        Ok(Self {
            hc,
            scratch: s.allocate(rows as usize * 4096 * 4 + GUARD)?,
            input,
            query,
            qa: s.allocate(rows as usize * 1280 * 4 + GUARD)?,
            qb: s.allocate(rows as usize * 8192 * 4 + GUARD)?,
            work: s.allocate(qa.workspace() as usize + GUARD)?,
        })
    }
    fn poison(&self, s: &Session, rows: u32, sticky: bool) -> LabResult<()> {
        for b in self
            .hc
            .iter()
            .chain([&self.scratch, &self.qa, &self.qb, &self.work])
        {
            upload(b, &vec![0xa5; b.byte_len()], &s.stream)?;
        }
        self.input.poison(s, rows, sticky)?;
        self.query.poison(s, rows, sticky)
    }
    fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        let mut values = self.input.read(s)?;
        values.extend(self.query.read(s)?);
        for b in self
            .hc
            .iter()
            .chain([&self.scratch, &self.qa, &self.qb, &self.work])
        {
            let mut v = download(b, b.byte_len(), &s.stream)?;
            if !v[v.len() - GUARD..].iter().all(|b| *b == 0xa5) {
                return Err("context guard changed".into());
            }
            v.truncate(v.len() - GUARD);
            values.push(v);
        }
        Ok(values)
    }
}
struct Prelude {
    qa: mxfp8_fold_cluster_fused::Entry,
    qb: mxfp8_fold_local_wide::Entry,
    rows: u32,
    chain: bool,
}
impl Prelude {
    fn run(
        &self,
        k: &Loaded,
        i: &Inputs,
        o: &Frame,
        arms: (usize, usize),
        stream: &Stream,
        pdl: bool,
    ) -> LabResult<()> {
        let hc = sm120::hc_round_expand_mix::entry(&FLASH);
        let chain = sm120::hc_round_chain::entry(&FLASH);
        let (symbol, mut plan) = if self.chain {
            (
                chain.symbols()[1].clone(),
                chain
                    .plan(self.rows, self.rows, 8)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            (
                hc.symbols()[0].clone(),
                hc.plans(self.rows).map_err(|e| e.to_string())?[0],
            )
        };
        plan.pdl = pdl;
        let first = [
            A::Buffer(&i.hc[0]),
            A::Buffer(&i.hc[1]),
            A::Buffer(&o.hc[0]),
            A::Buffer(&i.hc[2]),
            A::Buffer(&i.hc[3]),
            A::Buffer(&o.hc[1]),
            A::Buffer(&i.hc[4]),
            A::Buffer(&o.scratch),
            A::Buffer(&i.input.live),
        ];
        k.functions[&symbol]
            .launch(stream, launch::config(plan)?, &first)
            .map_err(|e| e.to_string())?;
        let mut plan = hc.plans(self.rows).map_err(|e| e.to_string())?[1];
        plan.pdl = pdl;
        let finish = [
            A::Buffer(&o.hc[1]),
            A::Buffer(&i.hc[5]),
            A::Buffer(&i.hc[6]),
            A::Buffer(&i.hc[7]),
            A::Buffer(&o.hc[2]),
            A::Buffer(&o.hc[3]),
            A::Buffer(&o.hc[4]),
            A::Buffer(&o.hc[5]),
            A::Buffer(&o.scratch),
            A::Buffer(&i.input.live),
        ];
        k.functions[&hc.symbols()[1]]
            .launch(stream, launch::config(plan)?, &finish)
            .map_err(|e| e.to_string())?;
        call(
            &cases()[1],
            k,
            arms.0,
            self.rows,
            &i.input,
            &o.hc[2],
            &o.input,
            stream,
            pdl,
        )?;
        let mut plan = self.qa.plan(self.rows).map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        let qa = [
            A::Buffer(&o.input.buffers[2]),
            A::Buffer(&o.input.buffers[3]),
            A::Buffer(&i.qa_weights),
            A::Buffer(&i.qa_scales),
            A::Buffer(&o.qa),
            A::NullBuffer,
            A::NullBuffer,
            A::NullBuffer,
            A::NullBuffer,
            A::Buffer(&o.work),
            A::Buffer(&i.input.live),
            A::Buffer(&o.input.buffers[4]),
            A::U64(OFFSET as u64),
            A::U32(self.rows),
        ];
        k.functions[&self.qa.packed_spec().map_err(|e| e.to_string())?.symbol]
            .launch(stream, launch::config(plan)?, &qa)
            .map_err(|e| e.to_string())?;
        call(
            &cases()[2],
            k,
            arms.1,
            self.rows,
            &i.query,
            &o.qa,
            &o.query,
            stream,
            pdl,
        )?;
        let mut plan = self.qb.plan(self.rows).map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        let qb = [
            A::Buffer(&o.query.buffers[2]),
            A::Buffer(&o.query.buffers[3]),
            A::Buffer(&i.qb_weights),
            A::Buffer(&i.qb_scales),
            A::NullBuffer,
            A::NullBuffer,
            A::Buffer(&o.qb),
            A::NullBuffer,
            A::Range {
                buffer: &o.query.buffers[4],
                offset: OFFSET * 4,
                byte_len: self.rows as usize * 4,
            },
            A::Buffer(&i.query.live),
            A::U32(self.rows),
            A::U32(0),
        ];
        k.functions[&self.qb.packed_spec().map_err(|e| e.to_string())?.symbol]
            .launch(stream, launch::config(plan)?, &qb)
            .map_err(|e| e.to_string())
    }
}
fn compare_context(a: &[Vec<u8>], b: &[Vec<u8>]) -> LabResult<()> {
    if a != b {
        return Err(format!(
            "complete prelude mismatch in role {:?}",
            a.iter().zip(b).position(|(a, b)| a != b)
        ));
    }
    Ok(())
}
#[test]
fn context_uses_the_served_packed_projection_and_hc_symbols() {
    assert_eq!(symbols().len(), 5);
    for symbol in symbols() {
        assert!(sm120::fold_cubin_for_symbol(&symbol).is_some(), "{symbol}");
    }
    for rows in ROWS {
        assert_eq!(
            sm120::fold_fused::fused("attention.q-a", rows)
                .unwrap()
                .cluster
                .base
                .query
                .k,
            5120
        );
        assert_eq!(sm120::fold_fused::wide(rows).unwrap().base.query.n, 8192);
    }
}
#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_SMALL_NORM_ARTIFACTS"]
fn complete_prelude_bitwise_all_combinations_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let loaded = Loaded::load(s)?;
        let mut comparisons = 0;
        let mut negatives = 0;
        for rank in 0..4 {
            for rows in ROWS {
                let input = Inputs::new(s, rows, rank)?;
                for chain in [false, true] {
                    let prelude = Prelude {
                        qa: sm120::fold_fused::fused("attention.q-a", rows).unwrap(),
                        qb: sm120::fold_fused::wide(rows).unwrap(),
                        rows,
                        chain,
                    };
                    let frames = (0..16)
                        .map(|_| Frame::new(s, rows, &prelude.qa))
                        .collect::<LabResult<Vec<_>>>()?;
                    let graphs = [false, true]
                        .into_iter()
                        .map(|pdl| {
                            (0..16)
                                .map(|arm| {
                                    s.capture(|stream| {
                                        prelude.run(
                                            &loaded,
                                            &input,
                                            &frames[arm],
                                            (arm / 4, arm % 4),
                                            stream,
                                            pdl,
                                        )
                                    })
                                })
                                .collect::<LabResult<Vec<_>>>()
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    for sticky in [false, true] {
                        for live in 0..=rows {
                            upload(&input.input.live, &live.to_ne_bytes(), &s.stream)?;
                            let mut first_mode: Option<Vec<Vec<u8>>> = None;
                            for mode in 0..3 {
                                for frame in &frames {
                                    frame.poison(s, rows, sticky)?;
                                }
                                for (arm, frame) in frames.iter().enumerate() {
                                    if mode == 0 {
                                        prelude.run(
                                            &loaded,
                                            &input,
                                            frame,
                                            (arm / 4, arm % 4),
                                            &s.stream,
                                            false,
                                        )?;
                                    } else {
                                        graphs[mode - 1][arm]
                                            .launch(&s.stream)
                                            .map_err(|e| e.to_string())?;
                                    }
                                }
                                s.stream.synchronize().map_err(|e| e.to_string())?;
                                let baseline = frames[0].read(s)?;
                                if let Some(previous) = &first_mode {
                                    compare_context(previous, &baseline)?;
                                } else {
                                    first_mode = Some(baseline.clone());
                                }
                                if !sticky && live > 0 {
                                    let bytes = download(
                                        &frames[0].qb,
                                        live as usize * 8192 * 4,
                                        &s.stream,
                                    )?;
                                    if bytes.iter().all(|b| *b == 0)
                                        || bytes.iter().all(|b| *b == 0xa5)
                                    {
                                        return Err(
                                            "complete prelude produced no finite output witness"
                                                .into(),
                                        );
                                    }
                                }
                                for frame in &frames[1..] {
                                    compare_context(&baseline, &frame.read(s)?)?;
                                    comparisons += 1;
                                }
                            }
                        }
                    }
                    let original = download(&frames[1].qb, frames[1].qb.byte_len(), &s.stream)?;
                    let mut bad = original.clone();
                    bad[0] ^= 1;
                    upload(&frames[1].qb, &bad, &s.stream)?;
                    if compare_context(&frames[0].read(s)?, &frames[1].read(s)?).is_ok() {
                        return Err("context output mutation escaped".into());
                    }
                    negatives += 1;
                }
            }
        }
        if comparisons != 4 * 2 * 2 * (3 + 5) * 3 * 15 || negatives != 16 {
            return Err(format!("context coverage: {comparisons}, {negatives}"));
        }
        eprintln!("SMALL_NORM_CONTEXT comparisons={comparisons} negatives={negatives} PASS");
        s.receipt(
            "small-norm-context",
            BTreeMap::from([
                ("comparisons".into(), json!(comparisons)),
                ("negative_controls".into(), json!(negatives)),
                ("hc_producers".into(), json!(["main", "small-row-chain8"])),
                ("logical_ranks".into(), json!(4)),
            ]),
        )
    });
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_SMALL_NORM_ARTIFACTS and CHOPINFER_SMALL_NORM_SELECTION"]
fn complete_prelude_selected_pair_timing_on_gpu() {
    let path =
        std::env::var_os("CHOPINFER_SMALL_NORM_SELECTION").expect("qualified pair selection");
    let selection: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(selection["qualified"], true);
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let loaded = Loaded::load(s)?;
        let mut records = vec![];
        for rank in 0..4 {
            for rows in ROWS {
                let picked = &selection["rows"][rows.to_string()];
                let index = |kind: &str| -> LabResult<usize> {
                    let w = picked[kind].as_u64().ok_or("missing selected warp count")? as u32;
                    WARPS
                        .iter()
                        .position(|x| *x == w)
                        .map(|i| i + 1)
                        .ok_or("invalid selected warp count".into())
                };
                let selected = (index("input")?, index("query")?);
                let input = Inputs::new(s, rows, rank)?;
                for chain in [false, true] {
                    let prelude = Prelude {
                        qa: sm120::fold_fused::fused("attention.q-a", rows).unwrap(),
                        qb: sm120::fold_fused::wide(rows).unwrap(),
                        rows,
                        chain,
                    };
                    let frames = [
                        Frame::new(s, rows, &prelude.qa)?,
                        Frame::new(s, rows, &prelude.qa)?,
                    ];
                    for frame in &frames {
                        frame.poison(s, rows, false)?;
                    }
                    for (cache, repeats) in [("cold", 1), ("warm", 8)] {
                        let graphs = (0..2)
                            .map(|arm| {
                                s.capture(|stream| {
                                    (0..repeats).try_for_each(|_| {
                                        prelude.run(
                                            &loaded,
                                            &input,
                                            &frames[arm],
                                            if arm == 0 { (0, 0) } else { selected },
                                            stream,
                                            true,
                                        )
                                    })
                                })
                            })
                            .collect::<LabResult<Vec<_>>>()?;
                        let label = format!(
                            "small-norm-context:rank={rank}:rows={rows}:hc-chain={chain}:{cache}"
                        );
                        let pair = |candidate: usize| {
                            kernel_harness::with_timing_label(
                                &format!("{label}:candidate={candidate}"),
                                || {
                                    kernel_harness::time_ab_prepared(
                                        s,
                                        3,
                                        15,
                                        [EvictionMode::Read; 2],
                                        |side, stream| {
                                            if cache == "warm" {
                                                graphs[if side == 0 { 0 } else { candidate }]
                                                    .launch(stream)
                                                    .map_err(|e| e.to_string())?;
                                            }
                                            Ok(())
                                        },
                                        |side, stream| {
                                            graphs[if side == 0 { 0 } else { candidate }]
                                                .launch(stream)
                                                .map_err(|e| e.to_string())
                                        },
                                    )
                                },
                            )
                        };
                        let aa = pair(0)?;
                        let ab = pair(1)?;
                        s.stream.synchronize().map_err(|e| e.to_string())?;
                        compare_context(&frames[0].read(s)?, &frames[1].read(s)?)?;
                        records.push(json!({"rank":rank,"rows":rows,"hc_chain":chain,"cache":cache,"repeats":repeats,
                            "input_warps":WARPS[selected.0-1],"query_warps":WARPS[selected.1-1],"aa_ratio":aa.paired_ratio,"ratio":ab.paired_ratio,
                            "old_us":ab.arms[0].median_ms*1000./repeats as f32,"new_us":ab.arms[1].median_ms*1000./repeats as f32}));
                    }
                }
            }
        }
        if records.len() != 32 {
            return Err("complete-chain timing coverage differs".into());
        }
        eprintln!("SMALL_NORM_CONTEXT_TIMING cases=32 PASS");
        s.receipt(
            "small-norm-context-timing",
            BTreeMap::from([
                ("cases".into(), json!(records)),
                ("selection".into(), selection.clone()),
            ]),
        )
    });
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
