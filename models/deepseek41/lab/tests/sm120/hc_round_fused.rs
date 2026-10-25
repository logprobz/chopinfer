//! Old round, expansion and split mixer versus reusable fused tasks.
use super::{
    fusion_checks::{Failures, compare_roles, live_counts},
    norm_quantize::{download, upload},
    round_artifacts as artifacts,
};
use bundle_deepseek41::sm120::hc_round_expand_mix as hc_expand_mix;
use kernel_core::contract::LaunchPlan;
use kernel_harness::{LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::sinkhorn_hc::{Geometry, lagged, lagged_split};
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::json;
use std::collections::BTreeMap;
const ROWS: [u32; 6] = [1, 2, 4, 8, 16, 32];
const PATTERNS: [&str; 17] = [
    "finite",
    "zero",
    "tiny",
    "wide",
    "nan_first",
    "nan_last",
    "inf",
    "weight_nan",
    "post_nan",
    "comb_inf",
    "cancellation",
    "sum_tie",
    "sum_nan",
    "sum_inf",
    "sum_subnormal",
    "sum_negative_zero",
    "sum_overflow",
];
const ROLES: [&str; 6] = ["expanded", "collapsed", "pre", "post", "comb", "rounded"];
const H: usize = 5120;
const N: usize = 4;
const MIXES: usize = 24;
#[path = "hc_small_rows.rs"]
mod small_rows;
pub(super) fn symbols() -> Vec<String> {
    [
        "chop_row_rounding_round_deepseek41_w5120_v1".into(),
        "chop_sinkhorn_hc_deepseek41_expand_f32_v1".into(),
        "chop_sinkhorn_hc_deepseek41_lagged_mix_collapse_split_v1".into(),
    ]
    .into_iter()
    .chain(hc_expand_mix::entry(&FLASH).symbols())
    .collect()
}
pub(super) fn coverage() -> serde_json::Value {
    json!({"cases":ROWS.map(|rows|json!({"label":"round_expand_mix","rows":rows})),"patterns":PATTERNS,"bitwise":ROWS.len()*PATTERNS.len(),"negative":ROWS.len()*ROLES.len(),"arithmetic":ROWS.len(),"timings":ROWS.len()})
}
fn plans(rows: u32, live: u32) -> LabResult<[LaunchPlan; 5]> {
    if rows == 0 || rows > 32 || live > rows {
        return Err(format!("invalid HC replay rows={rows} live={live}"));
    }
    let old = lagged_split::entry(
        lagged::LAB_PROFILE,
        Geometry {
            capacity: 32,
            ..lagged::LAB
        },
        false,
    );
    let expand = lagged::entries(
        lagged::LAB_PROFILE,
        Geometry {
            capacity: 32,
            ..lagged::LAB
        },
        false,
    )
    .into_iter()
    .find(|e| e.op == lagged::Op::Expand)
    .unwrap();
    let [partial, finish] = hc_expand_mix::entry(&FLASH)
        .plans(rows)
        .map_err(|e| e.to_string())?;
    Ok([
        kernels_sm120::row::rounding::plan(rows).map_err(|e| e.to_string())?,
        expand.plan(rows, live).map_err(|e| e.to_string())?,
        old.plan(rows, live).map_err(|e| e.to_string())?,
        partial,
        finish,
    ])
}
struct Input {
    x: DeviceBuffer,
    y: DeviceBuffer,
    fn_: DeviceBuffer,
    base: DeviceBuffer,
    scale: DeviceBuffer,
    pre: DeviceBuffer,
    post_in: DeviceBuffer,
    comb_in: DeviceBuffer,
    live: DeviceBuffer,
}
struct Output {
    data: Vec<DeviceBuffer>,
    lengths: Vec<usize>,
    scratch: DeviceBuffer,
    scratch_len: usize,
    tickets: DeviceBuffer,
}
impl Output {
    fn new(s: &Session, rows: u32, fused: bool) -> LabResult<Self> {
        let r = rows as usize;
        let lengths = vec![
            r * N * H * 2,
            r * H * 2,
            r * N * 4,
            r * N * 4,
            r * N * N * 4,
            r * H * 2,
        ];
        Ok(Self {
            data: lengths
                .iter()
                .map(|&n| s.allocate(n))
                .collect::<LabResult<_>>()?,
            lengths,
            scratch: s.allocate(r * (MIXES * (N * H / 128) + if fused { 256 } else { 0 }) * 4)?,
            scratch_len: r * (MIXES * (N * H / 128) + if fused { 256 } else { 0 }) * 4,
            tickets: s.allocate(r * 4)?,
        })
    }
    fn reset(&self, s: &Session, rows: u32, pattern: &str) -> LabResult<Vec<Vec<u8>>> {
        let mut initial = self
            .lengths
            .iter()
            .map(|&n| vec![0xa5; n])
            .collect::<Vec<_>>();
        let mut post = artifacts::random(rows as usize * N, 31, 1.0);
        let mut comb = artifacts::random(rows as usize * N * N, 47, 0.5);
        if pattern == "post_nan" {
            post[N - 1] = f32::NAN;
        }
        if pattern == "comb_inf" {
            comb[N * N - 1] = f32::INFINITY;
        }
        initial[3] = artifacts::f32_bytes(&post);
        initial[4] = artifacts::f32_bytes(&comb);
        for (b, v) in self.data.iter().zip(&initial) {
            upload(b, v, &s.stream)?;
        }
        upload(&self.scratch, &vec![0xa5; self.scratch_len], &s.stream)?;
        Ok(initial)
    }
    fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.data
            .iter()
            .zip(&self.lengths)
            .map(|(b, &n)| download(b, n, &s.stream))
            .collect()
    }
}
fn set_inputs(s: &Session, rows: u32, pattern: &str, i: &Input) -> LabResult<()> {
    let mut x = artifacts::random(rows as usize * N * H, 123, 4.0);
    match pattern {
        "zero" => x.fill(-0.0),
        "tiny" => x.fill(f32::from_bits(1)),
        "wide" => {
            for (v, n) in x.iter_mut().zip(0..) {
                *v *= if n % 2 == 0 { 1e30 } else { 1e-30 };
            }
        }
        "nan_first" => x[0] = f32::NAN,
        "nan_last" => x[N * H - 1] = f32::from_bits(0xff81_0000),
        "inf" => x[N * H - 1] = f32::INFINITY,
        "cancellation" => {
            for (v, n) in x.iter_mut().zip(0..) {
                *v = if n % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        _ => {}
    }
    let mut weights = artifacts::random(MIXES * N * H, 853, 0.02);
    if pattern == "weight_nan" {
        weights[MIXES * N * H - 1] = f32::NAN;
    }
    upload(&i.x, &artifacts::bf16_bytes(&x), &s.stream)?;
    let mut sum = artifacts::random(rows as usize * H, 71, 2.0);
    match pattern {
        "sum_tie" => sum.fill(f32::from_bits(0x3f80_8001)),
        "sum_nan" => sum[H - 1] = f32::from_bits(0xff80_0001),
        "sum_inf" => sum[0] = f32::INFINITY,
        "sum_subnormal" => sum.fill(f32::from_bits(0x0001_0001)),
        "sum_negative_zero" => sum.fill(-0.0),
        "sum_overflow" => sum[0] = f32::MAX,
        _ => {}
    }
    upload(&i.y, &artifacts::f32_bytes(&sum), &s.stream)?;
    upload(&i.fn_, &artifacts::f32_bytes(&weights), &s.stream)?;
    upload(
        &i.base,
        &artifacts::f32_bytes(&artifacts::random(MIXES, 61, 1.0)),
        &s.stream,
    )?;
    upload(
        &i.scale,
        &artifacts::f32_bytes(&[0.5, 0.25, 1.0]),
        &s.stream,
    )?;
    upload(
        &i.pre,
        &artifacts::f32_bytes(&artifacts::random(rows as usize * N, 13, 1.0)),
        &s.stream,
    )?;
    let mut post = artifacts::random(rows as usize * N, 31, 1.0);
    let mut comb = artifacts::random(rows as usize * N * N, 47, 0.5);
    if pattern == "post_nan" {
        post[N - 1] = f32::NAN;
    }
    if pattern == "comb_inf" {
        comb[N * N - 1] = f32::INFINITY;
    }
    upload(&i.post_in, &artifacts::f32_bytes(&post), &s.stream)?;
    upload(&i.comb_in, &artifacts::f32_bytes(&comb), &s.stream)
}
fn run(
    f: &[Function],
    fused: bool,
    aliased: bool,
    p: &[LaunchPlan; 5],
    i: &Input,
    o: &Output,
    stream: &Stream,
) -> LabResult<()> {
    use runtime_cuda::KernelArgument as A;
    let d = &o.data;
    let post = if aliased { &d[3] } else { &i.post_in };
    let comb = if aliased { &d[4] } else { &i.comb_in };
    if !fused {
        f[0].launch(
            stream,
            launch::config(p[0])?,
            &[A::Buffer(&i.y), A::Buffer(&d[5]), A::Buffer(&i.live)],
        )
        .map_err(|e| e.to_string())?;
    }
    let first = if fused {
        vec![
            A::Buffer(&i.x),
            A::Buffer(&i.y),
            A::Buffer(&d[5]),
            A::Buffer(post),
            A::Buffer(comb),
            A::Buffer(&d[0]),
            A::Buffer(&i.fn_),
            A::Buffer(&o.scratch),
            A::Buffer(&i.live),
        ]
    } else {
        vec![
            A::Buffer(&i.x),
            A::Buffer(&d[5]),
            A::Buffer(post),
            A::Buffer(comb),
            A::Buffer(&d[0]),
            A::Buffer(&i.live),
        ]
    };
    let mut second = vec![A::Buffer(&d[0])];
    if !fused {
        second.push(A::Buffer(&i.fn_));
    }
    second.extend([
        A::Buffer(&i.base),
        A::Buffer(&i.scale),
        A::Buffer(&i.pre),
        A::Buffer(&d[1]),
        A::Buffer(&d[2]),
        A::Buffer(&d[3]),
        A::Buffer(&d[4]),
        A::Buffer(&o.scratch),
    ]);
    if !fused {
        second.push(A::Buffer(&o.tickets));
    }
    second.push(A::Buffer(&i.live));
    let at = if fused { 3 } else { 1 };
    f[at]
        .launch(stream, launch::config(p[at])?, &first)
        .map_err(|e| e.to_string())?;
    f[at + 1]
        .launch(stream, launch::config(p[at + 1])?, &second)
        .map_err(|e| e.to_string())
}

#[test]
fn case_parameters_and_served_symbols_are_complete() {
    assert_eq!(hc_expand_mix::entry(&FLASH).query.mix, lagged::LAB.mix);
    for symbol in symbols() {
        assert!(
            bundle_deepseek41::sm120::fold_cubin_for_symbol(&symbol).is_some(),
            "{symbol}"
        );
    }
    for rows in ROWS {
        for live in live_counts(rows) {
            let p = plans(rows, live).unwrap();
            for (index, plan) in p.into_iter().enumerate() {
                assert_eq!(plan.grid[if index == 0 { 0 } else { 1 }], rows);
                assert_eq!(plan.block, [256, 1, 1]);
            }
        }
        assert!(plans(rows, rows + 7).is_err());
    }
    assert!(plans(0, 0).is_err());
    assert!(plans(33, 1).is_err());
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_ROUND_ARTIFACTS"]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    for rows in ROWS {
        for live in live_counts(rows) {
            plans(rows, live).unwrap();
        }
    }
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("HC fusion needs an idle GPU").unwrap();
        return;
    }
    let result =
        kernel_harness::gpu_test_with(artifacts::timing_cubins().unwrap(), 1 << 30, compare_on_gpu)
            .unwrap();
    assert!(result.is_some());
}
fn compare_on_gpu(s: &Session) -> LabResult<()> {
    let loaded = artifacts::load(s, &symbols())?;
    let f = &loaded.functions;
    let mut failures = Failures::new("ROUND_HC");
    let mut bitwise = 0;
    let mut negative = 0;
    let mut arithmetic = 0;
    let mut timings = Vec::new();
    for rows in ROWS {
        let mark = failures.len();
        let r = rows as usize;
        let p = plans(rows, rows)?;
        let label = format!("round_expand_mix rows={rows}");
        let input = Input {
            x: s.allocate(r * N * H * 2)?,
            y: s.allocate(r * H * 4)?,
            fn_: s.allocate(MIXES * N * H * 4)?,
            base: s.allocate(MIXES * 4)?,
            scale: s.allocate(12)?,
            pre: s.allocate(r * N * 4)?,
            post_in: s.allocate(r * N * 4)?,
            comb_in: s.allocate(r * N * N * 4)?,
            live: s.allocate(4)?,
        };
        let outputs = [Output::new(s, rows, false)?, Output::new(s, rows, true)?];
        for out in &outputs {
            upload(&out.tickets, &vec![0; r * 4], &s.stream)?;
        }
        let graphs = (0..2)
            .map(|arm| {
                s.capture(|stream| run(f, arm == 1, true, &p, &input, &outputs[arm], stream))
            })
            .collect::<LabResult<Vec<_>>>()?;
        for pattern in PATTERNS {
            let case_mark = failures.len();
            set_inputs(s, rows, pattern, &input)?;
            for (live_step, live) in live_counts(rows).into_iter().enumerate() {
                plans(rows, live)?;
                upload(&input.live, &live.to_ne_bytes(), &s.stream)?;
                let mut eager: Option<Vec<Vec<u8>>> = None;
                for replay in 0..3 {
                    let context = format!(
                        "{label} pattern={pattern} live={live} live_step={live_step} replay={replay}"
                    );
                    let mut initial = Vec::new();
                    for out in &outputs {
                        initial.push(out.reset(s, rows, pattern)?);
                    }
                    for arm in 0..2 {
                        if replay == 0 {
                            run(f, arm == 1, true, &p, &input, &outputs[arm], &s.stream)?;
                        } else {
                            graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                        }
                    }
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                    let old = outputs[0].read(s)?;
                    let new = outputs[1].read(s)?;
                    failures.record(&context, compare_roles(&ROLES, &old, &new));
                    if let Some(ref previous) = eager {
                        failures.record(
                            &format!("{context} eager-vs-replay"),
                            compare_roles(&ROLES, previous, &new),
                        );
                    } else {
                        eager = Some(old.clone());
                    }
                    for (arm, data) in [&old, &new].into_iter().enumerate() {
                        for (role, bytes) in data.iter().enumerate() {
                            let start = bytes.len() / r * live as usize;
                            failures.require(
                                &format!("{context} arm={arm} role={}", ROLES[role]),
                                bytes[start..] == initial[arm][role][start..],
                                "inactive output changed",
                            );
                        }
                    }
                    for (arm, out) in outputs.iter().enumerate() {
                        let bytes = download(&out.scratch, out.scratch_len, &s.stream)?;
                        failures.require(
                            &format!("{context} scratch arm={arm}"),
                            bytes[out.scratch_len / r * live as usize..]
                                .iter()
                                .all(|&b| b == 0xa5),
                            "inactive scratch changed",
                        );
                    }
                    failures.require(
                        &format!("{context} old tickets"),
                        download(&outputs[0].tickets, r * 4, &s.stream)?
                            .iter()
                            .all(|&b| b == 0),
                        "split tickets did not reset",
                    );
                }
            }
            bitwise += 1;
            eprintln!(
                "ROUND_HC_BITWISE {label} pattern={pattern} {}",
                if failures.len() == case_mark {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }
        set_inputs(s, rows, "finite", &input)?;
        upload(&input.live, &rows.to_ne_bytes(), &s.stream)?;
        for out in &outputs {
            out.reset(s, rows, "finite")?;
        }
        for g in &graphs {
            g.launch(&s.stream).map_err(|e| e.to_string())?;
        }
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let original = outputs[1].read(s)?;
        failures.record(
            &format!("{label} controls"),
            compare_roles(&ROLES, &outputs[0].read(s)?, &original),
        );
        for role in 0..ROLES.len() {
            let mut bad = original[role].clone();
            bad[0] ^= 1;
            upload(&outputs[1].data[role], &bad, &s.stream)?;
            failures.require(
                &format!("{label} negative={}", ROLES[role]),
                compare_roles(&ROLES, &original, &outputs[1].read(s)?).is_err(),
                "byte control escaped",
            );
            upload(&outputs[1].data[role], &original[role], &s.stream)?;
            negative += 1;
        }
        upload(
            &input.pre,
            &artifacts::f32_bytes(&vec![32.0; r * N]),
            &s.stream,
        )?;
        outputs[1].reset(s, rows, "finite")?;
        graphs[1].launch(&s.stream).map_err(|e| e.to_string())?;
        s.stream.synchronize().map_err(|e| e.to_string())?;
        failures.require(
            &format!("{label} negative=pre"),
            compare_roles(&ROLES, &original, &outputs[1].read(s)?).is_err(),
            "changed collapse weights escaped",
        );
        arithmetic += 1;
        if failures.len() == mark {
            set_inputs(s, rows, "finite", &input)?;
            // Fixed inputs for timing: separate previous and next post/comb.
            // Compare this form with the aliased production form before timing.
            let timing_graphs = (0..2)
                .map(|arm| {
                    s.capture(|stream| {
                        for _ in 0..32 {
                            run(f, arm == 1, false, &p, &input, &outputs[arm], stream)?;
                        }
                        Ok(())
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            for out in &outputs {
                out.reset(s, rows, "finite")?;
            }
            for g in &timing_graphs {
                g.launch(&s.stream).map_err(|e| e.to_string())?;
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            failures.record(
                &format!("{label} immutable timing inputs old"),
                compare_roles(&ROLES, &original, &outputs[0].read(s)?),
            );
            failures.record(
                &format!("{label} immutable timing inputs fused"),
                compare_roles(&ROLES, &original, &outputs[1].read(s)?),
            );
            if failures.len() != mark {
                eprintln!("ROUND_HC_TIMING_SKIPPED {label} reason=bitwise_failure");
                continue;
            }
            let timing = kernel_harness::with_timing_label(&label, || {
                kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                    timing_graphs[arm].launch(stream).map_err(|e| e.to_string())
                })
            })?;
            eprintln!(
                "ROUND_HC_TIMING {label} old_us={} new_us={} ratio={}",
                timing.arms[0].median_ms * 1000. / 32.,
                timing.arms[1].median_ms * 1000. / 32.,
                timing.paired_ratio
            );
            timings.push(json!({"rows":rows,"old_us":timing.arms[0].median_ms*1000./32.,"new_us":timing.arms[1].median_ms*1000./32.,"ratio":timing.paired_ratio}));
        } else {
            eprintln!("ROUND_HC_TIMING_SKIPPED {label} reason=bitwise_failure");
        }
    }
    s.receipt(
        "round-expand-mix",
        BTreeMap::from([
            ("bitwise".into(), json!(bitwise)),
            ("negative".into(), json!(negative)),
            ("arithmetic".into(), json!(arithmetic)),
            ("timings".into(), json!(timings)),
            ("mismatches".into(), json!(failures.messages())),
        ]),
    )?;
    eprintln!(
        "ROUND_HC_COMPLETE bitwise={bitwise} negative={negative} arithmetic_negative={arithmetic} timings={} mismatches={}",
        timings.len(),
        failures.len()
    );
    failures.finish()
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_ROUND_ARTIFACTS"]
fn exclusive_profile_on_gpu() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("HC profile needs an idle GPU").unwrap();
        return;
    }
    let result = kernel_harness::gpu_test_with(artifacts::timing_cubins().unwrap(), 1 << 30, |s| {
        let loaded = artifacts::load(s, &symbols())?;
        for rows in ROWS {
            let r = rows as usize;
            let p = plans(rows, rows)?;
            let i = Input {
                x: s.allocate(r * N * H * 2)?,
                y: s.allocate(r * H * 4)?,
                fn_: s.allocate(MIXES * N * H * 4)?,
                base: s.allocate(MIXES * 4)?,
                scale: s.allocate(12)?,
                pre: s.allocate(r * N * 4)?,
                post_in: s.allocate(r * N * 4)?,
                comb_in: s.allocate(r * N * N * 4)?,
                live: s.allocate(4)?,
            };
            let outputs = [Output::new(s, rows, false)?, Output::new(s, rows, true)?];
            set_inputs(s, rows, "finite", &i)?;
            upload(&i.live, &rows.to_ne_bytes(), &s.stream)?;
            for out in &outputs {
                upload(&out.tickets, &vec![0; r * 4], &s.stream)?;
                out.reset(s, rows, "finite")?;
            }
            let graphs = (0..2)
                .map(|arm| {
                    s.capture(|stream| {
                        run(
                            &loaded.functions,
                            arm == 1,
                            false,
                            &p,
                            &i,
                            &outputs[arm],
                            stream,
                        )
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            for _ in 0..128 {
                for g in &graphs {
                    g.launch(&s.stream).map_err(|e| e.to_string())?;
                }
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            eprintln!("ROUND_HC_PROFILE rows={rows} calls=128");
        }
        Ok(())
    })
    .unwrap();
    assert!(result.is_some());
}
