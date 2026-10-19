//! Exact three-launch versus one-launch window writes, including retained cells.
use super::fusion_checks::{Failures, compare_roles, guards, live_counts, validate_replay};
use super::norm_quantize::{PROBE_SYMBOLS, artifacts, download, upload, verify_modules};
use bundle_deepseek41::sm120::{
    ceil_parallel, fold_cubin_for_symbol, window_fused as registration,
};
use kernel_core::contract::{DType, LaunchPlan};
use kernel_core::latent_attention::LatentKvFormat;
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::{latent::window_fused::Entry, row::block_quantize};
use model_deepseek41_contract::{
    geometry::FLASH,
    own_ops::rope::{self, RopeTable},
    serving::DECODE_BATCH,
};
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

const ROWS: [u32; 6] = [1, 2, 4, 8, 16, DECODE_BATCH];
const PATTERNS: [&str; 16] = [
    "finite",
    "zero",
    "wide",
    "tiny",
    "nan_first",
    "nan_last",
    "inf",
    "weight_nan",
    "rope_nan",
    "sticky",
    "negative",
    "bad_sequence",
    "stale",
    "future",
    "staged_left",
    "ties",
];
const OFFSET: usize = 3;
const SLOTS: usize = 136;
struct Case {
    entry: Entry,
    staged: bool,
}
impl Case {
    fn bf16(&self) -> bool {
        self.entry.kv == LatentKvFormat::Bf16
    }
    fn label(&self) -> String {
        format!("{}:staged={}", self.entry.symbol, self.staged)
    }
    fn code_stride(&self) -> usize {
        SLOTS * if self.bf16() { 1024 } else { 512 } + 66
    }
    fn scale_stride(&self) -> usize {
        SLOTS * 16 + 7
    }
    fn quant(&self) -> block_quantize::Entry {
        ceil_parallel::quantizers(&FLASH, 1)
            .unwrap()
            .into_iter()
            .find(|e| e.query.width == 512 && e.query.input == DType::F32)
            .unwrap()
    }
    fn plans(&self, rows: u32) -> LabResult<Vec<LaunchPlan>> {
        let row = self.entry.plan(rows).map_err(|e| e.to_string())?;
        Ok(vec![
            row,
            self.quant().plan(rows).map_err(|e| e.to_string())?,
            row,
            row,
        ])
    }
    fn symbols(&self) -> [String; 4] {
        [
            self.entry.symbol.clone(),
            "chop_deepseek41_latent_window_rope_v1".into(),
            self.quant().symbol,
            if self.bf16() {
                "chop_deepseek41_window_store_bf16_v1"
            } else {
                "chop_deepseek41_latent_window_store_v1"
            }
            .into(),
        ]
    }
}
fn cases() -> Vec<Case> {
    registration::entries()
        .into_iter()
        .flat_map(|entry| {
            [false, true].map(|staged| Case {
                entry: entry.clone(),
                staged,
            })
        })
        .collect()
}
pub(super) fn symbols() -> Vec<String> {
    cases()
        .iter()
        .flat_map(Case::symbols)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
pub(super) fn coverage() -> serde_json::Value {
    json!({"entries":cases().iter().map(|c|c.label()).collect::<Vec<_>>(),"rows":ROWS,"patterns":PATTERNS,"bitwise_cases":cases().len()*ROWS.len()*PATTERNS.len(),"timing_cases":cases().len()*ROWS.len(),"negative_controls":cases().len()*ROWS.len()*6})
}
struct Inputs {
    x: DeviceBuffer,
    weight: DeviceBuffer,
    table: DeviceBuffer,
    live: DeviceBuffer,
    positions: DeviceBuffer,
    batches: DeviceBuffer,
    first: DeviceBuffer,
    last: DeviceBuffer,
}
struct Output {
    buffers: [DeviceBuffer; 6],
    lengths: [usize; 6],
}
impl Output {
    fn new(s: &Session, c: &Case, rows: u32) -> LabResult<Self> {
        let n = rows as usize;
        let lengths = [
            n * 512 * 4,
            n * 512,
            n * 16,
            n * c.code_stride(),
            n * c.scale_stride(),
            (n + OFFSET + 2) * 4,
        ];
        Ok(Self {
            buffers: [
                s.allocate(lengths[0])?,
                s.allocate(lengths[1])?,
                s.allocate(lengths[2])?,
                s.allocate(lengths[3])?,
                s.allocate(lengths[4])?,
                s.allocate(lengths[5])?,
            ],
            lengths,
        })
    }
    fn poison(&self, s: &Session, rows: u32, sticky: bool) -> LabResult<()> {
        for i in 0..5 {
            upload(&self.buffers[i], &vec![0xa5; self.lengths[i]], &s.stream)?;
        }
        let mut flags = vec![0x1357_2468u32; rows as usize + OFFSET + 2];
        flags[OFFSET..OFFSET + rows as usize].fill(if sticky { 0x80 } else { 0 });
        upload(
            &self.buffers[5],
            &flags
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
            &s.stream,
        )
    }
    fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.buffers
            .iter()
            .zip(self.lengths)
            .map(|(b, n)| download(b, n, &s.stream))
            .collect()
    }
}
const ROLES: [&str; 6] = [
    "latent",
    "quantized_codes",
    "quantized_scales",
    "window_codes",
    "window_scales",
    "status",
];
fn compare(a: &[Vec<u8>], b: &[Vec<u8>]) -> LabResult<()> {
    compare_roles(&ROLES, a, b)
}
fn validate_cases() -> LabResult<()> {
    for case in cases() {
        for rows in ROWS {
            let plans = case.plans(rows)?;
            for live in live_counts(rows) {
                validate_replay(rows, live, DECODE_BATCH, &plans)?;
            }
        }
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn run(
    c: &Case,
    f: &[Function],
    fused: bool,
    rows: u32,
    i: &Inputs,
    o: &Output,
    stream: &Stream,
    eps: u32,
) -> LabResult<()> {
    use runtime_cuda::KernelArgument as A;
    let [latent, qcodes, qscales, codes, scales, status] = &o.buffers;
    let mut norm = vec![
        A::Buffer(&i.x),
        A::Buffer(&i.weight),
        A::Buffer(latent),
        A::Buffer(&i.table),
        A::Buffer(&i.live),
        A::U32(eps),
        A::Buffer(status),
        A::U64(OFFSET as u64),
    ];
    let mut store = vec![
        A::Buffer(qcodes),
        A::Buffer(qscales),
        A::Buffer(&i.positions),
        A::Buffer(&i.batches),
        if c.staged {
            A::Buffer(&i.first)
        } else {
            A::NullBuffer
        },
        A::Buffer(&i.last),
        A::Buffer(codes),
        if c.bf16() {
            A::NullBuffer
        } else {
            A::Buffer(scales)
        },
        A::Buffer(&i.live),
        A::U32(rows),
        A::U32(SLOTS as u32),
        A::U32(u32::from(!c.staged)),
        A::U64(c.code_stride() as u64),
        A::U64(if c.bf16() { 0 } else { c.scale_stride() as u64 }),
        A::Buffer(status),
        A::U64(OFFSET as u64),
    ];
    let plan = c.entry.plan(rows).unwrap();
    if fused {
        norm.extend([A::Buffer(qcodes), A::Buffer(qscales)]);
        // Remove the repeated quantized inputs, live count and status tail.
        store.truncate(14);
        store.remove(8);
        store.drain(..2);
        norm.extend(store);
        f[0].launch(stream, launch::config(plan)?, &norm)
            .map_err(|e| e.to_string())
    } else {
        f[1].launch(stream, launch::config(plan)?, &norm)
            .map_err(|e| e.to_string())?;
        f[2].launch(
            stream,
            launch::config(c.quant().plan(rows).unwrap())?,
            &[
                A::Buffer(latent),
                A::Buffer(qcodes),
                A::Buffer(qscales),
                A::Buffer(&i.live),
                A::Buffer(status),
                A::U64(OFFSET as u64),
            ],
        )
        .map_err(|e| e.to_string())?;
        f[3].launch(stream, launch::config(plan)?, &store)
            .map_err(|e| e.to_string())
    }
}
fn values(rows: u32, pattern: &str) -> Vec<u8> {
    let mut state = 0x9ac1_04d2_1122_3344u64;
    (0..rows as usize * 512)
        .flat_map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = state as u32 & 0x8000_0000;
            let bits = match pattern {
                "zero" => sign,
                "tiny" => sign | ((state as u32 & 0x7f_ffff).max(1)),
                "wide" => {
                    sign | (((state >> 32) as u32 % 254 + 1) << 23) | (state as u32 & 0x7f_ffff)
                }
                "nan_first" if i == 0 => 0x7fc1_0000,
                "nan_last" if i == 511 => 0xff81_0000,
                "inf" if i == 511 => 0x7f80_0000,
                "ties" => sign | 0x3f80_8000 | ((i as u32 % 7) << 16),
                _ => sign | (((state >> 32) as u32 % 20 + 117) << 23) | (state as u32 & 0x7f_ffff),
            };
            bits.to_ne_bytes()
        })
        .collect()
}
fn set_inputs(s: &Session, c: &Case, rows: u32, pattern: &str, i: &Inputs) -> LabResult<()> {
    upload(&i.x, &values(rows, pattern), &s.stream)?;
    let weights = (0..512)
        .flat_map(|n| {
            if pattern == "weight_nan" && n == 511 {
                0x7fc1u16
            } else {
                0x3f00 + n as u16 % 128
            }
            .to_ne_bytes()
        })
        .collect::<Vec<_>>();
    upload(&i.weight, &weights, &s.stream)?;
    let positions = (0..rows)
        .map(|r| match pattern {
            "negative" => -1,
            "stale" => 17,
            "future" => 200,
            "staged_left" => 127,
            _ => 135 + (r % 2) as i32,
        })
        .collect::<Vec<_>>();
    let mut table = rope::reference(
        if c.staged {
            RopeTable::Compressed
        } else {
            RopeTable::Window
        },
        &(0..rows).map(|r| 135 + r).collect::<Vec<_>>(),
    )
    .unwrap();
    if pattern == "rope_nan" {
        table[63] = f32::NAN;
    }
    upload(
        &i.table,
        &table
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
        &s.stream,
    )?;
    let last = positions
        .iter()
        .map(|p| match pattern {
            "stale" => p + SLOTS as i32,
            "future" => p - 1,
            _ => *p,
        })
        .collect::<Vec<_>>();
    for (b, values) in [
        (&i.positions, positions),
        (
            &i.batches,
            (0..rows)
                .map(|r| {
                    if pattern == "bad_sequence" {
                        rows as i32
                    } else {
                        r as i32
                    }
                })
                .collect(),
        ),
        (&i.first, vec![128i32; rows as usize]),
        (&i.last, last),
    ] {
        upload(
            b,
            &values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
            &s.stream,
        )?;
    }
    Ok(())
}
#[test]
fn case_parameters_match_allocations_and_reject_oversized_live_counts() {
    validate_cases().unwrap();
    for case in cases() {
        for rows in ROWS {
            let plans = case.plans(rows).unwrap();
            assert!(validate_replay(rows, rows + 7, DECODE_BATCH, &plans).is_err());
            assert!(validate_replay(0, 0, DECODE_BATCH, &plans).is_err());
            assert!(validate_replay(DECODE_BATCH + 1, 1, DECODE_BATCH, &plans).is_err());
            let mut oversized = plans.clone();
            oversized[0].grid[0] = rows + 1;
            assert!(validate_replay(rows, rows, DECODE_BATCH, &oversized).is_err());
        }
    }
}

#[test]
fn window_fixture_inventory_and_negative_controls_are_complete() {
    assert_eq!(cases().len(), 4);
    for symbol in symbols() {
        assert!(fold_cubin_for_symbol(&symbol).is_some(), "{symbol}");
    }
    for pattern in PATTERNS {
        assert_eq!(values(2, pattern).len(), 2 * 512 * 4);
    }
    let original = vec![vec![0; 8]; 6];
    for role in 0..6 {
        let mut bad = original.clone();
        bad[role][0] = 1;
        assert!(compare(&original, &bad).is_err());
    }
}
#[test]
fn window_bitwise_graph_live_rows_and_timing_on_gpu() {
    validate_cases().unwrap();
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("window fusion comparison needs an idle GPU").unwrap();
        return;
    }
    verify_modules().unwrap();
    let probe = Box::leak(
        std::fs::read(artifacts().join("probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    let cubins = Cubins::new(
        Box::leak(Box::new([("probe_sm120", &probe[..])])),
        &[("probe_sm120", PROBE_SYMBOLS)],
    );
    let result = kernel_harness::gpu_test_with(cubins, 1 << 30, compare_on_gpu).unwrap();
    assert!(result.is_some());
}

fn compare_on_gpu(s: &Session) -> LabResult<()> {
    let mut bitwise = 0;
    let mut negative = 0;
    let mut arithmetic_negative = 0;
    let mut timings = Vec::new();
    let mut failures = Failures::new("WINDOW");
    for c in cases() {
        let modules = c
            .symbols()
            .iter()
            .map(|symbol| {
                let (name, _) = fold_cubin_for_symbol(symbol).unwrap();
                s.load_cubin(
                    &std::fs::read(artifacts().join(format!("{name}.cubin")))
                        .map_err(|e| e.to_string())?,
                )
            })
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .zip(c.symbols())
            .map(|(m, symbol)| m.function(&symbol).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        for rows in ROWS {
            let row_mark = failures.len();
            let label = format!("{} rows={rows}", c.label());
            let plans = c.plans(rows)?;
            validate_replay(rows, rows, DECODE_BATCH, &plans)?;
            let r = rows as usize;
            let input = Inputs {
                x: s.allocate(r * 512 * 4)?,
                weight: s.allocate(512 * 2)?,
                table: s.allocate(r * 64 * 4)?,
                live: s.allocate(4)?,
                positions: s.allocate(r * 4)?,
                batches: s.allocate(r * 4)?,
                first: s.allocate(r * 4)?,
                last: s.allocate(r * 4)?,
            };
            let outputs = [Output::new(s, &c, rows)?, Output::new(s, &c, rows)?];
            let eps = 1e-20_f32.to_bits();
            let graphs = (0..2)
                .map(|arm| {
                    s.capture(|stream| {
                        run(
                            &c,
                            &functions,
                            arm == 1,
                            rows,
                            &input,
                            &outputs[arm],
                            stream,
                            eps,
                        )
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            for pattern in PATTERNS {
                let case_mark = failures.len();
                set_inputs(s, &c, rows, pattern, &input)?;
                for (live_step, live_rows) in live_counts(rows).into_iter().enumerate() {
                    validate_replay(rows, live_rows, DECODE_BATCH, &plans)?;
                    upload(&input.live, &live_rows.to_ne_bytes(), &s.stream)?;
                    let mut eager: Option<Vec<Vec<u8>>> = None;
                    for replay in 0..3 {
                        let graph = replay != 0;
                        let context = format!(
                            "{label} pattern={pattern} live={live_rows} live_step={live_step} graph={graph} replay={replay}"
                        );
                        for output in &outputs {
                            output.poison(s, rows, pattern == "sticky")?;
                        }
                        for arm in 0..2 {
                            if graph {
                                graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                            } else {
                                run(
                                    &c,
                                    &functions,
                                    arm == 1,
                                    rows,
                                    &input,
                                    &outputs[arm],
                                    &s.stream,
                                    eps,
                                )?;
                            }
                        }
                        s.stream.synchronize().map_err(|e| e.to_string())?;
                        let old = outputs[0].read(s)?;
                        let new = outputs[1].read(s)?;
                        failures.record(&format!("{context} old-vs-fused"), compare(&old, &new));
                        if let Some(previous) = &eager {
                            failures.record(
                                &format!("{context} eager-vs-replay"),
                                compare(previous, &new),
                            );
                        }
                        for (arm, output) in [("old", &old), ("fused", &new)] {
                            let untouched: &[usize] = if c.bf16() { &[4] } else { &[] };
                            failures.record(
                                &format!("{context} arm={arm}"),
                                guards(
                                    &ROLES,
                                    output,
                                    rows,
                                    live_rows,
                                    untouched,
                                    OFFSET,
                                    pattern == "sticky",
                                ),
                            );
                            if pattern == "finite" && live_rows > 0 {
                                failures.require(
                                    &format!("{context} arm={arm}"),
                                    output[3][..c.code_stride()].iter().any(|b| *b != 0xa5),
                                    "finite fixture left its first window unchanged",
                                );
                            }
                        }
                        if eager.is_none() {
                            eager = Some(old);
                        }
                    }
                }
                bitwise += 1;
                let outcome = if failures.len() == case_mark {
                    "PASS"
                } else {
                    "FAIL"
                };
                eprintln!("WINDOW_BITWISE {label} pattern={pattern} {outcome}");
            }
            set_inputs(s, &c, rows, "finite", &input)?;
            upload(&input.live, &rows.to_ne_bytes(), &s.stream)?;
            for output in &outputs {
                output.poison(s, rows, false)?;
            }
            for graph in &graphs {
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            let expected = outputs[0].read(s)?;
            let original = outputs[1].read(s)?;
            failures.record(
                &format!("{label} control-baseline"),
                compare(&expected, &original),
            );
            for role in 0..6 {
                let mut bad = original[role].clone();
                bad[if role == 5 { OFFSET * 4 } else { 0 }] ^= 1;
                upload(&outputs[1].buffers[role], &bad, &s.stream)?;
                failures.require(
                    &format!("{label} negative={}", ROLES[role]),
                    compare(&original, &outputs[1].read(s)?).is_err(),
                    "byte mutation escaped the equality gate",
                );
                upload(&outputs[1].buffers[role], &original[role], &s.stream)?;
                negative += 1;
            }
            run(
                &c,
                &functions,
                true,
                rows,
                &input,
                &outputs[1],
                &s.stream,
                1e6_f32.to_bits(),
            )?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
            failures.require(
                &format!("{label} negative=epsilon"),
                compare(&original, &outputs[1].read(s)?).is_err(),
                "changed epsilon escaped the equality gate",
            );
            arithmetic_negative += 1;
            if failures.len() == row_mark {
                for output in &outputs {
                    output.poison(s, rows, false)?;
                }
                for graph in &graphs {
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                }
                s.stream.synchronize().map_err(|e| e.to_string())?;
                let timing_label = format!("window:{label}");
                let timing = kernel_harness::with_timing_label(&timing_label, || {
                    kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                        graphs[arm].launch(stream).map_err(|e| e.to_string())
                    })
                })?;
                eprintln!(
                    "WINDOW_TIMING {label} old_us={} new_us={} ratio={}",
                    timing.arms[0].median_ms * 1000.,
                    timing.arms[1].median_ms * 1000.,
                    timing.paired_ratio
                );
                timings.push(json!({"label":c.label(),"rows":rows,"old_us":timing.arms[0].median_ms*1000.,"new_us":timing.arms[1].median_ms*1000.,"ratio":timing.paired_ratio}));
            } else {
                eprintln!("WINDOW_TIMING_SKIPPED {label} reason=bitwise_failure");
            }
        }
    }
    assert_eq!(bitwise, cases().len() * ROWS.len() * PATTERNS.len());
    assert_eq!(negative, cases().len() * ROWS.len() * 6);
    assert_eq!(arithmetic_negative, cases().len() * ROWS.len());
    s.receipt(
        "window-fused",
        BTreeMap::from([
            ("bitwise_cases".into(), json!(bitwise)),
            ("negative_controls".into(), json!(negative)),
            ("arithmetic_controls".into(), json!(arithmetic_negative)),
            ("timings".into(), json!(timings)),
            ("mismatches".into(), json!(failures.messages())),
        ]),
    )?;
    eprintln!(
        "WINDOW_COMPLETE bitwise={bitwise} negative={negative} arithmetic_negative={arithmetic_negative} timings={} mismatches={}",
        timings.len(),
        failures.len()
    );
    failures.finish()
}
