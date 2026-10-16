//! Position coverage independent of the table supplied to the device.
//! Keep this a child of `row` to reuse its launch fixture without changing it.
use super::*;
use std::collections::BTreeSet;

const MAX_POSITION: usize = (1 << 20) - 1;
pub(crate) const SEEDS: [u64; 4] = [0x9a0, 0x5511, 0x12345678, 0xdeadbeef];
const QH: usize = 16;
const QK: usize = 192;
const V: usize = 128;
const PAIRS: usize = 32;

/// Shards partition positions by residue so every context limit is covered.
pub(crate) fn shard() -> (usize, usize) {
    let value = std::env::var("CHOPINFER_ROPE_SHARD").unwrap_or_else(|_| "0/1".into());
    parse_shard(&value)
}

fn parse_shard(value: &str) -> (usize, usize) {
    let (index, count) = value.split_once('/').expect("RoPE shard i/n");
    let index: usize = index.parse().expect("integer shard index");
    let count: usize = count.parse().expect("integer shard count");
    assert!(
        count > 0 && count <= 262144 && index < count,
        "invalid RoPE shard"
    );
    (index, count)
}

#[test]
fn position_shards_partition_both_contexts() {
    for count in [1, 2, 3, 7] {
        for limit in [262144, MAX_POSITION + 1] {
            let mut visits = vec![0_u8; limit];
            for index in 0..count {
                assert_eq!(parse_shard(&format!("{index}/{count}")), (index, count));
                for p in (index..limit).step_by(count) {
                    visits[p] += 1;
                }
            }
            assert!(visits.iter().all(|&n| n == 1));
        }
    }
}

pub(crate) fn positions(exhaustive: bool) -> Vec<usize> {
    if exhaustive {
        let (index, count) = shard();
        return (index..=MAX_POSITION).step_by(count).collect();
    }
    let mut result: BTreeSet<_> = (0..512).collect();
    for power in 9..=20 {
        let boundary = 1_usize << power;
        for offset in -32_isize..32 {
            let position = boundary.saturating_add_signed(offset);
            if position <= MAX_POSITION {
                result.insert(position);
            }
        }
    }
    result.into_iter().collect()
}

pub(crate) struct Stats {
    reference: oracle::OracleOutput,
    pub(crate) maximum_position: usize,
    checked: u64,
    pub(crate) failures: u64,
    table_bit_differences: u64,
    table_bit_comparison: bool,
    pub(crate) oracle_description: &'static str,
    oracle_bit_differences: u64,
    worst: f64,
    worst_position: usize,
    worst_pair: usize,
    first_failure: Option<usize>,
    pair_worst: [f64; PAIRS],
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            maximum_position: MAX_POSITION,
            reference: oracle::OracleOutput::new(vec![0.0], vec![2.0], 2)
                .expect("one-element oracle"),
            checked: 0,
            failures: 0,
            table_bit_differences: 0,
            table_bit_comparison: false,
            oracle_description: "HF F32 angle; accurate trig; BF16 cos/sin; F32 rotation; K=2 S=2",
            oracle_bit_differences: 0,
            worst: 0.0,
            worst_position: 0,
            worst_pair: 0,
            first_failure: None,
            pair_worst: [0.0; PAIRS],
        }
    }
}

impl Stats {
    pub(crate) fn check(&mut self, actual: u16, expected: f64, position: usize, pair: usize) {
        if actual != f32_to_bf16(expected as f32) {
            self.oracle_bit_differences += 1;
            self.failures += 1;
            self.first_failure = Some(self.first_failure.map_or(position, |p| p.min(position)));
        }
        self.check_historical_tolerance(actual, expected, position, pair);
    }

    /// Historical arithmetic experiments only; never the served/table contract.
    fn check_historical_tolerance(
        &mut self,
        actual: u16,
        expected: f64,
        position: usize,
        pair: usize,
    ) {
        // Match rope_case: the seeded operands are bounded by 0.7, so S=2
        // bounds both products, and the reduction length is two.
        self.reference.values[0] = expected;
        let bound = oracle::numeric_bound(&self.reference, 0, DType::Bf16);
        self.check_bound(
            f64::from(bf16_to_f32(actual)),
            expected,
            bound,
            position,
            pair,
        );
    }

    pub(crate) fn check_selected(
        &mut self,
        actual: u16,
        expected: f64,
        position: usize,
        pair: usize,
    ) {
        if matches!(arm().as_str(), "served" | "table") {
            self.check(actual, expected, position, pair);
        } else {
            self.oracle_bit_differences += u64::from(actual != f32_to_bf16(expected as f32));
            self.check_historical_tolerance(actual, expected, position, pair);
        }
    }

    pub(crate) fn check_bound(
        &mut self,
        value: f64,
        expected: f64,
        bound: f64,
        position: usize,
        pair: usize,
    ) {
        let ratio = if value.is_finite() {
            (value - expected).abs() / bound
        } else {
            f64::MAX
        };
        self.checked += 1;
        if ratio > 1.0 {
            self.failures += 1;
            self.first_failure = Some(self.first_failure.map_or(position, |p| p.min(position)));
        }
        if ratio > self.worst {
            self.worst = ratio;
            self.worst_position = position;
            self.worst_pair = pair;
        }
        self.pair_worst[pair] = self.pair_worst[pair].max(ratio);
    }

    pub(crate) fn receipt(
        &self,
        session: &Session,
        name: &str,
        rows: usize,
        seed: u64,
        exhaustive: bool,
    ) -> LabResult<()> {
        let data = BTreeMap::from([
            (
                "oracle_bit_differences".into(),
                serde_json::json!(self.oracle_bit_differences),
            ),
            ("entry".into(), json!(name)),
            ("arm".into(), json!(arm())),
            (
                "table_bit_comparison".into(),
                json!(self.table_bit_comparison),
            ),
            (
                "table_bit_differences".into(),
                json!(self.table_bit_differences),
            ),
            ("rows".into(), json!(rows)),
            ("seed".into(), json!(seed)),
            ("exhaustive_positions".into(), json!(exhaustive)),
            ("position_shard".into(), json!(shard())),
            ("maximum_position".into(), json!(self.maximum_position)),
            ("checked_elements".into(), json!(self.checked)),
            ("failures".into(), json!(self.failures)),
            ("first_failure".into(), json!(self.first_failure)),
            ("worst_error_over_bound".into(), json!(self.worst)),
            ("worst_position".into(), json!(self.worst_position)),
            ("worst_pair".into(), json!(self.worst_pair)),
            ("pair_worst".into(), json!(self.pair_worst)),
            ("oracle".into(), json!(self.oracle_description)),
        ]);
        session.receipt("rope_positions", data)?;
        eprintln!(
            "RoPE {name} rows={rows} seed={seed}: {} checked, {} failures, worst={}, position={}, pair={}",
            self.checked, self.failures, self.worst, self.worst_position, self.worst_pair
        );
        Ok(())
    }
}

fn setup<'a>(
    session: &'a Session,
    rows: usize,
    ring: bool,
    fp8: bool,
    theta: f32,
    table: bool,
) -> LabResult<Fixture<'a>> {
    let kvh = if ring { 2 } else { 1 };
    let page = if ring { 32 } else { 64 };
    let cache_tokens = if ring { 128 } else { page };
    let cache_bytes = if fp8 { 1 } else { 2 };
    let max_pages = MAX_POSITION / page + 1;
    let mut f = Fixture::new(session, rows)?;
    f.zero("q", rows * QH * QK * 2)?;
    f.zero("k_new", rows * kvh * QK * 2)?;
    f.zero("v_new", rows * kvh * V * 2)?;
    f.zero("k_pool", rows * cache_tokens * kvh * QK * cache_bytes)?;
    f.zero("v_pool", rows * cache_tokens * kvh * V * cache_bytes)?;
    // Each live row has its own slot/page. Arbitrary sampled positions must
    // not alias a ring slot or cause concurrent stores to one physical page.
    f.i32(
        "block_tables",
        &(0..rows)
            .flat_map(|row| std::iter::repeat_n(row as i32, max_pages))
            .collect::<Vec<_>>(),
    )?;
    let ids: Vec<_> = (0..rows as i32).collect();
    f.i32("slot_ids", &ids)?;
    f.i32("token_batch", &ids)?;
    if table {
        let values: Vec<_> = (0..=MAX_POSITION)
            .flat_map(|position| {
                (0..PAIRS).flat_map(move |pair| {
                    let (c, s) = kernel_core::hf_rope::cos_sin(position, pair, f64::from(theta));
                    [c as f32, s as f32]
                })
            })
            .collect();
        f.f32("cos_sin", &values)?;
    }
    f.scalar("max_pages", max_pages as u32);
    f.scalar("ring_size", 128);
    f.scalar64(
        "k_page_stride_bytes",
        (page * kvh * QK * cache_bytes) as u64,
    );
    f.scalar64("v_page_stride_bytes", (page * kvh * V * cache_bytes) as u64);
    f.scalar("theta_bits", theta.to_bits());
    f.scalar("v_multiplier_bits", 0.707_f32.to_bits());
    f.scalar("k_scale_bits", 1.0_f32.to_bits());
    f.scalar("v_scale_bits", 1.0_f32.to_bits());
    Ok(f)
}

fn run_case(
    session: &Session,
    module: &Module,
    entry: &RowInstantiation,
    name: &str,
    rows: usize,
    exhaustive: bool,
) -> LabResult<u64> {
    let ring = name.contains("swa");
    let kvh = if ring { 2 } else { 1 };
    let theta = if ring { 1e4_f32 } else { 1e7_f32 };
    let stride = QH * QK + kvh * (QK + V);
    let compare_table = arm() == "computed";
    let baseline = super::module(session)?;
    let entries = catalog();
    let table_entry = find(&entries, &name.replace("_computed_", "_table_"));
    let mut f = setup(
        session,
        rows,
        ring,
        name.contains("fp8"),
        theta,
        name.contains("table") || compare_table,
    )?;
    let mut failures = 0;
    let selected = positions(exhaustive);
    for seed in SEEDS {
        let mut stats = Stats {
            table_bit_comparison: compare_table,
            ..Stats::default()
        };
        for chunk in selected.chunks(rows) {
            let live = chunk.len();
            let input: Vec<_> = chunk
                .iter()
                .flat_map(|&position| seeded_bf16(stride, seed ^ ((position as u64 + 1) << 24)))
                .collect();
            f.bf16("qkv", &input)?;
            f.i32(
                "positions",
                &chunk.iter().map(|&p| p as i32).collect::<Vec<_>>(),
            )?;
            f.u32("live_tokens", &[live as u32])?;
            // BF16 NaNs detect unwritten outputs, including exact zero references.
            f.bf16("q", &vec![0x7fc1; rows * QH * QK])?;
            f.bf16("k_new", &vec![0x7fc1; rows * kvh * QK])?;
            f.launch(module, entry, rows as u32, live as u32)?;
            let q = f.read_bf16("q", live * QH * QK)?;
            let k = f.read_bf16("k_new", live * kvh * QK)?;
            if compare_table {
                f.launch(&baseline, table_entry, rows as u32, live as u32)?;
                let table_q = f.read_bf16("q", q.len())?;
                let table_k = f.read_bf16("k_new", k.len())?;
                let differences = q
                    .iter()
                    .chain(&k)
                    .zip(table_q.iter().chain(&table_k))
                    .filter(|(a, b)| a != b)
                    .count() as u64;
                stats.table_bit_differences += differences;
                if arm() == "computed" {
                    stats.failures += differences;
                }
            }
            for (row, &position) in chunk.iter().enumerate() {
                let heads = QH + kvh;
                let exact = hf_rope_bf16(
                    &as_f64(&input[row * stride..][..heads * QK]),
                    &vec![position; heads],
                    QK,
                    f64::from(theta),
                )?;
                for head in 0..heads {
                    let source = &input[row * stride + head * QK..][..QK];
                    let actual = if head < QH {
                        &q[(row * QH + head) * QK..][..QK]
                    } else {
                        &k[(row * kvh + head - QH) * QK..][..QK]
                    };
                    let exact = &exact[head * QK..][..QK];
                    for pair in 0..PAIRS {
                        stats.check_selected(actual[pair], exact[pair], position, pair);
                        stats.check_selected(
                            actual[pair + PAIRS],
                            exact[pair + PAIRS],
                            position,
                            pair,
                        );
                    }
                    if actual[64..] != source[64..] {
                        return Err(format!(
                            "{name}: nonrotary dimensions changed at {position}"
                        ));
                    }
                }
            }
        }
        stats.receipt(session, name, rows, seed, exhaustive)?;
        failures += stats.failures;
    }
    Ok(failures)
}

#[test]
fn row_rope_position_sweep() {
    gpu_test(2 << 30, |session| {
        let module = super::module(session)?;
        let entries = catalog();
        let exhaustive = std::env::var("CHOPINFER_ROPE_EXHAUSTIVE").as_deref() == Ok("1");
        let mut failures = 0;
        for kind in [if uses_tables() { "table" } else { "computed" }] {
            for suffix in ["ga_bf16", "ga_fp8", "swa_decode", "swa_prefill"] {
                let page = if suffix.starts_with("ga") { 64 } else { 32 };
                let name = format!("rope_mimo_{suffix}_{kind}_p{page}_pos32");
                for rows in [1, 32, 128] {
                    let entry = find(&entries, &name).clone();
                    failures += run_case(
                        session,
                        &module,
                        &entry,
                        &name,
                        rows,
                        exhaustive && rows > 1,
                    )?;
                }
            }
        }
        if failures != 0 {
            return Err(format!(
                "RoPE sweep: {failures} elements exceed the existing bound; see receipts"
            ));
        }
        Ok(())
    })
    .expect("RoPE positions through the MiMo context limit");
}

#[test]
fn position_selection_covers_failures_and_context_edges() {
    let selected = positions(false);
    for p in [0, 36, 46, 127, 511, 4095, 65535, 262143, MAX_POSITION] {
        assert!(selected.binary_search(&p).is_ok(), "missing {p}");
    }
    let (index, count) = shard();
    let exhaustive = positions(true);
    assert_eq!(exhaustive.len(), (MAX_POSITION + 1 - index).div_ceil(count));
    assert!(
        exhaustive
            .iter()
            .all(|&p| p <= MAX_POSITION && p % count == index)
    );
}

/// Baseline and exact-table launch costs. Timings do not qualify a failing
/// arithmetic path; the separate position sweep supplies correctness evidence.
#[test]
fn row_rope_position_costs() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        eprintln!("set CHOPINFER_KERNEL_BENCH=1 to measure RoPE launch costs");
        return;
    }
    gpu_test(4 << 30, |session| {
        let module = super::module(session)?;
        let entries = catalog();
        for suffix in ["ga_bf16", "ga_fp8", "swa_decode", "swa_prefill"] {
            for kind in [if uses_tables() { "table" } else { "computed" }] {
                let ring = suffix.starts_with("swa");
                let kvh = if ring { 2 } else { 1 };
                let page = if ring { 32 } else { 64 };
                let theta = if ring { 1e4_f32 } else { 1e7_f32 };
                let name = format!("rope_mimo_{suffix}_{kind}_p{page}_pos32");
                let entry = find(&entries, &name);
                let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
                for rows in [1, 2, 4, 8, 16, 32, 128, 512, 2048] {
                    let mut f = setup(
                        session,
                        rows,
                        ring,
                        suffix.contains("fp8"),
                        theta,
                        kind == "table",
                    )?;
                    let prefill_shape = rows > 32 && suffix != "swa_decode";
                    if prefill_shape {
                        f.i32("token_batch", &vec![0; rows])?;
                        // One chunk in one sequence, with distinct physical
                        // pages for every logical page touched by the chunk.
                        f.i32(
                            "block_tables",
                            &(0..=MAX_POSITION / page)
                                .map(|p| (p % rows) as i32)
                                .collect::<Vec<_>>(),
                        )?;
                    }
                    f.bf16(
                        "qkv",
                        &seeded_bf16(rows * (QH * QK + kvh * (QK + V)), SEEDS[0]),
                    )?;
                    f.i32(
                        "positions",
                        &(MAX_POSITION + 1 - rows..=MAX_POSITION)
                            .map(|p| p as i32)
                            .collect::<Vec<_>>(),
                    )?;
                    let graph = session
                        .capture(|_| f.launch_bound(&function, entry, rows as u32, rows as u32))?;
                    let timing = time(session, 3, 15, |stream| {
                        graph.launch(stream).map_err(|e| e.to_string())
                    })?;
                    session.receipt(
                        "rope_cost",
                        BTreeMap::from([
                            ("entry".into(), json!(name)),
                            ("rows".into(), json!(rows)),
                            ("position_end".into(), json!(MAX_POSITION)),
                            ("arm".into(), json!(arm())),
                            ("single_sequence_chunk".into(), json!(prefill_shape)),
                            ("median_us".into(), json!(timing.median_ms * 1000.0)),
                            ("min_us".into(), json!(timing.min_ms * 1000.0)),
                            ("rounds".into(), json!(timing.rounds)),
                            (
                                "scope".into(),
                                json!(
                                    "isolated captured RoPE launch, cold L2; not end-to-end serving"
                                ),
                            ),
                        ]),
                    )?;
                }
            }
        }
        Ok(())
    })
    .expect("RoPE launch cost measurements");
}

pub(crate) fn arm() -> String {
    let arm = std::env::var("CHOPINFER_ROPE_ARM").unwrap_or_else(|_| "served".into());
    assert!(
        ["baseline", "served", "table", "computed"].contains(&arm.as_str()),
        "unknown RoPE arm"
    );
    arm
}

pub(crate) fn uses_tables() -> bool {
    table_arm(&arm())
}

fn table_arm(arm: &str) -> bool {
    matches!(arm, "served" | "table")
}

#[test]
fn served_arm_selects_only_table_paths() {
    assert!(table_arm("served"));
    assert!(table_arm("table"));
    for arm in ["computed", "baseline"] {
        assert!(!table_arm(arm));
    }
}

#[test]
fn dflash_rope_position_sweep() {
    gpu_test(1 << 30, |session| {
        let module = super::module(session)?;
        let entries = catalog();
        let exhaustive = std::env::var("CHOPINFER_ROPE_EXHAUSTIVE").as_deref() == Ok("1");
        let mut failures = 0;
        for (kind, heads) in [("q", 16), ("k", 2)] {
            let name = format!(
                "dflash_{kind}_rope{}_pos32",
                if uses_tables() { "_table" } else { "" }
            );
            let entry = find(&entries, &name).clone();
            for rows in [1, 32, 128] {
                let mut f = Fixture::new(session, rows)?;
                if uses_tables() {
                    let values: Vec<f32> = (0..=MAX_POSITION)
                        .flat_map(|p| {
                            (0..32).flat_map(move |pair| {
                                let (c, s) = kernel_core::hf_rope::cos_sin(p, pair, 1e4);
                                [c as f32, s as f32]
                            })
                        })
                        .collect();
                    f.f32("cos_sin", &values)?;
                }
                f.scalar("input_stride", (heads * 128) as u32);
                f.scalar("output_stride", (heads * 128) as u32);
                for seed in SEEDS {
                    let mut stats = Stats::default();
                    for chunk in positions(exhaustive && rows > 1).chunks(rows) {
                        let input: Vec<_> = chunk
                            .iter()
                            .flat_map(|&p| seeded_bf16(heads * 128, seed ^ ((p as u64 + 1) << 24)))
                            .collect();
                        f.bf16("input", &input)?;
                        f.bf16("output", &vec![0x7fc1; input.len()])?;
                        f.i32(
                            "positions",
                            &chunk.iter().map(|&p| p as i32).collect::<Vec<_>>(),
                        )?;
                        f.u32("live_tokens", &[chunk.len() as u32])?;
                        f.launch(&module, &entry, rows as u32, chunk.len() as u32)?;
                        let actual = f.read_bf16("output", input.len())?;
                        let expanded: Vec<_> = chunk
                            .iter()
                            .flat_map(|&p| std::iter::repeat_n(p, heads))
                            .collect();
                        let expected = hf_rope_bf16(&as_f64(&input), &expanded, 128, 1e4)?;
                        for (row, &p) in expanded.iter().enumerate() {
                            for d in 0..64 {
                                stats.check_selected(
                                    actual[row * 128 + d],
                                    expected[row * 128 + d],
                                    p,
                                    d % 32,
                                );
                            }
                            if actual[row * 128 + 64..(row + 1) * 128]
                                != input[row * 128 + 64..(row + 1) * 128]
                            {
                                return Err("DFlash changed nonrotary dimensions".into());
                            }
                        }
                    }
                    stats.receipt(session, &name, rows, seed, exhaustive && rows > 1)?;
                    failures += stats.failures;
                }
            }
        }
        if failures > 0 {
            return Err(format!("DFlash RoPE: {failures} elements exceed the bound"));
        }
        Ok(())
    })
    .expect("DFlash Q/K position sweep");
}

#[test]
fn dflash_rope_position_costs() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        return;
    }
    gpu_test(1 << 30, |session| {
        let module = super::module(session)?;
        let entries = catalog();
        for (kind, heads) in [("q", 16), ("k", 2)] {
            let name = format!(
                "dflash_{kind}_rope{}_pos32",
                if uses_tables() { "_table" } else { "" }
            );
            let entry = find(&entries, &name).clone();
            let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
            for rows in [1, 2, 4, 8, 16, 32, 128, 512, 2048] {
                let mut f = Fixture::new(session, rows)?;
                if uses_tables() {
                    let values: Vec<f32> = (0..=MAX_POSITION)
                        .flat_map(|p| {
                            (0..32).flat_map(move |pair| {
                                let (c, s) = kernel_core::hf_rope::cos_sin(p, pair, 1e4);
                                [c as f32, s as f32]
                            })
                        })
                        .collect();
                    f.f32("cos_sin", &values)?;
                }
                f.scalar("input_stride", (heads * 128) as u32);
                f.scalar("output_stride", (heads * 128) as u32);
                f.bf16("input", &seeded_bf16(rows * heads * 128, SEEDS[0]))?;
                f.zero("output", rows * heads * 128 * 2)?;
                f.i32(
                    "positions",
                    &(MAX_POSITION + 1 - rows..=MAX_POSITION)
                        .map(|p| p as i32)
                        .collect::<Vec<_>>(),
                )?;
                let graph = session
                    .capture(|_| f.launch_bound(&function, &entry, rows as u32, rows as u32))?;
                let timing = time(session, 3, 15, |stream| {
                    graph.launch(stream).map_err(|e| e.to_string())
                })?;
                session.receipt(
                    "dflash_rope_cost",
                    BTreeMap::from([
                        ("arm".into(), json!(arm())),
                        ("entry".into(), json!(name)),
                        ("rows".into(), json!(rows)),
                        ("position_end".into(), json!(MAX_POSITION)),
                        ("median_us".into(), json!(timing.median_ms * 1000.0)),
                    ]),
                )?;
            }
        }
        Ok(())
    })
    .expect("DFlash RoPE costs");
}

#[test]
fn hf_frequency_bits_match_cuda_constants() {
    let header = include_str!("../../../kernels-sm120/cuda/common/rotary.cuh");
    let array = header
        .split("hf_inverse_bits[2][32] = {")
        .nth(1)
        .unwrap()
        .split("};")
        .next()
        .unwrap();
    let actual: Vec<u32> = array
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | '{' | '}'))
        .filter(|s| s.starts_with("0x"))
        .map(|s| u32::from_str_radix(s.trim_start_matches("0x").trim_end_matches('u'), 16).unwrap())
        .collect();
    assert_eq!(
        actual,
        kernel_core::hf_rope::INVERSE_BITS
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    );
}

#[test]
fn exact_checks_reject_adjacent_bits_and_signed_zero() {
    for (actual, expected) in [(0x3801, 0x3800), (0x8000, 0x0000)] {
        let mut stats = Stats::default();
        stats.check(actual, f64::from(bf16_to_f32(expected)), 511, 1);
        assert_eq!(stats.oracle_bit_differences, 1);
        assert!(
            stats.failures > 0,
            "exact contract must not depend on an environment flag"
        );
    }
}
