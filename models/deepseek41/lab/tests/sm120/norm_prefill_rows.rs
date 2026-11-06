//! GPU lab exactness and pair timing of the fused prefill norm-ceil
//! (`sm120::norm_prefill`, the served cached entry) against the unfused
//! chain it replaces at the rows `norm.input_admit.prefill` admits: the row
//! RMSNorm, the rounding widen of its rows and the serial sublayer input
//! quantizer (`sm120::norm_prefill::chain`).
//!
//! The ignored `norm_prefill_rows_bitwise_every_row` test, on one GPU,
//! compares the two at every row count the table admits, 33 to 8192, or
//! `CHOPINFER_NORM_ROWS_SWEEP=FIRST..=LAST`: the normed BF16 rows, widened
//! F32 rows, codes, scales and row status must be equal bit for bit, a guard
//! row and the guard status words must keep each arm's poison, and every row
//! with a nonfinite input must be refused; the embedded cubins must match
//! their lock records. The ignored `norm_prefill_rows_pair` test, on two
//! GPUs with a 295-second process timeout, repeats that check at the served
//! rows and then times an A/A control (the chain in both arms) and the A/B
//! pair in alternating order.
use bundle_deepseek41::sm120;
use kernel_core::{
    contract::{Argument, ArgumentList, LaunchPlan},
    rmsnorm::RmsNormBuffers,
    rmsnorm_quantize::RmsNormQuantizeBuffers,
};
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch::Bindings};
use kernels_sm120::row::{
    block_quantize,
    rmsnorm::{RmsNorm, RmsNormTarget},
    rmsnorm_prefill, rounding,
};
use model_deepseek41_contract::{geometry::FLASH, rows, serving::PREFILL_TOKENS};
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::json;
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

/// Status words before the first row, so a stray write is visible.
const OFFSET: usize = 3;
/// The first row count the table admits: fewer rows take the
/// small-bucket quantizer in the chain.
const FIRST: u32 = 33;
/// Rows `r` with `r % REFUSED == 1` carry one nonfinite input value.
const REFUSED: usize = 61;
/// Each arm's poison byte (scales and widened rows take its complement), so
/// a byte either arm leaves unwritten differs between them.
const POISON: [u8; 2] = [0xa5, 0x3c];

/// The served fused entry: the cached strategy the Flash table selects.
fn fused() -> rmsnorm_prefill::Entry {
    let entry = sm120::norm_prefill::entries()
        .unwrap()
        .into_iter()
        .find(|e| e.cache_groups)
        .unwrap();
    assert_eq!(
        entry.reference.symbol,
        "chop_prefill_norm_ceil_deepseek41_cached_v1"
    );
    entry
}

/// The chain's three entries at the hidden width.
struct Chain {
    norm: RmsNorm,
    norm_symbol: String,
    widen_symbol: String,
    ceil: block_quantize::Entry,
}

fn chain() -> Chain {
    let (norm_symbol, widen_symbol, ceil) = sm120::norm_prefill::chain(&FLASH, 4).unwrap();
    Chain {
        norm: RmsNorm::new(
            rows::norm(&FLASH, FLASH.shape.hidden),
            PREFILL_TOKENS,
            RmsNormTarget::SM120F,
        )
        .unwrap(),
        norm_symbol,
        widen_symbol,
        ceil,
    }
}

/// The sweep's row counts: `FIRST..=PREFILL_TOKENS`, or a part of it.
fn sweep_rows() -> LabResult<std::ops::RangeInclusive<u32>> {
    let Some(value) = std::env::var_os("CHOPINFER_NORM_ROWS_SWEEP") else {
        return Ok(FIRST..=PREFILL_TOKENS);
    };
    let usage = "CHOPINFER_NORM_ROWS_SWEEP must be FIRST..=LAST within 33..=8192";
    let value = value.into_string().map_err(|_| usage)?;
    let (first, last) = value.split_once("..=").ok_or(usage)?;
    let (first, last) = (
        first.parse::<u32>().map_err(|_| usage)?,
        last.parse::<u32>().map_err(|_| usage)?,
    );
    if first < FIRST || last > PREFILL_TOKENS || first > last {
        return Err(usage.into());
    }
    Ok(first..=last)
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// BF16 rows over a wide exponent range, with one nonfinite value in each
/// row `r` with `r % REFUSED == 1`: infinity and NaN alternate, each in a
/// different group of its row.
fn input(width: usize, rows: usize) -> Vec<u8> {
    let mut state = 0x6e0f_ac71_u64;
    let mut values: Vec<u16> = (0..width * rows)
        .map(|_| {
            let s = xorshift(&mut state);
            let bits = (s as u32 & 0x8000_0000)
                | (((s >> 32) as u32 % 60 + 97) << 23)
                | (s as u32 & 0x7f_ffff);
            (bits >> 16) as u16
        })
        .collect();
    for row in (1..rows).step_by(REFUSED) {
        let group = (row * 37) % (width / 32);
        values[row * width + group * 32 + 5] = if (row / REFUSED) % 2 == 0 {
            0x7f80
        } else {
            0x7fc0
        };
    }
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Finite BF16 norm weights of either sign.
fn weight(width: usize) -> Vec<u8> {
    let mut state = 0x91e1_0a2b_u64;
    (0..width)
        .flat_map(|_| {
            let s = xorshift(&mut state);
            let bits = (s as u32 & 0x8000_0000)
                | ((s >> 32) as u32 % 6 + 124) << 23
                | (s as u32 & 0x7f_ffff);
            ((bits >> 16) as u16).to_ne_bytes()
        })
        .collect()
}

/// One arm's output planes: normed BF16, widened F32, codes, scales and
/// status words with `OFFSET` guard words before the first row.
struct Planes {
    normed: DeviceBuffer,
    widened: DeviceBuffer,
    codes: DeviceBuffer,
    scales: DeviceBuffer,
    status: DeviceBuffer,
}

/// Each plane's bytes per row.
fn row_bytes(width: usize) -> [usize; 4] {
    [width * 2, width * 4, width, width / 32]
}

impl Planes {
    fn new(s: &Session, width: usize, rows: usize) -> LabResult<Self> {
        let [normed, widened, codes, scales] = row_bytes(width).map(|b| b * rows);
        Ok(Self {
            normed: s.allocate(normed)?,
            widened: s.allocate(widened)?,
            codes: s.allocate(codes)?,
            scales: s.allocate(scales)?,
            status: s.allocate((rows + OFFSET + 2) * 4)?,
        })
    }
    fn planes(&self) -> [&DeviceBuffer; 4] {
        [&self.normed, &self.widened, &self.codes, &self.scales]
    }
}

fn function(s: &Session, symbol: &str) -> LabResult<Function> {
    let (_, cubin) =
        sm120::fold_cubin_for_symbol(symbol).ok_or_else(|| format!("missing cubin: {symbol}"))?;
    let module = s.load_cubin(cubin)?;
    module.function(symbol).map_err(|e| e.to_string())
}

/// `symbol`'s embedded cubin and its digest, which must match the cubin's
/// lock record.
fn locked_cubin(symbol: &str) -> LabResult<(String, String)> {
    let (name, bytes) =
        sm120::fold_cubin_for_symbol(symbol).ok_or_else(|| format!("missing cubin: {symbol}"))?;
    let digest = kernel_build::digest(bytes);
    let locked = [
        include_str!("../../../bundle/kernels.lock.d/norm_prefill_fused.lock"),
        include_str!("../../../bundle/kernels.lock.d/rows.lock"),
        include_str!("../../../bundle/kernels.lock.d/row_rounding.lock"),
        include_str!("../../../bundle/kernels.lock.d/routed_moe.lock"),
    ]
    .iter()
    .flat_map(|text| serde_json::from_str::<Vec<serde_json::Value>>(text).unwrap())
    .any(|record| record["name"] == name && record["cubin_sha256"] == digest.as_str());
    if !locked {
        return Err(format!("{name} differs from its lock"));
    }
    Ok((name.to_owned(), digest))
}

/// The handles the arms' arguments name, bound to their buffers.
mod handle {
    pub const INPUT: u64 = 1;
    pub const WEIGHT: u64 = 2;
    pub const LIVE: u64 = 3;
    /// An arm's planes: `base + [normed, widened, codes, scales, status]`.
    pub const fn arm(arm: usize) -> u64 {
        10 * (arm as u64 + 1)
    }
}

/// The fused arm (0) or the chain (1), each launch with its symbol, plan
/// and arguments.
fn launches(
    arm: usize,
    fused: &rmsnorm_prefill::Entry,
    chain: &Chain,
    rows: u32,
) -> LabResult<Vec<(String, LaunchPlan, ArgumentList)>> {
    let base = handle::arm(arm);
    let [normed, widened, codes, scales, status] = [0, 1, 2, 3, 4].map(|i| base + i);
    let e = |r: kernel_core::contract::Refusal| r.to_string();
    if arm == 0 {
        let arguments = fused
            .reference
            .arguments(RmsNormQuantizeBuffers {
                input: handle::INPUT,
                weight: handle::WEIGHT,
                normed,
                widened: Some(widened),
                codes,
                scales,
                live_rows: handle::LIVE,
                status,
                status_row_offset: OFFSET as u64,
            })
            .map_err(e)?;
        return Ok(vec![(
            fused.reference.symbol.clone(),
            fused.plan(rows).map_err(e)?,
            arguments,
        )]);
    }
    let norm = chain
        .norm
        .arguments(RmsNormBuffers {
            x: handle::INPUT,
            weight: handle::WEIGHT,
            residual: None,
            y: normed,
            live_rows: handle::LIVE,
            status: None,
            status_row_offset: 0,
        })
        .map_err(e)?;
    let ceil = ArgumentList(vec![
        Argument::DeviceBuffer(normed),
        Argument::DeviceBuffer(codes),
        Argument::DeviceBuffer(scales),
        Argument::DeviceBuffer(handle::LIVE),
        Argument::DeviceBuffer(status),
        Argument::U64(OFFSET as u64),
    ]);
    Ok(vec![
        (
            chain.norm_symbol.clone(),
            chain.norm.plan(rows).map_err(e)?,
            norm,
        ),
        (
            chain.widen_symbol.clone(),
            rounding::plan(rows).map_err(e)?,
            rounding::arguments(normed, widened, handle::LIVE).map_err(e)?,
        ),
        (
            chain.ceil.symbol.clone(),
            chain.ceil.plan(rows).map_err(e)?,
            ceil,
        ),
    ])
}

#[test]
fn the_sweep_covers_every_admitted_row_with_the_served_entries() {
    assert_eq!(sweep_rows().unwrap(), 33..=8192);
    let (fused, chain) = (fused(), chain());
    assert_eq!(
        fused.reference.query.norm.epsilon_bits,
        rows::norm(&FLASH, FLASH.shape.hidden).epsilon_bits
    );
    for symbol in [
        &fused.reference.symbol,
        &chain.norm_symbol,
        &chain.widen_symbol,
        &chain.ceil.symbol,
    ] {
        assert!(locked_cubin(symbol).is_ok(), "{symbol}");
    }
    for rows in [FIRST, 1500, PREFILL_TOKENS] {
        assert_eq!(launches(0, &fused, &chain, rows).unwrap().len(), 1);
        assert_eq!(launches(1, &fused, &chain, rows).unwrap().len(), 3);
    }
    let width = FLASH.shape.hidden as usize;
    let bytes = input(width, 200);
    for row in 0..200 {
        let nonfinite = (0..width)
            .filter(|&c| {
                let bits = u16::from_ne_bytes([
                    bytes[(row * width + c) * 2],
                    bytes[(row * width + c) * 2 + 1],
                ]);
                !f32::from_bits(u32::from(bits) << 16).is_finite()
            })
            .count();
        assert_eq!(nonfinite, usize::from(row % REFUSED == 1), "row {row}");
    }
}

/// One device's buffers for both arms at full capacity, the loaded entry
/// functions and each arm's poison planes.
struct Rig {
    functions: BTreeMap<String, Function>,
    input: DeviceBuffer,
    weight: DeviceBuffer,
    live: DeviceBuffer,
    planes: [Planes; 2],
    poison: Vec<Vec<DeviceBuffer>>,
    fused: rmsnorm_prefill::Entry,
    chain: Chain,
    host: Vec<Vec<Vec<u8>>>,
}

impl Rig {
    fn new(s: &Session) -> LabResult<Self> {
        let (fused, chain) = (fused(), chain());
        let width = FLASH.shape.hidden as usize;
        let capacity = PREFILL_TOKENS as usize;
        let symbols = [
            &fused.reference.symbol,
            &chain.norm_symbol,
            &chain.widen_symbol,
            &chain.ceil.symbol,
        ];
        let functions = symbols
            .iter()
            .map(|symbol| Ok((symbol.to_string(), function(s, symbol)?)))
            .collect::<LabResult<_>>()?;
        let upload = |bytes: &[u8]| -> LabResult<DeviceBuffer> {
            let buffer = s.allocate(bytes.len())?;
            buffer
                .copy_from_bytes(bytes, &s.stream)
                .map_err(|e| e.to_string())?;
            Ok(buffer)
        };
        let sizes = row_bytes(width);
        let poison = POISON
            .iter()
            .map(|&value| {
                sizes
                    .iter()
                    .enumerate()
                    .map(|(plane, &bytes)| {
                        upload(&vec![
                            if plane % 2 == 0 { value } else { !value };
                            bytes * capacity
                        ])
                    })
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            functions,
            input: upload(&input(width, capacity))?,
            weight: upload(&weight(width))?,
            live: s.allocate(4)?,
            planes: [
                Planes::new(s, width, capacity)?,
                Planes::new(s, width, capacity)?,
            ],
            poison,
            fused,
            chain,
            host: (0..2)
                .map(|_| {
                    sizes
                        .iter()
                        .map(|b| vec![0; b * capacity])
                        .chain([vec![0; (capacity + OFFSET + 2) * 4]])
                        .collect()
                })
                .collect(),
        })
    }

    fn bindings(&self) -> Bindings<'_> {
        let mut buffers = BTreeMap::from([
            (handle::INPUT, &self.input),
            (handle::WEIGHT, &self.weight),
            (handle::LIVE, &self.live),
        ]);
        for (arm, out) in self.planes.iter().enumerate() {
            for (i, buffer) in out.planes().into_iter().chain([&out.status]).enumerate() {
                buffers.insert(handle::arm(arm) + i as u64, buffer);
            }
        }
        Bindings { buffers }
    }

    /// Issues `arm`'s launches over `rows` rows on `stream`.
    fn issue(
        &self,
        s: &Session,
        arm: usize,
        rows: u32,
        stream: &runtime_cuda::Stream,
    ) -> LabResult<()> {
        let bindings = self.bindings();
        for (symbol, plan, arguments) in launches(arm, &self.fused, &self.chain, rows)? {
            kernel_harness::launch::launch(
                &s.context,
                &self.functions[&symbol],
                stream,
                plan,
                &arguments,
                &bindings,
            )?;
        }
        Ok(())
    }

    /// Poisons both arms over `rows` rows and a guard row, runs both, and
    /// checks every output byte, the guard row and guard words, and the
    /// refusal of every nonfinite row. Returns the refused rows checked.
    fn compare(&mut self, s: &Session, count: u32) -> LabResult<usize> {
        let width = FLASH.shape.hidden as usize;
        let capacity = PREFILL_TOKENS as usize;
        let sizes = row_bytes(width);
        let n = count as usize;
        let guarded = (n + 1).min(capacity);
        let mut words = vec![0x1357_2468u32; n + OFFSET + 2];
        words[OFFSET..OFFSET + n].fill(0);
        let status: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        self.live
            .copy_from_bytes(&count.to_ne_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        for arm in 0..2 {
            let out = &self.planes[arm];
            for (plane, buffer) in out.planes().into_iter().enumerate() {
                buffer
                    .copy_range_from_async(
                        0,
                        &self.poison[arm][plane],
                        0,
                        guarded * sizes[plane],
                        &s.stream,
                    )
                    .map_err(|e| e.to_string())?;
            }
            out.status
                .copy_from_bytes(&status, &s.stream)
                .map_err(|e| e.to_string())?;
            self.issue(s, arm, count, &s.stream)?;
        }
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let lens: Vec<usize> = sizes
            .iter()
            .map(|b| guarded * b)
            .chain([status.len()])
            .collect();
        for arm in 0..2 {
            let out = &self.planes[arm];
            for (i, buffer) in out.planes().into_iter().chain([&out.status]).enumerate() {
                buffer
                    .copy_to_bytes(&mut self.host[arm][i][..lens[i]], &s.stream)
                    .map_err(|e| e.to_string())?;
            }
        }
        let (a, b) = (&self.host[0], &self.host[1]);
        for (plane, role) in ["normed", "widened", "codes", "scales"].iter().enumerate() {
            let written = n * sizes[plane];
            if a[plane][..written] != b[plane][..written] {
                let at = a[plane]
                    .iter()
                    .zip(&b[plane])
                    .position(|(x, y)| x != y)
                    .unwrap();
                return Err(format!(
                    "rows={count}: {role} differ in row {} at byte {at}",
                    at / sizes[plane]
                ));
            }
            for (arm, value) in POISON.iter().enumerate() {
                let value = if plane % 2 == 0 { *value } else { !*value };
                if self.host[arm][plane][written..lens[plane]]
                    .iter()
                    .any(|&x| x != value)
                {
                    return Err(format!(
                        "rows={count}: arm {arm} wrote {role} past the last row"
                    ));
                }
            }
        }
        if a[4][..lens[4]] != b[4][..lens[4]] {
            return Err(format!("rows={count}: status words differ"));
        }
        let word = |i: usize| u32::from_ne_bytes(a[4][i * 4..(i + 1) * 4].try_into().unwrap());
        if !(0..OFFSET)
            .chain([OFFSET + n, OFFSET + n + 1])
            .all(|i| word(i) == 0x1357_2468)
        {
            return Err(format!("rows={count}: a status guard word changed"));
        }
        let mut refused = 0;
        for row in (1..n).step_by(REFUSED) {
            if word(OFFSET + row) == 0
                || a[2][row * width..(row + 1) * width].iter().any(|&c| c != 0)
            {
                return Err(format!("rows={count}: nonfinite row {row} was not refused"));
            }
            refused += 1;
        }
        Ok(refused)
    }
}

#[test]
#[ignore = "GPU lab: one idle GPU, bitwise only, about eight minutes"]
fn norm_prefill_rows_bitwise_every_row() {
    let start = Instant::now();
    let rows = sweep_rows().unwrap();
    let result = kernel_harness::gpu_test(3 << 30, |s| {
        let initial = gpu_snapshot(s)?;
        let mut rig = Rig::new(s)?;
        let (mut cases, mut refused) = (0usize, 0usize);
        for count in rows.clone() {
            if start.elapsed() >= Duration::from_secs(1500) {
                return Err(format!(
                    "time budget exhausted at rows={count}; no complete result"
                ));
            }
            refused += rig.compare(s, count)?;
            cases += 1;
            if count % 512 == 0 || count == *rows.end() {
                println!(
                    "norm-rows sweep: rows {}..={count} bitwise, {:.0} s",
                    rows.start(),
                    start.elapsed().as_secs_f64()
                );
            }
        }
        let locked = rig
            .functions
            .keys()
            .map(|symbol| {
                locked_cubin(symbol).map(
                    |(name, digest)| json!({"symbol": symbol, "cubin": name, "sha256": digest}),
                )
            })
            .collect::<LabResult<Vec<_>>>()?;
        s.receipt(
            "deepseek41-norm-prefill-rows-sweep",
            BTreeMap::from([
                ("rows".into(), json!([rows.start(), rows.end()])),
                ("cubins".into(), json!(locked)),
                ("cases".into(), json!(cases)),
                ("refused_rows_checked".into(), json!(refused)),
                ("seconds".into(), json!(start.elapsed().as_secs_f64())),
                ("gpu_start".into(), initial),
                ("gpu_end".into(), gpu_snapshot(s)?),
            ]),
        )?;
        println!(
            "NORM_ROWS_SWEEP_COMPLETE rows={}..={} cases={cases} refused={refused}",
            rows.start(),
            rows.end()
        );
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "the bitwise sweep must run");
}

/// The pair timing's rows: the profiled 512-row pass, the 1.5K-3K served
/// range with unaligned counts, and the last count below the measured
/// full chunk.
const PAIR_ROWS: [u32; 9] = [512, 1500, 1536, 2048, 2300, 2304, 3000, 3072, 8191];

/// Pair-lab timing of the fused launch against the chain it replaces at
/// `PAIR_ROWS`, after a bitwise pre-check at each count, with three visits of
/// an A/A control (the chain's graph in both arms) and the A/B pair in
/// alternating order on each of two GPUs.
#[test]
#[ignore = "pair-lab: two idle GPUs, CUDA events, A/A control, at most five minutes"]
fn norm_prefill_rows_pair() {
    assert_eq!(
        kernel_harness::selected_ordinals()
            .unwrap()
            .expect("select two idle GPUs")
            .len(),
        2
    );
    let start = Instant::now();
    kernel_harness::gpu_test_multi_with(probe().unwrap(), 3 << 30, |sessions| {
        let job_start = sessions.iter().map(gpu_snapshot).collect::<LabResult<Vec<_>>>()?;
        for (session, initial) in sessions.iter().zip(&job_start) {
            let limit = initial["enforced_power_w"]
                .as_f64()
                .ok_or("missing enforced power limit")?;
            let snapshot = || -> LabResult<serde_json::Value> {
                let state = gpu_snapshot(session)?;
                if state["enforced_power_w"].as_f64() != Some(limit) {
                    return Err("power limit changed; pair-lab job is void".into());
                }
                Ok(state)
            };
            let s = session;
            let mut rig = Rig::new(s)?;
            let mut results = Vec::new();
            for rows in PAIR_ROWS {
                rig.compare(s, rows)?;
                let rig = &rig;
                let graphs = (0..2)
                    .map(|arm| s.capture(|stream| rig.issue(s, arm, rows, stream)))
                    .collect::<LabResult<Vec<_>>>()?;
                let label = format!("norm-rows:rows={rows}");
                let mut visits = Vec::new();
                for visit in 0..3 {
                    if start.elapsed() >= Duration::from_secs(280) {
                        return Err("pair-lab time budget exhausted; no complete result".into());
                    }
                    // Alternate the control/comparison order each visit. Both A/A arms
                    // issue the very same chain graph over the very same operands.
                    let visit_start = snapshot()?;
                    let mut timings = Vec::new();
                    for step in 0..2 {
                        let control = (visit + step) % 2 == 0;
                        let timing = kernel_harness::with_timing_label(
                            &format!("{label}-{}", if control { "aa" } else { "ab" }),
                            || {
                                kernel_harness::time_ab(s, 1, 5, [EvictionMode::Read; 2], |arm, stream| {
                                    graphs[if control { 1 } else { 1 - arm }]
                                        .launch(stream)
                                        .map_err(|e| e.to_string())
                                })
                            },
                        )?;
                        timings.push((control, timing));
                    }
                    let aa = &timings.iter().find(|(control, _)| *control).unwrap().1;
                    let ab = &timings.iter().find(|(control, _)| !*control).unwrap().1;
                    let spread = (1.0 - aa.paired_ratio).abs() * 100.0;
                    let gain = (1.0 - ab.paired_ratio) * 100.0;
                    println!(
                        "device={} {label} visit={visit} chain_ms={} fused_ms={} gain_pct={gain} aa_spread_pct={spread}",
                        s.ordinal, ab.arms[0].median_ms, ab.arms[1].median_ms
                    );
                    if spread > 3.0 {
                        println!("{label}: wide A/A spread ({spread}%); rerun on an idle host");
                    }
                    let visit_end = snapshot()?;
                    visits.push(json!({"gpu_start":visit_start,"gpu_end":visit_end,"chain_ms":ab.arms[0].median_ms,
                        "fused_ms":ab.arms[1].median_ms,"gain_pct":gain,"aa_spread_pct":spread,"gain_exceeds_control":gain > spread}));
                }
                results.push(json!({"rows":rows,"bitwise":true,"visits":visits}));
            }
            s.receipt(
                "deepseek41-norm-prefill-rows-pair",
                BTreeMap::from([
                    ("results".into(), json!(results)),
                    ("gpu_start".into(), initial.clone()),
                    ("gpu_end".into(), snapshot()?),
                ]),
            )?;
        }
        let job_end = sessions.iter().map(gpu_snapshot).collect::<LabResult<Vec<_>>>()?;
        for (before, after) in job_start.iter().zip(&job_end) {
            if before["enforced_power_w"] != after["enforced_power_w"] {
                return Err("power limit changed across the two-device job; result is void".into());
            }
        }
        sessions[0].receipt(
            "deepseek41-norm-prefill-rows-pair-job",
            BTreeMap::from([
                ("gpu_start".into(), json!(job_start)),
                ("gpu_end".into(), json!(job_end)),
            ]),
        )?;
        Ok(())
    })
    .unwrap()
    .expect("the two-GPU probe must execute");
    assert!(start.elapsed() < Duration::from_secs(295));
}

fn probe() -> LabResult<Cubins> {
    let path = std::env::var_os("CHOPINFER_KERNEL_PROBE_CUBIN")
        .ok_or("set CHOPINFER_KERNEL_PROBE_CUBIN to the staged probe_sm120.cubin")?;
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let locks: Vec<serde_json::Value> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../crates/kernel-lab/kernels.lock"
    )))
    .map_err(|e| e.to_string())?;
    let digest = locks
        .iter()
        .find(|r| r["name"] == "probe_sm120")
        .and_then(|r| r["cubin_sha256"].as_str())
        .ok_or("missing locked timing probe")?;
    if kernel_build::digest(&bytes) != digest {
        return Err("timing probe differs from its lock".into());
    }
    Ok(Cubins::new(
        Box::leak(vec![("probe_sm120", &*Box::leak(bytes.into_boxed_slice()))].into_boxed_slice()),
        &[],
    ))
}

fn gpu_snapshot(session: &Session) -> LabResult<serde_json::Value> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=enforced.power.limit,clocks.sm",
            "--format=csv,noheader,nounits",
            "--id",
            session.metadata["uuid"]
                .as_str()
                .ok_or("missing device identity")?,
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("GPU telemetry query failed".into());
    }
    let text = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let values = text
        .trim()
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    if values.len() != 2 || values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("GPU power and clock telemetry is unavailable".into());
    }
    Ok(json!({"enforced_power_w":values[0], "sm_mhz":values[1]}))
}
