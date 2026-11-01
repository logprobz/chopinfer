//! A prototype of the composed form's deferred pair in one launch: where
//! the next HC mix takes the round, a decode or verify step issues the F32
//! reduce-scatter then the all-gather into every rank's F32 sum
//! ([`deferred`]); the phased F32-output entry
//! (`chop_comm_allreduce_phased_f32_v1`, the one-shot
//! `chop_comm_allreduce_phased_oneshot_f32_v1` at its rows) does both in
//! one launch. No serving path launches it, and the bundle's comm cubin
//! holds only the rounding entries, so the lab loads kernel-lab's locked
//! `comm_phased_f32_sm120` cubin, which `CHOPINFER_PHASED_F32_CUBIN` names
//! ([`lab_cubin`]).
//!
//! [`Deferred`] wraps the candidate registration's rounded all-reduce
//! (`phased_backend`): each row count's phased rounding launch becomes the
//! F32-output entry of the same protocol, senders and workers, which
//! writes every rank's F32 sum (a state region after the phased state)
//! instead of the BF16 outputs and sums every row, as the pair does, with
//! no live count. Its deferred round reads this rank's sum, as the
//! composed form's does, so the next HC mix's one round sees the entry's
//! sums. The entry sums each element's rank partials in F32 in rank order
//! from zero, the pair's bits (kernel-lab `comm_phased_f32`).
//! [`phased_deferred_matches_the_deferred_pair_on_gpu`] checks that at the
//! bundle's width on adversarial patterns, and the `timing` bench's
//! `phased_deferred_timing` times it against the pair.
use super::phased_parity::{
    CAPACITY, PATTERNS, ROWS, deferred, pattern, phased_backend, rounded_op, served_backend,
    served_symbols, sum_region, upload_bits,
};
use super::{Fixture, Op, WIDTH, err, regular, synchronize};
use kernel_core::{
    backend::{Counts, DeferredBf16Round, Launch, OpInfo, PreparedOp, View},
    collective::AllReduceBuffers,
    contract::{Argument, Refusal, Validate},
};
use kernel_harness::{LabResult, Session, topology};
use kernels_sm120::comm::phased_f32;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// The entries of kernel-lab's cubin: the rounding entries a complete step
/// launches and the F32-output entries a deferred step launches.
pub(super) const SYMBOLS: [&str; 4] = [
    phased_f32::ONE_SHOT_ROUND_SYMBOL,
    phased_f32::ROUND_SYMBOL,
    phased_f32::ONE_SHOT_SYMBOL,
    phased_f32::SYMBOL,
];
/// Row counts past [`ROWS`]: the verify widths whose phased rounding form
/// lost to the pair (6, 10, 13) and the C1 verify width (6).
const EXTRA_ROWS: [u32; 3] = [6, 10, 13];
/// The byte every sum is poisoned with before a call.
const POISON: u8 = 0xa5;

/// Kernel-lab's `comm_phased_f32_sm120` cubin that
/// `CHOPINFER_PHASED_F32_CUBIN` names, refused unless it is the one
/// kernel-lab's lock pins.
pub(super) fn lab_cubin() -> LabResult<&'static [u8]> {
    let path = std::env::var_os("CHOPINFER_PHASED_F32_CUBIN")
        .ok_or("CHOPINFER_PHASED_F32_CUBIN must name kernel-lab's comm_phased_f32_sm120 cubin")?;
    let bytes = regular(Path::new(&path))?;
    let lock: Value = serde_json::from_str(include_str!(
        "../../../../../../crates/kernel-lab/kernels.lock.d/comm_phased_f32.lock"
    ))
    .map_err(err)?;
    let pinned = lock
        .as_array()
        .and_then(|records| records.iter().find(|r| r["name"] == phased_f32::LAB_CUBIN))
        .ok_or("kernel-lab pins no phased F32 cubin")?;
    if pinned["cubin_sha256"] != kernel_build::digest(&bytes) {
        return Err("the phased F32 cubin differs from kernel-lab's lock".into());
    }
    let entries: Vec<&str> = pinned["entries"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry["symbol"].as_str())
        .collect();
    if SYMBOLS.iter().any(|symbol| !entries.contains(symbol)) {
        return Err("the phased F32 cubin lacks an entry".into());
    }
    Ok(Box::leak(bytes.into_boxed_slice()))
}

/// The F32-output entry of the rounding entry `symbol`'s protocol.
fn f32_entry(symbol: &str) -> Option<&'static str> {
    if symbol == phased_f32::ONE_SHOT_ROUND_SYMBOL {
        Some(phased_f32::ONE_SHOT_SYMBOL)
    } else if symbol == phased_f32::ROUND_SYMBOL {
        Some(phased_f32::SYMBOL)
    } else {
        None
    }
}

/// Rank `rank`'s rounded all-reduce whose deferred round follows the
/// phased F32-output entry.
pub(super) struct Deferred {
    inner: Op,
    rank: usize,
    /// The state region of every rank's F32 sum.
    sum: usize,
    info: OpInfo,
}

impl Deferred {
    pub(super) fn new(rank: usize) -> LabResult<Self> {
        let inner = rounded_op(&phased_backend(), rank)?;
        let mut info = inner.info().clone();
        let sum = info.state.len();
        info.state.push(u64::from(CAPACITY) * WIDTH as u64 * 4);
        info.implementation = format!("{}+deferred_f32", info.implementation);
        info.symbols = SYMBOLS.map(|symbol| (symbol.to_owned(), 0)).to_vec();
        Ok(Self {
            inner,
            rank,
            sum,
            info,
        })
    }

    /// The F32-output launch of `counts` rows and this rank's sum.
    fn call(
        &self,
        buffers: &AllReduceBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<(Launch, View), Refusal> {
        let launches = self.inner.launches(buffers, workspace, counts)?;
        let [mut launch] = <[Launch; 1]>::try_from(launches)
            .map_err(|_| Refusal("a phased form launches one entry"))?;
        launch.symbol = f32_entry(&launch.symbol)
            .ok_or(Refusal("a phased form launches a rounding entry"))?
            .into();
        let elements = u64::from(counts.rows) * WIDTH as u64;
        let sums = buffers
            .ranks
            .iter()
            .map(|rank| {
                let sum = *rank
                    .state
                    .get(self.sum)
                    .ok_or(Refusal("an all-reduce rank lacks its sum"))?;
                sum.covers(elements * 4)?;
                Ok(sum)
            })
            .collect::<Result<Vec<View>, Refusal>>()?;
        let own = sums[self.rank];
        let arguments = &mut launch.arguments.0;
        if arguments.len() != 12 {
            return Err(Refusal("the phased entries take twelve arguments"));
        }
        // Output, every rank's output and the live count.
        arguments[1] = Argument::DeviceBuffer(own.handle);
        arguments[3] = Argument::DevicePointerArray(sums.iter().map(|sum| sum.handle).collect());
        arguments[8] = Argument::NullDevicePointer;
        launch.arguments.validate()?;
        Ok((launch, own))
    }
}

impl PreparedOp<AllReduceBuffers<View>> for Deferred {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn defer_bf16_round(
        &self,
        buffers: &AllReduceBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Option<DeferredBf16Round>, Refusal> {
        let (launch, own) = self.call(buffers, workspace, counts)?;
        Ok(Some(DeferredBf16Round {
            launches: vec![launch],
            input: own,
            output: buffers.output,
            live_rows: buffers
                .live_rows
                .ok_or(Refusal("a deferred round reads a device live count"))?,
            width: WIDTH as u32,
        }))
    }

    /// A complete step: the phased rounding entry.
    fn launches(
        &self,
        buffers: &AllReduceBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.inner.launches(buffers, workspace, counts)
    }
}

/// Every rank's [`Deferred`] over kernel-lab's `cubin`.
pub(super) fn fixture(sessions: &[Session], cubin: &'static [u8]) -> LabResult<Fixture> {
    Fixture::with_cubins(
        sessions,
        CAPACITY,
        &SYMBOLS,
        |_| Ok(cubin),
        |rank| Ok(Box::new(Deferred::new(rank)?) as Op),
    )
}
/// The served registration's ranks, launching the composed form.
pub(super) fn served(sessions: &[Session]) -> LabResult<Fixture> {
    let names = served_symbols();
    let names: Vec<_> = names.iter().map(String::as_str).collect();
    Fixture::with_ops(sessions, CAPACITY, &names, |rank| {
        rounded_op(&served_backend(), rank)
    })
}

/// Every rank's whole F32 sum region of `fixture` as bits.
fn sums(sessions: &[Session], fixture: &Fixture) -> LabResult<Vec<Vec<u32>>> {
    synchronize(sessions)?;
    (0..4)
        .map(|rank| {
            let region = sum_region(&fixture.ops[rank], rank)?;
            let values = sessions[rank].download_f32(
                &fixture.ranks[rank].state[region],
                CAPACITY as usize * WIDTH,
            )?;
            Ok(values.iter().map(|x| x.to_bits()).collect())
        })
        .collect()
}
/// Poisons every rank's sum region of `fixture`.
fn poison(sessions: &[Session], fixture: &Fixture) -> LabResult<()> {
    for (rank, session) in sessions.iter().enumerate() {
        let region = sum_region(&fixture.ops[rank], rank)?;
        let buffer = &fixture.ranks[rank].state[region];
        buffer
            .copy_from_bytes(&vec![POISON; buffer.byte_len()], &session.stream)
            .map_err(err)?;
    }
    synchronize(sessions)
}
/// Every rank's sums after a deferred step of `fixture` at `rows` rows on
/// `values` with `live` live rows, poisoned first.
fn deferred_sums(
    sessions: &[Session],
    fixture: &Fixture,
    values: &[Vec<f32>],
    rows: u32,
    live: u32,
) -> LabResult<Vec<Vec<u32>>> {
    poison(sessions, fixture)?;
    upload_bits(sessions, fixture, values, live)?;
    topology::each_rank(4, |r| fixture.launch_with(&sessions[r], r, rows, deferred))?;
    sums(sessions, fixture)
}

/// Counts of one comparison.
#[derive(Default)]
pub(super) struct Tally {
    pub(super) elements: usize,
    pub(super) nan_sums: usize,
    pub(super) infinite_sums: usize,
    pub(super) subnormal_sums: usize,
    /// NaN sums whose payload differs from the host's, which the bits of
    /// the pair decide.
    pub(super) nan_payloads: usize,
}

/// Runs the pair of `served` and the prototype of `candidate` at `rows`
/// rows on `values` with `live` live rows: on every rank, every row's F32
/// sum (live or not: both forms sum every row) is the pair's in every bit
/// and the host's F32 sum from zero in rank order (NaN for NaN), the same
/// on every rank, and nothing past the rows is written.
pub(super) fn compare(
    sessions: &[Session],
    served: &Fixture,
    candidate: &Fixture,
    values: &[Vec<f32>],
    rows: u32,
    live: u32,
) -> LabResult<Tally> {
    let n = rows as usize * WIDTH;
    let pair = deferred_sums(sessions, served, values, rows, live)?;
    let new = deferred_sums(sessions, candidate, values, rows, live)?;
    let poisoned = u32::from_ne_bytes([POISON; 4]);
    let mut tally = Tally {
        elements: n,
        ..Tally::default()
    };
    for rank in 0..4 {
        if new[rank][n..].iter().any(|&x| x != poisoned) {
            return Err(format!(
                "the phased F32 entry wrote past {rows} rows on rank {rank}"
            ));
        }
        for i in 0..n {
            let host = values.iter().fold(0.0_f32, |sum, rank| sum + rank[i]);
            let got = new[rank][i];
            let at = |what: &str| {
                format!(
                    "{what} at rows={rows} live={live} rank={rank} row={} col={} \
                     inputs={:08x?} pair={:08x} phased={got:08x} host={:08x}",
                    i / WIDTH,
                    i % WIDTH,
                    values.iter().map(|r| r[i].to_bits()).collect::<Vec<_>>(),
                    pair[rank][i],
                    host.to_bits(),
                )
            };
            if got != pair[rank][i] {
                return Err(at("the phased F32 sum differs from the deferred pair's"));
            }
            if got != new[0][i] {
                return Err(at("the phased F32 sums differ between ranks"));
            }
            let sum = f32::from_bits(got);
            if sum.is_nan() != host.is_nan() || (!host.is_nan() && got != host.to_bits()) {
                return Err(at("the phased F32 sum differs from the host's"));
            }
            if rank == 0 {
                tally.nan_sums += usize::from(sum.is_nan());
                tally.nan_payloads += usize::from(sum.is_nan() && got != host.to_bits());
                tally.infinite_sums += usize::from(sum.is_infinite());
                tally.subnormal_sums += usize::from(sum.is_subnormal());
            }
        }
    }
    Ok(tally)
}

/// Graph replays of the prototype at `rows` rows: each replay's sums are
/// the eager call's.
fn replays(
    sessions: &[Session],
    candidate: &Fixture,
    values: &[Vec<f32>],
    rows: u32,
) -> LabResult<()> {
    let expected = deferred_sums(sessions, candidate, values, rows, rows)?;
    let graph =
        topology::RankReplay::new(sessions, |r, s| candidate.launch_with(s, r, rows, deferred))?;
    for _ in 0..3 {
        poison(sessions, candidate)?;
        upload_bits(sessions, candidate, values, rows)?;
        graph.launch(sessions)?;
        if sums(sessions, candidate)? != expected {
            return Err(format!("a phased F32 graph replay differs at {rows} rows"));
        }
    }
    Ok(())
}

/// The prototype's row counts.
fn rows() -> Vec<u32> {
    let mut rows: Vec<u32> = ROWS.iter().chain(&EXTRA_ROWS).copied().collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

#[test]
fn prototype_replaces_the_rounding_entry_with_its_f32_output_twin() {
    for rank in 0..4 {
        let op = Deferred::new(rank).unwrap();
        let inner = rounded_op(&phased_backend(), rank).unwrap();
        let served = rounded_op(&served_backend(), rank).unwrap();
        let buffers = |op: &dyn PreparedOp<AllReduceBuffers<View>>| {
            super::views(op.info().state.len(), |_| u64::MAX / 4)
        };
        let region = sum_region(&(Box::new(Deferred::new(rank).unwrap()) as Op), rank).unwrap();
        assert_eq!(region, inner.info().state.len());
        assert_eq!(
            op.info().state[region],
            u64::from(CAPACITY) * WIDTH as u64 * 4
        );
        for rows in 1..=CAPACITY {
            let counts = Counts::rows(rows);
            let rounding = inner
                .launches(&buffers(&*inner), View::EMPTY, counts)
                .unwrap();
            let round = op
                .defer_bf16_round(&buffers(&op), View::EMPTY, counts)
                .unwrap()
                .unwrap();
            let [launch] = &round.launches[..] else {
                panic!("{rows} rows launch one entry");
            };
            assert_eq!(rounding.len(), 1);
            assert_eq!(Some(launch.symbol.as_str()), f32_entry(&rounding[0].symbol));
            assert_eq!(phased_f32::is_one_shot(&launch.symbol), rows <= 4);
            // The same plan and protocol arguments; the outputs are every
            // rank's sum and no live count limits the rows.
            assert_eq!(launch.plan, rounding[0].plan);
            let (old, new) = (&rounding[0].arguments.0, &launch.arguments.0);
            for i in [0, 2, 4, 5, 6, 7, 9, 10, 11] {
                assert_eq!(old[i], new[i], "{rows} rows argument {i}");
            }
            let base = 1000 + region as u64;
            assert_eq!(new[1], Argument::DeviceBuffer(base + rank as u64 * 100));
            assert_eq!(
                new[3],
                Argument::DevicePointerArray((0..4).map(|r| base + r * 100).collect())
            );
            assert_eq!(new[8], Argument::NullDevicePointer);
            assert_eq!(round.input.handle, base + rank as u64 * 100);
            // The served form defers the same round to the same consumer.
            let pair = served
                .defer_bf16_round(&buffers(&*served), View::EMPTY, counts)
                .unwrap()
                .unwrap();
            assert_eq!(
                (round.output, round.live_rows, round.width),
                (pair.output, pair.live_rows, pair.width)
            );
        }
    }
    let rows = rows();
    assert!(rows.windows(2).all(|w| w[0] < w[1]));
    assert!(rows.iter().all(|&r| (1..=CAPACITY).contains(&r)));
}

#[test]
#[ignore = "requires four idle GPUs and CHOPINFER_PHASED_F32_CUBIN"]
fn phased_deferred_matches_the_deferred_pair_on_gpu() {
    let cubin = lab_cubin().unwrap();
    let result = kernel_harness::gpu_test_multi(4 << 30, |sessions| {
        if sessions.len() != 4 {
            return Err("exactly four ranks required".into());
        }
        let served = served(sessions)?;
        let candidate = fixture(sessions, cubin)?;
        let mut cases = Vec::new();
        for rows in rows() {
            for name in PATTERNS {
                let values = pattern(name, rows as usize);
                let lives = if rows > 1 {
                    vec![rows, rows - 1]
                } else {
                    vec![rows]
                };
                for live in lives {
                    let t = compare(sessions, &served, &candidate, &values, rows, live)?;
                    cases.push(json!({"rows": rows, "live_rows": live, "pattern": name,
                        "elements_per_rank": t.elements, "nan_sums": t.nan_sums,
                        "nan_payloads_unlike_host": t.nan_payloads,
                        "infinite_sums": t.infinite_sums, "subnormal_sums": t.subnormal_sums,
                        "bitwise": true}));
                }
                replays(sessions, &candidate, &values, rows)?;
            }
            eprintln!(
                "PHASED_DEFERRED_PARITY rows={rows} patterns={} PASS",
                PATTERNS.len()
            );
        }
        let report: Value = json!({"status": "passed", "ranks": 4, "width": WIDTH,
            "capacity": CAPACITY, "rows": rows(), "patterns": PATTERNS,
            "served": served_symbols(), "prototype": SYMBOLS,
            "comparisons": "every rank's F32 sum of every row == the deferred pair's == the \
                host's rank-ordered F32 sum (NaN for NaN); nothing past the rows written; \
                graph replays equal the eager call",
            "cases": cases});
        if let Some(path) = std::env::var_os("CHOPINFER_PHASED_DEFERRED_REPORT") {
            std::fs::write(
                PathBuf::from(path),
                serde_json::to_vec_pretty(&report).map_err(err)?,
            )
            .map_err(err)?;
        }
        eprintln!(
            "PHASED_DEFERRED_PARITY cases={} PASS",
            report["cases"].as_array().unwrap().len()
        );
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "the job must run on four GPUs");
}
