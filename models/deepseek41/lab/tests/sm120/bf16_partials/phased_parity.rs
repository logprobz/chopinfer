//! The phased F32 rounding entries against decode and verify's served
//! rounded all-reduce, bit for bit, on four ranks.
//!
//! Decode and verify reduce through the bundle's registration
//! (`reduce::register`), whose rounded all-reduce takes the composed form:
//! the F32 reduce-scatter and all-gather into every rank's F32 sum, then
//! the round row, or only the pair where the next HC mix takes the round
//! (`ReduceOps::defer_round`). Registering the phased F32 entries on the
//! same backend (`with_phased_f32_all_reduce`, as
//! `Kernels::prefill_reduction_backend` does) moves every row count to the
//! phased forms: the one-shot round entry at 1 to 4 rows, the round
//! two-shot from 5 rows, with workers above 32.
//!
//! Each form sums every element's rank partials in F32 in rank order from
//! zero and rounds the sum once with `cvt.rn.bf16.f32`; the HC mixes that
//! take a deferred round convert the pair's F32 sum with the same
//! instruction before any other arithmetic (`round_expand_mix.cuh`,
//! `round_cols.cuh`). So the phased output must equal the served complete
//! output, and the round of the deferred pair's sum, in every bit,
//! whatever the inputs. [`phased_f32_matches_the_served_reduction_on_gpu`]
//! checks that on adversarial patterns at every timed row count, with dead
//! rows below the device live count and in graph replay.
use super::{Fixture, Op, Plan, WIDTH, err, every, reduce, synchronize, synthetic};
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, View},
    codec::f32_to_bf16,
    collective::AllReduceBuffers,
    contract::CommQuery,
    oracle,
};
use kernel_harness::{LabResult, Session, topology};
use kernels_sm120::{backend::Sm120, comm::phased_f32, row::rounding::Rounding};
use model_deepseek41_contract::geometry::FLASH;
use serde_json::{Value, json};
use std::path::PathBuf;

/// Every timed row count, the phased forms' boundaries (4/5, 32/33) and
/// the package's 192 verify rows.
const ROWS: [u32; 11] = [1, 2, 4, 5, 8, 16, 32, 33, 64, 128, 192];
const CAPACITY: u32 = 192;
const PATTERNS: [&str; 11] = [
    "finite",
    "cancellation",
    "dynamic_range",
    "subnormal",
    "signed_zero",
    "overflow",
    "nonfinite",
    "bf16_ties",
    "bf16_overflow",
    "random_bits",
    "rank_order",
];

/// The served registration: the rounding rows alone.
fn served_backend() -> Sm120 {
    reduce::register(Sm120::new(), &FLASH)
}
/// The candidate registration: the served one with the phased F32 entries.
fn phased_backend() -> Sm120 {
    served_backend().with_phased_f32_all_reduce()
}
fn rounded_op(backend: &Sm120, rank: usize) -> LabResult<Op> {
    backend
        .all_reduce(
            &CommQuery::AllReduceRounded {
                ranks: 4,
                elements: WIDTH as u64,
            },
            rank as u32,
            Need::rows(CAPACITY),
        )
        .map_err(err)
}
fn served_symbols() -> Vec<String> {
    vec![
        reduce::SCATTER_SYMBOL.into(),
        reduce::GATHER_SYMBOL.into(),
        reduce::rounding_symbol(&FLASH, Rounding::Round),
    ]
}
fn phased_symbols() -> Vec<String> {
    vec![
        phased_f32::ONE_SHOT_ROUND_SYMBOL.into(),
        phased_f32::ROUND_SYMBOL.into(),
    ]
}
/// The phased entry `rows` rows launch.
fn phased_symbol(rows: u32) -> &'static str {
    if rows <= 4 {
        phased_f32::ONE_SHOT_ROUND_SYMBOL
    } else {
        phased_f32::ROUND_SYMBOL
    }
}

/// The pair a step whose HC mix takes the round issues, and the index of
/// the op's state region that receives this rank's F32 sum.
fn deferred(op: &Op, buffers: &AllReduceBuffers<View>, counts: Counts) -> LabResult<Vec<Launch>> {
    Ok(op
        .defer_bf16_round(buffers, View::EMPTY, counts)
        .map_err(err)?
        .ok_or("the served reduction no longer defers its round")?
        .launches)
}
fn sum_region(op: &Op, rank: usize) -> LabResult<usize> {
    let buffers = super::views(op.info().state.len(), |_| u64::MAX / 4);
    let round = op
        .defer_bf16_round(&buffers, View::EMPTY, Counts::rows(1))
        .map_err(err)?
        .ok_or("the served reduction no longer defers its round")?;
    let base = 1000 + rank as u64 * 100;
    round
        .input
        .handle
        .checked_sub(base)
        .filter(|&i| i < 100)
        .map(|i| i as usize)
        .ok_or_else(|| "the deferred sum is not this rank's state".into())
}

/// A deterministic 64-bit stream (splitmix64) per pattern, rank and
/// element.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
fn bits(seed: u64, rank: usize, i: usize) -> u64 {
    mix(seed ^ mix((rank as u64) << 40 ^ i as u64))
}

/// Each rank's `rows` rows of `pattern`.
fn pattern(name: &str, rows: usize) -> Vec<Vec<f32>> {
    if name == "finite" {
        return synthetic(rows);
    }
    let seed = mix(name
        .bytes()
        .fold(0_u64, |h, b| h.wrapping_mul(131).wrapping_add(u64::from(b))));
    let value = |rank: usize, i: usize| -> f32 {
        let r = bits(seed, rank, i);
        let sign = (r >> 63) as u32;
        // One template per element for patterns that pair the ranks.
        let t = bits(seed, 7, i);
        match name {
            "cancellation" => {
                let big = f32::from_bits(((t >> 8) as u32 % 64 + 140) << 23);
                let small = f32::from_bits(((t >> 16) as u32 % 64 + 100) << 23 | (r as u32 >> 9));
                let template = [
                    [big, small, small, -big],
                    [small, big, -big, small],
                    [-big, big, small, -small],
                    [big, -big, big, -big],
                ][(t % 4) as usize];
                template[rank]
            }
            "dynamic_range" => {
                let exponent = 1 + (r >> 32) as u32 % 254;
                f32::from_bits(sign << 31 | exponent << 23 | (r as u32 & 0x7f_ffff))
            }
            "subnormal" => match r % 8 {
                0 => f32::from_bits(sign << 31),
                1 => f32::from_bits(sign << 31 | 0x80_0000),
                _ => f32::from_bits(sign << 31 | (r as u32 >> 9).max(1)),
            },
            "signed_zero" => match t % 3 {
                0 => -0.0,
                1 => [0.0, -0.0][(r % 2) as usize],
                _ => {
                    let x = f32::from_bits(0x3f00_0000 | ((r >> 32) as u32 & 0x7f_ffff));
                    [x, -x, -0.0, 0.0][rank]
                }
            },
            "overflow" => {
                let big = f32::from_bits(0x7f7f_0000 | (r as u32 & 0xffff));
                [
                    [big, big, -big, -big],
                    [-big, -big, big, big],
                    [big, -big, big, big],
                    [big, big, big, -f32::MAX],
                ][(t % 4) as usize][rank]
            }
            "nonfinite" => {
                let special = [
                    0x7fc0_0000_u32,
                    0x7fc1_2345,
                    0xffc0_0000,
                    0x7f80_0001,
                    0xff80_0123,
                    0x7f80_0000,
                    0xff80_0000,
                ];
                // About one element in four holds a special value on some
                // rank; Inf and -Inf meet on others.
                match (t % 8, rank as u64) {
                    (0, k) if k == (t >> 8) % 4 => f32::from_bits(special[(r % 7) as usize]),
                    (1, 0) => f32::INFINITY,
                    (1, 2) => f32::NEG_INFINITY,
                    _ => f32::from_bits(
                        sign << 31 | (120 + (r >> 40) as u32 % 16) << 23 | (r as u32 >> 9),
                    ),
                }
            }
            "bf16_ties" => {
                // The first rank's value lies on or next to a BF16 halfway
                // point; the others add zeros or one smallest subnormal.
                let base = 0x3800_0000 + ((t >> 16) as u32 & 0x0fff_0000);
                let tie = f32::from_bits(base | 0x8000);
                match rank {
                    0 => tie,
                    1 => [0.0, f32::from_bits(1), -f32::from_bits(1), -0.0][(t % 4) as usize],
                    _ => 0.0,
                }
            }
            "bf16_overflow" => {
                let near =
                    [0x7f7f_8000_u32, 0x7f7f_7fff, 0x7f7f_8001, 0x7f7e_8000][(t % 4) as usize];
                let sign = ((t >> 8) % 2) as u32;
                match rank {
                    0 => f32::from_bits(sign << 31 | near),
                    _ => [0.0, -0.0][(r % 2) as usize],
                }
            }
            "random_bits" => f32::from_bits((r >> 32) as u32),
            // Sums whose BF16 round differs with the order of the ranks.
            "rank_order" => {
                let one = f32::from_bits(((t >> 8) as u32 % 32 + 112) << 23);
                let ulp = one * f32::EPSILON;
                [
                    [one, ulp / 2.0, ulp / 2.0, -one],
                    [ulp / 2.0, one, ulp / 2.0, -ulp],
                ][(t % 2) as usize][rank]
            }
            _ => unreachable!("unknown pattern"),
        }
    };
    (0..4)
        .map(|rank| (0..rows * WIDTH).map(|i| value(rank, i)).collect())
        .collect()
}

/// Every rank's BF16 output: the live rows, after checking the rest kept
/// their sentinel.
fn outputs(sessions: &[Session], fixture: &Fixture, live: usize) -> LabResult<Vec<Vec<u16>>> {
    synchronize(sessions)?;
    fixture
        .ranks
        .iter()
        .zip(sessions)
        .map(|(r, s)| {
            let all = s.download_bf16(&r.output, fixture.capacity as usize * WIDTH)?;
            if all[live * WIDTH..].iter().any(|&x| x != 0xa5a5) {
                return Err("an all-reduce wrote past the live rows".into());
            }
            Ok(all[..live * WIDTH].to_vec())
        })
        .collect()
}
/// Uploads `values` and runs `plan` on every rank.
fn run(
    sessions: &[Session],
    fixture: &Fixture,
    values: &[Vec<f32>],
    rows: u32,
    live: u32,
    plan: Plan,
) -> LabResult<()> {
    upload_bits(sessions, fixture, values, live)?;
    topology::each_rank(4, |r| fixture.launch_with(&sessions[r], r, rows, plan))
}
/// [`Fixture::upload`] with every input's bits kept, NaN payloads
/// included.
fn upload_bits(
    sessions: &[Session],
    fixture: &Fixture,
    values: &[Vec<f32>],
    live: u32,
) -> LabResult<()> {
    for (r, (buffers, session)) in fixture.ranks.iter().zip(sessions).enumerate() {
        let bytes: Vec<u8> = values[r]
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect();
        buffers
            .input
            .copy_from_bytes(&bytes, &session.stream)
            .map_err(err)?;
        buffers
            .live
            .copy_from_bytes(&live.to_ne_bytes(), &session.stream)
            .map_err(err)?;
        buffers
            .output
            .copy_from_bytes(
                &vec![0xa5; fixture.capacity as usize * WIDTH * 2],
                &session.stream,
            )
            .map_err(err)?;
    }
    synchronize(sessions)
}

/// Counts of one case's comparisons.
#[derive(Default)]
struct Tally {
    elements: usize,
    nan_sums: usize,
    infinite_sums: usize,
    subnormal_sums: usize,
    negative_zero_inputs: usize,
    /// The BF16 NaNs the forms give for NaN sums.
    nan_outputs: std::collections::BTreeSet<u16>,
    oracle_differences: usize,
}

fn compare_case(
    sessions: &[Session],
    served: &Fixture,
    phased: &Fixture,
    sums: &[usize],
    values: &[Vec<f32>],
    rows: u32,
    live: u32,
) -> LabResult<Tally> {
    let n = live as usize * WIDTH;
    run(sessions, served, values, rows, live, every)?;
    let complete = outputs(sessions, served, live as usize)?;
    run(sessions, served, values, rows, live, deferred)?;
    synchronize(sessions)?;
    let pair: Vec<Vec<f32>> = served
        .ranks
        .iter()
        .zip(sessions)
        .zip(sums)
        .map(|((r, s), &i)| {
            let bits: Vec<u32> = s
                .download_f32(&r.state[i], n)?
                .iter()
                .map(|x| x.to_bits())
                .collect();
            Ok(bits.into_iter().map(f32::from_bits).collect())
        })
        .collect::<LabResult<_>>()?;
    run(sessions, phased, values, rows, live, every)?;
    let candidate = outputs(sessions, phased, live as usize)?;
    let host: Vec<Vec<f32>> = values.iter().map(|rank| rank[..n].to_vec()).collect();
    let expected = oracle::all_reduce_rounded(&host).map_err(err)?;
    let mut tally = Tally {
        elements: n,
        ..Tally::default()
    };
    for rank in 0..4 {
        for (i, &reference) in expected.iter().enumerate() {
            let at = |what: &str| {
                format!(
                    "{what} at rows={rows} live={live} rank={rank} row={} col={} \
                     inputs={:08x?} served={:04x} phased={:04x} sum={:08x}",
                    i / WIDTH,
                    i % WIDTH,
                    (0..4).map(|r| values[r][i].to_bits()).collect::<Vec<_>>(),
                    complete[rank][i],
                    candidate[rank][i],
                    pair[rank][i].to_bits(),
                )
            };
            if candidate[rank][i] != complete[rank][i] {
                return Err(at("phased F32 differs from the served complete form"));
            }
            // The deferred pair's sum is the same on every rank, and its
            // round, as the HC mix and the round row take it, is the
            // served output.
            if pair[rank][i].to_bits() != pair[0][i].to_bits() {
                return Err(at("the deferred pair's sums differ between ranks"));
            }
            let sum = pair[rank][i];
            let out = complete[rank][i];
            let rounded = match sum.is_nan() {
                true => out & 0x7f80 == 0x7f80 && out & 0x7f != 0,
                false => f32_to_bf16(sum) == out,
            };
            if !rounded {
                return Err(at("the served output is not the round of the deferred sum"));
            }
            if rank == 0 {
                if sum.is_nan() {
                    tally.nan_outputs.insert(out);
                }
                tally.nan_sums += usize::from(sum.is_nan());
                tally.infinite_sums += usize::from(sum.is_infinite());
                tally.subnormal_sums += usize::from(sum.is_subnormal());
                tally.negative_zero_inputs +=
                    usize::from((0..4).any(|r| values[r][i].to_bits() == 0x8000_0000));
                tally.oracle_differences += usize::from(reference != candidate[rank][i]);
            }
        }
    }
    Ok(tally)
}

/// Graph replays of both forms at `rows` rows on `values`.
fn replays(
    sessions: &[Session],
    served: &Fixture,
    phased: &Fixture,
    values: &[Vec<f32>],
    rows: u32,
) -> LabResult<()> {
    let old = topology::RankReplay::new(sessions, |r, s| served.launch_with(s, r, rows, every))?;
    let new = topology::RankReplay::new(sessions, |r, s| phased.launch_with(s, r, rows, every))?;
    upload_bits(sessions, served, values, rows)?;
    old.launch(sessions)?;
    let expected = outputs(sessions, served, rows as usize)?;
    for _ in 0..3 {
        upload_bits(sessions, phased, values, rows)?;
        new.launch(sessions)?;
        if outputs(sessions, phased, rows as usize)? != expected {
            return Err(format!("phased F32 graph replay differs at {rows} rows"));
        }
    }
    Ok(())
}

#[test]
fn candidate_registration_serves_the_phased_entries_at_every_row() {
    for rank in 0..4 {
        let served = rounded_op(&served_backend(), rank).unwrap();
        let phased = rounded_op(&phased_backend(), rank).unwrap();
        assert!(served.info().implementation.ends_with("[composed]"));
        assert!(phased.info().implementation.ends_with("[phased]"));
        let buffers = |op: &Op| super::views(op.info().state.len(), |_| u64::MAX / 4);
        for rows in 1..=CAPACITY {
            let counts = Counts::rows(rows);
            let calls = every(&phased, &buffers(&phased), counts).unwrap();
            assert_eq!(
                calls.iter().map(|c| c.symbol.as_str()).collect::<Vec<_>>(),
                [phased_symbol(rows)]
            );
            // A phased form exposes no deferred round.
            assert!(
                phased
                    .defer_bf16_round(&buffers(&phased), View::EMPTY, counts)
                    .unwrap()
                    .is_none()
            );
            let calls = every(&served, &buffers(&served), counts).unwrap();
            assert_eq!(
                calls.iter().map(|c| c.symbol.clone()).collect::<Vec<_>>(),
                served_symbols()
            );
        }
        sum_region(&served, rank).unwrap();
    }
    for name in PATTERNS {
        let values = pattern(name, 2);
        assert_eq!(values.len(), 4);
        assert!(values.iter().all(|rank| rank.len() == 2 * WIDTH));
    }
    let ties = pattern("bf16_ties", 1);
    assert!(ties[0].iter().all(|x| x.to_bits() & 0xffff == 0x8000));
    assert!(
        pattern("nonfinite", 1)
            .iter()
            .flatten()
            .any(|x| x.to_bits() == 0x7f80_0001)
    );
}

#[test]
#[ignore = "requires four idle GPUs"]
fn phased_f32_matches_the_served_reduction_on_gpu() {
    let result = kernel_harness::gpu_test_multi(4 << 30, |sessions| {
        if sessions.len() != 4 {
            return Err("exactly four ranks required".into());
        }
        let names = served_symbols();
        let names: Vec<_> = names.iter().map(String::as_str).collect();
        let served = Fixture::with_ops(sessions, CAPACITY, &names, |rank| {
            rounded_op(&served_backend(), rank)
        })?;
        let names = phased_symbols();
        let names: Vec<_> = names.iter().map(String::as_str).collect();
        let phased = Fixture::with_ops(sessions, CAPACITY, &names, |rank| {
            rounded_op(&phased_backend(), rank)
        })?;
        let sums = (0..4)
            .map(|rank| sum_region(&served.ops[rank], rank))
            .collect::<LabResult<Vec<_>>>()?;
        let mut cases = Vec::new();
        for rows in ROWS {
            for name in PATTERNS {
                let values = pattern(name, rows as usize);
                let lives = if rows > 1 { vec![rows, rows - 1] } else { vec![rows] };
                for live in lives {
                    let t = compare_case(sessions, &served, &phased, &sums, &values, rows, live)?;
                    cases.push(json!({"rows": rows, "live_rows": live, "pattern": name,
                        "elements_per_rank": t.elements, "nan_sums": t.nan_sums,
                        "infinite_sums": t.infinite_sums, "subnormal_sums": t.subnormal_sums,
                        "negative_zero_inputs": t.negative_zero_inputs,
                        "nan_outputs": t.nan_outputs.iter().map(|x| format!("{x:04x}")).collect::<Vec<_>>(),
                        "host_oracle_differences": t.oracle_differences, "bitwise": true}));
                }
                replays(sessions, &served, &phased, &values, rows)?;
            }
            eprintln!("PHASED_F32_PARITY rows={rows} patterns={} PASS", PATTERNS.len());
        }
        let report: Value = json!({"status": "passed", "ranks": 4, "width": WIDTH,
            "capacity": CAPACITY, "rows": ROWS, "patterns": PATTERNS,
            "served": served_symbols(), "phased": phased_symbols(),
            "comparisons": "phased output == served complete output == round of the deferred \
                pair's F32 sum, on every rank and live row; dead rows untouched; graph replays",
            "cases": cases});
        if let Some(path) = std::env::var_os("CHOPINFER_PHASED_PARITY_REPORT") {
            std::fs::write(PathBuf::from(path), serde_json::to_vec_pretty(&report).map_err(err)?)
                .map_err(err)?;
        }
        eprintln!("PHASED_F32_PARITY cases={} PASS", report["cases"].as_array().unwrap().len());
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "the job must run on four GPUs");
}
