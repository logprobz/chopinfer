//! The low-latency all-reduce's residency admission at create: its CTAs
//! spin on their counterparts on the peer ranks, so an instance serves it
//! only when every rank's device reports the entry's largest grid resident,
//! and the MoE sublayers' low-latency round entry's as well. A shortfall of
//! either, an MPS client or no report on any rank turns both off on every
//! rank, with a log line, and the phased one-shots reduce their rows; create
//! still succeeds. The fake host reports what the target's driver gives for the
//! entry's locked resources at the exact launch create asks about
//! (`crate::sm120::device`); the refusals replace that model
//! with fixed reports.

use super::*;
use crate::sm120::device::residency;
use bundle_runtime::residency::{Verdict, check};
use model_glm5next_contract::sm120::engines::{self as selected, Bf16Tuned};

const OFF: &str = "low-latency all-reduce off, the phased one-shot serves its rows";
const ON: &str = "low-latency all-reduce on: ";

/// The entry's largest grid: two CTAs per row at `BF16_LOW_LATENCY_ROWS`
/// rows.
const GRID: u32 = 2 * selected::BF16_LOW_LATENCY_ROWS;

/// The round entry's grid at its one row: four CTAs of 1,024 elements.
const ROUND_GRID: u32 = 4;

/// What one instance did: its logs about the entry, the BF16-partials
/// all-reduce symbols of a nine-row prefill's and a one-row decode step's
/// launches (every stream), the decode step's rounded all-reduce symbols,
/// each request's all-reduce calls (one listed launch each: an entry's, or
/// the comm engine's pair's scatter, which the MoE sublayers' composed form
/// launches too), and the entry's and the round entry's residency queries.
struct Outcome {
    on: Vec<String>,
    off: Vec<String>,
    prefill: Vec<String>,
    decode: Vec<String>,
    rounded: Vec<String>,
    calls: Vec<usize>,
    queries: Vec<(String, u32)>,
    round_queries: Vec<(String, u32)>,
}

/// The rounded all-reduce launches from `from` on: the low-latency round
/// entry's and the phased F32 rounding entries'.
fn rounded_reduces(host: &FakeHost, from: usize) -> Vec<String> {
    host.device.journal.borrow()[from..]
        .iter()
        .filter_map(|line| {
            let symbol = line.strip_prefix("launch ")?.split(' ').nth(1)?;
            (symbol == selected::LL_ROUND_SYMBOL
                || selected::PHASED_ROUND_SYMBOLS.contains(&symbol))
            .then(|| symbol.to_owned())
        })
        .collect()
}

fn reduces(host: &FakeHost, from: usize) -> Vec<String> {
    host.device.journal.borrow()[from..]
        .iter()
        .filter_map(|line| {
            let symbol = line.strip_prefix("launch ")?.split(' ').nth(1)?;
            [
                selected::LL_SYMBOL,
                selected::ONE_SHOT_SYMBOL,
                selected::SCATTER_SYMBOL,
            ]
            .into_iter()
            .chain(selected::PHASED_SYMBOLS)
            .any(|reduce| reduce == symbol)
            .then(|| symbol.to_owned())
        })
        .collect()
}

/// Creates a one-slot instance on a host that `setup` configured, then
/// runs a nine-row prefill and the first decode step.
fn run(setup: impl FnOnce(&FakeHost)) -> Outcome {
    let host = FakeHost::new((0, 0));
    setup(&host);
    let mut instance = create_one_slot(&host);
    let mut requests = Vec::new();
    let mut rounded = Vec::new();
    let mut calls = Vec::new();
    for request in [prompt_chunk(7, 0, 9, true), step(&[(7, 0, 9, &[1])])] {
        let start = host.device.journal.borrow().len();
        instance
            .execute(&request, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        requests.push(reduces(&host, start));
        rounded = rounded_reduces(&host, start);
        calls.push(requests.last().unwrap().len() + rounded.len());
    }
    instance.destroy().unwrap();
    let decode = requests.pop().unwrap();
    let prefill = requests.pop().unwrap();
    let logged = |text: &str| -> Vec<String> {
        host.logs
            .borrow()
            .iter()
            .filter(|line| line.contains(text))
            .cloned()
            .collect()
    };
    let (on, off) = (logged(ON), logged(OFF));
    let queries_of = |symbol: &str| {
        host.device
            .residency_queries
            .borrow()
            .iter()
            .filter(|(name, _)| name == symbol)
            .cloned()
            .collect()
    };
    Outcome {
        on,
        off,
        prefill,
        decode,
        rounded,
        calls,
        queries: queries_of(selected::LL_SYMBOL),
        round_queries: queries_of(selected::LL_ROUND_SYMBOL),
    }
}

/// The handle slot of each rank's resolution of the entry, in rank order.
fn entry_slots(host: &FakeHost) -> Vec<u32> {
    host.device
        .journal
        .borrow()
        .iter()
        .filter_map(|line| {
            let rest = line.strip_prefix("resolve ")?;
            let (id, name) = rest.split_once(' ')?;
            (name == selected::LL_SYMBOL).then(|| id.parse().unwrap())
        })
        .collect()
}

/// `run` with fixed reports in place of the device model: by default the
/// modelled report's 352 CTAs, then what `setup` sets.
fn run_reported(setup: impl FnOnce(&FakeHost)) -> Outcome {
    run(|host| {
        *host.device.modelled.borrow_mut() = None;
        host.device.residency.set(Some(residency(352, 0)));
        setup(host);
    })
}

/// The phased entry that reduces `rows` rows without the low-latency
/// entry: the one-shot in its place, else the entry the table gives.
fn engine(rows: u32) -> &'static str {
    match selected::bf16_tuned(rows) {
        Bf16Tuned::LowLatency => Bf16Tuned::OneShot,
        form => form,
    }
    .symbol()
}

/// Every rank's all-reduces of the prefill and the decode step went
/// through `entry(rows)` and none through the entry, and one log line says
/// why.
fn assert_off(outcome: &Outcome, reason: &str, entry: fn(u32) -> &'static str) {
    assert_eq!(outcome.off.len(), 1, "{:?}", outcome.off);
    assert!(outcome.off[0].contains(reason), "{:?}", outcome.off);
    assert!(outcome.on.is_empty(), "{:?}", outcome.on);
    for (reduces, rows) in [(&outcome.prefill, 9), (&outcome.decode, 1)] {
        assert!(!reduces.is_empty());
        assert!(
            reduces.iter().all(|symbol| symbol == entry(rows)),
            "{reduces:?}"
        );
    }
}

/// The comm engine's reduce-scatter, which a host that turns the phased
/// entries off as well reduces every row count through.
fn pair(_: u32) -> &'static str {
    selected::SCATTER_SYMBOL
}

/// The target's report for the entry's resources admits it on every rank:
/// each rank asks once, at the largest grid, and once for the round entry;
/// the decode step's all-reduces all take them, and the nine-row prefill's
/// take whichever entry SM120's table gives nine rows
/// (`engines::bf16_tuned`).
#[test]
fn the_entry_serves_when_every_rank_reports_it_resident() {
    let outcome = run(|_| {});
    assert!(outcome.off.is_empty(), "{:?}", outcome.off);
    // One line names what each rank needs of what it keeps resident.
    assert_eq!(outcome.on.len(), 1, "{:?}", outcome.on);
    for rank in 0..4 {
        assert!(
            outcome.on[0].contains(&format!("rank {rank} {GRID} of 352 CTAs")),
            "{:?}",
            outcome.on
        );
    }
    assert_eq!(
        outcome.queries,
        vec![(selected::LL_SYMBOL.to_owned(), GRID); 4]
    );
    assert_eq!(
        outcome.round_queries,
        vec![(selected::LL_ROUND_SYMBOL.to_owned(), ROUND_GRID); 4]
    );
    assert!(!outcome.rounded.is_empty());
    assert!(
        outcome
            .rounded
            .iter()
            .all(|symbol| symbol == selected::LL_ROUND_SYMBOL),
        "{:?}",
        outcome.rounded
    );
    assert!(!outcome.decode.is_empty());
    assert!(
        outcome
            .decode
            .iter()
            .all(|symbol| symbol == selected::LL_SYMBOL)
    );
    let prefill = selected::bf16_tuned(9).symbol();
    assert!(!outcome.prefill.is_empty());
    assert!(
        outcome.prefill.iter().all(|symbol| symbol == prefill),
        "{:?}",
        outcome.prefill
    );
}

/// A device that keeps fewer CTAs resident than the grid, an MPS client and
/// a host that reports nothing each turn the entry off: the phased entries
/// serve its rows where the device vouches for theirs, else the comm
/// engine's composition.
#[test]
fn others_serve_when_a_device_cannot_vouch_for_the_grid() {
    let short = run_reported(|host| {
        host.device
            .residency_by_symbol
            .borrow_mut()
            .insert(selected::LL_SYMBOL.to_owned(), residency(GRID - 1, 0));
    });
    assert_off(&short, "fewer CTAs resident", engine);
    assert_eq!(short.queries, vec![(selected::LL_SYMBOL.to_owned(), GRID)]);
    // The decode step's MoE sublayers take the phased one-shot round entry.
    assert!(!short.rounded.is_empty());
    assert!(
        short
            .rounded
            .iter()
            .all(|symbol| symbol == selected::PHASED_ROUND_SYMBOLS[0]),
        "{:?}",
        short.rounded
    );
    // The round entry short alone turns both off.
    let round_short = run_reported(|host| {
        host.device.residency_by_symbol.borrow_mut().insert(
            selected::LL_ROUND_SYMBOL.to_owned(),
            residency(ROUND_GRID - 1, 0),
        );
    });
    assert_off(&round_short, "fewer CTAs resident", engine);
    assert_eq!(
        round_short.round_queries,
        vec![(selected::LL_ROUND_SYMBOL.to_owned(), ROUND_GRID)]
    );
    assert_eq!(round_short.rounded, short.rounded);
    let mps = run_reported(|host| {
        host.device
            .residency
            .set(Some(residency(46, BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT)));
    });
    assert_off(&mps, "MPS client", pair);
    let unreported = run_reported(|host| {
        host.device
            .residency
            .set(Some(BundleRawClusterResidencyV1::unreported()));
    });
    assert_off(&unreported, "reports no cluster occupancy", pair);
    // As many all-reduce calls run with the entry on or off.
    let on = run(|_| {});
    for off in [&short, &round_short, &mps, &unreported] {
        assert_eq!(off.calls, on.calls);
    }
}

/// One rank short turns the entry off on every rank (else ranks would run
/// different protocols over one call): the last rank's resolution of the
/// entry reports a shortfall, the others the target's report.
#[test]
fn one_rank_short_turns_the_entry_off_on_every_rank() {
    let probe = FakeHost::new((0, 0));
    create_one_slot(&probe).destroy().unwrap();
    let slots = entry_slots(&probe);
    assert_eq!(slots.len(), 4);
    let last = *slots.last().unwrap();
    let outcome = run_reported(|host| {
        host.device
            .residency_by_slot
            .borrow_mut()
            .insert(last, residency(GRID - 1, 0));
    });
    assert_off(&outcome, "rank 3:", engine);
    assert_eq!(outcome.queries.len(), 4);
}

/// The launch create asks about is the selected plan's own: `GRID` CTAs
/// (two per row at `BF16_LOW_LATENCY_ROWS` rows) of 256 threads with no
/// cluster (a launch the host reports only since it asks about clusters of
/// one CTA). For it the target's driver, modelled from the entry's locked
/// resources, keeps enough CTAs resident for the grid; a host that reports
/// nothing for it refuses the entry.
#[test]
fn the_selected_unclustered_plan_is_admitted_on_its_modelled_report() {
    let plan = crate::sm120::low_latency_plan(Shape::TP4).unwrap();
    let config = bundle_runtime::dispatch::launch_config(plan).unwrap();
    assert_eq!((config.grid_x, config.grid_y, config.grid_z), (GRID, 1, 1));
    assert!(config.cluster.is_none());
    let resources = ll_resources()[selected::LL_SYMBOL];
    let report = crate::sm120::device::report(resources, config);
    assert!(report.max_active_clusters >= GRID, "{report:?}");
    assert_eq!(check(&plan, 1, &report), Verdict::Fits);
    assert!(matches!(
        check(&plan, 1, &BundleRawClusterResidencyV1::unreported()),
        Verdict::Unknown(_)
    ));
}
