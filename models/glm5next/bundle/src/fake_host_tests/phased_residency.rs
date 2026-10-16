//! The phased all-reduce's residency admission at create: every CTA of its
//! entries (the BF16 partials' and the MoE sublayers' F32 rounding entries)
//! spins on flags its own grid and the peer ranks release, so an instance
//! serves them only when every rank's device reports each served entry's
//! grid resident. A shortfall, an MPS client or no report for any entry on
//! any rank turns all of them off on every rank, with a log line, and the
//! round row, its widening, the phased pair and the round row reduce the
//! BF16 partials' rows, the F32 pair and the round row the MoE sublayers';
//! create still succeeds. The fake host reports what the target's driver
//! gives for the entries' locked resources at the exact launch create asks
//! about (`crate::sm120::device`); the refusals replace that
//! model with fixed reports.

use super::*;
use crate::sm120::device::residency;
use bundle_runtime::residency::{Verdict, check};
use model_glm5next_contract::sm120::engines::{self as selected};

const OFF: &str =
    "phased all-reduce off, the round row, the phased pair and the round row serve its rows";
const ON: &str = "phased all-reduce on: ";

/// The served entries in the order create asks about them: the BF16
/// partials', then the F32 rounding entries.
fn served() -> Vec<&'static str> {
    [
        &selected::PHASED_SYMBOLS[..],
        &selected::PHASED_ROUND_SYMBOLS[..],
    ]
    .concat()
}

/// The grid create asks about for `symbol`: the one-shots' 16 sending
/// CTAs, the two-shots' widest geometry.
fn grid(symbol: &str) -> u32 {
    let mut plans = selected::bf16_phased_admission_plans().unwrap();
    plans.extend(selected::rounded_phased_admission_plans().unwrap());
    let (_, plan) = plans.iter().find(|(name, _)| *name == symbol).unwrap();
    plan.grid[0]
}

/// What creating one instance logged about the entries, and their
/// residency queries in issue order.
struct Outcome {
    on: Vec<String>,
    off: Vec<String>,
    queries: Vec<(String, u32)>,
}

/// Creates and destroys a one-slot instance on a host that `setup`
/// configured.
fn created(setup: impl FnOnce(&FakeHost)) -> Outcome {
    let host = FakeHost::new((0, 0));
    setup(&host);
    create_one_slot(&host).destroy().unwrap();
    let logged = |text: &str| -> Vec<String> {
        host.logs
            .borrow()
            .iter()
            .filter(|line| line.contains(text))
            .cloned()
            .collect()
    };
    let queries = host
        .device
        .residency_queries
        .borrow()
        .iter()
        .filter(|(name, _)| served().contains(&name.as_str()))
        .cloned()
        .collect();
    Outcome {
        on: logged(ON),
        off: logged(OFF),
        queries,
    }
}

/// `created` with fixed reports in place of the device model: by default
/// 352 CTAs, then what `setup` sets.
fn created_reported(setup: impl FnOnce(&FakeHost)) -> Outcome {
    created(|host| {
        *host.device.modelled.borrow_mut() = None;
        host.device.residency.set(Some(residency(352, 0)));
        setup(host);
    })
}

/// The all-reduce and round launches of a two-item decode step (every
/// stream) on an instance created on a host that `setup` configured, the
/// rounded all-reduce's included.
fn decode_reduces(setup: impl FnOnce(&FakeHost)) -> Vec<String> {
    let host = FakeHost::new((0, 0));
    setup(&host);
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::seed(handle, 7, 0, 100);
    crate::instance::seed(handle, 8, 5, 64);
    crate::instance::run_lowered_stages_alone(handle);
    let start = host.device.journal.borrow().len();
    let decode = step(&[(7, 0, 100, &[11]), (8, 5, 64, &[12])]);
    instance
        .execute(&decode, &mut NoRows::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let reduces = issued(&host, start)
        .iter()
        .filter_map(|line| {
            let symbol = line.strip_prefix("launch ")?.split(' ').nth(1)?;
            (served().contains(&symbol)
                || [
                    selected::LL_SYMBOL,
                    selected::SCATTER_SYMBOL,
                    selected::GATHER_SYMBOL,
                ]
                .contains(&symbol)
                || symbol == selected::ModelRow::Round.symbol())
            .then(|| symbol.to_owned())
        })
        .collect();
    instance.commit().unwrap();
    instance.destroy().unwrap();
    reduces
}

/// Admitted, a two-row decode bucket's all-reduces each run as the phased
/// one-shot alone, the MoE sublayers' as the one-shot round entry; refused,
/// as the round row, its widening (not listed: the embedding's all-reduce
/// widens its rows first), the phased pair and the round row, and the MoE
/// sublayers' as the F32 pair and the round row, as many of them.
#[test]
fn admitted_entries_reduce_the_decode_rows_and_refused_ones_leave_the_pair() {
    let one_shot_round = selected::PHASED_ROUND_SYMBOLS[0];
    let on = decode_reduces(|_| {});
    assert!(!on.is_empty());
    assert!(
        on.iter()
            .all(|symbol| [selected::PHASED_SYMBOLS[0], one_shot_round].contains(&symbol.as_str())),
        "{on:?}"
    );
    let count = |symbol: &str| on.iter().filter(|s| *s == symbol).count();
    let (bf16, rounded) = (count(selected::PHASED_SYMBOLS[0]), count(one_shot_round));
    assert!(bf16 > 0 && rounded > 0, "{on:?}");
    let off = decode_reduces(|host| {
        *host.device.modelled.borrow_mut() = None;
        host.device.residency.set(Some(residency(352, 0)));
        host.device.residency_by_symbol.borrow_mut().insert(
            selected::PHASED_SYMBOLS[0].to_owned(),
            residency(grid(selected::PHASED_SYMBOLS[0]) - 1, 0),
        );
    });
    let round = selected::ModelRow::Round.symbol();
    let f32_pair = [
        selected::SCATTER_SYMBOL.to_owned(),
        selected::GATHER_SYMBOL.to_owned(),
        round.clone(),
    ];
    let (mut pairs, mut f32_pairs, mut at) = (0, 0, 0);
    while at < off.len() {
        if off[at] == round {
            assert_eq!(off[at + 1..at + 4], f32_pair, "{off:?}");
            (pairs, at) = (pairs + 1, at + 4);
        } else {
            assert_eq!(off[at..at + 3], f32_pair, "{off:?}");
            (f32_pairs, at) = (f32_pairs + 1, at + 3);
        }
    }
    assert_eq!((pairs, f32_pairs), (bf16, rounded));
}

fn assert_off(outcome: &Outcome, reason: &str) {
    assert_eq!(outcome.off.len(), 1, "{:?}", outcome.off);
    assert!(outcome.off[0].contains(reason), "{:?}", outcome.off);
    assert!(outcome.on.is_empty(), "{:?}", outcome.on);
}

/// The target's report for the entries' resources admits them on every
/// rank: each rank asks once per served entry at its largest grid, and one
/// log line names what each rank needs of what it keeps resident.
#[test]
fn the_entries_serve_when_every_rank_reports_them_resident() {
    let outcome = created(|_| {});
    assert!(outcome.off.is_empty(), "{:?}", outcome.off);
    assert_eq!(outcome.on.len(), 1, "{:?}", outcome.on);
    for symbol in served() {
        let named = outcome.on[0]
            .split("; ")
            .find(|part| part.contains(&format!("{symbol} needed of resident: ")))
            .unwrap_or_else(|| panic!("{symbol}: {:?}", outcome.on));
        for rank in 0..4 {
            assert!(
                named.contains(&format!("rank {rank} {} of ", grid(symbol))),
                "{named}"
            );
        }
    }
    let expected = served()
        .iter()
        .flat_map(|symbol| std::iter::repeat_n((symbol.to_string(), grid(symbol)), 4))
        .collect::<Vec<_>>();
    assert_eq!(outcome.queries, expected);
}

/// One entry short on the devices (a BF16 partials' entry or an F32
/// rounding entry), an MPS client and a host that reports nothing each turn
/// every entry off.
#[test]
fn every_entry_turns_off_when_a_device_cannot_vouch_for_one() {
    let short = created_reported(|host| {
        host.device.residency_by_symbol.borrow_mut().insert(
            selected::PHASED_SYMBOLS[1].to_owned(),
            residency(grid(selected::PHASED_SYMBOLS[1]) - 1, 0),
        );
    });
    assert_off(&short, "fewer CTAs resident");
    assert!(
        short.off[0].contains(&format!("rank 0: {}", selected::PHASED_SYMBOLS[1])),
        "{:?}",
        short.off
    );
    let round = selected::PHASED_ROUND_SYMBOLS[1];
    let round_short = created_reported(|host| {
        host.device
            .residency_by_symbol
            .borrow_mut()
            .insert(round.to_owned(), residency(grid(round) - 1, 0));
    });
    assert_off(&round_short, "fewer CTAs resident");
    assert!(
        round_short.off[0].contains(&format!("rank 0: {round}")),
        "{:?}",
        round_short.off
    );
    let mps = created_reported(|host| {
        host.device
            .residency
            .set(Some(residency(46, BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT)));
    });
    assert_off(&mps, "MPS client");
    let unreported = created_reported(|host| {
        host.device
            .residency
            .set(Some(BundleRawClusterResidencyV1::unreported()));
    });
    assert_off(&unreported, "reports no cluster occupancy");
}

/// The launches create asks about are each served entry's largest: CTAs of
/// 256 threads with no cluster. For them the target's driver, modelled from
/// each entry's locked resources, keeps enough CTAs resident.
#[test]
fn the_phased_plans_are_admitted_on_their_modelled_reports() {
    let resources = phased_resources();
    let mut plans = selected::bf16_phased_admission_plans().unwrap();
    plans.extend(selected::rounded_phased_admission_plans().unwrap());
    assert_eq!(
        plans.iter().map(|(symbol, _)| *symbol).collect::<Vec<_>>(),
        served()
    );
    for (symbol, plan) in plans {
        let config = bundle_runtime::dispatch::launch_config(plan).unwrap();
        let grid = grid(symbol);
        assert_eq!((config.grid_x, config.grid_y, config.grid_z), (grid, 1, 1));
        assert!(config.cluster.is_none());
        let report = crate::sm120::device::report(resources[symbol], config);
        assert!(report.max_active_clusters >= grid, "{symbol}: {report:?}");
        assert_eq!(check(&plan, 1, &report), Verdict::Fits);
    }
}
