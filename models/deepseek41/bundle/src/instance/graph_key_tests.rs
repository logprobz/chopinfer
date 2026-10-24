//! A target graph replays at every phase its key admits, so target
//! lowering must be a function of the key alone. Each case here lowers
//! afresh (no graph is captured), and every reachable verify phase of the
//! same route and rows must plan the same launches, with the same resolved
//! arguments, plans, collectives and stages on every rank, whatever its
//! live items, row split, request order, physical slots, pages and
//! contexts. Item-dependent device data reaches the graphs only through
//! the per-step uploads.
use std::collections::BTreeMap;

use kernel_core::contract::LaunchPlan;
use runtime_bundle_abi::KernelArgumentV1;

use super::*;
use crate::config::{KV_LAYOUTS, KvLayout};
use crate::fake_host_tests::indexer_workloads::complete_host;
use bundle_runtime::dispatch;
use bundle_runtime::geometry::Ranks;
use model_deepseek41_contract::state::KvStorage;

/// One verify phase: each item's rows, request end and physical slot,
/// whether the items run in reverse request order, and the page seed.
#[derive(Clone, Debug)]
struct Case {
    split: Vec<u32>,
    ends: Vec<u32>,
    slots: Vec<i32>,
    reversed: bool,
    seed: u32,
}

fn phase_inputs(case: &Case) -> PhaseInputs {
    let items = case.split.len();
    let ends: Vec<i32> = case.ends.iter().map(|&end| end as i32).collect();
    let prefix: Vec<i32> = ends
        .iter()
        .zip(&case.split)
        .map(|(&end, &rows)| end - rows as i32)
        .collect();
    let starts: Vec<i32> = case
        .split
        .iter()
        .scan(0, |at, &rows| {
            let start = *at;
            *at += rows as i32;
            Some(start)
        })
        .collect();
    let requests: Vec<u64> = if case.reversed {
        (1..=items as u64).rev().collect()
    } else {
        (1..=items as u64).collect()
    };
    PhaseInputs {
        decode: true,
        verify: true,
        verify_rows: case.split.iter().map(|&rows| rows > 1).collect(),
        item_order: if case.reversed {
            (0..items).rev().collect()
        } else {
            (0..items).collect()
        },
        request_ids: requests,
        emits_row: vec![true; items],
        prompt_rows: vec![false; items],
        anchors: vec![false; items],
        token_ids: (0..case.split.iter().sum::<u32>()).map(|n| 7 + n).collect(),
        positions: prefix
            .iter()
            .zip(&case.split)
            .flat_map(|(&first, &rows)| (0..rows as i32).map(move |row| first + row))
            .collect(),
        token_batch: case
            .split
            .iter()
            .enumerate()
            .flat_map(|(item, &rows)| vec![item as i32; rows as usize])
            .collect(),
        block_tables: ends
            .iter()
            .zip(&case.slots)
            .map(|(&end, &slot)| {
                (0..(end as u32).div_ceil(64))
                    .map(|page| (page * 32 + (slot as u32 + case.seed) % 32) as i32)
                    .collect()
            })
            .collect(),
        slot_ids: case.slots.clone(),
        q_start: starts,
        q_len: case.split.iter().map(|&rows| rows as i32).collect(),
        prefix_len: prefix,
        last_positions: ends.iter().map(|&end| i64::from(end - 1)).collect(),
        seq_lens: ends,
        ..PhaseInputs::default()
    }
}

/// Splits of `rows` over `items` items of one to `MAX_VERIFY_ROWS` rows
/// each: the longest items first, as even as possible, and the longest
/// items last.
fn splits(rows: u32, items: u32) -> Vec<Vec<u32>> {
    let mut front = Vec::new();
    let mut left = rows - items;
    for _ in 0..items {
        let extra = left.min(MAX_VERIFY_ROWS - 1);
        left -= extra;
        front.push(extra + 1);
    }
    let even = (0..items)
        .map(|i| rows / items + u32::from(i < rows % items))
        .collect::<Vec<_>>();
    let back = front.iter().rev().copied().collect::<Vec<_>>();
    let mut all = vec![front, even, back];
    all.dedup();
    all
}

/// Request ends a little below each of `bands` in turn, one per item.
fn staggered(items: usize, bands: &[u32], step: u32) -> Vec<u32> {
    (0..items)
        .map(|item| bands[item % bands.len()] - item as u32 % 3 * step)
        .collect()
}

/// The ordinary placement: slots in item order, requests in order.
fn placed(split: Vec<u32>, ends: Vec<u32>) -> Case {
    Case {
        slots: (0..split.len() as i32).collect(),
        split,
        ends,
        reversed: false,
        seed: 0,
    }
}

/// The same split moved: the last physical slots in reverse, requests in
/// reverse order, other pages and other request ends.
fn moved(split: Vec<u32>, ends: Vec<u32>, served: u32) -> Case {
    Case {
        slots: (0..split.len() as i32)
            .map(|item| served as i32 - 1 - item)
            .collect(),
        split,
        ends,
        reversed: true,
        seed: 13,
    }
}

const SHORT: u32 = 2048;
const MEDIUM: u32 = 32768;
const LONG: u32 = 131072;
const LONGEST: u32 = 262144;
/// Request-end bands: within one band, the long-context routes' bounds and
/// mixed bands.
const BANDS: [&[u32]; 6] = [
    &[MEDIUM],
    &[LONG],
    &[LONGEST],
    &[LONGEST + 1],
    &[SHORT, LONG],
    &[MEDIUM, LONG],
];

/// Reachable phases of `rows` at `served` items, one or more per item
/// count: every split at short contexts when `every`, and one split moved;
/// then one split in each long or mixed band, one band per item count in
/// turn.
fn cases(rows: u32, served: u32, every: bool) -> Vec<Case> {
    let mut all = Vec::new();
    for items in rows.div_ceil(MAX_VERIFY_ROWS)..=rows.min(served) {
        let n = items as usize;
        let splits = splits(rows, items);
        if every {
            for split in &splits {
                all.push(placed(split.clone(), staggered(n, &[SHORT], 5)));
            }
        }
        let band = BANDS[(rows + items) as usize % BANDS.len()];
        let (first, last) = (splits[0].clone(), splits[splits.len() - 1].clone());
        all.push(moved(first, staggered(n, &[SHORT + 700], 9), served));
        all.push(moved(last, staggered(n, band, 5), served));
    }
    all
}

/// The route witnesses at 32 rows over 8 items: four rows each against
/// the same rows split unevenly, at the page-rounded bounds of each band.
/// Request ends round up to whole 64-row pages before the band test.
fn witnesses(served: u32) -> Vec<Case> {
    let mut all = Vec::new();
    for end in [
        SHORT,
        MEDIUM - 64,
        MEDIUM - 63,
        MEDIUM,
        LONG - 64,
        LONG - 63,
        LONG,
        LONGEST,
        LONGEST + 1,
    ] {
        for split in [vec![4; 8], vec![1, 1, 5, 5, 5, 5, 5, 5]] {
            all.push(placed(split.clone(), vec![end; 8]));
            all.push(moved(split, vec![end; 8], served));
        }
    }
    all
}

/// One launch as planned: its stage, whether it waits for its peers, its
/// symbol and rows, its plan and its resolved arguments.
type Launch = (Stage, bool, String, u32, LaunchPlan, Vec<KernelArgumentV1>);

/// Everything a planned graph run hands the graph caches or the device,
/// besides its per-step uploads: the scored rows, the input layout and
/// each rank's launches in program order.
#[derive(Debug, PartialEq)]
struct Planned {
    scored: u32,
    layout: InputLayout,
    ranks: Vec<Vec<Launch>>,
}

fn planned(run: &Run) -> Planned {
    assert!(run.graph.is_some() && !run.phase.prefill());
    assert!(run.skip.is_none() && run.dma.is_none() && run.micro.is_none());
    assert!(run.overlap.is_none() && run.prefill_engram.is_none());
    assert!(run.cache.is_none() && run.media.is_none());
    let ranks = run
        .programs
        .iter()
        .enumerate()
        .map(|(rank, launches)| {
            assert!(!launches.is_empty(), "rank {rank} was not lowered");
            assert_eq!(launches.len(), run.collectives[rank].len());
            assert_eq!(launches.len(), run.stages[rank].len());
            launches
                .iter()
                .zip(&run.collectives[rank])
                .zip(&run.stages[rank])
                .map(|((launch, &collective), &stage)| {
                    let arguments =
                        dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
                    (
                        stage,
                        collective,
                        launch.symbol.clone(),
                        launch.rows,
                        launch.plan,
                        arguments,
                    )
                })
                .collect()
        })
        .collect();
    Planned {
        scored: run.head_rows,
        layout: run.layout,
        ranks,
    }
}

/// A graph key as the caches hold it: the route's bank and the key in it.
type Key = (IndexerRoute, (u32, u32));

/// The first launch that differs between two plans, for the failure.
fn difference(a: &Planned, b: &Planned) -> String {
    if (a.scored, a.layout) != (b.scored, b.layout) {
        return format!(
            "scored/layout {:?} vs {:?}",
            (a.scored, a.layout),
            (b.scored, b.layout)
        );
    }
    for (rank, (a, b)) in a.ranks.iter().zip(&b.ranks).enumerate() {
        if let Some(at) = a.iter().zip(b).position(|(x, y)| x != y) {
            return format!("rank {rank} launch {at}:\n{:?}\n{:?}", a[at], b[at]);
        }
        if a.len() != b.len() {
            return format!("rank {rank}: {} vs {} launches", a.len(), b.len());
        }
    }
    String::new()
}

/// Lowers each case of `cases` afresh on one instance, after `mutate`,
/// and checks that the cases of one key plan the same work. Returns each
/// key's case count.
fn sweep(
    ranks: Ranks,
    layout: KvLayout,
    dspark: bool,
    served: u32,
    cases: &[Case],
    mutate: fn(&mut Run),
) -> BTreeMap<Key, usize> {
    let (host, mut instance) = complete_host(ranks, layout, dspark, served);
    let mut counts = BTreeMap::new();
    super::super::INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let live = table.get_mut(instance.handle()).unwrap();
        assert_eq!(live.items().get(), served);
        let mut plans: BTreeMap<Key, (Case, Planned)> = BTreeMap::new();
        for case in cases {
            let mut run = live.plan_run(phase_inputs(case), None).unwrap();
            let rows = case.split.iter().sum();
            let items = case.split.len() as u32;
            assert_eq!(run.phase, Phase::Verify { rows, items });
            mutate(&mut run);
            let key = (run.bank.route, run.graph.unwrap());
            assert_eq!(key.1, verify_graph_key(rows));
            let plan = planned(&run);
            *counts.entry(key).or_default() += 1;
            match plans.get(&key) {
                None => {
                    plans.insert(key, (case.clone(), plan));
                }
                Some((first, expected)) => assert!(
                    plan == *expected,
                    "{ranks:?} {:?} dspark {dspark} served {served}: key {key:?} plans \
                     differ between {first:?} and {case:?} at {}",
                    layout.storage,
                    difference(expected, &plan),
                ),
            }
        }
    });
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    counts
}

fn unchanged(_: &mut Run) {}

/// The most verify rows at `served` items: every item at its deepest
/// block, within the rows `complete_host`'s instance serves (the default,
/// the widest decode bucket).
fn widest(served: u32) -> u32 {
    (served * MAX_VERIFY_ROWS).min(model_deepseek41_contract::serving::DECODE_BATCH)
}

/// Every reachable phase of each row count at `served` items.
fn reachable(served: u32) -> Vec<Case> {
    (2..=widest(served))
        .flat_map(|rows| cases(rows, served, served == 32))
        .collect()
}

#[test]
fn target_programs_are_a_function_of_their_graph_key() {
    let mut all = reachable(32);
    all.extend(witnesses(32));
    let counts = sweep(Ranks::TP4, KV_LAYOUTS[0], true, 32, &all, unchanged);
    // One Accepted key per row count, and the two context banks at 32
    // rows over eight items of four.
    let mut keys: Vec<Key> = (2..=32)
        .map(|rows| (IndexerRoute::Accepted, verify_graph_key(rows)))
        .collect();
    keys.push((IndexerRoute::VerifyHierarchy, verify_graph_key(32)));
    keys.push((IndexerRoute::Combined, verify_graph_key(32)));
    keys.sort();
    assert_eq!(counts.keys().copied().collect::<Vec<_>>(), keys);
    // Both witnesses' splits at each bound: four rows each reach a context
    // bank between its page-rounded bounds, and every uneven split stays
    // accepted.
    let banks = |route| counts[&(route, verify_graph_key(32))];
    assert_eq!(banks(IndexerRoute::VerifyHierarchy), 6);
    assert_eq!(banks(IndexerRoute::Combined), 6);
    let pairs: usize = (2..=32_u32)
        .map(|rows| (rows.div_ceil(MAX_VERIFY_ROWS)..=rows).count())
        .sum();
    assert_eq!(pairs, 31 + 426);
    assert!(counts.values().sum::<usize>() >= 3 * pairs + 36);
}

#[test]
fn target_programs_are_a_function_of_their_graph_key_at_each_served_count() {
    for served in [1, 8, 16, 24] {
        let mut all = reachable(served);
        if served >= 8 {
            all.extend(witnesses(served));
        }
        let counts = sweep(Ranks::TP4, KV_LAYOUTS[0], true, served, &all, unchanged);
        let banks = if served >= 8 { 2 } else { 0 };
        assert_eq!(
            counts.len(),
            widest(served) as usize - 1 + banks,
            "served {served}"
        );
    }
}

#[test]
fn target_programs_are_a_function_of_their_graph_key_at_every_topology_and_layout() {
    let rows = [2, 3, 5, 6, 7, 12, 13, 24, 31, 32];
    let mut all: Vec<Case> = rows
        .iter()
        .flat_map(|&rows| cases(rows, 32, false))
        .collect();
    all.extend(witnesses(32));
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        for layout in KV_LAYOUTS {
            for dspark in [false, true] {
                // The full sweep covers native TP4 DSpark; BF16 KV serves
                // autoregressive steps only, so its verify phase is ordinary
                // lowering without the drafter.
                let skip = (ranks, layout.storage, dspark) == (Ranks::TP4, KvStorage::Native, true)
                    || (layout.storage == KvStorage::Bf16 && dspark);
                if skip {
                    continue;
                }
                let counts = sweep(ranks, layout, dspark, 32, &all, unchanged);
                let banks = counts
                    .keys()
                    .filter(|(route, _)| *route != IndexerRoute::Accepted)
                    .count();
                let native = layout.storage == KvStorage::Native;
                assert_eq!(banks, if native { 2 } else { 0 }, "{ranks:?} {layout:?}");
                assert_eq!(counts.len(), rows.len() + banks);
            }
        }
    }
}

/// A launch that depends on the live items without the key would replay
/// wrongly at other item counts: the sweep refuses it.
#[test]
fn target_program_sweep_refuses_an_unkeyed_item_dependence() {
    fn unkeyed(run: &mut Run) {
        let Phase::Verify { items, .. } = run.phase else {
            unreachable!()
        };
        let launch = run.programs[0].last_mut().unwrap();
        launch.plan.grid[0] = items;
    }
    let all = cases(12, 32, true);
    let failed = std::panic::catch_unwind(|| {
        sweep(Ranks::TP1, KV_LAYOUTS[0], true, 32, &all, unkeyed);
    })
    .unwrap_err();
    let message = failed
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or_default();
    assert!(message.contains("plans differ"), "{message}");
    // The same cases agree once nothing depends on the live items.
    assert_eq!(
        sweep(Ranks::TP1, KV_LAYOUTS[0], true, 32, &all, unchanged).len(),
        1
    );
}

/// The route banks plan different launches at the same rows and items, so
/// a key without its route would replay the wrong schedule.
#[test]
fn route_banks_plan_different_launches_at_the_same_rows_and_items() {
    let (host, mut instance) = complete_host(Ranks::TP1, KV_LAYOUTS[0], true, 32);
    super::super::INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let live = table.get_mut(instance.handle()).unwrap();
        let plan = |split: Vec<u32>, end| {
            let run = live
                .plan_run(phase_inputs(&placed(split, vec![end; 8])), None)
                .unwrap();
            assert_eq!(run.phase, Phase::Verify { rows: 32, items: 8 });
            (run.bank.route, planned(&run))
        };
        let uneven = vec![1, 1, 5, 5, 5, 5, 5, 5];
        let (accepted, base) = plan(uneven.clone(), MEDIUM);
        assert_eq!(plan(uneven, LONG).0, IndexerRoute::Accepted);
        let (hierarchy, medium) = plan(vec![4; 8], MEDIUM);
        let (combined, long) = plan(vec![4; 8], LONG);
        assert_eq!(
            [accepted, hierarchy, combined],
            [
                IndexerRoute::Accepted,
                IndexerRoute::VerifyHierarchy,
                IndexerRoute::Combined
            ]
        );
        assert_ne!(base, medium);
        assert_ne!(base, long);
        assert_ne!(medium, long);
    });
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
