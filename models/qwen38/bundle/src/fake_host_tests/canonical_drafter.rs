//! The canonical drafter candidate (`CHOPINFER_QWEN38_CANONICAL_DRAFTER`):
//! every drafter row a prefilling item's step writes (its chunk-start
//! catch-up rows and every row its chunk completes) runs in the prefill
//! phase at every row count, never in a verify graph bucket. Without it a
//! continuing prefill's catch-up row, and the rows of a chunk of at most 33
//! tokens, run the decode entries of a graph bucket, so persistent drafter
//! state depends on where chunks started. These steps run with the
//! process's default execution: no test forces eager runs.

use super::speculation::{
    assert_drafter_rows, batch, committed_state, create_drafter, drafted, drafter_host_at, execute,
    kept, plain, truth, truths,
};
use super::*;
use crate::program::{Stage, Sublayer};

/// What one step's drafter did: per run, whether it holds rows a
/// prefilling item's step writes, whether it ran in the prefill phase, and
/// its rows; the graph replays the step issued; and, by stage, the symbols
/// its prefill-phase runs launched for the stages that write persistent
/// drafter state.
struct Drafted {
    runs: Vec<(bool, bool, u32)>,
    replays: u32,
    persistent: BTreeMap<String, BTreeSet<String>>,
}

impl Drafted {
    /// Every run of rows a prefilling item's step writes ran in the prefill
    /// phase.
    fn prefill_rows_in_prefill(&self) -> bool {
        self.runs
            .iter()
            .all(|&(prefill, phase, _)| !prefill || phase)
    }

    /// A step without drafting items ran every drafter run in the prefill
    /// phase.
    fn all_in_prefill(&self) -> bool {
        self.runs.iter().all(|run| run.1)
    }

    fn prefill_runs(&self) -> Vec<u32> {
        self.runs
            .iter()
            .filter(|run| run.0)
            .map(|run| run.2)
            .collect()
    }
}

/// Whether `stage` of the drafter layer feeds the state later rows read:
/// its K, V, QSA block keys and raw-key ring, through `IndexKeyWrite`. The
/// stages after it (selection, attention, the MLP and the reductions) feed
/// only the drafts.
pub(super) fn persistent(stage: Stage) -> bool {
    let mtp = Shape::TP4.mtp_layer();
    match stage {
        Stage::MtpGather
        | Stage::Embedding
        | Stage::MtpNorms
        | Stage::MtpFc
        | Stage::MtpCombine
        | Stage::FirstNorm => true,
        Stage::HcIn(layer, Sublayer::Attention)
        | Stage::HcBottleneck(layer, Sublayer::Attention)
        | Stage::HcUp(layer, Sublayer::Attention)
        | Stage::HcMix(layer, Sublayer::Attention)
        | Stage::AttnInProj(layer)
        | Stage::AttnQkNormRopeKv(layer)
        | Stage::IndexKeyWrite(layer) => layer == mtp,
        _ => false,
    }
}

/// A prefill item of `inputs` continuing at `previous`, of any length
/// (`plain` makes a one-token continuation a decode item).
fn prompt(
    request: u64,
    slot: u64,
    previous: u64,
    inputs: &[u32],
) -> (BatchItemV1, BatchAppendV1, u64) {
    let (_, append, slot) = plain(request, slot, previous, inputs);
    let item = BatchItemV1::new(request, false, inputs.len() as u32, true, inputs.to_vec());
    (item, append, slot)
}

/// An instance of `shape` with the drafter, the candidate on or off.
fn instance(shape: Shape, canonical: bool) -> (Rc<FakeHost>, BundleInstanceV1) {
    let host = drafter_host_at(3, shape);
    let instance = crate::instance::with_canonical_drafter(canonical, || create_drafter(&host));
    (host, instance)
}

/// Executes `request` and reports its drafter work.
fn step(host: &FakeHost, instance: &mut BundleInstanceV1, request: &BatchRequestV1) -> Drafted {
    crate::instance::DRAFTER_RUNS.with(|runs| runs.borrow_mut().clear());
    crate::instance::DRAFTER_LAUNCHES.with(|launches| launches.borrow_mut().clear());
    let replays = host.replays.get();
    execute(host, instance, request);
    let mut persistent_symbols = BTreeMap::<String, BTreeSet<String>>::new();
    for (prefill, stage, symbol) in
        crate::instance::DRAFTER_LAUNCHES.with(|launches| launches.take())
    {
        if prefill && persistent(stage) {
            persistent_symbols
                .entry(format!("{stage:?}"))
                .or_default()
                .insert(symbol);
        }
    }
    Drafted {
        runs: crate::instance::DRAFTER_RUNS.with(|runs| runs.take()),
        replays: host.replays.get() - replays,
        persistent: persistent_symbols,
    }
}

/// Prefills request 10 in slot 1 in `chunks` and returns the last chunk's
/// drafter work; every drafter row read the right target row and token.
fn prefill(shape: Shape, canonical: bool, chunks: &[u64]) -> Drafted {
    let (host, mut instance) = instance(shape, canonical);
    let mut start = 0;
    let mut last = None;
    for &rows in chunks {
        last = Some(step(
            &host,
            &mut instance,
            &batch(vec![prompt(10, 1, start, &truths(start..start + rows))]),
        ));
        instance.commit().unwrap();
        start += rows;
    }
    assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..start + 1))]));
    assert!(
        committed_state(&host, &instance, 1)
            .iter()
            .all(|history| *history == truths(0..start))
    );
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
    last.unwrap()
}

/// A continuing prefill's catch-up row runs in the prefill phase and, at
/// every stage that writes persistent drafter state, launches the entries
/// the same prompt's one-chunk prefill launches there: without the
/// candidate it replays a verify graph of the decode entries.
#[test]
fn a_continuing_prefill_catches_up_with_the_one_chunk_entries() {
    for shape in [Shape::TP2, Shape::TP4] {
        let whole = prefill(shape, true, &[129]);
        let chunked = prefill(shape, true, &[64, 65]);
        assert_eq!(chunked.prefill_runs(), [1, 64], "{shape:?}");
        assert!(chunked.all_in_prefill(), "{shape:?}");
        assert_eq!(chunked.replays, 0, "{shape:?}");
        assert_eq!(chunked.persistent, whole.persistent, "{shape:?}");
        assert_eq!(chunked.persistent.len(), 13, "{shape:?}");
        // The default runs the catch-up row in a verify graph bucket.
        let default = prefill(shape, false, &[64, 65]);
        assert_eq!(default.runs.iter().filter(|run| !run.1).count(), 1);
        assert!(default.replays > 0);
    }
}

/// Continuations of 1, 2, 32, 33 and 34 tokens (post-target runs of 0 to
/// 33 rows, both sides of the largest graph bucket) run every row they
/// write in the prefill phase and replay no graph.
#[test]
fn short_continuations_write_their_rows_in_the_prefill_phase() {
    for shape in [Shape::TP2, Shape::TP4] {
        for suffix in [1, 2, 32, 33, 34] {
            let canonical = prefill(shape, true, &[64, suffix]);
            let rows = if suffix > 1 {
                vec![1, suffix as u32 - 1]
            } else {
                vec![1]
            };
            assert_eq!(canonical.prefill_runs(), rows, "{shape:?} {suffix}");
            assert!(canonical.all_in_prefill(), "{shape:?} {suffix}");
            assert_eq!(canonical.replays, 0, "{shape:?} {suffix}");
            // The default runs every run of at most 32 rows in a graph.
            let default = prefill(shape, false, &[64, suffix]);
            let graphs = default.runs.iter().filter(|run| !run.1).count();
            assert_eq!(graphs, if suffix <= 33 && suffix > 1 { 2 } else { 1 });
        }
    }
}

/// A step with a drafting K3 verify item and a continuing prefill: the
/// verify item still catches up and drafts through its graph buckets, and
/// the prefill item's rows run in the prefill phase, apart from it. Both
/// requests' drafts, drafter rows and commits are right.
#[test]
fn a_mixed_step_drafts_in_graphs_and_writes_prefill_rows_in_the_prefill_phase() {
    for shape in [Shape::TP2, Shape::TP4] {
        let (host, mut instance) = instance(shape, true);
        execute(
            &host,
            &mut instance,
            &batch(vec![
                plain(10, 1, 0, &truths(0..70)),
                plain(11, 2, 0, &truths(0..64)),
            ]),
        );
        instance.commit().unwrap();
        let mixed = step(
            &host,
            &mut instance,
            &batch(vec![
                drafted(10, 1, 70, truth(70), 3),
                plain(11, 2, 64, &truths(64..129)),
            ]),
        );
        assert!(mixed.prefill_rows_in_prefill(), "{shape:?}");
        assert_eq!(mixed.prefill_runs(), [1, 64], "{shape:?}");
        // The verify item's first run and its two chained runs.
        let drafting = mixed.runs.iter().filter(|run| !run.0).collect::<Vec<_>>();
        assert_eq!(drafting.len(), 3, "{shape:?}");
        assert!(drafting.iter().all(|run| !run.1), "{shape:?}");
        assert!(mixed.replays > 0, "{shape:?}");
        instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
        let sink = execute(
            &host,
            &mut instance,
            &batch(vec![drafted(10, 1, 72, truth(72), 3)]),
        );
        assert_eq!(sink.draft_tokens(), [(10, truths(73..76))]);
        instance.commit_accepted(&kept(&[(10, 4)])).unwrap();
        assert_drafter_rows(
            &host,
            &BTreeMap::from([(1, truths(0..80)), (2, truths(0..130))]),
        );
        assert!(
            committed_state(&host, &instance, 2)
                .iter()
                .all(|history| *history == truths(0..129))
        );
        instance.destroy().unwrap();
    }
}

/// Eight rows outstanding after decodes, then a full chunk: the catch-up
/// run takes the eight rows, the post-target run the chunk's rows minus
/// one, both in the prefill phase and within the drafter's rows; the state
/// equals one token at a time and drafting resumes from the chunk's end.
#[test]
fn a_lagging_slot_catches_up_before_a_full_chunk_in_the_prefill_phase() {
    let (host, mut instance) = instance(Shape::TP4, true);
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(10, 0, 0, &truths(0..3))]),
    );
    instance.commit().unwrap();
    for length in 3..10 {
        execute(
            &host,
            &mut instance,
            &batch(vec![plain(10, 0, length, &[truth(length)])]),
        );
        instance.commit().unwrap();
    }
    let chunk = truths(10..10 + u64::from(PREFILL_TOKENS));
    let full = step(&host, &mut instance, &batch(vec![plain(10, 0, 10, &chunk)]));
    assert_eq!(full.prefill_runs(), [8, PREFILL_TOKENS - 1]);
    assert!(full.all_in_prefill());
    instance.commit().unwrap();
    let length = 10 + u64::from(PREFILL_TOKENS);
    assert!(
        committed_state(&host, &instance, 0)
            .iter()
            .all(|history| *history == truths(0..length))
    );
    assert_drafter_rows(&host, &BTreeMap::from([(0, truths(0..length + 1))]));
    let sink = execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 0, length, truth(length), 1)]),
    );
    assert_eq!(sink.draft_tokens(), [(10, vec![truth(length + 1)])]);
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    instance.destroy().unwrap();
}

/// A snapshot whose drafter frontier lags its length (a capture after a
/// verify commit) restores, and a prefill continuing from it catches the
/// lagging rows up in the prefill phase.
#[test]
fn a_restored_lagging_frontier_catches_up_in_the_prefill_phase() {
    for shape in [Shape::TP2, Shape::TP4] {
        let (host, mut instance) = instance(shape, true);
        host.backing.set(true);
        let pages = instance.describe_prefix_state().unwrap();
        let snapshot = (500..500 + pages).collect::<Vec<_>>();
        execute(
            &host,
            &mut instance,
            &batch(vec![plain(10, 1, 0, &truths(0..70))]),
        );
        instance.commit().unwrap();
        execute(
            &host,
            &mut instance,
            &batch(vec![drafted(10, 1, 70, truth(70), 3)]),
        );
        instance.commit_accepted(&kept(&[(10, 3)])).unwrap();
        // Length 73, frontier 70.
        instance.capture_prefix_state(10, 73, &snapshot).unwrap();
        instance.retire_request(10).unwrap();
        assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..80))]));
        instance
            .restore_prefix_state_in_slot(11, &truths(0..73), &snapshot, 1)
            .unwrap();
        let resumed = step(
            &host,
            &mut instance,
            &batch(vec![plain(11, 1, 73, &truths(73..140))]),
        );
        assert_eq!(resumed.prefill_runs(), [3, 66], "{shape:?}");
        assert!(resumed.all_in_prefill(), "{shape:?}");
        assert_eq!(resumed.replays, 0, "{shape:?}");
        instance.commit().unwrap();
        assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..141))]));
        assert!(
            committed_state(&host, &instance, 1)
                .iter()
                .all(|history| *history == truths(0..140))
        );
        instance.destroy().unwrap();
    }
}

/// A cancelled continuing prefill retires its request (in-place recurrent
/// cancellation); the prompt prefilled again runs the same drafter runs
/// and commits the rows and state of an uncancelled run.
#[test]
fn a_cancelled_continuing_prefill_runs_again_as_before() {
    for shape in [Shape::TP2, Shape::TP4] {
        let (host, mut instance) = instance(shape, true);
        let first = batch(vec![plain(10, 1, 0, &truths(0..64))]);
        let continuing = batch(vec![plain(10, 1, 64, &truths(64..90))]);
        execute(&host, &mut instance, &first);
        instance.commit().unwrap();
        let cancelled = step(&host, &mut instance, &continuing);
        instance.cancel().unwrap();
        instance.retire_request(10).unwrap();
        execute(&host, &mut instance, &first);
        instance.commit().unwrap();
        let again = step(&host, &mut instance, &continuing);
        assert_eq!(cancelled.runs, again.runs, "{shape:?}");
        assert!(again.all_in_prefill(), "{shape:?}");
        instance.commit().unwrap();
        assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..91))]));
        assert!(
            committed_state(&host, &instance, 1)
                .iter()
                .all(|history| *history == truths(0..90))
        );
        instance.destroy().unwrap();
    }
}
