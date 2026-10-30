//! The canonical drafter candidate (`CHOPINFER_QWEN38_CANONICAL_DRAFTER`):
//! every drafter row a prefilling item's step writes (its chunk-start
//! catch-up rows and every row its chunk completes) runs in the prefill
//! phase at every row count, never in a verify graph bucket. Without it a
//! continuing prefill's catch-up row, and the rows of a chunk of at most 33
//! tokens, run the decode entries of a graph bucket, so persistent drafter
//! state depends on where chunks started. `Full` (`1`) runs the whole
//! drafter program; `Persistent` (`2`) runs only the stages whose writes a
//! later run reads, and must launch exactly `Full`'s launches of those
//! stages. Without the candidate, runs of prefilling rows already in the
//! prefill phase run only the persistent stages too. These steps run with
//! the process's default execution: no test forces eager runs unless it says
//! so.

use super::speculation::{
    assert_drafter_rows, batch, committed_state, create_drafter, drafted, drafter_host_at, execute,
    kept, plain, truth, truths,
};
use super::*;
use crate::instance::{Canonical, DrafterLaunch};
use crate::program::{Stage, Sublayer};

/// Both candidates: the whole drafter program, and its persistent stages.
const CANDIDATES: [Canonical; 2] = [Canonical::Full, Canonical::Persistent];

/// What one step's drafter did: per run, whether it holds rows a
/// prefilling item's step writes, whether it ran in the prefill phase, and
/// its rows; the graph replays the step issued; the drafts it emitted;
/// and, by stage, the symbols its prefill-phase runs launched for the
/// stages that write persistent drafter state.
struct Drafted {
    runs: Vec<(bool, bool, u32)>,
    replays: u32,
    drafts: Vec<(u64, Vec<u32>)>,
    persistent: BTreeMap<String, BTreeSet<String>>,
    /// Every launch the step's drafter runs lowered.
    launches: Vec<DrafterLaunch>,
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
        | Stage::HcDown(layer, Sublayer::Attention)
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

/// An instance of `shape` with the drafter and the candidate `mode`.
fn instance(shape: Shape, mode: Canonical) -> (Rc<FakeHost>, BundleInstanceV1) {
    let host = drafter_host_at(3, shape);
    let instance = crate::instance::with_canonical_drafter(mode, || create_drafter(&host));
    (host, instance)
}

/// Executes `request` and reports its drafter work.
fn step(host: &FakeHost, instance: &mut BundleInstanceV1, request: &BatchRequestV1) -> Drafted {
    crate::instance::DRAFTER_RUNS.with(|runs| runs.borrow_mut().clear());
    crate::instance::DRAFTER_LAUNCHES.with(|launches| launches.borrow_mut().clear());
    let replays = host.replays.get();
    let sink = execute(host, instance, request);
    let launches = crate::instance::DRAFTER_LAUNCHES.with(|launches| launches.take());
    let mut persistent_symbols = BTreeMap::<String, BTreeSet<String>>::new();
    for launch in &launches {
        if launch.prefill && persistent(launch.stage) {
            persistent_symbols
                .entry(format!("{:?}", launch.stage))
                .or_default()
                .insert(launch.symbol.clone());
        }
    }
    Drafted {
        runs: crate::instance::DRAFTER_RUNS.with(|runs| runs.take()),
        replays: host.replays.get() - replays,
        drafts: sink.draft_tokens(),
        persistent: persistent_symbols,
        launches,
    }
}

/// Prefills request 10 in slot 1 in `chunks` and returns the last chunk's
/// drafter work; every drafter row read the right target row and token.
fn prefill(shape: Shape, mode: Canonical, chunks: &[u64]) -> Drafted {
    prefill_steps(shape, mode, chunks).pop().unwrap()
}

/// [`prefill`], returning every chunk's drafter work.
fn prefill_steps(shape: Shape, mode: Canonical, chunks: &[u64]) -> Vec<Drafted> {
    let (host, mut instance) = instance(shape, mode);
    let mut start = 0;
    let mut steps = Vec::new();
    for &rows in chunks {
        steps.push(step(
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
    steps
}

/// Every shape with each candidate.
fn shapes_and_candidates() -> Vec<(Shape, Canonical)> {
    [Shape::TP2, Shape::TP4]
        .into_iter()
        .flat_map(|shape| CANDIDATES.map(|mode| (shape, mode)))
        .collect()
}

/// A continuing prefill's catch-up row runs in the prefill phase and, at
/// every stage that writes persistent drafter state, launches the entries
/// the same prompt's one-chunk prefill launches there: without the
/// candidate it replays a verify graph of the decode entries.
#[test]
fn a_continuing_prefill_catches_up_with_the_one_chunk_entries() {
    for (shape, mode) in shapes_and_candidates() {
        let whole = prefill(shape, mode, &[129]);
        let chunked = prefill(shape, mode, &[64, 65]);
        assert_eq!(chunked.prefill_runs(), [1, 64], "{shape:?} {mode:?}");
        assert!(chunked.all_in_prefill(), "{shape:?} {mode:?}");
        assert_eq!(chunked.replays, 0, "{shape:?} {mode:?}");
        assert_eq!(chunked.persistent, whole.persistent, "{shape:?} {mode:?}");
        assert_eq!(chunked.persistent.len(), 13, "{shape:?} {mode:?}");
    }
    for shape in [Shape::TP2, Shape::TP4] {
        // The default runs the catch-up row in a verify graph bucket.
        let default = prefill(shape, Canonical::Off, &[64, 65]);
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
            let rows = if suffix > 1 {
                vec![1, suffix as u32 - 1]
            } else {
                vec![1]
            };
            for mode in CANDIDATES {
                let canonical = prefill(shape, mode, &[64, suffix]);
                assert_eq!(
                    canonical.prefill_runs(),
                    rows,
                    "{shape:?} {mode:?} {suffix}"
                );
                assert!(canonical.all_in_prefill(), "{shape:?} {mode:?} {suffix}");
                assert_eq!(canonical.replays, 0, "{shape:?} {mode:?} {suffix}");
            }
            // The default runs every run of at most 32 rows in a graph.
            let default = prefill(shape, Canonical::Off, &[64, suffix]);
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
    for (shape, mode) in shapes_and_candidates() {
        mixed_steps(shape, mode);
    }
}

/// [`a_mixed_step_drafts_in_graphs_and_writes_prefill_rows_in_the_prefill_phase`]'s
/// mixed step and the drafted step after it.
fn mixed_steps(shape: Shape, mode: Canonical) -> Vec<Drafted> {
    let (host, mut instance) = instance(shape, mode);
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
    if mode.on() {
        assert!(mixed.prefill_rows_in_prefill(), "{shape:?} {mode:?}");
        assert_eq!(mixed.prefill_runs(), [1, 64], "{shape:?} {mode:?}");
    } else {
        // The catch-up row joins the verify item's first run.
        assert_eq!(mixed.prefill_runs(), [64], "{shape:?} {mode:?}");
    }
    // The verify item's first run and its two chained runs.
    let drafting = mixed.runs.iter().filter(|run| !run.0).collect::<Vec<_>>();
    assert_eq!(drafting.len(), 3, "{shape:?} {mode:?}");
    assert!(drafting.iter().all(|run| !run.1), "{shape:?} {mode:?}");
    assert!(mixed.replays > 0, "{shape:?} {mode:?}");
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    let after = step(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 1, 72, truth(72), 3)]),
    );
    assert_eq!(after.drafts, [(10, truths(73..76))]);
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
    vec![mixed, after]
}

/// Eight rows outstanding after decodes, then a full chunk: the catch-up
/// run takes the eight rows, the post-target run the chunk's rows minus
/// one, both in the prefill phase and within the drafter's rows; the state
/// equals one token at a time and drafting resumes from the chunk's end.
#[test]
fn a_lagging_slot_catches_up_before_a_full_chunk_in_the_prefill_phase() {
    for mode in CANDIDATES {
        lagging_steps(mode);
    }
}

/// [`a_lagging_slot_catches_up_before_a_full_chunk_in_the_prefill_phase`]'s
/// full chunk and the drafted step after it.
fn lagging_steps(mode: Canonical) -> Vec<Drafted> {
    let (host, mut instance) = instance(Shape::TP4, mode);
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
    if mode.on() {
        assert_eq!(full.prefill_runs(), [8, PREFILL_TOKENS - 1], "{mode:?}");
        assert!(full.all_in_prefill(), "{mode:?}");
    } else {
        assert_eq!(full.prefill_runs(), [PREFILL_TOKENS - 1], "{mode:?}");
    }
    instance.commit().unwrap();
    let length = 10 + u64::from(PREFILL_TOKENS);
    assert!(
        committed_state(&host, &instance, 0)
            .iter()
            .all(|history| *history == truths(0..length))
    );
    assert_drafter_rows(&host, &BTreeMap::from([(0, truths(0..length + 1))]));
    let after = step(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 0, length, truth(length), 1)]),
    );
    assert_eq!(after.drafts, [(10, vec![truth(length + 1)])]);
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    instance.destroy().unwrap();
    vec![full, after]
}

/// A snapshot whose drafter frontier lags its length (a capture after a
/// verify commit) restores, and a prefill continuing from it catches the
/// lagging rows up in the prefill phase.
#[test]
fn a_restored_lagging_frontier_catches_up_in_the_prefill_phase() {
    for (shape, mode) in shapes_and_candidates() {
        restored_steps(shape, mode);
    }
}

/// [`a_restored_lagging_frontier_catches_up_in_the_prefill_phase`]'s
/// continuing prefill.
fn restored_steps(shape: Shape, mode: Canonical) -> Vec<Drafted> {
    let (host, mut instance) = instance(shape, mode);
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
    if mode.on() {
        assert_eq!(resumed.prefill_runs(), [3, 66], "{shape:?} {mode:?}");
        assert!(resumed.all_in_prefill(), "{shape:?} {mode:?}");
        assert_eq!(resumed.replays, 0, "{shape:?} {mode:?}");
    } else {
        assert_eq!(resumed.prefill_runs(), [66], "{shape:?} {mode:?}");
    }
    instance.commit().unwrap();
    assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..141))]));
    assert!(
        committed_state(&host, &instance, 1)
            .iter()
            .all(|history| *history == truths(0..140))
    );
    instance.destroy().unwrap();
    vec![resumed]
}

/// A cancelled continuing prefill retires its request (in-place recurrent
/// cancellation); the prompt prefilled again runs the same drafter runs
/// and commits the rows and state of an uncancelled run.
#[test]
fn a_cancelled_continuing_prefill_runs_again_as_before() {
    for (shape, mode) in shapes_and_candidates() {
        cancelled_steps(shape, mode);
    }
}

/// [`a_cancelled_continuing_prefill_runs_again_as_before`]'s cancelled and
/// repeated continuations.
fn cancelled_steps(shape: Shape, mode: Canonical) -> Vec<Drafted> {
    let (host, mut instance) = instance(shape, mode);
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
    assert_eq!(cancelled.runs, again.runs, "{shape:?} {mode:?}");
    assert!(!mode.on() || again.all_in_prefill(), "{shape:?} {mode:?}");
    instance.commit().unwrap();
    assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..91))]));
    assert!(
        committed_state(&host, &instance, 1)
            .iter()
            .all(|history| *history == truths(0..90))
    );
    instance.destroy().unwrap();
    vec![cancelled, again]
}

/// The chunk lists the audit and the launch comparison prefill.
const AUDITED_CHUNKS: [&[u64]; 9] = [
    &[129],
    &[64, 65],
    &[64, 1],
    &[64, 2],
    &[64, 32],
    &[64, 33],
    &[64, 34],
    &[3, 66],
    &[64, PREFILL_TOKENS as u64],
];

/// A cold prompt of 150 tokens in slot 3 that keeps an anchor.
fn anchored_step(shape: Shape, mode: Canonical) -> Drafted {
    let (host, mut instance) = instance(shape, mode);
    let (item, append, slot) = prompt(13, 3, 0, &truths(0..150));
    let anchored = step(
        &host,
        &mut instance,
        &batch(vec![(item.with_anchor(true), append, slot)]),
    );
    assert!(anchored.all_in_prefill(), "{shape:?} {mode:?}");
    instance.commit().unwrap();
    instance.destroy().unwrap();
    anchored
}

/// Every entry the candidate's prefill-phase runs launch for a stage that
/// writes persistent drafter state declares what those runs rely on:
/// - rows keep their bits at any cut (`Chunk::Any`), since a run starts at
///   a continuing item's frontier (row `A - 1` after a hit at `A`, a lagging
///   restore's) or at its chunk's start, and holds from one row to a full
///   chunk;
/// - rows keep their bits whatever other items share the launch
///   (`Batch::Independent`);
/// - per-slot state is kept at an anchor frontier at least `QSA_RING` rows
///   past the item's first row in the run, the only anchors the drafter
///   asks a run to keep (`run_drafter_with`).
///
/// Chunk lists ([`AUDITED_CHUNKS`]) cover runs of 1 to 128 rows, both
/// sides of the largest graph bucket, a lagging catch-up of three rows and
/// a full chunk; one prompt keeps an anchor. Both candidates are audited
/// (`Persistent` launches `Full`'s entries but the inject). A mixed
/// component recording the MTP layer launches the fixed program's entries
/// (`mixed_mtp`), so this covers it too.
#[test]
fn prefill_rows_launch_entries_that_keep_rows_at_any_cut() {
    use kernel_core::properties::{Batch, Chunk};
    let owned = model_qwen38_contract::owned::declared();
    let catalogs = model_qwen38_contract::sm120::catalogs();
    let mut violations = BTreeSet::new();
    for (shape, mode) in shapes_and_candidates() {
        let mut launched = BTreeMap::<String, BTreeSet<String>>::new();
        let mut record = |drafted: &Drafted| {
            for (stage, symbols) in &drafted.persistent {
                launched
                    .entry(stage.clone())
                    .or_default()
                    .extend(symbols.iter().cloned());
            }
        };
        for chunks in AUDITED_CHUNKS {
            for drafted in prefill_steps(shape, mode, chunks) {
                assert!(drafted.all_in_prefill(), "{shape:?} {mode:?} {chunks:?}");
                record(&drafted);
            }
        }
        record(&anchored_step(shape, mode));
        assert_eq!(launched.len(), 13, "{shape:?} {mode:?}");
        for (stage, symbols) in &launched {
            for symbol in symbols {
                let declared = owned
                    .iter()
                    .find(|declared| declared.symbol == *symbol)
                    .cloned()
                    .or_else(|| super::sm120::declaration(catalogs, symbol));
                let Some(declared) = declared else {
                    violations.insert(format!("{stage} {symbol}: undeclared"));
                    continue;
                };
                let invariance = declared.invariance;
                if invariance.chunk != Chunk::Any {
                    violations.insert(format!("{stage} {symbol}: chunks {:?}", invariance.chunk));
                }
                if invariance.batch != Batch::Independent {
                    violations.insert(format!("{stage} {symbol}: batch {:?}", invariance.batch));
                }
                if !invariance.anchors.supported_after(QSA_RING) {
                    violations.insert(format!(
                        "{stage} {symbol}: anchors {:?}",
                        invariance.anchors
                    ));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "{}",
        violations.into_iter().collect::<Vec<_>>().join("\n")
    );
}

/// Whether `launch` is `HcIn`'s inject projection, which `Stage::HcDown`
/// leaves out.
fn inject(launch: &DrafterLaunch) -> bool {
    matches!(launch.stage, Stage::HcIn(..)) && launch.symbol.contains("hc_inject")
}

/// A step's launches by run and rank, each in lowering order.
fn by_run_and_rank(
    launches: impl IntoIterator<Item = DrafterLaunch>,
) -> BTreeMap<(usize, usize), Vec<DrafterLaunch>> {
    let mut grouped = BTreeMap::<_, Vec<_>>::new();
    for launch in launches {
        grouped
            .entry((launch.run, launch.rank))
            .or_default()
            .push(launch);
    }
    grouped
}

/// Which runs are trimmed, by `(prefilling rows, prefill phase, rows)`:
/// with `Persistent` every run of prefilling rows, by default those of them
/// in the prefill phase.
type Trims = fn(&(bool, bool, u32)) -> bool;

/// `Persistent` against `Full`.
fn every_prefilling_run(run: &(bool, bool, u32)) -> bool {
    run.0
}

/// The default.
fn prefill_phase_prefilling_runs(run: &(bool, bool, u32)) -> bool {
    run.0 && run.1
}

/// The launches a trimming candidate must lower for a step that the program
/// it trims lowered as `full`: a run it trims keeps only its launches of the
/// persistent stages, without the inject and with `HcIn`'s as `HcDown`'s;
/// every other run keeps every launch.
fn persistent_launches(
    full: &Drafted,
    trims: Trims,
) -> BTreeMap<(usize, usize), Vec<DrafterLaunch>> {
    let trimmed = |launch: &DrafterLaunch| trims(&full.runs[launch.run]);
    let kept = full
        .launches
        .iter()
        .filter(|launch| !trimmed(launch) || (persistent(launch.stage) && !inject(launch)))
        .cloned()
        .map(|mut launch| {
            if trimmed(&launch)
                && let Stage::HcIn(layer, sublayer) = launch.stage
            {
                launch.stage = Stage::HcDown(layer, sublayer);
            }
            launch
        });
    by_run_and_rank(kept)
}

/// The trimming candidate ran each step of `steps` as the program it trims
/// ran `full`'s: the same runs, phases and replays, and per run and rank the
/// same launches (kind, plan and arguments as the host receives them), but
/// for the stages and the inject a run it trims leaves out. Returns how many
/// inject launches it left out.
fn assert_persistent_launches(
    full: &[Drafted],
    steps: &[Drafted],
    trims: Trims,
    case: &str,
) -> usize {
    assert_eq!(full.len(), steps.len(), "{case}");
    let mut injects = 0;
    for (index, (full, persistent)) in full.iter().zip(steps).enumerate() {
        assert_eq!(full.runs, persistent.runs, "{case} step {index}");
        assert_eq!(full.replays, persistent.replays, "{case} step {index}");
        assert_eq!(full.drafts, persistent.drafts, "{case} step {index}");
        let expected = persistent_launches(full, trims);
        let launched = by_run_and_rank(persistent.launches.iter().cloned());
        assert_eq!(
            expected.keys().collect::<Vec<_>>(),
            launched.keys().collect::<Vec<_>>(),
            "{case} step {index}"
        );
        for (key, expected) in &expected {
            assert!(
                launched[key] == *expected,
                "{case} step {index} run and rank {key:?}: {:#?}\nexpected {expected:#?}",
                launched[key]
            );
        }
        injects += full
            .launches
            .iter()
            .filter(|launch| trims(&full.runs[launch.run]) && inject(launch))
            .count();
    }
    injects
}

/// Each flow's steps under `mode`.
fn flows(shape: Shape, mode: Canonical) -> Vec<(String, Vec<Drafted>)> {
    let mut flows = AUDITED_CHUNKS
        .iter()
        .map(|chunks| {
            (
                format!("chunks {chunks:?}"),
                prefill_steps(shape, mode, chunks),
            )
        })
        .collect::<Vec<_>>();
    flows.push(("anchored".into(), vec![anchored_step(shape, mode)]));
    flows.push(("mixed".into(), mixed_steps(shape, mode)));
    flows.push(("restored".into(), restored_steps(shape, mode)));
    flows.push(("cancelled".into(), cancelled_steps(shape, mode)));
    flows.push(("wide catch-up".into(), wide_steps(shape, mode, 0)));
    flows.push(("wide drafting".into(), wide_steps(shape, mode, 1)));
    if shape == Shape::TP4 {
        flows.push(("lagging".into(), lagging_steps(mode)));
    }
    flows
}

/// In every flow `Persistent` lowers exactly `Full`'s launches of the
/// persistent stages for each run of prefilling rows (the down projection
/// of `HcIn` alone), at every rank, with the same plans and arguments; it
/// lowers every other run exactly as `Full` does. So it writes the same
/// persistent drafter state wherever `Full` does, and drafts the same.
#[test]
fn persistent_runs_lower_the_full_programs_persistent_launches() {
    for shape in [Shape::TP2, Shape::TP4] {
        let full = flows(shape, Canonical::Full);
        let persistent = flows(shape, Canonical::Persistent);
        let mut injects = 0;
        for ((case, full), (_, steps)) in full.iter().zip(&persistent) {
            injects += assert_persistent_launches(
                full,
                steps,
                every_prefilling_run,
                &format!("{shape:?} {case}"),
            );
        }
        assert!(injects > 0, "{shape:?}");
    }
}

/// Context-parallel size 2 at TP4: a continuing prefill's runs lower
/// `Full`'s persistent launches, owner-routed key writes included, and a
/// verify step after it lowers every launch, queued whole on each rank
/// without refusing the ranks' collectives.
#[test]
fn context_parallel_persistent_runs_lower_the_full_programs_persistent_launches() {
    let full = context_parallel_steps(Canonical::Full);
    let persistent = context_parallel_steps(Canonical::Persistent);
    assert_eq!(full[1].prefill_runs(), [1, 64]);
    assert!(assert_persistent_launches(&full, &persistent, every_prefilling_run, "dcp2") > 0);
}

/// Two chunks of a prompt at context-parallel size 2, then a verify step.
fn context_parallel_steps(mode: Canonical) -> Vec<Drafted> {
    use super::context_parallel::{create_at, item, pages};
    crate::context_parallel::serve_in_tests(true);
    let _one_lane = legacy_overlap_off();
    {
        let host = drafter_host_at(3, Shape::TP4);
        let mut instance = crate::instance::with_canonical_drafter(mode, || create_at(&host, 2));
        let mut steps = Vec::new();
        for (previous, end) in [(0, 64), (64, 129)] {
            steps.push(step(
                &host,
                &mut instance,
                &batch(vec![item(10, 1, previous, &truths(previous..end), 2)]),
            ));
            instance.commit().unwrap();
        }
        let (verify, _, slot) = drafted(10, 1, 129, truth(129), 3);
        let append = BatchAppendV1::new(129, 4, 133, pages(1, 133, 2), None);
        steps.push(step(
            &host,
            &mut instance,
            &batch(vec![(verify, append, slot)]),
        ));
        instance.commit_accepted(&kept(&[(10, 4)])).unwrap();
        instance.retire_request(10).unwrap();
        instance.destroy().unwrap();
        steps
    }
}

/// Five requests in reordered high slots, prefilled with three tokens each
/// and decoded to length 10 without drafting, which leaves eight rows each
/// outstanding; then a step of `drafts` drafts each. Its first run holds
/// 40 rows, too many for a graph bucket: without drafts a decode catch-up
/// run, with them a drafting run, both in the prefill phase and neither a
/// run of prefilling rows. Returns that step.
fn wide_steps(shape: Shape, mode: Canonical, drafts: u32) -> Vec<Drafted> {
    const SLOTS: [u64; 5] = [31, 27, 30, 28, 29];
    let (host, mut instance) = instance(shape, mode);
    let requests = || {
        SLOTS
            .iter()
            .enumerate()
            .map(|(index, &slot)| (20 + index as u64, slot))
    };
    // High state slots on low KV pages: the fixture's page pool is smaller
    // than slot * 64 there.
    let group = |request: u64| request - 19;
    let in_slot = |(item, append, _): (BatchItemV1, BatchAppendV1, u64), slot| (item, append, slot);
    execute(
        &host,
        &mut instance,
        &batch(
            requests()
                .map(|(request, slot)| {
                    in_slot(plain(request, group(request), 0, &truths(0..3)), slot)
                })
                .collect(),
        ),
    );
    instance.commit().unwrap();
    for length in 3..10 {
        let decoded = step(
            &host,
            &mut instance,
            &batch(
                requests()
                    .map(|(request, slot)| {
                        in_slot(
                            plain(request, group(request), length, &[truth(length)]),
                            slot,
                        )
                    })
                    .collect(),
            ),
        );
        assert!(decoded.runs.is_empty(), "{shape:?} {mode:?} {length}");
        instance.commit().unwrap();
    }
    let items = requests()
        .map(|(request, slot)| {
            let item = if drafts == 0 {
                plain(request, group(request), 10, &[truth(10)])
            } else {
                drafted(request, group(request), 10, truth(10), drafts)
            };
            in_slot(item, slot)
        })
        .collect();
    let wide = step(&host, &mut instance, &batch(items));
    assert_eq!(
        wide.runs[0],
        (false, true, 40),
        "{shape:?} {mode:?} {drafts}"
    );
    let expected = requests()
        .filter(|_| drafts > 0)
        .map(|(request, _)| (request, truths(11..11 + u64::from(drafts))))
        .collect::<Vec<_>>();
    assert_eq!(wide.drafts, expected, "{shape:?} {mode:?}");
    if drafts == 0 {
        instance.commit().unwrap();
    } else {
        let kept_rows = requests()
            .map(|(request, _)| (request, drafts + 1))
            .collect::<Vec<_>>();
        instance.commit_accepted(&kept(&kept_rows)).unwrap();
    }
    assert_drafter_rows(
        &host,
        &SLOTS
            .iter()
            .map(|&slot| (slot as u32, truths(0..20)))
            .collect::<BTreeMap<_, _>>(),
    );
    instance.destroy().unwrap();
    vec![wide]
}

/// Per run and rank, the stages a step's drafter lowered, in order.
fn stages(drafted: &Drafted) -> BTreeMap<(usize, usize), Vec<Stage>> {
    by_run_and_rank(drafted.launches.iter().cloned())
        .into_iter()
        .map(|(key, launches)| {
            let mut stages = launches
                .iter()
                .map(|launch| launch.stage)
                .collect::<Vec<_>>();
            stages.dedup();
            (key, stages)
        })
        .collect()
}

/// Whether `stages` keep the order of `program`'s, some perhaps left out
/// (a stage a graph bucket fuses into another lowers nothing).
fn within(stages: &[Stage], program: &[Stage]) -> bool {
    let mut program = program.iter();
    stages
        .iter()
        .all(|stage| program.any(|other| other == stage))
}

/// `Persistent` truncates by a run's purpose, not its phase: each run of
/// prefilling rows lowers exactly the persistent stages, and every other
/// run its whole program through its last stage, whether a graph bucket, a
/// wide run in the prefill phase (a decode catch-up or a drafting run of 40
/// rows) or, with every drafter run forced eager, a drafting run in the
/// prefill phase.
#[test]
fn only_runs_of_prefilling_rows_stop_at_the_key_write() {
    let check = |shape: Shape, case: &str, steps: &[Drafted]| {
        check_trims(shape, case, steps, every_prefilling_run).1
    };
    for shape in [Shape::TP2, Shape::TP4] {
        let mode = Canonical::Persistent;
        assert!(
            check(shape, "mixed", &mixed_steps(shape, mode)) > 0,
            "{shape:?}"
        );
        let catch_up = wide_steps(shape, mode, 0);
        assert!(check(shape, "wide catch-up", &catch_up) > 0, "{shape:?}");
        let last = *stages(&catch_up[0])
            .values()
            .next()
            .unwrap()
            .last()
            .unwrap();
        assert_eq!(last, Stage::Reduce(shape.mtp_layer(), Sublayer::Mlp));
        let drafting = wide_steps(shape, mode, 1);
        assert!(
            drafting[0]
                .launches
                .iter()
                .any(|launch| launch.stage == Stage::DraftFinal)
        );
        assert!(check(shape, "wide drafting", &drafting) > 0, "{shape:?}");
        // Forced eager, the drafting runs run in the prefill phase too.
        crate::instance::FORCE_EAGER.with(|eager| eager.set(Some(true)));
        let eager = eager_mixed_step(shape, mode);
        crate::instance::FORCE_EAGER.with(|eager| eager.set(None));
        assert!(eager.all_in_prefill(), "{shape:?}");
        assert!(eager.runs.iter().any(|run| !run.0), "{shape:?}");
        assert!(check(shape, "eager mixed", &[eager]) > 0, "{shape:?}");
    }
}

/// Checks that in `steps` each run `trims` picks lowered exactly the
/// persistent stages and every other run its whole program through its last
/// stage. Returns how many runs of each kind it checked.
fn check_trims(shape: Shape, case: &str, steps: &[Drafted], trims: Trims) -> (usize, usize) {
    let (mut trimmed, mut whole) = (0, 0);
    for (index, drafted) in steps.iter().enumerate() {
        for ((run, rank), stages) in stages(drafted) {
            let at = format!("{shape:?} {case} step {index} run {run} rank {rank}");
            if trims(&drafted.runs[run]) {
                assert_eq!(
                    stages,
                    crate::program::drafter_persistent_stages(shape),
                    "{at}"
                );
                trimmed += 1;
                continue;
            }
            let drafting = drafted
                .launches
                .iter()
                .any(|launch| launch.run == run && launch.stage == Stage::DraftFinal);
            let program = crate::program::drafter_stages_for(shape, drafting);
            assert!(within(&stages, &program), "{at}: {stages:?}");
            assert_eq!(stages.last(), program.last(), "{at}");
            whole += 1;
        }
    }
    (trimmed, whole)
}

/// By default runs are trimmed by purpose and phase: a run of prefilling
/// rows in the prefill phase (a post-target run of more than 32 rows, or any
/// such run with every drafter run forced eager) lowers exactly the
/// persistent stages; a run of prefilling rows in a graph bucket (a chunk of
/// 33 tokens), a wide decode catch-up or drafting run of 40 rows in the
/// prefill phase, and every drafting run lower their whole programs.
#[test]
fn only_prefill_phase_runs_of_prefilling_rows_are_trimmed() {
    let mode = Canonical::Off;
    for shape in [Shape::TP2, Shape::TP4] {
        let check = |case: &str, steps: &[Drafted]| {
            check_trims(shape, case, steps, prefill_phase_prefilling_runs)
        };
        let mixed = mixed_steps(shape, mode);
        assert_eq!(mixed[0].runs.iter().filter(|run| run.0).count(), 1);
        let (trimmed, whole) = check("mixed", &mixed);
        assert!(trimmed > 0 && whole > 0, "{shape:?} mixed");
        // The second chunk's 32 rows replay a whole graph bucket.
        let chunks = prefill_steps(shape, mode, &[64, 33]);
        assert_eq!(
            chunks[1]
                .runs
                .iter()
                .filter(|run| run.0)
                .collect::<Vec<_>>(),
            [&(true, false, 32)]
        );
        let (trimmed, whole) = check("chunks [64, 33]", &chunks);
        assert!(trimmed > 0 && whole > 0, "{shape:?} chunks");
        for drafts in [0, 1] {
            let wide = wide_steps(shape, mode, drafts);
            let (trimmed, whole) = check("wide", &wide);
            assert!(trimmed == 0 && whole > 0, "{shape:?} wide {drafts}");
        }
        crate::instance::FORCE_EAGER.with(|eager| eager.set(Some(true)));
        let eager = eager_mixed_step(shape, mode);
        crate::instance::FORCE_EAGER.with(|eager| eager.set(None));
        assert!(eager.all_in_prefill(), "{shape:?}");
        let (trimmed, whole) = check("eager mixed", &[eager]);
        assert!(trimmed > 0 && whole > 0, "{shape:?} eager mixed");
    }
}

/// [`mixed_steps`]' mixed step with every drafter run eager.
fn eager_mixed_step(shape: Shape, mode: Canonical) -> Drafted {
    let (host, mut instance) = instance(shape, mode);
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
    instance.commit_accepted(&kept(&[(10, 4)])).unwrap();
    assert_drafter_rows(
        &host,
        &BTreeMap::from([(1, truths(0..80)), (2, truths(0..130))]),
    );
    instance.destroy().unwrap();
    mixed
}
