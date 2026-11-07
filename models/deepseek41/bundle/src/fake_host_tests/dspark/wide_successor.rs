//! Successors of 9 to 16 drafting items (`dspark_successor_items`). With
//! the bound at 16, a verify step's device result queues the private
//! projection of every item's accepted rows, in launches of at most 32
//! rows, and its draft, in one forward through twelve items and two above,
//! and the next step reuses both: its target work and boundary readback are
//! the inline path's. The default bound of eight queues none above eight
//! items, and steps of at most eight items are unchanged at any bound.
use super::results::{DeviceSink, batch, metadata};
use super::*;
use model_deepseek41_contract::execution_options::{
    DSPARK_GRAPH_HANDLES_PER_RANK, DSPARK_SUCCESSOR_ITEMS,
};
use runtime_bundle_abi::device_results::*;

/// The drafter forwards one draft of `items` items takes at the default
/// chunk of twelve.
fn chunks(items: u32) -> u32 {
    items.div_ceil(12)
}

/// The boundary readback of a step of `items` verify items of `k` drafts
/// that caught up `projected` rows: each rank's saved status (both planes
/// of every projected and draft row, and every target row's quantizer and
/// index statuses), the drafts and confidence, then the acceptance packet,
/// every item accepting `accepted` drafts with bonus 7.
fn boundary(items: u32, k: u32, projected: u32, accepted: u32, epoch: u64) -> Vec<u8> {
    let (items, rows, projected) = (items as usize, (k + 1) as usize, projected as usize);
    let status: usize = (0..4)
        .map(|rank| {
            let indexes = if rank == 0 {
                FLASH.shape.csa.index_sources.len()
            } else {
                0
            };
            4 * (2 * (projected + items * 5) + items * rows * (indexes + 2))
        })
        .sum();
    let mut bytes = vec![0; status + items * 40];
    let offset = bytes.len();
    let samples = crate::dspark::results::SAMPLE_OFFSET as usize;
    bytes.resize(offset + samples + items * rows * 256, 0);
    for index in 0..items {
        let record = BundleAcceptanceResultV1 {
            cycle_epoch: epoch,
            accepted_drafts: accepted,
            emitted_rows: accepted + 1,
            flags: if accepted < k {
                BUNDLE_ACCEPTANCE_REJECTED_V1
            } else {
                0
            },
            status: 0,
            failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
            verified_drafts: 0,
        };
        bytes[offset + index * 32..offset + (index + 1) * 32]
            .copy_from_slice(&record.to_le_bytes());
        let sample = offset + samples + (index * rows + accepted as usize) * 256;
        bytes[sample..sample + 4].copy_from_slice(&7_u32.to_le_bytes());
    }
    bytes
}

/// An instance of `items` slots serving `verify` rows with the successor
/// bound `bound` and `options`, each item's three-token prompt committed.
fn prompted(
    items: u32,
    verify: u32,
    bound: u32,
    options: &[(&str, serde_json::Value)],
) -> (Rc<Host>, BundleInstanceV1) {
    let mut options = options.to_vec();
    options.push((DSPARK_SUCCESSOR_ITEMS, bound.into()));
    let (host, mut instance) = create_with_options(
        Shape::new(&FLASH, Ranks::TP4).unwrap(),
        items,
        verify,
        &options,
    );
    let prompt = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (1..=items)
            .map(|request| BatchItemV1::new(u64::from(request), false, 3, true, vec![7; 3]))
            .collect(),
        (0..items)
            .map(|slot| BatchAppendV1::new(0, 3, 3, vec![u64::from(slot)], None))
            .collect(),
    )
    .with_state_slots((0..items).map(u64::from).collect())
    .encode()
    .unwrap();
    execute(&mut instance, &host, &prompt);
    instance.commit().unwrap();
    (host, instance)
}

/// One device-result cycle: every item verifies `k` drafts after `start`
/// committed tokens, `projected` of its rows caught up, each accepting
/// `accepted`. Returns the step's journal, or the failed execute.
fn cycle(
    host: &Rc<Host>,
    instance: &mut BundleInstanceV1,
    (items, k, start): (u32, u32, u64),
    (projected, accepted): (u32, u32),
    epoch: u64,
) -> Result<Vec<String>, BundleAbiError> {
    host.injected
        .replace(Some(boundary(items, k, projected, accepted, epoch)));
    let begin = host.journal.borrow().len();
    let mut sink = DeviceSink {
        host: Rc::clone(host),
        metadata: Some(metadata(items, k, epoch)),
        legacy: Sink::default(),
        drafts: Vec::new(),
        samples: Vec::new(),
        completion: None,
    };
    instance.execute(&batch(items, start, k), &mut sink)?;
    assert!(
        host.injected.borrow().is_none(),
        "the boundary packet's length is the step's"
    );
    assert!(sink.completion.is_some());
    Ok(host.journal.borrow()[begin..].to_vec())
}

fn position(step: &[String], prefix: &str) -> usize {
    step.iter()
        .position(|line| line.starts_with(prefix))
        .unwrap_or_else(|| panic!("no {prefix:?} line"))
}

/// The rank-0 hash launch: every drafter and catch-up launch of a step
/// precedes it.
fn hash(host: &Host, step: &[String]) -> usize {
    let stream = host.rank_streams.borrow()[&0];
    position(
        step,
        &format!(
            "launch {stream} {} ",
            crate::sm120::engram_hash_rows::SYMBOL
        ),
    )
}

fn count(lines: &[String], prefix: &str) -> usize {
    lines.iter().filter(|line| line.starts_with(prefix)).count()
}

/// Commits every item's `kept` rows, then retires every item and destroys
/// the instance.
fn finish(host: &Host, mut instance: BundleInstanceV1, items: u32, kept_rows: u32) {
    instance
        .commit_accepted(
            &(1..=items)
                .map(|id| kept(u64::from(id), kept_rows))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for request in 1..=items {
        instance.retire_request(u64::from(request)).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// At the bound of 16, a step of 9, 12, 13 or 16 items queues its
/// successor after the boundary readback, with no other readback: one
/// projection launch a 32 rows of the accepted rows and the draft's one or
/// two forwards a rank. The next step launches neither before its hash,
/// and from the hash through its boundary readback it issues the inline
/// path's journal. At the default bound neither step queues one.
#[test]
fn wide_device_feedback_queues_a_successor_the_next_step_reuses() {
    for items in [9, 12, 13, 16] {
        let mut targets = Vec::new();
        for bound in [8, 16] {
            let (host, mut instance) = prompted(items, 64, bound, &[]);
            let mut steps = Vec::new();
            for (epoch, start, projected) in [(1, 3, items * 3), (2, 7, items * 4)] {
                let step = cycle(
                    &host,
                    &mut instance,
                    (items, 3, start),
                    (projected, 3),
                    epoch,
                )
                .unwrap();
                instance
                    .commit_accepted(
                        &(1..=items)
                            .map(|id| kept(u64::from(id), 4))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                steps.push(step);
            }
            for (index, step) in steps.iter().enumerate() {
                let download = position(step, "download ");
                assert_eq!(count(step, "download "), 1, "items={items} bound={bound}");
                let queued = if bound == 16 {
                    4 * ((items * 4).div_ceil(32) + chunks(items)) as usize
                } else {
                    0
                };
                assert_eq!(
                    count(&step[download..], "replay "),
                    queued,
                    "items={items} bound={bound} step={index}: the successor's projection and draft"
                );
                let hash = hash(&host, step);
                let inline = count(&step[..hash], "replay ");
                if index == 1 && bound == 16 {
                    assert_eq!(inline, 0, "items={items}: the successor is reused");
                } else {
                    assert!(inline > 0, "items={items} bound={bound} step={index}");
                }
            }
            assert_eq!(
                crate::instance::dspark_successor_for_test(instance.handle()),
                (bound == 16).then_some(items as usize)
            );
            let step = &steps[1];
            targets.push(step[hash(&host, step)..=position(step, "download ")].to_vec());
            for request in 1..=items {
                instance.retire_request(u64::from(request)).unwrap();
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
        assert_eq!(targets[0], targets[1], "items={items}");
    }
}

/// At most eight items queue the same successors at every bound: the
/// whole journal of two cycles at C8, depth three, and C6, depth five,
/// whose six-row items pass one launch and plan none, is the default's.
#[test]
fn at_most_eight_items_keep_the_default_journal_at_every_bound() {
    for (items, k, accepted) in [(8, 3, 3), (6, 5, 5), (6, 5, 4)] {
        let mut journals = Vec::new();
        for bound in [8, 9, 16] {
            let (host, mut instance) = prompted(items, 64, bound, &[]);
            let rows = accepted + 1;
            for (epoch, start, projected) in
                [(1, 3, items * 3), (2, 3 + u64::from(rows), items * rows)]
            {
                cycle(
                    &host,
                    &mut instance,
                    (items, k, start),
                    (projected, accepted),
                    epoch,
                )
                .unwrap();
                instance
                    .commit_accepted(
                        &(1..=items)
                            .map(|id| kept(u64::from(id), rows))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
            for request in 1..=items {
                instance.retire_request(u64::from(request)).unwrap();
            }
            instance.destroy().unwrap();
            let journal = host.journal.borrow().clone();
            journals.push(
                journal
                    .into_iter()
                    .filter(|line| !line.starts_with("allocate "))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(journals[0], journals[1], "items={items} k={k}");
        assert_eq!(journals[0], journals[2], "items={items} k={k}");
    }
}

/// Above eight items both paths compute five heads, so a wide successor
/// serves the next step at a shallower or a deeper draft budget: depth
/// five to three and three to five each reuse it, in one or two forwards.
#[test]
fn wide_successors_serve_depth_changes_in_both_directions() {
    for items in [9, 13] {
        for (first, next) in [(5, 3), (3, 5)] {
            let (host, mut instance) = prompted(items, 96, 16, &[]);
            cycle(
                &host,
                &mut instance,
                (items, first, 3),
                (items * 3, first),
                1,
            )
            .unwrap();
            let rows = first + 1;
            instance
                .commit_accepted(
                    &(1..=items)
                        .map(|id| kept(u64::from(id), rows))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let step = cycle(
                &host,
                &mut instance,
                (items, next, 3 + u64::from(rows)),
                (items * rows, 0),
                2,
            )
            .unwrap();
            assert_eq!(
                count(&step[..hash(&host, &step)], "replay "),
                0,
                "items={items} {first} to {next}: the successor is reused"
            );
            finish(&host, instance, items, 1);
        }
    }
}

/// Under graph pressure a wide successor launches eagerly and is still
/// reused: with no graph handles every projection and draft launches
/// eagerly, and with room for a few graphs the private draft graphs share
/// the budget, evicting nothing the ledger keeps.
#[test]
fn graph_pressure_keeps_wide_successors_eager_and_reused() {
    let quantize = "chop_deepseek41_dspark_quantize_w15360_bf16_parallel_v1";
    for handles in [0_u64, 6] {
        let items = 16;
        let (host, mut instance) = prompted(
            items,
            64,
            16,
            &[(DSPARK_GRAPH_HANDLES_PER_RANK, handles.into())],
        );
        let first = cycle(&host, &mut instance, (items, 3, 3), (items * 3, 3), 1).unwrap();
        let download = position(&first, "download ");
        let projections = first[download..]
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(quantize))
            .count();
        if handles == 0 {
            assert_eq!(count(&first, "replay "), 0);
            assert_eq!(count(&first, "capture "), 0);
            assert_eq!(projections, 4 * 2, "two eager 32-row projections a rank");
        }
        instance
            .commit_accepted(
                &(1..=items)
                    .map(|id| kept(u64::from(id), 4))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let step = cycle(&host, &mut instance, (items, 3, 7), (items * 4, 0), 2).unwrap();
        let hash = hash(&host, &step);
        assert_eq!(count(&step[..hash], "replay "), 0, "handles={handles}");
        for symbol in [quantize, crate::sm120::confidence_parallel::SYMBOL] {
            assert!(
                !step[..hash]
                    .iter()
                    .any(|line| line.starts_with("launch ") && line.contains(symbol)),
                "handles={handles}: no inline catch-up or draft"
            );
        }
        finish(&host, instance, items, 1);
    }
}

/// A queue phase that fails poisons the drafter with no successor: the
/// first ring backup copy, the first projection launch, the second draft
/// forward and the join's first event record each fail the step after the
/// target published, and the instance refuses every later step and cancel.
#[test]
fn a_failed_wide_successor_phase_poisons_with_no_adoption() {
    let items = 16;
    let reference = {
        let (host, mut instance) = prompted(items, 64, 16, &[]);
        let rings = crate::instance::dspark_successor_rings_for_test(instance.handle());
        let step = cycle(&host, &mut instance, (items, 3, 3), (items * 3, 3), 1).unwrap();
        finish(&host, instance, items, 4);
        (step, rings)
    };
    let (step, rings) = reference;
    let download = position(&step, "download ");
    let backup = format!(" {:?} {:?}", rings[0].0, rings[0].1);
    let backup = step
        .iter()
        .position(|line| line.starts_with("copy ") && line.ends_with(&backup))
        .expect("rank 0's ring backup");
    assert!(backup > download);
    let replays: Vec<_> = (download..step.len())
        .filter(|&at| step[at].starts_with("replay "))
        .collect();
    // Per rank: two 32-row projections, then two eight-item forwards.
    assert_eq!(replays.len(), 4 * 4);
    let second_forward = replays[4 * 3];
    let join = (replays[replays.len() - 1]..step.len())
        .find(|&at| step[at].starts_with("record "))
        .expect("the join records");
    for (phase, at, replayed) in [
        ("ring backup", backup, false),
        ("projection", replays[0], true),
        ("second forward", second_forward, true),
        ("join", join, false),
    ] {
        let (host, mut instance) = prompted(items, 64, 16, &[]);
        let prefix = &step[..at];
        match phase {
            "ring backup" => host
                .device
                .fail_copy
                .set(Some(count(prefix, "copy ") as u32)),
            "join" => host
                .device
                .fail_record
                .set(Some(count(prefix, "record ") as u32)),
            _ => host
                .device
                .fail_graph_launch
                .set(Some(count(prefix, "replay ") as u32)),
        }
        let failed = cycle(&host, &mut instance, (items, 3, 3), (items * 3, 3), 1);
        assert_eq!(
            failed.err(),
            Some(BundleAbiError::Status(BundleStatusV1::BundleError)),
            "{phase}"
        );
        assert_eq!(
            crate::instance::dspark_successor_for_test(instance.handle()),
            None,
            "{phase}"
        );
        assert!(instance.cancel().is_err(), "{phase}");
        let begin = host.journal.borrow().len();
        let mut sink = Sink::default();
        assert!(
            instance.execute(&batch(items, 7, 3), &mut sink).is_err(),
            "{phase}"
        );
        assert!(
            host.journal.borrow()[begin..]
                .iter()
                .all(|line| !line.starts_with("launch ") && !line.starts_with("replay ")),
            "{phase}"
        );
        // A failed graph replay of the collectives poisons the instance
        // itself, which then refuses destroy and keeps every handle; after a
        // failed copy or record the drafter alone is poisoned.
        let released = host.released.get();
        if replayed {
            assert!(instance.destroy().is_err(), "{phase}");
            assert!(instance.is_poisoned(), "{phase}");
            assert_eq!(host.released.get(), released, "{phase}");
        } else {
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get(), "{phase}");
        }
    }
}
