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

/// The serving pages: one a slot for the verify items, then a prompt's and
/// a snapshot's.
const KV_PAGES: u64 = 1024;
const PROMPT_PAGE: u64 = 64;
const SNAPSHOT_PAGE: u64 = 512;

/// An instance of `items` slots serving `verify` rows with the successor
/// bound `bound` and `options`, each item's three-token prompt committed.
pub(super) fn prompted(
    items: u32,
    verify: u32,
    bound: u32,
    options: &[(&str, serde_json::Value)],
) -> (Rc<Host>, BundleInstanceV1) {
    prompted_in(items, items, verify, bound, options)
}

/// [`prompted`] on an instance of `slots` slots whose pool holds
/// [`KV_PAGES`] pages.
pub(super) fn prompted_in(
    items: u32,
    slots: u32,
    verify: u32,
    bound: u32,
    options: &[(&str, serde_json::Value)],
) -> (Rc<Host>, BundleInstanceV1) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, verify)));
    host.kv_pages.set(KV_PAGES);
    let mut record = crate::config::tests::record(&shape, host.kv, slots, false);
    for (name, value) in options
        .iter()
        .cloned()
        .chain([(DSPARK_SUCCESSOR_ITEMS, bound.into())])
    {
        record = crate::config::tests::with_option(record, name, value);
    }
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
            &record,
        )
        .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
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
pub(super) fn cycle(
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

/// `dspark_successor_prefill` changes nothing a decode step after an
/// unallocated one runs: at C8, depth three, the compact successor is the
/// step's own three heads, and above eight items both compute five, so
/// with the bound at 16 every journal of two cycles is the same with the
/// option on or off.
#[test]
fn the_prefill_option_keeps_the_decode_journals() {
    use model_deepseek41_contract::execution_options::DSPARK_SUCCESSOR_PREFILL;
    for items in [8, 13] {
        let mut journals = Vec::new();
        for prefill in [false, true] {
            let (host, mut instance) =
                prompted(items, 64, 16, &[(DSPARK_SUCCESSOR_PREFILL, prefill.into())]);
            for (epoch, start, projected) in [(1, 3, items * 3), (2, 7, items * 4)] {
                cycle(
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
            }
            for request in 1..=items {
                instance.retire_request(u64::from(request)).unwrap();
            }
            instance.destroy().unwrap();
            journals.push(host.journal.borrow().clone());
        }
        assert_eq!(journals[0], journals[1], "items={items}");
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

/// A C16 successor at depth five whose items accept every draft fills its
/// status to the bound: 96 projected rows in three 32-row launches and 80
/// draft rows in two eight-item forwards a rank, both planes of each, 1,408
/// bytes. The next step reuses it and reads that status back ahead of its
/// own target rows. A nonzero word in it, in the first or a later launch
/// on rank 0 or a later rank, fails that step before the host reads any of
/// its results, naming the rank and row: its rows commit nothing and no
/// successor follows it.
#[test]
fn a_full_c16_successor_status_is_checked_in_every_launch_and_rank() {
    let items = 16;
    // Each rank's successor status parts, in rows: three projections, then
    // two forwards, each saving its row and shared planes.
    let parts = [32_usize, 32, 32, 40, 40];
    assert_eq!(
        2 * 4 * parts.iter().sum::<usize>(),
        crate::dspark::prefetch::status_rows(16) as usize * 8
    );
    // Each rank's boundary status: the successor's, then the target rows'.
    let rank_bytes = |rank: usize| {
        let indexes = if rank == 0 {
            FLASH.shape.csa.index_sources.len()
        } else {
            0
        };
        4 * (2 * (96 + 80) + 16 * 6 * (indexes + 2))
    };
    // (rank, part, plane, row) of the nonzero word.
    for failure in [
        None,
        Some((0, 0, 0, 0)),
        Some((0, 2, 1, 31)),
        Some((2, 1, 0, 7)),
        Some((3, 3, 0, 0)),
        Some((3, 4, 1, 39)),
    ] {
        let (host, mut instance) = prompted(items, 96, 16, &[]);
        let first = cycle(&host, &mut instance, (items, 5, 3), (items * 3, 5), 1).unwrap();
        let download = position(&first, "download ");
        assert_eq!(
            count(&first[download..], "replay "),
            4 * (3 + 2),
            "three projections and two forwards a rank"
        );
        instance
            .commit_accepted(
                &(1..=items)
                    .map(|id| kept(u64::from(id), 6))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert_eq!(
            crate::instance::dspark_successor_for_test(instance.handle()),
            Some(16)
        );
        let mut packet = boundary(items, 5, 96, 0, 2);
        if let Some((rank, part, plane, row)) = failure {
            let at = (0..rank).map(rank_bytes).sum::<usize>()
                + 4 * (2 * parts[..part].iter().sum::<usize>() + plane * parts[part] + row);
            packet[at] = 1;
        }
        host.injected.replace(Some(packet));
        let logs = host.logs.borrow().len();
        let begin = host.journal.borrow().len();
        let mut sink = DeviceSink {
            host: Rc::clone(&host),
            metadata: Some(metadata(items, 5, 2)),
            legacy: Sink::default(),
            drafts: Vec::new(),
            samples: Vec::new(),
            completion: None,
        };
        let result = instance.execute(&batch(items, 9, 5), &mut sink);
        let step = host.journal.borrow()[begin..].to_vec();
        assert!(
            host.injected.borrow().is_none(),
            "{failure:?}: the boundary's length holds the successor's status"
        );
        assert_eq!(
            count(&step[..hash(&host, &step)], "replay "),
            0,
            "{failure:?}: the successor is reused"
        );
        let Some((rank, _, _, row)) = failure else {
            result.unwrap();
            assert!(sink.completion.is_some());
            finish(&host, instance, items, 1);
            continue;
        };
        assert_eq!(
            result.err(),
            Some(BundleAbiError::Status(BundleStatusV1::BundleError)),
            "{failure:?}"
        );
        let message = format!("DSpark quantized input on rank {rank} is invalid in row {row}");
        assert!(
            host.logs.borrow()[logs..]
                .iter()
                .any(|line| line.contains(&message)),
            "{failure:?}"
        );
        assert_eq!(
            crate::instance::dspark_successor_for_test(instance.handle()),
            None,
            "{failure:?}"
        );
        assert!(
            instance
                .commit_accepted(
                    &(1..=items)
                        .map(|id| kept(u64::from(id), 1))
                        .collect::<Vec<_>>(),
                )
                .is_err(),
            "{failure:?}"
        );
        for request in 1..=u64::from(items) {
            assert_eq!(
                crate::instance::dspark_context(instance.handle(), request).map(|c| c.end),
                Some(9),
                "{failure:?}"
            );
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get(), "{failure:?}");
    }
}

/// Which host call a queue failure fails.
#[derive(Clone, Copy, Debug)]
enum Fault {
    Copy,
    Record,
    Wait,
    Replay,
}

/// A queue phase that fails poisons the drafter with no successor. Each
/// rank's ring backup copy, a projection replay of the first launch on
/// rank 0 or of the second on rank 3, the status save of a projection or a
/// forward, the second draft forward's replay, its chunk retention and the
/// chunk restore copies, the join's first event record and its first and
/// last waits, and the next and undo ring copies after the join each fail
/// the step after the target published, and the instance refuses every
/// later step and cancel.
#[test]
fn a_failed_wide_successor_phase_poisons_with_no_adoption() {
    let items = 16;
    let (step, rings, status) = {
        let (host, mut instance) = prompted(items, 64, 16, &[]);
        let rings = crate::instance::dspark_successor_rings_for_test(instance.handle());
        let status = crate::instance::dspark_successor_status_for_test(instance.handle());
        let step = cycle(&host, &mut instance, (items, 3, 3), (items * 3, 3), 1).unwrap();
        finish(&host, instance, items, 4);
        (step, rings, status)
    };
    let download = position(&step, "download ");
    let copy = |source: DeviceBufferSpanV1, destination: DeviceBufferSpanV1| {
        let suffix = format!(" {source:?} {destination:?}");
        (download..step.len())
            .find(|&at| step[at].starts_with("copy ") && step[at].ends_with(&suffix))
            .unwrap_or_else(|| panic!("no copy {suffix}"))
    };
    let into = |destination: DeviceBufferSpanV1| {
        let suffix = format!(" {destination:?}");
        (download..step.len())
            .find(|&at| step[at].starts_with("copy ") && step[at].ends_with(&suffix))
            .unwrap_or_else(|| panic!("no copy into {destination:?}"))
    };
    let within = |span: DeviceBufferSpanV1, offset: u64, length: u64| {
        DeviceBufferSpanV1::new(span.buffer, span.offset + offset, length)
    };
    let replays: Vec<_> = (download..step.len())
        .filter(|&at| step[at].starts_with("replay "))
        .collect();
    // Per rank: two 32-row projections, then two eight-item forwards.
    assert_eq!(replays.len(), 4 * 4);
    let join = (replays[replays.len() - 1]..step.len())
        .find(|&at| step[at].starts_with("record "))
        .expect("the join records");
    let waits: Vec<_> = (join..step.len())
        .filter(|&at| step[at].starts_with("wait "))
        .take(6)
        .collect();
    // Four rank records and three waits for them, then the joined record
    // and three waits for it.
    assert_eq!(count(&step[join..waits[5]], "record "), 5);
    // Each rank's saved status: both planes of the two 32-row projections
    // (128 bytes each), then of the two forwards (160 bytes each). The
    // chunk output holds every item's draft IDs, then their confidence.
    let (_, status, chunk) = status[3];
    let plane = u64::from(model_deepseek41_contract::serving::DECODE_BATCH) * 5 * 4;
    for (phase, at, fault) in [
        (
            "first ring backup",
            copy(rings[0].0, rings[0].1),
            Fault::Copy,
        ),
        (
            "last ring backup",
            copy(rings[3].0, rings[3].1),
            Fault::Copy,
        ),
        ("first projection", replays[0], Fault::Replay),
        ("second projection on rank 3", replays[4 + 3], Fault::Replay),
        (
            "projection status",
            into(within(status, 384, 128)),
            Fault::Copy,
        ),
        ("second forward", replays[4 * 3], Fault::Replay),
        (
            "forward status",
            into(within(status, 992, 160)),
            Fault::Copy,
        ),
        (
            "chunk retention",
            into(within(chunk, plane + 160, 160)),
            Fault::Copy,
        ),
        (
            "chunk restore",
            (download..step.len())
                .find(|&at| {
                    step[at].starts_with("copy ")
                        && step[at].contains(&format!(" {:?} ", within(chunk, plane, 320)))
                })
                .expect("rank 3's confidence restore"),
            Fault::Copy,
        ),
        ("join record", join, Fault::Record),
        ("first join wait", waits[0], Fault::Wait),
        ("last join wait", waits[5], Fault::Wait),
        ("next ring copy", copy(rings[0].0, rings[0].2), Fault::Copy),
        ("ring undo copy", copy(rings[3].1, rings[3].0), Fault::Copy),
    ] {
        let (host, mut instance) = prompted(items, 64, 16, &[]);
        let prefix = &step[..at];
        let (switch, lines) = match fault {
            Fault::Copy => (&host.device.fail_copy, "copy "),
            Fault::Record => (&host.device.fail_record, "record "),
            Fault::Wait => (&host.device.fail_wait, "wait "),
            Fault::Replay => (&host.device.fail_graph_launch, "replay "),
        };
        switch.set(Some(count(prefix, lines) as u32));
        let begin = host.journal.borrow().len();
        let failed = cycle(&host, &mut instance, (items, 3, 3), (items * 3, 3), 1);
        assert_eq!(
            failed.err(),
            Some(BundleAbiError::Status(BundleStatusV1::BundleError)),
            "{phase}"
        );
        let journal = host.journal.borrow()[begin..].to_vec();
        assert_eq!(journal[..at], step[..at], "{phase}: the calls before it");
        assert_ne!(journal.get(at), Some(&step[at]), "{phase}: the failed call");
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
        // failed copy, record or wait the drafter alone is poisoned.
        let released = host.released.get();
        if matches!(fault, Fault::Replay) {
            assert!(instance.destroy().is_err(), "{phase}");
            assert!(instance.is_poisoned(), "{phase}");
            assert_eq!(host.released.get(), released, "{phase}");
        } else {
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get(), "{phase}");
        }
    }
}

/// The successor records logged since `from`.
fn events(host: &Host, from: usize) -> Vec<serde_json::Value> {
    host.logs.borrow()[from..]
        .iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|record| record["phase"] == "dspark-prefetch")
        .collect()
}

/// A step of the 16 verify items of [`batch`], each `k` drafts from
/// `start`, after the last `rows` rows of `request`'s prompt in `slot`,
/// which `prefix` tokens restored.
fn c16_mixed(request: u64, slot: u64, prefix: u64, rows: u64, start: u64, k: u32) -> Vec<u8> {
    let end = prefix + rows;
    let mut items = vec![BatchItemV1::new(
        request,
        false,
        rows as u32,
        true,
        (prefix..end).map(|at| at as u32).collect(),
    )];
    let mut appends = vec![BatchAppendV1::new(
        prefix,
        rows,
        end,
        (0..end.div_ceil(64))
            .map(|page| PROMPT_PAGE + page)
            .collect(),
        None,
    )];
    let mut slots = vec![slot];
    for slot in 0..16 {
        items.push(BatchItemV1::verify(slot + 1, k + 1, true, vec![7]));
        appends.push(BatchAppendV1::new(
            start,
            u64::from(k + 1),
            start + u64::from(k + 1),
            vec![slot],
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

/// A wide successor follows the request lifecycle. A cancel of the step
/// that queued it retires every item, and the next step discards it; an
/// item that retires before the next step discards it; and so does a
/// prefix restore, unless `dspark_successor_prefill` lets the restored
/// prompt join the 16 decoders, a C16 mixed step, which then reuses the
/// five-head successor as it is and, from its draft readback, issues the
/// inline step's journal.
#[test]
fn wide_successors_follow_cancel_retirement_and_restore() {
    use model_deepseek41_contract::execution_options::{
        DIAGNOSTIC_STEP_LOGS, DSPARK_SUCCESSOR_PREFILL,
    };
    let items = 16;
    let logs = [(DIAGNOSTIC_STEP_LOGS, serde_json::Value::from(true))];

    let (host, mut instance) = prompted(items, 64, 16, &logs);
    cycle(&host, &mut instance, (items, 3, 3), (items * 3, 3), 1).unwrap();
    assert_eq!(
        crate::instance::dspark_successor_for_test(instance.handle()),
        Some(16)
    );
    instance.cancel().unwrap();
    for request in 1..=items {
        instance.retire_request(u64::from(request)).unwrap();
    }
    let from = host.logs.borrow().len();
    execute(
        &mut instance,
        &host,
        &super::super::prompt(100, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    let records = events(&host, from);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["event"], "discarded");
    assert_eq!(
        crate::instance::dspark_successor_for_test(instance.handle()),
        None
    );
    instance.retire_request(100).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());

    let (host, mut instance) = prompted(items, 64, 16, &logs);
    cycle(&host, &mut instance, (items, 3, 3), (items * 3, 3), 1).unwrap();
    instance
        .commit_accepted(
            &(1..=items)
                .map(|id| kept(u64::from(id), 4))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    instance.retire_request(u64::from(items)).unwrap();
    let from = host.logs.borrow().len();
    let step = cycle(
        &host,
        &mut instance,
        (items - 1, 3, 7),
        ((items - 1) * 4, 3),
        2,
    )
    .unwrap();
    assert_eq!(events(&host, from)[0]["event"], "discarded");
    assert!(count(&step[..hash(&host, &step)], "replay ") > 0);
    finish(&host, instance, items - 1, 4);

    let ids: Vec<u32> = (0..141).collect();
    let mut targets = Vec::new();
    let mut drafts = Vec::new();
    for prefill in [false, true] {
        let options = [logs[0].clone(), (DSPARK_SUCCESSOR_PREFILL, prefill.into())];
        let (host, mut instance) = prompted_in(items, 20, 64, 16, &options);
        execute(
            &mut instance,
            &host,
            &super::super::prompt(200, 16, 0, 141, true),
        );
        instance.commit().unwrap();
        let count_pages = instance.describe_prefix_state().unwrap();
        assert!(SNAPSHOT_PAGE + count_pages <= KV_PAGES);
        let pages: Vec<_> = (0..count_pages).map(|page| SNAPSHOT_PAGE + page).collect();
        instance.capture_prefix_state(200, 141, &pages).unwrap();
        instance.retire_request(200).unwrap();
        cycle(&host, &mut instance, (items, 3, 3), (items * 3, 3), 1).unwrap();
        instance
            .commit_accepted(
                &(1..=items)
                    .map(|id| kept(u64::from(id), 4))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        instance
            .restore_prefix_state_in_slot(201, &ids, &pages, 17)
            .unwrap();
        let from = host.logs.borrow().len();
        let begin = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &c16_mixed(201, 17, 141, 20, 7, 3));
        let step = host.journal.borrow()[begin..].to_vec();
        let records = events(&host, from);
        assert_eq!(records.len(), 1, "prefill={prefill}");
        if prefill {
            assert_eq!(records[0]["event"], "reused");
            assert_eq!(records[0]["prefill_items"], 1);
            assert_eq!(records[0]["heads"], 5);
        } else {
            assert_eq!(records[0]["event"], "discarded");
            assert_eq!(records[0]["relaxed"], true);
        }
        let (outputs, heads) = crate::instance::dspark_draft_outputs_for_test(instance.handle());
        assert_eq!(heads, 5);
        let ids_span = outputs[0].0;
        let readback = format!(
            "download {} {:?}",
            host.rank_streams.borrow()[&0],
            DeviceBufferSpanV1::new(ids_span.buffer, ids_span.offset, u64::from(items) * 20)
        );
        let at = step
            .iter()
            .position(|line| *line == readback)
            .expect("the draft ID readback");
        targets.push((count(&step[..at], "replay "), step[at..].to_vec()));
        drafts.push(sink.draft_tokens());
        instance
            .commit_accepted(
                &(1..=items)
                    .map(|id| kept(u64::from(id), 4))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for request in (1..=u64::from(items)).chain([201]) {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
    // Per rank: the decoders' two 32-row catch-up launches and two forwards.
    assert_eq!(targets[0].0 - targets[1].0, 4 * 4);
    assert_eq!(targets[0].1, targets[1].1);
    assert_eq!(drafts[0], drafts[1]);
}
