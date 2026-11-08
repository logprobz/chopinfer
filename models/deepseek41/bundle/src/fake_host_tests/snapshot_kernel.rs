//! Prefix snapshots under the `prefix_snapshot_copy_kernel` candidate, on
//! by default: each rank moves its snapshot's pieces with one table upload
//! and one launch of the shared snapshot copy kernel, which the fake host
//! runs as the device copies its arguments and uploaded table name
//! (`FakeHost::run_snapshot_copy`). The pages and restored slots must hold
//! what the device copies leave there, byte origin for byte origin, and the
//! lifecycle keeps its rules: every rank that received work drains, an
//! unconfirmed drain is INTERNAL_ERROR, which poisons, a refused plan
//! submits nothing, and a failed or refused restore leaves the slot free.
//! The prefix tests of this file's parent rerun here with the option off
//! ([`with_copies`]).

use super::*;
use model_deepseek41_contract::execution_options::PREFIX_SNAPSHOT_COPY_KERNEL;

/// Runs `body` with the candidate off in every instance it creates, and
/// fails if any snapshot went through the kernel.
pub(crate) fn with_copies<T>(body: impl FnOnce() -> T) -> T {
    let before = SNAPSHOT_COPY_LAUNCHES.with(Cell::get);
    let result = crate::config::tests::with_record_options(
        &[(PREFIX_SNAPSHOT_COPY_KERNEL, false.into())],
        body,
    );
    assert_eq!(
        SNAPSHOT_COPY_LAUNCHES.with(Cell::get),
        before,
        "a snapshot went through the snapshot copy kernel"
    );
    result
}

#[test]
fn prefix_snapshots_round_trip_through_the_pages_with_device_copies() {
    with_copies(super::prefix_snapshots_round_trip_through_the_pages);
}

#[test]
fn a_state_slot_takes_a_new_request_once_its_holder_retires_with_device_copies() {
    with_copies(super::a_state_slot_takes_a_new_request_once_its_holder_retires);
}

#[test]
fn the_lifecycle_verifies_cancels_preempts_and_recycles_slots_with_device_copies() {
    with_copies(super::the_lifecycle_verifies_cancels_preempts_and_recycles_slots);
}

/// The rank of the stream a journal line names (its second word).
fn rank_of(host: &FakeHost, line: &str) -> Option<u32> {
    let stream: u32 = line.split_whitespace().nth(1)?.parse().ok()?;
    host.device.stream_ranks.borrow().get(&stream).copied()
}

/// The ranks of the journal lines from `from` on that start with `kind`
/// and contain `text`, in order.
fn journal_ranks(host: &FakeHost, from: usize, kind: &str, text: &str) -> Vec<u32> {
    host.device.journal.borrow()[from..]
        .iter()
        .filter(|line| line.starts_with(kind) && line.contains(text))
        .filter_map(|line| rank_of(host, line))
        .collect()
}

/// An instance of `ranks` and `kv` with request 7's 141-token prompt
/// committed in slot 0, and the pages its snapshot takes.
fn prefilled(ranks: Ranks, kv: KvLayout, kernel: bool) -> (Rc<Host>, BundleInstanceV1, Vec<u64>) {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let host = FakeHost::new(shape, kv);
    let options = [(PREFIX_SNAPSHOT_COPY_KERNEL, kernel.into())];
    let mut instance = create_with_options(&host, 4, &options);
    instance
        .execute(&prompt(7, 0, 0, 141, true), &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    let count = instance.describe_prefix_state().unwrap();
    let pages = (0..count).map(|page| 40 + page).collect();
    (host, instance, pages)
}

/// The origins of every byte of the snapshot's pages and of the state view
/// on each rank.
fn origins(host: &FakeHost, pages: &[u64]) -> Vec<Vec<(u64, Origin)>> {
    let memory = host.memory.borrow();
    let page_bytes = host.page_bytes();
    let mut origins = Vec::new();
    for rank in 0..host.shape.ranks() {
        let pool = BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank);
        for &page in pages {
            origins.push(memory.read(DeviceBufferSpanV1::new(pool, page * page_bytes, page_bytes)));
        }
        let state = BundleDeviceBufferHandleV1::from_parts(1, STATE_SLOT + rank);
        origins.push(memory.read(DeviceBufferSpanV1::new(state, 0, host.state_bytes.get())));
    }
    origins
}

/// A capture and a restore through the kernel leave every page of the
/// snapshot and every byte of the state view with the origins the device
/// copies give them, at TP2 and TP4 in both KV layouts; each rank issues
/// one table upload and one launch per snapshot and no device copy, and
/// its launches make the device copies, in their order.
#[test]
fn the_kernel_leaves_the_pages_and_slots_the_device_copies_leave() {
    for (ranks, kv) in PREFIX_RANKS
        .into_iter()
        .flat_map(|ranks| KV_LAYOUTS.map(|kv| (ranks, kv)))
    {
        let mut after = Vec::new();
        let mut moved = Vec::new();
        for kernel in [false, true] {
            let (host, mut instance, pages) = prefilled(ranks, kv, kernel);
            let start = host.device.journal.borrow().len();
            instance.capture_prefix_state(7, 141, &pages).unwrap();
            let captured = origins(&host, &pages);
            let launches = ranks_of_launches(&host, start);
            instance.retire_request(7).unwrap();
            let ids = (0..141).collect::<Vec<u32>>();
            instance
                .restore_prefix_state_in_slot(9, &ids, &pages, 2)
                .unwrap();
            assert_state_moved(&host, 4, 0, 0, 2);
            let copies = journal_ranks(&host, start, "copy ", "");
            let all = (0..ranks.get()).collect::<Vec<_>>();
            if kernel {
                // One launch a rank for the capture, one for the restore.
                assert_eq!(launches, all, "{ranks:?} {kv:?}");
                assert_eq!(ranks_of_launches(&host, start).len(), 2 * all.len());
                assert!(copies.is_empty(), "{ranks:?} {kv:?}: {copies:?}");
            } else {
                assert!(launches.is_empty());
                assert!(copies.len() > 2 * all.len());
            }
            let kind = if kernel { "kernel copy " } else { "copy " };
            moved.push(
                host.device.journal.borrow()[start..]
                    .iter()
                    .filter_map(|line| line.strip_prefix(kind).map(str::to_owned))
                    .collect::<Vec<_>>(),
            );
            after.push((captured, origins(&host, &pages)));
            instance.retire_request(9).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
        assert!(after[0] == after[1], "{ranks:?} {kv:?}");
        assert!(!moved[0].is_empty());
        assert_eq!(moved[0], moved[1], "{ranks:?} {kv:?}");
    }
}

/// The ranks of the snapshot copy launches from journal line `from` on.
fn ranks_of_launches(host: &FakeHost, from: usize) -> Vec<u32> {
    journal_ranks(host, from, "launch ", kernels_sm120::snapshot_copy::SYMBOL)
}

/// With the candidate off no instance loads the kernel's module; with it
/// on, each rank loads it once and resolves its entry.
#[test]
fn only_a_candidate_instance_loads_the_kernel() {
    for kernel in [false, true] {
        let (host, mut instance, _) = prefilled(Ranks::TP2, KV_LAYOUTS[0], kernel);
        let loaded = host
            .device
            .module_cubins
            .borrow()
            .values()
            .filter(|cubin| **cubin == crate::sm120::snapshot_copy::CUBIN)
            .count();
        assert_eq!(loaded, if kernel { 2 } else { 0 });
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// A launch that fails on rank 1 ends the capture with the host's error
/// once ranks 0 and 1, which received work, are drained; ranks 2 and 3
/// receive nothing. The instance stays usable and the next capture of the
/// same state succeeds; a restore whose launch fails leaves its slot free
/// for a restore that succeeds.
#[test]
fn a_failed_launch_drains_the_ranks_that_received_work() {
    let (host, mut instance, pages) = prefilled(Ranks::TP4, KV_LAYOUTS[0], true);
    let start = host.device.journal.borrow().len();
    host.device.fail_launch.set(Some(1));
    assert_eq!(
        instance.capture_prefix_state(7, 141, &pages),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert_eq!(ranks_of_launches(&host, start), [0]);
    assert_eq!(journal_ranks(&host, start, "synchronize ", ""), [0, 1]);
    let uploaded = journal_ranks(&host, start, "upload ", "")
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(uploaded, [0, 1].into());
    assert!(!instance.is_poisoned());
    instance.capture_prefix_state(7, 141, &pages).unwrap();
    instance.retire_request(7).unwrap();
    let ids = (0..141).collect::<Vec<u32>>();
    let start = host.device.journal.borrow().len();
    host.device.fail_launch.set(Some(2));
    assert!(
        instance
            .restore_prefix_state_in_slot(9, &ids, &pages, 1)
            .is_err()
    );
    assert_eq!(journal_ranks(&host, start, "synchronize ", ""), [0, 1, 2]);
    assert!(!instance.is_poisoned());
    instance
        .restore_prefix_state_in_slot(9, &ids, &pages, 1)
        .unwrap();
    assert_state_moved(&host, 4, 0, 0, 1);
    instance.retire_request(9).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A drain that cannot be confirmed makes the capture INTERNAL_ERROR once
/// every other rank drained, so the host keeps the pages and the client is
/// poisoned; a restore's does the same.
#[test]
fn an_unconfirmed_drain_is_an_internal_error_that_poisons() {
    for restore in [false, true] {
        let (host, mut instance, pages) = prefilled(Ranks::TP4, KV_LAYOUTS[0], true);
        if restore {
            instance.capture_prefix_state(7, 141, &pages).unwrap();
            instance.retire_request(7).unwrap();
        }
        let start = host.device.journal.borrow().len();
        host.fail_synchronize.set(Some(1));
        let result = if restore {
            let ids = (0..141).collect::<Vec<u32>>();
            instance.restore_prefix_state_in_slot(9, &ids, &pages, 1)
        } else {
            instance.capture_prefix_state(7, 141, &pages)
        };
        assert_eq!(
            result,
            Err(BundleAbiError::Status(BundleStatusV1::InternalError))
        );
        assert_eq!(ranks_of_launches(&host, start), [0, 1, 2, 3]);
        // Rank 1's drain failed; the others drained.
        assert_eq!(journal_ranks(&host, start, "synchronize ", ""), [0, 2, 3]);
        assert!(instance.is_poisoned());
    }
}

/// Pages the pool does not hold, or repeated, are refused before any rank
/// uploads or launches anything, as without the kernel.
#[test]
fn refused_pages_submit_nothing() {
    let (host, mut instance, pages) = prefilled(Ranks::TP2, KV_LAYOUTS[0], true);
    let past = host.kv_pages.get();
    let mut outside = pages.clone();
    *outside.last_mut().unwrap() = past;
    let mut repeated = pages.clone();
    repeated[1] = repeated[0];
    for bad in [outside, repeated, pages[1..].to_vec()] {
        let start = host.device.journal.borrow().len();
        assert_eq!(
            instance.capture_prefix_state(7, 141, &bad),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
        );
        assert!(
            host.device.journal.borrow()[start..].is_empty(),
            "{:?}",
            &host.device.journal.borrow()[start..]
        );
    }
    assert!(!instance.is_poisoned());
    instance.capture_prefix_state(7, 141, &pages).unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
}

/// The pinned Flash journals of each topology and KV layout with the
/// candidate off (`journal_steps_with`). No rank allocates the kernel's
/// table or resolves its symbol at acquire, or releases the table, the
/// symbol and the module at destroy: five lines fewer a rank. The
/// lifecycle captures no snapshot, so nothing else changes.
#[test]
fn the_flash_journals_with_device_copies_drop_each_ranks_kernel_and_table() {
    let mut mismatches = Vec::new();
    for (topology, storage, (default_lines, _)) in super::FLASH_JOURNALS {
        let shape = crate::shape::topologies()
            .into_iter()
            .find(|shape| shape.topology() == topology)
            .unwrap();
        let kv = KV_LAYOUTS
            .into_iter()
            .find(|kv| kv.storage == storage)
            .unwrap();
        let host = FakeHost::new(shape, kv);
        super::journal_steps_with(&host, &[(PREFIX_SNAPSHOT_COPY_KERNEL, false.into())]);
        assert_eq!(host.acquired.get(), host.released.get(), "{topology}");
        let lines = host.device.journal.borrow().clone();
        assert_eq!(
            lines.len() + 5 * shape.ranks() as usize,
            default_lines,
            "{topology} {storage:?}"
        );
        let layout = format!("{storage:?}").to_lowercase();
        let (_, _, expected) = COPIES_JOURNALS
            .iter()
            .find(|(id, kind, _)| *id == topology && *kind == storage)
            .unwrap();
        if let Err(observed) = bundle_runtime::fake_host::check_journal(
            &lines,
            *expected,
            &format!("{topology}.{layout}.copies.journal"),
        ) {
            mismatches.push(format!("{topology} {layout}: {observed}"));
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The pins of [`the_flash_journals_with_device_copies_drop_each_ranks_kernel_and_table`].
const COPIES_JOURNALS: [(&str, KvStorage, (usize, u64)); 6] = [
    (
        "deepseek41-flash-tp1",
        KvStorage::Native,
        (15525, 0x7f80da60657b8f51),
    ),
    (
        "deepseek41-flash-tp1",
        KvStorage::Bf16,
        (15379, 0xe419bcbd959c831e),
    ),
    (
        "deepseek41-flash-tp2",
        KvStorage::Native,
        (30975, 0x71eb2d4066b5928e),
    ),
    (
        "deepseek41-flash-tp2",
        KvStorage::Bf16,
        (30683, 0x237c390684736b5e),
    ),
    (
        "deepseek41-flash-tp4",
        KvStorage::Native,
        (76801, 0xc9be576878c4bfc2),
    ),
    (
        "deepseek41-flash-tp4",
        KvStorage::Bf16,
        (75181, 0xf66925c8a454ce45),
    ),
];
