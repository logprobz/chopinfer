//! Prefix snapshots through the shared snapshot copy kernel: each rank
//! moves its snapshot's pieces with one table upload and one launch, which
//! the fake host runs as the device copies its arguments and uploaded table
//! name (`FakeHost::run_snapshot_copy`). The lifecycle keeps its rules:
//! every rank that received work drains, an unconfirmed drain is
//! INTERNAL_ERROR, which poisons, a refused plan submits nothing, and a
//! failed or refused restore leaves the slot free.

use super::*;

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
fn prefilled(ranks: Ranks, kv: KvLayout) -> (Rc<Host>, BundleInstanceV1, Vec<u64>) {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let host = FakeHost::new(shape, kv);
    let mut instance = create(&host, 4);
    instance
        .execute(&prompt(7, 0, 0, 141, true), &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    let count = instance.describe_prefix_state().unwrap();
    let pages = (0..count).map(|page| 40 + page).collect();
    (host, instance, pages)
}

/// A capture and a restore through the kernel at TP2 and TP4 in both KV
/// layouts: each rank issues one launch per snapshot and no device copy,
/// and every captured piece of the restored slot holds the bytes of the
/// captured slot's (`assert_state_moved`).
#[test]
fn each_rank_moves_a_snapshot_in_one_launch() {
    for (ranks, kv) in PREFIX_RANKS
        .into_iter()
        .flat_map(|ranks| KV_LAYOUTS.map(|kv| (ranks, kv)))
    {
        let (host, mut instance, pages) = prefilled(ranks, kv);
        let start = host.device.journal.borrow().len();
        instance.capture_prefix_state(7, 141, &pages).unwrap();
        let all = (0..ranks.get()).collect::<Vec<_>>();
        assert_eq!(ranks_of_launches(&host, start), all, "{ranks:?} {kv:?}");
        instance.retire_request(7).unwrap();
        let ids = (0..141).collect::<Vec<u32>>();
        instance
            .restore_prefix_state_in_slot(9, &ids, &pages, 2)
            .unwrap();
        assert_state_moved(&host, 4, 0, 0, 2);
        // One launch a rank for the capture, one for the restore.
        assert_eq!(ranks_of_launches(&host, start).len(), 2 * all.len());
        let copies = journal_ranks(&host, start, "copy ", "");
        assert!(copies.is_empty(), "{ranks:?} {kv:?}: {copies:?}");
        let journal = host.device.journal.borrow();
        assert!(
            journal[start..]
                .iter()
                .any(|line| line.starts_with("kernel copy "))
        );
        drop(journal);
        instance.retire_request(9).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// The ranks of the snapshot copy launches from journal line `from` on.
fn ranks_of_launches(host: &FakeHost, from: usize) -> Vec<u32> {
    journal_ranks(host, from, "launch ", kernels_sm120::snapshot_copy::SYMBOL)
}

/// Each rank loads the kernel's module once.
#[test]
fn every_instance_loads_the_kernel_on_each_rank() {
    let (host, mut instance, _) = prefilled(Ranks::TP2, KV_LAYOUTS[0]);
    let loaded = host
        .device
        .module_cubins
        .borrow()
        .values()
        .filter(|cubin| **cubin == crate::sm120::snapshot_copy::CUBIN)
        .count();
    assert_eq!(loaded, 2);
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A launch that fails on rank 1 ends the capture with the host's error
/// once ranks 0 and 1, which received work, are drained; ranks 2 and 3
/// receive nothing. The instance stays usable and the next capture of the
/// same state succeeds; a restore whose launch fails leaves its slot free
/// for a restore that succeeds.
#[test]
fn a_failed_launch_drains_the_ranks_that_received_work() {
    let (host, mut instance, pages) = prefilled(Ranks::TP4, KV_LAYOUTS[0]);
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
        let (host, mut instance, pages) = prefilled(Ranks::TP4, KV_LAYOUTS[0]);
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
/// uploads or launches anything.
#[test]
fn refused_pages_submit_nothing() {
    let (host, mut instance, pages) = prefilled(Ranks::TP2, KV_LAYOUTS[0]);
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
