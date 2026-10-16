//! The bytes a prefix snapshot holds and restores, followed through the
//! fake host's uploads, device copies, copy-kernel launches and readbacks.
//! The copy kernel is emulated from the addressing of `cuda/mimo_ring.cuh`,
//! not from the bundle's piece list, so a change to how the bundle builds
//! its copies shows up here. Set `CHOPINFER_MIMO_SNAPSHOT_DUMP` to a
//! directory to write every configuration's bytes and journal there, to
//! compare two commits.
use std::fmt::Write as _;

use model_mimo26_contract::state::{Drafter, request_state};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

use bundle_runtime::fake_host::dump::{Memory, Origin, describe, paged_copies};

use super::*;

fn span(arguments: &[KernelArgumentV1], index: usize) -> Option<DeviceBufferSpanV1> {
    match arguments[index] {
        KernelArgumentV1::DeviceBuffer(span) => Some(span),
        KernelArgumentV1::NullDevicePointer => None,
        ref other => panic!("argument {index}: {other:?}"),
    }
}

fn scalar(arguments: &[KernelArgumentV1], index: usize) -> u64 {
    match arguments[index] {
        KernelArgumentV1::ScalarU32(value) => value.into(),
        KernelArgumentV1::ScalarU64(value) => value,
        ref other => panic!("argument {index}: {other:?}"),
    }
}

/// `prefix_copy` of `cuda/mimo_ring.cuh` in a ring of `slots` slots: every
/// layer's K then V slot entry, then the drafter bytes, end to end across
/// the pages. Returns the spans it wrote.
fn prefix_copy(
    memory: &mut Memory,
    tp: Tp,
    slots: u64,
    arguments: &[KernelArgumentV1],
) -> Vec<DeviceBufferSpanV1> {
    assert_eq!(arguments.len(), 9);
    let state = span(arguments, 0).unwrap();
    let drafter = span(arguments, 1);
    let drafter_bytes = scalar(arguments, 2);
    let pool = span(arguments, 3).unwrap();
    let page_count = scalar(arguments, 5);
    let page_bytes = scalar(arguments, 6);
    let slot = scalar(arguments, 7);
    let to_pages = scalar(arguments, 8) != 0;
    let table = memory.bytes(span(arguments, 4).unwrap());
    let pages = table
        .chunks_exact(8)
        .take(page_count as usize)
        .map(|page| u64::from_le_bytes(page.try_into().unwrap()))
        .collect::<Vec<_>>();
    let heads = tp.kv_heads(false);
    let (k_entry, v_entry) = (heads * 192 * 2, heads * 128 * 2);
    let (k_slot, v_slot) = (128 * k_entry, 128 * v_entry);
    let (k_layer, layer) = (slots * k_slot, slots * (k_slot + v_slot));
    let mut held = Vec::new();
    for index in 0..39 {
        let base = state.offset + index * layer;
        held.push((base + slot * k_slot, k_slot));
        held.push((base + k_layer + slot * v_slot, v_slot));
    }
    let held = held
        .into_iter()
        .map(|(offset, length)| DeviceBufferSpanV1::new(state.buffer, offset, length))
        .chain(drafter.map(|span| DeviceBufferSpanV1::new(span.buffer, span.offset, drafter_bytes)))
        .filter(|span| span.length > 0);
    let mut wrote = Vec::new();
    for (from, to) in paged_copies(held, pool, &pages, page_bytes, to_pages) {
        memory.copy(from, to);
        wrote.push(to);
    }
    wrote
}

/// What one rank's copies did to its memory.
#[derive(Default)]
struct Rank {
    memory: Memory,
    /// The serving pool, from the copy kernel's arguments.
    pool: Option<DeviceBufferSpanV1>,
    /// The spans device copies and copy kernels wrote, in order.
    written: Vec<DeviceBufferSpanV1>,
    reads: Vec<Vec<(u64, Origin)>>,
}

fn replay(host: &FakeHost, ranks: &mut BTreeMap<u32, Rank>, ops: Vec<SnapshotOp>) {
    let wrote = |rank: &mut Rank, to: DeviceBufferSpanV1| rank.written.push(to);
    for op in ops {
        match op {
            SnapshotOp::Upload(stream, to, bytes) => {
                let rank = ranks.entry(stream).or_default();
                // The page table; a capture's header is copied on from here.
                rank.memory
                    .write(to, vec![(to.length, Origin::Uploaded(bytes.into(), 0))]);
            }
            SnapshotOp::Copy(stream, from, to) => {
                let rank = ranks.entry(stream).or_default();
                rank.memory.copy(from, to);
                wrote(rank, to);
            }
            SnapshotOp::Launch(stream, arguments) => {
                let rank = ranks.entry(stream).or_default();
                rank.pool = span(&arguments, 3);
                let slots = u64::from(host.ring_slots());
                for to in prefix_copy(&mut rank.memory, host.tp(), slots, &arguments) {
                    wrote(rank, to);
                }
            }
            SnapshotOp::Read(stream, from) => {
                let rank = ranks.entry(stream).or_default();
                let parts = rank.memory.read(from);
                rank.reads.push(parts);
            }
        }
    }
}

/// Request 7 commits a prefill and a verify step in slot `served - 2`, its
/// snapshot goes to odd pages in descending order, and request 8 restores it
/// into slot `served - 1`. Returns the dump, and how any rank failed to
/// capture or restore exactly the declared snapshot's bytes.
fn snapshot(tp: Tp, drafter: Drafter, fp8: bool, served: u32) -> (String, Vec<String>) {
    let host = Rc::new(FakeHost::at(tp));
    // Allocated buffers take IDs past the serving pools' (100 + rank), so no
    // fake handle names two buffers.
    host.next.set(1000);
    host.served.set(served);
    host.fp8_kv.set(fp8);
    host.pool_pages.set(Some(512));
    host.speculation.set(Some((7, 32)));
    host.dflash.set(drafter == Drafter::Dflash);
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_serving(tp, host.kv(), drafter != Drafter::None, served),
        )
        .unwrap();
    let residents = u64::from(served) - 2;
    let hold = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..residents)
            .map(|request| BatchItemV1::new(1000 + request, false, 1, true, vec![5]))
            .collect(),
        (0..residents)
            .map(|_| BatchAppendV1::new(0, 1, 1, vec![127], None))
            .collect(),
    );
    instance
        .execute(&hold.encode().unwrap(), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    let step = |request, start: u64, tokens: Vec<u32>, rows: u32, verify: bool| {
        let item = if verify {
            BatchItemV1::verify(request, rows, drafter != Drafter::None, tokens)
        } else {
            BatchItemV1::new(request, start > 0, rows, true, tokens)
        };
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![item],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                start + u64::from(rows),
                (0..(start + u64::from(rows)).div_ceil(PAGE_TOKENS)).collect(),
                None,
            )],
        )
        .encode()
        .unwrap()
    };
    let mut tokens = (0..1022).collect::<Vec<u32>>();
    instance
        .execute(
            &step(7, 0, tokens.clone(), 1022, false),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    *host.readback_tokens.borrow_mut() = vec![1022, 1023, 1024, 1025];
    let input = if drafter == Drafter::None {
        vec![1022, 1023, 1024, 1025]
    } else {
        vec![1022]
    };
    instance
        .execute(&step(7, 1022, input, 4, true), &mut DraftSink::default())
        .unwrap();
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    tokens.extend([1022, 1023]);
    let boundary = tokens.len() as u64;
    let count = instance.describe_prefix_state().unwrap();
    let pages = (0..count)
        .rev()
        .map(|page| 2 * page + 1)
        .collect::<Vec<_>>();

    let mut dump =
        format!("tp={tp:?} drafter={drafter:?} fp8={fp8} served={served} pages={count}\n");
    let mut ranks = BTreeMap::<u32, Rank>::new();
    let mut captured_spans = BTreeMap::new();
    for restore in [false, true] {
        if restore {
            captured_spans = ranks
                .iter()
                .map(|(&stream, rank)| (stream, rank.written.len()))
                .collect();
        }
        let start = host.journal.borrow().len();
        *host.snapshot_ops.borrow_mut() = Some(Vec::new());
        if restore {
            instance.restore_prefix_state(8, &tokens, &pages).unwrap();
        } else {
            instance.capture_prefix_state(7, boundary, &pages).unwrap();
        }
        let ops = host.snapshot_ops.borrow_mut().take().unwrap();
        let name = if restore { "restore" } else { "capture" };
        writeln!(dump, "{name} journal").unwrap();
        for line in &host.journal.borrow()[start..] {
            writeln!(dump, "  {line}").unwrap();
        }
        replay(&host, &mut ranks, ops);
        if !restore {
            for (stream, rank) in &ranks {
                writeln!(dump, "stream {stream} pages after capture").unwrap();
                let pool = rank.pool.unwrap().buffer.raw;
                for (buffer, start, length, origin) in rank.memory.runs(|buffer, _| buffer == pool)
                {
                    writeln!(
                        dump,
                        "  {buffer:#x}+{start} {length} <- {}",
                        describe(&origin, length, None)
                    )
                    .unwrap();
                }
            }
        }
    }
    let state = request_state(tp, host.kv(), drafter);
    let mut failures = Vec::new();
    if ranks.len() != tp.count() {
        failures.push(format!("{} ranks copied", ranks.len()));
    }
    for (stream, rank) in &ranks {
        let pool = rank.pool.unwrap().buffer.raw;
        let captured = rank
            .memory
            .runs(|buffer, _| buffer == pool)
            .iter()
            .map(|run| run.2)
            .sum::<u64>();
        if captured != state.snapshot_bytes() {
            failures.push(format!("stream {stream} captured {captured} bytes"));
        }
        writeln!(dump, "stream {stream} restored").unwrap();
        let mut restored = 0;
        for &to in &rank.written[captured_spans[stream]..] {
            let parts = rank.memory.read(to);
            let (buffer, mut at) = (to.buffer.raw, to.offset);
            for (length, origin) in parts {
                if matches!(origin, Origin::Held(held, _) if held == pool) {
                    failures.push(format!("stream {stream} restored unwritten page bytes"));
                }
                writeln!(
                    dump,
                    "  {buffer:#x}+{at} {length} <- {}",
                    describe(&origin, length, None)
                )
                .unwrap();
                at += length;
            }
            restored += to.length;
        }
        if restored != state.snapshot_bytes() {
            failures.push(format!("stream {stream} restored {restored} bytes"));
        }
        for read in &rank.reads {
            writeln!(dump, "stream {stream} read").unwrap();
            for (length, origin) in read {
                writeln!(dump, "  {length} <- {}", describe(origin, *length, None)).unwrap();
            }
        }
    }
    instance.destroy().unwrap();
    super::launch_journal::export_host(
        &format!(
            "snapshot-{}-{drafter:?}-{fp8}-{served}",
            crate::topology::topology_id(tp)
        ),
        &host,
    );
    (dump, failures)
}

#[test]
fn snapshot_bytes_follow_the_copy_kernel_and_the_projected_copies() {
    let mut cases = Vec::new();
    for tp in Tp::SUPPORTED {
        for drafter in [Drafter::None, Drafter::Mtp, Drafter::Dflash] {
            for fp8 in [false, true] {
                cases.push((tp, drafter, fp8, 32));
            }
        }
    }
    for served in [100, 256] {
        for drafter in [Drafter::None, Drafter::Dflash] {
            cases.push((Tp::TP4, drafter, false, served));
        }
    }
    let directory = std::env::var_os("CHOPINFER_MIMO_SNAPSHOT_DUMP").map(std::path::PathBuf::from);
    let mut failures = Vec::new();
    for (tp, drafter, fp8, served) in cases {
        let name = format!(
            "tp{}-{drafter:?}-{}-{served}",
            tp.ranks(),
            if fp8 { "fp8" } else { "bf16" }
        );
        let (dump, failed) = snapshot(tp, drafter, fp8, served);
        if let Some(directory) = &directory {
            std::fs::write(directory.join(format!("{name}.txt")), dump).unwrap();
        }
        failures.extend(
            failed
                .into_iter()
                .map(|failure| format!("{name}: {failure}")),
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A batch of one item of `request`: `rows` rows from `start`, a verify item
/// when `verify`, drafted by the bundle when it has a drafter.
fn one_item(
    drafter: Drafter,
    request: u64,
    start: u64,
    tokens: Vec<u32>,
    rows: u32,
    verify: bool,
) -> Vec<u8> {
    let item = if verify {
        BatchItemV1::verify(request, rows, drafter != Drafter::None, tokens)
    } else {
        BatchItemV1::new(request, start > 0, rows, true, tokens)
    };
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![item],
        vec![BatchAppendV1::new(
            start,
            u64::from(rows),
            start + u64::from(rows),
            (0..(start + u64::from(rows)).div_ceil(PAGE_TOKENS)).collect(),
            None,
        )],
    )
    .encode()
    .unwrap()
}

/// One request's life across the calls that change its state's bytes, with
/// every device copy, upload and snapshot copy kernel followed through the
/// ranks' memory from creation on. Kernels other than the snapshot copy are
/// not modeled: their launches are in the journal.
///
/// Request 7 commits a prefill in slot `served - 2`; a decode is cancelled;
/// a verify keeps two of four rows; a second verify is cancelled. Its
/// snapshot goes to odd pages. It is preempted (retired) and resumes from
/// the snapshot under the same request, then decodes a token. A second
/// snapshot goes to even pages; it is retired again, the host reloads that
/// snapshot into other pages (an offload), and it resumes from those and
/// decodes. Request 8 then restores the same anchor into slot
/// `served - 1` and decodes. Every phase dumps its journal and the bytes
/// its copies wrote; captures and restores must move exactly the declared
/// snapshot's bytes.
fn lifecycle(tp: Tp, drafter: Drafter, fp8: bool, served: u32) -> (String, Vec<String>) {
    let host = Rc::new(FakeHost::at(tp));
    host.next.set(1000);
    host.served.set(served);
    host.fp8_kv.set(fp8);
    host.pool_pages.set(Some(512));
    host.speculation.set(Some((7, 32)));
    host.dflash.set(drafter == Drafter::Dflash);
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_serving(tp, host.kv(), drafter != Drafter::None, served),
        )
        .unwrap();
    let state = request_state(tp, host.kv(), drafter);
    let snapshot_bytes = state.snapshot_bytes();
    let mut dump = format!("tp={tp:?} drafter={drafter:?} fp8={fp8} served={served} lifecycle\n");
    let mut failures = Vec::new();
    let mut ranks = BTreeMap::<u32, Rank>::new();
    let mut phase = |name: &str,
                     kind: Option<bool>,
                     ranks: &mut BTreeMap<u32, Rank>,
                     run: &mut dyn FnMut() -> Result<(), BundleAbiError>| {
        let marks = ranks
            .iter()
            .map(|(&stream, rank)| (stream, (rank.written.len(), rank.reads.len())))
            .collect::<BTreeMap<_, _>>();
        let start = host.journal.borrow().len();
        *host.snapshot_ops.borrow_mut() = Some(Vec::new());
        let result = run();
        let ops = host.snapshot_ops.borrow_mut().take().unwrap();
        writeln!(dump, "phase {name}: {result:?}").unwrap();
        if result.is_err() {
            failures.push(format!("{name}: {result:?}"));
        }
        for line in &host.journal.borrow()[start..] {
            writeln!(dump, "  {line}").unwrap();
        }
        replay(&host, ranks, ops);
        for (stream, rank) in ranks.iter() {
            let (written, reads) = marks.get(stream).copied().unwrap_or_default();
            let pool = rank.pool.map(|pool| pool.buffer.raw);
            let mut moved = 0;
            for &to in &rank.written[written..] {
                let (buffer, mut at) = (to.buffer.raw, to.offset);
                writeln!(dump, "stream {stream} wrote {buffer:#x}+{at} {}", to.length).unwrap();
                for (length, origin) in rank.memory.read(to) {
                    if kind == Some(false)
                        && matches!(origin, Origin::Held(held, _) if Some(held) == pool)
                    {
                        failures.push(format!(
                            "{name}: stream {stream} restored unwritten page bytes"
                        ));
                    }
                    writeln!(
                        dump,
                        "  {at} {length} <- {}",
                        describe(&origin, length, None)
                    )
                    .unwrap();
                    at += length;
                }
                // A capture writes the pages; a restore writes the state.
                if kind.is_some_and(|capture| capture == (Some(buffer) == pool)) {
                    moved += to.length;
                }
            }
            for read in &rank.reads[reads..] {
                writeln!(dump, "stream {stream} read").unwrap();
                for (length, origin) in read {
                    writeln!(dump, "  {length} <- {}", describe(origin, *length, None)).unwrap();
                }
            }
            if kind.is_some() && moved != snapshot_bytes {
                failures.push(format!("{name}: stream {stream} moved {moved} bytes"));
            }
        }
        if kind.is_some() && ranks.len() != tp.count() {
            failures.push(format!("{name}: {} ranks copied", ranks.len()));
        }
    };
    let ok = |_| ();

    let residents = u64::from(served) - 2;
    let hold = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..residents)
            .map(|request| BatchItemV1::new(1000 + request, false, 1, true, vec![5]))
            .collect(),
        (0..residents)
            .map(|_| BatchAppendV1::new(0, 1, 1, vec![127], None))
            .collect(),
    )
    .encode()
    .unwrap();
    let verify_input = |token: u32| {
        if drafter == Drafter::None {
            (token..token + 4).collect()
        } else {
            vec![token]
        }
    };
    let mut tokens = (0..1022).collect::<Vec<u32>>();
    phase("residents", None, &mut ranks, &mut || {
        instance.execute(&hold, &mut DraftSink::default()).map(ok)?;
        instance.commit()
    });
    phase("prefill", None, &mut ranks, &mut || {
        let step = one_item(drafter, 7, 0, tokens.clone(), 1022, false);
        instance.execute(&step, &mut DraftSink::default()).map(ok)?;
        instance.commit()
    });
    phase("decode-cancel", None, &mut ranks, &mut || {
        let step = one_item(drafter, 7, 1022, vec![1022], 1, false);
        instance.execute(&step, &mut DraftSink::default()).map(ok)?;
        instance.cancel()
    });
    *host.readback_tokens.borrow_mut() = vec![1022, 1023, 1024, 1025];
    phase("verify-keep-2", None, &mut ranks, &mut || {
        let step = one_item(drafter, 7, 1022, verify_input(1022), 4, true);
        instance.execute(&step, &mut DraftSink::default()).map(ok)?;
        instance.commit_accepted(&[kept(7, 2)])
    });
    tokens.extend([1022, 1023]);
    *host.readback_tokens.borrow_mut() = vec![1024, 1025, 1026, 1027];
    phase("verify-cancel", None, &mut ranks, &mut || {
        let step = one_item(drafter, 7, 1024, verify_input(1024), 4, true);
        instance.execute(&step, &mut DraftSink::default()).map(ok)?;
        instance.cancel()
    });
    let count = instance.describe_prefix_state().unwrap();
    let odd = (0..count)
        .rev()
        .map(|page| 2 * page + 1)
        .collect::<Vec<_>>();
    let even = (0..count)
        .rev()
        .map(|page| 2 * page + 2)
        .collect::<Vec<_>>();
    let reloaded = (0..count).map(|page| 300 + page).collect::<Vec<_>>();
    phase(
        &format!("capture-1 ({count} pages)"),
        Some(true),
        &mut ranks,
        &mut || instance.capture_prefix_state(7, 1024, &odd),
    );
    phase("preempt", None, &mut ranks, &mut || {
        instance.retire_request(7)
    });
    phase("resume", Some(false), &mut ranks, &mut || {
        instance.restore_prefix_state(7, &tokens, &odd)
    });
    phase("resume-decode", None, &mut ranks, &mut || {
        let step = one_item(drafter, 7, 1024, vec![1024], 1, false);
        instance.execute(&step, &mut DraftSink::default()).map(ok)?;
        instance.commit()
    });
    tokens.push(1024);
    phase("capture-2", Some(true), &mut ranks, &mut || {
        instance.capture_prefix_state(7, 1025, &even)
    });
    phase("offload", None, &mut ranks, &mut || {
        instance.retire_request(7)
    });
    // The host reloads the offloaded snapshot into other pages.
    let page_bytes = state.page_bytes();
    for rank in ranks.values_mut() {
        let pool = rank.pool.unwrap();
        for (&from, &to) in even.iter().zip(&reloaded) {
            let page = |page: u64| {
                DeviceBufferSpanV1::new(pool.buffer, pool.offset + page * page_bytes, page_bytes)
            };
            rank.memory.copy(page(from), page(to));
            let mut memory = host.snapshot_memory.borrow_mut();
            let bytes = memory
                .range(
                    (pool.buffer.raw, page(from).offset)
                        ..(pool.buffer.raw, page(from).offset + page_bytes),
                )
                .map(|(&(_, offset), &byte)| (offset - page(from).offset, byte))
                .collect::<Vec<_>>();
            for (offset, byte) in bytes {
                memory.insert((pool.buffer.raw, page(to).offset + offset), byte);
            }
        }
    }
    phase("offload-restore", Some(false), &mut ranks, &mut || {
        instance.restore_prefix_state(7, &tokens, &reloaded)
    });
    phase("offload-decode", None, &mut ranks, &mut || {
        let step = one_item(drafter, 7, 1025, vec![1025], 1, false);
        instance.execute(&step, &mut DraftSink::default()).map(ok)?;
        instance.commit()
    });
    phase("restore-other", Some(false), &mut ranks, &mut || {
        instance.restore_prefix_state(8, &tokens, &reloaded)
    });
    phase("other-decode", None, &mut ranks, &mut || {
        let step = one_item(drafter, 8, 1025, vec![1025], 1, false);
        instance.execute(&step, &mut DraftSink::default()).map(ok)?;
        instance.commit()
    });
    instance.destroy().unwrap();
    super::launch_journal::export_host(
        &format!(
            "lifecycle-{}-{drafter:?}-{fp8}-{served}",
            crate::topology::topology_id(tp)
        ),
        &host,
    );
    (dump, failures)
}

#[test]
fn lifecycle_snapshot_bytes_follow_cancels_preemption_and_offload() {
    let mut cases = Vec::new();
    for drafter in [Drafter::None, Drafter::Mtp, Drafter::Dflash] {
        cases.push((Tp::TP4, drafter, false, 32));
    }
    // 33 sequences: the restore lands in slot 32, the first of the 64-slot
    // bucket's second half.
    for drafter in [Drafter::None, Drafter::Dflash] {
        cases.push((Tp::TP4, drafter, false, 33));
    }
    cases.extend([
        (Tp::TP4, Drafter::Dflash, true, 32),
        (Tp::TP2, Drafter::Dflash, false, 32),
        (Tp::TP2, Drafter::Mtp, false, 32),
        (Tp::TP2, Drafter::None, true, 32),
    ]);
    let directory = std::env::var_os("CHOPINFER_MIMO_SNAPSHOT_DUMP").map(std::path::PathBuf::from);
    let mut failures = Vec::new();
    for (tp, drafter, fp8, served) in cases {
        let name = format!(
            "lifecycle-tp{}-{drafter:?}-{}-{served}",
            tp.ranks(),
            if fp8 { "fp8" } else { "bf16" }
        );
        let (dump, failed) = lifecycle(tp, drafter, fp8, served);
        if let Some(directory) = &directory {
            std::fs::write(directory.join(format!("{name}.txt")), dump).unwrap();
        }
        failures.extend(
            failed
                .into_iter()
                .map(|failure| format!("{name}: {failure}")),
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
