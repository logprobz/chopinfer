//! The bytes prefix snapshots hold and restore across one request's life,
//! followed through the fake host's uploads, device copies, copy-kernel
//! launches and readbacks. The copy kernel is emulated from the addressing
//! of `cuda/qwen38_bundle.cuh` (`prefix_copy_spans`), not from the bundle's
//! piece list, so a change to how the bundle builds its copies shows up
//! here. Set `CHOPINFER_QWEN_SNAPSHOT_DUMP` to a directory to write every
//! configuration's bytes and journal there, to compare two commits.

use std::fmt::Write as _;

use bundle_runtime::fake_host::dump::{Memory, Origin, describe};

use super::speculation::{SpeculativeSink, batch, kept, truth};
use super::*;
use kernel_core::precision::execution::PrecisionDocument;

/// What one rank's copies did.
#[derive(Default)]
struct Rank {
    /// The serving pool, from the copy kernel's arguments.
    pool: Option<DeviceBufferSpanV1>,
    /// The spans device copies, copy kernels and header uploads wrote, in
    /// order.
    written: Vec<DeviceBufferSpanV1>,
    reads: Vec<Vec<(u64, Origin)>>,
}

#[derive(Default)]
struct Replay {
    memory: Memory,
    ranks: BTreeMap<u32, Rank>,
}

impl Replay {
    fn apply(&mut self, host: &FakeHost, ops: Vec<SnapshotOp>) {
        let rank_of = |stream: u32| host.stream_ranks.borrow()[&stream];
        for op in ops {
            match op {
                SnapshotOp::Upload(stream, to, bytes) => {
                    let rank = self.ranks.entry(rank_of(stream)).or_default();
                    self.memory
                        .write(to, vec![(to.length, Origin::Uploaded(bytes.into(), 0))]);
                    // A capture's header goes straight into the pages.
                    if rank.pool.is_some_and(|pool| pool.buffer == to.buffer) {
                        rank.written.push(to);
                    }
                }
                SnapshotOp::Copy(stream, from, to) => {
                    self.memory.copy(from, to);
                    let rank = self.ranks.entry(rank_of(stream)).or_default();
                    rank.written.push(to);
                }
                SnapshotOp::Launch(stream, arguments) => {
                    let KernelArgumentV1::DeviceBuffer(pool) = arguments[5] else {
                        panic!("the pool is {:?}", arguments[5]);
                    };
                    let KernelArgumentV1::DeviceBuffer(table) = arguments[6] else {
                        panic!("the page table is {:?}", arguments[6]);
                    };
                    let pages = self
                        .memory
                        .bytes(table)
                        .chunks_exact(8)
                        .map(|page| u64::from_le_bytes(page.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let rank = self.ranks.entry(rank_of(stream)).or_default();
                    rank.pool = Some(pool);
                    for (from, to) in prefix_copy_spans(host.shape(), &arguments, &pages) {
                        self.memory.copy(from, to);
                        rank.written.push(to);
                    }
                }
                SnapshotOp::Read(stream, from) => {
                    let parts = self.memory.read(from);
                    self.ranks
                        .entry(rank_of(stream))
                        .or_default()
                        .reads
                        .push(parts);
                }
            }
        }
    }
}

/// One served configuration: topology, whether the bundle's MTP drafter
/// drafts (with the MTP layout), the MTP layout serving without
/// speculation, FP8 main-layer KV, and the served sequences.
#[derive(Clone, Copy)]
struct Case {
    shape: Shape,
    drafter: bool,
    mtp_layout: bool,
    fp8: bool,
    served: u32,
}

// Keep Flash's snapshot headers stable. Update this field list whenever Shape changes.
struct SnapshotShape(Shape);

impl std::fmt::Debug for SnapshotShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.0;
        let mut debug = f.debug_struct("Shape");
        if shape.geometry.id != model_qwen38_contract::geometry::FLASH.id {
            debug.field("geometry", &shape.geometry.id);
        }
        debug
            .field("ranks", &shape.ranks)
            .field("vocab_per_rank", &shape.vocab_per_rank)
            .field("key_heads", &shape.key_heads)
            .field("value_heads", &shape.value_heads)
            .field("linear_qkv", &shape.linear_qkv)
            .field("linear_value", &shape.linear_value)
            .field("linear_ab_stride", &shape.linear_ab_stride)
            .field("q_heads", &shape.q_heads)
            .field("attn_row", &shape.attn_row)
            .field("attn_value", &shape.attn_value)
            .field("local_experts", &shape.local_experts)
            .field("shared_intermediate", &shape.shared_intermediate)
            .field("ngram_local_heads", &shape.ngram_local_heads)
            .field("ngram_local", &shape.ngram_local)
            .field("conv_entry_bytes", &shape.conv_entry_bytes)
            .field("delta_entry_bytes", &shape.delta_entry_bytes)
            .finish()
    }
}

impl std::fmt::Debug for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Case")
            .field("shape", &SnapshotShape(self.shape))
            .field("drafter", &self.drafter)
            .field("mtp_layout", &self.mtp_layout)
            .field("fp8", &self.fp8)
            .field("served", &self.served)
            .finish()
    }
}

impl Case {
    fn name(self) -> String {
        format!(
            "tp{}-{}-{}-{}",
            self.shape.ranks(),
            match (self.drafter, self.mtp_layout) {
                (true, _) => "mtp",
                (false, true) => "mtp-layout",
                (false, false) => "none",
            },
            if self.fp8 { "fp8" } else { "bf16" },
            self.served
        )
    }

    fn speculates(self) -> bool {
        self.drafter || !self.mtp_layout
    }
}

/// How an item runs.
#[derive(Clone, Copy)]
enum Kind {
    /// A prefill chunk, keeping its anchor when set.
    Prefill(bool),
    Decode,
    /// A verify item of this many rows.
    Verify(u32),
}

/// One item of a step: (request, slot, tokens before, inputs, first page,
/// kind).
type Item<'a> = (u64, u32, u64, &'a [u32], u64, Kind);

/// Request 7's life in slot `served - 3`, with every device copy, upload
/// and snapshot copy kernel followed through the ranks' memory from the
/// first step on. Kernels other than the snapshot copy are not modeled:
/// their launches are in the journal.
///
/// Residents fill the other slots below. Request 7 prefills 150 tokens
/// keeping its anchor at 128, which is captured; it decodes; a verify step
/// keeps two of its four rows beside a resident's decode and a resident's
/// three kept rows; its snapshot goes to odd pages. A verify step is
/// cancelled, which invalidates its slots; it is preempted (retired) and
/// resumes from the snapshot in its slot, then decodes. A second snapshot
/// goes to even pages; a prefill chunk is cancelled; it is retired, the
/// host reloads that snapshot into other pages (an offload), and it resumes
/// from those and decodes. Request 8 restores the anchor into slot
/// `served - 2` and prefills on; request 9 restores the reloaded snapshot
/// into slot `served - 1` and decodes. Without speculation, verify steps
/// are decode steps. Every phase dumps its journal and the bytes its copies
/// wrote; captures and restores must move exactly the declared snapshot's
/// bytes, the header on rank 0. A precision document serves the projections
/// through the package's prepared component. A `rows` scenario exports the
/// launches' rows (`export_rows`), checked against the host's launches.
fn lifecycle(
    case: Case,
    precision: Option<PrecisionDocument>,
    rows: Option<&str>,
) -> (String, Vec<String>) {
    let Case {
        shape,
        drafter,
        mtp_layout,
        fp8,
        served,
    } = case;
    let host = Rc::new(FakeHost::at(shape));
    *host.precision.borrow_mut() = precision;
    host.ngram_host.set(ngram_in_host_memory(shape));
    host.drafter.set(mtp_layout);
    host.fp8_kv.set(fp8);
    host.served.set(Some(served));
    host.speculation.set(if case.speculates() {
        (3, MAX_VERIFY_TOKENS)
    } else {
        (0, 0)
    });
    crate::pipeline::take_recorded();
    let mut instance = create(&host);
    host.model(instance.handle());
    host.backing.set(true);
    let kv = &shape.kv_formats()[usize::from(mtp_layout) + 2 * usize::from(fp8)];
    let state = model_qwen38_contract::state::request_state(shape, kv, drafter);
    let (snapshot_bytes, header_bytes) = (state.snapshot_bytes(), state.header_bytes);
    let mut dump = format!("{case:?} lifecycle\n");
    let mut failures = Vec::new();
    let mut replay = Replay::default();
    let mut phase = |name: &str,
                     kind: Option<bool>,
                     replay: &mut Replay,
                     run: &mut dyn FnMut() -> Result<(), BundleAbiError>| {
        let marks = replay
            .ranks
            .iter()
            .map(|(&rank, state)| (rank, (state.written.len(), state.reads.len())))
            .collect::<BTreeMap<_, _>>();
        let start = host.journal.borrow().len();
        *host.snapshot_ops.borrow_mut() = Some(Vec::new());
        let result = run();
        let ops = host.snapshot_ops.borrow_mut().take().unwrap();
        writeln!(dump, "phase {name}: {result:?}").unwrap();
        if result.is_err() {
            failures.push(format!("{name}: {result:?} {:?}", host.logs.borrow()));
        }
        for line in &host.journal.borrow()[start..] {
            writeln!(dump, "  {line}").unwrap();
        }
        replay.apply(&host, ops);
        let mut copied = 0;
        for (&rank, state) in &replay.ranks {
            let (written, reads) = marks.get(&rank).copied().unwrap_or_default();
            let pool = state.pool.map(|pool| pool.buffer.raw);
            let mut moved = 0;
            for &to in &state.written[written..] {
                let (buffer, mut at) = (to.buffer.raw, to.offset);
                writeln!(dump, "rank {rank} wrote {buffer:#x}+{at} {}", to.length).unwrap();
                for (length, origin) in replay.memory.read(to) {
                    if kind == Some(false)
                        && matches!(origin, Origin::Held(held, _) if Some(held) == pool)
                    {
                        failures.push(format!("{name}: rank {rank} restored unwritten page bytes"));
                    }
                    writeln!(
                        dump,
                        "  {at} {length} <- {}",
                        describe(&origin, length, Some(64))
                    )
                    .unwrap();
                    at += length;
                }
                // A capture writes the pages; a restore writes the state.
                if kind.is_some_and(|capture| capture == (Some(buffer) == pool)) {
                    moved += to.length;
                }
            }
            for read in &state.reads[reads..] {
                writeln!(dump, "rank {rank} read").unwrap();
                for (length, origin) in read {
                    writeln!(
                        dump,
                        "  {length} <- {}",
                        describe(origin, *length, Some(64))
                    )
                    .unwrap();
                }
            }
            if let Some(capture) = kind {
                copied += 1;
                // Rank 0 writes the header after a capture's pieces.
                let expected = if capture && rank == 0 {
                    snapshot_bytes
                } else {
                    snapshot_bytes - header_bytes
                };
                if moved != expected {
                    failures.push(format!("{name}: rank {rank} moved {moved} bytes"));
                }
            }
        }
        if kind.is_some() && copied != shape.ranks.count() {
            failures.push(format!("{name}: {copied} ranks copied"));
        }
    };
    let ok = |_| ();
    let step = |items: &[Item<'_>]| {
        let items = items
            .iter()
            .map(|&(request, slot, start, inputs, first_page, kind)| {
                let rows = match kind {
                    Kind::Prefill(_) => inputs.len() as u32,
                    Kind::Decode => 1,
                    Kind::Verify(rows) => rows,
                };
                let end = start + u64::from(rows);
                let item = match kind {
                    Kind::Prefill(anchor) => {
                        BatchItemV1::new(request, false, rows, true, inputs.to_vec())
                            .with_anchor(anchor)
                    }
                    Kind::Decode => BatchItemV1::new(request, true, 1, true, inputs[..1].to_vec()),
                    Kind::Verify(_) if drafter => {
                        BatchItemV1::verify(request, rows, true, inputs[..1].to_vec())
                    }
                    Kind::Verify(_) => {
                        BatchItemV1::verify(request, rows, false, inputs[..rows as usize].to_vec())
                    }
                };
                let pages = (0..end.div_ceil(PAGE_TOKENS))
                    .map(|page| first_page + page)
                    .collect();
                (
                    item,
                    BatchAppendV1::new(start, u64::from(rows), end, pages, None),
                    u64::from(slot),
                )
            })
            .collect();
        batch(items).encode().unwrap()
    };
    let truths = |range: std::ops::Range<u64>| range.map(truth).collect::<Vec<_>>();
    let execute = |instance: &mut BundleInstanceV1, request: Vec<u8>| {
        instance
            .execute(&request, &mut SpeculativeSink::default())
            .map(ok)
    };
    // Without speculation a verify item is a decode item, and commits whole.
    let verify = |rows| {
        if case.speculates() {
            Kind::Verify(rows)
        } else {
            Kind::Decode
        }
    };
    let accepted = |instance: &mut BundleInstanceV1, records: &[(u64, u32)]| {
        if case.speculates() {
            instance.commit_accepted(&kept(records))
        } else {
            instance.commit()
        }
    };

    let (slot7, slot8, slot9) = (served - 3, served - 2, served - 1);
    let residents = (0..served - 3)
        .map(|slot| (1000 + u64::from(slot), slot))
        .collect::<Vec<_>>();
    phase("residents", None, &mut replay, &mut || {
        let one = [5];
        let items = residents
            .iter()
            .map(|&(request, slot)| (request, slot, 0, &one[..], 1000, Kind::Prefill(false)))
            .collect::<Vec<_>>();
        execute(&mut instance, step(&items))?;
        instance.commit()
    });
    let prompt = truths(0..150);
    phase("prefill-anchor", None, &mut replay, &mut || {
        execute(
            &mut instance,
            step(&[(7, slot7, 0, &prompt, 900, Kind::Prefill(true))]),
        )?;
        instance.commit()
    });
    let count = instance.describe_prefix_state().unwrap();
    let pages = |first: u64, stride: u64| {
        (0..count)
            .rev()
            .map(|page| first + stride * page)
            .collect::<Vec<_>>()
    };
    let (anchor_pages, odd, even) = (pages(600, 1), pages(1, 2), pages(2, 2));
    let reloaded = (0..count).map(|page| 300 + page).collect::<Vec<_>>();
    phase(
        &format!("capture-anchor ({count} pages)"),
        Some(true),
        &mut replay,
        &mut || instance.capture_prefix_state(7, 128, &anchor_pages),
    );
    phase("decode", None, &mut replay, &mut || {
        execute(
            &mut instance,
            step(&[(7, slot7, 150, &[truth(150)], 900, Kind::Decode)]),
        )?;
        instance.commit()
    });
    let resident_tokens = [5, truth(1), truth(2), truth(3)];
    phase("verify-keep", None, &mut replay, &mut || {
        let verify_tokens = truths(151..155);
        execute(
            &mut instance,
            step(&[
                (1000, 0, 1, &resident_tokens[1..2], 1000, Kind::Decode),
                (7, slot7, 151, &verify_tokens, 900, verify(4)),
                (1001, 1, 1, &resident_tokens[1..], 1000, verify(3)),
            ]),
        )?;
        accepted(&mut instance, &[(7, 2), (1001, 3)])
    });
    let mut tokens = truths(0..153);
    if !case.speculates() {
        tokens.truncate(152);
    }
    let length = tokens.len() as u64;
    phase("capture-1", Some(true), &mut replay, &mut || {
        instance.capture_prefix_state(7, length, &odd)
    });
    phase("verify-cancel", None, &mut replay, &mut || {
        let verify_tokens = truths(length..length + 4);
        execute(
            &mut instance,
            step(&[
                (7, slot7, length, &verify_tokens, 900, verify(4)),
                (1002, 2, 1, &resident_tokens[1..2], 1000, Kind::Decode),
            ]),
        )?;
        instance.cancel()
    });
    phase("preempt", None, &mut replay, &mut || {
        instance.retire_request(1002)?;
        instance.retire_request(7)
    });
    phase("resume", Some(false), &mut replay, &mut || {
        instance.restore_prefix_state_in_slot(7, &tokens, &odd, u64::from(slot7))
    });
    phase("resume-decode", None, &mut replay, &mut || {
        execute(
            &mut instance,
            step(&[(7, slot7, length, &[truth(length)], 900, Kind::Decode)]),
        )?;
        instance.commit()
    });
    tokens.push(truth(length));
    let length = tokens.len() as u64;
    phase("capture-2", Some(true), &mut replay, &mut || {
        instance.capture_prefix_state(7, length, &even)
    });
    phase("chunk-cancel", None, &mut replay, &mut || {
        let chunk = truths(length..length + 5);
        execute(
            &mut instance,
            step(&[(7, slot7, length, &chunk, 900, Kind::Prefill(false))]),
        )?;
        instance.cancel()
    });
    phase("offload", None, &mut replay, &mut || {
        instance.retire_request(7)
    });
    // The host reloads the offloaded snapshot into other pages.
    let page_bytes = kv.page_bytes;
    for state in replay.ranks.values() {
        let pool = state.pool.unwrap();
        let page = |page: u64| {
            DeviceBufferSpanV1::new(pool.buffer, pool.offset + page * page_bytes, page_bytes)
        };
        for (&from, &to) in even.iter().zip(&reloaded) {
            replay.memory.copy(page(from), page(to));
            host.write(page(to), &host.read(page(from)));
            host.sim
                .borrow_mut()
                .copy(page(to), page(from), &host.extents.borrow());
        }
    }
    phase("offload-restore", Some(false), &mut replay, &mut || {
        instance.restore_prefix_state_in_slot(7, &tokens, &reloaded, u64::from(slot7))
    });
    phase("offload-decode", None, &mut replay, &mut || {
        execute(
            &mut instance,
            step(&[(7, slot7, length, &[truth(length)], 900, Kind::Decode)]),
        )?;
        instance.commit()
    });
    phase("restore-anchor", Some(false), &mut replay, &mut || {
        instance.restore_prefix_state_in_slot(8, &prompt[..128], &anchor_pages, u64::from(slot8))
    });
    phase("anchor-prefill", None, &mut replay, &mut || {
        execute(
            &mut instance,
            step(&[(8, slot8, 128, &prompt[128..], 910, Kind::Prefill(false))]),
        )?;
        instance.commit()
    });
    phase("restore-other", Some(false), &mut replay, &mut || {
        instance.restore_prefix_state_in_slot(9, &tokens, &reloaded, u64::from(slot9))
    });
    phase("other-decode", None, &mut replay, &mut || {
        execute(
            &mut instance,
            step(&[(9, slot9, length, &[truth(length)], 920, Kind::Decode)]),
        )?;
        instance.commit()
    });
    instance.destroy().unwrap();
    if let Some(scenario) = rows {
        super::export_rows(scenario, &[&host]);
    }
    (dump, failures)
}

#[test]
fn lifecycle_snapshot_bytes_follow_cancels_preemption_and_offload() {
    let mut cases = Vec::new();
    for shape in Shape::ALL {
        for drafter in [false, true] {
            for fp8 in [false, true] {
                cases.push(Case {
                    shape,
                    drafter,
                    mtp_layout: drafter,
                    fp8,
                    served: 32,
                });
            }
        }
    }
    let case = |shape, drafter, mtp_layout, served| Case {
        shape,
        drafter,
        mtp_layout,
        fp8: false,
        served,
    };
    cases.extend([
        // The MTP layout serving without speculation: a ring with the
        // drafter's layer that no drafter writes.
        case(Shape::TP4, false, true, 32),
        // 33 sequences: request 9 restores into slot 32, the first of the
        // 64-slot bucket's second half.
        case(Shape::TP4, false, false, 33),
        case(Shape::TP4, true, true, 33),
        case(Shape::TP4, false, false, 100),
        case(Shape::TP4, false, false, 256),
    ]);
    let directory = std::env::var_os("CHOPINFER_QWEN_SNAPSHOT_DUMP").map(std::path::PathBuf::from);
    let mut failures = Vec::new();
    for case in cases {
        let (dump, failed) = lifecycle(case, None, Some(&format!("prefix-{}", case.name())));
        if let Some(directory) = &directory {
            std::fs::write(directory.join(format!("{}.txt", case.name())), dump).unwrap();
        }
        failures.extend(
            failed
                .into_iter()
                .map(|failure| format!("{}: {failure}", case.name())),
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_geometry_snapshot_covers_fp8_and_mtp_layouts() {
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::of_geometry(geometry, ranks).unwrap();
            for fp8 in [false, true] {
                for (drafter, mtp_layout) in [(false, false), (false, true), (true, true)] {
                    let case = Case {
                        shape,
                        drafter,
                        mtp_layout,
                        fp8,
                        served: 32,
                    };
                    let (_, failures) = lifecycle(case, None, None);
                    assert!(failures.is_empty(), "{case:?}: {}", failures.join("\n"));
                }
            }
        }
    }
}

/// The lifecycle with each kind of prepared precision component: FP8 legacy
/// projections at both degrees (BF16 and FP8 KV, with and without the MTP
/// layout), NVFP4 experts, and NVFP4 experts with MXFP8 dense and the NVFP4
/// n-gram lookup and its hash, at decode buckets up to 256. The dumps' SHA-256
/// digests were recorded from the lowering before the shared projection
/// runtime and must not change. `CHOPINFER_QWEN_PRECISION_DUMP` names a
/// directory to write the dumps to.
#[test]
fn precision_lifecycles_keep_their_recorded_bytes() {
    let mut cases = Vec::new();
    let case = |shape, mtp_layout, fp8, served| Case {
        shape,
        drafter: false,
        mtp_layout,
        fp8,
        served,
    };
    for shape in Shape::ALL {
        for mtp_layout in [false, true] {
            for fp8 in [false, true] {
                cases.push(("fp8", false, false, case(shape, mtp_layout, fp8, 32)));
                if shape.ranks() == 4 {
                    cases.push(("fp8", false, false, case(shape, mtp_layout, fp8, 100)));
                }
            }
        }
    }
    for (label, mx) in [("nvidia", false), ("mxfp8", true)] {
        for served in [33, 100, 256] {
            cases.push((label, true, mx, case(Shape::TP4, false, false, served)));
        }
        cases.push((label, true, mx, case(Shape::TP4, false, true, 100)));
    }
    let directory = std::env::var_os("CHOPINFER_QWEN_PRECISION_DUMP").map(std::path::PathBuf::from);
    let mut digests = String::new();
    for (label, nv, mx, case) in cases {
        let document = super::precision::document_for(case.shape, nv, mx, false);
        let name = format!("precision-{label}-{}", case.name());
        let (dump, failures) = lifecycle(case, Some(document), Some(&name));
        assert!(
            failures.is_empty(),
            "{label} {case:?}: {}",
            failures.join("\n")
        );
        if let Some(directory) = &directory {
            std::fs::write(directory.join(format!("{name}.txt")), &dump).unwrap();
        }
        writeln!(
            digests,
            "{}  {name}.txt",
            kernel_build::digest(dump.as_bytes())
        )
        .unwrap();
    }
    assert_eq!(
        digests,
        include_str!("../../tests/fixtures/precision-lifecycles.sha256"),
        "precision lifecycle dumps changed"
    );
}
