//! The bytes a prefix snapshot holds across one request's life, in the
//! autoregressive and MTP configurations over both latent KV formats, one
//! served item with the precision component bound: the request prefills,
//! decodes (or verifies, with the drafter), describes its prefix state and
//! captures it, and the snapshot's bytes are pinned by digest. A second run
//! retires the request after the capture, overwrites its slot, restores the
//! snapshot into it and continues; the continuation issues exactly the
//! uninterrupted run's journal and leaves the same state.
//!
//! The host runs no kernel, so the slot's KDA state, convolution history,
//! indexer tail and (with the drafter) retained rows are filled with known
//! bytes after the last step, standing in for what the kernels wrote. The
//! latent KV and pooled keys live in the serving pages, which the host
//! holds, so the snapshot does not copy them; the continuation names the
//! same pages in both runs. Set `CHOPINFER_GLM5NEXT_SNAPSHOT_DUMP` to a
//! directory to write every configuration's pieces, header and journal
//! there, to compare two commits.

use std::fmt::Write as _;

use bundle_runtime::fake_host::fnv1a;

use super::*;

/// A configuration: its KV format (an index into `Shape::kv_formats`), the
/// host's (drafts, verify tokens), and the snapshot's pinned pages and
/// byte digest (every rank's pieces and header, in rank order). The pieces
/// do not depend on the latent format, so both formats share a digest and
/// differ in pages only.
struct Case {
    name: &'static str,
    format: usize,
    speculation: (u32, u32),
    pages: u64,
    digest: u64,
}

const CASES: [Case; 4] = [
    Case {
        name: "ar-fp8",
        format: 0,
        speculation: (0, 0),
        pages: 89,
        digest: 0x27e4_53a5_d60e_d3a1,
    },
    Case {
        name: "ar-bf16",
        format: 2,
        speculation: (0, 0),
        pages: 49,
        digest: 0x27e4_53a5_d60e_d3a1,
    },
    Case {
        name: "mtp-fp8",
        format: 1,
        speculation: (3, 32),
        pages: 82,
        digest: 0x477b_4237_b7cc_4cad,
    },
    Case {
        name: "mtp-bf16",
        format: 3,
        speculation: (3, 32),
        pages: 45,
        digest: 0x477b_4237_b7cc_4cad,
    },
];

/// Prompt tokens of request 7.
const PROMPT: u64 = 100;

/// What a run left: the snapshot's bytes per rank, the continuation's
/// journal, and the state after it.
struct Run {
    pages: u64,
    snapshot: Vec<Vec<u8>>,
    continuation: Vec<String>,
    after: Vec<Vec<u8>>,
    holder: (Option<(u64, u64)>, bool),
    frontier: Option<u64>,
    dump: String,
}

impl Case {
    fn format(&self) -> KvFormat {
        Shape::TP4.kv_formats()[self.format]
    }

    fn drafting(&self) -> bool {
        has_mtp_layer(&self.format()) && self.speculation.0 > 0
    }

    /// Runs request 7's life; with `interrupt`, the request is retired
    /// after its capture and restored from the snapshot into an overwritten
    /// slot before it continues.
    fn run(&self, interrupt: bool) -> Run {
        let format = self.format();
        let host = FakeHost::with_format(self.speculation, format);
        host.items.set(1);
        host.backing.set(true);
        let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
        let record = crate::config::tests::record_naming(1, &options(2048), format, &roles);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap_or_else(|error| panic!("{}: {error:?} {:?}", self.name, host.logs.borrow()));
        let handle = instance.handle();
        let mut dump = format!("{} interrupt {interrupt}\n", self.name);
        let phase = |name: &str, dump: &mut String, from: usize| {
            writeln!(dump, "phase {name}").unwrap();
            for line in &host.device.journal.borrow()[from..] {
                writeln!(dump, "  {line}").unwrap();
            }
        };
        let fail = |error: BundleAbiError| -> ! {
            panic!("{}: {error:?} {:?}", self.name, host.logs.borrow())
        };
        let mut sink = DraftSink::default();
        let mut tokens = (0..PROMPT as u32).collect::<Vec<_>>();

        let from = host.device.journal.borrow().len();
        instance
            .execute(&prompt_chunk(7, 0, PROMPT, true), &mut sink)
            .unwrap_or_else(|error| fail(error));
        instance.commit().unwrap_or_else(|error| fail(error));
        phase("prefill", &mut dump, from);

        // A verify step the drafter drafts keeps two rows (the bonus token
        // and the first draft, which the fake device reads back as zero);
        // without speculation a decode step.
        let from = host.device.journal.borrow().len();
        if self.drafting() {
            instance
                .execute(&drafted(7, PROMPT, 4, 21), &mut sink)
                .unwrap_or_else(|error| fail(error));
            instance
                .commit_accepted(&[kept(7, 2)])
                .unwrap_or_else(|error| fail(error));
            tokens.extend([21, 0]);
        } else {
            let decode = step(&[(7, 0, PROMPT, &[21])]);
            instance
                .execute(&decode, &mut sink)
                .unwrap_or_else(|error| fail(error));
            instance.commit().unwrap_or_else(|error| fail(error));
            tokens.push(21);
        }
        phase("step", &mut dump, from);
        let length = tokens.len() as u64;
        assert_eq!(
            crate::instance::ledger_state(handle, 0),
            (Some((7, length)), false)
        );
        let frontier = crate::instance::drafter_frontier(handle, 0);
        assert_eq!(frontier, self.drafting().then_some(PROMPT));

        let count = instance
            .describe_prefix_state()
            .unwrap_or_else(|error| fail(error));
        // What the kernels wrote, in every declared piece on every rank.
        let source = fill_slot(&host, handle, 0, 1);
        let (_, table) = crate::instance::snapshot_layout(handle, 0);
        assert_eq!(count, table.pages());
        // Pages past the request's own, in reverse order.
        let pages = (0..count).rev().map(|page| 100 + page).collect::<Vec<_>>();
        let from = host.device.journal.borrow().len();
        instance
            .capture_prefix_state(7, length, &pages)
            .unwrap_or_else(|error| fail(error));
        phase(&format!("capture ({count} pages)"), &mut dump, from);
        let snapshot = (0..4)
            .map(|rank| snapshot_bytes(&host, &table, rank, &pages))
            .collect::<Vec<_>>();
        let state =
            model_glm5next_contract::state::request_state(Shape::TP4, &format, self.drafting());
        for (rank, bytes) in snapshot.iter().enumerate() {
            writeln!(dump, "rank {rank} snapshot {} bytes", bytes.len()).unwrap();
            for piece in &table.descriptors {
                let at = piece.at as usize..(piece.at + piece.length) as usize;
                assert!(
                    bytes[at.clone()] == source[rank][at.clone()],
                    "{} rank {rank} piece {piece:?}",
                    self.name
                );
                writeln!(
                    dump,
                    "  {} +{} {} at {} {:016x}",
                    state.buffers[piece.buffer].name,
                    piece.offset,
                    piece.length,
                    piece.at,
                    fnv1a(&bytes[at])
                )
                .unwrap();
            }
            let header = &bytes[table.header_at() as usize..];
            assert_eq!(header.iter().any(|&byte| byte != 0), rank == 0);
            writeln!(dump, "  header {header:02x?}").unwrap();
        }

        if interrupt {
            let from = host.device.journal.borrow().len();
            instance
                .retire_request(7)
                .unwrap_or_else(|error| fail(error));
            fill_slot(&host, handle, 0, 2);
            instance
                .restore_prefix_state_in_slot(7, &tokens, &pages, 0)
                .unwrap_or_else(|error| fail(error));
            phase("retire and restore", &mut dump, from);
            assert!(
                slot_bytes(&host, handle, 0) == source,
                "{}: the restore filled the slot from the pages",
                self.name
            );
            assert_eq!(
                crate::instance::ledger_state(handle, 0),
                (Some((7, length)), false)
            );
            assert_eq!(crate::instance::drafter_frontier(handle, 0), frontier);
        }

        let from = host.device.journal.borrow().len();
        if self.drafting() {
            instance
                .execute(&drafted(7, length, 4, 22), &mut sink)
                .unwrap_or_else(|error| fail(error));
            instance
                .commit_accepted(&[kept(7, 3)])
                .unwrap_or_else(|error| fail(error));
        } else {
            let decode = step(&[(7, 0, length, &[22])]);
            instance
                .execute(&decode, &mut sink)
                .unwrap_or_else(|error| fail(error));
            instance.commit().unwrap_or_else(|error| fail(error));
        }
        phase("continue", &mut dump, from);
        let continuation = host.device.journal.borrow()[from..].to_vec();
        let run = Run {
            pages: count,
            snapshot,
            continuation,
            after: slot_bytes(&host, handle, 0),
            holder: crate::instance::ledger_state(handle, 0),
            frontier: crate::instance::drafter_frontier(handle, 0),
            dump,
        };
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        run
    }
}

#[test]
fn prefix_snapshots_restore_and_continue_as_the_uninterrupted_run() {
    let directory =
        std::env::var_os("CHOPINFER_GLM5NEXT_SNAPSHOT_DUMP").map(std::path::PathBuf::from);
    let mut failures = Vec::new();
    for case in &CASES {
        let uninterrupted = case.run(false);
        let restored = case.run(true);
        if let Some(directory) = &directory {
            for (run, suffix) in [(&uninterrupted, "uninterrupted"), (&restored, "restored")] {
                let path = directory.join(format!("{}-{suffix}.txt", case.name));
                std::fs::write(path, &run.dump).unwrap();
            }
        }
        assert!(
            restored.snapshot == uninterrupted.snapshot,
            "{}: both runs capture the same bytes",
            case.name
        );
        assert_eq!(
            restored.continuation, uninterrupted.continuation,
            "{}: the restored request continues as the uninterrupted one",
            case.name
        );
        assert!(restored.after == uninterrupted.after, "{}", case.name);
        assert_eq!(restored.holder, uninterrupted.holder, "{}", case.name);
        assert_eq!(restored.frontier, uninterrupted.frontier, "{}", case.name);
        let pages = uninterrupted.pages;
        let digest = fnv1a(&uninterrupted.snapshot.concat());
        if (pages, digest) != (case.pages, case.digest) {
            failures.push(format!(
                "{}: pages {pages}, digest {digest:#018x}",
                case.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "snapshot bytes moved from the pinned digests:\n{}",
        failures.join("\n")
    );
}
