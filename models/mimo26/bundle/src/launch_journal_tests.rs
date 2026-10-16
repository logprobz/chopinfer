//! The fake-host launch journal of fixed MiMo steps at TP2 and TP4.
//!
//! Each scenario records device-facing host service calls, including launch
//! configurations and arguments, upload hashes, copies, graph and event
//! operations, allocations, logits views and drafts. A mismatch writes the
//! journal to `TMPDIR` for diffing. Refactors preserve the pinned digests.
//!
//! The baselines include the shared runtime extraction, prefix snapshots,
//! graph preflight and release order, decode and round-once prefill
//! collectives, tuned expert schedules, bucketed row grids and HF RoPE
//! tables. Exact verify rows and shared-key attention changed the verify
//! launches and entry resolutions. Their recorded updates compared the
//! unaffected launches, uploads, copies and allocations with handles mapped
//! by first appearance. The row-local paged decode split (2026-10-04) added
//! one argument, `split_units`, after `split_target` to the global-attention
//! decode partial, merge and exact-rows launches; the base and new journals
//! of every scenario differ in those lines only. Row-local verify
//! (2026-10-05) added the same argument to the verify partial and merge
//! launches; only the DFlash block attention launches of the DFlash
//! scenarios change, since every scenario's verify steps run exact rows. Without stream-K experts
//! (2026-10-04), TP4 no longer resolves the wide verify entries, and down's
//! decode entry at 8 rows, now a ring of one K split, replaces its exact
//! entry: the launches of the exact verify rows name the decode entry with
//! the same geometry, and the resolve, configure and release lines of the
//! dropped entries left the journals; every other launch is equal. With
//! the stream-K entries deleted, neither topology resolves the 16- and
//! 32-row decode entries, nor TP4 the 16- and 32-row fused down-combine
//! entries, which no step launched; launches and allocations are
//! unchanged. With ring decode entries at 16 rows (2026-10-05) both
//! topologies resolve those entries again (and TP4 the 16-row fused
//! down-combine entry); no scenario step has 16 rows, so launches and
//! allocations are unchanged. The wide-count prefill rows (2026-10-05:
//! TP4 down, TP2 gate/up and down on shallower rings) change only the
//! dynamic shared memory of those three entries, in their configure and
//! launch lines; every other field and line is equal.

use bundle_runtime::fake_host::check_journal;

use super::*;
use runtime_bundle_sdk::{
    BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaSectionV3, BatchRequestV1,
    BatchWireVersionV1,
};

type JournalPin = (usize, u64);

/// (scenario, line count and FNV-1a digest). Plain decode v1 left the comm
/// cubin on 2026-10-05: each rank resolves and releases one entry fewer,
/// and launch geometry, arguments, uploads and copies are unchanged after
/// mapping handles.
/// TP4's baselines include protocol-sized comm inboxes, separate parallel
/// route tickets and round-once prefill all-reduces (2026-09-30). The encoder
/// catalog and per-modality recorded programs were added on 2026-10-01;
/// the vision and audio F32 pairs and residual quantizer followed on
/// 2026-10-02. Those updates checked that unaffected launch geometry,
/// arguments, uploads and copies stayed equal after mapping handle IDs.
/// MTP's EH shard moved onto the shared column slice on 2026-10-10
/// (identity change, accuracy unaffected): each rank loads the slice's
/// module and resolves and releases its entry, and each pack launch became
/// the slice's at the same grid, stream and buffers (256 threads, shape
/// arguments for the shard index); every other line is equal after mapping
/// handles. Deleting the pack's cubins then left each rank one module and
/// entry fewer to load, resolve and release, with the same line counts as
/// before the move.
const EXPECTED: [(&str, JournalPin); 7] = [
    ("target", (23713, 0xad58_6944_0148_7fd6)),
    ("mtp", (12434, 0xe683_2d67_de17_366a)),
    ("dflash", (11739, 0x97ff_bbf4_3cd9_da33)),
    ("vision", (614, 0x94fb_6d47_38b0_5b9f)),
    ("vision_parallel", (614, 0x94fb_6d47_38b0_5b9f)),
    ("audio", (718, 0xe68e_7a02_3186_0f58)),
    ("audio_parallel", (718, 0xe68e_7a02_3186_0f58)),
];

fn note_views(host: &FakeHost, step: &str, views: &[OwnedLogitsViewV1]) {
    for view in views {
        host.note(format!("{step} view {:?}", view.raw()));
    }
}

fn note_drafts(host: &FakeHost, step: &str, sink: &DraftSink) {
    for drafts in &sink.drafts {
        host.note(format!("{step} drafts {drafts:?}"));
    }
    note_views(host, step, &sink.views);
}

fn api() -> BundleApiV1 {
    BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).expect("bundle API")
}

fn run(
    host: &FakeHost,
    instance: &mut runtime_bundle_abi::BundleInstanceV1,
    step: &str,
    request: BatchRequestV1,
) {
    let mut sink = ViewSink::default();
    instance
        .execute(&request.encode().unwrap(), &mut sink)
        .unwrap_or_else(|error| panic!("{step}: {error:?}"));
    note_views(host, step, &sink.0);
}

/// Prefill, decode, verify with a kept prefix, a cancelled decode, and a
/// pipelined prefill mixed with a decode, without a drafter.
fn target() -> Rc<FakeHost> {
    target_with(false)
}

fn target_with(fp8_kv: bool) -> Rc<FakeHost> {
    target_with_scales(fp8_kv, None)
}

fn target_with_scales(fp8_kv: bool, scales: Option<&'static [[f32; 2]]>) -> Rc<FakeHost> {
    target_on(Tp::TP4, fp8_kv, scales)
}

fn target_at(tp: Tp) -> Rc<FakeHost> {
    target_on(tp, false, None)
}

fn target_on(tp: Tp, fp8_kv: bool, scales: Option<&'static [[f32; 2]]>) -> Rc<FakeHost> {
    target_on_format(tp, crate::contract::KV_FORMATS[usize::from(fp8_kv)], scales)
}

fn target_on_format(
    tp: Tp,
    format: bundle_runtime::kv::KvFormat,
    scales: Option<&'static [[f32; 2]]>,
) -> Rc<FakeHost> {
    let host = Rc::new(FakeHost::at(tp));
    host.speculation.set(Some((7, 32)));
    host.fp8_kv.set(format.fp8());
    host.kv_layout_id.set(Some(format.layout_id));
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_for_at(tp, &format, false),
        )
        .expect("create");
    if let Some(scales) = scales {
        crate::instance::force_kv_scales(instance.handle(), scales);
    }
    crate::instance::force_overlap(
        instance.handle(),
        Some(crate::overlap::Config {
            sms: 188,
            reserved_sms: 16,
            comm_ctas: Some(24),
            min_tokens: 1024,
        }),
    );
    run(
        &host,
        &mut instance,
        "prefill",
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(50, false, 3, true, vec![1, 2, 3]),
                BatchItemV1::new(51, false, 3, true, vec![4, 5, 6]),
            ],
            vec![
                BatchAppendV1::new(0, 3, 3, vec![1], None),
                BatchAppendV1::new(0, 3, 3, vec![2], None),
            ],
        ),
    );
    instance.commit().expect("commit prefill");
    let decode = |previous: u64, tokens: [u32; 2]| {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(50, true, 1, true, vec![tokens[0]]),
                BatchItemV1::new(51, true, 1, true, vec![tokens[1]]),
            ],
            vec![
                BatchAppendV1::new(previous, 1, previous + 1, vec![1], None),
                BatchAppendV1::new(previous, 1, previous + 1, vec![2], None),
            ],
        )
    };
    run(&host, &mut instance, "decode", decode(3, [7, 8]));
    instance.commit().expect("commit decode");
    run(
        &host,
        &mut instance,
        "verify",
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::verify(50, 4, false, vec![10, 11, 12, 13]),
                BatchItemV1::new(51, true, 1, true, vec![9]),
            ],
            vec![
                BatchAppendV1::new(4, 4, 8, vec![1], None),
                BatchAppendV1::new(4, 1, 5, vec![2], None),
            ],
        ),
    );
    instance
        .commit_accepted(&[kept(50, 2)])
        .expect("keep two verify rows");
    run(
        &host,
        &mut instance,
        "cancelled decode",
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(50, true, 1, true, vec![14]),
                BatchItemV1::new(51, true, 1, true, vec![15]),
            ],
            vec![
                BatchAppendV1::new(6, 1, 7, vec![1], None),
                BatchAppendV1::new(5, 1, 6, vec![2], None),
            ],
        ),
    );
    instance.cancel().expect("cancel decode");
    run(
        &host,
        &mut instance,
        "pipelined prefill",
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(52, false, 1100, true, (0..1100).collect()),
                BatchItemV1::new(50, true, 1, true, vec![16]),
            ],
            vec![
                BatchAppendV1::new(0, 1100, 1100, (3..21).collect(), None),
                BatchAppendV1::new(6, 1, 7, vec![1], None),
            ],
        ),
    );
    instance.commit().expect("commit pipelined prefill");
    for request in [50, 51, 52] {
        instance.retire_request(request).expect("retire");
    }
    instance.destroy().expect("destroy");
    export_host(
        &format!(
            "{}-target-{}-{}",
            crate::topology::topology_id(tp),
            std::str::from_utf8(format.layout_id).unwrap(),
            if scales.is_some() {
                "scaled"
            } else {
                "default"
            }
        ),
        &host,
    );
    host
}

fn drafted(start: u64, tokens: Vec<u32>, verify: bool) -> Vec<u8> {
    let rows = if verify { 4 } else { tokens.len() as u32 };
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![if verify {
            BatchItemV1::verify(7, rows, true, tokens)
        } else {
            BatchItemV1::new(7, false, rows, true, tokens)
        }],
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

/// A drafter instance: prefill, two bundle-drafted verify steps with kept
/// prefixes, and a cancelled one between them.
fn drafter(dflash: bool) -> Rc<FakeHost> {
    drafter_with(dflash, false)
}

fn drafter_with(dflash: bool, fp8_kv: bool) -> Rc<FakeHost> {
    drafter_on(Tp::TP4, dflash, fp8_kv)
}

fn drafter_at(tp: Tp, dflash: bool) -> Rc<FakeHost> {
    drafter_on(tp, dflash, false)
}

fn drafter_on(tp: Tp, dflash: bool, fp8_kv: bool) -> Rc<FakeHost> {
    let host = drafter_steps(tp, dflash, fp8_kv);
    export_host(
        &format!(
            "{}-{}-{}",
            crate::topology::topology_id(tp),
            if dflash { "dflash" } else { "mtp" },
            if fp8_kv { "fp8" } else { "bf16" }
        ),
        &host,
    );
    host
}

/// The drafter scenario's steps, unexported.
fn drafter_steps(tp: Tp, dflash: bool, fp8_kv: bool) -> Rc<FakeHost> {
    let host = Rc::new(FakeHost::at(tp));
    host.speculation.set(Some((7, 32)));
    host.dflash.set(dflash);
    host.fp8_kv.set(fp8_kv);
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_for_at(tp, host.kv(), true),
        )
        .expect("create");
    let step = |instance: &mut runtime_bundle_abi::BundleInstanceV1,
                name: &str,
                request: Vec<u8>,
                readback: Vec<u32>| {
        *host.readback_tokens.borrow_mut() = readback;
        let mut sink = DraftSink::default();
        instance
            .execute(&request, &mut sink)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        note_drafts(&host, name, &sink);
    };
    let verify = || vec![90, 91, 92, 93];
    step(
        &mut instance,
        "prefill",
        drafted(0, vec![40, 41, 42], false),
        Vec::new(),
    );
    instance.commit().expect("commit prefill");
    step(
        &mut instance,
        "verify",
        drafted(3, vec![43], true),
        vec![43, 44, 45, 46],
    );
    instance.commit_accepted(&[kept(7, 2)]).expect("keep two");
    step(
        &mut instance,
        "cancelled verify",
        drafted(5, vec![90], true),
        verify(),
    );
    instance.cancel().expect("cancel verify");
    step(
        &mut instance,
        "verify again",
        drafted(5, vec![90], true),
        verify(),
    );
    instance.commit_accepted(&[kept(7, 1)]).expect("keep one");
    instance.retire_request(7).expect("retire");
    instance.destroy().expect("destroy");
    host
}

/// The drafter scenarios run eagerly, as `CHOPINFER_MIMO_MTP_EAGER=1` or an
/// eager instance runs them: each step launches at its live rows (an MTP
/// catch-up of fewer rows than its verify bucket), where graphs launch
/// their buckets' rows. Their journals (each launch's grid and arguments)
/// and (rank, symbol, rows) records go to `eager-<topology>-<drafter>`
/// files for comparison; no journal is pinned. The ranks queue their eager
/// programs collective by collective (`pipeline::run_ranks`).
#[test]
fn eager_drafter_launch_rows() {
    if journal_switch_set() {
        return;
    }
    crate::pipeline::take_recorded();
    for tp in [Tp::TP2, Tp::TP4] {
        for dflash in [false, true] {
            crate::instance::FORCE_EAGER.with(|eager| eager.set(Some(true)));
            let host = drafter_steps(tp, dflash, false);
            crate::instance::FORCE_EAGER.with(|eager| eager.set(None));
            assert!(!host.collectives.borrow().is_empty());
            // Eager decode and verify programs go collective by collective
            // (476 launches behind a collective for MTP and 545 for DFlash
            // when queued rank by rank); the staged prefill queues a stage
            // at a time, and DFlash's hidden-row copy follows a stage's
            // all-reduce.
            assert_eq!(
                host.launches_past_unmatched_collectives(),
                usize::from(dflash),
                "{tp:?}"
            );
            let name = format!(
                "eager-{}-{}",
                crate::topology::topology_id(tp),
                if dflash { "dflash" } else { "mtp" }
            );
            export_rows(&name, &[&host]);
            export_host(&name, &host);
        }
    }
}

/// A traced step (`CHOPINFER_MIMO_TRACE_LAUNCHES`) queues its eager
/// programs collective by collective too, the ring backup alone first.
/// The fake host does not wait on events, so the step fails once every
/// launch is queued, when it waits on the first one's event.
#[test]
fn traced_steps_queue_collective_by_collective() {
    if journal_switch_set() {
        return;
    }
    crate::pipeline::take_recorded();
    for tp in [Tp::TP2, Tp::TP4] {
        let host = Rc::new(FakeHost::at(tp));
        let mut instance = api()
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_for_at(tp, host.kv(), false),
            )
            .expect("create");
        crate::instance::FORCE_TRACE.with(|trace| trace.set(true));
        let traced = instance.execute(
            &drafted(0, vec![40, 41, 42], false),
            &mut DraftSink::default(),
        );
        crate::instance::FORCE_TRACE.with(|trace| trace.set(false));
        assert!(traced.is_err(), "{tp:?}");
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.starts_with("trace rank 0 launch 0 ")),
            "{tp:?}"
        );
        assert!(!host.collectives.borrow().is_empty());
        assert_eq!(host.launches_past_unmatched_collectives(), 0, "{tp:?}");
    }
    crate::pipeline::take_recorded();
}

/// The record of `encoder` (vision 0, audio 1) on the shared encoder
/// runtime on one rank: relocated (`parallel`) or serial, its regions where
/// the pre-port bundle laid out the buffers of the same roles
/// (`super::shared_encoder`).
fn media(encoder: u32, parallel: bool) -> Rc<FakeHost> {
    let host = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(Rc::clone(&host));
    let services = table.caller();
    let (mut shared, symbols) =
        super::shared_encoder(services, &[BundleDeviceBufferHandleV1::from_parts(1, 1)]);
    shared.mode = if parallel {
        bundle_runtime::encoder::Mode::Relocated
    } else {
        bundle_runtime::encoder::Mode::Serial
    };
    let request = BatchRequestV1::new(BatchWireVersionV1::V3, Vec::new(), Vec::new())
        .with_media(media_section(encoder))
        .encode()
        .unwrap();
    let injects = shared
        .run(
            services,
            &[BundleStreamHandleV1::from_parts(1, 1)],
            &symbols,
            &request,
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
    assert!(!injects);
    host
}

/// The media scenarios' record of `encoder`: a two-grid video of 4 x 6
/// patches, or audio of 6,001 frames (two segments).
fn media_section(encoder: u32) -> BatchMediaSectionV3 {
    if encoder == 0 {
        return BatchMediaSectionV3 {
            encodes: vec![BatchMediaEncodeV3 {
                encoder: 0,
                geometry: [2, 4, 6, 0],
                output_rows: 12,
                pages: vec![7],
                payload_offset: 0,
                payload_bytes: 2 * 4 * 6 * 1536 * 4,
            }],
            payload: vec![0; 2 * 4 * 6 * 1536 * 4],
            ..BatchMediaSectionV3::default()
        };
    }
    let rows = 6001;
    BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder: 1,
            geometry: [rows, 128, 0, 0],
            output_rows: 376,
            pages: vec![9, 3, 8, 2, 7, 1],
            payload_offset: 0,
            payload_bytes: u64::from(rows) * 128 * 4,
        }],
        payload: vec![0; rows as usize * 128 * 4],
        ..BatchMediaSectionV3::default()
    }
}

/// Writes `journal` to `$CHOPINFER_MIMO26_JOURNAL_DIR/<scenario>.txt` when
/// the variable is set, whether or not it matches its digest, so a base
/// checkout's journals can be kept for comparison.
const JOURNAL_DIR: &str = "CHOPINFER_MIMO26_JOURNAL_DIR";

fn export(scenario: &str, journal: &str) {
    if let Some(root) = std::env::var_os(JOURNAL_DIR) {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).expect("create journal directory");
        std::fs::write(root.join(format!("{scenario}.txt")), journal).expect("export journal");
    }
}

/// Writes the launches this thread recorded since the last call
/// (`crate::pipeline::take_recorded`: rank or stream, symbol, rows) to
/// `$CHOPINFER_MIMO26_JOURNAL_DIR/<scenario>.rows.tsv` when the variable is
/// set, so the rows of every launch, which journals omit, can be compared.
/// They must be every launch `hosts` received, in the order the hosts were
/// created, by stream and symbol, each stream's in its order: no submission
/// bypasses the recorder. Eager programs are recorded rank by rank when
/// lowered and queued across the ranks collective by collective
/// (`pipeline::run_ranks`), so only each stream's order is compared. The
/// media scenarios pass no host, since their encoder launches go through
/// the shared encoder runtime, not the bundle's pipeline.
pub(super) fn export_rows(scenario: &str, hosts: &[&FakeHost]) {
    let recorded = crate::pipeline::take_recorded();
    if !hosts.is_empty() {
        let by_stream = |mut launches: Vec<(u32, String)>| {
            launches.sort_by_key(|(stream, _)| *stream);
            launches
        };
        let mut seen = recorded
            .iter()
            .map(|launch| (launch.stream, launch.symbol.clone()));
        assert!(!recorded.is_empty(), "{scenario}: no launch was recorded");
        for host in hosts {
            let launched = host
                .launch_records
                .borrow()
                .iter()
                .map(|(stream, symbol, ..)| (*stream, symbol.clone()))
                .collect::<Vec<_>>();
            let own = seen.by_ref().take(launched.len()).collect::<Vec<_>>();
            assert!(
                by_stream(own) == by_stream(launched),
                "{scenario}: the recorded launches are not the host's"
            );
        }
        assert_eq!(
            seen.count(),
            0,
            "{scenario}: launches were recorded past the hosts'"
        );
    }
    let rows = recorded
        .into_iter()
        .map(|launch| launch.line)
        .collect::<Vec<_>>();
    if let Some(root) = std::env::var_os(JOURNAL_DIR) {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).expect("create journal directory");
        std::fs::write(root.join(format!("{scenario}.rows.tsv")), rows.join("\n"))
            .expect("export launch rows");
    }
}

/// Records every allocation in issue order with its rank and physical bytes.
/// The media-only host has no rank stream mapping and uses rank zero.
pub(super) fn export_host(scenario: &str, host: &FakeHost) {
    export(scenario, &host.journal.borrow().join("\n"));
    if let Some(root) = std::env::var_os(JOURNAL_DIR) {
        use std::fmt::Write as _;
        let mut rows = String::from("index\trank\thandle\tbytes\n");
        let ranks = host.allocation_ranks.borrow();
        let bytes = host.allocation_bytes.borrow();
        for (index, &handle) in host.buffer_order.borrow().iter().enumerate() {
            let (rank, length) = ranks
                .get(&handle)
                .copied()
                .unwrap_or_else(|| (0, bytes[&handle]));
            writeln!(rows, "{index}\t{rank}\t{handle}\t{length}").unwrap();
        }
        assert_eq!(
            host.buffer_order.borrow().len(),
            host.journal
                .borrow()
                .iter()
                .filter(|line| line.starts_with("allocate "))
                .count()
        );
        std::fs::write(
            std::path::PathBuf::from(root).join(format!("{scenario}.allocations.tsv")),
            rows,
        )
        .unwrap();
    }
}

/// Whether an unpinned diagnostic selection is active.
fn journal_switch_set() -> bool {
    for name in [
        "CHOPINFER_MIMO_ATTENTION_FOLD",
        "CHOPINFER_MIMO_MTP_EAGER",
        "CHOPINFER_MIMO_VERIFY_EAGER",
        "CHOPINFER_MIMO_TRACE_LAUNCHES",
        "CHOPINFER_MIMO_F32_LOGITS",
        "CHOPINFER_MIMO_ROW_OWNED_LOGITS",
        "CHOPINFER_MIMO_ROW_OWNED_SPECULATION",
        "CHOPINFER_MIMO_WIDE_COMM",
        "CHOPINFER_MIMO_VERIFY_DENSE_MID",
        crate::selection::PREFILL_ROW_GRIDS,
        crate::selection::VERIFY_EXACT,
        crate::selection::EXACT_ATTENTION,
        crate::selection::VERIFY_ATTENTION,
    ] {
        if std::env::var_os(name).is_some() {
            eprintln!("launch journal skipped: {name} is set");
            return true;
        }
    }
    false
}

#[test]
fn mimo_launch_journal_is_unchanged() {
    if journal_switch_set() {
        return;
    }
    let mut failures = Vec::new();
    crate::pipeline::take_recorded();
    for (name, (lines, digest)) in EXPECTED {
        let host = match name {
            "target" => target(),
            "mtp" => drafter(false),
            "dflash" => drafter(true),
            "vision" => media(0, false),
            "vision_parallel" => media(0, true),
            "audio" => media(1, false),
            _ => media(1, true),
        };
        let checked: &[&FakeHost] = match name {
            "target" | "mtp" | "dflash" => &[&host],
            _ => &[],
        };
        export_rows(name, checked);
        export_host(name, &host);
        let journal = host.journal.borrow();
        export(name, &journal.join("\n"));
        if let Err(error) = check_journal(
            &journal,
            (lines, digest),
            &format!("bundle-mimo26-journal-{name}.txt"),
        ) {
            failures.push(format!("{name}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Decode entries use only static shared memory: plain decode v2 is
/// resolved on every rank without a dynamic-shared configure call.
#[test]
fn decode_create_uses_only_static_shared_memory() {
    for tp in [Tp::TP2, Tp::TP4] {
        let host = target_at(tp);
        let journal = host.journal.borrow();
        export(
            &format!("tp{}-decode-create", tp.ranks()),
            &journal.join("\n"),
        );
        let handles: Vec<_> = journal
            .iter()
            .filter_map(|line| {
                let mut words = line.split_whitespace();
                if words.next()? != "resolve" {
                    return None;
                }
                let handle = words.next()?;
                let symbol = words.next()?;
                (symbol == crate::sm120::comm::DECODE_SYMBOL_V2).then_some(handle)
            })
            .collect();
        assert_eq!(handles.len(), tp.ranks() as usize);
        for handle in handles {
            let prefix = format!("configure {handle} ");
            assert!(
                journal.iter().all(|line| !line.starts_with(&prefix)),
                "plain decode v2 requests dynamic shared memory: {prefix}"
            );
        }
    }
}

/// (scenario, pin) of the FP8 global-KV journals, re-recorded with
/// `EXPECTED` (last: the 512-row scored cap, which changes only the sizes
/// of the logits buffers, in their allocations and in the spans the heads
/// and logits gathers bind; line count, order, symbols, geometry and
/// uploads are unchanged; then the prefill row grids and the round-once
/// comm entries, then the generated encoder catalog).
const EXPECTED_FP8: [(&str, JournalPin); 3] = [
    ("target", (23713, 0xdd89_928a_1187_223c)),
    ("mtp", (12434, 0x1d98_63df_2ed6_c75a)),
    ("dflash", (11739, 0xcfd5_fb42_eb5e_baa1)),
];

/// The FP8 global-KV layout runs the BF16 program: the
/// target, MTP and DFlash journals hold the same lines as BF16's except the
/// launches that write or read the global-attention pages, which switch to
/// the FP8 entries of the same stage.
#[test]
fn mimo_fp8_kv_journals_differ_only_in_global_kv_launches() {
    // With a journal switch set only the digests are skipped.
    let digests = !journal_switch_set();
    let symbol = |line: &str| line.split(' ').nth(2).unwrap_or_default().to_owned();
    // The scenarios' verify steps (at most eight rows) run exact rows.
    let drafted = [
        "chop_attn_mimo_ga_fp8_decode_rows_merge_v1",
        "chop_attn_mimo_ga_fp8_decode_rows_shared_v1",
        "chop_attn_mimo_ga_fp8_prefill_direct_v1",
        "chop_row_rope_mimo_ga_fp8_table_p64_pos32_v1",
    ];
    let target = [
        "chop_attn_mimo_ga_fp8_decode_merge_v1",
        "chop_attn_mimo_ga_fp8_decode_rope_table_v1",
        "chop_attn_mimo_ga_fp8_prefill_merge_v1",
    ];
    let mut failures = Vec::new();
    crate::pipeline::take_recorded();
    for (name, (lines, digest)) in EXPECTED_FP8 {
        let (bf16, fp8) = match name {
            "target" => (target_with(false), target_with(true)),
            "mtp" => (drafter_with(false, false), drafter_with(false, true)),
            _ => (drafter_with(true, false), drafter_with(true, true)),
        };
        export_rows(&format!("{name}-bf16-and-fp8"), &[&bf16, &fp8]);
        let (bf16, fp8) = (bf16.journal.borrow(), fp8.journal.borrow());
        assert_eq!(bf16.len(), fp8.len(), "{name}");
        let mut changed = BTreeSet::new();
        for (bf16, fp8) in bf16.iter().zip(fp8.iter()) {
            if bf16 == fp8 {
                continue;
            }
            assert!(
                bf16.starts_with("launch ") && fp8.starts_with("launch "),
                "{name}: {bf16}\n{fp8}"
            );
            let (from, to) = (symbol(bf16), symbol(fp8));
            assert_eq!(from.replace("bf16", "fp8"), to, "{name}: {bf16}\n{fp8}");
            changed.insert(to);
        }
        let expected = drafted
            .iter()
            .chain(if name == "target" { &target[..] } else { &[] })
            .map(|symbol| symbol.to_string())
            .collect::<BTreeSet<_>>();
        assert_eq!(changed, expected, "{name}");
        export(&format!("{name}-fp8"), &fp8.join("\n"));
        if digests
            && let Err(error) = check_journal(
                &fp8,
                (lines, digest),
                &format!("bundle-mimo26-journal-{name}-fp8.txt"),
            )
        {
            failures.push(format!("{name}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// (topology ranks, scenario, pin) of the same steps at TP2, recorded when
/// those topologies were added and again when the comm cubin gained the decode
/// all-reduce entries: each rank resolves and releases two more entries, the
/// handle, stream, event and graph IDs after them move, and the hidden
/// all-reduces that `comm::push_symbols` sends to the decode entries change
/// only their symbols (geometry, buffers, sizes and order are unchanged), and
/// again when the medium-M GEMM cubin became default: each rank loads it and
/// resolves, configures and releases its ten entries, and again with the
/// prefill row grids, the round-once comm entries and the generated encoder
/// catalog (see `EXPECTED`).
const OTHER_TOPOLOGIES: [(u32, &str, JournalPin); 3] = [
    (2, "target", (12360, 0x922f_c275_11b2_487e)),
    (2, "mtp", (6719, 0x0ca7_b6bc_d290_4799)),
    (2, "dflash", (6377, 0x4bc6_7ca7_0697_1679)),
];

#[test]
fn mimo_launch_journal_at_tp2_is_unchanged() {
    if journal_switch_set() {
        return;
    }
    let mut failures = Vec::new();
    crate::pipeline::take_recorded();
    for (ranks, name, (lines, digest)) in OTHER_TOPOLOGIES {
        let tp = Tp::new(ranks).unwrap();
        let host = match name {
            "target" => target_at(tp),
            "mtp" => drafter_at(tp, false),
            _ => drafter_at(tp, true),
        };
        export_rows(&format!("tp{ranks}-{name}"), &[&host]);
        let journal = host.journal.borrow();
        export(&format!("tp{ranks}-{name}"), &journal.join("\n"));
        if let Err(error) = check_journal(
            &journal,
            (lines, digest),
            &format!("bundle-mimo26-journal-tp{ranks}-{name}.txt"),
        ) {
            failures.push(format!("({ranks}, \"{name}\") {error}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Distinct K and V scales per global layer.
static DISTINCT_SCALES: [[f32; 2]; GA_LAYERS as usize] = {
    let mut scales = [[0.0; 2]; GA_LAYERS as usize];
    let mut layer = 0;
    while layer < GA_LAYERS as usize {
        scales[layer] = [1.0 / (2 << layer) as f32, 1.0 / (4096 << layer) as f32];
        layer += 1;
    }
    scales
};

/// Each global layer's K and V scales, in that order, reach the launches
/// that write or read that layer's pages: the RoPE paged write and the
/// attention partial or direct entries of every phase, the decode fold
/// included. The layer is read back from the K plane each launch binds.
#[test]
fn mimo_fp8_kv_scales_reach_each_global_layers_launches() {
    let host = target_with_scales(true, Some(&DISTINCT_SCALES));
    let layer_bytes = GA_PAGE_BYTES_FP8 / GA_LAYERS;
    let mut seen = BTreeMap::<String, BTreeSet<u64>>::new();
    for line in host.journal.borrow().iter() {
        let symbol = line.split(' ').nth(2).unwrap_or_default();
        if !line.starts_with("launch ")
            || !(symbol.contains("mimo_ga_fp8") || symbol.contains("_ga_fp8_"))
            || symbol.contains("merge")
        {
            continue;
        }
        let k_plane = (0..RANKS)
            .find_map(|rank| {
                let pool = format!(
                    "raw: {} }}, offset: ",
                    (1_u64 << 32) + 100 + u64::from(rank)
                );
                let at = line.find(&pool)? + pool.len();
                line[at..].split(',').next()?.parse::<u64>().ok()
            })
            .unwrap_or_else(|| panic!("{symbol} binds no pool plane"));
        let ordinal = k_plane / layer_bytes;
        assert_eq!(k_plane % layer_bytes, 0, "{symbol} binds a V plane first");
        let [k, v] = DISTINCT_SCALES[ordinal as usize];
        let pair = format!("ScalarU32({}), ScalarU32({})", k.to_bits(), v.to_bits());
        assert!(
            line.contains(&pair),
            "{symbol} layer {ordinal} lacks its scales"
        );
        seen.entry(symbol.to_owned()).or_default().insert(ordinal);
    }
    assert_eq!(
        seen.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "chop_attn_mimo_ga_fp8_decode_rope_table_v1",
            "chop_attn_mimo_ga_fp8_decode_rows_shared_v1",
            "chop_attn_mimo_ga_fp8_prefill_direct_v1",
            "chop_row_rope_mimo_ga_fp8_table_p64_pos32_v1",
        ]
    );
    for (symbol, ordinals) in &seen {
        assert_eq!(ordinals.len(), GA_LAYERS as usize, "{symbol}");
    }
}

/// Rebuilt data-only tables must reach launches through create, without the
/// scale override used by the independent argument-wiring test above.
#[test]
fn compiled_fp8_layouts_select_their_scales_at_create() {
    let mut tables = BTreeSet::new();
    for tp in model_mimo26_contract::geometry::GEOMETRIES
        .iter()
        .flat_map(|geometry| [2, 4].map(|ranks| Tp::of_geometry(geometry, ranks).unwrap()))
    {
        let formats = model_mimo26_contract::serving::kv_formats(tp);
        for format in formats.into_iter().filter(|format| format.fp8()) {
            tables.insert((tp.geometry.id, format.layout_id));
            let host = target_on_format(tp, format, None);
            let layer_bytes = format.page_bytes / tp.geometry.global_layers() as u64;
            let mut seen = BTreeSet::new();
            for line in host
                .journal
                .borrow()
                .iter()
                .filter(|line| line.starts_with("launch "))
            {
                let symbol = line.split(' ').nth(2).unwrap();
                if !(symbol.contains("mimo_ga_fp8") || symbol.contains("_ga_fp8_"))
                    || symbol.contains("merge")
                {
                    continue;
                }
                let k_plane = (0..tp.ranks())
                    .find_map(|rank| {
                        let pool = format!(
                            "raw: {} }}, offset: ",
                            (1_u64 << 32) + 100 + u64::from(rank)
                        );
                        let at = line.find(&pool)? + pool.len();
                        line[at..].split(',').next()?.parse::<u64>().ok()
                    })
                    .expect("K plane");
                assert_eq!(k_plane % layer_bytes, 0);
                let ordinal = (k_plane / layer_bytes) as usize;
                let [k, v] = format.scales[ordinal];
                assert!(
                    line.contains(&format!(
                        "ScalarU32({}), ScalarU32({})",
                        k.to_bits(),
                        v.to_bits()
                    )),
                    "{symbol} layer {ordinal}"
                );
                seen.insert(ordinal);
            }
            assert_eq!(seen.len(), tp.geometry.global_layers());
        }
    }
    assert_eq!(
        tables.len(),
        model_mimo26_contract::serving::COMPILED_FP8_KV.len()
    );
    assert!(!tables.is_empty(), "no compiled FP8 tables exercised");
}

/// Every launch of the target, MTP and DFlash scenarios at every rank count
/// stays within its entry's declared threads. The bundle-owned kernels that
/// index one vector per thread or return early on another block size (the
/// ring kernels and the DFlash cache write) launch exactly their declared
/// threads, so a planner and `build.rs` cannot drift apart silently.
#[test]
fn launch_blocks_match_their_entries_at_every_topology() {
    if journal_switch_set() {
        return;
    }
    let field = |line: &str, name: &str| -> u32 {
        let at = line
            .find(name)
            .unwrap_or_else(|| panic!("{name} in {line}"))
            + name.len();
        line[at..]
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let exact = |symbol: &str| {
        crate::CUBINS
            .iter()
            .any(|cubin| cubin.name.ends_with("_ring") && cubin.symbols.contains(&symbol))
            || symbol.ends_with("_dflash_cache_write_v1")
    };
    for tp in Tp::SUPPORTED {
        let mut checked = BTreeSet::new();
        for host in [target_at(tp), drafter_at(tp, false), drafter_at(tp, true)] {
            for line in host
                .journal
                .borrow()
                .iter()
                .filter(|l| l.starts_with("launch "))
            {
                let symbol = line.split(' ').nth(2).unwrap();
                let threads =
                    field(line, "block_x: ") * field(line, "block_y: ") * field(line, "block_z: ");
                let declared = crate::topology::entry_threads(symbol)
                    .unwrap_or_else(|| panic!("{symbol} is not embedded"));
                assert!(threads <= declared, "{symbol}: {threads} > {declared}");
                if exact(symbol) {
                    assert_eq!(threads, declared, "{symbol}");
                    checked.insert(symbol.to_owned());
                }
            }
        }
        for name in ["ring_backup", "dflash_cache_write"] {
            let symbol = crate::topology::bundle_symbol(tp, name);
            assert!(checked.contains(&symbol), "{tp:?} never launched {symbol}");
        }
    }
}

#[test]
fn every_geometry_topology_runs_the_fake_host_journal_and_lifecycle() {
    geometry_journals(&[2, 4]);
}

#[test]
fn every_geometry_tp2_runs_the_fake_host_journal_and_lifecycle() {
    geometry_journals(&[2]);
}

fn geometry_journals(rank_counts: &[u32]) {
    for geometry in model_mimo26_contract::geometry::GEOMETRIES {
        for &ranks in rank_counts {
            let tp = Tp::of_geometry(geometry, ranks).unwrap();
            for (scenario, host) in [
                ("target", target_on(tp, false, None)),
                ("mtp", drafter_on(tp, false, false)),
                ("dflash", drafter_on(tp, true, false)),
            ] {
                for symbol in host.symbol_names.borrow().values() {
                    assert!(
                        crate::CUBINS
                            .iter()
                            .any(|cubin| crate::topology::cubin_serves(cubin.name, tp)
                                && cubin.symbols.contains(&symbol.as_str())),
                        "{tp:?}: {symbol}"
                    );
                }
                if let Some(dir) = std::env::var_os("CHOPINFER_MIMO26_GEOMETRY_JOURNAL_DIR") {
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir).unwrap();
                    std::fs::write(
                        dir.join(format!(
                            "{}-{scenario}.journal",
                            crate::topology::topology_id(tp)
                        )),
                        host.journal.borrow().join("\n"),
                    )
                    .unwrap();
                }
            }
        }
    }
}

#[test]
fn fp8_drafter_journals_run_at_every_topology() {
    assert!(
        !journal_switch_set(),
        "preservation capture requires default journal switches"
    );
    for tp in crate::topology::all() {
        for dflash in [false, true] {
            let host = drafter_on(tp, dflash, true);
            assert!(!host.allocation_ranks.borrow().is_empty());
            assert!(
                host.journal
                    .borrow()
                    .iter()
                    .any(|line| line.starts_with("launch "))
            );
        }
    }
}

#[test]
fn lifecycle_failure_boundaries_preserve_drafter_and_ledger_effect_order() {
    for tp in [Tp::TP2, Tp::TP4] {
        for dflash in [false, true] {
            for cancel in [false, true] {
                let host = Rc::new(FakeHost::at(tp));
                host.speculation.set(Some((7, 32)));
                host.dflash.set(dflash);
                let mut instance = api()
                    .create(
                        HostServicesTableV1::new(Rc::clone(&host)),
                        &configuration_for_at(tp, host.kv(), true),
                    )
                    .unwrap();
                instance
                    .execute(
                        &drafted(0, vec![40, 41, 42], false),
                        &mut DraftSink::default(),
                    )
                    .unwrap();
                instance.commit().unwrap();
                *host.readback_tokens.borrow_mut() = vec![43, 44, 45, 46];
                instance
                    .execute(&drafted(3, vec![43], true), &mut DraftSink::default())
                    .unwrap();
                assert_eq!(
                    crate::instance::lifecycle_test_state(instance.handle(), 7),
                    (true, Some(3), false)
                );
                let before = host.device_copies.borrow().len();
                if cancel {
                    // MTP restores all drafter ranks before the first target cell.
                    // DFlash cancels bookkeeping without a device copy.
                    let copies = if dflash { 0 } else { tp.ranks() };
                    host.fail_copy.set(Some(copies));
                    assert_eq!(
                        instance.cancel(),
                        Err(BundleAbiError::Status(BundleStatusV1::HostError))
                    );
                    assert_eq!(host.fail_copy.get(), None);
                    let restored = host.device_copies.borrow()[before..].to_vec();
                    assert_eq!(restored.len(), copies as usize);
                    let ring_bytes = 32
                        * tp.kv_heads(false)
                        * u64::from(tp.geometry.window())
                        * (tp.qk_dim(false) + tp.value_dim(false))
                        * 2;
                    assert!(
                        restored
                            .iter()
                            .all(|(source, destination)| source.length == ring_bytes
                                && destination.length == ring_bytes
                                && source.buffer != destination.buffer)
                    );
                    // Target restoration failed, so ledger cancellation has not run.
                    assert_eq!(
                        crate::instance::lifecycle_test_state(instance.handle(), 7),
                        (true, Some(3), true)
                    );
                } else {
                    // A verify commit has no staged prefill history copies. Fail
                    // the first rejected-tail launch after drafter and ledger apply.
                    host.fail_launch.set(Some(0));
                    assert_eq!(
                        instance.commit_accepted(&[kept(7, 2)]),
                        Err(BundleAbiError::Status(BundleStatusV1::HostError))
                    );
                    assert_eq!(host.fail_launch.get(), None);
                    assert_eq!(
                        crate::instance::lifecycle_test_state(instance.handle(), 7),
                        (false, Some(5), true)
                    );
                }
                assert!(instance.is_poisoned());
                let journal = host.journal.borrow().clone();
                assert_eq!(instance.cancel(), Err(BundleAbiError::Poisoned));
                assert_eq!(*host.journal.borrow(), journal);
            }
        }
    }
}

fn prefill_history_failure_keeps_ledger_open(accepted: bool) {
    for tp in [Tp::TP2, Tp::TP4] {
        for dflash in [false, true] {
            let host = Rc::new(FakeHost::at(tp));
            host.speculation.set(Some((7, 32)));
            host.dflash.set(dflash);
            let mut instance = api()
                .create(
                    HostServicesTableV1::new(Rc::clone(&host)),
                    &configuration_for_at(tp, host.kv(), true),
                )
                .unwrap();
            instance
                .execute(
                    &drafted(0, vec![40, 41, 42], false),
                    &mut DraftSink::default(),
                )
                .unwrap();
            if accepted {
                instance.commit().unwrap();
                *host.readback_tokens.borrow_mut() = vec![43, 44, 45, 46];
                // Wire order is verify then prefill; drafter history is prefill first.
                let mixed = BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    vec![
                        BatchItemV1::verify(7, 4, true, vec![43]),
                        BatchItemV1::new(8, false, 3, true, vec![50, 51, 52]),
                    ],
                    vec![
                        BatchAppendV1::new(3, 4, 7, vec![0], None),
                        BatchAppendV1::new(0, 3, 3, vec![1], None),
                    ],
                );
                instance
                    .execute(&mixed.encode().unwrap(), &mut DraftSink::default())
                    .unwrap();
            }
            let request = if accepted { 8 } else { 7 };
            assert_eq!(
                crate::instance::lifecycle_test_state(instance.handle(), request),
                (true, Some(0), false)
            );
            let before = host.device_copies.borrow().len();
            host.fail_copy.set(Some(0));
            let result = if accepted {
                instance.commit_accepted(&[kept(7, 2)])
            } else {
                instance.commit()
            };
            assert_eq!(
                result,
                Err(BundleAbiError::Status(BundleStatusV1::HostError))
            );
            assert_eq!(
                host.fail_copy.get(),
                None,
                "prefill-history fault was reached"
            );
            assert_eq!(host.device_copies.borrow().len(), before);
            assert_eq!(
                crate::instance::lifecycle_test_state(instance.handle(), request),
                (true, Some(0), false),
                "ledger apply must follow drafter history copies"
            );
            if accepted {
                assert_eq!(
                    crate::instance::lifecycle_test_state(instance.handle(), 7),
                    (true, Some(3), false)
                );
            }
            assert!(instance.is_poisoned());
            let journal = host.journal.borrow().clone();
            assert_eq!(instance.commit(), Err(BundleAbiError::Poisoned));
            assert_eq!(*host.journal.borrow(), journal);
        }
    }
}

#[test]
fn plain_commit_history_failure_precedes_ledger_apply() {
    prefill_history_failure_keeps_ledger_open(false);
}

#[test]
fn accepted_commit_history_failure_precedes_ledger_apply() {
    prefill_history_failure_keeps_ledger_open(true);
}
