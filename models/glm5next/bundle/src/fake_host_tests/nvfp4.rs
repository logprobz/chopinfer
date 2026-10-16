//! Fake-host tests of the NVFP4 exports' packages: create acquires the
//! layout's rank images (checked against `direct::checkpoint_layout`), its
//! precision component and the format cubins, and decode and verify steps
//! launch each chained projection and lowered projection in the layout's
//! format. On the MTP KV format, a prompt chunk and a drafted step launch
//! the MTP drafter's experts in the layout's MTP format (MXFP8 for
//! local-inference-lab, BF16 for NVIDIA). Every journal is pinned per
//! export.

use super::*;
use crate::config::tests::options_of;
use model_glm5next_contract::quantization::{Layout, NVFP4_DENSE_ROUTED, NVFP4_ROUTED};

/// One export's package and its pinned journals.
pub(super) struct Export {
    pub(super) layout: &'static Layout,
    config: &'static [u8],
    /// Symbols of the chained engine and projections that only this
    /// layout's steps launch.
    decode_symbols: &'static [&'static str],
    /// Symbols of the MTP drafter's experts in the layout's MTP format.
    drafter_symbols: &'static [&'static str],
    create: (usize, u64),
    decode: (usize, u64),
    verify: (usize, u64),
    /// A prompt chunk and a drafted step on the MTP KV format. The drafted
    /// steps were re-pinned when the drafter's head, its draft rows' gather
    /// and argmax took the run's item bucket (`program::draft_rows`). Every
    /// journal was re-pinned when commit's state copy symbol joined create
    /// (the handles), and when the phased all-reduce joined create (the
    /// handles); the decode, verify and drafted steps when it reduced their
    /// rows.
    prefill: (usize, u64),
    drafted: (usize, u64),
}

/// Each export's journals; create, decode, verify and drafted re-pinned when the shared
/// gate/up overlap became the default (the side streams and events; the
/// shared gate/ups on them, proven the one-stream program's by
/// `fake_host_tests::shared_overlap::nvfp4_exports_overlap_alike`). Every
/// journal re-pinned when the strip-chain and prefill warp-tile entries took
/// the kernels that fence their ring reads before release (their symbols,
/// and their resolve order in create).
/// Every journal re-pinned when prefill mixers from 144 rows ran the tiled
/// mix-collapse and its two symbols joined create (the handles). Create
/// re-pinned when the projections' decoded-tile cubin took the balanced
/// decoded tiles (its six symbols). Every journal re-pinned when the
/// projections' staged-tile cubin joined create (the handles) and the BF16
/// prefill projections it holds took its entries. Every journal re-pinned
/// when the stream expansion and the score compaction launched the shared
/// row copies (identity change, accuracy unaffected): their symbols and shape
/// arguments, and in create the row cubin's two symbols fewer and
/// `row_copy_sm120`'s two per rank (the handles). The prefill and drafted
/// steps re-pinned when the drafter's argmax moved onto the shared sharded
/// argmax (its partial, the pairs' two gathers and its final per rank, and
/// their bindings). Every journal re-pinned when GLM's own draft argmax
/// kernels left its row cubin (two fewer resolves per rank at create; the
/// ids issued after them renumber).
/// Create re-pinned when the KDA state pools moved before the serving KV
/// pool: the one `serving kv` line follows their allocations and zeroes,
/// every other line and handle as before.
pub(super) const LOCAL_INFERENCE_LAB: Export = Export {
    layout: &NVFP4_ROUTED,
    config: include_bytes!("../../../contract/fixtures/nvfp4-config.json"),
    decode_symbols: &[
        "chop_moe_chain_gateup_nvfp4_e288k8_n1024_k4096_r16",
        "chop_moe_chain_down_nvfp4_e288k8_n4096_k512_r16",
        "chop_moe_chain_gateup_bf16_dense_n6144_k4096_r16",
        "chop_moe_chain_gateup_bf16_dense_n1024_k4096_r16",
    ],
    drafter_symbols: &[
        "chop_moe_chain_gateup_mxfp8_e288k8_n1024_k4096_r16",
        "chop_moe_chain_down_mxfp8_e288k8_n4096_k512_r16",
    ],
    create: (2926, 0x0280645cef4c0b01),
    decode: (4296, 0x86a32217869d4d22),
    verify: (4296, 0x999e4fb722561678),
    prefill: (6192, 0x96edd3c5ac00721e),
    drafted: (4720, 0x0d5a83a35fff3307),
};

pub(super) const NVIDIA: Export = Export {
    layout: &NVFP4_DENSE_ROUTED,
    config: include_bytes!("../../../contract/fixtures/nvidia-nvfp4-config.json"),
    decode_symbols: &[
        "chop_moe_chain_gateup_nvfp4_e288k8_n1024_k4096_r16",
        "chop_moe_chain_down_nvfp4_e288k8_n4096_k512_r16",
        "chop_moe_chain_gateup_nvfp4_dense_n6144_k4096_r16",
        "chop_moe_chain_gateup_bf16_dense_n1024_k4096_r16",
        "chop_proj_nvfp4_dense_n4096_k3072_",
    ],
    drafter_symbols: &[
        "chop_moe_chain_gateup_bf16_e288k8_n1024_k4096_r16",
        "chop_moe_chain_down_bf16_e288k8_n4096_k512_r16",
    ],
    create: (2966, 0x1e8f22369091453d),
    decode: (4296, 0xaa9a6ff6c6b2b828),
    verify: (4296, 0x7ad95683d48a8800),
    prefill: (6192, 0x7cca938f717a31ce),
    drafted: (4720, 0x5ea8a1980cddb9d7),
};

impl Export {
    pub(super) fn host(&self) -> Rc<Host> {
        FakeHost::with_layout((3, 32), self.layout, self.config)
    }

    /// Create from a record naming the layout and the precision component.
    pub(super) fn create(&self, host: &Rc<Host>) -> BundleInstanceV1 {
        let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
        let record = crate::config::tests::record_naming(
            DECODE_BATCH,
            &options_of(self.layout.id, 2048),
            host.format,
            &roles,
        );
        api()
            .create(HostServicesTableV1::new(Rc::clone(host)), &record)
            .unwrap_or_else(|error| {
                panic!(
                    "{} create: {error:?} {:?}",
                    self.layout.id,
                    host.logs.borrow()
                )
            })
    }

    fn file(&self, what: &str) -> String {
        format!("glm5next-{}-{what}.txt", self.layout.id)
    }

    /// Create, then a decode and a verify step through the lowered stages.
    fn run(&self) {
        self.run_on(&self.host());
    }

    /// [`Self::run`] on `host`.
    fn run_on(&self, host: &Rc<Host>) {
        let mut instance = self.create(host);
        let journal = host.device.journal.borrow().clone();
        // The layout's format cubins load: every NVFP4 and MXFP8 chained
        // entry and the added projections resolve.
        assert!(
            journal
                .iter()
                .any(|line| line.contains("resolve") && line.contains("_nvfp4_e288k8_")),
            "{}",
            self.layout.id
        );
        check_journal(&journal, self.create, &self.file("create")).unwrap();
        let handle = instance.handle();
        crate::instance::seed(handle, 7, 0, 100);
        crate::instance::seed(handle, 8, 5, 64);
        crate::instance::run_lowered_stages_alone(handle);
        let decode = step(&[(7, 0, 100, &[11]), (8, 5, 64, &[12])]);
        let start = host.device.journal.borrow().len();
        instance.execute(&decode, &mut NoRows::default()).unwrap();
        let lines = issued(host, start);
        for symbol in self.decode_symbols {
            assert!(
                lines
                    .iter()
                    .any(|line| line.starts_with("launch ") && line.contains(symbol)),
                "{}: {symbol}",
                self.layout.id
            );
        }
        // No block-FP8 chained projection or projection runs.
        assert!(
            !lines.iter().any(|line| line.starts_with("launch ")
                && (line.contains("_fp8block_") || line.contains("chop_proj_fp8"))),
            "{}",
            self.layout.id
        );
        check_journal(&lines, self.decode, &self.file("decode-step")).unwrap();
        instance.commit().unwrap();
        let verify = step(&[(7, 0, 101, &[21, 22, 23, 24]), (8, 5, 65, &[13])]);
        let start = host.device.journal.borrow().len();
        instance.execute(&verify, &mut NoRows::default()).unwrap();
        let lines = issued(host, start);
        check_journal(&lines, self.verify, &self.file("verify-step")).unwrap();
        instance.commit_accepted(&[kept(7, 2)]).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }

    /// On the MTP KV format: a prompt chunk (the target's prefill, then the
    /// drafter's extend over the prompt) and a drafted verify step (the
    /// target's verify, the drafter's extend and its draft steps), each
    /// launching the drafter's experts in the layout's MTP format.
    fn drafts(&self) {
        self.drafts_on(&self.mtp_host());
    }

    /// A host of the package on the MTP KV format.
    pub(super) fn mtp_host(&self) -> Rc<Host> {
        let format = Shape::TP4.kv_formats()[1];
        assert!(model_glm5next_contract::contract::has_mtp_layer(&format));
        FakeHost::with_layout_in((3, 32), self.layout, self.config, format)
    }

    /// [`Self::drafts`] on `host`.
    fn drafts_on(&self, host: &Rc<Host>) {
        let mut instance = self.create(host);
        let handle = instance.handle();
        crate::instance::run_lowered_stages_alone(handle);
        // The prompt chunk's journal holds its eager launches.
        crate::instance::keep_last_chunks_eager(handle);
        let drafter_experts = |lines: &[String]| {
            for symbol in self.drafter_symbols {
                assert!(
                    lines
                        .iter()
                        .any(|line| line.starts_with("launch ") && line.contains(symbol)),
                    "{}: {symbol}",
                    self.layout.id
                );
            }
        };
        let start = host.device.journal.borrow().len();
        instance
            .execute(&prompt_chunk(7, 0, 10, true), &mut DraftSink::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        let lines = issued(host, start);
        drafter_experts(&lines);
        check_journal(&lines, self.prefill, &self.file("prefill-step")).unwrap();
        instance.commit().unwrap();
        assert_eq!(crate::instance::drafter_frontier(handle, 0), Some(9));
        let start = host.device.journal.borrow().len();
        let mut sink = DraftSink::default();
        instance
            .execute(&drafted(7, 10, 4, 21), &mut sink)
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        assert_eq!(sink.drafts.len(), 1);
        let lines = issued(host, start);
        drafter_experts(&lines);
        check_journal(&lines, self.drafted, &self.file("drafted-step")).unwrap();
        instance.commit_accepted(&[kept(7, 2)]).unwrap();
        assert_eq!(crate::instance::drafter_frontier(handle, 0), Some(10));
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn the_local_inference_lab_package_creates_and_steps() {
    LOCAL_INFERENCE_LAB.run();
}

#[test]
fn the_nvidia_package_creates_and_steps() {
    NVIDIA.run();
}

/// Each configured export's real plans (every rank planned from its
/// headers and prepared as the host does) create, step and draft on the
/// fake host with the pinned journals; skipped when its checkpoint variable
/// is unset.
#[test]
fn the_real_exports_create_and_step_from_their_plans() {
    for (export, variable) in [
        (&LOCAL_INFERENCE_LAB, "CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT"),
        (&NVIDIA, "CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT"),
    ] {
        let Some(planned) = super::host_plan::real_planned(variable) else {
            eprintln!("{variable} unset: skipped");
            continue;
        };
        for (host, run) in [
            (export.host(), Export::run_on as fn(&Export, &Rc<Host>)),
            (export.mtp_host(), Export::drafts_on),
        ] {
            let mut host = Rc::into_inner(host).expect("a new host").into_inner();
            host.planned = planned.clone();
            run(export, &Rc::new(Backend::new(host)));
        }
    }
}

#[test]
fn the_local_inference_lab_drafter_runs_its_mxfp8_experts() {
    LOCAL_INFERENCE_LAB.drafts();
}

#[test]
fn the_nvidia_drafter_runs_its_bf16_experts() {
    NVIDIA.drafts();
}

/// An NVFP4 package's rank images are refused under another layout's
/// model options: create checks each rank index against that layout's
/// plan, before it reads the precision component.
#[test]
fn a_rank_image_of_another_layout_is_refused() {
    let host = LOCAL_INFERENCE_LAB.host();
    let error = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &record(DECODE_BATCH, &options_of(NVIDIA.layout.id, 2048)),
        )
        .err()
        .unwrap();
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("differs from this build's plan")),
        "{error:?} {:?}",
        host.logs.borrow()
    );
}
