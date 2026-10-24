//! Fake-host tests of the packages of the mixed representative layouts,
//! each from a synthetic config of its class mix: create acquires the
//! layout's rank images (checked against `direct::checkpoint_layout`), its
//! precision component and the format cubins, and decode and verify steps
//! launch each chained projection and lowered projection in the layout's
//! format. On the MTP KV format, a prompt chunk and a drafted step launch
//! the MTP drafter's experts in the layout's MTP format (MXFP8, BF16 or
//! block FP8). Every journal is pinned per layout.

use super::*;
use crate::config::tests::options_of;
use model_glm5next_contract::quantization::{
    Encoding, FP8_NVFP4_ROUTED, Layout, NVFP4_DENSE_ROUTED, NVFP4_ROUTED,
    test_support::{config_of, extra_checkpoints},
};

/// One mixed layout's package and its pinned journals.
pub(super) struct Mix {
    pub(super) layout: &'static Layout,
    config: fn() -> Vec<u8>,
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

/// Each mix's journals; create, decode, verify and drafted re-pinned when the shared
/// gate/up overlap became the default (the side streams and events; the
/// shared gate/ups on them, proven the one-stream program's by
/// `fake_host_tests::shared_overlap::mixed_layouts_overlap_alike_*`). Every
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
pub(super) const ROUTED: Mix = Mix {
    layout: &NVFP4_ROUTED,
    config: || config_of(&NVFP4_ROUTED),
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
    create: (2938, 0xcde7_e59f_164e_ebe2),
    decode: (4296, 0xc700_ff51_1444_12ca),
    verify: (4296, 0x26af_6f1a_cd65_8e92),
    prefill: (6192, 0x0d09_4c3a_bdbc_49cc),
    drafted: (4720, 0x6911_5ce0_cc56_da43),
};

pub(super) const DENSE_ROUTED: Mix = Mix {
    layout: &NVFP4_DENSE_ROUTED,
    config: || config_of(&NVFP4_DENSE_ROUTED),
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
    create: (2978, 0xaaeb_9b6d_e283_0a96),
    decode: (4296, 0x08eb_cfa7_0d39_73b8),
    verify: (4296, 0x8781_dce1_7fde_5f5a),
    prefill: (6192, 0xe91f_6fb6_bc28_3952),
    drafted: (4720, 0x0c36_7f7e_a707_525f),
};

/// Block FP8 everywhere but the main layers' routed experts, which are
/// NVFP4, from a synthetic config of that mix: its dense MLPs and shared
/// experts run the block-FP8 chained projections beside the NVFP4 experts.
pub(super) const FP8_ROUTED: Mix = Mix {
    layout: &FP8_NVFP4_ROUTED,
    config: || config_of(&FP8_NVFP4_ROUTED),
    decode_symbols: &[
        "chop_moe_chain_gateup_nvfp4_e288k8_n1024_k4096_r16",
        "chop_moe_chain_down_nvfp4_e288k8_n4096_k512_r16",
        "chop_moe_chain_gateup_fp8block_dense_n6144_k4096_r16",
        "chop_moe_chain_gateup_fp8block_dense_n1024_k4096_r16",
    ],
    drafter_symbols: &[
        "chop_moe_chain_gateup_fp8block_e288k8_n1024_k4096_r16",
        "chop_moe_chain_down_fp8block_e288k8_n4096_k512_r16",
    ],
    create: (3498, 0x328d_459c_e187_80a2),
    decode: (4296, 0xef30_c9cf_123a_856a),
    verify: (4296, 0x21f7_08ff_0b8c_4408),
    prefill: (6192, 0xb27c_a292_a085_8900),
    drafted: (4720, 0xb9da_373e_de16_adf3),
};

impl Mix {
    pub(super) fn host(&self) -> Rc<Host> {
        FakeHost::with_layout((3, 32), self.layout, &(self.config)())
    }

    /// Whether the layout stores some class in block FP8, whose chained
    /// projections and projections its steps then launch.
    fn stores_fp8(&self) -> bool {
        let fp8 = Encoding::Fp8Block128;
        let l = self.layout;
        [l.projections, l.dense, l.routed, l.mtp_routed].contains(&fp8)
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
        // Block-FP8 chained projections or projections run only for a
        // layout that stores a class in block FP8.
        assert_eq!(
            lines.iter().any(|line| line.starts_with("launch ")
                && (line.contains("_fp8block_") || line.contains("chop_proj_fp8"))),
            self.stores_fp8(),
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
        FakeHost::with_layout_in((3, 32), self.layout, &(self.config)(), format)
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
fn the_nvfp4_routed_package_creates_and_steps() {
    ROUTED.run();
}

#[test]
fn the_nvfp4_dense_routed_package_creates_and_steps() {
    DENSE_ROUTED.run();
}

#[test]
fn the_block_fp8_and_nvfp4_package_creates_and_steps() {
    FP8_ROUTED.run();
}

#[test]
fn the_block_fp8_and_nvfp4_drafter_runs_its_fp8_experts() {
    FP8_ROUTED.drafts();
}

/// Each extra checkpoint of a mixed layout pinned here
/// (`CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS`): its real plans (every rank
/// planned from its headers and prepared as the host does) create, step
/// and draft on the fake host with that layout's pinned journals. A
/// checkpoint of another layout is skipped.
#[test]
fn extra_checkpoints_create_and_step_from_their_plans() {
    for root in extra_checkpoints() {
        let layout = super::host_plan::real_layout(&root);
        let Some(mix) = [&ROUTED, &DENSE_ROUTED, &FP8_ROUTED]
            .into_iter()
            .find(|mix| mix.layout == layout)
        else {
            eprintln!(
                "{}: no journals pinned for {}: skipped",
                root.display(),
                layout.id
            );
            continue;
        };
        let planned = super::host_plan::real_planned(&root);
        for (host, run) in [
            (mix.host(), Mix::run_on as fn(&Mix, &Rc<Host>)),
            (mix.mtp_host(), Mix::drafts_on),
        ] {
            let mut host = Rc::into_inner(host).expect("a new host").into_inner();
            host.planned = planned.clone();
            run(mix, &Rc::new(Backend::new(host)));
        }
    }
}

#[test]
fn the_nvfp4_routed_drafter_runs_its_mxfp8_experts() {
    ROUTED.drafts();
}

#[test]
fn the_nvfp4_dense_routed_drafter_runs_its_bf16_experts() {
    DENSE_ROUTED.drafts();
}

/// A mixed package's rank images are refused under another layout's
/// model options: create checks each rank index against that layout's
/// plan, before it reads the precision component.
#[test]
fn a_rank_image_of_another_layout_is_refused() {
    let host = ROUTED.host();
    let error = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &record(DECODE_BATCH, &options_of(DENSE_ROUTED.layout.id, 2048)),
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
