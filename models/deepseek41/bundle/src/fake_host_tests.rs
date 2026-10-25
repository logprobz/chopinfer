//! The bundle on a CPU fake host: the package's components and rank images,
//! the serving KV pool and the per-item state, served through the shared
//! fake host (`bundle_runtime::fake_host`). No kernel runs: the device
//! journal pins every host call. These tests prove create, execute, commit,
//! retire and release mechanics, not numerics.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use bundle_runtime::fake_host::dump::{Memory, Origin};
use bundle_runtime::fake_host::{Backend, FakeDevice, FakeIndex, FakePackage, Handle, NoObserver};
use bundle_runtime::geometry::Ranks;
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::serving::{INDEX_SCHEMA, RANK_IMAGE_ROLE, index_role};
use model_deepseek41_contract::weight_plan::RoutedLayout;
use runtime_bundle_abi::*;

use crate::config::{KV_LAYOUTS, KvLayout};

mod ar_engram;
mod ar_status;
mod batch_limits;
mod bf16_decode;
mod bf16_partials;
mod ceil_split;
mod conformance;
mod decode_attention;
mod decode_join;
mod device_results;
mod dspark;
mod fold_fused;
mod fold_stream;
mod head_last;
mod index_mma;
mod index_query;
pub(crate) mod indexer_workloads;
mod media;
mod moe_pipeline;
mod norm_quantize;
mod position_valid;
mod prefill_attention;
mod prefill_cache;
mod prefill_cache_capacity;
mod prefill_dma;
mod prefill_gate_journal;
mod prefill_matrix;
mod prefill_micro;
mod prefill_overlap;
mod prefill_plan_cache;
mod prefill_reduce;
#[cfg(feature = "diagnostic-taps")]
mod prefill_selected;
mod prefill_skip;
mod prefill_tma;
mod prefill_tma_scales;
mod producer_epilogues;
mod projection_reuse;
mod query_window;
mod router_gaps;
mod router_prefill;
mod scale_banks;
mod scratch_regions;
mod selected_cache;
mod shared_overlap;
mod short_prefill;
mod startup_memory;
mod verify_hc;
mod wide_hc;
use crate::shape::Shape;
use model_deepseek41_contract::state::KvStorage;

#[path = "lower/dspark/journal.rs"]
mod dspark_journal;

const KV_PAGES: u64 = 256;
const IMAGE_SLOT: u32 = 1_000_000;
const KV_SLOT: u32 = 2_000_000;
const STATE_SLOT: u32 = 3_000_000;
const INDEX_COMPONENT: u32 = 100;
const IDENTITY_COMPONENT: u32 = 200;
const VIEW_ALIGNMENT: u64 = 4096;

#[test]
fn latent_sources_configure_shared_memory_before_serving_launches() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    journal_steps(&host);
    let lines = host.device.journal.borrow();
    assert_launches_fit_configured_shared(&lines, &shape.topology());
    let mut sources = std::collections::BTreeMap::new();
    for line in lines.iter().filter(|line| line.starts_with("launch ")) {
        let symbol = line.split_whitespace().nth(2).unwrap();
        if symbol.starts_with("chop_latent_sources_")
            && !symbol.contains("_merge_")
            && !symbol.contains("_cache_decode_control0_")
        {
            let shared = if symbol.ends_with("_reuse2_v1") {
                85504
            } else if symbol.contains("_cached_profile0_") {
                73984
            } else {
                83200
            };
            assert!(
                line.contains(&format!("shared_memory_bytes: {shared},")),
                "{line}"
            );
            sources.insert(symbol, shared);
        }
    }
    assert!(!sources.is_empty());
    let mut configured = 0;
    for line in lines.iter().filter(|line| line.starts_with("resolve ")) {
        let words = line.split_whitespace().collect::<Vec<_>>();
        if let Some(shared) = sources.get(words[2]) {
            let expected = format!("configure {} {shared}", words[1]);
            assert!(
                lines.contains(&expected),
                "missing {expected} for {}",
                words[2]
            );
            configured += 1;
        }
    }
    assert_eq!(configured, sources.len() * shape.ranks() as usize);
}

#[test]
fn exact_f32_classifies_the_head_once_and_splits_decode_without_rescanning_weights() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    journal_steps(&host);
    let journal = host.device.journal.borrow();
    let launches: Vec<_> = journal
        .iter()
        .filter(|line| line.starts_with("launch "))
        .collect();
    let head: Vec<_> = launches
        .iter()
        .filter(|line| line.contains("_head_f32_m"))
        .collect();
    assert_eq!(
        head.iter()
            .filter(|line| line.contains("_split_weights_v2"))
            .count(),
        shape.ranks() as usize
    );
    assert!(head.iter().any(|line| line.contains("_cached_decode_v1")));
    assert!(head.iter().any(|line| line.contains("_cached_reduce_v1")));
    // This lifecycle gathers only a few head rows; its longer prompt
    // projections exercise the fused path before gathering the head.
    assert!(
        launches
            .iter()
            .any(|line| line.contains("_split_prepare_v2"))
    );
    for line in &head {
        if line.contains("_split_weights_v2") {
            continue;
        }
        let (_, rows) = line.rsplit_once("U32(").expect("head row argument");
        let rows: u32 = rows.split_once(')').unwrap().0.parse().unwrap();
        assert_eq!(line.contains("_cached"), rows <= 32, "{line}");
        assert_eq!(
            line.contains("_split"),
            (33..=256).contains(&rows),
            "{line}"
        );
    }
    // The create scans precede the first decode projection, and none can
    // be captured or repeated by a graph replay.
    let first_decode = launches
        .iter()
        .position(|line| line.contains("_cached_decode_v1") || line.contains("_split_v2"))
        .unwrap();
    assert!(
        launches[first_decode..]
            .iter()
            .all(|line| !line.contains("_split_weights_v2"))
    );
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn exact_f32_prefill_router_covers_its_row_range_and_preserves_small_rows() {
    let mut cases: Vec<_> = (320..=2496)
        .step_by(128)
        .map(|rows| (rows, "_blocked_m32_v1"))
        .collect();
    cases.extend([
        (319, ""),
        (321, ""),
        (2497, ""),
        (512, "_blocked_m32_v1"),
        (2048, "_blocked_m32_v1"),
        (4094, ""),
        (4095, "_tiles_n64_v1"),
        (4096, "_blocked_m32_v1"),
        (4097, ""),
        (8192, "_wide_m32_n64_v1"),
    ]);
    for (rows, _) in cases {
        // At four ranks every prefill count from 320 uses the independent
        // row certificates, including packed steps between measured counts.
        let certified = (320..=8192).contains(&rows);
        // At 8192 rows the router runs in the MoE-input norm instead
        // (`sm120::router_norm`), with no prepare.
        let in_norm = rows == 8192;
        let suffix = if in_norm {
            "_certified_m32_f64_normed_v1"
        } else if certified {
            "_zero_m32_v1"
        } else if (257..=319).contains(&rows) {
            "_tiles_n32_v1"
        } else {
            ""
        };
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 1);
        step(
            &mut instance,
            &prompt(7, 0, 0, rows, true),
            &mut Views::default(),
        );
        let journal = host.device.journal.borrow();
        assert_launches_fit_configured_shared(&journal, &shape.topology());
        let router = "chop_deepseek41_dense_f32_m8192_n384_k5120_v1";
        let expected = format!("{router}{suffix}");
        let calls: Vec<_> = journal
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(router))
            .collect();
        assert!(
            calls
                .iter()
                .any(|line| line.contains(&format!("{expected} LaunchConfig"))),
            "rows={rows}"
        );
        assert!(
            calls
                .iter()
                .all(|line| !line.contains("_tiles_n") || !line.contains("_weights"))
        );
        assert_eq!(
            calls.iter().any(|line| line.contains("_tiles_n")),
            (257..=319).contains(&rows)
        );
        assert!(calls.iter().all(|line| !line.contains("_blocked_m")));
        assert!(calls.iter().all(|line| !line.contains("_wide_m")));
        assert_eq!(
            calls.iter().any(|line| line.contains("_zero_prepare_v1")),
            certified && !in_norm,
            "rows={rows}"
        );
        drop(journal);
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn runtime_candidate_pool_reaches_its_separate_producer() {
    for kv in KV_LAYOUTS {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, kv);
        let mut record = crate::config::tests::record(&shape, kv, 1, false);
        let at = record
            .windows(MODEL_OPTIONS_TAG_V1.len())
            .position(|w| w == MODEL_OPTIONS_TAG_V1)
            .unwrap();
        record.truncate(at);
        let options = br#"{"text_config.index_topk":256,"text_config.candidate_topk_blocks":1024}"#;
        record.extend_from_slice(MODEL_OPTIONS_TAG_V1);
        record.extend_from_slice(&(options.len() as u32).to_le_bytes());
        record.extend_from_slice(options);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
        instance
            .execute(&prompt(11, 0, 0, 70, true), &mut Views::default())
            .unwrap();
        let lines = host.device.journal.borrow();
        let producers: Vec<_> = lines
            .iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && line.contains("chop_candidate_block_ids_")
                    && line.contains("_gather_v3")
            })
            .collect();
        assert_eq!(producers.len(), 16);
        for suffix in ["_valid_v1", "_valid_merge_v1"] {
            assert_eq!(
                producers
                    .iter()
                    .filter(|line| line.contains(suffix))
                    .count(),
                8
            );
        }
        for line in producers {
            assert!(line.contains("_runtime_gather_v3"), "{line}");
            assert!(line.contains("U32(1024),"), "{line}");
        }
        drop(lines);
        instance.cancel().unwrap();
        instance.destroy().unwrap();
    }
}

#[test]
fn runtime_index_topk_reaches_dense_and_compact_launches() {
    for kv in KV_LAYOUTS {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, kv);
        let mut record = crate::config::tests::record(&shape, kv, 1, false);
        let at = record
            .windows(MODEL_OPTIONS_TAG_V1.len())
            .position(|w| w == MODEL_OPTIONS_TAG_V1)
            .unwrap();
        record.truncate(at);
        let options = br#"{"text_config.index_topk":256}"#;
        record.extend_from_slice(MODEL_OPTIONS_TAG_V1);
        record.extend_from_slice(&(options.len() as u32).to_le_bytes());
        record.extend_from_slice(options);
        let acquired = host.acquired.get();
        assert!(matches!(
            api().create(HostServicesTableV1::new(Rc::clone(&host)), &record),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
        ));
        assert_eq!(host.acquired.get(), acquired);
        assert!(host.logs.borrow().iter().any(|line| line.contains(
            "the create record is missing resolved model option text_config.candidate_topk_blocks"
        )));
        record.truncate(at);
        let options = br#"{"text_config.index_topk":256,"text_config.candidate_topk_blocks":2048}"#;
        record.extend_from_slice(MODEL_OPTIONS_TAG_V1);
        record.extend_from_slice(&(options.len() as u32).to_le_bytes());
        record.extend_from_slice(options);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
        let mut views = Views::default();
        instance
            .execute(&prompt(11, 0, 0, 70, true), &mut views)
            .unwrap();
        let lines = host.device.journal.borrow();
        for family in ["chop_position_indexer_", "chop_position_blocks_"] {
            let selections: Vec<_> = lines
                .iter()
                .filter(|line| {
                    line.starts_with("launch ")
                        && line.contains(family)
                        && line.contains("_select_v3")
                })
                .collect();
            assert!(!selections.is_empty(), "{family}");
            for line in selections {
                assert!(line.contains("U32(256),"), "{line}");
            }
        }
        drop(lines);
        instance.cancel().unwrap();
        instance.destroy().unwrap();
    }
}

/// The fake host of one package topology and KV layout.
pub(crate) struct FakeHost {
    pub device: FakeDevice,
    pub logs: RefCell<Vec<String>>,
    shape: Shape,
    kv: KvLayout,
    /// The host's speculation: (drafts, verify tokens), none when off.
    speculation: Cell<Option<(u32, u32)>>,
    /// The per-item state bytes create asked for.
    state_bytes: Cell<u64>,
    kv_pages: Cell<u64>,
    drafter: Cell<bool>,
    /// Every upload's bytes by its destination, which a download of the
    /// same span reads back (a snapshot's header).
    uploads: RefCell<std::collections::BTreeMap<(u64, u64), Vec<u8>>>,
    /// Bytes the next read back of their length returns instead (a device
    /// status a kernel would have written).
    injected: RefCell<Option<Vec<u8>>>,
    /// Read backs of the injected length that pass before it applies.
    injected_after: Cell<usize>,
    /// Device memory as the origins of its bytes: every device copy moves
    /// them, so a test can tell where a span's bytes came from.
    memory: RefCell<Memory>,
    /// Layout IDs that replace a view's own in every rank index.
    layouts: RefCell<BTreeMap<String, String>>,
    /// Refuse a peer copy on the source's stream, as a host from before
    /// pushes does.
    pub pull_only: Cell<bool>,
}

pub(crate) type Host = Backend<FakeHost>;

impl std::ops::Deref for FakeHost {
    type Target = FakeDevice;

    fn deref(&self) -> &FakeDevice {
        &self.device
    }
}

/// The views the program binds today, with their bytes and element type,
/// in the order a rank image lays them out. The weight plan names every
/// view once it lands; the fake host serves exactly what the lowering
/// reads.
fn views(shape: &Shape) -> Vec<(String, u64, &'static str)> {
    let g = shape.geometry;
    let d = shape.dims();
    let hidden = u64::from(d.hidden);
    let mixes = u64::from(g.hc_mix());
    let mut views = vec![(
        "embed.weight".to_owned(),
        u64::from(shape.vocab_rows()) * hidden * 2,
        "BF16",
    )];
    for layer in 0..d.layers {
        for norm in ["attn_norm", "ffn_norm"] {
            views.push((format!("layers.{layer}.{norm}.weight"), hidden * 2, "BF16"));
        }
        for (norm, width) in [
            ("q_norm", d.attention.q_rank),
            ("kv_norm", d.attention.head_dim),
        ] {
            let bytes = u64::from(width) * 2;
            views.push((format!("layers.{layer}.attn.{norm}.weight"), bytes, "BF16"));
        }
        // The rank's heads' sinks.
        views.push((
            format!("layers.{layer}.attn.attn_sink"),
            u64::from(shape.local_heads()) * 4,
            "F32",
        ));
        let experts = u64::from(d.moe.experts);
        let inter = u64::from(shape.local_inter());
        views.push((format!("layers.{layer}.ffn.gate.bias"), experts * 4, "F32"));
        views.push((
            format!("layers.{layer}.ffn.gate.bias_vl"),
            experts * 4,
            "F32",
        ));
        for (bank, elements) in [
            ("gate_up", experts * 2 * inter * hidden),
            ("down", experts * hidden * inter),
        ] {
            let name = |part: &str| format!("layers.{layer}.ffn.experts.{bank}.{part}");
            views.push((name("weight"), elements / 2, "U8"));
            views.push((name("scale"), elements / 32, "U8"));
        }
        // The dense fold projections' MXFP8 rows and scales; the indexer's
        // query projection is its index sources' alone, `wkv` its Engram
        // layers'.
        for dense in crate::ops::dense::projections(shape) {
            if dense.id == "indexer.q" && !g.shape.csa.index_sources.contains(&layer)
                || dense.id == "engram.kv" && !g.is_engram_layer(layer)
            {
                continue;
            }
            let q = dense.query;
            let codes = u64::from(q.groups) * u64::from(q.n) * u64::from(q.k);
            let name = |part: &str| format!("layers.{layer}.{}.{part}", dense.tensor);
            views.push((name("weight"), codes, "U8"));
            views.push((name("scale"), codes / 32, "U8"));
        }
        if g.is_engram_layer(layer) {
            for weight in ["q_weight", "k_weight"] {
                let bytes = u64::from(d.hc.streams) * hidden * 2;
                views.push((format!("layers.{layer}.engram.{weight}"), bytes, "BF16"));
            }
        }
        for mixer in ["hc_attn", "hc_ffn"] {
            let f32 = |name: &str, elements: u64| {
                (
                    format!("layers.{layer}.{mixer}_{name}"),
                    elements * 4,
                    "F32",
                )
            };
            views.push(f32("fn", mixes * u64::from(g.hc_width())));
            views.push(f32("base", mixes));
            views.push(f32("scale", 3));
        }
    }
    // The exact F32 projections' BF16 weights: every router, each KV
    // source's compressor and indexer key, each index source's weights.
    for layer in 0..d.layers {
        views.push((
            format!("layers.{layer}.ffn.gate.weight"),
            u64::from(d.moe.experts) * hidden * 2,
            "BF16",
        ));
    }
    for &layer in g.shape.csa.kv_sources {
        let attention = |name: &str| format!("layers.{layer}.attn.{name}.weight");
        let head_dim = u64::from(d.attention.head_dim);
        views.push((attention("compressor.norm"), head_dim * 2, "BF16"));
        let index_dim = u64::from(d.indexer.dim);
        views.push((attention("indexer.k_norm"), index_dim * 2, "BF16"));
        views.push((attention("compressor.wkv"), head_dim * hidden * 2, "BF16"));
        if g.compress_ratio(layer) > 1 {
            views.push((attention("compressor.wgate"), head_dim * hidden * 2, "BF16"));
        }
        views.push((attention("indexer.wk"), index_dim * head_dim * 2, "BF16"));
    }
    for &layer in g.shape.csa.index_sources {
        views.push((
            format!("layers.{layer}.attn.indexer.weights_proj.weight"),
            u64::from(d.indexer.heads) * hidden * 2,
            "BF16",
        ));
    }
    views.push(("norm.weight".into(), hidden * 2, "BF16"));
    views.push((
        "head.weight".into(),
        u64::from(shape.vocab_rows()) * hidden * 2,
        "BF16",
    ));
    views
}

/// The layout the weight plan stores view `name` of `dtype` in at
/// `shape`'s rank count: a dense projection's E4M3 codes (target, DSpark and
/// main_proj) as packed fold codes, an expert bank's MXFP4 rows, or the
/// backbone gate/up bank in the served layout ([`RoutedLayout::served`]),
/// every other view row-major.
pub(super) fn plan_layout(shape: &Shape, name: &str, dtype: &str) -> (u64, &'static str) {
    let dense = || {
        name == "mtp.0.main_proj.weight"
            || crate::ops::dense::projections(shape).iter().any(|d| {
                ["layers.", "mtp."].iter().any(|prefix| {
                    name.strip_prefix(prefix)
                        .and_then(|rest| rest.strip_suffix(&format!(".{}.weight", d.tensor)))
                        .is_some_and(|layer| layer.parse::<u32>().is_ok())
                })
            })
    };
    match dtype {
        "F32" => (4, "f32_row_major_v1"),
        "U8" if dense() => (1, kernel_core::weight_transform::mxfp8_fold::LAYOUT_ID),
        "U8" if name.contains(".ffn.experts.") => {
            let layout = if name.starts_with("layers.") && name.contains(".ffn.experts.gate_up.") {
                RoutedLayout::served(shape.geometry, shape.ranks())
            } else {
                RoutedLayout::Rows
            };
            let layouts = layout.view_layouts().unwrap();
            (1, layouts[usize::from(name.ends_with(".scale"))])
        }
        "U8" => (1, "row-major"),
        _ => (2, "bf16_row_major_v1"),
    }
}

/// Rank `rank`'s index and image bytes: each view in the layout the weight
/// plan stores it in (`plan_layout`), unless `layouts` names its layout.
fn index(shape: &Shape, rank: u32, layouts: &BTreeMap<String, String>) -> (Vec<u8>, u64) {
    let mut index = FakeIndex::new(INDEX_SCHEMA, rank);
    for (name, bytes, dtype) in views(shape) {
        let (element, layout) = plan_layout(shape, &name, dtype);
        let layout = layouts.get(&name).map_or(layout, String::as_str);
        index.push(
            name,
            bytes,
            dtype,
            vec![bytes / element],
            layout,
            VIEW_ALIGNMENT,
        );
    }
    let image_bytes = index.end(VIEW_ALIGNMENT);
    (index.bytes(image_bytes), image_bytes)
}

impl FakeHost {
    pub(crate) fn new(shape: Shape, kv: KvLayout) -> Rc<Host> {
        Rc::new(Backend::new(FakeHost {
            device: FakeDevice::default(),
            logs: RefCell::new(Vec::new()),
            shape,
            kv,
            speculation: Cell::new(None),
            state_bytes: Cell::new(0),
            kv_pages: Cell::new(KV_PAGES),
            drafter: Cell::new(false),
            uploads: RefCell::new(std::collections::BTreeMap::new()),
            injected: RefCell::new(None),
            injected_after: Cell::new(0),
            memory: RefCell::new(Memory::default()),
            layouts: RefCell::new(BTreeMap::new()),
            pull_only: Cell::new(false),
        }))
    }

    /// The rank's index bytes and image size, built once per test process
    /// for each shape, rank, layout override set and drafter. Create reads
    /// each index several times, and rebuilding it took a third of an
    /// allocation sweep's CPU time (2026-10-21).
    fn rank_index(&self, rank: u32) -> (Vec<u8>, u64) {
        type Key = (Shape, u32, BTreeMap<String, String>, bool);
        type Built = std::sync::Arc<std::sync::OnceLock<(Vec<u8>, u64)>>;
        static INDEXES: std::sync::Mutex<Vec<(Key, Built)>> = std::sync::Mutex::new(Vec::new());
        let key = (
            self.shape,
            rank,
            self.layouts.borrow().clone(),
            self.drafter.get(),
        );
        let built = {
            let mut indexes = INDEXES.lock().unwrap();
            let at = match indexes.iter().position(|(k, _)| *k == key) {
                Some(at) => at,
                None => {
                    indexes.push((key.clone(), Built::default()));
                    indexes.len() - 1
                }
            };
            std::sync::Arc::clone(&indexes[at].1)
        };
        built
            .get_or_init(|| {
                let (shape, rank, layouts, drafter) = &key;
                if *drafter {
                    dspark::index(shape, *rank, layouts)
                } else {
                    index(shape, *rank, layouts)
                }
            })
            .clone()
    }

    fn page_bytes(&self) -> u64 {
        model_deepseek41_contract::state::request_state(self.shape.geometry, self.kv.storage, false)
            .page_bytes()
    }
}

impl FakePackage for FakeHost {
    type Observer = NoObserver;

    fn device(&self) -> &FakeDevice {
        &self.device
    }
    fn check_launch(&self, symbol: &str, config: LaunchConfigV1) {
        if symbol.starts_with("chop_latent_prefill_")
            || ((symbol.starts_with("chop_deepseek41_mxfp8_stream_")
                || symbol.starts_with("chop_deepseek41_mxfp8_direct_"))
                && symbol.ends_with("_prepare_v1"))
        {
            assert_eq!(
                config.flags, 0,
                "operand preparation must serialize normally"
            );
        } else if config.flags == 0
            && symbol.contains("gate")
            && self
                .device
                .journal
                .borrow()
                .last()
                .is_some_and(|line| line.starts_with("wait "))
        {
            // The routed gate directly after an r2 shared-expert join
            // waits for the shared expert in full.
        } else {
            self.device.check_launch_for(symbol, config);
        }
    }
    fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
        crate::CUBINS.iter().map(|cubin| (cubin.name, cubin.bytes))
    }
    fn log(&self, message: &[u8]) {
        self.logs
            .borrow_mut()
            .push(String::from_utf8_lossy(message).into_owned());
    }
    fn on_acquire(&self, _handle: Handle) {}
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        let mut injected = self.injected.borrow_mut();
        if injected
            .as_ref()
            .is_some_and(|bytes| bytes.len() == destination.len())
        {
            match self.injected_after.get() {
                0 => {
                    destination.copy_from_slice(&injected.take().unwrap_or_default());
                    return Ok(());
                }
                after => self.injected_after.set(after - 1),
            }
        }
        drop(injected);
        match self
            .uploads
            .borrow()
            .get(&(source.buffer.raw, source.offset))
        {
            Some(bytes) if bytes.len() == destination.len() => destination.copy_from_slice(bytes),
            _ => destination.fill(0),
        }
        Ok(())
    }
    fn on_upload(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.uploads.borrow_mut().insert(
            (destination.buffer.raw, destination.offset),
            source.to_vec(),
        );
        Ok(())
    }
    fn on_copy(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.memory.borrow_mut().copy(source, destination);
        Ok(())
    }
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        // The destination's stream pulls, the source's pushes.
        let issuer = self.device.stream_ranks.borrow()[&stream.slot()];
        let (to, from) = (destination_context.slot() - 1, source_context.slot() - 1);
        assert!(
            issuer == to || issuer == from,
            "a peer copy on a third rank's stream"
        );
        if issuer != to && self.pull_only.get() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if self.device.fail_copy.fails() {
            return Err(BundleStatusV1::HostError);
        }
        assert_eq!(destination.length, source.length);
        self.check_span(source);
        self.check_span(destination);
        self.device.note(format!(
            "peer-copy {} {} {} {source:?} {destination:?}",
            stream.slot(),
            source_context.slot(),
            destination_context.slot()
        ));
        self.memory.borrow_mut().copy(source, destination);
        Ok(())
    }
    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        assert_eq!(role, RANK_IMAGE_ROLE.as_bytes());
        Ok(BundleCompiledImageHandleV1::from_parts(1, self.id()))
    }
    fn upload_rank_arena(
        &self,
        _context: BundleContextHandleV1,
        _pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        _image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        self.note(format!("upload arena {rank}"));
        Ok(BundleRankArenaHandleV1::from_parts(1, self.id()))
    }
    fn resolve_compiled_image_view(
        &self,
        _image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        rank: u32,
        _view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        let (index, mut bytes) = self.rank_index(rank);
        if self.drafter.get() {
            let index = bundle_runtime::index::parse(&index, rank, INDEX_SCHEMA).unwrap();
            bytes -= index
                .views
                .iter()
                .filter(|(name, _)| name.contains(".engram.embed."))
                .map(|(_, view)| view.bytes.next_multiple_of(VIEW_ALIGNMENT))
                .sum::<u64>();
        }
        let slot = IMAGE_SLOT + rank;
        self.device.extents.borrow_mut().insert(slot, bytes);
        Ok(CompiledImageSpanV1::new(
            rank,
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, slot), 0, bytes),
            VIEW_ALIGNMENT,
        ))
    }
    fn open_authenticated_component(
        &self,
        role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        let ranks = self.shape.ranks();
        let identity = format!("chop.compiled-image-identity.v1/{RANK_IMAGE_ROLE}/");
        for rank in 0..ranks {
            if role == index_role(rank).as_bytes() {
                return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                    1,
                    INDEX_COMPONENT + rank,
                ));
            }
            if role == format!("{identity}{rank}").as_bytes() {
                return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                    1,
                    IDENTITY_COMPONENT + rank,
                ));
            }
        }
        Err(BundleStatusV1::NotFound)
    }
    fn component(&self, slot: u32) -> (String, Vec<u8>) {
        if slot >= IDENTITY_COMPONENT {
            let rank = slot - IDENTITY_COMPONENT;
            let (_, bytes) = self.rank_index(rank);
            let identity = serde_json::json!({"byte_length": bytes, "sha256": "a".repeat(64)});
            return (
                format!("chop.compiled-image-identity.v1/{RANK_IMAGE_ROLE}/{rank}"),
                serde_json::to_vec(&identity).unwrap(),
            );
        }
        let rank = slot - INDEX_COMPONENT;
        (index_role(rank), self.rank_index(rank).0)
    }
    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64> {
        assert_eq!(layout_id, self.kv.layout_id);
        self.note("serving kv".into());
        Ok(self.kv_pages.get())
    }
    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let slot = KV_SLOT + rank;
        let bytes = self.kv_pages.get() * self.page_bytes();
        self.device.extents.borrow_mut().insert(slot, bytes);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, slot),
            0,
            bytes,
        ))
    }
    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1> {
        assert_eq!(
            request.state_id,
            model_deepseek41_contract::serving::REQUEST_STATE_ID.as_bytes()
        );
        self.state_bytes.set(request.byte_length);
        self.note(format!("state {}", request.byte_length));
        Ok(BundleStateHandleV1::from_parts(1, self.id()))
    }
    fn resolve_state_view(
        &self,
        _state: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let slot = STATE_SLOT + rank;
        let bytes = self.state_bytes.get();
        self.device.extents.borrow_mut().insert(slot, bytes);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, slot),
            0,
            bytes,
        ))
    }
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
        let mut speculation = BundleRawSpeculationV1::disabled();
        if let Some((drafts, rows)) = self.speculation.get() {
            speculation.maximum_draft_tokens = drafts;
            speculation.maximum_verify_tokens = rows;
        }
        Ok(speculation)
    }
}

pub(crate) fn api() -> BundleApiV1 {
    BundleApiV1::new_with_prefix_state_slot(crate::api::BUNDLE_DEEPSEEK41_EXTENDED_API_V1)
        .expect("bundle API")
        .with_device_results_api(crate::api::BUNDLE_DEEPSEEK41_DEVICE_RESULTS_API_V1)
        .expect("device result API")
}

/// Creates an instance serving `items` with only its lowered stages
/// (`crate::instance::PARTIAL`).
pub(crate) fn create(host: &Rc<Host>, items: u32) -> BundleInstanceV1 {
    if host.speculation.get().is_some_and(|(drafts, _)| drafts > 0) {
        host.drafter.set(true);
    }
    let record = crate::config::tests::record(&host.shape, host.kv, items, false);
    crate::instance::PARTIAL.with(|partial| partial.set(true));
    let services = if host.drafter.get() {
        HostServicesTableV1::new(Rc::new(dspark::host::MappedHost::new(Rc::clone(host))))
    } else {
        HostServicesTableV1::new(Rc::clone(host))
    };
    let created = api().create(services, &record);
    crate::instance::PARTIAL.with(|partial| partial.set(false));
    created.unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

/// A sink that keeps the logits views.
#[derive(Default)]
pub(crate) struct Views(pub(crate) Vec<OwnedLogitsViewV1>);

impl Views {
    /// The logits rows the views hold.
    pub(crate) fn rows(&self) -> u32 {
        self.0.iter().map(|view| view.raw().row_count).sum()
    }
}

impl LogitsRowSinkV1 for Views {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.0.push(view);
        Ok(())
    }
}

/// A prompt chunk of `count` tokens after `previous` of `request` in
/// `slot`, scoring its last row when `last`.
pub(crate) fn prompt(request: u64, slot: u64, previous: u64, count: u64, last: bool) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + count;
    let inputs = (previous..end).map(|token| token as u32).collect();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, false, count as u32, last, inputs)],
        vec![BatchAppendV1::new(
            previous,
            count,
            end,
            (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
            None,
        )],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

/// One decode step of `items`: (request, slot, previous tokens, token).
pub(crate) fn decode(items: &[(u64, u64, u64, u32)]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for &(request, slot, previous, token) in items {
        let end = previous + 1;
        batch.push(BatchItemV1::new(request, true, 1, true, vec![token]));
        appends.push(BatchAppendV1::new(
            previous,
            1,
            end,
            (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

/// Executes `batch` and commits it.
#[track_caller]
fn step(instance: &mut BundleInstanceV1, batch: &[u8], views: &mut Views) {
    if let Err(error) = instance.execute(batch, views) {
        panic!("{error:?}: {:?}", instance.last_execute_refusal());
    }
    instance.commit().unwrap();
}

/// The journal's lifecycle on `host`: create; a prompt prefilled in two
/// chunks, the first a whole page, and a second prompt; a decode step of
/// both that is cancelled, which invalidates the requests it touched
/// (`Cancel::Invalidate`), so both retire and start again as new requests
/// in their slots, the first in one chunk; a decode of both; retire and
/// destroy. Each scored row reaches the sink.
fn journal_steps(host: &Rc<Host>) {
    let mut instance = create(host, 4);
    let mut views = Views::default();
    step(&mut instance, &prompt(7, 0, 0, 64, false), &mut views);
    assert_eq!(views.rows(), 0);
    step(&mut instance, &prompt(7, 0, 64, 6, true), &mut views);
    assert_eq!(views.rows(), 1);
    step(&mut instance, &prompt(9, 1, 0, 3, true), &mut views);
    instance
        .execute(
            &decode(&[(7, 0, 70, 1), (9, 1, 3, 2)]),
            &mut Views::default(),
        )
        .unwrap();
    instance.cancel().unwrap();
    instance.retire_request(7).unwrap();
    instance.retire_request(9).unwrap();
    step(&mut instance, &prompt(11, 0, 0, 70, true), &mut views);
    step(&mut instance, &prompt(13, 1, 0, 3, true), &mut views);
    step(
        &mut instance,
        &decode(&[(11, 0, 70, 1), (13, 1, 3, 2)]),
        &mut views,
    );
    assert_eq!(views.rows(), 6);
    instance.retire_request(11).unwrap();
    instance.retire_request(13).unwrap();
    instance.destroy().unwrap();
}

/// The pinned journal digest of each Flash topology and KV layout. The
/// workspace includes attention partials at the full token budget; its
/// allocation and bound span lengths are part of these journals. Decode
/// combines split mixing, measured projection and expert buckets, live
/// position selection, parallel queries, native FP4 scores with exact query
/// repacking, measured weight prefetch and parallel ceil operations.
/// Prefill uses valid candidate work, exact-row TP4 folds through 64 rows,
/// staged MXFP8 tiles for full rows and measured exact F32 row buckets.
/// Qualified retained mixed64 rows use paired native-key attention.
/// Large-row dense ceil uses the qualified prefill warp counts. Prefill HC
/// row tiles add a separate entry while small-row mixer paths stay unchanged.
/// Measured F32 prefill adds eight compute handles and two modules per rank,
/// with one retained tiled entry configured for its existing shared-memory limit.
/// Shared expert scratch stays independent across every fork/join region.
/// TP4 decode and verify use the measured direct-load MoE buckets, and the
/// paired shared gate/up keeps no single packed folds.
/// The qualified warp mixer retains its full cubin identity. Loading its
/// eleven entries adds eleven resolves and twelve releases per rank;
/// C1/C8 decode changes three mixer symbols and their null stamp args.
/// The qualified input norm tiles add two resolves and three releases per
/// rank. Cached quantization replaces those C1/C8 norm tiles; rounded HC
/// uses 160-CTA producers with the same finisher and scratch. The two
/// additional modules add five resolves and seven releases per rank.
/// The qualified finish/norm module adds two resolves and three releases
/// per rank. C1/C8 replaces 77 adjacent finish/norm pairs with one exact
/// tail each, preserving all constituent arguments and capture dependencies.
/// Full mixed prefill head staging adds two immutable modules per TP4 rank,
/// eighteen resolves, four shared-memory configurations and twenty releases.
/// Every launch retains its argument list and numerical contract.
/// Native TP4 adds the selected-key cache: 24 resolves, six shared-memory
/// opt-ins, one separate cache allocation and 26 releases per rank.
/// This two-row lifecycle retains its source launches. The image lifecycle
/// selects the C1 producer and cached source; its ordered merge is unchanged.
/// Retained launches keep their argument lists and numerical contracts.
/// Decode uses the qualified four-group ordered position merge and disjoint
/// score-tail lists. Prefill and verify retain their original merge.
/// Measured router epilogues add one module. These two-item captures retain
/// their original router launches; the one-item media capture audits fusion.
/// Measured direct folds use per-operation state sized across all selected row buckets.
/// TP4 row reuse adds two modules, seventeen resolves and nineteen
/// releases per rank. This lifecycle keeps every launch and allocation;
/// only resource identities move. Larger-row projection graphs are audited
/// separately with the same loaded catalog in reference and serving modes.
/// The tiled indexer adds twelve resolves and thirteen releases per native
/// rank; these short-context fixtures retain their prior indexer launches.
/// Qualified HC row tiles and prefill mixers retain their exact operands.
/// Fused prefill heads add only their declared module resources here; their
/// active output/merge substitutions are audited in full prefill lifetimes.
/// The ordered head adds 24 resolves and 25 releases per rank. Sixteen
/// one-row heads replace three launches with two; allocations are unchanged.
/// The full-chunk query norm adds two resolves and three releases per rank.
/// These short-row lifecycles retain every launch, operand and dependency.
/// Qualified prefill scoring adds one module, eight resolves and nine
/// releases per native rank. These small-row launches remain unchanged.
/// At four ranks the grouped 8192-row down binding adds one module, two
/// resolves, one configure and three releases per rank; no short-row launch
/// uses it. The certified router rows add one module, two resolves and three
/// releases per rank. At four ranks the router projection's workspace view
/// grows from 251691008 to 251756544 bytes on its own launches; no launch
/// count, grid or allocation changes, and no short-row count selects them.
/// TP4 defaults to native MoE prefill. Against the exact expert path,
/// the audited lifecycle replaces 800 column gate/up and 800 column down
/// launches with native pack and tile launches, adds 16 resolves and 20
/// releases, and removes four shared-memory configures: 1632 lines.
/// The resident small gate adds one module, two resolves and three releases
/// per TP4 rank; no lifecycle launch or allocation changes. Qualified matrix
/// and TMA tiles add two modules, 138 resolves, ten shared-memory
/// configurations and 140 releases per TP4 rank; no short-row step launches
/// them. The fused candidate publisher adds one module, one resolve and two
/// releases per native rank, and replaces each layer-20 producer triple
/// (scores, local gather, merge) with one launch on the merge's spans. In
/// these steps of at most 2048 rows the fused ceil replaces the 800 gate/up
/// tile launches with the fused tiles and the 800 intermediate quantizer
/// launches with its fixup, and adds one module, two resolves, one configure
/// and three releases per TP4 rank. Per native TP4 rank the dedicated prefill
/// arena adds two allocations and their releases, and verify-4 head grouping
/// adds one module, 24 resolves, three configures and 25 releases; no step
/// here verifies four rows. The combined journals were audited against
/// main's: after removing exactly these resources and swaps, every other
/// operation is identical after handle mapping.
/// The early-stop selectors add one module, six resolves and seven releases
/// per native rank, and rename each prefill local selector and merge; their
/// streams, grids and arguments are unchanged. The r2 router finish module
/// adds one module, 14 resolves and 15 releases per TP4 rank; no launch or
/// allocation changes. Exact-row folds replace only qualified short prefill
/// launches. Staged scales add one module, fifteen resolves, four
/// configurations and sixteen releases per TP4 rank, and allocations are
/// unchanged. The combined journals were audited against main's and against
/// the staged-scale journals alone: the selectors and the r2 module add
/// exactly the resources and renames above, the short-row and staged-scale
/// delta is the same with or without them, and every other operation is
/// identical after handle mapping. The wide gate/up module
/// (`moe_native_entries::WIDE_CUBIN`) adds two resolves and three releases
/// per TP4 rank, and the shared-memory configure moves from the narrow
/// gate/up tile to the wide one; steps above 2048 rows launch it, so these
/// journals do not. The gate finisher loads module adds one module, 2
/// resolves and 3 releases per TP4 rank. Each was audited against the
/// combined journals before it: exactly these lines differ.
/// The one-launch HC boundary module adds ten resolves and eleven releases
/// per rank; these fixtures launch no C1 or C8 boundary.
/// The cached query norm module adds three resolves and four releases per
/// rank; these two-row decodes keep the served query norm.
/// operation is identical after handle mapping. The router certificate
/// (`router_certified`) adds one allocation and its release per rank; at
/// four ranks the router-in-norm module (`sm120::router_norm`) adds two
/// resolves and three releases per rank. No launch changes: only 8192-row
/// steps run the router in the norm.
/// Combined C1 query/KV adds one module, two resolves and 120 private
/// allocations per TP4 rank (108175360 bytes), with 40 weight initializers.
/// Full lifecycle audits retain every other operation after handle mapping;
/// each C1 query plus KV triple becomes one launch with identical operands.
/// Full-row Engram staged scales add one 68608-byte shared-memory configure
/// per TP4 rank. These short-row lifecycles retain every other line exactly.
/// The lean native module (`moe_native_entries::LEAN_CUBIN`) adds three
/// resolves and four releases per TP4 rank, and the unfused native MoE
/// steps of these lifecycles run its tiles. The scanned loop gives the
/// module's two tile resolves the scanned names and adds the two admission
/// scans' resolves and releases per rank.
/// The local C32 query module adds eight resolves and nine releases per
/// TP4 rank. These short-row lifecycles retain every launch and allocation;
/// the complete histories were compared under typed handle mapping.
/// The router's 64-expert certified projection renames the module's router
/// resolve on each TP4 rank (four lines), audited against the journals
/// before it: no other line changes. Its narrow form (no widened rows)
/// renames the same four resolves again, nothing else.
/// The two-CTA HC finish module adds six resolves and seven releases per
/// rank; these fixtures launch no Verify-6 tail.
/// The two-strip routed down module adds two resolves and three releases
/// per TP4 rank. With its resolves removed, every other line is identical
/// after typed handle mapping; these two-row decodes launch no two-strip
/// queue.
/// Each decode capture's window KV projection binds the shared expert's
/// workspace and row status and runs on the side stream beside its query
/// chain: per layer and rank, its launches change stream and those two
/// spans, and its fork and join add four event lines (audited by
/// `shared_overlap::the_window_kv_overlap_changes_only_its_projection`).
/// These lifecycles capture one Decode-2 graph: 160 lines more per rank.
/// The row-local rings (`fold_local_rows::SERVED`) add eight resolves and
/// eight releases per TP4 rank; these lifecycles launch no 16- or 32-row
/// bucket.
/// The column-block HC pair's two modules add fourteen resolves and sixteen
/// releases per rank; every other line is retained after handle mapping.
/// The cluster folds (`fold_cluster::SERVED`) add sixteen resolves, one
/// configure (WQ_a's 32-row ring) and seventeen releases per TP4 rank and
/// shift ids only.
/// The status folds (`sm120::fold_fused`) add eleven resolves, one
/// allocation (WO_a's 524288 block flag bytes) and thirteen releases per
/// TP4 rank, and move WQ_a's 32-row configure from its cluster fold to the
/// status twin; these lifecycles launch no 16- or 32-row bucket, and every
/// other line is identical after handle mapping.
/// Sharing fold state and workspace across layers removes 406 allocations
/// and their releases per TP1 rank and 484 per TP2 and TP4 rank (at TP4 the
/// dual fold's window-key operation shares as well). Packed weights,
/// initialization count, every launch and every span length stay unchanged.
/// All six journals match their predecessors after exactly those removals
/// and the state/workspace aliases and resulting handle renumbering.
/// The scratch regions (`buffers::REGIONS`) allocate each region once in
/// place of one allocation per member: ten fewer allocations and releases
/// per rank. Audited against the same lifecycles bound without
/// regions: every other line is identical once handle, symbol, stream,
/// event and graph ids are mapped, no offset or length changes, and the
/// only handles that merge are each rank's members of one region.
/// No instance loads the retired one-launch HC boundary module: ten
/// resolves and eleven releases per rank fewer, ids shifted only.
/// The status folds serve the 1-, 2- and 4-row buckets on their 16-row
/// tile. In the two-row decode of these lifecycles each layer's ten
/// launches (the WQ_a and WKV prefetch initialize/fold pairs, WQ_b's stream
/// fold, WO_a's input quantizer, prefetch pair and WO_b's F32 input
/// quantizer, and WO_b's direct fold) become six with the same operands,
/// the packed fold's weights and state replaced by the layer's weights: 640
/// launches and captured nodes fewer per TP4 lifecycle, audited line by
/// line against the 16-row status folds. No other line changes.
/// Every MXFP8 fold reads the plan's packed codes
/// (`mxfp8-fold-packed-8x128-v1`): each weight-reading fold resolve and
/// launch takes its packed entry (`_packed_v1`), and a stream, direct or
/// dual fold binds the weight's own view as its packed bank. Create no
/// longer allocates packed copies or launches their prepares (250 prepares
/// per rank; 290 fewer allocations and releases per TP1 and TP2 rank, 250
/// per TP4 rank) and zeroes each fold state span once instead (eight zero
/// lines per rank). The cluster, row-local and TMA matrix modules no longer
/// load: 62 fewer resolves and one fewer configure per TP4 rank. Audited
/// against the previous journals, with each fold launch's packed bank
/// mapped from its private copy to a view of the same length: every other
/// line is identical once handle, symbol, stream, event and graph ids are
/// mapped.
/// Flash TP4 stores the backbone gate/up bank in row tiles: the row-tiled
/// builds of its gate/up entries replace the row-major ones in place, so
/// every rank count's resolves of the column module and the TP4 launches
/// of those entries name the tiled builds. No other line changes.
/// Its codes are in quad-lane order: the quad-lane builds replace the
/// row-tiled ones the same way, renaming the same resolves and launches.
/// No other line changes.
/// Native TP4 pairs prefill attention rows at every row count, so these
/// short prompts' window, mixed32 and mixed64 attention launches (3, 6, 64
/// and 70 rows) name the loaded paired entries, with grid x halved and
/// rounded up, twice the threads and 85,504 shared bytes. Audited against
/// the previous journal: those 720 launches differ in exactly these fields,
/// and every other line, argument, stream and handle is identical.
/// Outside the cache's admitted chunks, the mixed32 and mixed64 layers then
/// fuse with shared heads: each of those 760 paired attention launches and
/// its merge become one finish launch on the same stream, with grid (rows,
/// 1, 1), 256 threads and 83,200 shared bytes. Its arguments are the
/// attention's with the partial workspace null, then the merge's sinks,
/// RoPE and output. Audited line by line; every other line is identical.
/// The small-router admission changes Decode-2 at TP4 only in this
/// lifecycle. Across 160 layer/rank groups it removes 320 launches. The
/// cached producer, fused operands, workspace shift and full dependency
/// after the shared-expert join were audited against the previous journal;
/// all other calls, captures, allocations and releases are identical.
/// TP1 and TP2 journals are unchanged in both KV layouts.
/// The small-row admission changes exactly 308 two-row HC producers in
/// each TP4 history, retaining their operands and allocations. Two existing
/// wide-fold modules per rank add 56 resolves, 12 shared-memory settings and
/// 64 releases. `lab/audit_small_row_lifecycle.py` checks every retained
/// record through typed handle bijections and rejects mutated operands,
/// launch geometry, retained bindings and missing lifecycle operations.
/// TP1 and TP2 remain byte-identical.
/// The cached small-decode norms change exactly 320 input and 160 query
/// launches in each TP4 history, only their symbol, grid and block width.
/// `lab/audit_norm_small_rows.py` verifies every other byte, the complete
/// handle tables and eight mutation controls against the prior pinned history.
/// Decode-2 selected-key caching replaces each of 160 source launches with
/// its predecode and cached source. Original source operands and all ordered
/// merges remain identical. The unchanged graph splitter moves one complete
/// five-launch KV fork/join unit from the third to the second piece per rank;
/// piece node counts change from 44/201/1288 to 45/211/1317. The audit retains
/// every other operation and allocation, and all five other journals exactly.
const FLASH_JOURNALS: [(&str, KvStorage, (usize, u64)); 6] = [
    (
        "deepseek41-flash-tp1",
        KvStorage::Native,
        (15864, 0x0ecdca5645c278de),
    ),
    (
        "deepseek41-flash-tp1",
        KvStorage::Bf16,
        (15718, 0x7b4ce841cbca7856),
    ),
    (
        "deepseek41-flash-tp2",
        KvStorage::Native,
        (31603, 0x5ac02bac440c0073),
    ),
    (
        "deepseek41-flash-tp2",
        KvStorage::Bf16,
        (31311, 0x9174be3bab2de486),
    ),
    (
        "deepseek41-flash-tp4",
        KvStorage::Native,
        // One dedicated data arena and table per rank, reserved before KV.
        // The full journal audit retains every other operation exactly.
        (77735, 0x4bc1c6a7a12d2a2b),
    ),
    (
        "deepseek41-flash-tp4",
        KvStorage::Bf16,
        (75979, 0x3a2f0a60c34638dd),
    ),
];

/// Every compiled geometry at every topology and KV layout runs the
/// journal's lifecycle, loading only the cubins that serve its geometry,
/// and releases everything it acquired; the Flash journals are pinned, and
/// hold whatever other geometries the build compiles. With `CHOPINFER_DEEPSEEK41_GEOMETRY_JOURNAL_DIR`
/// set, each journal is written there as `<topology>.<layout>.journal`.
#[test]
fn every_geometry_topology_runs_the_fake_host_journal_and_lifecycle() {
    let export =
        std::env::var_os("CHOPINFER_DEEPSEEK41_GEOMETRY_JOURNAL_DIR").map(std::path::PathBuf::from);
    let mut mismatches = Vec::new();
    for shape in crate::shape::topologies() {
        for kv in KV_LAYOUTS {
            let host = FakeHost::new(shape, kv);
            journal_steps(&host);
            let topology = shape.topology();
            assert_eq!(
                host.acquired.get(),
                host.released.get(),
                "{topology} {:?}",
                kv.storage
            );
            // Each loaded cubin serves the topology's geometry.
            for cubin in host.device.module_cubins.borrow().values() {
                assert!(
                    !cubin.starts_with("streamed_moe_") || shape.ranks() == 4,
                    "streamed candidate loaded outside TP4: {cubin}"
                );
                assert!(
                    !cubin.starts_with("position_candidates_runtime_"),
                    "default loaded {cubin}"
                );
                // The column-block pair retired the one-launch HC boundary.
                assert_ne!(*cubin, crate::sm120::hc_completion::CUBIN, "{topology}");
                assert!(crate::serves(cubin, &shape), "{topology}: {cubin}");
            }
            let lines = host.device.journal.borrow().clone();
            assert_launches_fit_configured_shared(&lines, &topology);
            assert_small_row_ceil_bindings(&lines);
            index_query::assert_query_bindings(&lines);
            let layout = format!("{:?}", kv.storage).to_lowercase();
            if let Some(dir) = &export {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(
                    dir.join(format!("{topology}.{layout}.journal")),
                    lines.join("\n"),
                )
                .unwrap();
                decode_attention::export_handles(
                    &host.device,
                    dir,
                    &format!("{topology}.{layout}.journal"),
                );
            }
            if let Some(&(_, _, expected)) = FLASH_JOURNALS
                .iter()
                .find(|(id, storage, _)| *id == topology && *storage == kv.storage)
                && let Err(observed) = bundle_runtime::fake_host::check_journal(
                    &lines,
                    expected,
                    &format!("{topology}.{layout}.journal"),
                )
            {
                mismatches.push(format!("{topology} {layout}: {observed}"));
            }
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The Flash TP4 journals with the explicit native MoE prefill switch
/// (`sm120::moe_native`, `CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL=1`): the
/// native modules load, every prefill expert projection launches the
/// native tiles within their configured shared memory, decode keeps its
/// experts, and the lifecycle releases everything. Every prefill step's
/// 40 layers launch gate/up's pack, fused ceil tiles and fixup and down's
/// pack and tiles on each rank in place of the exact expert launches and
/// the intermediate quantizer: every prefill step here has at most
/// `moe_native::FUSED_CEIL_ROWS` rows (the wrapper's unit test covers the
/// larger steps, which keep the quantizer). The explicit setting and the
/// default must produce the same pinned journals.
#[test]
fn native_moe_prefill_switch_runs_the_tp4_journal() {
    let shape = crate::shape::topologies()
        .into_iter()
        .find(|s| s.topology() == "deepseek41-flash-tp4")
        .unwrap();
    let mut mismatches = Vec::new();
    for kv in KV_LAYOUTS {
        let host = crate::sm120::moe_native::with_native(true, || {
            let host = FakeHost::new(shape, kv);
            journal_steps(&host);
            host
        });
        assert_eq!(host.acquired.get(), host.released.get(), "{:?}", kv.storage);
        assert!(
            host.device
                .module_cubins
                .borrow()
                .values()
                .any(|c| *c == crate::sm120::moe_native::CUBIN)
        );
        let lines = host.device.journal.borrow().clone();
        assert_launches_fit_configured_shared(&lines, "deepseek41-flash-tp4");
        let native = lines
            .iter()
            .filter(|l| l.starts_with("launch ") && l.contains("chop_moe_native_"))
            .count();
        // Gate/up's pack, fused tiles and fixup; down's pack and tiles. No
        // separate intermediate quantizer runs in a native step.
        assert!(native > 0 && native % 5 == 0, "{native} native launches");
        let fixups = lines
            .iter()
            .filter(|l| l.starts_with("launch ") && l.contains("chop_moe_native_ceil_fixup_"))
            .count();
        assert_eq!(fixups * 5, native);
        assert!(
            !lines
                .iter()
                .any(|l| l.starts_with("launch ") && l.contains("_moe_ceil_w3456_")),
            "a native step kept the separate intermediate quantizer"
        );
        let layout = format!("{:?}", kv.storage).to_lowercase();
        let (_, _, expected) = FLASH_JOURNALS
            .iter()
            .find(|(id, storage, _)| *id == "deepseek41-flash-tp4" && *storage == kv.storage)
            .unwrap();
        if let Err(observed) = bundle_runtime::fake_host::check_journal(
            &lines,
            *expected,
            &format!("deepseek41-flash-tp4.native-moe.{layout}.journal"),
        ) {
            mismatches.push(format!("{layout}: {observed}"));
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// Opting out of native MoE prefill refuses a Flash TP4 instance, and says
/// why: its exact prefill kernels read the backbone gate/up bank as rows,
/// which TP4 stores in quad-lane row tiles (`sm120::moe_row_tiles_op`).
/// Nothing stays held.
#[test]
fn native_moe_prefill_opt_out_refuses_the_row_tiled_tp4_bank() {
    crate::sm120::moe_native::with_native(false, || {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        for kv in KV_LAYOUTS {
            let host = FakeHost::new(shape, kv);
            let record = crate::config::tests::record(&shape, kv, 4, false);
            let created = api().create(HostServicesTableV1::new(Rc::clone(&host)), &record);
            assert!(created.is_err(), "{:?}", kv.storage);
            let reason = crate::sm120::moe_row_tiles_op::NATIVE_OFF;
            assert!(
                host.logs.borrow().iter().any(|line| line.contains(reason)),
                "{:?}: {:?}",
                kv.storage,
                host.logs.borrow()
            );
            assert_eq!(host.acquired.get(), host.released.get());
        }
    });
}

/// Every launch of `lines` fits its entry's dynamic shared memory: the
/// default without a configure, else the bytes its resolved symbol was
/// configured for. CUDA refuses a launch past either.
fn assert_launches_fit_configured_shared(lines: &[String], topology: &str) {
    let mut names = BTreeMap::<&str, &str>::new();
    let mut configured = BTreeMap::<&str, u32>::new();
    let mut launched = 0;
    for line in lines {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["resolve", slot, symbol, ..] => {
                names.insert(slot, symbol);
            }
            ["configure", slot, bytes] => {
                let symbol = names[slot];
                let bytes: u32 = bytes.parse().unwrap();
                let most = configured.entry(symbol).or_insert(0);
                *most = (*most).max(bytes);
            }
            ["launch", _, symbol, ..] => {
                let shared: u32 = line
                    .split("shared_memory_bytes: ")
                    .nth(1)
                    .and_then(|rest| rest.split(',').next())
                    .unwrap()
                    .parse()
                    .unwrap();
                let limit = configured
                    .get(symbol)
                    .copied()
                    .unwrap_or(bundle_runtime::acquire::DEFAULT_SHARED_BYTES);
                assert!(
                    shared <= limit,
                    "{topology}: {symbol} launches with {shared} bytes of dynamic shared memory, configured for {limit}"
                );
                launched += 1;
            }
            _ => {}
        }
    }
    assert!(launched > 0, "{topology}");
}

/// Create refuses a rank index that stores a weight view an exact F32
/// projection, a dense projection or the head reads in another layout than
/// its ops read: a router or head packed in MMA fragments, or row-major
/// codes under a packed fold, binds and runs, on scrambled weights.
#[test]
fn create_refuses_a_weight_view_its_op_reads_in_another_layout() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let packed = kernel_core::weight_transform::mxfp8_fold::LAYOUT_ID;
    for (view, stored, read) in [
        (
            "layers.0.ffn.gate.weight",
            "bf16_mma_fragments_16x16_v1",
            "op reads bf16_row_major_v1",
        ),
        (
            "head.weight",
            "bf16_mma_fragments_16x16_v1",
            "op reads bf16_row_major_v1",
        ),
        (
            "layers.3.attn.wq_a.weight",
            "mxfp8_e4m3_rows_v1",
            &*format!("ops read {packed}"),
        ),
        (
            "layers.0.ffn.shared_experts.w3.weight",
            "mxfp8_e4m3_rows_v1",
            &*format!("ops read {packed}"),
        ),
    ] {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.layouts.borrow_mut().insert(view.into(), stored.into());
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 4, false);
        let created = api().create(HostServicesTableV1::new(Rc::clone(&host)), &record);
        assert!(created.is_err(), "{view}");
        let expected = format!("{view} is stored as {stored}, but its {read}");
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains(&expected)),
            "{view}: {:?}",
            host.logs.borrow()
        );
        assert_eq!(host.acquired.get(), host.released.get(), "{view}");
    }
}

/// Create refuses a rank index that stores a backbone expert bank in
/// another layout than its routed ops read: at TP4 the gate/up consumers
/// read quad-lane row tiles (codes in row tiles alone are another layout)
/// and the down consumers rows, at TP2 both read rows (the drafter's bank:
/// `dspark`).
#[test]
fn create_refuses_an_expert_bank_its_routed_ops_read_in_another_layout() {
    let rows = RoutedLayout::Rows.view_layouts().unwrap();
    let tiles = RoutedLayout::Tiles.view_layouts().unwrap();
    let lanes = RoutedLayout::Lanes.view_layouts().unwrap();
    assert_eq!(RoutedLayout::served(&FLASH, 4), RoutedLayout::Lanes);
    for (ranks, view, part, stored, read) in [
        (
            Ranks::TP4,
            "layers.0.ffn.experts.gate_up.weight",
            0,
            rows,
            lanes,
        ),
        (
            Ranks::TP4,
            "layers.0.ffn.experts.gate_up.weight",
            0,
            tiles,
            lanes,
        ),
        (
            Ranks::TP4,
            "layers.7.ffn.experts.gate_up.scale",
            1,
            rows,
            lanes,
        ),
        (
            Ranks::TP4,
            "layers.0.ffn.experts.down.weight",
            0,
            tiles,
            rows,
        ),
        (
            Ranks::TP2,
            "layers.0.ffn.experts.gate_up.weight",
            0,
            tiles,
            rows,
        ),
    ] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.layouts
            .borrow_mut()
            .insert(view.into(), stored[part].into());
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 4, false);
        let created = api().create(HostServicesTableV1::new(Rc::clone(&host)), &record);
        assert!(created.is_err(), "{ranks:?} {view}");
        let expected = format!(
            "{view} is stored as {}, but its op reads {}",
            stored[part], read[part]
        );
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains(&expected)),
            "{ranks:?} {view}: {:?}",
            host.logs.borrow()
        );
        assert_eq!(host.acquired.get(), host.released.get(), "{view}");
    }
}

/// An instance that runs the lowered stages alone launches the prepared
/// ops through their cubins' entries in program order.
#[test]
fn a_partial_instance_launches_the_lowered_stages_in_program_order() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    // The partial instance launches the row ops', the all-reduce's and the
    // mixers' entries: the embedding widened, reduced through the F32 pair
    // and rounded, before the streams take it; the first sublayer's mixer
    // before its input norm; the F32 head collapse, then the F32 final
    // norm, last.
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    if let Err(error) = instance.execute(&prompt(7, 0, 0, 3, true), &mut Views::default()) {
        panic!("{error:?}: {:?}", host.logs.borrow());
    }
    let journal = host.device.journal.borrow().clone();
    let first = |symbol: &str| {
        journal
            .iter()
            .position(|line| line.starts_with("launch") && line.contains(symbol))
            .unwrap_or_else(|| panic!("{symbol}"))
    };
    let order = [
        "chop_row_embedding_deepseek41_tp4_v1",
        "chop_row_rounding_widen_deepseek41_w5120_v1",
        "chop_comm_reduce_scatter_f32_v1",
        "chop_comm_allgather_scattered_f32_v1",
        "chop_row_rounding_round_deepseek41_w5120_v1",
        kernels_sm120_broadcast(),
        "chop_sinkhorn_hc_deepseek41_lagged_mix_collapse_split_v1",
        "chop_row_rmsnorm_deepseek41_w5120_v1",
        "chop_sinkhorn_hc_deepseek41_expand_f32_v1",
        "chop_sinkhorn_hc_deepseek41_weighted_head_f32_v1",
        "chop_row_rmsnorm_deepseek41_f32_w5120_v1",
    ]
    .map(first);
    assert!(order.is_sorted(), "{order:?}");
    instance.commit().unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
}

/// Both KV layouts lower every stage: a full instance, not the lowered
/// stages alone, creates, prefills and decodes, decoding each Engram
/// layer's rows in each step. The BF16 layout launches the same stages
/// through its BF16 window writes and staging, index-key writes and indexer
/// reads, and never the native ones.
#[test]
fn every_kv_layout_lowers_every_stage() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let mut counts = Vec::new();
    for kv in KV_LAYOUTS {
        let host = FakeHost::new(shape, kv);
        let record = crate::config::tests::record(&shape, kv, 4, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap_or_else(|error| {
                panic!(
                    "{:?} create: {error:?} {:?}",
                    kv.storage,
                    host.logs.borrow()
                )
            });
        for batch in [prompt(7, 0, 0, 3, true), decode(&[(7, 0, 3, 3)])] {
            if let Err(error) = instance.execute(&batch, &mut Views::default()) {
                panic!("{:?} {error:?}: {:?}", kv.storage, host.logs.borrow());
            }
            instance.commit().unwrap();
        }
        let journal = host.device.journal.borrow().clone();
        let launches = |symbol: &str| {
            journal
                .iter()
                .filter(|line| line.starts_with("launch") && line.contains(symbol))
                .count()
        };
        // Two Engram layers in each of the two steps, on each of four ranks.
        assert_eq!(launches("chop_row_mxfp8_decode_v1"), 2 * 2 * 4);
        let bf16 = launches("bf16");
        match kv.storage {
            KvStorage::Native => assert_eq!(launches("window_store_bf16"), 0),
            KvStorage::Bf16 => assert!(launches("window_store_bf16") > 0),
        }
        let packed = launches("_pack_fp4_guard_v1");
        let native_scores = launches("_mma_fp4_guard_fast_v1");
        let expected_packs = if kv.storage == KvStorage::Native {
            2 * shape.geometry.shape.csa.index_sources.len() * shape.ranks() as usize
        } else {
            0
        };
        assert_eq!(packed, expected_packs);
        assert_eq!(native_scores, expected_packs);
        let cache_producers = launches("_cache_decode_control0_");
        assert_eq!(
            cache_producers,
            if kv.storage == KvStorage::Native {
                FLASH.layers() as usize * 4
            } else {
                0
            }
        );
        // Native decode replaces each ordered merge with a group/final pair.
        let hierarchical_groups = launches("_hier_group_v1 ");
        assert_eq!(launches("_hier_final_v1 "), hierarchical_groups);
        assert_eq!(
            hierarchical_groups,
            if kv.storage == KvStorage::Native {
                shape.geometry.shape.csa.index_sources.len() * shape.ranks() as usize
            } else {
                0
            }
        );
        // Native prefill publishes each layer-20 candidate batch with one
        // fused launch in place of the scores, local-gather and merge triple.
        let fused_candidates = launches("_gather_v3_cached4_v1 ");
        assert_eq!(
            fused_candidates,
            if kv.storage == KvStorage::Native {
                shape.ranks() as usize
            } else {
                0
            }
        );
        // Commit e2292c26be replaces each native mixed prefill source/merge
        // pair with one finish: 38 layers on four ranks save 152 launches.
        // BF16 and the two window-only layers retain separate merges.
        let fused_prefill = launches("_head_shared_finish_v1");
        assert_eq!(
            fused_prefill,
            if kv.storage == KvStorage::Native {
                38 * shape.ranks() as usize
            } else {
                0
            }
        );
        counts.push((
            kv.storage,
            journal.iter().filter(|l| l.starts_with("launch")).count()
                + 2 * fused_candidates
                + fused_prefill
                - packed
                - cache_producers
                - hierarchical_groups,
            bf16,
        ));
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
    // After the validated query packs, cache producers, group/final merges
    // and fused candidates and prefill finishes, both layouts lower the same stages.
    assert_eq!(counts[0].1, counts[1].1, "{counts:?}");
}

/// The shared stream broadcast's symbol.
fn kernels_sm120_broadcast() -> &'static str {
    "chop_row_copy_broadcast_streams_v1"
}

/// Checks that every captured piece of `to`'s request state on each rank
/// holds exactly the bytes of the same piece of `from`'s, moved through the
/// snapshot pages, and that `from`'s state is still its own: every layer,
/// plane and slot lands where it belongs, and no byte of `to` is stale.
fn assert_state_moved(host: &FakeHost, items: u32, verify_rows: u32, from: u32, to: u32) {
    let state = model_deepseek41_contract::state::request_state(
        host.shape.geometry,
        host.kv.storage,
        false,
    );
    let extent = bundle_runtime::state::Extent {
        served: items,
        bucket: bundle_runtime::geometry::Items::new(items)
            .unwrap()
            .capacity(),
        verify_rows,
    };
    let holder = |slot| bundle_runtime::state::Holder::Slot(slot);
    let sources = state.pieces(extent, holder(from)).unwrap();
    let targets = state.pieces(extent, holder(to)).unwrap();
    assert_eq!(sources.len(), targets.len());
    let captured: std::collections::BTreeSet<usize> =
        sources.iter().map(|piece| piece.buffer).collect();
    // The rings and tails are captured whole: every layer's and plane's
    // complete entry of the slot.
    for name in ["window.ring", "compressor.tail"] {
        let index = state.buffers.iter().position(|b| b.name == name).unwrap();
        assert!(captured.contains(&index), "{name} is not captured");
        let bytes: u64 = sources
            .iter()
            .filter(|piece| piece.buffer == index)
            .map(|piece| piece.length)
            .sum();
        assert_eq!(bytes, state.buffers[index].entry_bytes(), "{name}");
    }
    // Each buffer's offset in the state view, end to end.
    let mut starts = Vec::new();
    let mut at = 0;
    for buffer in &state.buffers {
        starts.push(at);
        at += buffer.bytes(extent);
    }
    let memory = host.memory.borrow();
    for rank in 0..host.shape.ranks() {
        let view = BundleDeviceBufferHandleV1::from_parts(1, STATE_SLOT + rank);
        let span = |piece: &bundle_runtime::state::Piece| {
            assert!(!piece.spare);
            DeviceBufferSpanV1::new(view, starts[piece.buffer] + piece.offset, piece.length)
        };
        for (source, target) in sources.iter().zip(&targets) {
            assert_eq!(
                (source.buffer, source.length),
                (target.buffer, target.length)
            );
            let (from_span, to_span) = (span(source), span(target));
            let mut expected = from_span.offset;
            for (length, origin) in memory.read(to_span) {
                assert_eq!(
                    origin,
                    Origin::Held(view.raw, expected),
                    "rank {rank} {}: a restored byte did not come from slot {from}",
                    state.buffers[target.buffer].name
                );
                expected += length;
            }
            assert_eq!(expected, from_span.offset + from_span.length);
            assert_eq!(
                memory.read(from_span),
                [(from_span.length, Origin::Held(view.raw, from_span.offset))]
            );
        }
    }
}

/// The rank counts the prefix-cache tests serve flash at.
const PREFIX_RANKS: [Ranks; 2] = [Ranks::TP2, Ranks::TP4];

/// A request's rings and tails reach snapshot pages after a prompt and
/// come back into the host's state slot of a new request, which then
/// decodes; a restore that names other tokens is refused and leaves the
/// slot free. The capture ends at an odd position past the 136-cell ring's
/// wrap, and the restored slot holds exactly the captured slot's bytes on
/// every rank, at TP2 and TP4 in both KV layouts.
#[test]
fn prefix_snapshots_round_trip_through_the_pages() {
    for (ranks, kv) in PREFIX_RANKS
        .into_iter()
        .flat_map(|ranks| KV_LAYOUTS.map(|kv| (ranks, kv)))
    {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, kv);
        let mut instance = create(&host, 4);
        let mut views = Views::default();
        let tokens = 141;
        instance
            .execute(&prompt(7, 0, 0, tokens, true), &mut views)
            .unwrap();
        instance.commit().unwrap();
        let count = instance.describe_prefix_state().unwrap();
        assert!(count > 0);
        let pages = (0..count).map(|page| 40 + page).collect::<Vec<_>>();
        instance.capture_prefix_state(7, tokens, &pages).unwrap();
        instance.retire_request(7).unwrap();
        let ids = (0..tokens as u32).collect::<Vec<u32>>();
        assert!(
            instance
                .restore_prefix_state_in_slot(9, &ids[..tokens as usize - 1], &pages, 2)
                .is_err()
        );
        instance
            .restore_prefix_state_in_slot(9, &ids, &pages, 2)
            .unwrap();
        assert_state_moved(&host, 4, 0, 0, 2);
        instance
            .execute(&decode(&[(9, 2, tokens, 5)]), &mut views)
            .unwrap();
        instance.commit().unwrap();
        assert_eq!(views.rows(), 2);
        instance.retire_request(9).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// A host state slot takes another request only once its holder retires,
/// by restore or by prompt: a restore or a prompt into a held slot, or a
/// restore into a slot past the served items, is refused and changes
/// nothing; after the retire the same slot takes a restored request, which
/// holds exactly the captured state and decodes, and a new prompt. At TP2
/// and TP4 in both KV layouts.
#[test]
fn a_state_slot_takes_a_new_request_once_its_holder_retires() {
    for (ranks, kv) in PREFIX_RANKS
        .into_iter()
        .flat_map(|ranks| KV_LAYOUTS.map(|kv| (ranks, kv)))
    {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, kv);
        let mut instance = create(&host, 4);
        let tokens = 141;
        instance
            .execute(&prompt(7, 0, 0, tokens, true), &mut Views::default())
            .unwrap();
        instance.commit().unwrap();
        let count = instance.describe_prefix_state().unwrap();
        let pages = (0..count).map(|page| 40 + page).collect::<Vec<_>>();
        instance.capture_prefix_state(7, tokens, &pages).unwrap();
        let ids = (0..tokens as u32).collect::<Vec<u32>>();
        instance
            .restore_prefix_state_in_slot(9, &ids, &pages, 1)
            .unwrap();
        assert_state_moved(&host, 4, 0, 0, 1);
        // Slots 0 and 1 are held, and the instance serves slots 0 to 3.
        for slot in [0, 1, 4] {
            assert!(
                instance
                    .restore_prefix_state_in_slot(11, &ids, &pages, slot)
                    .is_err(),
                "slot {slot}"
            );
        }
        for slot in [0, 1] {
            assert!(
                instance
                    .execute(&prompt(11, slot, 0, 3, true), &mut Views::default())
                    .is_err(),
                "slot {slot}"
            );
        }
        // The holders still decode in their slots.
        instance
            .execute(
                &decode(&[(7, 0, tokens, 1), (9, 1, tokens, 2)]),
                &mut Views::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        // Once request 9 retires, its slot takes request 11's restore.
        instance.retire_request(9).unwrap();
        instance
            .restore_prefix_state_in_slot(11, &ids, &pages, 1)
            .unwrap();
        assert_state_moved(&host, 4, 0, 0, 1);
        let mut views = Views::default();
        instance
            .execute(&decode(&[(11, 1, tokens, 3)]), &mut views)
            .unwrap();
        instance.commit().unwrap();
        assert_eq!(views.rows(), 1);
        // Once request 7 retires, its slot takes a new prompt.
        instance.retire_request(7).unwrap();
        instance
            .execute(&prompt(13, 0, 0, 3, true), &mut Views::default())
            .unwrap();
        instance.commit().unwrap();
        instance.retire_request(11).unwrap();
        instance.retire_request(13).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// The origin check itself fails when a restore moves nothing: a slot's
/// own bytes are not another slot's.
#[test]
#[should_panic(expected = "a restored byte did not come from slot 0")]
fn the_state_check_catches_a_restore_that_moved_nothing() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    assert_state_moved(&host, 4, 0, 0, 2);
}

/// One verify step of `items`: (request, slot, previous tokens, the
/// row's tokens: the last sampled one, then the drafts).
fn verify(items: &[(u64, u64, u64, &[u32])]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for &(request, slot, previous, tokens) in items {
        let rows = tokens.len() as u64;
        let end = previous + rows;
        batch.push(BatchItemV1::verify(
            request,
            rows as u32,
            false,
            tokens.to_vec(),
        ));
        appends.push(BatchAppendV1::new(
            previous,
            rows,
            end,
            (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

fn kept(request_id: u64, kept_rows: u32) -> BundleKeptRowsV1 {
    BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id,
        kept_rows,
        reserved_tail: 0,
    }
}

/// With speculation on: a verify step keeps two of one request's four
/// rows and one of another's two (three drafts rejected), the next keeps
/// all four; a cancelled step invalidates the request it
/// touched, which retires, while another request decodes on; a request preempted by a snapshot resumes as a
/// new request in a recycled slot, which holds exactly the captured state
/// (none of its previous request's), and decodes on. Each verify row's
/// logits reach the sink.
#[test]
fn the_lifecycle_verifies_cancels_preempts_and_recycles_slots() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.speculation.set(Some((3, 16)));
    let mut instance = create(&host, 4);
    let mut views = Views::default();
    for (request, slot, count) in [(7, 0, 70), (9, 1, 5)] {
        instance
            .execute(&prompt(request, slot, 0, count, true), &mut views)
            .unwrap();
        instance.commit().unwrap();
    }
    assert_eq!(views.rows(), 2);
    // Request 7 verifies three drafts and keeps two rows; request 9
    // verifies one and keeps only its first row.
    let mut views = Views::default();
    instance
        .execute(
            &verify(&[(7, 0, 70, &[1, 2, 3, 4]), (9, 1, 5, &[6, 7])]),
            &mut views,
        )
        .unwrap();
    assert_eq!(views.rows(), 6);
    instance.commit_accepted(&[kept(7, 2), kept(9, 1)]).unwrap();
    // Every draft accepted from the kept position.
    let mut views = Views::default();
    instance
        .execute(&verify(&[(7, 0, 72, &[5, 6, 7, 8])]), &mut views)
        .unwrap();
    assert_eq!(views.rows(), 4);
    instance.commit_accepted(&[kept(7, 4)]).unwrap();
    // A step from a stale position is refused and changes nothing.
    assert!(
        instance
            .execute(&decode(&[(7, 0, 72, 1)]), &mut Views::default())
            .is_err()
    );
    // A cancelled decode invalidates the request it touched, and only it.
    instance
        .execute(&decode(&[(7, 0, 76, 1)]), &mut Views::default())
        .unwrap();
    instance.cancel().unwrap();
    assert!(
        instance
            .execute(&decode(&[(7, 0, 76, 1)]), &mut Views::default())
            .is_err()
    );
    instance
        .execute(&decode(&[(9, 1, 6, 2)]), &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    instance.retire_request(7).unwrap();
    instance.retire_request(9).unwrap();
    // Request 11 runs past the ring's wrap in slot 2 and is preempted: its
    // state is captured and it retires; request 13 then uses slot 2.
    instance
        .execute(&prompt(11, 2, 0, 141, true), &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    let count = instance.describe_prefix_state().unwrap();
    let pages = (0..count).map(|page| 40 + page).collect::<Vec<_>>();
    instance.capture_prefix_state(11, 141, &pages).unwrap();
    instance.retire_request(11).unwrap();
    instance
        .execute(&prompt(13, 2, 0, 9, true), &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    // Request 11 resumes as request 15 in slot 0, which request 7 held.
    let ids = (0..141).collect::<Vec<u32>>();
    instance
        .restore_prefix_state_in_slot(15, &ids, &pages, 0)
        .unwrap();
    assert_state_moved(&host, 4, 4, 2, 0);
    let mut views = Views::default();
    instance
        .execute(&decode(&[(15, 0, 141, 3), (13, 2, 9, 4)]), &mut views)
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(views.rows(), 2);
    instance.retire_request(13).unwrap();
    instance.retire_request(15).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Marks a test's child process (`in_child`).
const TAPS_CHILD: &str = "DEEPSEEK41_TAPS_CHILD";

/// Runs the test `name` in a child process with `CHOPINFER_TAPS_DIR` set to
/// `directory` and the taps variables `env`, and fails unless it passed.
fn in_child(name: &str, directory: &std::path::Path, env: &[(&str, &str)]) {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([name, "--exact", "--test-threads=1"])
        .env(TAPS_CHILD, "1")
        .env(runtime_bundle_sdk::taps::TAPS_ENV.dir, directory);
    for variable in ["SITES", "LAYERS", "FROM", "BYTES", "SCOPE_JSON"] {
        command.env_remove(format!("CHOPINFER_TAPS_{variable}"));
    }
    let output = command.envs(env.iter().copied()).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A serving bundle refuses to create while a taps variable is set.
#[cfg(not(feature = "diagnostic-taps"))]
#[test]
fn serving_create_refuses_taps() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    if std::env::var_os(TAPS_CHILD).is_some() {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 4, false);
        let refused = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .err();
        assert!(refused.is_some(), "the create was not refused");
        let logs = host.logs.borrow();
        assert!(
            logs.iter()
                .any(|line| line.contains("CHOPINFER_TAPS_DIR") && line.contains("diagnostic-taps")),
            "{logs:?}"
        );
        assert_eq!(host.released.get(), host.acquired.get());
        return;
    }
    let directory =
        std::env::temp_dir().join(format!("deepseek41-taps-serving-{}", std::process::id()));
    in_child(
        "fake_host_tests::serving_create_refuses_taps",
        &directory,
        &[],
    );
    assert!(!directory.exists(), "a refused create wrote taps");
}

/// The launch lines of the journal from line `from`, by stream in issue
/// order.
#[cfg(feature = "diagnostic-taps")]
fn launch_lines(host: &FakeHost, from: usize) -> std::collections::BTreeMap<String, Vec<String>> {
    let mut streams = std::collections::BTreeMap::<String, Vec<String>>::new();
    for line in &host.device.journal.borrow()[from..] {
        if let Some(rest) = line.strip_prefix("launch ") {
            let stream = rest.split(' ').next().unwrap_or_default().to_owned();
            streams.entry(stream).or_default().push(line.clone());
        }
    }
    streams
}

/// A diagnostic bundle's prefill runs stage by stage and issues exactly
/// the launches of an untapped prefill; its committed step holds every
/// selected tap of the prompt's rows in program order from rank 0, and a
/// cancelled step's capture is removed.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_prefills_write_the_taps() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let sites = [
        (
            "CHOPINFER_TAPS_SITES",
            "embed.out,in.streams,x.hc.attn.pre,attn.in,router.*,out,final.norm",
        ),
        ("CHOPINFER_TAPS_LAYERS", "all"),
    ];
    if std::env::var_os(TAPS_CHILD).is_some() {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        let directory = std::env::var_os(runtime_bundle_sdk::taps::TAPS_ENV.dir).unwrap();
        for request in [7, 9] {
            let start = host.device.journal.borrow().len();
            instance
                .execute(&prompt(request, 0, 0, 5, true), &mut Views::default())
                .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
            if request == 7 {
                std::fs::write(
                    std::path::Path::new(&directory).join("launches.json"),
                    serde_json::to_vec(&launch_lines(&host, start)).unwrap(),
                )
                .unwrap();
                instance.commit().unwrap();
            } else {
                instance.cancel().unwrap();
            }
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        return;
    }
    // The launches of the same prefill without taps.
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    let start = host.device.journal.borrow().len();
    instance
        .execute(&prompt(7, 0, 0, 5, true), &mut Views::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let untapped = launch_lines(&host, start);
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let directory = std::env::temp_dir().join(format!("deepseek41-taps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    in_child(
        "fake_host_tests::diagnostic_prefills_write_the_taps",
        &directory,
        &sites,
    );
    let tapped: std::collections::BTreeMap<String, Vec<String>> =
        serde_json::from_slice(&std::fs::read(directory.join("launches.json")).unwrap()).unwrap();
    assert_eq!(untapped.len(), 2);
    assert!(
        untapped.values().all(|lines| lines.len() > 100),
        "{untapped:?}"
    );
    assert_eq!(tapped, untapped);
    let step = directory.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(step.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("capture.json")["model"], "deepseek41");
    assert_eq!(read("capture.json")["program"], "Prefill");
    assert_eq!(read("completion.json")["completion"], "full");
    assert_eq!(read("items.json")["items"][0]["request"], 7);
    let index = std::fs::read_to_string(step.join("index.jsonl")).unwrap();
    let written: Vec<serde_json::Value> = index
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let selected = [
        "embed.out",
        "in.streams",
        "x.hc.attn.pre",
        "attn.in",
        "router.weights",
        "router.ids",
        "out",
        "norm",
    ];
    let expected: Vec<String> = crate::taps::text_specs(&shape)
        .into_iter()
        .filter(|spec| selected.contains(&runtime_bundle_sdk::taps::site(&spec.name)))
        .map(|spec| spec.name)
        .collect();
    let names: Vec<&str> = written
        .iter()
        .map(|entry| entry["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, expected);
    for entry in &written {
        assert_eq!(entry["rank"], 0, "{entry}");
        assert_eq!(entry["rows"], 5, "{entry}");
    }
    // The cancelled step's capture is removed.
    assert!(!directory.join("step0001").exists());
    std::fs::remove_dir_all(&directory).unwrap();
}

/// An invalid quantized input in one live row of one rank refuses that
/// step with a named error and emits no logits, after every index status
/// read clean; the request retires and the instance serves on.
#[test]
fn an_invalid_quantized_input_refuses_its_step_alone() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    let mut views = Views::default();
    step(&mut instance, &prompt(9, 1, 0, 3, true), &mut views);
    // Rank 1 flags row 3 in the next step's prefill: its read back follows
    // the index sources' and both of rank 0's status planes.
    let mut status = vec![0_u8; 5 * 4];
    status[3 * 4] = 1;
    host.injected.replace(Some(status));
    host.injected_after
        .set(FLASH.shape.csa.index_sources.len() + 2);
    let mut refused = Views::default();
    let error = instance
        .execute(&prompt(7, 0, 0, 5, true), &mut refused)
        .unwrap_err();
    assert_eq!(error, BundleAbiError::Status(BundleStatusV1::BundleError));
    assert_eq!(refused.rows(), 0);
    assert!(
        host.logs.borrow().iter().any(|line| line
            .contains("a quantized input on rank 1 was invalid in row 3 (request Some(7))")),
        "{:?}",
        host.logs.borrow()
    );
    assert!(host.injected.borrow().is_none());
    assert!(instance.commit().is_err());
    instance.retire_request(7).unwrap();
    let mut views = Views::default();
    step(&mut instance, &decode(&[(9, 1, 3, 5)]), &mut views);
    assert_eq!(views.rows(), 1);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A non-finite index score in one live row refuses that step with a
/// named error and emits no logits; its request is invalidated and retires,
/// while the other request in the instance decodes on, and the slot serves
/// a new request.
#[test]
fn a_non_finite_index_score_refuses_its_step_alone() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    let mut views = Views::default();
    step(&mut instance, &prompt(9, 1, 0, 3, true), &mut views);
    assert_eq!(views.rows(), 1);
    // Row 2 of the first index source fails in the next step's prefill.
    let mut status = vec![0_u8; 5 * 4];
    status[2 * 4] = 1;
    host.injected.replace(Some(status));
    let mut refused = Views::default();
    let error = instance
        .execute(&prompt(7, 0, 0, 5, true), &mut refused)
        .unwrap_err();
    assert_eq!(error, BundleAbiError::Status(BundleStatusV1::BundleError));
    assert_eq!(refused.rows(), 0);
    let layer = FLASH.shape.csa.index_sources[0];
    assert!(
        host.logs.borrow().iter().any(|line| line.contains(&format!(
            "the index selection of layer {layer} met a non-finite score in row 2 (request Some(7))"
        ))),
        "{:?}",
        host.logs.borrow()
    );
    assert!(host.injected.borrow().is_none());
    // No step is open; the failed request retires, the other decodes on.
    assert!(instance.commit().is_err());
    instance.retire_request(7).unwrap();
    let mut views = Views::default();
    step(&mut instance, &decode(&[(9, 1, 3, 5)]), &mut views);
    assert_eq!(views.rows(), 1);
    let mut views = Views::default();
    step(&mut instance, &prompt(11, 0, 0, 4, true), &mut views);
    assert_eq!(views.rows(), 1);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Every captured small ceil launch uses a decode entry (parallel, cached
/// at the measured widths, or the status folds' sliced WO_a input
/// quantizer), including ops originally prepared once at prefill capacity.
/// Larger rows keep the serial entry.
fn assert_small_row_ceil_bindings(lines: &[String]) {
    let cached: Vec<String> = model_deepseek41_contract::geometry::GEOMETRIES
        .iter()
        .flat_map(|g| {
            model_deepseek41_contract::admission::SUPPORTED_RANKS
                .iter()
                .filter_map(move |&ranks| crate::sm120::ceil_parallel::quantizers(g, ranks).ok())
        })
        .flatten()
        .filter_map(|e| crate::sm120::ceil_cached::cached(&e))
        .map(|e| e.symbol)
        .chain(
            crate::sm120::fold_fused::ROWS
                .iter()
                .filter_map(|&rows| crate::sm120::fold_fused::sliced(rows))
                .map(|q| q.symbol),
        )
        .collect();
    let mut checked = 0;
    let mut capturing = false;
    for line in lines {
        if line.starts_with("capture ") {
            capturing = true;
        }
        if line.starts_with("captured ") {
            capturing = false;
        }
        if !line.starts_with("launch ") {
            continue;
        }
        let symbol = line.split_whitespace().nth(2).unwrap();
        if !(symbol.contains("_ceil") || symbol.contains("_shared_swiglu_")) {
            continue;
        }
        let rows: u32 = line
            .split("grid_x: ")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        if capturing || rows > model_deepseek41_contract::serving::DECODE_BATCH {
            assert_eq!(
                symbol.ends_with("_parallel_v1") || cached.iter().any(|c| c == symbol),
                rows <= model_deepseek41_contract::serving::DECODE_BATCH,
                "{line}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

mod norm_prefill;
mod prefill_engram;
mod query_prefill;
