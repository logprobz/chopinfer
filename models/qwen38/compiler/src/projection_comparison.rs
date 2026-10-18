//! Freeze comparisons use full served header shapes and the existing compiler.
use crate::mixed_layouts::headers;
use headers::exports::Layout;

// The synthetic projection payloads every model shares, from the SDK's
// test support sources (outside its module tree).
#[path = "../../../../crates/model-compiler-sdk/test-support/synthetic.rs"]
mod synthetic;

use crate::{catalog, precision_source::SourcePrecision, projection_program};
use kernel_core::contract::DType;
use kernel_core::precision::capability::{CapabilityDocument, CapabilityInventory};
use kernel_core::precision::format::{A16Recipe, KernelFamily, registry};
use kernel_core::precision::{TensorSpan, dtype_bits};
use model_compiler_sdk::quantization::{
    ActivationPolicy, import::ExportMetadata, source::TensorInventory,
};
use model_qwen38_contract::{
    backend::{Kernels, Selected},
    contract::{Ranks, Shape},
    precision::{self, DenseRole},
    projections,
};
use projection_spec::derive::{
    GeometryMode, ImportedProjection, InstanceKey, ProjectionContext, SourceMatrix, Topology,
};
use std::collections::{BTreeMap, BTreeSet};
use synthetic::{SyntheticPayload, selected_source_payload_with, synthetic_payload};

fn fill_source(span: &TensorSpan, offset: u64, out: &mut [u8]) {
    synthetic::fill_source(span, offset, out);
    if span.dtype == DType::Ue8M0 {
        // Qwen's A16 consumer requires exact BF16 scaled operands. Every
        // finite E4M3 code is exact at UE8M0 scales 3..=246, so shift the
        // offset-dependent ramp out of the underflowing scales 0..=2.
        for byte in out {
            *byte += 3;
        }
    }
}

fn synthetic_source(span: &TensorSpan) -> Vec<u8> {
    let mut bytes = vec![0; span.byte_length as usize];
    fill_source(span, 0, &mut bytes);
    bytes
}

fn dtype(name: &str) -> DType {
    match name {
        "BF16" => DType::Bf16,
        "F32" => DType::F32,
        "I64" => DType::I64,
        "U8" => DType::U8,
        "F8_E4M3" => DType::Fp8E4M3,
        "F8_E8M0" => DType::Ue8M0,
        other => panic!("unsupported fixture dtype {other}"),
    }
}

fn source(m: &projection_spec::bind::CheckpointMetadata, shape: Shape) -> SourcePrecision {
    let mut names: BTreeSet<_> = (0..shape.layers())
        .flat_map(|layer| {
            DenseRole::roles_for(shape, layer)
                .unwrap()
                .into_iter()
                .flat_map(move |r| r.operand_names(shape, layer).unwrap())
        })
        .collect();
    // Retain an authenticated expert leaf so the export-wide image policy
    // sees the same format family as the production source program.
    names.insert("model.language_model.layers.0.mlp.experts.0.gate_proj.weight".into());
    source_for_names(m, shape, &names)
}

fn source_for_names(
    m: &projection_spec::bind::CheckpointMetadata,
    shape: Shape,
    names: &BTreeSet<String>,
) -> SourcePrecision {
    let tensors = m
        .tensors
        .iter()
        .filter(|(n, _)| {
            names.contains(*n)
                || n.rsplit_once('.')
                    .is_some_and(|(stem, _)| names.contains(&format!("{stem}.weight")))
        })
        .map(|(name, info)| {
            let dtype = dtype(&info.dtype);
            let span = TensorSpan {
                tensor: name.clone(),
                dtype,
                shape: info.shape.clone(),
                offset: 0,
                byte_length: info.shape.iter().product::<u64>() * dtype_bits(dtype) / 8,
            };
            (name.clone(), span)
        })
        .collect();
    let mut config =
        crate::config::Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
    config.quantization_config = m.quantization_config.clone();
    let logical = catalog::expected_tensors_for(shape.geometry)
        .unwrap()
        .into_iter()
        .filter(|(n, _)| names.contains(n))
        .collect();
    SourcePrecision::admit(
        &config,
        ExportMetadata::from_documents(Some(&m.quantization_config), None).unwrap(),
        TensorInventory {
            tensors,
            spans: BTreeMap::new(),
        },
        logical,
    )
    .unwrap()
}

fn projection(role: DenseRole, full: bool) -> &'static str {
    match role {
        DenseRole::LinearQkv => "linear.qkv",
        DenseRole::LinearZ => "linear.z",
        DenseRole::LinearA => "linear.a",
        DenseRole::LinearB => "linear.b",
        DenseRole::AttentionQ => "full.q",
        DenseRole::AttentionK => "full.k",
        DenseRole::AttentionV => "full.v",
        DenseRole::IndexQk => "full.index",
        DenseRole::AttentionOutput => "full.output",
        DenseRole::Output if full => "full.output",
        DenseRole::Output => "linear.output",
        DenseRole::SharedGateUp => "main.shared.gate-up",
        DenseRole::SharedDown => "main.shared.down",
    }
}

/// How much payload production a served comparison runs. Payload bytes back
/// the record, packing and slice checks that run on every layer, rank and use,
/// so the default compares them once per payload class on the first and last
/// rank, and for the first and last member of an expert bank; `Every` is the
/// opt-in full sweep.
#[derive(Clone, Copy, PartialEq)]
enum Sweep {
    Default,
    Every,
}

/// The rank whose payload bytes a comparison checks. Interior ranks select an
/// offset window like the last rank, so the default checks the first and last.
fn payload_rank(rank: u32, ranks: u32, sweep: Sweep) -> Option<u32> {
    (sweep == Sweep::Every || rank == 0 || rank + 1 == ranks).then_some(rank)
}

/// Projections with one weight format, packed layout, row-run count and k
/// window gather and pack alike, so the default compares the smallest of them.
fn payload_class(operation: &projection_spec::derive::DerivedOperation) -> String {
    let slice = &operation.slices[0];
    let weight = &slice.source.weight;
    format!(
        "{:?} {:?} {:?} {} {}",
        weight.format,
        operation.packing.layout,
        operation.packing.columns,
        slice.rows.len(),
        slice.k == (0..weight.logical_shape[1])
    )
}

const FP8: &str = headers::exports::FP8;

/// The NVFP4-routed layout's header fixture.
fn nvfp4_routed() -> &'static str {
    Layout::Nvfp4RoutedFp8Mtp.text()
}

/// The MXFP8-dense layout's header fixture.
fn mxfp8_dense() -> &'static str {
    Layout::Mxfp8DenseNvfp4.text()
}

/// What a served comparison covered: every (layer, projection, rank, use) it
/// derived, its OperationSpec and source selection counts, and the payloads
/// it produced as (projection or bank member, rank).
#[derive(Default)]
struct Coverage {
    keys: BTreeSet<(Option<u64>, String, u32, usize)>,
    specs: usize,
    selections: usize,
    payloads: BTreeSet<(String, u32)>,
}

/// The (layer, projection, rank, use) keys the declaration serves for `ids`
/// on the stem layers in `layers`, independent of the comparison loops.
fn declared_keys(
    ranks: u32,
    ids: &[&str],
    layers: std::ops::Range<u64>,
) -> BTreeSet<(Option<u64>, String, u32, usize)> {
    let g = projections::geometry(ranks).unwrap();
    let d = projections::declaration(ranks)
        .unwrap()
        .expanded_profiles(&g)
        .unwrap();
    let mut keys = BTreeSet::new();
    for id in ids {
        let p = d.projections.iter().find(|p| p.id == *id).unwrap();
        let stem = d
            .layer_stems
            .iter()
            .find(|s| Some(s.id.as_str()) == p.stem())
            .unwrap();
        for layer in stem.indices.resolve(&g).unwrap() {
            if !layers.contains(&layer) {
                continue;
            }
            for rank in 0..ranks {
                for usage in 0..p.uses.len() {
                    keys.insert((Some(layer), id.to_string(), rank, usage));
                }
            }
        }
    }
    keys
}

/// The dense projections the compiler comparison serves; the QKV source has
/// declared part windows and its legacy fusion is compared elsewhere.
const DENSE: [&str; 11] = [
    "linear.z",
    "linear.a",
    "linear.b",
    "linear.output",
    "full.q",
    "full.k",
    "full.v",
    "full.index",
    "full.output",
    "main.shared.gate-up",
    "main.shared.down",
];

/// The smallest projection of each payload class, and the two-weight shared
/// gate/up, whose payload bytes the dense comparisons produce.
const DENSE_PAYLOADS: [&str; 3] = ["full.k", "main.shared.down", "main.shared.gate-up"];

/// The MXFP8-dense layout's smallest MXFP8 projection is `linear.a`.
const MXFP8_PAYLOADS: [&str; 3] = ["linear.a", "main.shared.down", "main.shared.gate-up"];

/// Check one default dense comparison against its declared keys, exact
/// counts and the payloads it samples on the first and last rank.
fn dense(text: &str, ranks: Ranks, counts: (usize, usize, usize), payloads: &[&str]) {
    let coverage = served_dense(text, ranks, Sweep::Default);
    let tp = Shape::flash(ranks).ranks();
    assert_eq!(coverage.keys, declared_keys(tp, &DENSE, 0..u64::MAX));
    assert_eq!(
        (coverage.keys.len(), coverage.specs, coverage.selections),
        counts
    );
    assert_eq!(
        coverage.payloads,
        payloads
            .iter()
            .flat_map(|id| [(id.to_string(), 0), (id.to_string(), tp - 1)])
            .collect()
    );
}

#[test]
fn fp8_tp2_served_dense_operations_and_rank_selections_match_the_current_compiler() {
    dense(FP8, Ranks::TP2, (11400, 7296, 13224), &DENSE_PAYLOADS);
}

#[test]
fn fp8_tp4_served_dense_operations_and_rank_selections_match_the_current_compiler() {
    dense(FP8, Ranks::TP4, (22800, 14592, 26448), &DENSE_PAYLOADS);
}

#[test]
fn nvfp4_routed_tp4_served_dense_operations_and_rank_selections_match_the_current_compiler() {
    dense(
        nvfp4_routed(),
        Ranks::TP4,
        (22800, 14592, 26448),
        &DENSE_PAYLOADS,
    );
}

#[test]
fn nvfp4_routed_tp2_served_dense_operations_and_rank_selections_match_the_current_compiler() {
    dense(
        nvfp4_routed(),
        Ranks::TP2,
        (11400, 7296, 13224),
        &DENSE_PAYLOADS,
    );
}

/// The MXFP8-dense layout's dense ports split by rows (column-split
/// projections) and by K (row-split outputs) at TP2 and TP4.
#[test]
fn mxfp8_dense_tp2_served_dense_operations_and_rank_selections_match_the_current_compiler() {
    // No input fusion is all BF16, so every port resolves its own spec.
    dense(
        mxfp8_dense(),
        Ranks::TP2,
        (11400, 11400, 13224),
        &MXFP8_PAYLOADS,
    );
}

#[test]
fn mxfp8_dense_tp4_served_dense_operations_and_rank_selections_match_the_current_compiler() {
    dense(
        mxfp8_dense(),
        Ranks::TP4,
        (22800, 22800, 26448),
        &MXFP8_PAYLOADS,
    );
}

#[test]
fn mxfp8_dense_synthetic_scales_are_exact_but_underflow_is_refused() {
    use kernel_core::contract::WeightFormat;
    use kernel_core::mxfp8_gemm::admit_operand;
    use kernel_core::precision::NibbleOrder;
    use kernel_core::precision::packed::{MatrixView, ScaleBytes};

    // The importer gives ModelOpt's U8 scale tensors their UE8M0 dtype.
    let span = TensorSpan {
        tensor: "matrix.weight_scale".into(),
        dtype: DType::Ue8M0,
        shape: vec![256, 1],
        offset: 0,
        byte_length: 256,
    };
    let mut scales = synthetic_source(&span);
    // The smallest E4M3 subnormal exposes BF16 underflow at scales 0..=2.
    let codes = vec![0x01; 256 * 32];
    let admit = |scales: &[u8]| {
        admit_operand(MatrixView {
            tensor: "matrix.weight",
            format: WeightFormat::Mxfp8,
            rows: 256,
            k: 32,
            row_origin: 0,
            k_origin: 0,
            nibble_order: NibbleOrder::EvenLow,
            codes: &codes,
            local: Some(ScaleBytes {
                dtype: DType::Ue8M0,
                bytes: scales,
            }),
            global: None,
        })
    };
    admit(&scales).unwrap();
    scales[0] = 2;
    assert_eq!(
        admit(&scales).unwrap_err().to_string(),
        "MXFP8 fold operands must be finite and exact in BF16"
    );
}

/// A header-only checkpoint of `text`'s served shapes in a fresh directory:
/// its configuration with the export's quantization rules, one index and
/// one sparse zero shard, so capture, import and placement run as they do
/// on the export without reading a weight. A layout without a vision tower
/// takes the FP8 export's BF16 tower.
fn sparse_checkpoint(text: &str, name: &str) -> std::path::PathBuf {
    let mut m = headers::fixture(text);
    for (tensor, info) in headers::fixture(FP8).tensors {
        if tensor.starts_with("model.visual.") {
            m.tensors.entry(tensor).or_insert(info);
        }
    }
    let base = std::env::var_os("CHOPINFER_SCRATCH").map_or_else(std::env::temp_dir, Into::into);
    let root = base.join(format!(
        "qwen38-placed-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let mut config: serde_json::Value =
        serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
    config["quantization_config"] = m.quantization_config.clone();
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut header = serde_json::Map::new();
    let mut weight_map = serde_json::Map::new();
    let mut offset = 0_u64;
    for (tensor, info) in &m.tensors {
        let bytes = info.shape.iter().product::<u64>() * dtype_bits(dtype(&info.dtype)) / 8;
        header.insert(
            tensor.clone(),
            serde_json::json!({
                "dtype": info.dtype, "shape": info.shape, "data_offsets": [offset, offset + bytes]
            }),
        );
        weight_map.insert(tensor.clone(), "model.safetensors".into());
        offset += bytes;
    }
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": weight_map})).unwrap(),
    )
    .unwrap();
    let header = serde_json::to_vec(&header).unwrap();
    let mut shard = std::fs::File::create(root.join("model.safetensors")).unwrap();
    std::io::Write::write_all(&mut shard, &(header.len() as u64).to_le_bytes()).unwrap();
    std::io::Write::write_all(&mut shard, &header).unwrap();
    shard.set_len(8 + header.len() as u64 + offset).unwrap();
    root
}

/// Each rank index the placement plans for a whole mixed export, from its
/// captured checkpoint through import, the catalog and the precision
/// resolved against the published capabilities, passes the view validation
/// the bundle runs at create, the MTP drafter's included. The fake host
/// builds its indexes from the same contract, so only this placement output
/// shows a view the contract names differently (MXFP8 scales were once
/// planned as `U8`, which the contract reads as `F8_E8M0`).
#[test]
fn planned_mixed_rank_indexes_pass_the_bundles_view_validation() {
    let published = precision::capabilities().unwrap();
    let inventory =
        CapabilityInventory::new(published.clone(), published.entries(), "0".repeat(64)).unwrap();
    for layout in Layout::ALL {
        let (name, text) = (layout.name(), layout.text());
        let root = sparse_checkpoint(text, name);
        let files = catalog::capture(&root).unwrap();
        let source = SourcePrecision::from_captured(&files).unwrap();
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::flash(ranks);
            let catalog =
                catalog::Catalog::from_imported(&files, source.clone(), shape.ranks() as usize)
                    .unwrap();
            let document = projection_program::SourceProgram::new(&source, shape)
                .unwrap()
                .resolve(ActivationPolicy::Auto, &inventory)
                .unwrap();
            model_qwen38_contract::prepared::bind(&document, shape, &inventory).unwrap();
            assert!(
                model_qwen38_contract::prepared::records_drafter(&document, shape).unwrap(),
                "{name} TP{}",
                shape.ranks()
            );
            let plan =
                crate::placement::ImagePlan::with_precision(&catalog, None, &document).unwrap();
            for index in &plan.ranks {
                let mut index = index.clone();
                index.sha256 = "0".repeat(64);
                let parsed = model_qwen38_contract::views::parse(
                    &serde_json::to_vec(&index).unwrap(),
                    index.rank as u32,
                )
                .unwrap();
                let context = format!("{name} TP{} rank {}", shape.ranks(), parsed.rank);
                model_qwen38_contract::prepared::validate_views(&parsed, shape, &document)
                    .unwrap_or_else(|e| panic!("{context}: {e}"));
                model_qwen38_contract::prepared::validate_drafter_views(&parsed, shape, &document)
                    .unwrap_or_else(|e| panic!("{context}: {e}"));
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// The first view `validate_views` or `validate_drafter_views` requires of
/// `index` (with each F32 norm `norms::admit` accepts from it) that is
/// neither its fixed view, which the legacy contract covers, nor a row of
/// `admitted` with its dtype, layout and bytes under its key.
fn uncovered(
    document: &kernel_core::precision::execution::PrecisionDocument,
    index: &model_qwen38_contract::views::RankIndex,
    shape: Shape,
    admitted: &[model_qwen38_contract::prepared::AdmittedView],
) -> Option<String> {
    use kernel_core::precision::Calibration;
    use model_qwen38_contract::{prepared, views};
    let ports = projections::ports::get(shape).unwrap();
    let fixed: BTreeMap<_, _> = views::required(index.rank, shape)
        .into_iter()
        .chain(views::drafter_required(shape))
        .map(|view| (view.name.clone(), view))
        .collect();
    let mut required = prepared::required(document, index.rank, shape).unwrap();
    required.extend(prepared::drafter_required(document, index.rank, shape).unwrap());
    model_qwen38_contract::norms::admit(index, &mut required);
    for view in required {
        if fixed.get(&view.name) == Some(&view) {
            continue;
        }
        let key = document
            .operations
            .iter()
            .filter_map(|s| {
                let suffix = view.name.strip_prefix(&format!("{}.", s.effective.id))?;
                Some((s, suffix))
            })
            .max_by_key(|(s, _)| s.effective.id.len())
            .map(|(s, suffix)| {
                let projection = ports.operation_projection(&s.source.id).unwrap().unwrap();
                let port = ports
                    .role(projection)
                    .map_or(ports.published(projection), |role| {
                        role.projection_id(shape, None)
                    });
                let calibrated = s.source.activations.iter().any(|a| {
                    a.quantizer
                        .as_ref()
                        .is_some_and(|q| matches!(q.calibration, Calibration::Static(_)))
                });
                prepared::ViewKey::Port {
                    port: port.into(),
                    format: registry::by_weight(&s.effective.weights[0]).unwrap().id(),
                    activation_scale: if calibrated {
                        prepared::ActivationScale::Static
                    } else {
                        prepared::ActivationScale::Absent
                    },
                    suffix: suffix.into(),
                }
            })
            .unwrap_or_else(|| prepared::ViewKey::F32Norm {
                name: view.name.clone(),
            });
        if !admitted.iter().any(|row| {
            row.key == key
                && row.dtype == view.dtype
                && row.layout_id == view.layout_id
                && row.bytes == view.bytes
        }) {
            return Some(format!("{view:?} under {key:?}"));
        }
    }
    None
}

/// The prepared identity's admitted table covers every packed layout and
/// every view create requires of each rank index the placement plans for a whole mixed
/// export, its MTP drafter's included, at TP2 and TP4: statically
/// calibrated NVFP4 experts need the `input_scale` rows and F32
/// hyper-connection norms the F32 norm rows.
#[test]
fn admitted_views_cover_every_view_create_requires_of_mixed_exports() {
    use model_qwen38_contract::prepared::{self, ViewKey};
    let published = precision::capabilities().unwrap();
    let inventory =
        CapabilityInventory::new(published.clone(), published.entries(), "0".repeat(64)).unwrap();
    for (layout, needs) in [
        (Layout::Nvfp4RoutedFp8Mtp, "input_scale"),
        (Layout::Mxfp8DenseNvfp4, "F32 norm"),
    ] {
        let (name, text) = (layout.name(), layout.text());
        let root = sparse_checkpoint(text, name);
        let files = catalog::capture(&root).unwrap();
        let source = SourcePrecision::from_captured(&files).unwrap();
        let mut needed = false;
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::flash(ranks);
            let catalog =
                catalog::Catalog::from_imported(&files, source.clone(), shape.ranks() as usize)
                    .unwrap();
            let document = projection_program::SourceProgram::new(&source, shape)
                .unwrap()
                .resolve(ActivationPolicy::Auto, &inventory)
                .unwrap();
            assert!(prepared::records_drafter(&document, shape).unwrap());
            let layouts = &prepared::admitted(shape).unwrap().layouts;
            for weight in document
                .operations
                .iter()
                .flat_map(|s| &s.effective.weights)
            {
                let layout = (weight.layout.id.clone(), weight.layout.version);
                assert!(layouts.contains(&layout), "{name}: {layout:?}");
            }
            let plan =
                crate::placement::ImagePlan::with_precision(&catalog, None, &document).unwrap();
            for index in &plan.ranks {
                let mut index = index.clone();
                index.sha256 = "0".repeat(64);
                let parsed = model_qwen38_contract::views::parse(
                    &serde_json::to_vec(&index).unwrap(),
                    index.rank as u32,
                )
                .unwrap();
                let context = format!("{name} TP{} rank {}", shape.ranks(), parsed.rank);
                let admitted = prepared::admitted_views(parsed.rank, shape).unwrap();
                if let Some(view) = uncovered(&document, &parsed, shape, admitted) {
                    panic!("{context}: the admitted table lacks {view}");
                }
                // The table less the rows this export needs.
                let without: Vec<_> = admitted
                    .iter()
                    .filter(|row| match &row.key {
                        ViewKey::Port { suffix, .. } => {
                            needs != "input_scale" || suffix != "input_scale"
                        }
                        ViewKey::F32Norm { .. } => needs != "F32 norm",
                    })
                    .cloned()
                    .collect();
                needed |= uncovered(&document, &parsed, shape, &without).is_some();
            }
        }
        assert!(needed, "{name} needs no {needs} row");
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// A rank index holding exactly the views `document` makes create require
/// of `rank`, laid out so each full layer's Q, K and V are contiguous.
fn exact_index(
    document: &kernel_core::precision::execution::PrecisionDocument,
    rank: u32,
    shape: Shape,
) -> model_qwen38_contract::views::RankIndex {
    use model_qwen38_contract::{prepared, views};
    let mut required = prepared::required(document, rank, shape).unwrap();
    required.extend(prepared::drafter_required(document, rank, shape).unwrap());
    let order = |name: &str| {
        name.replace(".q_proj.weight", ".0_proj.weight")
            .replace(".k_proj.weight", ".1_proj.weight")
            .replace(".v_proj.weight", ".2_proj.weight")
    };
    required.sort_by_key(|view| order(&view.name));
    let mut index = views::RankIndex {
        rank,
        image: format!("rank{rank}.img"),
        image_bytes: 0,
        sha256: Some("0".repeat(64)),
        direct_identity: None,
        views: Default::default(),
    };
    let mut offset = 0;
    for view in required {
        index.views.insert(
            view.name,
            views::View {
                offset,
                bytes: view.bytes,
                dtype: view.dtype.into(),
                shape: vec![view.bytes],
                layout_id: view.layout_id.into(),
            },
        );
        offset = (offset + view.bytes).next_multiple_of(4096);
    }
    index.image_bytes = offset;
    index
}

/// Components create admits that no importer produces are covered too: in
/// the NVFP4-routed layout's component, each MTP expert bank keeps its registered FP8 block
/// weights under BF16 weight reuse but records the main experts' static
/// NVFP4 source activation. Binding, the MTP admission and both view
/// validators accept it at TP2 and TP4, and the `input_scale` view it
/// requires (2048 and 1024 bytes at TP2, 1024 and 512 at TP4) is admitted.
#[test]
fn admitted_views_cover_static_scales_on_registered_fp8_mtp_experts() {
    use model_qwen38_contract::prepared;
    let published = precision::capabilities().unwrap();
    let inventory =
        CapabilityInventory::new(published.clone(), published.entries(), "0".repeat(64)).unwrap();
    let root = sparse_checkpoint(nvfp4_routed(), "nvfp4-routed-static-mtp");
    let source = SourcePrecision::from_captured(&catalog::capture(&root).unwrap()).unwrap();
    std::fs::remove_dir_all(&root).unwrap();
    for (ranks, gate_up, down) in [(Ranks::TP2, 2048, 1024), (Ranks::TP4, 1024, 512)] {
        let shape = Shape::flash(ranks);
        let document = projection_program::SourceProgram::new(&source, shape)
            .unwrap()
            .resolve(ActivationPolicy::Auto, &inventory)
            .unwrap();
        let calibrated = document
            .operations
            .iter()
            .flat_map(|s| &s.source.activations)
            .find(|a| {
                a.quantizer.as_ref().is_some_and(|q| {
                    matches!(
                        q.calibration,
                        kernel_core::precision::Calibration::Static(_)
                    )
                })
            })
            .unwrap()
            .clone();
        let drafter = prepared::drafter_operations(shape).unwrap();
        for (id, bytes) in drafter[drafter.len() - 2..].iter().zip([gate_up, down]) {
            let mut altered = document.clone();
            let selection = altered
                .operations
                .iter_mut()
                .find(|s| &s.source.id == id)
                .unwrap();
            assert!(matches!(
                selection.source.weights[0].format,
                kernel_core::contract::WeightFormat::Fp8Block128x128(_)
            ));
            assert!(selection.reuse.is_some());
            selection.source.activations.fill(calibrated.clone());
            altered.validate().unwrap();
            prepared::bind(&altered, shape, &inventory).unwrap();
            projections::support::mtp(shape, &altered).unwrap();
            let required = prepared::drafter_required(&altered, 0, shape).unwrap();
            let scale = required
                .iter()
                .find(|v| v.name == format!("{id}.input_scale"))
                .unwrap();
            assert_eq!(scale.bytes, bytes, "{id} TP{}", shape.ranks());
            for rank in 0..shape.ranks() {
                let index = exact_index(&altered, rank, shape);
                prepared::validate_views(&index, shape, &altered).unwrap();
                prepared::validate_drafter_views(&index, shape, &altered).unwrap();
                let admitted = prepared::admitted_views(rank, shape).unwrap();
                if let Some(view) = uncovered(&altered, &index, shape, admitted) {
                    panic!(
                        "{id} TP{} rank {rank}: the admitted table lacks {view}",
                        shape.ranks()
                    );
                }
            }
        }
        // Every packed layout the component binds is admitted.
        let layouts = &prepared::admitted(shape).unwrap().layouts;
        for selection in &document.operations {
            for weight in &selection.effective.weights {
                assert!(
                    layouts.contains(&(weight.layout.id.clone(), weight.layout.version)),
                    "{} {:?}",
                    selection.source.id,
                    weight.layout
                );
            }
        }
    }
}

/// The MXFP8-dense layout's whole precision component at TP4, its MTP
/// layer recorded, and the inventory it binds.
fn mxfp8_dense_document() -> (
    kernel_core::precision::execution::PrecisionDocument,
    CapabilityInventory,
) {
    let published = precision::capabilities().unwrap();
    let inventory =
        CapabilityInventory::new(published.clone(), published.entries(), "0".repeat(64)).unwrap();
    let root = sparse_checkpoint(mxfp8_dense(), "mxfp8-dense-mtp-bind");
    let source = SourcePrecision::from_captured(&catalog::capture(&root).unwrap()).unwrap();
    std::fs::remove_dir_all(&root).unwrap();
    let document = projection_program::SourceProgram::new(&source, Shape::TP4)
        .unwrap()
        .resolve(ActivationPolicy::Auto, &inventory)
        .unwrap();
    model_qwen38_contract::prepared::bind(&document, Shape::TP4, &inventory).unwrap();
    assert!(model_qwen38_contract::prepared::records_drafter(&document, Shape::TP4).unwrap());
    (document, inventory)
}

/// Binding refuses a component whose MTP dense port is not BF16 through
/// that port's declared composition, naming its operation: the MTP Q port
/// itself, and the K port that is a later fusion peer of a BF16 Q.
#[test]
fn bind_refuses_non_bf16_mtp_dense_ports_through_their_compositions() {
    let shape = Shape::TP4;
    let (document, inventory) = mxfp8_dense_document();
    let ports = projections::ports::get(shape).unwrap();
    let mxfp8 = document
        .operations
        .iter()
        .find(|s| s.source.weights[0].format == kernel_core::contract::WeightFormat::Mxfp8)
        .unwrap();
    let layer = shape.mtp_layer();
    for projection in ["mtp.q", "mtp.k"] {
        let id = ports.layer_operation(layer, projection).unwrap();
        let mut altered = document.clone();
        let selection = altered
            .operations
            .iter_mut()
            .find(|s| s.source.id == id)
            .unwrap();
        *selection = mxfp8.clone();
        selection.source.id = id.into();
        selection.effective.id = id.into();
        assert_eq!(
            model_qwen38_contract::prepared::bind(&altered, shape, &inventory).unwrap_err(),
            format!("operation {id}: {projection} composes no Mxfp8 MTP operation")
        );
    }
}

/// Binding refuses a well-formed component whose BF16 MTP Q port declares
/// a quantized activation, which the importer never records: the module
/// supplies no BF16-weight composition with a quantized activation, and the
/// refusal names the operation. The MTP layer has no separate legacy-BF16
/// rule.
#[test]
fn bind_refuses_a_bf16_mtp_dense_port_with_a_quantized_activation() {
    let shape = Shape::TP4;
    let (document, inventory) = mxfp8_dense_document();
    let quantized = document
        .operations
        .iter()
        .find(|s| {
            s.source.kind == kernel_core::precision::OperationKind::Dense
                && s.source.activations.iter().any(|a| !a.is_bf16())
        })
        .unwrap()
        .source
        .activations[0]
        .clone();
    let id = projections::ports::get(shape)
        .unwrap()
        .layer_operation(shape.mtp_layer(), "mtp.q")
        .unwrap();
    let mut altered = document.clone();
    let selection = altered
        .operations
        .iter_mut()
        .find(|s| s.source.id == id)
        .unwrap();
    assert_eq!(
        selection.source.weights[0].format,
        kernel_core::contract::WeightFormat::Bf16
    );
    selection.source.activations.fill(quantized.clone());
    selection.effective.activations.fill(quantized);
    for binding in &mut selection.bindings {
        binding.implementation.activations = selection.effective.activations.clone();
    }
    altered.validate().unwrap();
    let error = model_qwen38_contract::prepared::bind(&altered, shape, &inventory).unwrap_err();
    assert!(
        error.starts_with(&format!("operation {id}: "))
            && error.contains("requested Mxfp8 activation operands")
            && error.contains("supplied composition requires Bf16 operands"),
        "{error}"
    );
}

#[test]
#[ignore = "payload bytes on every rank; about 4 minutes, run with --ignored"]
fn served_dense_operations_and_rank_selections_match_the_current_compiler() {
    for (text, ranks) in [
        (FP8, Ranks::TP2),
        (FP8, Ranks::TP4),
        (nvfp4_routed(), Ranks::TP2),
        (nvfp4_routed(), Ranks::TP4),
        (mxfp8_dense(), Ranks::TP2),
        (mxfp8_dense(), Ranks::TP4),
    ] {
        let coverage = served_dense(text, ranks, Sweep::Every);
        let tp = Shape::flash(ranks).ranks();
        assert_eq!(coverage.keys, declared_keys(tp, &DENSE, 0..u64::MAX));
    }
}

fn served_dense(text: &str, ranks: Ranks, sweep: Sweep) -> Coverage {
    let mut coverage = Coverage::default();
    let shape = Shape::flash(ranks);
    let mut m = headers::fixture(text);
    let d = headers::owner_scope(&projections::declaration(shape.ranks()).unwrap(), &mut m);
    let g = projections::geometry(shape.ranks()).unwrap();
    let b = headers::bind(&d, &g, &m).unwrap();
    let d = d.expanded_profiles(&g).unwrap();
    let source = source(&m, shape);
    let record_catalog = source_catalog(&source, shape);
    let tensors = source
        .operands
        .iter()
        .map(|(name, operand)| {
            let (placement, scale) =
                catalog::classify_for(shape.geometry, name, shape.ranks() as usize).unwrap();
            (
                name.clone(),
                catalog::TensorRecord {
                    name: name.clone(),
                    dtype: runtime_weights::DType::Bf16,
                    shape: operand.weight.logical_shape.clone(),
                    file: String::new(),
                    file_start: 0,
                    bytes: operand.weight.codes.byte_length,
                    placement,
                    scale,
                },
            )
        })
        .collect();
    let catalog = catalog::Catalog {
        geometry: shape.geometry,
        tensors,
        ranks: shape.ranks() as usize,
        precision: Some(source.clone()),
    };
    let mut payload_cases = BTreeSet::new();
    let mut payloads: BTreeMap<_, (u64, _, _)> = BTreeMap::new();
    let mut capabilities = model_qwen38_contract::sm120::generated::inventory(shape)
        .unwrap()
        .document
        .operations;
    capabilities.extend(precision::legacy_capabilities_for(shape.geometry).unwrap());
    let document = CapabilityDocument {
        version: 1,
        operations: capabilities,
    };
    let inventory =
        CapabilityInventory::new(document.clone(), document.entries(), "0".repeat(64)).unwrap();
    for layer in 0..shape.layers() {
        for role in DenseRole::roles_for(shape, layer).unwrap() {
            let old = projection_program::dense(&source, layer, role, shape).unwrap();
            let id = projection(role, shape.is_full(layer));
            let p = d.projections.iter().find(|p| p.id == id).unwrap();
            // The QKV source has declared part windows. A4 supplies its legacy fusion.
            if !p.parts.is_empty() {
                continue;
            }
            let format = registry::by_weight(&old.source.weights[0]).unwrap().id();
            let recipe = p
                .composition
                .as_ref()
                .and_then(|c| c.numerical_contracts.get(&format));
            let mut arithmetic = old.source.arithmetic.clone();
            arithmetic.numerical_contract = recipe
                .map_or(&p.numerical_contract, |r| &r.numerical_contract)
                .clone();
            // A quantized export's own activation recipe (such as MXFP8's)
            // is admitted as the declared A16 BF16 operands.
            let mut activations = old.source.activations.clone();
            if !old.legacy {
                activations.fill(kernel_core::precision::ActivationSpec::bf16());
                arithmetic.weight_operand = DType::Bf16;
                arithmetic.decode_order = registry::by_id(format)
                    .unwrap()
                    .decode_order(
                        KernelFamily::Dense,
                        kernel_core::contract::ActivationFormat::Bf16,
                        recipe.map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe),
                    )
                    .unwrap();
            }
            let imported = ImportedProjection {
                operation_id: old.source.id.clone(),
                target: Selected::TARGET,
                family: KernelFamily::Dense,
                format,
                geometry: GeometryMode::RankLocal,
                weights: old.source.weights.clone(),
                activations,
                arithmetic,
                a16_recipe: recipe.map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe),
                matrices: old
                    .source
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(operand, w)| SourceMatrix {
                        weight: w.clone(),
                        expert: 0,
                        operand: operand as u32,
                        source_expert: 0,
                        target_rows: 0..w.logical_shape[0],
                        source_rows: 0..w.logical_shape[0],
                        target_k: 0..w.logical_shape[1],
                        source_k: 0..w.logical_shape[1],
                    })
                    .collect(),
                preparation: None,
            };
            let scoped = projection_spec::bind::BoundDeclaration {
                operands: b
                    .operands
                    .iter()
                    .filter(|o| {
                        old.source
                            .weights
                            .iter()
                            .any(|w| w.codes.tensor == o.imported.weight.codes.tensor)
                    })
                    .cloned()
                    .collect(),
                assemblies: b
                    .assemblies
                    .iter()
                    .filter(|a| a.projection == id && a.layer == Some(layer.into()))
                    .cloned()
                    .collect(),
                conditions: b.conditions.clone(),
                layer_aliases: b.layer_aliases.clone(),
                ..Default::default()
            };
            let selected = if (old.legacy
                && matches!(role, DenseRole::LinearA | DenseRole::LinearB))
                || p.composition
                    .as_ref()
                    .and_then(|c| c.implementation_for(format))
                    .is_some()
            {
                None
            } else {
                Some(old.resolve(ActivationPolicy::Auto, &inventory).unwrap())
            };
            if let Some(selected) = &selected
                && layer < 2
            {
                assert_planned_views_follow_the_contract(
                    &record_catalog,
                    selected.selection(),
                    shape,
                );
            }
            let templates = selected.as_ref().map(|selected| {
                crate::precision_packing::templates(
                    &catalog,
                    &kernel_core::precision::execution::PrecisionDocument {
                        version: 1,
                        operations: vec![selected.selection().clone()],
                    },
                )
                .unwrap()
            });
            for rank in 0..shape.ranks() {
                for (use_index, usage) in p.uses.iter().enumerate() {
                    coverage
                        .keys
                        .insert((Some(layer.into()), id.to_string(), rank, use_index));
                    let derived = ProjectionContext {
                        bound: &scoped,
                        source_instance: InstanceKey {
                            projection: id.into(),
                            layer: Some(layer.into()),
                        },
                        selections: None,
                        declaration: &d,
                        projection: p,
                        geometry: &g,
                        topology: Topology {
                            ranks: shape.ranks(),
                            rank,
                        },
                        layer: Some(layer.into()),
                        source_layer: Some(layer.into()),
                        usage,
                    }
                    .ordinary(&imported)
                    .unwrap_or_else(|e| panic!("{id}/{layer}/TP{}/rank{rank}: {e}", shape.ranks()));
                    for slice in &derived.slices {
                        let w = &slice.source.weight;
                        let stored = &source.inventory.tensors[&w.codes.tensor];
                        let (placement, scale) = catalog::classify_for(
                            shape.geometry,
                            &w.codes.tensor,
                            shape.ranks() as usize,
                        )
                        .unwrap();
                        let tensor = catalog::TensorRecord {
                            name: w.codes.tensor.clone(),
                            dtype: match stored.dtype {
                                DType::Bf16 => runtime_weights::DType::Bf16,
                                DType::U8 => runtime_weights::DType::U8,
                                _ => runtime_weights::DType::F8E4M3,
                            },
                            shape: w.logical_shape.clone(),
                            file: String::new(),
                            file_start: 0,
                            bytes: stored.byte_length,
                            placement,
                            scale,
                        };
                        let old_ranges = tensor.rank_slices_for(
                            shape.geometry,
                            rank as usize,
                            shape.ranks() as usize,
                        );
                        if old_ranges[0].0 == 0 {
                            assert_eq!(
                                slice.rows,
                                old_ranges.into_iter().map(|(_, r)| r).collect::<Vec<_>>(),
                                "{id}/{layer}"
                            );
                            assert_eq!(slice.k, 0..w.logical_shape[1]);
                        } else {
                            assert_eq!(slice.k, old_ranges[0].1, "{id}/{layer}");
                            assert_eq!(slice.rows, vec![0..w.logical_shape[0]]);
                        }
                        coverage.selections += 1;
                    }
                    if let Some(selected) = &selected {
                        let mut expected = selected.selection().effective.clone();
                        expected.uses = vec![derived.requested];
                        let actual = projection_program::DenseOperation {
                            source: derived.operation.clone(),
                            role,
                            shape,
                            legacy: old.legacy,
                        }
                        .resolve(ActivationPolicy::Auto, &inventory)
                        .unwrap();
                        assert_eq!(
                            actual.selection().effective,
                            expected,
                            "{id}/{layer}/TP{}/rank{rank}",
                            shape.ranks()
                        );
                        assert_eq!(
                            derived.operation.matrix_geometry().unwrap(),
                            precision::ports(shape)
                                .unwrap()
                                .weight_geometry(
                                    role.projection_id(shape, None),
                                    expected.weights[0].format
                                )
                                .unwrap()
                        );
                        assert_eq!(
                            derived.compiled_capacity,
                            precision::projection_capacity(derived.requested).unwrap()
                        );
                        if usage == &p.uses[0] && derived.operation.weights.len() == 1 {
                            let name = &imported.weights[0].codes.tensor;
                            let old_piece = templates.as_ref().unwrap()[name]
                                .place(&catalog, rank as usize)
                                .unwrap()
                                .unwrap();
                            assert_eq!(
                                derived.packing, old_piece.packing.plan,
                                "packing {id}/{layer}"
                            );
                            let old_record = old_piece
                                .packing
                                .record(&imported.weights[0], &old_piece.slices)
                                .unwrap();
                            let records =
                                model_compiler_sdk::quantization::projection::records(&derived)
                                    .unwrap();
                            assert_eq!(records[0].record, old_record, "record {id}/{layer}");
                            if let Some(rank) = payload_rank(rank, shape.ranks(), sweep) {
                                let class = match sweep {
                                    Sweep::Default => payload_class(&derived),
                                    Sweep::Every => format!("{id} {format:?}"),
                                };
                                let bytes = imported.weights[0].codes.byte_length;
                                let key = (class, rank);
                                if payloads.get(&key).is_none_or(|(size, ..)| bytes < *size) {
                                    payloads
                                        .insert(key, (bytes, derived.clone(), old_record.clone()));
                                }
                            }
                        }
                        if usage == &p.uses[0] && derived.operation.weights.len() == 2 {
                            let payload = payload_rank(rank, shape.ranks(), sweep)
                                .is_some_and(|rank| payload_cases.insert((id, format, rank)));
                            compare_served_records(
                                &derived,
                                &record_catalog,
                                actual.selection(),
                                rank as usize,
                                payload,
                                sweep,
                            );
                            if payload {
                                coverage.payloads.insert((id.to_string(), rank));
                            }
                        }
                        coverage.specs += 1;
                    }
                }
            }
        }
    }
    for ((_, rank), (_, derived, old_record)) in &payloads {
        compare_payload(derived, old_record, None);
        coverage
            .payloads
            .insert((derived.instance.projection.clone(), *rank));
    }
    coverage
}

fn compare_payload(
    operation: &projection_spec::derive::DerivedOperation,
    baseline: &kernel_core::weight_transform::WeightRecord,
    baseline_local: Option<&kernel_core::weight_transform::WeightRecord>,
) {
    let SyntheticPayload {
        payloads,
        packed: actual,
    } = synthetic_payload(operation);
    let expected = baseline
        .clone()
        .prepare(baseline.output_bytes().unwrap())
        .unwrap()
        .produce_whole(&mut |index, offset, out| {
            out.copy_from_slice(&payloads[index][offset as usize..offset as usize + out.len()]);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        actual.codes, expected,
        "payload {}",
        operation.instance.projection
    );
    if let Some(local) = baseline_local {
        let expected_local = local
            .clone()
            .prepare(local.output_bytes().unwrap())
            .unwrap()
            .produce_whole(&mut |_, offset, out| {
                out.copy_from_slice(&payloads[1][offset as usize..offset as usize + out.len()]);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            actual.local_scales, expected_local,
            "local scale payload {}",
            operation.instance.projection
        );
    }
    let records = model_compiler_sdk::quantization::projection::records(operation).unwrap();
    for record in records {
        // A record equal to the baseline produces the bytes compared above.
        if record.record == *baseline {
            assert_eq!(
                record.section,
                kernel_core::weight_transform::PackedSection::Codes
            );
            continue;
        }
        let bytes = record
            .record
            .clone()
            .prepare(record.record.output_bytes().unwrap())
            .unwrap()
            .produce_whole(&mut |index, offset, out| {
                out.copy_from_slice(&payloads[index][offset as usize..offset as usize + out.len()]);
                Ok(())
            })
            .unwrap();
        let expected = match record.section {
            kernel_core::weight_transform::PackedSection::Codes => &actual.codes,
            kernel_core::weight_transform::PackedSection::LocalScales => &actual.local_scales,
            kernel_core::weight_transform::PackedSection::GlobalScales => &actual.global_scales,
        };
        assert_eq!(
            &bytes, expected,
            "persistent payload {}",
            operation.instance.projection
        );
    }
}

/// The expert layers of each default expert test; together they cover the
/// bank stem, so every layer's OperationSpecs, selections, ownership, slices,
/// records and views are compared. Payload bytes run at layer 0.
const EXPERT_LAYERS: [std::ops::Range<u32>; 8] =
    [0..6, 6..12, 12..18, 18..24, 24..30, 30..36, 36..42, 42..48];

const EXPERTS: [&str; 2] = ["main.experts.gate-up", "main.experts.down"];

#[test]
fn expert_tests_cover_every_bank_layer() {
    let keys = declared_keys(4, &EXPERTS, 0..u64::MAX);
    let covered: BTreeSet<_> = EXPERT_LAYERS
        .iter()
        .flat_map(|layers| declared_keys(4, &EXPERTS, layers.start.into()..layers.end.into()))
        .collect();
    assert_eq!(covered, keys);
    assert_eq!(keys.len(), 48 * 2 * 4 * 19);
}

/// Check one default expert test against its declared keys, exact counts and
/// payload samples.
fn experts(chunk: usize, payloads: &[(String, u32)]) {
    let layers = EXPERT_LAYERS[chunk].clone();
    let coverage = served_native_mixed_experts(layers.clone(), Sweep::Default);
    assert_eq!(
        coverage.keys,
        declared_keys(4, &EXPERTS, layers.start.into()..layers.end.into())
    );
    let count = layers.len() * 2 * 4 * 19;
    assert_eq!(
        (coverage.keys.len(), coverage.specs, coverage.selections),
        (count, count, count / 2 * 128 * 3)
    );
    assert_eq!(coverage.payloads, payloads.iter().cloned().collect());
}

#[test]
fn served_native_mixed_expert_specs_and_rank_ownership_match_in_layers_0_to_6() {
    // Layer 0 compares the bytes of each section's first and last bank
    // member on the first and last rank: the first expert's gate, the last
    // one's up, and both downs.
    let mut payloads = Vec::new();
    for (rank, first, last) in [(0, 0, 127), (3, 384, 511)] {
        for member in [
            format!("{first}.gate_proj"),
            format!("{last}.up_proj"),
            format!("{first}.down_proj"),
            format!("{last}.down_proj"),
        ] {
            for section in ["weight", "weight_scale", "weight_scale_2"] {
                let name = format!("model.language_model.layers.0.mlp.experts.{member}.{section}");
                payloads.push((name, rank));
            }
        }
    }
    experts(0, &payloads);
}

#[test]
fn served_native_mixed_expert_specs_and_rank_ownership_match_in_layers_6_to_12() {
    experts(1, &[]);
}

#[test]
fn served_native_mixed_expert_specs_and_rank_ownership_match_in_layers_12_to_18() {
    experts(2, &[]);
}

#[test]
fn served_native_mixed_expert_specs_and_rank_ownership_match_in_layers_18_to_24() {
    experts(3, &[]);
}

#[test]
fn served_native_mixed_expert_specs_and_rank_ownership_match_in_layers_24_to_30() {
    experts(4, &[]);
}

#[test]
fn served_native_mixed_expert_specs_and_rank_ownership_match_in_layers_30_to_36() {
    experts(5, &[]);
}

#[test]
fn served_native_mixed_expert_specs_and_rank_ownership_match_in_layers_36_to_42() {
    experts(6, &[]);
}

#[test]
fn served_native_mixed_expert_specs_and_rank_ownership_match_in_layers_42_to_48() {
    experts(7, &[]);
}

#[test]
#[ignore = "payload bytes for every rank and bank member; about 4 minutes, run with --ignored"]
fn served_native_mixed_expert_specs_and_rank_ownership_match_the_current_compiler() {
    for ranks in [Ranks::TP2, Ranks::TP4] {
        let coverage = served_native_mixed_experts_at(ranks, 0..48, Sweep::Every);
        let tp = Shape::flash(ranks).ranks();
        assert_eq!(coverage.keys, declared_keys(tp, &EXPERTS, 0..u64::MAX));
    }
}

/// TP2 owns 256 experts per rank: layer 0 compares the bytes of the first
/// and last bank member of each section on both ranks, so the expert
/// 255/256 boundary between the ranks is compared byte for byte.
#[test]
fn tp2_native_mixed_expert_specs_and_rank_ownership_match_in_layers_0_to_6() {
    let layers = EXPERT_LAYERS[0].clone();
    let coverage = served_native_mixed_experts_at(Ranks::TP2, layers.clone(), Sweep::Default);
    assert_eq!(
        coverage.keys,
        declared_keys(2, &EXPERTS, layers.start.into()..layers.end.into())
    );
    let count = layers.len() * 2 * 2 * 19;
    assert_eq!(
        (coverage.keys.len(), coverage.specs, coverage.selections),
        (count, count, count / 2 * 256 * 3)
    );
    let mut payloads = BTreeSet::new();
    for (rank, first, last) in [(0, 0, 255), (1, 256, 511)] {
        for member in [
            format!("{first}.gate_proj"),
            format!("{last}.up_proj"),
            format!("{first}.down_proj"),
            format!("{last}.down_proj"),
        ] {
            for section in ["weight", "weight_scale", "weight_scale_2"] {
                let name = format!("model.language_model.layers.0.mlp.experts.{member}.{section}");
                payloads.insert((name, rank));
            }
        }
    }
    assert_eq!(coverage.payloads, payloads);
}

fn served_native_mixed_experts(layers: std::ops::Range<u32>, sweep: Sweep) -> Coverage {
    served_native_mixed_experts_at(Ranks::TP4, layers, sweep)
}

/// [`served_native_mixed_experts`] at `ranks`.
fn served_native_mixed_experts_at(
    ranks: Ranks,
    layers: std::ops::Range<u32>,
    sweep: Sweep,
) -> Coverage {
    use model_compiler_sdk::quantization::resolve_operation;
    use model_qwen38_contract::precision::Projection;
    let shape = Shape::flash(ranks);
    let tp = shape.ranks();
    let mut m = headers::fixture(nvfp4_routed());
    let d = headers::owner_scope(&projections::declaration(tp).unwrap(), &mut m);
    let g = projections::geometry(tp).unwrap();
    let b = headers::bind(&d, &g, &m).unwrap();
    let d = d.expanded_profiles(&g).unwrap();
    let names = b
        .operands
        .iter()
        .filter(|o| o.projection.starts_with("main.experts."))
        .map(|o| o.imported.weight.codes.tensor.clone())
        .collect();
    let source = source_for_names(&m, shape, &names);
    let record_catalog = source_catalog(&source, shape);
    let document = CapabilityDocument {
        version: 1,
        operations: model_qwen38_contract::sm120::generated::inventory(shape)
            .unwrap()
            .document
            .operations,
    };
    let inventory =
        CapabilityInventory::new(document.clone(), document.entries(), "0".repeat(64)).unwrap();
    let mut coverage = Coverage::default();
    for layer in layers {
        for (id, projection) in [
            ("main.experts.gate-up", Projection::GateUp),
            ("main.experts.down", Projection::Down),
        ] {
            let p = d.projections.iter().find(|p| p.id == id).unwrap();
            let operands: Vec<_> = b
                .operands
                .iter()
                .filter(|o| o.projection == id && o.layer == Some(layer.into()))
                .cloned()
                .collect();
            let old = projection_program::routed(&source, layer, projection, shape).unwrap();
            let format = registry::by_weight(&old.source.weights[0]).unwrap().id();
            assert!(
                p.composition
                    .as_ref()
                    .unwrap()
                    .implementation_for(format)
                    .is_none(),
                "native {format:?}"
            );
            let expected = model_compiler_sdk::quantization::resolve_operation(
                &old.source,
                ActivationPolicy::Bf16,
                &inventory,
            )
            .unwrap();
            if layer == 0 {
                assert_planned_views_follow_the_contract(
                    &record_catalog,
                    expected.selection(),
                    shape,
                );
            }
            let mut admitted = old.source.clone();
            admitted
                .activations
                .fill(kernel_core::precision::ActivationSpec::bf16());
            admitted.arithmetic.weight_operand = DType::Bf16;
            admitted.arithmetic.numerical_contract =
                p.composition.as_ref().unwrap().numerical_contracts[&format]
                    .numerical_contract
                    .clone();
            admitted.arithmetic.decode_order = registry::by_id(format)
                .unwrap()
                .decode_order(
                    KernelFamily::GroupedExpert,
                    kernel_core::contract::ActivationFormat::Bf16,
                    A16Recipe::ScaledWeight,
                )
                .unwrap();
            let imported = ImportedProjection {
                operation_id: admitted.id.clone(),
                target: admitted.target,
                family: KernelFamily::GroupedExpert,
                format,
                geometry: GeometryMode::RankLocal,
                weights: admitted.weights.clone(),
                activations: admitted.activations.clone(),
                arithmetic: admitted.arithmetic.clone(),
                a16_recipe: A16Recipe::ScaledWeight,
                matrices: operands
                    .iter()
                    .map(|o| {
                        let w = &source.operands[&o.imported.weight.codes.tensor].weight;
                        assert_eq!(w, &o.imported.weight);
                        SourceMatrix {
                            weight: w.clone(),
                            expert: o.expert.unwrap() as u32,
                            operand: o.operand.unwrap_or(0) as u32,
                            source_expert: 0,
                            target_rows: 0..w.logical_shape[0],
                            source_rows: 0..w.logical_shape[0],
                            target_k: 0..w.logical_shape[1],
                            source_k: 0..w.logical_shape[1],
                        }
                    })
                    .collect(),
                preparation: None,
            };
            let scope = projection_spec::bind::BoundDeclaration {
                operands,
                conditions: b.conditions.clone(),
                ..Default::default()
            };
            for rank in 0..tp {
                for (use_index, usage) in p.uses.iter().enumerate() {
                    coverage
                        .keys
                        .insert((Some(layer.into()), id.to_string(), rank, use_index));
                    let derived = ProjectionContext {
                        bound: &scope,
                        source_instance: InstanceKey {
                            projection: id.into(),
                            layer: Some(layer.into()),
                        },
                        selections: None,
                        declaration: &d,
                        projection: p,
                        geometry: &g,
                        topology: Topology { ranks: tp, rank },
                        layer: Some(layer.into()),
                        source_layer: Some(layer.into()),
                        usage,
                    }
                    .ordinary(&imported)
                    .unwrap_or_else(|e| panic!("{id}/{layer}/rank{rank}: {e}"));
                    let mut expected_source = admitted.clone();
                    expected_source.uses = vec![derived.requested];
                    assert_eq!(derived.operation, expected_source);
                    let actual =
                        resolve_operation(&derived.operation, ActivationPolicy::Bf16, &inventory)
                            .unwrap();
                    let mut expected = expected.selection().effective.clone();
                    expected.uses = vec![derived.requested];
                    assert_eq!(actual.selection().effective, expected);
                    if usage == &p.uses[0] {
                        let compared = compare_served_records(
                            &derived,
                            &record_catalog,
                            actual.selection(),
                            rank as usize,
                            layer == 0 && payload_rank(rank, tp, sweep).is_some(),
                            sweep,
                        );
                        coverage
                            .payloads
                            .extend(compared.into_iter().map(|name| (name, rank)));
                    }
                    for slice in &derived.slices {
                        let w = &slice.source.weight;
                        let (placement, scale) =
                            catalog::classify_for(shape.geometry, &w.codes.tensor, tp as usize)
                                .unwrap();
                        let tensor = catalog::TensorRecord {
                            name: w.codes.tensor.clone(),
                            dtype: runtime_weights::DType::U8,
                            shape: w.logical_shape.clone(),
                            file: String::new(),
                            file_start: 0,
                            bytes: w.codes.byte_length,
                            placement,
                            scale,
                        };
                        assert_eq!(
                            tensor.rank_slices_for(shape.geometry, rank as usize, tp as usize),
                            vec![(0, 0..w.logical_shape[0])]
                        );
                        assert_eq!(slice.rows, vec![0..w.logical_shape[0]]);
                        assert_eq!(slice.k, 0..w.logical_shape[1]);
                        assert_eq!(slice.source.expert / shape.local_experts, rank);
                        assert_eq!(slice.expert, slice.source.expert % shape.local_experts);
                        coverage.selections += 1;
                    }
                    coverage.specs += 1;
                }
            }
        }
    }
    coverage
}

/// Construct physical headers with the same companion-role policy as admission.
fn source_catalog(source: &SourcePrecision, shape: Shape) -> catalog::Catalog {
    let roles = catalog::precision_roles(source, shape.ranks() as usize, shape.geometry).unwrap();
    let tensors = roles
        .into_iter()
        .map(|(name, (placement, scale))| {
            let span = &source.inventory.tensors[&name];
            let record = catalog::TensorRecord {
                name: name.clone(),
                dtype: match span.dtype {
                    DType::Bf16 => runtime_weights::DType::Bf16,
                    DType::F32 => runtime_weights::DType::F32,
                    DType::U8 => runtime_weights::DType::U8,
                    DType::Fp8E4M3 => runtime_weights::DType::F8E4M3,
                    DType::Ue8M0 => runtime_weights::DType::F8E8M0,
                    other => panic!("unexpected source dtype {other:?}"),
                },
                shape: if span.shape.is_empty() {
                    vec![1]
                } else {
                    span.shape.clone()
                },
                bytes: span.byte_length,
                file: String::new(),
                file_start: 0,
                placement,
                scale,
            };
            (name, record)
        })
        .collect();
    catalog::Catalog {
        geometry: shape.geometry,
        tensors,
        ranks: shape.ranks() as usize,
        precision: Some(source.clone()),
    }
}

fn compare_served_records(
    operation: &projection_spec::derive::DerivedOperation,
    catalog: &catalog::Catalog,
    selection: &kernel_core::precision::execution::PrecisionSelection,
    rank: usize,
    payload: bool,
    sweep: Sweep,
) -> Vec<String> {
    use kernel_core::weight_transform::PackedSection;
    use model_compiler_sdk::quantization::projection::{bind_record_policy, records};
    let mut compared = Vec::new();
    let names: BTreeSet<_> = operation
        .slices
        .iter()
        .flat_map(|s| {
            let w = &s.source.weight;
            std::iter::once(w.codes.tensor.clone())
                .chain(w.local_scale.iter().map(|s| s.tensor.tensor.clone()))
                .chain(w.global_scale.iter().map(|s| s.tensor.tensor.clone()))
        })
        .collect();
    let original = catalog.precision.as_ref().unwrap();
    let mut operands: BTreeMap<_, _> = original
        .operands
        .iter()
        .filter(|(n, _)| names.contains(*n))
        .map(|(n, o)| (n.clone(), o.clone()))
        .collect();
    if !original.legacy_text_layout() {
        let (name, marker) = original
            .operands
            .iter()
            .find(|(_, o)| {
                !matches!(
                    o.weight.format,
                    kernel_core::contract::WeightFormat::Bf16
                        | kernel_core::contract::WeightFormat::Fp8Block128x128(_)
                        | kernel_core::contract::WeightFormat::Fp8PerTensor
                )
            })
            .unwrap();
        operands.insert(name.clone(), marker.clone());
    }
    let scoped_source = SourcePrecision {
        operands,
        metadata: original.metadata.clone(),
        inventory: TensorInventory {
            tensors: original
                .inventory
                .tensors
                .iter()
                .filter(|(n, _)| names.contains(*n))
                .map(|(n, s)| (n.clone(), s.clone()))
                .collect(),
            spans: BTreeMap::new(),
        },
    };
    let scoped = catalog::Catalog {
        tensors: catalog
            .tensors
            .iter()
            .filter(|(n, _)| names.contains(*n))
            .map(|(n, t)| (n.clone(), t.clone()))
            .collect(),
        geometry: catalog.geometry,
        ranks: catalog.ranks,
        precision: Some(scoped_source),
    };
    let plan = crate::placement::ImagePlan::with_precision(
        &scoped,
        None,
        &kernel_core::precision::execution::PrecisionDocument {
            version: 1,
            operations: vec![selection.clone()],
        },
    )
    .unwrap();
    let operation = bind_record_policy(
        operation.clone(),
        crate::precision_packing::projection_record_policy(original),
    )
    .unwrap();
    let actual = records(&operation).unwrap();
    let slices = operation.slices.as_slice();
    let mut count = 0;
    for section in [
        PackedSection::Codes,
        PackedSection::LocalScales,
        PackedSection::GlobalScales,
    ] {
        let selected: Vec<_> = actual.iter().filter(|r| r.section == section).collect();
        if selected.is_empty() {
            continue;
        }
        let mut baseline: Vec<_> = plan
            .sources
            .iter()
            .flat_map(|(name, pieces)| {
                pieces
                    .iter()
                    .filter(move |p| p.rank == rank && selected_names(slices, section, name))
                    .map(move |piece| (name, piece))
            })
            .collect();
        baseline.sort_by_key(|(_, p)| p.offset);
        assert_eq!(selected.len(), baseline.len(), "record count {:?}", section);
        let view = &operation
            .views
            .iter()
            .find(|v| v.role == projection_spec::derive::ViewRole::Weight(section))
            .unwrap();
        let loader_view = &plan.ranks[rank].views[baseline[0].1.view];
        assert_eq!(view.bytes, loader_view.bytes);
        assert_eq!(view.layout, loader_view.layout_id);
        let last = selected.len() - 1;
        for (index, (record, (name, piece))) in selected.iter().zip(baseline).enumerate() {
            let (old, tensors) = if let Some(packing) = &piece.packing {
                let weight = &catalog.precision.as_ref().unwrap().operands[&packing.operand].weight;
                (
                    packing.record(weight, &piece.slices).unwrap(),
                    std::iter::once(weight.codes.tensor.clone())
                        .chain(weight.local_scale.iter().map(|s| s.tensor.tensor.clone()))
                        .chain(weight.global_scale.iter().map(|s| s.tensor.tensor.clone()))
                        .collect::<Vec<_>>(),
                )
            } else {
                (
                    non_packed_baseline(catalog, name, piece, &loader_view.layout_id),
                    vec![name.clone()],
                )
            };
            assert_eq!(record.tensors, tensors);
            assert_eq!(record.view_offset, piece.offset - loader_view.offset);
            // The shared section records check the scales they copy; the
            // loader's plain copy stores the same bytes.
            let mut derived = record.record.clone();
            if derived.transform == kernel_core::weight_transform::WeightTransform::CopyScales {
                derived.transform = kernel_core::weight_transform::WeightTransform::Copy;
            }
            assert_eq!(derived, old, "record {name}");
            assert_eq!(
                derived.transform.encode_params().unwrap(),
                old.transform.encode_params().unwrap()
            );
            // Bank members repeat one recipe; the default compares the bytes
            // of the first and last member of each section.
            if payload && (sweep == Sweep::Every || index == 0 || index == last) {
                compare_record_bytes(&operation, record, &old, &tensors);
                compared.push(name.clone());
            }
            count += 1;
        }
    }
    assert_eq!(count, actual.len());
    compared
}

/// The rank indexes a mixed image plan writes for one prepared operation
/// follow the bundle's prepared view contract: every view the contract
/// requires has its dtype, layout and length. The fake host builds its
/// indexes from that contract, so only a planned index shows a placement
/// that disagrees with it.
fn assert_planned_views_follow_the_contract(
    catalog: &catalog::Catalog,
    selection: &kernel_core::precision::execution::PrecisionSelection,
    shape: Shape,
) {
    let bindings = model_qwen38_contract::projections::ports::get(shape)
        .unwrap()
        .bindings()
        .unwrap();
    let members: BTreeSet<_> = bindings
        .iter()
        .filter(|(_, b)| b.operation == selection.source.id)
        .map(|(name, _)| name.trim_end_matches(".weight").to_owned())
        .collect();
    let owned = |name: &str| {
        members.iter().any(|member| {
            name.strip_prefix(member.as_str())
                .is_some_and(|rest| rest.starts_with('.'))
        })
    };
    let original = catalog.precision.as_ref().unwrap();
    let scoped = catalog::Catalog {
        tensors: catalog
            .tensors
            .iter()
            .filter(|(n, _)| owned(n))
            .map(|(n, t)| (n.clone(), t.clone()))
            .collect(),
        geometry: catalog.geometry,
        ranks: catalog.ranks,
        precision: Some(SourcePrecision {
            operands: original
                .operands
                .iter()
                .filter(|(n, _)| owned(n))
                .map(|(n, o)| (n.clone(), o.clone()))
                .collect(),
            metadata: original.metadata.clone(),
            inventory: TensorInventory {
                tensors: original
                    .inventory
                    .tensors
                    .iter()
                    .filter(|(n, _)| owned(n))
                    .map(|(n, s)| (n.clone(), s.clone()))
                    .collect(),
                spans: BTreeMap::new(),
            },
        }),
    };
    let document = kernel_core::precision::execution::PrecisionDocument {
        version: 1,
        operations: vec![selection.clone()],
    };
    let plan = crate::placement::ImagePlan::with_precision(&scoped, None, &document).unwrap();
    for (rank, index) in plan.ranks.iter().enumerate() {
        let required: BTreeMap<_, _> =
            model_qwen38_contract::prepared::required(&document, rank as u32, shape)
                .unwrap()
                .into_iter()
                .map(|r| (r.name.clone(), r))
                .collect();
        // Every planned view is one the contract requires, and every view
        // the contract requires of the operation is planned: codes and
        // scales alike.
        for view in &index.views {
            let contract = required.get(&view.name).unwrap_or_else(|| {
                panic!(
                    "{} rank {rank} plans view {} the contract does not require",
                    selection.source.id, view.name
                )
            });
            assert_eq!(
                (view.dtype.as_str(), view.layout_id.as_str(), view.bytes),
                (contract.dtype, contract.layout_id, contract.bytes),
                "{} rank {rank} view {}",
                selection.source.id,
                view.name
            );
        }
        let planned: BTreeSet<_> = index.views.iter().map(|v| v.name.as_str()).collect();
        let operation = format!("{}.", selection.source.id);
        let owned_required: Vec<_> = required
            .keys()
            .filter(|name| owned(name) || name.starts_with(&operation))
            .collect();
        assert!(
            !owned_required.is_empty(),
            "{} rank {rank}",
            selection.source.id
        );
        for name in owned_required {
            assert!(
                planned.contains(name.as_str()),
                "{} rank {rank} plans no required view {name}",
                selection.source.id
            );
        }
    }
}

fn selected_names(
    slices: &[projection_spec::derive::OperandSlice],
    section: kernel_core::weight_transform::PackedSection,
    name: &str,
) -> bool {
    use kernel_core::weight_transform::PackedSection;
    slices.iter().any(|s| match section {
        PackedSection::Codes => s.source.weight.codes.tensor == name,
        PackedSection::LocalScales => s
            .source
            .weight
            .local_scale
            .as_ref()
            .is_some_and(|s| s.tensor.tensor == name),
        PackedSection::GlobalScales => s
            .source
            .weight
            .global_scale
            .as_ref()
            .is_some_and(|s| s.tensor.tensor == name),
    })
}

fn derived_piece_bytes(
    slice: &projection_spec::derive::OperandSlice,
    packing: &kernel_core::precision::layout::PackingPlan,
    section: kernel_core::weight_transform::PackedSection,
) -> Vec<u8> {
    use kernel_core::precision::packed::MatrixSlice;
    let matrix = selected_source_payload_with(slice, section, synthetic_source);
    let mut plan = packing.clone();
    plan.experts = 1;
    plan.projections = 1;
    let mut builder =
        model_compiler_sdk::quantization::packing::PackedWeightBuilder::new(plan).unwrap();
    builder
        .write_matrix(
            0,
            0,
            matrix.view("derived"),
            &MatrixSlice {
                rows: (0..matrix.rows).collect(),
                k_start: 0,
                k: matrix.k,
                padded_rows: packing.rows as usize,
                padded_k: packing.k as usize,
                nibble_order: matrix.nibble_order,
            },
        )
        .unwrap();
    let output = builder.finish().unwrap();
    match section {
        kernel_core::weight_transform::PackedSection::Codes => output.codes,
        kernel_core::weight_transform::PackedSection::LocalScales => output.local_scales,
        kernel_core::weight_transform::PackedSection::GlobalScales => output.global_scales,
    }
}
/// Compare loader execution with separately gathered and packed derivation bytes.
fn compare_record_bytes(
    operation: &projection_spec::derive::DerivedOperation,
    record: &projection_spec::derive::VariantRecord,
    old: &kernel_core::weight_transform::WeightRecord,
    names: &[String],
) {
    let slice = operation
        .slices
        .iter()
        .find(|s| {
            s.source.weight.codes.tensor == names[0]
                || s.source
                    .weight
                    .local_scale
                    .as_ref()
                    .is_some_and(|g| g.tensor.tensor == names[0])
                || s.source
                    .weight
                    .global_scale
                    .as_ref()
                    .is_some_and(|g| g.tensor.tensor == names[0])
        })
        .unwrap();
    let actual = derived_piece_bytes(slice, &operation.packing, record.section);
    assert_eq!(actual.len() as u64, record.record.output_bytes().unwrap());
    let weight = &slice.source.weight;
    let sources = std::iter::once(&weight.codes)
        .chain(weight.local_scale.iter().map(|s| &s.tensor))
        .chain(weight.global_scale.iter().map(|s| &s.tensor))
        .collect::<Vec<_>>();
    let sources = names
        .iter()
        .map(|name| *sources.iter().find(|s| s.tensor == *name).unwrap())
        .collect::<Vec<_>>();
    let expected = old
        .clone()
        .prepare(old.output_bytes().unwrap())
        .unwrap()
        .produce_whole(&mut |index, offset, out| {
            fill_source(sources[index], offset, out);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, expected, "independent synthetic payload {names:?}");
}

#[test]
fn unbaselined_header_families_pin_fixed_bf16_and_converted_mtp_banks() {
    for (text, converted) in [
        (FP8, "F8_E4M3"),
        (nvfp4_routed(), "F8_E4M3"),
        (mxfp8_dense(), "U8"),
    ] {
        let fixture: serde_json::Value = serde_json::from_str(text).unwrap();
        let mut fixed = 0;
        let mut expert_sources = 0;
        for family in fixture["families"].as_array().unwrap() {
            let name = family["tensor"].as_str().unwrap();
            let count: u64 = family["axes"]
                .as_object()
                .unwrap()
                .values()
                .map(|axis| {
                    if let Some(indices) = axis.get("indices") {
                        indices.as_array().unwrap().len() as u64
                    } else {
                        axis["range"][1].as_u64().unwrap() - axis["range"][0].as_u64().unwrap()
                    }
                })
                .product();
            if !name.ends_with(".weight") || family["shape"].as_array().unwrap().len() != 2 {
                continue;
            }
            if name.starts_with("mtp.") && name.contains(".experts.") {
                assert_eq!(family["dtype"], converted, "{name}");
                assert_eq!(
                    family["axes"]["{axis0}"]["range"],
                    serde_json::json!([0, 1])
                );
                assert_eq!(
                    family["axes"]["{axis1}"]["range"],
                    serde_json::json!([0, 512])
                );
                expert_sources += count;
            } else if name.contains("hyper_connection")
                || name.ends_with(".mlp.gate.weight")
                || name.ends_with(".shared_expert_gate.weight")
                || name.contains(".ple.key_proj.")
                || name.contains(".ple.value_proj.")
                || name == "lm_head.weight"
                || name == "model.language_model.embed_tokens.weight"
                || name.starts_with("mtp.")
            {
                assert_eq!(family["dtype"], "BF16", "{name}");
                fixed += count;
            }
        }
        // The embedding has a lookup baseline; retain it as a format exception.
        assert_eq!(fixed, 410);
        assert_eq!(expert_sources, 1536);
    }
}

#[test]
fn record_policy_follows_authenticated_source_precision() {
    use model_compiler_sdk::quantization::{format::RecordLayout, projection::RecordPolicy};
    let names =
        BTreeSet::from(["model.language_model.layers.0.mlp.experts.0.gate_proj.weight".into()]);
    for (text, expected) in [
        (FP8, RecordPolicy::Sections(RecordLayout::Fragment)),
        (nvfp4_routed(), RecordPolicy::PackedPieces),
    ] {
        let headers = headers::fixture(text);
        let source = source_for_names(&headers, Shape::TP4, &names);
        assert_eq!(
            crate::precision_packing::projection_record_policy(&source),
            expected
        );
    }
}

fn non_packed_baseline(
    catalog: &catalog::Catalog,
    name: &str,
    piece: &crate::placement::Piece,
    layout: &str,
) -> kernel_core::weight_transform::WeightRecord {
    use kernel_core::weight_transform::{SourceTensor, WeightRecord, WeightTransform};
    use model_qwen38_contract::{catalog as native, placement as native_placement};
    use runtime_bundle_abi::weight_plan::{CheckpointFileV1, WeightPlanRequestV1};
    let t = &catalog.tensors[name];
    let placement = match &t.placement {
        catalog::Placement::Replicated => native::Placement::Replicated,
        catalog::Placement::Axis { axis, parts } => native::Placement::Axis {
            axis: *axis,
            parts: *parts,
        },
        catalog::Placement::LinearQkv => native::Placement::LinearQkv,
        catalog::Placement::Expert { rank, local_expert } => native::Placement::Expert {
            rank: *rank,
            local_expert: *local_expert,
        },
        catalog::Placement::Ngram { rows } => native::Placement::Ngram { rows: rows.clone() },
    };
    let scale = match &t.scale {
        catalog::ScaleLayout::None => native::ScaleLayout::None,
        catalog::ScaleLayout::Block128 { scale } => native::ScaleLayout::Block128 {
            scale: scale.clone(),
        },
        catalog::ScaleLayout::PerTensor { scale } => native::ScaleLayout::PerTensor {
            scale: scale.clone(),
        },
        catalog::ScaleLayout::BlockInverseScale => native::ScaleLayout::BlockInverseScale,
        catalog::ScaleLayout::TensorScale => native::ScaleLayout::TensorScale,
        catalog::ScaleLayout::QuantizedWeight {
            format,
            local_scale,
            global_scale,
        } => native::ScaleLayout::QuantizedWeight {
            format: *format,
            local_scale: local_scale.clone(),
            global_scale: global_scale.clone(),
        },
        catalog::ScaleLayout::QuantizedBlockScale { format } => {
            native::ScaleLayout::QuantizedBlockScale { format: *format }
        }
        catalog::ScaleLayout::GlobalScale => native::ScaleLayout::GlobalScale,
        catalog::ScaleLayout::ActivationScale => native::ScaleLayout::ActivationScale,
    };
    let tensor = native::TensorRecord {
        name: t.name.clone(),
        dtype: t.dtype,
        shape: t.shape.clone(),
        file: t.file.clone(),
        file_start: t.file_start,
        bytes: t.bytes,
        placement,
        scale,
    };
    let native_catalog = native::Catalog {
        geometry: catalog.geometry,
        ranks: catalog.ranks,
        precision: None,
        tensors: BTreeMap::new(),
    };
    let piece = native_placement::Piece {
        rank: piece.rank,
        view: piece.view,
        offset: piece.offset,
        bytes: piece.bytes,
        slices: piece.slices.clone(),
        shape: piece.shape.clone(),
        packing: None,
    };
    let request = WeightPlanRequestV1 {
        image_role: String::new(),
        ranks: catalog.ranks as u32,
        rank: piece.rank as u32,
        checkpoint_identity: [0; 32],
        implementation: Default::default(),
        files: vec![CheckpointFileV1 {
            name: t.file.clone(),
            byte_length: 0,
            data_offset: 0,
            header: b"{}".to_vec(),
        }],
        documents: vec![],
        components: vec![],
        host_views: vec![],
    };
    let files = model_compiler_sdk::weight_plan::RequestFiles::new(&request);
    let produced = model_qwen38_contract::weight_plan::record(
        &native_catalog,
        &files,
        &tensor,
        &piece,
        layout,
    )
    .unwrap();
    assert_eq!(produced.byte_length, piece.bytes);
    assert_eq!(produced.arena_offset, piece.offset);
    WeightRecord {
        transform: WeightTransform::decode(produced.transform, &produced.params).unwrap(),
        sources: produced
            .sources
            .iter()
            .map(|source| {
                let t = &catalog.tensors[&source.tensor];
                SourceTensor {
                    dtype: t.dtype.safetensors_name().into(),
                    shape: catalog
                        .precision
                        .as_ref()
                        .and_then(|s| s.inventory.tensors.get(&source.tensor))
                        .map_or_else(|| t.shape.clone(), |s| s.shape.clone()),
                    byte_length: t.bytes,
                }
            })
            .collect(),
        axis: produced.axis,
        ranges: produced.ranges,
        stored_shape: produced.stored_shape,
    }
}

#[test]
fn non_packed_loader_baseline_preserves_bf16_scale_widening() {
    use kernel_core::weight_transform::WeightTransform;
    let tensor = catalog::TensorRecord {
        name: "matrix.weight_scale_inv".into(),
        dtype: runtime_weights::DType::Bf16,
        shape: vec![2, 2],
        bytes: 8,
        file: "source.safetensors".into(),
        file_start: 0,
        placement: catalog::Placement::Replicated,
        scale: catalog::ScaleLayout::BlockInverseScale,
    };
    let catalog = catalog::Catalog {
        geometry: Shape::TP4.geometry,
        ranks: 4,
        precision: None,
        tensors: BTreeMap::from([(tensor.name.clone(), tensor.clone())]),
    };
    let piece = crate::placement::Piece {
        rank: 0,
        view: 0,
        offset: 0,
        bytes: 16,
        slices: vec![(0, 0..2)],
        shape: vec![2, 2],
        packing: None,
    };
    let layout = crate::placement::layout(&tensor);
    let record = non_packed_baseline(&catalog, &tensor.name, &piece, &layout);
    assert_eq!(record.transform, WeightTransform::WidenBf16F32);
    assert_eq!(record.output_bytes().unwrap(), 16);
    let payload = [0x3f80u16, 0x4000, 0x4040, 0x4080]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let result = record
        .prepare(16)
        .unwrap()
        .produce_whole(&mut |_, offset, out| {
            out.copy_from_slice(&payload[offset as usize..offset as usize + out.len()]);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        result,
        [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn independent_payload_proof_observes_changed_rows_with_records_held_fixed() {
    use kernel_core::contract::{SwigluColumns, WeightFormat};
    use kernel_core::precision::{CodeEncoding, MatrixGeometry, WeightLayout, WeightSpec};
    use kernel_core::weight_transform::{
        PackedSection, SourceTensor, WeightRecord, WeightTransform,
    };
    let weight = WeightSpec {
        codes: TensorSpan {
            tensor: "matrix.weight".into(),
            dtype: DType::Bf16,
            shape: vec![8, 16],
            offset: 0,
            byte_length: 256,
        },
        logical_shape: vec![8, 16],
        format: WeightFormat::Bf16,
        encoding: CodeEncoding::Bf16LittleEndian,
        layout: WeightLayout::row_major(),
        local_scale: None,
        global_scale: None,
        transforms: vec![],
    };
    let source = SourceMatrix {
        weight: weight.clone(),
        expert: 0,
        operand: 0,
        source_expert: 0,
        target_rows: 0..8,
        source_rows: 0..8,
        target_k: 0..16,
        source_k: 0..16,
    };
    let mut slice = projection_spec::derive::OperandSlice {
        source,
        format: kernel_core::precision::format::FormatId::Bf16,
        rows: vec![4..8],
        k: 0..16,
        target_rows: vec![0..4],
        target_k: 0..16,
        expert: 0,
        operand: 0,
        geometry: MatrixGeometry { rows: 64, k: 16 },
    };
    let plan = kernel_core::precision::layout::PackingPlan {
        layout: WeightLayout {
            id: "bf16-fragment".into(),
            version: 1,
        },
        format: WeightFormat::Bf16,
        experts: 1,
        projections: 1,
        rows: 64,
        k: 16,
        columns: SwigluColumns::ConcatenatedHalves,
        local_scale_dtype: None,
    };
    let record = WeightRecord {
        transform: WeightTransform::Fragment(
            kernel_core::transform::fragment::FragmentWeightFormat::Bf16,
        ),
        sources: vec![SourceTensor {
            dtype: "BF16".into(),
            shape: vec![8, 16],
            byte_length: 256,
        }],
        axis: Some(0),
        ranges: vec![4..8],
        stored_shape: vec![64, 16],
    };
    let payload = synthetic_source(&weight.codes);
    let loader = record
        .clone()
        .prepare(record.output_bytes().unwrap())
        .unwrap()
        .produce_whole(&mut |_, offset, out| {
            out.copy_from_slice(&payload[offset as usize..offset as usize + out.len()]);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        derived_piece_bytes(&slice, &plan, PackedSection::Codes),
        loader
    );
    use kernel_core::precision::{
        ActivationSpec, ArithmeticSpec, DecodeOrder, ExecutionPhase, OperationKind, OperationSpec,
        OperationUse, Rounding,
    };
    let usage = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 1,
        graph: true,
    };
    let identity = InstanceKey {
        projection: "matrix".into(),
        layer: None,
    };
    let mut operation = projection_spec::derive::DerivedOperation {
        instance: identity.clone(),
        source_layer: None,
        source_instance: identity,
        requested: usage,
        compiled_capacity: 1,
        row_multiplier: 1,
        chunk_rows: None,
        per_item_verify: None,
        aggregate_verify: None,
        exact_rows: false,
        operation: OperationSpec {
            id: "matrix".into(),
            kind: OperationKind::Dense,
            target: Selected::TARGET,
            weights: vec![weight.clone()],
            geometry: Some(slice.geometry),
            activations: vec![ActivationSpec::bf16()],
            arithmetic: ArithmeticSpec {
                weight_operand: DType::Bf16,
                accumulator: DType::F32,
                output: DType::Bf16,
                decode_order: DecodeOrder::ScaleThenCode,
                rounding: Rounding::NearestEven,
                numerical_contract: "matrix.v1".into(),
            },
            epilogues: Default::default(),
            uses: vec![usage],
        },
        logical_geometry: MatrixGeometry { rows: 4, k: 16 },
        family: KernelFamily::Dense,
        a16_recipe: A16Recipe::ScaledWeight,
        slices: vec![slice.clone()],
        packing: plan.clone(),
        views: vec![],
        stages: vec![],
        inputs: vec![],
        variant_records: vec![],
        record_sources: vec![weight.codes.clone()],
        source_preparation: None,
    };
    let emitted = projection_spec::derive::VariantRecord {
        view: "matrix.codes".into(),
        view_offset: 0,
        section: PackedSection::Codes,
        record: record.clone(),
        tensors: vec![weight.codes.tensor.clone()],
    };
    compare_record_bytes(&operation, &emitted, &record, &emitted.tensors);
    slice.rows = vec![0..4];
    operation.slices[0] = slice.clone();
    assert!(
        std::panic::catch_unwind(|| compare_record_bytes(
            &operation,
            &emitted,
            &record,
            &emitted.tensors
        ))
        .is_err(),
        "the comparison must detect changed rows while the two records stay identical"
    );
    assert_ne!(
        derived_piece_bytes(&slice, &plan, PackedSection::Codes),
        loader,
        "a changed derivation slice must fail byte parity even when records compare equal"
    );
}
