//! MiMo materialized tensor contracts shared by compiler and bundle admission.
use crate::geometry::V26_FLASH as GEOMETRY;

use runtime_bundle::rank_index::ViewContract;
use std::collections::BTreeMap;

pub mod backend;
pub mod frontend;
pub mod geometry;
mod geometry_checks;
pub mod kv_calibration;
pub mod note;
pub mod prepared;
pub mod projections;
#[cfg(test)]
mod rank_shape_tests;
// MiMo's SM120 kernel catalogs.
pub mod sm120;
mod supported_ranks;
pub mod tp;
pub use tp::Tp;

// The packager's precision admission, compiled here unchanged so the bundle
// plans direct loads with the code that admits the checkpoint
// (`ci/source_dependencies.toml` records the edge). It names its error type,
// tensor record and this crate from the crate root, as in the compiler.
extern crate self as model_mimo26_contract;
pub type MimoResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path = "../../compiler/src/precision.rs"]
pub mod precision;
pub mod weight_plan;

/// The packager's checkpoint tensor record, as its placement reads it.
pub mod catalog {
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct TensorRecord {
        pub name: String,
        pub dtype: runtime_weights::DType,
        pub shape: Vec<u64>,
        /// The file name within the checkpoint directory (`dir/file` below it).
        pub file: String,
        /// Absolute offset of the tensor's bytes in the file.
        pub file_start: u64,
        pub bytes: u64,
    }
}

pub const GLOBAL_LAYERS: [u32; GEOMETRY.global_layers()] = GEOMETRY.global_layer_ids();

fn view(shape: &[u64], dtype: &'static str, layout_id: &'static str) -> ViewContract {
    let width = match dtype {
        "BF16" => 2,
        "F32" => 4,
        _ => 1,
    };
    ViewContract {
        shape: shape.to_vec(),
        dtype,
        layout_id,
        bytes: shape.iter().product::<u64>() * width,
    }
}

/// The released checkpoints' export metadata: FP8 E4M3 128x128 blocks with
/// MXFP4 experts, attention outputs left BF16.
pub fn served_quantization_config(geometry: &geometry::Geometry) -> serde_json::Value {
    let ignored: Vec<String> = (0..geometry.layers())
        .map(|layer| format!("model.layers.{layer}.self_attn.o_proj"))
        .chain(["model.decoder.self_attn.o_proj".into()])
        .collect();
    serde_json::json!({
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128], "store_dtype": "mxfp4", "mxfp4_block_size": 32,
        "ignored_layers": ignored,
    })
}

/// One static copy of each contract string.
fn intern(text: &str) -> &'static str {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<&'static str>>> =
        std::sync::OnceLock::new();
    let mut seen = SEEN.get_or_init(Default::default).lock().expect("intern");
    match seen.get(text) {
        Some(found) => found,
        None => {
            let leaked: &'static str = Box::leak(text.to_owned().into_boxed_str());
            seen.insert(leaked);
            leaked
        }
    }
}

/// Text image contracts: the direct-load plan of the served checkpoint's
/// headers. Rank ownership changes selections, not the view schema.
pub fn text_views(tp: Tp) -> BTreeMap<String, ViewContract> {
    fn build(tp: Tp) -> BTreeMap<String, ViewContract> {
        let roles = precision::text_roles(tp.geometry, |layer| tp.geometry.is_global(layer as u32));
        let headers = precision::expected_source_tensors(&roles).expect("served source encodings");
        let tensors = headers
            .into_iter()
            .map(|(name, (dtype, shape))| {
                let width = match dtype {
                    runtime_weights::DType::Bf16 => 2,
                    runtime_weights::DType::F32 => 4,
                    _ => 1,
                };
                let bytes = shape.iter().product::<u64>() * width;
                (
                    name.clone(),
                    catalog::TensorRecord {
                        name,
                        dtype,
                        shape,
                        bytes,
                        file: String::new(),
                        file_start: 0,
                    },
                )
            })
            .collect();
        let metadata =
            precision::export_metadata(Some(&served_quantization_config(tp.geometry)), None)
                .expect("served export metadata");
        let admitted = precision::admit(tp.geometry, &metadata, &tensors, &roles)
            .expect("served text operands");
        weight_plan::text_views(&tensors, &admitted, None, 0, tp)
            .expect("declared text image")
            .into_iter()
            .map(|planned| {
                (
                    planned.name,
                    ViewContract {
                        dtype: intern(&planned.dtype),
                        shape: planned.shape,
                        bytes: planned.bytes,
                        layout_id: intern(&planned.layout_id),
                    },
                )
            })
            .collect()
    }
    match retained(tp) {
        Some(cache) => cache.get_or_init(|| build(tp)).clone(),
        None => build(tp),
    }
}

/// The pinned digests of [`text_views`] (`text-contracts.sha256`).
const TEXT_CONTRACT_PINS: &str = include_str!("../text-contracts.sha256");

/// The pinned [`runtime_bundle::rank_index::contract_sha256`] of
/// [`text_views`] at `tp`, when its geometry has one: rank indexes are then
/// checked against the digest instead of rederived contracts.
pub fn text_contract_pin(tp: Tp) -> Option<&'static str> {
    TEXT_CONTRACT_PINS
        .lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            let (geometry, ranks, digest) = (fields.next()?, fields.next()?, fields.next()?);
            (geometry == tp.geometry.id && ranks == tp.ranks().to_string()).then_some(digest)
        })
}

// Serving repeatedly validates complete indexes and device views. Retain
// the immutable contracts instead of rebuilding every source bank.
fn retained(tp: Tp) -> Option<&'static std::sync::OnceLock<BTreeMap<String, ViewContract>>> {
    static TP2: std::sync::OnceLock<BTreeMap<String, ViewContract>> = std::sync::OnceLock::new();
    static TP4: std::sync::OnceLock<BTreeMap<String, ViewContract>> = std::sync::OnceLock::new();
    (tp.geometry == &GEOMETRY).then(|| match tp.ranks() {
        2 => &TP2,
        4 => &TP4,
        _ => unreachable!("served topology"),
    })
}

pub mod capabilities;
pub mod encoders;

pub mod drafters;

pub mod dflash;
pub mod encoder_index;
pub mod graphs;
pub mod index;
pub mod mtp;
pub mod own_ops;
pub mod resources;
pub mod serving;
pub mod state;
pub mod topology;

#[cfg(test)]
mod geometry_goldens;

#[cfg(test)]
mod text_contract_pins {
    use super::*;

    const REGENERATE: &str = "CHOPINFER_UPDATE_TEXT_CONTRACT_PINS=1 cargo test -p model-mimo26-contract --lib text_contract_pins";

    /// Every pin is the digest of the contracts derived now, and the served
    /// geometry is pinned at every supported rank count. Geometries without
    /// pins (data-only additions) derive their contracts instead.
    #[test]
    fn text_contract_pins_match_the_derivation() {
        let header: String = TEXT_CONTRACT_PINS
            .lines()
            .take_while(|line| line.starts_with('#'))
            .map(|line| format!("{line}\n"))
            .collect();
        let mut pinned: Vec<&str> = TEXT_CONTRACT_PINS
            .lines()
            .filter(|line| !line.starts_with('#'))
            .filter_map(|line| line.split_whitespace().next())
            .collect();
        if !pinned.contains(&GEOMETRY.id) {
            pinned.insert(0, GEOMETRY.id);
        }
        pinned.dedup();
        let mut rendered = header;
        for id in pinned {
            let Some(geometry) = geometry::GEOMETRIES.iter().find(|g| g.id == id) else {
                continue;
            };
            for ranks in Tp::SUPPORTED {
                let tp = Tp::of_geometry(geometry, ranks.ranks()).unwrap();
                let digest = runtime_bundle::rank_index::contract_sha256(&text_views(tp));
                rendered.push_str(&format!("{id} {} {digest}\n", tp.ranks()));
            }
        }
        if std::env::var("CHOPINFER_UPDATE_TEXT_CONTRACT_PINS").as_deref() == Ok("1") {
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/text-contracts.sha256");
            std::fs::write(path, rendered).unwrap();
            return;
        }
        assert!(
            TEXT_CONTRACT_PINS == rendered,
            "text-contracts.sha256 differs from the derived text contracts; after an \
             intended contract change regenerate it with `{REGENERATE}`.\nderived:\n{rendered}"
        );
    }
}
