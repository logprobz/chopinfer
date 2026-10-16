//! Shared encoder image geometry and resource declarations.
pub mod views;

use std::sync::OnceLock;

use crate::backend::{Kernels, Selected};
use encoder_spec::{Declaration, Environment, Error};

/// One kernel the encoder declarations run, as the backend compiles it.
pub type Entry = <Selected as Kernels>::EncoderKernel;

type DeclarationCache = Vec<(&'static str, Result<[Declaration; 1], Error>)>;
type KernelCache = Vec<(&'static str, Result<Vec<Entry>, Error>)>;

/// The vision encoder's declaration.
pub const MAX_IMAGE_PIXELS: u32 = 16_777_216;

pub const VISION: &str = include_str!("../../encoders/qwen38-vision.json");

/// Qwen3.8's declarations in the frontend's encoder order. Parsed and
/// validated once.
pub fn declarations() -> Result<&'static [Declaration; 1], Error> {
    static DECLARATIONS: OnceLock<Result<[Declaration; 1], Error>> = OnceLock::new();
    DECLARATIONS
        .get_or_init(|| Ok([encoder_spec::parse(VISION, &Environment::new("qwen38"))?]))
        .as_ref()
        .map_err(Clone::clone)
}

/// The vision declaration specialized to a listed geometry.
pub fn declarations_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<&'static [Declaration; 1], Error> {
    if geometry.id == crate::geometry::FLASH.id {
        return declarations();
    }
    static DECLARATIONS: OnceLock<DeclarationCache> = OnceLock::new();
    let declarations = DECLARATIONS.get_or_init(|| {
        crate::geometry::GEOMETRIES
            .iter()
            .map(|g| {
                let result = (|| {
                    let mut value: serde_json::Value =
                        serde_json::from_str(VISION).expect("vision template");
                    let config: serde_json::Value =
                        serde_json::from_str(g.document).expect("geometry document");
                    value["name"] = format!("{}-vision", g.kernel_tag.replace('_', "-")).into();
                    let pinned = value["checkpoint"]["config"][0]["values"]
                        .as_object_mut()
                        .expect("vision fields");
                    for (key, field) in pinned {
                        *field = config["vision_config"][key].clone();
                    }
                    value["input"]["recipes"][0]["patch"] = g.vision_patch().into();
                    value["input"]["recipes"][0]["temporal_patch"] =
                        g.vision_temporal_patch().into();
                    value["input"]["recipes"][0]["merge"] = g.vision_merge().into();
                    value["input"]["columns"] =
                        (g.vision_channels() * g.vision_temporal_patch() * g.vision_patch().pow(2))
                            .into();
                    value["input"]["capacity_rows"] =
                        (MAX_IMAGE_PIXELS / g.vision_patch().pow(2)).into();
                    let side = g.vision_positions().isqrt();
                    assert_eq!(side * side, g.vision_positions(), "square position table");
                    for stage in value["stages"].as_array_mut().expect("vision stages") {
                        match stage["op"].as_str().expect("vision stage kind") {
                            "position.learned-grid.bilinear-align-corners.f32.v1" => {
                                stage["table"] = serde_json::json!([side, side])
                            }
                            "merge.consecutive.v1" => {
                                stage["rows"] = g.vision_merge().pow(2).into()
                            }
                            "linear.pairs.v1" => {
                                let tensor = stage["weights"]["weight"]
                                    .as_str()
                                    .or_else(|| stage["weights"]["weight"]["tensor"].as_str())
                                    .expect("linear weight");
                                stage["out"] = match tensor {
                                    "patch_embed.proj.weight" => g.vision_hidden(),
                                    "merger.linear_fc1.weight" => {
                                        g.vision_hidden() * g.vision_merge().pow(2)
                                    }
                                    "merger.linear_fc2.weight" => g.vision_out(),
                                    _ => panic!("unknown vision linear weight"),
                                }
                                .into();
                            }
                            "transformer.pre-norm.v1" => {
                                stage["blocks"] = g.vision_blocks().into();
                                stage["width"] = g.vision_hidden().into();
                                stage["attention"]["q_heads"] = g.vision_heads().into();
                                stage["attention"]["kv_heads"] = g.vision_heads().into();
                                stage["attention"]["head_dim"] =
                                    (g.vision_hidden() / g.vision_heads()).into();
                                stage["rotary"]["pairs_per_axis"] =
                                    (g.vision_hidden() / g.vision_heads() / 4).into();
                                let padded = g.vision_intermediate().next_multiple_of(128);
                                stage["mlp"]["hidden"] = g.vision_intermediate().into();
                                stage["mlp"]["pad_to"] = padded.into();
                                for name in ["up.weight", "up.bias"] {
                                    stage["weights"][name]["transforms"][0]["pad_rows"] =
                                        padded.into();
                                }
                                stage["weights"]["down.weight"]["transforms"][0]["pad_cols"] =
                                    padded.into();
                                if padded == g.vision_intermediate() {
                                    stage["mlp"].as_object_mut().unwrap().remove("pad_to");
                                    for name in ["up.weight", "up.bias", "down.weight"] {
                                        stage["weights"][name]
                                            .as_object_mut()
                                            .unwrap()
                                            .remove("transforms");
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    value["output"]["hidden"] = g.vision_out().into();
                    Ok([encoder_spec::parse(
                        &serde_json::to_string(&value).expect("vision JSON"),
                        &Environment::new("qwen38"),
                    )?])
                })();
                (g.id, result)
            })
            .collect()
    });
    declarations
        .iter()
        .find(|(id, _)| *id == geometry.id)
        .expect("listed geometry")
        .1
        .as_ref()
        .map_err(Clone::clone)
}

pub fn kernels_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<&'static [Entry], Error> {
    if geometry.id == crate::geometry::FLASH.id {
        return kernels();
    }
    static KERNELS: OnceLock<KernelCache> = OnceLock::new();
    let kernels = KERNELS.get_or_init(|| {
        crate::geometry::GEOMETRIES
            .iter()
            .map(|g| {
                (
                    g.id,
                    declarations_for(g).and_then(|d| Selected::encoder_kernels(d)),
                )
            })
            .collect()
    });
    kernels
        .iter()
        .find(|(id, _)| *id == geometry.id)
        .expect("listed geometry")
        .1
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// The encoder kernel catalog: the entries the declarations run. Built once.
pub fn kernels() -> Result<&'static [Entry], Error> {
    static KERNELS: OnceLock<Result<Vec<Entry>, Error>> = OnceLock::new();
    KERNELS
        .get_or_init(|| Selected::encoder_kernels(declarations()?))
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// The capabilities a package with the encoder image declares.
pub const MEDIA_CAPABILITY: &str = "chop.media-inputs.v1";
pub const POSITION_AXES_CAPABILITY: &str = "chop.position-axes.v1";

/// The checkpoint's MLP width and the zero-padded width the GEMMs run.
pub const MLP: u32 = crate::geometry::FLASH.vision_intermediate();
pub const MLP_PADDED: u32 = MLP.next_multiple_of(128);

/// Merger output width: the language model's hidden size.
pub const VIT_OUT: u32 = crate::geometry::FLASH.vision_out();
pub const PATCH_COLUMNS: u32 = crate::geometry::FLASH.vision_channels()
    * crate::geometry::FLASH.vision_temporal_patch()
    * crate::geometry::FLASH.vision_patch()
    * crate::geometry::FLASH.vision_patch();
pub const MERGE: u32 = crate::geometry::FLASH.vision_merge();
/// Patches of the largest image, one attention sequence: 16,777,216
/// pixels over 16 x 16 patches.
pub const VIT_ROWS: u32 = MAX_IMAGE_PIXELS / crate::geometry::FLASH.vision_patch().pow(2);

/// The media arena: rows of `VIT_OUT` BF16 in pages of `PAGE_ROWS`. Four
/// of the largest images (16,384 rows each) fit.
pub const ARENA_ROWS: u64 = 65_536;
pub const PAGE_ROWS: u64 = 64;
/// One image's F32 patch rows at most.
pub const MAX_PAYLOAD_BYTES: u64 = max_payload_bytes(&crate::geometry::FLASH);

/// F32 patch payload for the largest admitted image.
pub const fn max_payload_bytes(g: &crate::geometry::Geometry) -> u64 {
    let patch = g.vision_patch() as u64;
    let rows = MAX_IMAGE_PIXELS as u64 / (patch * patch);
    rows * g.vision_channels() as u64 * g.vision_temporal_patch() as u64 * patch * patch * 4
}

/// Every encoder allocation on one rank, including the media arena.
/// The bundle checks this bound against its actual allocation requests.
pub const BUFFER_BYTES: u64 = 2_101_411_856;

pub fn buffer_bytes(geometry: &'static crate::geometry::Geometry) -> u64 {
    let declarations = declarations_for(geometry).expect("listed encoder geometry");
    encoder_spec::buffers::plan(
        &[&declarations[0]],
        u64::from(MAX_IMAGE_PIXELS / geometry.vision_patch().pow(2)),
        encoder_spec::buffers::Serving {
            arena_rows: ARENA_ROWS,
            prefill_rows: u64::from(crate::contract::PREFILL_TOKENS),
            staging_bytes: 0,
        },
    )
    .expect("encoder buffer plan")
    .bytes()
}
