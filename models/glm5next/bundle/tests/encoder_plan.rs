//! The encoder image's weight plan executed as a host executes it
//! (`runtime_host::direct`, with this bundle's exported transforms), read
//! back through the bundle's index parser, view by view against bytes
//! built here from the checkpoint tensors.
//!
//! `a_tiny_tower_loads_bitwise` writes a synthetic checkpoint of a small
//! tower under the vision declaration's mapping (2 blocks of width 16,
//! 2 x 2 x 2 x 3 patches padded from 24 to 32 columns, a 16-channel 2 x 2
//! downsample). `real_views_dump` materializes chosen views of rank 0 from
//! the pinned checkpoint into `CHOPINFER_GLM5NEXT_ENCODER_DUMP` (skipped
//! when unset), for the torch comparison in its README.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use encoder_spec::weights::{Source, Transform};
use encoder_spec::{Declaration, Environment};
use model_glm5next_contract::encoder::views::{self, Content, Stored};
use model_glm5next_contract::encoder::{VISION, declarations};
use model_glm5next_contract::encoder_plan;
use runtime_bundle_abi::weight_plan::WeightPlanV1;
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};
use serde_json::{Value, json};

const RANKS: u32 = 4;

/// The vision declaration at the tiny tower's sizes.
fn tiny() -> Declaration {
    let mut data: Value = serde_json::from_str(VISION).unwrap();
    data["checkpoint"]["files"] = json!({"model.visual.": "model.safetensors"});
    data["input"]["columns"] = json!(24);
    data["input"]["recipes"][0]["patch"] = json!(2);
    data["input"]["recipes"][0]["maximum_tokens"] = json!(256);
    data["input"]["capacity_rows"] = json!(1024);
    let stages = data["stages"].as_array_mut().unwrap();
    stages[0]["columns"] = json!(32);
    stages[1]["out"] = json!(16);
    stages[1]["weights"]["weight"]["transforms"] = json!(["flatten", {"pad_cols": 32}]);
    stages[2]["blocks"] = json!(2);
    stages[2]["width"] = json!(16);
    stages[2]["attention"]["q_heads"] = json!(2);
    stages[2]["attention"]["kv_heads"] = json!(2);
    stages[2]["attention"]["head_dim"] = json!(8);
    stages[2]["rotary"]["pairs_per_axis"] = json!(2);
    stages[2]["mlp"]["hidden"] = json!(12);
    stages[5]["out"] = json!(8);
    stages[6]["out"] = json!(8);
    stages[7]["mlp"]["hidden"] = json!(10);
    data["output"]["hidden"] = json!(8);
    encoder_spec::parse(&data.to_string(), &Environment::new("glm5next")).unwrap()
}

/// The tiny tower's checkpoint shapes (HF's `Glm5NextVisionModel` at those
/// sizes), under the checkpoint prefix.
fn tiny_shape(name: &str) -> Vec<u64> {
    let name = name.strip_prefix("model.visual.").unwrap();
    let suffix = name
        .split_once("blocks.")
        .map_or(name, |(_, rest)| rest.split_once('.').unwrap().1);
    match suffix {
        "patch_embed.proj.weight" => vec![16, 3, 2, 2, 2],
        "attn.qkv.weight" => vec![48, 16],
        "attn.qkv.bias" => vec![48],
        "attn.q_norm.weight" | "attn.k_norm.weight" => vec![8],
        "attn.proj.weight" => vec![16, 16],
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" => vec![12, 16],
        "mlp.gate_proj.bias" | "mlp.up_proj.bias" => vec![12],
        "mlp.down_proj.weight" => vec![16, 12],
        "downsample.weight" => vec![8, 16, 2, 2],
        "merger.proj.weight" => vec![8, 8],
        "merger.gate_proj.weight" | "merger.up_proj.weight" => vec![10, 8],
        "merger.down_proj.weight" => vec![8, 10],
        "downsample.bias"
        | "merger.post_projection_norm.weight"
        | "merger.post_projection_norm.bias" => vec![8],
        _ => vec![16],
    }
}

/// Writes a safetensors file of every tensor `declaration` reads, BF16
/// with distinct bytes, and returns each tensor's shape and bytes.
fn write_checkpoint(
    root: &Path,
    declaration: &Declaration,
) -> BTreeMap<String, (Vec<u64>, Vec<u8>)> {
    let mut tensors = BTreeMap::new();
    let mut state = 0x9e37_79b9_u32;
    for name in declaration.tensors() {
        let shape = tiny_shape(&name);
        let bytes: Vec<u8> = (0..2 * shape.iter().product::<u64>())
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        tensors.insert(name, (shape, bytes));
    }
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, (shape, bytes)) in &tensors {
        header.insert(
            name.clone(),
            json!({"dtype": "BF16", "shape": shape,
                   "data_offsets": [data.len(), data.len() + bytes.len()]}),
        );
        data.extend_from_slice(bytes);
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(data);
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("model.safetensors"), file).unwrap();
    std::fs::write(root.join("config.json"), "{}").unwrap();
    tensors
}

/// One view's bytes built from the checkpoint tensors by the
/// declaration's mapping: each transform written out by index.
fn expected(
    view: &Stored,
    declaration: &Declaration,
    tensors: &BTreeMap<String, (Vec<u64>, Vec<u8>)>,
) -> Vec<u8> {
    let tensor = |name: &str| {
        let name = match view.block {
            Some(block) => name.replace("{i}", &block.to_string()),
            None => name.to_owned(),
        };
        &tensors[&format!("{}{name}", declaration.checkpoint.prefix)]
    };
    match (&view.content, &view.source) {
        (Content::Zeros, _) => vec![0; view.bytes() as usize],
        (Content::Role, Some(Source::Concat(concat))) => concat
            .concat_rows
            .iter()
            .flat_map(|part| match part {
                Source::Tensor(part) => tensor(&part.tensor).1.clone(),
                other => panic!("{other:?}"),
            })
            .collect(),
        (Content::Role, Some(Source::Tensor(part))) => {
            let (shape, bytes) = tensor(&part.tensor);
            let element = |index: usize| [bytes[2 * index], bytes[2 * index + 1]];
            match part.transforms[..] {
                [] | [Transform::Flatten] => bytes.clone(),
                [Transform::Flatten, Transform::PadCols(padded)] => {
                    let columns: u64 = shape[1..].iter().product();
                    let mut out = Vec::new();
                    for row in 0..shape[0] {
                        for column in 0..padded {
                            out.extend(if column < columns {
                                element((row * columns + column) as usize)
                            } else {
                                [0, 0]
                            });
                        }
                    }
                    out
                }
                [Transform::Flatten, Transform::SpaceToDepthCols(2)] => {
                    // [out, channel, ky, kx] to columns (ky * 2 + kx) * channels + c.
                    let (rows, channels) = (shape[0] as usize, shape[1] as usize);
                    let mut out = vec![0; bytes.len()];
                    for row in 0..rows {
                        for channel in 0..channels {
                            for ky in 0..2 {
                                for kx in 0..2 {
                                    let from = ((row * channels + channel) * 2 + ky) * 2 + kx;
                                    let column = (ky * 2 + kx) * channels + channel;
                                    let to = row * channels * 4 + column;
                                    out[2 * to..2 * to + 2].copy_from_slice(&element(from));
                                }
                            }
                        }
                    }
                    out
                }
                ref other => panic!("{other:?}"),
            }
        }
        other => panic!("{other:?}"),
    }
}

/// Every rank's plan of `declaration`'s image over the checkpoint at
/// `root`, prepared as the host prepares it, with this bundle's
/// transforms.
fn load(
    root: &Path,
    plan: impl Fn(&runtime_bundle_abi::weight_plan::WeightPlanRequestV1) -> WeightPlanV1,
) -> (PlannedImageV1, Vec<WeightPlanV1>) {
    let checkpoint = Arc::new(DirectCheckpointV1::open(root).unwrap());
    // Any module digest: the identity covers it, nothing reads it here.
    let implementation = implementation_v1([7; 32]);
    let role = std::str::from_utf8(views::IMAGE_ROLE).unwrap();
    let plans: Vec<WeightPlanV1> = (0..RANKS)
        .map(|rank| plan(&checkpoint.request(role, RANKS, rank, implementation, vec![], vec![])))
        .collect();
    let image = PlannedImageV1::new(
        checkpoint,
        plans.clone(),
        Some(bundle_glm5next::chop_bundle_weight_transform_v1),
        implementation,
        DirectLoadOptionsV1::for_ranks(RANKS as usize),
    )
    .unwrap();
    (image, plans)
}

/// Rank `rank`'s bytes `[offset, offset + bytes)` of `image`.
fn read(image: &PlannedImageV1, rank: u32, offset: u64, bytes: u64) -> Vec<u8> {
    let start: u64 = (0..rank as usize).map(|rank| image.arena_bytes(rank)).sum();
    let mut out = vec![0; bytes as usize];
    image.read(start + offset, &mut out).unwrap();
    out
}

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("glm5next-{name}-{}", std::process::id()))
}

#[test]
fn a_tiny_tower_loads_bitwise() {
    let declaration = tiny();
    let root = scratch("tiny-encoder-checkpoint");
    let tensors = write_checkpoint(&root, &declaration);
    let (image, plans) = load(&root, |request| {
        encoder_plan::plan_for(request, &declaration, &|name| Some(tiny_shape(name))).unwrap()
    });
    let layout = views::layout_for(&declaration).unwrap();
    // The patch and downsample views take the bundle's transforms.
    assert!(plans.iter().all(WeightPlanV1::uses_bundle_transforms));
    for (rank, plan) in plans.iter().enumerate() {
        let rank = rank as u32;
        let index = views::parse_for(&plan.components[0].bytes, rank, &layout).unwrap();
        let whole = read(&image, rank, 0, index.image_bytes);
        for stored in &layout {
            let view = &index.views[&stored.name];
            let bytes = &whole[view.offset as usize..(view.offset + view.bytes) as usize];
            assert!(
                bytes == expected(stored, &declaration, &tensors),
                "rank {rank}: {}",
                stored.name
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// The views of rank 0 the torch comparison reads: one of each way a view
/// is built.
const DUMPED: &[&str] = &[
    "model.visual.patch_embed.proj.weight",
    "model.visual.downsample.weight",
    "model.visual.blocks.0.mlp.gate_proj.weight",
    "model.visual.blocks.11.mlp.gate_proj.bias",
    "model.visual.merger.gate_proj.weight",
    "model.visual.blocks.23.attn.qkv.weight",
    "model.visual.blocks.7.attn.k_norm.weight",
    "model.visual.merger.proj.bias",
    "model.visual.merger.post_projection_norm.bias",
];

#[test]
fn real_views_dump() {
    let (Some(root), Some(out)) = (
        std::env::var_os("CHOPINFER_GLM5NEXT_CHECKPOINT"),
        std::env::var_os("CHOPINFER_GLM5NEXT_ENCODER_DUMP"),
    ) else {
        eprintln!("skipped: CHOPINFER_GLM5NEXT_ENCODER_DUMP is not set");
        return;
    };
    let (image, plans) = load(Path::new(&root), |request| {
        model_glm5next_contract::direct::plan(request)
            .unwrap()
            .unwrap()
    });
    let index = views::parse(&plans[0].components[0].bytes, 0).unwrap();
    let layout = views::layout().unwrap();
    let out = Path::new(&out);
    std::fs::create_dir_all(out).unwrap();
    let mut manifest = serde_json::Map::new();
    for name in DUMPED {
        let view = &index.views[*name];
        let stored = layout.iter().find(|stored| stored.name == *name).unwrap();
        let bytes = read(&image, 0, view.offset, view.bytes);
        std::fs::write(out.join(format!("{name}.bin")), &bytes).unwrap();
        manifest.insert(
            (*name).into(),
            json!({"shape": view.shape, "offset": view.offset,
                   "source": serde_json::to_value(&stored.source).unwrap()}),
        );
    }
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert_eq!(declarations().unwrap()[0].tensors().len(), 24 * 14 + 11);
}
