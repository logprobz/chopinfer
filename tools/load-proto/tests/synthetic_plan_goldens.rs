//! A small synthetic Qwen3.8 checkpoint planned by the bundle's contract and
//! loaded by the host's direct executor at 2, 4 and 8 ranks: every rank's
//! whole arena and view layout equal the digests recorded from the
//! rank-image materializer before it was deleted
//! (`fixtures/qwen38/weight-plan-goldens.json`; its arena digests were
//! re-derived once, when the synthetic FP8 codes stopped drawing NaN codes,
//! by the direct executor that had matched the materializer).
use model_qwen38_contract::contract::{RANK_IMAGE_ROLE, index_role};
use model_qwen38_contract::placement::ImagePlan;
use model_qwen38_contract::weight_plan::{Planned, rank_plan};
use model_qwen38_contract::{
    self as contract_model,
    catalog::{
        Catalog, Placement, ScaleLayout, TensorRecord, local_experts, ngram_rank_rows_flash,
    },
};
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};
use runtime_weights::{DType, parse_safetensors_header};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const GOLDENS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/qwen38/weight-plan-goldens.json"
);

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "qwen38-plan-goldens-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One tensor of every placement and layout the FP8 plan uses, placed for
/// `ranks`: FP8 experts with BF16 block scales (gate/up concatenation, BF16
/// to F32 scale tiles), axis-0 and axis-1 splits, linear q/k/v slices,
/// shared KV heads, the padded shared expert, widened `A_log` and an n-gram
/// shard across a rank boundary. Writes the checkpoint's one safetensors
/// file under `root` and returns its catalog.
fn synthetic(root: &Path, ranks: usize) -> Catalog {
    let mut tensors = BTreeMap::new();
    let mut data: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut add = |name: String, shape: Vec<u64>, dtype: DType, placement, scale| {
        let width = if dtype == DType::Bf16 { 2 } else { 1 };
        let bytes = shape.iter().product::<u64>() * width;
        let seed = data.len();
        // FP8 codes are valid E4M3: a NaN code's low bit cleared.
        let raw = (0..bytes as usize)
            .map(|n| (n.wrapping_mul(37) ^ (n >> 8) ^ seed) as u8)
            .map(|b| {
                if dtype == DType::F8E4M3 && b & 0x7f == 0x7f {
                    b ^ 1
                } else {
                    b
                }
            })
            .collect();
        data.insert(name.clone(), raw);
        tensors.insert(
            name.clone(),
            TensorRecord {
                name,
                shape,
                dtype,
                placement,
                scale,
                bytes,
                file: "weights.safetensors".into(),
                file_start: 0,
            },
        );
    };
    let local = local_experts(ranks).unwrap();
    for expert in 0..512 {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let prefix = format!("model.language_model.layers.0.mlp.experts.{expert}.{proj}");
            let placement = Placement::Expert {
                rank: expert / local,
                local_expert: expert % local,
            };
            let shape = if proj == "down_proj" {
                vec![32, 32]
            } else {
                vec![16, 32]
            };
            add(
                format!("{prefix}.weight"),
                shape,
                DType::F8E4M3,
                placement.clone(),
                ScaleLayout::Block128 {
                    scale: format!("{prefix}.weight_scale_inv"),
                },
            );
            add(
                format!("{prefix}.weight_scale_inv"),
                vec![1, 1],
                DType::Bf16,
                placement,
                ScaleLayout::BlockInverseScale,
            );
        }
    }
    let layer = "model.language_model.layers";
    let split = |axis| Placement::Axis { axis, parts: ranks };
    for (name, shape, placement) in [
        (
            "model.language_model.embed_tokens.weight".to_string(),
            vec![32, 32],
            Placement::Replicated,
        ),
        ("lm_head.weight".into(), vec![64, 32], split(0)),
        (
            format!("{layer}.0.linear_attn.in_proj_qkv.weight"),
            vec![10240, 32],
            Placement::LinearQkv,
        ),
        (
            format!("{layer}.0.linear_attn.conv1d.weight"),
            vec![10240, 1, 4],
            Placement::LinearQkv,
        ),
        (
            format!("{layer}.0.linear_attn.out_proj.weight"),
            vec![32, 128],
            split(1),
        ),
        (format!("{layer}.0.linear_attn.A_log"), vec![48], split(0)),
        (
            format!("{layer}.3.self_attn.q_proj.weight"),
            vec![12288, 32],
            split(0),
        ),
        (
            format!("{layer}.3.self_attn.k_proj.weight"),
            vec![512, 32],
            Placement::Axis { axis: 0, parts: 2 },
        ),
        (
            format!("{layer}.3.self_attn.v_proj.weight"),
            vec![512, 32],
            Placement::Axis { axis: 0, parts: 2 },
        ),
        (
            format!("{layer}.0.mlp.shared_expert.gate_proj.weight"),
            vec![640, 32],
            split(0),
        ),
        (
            format!("{layer}.0.mlp.shared_expert.up_proj.weight"),
            vec![640, 32],
            split(0),
        ),
        (
            format!("{layer}.0.mlp.shared_expert.down_proj.weight"),
            vec![32, 640],
            split(1),
        ),
    ] {
        add(name, shape, DType::Bf16, placement, ScaleLayout::None);
    }
    let boundary = ngram_rank_rows_flash(1, ranks).start;
    add(
        format!("{layer}.1.ple.ple_embedding.ngram_embedding.shard_0.weight"),
        vec![8, 160],
        DType::F8E4M3,
        Placement::Ngram {
            rows: boundary - 4..boundary + 4,
        },
        ScaleLayout::PerTensor {
            scale: format!("{layer}.1.ple.ple_embedding.ngram_embedding.weight_scale"),
        },
    );
    let mut header = serde_json::Map::new();
    let mut offset = 0;
    for (name, raw) in &data {
        let t = &tensors[name];
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": t.dtype.safetensors_name(), "shape": t.shape, "data_offsets": [offset, offset + raw.len()]}),
        );
        offset += raw.len();
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = std::fs::File::create(root.join("weights.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    for raw in data.values() {
        file.write_all(raw).unwrap();
    }
    drop(file);
    let mut file = std::fs::File::open(root.join("weights.safetensors")).unwrap();
    let len = file.metadata().unwrap().len();
    let parsed = parse_safetensors_header(&mut file, len).unwrap();
    for (name, info) in parsed.tensors() {
        tensors.get_mut(name).unwrap().file_start = parsed.data_offset() + info.data_offsets().0;
    }
    Catalog {
        geometry: &contract_model::geometry::FLASH,
        tensors,
        ranks,
        precision: None,
    }
}

/// One rank as loaded: its whole arena and its views as
/// `name offset bytes dtype shape layout` lines in offset order.
struct Rank {
    arena: Vec<u8>,
    layout: String,
}

/// Plans `catalog` as the bundle's planner does and loads every rank's
/// arena through the host's executor.
fn load(root: &Path, catalog: Catalog) -> Vec<Rank> {
    let ranks = catalog.ranks;
    let checkpoint = Arc::new(DirectCheckpointV1::open(root).unwrap());
    let plan = ImagePlan::new(&catalog, None).unwrap();
    let planned = Planned {
        catalog: Arc::new(catalog),
        plan,
        // Where the device capacity rule places it: host memory at TP2.
        host_ngram: model_qwen38_contract::package::ngram_in_host_memory(
            model_qwen38_contract::placement::bundle_shape_flash(ranks).unwrap(),
        ),
    };
    let implementation = implementation_v1([0; 32]);
    let role = std::str::from_utf8(RANK_IMAGE_ROLE).unwrap();
    let plans: Vec<_> = (0..ranks as u32)
        .map(|rank| {
            let request =
                checkpoint.request(role, ranks as u32, rank, implementation, vec![], vec![]);
            rank_plan(&request, &planned).unwrap()
        })
        .collect();
    let layouts: Vec<String> = plans
        .iter()
        .map(|plan| {
            let index = plan
                .components
                .iter()
                .find(|c| c.name == index_role(plan.rank))
                .unwrap();
            let index: serde_json::Value = serde_json::from_slice(&index.bytes).unwrap();
            layout(index["views"].as_array().unwrap())
        })
        .collect();
    let image = PlannedImageV1::new(
        checkpoint,
        plans,
        None,
        implementation,
        DirectLoadOptionsV1 {
            threads_per_rank: 2,
            read_ahead: false,
            direct_reads: false,
            shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
        },
    )
    .unwrap();
    layouts
        .into_iter()
        .enumerate()
        .map(|(rank, layout)| {
            let mut arena = vec![0; image.arena_bytes(rank) as usize];
            image
                .read(image.description().rank_images()[rank].offset, &mut arena)
                .unwrap();
            Rank { arena, layout }
        })
        .collect()
}

/// A rank index's views, one line each in offset order.
fn layout(views: &[serde_json::Value]) -> String {
    let mut lines: Vec<(u64, String)> = views
        .iter()
        .map(|view| {
            let shape: Vec<String> = view["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap().to_string())
                .collect();
            (
                view["offset"].as_u64().unwrap(),
                format!(
                    "{} {} {} {} {} {}\n",
                    view["name"].as_str().unwrap(),
                    view["offset"],
                    view["bytes"],
                    view["dtype"].as_str().unwrap(),
                    shape.join("x"),
                    view["layout_id"].as_str().unwrap(),
                ),
            )
        })
        .collect();
    lines.sort();
    lines.into_iter().map(|(_, line)| line).collect()
}

fn hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Every way `loaded` differs from the recorded ranks at its rank count.
fn differences(goldens: &serde_json::Value, loaded: &[Rank]) -> Vec<String> {
    let recorded = goldens["text"][loaded.len().to_string()]
        .as_array()
        .unwrap();
    let mut differences = Vec::new();
    if recorded.len() != loaded.len() {
        differences.push(format!(
            "{} ranks, {} recorded",
            loaded.len(),
            recorded.len()
        ));
    }
    for (rank, (current, recorded)) in loaded.iter().zip(recorded).enumerate() {
        for (field, value) in [
            ("arena_bytes", serde_json::json!(current.arena.len())),
            ("sha256", serde_json::json!(hex(&current.arena))),
            (
                "layout_sha256",
                serde_json::json!(hex(current.layout.as_bytes())),
            ),
        ] {
            if recorded[field] != value {
                differences.push(format!(
                    "tp{} rank {rank} {field}: {value}, recorded {}",
                    loaded.len(),
                    recorded[field]
                ));
            }
        }
    }
    differences
}

fn goldens() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(GOLDENS).unwrap()).unwrap()
}

#[test]
fn planned_arenas_equal_the_materializer_at_every_rank_count() {
    let goldens = goldens();
    for ranks in [2, 4] {
        let temp = Temp::new();
        let loaded = load(&temp.0, synthetic(&temp.0, ranks));
        let differences = differences(&goldens, &loaded);
        assert!(differences.is_empty(), "{differences:#?}");
    }
}

/// A byte moved within an arena, or two views that trade places, are not
/// the recorded load.
#[test]
fn a_permuted_byte_or_view_is_detected() {
    let goldens = goldens();
    let temp = Temp::new();
    let loaded = load(&temp.0, synthetic(&temp.0, 4));
    assert!(differences(&goldens, &loaded).is_empty());
    let mut permuted = load(&temp.0, synthetic(&temp.0, 4));
    let arena = &mut permuted[1].arena;
    let first = arena.iter().position(|&byte| byte != 0).unwrap();
    let second = (first + 1..arena.len())
        .find(|&at| arena[at] != 0 && arena[at] != arena[first])
        .unwrap();
    arena.swap(first, second);
    let found = differences(&goldens, &permuted);
    assert_eq!(found.len(), 1, "{found:#?}");
    assert!(found[0].starts_with("tp4 rank 1 sha256"), "{found:#?}");
    let mut moved = load(&temp.0, synthetic(&temp.0, 4));
    let mut lines: Vec<&str> = moved[2].layout.lines().collect();
    lines.swap(0, 1);
    moved[2].layout = lines.iter().map(|line| format!("{line}\n")).collect();
    let found = differences(&goldens, &moved);
    assert_eq!(found.len(), 1, "{found:#?}");
    assert!(
        found[0].starts_with("tp4 rank 2 layout_sha256"),
        "{found:#?}"
    );
}
