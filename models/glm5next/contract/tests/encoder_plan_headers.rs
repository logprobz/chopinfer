//! The encoder image's weight plan against the checkpoints' safetensors
//! headers: every rank's records tile every view of the layout exactly,
//! read tensors of the vision tower's file inside their data, and the
//! planned index is the one the bundle admits. Set
//! `CHOPINFER_GLM5NEXT_CHECKPOINT` (and `CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT`)
//! to the pinned checkpoint directories; a test skips without one. Only
//! headers are read, never tensor data.
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use model_glm5next_contract::encoder::views;
use model_glm5next_contract::encoder_plan;
use runtime_bundle_abi::weight_plan::{CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1};
use serde_json::Value;

const MAX_HEADER: u64 = 64 << 20;

/// The checkpoint's files and documents as a plan request names them,
/// with each file's header and each header's tensors by name.
fn request(root: &Path) -> (WeightPlanRequestV1, BTreeMap<String, (String, Value)>) {
    let index: Value =
        serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let mut shards: Vec<&str> = index["weight_map"]
        .as_object()
        .unwrap()
        .values()
        .map(|file| file.as_str().unwrap())
        .collect();
    shards.sort_unstable();
    shards.dedup();
    let mut files = Vec::new();
    let mut tensors = BTreeMap::new();
    for shard in shards {
        let path = root.join(shard);
        let mut file = std::fs::File::open(&path).unwrap();
        let mut length = [0; 8];
        file.read_exact(&mut length).unwrap();
        let length = u64::from_le_bytes(length);
        assert!(length <= MAX_HEADER, "{}", path.display());
        let mut header = vec![0; length as usize];
        file.read_exact(&mut header).unwrap();
        let parsed: serde_json::Map<String, Value> = serde_json::from_slice(&header).unwrap();
        for (name, entry) in parsed {
            tensors.insert(name, (shard.to_owned(), entry));
        }
        files.push(CheckpointFileV1 {
            name: shard.into(),
            byte_length: file.metadata().unwrap().len(),
            data_offset: 8 + length,
            header,
        });
    }
    let documents = ["config.json", "model.safetensors.index.json"]
        .into_iter()
        .map(|name| NamedBytesV1 {
            name: name.into(),
            bytes: std::fs::read(root.join(name)).unwrap(),
        })
        .collect();
    let request = WeightPlanRequestV1 {
        image_role: std::str::from_utf8(views::IMAGE_ROLE).unwrap().into(),
        ranks: 4,
        files,
        documents,
        ..Default::default()
    };
    (request, tensors)
}

fn check(variable: &str) {
    let Some(root) = std::env::var_os(variable) else {
        eprintln!("skipped: {variable} is not set");
        return;
    };
    let (mut request, tensors) = request(Path::new(&root));
    encoder_plan::check_tower(&request).unwrap();
    let layout = views::layout().unwrap();
    let mut identities = Vec::new();
    for rank in 0..4 {
        request.rank = rank;
        let plan = model_glm5next_contract::direct::plan(&request)
            .unwrap()
            .expect("the encoder image is planned");
        assert_eq!(plan.image_role.as_bytes(), views::IMAGE_ROLE);
        assert_eq!(plan.arena_bytes, views::image_bytes().unwrap());
        // Each view's records, back to back from its offset to its end.
        let mut records = plan.records.iter().peekable();
        for view in &layout {
            let mut at = view.offset;
            while at < view.offset + view.bytes() {
                let record = records
                    .next_if(|record| record.arena_offset == at)
                    .unwrap_or_else(|| panic!("{} is not planned at {at}", view.name));
                for source in &record.sources {
                    let (file, entry) = &tensors[&source.tensor];
                    assert_eq!(request.files[source.file as usize].name, *file);
                    let offsets = entry["data_offsets"].as_array().unwrap();
                    let bytes = offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap();
                    assert_eq!(entry["dtype"], "BF16", "{}", source.tensor);
                    assert!(bytes <= record.byte_length, "{}", source.tensor);
                }
                at += record.byte_length;
            }
            assert_eq!(at, view.offset + view.bytes(), "{}", view.name);
        }
        assert!(records.next().is_none());
        let [index] = &plan.components[..] else {
            panic!("one component")
        };
        let parsed = views::parse(&index.bytes, rank).unwrap();
        assert_eq!(parsed.views.len(), layout.len());
        identities.push(parsed.direct_identity.unwrap());
    }
    // Each rank's plan differs only by its rank.
    identities.sort_unstable();
    identities.dedup();
    assert_eq!(identities.len(), 4);
}

#[test]
fn the_bf16_checkpoint_plans_every_view() {
    check("CHOPINFER_GLM5NEXT_CHECKPOINT");
}

#[test]
fn the_nvfp4_export_plans_every_view() {
    check("CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT");
}
