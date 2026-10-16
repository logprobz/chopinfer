//! Sparse checkpoint and weight-plan proof for a compiled geometry.
use model_qwen38::{Result, catalog, config::Qwen38Config};
use runtime_bundle_abi::weight_plan::{
    CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1, WeightPlanV1,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::PathBuf,
};

// Count each catalog tensor's stored BF16 bytes independently of the view
// offsets and planner arena size, including the vision MLP's row padding.
fn encoder_bytes<'a>(tensors: impl Iterator<Item = (&'a str, &'a [u64])>) -> u64 {
    tensors
        .filter(|(name, _)| name.starts_with("model.visual."))
        .map(|(name, source_shape)| {
            let mut shape = source_shape.to_vec();
            if name.starts_with("model.visual.blocks.") {
                if name.ends_with(".mlp.linear_fc1.weight")
                    || name.ends_with(".mlp.linear_fc1.bias")
                {
                    shape[0] = shape[0].next_multiple_of(128);
                } else if name.ends_with(".mlp.linear_fc2.weight") {
                    shape[1] = shape[1].next_multiple_of(128);
                }
            }
            (2 * shape.iter().product::<u64>()).next_multiple_of(4096)
        })
        .sum()
}

fn check_destinations(plan: &WeightPlanV1, expected_bytes: u64) {
    assert_eq!(plan.arena_bytes, expected_bytes, "{}", plan.image_role);
    let mut end = 0;
    for record in &plan.records {
        assert!(
            record.arena_offset >= end,
            "overlapping weight-plan destinations"
        );
        end = record.arena_offset.checked_add(record.byte_length).unwrap();
        assert!(end <= plan.arena_bytes);
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: geometry-checkpoint CONFIG TOKENIZER_DIR OUTPUT".into());
    }
    let config = fs::read(&args[0])?;
    let geometry = Qwen38Config::from_bytes(&config)?.geometry()?.geometry;
    let value = serde_json::from_slice(&config)?;
    let tensors = catalog::expected_tensors_with_config(geometry, &value)?;
    let output = PathBuf::from(&args[2]);
    fs::create_dir(&output)?;
    fs::write(output.join("config.json"), &config)?;
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        fs::copy(PathBuf::from(&args[1]).join(name), output.join(name))?;
    }
    model_qwen38::tokenizer::load_tokenizer(&output)?;
    let mut header = BTreeMap::new();
    let mut weight_map = BTreeMap::new();
    let mut offset = 0_u64;
    for (name, (dtype, shape)) in tensors {
        let width = match dtype.as_str() {
            "BF16" => 2,
            "F32" => 4,
            "I64" => 8,
            "F8_E4M3" => 1,
            _ => return Err(format!("unsupported fixture dtype {dtype}").into()),
        };
        let bytes = shape
            .iter()
            .try_fold(width, |n: u64, &dim| n.checked_mul(dim))
            .ok_or("fixture tensor size overflow")?;
        let end = offset.checked_add(bytes).ok_or("fixture offset overflow")?;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape,
            "data_offsets": [offset, end]}),
        );
        weight_map.insert(name, "model.safetensors");
        offset = end;
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut file = fs::File::create(output.join("model.safetensors"))?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    let length = 8 + header.len() as u64 + offset;
    file.set_len(length)?;
    let index = serde_json::to_vec(&serde_json::json!({
        "metadata": {"total_size": offset}, "weight_map": weight_map}))?;
    fs::write(output.join("model.safetensors.index.json"), &index)?;
    for ranks in [2, 4] {
        let catalog = catalog::Catalog::load_ranks(&output, ranks as usize)?;
        assert_eq!(catalog.geometry.id, geometry.id);
        let shape = model_qwen38::placement::bundle_shape_for(geometry, ranks as usize)?;
        for role in [
            model_qwen38_contract::contract::RANK_IMAGE_ROLE,
            model_qwen38_contract::encoder::views::IMAGE_ROLE,
        ] {
            let mut consumed = BTreeSet::new();
            for rank in 0..ranks {
                let request = WeightPlanRequestV1 {
                    image_role: std::str::from_utf8(role)?.into(),
                    ranks,
                    rank,
                    files: vec![CheckpointFileV1 {
                        name: "model.safetensors".into(),
                        byte_length: length,
                        data_offset: 8 + header.len() as u64,
                        header: header.clone(),
                    }],
                    documents: vec![
                        NamedBytesV1 {
                            name: "config.json".into(),
                            bytes: config.clone(),
                        },
                        NamedBytesV1 {
                            name: "model.safetensors.index.json".into(),
                            bytes: index.clone(),
                        },
                    ],
                    ..Default::default()
                };
                let plan = model_qwen38_contract::weight_plan::plan(&request)?
                    .ok_or("image role was declined")?;
                assert!(!plan.records.is_empty());
                let expected_bytes = if role == model_qwen38_contract::contract::RANK_IMAGE_ROLE {
                    model_qwen38_contract::views::image_bytes(rank, shape)
                } else {
                    encoder_bytes(
                        catalog
                            .tensors
                            .iter()
                            .map(|(name, tensor)| (name.as_str(), tensor.shape.as_slice())),
                    )
                };
                check_destinations(&plan, expected_bytes);
                for record in &plan.records {
                    for source in &record.sources {
                        let tensor = &catalog.tensors[&source.tensor];
                        assert_eq!(source.file, 0);
                        consumed.insert(source.tensor.clone());
                        if let Some(axis) = record.axis {
                            let bound = tensor.shape[axis as usize];
                            for range in &record.ranges {
                                assert!(
                                    range.start < range.end && range.end <= bound,
                                    "{}: {:?} outside axis {axis} of {:?}",
                                    source.tensor,
                                    range,
                                    tensor.shape
                                );
                            }
                        } else {
                            assert!(record.ranges.is_empty());
                        }
                    }
                }
                println!(
                    "geometry {} TP{ranks} rank {rank}: {} planned",
                    geometry.id, request.image_role
                );
            }
            let expected = catalog
                .tensors
                .keys()
                .filter(|name| {
                    name.starts_with("model.visual.")
                        == (role == model_qwen38_contract::encoder::views::IMAGE_ROLE)
                })
                .cloned()
                .collect::<BTreeSet<_>>();
            assert_eq!(
                consumed,
                expected,
                "TP{ranks}: {} tensor coverage",
                std::str::from_utf8(role)?
            );
        }
    }
    println!(
        "sparse checkpoint: {} tensors, {length} logical bytes",
        weight_map.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::weight_plan::WeightRecordV1;

    fn plan() -> WeightPlanV1 {
        WeightPlanV1 {
            ranks: 2,
            rank: 0,
            image_role: "encoder".into(),
            alignment: 4096,
            arena_bytes: 8192,
            views: vec![],
            components: vec![],
            records: [0, 4096]
                .map(|arena_offset| WeightRecordV1 {
                    arena_offset,
                    byte_length: 4096,
                    transform: 0,
                    params: vec![],
                    sources: vec![],
                    axis: None,
                    ranges: vec![],
                    stored_shape: vec![2048],
                })
                .to_vec(),
        }
    }

    #[test]
    fn encoder_size_counts_catalog_shapes_and_padding() {
        let tensors = [
            ("model.visual.blocks.0.mlp.linear_fc1.weight", vec![129, 16]),
            ("model.visual.blocks.0.mlp.linear_fc1.bias", vec![129]),
            ("model.visual.blocks.0.mlp.linear_fc2.weight", vec![16, 129]),
            ("model.visual.merger.linear_fc1.bias", vec![129]),
            ("model.layers.0.input_layernorm.weight", vec![16]),
        ];
        assert_eq!(
            encoder_bytes(tensors.iter().map(|(n, s)| (*n, s.as_slice()))),
            24576
        );
        check_destinations(&plan(), 8192);
    }

    #[test]
    #[should_panic(expected = "overlapping weight-plan destinations")]
    fn reject_overlapping_destinations_inside_the_arena() {
        let mut plan = plan();
        plan.records[1].arena_offset = 2048;
        check_destinations(&plan, 8192);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn reject_an_arena_larger_than_the_catalog_sum() {
        let mut plan = plan();
        plan.arena_bytes = 12288;
        check_destinations(&plan, 8192);
    }
}
