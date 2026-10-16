//! Sparse MiMo text, DFlash and encoder checkpoint and complete image-plan proof.
use model_mimo26::{
    MimoResult,
    catalog::{Catalog, DrafterKind},
    config::MiMoConfig,
    precision,
};
use model_mimo26_contract::{
    Tp,
    encoders::{self, Encoders},
    weight_plan,
};
use runtime_bundle_abi::weight_plan::{CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1};
use runtime_weights::DType;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

type Tensors = BTreeMap<String, (DType, Vec<u64>)>;

fn shard(root: &Path, name: &str, tensors: &Tensors) -> MimoResult<(CheckpointFileV1, u64)> {
    let mut header = BTreeMap::new();
    let mut offset = 0_u64;
    for (name, (dtype, shape)) in tensors {
        let width = match dtype {
            DType::Bf16 => 2,
            DType::F32 => 4,
            DType::F8E4M3 | DType::U8 => 1,
            _ => return Err("unsupported fixture dtype".into()),
        };
        let bytes = shape
            .iter()
            .try_fold(width, |n: u64, &dim| n.checked_mul(dim))
            .ok_or("tensor size overflow")?;
        let end = offset.checked_add(bytes).ok_or("tensor offset overflow")?;
        header.insert(
            name,
            serde_json::json!({"dtype": dtype.safetensors_name(), "shape": shape,
            "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let data_offset = 8 + header.len() as u64;
    let path = root.join(name);
    fs::create_dir_all(path.parent().unwrap())?;
    let mut file = fs::File::create(path)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    let byte_length = data_offset
        .checked_add(offset)
        .ok_or("file size overflow")?;
    file.set_len(byte_length)?;
    Ok((
        CheckpointFileV1 {
            name: name.into(),
            byte_length,
            data_offset,
            header,
        },
        offset,
    ))
}

fn main() -> MimoResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: geometry-checkpoint CONFIG TOKENIZER_DIR MASK_ARCHIVE OUTPUT".into());
    }
    let config_bytes = fs::read(&args[0])?;
    let config: MiMoConfig = serde_json::from_slice(&config_bytes)?;
    config.validate()?;
    let geometry = config.geometry()?;
    let value: serde_json::Value = serde_json::from_slice(&config_bytes)?;
    let root = PathBuf::from(&args[3]);
    fs::create_dir(&root)?;
    fs::write(root.join("config.json"), &config_bytes)?;
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        fs::copy(PathBuf::from(&args[1]).join(name), root.join(name))?;
    }
    fs::create_dir(root.join("dflash"))?;
    fs::create_dir(root.join("audio_tokenizer"))?;
    let dflash_config = serde_json::to_vec(&value["dflash"])?;
    let audio_config = serde_json::to_vec(&value["audio_tokenizer_config"])?;
    fs::write(root.join("dflash/config.json"), &dflash_config)?;
    fs::write(root.join("audio_tokenizer/config.json"), &audio_config)?;
    let mask = fs::read(&args[2])?;
    weight_plan::mask_embedding(&mask)?;
    fs::write(root.join(weight_plan::MASK_ARCHIVE), &mask)?;
    let roles = precision::text_roles(geometry, |layer| geometry.is_global(layer as u32));
    let mut main_tensors = precision::expected_source_tensors(&roles)?;
    let mut audio_tensors = Tensors::new();
    for (name, (dtype, shape, source)) in encoders::expected_for(geometry) {
        let dtype = match dtype {
            "BF16" => DType::Bf16,
            "F32" => DType::F32,
            _ => return Err("unsupported encoder dtype".into()),
        };
        if source == encoders::MAIN_FILE {
            main_tensors.insert(name, (dtype, shape));
        } else if source == encoders::TOKENIZER_FILE {
            audio_tensors.insert(name, (dtype, shape));
        } else {
            return Err("unsupported encoder source".into());
        }
    }
    let (main, total_size) = shard(&root, encoders::MAIN_FILE, &main_tensors)?;
    let (audio, _) = shard(&root, encoders::TOKENIZER_FILE, &audio_tensors)?;
    let (draft, _) = shard(
        &root,
        weight_plan::DFLASH_FILE,
        &model_mimo26::catalog::expected_dflash_tensors(geometry),
    )?;
    let weight_map = main_tensors
        .keys()
        .map(|name| (name, encoders::MAIN_FILE))
        .collect::<BTreeMap<_, _>>();
    let index = serde_json::to_vec(
        &serde_json::json!({"metadata":{"total_size":total_size},"weight_map":weight_map}),
    )?;
    fs::write(root.join("model.safetensors.index.json"), &index)?;
    let files = vec![audio, draft, main];
    let documents = vec![
        NamedBytesV1 {
            name: "config.json".into(),
            bytes: config_bytes,
        },
        NamedBytesV1 {
            name: "model.safetensors.index.json".into(),
            bytes: index,
        },
        NamedBytesV1 {
            name: "dflash/config.json".into(),
            bytes: dflash_config,
        },
        NamedBytesV1 {
            name: "audio_tokenizer/config.json".into(),
            bytes: audio_config,
        },
        NamedBytesV1 {
            name: weight_plan::MASK_ARCHIVE.into(),
            bytes: mask,
        },
    ];
    let catalog = Catalog::load(&root, &config)?;
    let draft_catalog = Catalog::load_drafter_with_config(&root, DrafterKind::Dflash, &config)?;
    for ranks in [2, 4] {
        let tp = Tp::of_geometry(geometry, ranks)?;
        let text_bytes = model_mimo26::rank_bytes::estimated_rank_bytes(&catalog, None, tp)?;
        let draft_bytes = model_mimo26::rank_bytes::estimated_drafter_rank_bytes(
            &draft_catalog,
            DrafterKind::Dflash,
            tp,
        )?;
        let encoder_bytes = model_mimo26::encoders::image::estimated_rank_bytes(
            &catalog.encoders,
            tp,
            Encoders::ALL,
        )?;
        for (role, tensors, sizes) in [
            (weight_plan::TEXT_IMAGE_ROLE, &catalog.tensors, &text_bytes),
            (
                weight_plan::DRAFTER_IMAGE_ROLE,
                &draft_catalog.tensors,
                &draft_bytes,
            ),
            (
                weight_plan::ENCODER_IMAGE_ROLE,
                &catalog.encoders.tensors,
                &encoder_bytes,
            ),
        ] {
            let mut consumed = BTreeSet::new();
            for rank in 0..ranks {
                let request = WeightPlanRequestV1 {
                    ranks,
                    rank,
                    image_role: role.into(),
                    files: files.clone(),
                    documents: documents.clone(),
                    ..Default::default()
                };
                let plan = weight_plan::plan(&request)?.ok_or("image role was declined")?;
                assert_eq!(
                    plan.arena_bytes, sizes[rank as usize],
                    "{role} TP{ranks} rank {rank}"
                );
                assert!(!plan.records.is_empty());
                let mut end = 0;
                for record in &plan.records {
                    assert!(record.arena_offset >= end);
                    end = record.arena_offset.checked_add(record.byte_length).unwrap();
                    assert!(end <= plan.arena_bytes);
                    for source in &record.sources {
                        let tensor = &tensors[&source.tensor];
                        consumed.insert(source.tensor.clone());
                        assert_eq!(files[source.file as usize].name, tensor.file);
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
                    "geometry {} TP{ranks} rank {rank}: {role} planned",
                    geometry.id
                );
            }
            assert_eq!(
                consumed,
                tensors.keys().cloned().collect::<BTreeSet<_>>(),
                "{role} TP{ranks} tensor coverage"
            );
        }
    }
    println!(
        "sparse checkpoint: {} indexed tensors, {total_size} logical indexed bytes",
        main_tensors.len()
    );
    Ok(())
}
