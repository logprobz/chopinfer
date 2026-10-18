//! A checkpoint whose DFlash drafter does not check (here, its mask
//! archive is not the checked one) is served with MTP: the compiler does not
//! select DFlash, and the load-time plan of a package that drafts with MTP
//! leaves the DFlash assets unread. A package that names DFlash still
//! refuses them. The checkpoint is sparse past its headers.
use model_mimo26::catalog::{Catalog, DrafterKind};
use model_mimo26::config::MiMoConfig;
use model_mimo26::precision;
use model_mimo26_contract::{Tp, weight_plan};
use runtime_bundle_abi::weight_plan::{CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1};
use runtime_weights::DType;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

type Tensors = BTreeMap<String, (DType, Vec<u64>)>;

const TEXT_FILE: &str = "model-00001-of-00001.safetensors";

/// Writes `tensors` as the sparse safetensors file `name` under `root`.
fn shard(root: &Path, name: &str, tensors: &Tensors) -> (CheckpointFileV1, u64) {
    let mut header = BTreeMap::new();
    let mut offset = 0_u64;
    for (name, (dtype, shape)) in tensors {
        let width = match dtype {
            DType::Bf16 => 2,
            DType::F32 => 4,
            DType::F8E4M3 | DType::U8 => 1,
            _ => panic!("unsupported fixture dtype"),
        };
        let end = offset + shape.iter().product::<u64>() * width;
        header.insert(
            name,
            serde_json::json!({"dtype": dtype.safetensors_name(), "shape": shape,
            "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let data_offset = 8 + header.len() as u64;
    let path = root.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(data_offset + offset).unwrap();
    (
        CheckpointFileV1 {
            name: name.into(),
            byte_length: data_offset + offset,
            data_offset,
            header,
        },
        offset,
    )
}

struct Checkpoint {
    root: PathBuf,
    config: MiMoConfig,
    files: Vec<CheckpointFileV1>,
    documents: Vec<NamedBytesV1>,
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// The MiMo V2.6 Flash checkpoint of the fixture config, with its MTP
/// layers and a DFlash drafter whose mask archive is not the checked one.
fn checkpoint() -> Checkpoint {
    let root = std::env::temp_dir().join(format!("mimo-local-drafter-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    let config_bytes = include_bytes!("../fixtures/config.json").to_vec();
    let config: MiMoConfig = serde_json::from_slice(&config_bytes).unwrap();
    let geometry = config.geometry().unwrap();
    std::fs::write(root.join("config.json"), &config_bytes).unwrap();
    let text = precision::expected_source_tensors(&precision::text_roles(geometry, |layer| {
        geometry.is_global(layer as u32)
    }))
    .unwrap();
    let mtp = precision::expected_source_tensors(&precision::mtp_roles(geometry)).unwrap();
    let (text_file, text_bytes) = shard(&root, TEXT_FILE, &text);
    let (mtp_file, mtp_bytes) = shard(&root, weight_plan::MTP_FILE, &mtp);
    let (dflash_file, _) = shard(
        &root,
        weight_plan::DFLASH_FILE,
        &model_mimo26::catalog::expected_dflash_tensors(geometry),
    );
    let weight_map: BTreeMap<_, _> = text
        .keys()
        .map(|name| (name.as_str(), TEXT_FILE))
        .chain(
            mtp.keys()
                .map(|name| (name.as_str(), weight_plan::MTP_FILE)),
        )
        .collect();
    let index = serde_json::to_vec(&serde_json::json!({
        "metadata": {"total_size": text_bytes + mtp_bytes},
        "weight_map": weight_map,
    }))
    .unwrap();
    std::fs::write(root.join("model.safetensors.index.json"), &index).unwrap();
    // The geometry's own drafter config, which checks.
    let document: serde_json::Value = serde_json::from_str(geometry.document).unwrap();
    let dflash_config = serde_json::to_vec(&document["dflash"]).unwrap();
    std::fs::write(root.join("dflash/config.json"), &dflash_config).unwrap();
    let mask = vec![0_u8; 64];
    std::fs::write(root.join(weight_plan::MASK_ARCHIVE), &mask).unwrap();
    let named = |name: &str, bytes: Vec<u8>| NamedBytesV1 {
        name: name.into(),
        bytes,
    };
    let mut files = vec![dflash_file, mtp_file, text_file];
    files.sort_by(|a, b| a.name.cmp(&b.name));
    Checkpoint {
        root,
        config,
        files,
        documents: vec![
            named("config.json", config_bytes),
            named("dflash/config.json", dflash_config),
            named(weight_plan::MASK_ARCHIVE, mask),
            named("model.safetensors.index.json", index),
        ],
    }
}

impl Checkpoint {
    /// The request for `role` at `rank` of `tp`, from a package that names
    /// `drafter` (none: a request with no package components).
    fn request(&self, role: &str, tp: Tp, rank: u32, drafter: Option<&str>) -> WeightPlanRequestV1 {
        WeightPlanRequestV1 {
            image_role: role.into(),
            ranks: tp.ranks(),
            rank,
            files: self.files.clone(),
            documents: self.documents.clone(),
            components: drafter
                .map(|drafter| NamedBytesV1 {
                    name: weight_plan::DRAFTER_KIND_ROLE.into(),
                    bytes: serde_json::to_vec(&serde_json::json!({"drafter": drafter})).unwrap(),
                })
                .into_iter()
                .collect(),
            ..Default::default()
        }
    }
}

#[test]
fn a_dflash_drafter_that_does_not_check_leaves_mtp() {
    let checkpoint = checkpoint();
    let error = Catalog::load_drafter_with_config(
        &checkpoint.root,
        DrafterKind::Dflash,
        &checkpoint.config,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(
        error.contains("differs from the checked BF16 tensor archive"),
        "{error}"
    );
    let mtp =
        Catalog::load_drafter_with_config(&checkpoint.root, DrafterKind::Mtp, &checkpoint.config)
            .unwrap();
    for tp in Tp::SUPPORTED {
        let tp = Tp::of_geometry(checkpoint.config.geometry().unwrap(), tp.ranks()).unwrap();
        let drafter_bytes =
            model_mimo26::rank_bytes::estimated_drafter_rank_bytes(&mtp, DrafterKind::Mtp, tp)
                .unwrap();
        for rank in 0..tp.ranks() {
            // A package that drafts with MTP, and a request that names no
            // package, plan the text and MTP drafter images.
            for drafter in [Some("mtp"), None] {
                let text = checkpoint.request(weight_plan::TEXT_IMAGE_ROLE, tp, rank, drafter);
                assert!(
                    !weight_plan::plan(&text)
                        .unwrap()
                        .unwrap()
                        .records
                        .is_empty()
                );
                let draft = checkpoint.request(weight_plan::DRAFTER_IMAGE_ROLE, tp, rank, drafter);
                let plan = weight_plan::plan(&draft).unwrap().unwrap();
                assert_eq!(
                    plan.arena_bytes,
                    drafter_bytes[rank as usize],
                    "TP{} rank {rank}",
                    tp.ranks()
                );
            }
            // A package that names DFlash refuses its assets.
            let refused =
                checkpoint.request(weight_plan::TEXT_IMAGE_ROLE, tp, rank, Some("dflash"));
            let error = weight_plan::plan(&refused).unwrap_err();
            assert!(
                error.contains("differs from the checked BF16 tensor archive"),
                "{error}"
            );
        }
    }
}
