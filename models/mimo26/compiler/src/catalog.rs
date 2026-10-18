#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH as GEOMETRY;

use model_compiler_sdk::source_file::SourceFile;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use runtime_weights::{DType, read_safetensors_header};

use crate::MimoResult;
use crate::config::MiMoConfig;
use crate::encoders::catalog::EncoderCatalog;
use crate::precision;

pub use model_mimo26_contract::catalog::TensorRecord;

#[derive(Clone, Debug)]
pub struct Catalog {
    pub geometry: &'static model_mimo26_contract::geometry::Geometry,
    pub sources: BTreeMap<String, SourceFile>,
    pub tensors: BTreeMap<String, TensorRecord>,
    pub encoders: EncoderCatalog,
    /// Operands checked against the retained checkpoint headers and export rules.
    pub operands: BTreeMap<String, model_compiler_sdk::quantization::import::ImportedOperand>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrafterKind {
    Mtp,
    Dflash,
}

impl DrafterKind {
    /// The contract's name of the drafter a package's drafter image holds.
    pub const fn planned(self) -> model_mimo26_contract::weight_plan::Drafter {
        match self {
            Self::Mtp => model_mimo26_contract::weight_plan::Drafter::Mtp,
            Self::Dflash => model_mimo26_contract::weight_plan::Drafter::Dflash,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mtp => "mtp",
            Self::Dflash => "dflash",
        }
    }

    pub const fn directory(self) -> &'static str {
        match self {
            Self::Mtp => "mtp",
            Self::Dflash => "dflash-mopd",
        }
    }

    pub const fn source(self) -> &'static str {
        match self {
            Self::Mtp => "model_mtp.safetensors",
            Self::Dflash => "dflash/dflash_draft_model.safetensors",
        }
    }
}

impl Catalog {
    pub fn load_drafter(root: &Path, kind: DrafterKind) -> MimoResult<Self> {
        let config = MiMoConfig::load(root)?;
        Self::load_drafter_with_config(root, kind, &config)
    }

    pub fn load_drafter_with_config(
        root: &Path,
        kind: DrafterKind,
        config: &MiMoConfig,
    ) -> MimoResult<Self> {
        if kind == DrafterKind::Dflash {
            validate_dflash_config(root, config.geometry()?)?;
            // The load-time plan reads only the checked mask archive.
            let archive = crate::checkpoint::read_metadata(
                root,
                model_mimo26_contract::weight_plan::MASK_ARCHIVE,
                16 * 1024,
            )?;
            model_mimo26_contract::weight_plan::mask_embedding(&archive)?;
        }
        let file_name = kind.source();
        let path = root.join(file_name);
        let file = open_source(&path)?;
        let header = read_safetensors_header(&file)?;
        let mut tensors = BTreeMap::new();
        for (name, info) in header.tensors() {
            let (start, _) = info.data_offsets();
            tensors.insert(
                name.clone(),
                TensorRecord {
                    name: name.clone(),
                    dtype: info.dtype(),
                    shape: info.shape().to_vec(),
                    file: file_name.into(),
                    file_start: header
                        .data_offset()
                        .checked_add(start)
                        .ok_or("offset overflow")?,
                    bytes: info.byte_len(),
                },
            );
        }
        let operands = match kind {
            // The MTP layers share the text model's export metadata.
            DrafterKind::Mtp => {
                let metadata = export_metadata(root, config)?;
                precision::admit(
                    config.geometry()?,
                    &metadata,
                    &tensors,
                    &precision::mtp_roles(config.geometry()?),
                )?
            }
            DrafterKind::Dflash => {
                validate_dflash_tensors(config.geometry()?, &tensors)?;
                precision::admit_dflash(config.geometry()?, &tensors)?
            }
        };
        file.verify()?;
        Ok(Self {
            geometry: config.geometry()?,
            sources: BTreeMap::from([(file_name.to_owned(), file)]),
            tensors,
            encoders: EncoderCatalog::default(),
            operands,
        })
    }

    pub fn load(root: &Path, config: &MiMoConfig) -> MimoResult<Self> {
        let (sources, indexed_tensors) = load_indexed(root)?;
        let metadata = export_metadata(root, config)?;
        let tensors = validate_text_metadata(config, &metadata, indexed_tensors)?;
        let operands = precision::admit(
            config.geometry()?,
            &metadata,
            &tensors,
            &precision::text_roles(config.geometry()?, |layer| config.is_global(layer)),
        )?;
        Ok(Self {
            geometry: config.geometry()?,
            sources,
            tensors,
            encoders: EncoderCatalog::load(root)?,
            operands,
        })
    }

    pub fn source(&self, name: &str) -> MimoResult<&SourceFile> {
        let source = self
            .sources
            .get(name)
            .ok_or_else(|| format!("missing retained source {name}"))?;
        source.verify()?;
        Ok(source)
    }

    pub fn tensor(&self, name: &str) -> MimoResult<&TensorRecord> {
        self.tensors
            .get(name)
            .or_else(|| self.encoders.tensors.get(name))
            .ok_or_else(|| format!("missing tensor {name}").into())
    }

    /// Build a topology component from the retained, admitted text sources
    /// and an optional MTP or DFlash drafter catalog. Encoder imports have
    /// their own owner and do not enter the projection scope.
    pub fn precision_component(
        &self,
        tp: model_mimo26_contract::Tp,
        drafter: Option<&Self>,
    ) -> MimoResult<Vec<u8>> {
        if tp.geometry != self.geometry || drafter.is_some_and(|c| c.geometry != self.geometry) {
            return Err("precision topology differs from admitted source geometry".into());
        }
        let mut tensors = BTreeMap::new();
        let mut operands = BTreeMap::new();
        for catalog in std::iter::once(self).chain(drafter) {
            for source in catalog.sources.values() {
                source.verify()?;
            }
            for (name, record) in &catalog.tensors {
                if tensors.insert(name.clone(), record.clone()).is_some() {
                    return Err("duplicate precision source".into());
                }
            }
            for (name, operand) in &catalog.operands {
                if operands.insert(name.clone(), operand.clone()).is_some() {
                    return Err("duplicate admitted precision operand".into());
                }
            }
        }
        let graph =
            model_mimo26_contract::projections::imports::derive(tp, 0, &tensors, &operands)?;
        let bytes = model_mimo26_contract::prepared::encode(&graph, tp, &operands)?;
        for catalog in std::iter::once(self).chain(drafter) {
            for source in catalog.sources.values() {
                source.verify()?;
            }
        }
        Ok(bytes)
    }
}

/// Every served topology's precision component of the checkpoint at
/// `checkpoint`, with the packaged drafter's projections.
pub fn precision_components(
    checkpoint: &Path,
    drafter: Option<DrafterKind>,
) -> MimoResult<BTreeMap<String, Vec<u8>>> {
    let config = crate::checkpoint::registry()
        .verifier(checkpoint)?
        .config()?;
    let catalog = Catalog::load(checkpoint, &config)?;
    let drafter = drafter
        .map(|kind| Catalog::load_drafter_with_config(checkpoint, kind, &config))
        .transpose()?;
    model_mimo26_contract::Tp::SUPPORTED
        .into_iter()
        .map(|ranks| {
            let tp = model_mimo26_contract::Tp::of_geometry(catalog.geometry, ranks.ranks())?;
            Ok((
                model_mimo26_contract::prepared::component_id(tp),
                catalog.precision_component(tp, drafter.as_ref())?,
            ))
        })
        .collect()
}

/// The checkpoint's export metadata: `config.json`'s `quantization_config`
/// and `hf_quant_config.json` when the checkpoint has one.
pub fn export_metadata(
    root: &Path,
    config: &MiMoConfig,
) -> MimoResult<model_compiler_sdk::quantization::import::ExportMetadata> {
    let path = root.join(precision::HF_QUANT_CONFIG);
    let hf_quant_config = match std::fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
        Ok(_) => {
            let mut file = open_source(&path)?;
            let limit = model_compiler_sdk::quantization::import::MAX_EXPORT_METADATA_BYTES;
            if file.metadata()?.len() > limit as u64 {
                return Err(format!("{} exceeds {limit} bytes", precision::HF_QUANT_CONFIG).into());
            }
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            file.verify()?;
            Some(bytes)
        }
    };
    precision::export_metadata(
        config.quantization_config.as_ref(),
        hf_quant_config.as_deref(),
    )
}

/// Validate the complete text metadata after the loader has checked shard
/// ownership: every text tensor is a MiMo role or a companion the shared
/// importer read, in an encoding MiMo serves.
fn validate_text_metadata(
    config: &MiMoConfig,
    metadata: &model_compiler_sdk::quantization::import::ExportMetadata,
    indexed_tensors: impl IntoIterator<Item = (String, TensorRecord)>,
) -> MimoResult<BTreeMap<String, TensorRecord>> {
    let mut tensors = BTreeMap::new();
    for (name, tensor) in indexed_tensors {
        if !is_text(&name) || name.starts_with("model.mtp.") {
            continue;
        }
        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(format!("duplicate text tensor {name}").into());
        }
    }
    precision::admit(
        config.geometry()?,
        metadata,
        &tensors,
        &precision::text_roles(config.geometry()?, |layer| config.is_global(layer)),
    )?;
    Ok(tensors)
}

/// Read every indexed shard through retained no-follow descriptors and validate ownership.
pub(crate) fn load_indexed(
    root: &Path,
) -> MimoResult<(BTreeMap<String, SourceFile>, BTreeMap<String, TensorRecord>)> {
    let mut source = open_source(&root.join("model.safetensors.index.json"))?;
    let mut bytes = Vec::new();
    source.read_to_end(&mut bytes)?;
    let index = runtime_weights::parse_safetensors_index(&bytes)?;
    source.verify()?;
    let indexed: std::collections::BTreeSet<_> = index.shards().collect();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "safetensors")
            && !name.to_str().is_some_and(|name| indexed.contains(name))
        {
            return Err(format!("unindexed safetensors shard {}", entry.path().display()).into());
        }
    }
    let mut sources = BTreeMap::from([("model.safetensors.index.json".to_owned(), source)]);
    let mut tensors = BTreeMap::new();
    let mut total = 0u64;
    for filename in index.shards() {
        let file = open_source(&root.join(filename))?;
        let header = read_safetensors_header(&file)?;
        for (name, info) in header.tensors() {
            if index.shard_for(name) != Some(filename) {
                return Err(format!("index ownership mismatch for {name} in {filename}").into());
            }
            total = total
                .checked_add(info.byte_len())
                .ok_or("index byte count overflow")?;
            tensors.insert(
                name.clone(),
                TensorRecord {
                    name: name.clone(),
                    dtype: info.dtype(),
                    shape: info.shape().to_vec(),
                    file: filename.to_owned(),
                    file_start: header
                        .data_offset()
                        .checked_add(info.data_offsets().0)
                        .ok_or("file offset overflow")?,
                    bytes: info.byte_len(),
                },
            );
        }
        file.verify()?;
        sources.insert(filename.to_owned(), file);
    }
    if tensors.len() != index.tensor_count() || total != index.total_size() {
        return Err("index tensor count or total size disagrees with shard headers".into());
    }
    Ok((sources, tensors))
}

pub fn open_source(path: &Path) -> MimoResult<SourceFile> {
    Ok(SourceFile::open(path)?)
}

fn is_text(name: &str) -> bool {
    name == "lm_head.weight" || name.starts_with("model.")
}

/// The DFlash drafter's tensors, all BF16. Its own config carries no
/// export metadata.
pub fn expected_dflash_tensors(
    geometry: &model_mimo26_contract::geometry::Geometry,
) -> BTreeMap<String, (DType, Vec<u64>)> {
    let d = geometry.dflash();
    let hidden = u64::from(d.hidden_size);
    let head = u64::from(d.head_dim);
    let value = u64::from(d.v_head_dim);
    let q = u64::from(d.num_attention_heads);
    let kv = u64::from(d.num_key_value_heads);
    let intermediate = u64::from(d.intermediate_size);
    let mut expected = BTreeMap::new();
    let mut add = |name: String, dtype: DType, shape: &[u64]| {
        expected.insert(name, (dtype, shape.to_vec()));
    };
    add(
        "fc.weight".into(),
        DType::Bf16,
        &[
            hidden,
            d.dflash_config.target_layer_ids.len() as u64 * u64::from(d.target_hidden_size),
        ],
    );
    for name in ["hidden_norm.weight", "norm.weight"] {
        add(name.into(), DType::Bf16, &[hidden]);
    }
    for layer in 0..d.num_hidden_layers {
        let p = format!("layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            add(format!("{p}.{norm}.weight"), DType::Bf16, &[hidden]);
        }
        for norm in ["q_norm", "k_norm"] {
            add(format!("{p}.self_attn.{norm}.weight"), DType::Bf16, &[head]);
        }
        add(
            format!("{p}.self_attn.attention_sink_bias"),
            DType::Bf16,
            &[q],
        );
        for (projection, rows) in [
            ("q_proj", q * head),
            ("k_proj", kv * head),
            ("v_proj", kv * value),
        ] {
            add(
                format!("{p}.self_attn.{projection}.weight"),
                DType::Bf16,
                &[rows, hidden],
            );
        }
        add(
            format!("{p}.self_attn.o_proj.weight"),
            DType::Bf16,
            &[hidden, q * value],
        );
        for projection in ["gate_proj", "up_proj"] {
            add(
                format!("{p}.mlp.{projection}.weight"),
                DType::Bf16,
                &[intermediate, hidden],
            );
        }
        add(
            format!("{p}.mlp.down_proj.weight"),
            DType::Bf16,
            &[hidden, intermediate],
        );
    }

    expected
}

fn validate_dflash_tensors(
    geometry: &model_mimo26_contract::geometry::Geometry,
    tensors: &BTreeMap<String, TensorRecord>,
) -> MimoResult<()> {
    let expected = expected_dflash_tensors(geometry);
    for (name, tensor) in tensors {
        let Some((dtype, shape)) = expected.get(name) else {
            return Err(format!("unknown dflash tensor {name}").into());
        };
        if tensor.dtype != *dtype || tensor.shape != *shape {
            return Err(format!("{name} has wrong dtype or shape").into());
        }
    }
    if let Some(name) = expected.keys().find(|name| !tensors.contains_key(*name)) {
        return Err(format!("missing dflash tensor {name}").into());
    }
    Ok(())
}

pub(crate) fn validate_dflash_config(
    root: &Path,
    target: &'static model_mimo26_contract::geometry::Geometry,
) -> MimoResult<()> {
    let mut file = open_source(&root.join("dflash/config.json"))?;
    if file.metadata()?.len() > 1 << 20 {
        return Err("DFlash config exceeds 1 MiB".into());
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    model_mimo26_contract::dflash::validate_config_for(target, &bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(config: &MiMoConfig) -> Vec<(String, TensorRecord)> {
        precision::tests::served_tensors(&precision::text_roles(
            config.geometry().unwrap(),
            |layer| config.is_global(layer),
        ))
        .into_iter()
        .map(|(name, (dtype, shape))| {
            let tensor = precision::tests::record(&name, dtype, &shape);
            (name, tensor)
        })
        .collect()
    }

    fn export(config: &MiMoConfig) -> model_compiler_sdk::quantization::import::ExportMetadata {
        precision::export_metadata(config.quantization_config.as_ref(), None).unwrap()
    }

    #[test]
    fn portable_mapping_accepts_complete_metadata_and_rejects_drift() {
        let config = crate::config::fixture_config();
        let export = export(&config);
        let valid = metadata(&config);
        let tensors = validate_text_metadata(&config, &export, valid.clone()).unwrap();
        assert_eq!(tensors.len(), 72_574);
        for (name, dtype, shape) in [
            ("model.embed_tokens.weight", DType::Bf16, vec![152576, 4096]),
            (
                "model.layers.0.self_attn.qkv_proj.weight",
                DType::F8E4M3,
                vec![13568, 4096],
            ),
            (
                "model.layers.0.self_attn.qkv_proj.weight_scale_inv",
                DType::F32,
                vec![108, 32],
            ),
            (
                "model.layers.1.self_attn.qkv_proj.weight_scale_inv",
                DType::F32,
                vec![116, 32],
            ),
            (
                "model.layers.1.self_attn.o_proj.weight",
                DType::Bf16,
                vec![4096, 8192],
            ),
            (
                "model.layers.47.mlp.experts.255.down_proj.weight",
                DType::U8,
                vec![4096, 1024],
            ),
            (
                "model.layers.47.mlp.experts.255.down_proj.weight_scale",
                DType::U8,
                vec![4096, 64],
            ),
        ] {
            assert_eq!((tensors[name].dtype, &tensors[name].shape), (dtype, &shape));
        }
        for (name, error) in [
            ("lm_head.weight", "missing MiMo tensor lm_head.weight"),
            (
                "model.layers.1.mlp.experts.0.gate_proj.weight_scale",
                "missing companion tensor model.layers.1.mlp.experts.0.gate_proj.weight_scale",
            ),
        ] {
            let missing = valid.iter().filter(|(key, _)| key != name).cloned();
            let found = validate_text_metadata(&config, &export, missing)
                .unwrap_err()
                .to_string();
            assert!(found.contains(error), "{found}");
        }
        for (change, error) in [
            (
                "unknown",
                "neither a MiMo tensor role nor a companion its export metadata declares",
            ),
            ("duplicate", "duplicate text tensor"),
            ("shape", "stored shape disagrees"),
            ("dtype", "disagree with weight format"),
        ] {
            let mut changed = valid.clone();
            match change {
                "unknown" => changed.push((
                    "model.unknown.weight".into(),
                    precision::tests::record("model.unknown.weight", DType::Bf16, &[4096]),
                )),
                "duplicate" => changed.push(changed[0].clone()),
                "shape" => {
                    let (name, tensor) = &mut changed[0];
                    let shape = [tensor.shape[0] + 1, tensor.shape[1]];
                    *tensor = precision::tests::record(name, tensor.dtype, &shape);
                }
                _ => {
                    let (name, tensor) = &mut changed[0];
                    *tensor = precision::tests::record(name, DType::F32, &tensor.shape);
                }
            }
            let found = validate_text_metadata(&config, &export, changed)
                .unwrap_err()
                .to_string();
            assert!(found.contains(error), "{change}: {found}");
        }
    }

    #[test]
    fn indexed_layout_checks_ownership_size_missing_tensors_and_symlinks() {
        use std::fs;
        let root = std::env::temp_dir().join(format!("mimo-index-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let header = br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[3, 7]);
        fs::write(root.join("arbitrary.safetensors"), bytes).unwrap();
        let path = root.join("model.safetensors.index.json");
        let valid = serde_json::json!({"metadata":{"total_size":2},"weight_map":{"a":"arbitrary.safetensors"}});
        fs::write(&path, valid.to_string()).unwrap();
        let (_, tensors) = load_indexed(&root).unwrap();
        assert_eq!(tensors["a"].file, "arbitrary.safetensors");
        fs::write(root.join("README.md"), "metadata").unwrap();
        load_indexed(&root).unwrap();
        fs::write(root.join("unexpected.safetensors"), []).unwrap();
        assert!(
            load_indexed(&root)
                .unwrap_err()
                .to_string()
                .contains("unindexed safetensors shard")
        );
        fs::remove_file(root.join("unexpected.safetensors")).unwrap();
        for invalid in [
            serde_json::json!({"metadata":{"total_size":3},"weight_map":{"a":"arbitrary.safetensors"}}),
            serde_json::json!({"metadata":{"total_size":2},"weight_map":{"b":"arbitrary.safetensors"}}),
            serde_json::json!({"metadata":{"total_size":2},"weight_map":{"a":"arbitrary.safetensors","missing":"arbitrary.safetensors"}}),
            serde_json::json!({"metadata":{"total_size":2},"weight_map":{"a":"../arbitrary.safetensors"}}),
        ] {
            fs::write(&path, invalid.to_string()).unwrap();
            assert!(load_indexed(&root).is_err());
        }
        fs::write(&path, valid.to_string()).unwrap();
        fs::rename(root.join("arbitrary.safetensors"), root.join("real.bytes")).unwrap();
        std::os::unix::fs::symlink("real.bytes", root.join("arbitrary.safetensors")).unwrap();
        assert!(load_indexed(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn drafter_tensor_contracts_have_expected_counts() {
        let mtp = precision::mtp_roles(&GEOMETRY);
        assert_eq!(mtp.len(), 36);
        assert_eq!(precision::tests::served_tensors(&mtp).len(), 48);
        assert_eq!(expected_dflash_tensors(&GEOMETRY).len(), 63);
    }

    #[test]
    fn real_drafter_catalogs_match_checkpoint() {
        let Ok(root) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
            eprintln!("SKIP: set CHOPINFER_MIMO_CHECKPOINT for drafter catalog test");
            return;
        };
        for (kind, count) in [(DrafterKind::Mtp, 48), (DrafterKind::Dflash, 63)] {
            let catalog = Catalog::load_drafter(Path::new(&root), kind).unwrap();
            assert_eq!(catalog.tensors.len(), count);
            let mask = [0; model_mimo26_contract::drafters::MASK_BYTES];
            for tp in model_mimo26_contract::Tp::SUPPORTED {
                for rank in 0..tp.count() {
                    let views = model_mimo26_contract::weight_plan::drafter_views(
                        &catalog.tensors,
                        &catalog.operands,
                        kind.planned(),
                        Some(&mask),
                        rank,
                        tp,
                    )
                    .unwrap();
                    let read: std::collections::BTreeSet<_> = views
                        .iter()
                        .flat_map(|view| view.pieces.iter().flat_map(|p| &p.sources))
                        .collect();
                    assert_eq!(read.len(), count, "TP{} rank {rank}", tp.ranks());
                }
            }
        }
    }

    #[test]
    fn real_catalog() {
        let Ok(root) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
            eprintln!("SKIP: set CHOPINFER_MIMO_CHECKPOINT for MiMo catalog test");
            return;
        };
        let root = Path::new(&root);
        let config = MiMoConfig::load(root).unwrap();
        let catalog = Catalog::load(root, &config).unwrap();
        assert_eq!(catalog.tensors.len(), 72_574);
        assert_eq!(catalog.encoders.tensors.len(), 848);
        assert_eq!(
            catalog.tensor("visual.merger.ln_q.weight").unwrap().shape,
            [1280]
        );
        // Every rank's planned image reads every checkpoint tensor.
        for tp in model_mimo26_contract::Tp::SUPPORTED {
            for rank in 0..tp.count() {
                let views = model_mimo26_contract::weight_plan::text_views(
                    &catalog.tensors,
                    &catalog.operands,
                    None,
                    rank,
                    tp,
                )
                .unwrap();
                let read: std::collections::BTreeSet<_> = views
                    .iter()
                    .flat_map(|view| view.pieces.iter().flat_map(|p| &p.sources))
                    .collect();
                assert_eq!(
                    read.len(),
                    catalog.tensors.len(),
                    "TP{} rank {rank}",
                    tp.ranks()
                );
            }
        }
    }
}

#[cfg(test)]
mod geometry_golden {
    use super::*;
    // Preserve the pre-refactor expansion as the golden, independent of geometry.
    fn captured_dflash_tensors() -> BTreeMap<String, (DType, Vec<u64>)> {
        let mut expected = BTreeMap::new();
        let mut add = |name: String, dtype: DType, shape: &[u64]| {
            expected.insert(name, (dtype, shape.to_vec()));
        };
        add("fc.weight".into(), DType::Bf16, &[4096, 20480]);
        for name in ["hidden_norm.weight", "norm.weight"] {
            add(name.into(), DType::Bf16, &[4096]);
        }
        for layer in 0..5 {
            let p = format!("layers.{layer}");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{p}.{norm}.weight"), DType::Bf16, &[4096]);
            }
            for norm in ["q_norm", "k_norm"] {
                add(format!("{p}.self_attn.{norm}.weight"), DType::Bf16, &[128]);
            }
            add(
                format!("{p}.self_attn.attention_sink_bias"),
                DType::Bf16,
                &[64],
            );
            for (projection, rows) in [("q_proj", 8192), ("k_proj", 1024), ("v_proj", 1024)] {
                add(
                    format!("{p}.self_attn.{projection}.weight"),
                    DType::Bf16,
                    &[rows, 4096],
                );
            }
            add(
                format!("{p}.self_attn.o_proj.weight"),
                DType::Bf16,
                &[4096, 8192],
            );
            for projection in ["gate_proj", "up_proj"] {
                add(
                    format!("{p}.mlp.{projection}.weight"),
                    DType::Bf16,
                    &[16384, 4096],
                );
            }
            add(
                format!("{p}.mlp.down_proj.weight"),
                DType::Bf16,
                &[4096, 16384],
            );
        }

        expected
    }

    #[test]
    fn dflash_catalog_matches_captured_expansion() {
        assert_eq!(
            expected_dflash_tensors(&GEOMETRY),
            captured_dflash_tensors()
        );
    }
}
