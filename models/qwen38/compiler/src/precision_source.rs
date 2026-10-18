//! Qwen tensor roles and logical geometry. Shared importers own checkpoint
//! encodings, scale interpretation and activation recipes.
#[cfg(test)]
use crate::catalog::expected_tensors;
use crate::{Result, config::Qwen38Config};
use kernel_core::contract::{DType, WeightFormat};
use model_compiler_sdk::{
    quantization::{
        import::{
            ActivationRecipe, EncodingRecipe, ExportMetadata, HF_QUANT_CONFIG, ImportedOperand,
            OperandRole, import_operand,
        },
        source::TensorInventory,
    },
    source::SourceFiles,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct SourcePrecision {
    pub inventory: TensorInventory,
    pub metadata: ExportMetadata,
    pub operands: BTreeMap<String, ImportedOperand>,
}

fn table_hint(config: &Qwen38Config, metadata: &ExportMetadata) -> Result<Option<EncodingRecipe>> {
    let format = match config.text_config.ple_embedding_dtype.as_deref() {
        Some("nvfp4") => Some(WeightFormat::Nvfp4),
        Some("fp8") => Some(WeightFormat::Fp8PerTensor),
        Some("bfloat16" | "bf16") => Some(WeightFormat::Bf16),
        Some(other) => {
            return Err(format!("unsupported text_config.ple_embedding_dtype: {other}").into());
        }
        // HF's block recipe applies to linear projections. This family's
        // additional table conversion uses one table-global FP8 scale.
        None if metadata
            .default
            .is_some_and(|r| matches!(r.weight, WeightFormat::Fp8Block128x128(_))) =>
        {
            Some(WeightFormat::Fp8PerTensor)
        }
        None => None,
    };
    Ok(format.map(|weight| EncodingRecipe {
        weight,
        activation: Some(ActivationRecipe::Bf16),
    }))
}

fn role(
    config: &Qwen38Config,
    metadata: &ExportMetadata,
    name: &str,
    shape: Vec<u64>,
    binding: Option<&model_qwen38_contract::projections::ports::SourceBinding>,
) -> Result<OperandRole> {
    let stem = name
        .strip_suffix(".weight")
        .ok_or("expected a weight role")?;
    let table = binding
        .filter(|b| b.shard_rows.is_some())
        .map(|b| b.operation.clone());
    let lookup = table.is_some() || binding.is_some_and(|b| b.projection == "embedding");
    let mut targets = vec![stem.into()];
    if let Some(table) = &table {
        targets.push(table.clone());
        targets.push("ple.ple_embedding.ngram_embedding".into());
    }
    let declared = if table.is_some() {
        table_hint(config, metadata)?
    } else if lookup {
        Some(EncodingRecipe::bf16())
    } else {
        None
    };
    Ok(OperandRole {
        operation: stem.into(),
        weight: name.into(),
        targets,
        logical_shape: shape,
        lookup,
        shared_scale_stem: table,
        declared,
    })
}

impl SourcePrecision {
    pub fn from_captured(files: &SourceFiles) -> Result<Self> {
        let config =
            Qwen38Config::from_bytes(files.inputs.get("config.json").ok_or("missing config")?)?;
        let metadata = ExportMetadata::from_documents(
            Some(&config.quantization_config),
            files.inputs.get(HF_QUANT_CONFIG).map(Vec::as_slice),
        )?;
        let inventory = TensorInventory::from_captured(files, "model.safetensors.index.json")?;
        Self::admit(
            &config,
            metadata,
            inventory,
            crate::catalog::expected_tensors_with_config(
                config.geometry()?.geometry,
                &serde_json::to_value(&config)?,
            )?,
        )
    }

    pub(crate) fn admit(
        config: &Qwen38Config,
        metadata: ExportMetadata,
        inventory: TensorInventory,
        logical: BTreeMap<String, (String, Vec<u64>)>,
    ) -> Result<Self> {
        let definitions = model_qwen38_contract::projections::ports::get(
            model_qwen38_contract::contract::Shape::new(
                config.geometry()?.geometry,
                model_qwen38_contract::contract::Ranks::TP4,
            ),
        )?;
        let bindings = definitions.bindings()?;
        let mut accounted = BTreeSet::new();
        let mut projections = Vec::new();
        // The first error of the walk in logical order: a check or role
        // failing here stops the walk, and only the projections before it
        // import, so an earlier projection's import error still comes first.
        let mut walk_error = None;
        for (name, (dtype, shape)) in logical {
            // Companion names and physical dimensions follow the imported
            // encoding, never the reference checkpoint's companion list.
            if name.ends_with(".weight_scale") || name.ends_with(".weight_scale_inv") {
                continue;
            }
            if name.ends_with(".weight") && shape.len() == 2 {
                match role(config, &metadata, &name, shape, bindings.get(&name)) {
                    Ok(role) => projections.push((name, role)),
                    Err(error) => {
                        walk_error = Some(error);
                        break;
                    }
                }
            } else if let Err(error) = (|| -> Result<()> {
                let actual = inventory
                    .tensors
                    .get(&name)
                    .ok_or_else(|| format!("missing Qwen tensor role {name}"))?;
                let expected = match dtype.as_str() {
                    "BF16" => DType::Bf16,
                    "F32" => DType::F32,
                    "I64" => DType::I64,
                    _ => {
                        return Err(format!(
                            "unsupported non-projection role dtype {dtype}: {name}"
                        )
                        .into());
                    }
                };
                // A norm with an F32 consumer may be stored in F32: HF
                // computes `(1 + w)` in F32, so it is bound as stored and
                // never narrowed (`model_qwen38_contract::norms`).
                let f32_norm = expected == DType::Bf16
                    && actual.dtype == DType::F32
                    && model_qwen38_contract::norms::f32_consumer(&name);
                if (actual.dtype != expected && !f32_norm) || actual.shape != shape {
                    return Err(format!("tensor {name}: Qwen side calculation/state requires {dtype} {shape:?}, received {:?} {:?}", actual.dtype, actual.shape).into());
                }
                // Module weight rules cannot silently quantize a norm or
                // convolution. Their bias and integer state keep the explicit
                // dtype above, independent of a module's projection recipe.
                metadata.require_unconverted(&name)?;
                Ok(())
            })() {
                walk_error = Some(error);
                break;
            } else {
                accounted.insert(name);
            }
        }
        // Imports read the metadata and inventory only, and the recipes
        // registered above name norm and state stems, never a projection's;
        // so each projection imports on its own, on several threads, and the
        // results keep the logical order (the first error in that order).
        let imported = import_in_parallel(&metadata, &inventory, &projections);
        let mut operands = BTreeMap::new();
        for ((name, _), operand) in projections.into_iter().zip(imported) {
            let operand = operand?;
            accounted.extend(operand.source_tensors.iter().cloned());
            operands.insert(name, operand);
        }
        if let Some(error) = walk_error {
            return Err(error);
        }
        inventory.require_accounted(&accounted)?;
        Ok(Self {
            inventory,
            metadata,
            operands,
        })
    }

    /// The norm weights the export stores in F32 for their F32 consumers.
    pub fn f32_norms(&self) -> impl Iterator<Item = &kernel_core::precision::TensorSpan> {
        self.inventory.tensors.iter().filter_map(|(name, span)| {
            (span.dtype == DType::F32 && model_qwen38_contract::norms::f32_consumer(name))
                .then_some(span)
        })
    }

    /// Refuse a non-finite element of any F32 norm weight. Admission reads
    /// headers only; this reads those weights' payload from `files`.
    pub fn require_finite_f32_norms(&self, files: &SourceFiles) -> Result<()> {
        for span in self.f32_norms() {
            let bytes = self.inventory.read(files, span)?;
            if let Some(element) = bytes
                .chunks_exact(4)
                .position(|v| !f32::from_le_bytes(v.try_into().expect("four bytes")).is_finite())
            {
                return Err(format!(
                    "tensor {}: F32 norm weight element {element} is not finite",
                    span.tensor
                )
                .into());
            }
        }
        Ok(())
    }

    pub fn require_bf16_vision(&self) -> Result<()> {
        if let Some((name, operand)) = self.operands.iter().find(|(name, operand)| {
            name.starts_with("model.visual.") && operand.weight.format != WeightFormat::Bf16
        }) {
            return Err(format!("tensor {name}: requested vision operation requires an unavailable {:?} encoder contract; this bundle supplies BF16 vision", operand.weight.format).into());
        }
        Ok(())
    }

    /// Whether the old rank-image contract applies without a precision
    /// descriptor. Other source recipes need prepared operation bindings.
    pub fn legacy_text_layout(&self) -> bool {
        self.operands
            .iter()
            .filter(|(name, _)| !name.starts_with("model.visual."))
            .all(|(name, operand)| {
                let weight = &operand.weight;
                if name.contains(".ngram_embedding.") {
                    weight.format == WeightFormat::Fp8PerTensor
                        && weight
                            .local_scale
                            .as_ref()
                            .is_some_and(|s| s.tensor.dtype == DType::Bf16)
                } else if name.contains(".experts.") {
                    matches!(weight.format, WeightFormat::Fp8Block128x128(_))
                } else {
                    weight.format == WeightFormat::Bf16
                }
            })
    }
}

/// [`import_operand`] for every projection, in order, on up to eight threads.
fn import_in_parallel(
    metadata: &ExportMetadata,
    inventory: &TensorInventory,
    projections: &[(String, OperandRole)],
) -> Vec<std::result::Result<ImportedOperand, model_compiler_sdk::CompilerError>> {
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, 8);
    let chunk = projections.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let workers: Vec<_> = projections
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|(_, role)| import_operand(metadata, role, &inventory.tensors))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("an import worker panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixed_layouts::Layout;
    use kernel_core::precision::TensorSpan;

    fn config() -> Qwen38Config {
        Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap()
    }

    fn tensor(name: &str, dtype: DType, shape: &[u64]) -> TensorSpan {
        TensorSpan {
            tensor: name.into(),
            dtype,
            shape: shape.into(),
            offset: 0,
            byte_length: shape.iter().product::<u64>() * kernel_core::precision::dtype_bits(dtype)
                / 8,
        }
    }

    struct MixedFixture {
        files: SourceFiles,
        root: std::path::PathBuf,
    }

    impl Drop for MixedFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }

    fn mixed_fixture(layout: Layout) -> Result<MixedFixture> {
        mixed_fixture_with(layout, &[])
    }

    /// The layout's synthetic checkpoint with each listed F32 element
    /// `(tensor, element, value)` written into the otherwise zero shard
    /// before capture.
    fn mixed_fixture_with(layout: Layout, values: &[(&str, u64, f32)]) -> Result<MixedFixture> {
        use std::io::Write;
        use std::os::unix::fs::FileExt;
        let root = std::env::temp_dir().join(format!(
            "qwen38-mixed-{}-{}-{}",
            layout.name(),
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&root)?;
        let tensors = crate::mixed_layouts::tensors(layout);
        let weight_map: serde_json::Map<String, serde_json::Value> = tensors
            .iter()
            .map(|(name, _, _)| (name.clone(), "synthetic.safetensors".into()))
            .collect();
        for (document, value) in [
            ("config.json", crate::mixed_layouts::config(layout)),
            (
                HF_QUANT_CONFIG,
                crate::mixed_layouts::hf_quant_config(layout),
            ),
            (
                "model.safetensors.index.json",
                serde_json::json!({ "weight_map": weight_map }),
            ),
        ] {
            std::fs::write(root.join(document), serde_json::to_vec(&value)?)?;
        }
        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for (name, dtype, shape) in tensors {
            let width = match dtype.as_str() {
                "U8" | "F8_E4M3" => 1,
                "BF16" => 2,
                "F32" => 4,
                other => panic!("unsupported fixture dtype {other}"),
            };
            let end = offset + shape.iter().product::<u64>() * width;
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype, "shape": shape, "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }
        let starts: BTreeMap<String, u64> = header
            .iter()
            .map(|(name, entry)| (name.clone(), entry["data_offsets"][0].as_u64().unwrap()))
            .collect();
        let header = serde_json::to_vec(&header)?;
        let mut shard = std::fs::File::create(root.join("synthetic.safetensors"))?;
        shard.write_all(&(header.len() as u64).to_le_bytes())?;
        shard.write_all(&header)?;
        // Sparse zero data keeps the production shapes without reading weights.
        shard.set_len(8 + header.len() as u64 + offset)?;
        for &(tensor, element, value) in values {
            shard.write_all_at(
                &value.to_le_bytes(),
                8 + header.len() as u64 + starts[tensor] + 4 * element,
            )?;
        }
        let files = SourceFiles::capture(
            &root,
            &[
                ("config.json", 1024 * 1024),
                ("model.safetensors.index.json", 1024 * 1024),
            ],
            &[(HF_QUANT_CONFIG, 1024 * 1024)],
            "model.safetensors.index.json",
        )?;
        Ok(MixedFixture { files, root })
    }

    fn admit_mixed_fixture(files: &SourceFiles) -> Result<SourcePrecision> {
        let config = Qwen38Config::from_bytes(&files.inputs["config.json"])?;
        let metadata = ExportMetadata::from_documents(
            Some(&config.quantization_config),
            Some(&files.inputs[HF_QUANT_CONFIG]),
        )?;
        let inventory = TensorInventory::from_captured(files, "model.safetensors.index.json")?;
        let mut logical = crate::catalog::expected_tensors_with_config(
            config.geometry()?.geometry,
            &serde_json::to_value(&config)?,
        )?;
        // The fixture covers selected roles. Admission still accounts for
        // every physical tensor and every required companion of those roles.
        logical.retain(|name, _| inventory.tensors.contains_key(name));
        SourcePrecision::admit(&config, metadata, inventory, logical)
    }

    #[test]
    fn nvfp4_routed_fixture_is_admitted_without_fp8_block_size() {
        let fixture = mixed_fixture(Layout::Nvfp4RoutedFp8Mtp).unwrap();
        let files = &fixture.files;
        let source = admit_mixed_fixture(files).unwrap();
        let format = |name: &str| source.operands[name].weight.format;
        assert_eq!(
            format("model.language_model.layers.0.linear_attn.in_proj_a.weight"),
            WeightFormat::Bf16
        );
        assert_eq!(
            format("model.language_model.layers.0.mlp.experts.0.gate_proj.weight"),
            WeightFormat::Nvfp4
        );
        assert!(matches!(
            format("mtp.layers.0.mlp.experts.0.gate_proj.weight"),
            WeightFormat::Fp8Block128x128(_)
        ));
        assert_eq!(
            format(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
            ),
            WeightFormat::Fp8PerTensor
        );
        assert_eq!(source.operands.len(), 4);
        for ranks in [2, 4] {
            let catalog =
                crate::catalog::Catalog::from_imported(files, source.clone(), ranks).unwrap();
            assert_eq!(catalog.tensors.len(), source.inventory.tensors.len());
        }
    }

    const HC_NORM: &str = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight";

    /// The MXFP8 layout's F32 hyper-connection norm is admitted as stored,
    /// beside its MXFP8 and NVFP4 operands.
    #[test]
    fn mxfp8_dense_fixture_admits_its_f32_norm() {
        let fixture = mixed_fixture(Layout::Mxfp8DenseNvfp4).unwrap();
        let source = admit_mixed_fixture(&fixture.files).unwrap();
        assert_eq!(
            source
                .f32_norms()
                .map(|span| span.tensor.as_str())
                .collect::<Vec<_>>(),
            [HC_NORM]
        );
        assert_eq!(source.inventory.tensors[HC_NORM].dtype, DType::F32);
        assert_eq!(
            source.operands["model.language_model.layers.0.linear_attn.in_proj_a.weight"]
                .weight
                .format,
            WeightFormat::Mxfp8
        );
        source.require_finite_f32_norms(&fixture.files).unwrap();
    }

    #[test]
    fn non_finite_f32_norms_are_refused() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let fixture = mixed_fixture_with(
                Layout::Mxfp8DenseNvfp4,
                &[(HC_NORM, 7, 0.25), (HC_NORM, 9, value)],
            )
            .unwrap();
            let source = admit_mixed_fixture(&fixture.files).unwrap();
            let error = source
                .require_finite_f32_norms(&fixture.files)
                .unwrap_err()
                .to_string();
            assert_eq!(
                error,
                format!("tensor {HC_NORM}: F32 norm weight element 9 is not finite")
            );
        }
        let finite = mixed_fixture_with(Layout::Mxfp8DenseNvfp4, &[(HC_NORM, 9, -1.0002)]).unwrap();
        admit_mixed_fixture(&finite.files)
            .unwrap()
            .require_finite_f32_norms(&finite.files)
            .unwrap();
    }

    /// Norms whose consumers have no F32 form keep the BF16 requirement.
    #[test]
    fn f32_norms_without_an_f32_consumer_are_refused() {
        let config = config();
        let metadata = ExportMetadata::default();
        for (name, admitted) in [
            ("mtp.pre_fc_norm_hidden.weight", false),
            (
                "model.language_model.layers.0.linear_attn.norm.weight",
                false,
            ),
            ("model.language_model.layers.1.ple.norm_conv.weight", true),
            (HC_NORM, true),
        ] {
            let inventory = TensorInventory {
                tensors: BTreeMap::from([(name.into(), tensor(name, DType::F32, &[128]))]),
                spans: BTreeMap::new(),
            };
            let logical = BTreeMap::from([(name.into(), ("BF16".into(), vec![128]))]);
            let result = SourcePrecision::admit(&config, metadata.clone(), inventory, logical);
            if admitted {
                assert_eq!(result.unwrap().f32_norms().count(), 1, "{name}");
            } else {
                let error = result.unwrap_err().to_string();
                assert_eq!(
                    error,
                    format!(
                        "tensor {name}: Qwen side calculation/state requires BF16 [128], received F32 [128]"
                    )
                );
            }
        }
    }

    /// An export manifest may name MTP experts MXFP8 while the quantization
    /// rules and tensor headers are weight-only NVFP4. The catalog reads its
    /// configuration documents and headers alone, so a disagreeing manifest
    /// selects no encoding (nor the composition an encoding binds).
    #[test]
    fn a_disagreeing_export_manifest_selects_no_encoding() {
        let fixture = mixed_fixture(Layout::Mxfp8DenseNvfp4).unwrap();
        std::fs::write(
            fixture.root.join("export-manifest.json"),
            br#"{"precision": {"mtp_experts": "mxfp8_block32"}}"#,
        )
        .unwrap();
        let files = crate::catalog::capture(&fixture.root).unwrap();
        assert!(!files.inputs.contains_key("export-manifest.json"));
        let config = Qwen38Config::from_bytes(&files.inputs["config.json"]).unwrap();
        let metadata = ExportMetadata::from_documents(
            Some(&config.quantization_config),
            Some(&files.inputs[HF_QUANT_CONFIG]),
        )
        .unwrap();
        // The MTP experts alone: the fixture's F32 norm awaits its binding.
        let mut inventory =
            TensorInventory::from_captured(&files, "model.safetensors.index.json").unwrap();
        inventory.tensors.retain(|name, _| name.starts_with("mtp."));
        let mut logical = crate::catalog::expected_tensors_with_config(
            config.geometry().unwrap().geometry,
            &serde_json::to_value(&config).unwrap(),
        )
        .unwrap();
        logical.retain(|name, _| inventory.tensors.contains_key(name));
        let source = SourcePrecision::admit(&config, metadata, inventory, logical).unwrap();
        let operand = &source.operands["mtp.layers.0.mlp.experts.0.gate_proj.weight"];
        assert_eq!(operand.weight.format, WeightFormat::Nvfp4);
        assert!(operand.activation.as_ref().is_some_and(|a| a.is_bf16()));
    }

    #[test]
    fn the_first_fault_in_logical_order_is_reported() {
        let weight = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
        let scale = weight.replace(".weight", ".weight_scale");
        let global = weight.replace(".weight", ".weight_scale_2");
        let input = weight.replace(".weight", ".input_scale");
        // One side tensor before the projection and one after it, in the
        // logical order the admission walks.
        let before = "model.aaa.state";
        let after = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
        let inventory = TensorInventory {
            tensors: [
                tensor(weight, DType::U8, &[640, 1280]),
                tensor(&scale, DType::Fp8E4M3, &[640, 160]),
                tensor(&global, DType::F32, &[]),
                tensor(before, DType::I64, &[3]),
                tensor(after, DType::I64, &[3]),
            ]
            .into_iter()
            .map(|t| (t.tensor.clone(), t))
            .collect(),
            spans: BTreeMap::new(),
        };
        let mut metadata = ExportMetadata::default();
        metadata.add_document("quantization", br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION",
          "quantized_layers":{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16}}}"#).unwrap();
        let logical = BTreeMap::from([
            (weight.into(), ("F8_E4M3".into(), vec![640, 2560])),
            (before.into(), ("I64".into(), vec![3])),
            (after.into(), ("I64".into(), vec![3])),
        ]);
        let refusal = |inventory: &TensorInventory| {
            SourcePrecision::admit(
                &config(),
                metadata.clone(),
                inventory.clone(),
                logical.clone(),
            )
            .unwrap_err()
            .to_string()
        };
        // The projection lacks its input scale (an import fault) and the
        // later side tensor has the wrong dtype: the projection's comes first.
        let mut both = inventory.clone();
        both.tensors.get_mut(after).unwrap().dtype = DType::F32;
        let first = refusal(&both);
        assert!(first.contains(&input), "{first}");
        // A side tensor before the projection fails first when it is wrong.
        both.tensors.get_mut(before).unwrap().dtype = DType::F32;
        let first = refusal(&both);
        assert!(first.contains(before), "{first}");
    }

    #[test]
    fn logical_roles_import_packed_shapes_and_preserve_side_contracts() {
        let weight = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
        let scale = weight.replace(".weight", ".weight_scale");
        let global = weight.replace(".weight", ".weight_scale_2");
        let input = weight.replace(".weight", ".input_scale");
        let state = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
        let inventory = TensorInventory {
            tensors: [
                tensor(weight, DType::U8, &[640, 1280]),
                tensor(&scale, DType::Fp8E4M3, &[640, 160]),
                tensor(&global, DType::F32, &[]),
                tensor(&input, DType::F32, &[1]),
                tensor(state, DType::I64, &[3]),
            ]
            .into_iter()
            .map(|t| (t.tensor.clone(), t))
            .collect(),
            spans: BTreeMap::new(),
        };
        let mut metadata = ExportMetadata::default();
        metadata.add_document("quantization", br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION",
          "quantized_layers":{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16}}}"#).unwrap();
        let logical = BTreeMap::from([
            (weight.into(), ("F8_E4M3".into(), vec![640, 2560])),
            (
                weight.replace(".weight", ".weight_scale_inv"),
                ("BF16".into(), vec![5, 20]),
            ),
            (state.into(), ("I64".into(), vec![3])),
        ]);
        let admitted = SourcePrecision::admit(
            &config(),
            metadata.clone(),
            inventory.clone(),
            logical.clone(),
        )
        .unwrap();
        assert_eq!(admitted.operands[weight].weight.logical_shape, [640, 2560]);
        assert_eq!(admitted.operands[weight].source_tensors.len(), 4);
        let mut missing = inventory.clone();
        missing.tensors.remove(&input);
        assert!(
            SourcePrecision::admit(&config(), metadata.clone(), missing, logical.clone())
                .unwrap_err()
                .to_string()
                .contains(&input)
        );
        let mut changed = inventory.clone();
        changed.tensors.get_mut(state).unwrap().dtype = DType::F32;
        assert!(
            SourcePrecision::admit(&config(), metadata.clone(), changed, logical.clone())
                .unwrap_err()
                .to_string()
                .contains("side calculation/state")
        );
        let mut extra = inventory;
        extra
            .tensors
            .insert("unknown".into(), tensor("unknown", DType::U8, &[1]));
        assert!(
            SourcePrecision::admit(&config(), metadata, extra, logical)
                .unwrap_err()
                .to_string()
                .contains("unaccounted")
        );
    }

    #[test]
    fn checkpoint_precision_inventory() {
        let Ok(root) = std::env::var("CHOPINFER_QWEN38_CHECKPOINT") else {
            eprintln!("SKIP: set CHOPINFER_QWEN38_CHECKPOINT for complete source interpretation");
            return;
        };
        let files = SourceFiles::capture(
            std::path::Path::new(&root),
            &[
                ("config.json", 1024 * 1024),
                ("model.safetensors.index.json", 64 * 1024 * 1024),
            ],
            &[("hf_quant_config.json", 16 * 1024 * 1024)],
            "model.safetensors.index.json",
        )
        .unwrap();
        let catalog = crate::catalog::Catalog::from_captured(&files, 4).unwrap();
        let source = catalog.precision.as_ref().unwrap();
        assert_eq!(
            source.operands.len(),
            expected_tensors()
                .iter()
                .filter(|(name, (_, shape))| name.ends_with(".weight") && shape.len() == 2)
                .count()
        );
        let mut formats = BTreeMap::new();
        for operand in source.operands.values() {
            *formats
                .entry(format!("{:?}", operand.weight.format))
                .or_insert(0_usize) += 1;
        }
        eprintln!("interpreted logical operands: {formats:?}");
        let has_quantized_vision = source.operands.iter().any(|(name, operand)| {
            name.starts_with("model.visual.") && operand.weight.format != WeightFormat::Bf16
        });
        assert_eq!(source.require_bf16_vision().is_err(), has_quantized_vision);
        for (name, operand) in &source.operands {
            assert_eq!(
                catalog.tensors[name].bytes,
                operand.weight.codes.byte_length
            );
            for companion in &operand.source_tensors {
                assert!(catalog.tensors.contains_key(companion));
            }
        }
    }

    /// No operand a checkpoint can import, whatever its format, makes the
    /// declaration refuse TP2 or MTP: the kernels' compositions decide
    /// what a quantized tensor serves, not tensor-prefix scopes.
    #[test]
    fn source_formats_select_no_declared_topology_refusal() {
        use kernel_core::contract::ScaleMode;
        use kernel_core::precision::ExecutionPhase;
        use model_qwen38_contract::contract::Shape;
        use model_qwen38_contract::projections::{ports, support::Support};
        let formats = [
            WeightFormat::Bf16,
            WeightFormat::Fp8PerTensor,
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            WeightFormat::Mxfp4,
            WeightFormat::Mxfp8,
            WeightFormat::Nvfp4,
        ];
        for shape in [Shape::TP2, Shape::TP4] {
            let bindings = ports::get(shape).unwrap().bindings().unwrap();
            // Members of one projection share their outcome.
            let mut checked = BTreeMap::new();
            for (name, (_, logical)) in expected_tensors() {
                if !name.ends_with(".weight") || logical.len() != 2 {
                    continue;
                }
                for format in formats {
                    let key = (
                        bindings.get(&name).map(|b| b.projection.clone()),
                        format!("{format:?}"),
                    );
                    checked.entry(key).or_insert_with(|| {
                        let support = Support::sources(shape, [(&name, format)]).unwrap();
                        for projection in ["mtp.q", "linear.qkv"] {
                            assert!(
                                support.admit(projection, ExecutionPhase::Decode).is_ok(),
                                "{name} {format:?} refuses {projection} at TP{}",
                                shape.ranks()
                            );
                        }
                    });
                }
            }
            assert!(checked.len() > 100);
        }
    }
}
