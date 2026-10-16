//! One admitted source version for preparation and explicit oracle commands.
use crate::{Result, catalog::Catalog};
use model_compiler_sdk::source::{SourceFiles, SourceIdentity, TensorSpan};
use std::collections::BTreeMap;
use std::path::Path;

pub struct CheckpointSource {
    pub files: SourceFiles,
    pub catalog: Catalog,
    pub spans: BTreeMap<String, TensorSpan>,
}

impl CheckpointSource {
    pub fn open(root: &Path, ranks: usize) -> Result<Self> {
        let files = Self::capture(root)?;
        let catalog = Catalog::from_imported(&files, Self::import(&files)?, ranks)?;
        files.verify()?;
        let spans = catalog
            .tensors
            .iter()
            .map(|(name, t)| {
                (
                    name.clone(),
                    TensorSpan {
                        file: t.file.clone(),
                        offset: t.file_start,
                        bytes: t.bytes,
                    },
                )
            })
            .collect();
        Ok(Self {
            files,
            catalog,
            spans,
        })
    }

    /// The precision of the checkpoint whose `files` were captured, admitted
    /// from its headers and checked against the payload admission needs:
    /// every F32 norm weight must be finite. Preparation, package emission
    /// and the serving helper import through this; the header-only weight
    /// plan imports without reading tensor bytes.
    pub fn import(files: &SourceFiles) -> Result<crate::precision_source::SourcePrecision> {
        let precision = crate::precision_source::SourcePrecision::from_captured(files)?;
        precision.require_finite_f32_norms(files)?;
        Ok(precision)
    }

    /// The files a source reads: configuration, tokenizer, index and the
    /// safetensors headers, captured once whatever the rank count.
    pub fn capture(root: &Path) -> Result<SourceFiles> {
        SourceFiles::capture(
            root,
            &[
                ("config.json", 1024 * 1024),
                ("tokenizer.json", 32 * 1024 * 1024),
                ("tokenizer_config.json", 64 * 1024),
                ("model.safetensors.index.json", 64 * 1024 * 1024),
            ],
            &[
                ("chat_template.jinja", 64 * 1024),
                ("generation_config.json", 64 * 1024),
                ("hf_quant_config.json", 16 * 1024 * 1024),
            ],
            "model.safetensors.index.json",
        )
    }

    pub fn measure(&self, packed: &BTreeMap<String, String>) -> Result<(SourceIdentity, u64)> {
        self.files.measure(&self.spans, packed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sparse checkpoint of the served FP8 geometry in one shard, with
    /// the listed tensors stored as F32 and the listed F32 elements
    /// `(tensor, element, value)` written, removed when dropped.
    struct Sparse(std::path::PathBuf);

    impl Drop for Sparse {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sparse_checkpoint(f32: &[&str], values: &[(&str, u64, f32)]) -> Sparse {
        use std::io::Write;
        use std::os::unix::fs::FileExt;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let config: &[u8] = include_bytes!("../fixtures/config.json");
        let value: serde_json::Value = serde_json::from_slice(config).unwrap();
        let geometry = crate::config::Qwen38Config::from_bytes(config)
            .unwrap()
            .geometry()
            .unwrap()
            .geometry;
        let tensors = crate::catalog::expected_tensors_with_config(geometry, &value).unwrap();
        let root = std::env::temp_dir().join(format!(
            "qwen38-sparse-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let sparse = Sparse(root.clone());
        std::fs::write(root.join("config.json"), config).unwrap();
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            std::fs::write(root.join(name), b"{}").unwrap();
        }
        let (mut header, mut weight_map, mut starts) =
            (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        let mut offset = 0_u64;
        for (name, (dtype, shape)) in tensors {
            let dtype = if f32.contains(&name.as_str()) {
                assert_eq!(dtype, "BF16", "{name}");
                "F32".to_owned()
            } else {
                dtype
            };
            let width = match dtype.as_str() {
                "BF16" => 2,
                "F32" => 4,
                "I64" => 8,
                _ => 1,
            };
            let end = offset + width * shape.iter().product::<u64>();
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
            );
            starts.insert(name.clone(), offset);
            weight_map.insert(name, "model.safetensors");
            offset = end;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut file = std::fs::File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        let data = 8 + header.len() as u64;
        file.set_len(data + offset).unwrap();
        for &(tensor, element, value) in values {
            file.write_all_at(&value.to_le_bytes(), data + starts[tensor] + 4 * element)
                .unwrap();
        }
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
        sparse
    }

    const NORM: &str = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight";

    /// Every production path that imports a checkpoint for preparation
    /// refuses a non-finite F32 norm element by tensor and index: opening a
    /// source, package emission and the serving helper's emission.
    #[test]
    fn preparation_paths_refuse_non_finite_f32_norms() {
        let module = crate::test_native::module();
        let native = model_compiler_sdk::native_contract::NativeModule::inspect(&module).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let checkpoint = sparse_checkpoint(&[NORM], &[(NORM, 3, 0.5), (NORM, 11, value)]);
            let root = &checkpoint.0;
            let expected = format!("tensor {NORM}: F32 norm weight element 11 is not finite");
            let output = root.join("package");
            let package = crate::preparation::emit_direct_package(
                root,
                &native,
                model_compiler_sdk::quantization::ActivationPolicy::Auto,
                &[],
                &"0".repeat(64),
                crate::package::Drafter::None,
                None,
                Some(model_qwen38_contract::contract::Shape::TP4),
                &output,
            );
            assert_eq!(
                package.err().map(|e| e.to_string()),
                Some(expected.clone()),
                "{value}"
            );
            let request = runtime_bundle::preparation::PrepareDirectRequest {
                path: root.clone(),
                cache_root: std::path::PathBuf::new(),
                tensor_parallel_size: 4,
                activation_policy: None,
                modalities: Default::default(),
            };
            let helper = crate::preparation::emit_direct_with(
                root,
                &request,
                &"0".repeat(64),
                &output,
                &module,
                None,
            );
            assert_eq!(
                helper.err().map(|e| e.to_string()),
                Some(expected.clone()),
                "{value}"
            );
            let opened = CheckpointSource::open(root, 4);
            assert_eq!(
                opened.err().map(|e| e.to_string()),
                Some(expected),
                "{value}"
            );
            assert!(!output.exists());
        }
    }

    /// A checkpoint with the legacy FP8 projection layout and admitted F32
    /// norms places them as F32 row views under prepared precision, which
    /// prepared view validation accepts and binds; unprepared planning
    /// still refuses them.
    #[test]
    fn prepared_legacy_layouts_place_admitted_f32_norms() {
        use model_qwen38_contract::contract::Shape;
        let query = "model.language_model.layers.1.ple.norm_query.weight";
        let mlp = "model.language_model.layers.4.mlp_hyper_connection.hc_norm.weight";
        let checkpoint = sparse_checkpoint(&[query, mlp], &[(mlp, 2, -1.0002)]);
        let source = CheckpointSource::open(&checkpoint.0, 4).unwrap();
        let precision = source.catalog.precision.as_ref().unwrap();
        assert!(precision.legacy_text_layout());
        assert_eq!(precision.f32_norms().count(), 2);
        let error = crate::placement::ImagePlan::new(&source.catalog, None)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("F32 norm weights require prepared operation bindings"),
            "{error}"
        );
        let capabilities = model_qwen38_contract::precision::capabilities().unwrap();
        let inventory = kernel_core::precision::capability::CapabilityInventory::new(
            capabilities.clone(),
            capabilities.entries(),
            "0".repeat(64),
        )
        .unwrap();
        let document = crate::projection_program::SourceProgram::new(precision, Shape::TP4)
            .unwrap()
            .resolve(
                model_compiler_sdk::quantization::ActivationPolicy::Auto,
                &inventory,
            )
            .unwrap();
        model_qwen38_contract::prepared::bind(&document, Shape::TP4, &inventory).unwrap();
        let plan =
            crate::placement::ImagePlan::with_precision(&source.catalog, None, &document).unwrap();
        for index in &plan.ranks {
            let mut index = index.clone();
            index.sha256 = "0".repeat(64);
            let parsed = model_qwen38_contract::views::parse(
                &serde_json::to_vec(&index).unwrap(),
                index.rank as u32,
            )
            .unwrap();
            model_qwen38_contract::prepared::validate_views(&parsed, Shape::TP4, &document)
                .unwrap();
            assert!(model_qwen38_contract::views::validate(&parsed, Shape::TP4).is_err());
            let norms = model_qwen38_contract::norms::NormWeights::from_index(&parsed);
            for name in [query, mlp] {
                assert_eq!(
                    norms.dtype(name),
                    kernel_core::contract::DType::F32,
                    "{name}"
                );
            }
            assert_eq!(norms.dtype(NORM), kernel_core::contract::DType::Bf16);
        }
    }

    /// The local-inference-lab mixed export opens at TP4 with its F32 norm
    /// weights admitted as stored and finite: every main layer's two
    /// hyper-connection norms, the final mixer's and the three PLE norms,
    /// while its MTP norms stay BF16.
    #[test]
    #[ignore = "needs the local-inference-lab mixed export: run with --ignored and \
                CHOPINFER_QWEN38_LOCAL_LAB_CHECKPOINT set"]
    fn local_lab_checkpoint_admits_finite_f32_norms() {
        let root = std::env::var("CHOPINFER_QWEN38_LOCAL_LAB_CHECKPOINT")
            .expect("set CHOPINFER_QWEN38_LOCAL_LAB_CHECKPOINT to the local-lab mixed export");
        let source = CheckpointSource::open(Path::new(&root), 4).unwrap();
        let precision = source.catalog.precision.as_ref().unwrap();
        let norms = precision
            .f32_norms()
            .map(|span| span.tensor.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let model = "model.language_model";
        let layers = source.catalog.geometry.layers();
        let mut expected = std::collections::BTreeSet::new();
        for layer in 0..layers {
            for sublayer in ["attn", "mlp"] {
                expected.insert(format!(
                    "{model}.layers.{layer}.{sublayer}_hyper_connection.hc_norm.weight"
                ));
            }
        }
        expected.insert(format!("{model}.hyper_connection_mixer.hc_norm.weight"));
        for norm in ["norm_key", "norm_conv", "norm_query"] {
            expected.insert(format!("{model}.layers.1.ple.{norm}.weight"));
        }
        assert_eq!(norms, expected);
        assert_eq!(norms.len(), 100);
        for (name, record) in &source.catalog.tensors {
            if name.starts_with("mtp.") && name.contains("norm") {
                assert_eq!(record.dtype, runtime_weights::DType::Bf16, "{name}");
            }
        }
    }

    /// Prepared at TP4 with a built native module, local-lab's rank images
    /// store its 100 F32 norms as F32 row views, which prepared view
    /// admission accepts and from which each consumer binds its F32 form,
    /// and every rank index passes complete view validation.
    #[test]
    #[ignore = "needs the local-inference-lab mixed export and a native module: run with \
                --ignored and CHOPINFER_QWEN38_LOCAL_LAB_CHECKPOINT and \
                CHOPINFER_QWEN38_NATIVE_MODULE set"]
    fn local_lab_prepared_rank_views_bind_its_f32_norms() {
        use model_qwen38_contract::contract::Shape;
        let root = std::env::var("CHOPINFER_QWEN38_LOCAL_LAB_CHECKPOINT")
            .expect("set CHOPINFER_QWEN38_LOCAL_LAB_CHECKPOINT to the local-lab mixed export");
        let native = std::env::var("CHOPINFER_QWEN38_NATIVE_MODULE")
            .expect("set CHOPINFER_QWEN38_NATIVE_MODULE to a built libbundle_qwen38.so");
        let module =
            model_compiler_sdk::native_contract::NativeModule::inspect(Path::new(&native)).unwrap();
        let (source, precision) = crate::preparation::direct_precision(
            Path::new(&root),
            &module,
            Shape::TP4,
            model_compiler_sdk::quantization::ActivationPolicy::Auto,
            &[],
        )
        .unwrap();
        let plan =
            crate::placement::ImagePlan::with_precision(&source.catalog, None, &precision).unwrap();
        let expected = source
            .catalog
            .precision
            .as_ref()
            .unwrap()
            .f32_norms()
            .map(|span| span.tensor.clone())
            .collect::<Vec<_>>();
        assert_eq!(expected.len(), 100);
        for index in &plan.ranks {
            let mut index = index.clone();
            index.sha256 = "0".repeat(64);
            let bytes = serde_json::to_vec(&index).unwrap();
            let parsed = model_qwen38_contract::views::parse(&bytes, index.rank as u32).unwrap();
            assert!(model_qwen38_contract::views::validate(&parsed, Shape::TP4).is_err());
            let mut required =
                model_qwen38_contract::prepared::required(&precision, parsed.rank, Shape::TP4)
                    .unwrap();
            model_qwen38_contract::norms::admit(&parsed, &mut required);
            let mut admitted = 0;
            for view in required
                .iter()
                .filter(|view| model_qwen38_contract::norms::f32_consumer(&view.name))
            {
                let stored = &parsed.views[&view.name];
                assert_eq!(
                    (
                        stored.dtype.as_str(),
                        stored.layout_id.as_str(),
                        stored.bytes
                    ),
                    (view.dtype, view.layout_id, view.bytes),
                    "{}",
                    view.name
                );
                admitted += usize::from(view.dtype == "F32");
            }
            assert_eq!(admitted, expected.len());
            model_qwen38_contract::prepared::validate_views(&parsed, Shape::TP4, &precision)
                .unwrap_or_else(|error| panic!("rank {}: {error}", parsed.rank));
            let norms = model_qwen38_contract::norms::NormWeights::from_index(&parsed);
            for name in &expected {
                assert_eq!(
                    norms.dtype(name),
                    kernel_core::contract::DType::F32,
                    "{name}"
                );
                let view = &parsed.views[name];
                assert_eq!(
                    (view.dtype.as_str(), view.bytes),
                    ("F32", 4 * 10240),
                    "{name}"
                );
            }
        }
    }
}
