use std::collections::BTreeMap;
use std::path::Path;

use runtime_weights::{DType, read_safetensors_header};

use crate::MimoResult;
use crate::catalog::{TensorRecord, open_source};

pub const TOKENIZER_FILE: &str = "audio_tokenizer/model.safetensors";

/// The inference-only encoder catalog. EMA buffers and the audio decoder are excluded.
#[derive(Clone, Debug, Default)]
pub struct EncoderCatalog {
    pub sources: BTreeMap<String, model_compiler_sdk::source_file::SourceFile>,
    pub tensors: BTreeMap<String, TensorRecord>,
}

type Expected = BTreeMap<String, (DType, Vec<u64>, &'static str)>;

fn expected(geometry: &model_mimo26_contract::geometry::Geometry) -> Expected {
    model_mimo26_contract::encoders::expected_for(geometry)
        .into_iter()
        .map(|(name, (dtype, shape, file))| {
            (
                name,
                (
                    if dtype == "F32" {
                        DType::F32
                    } else {
                        DType::Bf16
                    },
                    shape,
                    file,
                ),
            )
        })
        .collect()
}

impl EncoderCatalog {
    pub fn load(root: &Path) -> MimoResult<Self> {
        let geometry = crate::config::MiMoConfig::load(root)?.geometry()?;
        let mut expected = expected(geometry);
        let mut tensors = BTreeMap::new();
        let mut sources = BTreeMap::new();
        let (indexed_sources, indexed_tensors) = crate::catalog::load_indexed(root)?;
        for (name, tensor) in indexed_tensors {
            if !(name.starts_with("visual.")
                || name.starts_with("audio_encoder.")
                || name.starts_with("speech_embeddings."))
            {
                continue;
            }
            let Some((dtype, shape, _)) = expected.remove(&name) else {
                return Err(format!("unexpected encoder tensor {name}").into());
            };
            if dtype != tensor.dtype || shape != tensor.shape {
                return Err(format!("encoder tensor {name} has wrong dtype or shape").into());
            }
            tensors.insert(name, tensor);
        }
        sources.extend(indexed_sources);
        for filename in [TOKENIZER_FILE] {
            let path = root.join(filename);
            let file = open_source(&path)?;
            let header = read_safetensors_header(&file)?;
            for (name, info) in header.tensors() {
                let belongs = name.starts_with("encoder.")
                    && !name.contains("._codebook.cluster_size")
                    && !name.contains("._codebook.embed_avg")
                    && !name.contains("._codebook.inited");
                if !belongs {
                    continue;
                }
                let Some((dtype, shape, source)) = expected.remove(name) else {
                    return Err(format!("unexpected encoder tensor {name}").into());
                };
                if dtype != info.dtype() || shape != info.shape() || filename != source {
                    return Err(
                        format!("encoder tensor {name} has wrong dtype, shape or source").into(),
                    );
                }
                let (start, _) = info.data_offsets();
                tensors.insert(
                    name.clone(),
                    TensorRecord {
                        name: name.clone(),
                        dtype,
                        shape,
                        file: filename.into(),
                        file_start: header
                            .data_offset()
                            .checked_add(start)
                            .ok_or("offset overflow")?,
                        bytes: info.byte_len(),
                    },
                );
            }
            file.verify()?;
            sources.insert(filename.to_owned(), file);
        }
        if let Some(name) = expected.keys().next() {
            return Err(format!("missing encoder tensor {name}").into());
        }
        Ok(Self { tensors, sources })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_name_count() {
        // 364 vision, 75 local audio, 20 embeddings, 389 tokenizer encoder.
        assert_eq!(
            expected(&model_mimo26_contract::geometry::V26_FLASH).len(),
            848
        );
    }

    #[test]
    fn real_checkpoint() {
        let Ok(root) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
            return;
        };
        let catalog = EncoderCatalog::load(Path::new(&root)).unwrap();
        assert_eq!(catalog.tensors.len(), 848);
    }
}
