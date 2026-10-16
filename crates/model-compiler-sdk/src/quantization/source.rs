//! Closed safetensors inventory and bounded reads shared by family adapters.
use crate::{
    CompilerError,
    source::{SourceFiles, TensorSpan as FileSpan},
};
use kernel_core::contract::DType;
use kernel_core::precision::TensorSpan;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInventory {
    pub tensors: BTreeMap<String, TensorSpan>,
    pub spans: BTreeMap<String, FileSpan>,
}

fn error(message: impl std::fmt::Display) -> CompilerError {
    CompilerError::validation(format!("checkpoint tensors: {message}"))
}

/// The precision contracts' dtype of a safetensors dtype the importers read.
pub fn source_dtype(dtype: runtime_weights::DType) -> Option<DType> {
    use runtime_weights::DType as Stored;
    Some(match dtype {
        Stored::Bf16 => DType::Bf16,
        Stored::F32 => DType::F32,
        Stored::I32 => DType::I32,
        Stored::I64 => DType::I64,
        Stored::U8 => DType::U8,
        Stored::F8E4M3 => DType::Fp8E4M3,
        Stored::F8E8M0 => DType::Ue8M0,
        _ => return None,
    })
}

impl TensorInventory {
    pub fn from_captured(files: &SourceFiles, index: &str) -> Result<Self, CompilerError> {
        #[derive(Deserialize)]
        struct Index {
            weight_map: BTreeMap<String, String>,
        }
        let bytes = files
            .inputs
            .get(index)
            .ok_or_else(|| error("missing captured index"))?;
        let mut index: Index = serde_json::from_slice(bytes).map_err(error)?;
        let mut tensors = BTreeMap::new();
        let mut spans = BTreeMap::new();
        for (file, header) in &files.headers {
            for (name, info) in header.tensors() {
                if index.weight_map.remove(name).as_ref() != Some(file)
                    || tensors.contains_key(name)
                {
                    return Err(error(format!(
                        "tensor {name}: duplicate or mismatched shard assignment"
                    )));
                }
                let dtype = source_dtype(info.dtype()).ok_or_else(|| {
                    error(format!(
                        "tensor {name}: unsupported source dtype {}",
                        info.dtype().safetensors_name()
                    ))
                })?;
                let tensor = TensorSpan {
                    tensor: name.clone(),
                    dtype,
                    shape: info.shape().to_vec(),
                    offset: 0,
                    byte_length: info.byte_len(),
                };
                tensor.validate().map_err(error)?;
                tensors.insert(name.clone(), tensor);
                spans.insert(
                    name.clone(),
                    FileSpan {
                        file: file.clone(),
                        offset: header
                            .data_offset()
                            .checked_add(info.data_offsets().0)
                            .ok_or_else(|| error("source offset overflow"))?,
                        bytes: info.byte_len(),
                    },
                );
            }
        }
        if tensors.is_empty() || !index.weight_map.is_empty() {
            return Err(error("index references missing tensors"));
        }
        files.verify().map_err(error)?;
        Ok(Self { tensors, spans })
    }

    /// The family accounts for architecture side tensors and every imported
    /// companion, including calibration retained only as source provenance.
    pub fn require_accounted(&self, names: &BTreeSet<String>) -> Result<(), CompilerError> {
        if let Some(name) = self.tensors.keys().find(|n| !names.contains(*n)) {
            return Err(error(format!(
                "tensor {name}: unaccounted source role or execution companion"
            )));
        }
        if let Some(name) = names.iter().find(|n| !self.tensors.contains_key(*n)) {
            return Err(error(format!(
                "tensor {name}: missing required source role"
            )));
        }
        Ok(())
    }

    pub fn read(
        &self,
        files: &SourceFiles,
        requested: &TensorSpan,
    ) -> Result<Vec<u8>, CompilerError> {
        self.read_inner(files, requested, false)
    }

    /// Read a bounded view of a captured tensor. The dtype stays fixed (apart
    /// from the declared UE8M0 byte alias), and byte offsets are tensor-relative.
    pub fn read_slice(
        &self,
        files: &SourceFiles,
        requested: &TensorSpan,
    ) -> Result<Vec<u8>, CompilerError> {
        self.read_inner(files, requested, true)
    }

    fn read_inner(
        &self,
        files: &SourceFiles,
        requested: &TensorSpan,
        sliced: bool,
    ) -> Result<Vec<u8>, CompilerError> {
        requested.validate().map_err(error)?;
        let original = self.tensors.get(&requested.tensor).ok_or_else(|| {
            error(format!(
                "tensor {}: outside the captured inventory",
                requested.tensor
            ))
        })?;
        let alias = original.dtype == DType::U8 && requested.dtype == DType::Ue8M0;
        let relative = requested.offset.checked_sub(original.offset);
        let width = kernel_core::precision::dtype_bits(original.dtype) / 8;
        if (original.dtype != requested.dtype && !alias)
            || (!sliced
                && (original.shape != requested.shape
                    || original.offset != requested.offset
                    || original.byte_length != requested.byte_length))
            || relative
                .and_then(|offset| offset.checked_add(requested.byte_length))
                .is_none_or(|end| end > original.byte_length)
            || width == 0
            || !requested.offset.is_multiple_of(width)
            || requested.byte_length > 2 * 1024 * 1024 * 1024
        {
            return Err(error(format!(
                "tensor {}: requested representation or bounds differ from the captured source",
                requested.tensor
            )));
        }
        let span = self
            .spans
            .get(&requested.tensor)
            .ok_or_else(|| error("missing file span"))?;
        if span.bytes != original.byte_length {
            return Err(error("source span length differs"));
        }
        let file = files.file(&span.file).map_err(error)?;
        let header = files
            .headers
            .get(&span.file)
            .ok_or_else(|| error("missing captured shard"))?;
        let tensor = header
            .tensors()
            .get(&requested.tensor)
            .ok_or_else(|| error("missing captured tensor"))?;
        if header.data_offset().checked_add(tensor.data_offsets().0) != Some(span.offset) {
            return Err(error("source span offset differs"));
        }
        let offset = span
            .offset
            .checked_add(relative.unwrap())
            .ok_or_else(|| error("source slice offset overflow"))?;
        let mut bytes = vec![0; requested.byte_length as usize];
        runtime_weights::read_exact_at(file, &mut bytes, offset).map_err(error)?;
        file.verify().map_err(error)?;
        Ok(bytes)
    }
}
