//! Named tensors read from bundle dumps as f64.

use runtime_weights::DType;

use super::weights::bf16;
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// BF16, F32, F64, I64 or U8 safetensors data as f64.
pub(crate) fn decode(name: &str, dtype: DType, raw: &[u8]) -> Result<Vec<f64>> {
    Ok(match dtype {
        DType::Bf16 => raw
            .chunks_exact(2)
            .map(|b| bf16(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        DType::F32 => raw
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes(b.try_into().expect("four bytes"))))
            .collect(),
        DType::F64 => raw
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().expect("eight bytes")))
            .collect(),
        DType::I64 => raw
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().expect("eight bytes")) as f64)
            .collect(),
        DType::U8 | DType::Bool => raw.iter().map(|&b| f64::from(b)).collect(),
        other => return Err(format!("{name}: unsupported dump dtype {other:?}").into()),
    })
}
