//! The vision encoder's SM120 kernels: the catalog of entries its
//! declaration (`crate::encoder::declarations`) runs and the two cubins
//! that build them, which `build.rs` compiles.

use std::sync::OnceLock;

use encoder_spec::Error;
use kernel_build::CubinSpec;
use kernels_sm120::encoder::{self, Entry};

use crate::encoder::{FAMILY, declarations};

/// The encoder kernel catalog: the entries the declaration runs. Built once.
pub fn kernels() -> Result<&'static [Entry], Error> {
    static KERNELS: OnceLock<Result<Vec<Entry>, Error>> = OnceLock::new();
    KERNELS
        .get_or_init(|| encoder::model_catalog(declarations()?, &encoder::Shared))
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// Whether `name` is one of the encoder's cubins.
pub fn is_cubin(name: &str) -> bool {
    name == encoder::ROWS_GEMM_CUBIN || name == encoder::ATTENTION_CUBIN
}

/// The encoder's two cubins (rows and GEMMs, attention), in the `FAMILY`
/// lock family.
pub fn cubin_specs() -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let entries = kernels().map_err(|error| error.0)?;
    let specs = encoder::cubin_specs(entries).map_err(|refusal| refusal.0.to_owned())?;
    Ok(specs.into_iter().map(|spec| (FAMILY, spec)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalog_holds_the_new_row_entries() {
        let entries = kernels().unwrap();
        for stem in [
            "chop_enc_pad_zero_columns_w1176_to1216_",
            "chop_enc_attention_varlen_qk_rms_pairs_qk_norm_q16_kv16_d64_",
            "chop_enc_mlp_swiglu_clamped_swiglu_h4096_",
            "chop_enc_mlp_swiglu_clamped_swiglu_h10240_",
            "chop_enc_projector_norm_gelu_mlp_gelu_w4096_",
            "chop_enc_linear_pairs_n1024_k1216_",
        ] {
            assert!(
                entries.iter().any(|entry| entry.symbol.starts_with(stem)),
                "{stem}"
            );
        }
        let specs = cubin_specs().unwrap();
        assert_eq!(specs.len(), 2);
        assert!(
            specs
                .iter()
                .all(|(family, spec)| *family == FAMILY && is_cubin(&spec.name))
        );
    }
}
