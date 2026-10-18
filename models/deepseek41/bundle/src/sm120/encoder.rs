//! The vision encoder's SM120 kernels: the catalog of shared entries its
//! declaration (`model_deepseek41_contract::encoder`) runs, the two cubins
//! the build compiles from it in the encoder lock family, and their
//! registration on `Sm120`. The lab reads this file for its conformance
//! cases.

use std::sync::OnceLock;

use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::encoder::{self, Entry, Shared};
use model_deepseek41_contract::encoder::{FAMILY, declarations};

/// The encoder kernel catalog: the entries the declaration runs. Built once.
pub fn kernels() -> Result<&'static [Entry], String> {
    static KERNELS: OnceLock<Result<Vec<Entry>, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            let declarations = declarations().map_err(|error| error.0.clone())?;
            encoder::model_catalog(declarations, &Shared).map_err(|error| error.0)
        })
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// Whether `name` is one of the encoder's cubins.
pub fn is_cubin(name: &str) -> bool {
    name == encoder::ROWS_GEMM_CUBIN || name == encoder::ATTENTION_CUBIN
}

/// The encoder's two cubins (rows and GEMMs, attention) in its lock family.
pub fn cubins() -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let specs = encoder::cubin_specs(kernels()?).map_err(|refusal| refusal.0.to_owned())?;
    Ok(specs.into_iter().map(|spec| (FAMILY, spec)).collect())
}

/// `sm120` serving the encoder's catalog entries.
pub fn register(sm120: Sm120) -> Result<Sm120, String> {
    Ok(sm120.with_encoders(kernels()?.iter().cloned()))
}
