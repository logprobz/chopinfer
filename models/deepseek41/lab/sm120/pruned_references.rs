//! Original mixed catalogs retained for comparisons after package pruning.
use bundle_deepseek41::sm120::{ceil_split, mxfp8_fold, prefill_matrix};
use kernel_build::CubinSpec;
use kernels_sm120::{gemm::mxfp8_prefill_matrix, row::block_quantize};
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::GEOMETRIES};

pub fn ceil_split() -> Result<Vec<CubinSpec>, String> {
    let mut specs = Vec::new();
    for geometry in GEOMETRIES {
        let mut entries = Vec::<block_quantize::Entry>::new();
        for &ranks in SUPPORTED_RANKS {
            for entry in ceil_split::candidates(geometry, ranks)? {
                if !entries.iter().any(|old| old.symbol == entry.symbol) {
                    entries.push(entry);
                }
            }
        }
        if !entries.is_empty() {
            specs.push(
                block_quantize::cubin_spec(&ceil_split::cubin_name(geometry), &entries)
                    .map_err(|e| e.to_string())?,
            );
        }
    }
    Ok(specs)
}

pub fn prefill_matrix() -> Result<Vec<CubinSpec>, String> {
    let entries = prefill_matrix::catalog()?;
    Ok(vec![mxfp8_fold::packed(
        mxfp8_prefill_matrix::cubin_spec(prefill_matrix::CUBIN, &entries),
        entries.iter().map(|entry| entry.packed_spec()),
    )?])
}
