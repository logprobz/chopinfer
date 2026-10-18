//! Exact native index-key products in the K.2 BF16 comparison layout.
//! A separate family leaves every existing compressor cubin unchanged.

use super::compressor::{CompressorKernels, WriteQuery};
use kernel_build::{BuildError, CubinSpec, EntryDeclaration, SelectedCubin};
use kernel_core::contract::{NumericalContract, Refusal};
use kernel_core::pooled_indexer::IndexerKeyFormat;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gated_pooling::key_write::{self, Entry};
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "compressor_bf16";

pub fn cubin_name(geometry: &Geometry) -> String {
    format!("{}_compressor_bf16_keys_v1", geometry.kernel_tag)
}

pub fn entries(geometry: &Geometry) -> Result<Vec<Entry>, Refusal> {
    CompressorKernels::serving(geometry)?
        .writes
        .into_iter()
        .filter_map(|(_, query)| match query {
            WriteQuery::Index(q) => Some(q),
            _ => None,
        })
        .map(|query| {
            Entry::index_with_storage(
                &query,
                IndexerKeyFormat::Bf16,
                &format!(
                    "chop_{}_index_key_bf16_r{}_v1",
                    geometry.kernel_tag, query.compression_ratio
                ),
            )
        })
        .collect()
}

pub fn register(backend: Sm120, geometry: &Geometry) -> Result<Sm120, String> {
    Ok(backend.with_fp4_key_writes(entries(geometry).map_err(|e| e.to_string())?))
}

pub fn selected(geometry: &Geometry) -> Result<SelectedCubin, BuildError> {
    let entries = entries(geometry).map_err(|e| BuildError(e.to_string()))?;
    let spec = key_write::cubin_spec(cubin_name(geometry), &entries)
        .map_err(|e| BuildError(e.to_string()))?;
    let declarations = CompressorKernels::serving(geometry)
        .map_err(|e| BuildError(e.to_string()))?
        .writes
        .into_iter()
        .filter_map(|(_, query)| match query {
            WriteQuery::Index(q) => Some(q),
            _ => None,
        })
        .zip(&entries)
        .map(|(query, entry)| {
            Ok(EntryDeclaration {
                symbol: entry.symbol().into(),
                operation: "Fp4KeyWrite".into(),
                contract: query
                    .numerical_contract_id()
                    .map_err(|e| BuildError(e.to_string()))?
                    .to_string(),
                implementation: "sm120.fp4-key-write.f32-tail-rope-status-or-zero.v2".into(),
                requirements: [
                    ("query".into(), format!("{query:?}")),
                    (
                        "storage".into(),
                        "exact native E2M1 times UE8M0 BF16 products; no scale plane".into(),
                    ),
                    (
                        "addressing".into(),
                        "K.2 index stream offset with 64-bit physical page stride".into(),
                    ),
                ]
                .into(),
                correctness_case: Some("fp4_key_write_bf16".into()),
            })
        })
        .collect::<Result<Vec<_>, BuildError>>()?;
    SelectedCubin::new(spec, declarations)
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, Refusal> {
    geometries
        .iter()
        .map(|geometry| {
            Ok((
                FAMILY,
                key_write::cubin_spec(cubin_name(geometry), &entries(geometry)?)?,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::GEOMETRIES;

    #[test]
    fn comparison_registration_has_typed_entries_and_a_separate_family() {
        for geometry in GEOMETRIES {
            selected(geometry).unwrap();
            let native = CompressorKernels::serving(geometry)
                .unwrap()
                .cubins()
                .unwrap();
            let comparison = cubins(std::slice::from_ref(geometry)).unwrap();
            assert_eq!(comparison.len(), 1);
            assert_eq!(comparison[0].0, FAMILY);
            assert!(native.iter().all(|c| c.name != comparison[0].1.name));
            let entries = entries(geometry).unwrap();
            let native_keys = CompressorKernels::serving(geometry)
                .unwrap()
                .writes
                .into_iter()
                .filter(|(_, query)| matches!(query, WriteQuery::Index(_)))
                .count();
            assert_eq!(entries.len(), native_keys);
            for entry in entries {
                assert!(!entry.packed());
                assert_eq!(entry.width(), geometry.shape.indexer.dim);
                assert_eq!(
                    entry.page_size() * entry.ratio(),
                    model_deepseek41_contract::serving::PAGE_TOKENS as u32
                );
            }
        }
    }
}
