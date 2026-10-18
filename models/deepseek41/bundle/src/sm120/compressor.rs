//! Compressor and key-write entries derived from the model geometry.
//! Projections are bound separately from the projection declaration. These
//! entries consume F32 outputs without choosing weight layouts. The only
//! BF16 conversion is the trained quantizer's input rounding.

use kernel_build::{BuildError, CubinSpec, EntryDeclaration, SelectedCubin};
use kernel_core::compressed_kv::{CompressedKvStorage, CompressedKvWriteQuery};
use kernel_core::contract::{NumericalContract, RecurrentPhase, Refusal};
use kernel_core::gated_pooling::GatedPoolingQuery;
use kernel_core::pooled_indexer::{IndexerCandidates, PositionIndexerQuery};
use kernel_core::rmsnorm::RmsNormRequirement;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gated_pooling::key_write;
use kernels_sm120::gated_pooling::{self, Entry, F32Norm};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving;
use std::collections::{BTreeMap, BTreeSet};

pub const FAMILY: &str = "compressor";

#[derive(Clone, Copy, Debug)]
pub enum WriteQuery {
    Index(PositionIndexerQuery),
    Compressed(CompressedKvWriteQuery),
}

/// One geometry's rank-invariant compressor math. Capacity, page size and
/// verify row bounds come from the program's scheduling and cache layout.
pub struct CompressorKernels {
    pub pooling: Vec<Entry>,
    pub writes: Vec<(key_write::Entry, WriteQuery)>,
    /// Ratio one uses a stateless F32 norm with promoted BF16 weights.
    pub ratio_one: Option<RmsNormRequirement>,
    ratio_one_entry: Option<(String, F32Norm)>,
    tag: String,
}

impl CompressorKernels {
    pub fn serving(geometry: &Geometry) -> Result<Self, Refusal> {
        Self::new(
            geometry,
            serving::PREFILL_TOKENS,
            serving::PAGE_TOKENS as u32,
            serving::MAX_VERIFY_ROWS,
        )
    }

    pub fn new(
        geometry: &Geometry,
        capacity: u32,
        page_size: u32,
        verify_rows: u32,
    ) -> Result<Self, Refusal> {
        let shape = geometry.shape;
        let ratios: BTreeSet<_> = shape
            .csa
            .kv_sources
            .iter()
            .map(|&layer| geometry.compress_ratio(layer))
            .collect();
        if ratios.is_empty() || ratios.iter().any(|r| !matches!(r, 1 | 2)) {
            return Err(Refusal("compressor source layers need ratios one or two"));
        }
        let tag = geometry.kernel_tag.to_owned();
        let eps = shape.eps as f32;
        let mut pooling = Vec::new();
        if ratios.contains(&2) {
            for (name, phase) in [
                ("prefill", RecurrentPhase::Prefill),
                ("decode", RecurrentPhase::Decode),
                (
                    "verify",
                    RecurrentPhase::Verify {
                        max_rows: verify_rows,
                    },
                ),
            ] {
                pooling.push(Entry::new(
                    GatedPoolingQuery {
                        group: 2,
                        width: shape.attention.head_dim,
                        norm_eps_bits: eps.to_bits(),
                        phase,
                        capacity,
                    },
                    &format!("chop_{tag}_compressor_{name}"),
                )?);
            }
        }
        let ratio_one_entry = if ratios.contains(&1) {
            Some((
                format!("chop_{tag}_compressor_ratio_one_v1"),
                gated_pooling::ratio_one_norm(shape.attention.head_dim, eps, capacity)?,
            ))
        } else {
            None
        };
        let ratio_one = ratio_one_entry
            .as_ref()
            .map(|(_, norm)| *norm.requirement());
        let mut writes = Vec::new();
        for ratio in ratios {
            if !page_size.is_multiple_of(ratio) {
                return Err(Refusal("serving pages must hold whole compression groups"));
            }
            // The serving layout counts token positions; writer contracts
            // count completed groups inside each physical page.
            let compressed_page_size = page_size / ratio;
            let query = PositionIndexerQuery {
                heads: shape.indexer.heads,
                head_dim: shape.indexer.dim,
                compression_ratio: ratio,
                tail_rope_dims: shape.attention.rope_dim,
                key_norm_eps_bits: eps.to_bits(),
                max_top_k: shape.indexer.top_k,
                page_size: compressed_page_size,
                candidates: IndexerCandidates::All,
                capacity,
            };
            writes.push((
                key_write::Entry::index(&query, &format!("chop_{tag}_index_key_r{ratio}_v1"))?,
                WriteQuery::Index(query),
            ));
            for (name, storage) in [
                ("packed", CompressedKvStorage::Packed),
                ("bf16", CompressedKvStorage::Bf16),
            ] {
                let query = CompressedKvWriteQuery {
                    width: shape.attention.head_dim,
                    tail_rope_dims: shape.attention.rope_dim,
                    compression_ratio: ratio,
                    page_size: compressed_page_size,
                    storage,
                    capacity,
                };
                writes.push((
                    key_write::Entry::compressed(
                        &query,
                        &format!("chop_{tag}_compressed_key_{name}_r{ratio}_v1"),
                    )?,
                    WriteQuery::Compressed(query),
                ));
            }
        }
        Ok(Self {
            pooling,
            writes,
            ratio_one,
            ratio_one_entry,
            tag,
        })
    }

    pub fn register(&self, mut backend: Sm120) -> Sm120 {
        if let Some((symbol, norm)) = &self.ratio_one_entry {
            backend = backend.with_f32_rmsnorm(*norm, symbol.clone());
        }
        backend
            .with_gated_pooling(self.pooling.clone())
            .with_fp4_key_writes(self.writes.iter().map(|(entry, _)| entry.clone()))
    }

    pub fn cubins(&self) -> Result<Vec<CubinSpec>, Refusal> {
        let mut specs = Vec::new();
        if !self.pooling.is_empty() {
            specs.push(gated_pooling::cubin_spec(
                format!("{}_compressor_pooling_v1", self.tag),
                &self.pooling,
            )?);
        }
        specs.push(key_write::cubin_spec(
            format!("{}_compressor_keys_v1", self.tag),
            &self
                .writes
                .iter()
                .map(|(entry, _)| entry.clone())
                .collect::<Vec<_>>(),
        )?);
        if let Some((symbol, norm)) = &self.ratio_one_entry {
            specs.push(norm.cubin_spec(format!("{}_compressor_ratio_one_v1", self.tag), symbol));
        }
        Ok(specs)
    }

    /// Typed declarations travel with the same entries that register on
    /// the backend. The model's build owns their source-family locks.
    pub fn selected(&self) -> Result<Vec<SelectedCubin>, BuildError> {
        let specs = self.cubins().map_err(|e| BuildError(e.0.into()))?;
        let mut declared: BTreeMap<String, EntryDeclaration> = BTreeMap::new();
        for entry in &self.pooling {
            let query = entry.query();
            for (stage, symbol) in ["pool", "norm"].into_iter().zip(entry.symbols()) {
                declared.insert(
                    symbol.into(),
                    EntryDeclaration {
                        symbol: symbol.into(),
                        operation: "GatedPooling".into(),
                        contract: query
                            .numerical_contract_id()
                            .map_err(|e| BuildError(e.0.into()))?
                            .to_string(),
                        implementation: "sm120.gated-pooling.pair-f32-rmsnorm.v1".into(),
                        requirements: [
                            ("query".into(), format!("{query:?}")),
                            ("stage".into(), stage.into()),
                            (
                                "state".into(),
                                "raw F32 key and score tail; exact copies".into(),
                            ),
                            (
                                "rounding".into(),
                                "F32 pool, weighted F32 RMSNorm, F32 output".into(),
                            ),
                        ]
                        .into(),
                        correctness_case: Some("gated_pooling_f32".into()),
                    },
                );
            }
        }
        for (entry, query) in &self.writes {
            let contract = match query {
                WriteQuery::Index(q) => q.numerical_contract_id(),
                WriteQuery::Compressed(q) => q.numerical_contract_id(),
            }
            .map_err(|e| BuildError(e.0.into()))?;
            declared.insert(
                entry.symbol().into(),
                EntryDeclaration {
                    symbol: entry.symbol().into(),
                    operation: "Fp4KeyWrite".into(),
                    contract: contract.to_string(),
                    implementation: "sm120.fp4-key-write.f32-tail-rope-bf16-quantizer.v1".into(),
                    requirements: [
                        ("query".into(), format!("{query:?}")),
                        (
                            "position".into(),
                            "compressed position times compression ratio".into(),
                        ),
                        (
                            "quantizer".into(),
                            "trained BF16-input FP4 with no global scale".into(),
                        ),
                        (
                            "addressing".into(),
                            "independent 64-bit code and scale page strides".into(),
                        ),
                    ]
                    .into(),
                    correctness_case: Some("fp4_key_write".into()),
                },
            );
        }
        if let Some((symbol, norm)) = &self.ratio_one_entry {
            declared.insert(symbol.clone(), norm.declaration(symbol));
        }
        specs
            .into_iter()
            .map(|spec| {
                let entries = spec
                    .entries
                    .iter()
                    .map(|entry| {
                        declared.get(&entry.symbol).cloned().ok_or_else(|| {
                            BuildError(format!(
                                "missing compressor declaration for {}",
                                entry.symbol
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                SelectedCubin::new(spec, entries)
            })
            .collect()
    }
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, Refusal> {
    let mut specs = Vec::new();
    for geometry in geometries {
        specs.extend(
            CompressorKernels::serving(geometry)?
                .cubins()?
                .into_iter()
                .map(|spec| (FAMILY, spec)),
        );
    }
    Ok(specs)
}
