//! The declaration's native MXFP8 matrices, including independent output
//! groups and the two shared-expert operands. Shapes come only from the
//! declaration and the compiled geometry catalog.

use kernel_build::CubinSpec;
use kernel_core::contract::{DType, Validate};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernel_core::precision::OperationUse;
use kernel_core::precision::execution::BlockScaledExecution;
use kernel_core::precision::format::FormatId;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold::Entry;
use model_deepseek41_contract::{admission, geometry::Geometry, projections};
use projection_spec::ProjectionKind;
use projection_spec::derive::{Topology, rank_selection};
use std::collections::BTreeMap;
use std::ops::{Range, RangeInclusive};

#[derive(Clone, Debug)]
pub struct Registered {
    pub projection: String,
    pub operand: u32,
    pub usage: OperationUse,
    pub source_groups: Range<u32>,
    /// Byte offsets within the packed codes and scale banks. Shared gate
    /// and up consume their own contiguous halves and write separate F32 rows.
    pub codes_offset: u64,
    pub scales_offset: u64,
    pub entry: Entry,
}

pub fn registrations(g: &Geometry, ranks: u32, rank: u32) -> Result<Vec<Registered>, String> {
    admission::check_ranks(g, ranks)?;
    if rank >= ranks {
        return Err("fold registration rank exceeds topology".into());
    }
    let keys = projections::geometry(g);
    let declaration = projections::declaration(g)
        .and_then(|d| d.expanded_profiles(&keys))
        .map_err(|e| e.0)?;
    let mut result = Vec::new();
    for p in declaration
        .projections
        .iter()
        .filter(|p| p.format == Some(FormatId::Fp8Block32))
    {
        if !matches!(
            p.kind,
            ProjectionKind::Dense {} | ProjectionKind::GroupedDense { .. }
        ) {
            return Err(format!("{} is not a dense fold projection", p.id));
        }
        if matches!(p.kind, ProjectionKind::GroupedDense { .. }) && p.operands.len() != 1 {
            return Err("grouped fold registration requires one operand per group".into());
        }
        let mut codes_offset = 0;
        for operand in 0..p.operands.len() as u32 {
            let selected =
                rank_selection(p, &keys, Topology { ranks, rank }, operand).map_err(|e| e.0)?;
            let n = u32::try_from(selected.physical.rows).map_err(|e| e.to_string())?;
            let k = u32::try_from(selected.physical.k).map_err(|e| e.to_string())?;
            let groups = selected.experts.end - selected.experts.start;
            let row_stride = groups.checked_mul(k).ok_or("fold input stride overflows")?;
            for usage in &p.uses {
                let m_capacity = u32::try_from(usage.capacity.resolve(&keys).map_err(|e| e.0)?)
                    .map_err(|e| e.to_string())?;
                let query = Mxfp8FoldGemmQuery {
                    m_capacity,
                    n,
                    k,
                    groups,
                    activation_row_stride: row_stride,
                    activation_group_stride: k,
                    output: DType::F32,
                    execution: BlockScaledExecution::NativeBlockScaledMma,
                };
                query.validate().map_err(|e| e.to_string())?;
                let entry = Entry::new(
                    query,
                    format!("chop_deepseek41_mxfp8_fold_m{m_capacity}_n{n}_k{k}_g{groups}_f32_v2"),
                )
                .map_err(|e| e.to_string())?;
                result.push(Registered {
                    projection: p.id.clone(),
                    operand,
                    usage: OperationUse {
                        phase: usage.phase,
                        capacity: m_capacity,
                        graph: usage.graph,
                    },
                    source_groups: selected.experts.clone(),
                    codes_offset,
                    scales_offset: codes_offset / 32,
                    entry,
                });
            }
            codes_offset += u64::from(groups) * u64::from(n) * u64::from(k);
        }
    }
    if result.is_empty() {
        return Err("geometry declares no executable native MXFP8 fold uses".into());
    }
    Ok(result)
}

pub fn entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    let mut entries = BTreeMap::new();
    for rank in 0..ranks {
        for registration in registrations(g, ranks, rank)? {
            let entry = registration.entry;
            if let Some(old) = entries.insert(entry.symbol.clone(), entry.clone())
                && old.query != entry.query
            {
                return Err("fold symbol collides across queries".into());
            }
        }
    }
    Ok(entries.into_values().collect())
}

pub fn catalog(g: &Geometry) -> Result<Vec<Entry>, String> {
    let mut all = BTreeMap::new();
    for &ranks in admission::SUPPORTED_RANKS {
        for entry in entries(g, ranks)? {
            all.insert(entry.symbol.clone(), entry);
        }
    }
    Ok(all.into_values().collect())
}

pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    let prefill: std::collections::BTreeSet<_> = prefill_entries(g, ranks)?
        .into_iter()
        .map(|entry| entry.symbol)
        .collect();
    let backend = entries(g, ranks)?
        .into_iter()
        .fold(backend, |backend, entry| {
            let backend = if prefill.contains(&entry.symbol) {
                backend.with_mxfp8_prefill(entry.clone())
            } else {
                backend
            };
            backend.with_mxfp8_fold(entry)
        });
    let backend = decode_entries(g, ranks)?
        .into_iter()
        .fold(backend, Sm120::with_mxfp8_fold_decode);
    Ok(ticket_entries(g, ranks)?
        .into_iter()
        .fold(backend, Sm120::with_mxfp8_fold_ticket))
}

pub const TICKET_FAMILY: &str = "mxfp8_fold_ticket";
pub const SMALL_CAPACITY: u32 = 64;

// Flash SM120, 250 W, 2026-10-19: ticket timings were within 3% of the
// split-fold baseline on both GPUs. Prefer the baseline for these ties.
// Each row is (N, K, row-count ranges); all are single projections at G=1.
const SPLIT_FOLD_TIES: &[(u32, u32, &[RangeInclusive<u32>])] = &[
    (16384, 1280, &[3..=4, 7..=11]),
    (5120, 1152, &[1..=2, 9..=16]),
    (5120, 576, &[1..=28]),
    (8192, 1280, &[1..=2, 5..=12, 16..=16]),
];

/// Keep split folds for the measured ties and the TP2 one/two-row regression.
/// Capacity selects an existing family; it does not change the arithmetic.
pub fn small_query(query: Mxfp8FoldGemmQuery, rows: u32) -> Mxfp8FoldGemmQuery {
    let tied = query.groups == 1
        && SPLIT_FOLD_TIES.iter().any(|(n, k, ranges)| {
            (query.n, query.k) == (*n, *k) && ranges.iter().any(|range| range.contains(&rows))
        });
    let split =
        tied || ((1..=2).contains(&rows) && (query.n, query.k, query.groups) == (16384, 1280, 1));
    Mxfp8FoldGemmQuery {
        m_capacity: if split { 32 } else { SMALL_CAPACITY },
        ..query
    }
}

pub fn ticket_entries(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<kernels_sm120::gemm::mxfp8_fold_ticket::Entry>, String> {
    decode_entries(g, ranks)?
        .into_iter()
        .map(|old| {
            let query = Mxfp8FoldGemmQuery {
                m_capacity: SMALL_CAPACITY,
                ..old.query
            };
            kernels_sm120::gemm::mxfp8_fold_ticket::Entry::new(
                query,
                format!(
                    "chop_deepseek41_mxfp8_ticket_n{}_k{}_g{}_{}_v2",
                    query.n,
                    query.k,
                    query.groups,
                    if old.paired { "pair_f32" } else { "f32" }
                ),
                old.paired,
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}
pub fn ticket_catalog(
    g: &Geometry,
) -> Result<Vec<kernels_sm120::gemm::mxfp8_fold_ticket::Entry>, String> {
    let mut all = BTreeMap::new();
    for &ranks in admission::SUPPORTED_RANKS {
        for entry in ticket_entries(g, ranks)? {
            all.insert(entry.symbol.clone(), entry);
        }
    }
    Ok(all.into_values().collect())
}
pub fn ticket_cubin_name(g: &Geometry) -> String {
    format!("mxfp8_fold_ticket_{}_sm120_v2", g.kernel_tag)
}
pub fn ticket_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                TICKET_FAMILY,
                kernels_sm120::gemm::mxfp8_fold_ticket::cubin_spec(
                    &ticket_cubin_name(g),
                    &ticket_catalog(g)?,
                )
                .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

pub const DECODE_BUCKETS: [u32; 6] = [1, 2, 4, 8, 16, 32];
pub const DECODE_FAMILY: &str = "mxfp8_fold_decode";

#[derive(Clone, Debug)]
pub struct DecodeRegistered {
    pub projection: String,
    pub operand: u32,
    pub usage: OperationUse,
    pub entry: kernels_sm120::gemm::mxfp8_fold_decode::Entry,
}

/// Bind every exact decode bucket to its declared projection. The same
/// compiled entry supports all six buckets, with a separately prepared
/// grid and workspace at each capacity. Prefill and verify keep v2.
pub fn decode_registrations(
    g: &Geometry,
    ranks: u32,
    rank: u32,
) -> Result<Vec<DecodeRegistered>, String> {
    use kernel_core::precision::ExecutionPhase;
    use kernels_sm120::gemm::mxfp8_fold_decode::Entry as DecodeEntry;
    let mut result = Vec::new();
    for r in registrations(g, ranks, rank)?
        .into_iter()
        .filter(|r| r.usage.phase == ExecutionPhase::Decode)
    {
        let query = Mxfp8FoldGemmQuery {
            m_capacity: 32,
            ..r.entry.query
        };
        let q = query;
        let symbol = format!(
            "chop_deepseek41_mxfp8_decode_n{}_k{}_g{}_f32_v1",
            q.n, q.k, q.groups
        );
        let entry = DecodeEntry::new(query, symbol, false).map_err(|e| e.to_string())?;
        for capacity in DECODE_BUCKETS {
            result.push(DecodeRegistered {
                projection: r.projection.clone(),
                operand: r.operand,
                usage: OperationUse {
                    capacity,
                    ..r.usage
                },
                entry: entry.clone(),
            });
            if r.projection == "layer.shared.gate-up" && r.operand == 0 {
                let paired = DecodeEntry::new(query, format!("{}_pair_v1", entry.symbol), true)
                    .map_err(|e| e.to_string())?;
                result.push(DecodeRegistered {
                    projection: r.projection.clone(),
                    operand: r.operand,
                    usage: OperationUse {
                        capacity,
                        ..r.usage
                    },
                    entry: paired,
                });
            }
        }
    }
    Ok(result)
}

pub fn decode_entries(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<kernels_sm120::gemm::mxfp8_fold_decode::Entry>, String> {
    let mut entries = BTreeMap::new();
    for rank in 0..ranks {
        for r in decode_registrations(g, ranks, rank)? {
            if let Some(old) = entries.insert(r.entry.symbol.clone(), r.entry.clone())
                && (old.query != r.entry.query || old.paired != r.entry.paired)
            {
                return Err("decode fold symbol collides across queries".into());
            }
        }
    }
    Ok(entries.into_values().collect())
}

pub fn decode_catalog(
    g: &Geometry,
) -> Result<Vec<kernels_sm120::gemm::mxfp8_fold_decode::Entry>, String> {
    let mut all = BTreeMap::new();
    for &ranks in admission::SUPPORTED_RANKS {
        for entry in decode_entries(g, ranks)? {
            all.insert(entry.symbol.clone(), entry);
        }
    }
    Ok(all.into_values().collect())
}

pub fn decode_cubin_name(g: &Geometry) -> String {
    format!("mxfp8_fold_decode_{}_sm120_v1", g.kernel_tag)
}

pub fn decode_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            let spec = kernels_sm120::gemm::mxfp8_fold_decode::cubin_spec(
                &decode_cubin_name(g),
                &decode_catalog(g)?,
            )
            .map_err(|e| e.to_string())?;
            Ok((DECODE_FAMILY, spec))
        })
        .collect()
}

/// The lock family of every geometry's fold cubin.
pub const FAMILY: &str = "mxfp8_fold";

/// A geometry's fold cubin, named by its kernel tag.
pub fn cubin_name(g: &Geometry) -> String {
    format!("mxfp8_fold_{}_sm120_v1", g.kernel_tag)
}

/// Each geometry's fold cubin, one per geometry in the fold family, so a
/// geometry added later only appends its record.
#[allow(dead_code, reason = "the bundle's build script compiles the cubins")]
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            let spec = kernels_sm120::gemm::mxfp8_fold::cubin_spec(&cubin_name(g), &catalog(g)?)
                .map_err(|e| e.to_string())?;
            Ok((FAMILY, spec))
        })
        .collect()
}

/// Only declared prefill uses opt into the staged engine. Decode and verify
/// retain their current entry and operand layout.
pub fn prefill_entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    let mut entries = BTreeMap::new();
    for r in registrations(g, ranks, 0)? {
        if r.usage.phase == kernel_core::precision::ExecutionPhase::Prefill {
            entries.insert(r.entry.symbol.clone(), r.entry);
        }
    }
    Ok(entries.into_values().collect())
}
pub fn prefill_catalog(g: &Geometry) -> Result<Vec<Entry>, String> {
    let mut entries = BTreeMap::new();
    for &ranks in admission::SUPPORTED_RANKS {
        for e in prefill_entries(g, ranks)? {
            entries.insert(e.symbol.clone(), e);
        }
    }
    Ok(entries.into_values().collect())
}
pub const PREFILL_FAMILY: &str = "mxfp8_prefill";
pub fn prefill_cubin_name(g: &Geometry) -> String {
    format!("mxfp8_prefill_{}_sm120", g.kernel_tag)
}
#[allow(dead_code, reason = "the build compiles the prefill entries")]
pub fn prefill_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                PREFILL_FAMILY,
                kernels_sm120::gemm::mxfp8_prefill::cubin_spec(
                    &prefill_cubin_name(g),
                    &prefill_catalog(g)?,
                )
                .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
