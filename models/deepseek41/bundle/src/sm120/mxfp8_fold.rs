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
use std::ops::Range;

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
    Ok(entries(g, ranks)?
        .into_iter()
        .fold(backend, Sm120::with_mxfp8_fold))
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
