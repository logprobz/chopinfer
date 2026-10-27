//! Retired decode module recipes, rebuilt only by the model lab.
#![allow(dead_code)]
// The single frozen reference implementation is shared by source so model labs
// never link kernel-lab. Its source root is the SDK crates directory.
#[path = "../../../../crates/kernel-lab/families/src/dense_decode_reference.rs"]
pub mod frozen;
use crate::dense_decode_reference::frozen::{
    mxfp8_fold_decode, mxfp8_fold_reuse, mxfp8_fold_ticket,
};
pub use bundle_deepseek41::sm120::mxfp8_fold::*;
use kernel_build::CubinSpec;
use kernel_core::{mxfp8_gemm::Mxfp8FoldGemmQuery, precision::OperationUse};
use model_deepseek41_contract::{
    admission,
    geometry::{FLASH, Geometry},
};
use std::collections::BTreeMap;
pub const TICKET_FAMILY: &str = "mxfp8_fold_ticket_reference";
pub const PREFETCH_FAMILY: &str = "mxfp8_fold_prefetch_reference";
pub const DECODE_FAMILY: &str = "mxfp8_fold_decode_reference";
pub const REUSE_FAMILY: &str = "mxfp8_fold_reuse_reference";
pub const REUSE_CUBIN: &str = "mxfp8_fold_reuse_deepseek41_sm120";
pub fn ticket_entries(g: &Geometry, ranks: u32) -> Result<Vec<mxfp8_fold_ticket::Entry>, String> {
    decode_entries(g, ranks)?
        .into_iter()
        .map(|old| {
            let query = Mxfp8FoldGemmQuery {
                m_capacity: SMALL_CAPACITY,
                ..old.query
            };
            mxfp8_fold_ticket::Entry::new(
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

pub fn prefetch_entries(g: &Geometry, ranks: u32) -> Result<Vec<mxfp8_fold_ticket::Entry>, String> {
    ticket_entries(g, ranks)?
        .into_iter()
        .map(|old| {
            let q = old.query;
            mxfp8_fold_ticket::Entry::new(
                q,
                format!(
                    "chop_deepseek41_mxfp8_prefetch_n{}_k{}_g{}_{}_v3",
                    q.n,
                    q.k,
                    q.groups,
                    if old.paired { "pair_f32" } else { "f32" }
                ),
                old.paired,
            )
            .map(|entry| entry.with_weight_prefetch())
            .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn prefetch_catalog(g: &Geometry) -> Result<Vec<mxfp8_fold_ticket::Entry>, String> {
    let mut all = BTreeMap::new();
    for &ranks in admission::SUPPORTED_RANKS {
        for entry in prefetch_entries(g, ranks)? {
            all.insert(entry.symbol.clone(), entry);
        }
    }
    Ok(all.into_values().collect())
}

pub fn prefetch_cubin_name(g: &Geometry) -> String {
    format!("mxfp8_fold_prefetch_{}_sm120_v3", g.kernel_tag)
}

pub fn prefetch_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                PREFETCH_FAMILY,
                packed(
                    mxfp8_fold_ticket::cubin_spec(&prefetch_cubin_name(g), &prefetch_catalog(g)?),
                    prefetch_catalog(g)?.iter().map(|e| e.packed_spec()),
                )?,
            ))
        })
        .collect()
}
pub fn ticket_catalog(g: &Geometry) -> Result<Vec<mxfp8_fold_ticket::Entry>, String> {
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
                packed(
                    mxfp8_fold_ticket::cubin_spec(&ticket_cubin_name(g), &ticket_catalog(g)?),
                    ticket_catalog(g)?.iter().map(|e| e.packed_spec()),
                )?,
            ))
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct DecodeRegistered {
    pub projection: String,
    pub operand: u32,
    pub usage: OperationUse,
    pub entry: mxfp8_fold_decode::Entry,
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
    use mxfp8_fold_decode::Entry as DecodeEntry;
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

pub fn decode_entries(g: &Geometry, ranks: u32) -> Result<Vec<mxfp8_fold_decode::Entry>, String> {
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

pub fn decode_catalog(g: &Geometry) -> Result<Vec<mxfp8_fold_decode::Entry>, String> {
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
            let spec = packed(
                mxfp8_fold_decode::cubin_spec(&decode_cubin_name(g), &decode_catalog(g)?),
                decode_catalog(g)?.iter().map(|e| e.packed_spec()),
            )?;
            Ok((DECODE_FAMILY, spec))
        })
        .collect()
}

pub fn reuse_entries() -> Result<Vec<mxfp8_fold_reuse::Entry>, String> {
    let mut entries = Vec::new();
    for stream in bundle_deepseek41::sm120::mxfp8_fold_stream::catalog(&FLASH)? {
        let q = stream.ticket.query;
        if !stream.ticket.paired
            && [
                (8192, 1280, 1),
                (5120, 2048, 1),
                (1024, 4096, 2),
                (1280, 5120, 1),
            ]
            .contains(&(q.n, q.k, q.groups))
        {
            for tile_n in [32, 64] {
                entries.push(mxfp8_fold_reuse::Entry {
                    ticket: mxfp8_fold_ticket::Entry::new(q, stream.ticket.symbol.clone(), false)
                        .map_err(|e| e.to_string())?,
                    tile_n,
                });
            }
        }
    }
    if entries.len() != 8 {
        return Err("missing qualified projection reuse shapes".into());
    }
    Ok(entries)
}
pub fn reuse_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        REUSE_FAMILY,
        packed(
            mxfp8_fold_reuse::cubin_spec(REUSE_CUBIN, &reuse_entries()?),
            reuse_entries()?.iter().map(|e| e.packed_spec()),
        )?,
    )])
}

pub struct ReferenceModule {
    pub name: String,
    pub digest: String,
    pub symbols: std::collections::BTreeSet<String>,
    bytes: std::sync::OnceLock<Result<Vec<u8>, String>>,
}

pub fn modules() -> &'static [ReferenceModule] {
    static MODULES: std::sync::OnceLock<Vec<ReferenceModule>> = std::sync::OnceLock::new();
    MODULES.get_or_init(|| {
        [
            include_str!("../kernels.lock.d/mxfp8_fold_decode_reference.lock"),
            include_str!("../kernels.lock.d/mxfp8_fold_ticket_reference.lock"),
            include_str!("../kernels.lock.d/mxfp8_fold_prefetch_reference.lock"),
            include_str!("../kernels.lock.d/mxfp8_fold_reuse_reference.lock"),
        ]
        .into_iter()
        .flat_map(|text| {
            serde_json::from_str::<Vec<serde_json::Value>>(text)
                .expect("reference lock")
                .into_iter()
                .map(|record| ReferenceModule {
                    name: record["name"].as_str().unwrap().into(),
                    digest: record["cubin_sha256"].as_str().unwrap().into(),
                    symbols: record["entries"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|entry| entry["symbol"].as_str().unwrap().into())
                        .collect(),
                    bytes: std::sync::OnceLock::new(),
                })
        })
        .collect()
    })
}

pub fn owns(symbol: &str) -> bool {
    modules().iter().any(|m| m.symbols.contains(symbol))
}

/// Load only a requested frozen module, authenticated by its moved lab lock.
pub fn load_symbol(symbol: &str) -> Result<(&'static str, &'static [u8]), String> {
    if let Some(found) = bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol) {
        return Ok(found);
    }
    let module = modules()
        .iter()
        .find(|m| m.symbols.contains(symbol))
        .ok_or_else(|| format!("no served or reference module owns {symbol}"))?;
    let bytes = module
        .bytes
        .get_or_init(|| {
            let directory = std::env::var_os("CHOPINFER_DEEPSEEK41_FOLD_CUBINS")
                .ok_or("set CHOPINFER_DEEPSEEK41_FOLD_CUBINS for retired lab references")?;
            let path = std::path::PathBuf::from(directory).join(format!("{}.cubin", module.name));
            let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
            if kernel_build::digest(&bytes) != module.digest {
                return Err(format!(
                    "{} differs from its retained lab lock",
                    module.name
                ));
            }
            Ok(bytes)
        })
        .as_ref()
        .map_err(Clone::clone)?;
    Ok((&module.name, bytes))
}
