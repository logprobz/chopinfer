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

#[path = "mxfp8_compact.rs"]
pub mod compact;

#[path = "mxfp8_fold_routing.rs"]
mod routing;
#[allow(unused_imports, reason = "the lab compares selected schedule classes")]
pub use routing::{DecodePath, decode_path};

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
                backend.with_mxfp8_prefill_split_policy(
                    entry.clone(),
                    prefill_tail_policy(entry.query),
                )
            } else {
                backend
            };
            backend.with_mxfp8_fold(entry)
        });
    let mut backend = backend;
    for base in prefill_entries(g, ranks)? {
        for rows in compact::ROWS {
            if let Some(stages) = compact::stages(g, base.query, rows) {
                let entry =
                    kernels_sm120::gemm::mxfp8_prefill_compact::Entry::new(base.clone(), stages)
                        .map_err(|e| e.to_string())?;
                backend = backend.with_mxfp8_prefill_compact(entry, rows);
            }
        }
    }
    for entry in dense_entries(g, ranks)? {
        backend = backend.with_dense_decode(entry, None);
    }
    Ok(backend)
}

pub const DENSE_FAMILY: &str = "dense_decode";

pub fn dense_knobs(
    g: &Geometry,
    q: Mxfp8FoldGemmQuery,
    rows: u32,
    paired: bool,
) -> Result<kernels_sm120::gemm::dense_decode::Knobs, String> {
    use kernels_sm120::gemm::dense_decode::Knobs;
    match tuning_row(
        g,
        model_deepseek41_contract::tuning::Family::DenseFold,
        "dense_fold.fold.decode",
        q,
        paired,
        rows,
    ) {
        Some(row) => {
            let (knobs, folds) = Knobs::from_row(row)?;
            if folds.is_some() {
                return Err(format!("{}: runtime folds belong to reuse", row.key));
            }
            Ok(knobs)
        }
        None => Ok(Knobs::ticket(q.n)),
    }
}

pub fn dense_entry(
    q: Mxfp8FoldGemmQuery,
    paired: bool,
    knobs: kernels_sm120::gemm::dense_decode::Knobs,
) -> Result<kernels_sm120::gemm::dense_decode::Entry, String> {
    kernels_sm120::gemm::dense_decode::Entry::new(
        q,
        &format!(
            "chop_deepseek41_dense_decode_n{}_k{}_g{}_{}",
            q.n,
            q.k,
            q.groups,
            if paired { "pair" } else { "single" }
        ),
        paired,
        knobs,
    )
    .map_err(|e| e.to_string())
}

/// One exact-row registration per shape and bank choice. Missing tuning rows
/// retain the ticket baseline; present rows supply every compile-time knob.
pub fn dense_entries(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<kernels_sm120::gemm::dense_decode::Entry>, String> {
    let mut result = Vec::new();
    for base in decode_entries(g, ranks)? {
        for rows in 1..=SMALL_CAPACITY {
            result.push(dense_entry(
                routed_query(base.query, rows),
                base.paired,
                dense_knobs(g, base.query, rows, base.paired)?,
            )?);
        }
    }
    Ok(result)
}

pub fn dense_reuse_entries(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<(kernels_sm120::gemm::dense_decode::Entry, u32)>, String> {
    if g.id != model_deepseek41_contract::geometry::FLASH.id || ranks != 4 {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    for base in decode_entries(g, ranks)?.into_iter().filter(|e| !e.paired) {
        for rows in 9..=32 {
            if let Some(row) = tuning_row(
                g,
                model_deepseek41_contract::tuning::Family::Reuse,
                "reuse.fold.decode",
                base.query,
                false,
                rows,
            ) {
                let (knobs, folds) = kernels_sm120::gemm::dense_decode::Knobs::from_row(row)?;
                let folds = folds
                    .ok_or_else(|| format!("{}: reuse requires a runtime partition", row.key))?;
                let entry = dense_entry(routed_query(base.query, rows), false, knobs)?;
                entry.plans(rows, folds).map_err(|e| e.to_string())?;
                result.push((entry, folds));
            }
        }
    }
    Ok(result)
}

pub fn dense_catalog(
    g: &Geometry,
) -> Result<Vec<kernels_sm120::gemm::dense_decode::Entry>, String> {
    let mut entries = BTreeMap::new();
    for &ranks in admission::SUPPORTED_RANKS {
        for entry in dense_entries(g, ranks)? {
            entries.insert(entry.symbol.clone(), entry);
        }
        for (entry, _) in dense_reuse_entries(g, ranks)? {
            entries.insert(entry.symbol.clone(), entry);
        }
    }
    Ok(entries.into_values().collect())
}

pub fn dense_cubin_name(g: &Geometry) -> String {
    format!("dense_decode_{}_sm120_v1", g.kernel_tag)
}

pub fn dense_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                DENSE_FAMILY,
                kernels_sm120::gemm::dense_decode::cubin_spec(
                    &dense_cubin_name(g),
                    &dense_catalog(g)?,
                    true,
                )
                .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

/// The wide verify fold module of each geometry that has one
/// ([`wide_catalog`]).
pub fn dense_wide_cubins(
    geometries: &[Geometry],
) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let entries = wide_catalog(g)?;
        if !entries.is_empty() {
            specs.push((
                DENSE_WIDE_FAMILY,
                kernels_sm120::gemm::dense_decode::cubin_spec(
                    &dense_wide_cubin_name(g),
                    &entries,
                    true,
                )
                .map_err(|e| e.to_string())?,
            ));
        }
    }
    Ok(specs)
}

pub const DENSE_WIDE_FAMILY: &str = "dense_decode_wide";

/// The module of the wide verify dense folds ([`wide_entry`]), which only
/// instances with wide verify buckets load.
pub fn dense_wide_cubin_name(g: &Geometry) -> String {
    format!("dense_decode_wide_{}_sm120_v1", g.kernel_tag)
}

/// Whether `g` at `ranks` has wide verify dense folds: Flash TP4, the
/// topology whose verify rows exceed the 64-row decode buckets.
pub fn wide_served(g: &Geometry, ranks: u32) -> bool {
    g.id == model_deepseek41_contract::geometry::FLASH.id && ranks == 4
}

/// The schedule of `q`'s wide verify fold at `bucket`, one of
/// `verify_bounds::WIDE`: its `dense_fold.wide.verify` row, or the ticket
/// baseline.
pub fn wide_knobs(
    g: &Geometry,
    q: Mxfp8FoldGemmQuery,
    paired: bool,
    bucket: u32,
) -> Result<kernels_sm120::gemm::dense_decode::Knobs, String> {
    use kernels_sm120::gemm::dense_decode::Knobs;
    let row = model_deepseek41_contract::tuning::select(
        g,
        model_deepseek41_contract::tuning::Family::DenseFold,
        "dense_fold.wide.verify",
        "mxfp8",
        &[
            ("n", q.n),
            ("k", q.k),
            ("groups", q.groups),
            ("paired", u32::from(paired)),
        ],
        bucket,
        bucket,
    );
    match row {
        Some(row) => {
            let (knobs, folds) = Knobs::from_row(row)?;
            if folds.is_some() {
                return Err(format!("{}: wide folds take no runtime partition", row.key));
            }
            Ok(knobs)
        }
        None => Ok(Knobs::ticket(q.n)),
    }
}

/// The decode folds a target verify step above 64 rows runs through the
/// decode template: each backbone projection's and the indexer query's,
/// single banked. Engram's wide buckets take its matrix tile; the
/// drafter's verify forwards stay within 64 rows.
pub fn wide_bases(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<kernels_sm120::gemm::dense_decode::Entry>, String> {
    let mut entries = BTreeMap::new();
    for rank in 0..ranks {
        for r in decode_registrations(g, ranks, rank)? {
            let served = (r.projection.starts_with("layer.") || r.projection == "csa.indexer.wq_b")
                && !r.entry.paired;
            if served {
                entries.insert(r.entry.symbol.clone(), r.entry);
            }
        }
    }
    Ok(entries.into_values().collect())
}

/// The wide verify fold of decode base `base` (a [`wide_bases`] entry)
/// for an instance bucket of `capacity` rows, which caps the wide
/// bucket `bucket` (`verify_bounds::WIDE`): the decode template with a
/// header of `dense_decode::WIDE_ROWS` rows and `bucket`'s schedule, so
/// one symbol serves every capacity of a schedule.
pub fn wide_entry(
    g: &Geometry,
    base: &kernels_sm120::gemm::dense_decode::Entry,
    bucket: u32,
    capacity: u32,
) -> Result<kernels_sm120::gemm::dense_decode::Entry, String> {
    let q = base.query;
    kernels_sm120::gemm::dense_decode::Entry::wide(
        routed_query(q, capacity),
        &format!(
            "chop_deepseek41_dense_decode_n{}_k{}_g{}_{}",
            q.n,
            q.k,
            q.groups,
            if base.paired { "pair" } else { "single" }
        ),
        base.paired,
        wide_knobs(g, q, base.paired, bucket)?,
        kernels_sm120::gemm::dense_decode::WIDE_ROWS,
    )
    .map_err(|e| e.to_string())
}

/// Every wide verify fold the module holds: each [`wide_bases`] entry at
/// each wide bucket's schedule.
pub fn wide_catalog(g: &Geometry) -> Result<Vec<kernels_sm120::gemm::dense_decode::Entry>, String> {
    let mut entries = BTreeMap::new();
    for &ranks in admission::SUPPORTED_RANKS {
        if !wide_served(g, ranks) {
            continue;
        }
        for base in wide_bases(g, ranks)? {
            for bucket in model_deepseek41_contract::verify_bounds::WIDE {
                let entry = wide_entry(g, &base, bucket, bucket)?;
                entries.insert(entry.symbol.clone(), entry);
            }
        }
    }
    Ok(entries.into_values().collect())
}

/// Registers the wide verify folds of an instance whose wide verify buckets
/// are `buckets` (`Shape::wide_verify_rows`), each capping the
/// `verify_bounds::WIDE` bucket at its position.
pub fn register_wide(
    mut backend: Sm120,
    g: &Geometry,
    ranks: u32,
    buckets: &[u32],
) -> Result<Sm120, String> {
    if buckets.is_empty() {
        return Ok(backend);
    }
    if !wide_served(g, ranks) {
        return Err("wide verify folds serve Flash TP4 only".into());
    }
    for base in wide_bases(g, ranks)? {
        for (&capacity, bucket) in buckets
            .iter()
            .zip(model_deepseek41_contract::verify_bounds::WIDE)
        {
            backend = backend.with_dense_decode(wide_entry(g, &base, bucket, capacity)?, None);
        }
    }
    Ok(backend)
}

pub const SMALL_CAPACITY: u32 = 64;

/// Exact graph capacity selects the qualified single or paired registration.
pub fn routed_query(query: Mxfp8FoldGemmQuery, rows: u32) -> Mxfp8FoldGemmQuery {
    Mxfp8FoldGemmQuery {
        m_capacity: rows,
        ..query
    }
}

/// The accepted ticket baseline: split folds for measured ties and the
/// TP2 one/two-row regression. The lab also uses this to compare candidates.
pub fn small_query(g: &Geometry, query: Mxfp8FoldGemmQuery, rows: u32) -> Mxfp8FoldGemmQuery {
    let split = tuning_row(
        g,
        model_deepseek41_contract::tuning::Family::DenseFold,
        "dense_fold.split.decode",
        query,
        false,
        rows,
    )
    .is_some();
    Mxfp8FoldGemmQuery {
        m_capacity: if split { 32 } else { SMALL_CAPACITY },
        ..query
    }
}

/// `spec` with its weight readers replaced by their packed `twins`: the
/// bundle stores every MXFP8 fold projection's codes as
/// `mxfp8-fold-packed-8x128-v1`, so its fold cubins hold only packed readers.
pub fn packed(
    spec: Result<CubinSpec, kernel_core::contract::Refusal>,
    twins: impl IntoIterator<Item = Result<kernel_build::EntrySpec, kernel_core::contract::Refusal>>,
) -> Result<CubinSpec, String> {
    packed_all(spec, twins.into_iter().map(|twin| twin.map(|t| vec![t])))
}

/// [`packed`] for families with several packed twins per entry.
pub fn packed_all(
    spec: Result<CubinSpec, kernel_core::contract::Refusal>,
    twins: impl IntoIterator<
        Item = Result<Vec<kernel_build::EntrySpec>, kernel_core::contract::Refusal>,
    >,
) -> Result<CubinSpec, String> {
    let twins = twins
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    kernels_sm120::gemm::mxfp8_fold_packed::replace(
        spec.map_err(|e| e.to_string())?,
        twins.into_iter().flatten(),
    )
    .map_err(|e| e.to_string())
}

pub use model_deepseek41_contract::serving::DECODE_BUCKETS;

#[derive(Clone, Debug)]
pub struct DecodeRegistered {
    pub projection: String,
    pub operand: u32,
    pub usage: OperationUse,
    pub entry: kernels_sm120::gemm::dense_decode::Entry,
}

/// Enumerate every decode bucket and declared projection. Registration
/// applies the selected row's schedule to this geometry and bank choice.
pub fn decode_registrations(
    g: &Geometry,
    ranks: u32,
    rank: u32,
) -> Result<Vec<DecodeRegistered>, String> {
    use kernel_core::precision::ExecutionPhase;
    let mut result = Vec::new();
    for r in registrations(g, ranks, rank)?
        .into_iter()
        .filter(|r| r.usage.phase == ExecutionPhase::Decode)
    {
        let query = Mxfp8FoldGemmQuery {
            m_capacity: 32,
            ..r.entry.query
        };
        let entry = dense_entry(
            query,
            false,
            kernels_sm120::gemm::dense_decode::Knobs::ticket(query.n),
        )?;
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
                let paired = dense_entry(query, true, entry.knobs)?;
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
) -> Result<Vec<kernels_sm120::gemm::dense_decode::Entry>, String> {
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
            let spec = packed(
                kernels_sm120::gemm::mxfp8_fold::cubin_spec(&cubin_name(g), &catalog(g)?),
                catalog(g)?.iter().map(Entry::packed_spec),
            )?;
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
                packed(
                    kernels_sm120::gemm::mxfp8_prefill::cubin_spec(
                        &prefill_cubin_name(g),
                        &prefill_catalog(g)?,
                    ),
                    prefill_catalog(g)?
                        .iter()
                        .map(kernels_sm120::gemm::mxfp8_prefill::packed_spec),
                )?,
            ))
        })
        .collect()
}

/// Split-tail candidate, isolated from the original TMA cubins.
pub const PREFILL_SPLIT_FAMILY: &str = "mxfp8_prefill_split";
pub fn prefill_split_cubin_name(g: &Geometry) -> String {
    format!("mxfp8_prefill_split_{}_sm120", g.kernel_tag)
}
#[allow(dead_code, reason = "the build compiles the split prefill entries")]
pub fn prefill_split_cubins(
    geometries: &[Geometry],
) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                PREFILL_SPLIT_FAMILY,
                packed_all(
                    kernels_sm120::gemm::mxfp8_prefill_split::cubin_spec(
                        &prefill_split_cubin_name(g),
                        &prefill_catalog(g)?,
                    ),
                    prefill_catalog(g)?
                        .iter()
                        .map(kernels_sm120::gemm::mxfp8_prefill_split::packed_specs),
                )?,
            ))
        })
        .collect()
}

/// Paired 250 W measurements (2026-10-19) favor split tails except on
/// wide q projections and larger grouped o projections. Retain TMA in
/// their measured regressions and near ties. Between sampled row counts,
/// use conservative bounds at 16-row fold-tile and chunk boundaries;
/// these bounds are interpolation, not measured crossover points.
pub fn prefill_tail_policy(
    q: Mxfp8FoldGemmQuery,
) -> kernels_sm120::gemm::mxfp8_prefill_split::TailPolicy {
    use kernels_sm120::gemm::mxfp8_prefill_split::TailPolicy;
    match (q.n, q.k, q.groups) {
        (16384 | 32768, 1280, 1) | (1024, 4096, 8) => TailPolicy {
            max_tail_rows: 16,
            ..TailPolicy::ALL
        },
        (8192, 1280, 1) => TailPolicy {
            max_tail_rows: 112,
            ..TailPolicy::ALL
        },
        (1024, 4096, 4) => TailPolicy {
            max_rows: 1024,
            ..TailPolicy::ALL
        },
        _ => TailPolicy::ALL,
    }
}

/// Select an exact-row registration from the 64-row fold catalog. Decode,
/// verify and qualified short prefills share these offers; their precedence
/// and prepared query bounds remain with the callers.
pub fn tuning_row(
    g: &Geometry,
    family: model_deepseek41_contract::tuning::Family,
    op: &str,
    q: Mxfp8FoldGemmQuery,
    paired: bool,
    rows: u32,
) -> Option<&'static kernel_build::tuning::Row> {
    model_deepseek41_contract::tuning::select(
        g,
        family,
        op,
        "mxfp8",
        &[
            ("n", q.n),
            ("k", q.k),
            ("groups", q.groups),
            ("paired", u32::from(paired)),
        ],
        SMALL_CAPACITY,
        rows,
    )
}

/// Status sites have fixed Flash TP4 shapes. Their producer and bank gates
/// are checked before these descriptors are used.
pub fn status_tuning(
    op: &str,
    bucket: u32,
    rows: u32,
) -> Option<&'static kernel_build::tuning::Row> {
    model_deepseek41_contract::tuning::select(
        &model_deepseek41_contract::geometry::FLASH,
        model_deepseek41_contract::tuning::Family::DenseFold,
        op,
        "mxfp8",
        &[("ranks", 4)],
        bucket,
        rows,
    )
}

pub fn tuning_count(row: &kernel_build::tuning::Row, name: &str) -> u32 {
    row.knobs
        .get(name)
        .and_then(|v| v.count())
        .unwrap_or_else(|| panic!("{}: missing count {name}", row.key))
}

/// Iterate measured rows without retaining a second registration list.
pub fn status_rows(op: &str, bucket: u32) -> impl Iterator<Item = u32> + '_ {
    model_deepseek41_contract::tuning::table(
        &model_deepseek41_contract::geometry::FLASH,
        model_deepseek41_contract::tuning::Family::DenseFold,
    )
    .into_iter()
    .flat_map(|table| table.rows())
    .filter(move |row| {
        row.key.op == op
            && row.key.weights == "mxfp8"
            && row.key.bucket == bucket
            && row.key.shape.len() == 1
            && row.key.shape.get("ranks") == Some(&4)
    })
    .flat_map(|row| {
        let [first, last] = row.key.rows.unwrap_or([1, row.key.bucket]);
        first..=last
    })
}
