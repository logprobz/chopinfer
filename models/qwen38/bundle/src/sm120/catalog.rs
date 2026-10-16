//! Qwen3.8's entries of the SM120 kernel catalogs (dense and Qwen-owned
//! GEMMs, experts and routes, QSA, the linear-attention engine), the
//! hyper-connection comm entries of each topology, the workspace and shared
//! memory they need, and, in tests, the reference launch builders the
//! prepared ops' launches were checked against.

use kernel_core::contract::{DType, Refusal, SigmoidMeanHyperConnectionQuery, StaticQuery};
use kernels_sm120::Profile;

use crate::contract::{DECODE_ENTRY_BUCKETS, DECODE_ENTRY_ROWS, PREFILL_TOKENS, Ranks, Shape};
use crate::scratch::Served;

// The test-only launch builders' names.
#[cfg(test)]
use crate::dispatch::BufferBindings;
#[cfg(test)]
use crate::pipeline::PreparedLaunch;
#[cfg(test)]
use kernel_core::contract::{Argument, Validate};
#[cfg(test)]
use runtime_bundle_abi::DeviceBufferSpanV1;

/// Whether an entry belongs to one of the Qwen3.8 profiles: the TP4 one,
/// which also holds the rank-invariant entries, or a per-rank one.
fn qwen(profile: Profile) -> bool {
    model_qwen38_contract::sm120::geometry::owner(profile).is_some()
}

/// The hyper-connection all-reduce with inject and the next grouped norm
/// over `shape`'s ranks.
pub fn hc_query(shape: Shape) -> SigmoidMeanHyperConnectionQuery {
    geometry_owner(shape).transport.query(shape.ranks())
}

pub fn geometry_owner(
    shape: Shape,
) -> &'static model_qwen38_contract::sm120::geometry::GeometryModel {
    model_qwen38_contract::sm120::geometry::owner(shape.profile())
        .expect("registered geometry profile")
}

/// Whether `symbol` is a phased prefill all-reduce entry, of either norm
/// weight dtype.
pub fn phased_comm(shape: Shape, symbol: &str) -> bool {
    [DType::Bf16, DType::F32].into_iter().any(|weight| {
        geometry_owner(shape)
            .transport_for(weight)
            .is_ok_and(|t| [t.scatter_symbol, t.gather_symbol].contains(&symbol))
    })
}

/// The spinning comm entries of an instance of `shape` whose graph buckets
/// reach `rows` rows, each at its largest plan: the decode entries at 32
/// rows, the general entry's one-shot and phased pair at its capacity and,
/// when graphs take it, the wide entry at its largest bucket. Instance
/// create checks each against the device
/// (`bundle_runtime::residency`). With pipelined prefill `lanes`, the
/// phased pair also runs at the comm lane's grid; a grid past what an SM120
/// GPU can hold at once (`overlap::MAX_COMM_CTAS`) is refused.
pub fn spinning_entries(
    shape: Shape,
    rows: u32,
    f32_norms: bool,
    lanes: Option<&bundle_runtime::lanes::Config>,
) -> Result<Vec<bundle_runtime::residency::SpinningEntry>, Refusal> {
    use bundle_runtime::lanes::{Lane, LaunchFacts, adjust_grid};
    use bundle_runtime::residency::largest;
    let mut entries = spinning_entries_for(shape, rows, DType::Bf16)?;
    // An image with F32 norm weights also launches their forms.
    if f32_norms {
        entries.extend(spinning_entries_for(shape, rows, DType::F32)?);
    }
    let mut entries = largest(entries);
    if let Some(lanes) = lanes {
        if lanes
            .comm_ctas
            .is_some_and(|ctas| ctas > super::overlap::MAX_COMM_CTAS)
        {
            return Err(Refusal(
                "the overlap's comm grid exceeds the CTAs an SM120 GPU keeps resident",
            ));
        }
        let phased = LaunchFacts {
            persistent: false,
            phased_comm: true,
            collective: true,
        };
        let adjusted = entries
            .iter()
            .copied()
            .filter(|entry| phased_comm(shape, entry.symbol))
            .map(|mut entry| {
                adjust_grid(&mut entry.plan.grid, Lane::Comm, lanes, phased);
                entry
            })
            .collect::<Vec<_>>();
        entries.extend(adjusted);
        entries = largest(entries);
    }
    Ok(entries)
}

fn spinning_entries_for(
    shape: Shape,
    rows: u32,
    weight: DType,
) -> Result<Vec<bundle_runtime::residency::SpinningEntry>, Refusal> {
    let query = hc_query(shape);
    let kernel_core::contract::SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("Qwen3.8 all-reduce query"));
    };
    let transport = &geometry_owner(shape).transport_for(weight)?;
    let decode = &geometry_owner(shape).decode_for(weight)?;
    Ok(
        kernels_sm120::comm::hyper_connection_decode::spinning_entries(
            transport,
            decode,
            query,
            capacity,
            shape.ranks(),
            &DECODE_ENTRY_BUCKETS,
            rows,
        )?
        .into_iter()
        .map(|entry| bundle_runtime::residency::SpinningEntry {
            symbol: entry.symbol,
            plan: entry.plan,
            wide: entry.wide,
        })
        .collect(),
    )
}

/// The decode all-reduce over a graph bucket of `rows` prepared rows: v3 at
/// the LL prepared rows, v2 above. Both preserve the decode arguments and
/// bits.
///
/// Lowering prepares the collectives through the backend (`crate::ops`);
/// this builder is the reference their launches are checked against.
#[cfg(test)]
pub fn allreduce_hc_decode(
    shape: Shape,
    weight: DType,
    call: model_qwen38_contract::sm120::comm::decode::Call,
    bindings: BufferBindings,
) -> Result<PreparedLaunch, Refusal> {
    use kernels_sm120::comm::hyper_connection_decode as hc;
    let decode = &geometry_owner(shape).decode_for(weight)?;
    let symbol = if call.rows <= hc::LL_MAX_ROWS {
        hc::symbol_v3(decode, shape.ranks())?
    } else {
        hc::symbol_v2(decode, shape.ranks())?
    };
    Ok(PreparedLaunch {
        symbol: symbol.into(),
        rows: call.rows,
        plan: model_qwen38_contract::sm120::comm::decode::plan_with(decode.shape, call.rows)?,
        arguments: model_qwen38_contract::sm120::comm::decode::arguments(call)?,
        bindings,
    })
}

/// Whether `symbol` is a vocabulary head entry of any Qwen3.8 topology,
/// `chop_gemm_{tag}_head_*`.
#[cfg(test)]
fn vocabulary_head(symbol: &str) -> bool {
    model_qwen38_contract::geometry::GEOMETRIES.iter().any(|g| {
        [Ranks::TP2, Ranks::TP4].iter().any(|r| {
            Shape::of_geometry(g, *r)
                .is_some_and(|s| symbol.starts_with(&format!("chop_gemm_{}_head_", s.tag())))
        })
    })
}

/// Refuses a row gather whose `rows` rows of `shard` BF16 values overrun
/// this rank's input or the target's output, which holds the shards of all
/// `ranks` ranks.
///
/// Lowering prepares the collectives through the backend (`crate::ops`);
/// this builder is the reference their launches are checked against.
#[cfg(test)]
pub fn check_gather_extents(
    input: DeviceBufferSpanV1,
    target_output: DeviceBufferSpanV1,
    rows: u32,
    shard: u64,
    ranks: u32,
) -> Result<(), Refusal> {
    let bytes = u64::from(rows)
        .checked_mul(shard)
        .and_then(|values| values.checked_mul(2))
        .ok_or(Refusal("Qwen3.8 row gather size overflows"))?;
    if ranks == 0 || input.length < bytes || target_output.length / u64::from(ranks) < bytes {
        return Err(Refusal("Qwen3.8 row gather exceeds its buffers"));
    }
    Ok(())
}

pub mod moe {
    use kernels_sm120::Profile;
    pub use kernels_sm120::moe::expert_parallel::RouteEntry;
    use model_qwen38_contract::sm120::moe as catalog;

    pub fn route_catalog() -> Vec<RouteEntry> {
        catalog::route_catalog()
            .into_iter()
            .chain(catalog::parallel_route_catalog())
            .collect()
    }
    pub use kernels_sm120::moe::{Phase, Projection, Variant};

    /// The compiled Qwen3.8 expert entries of every topology.
    pub fn variants() -> &'static [Variant] {
        static VARIANTS: std::sync::OnceLock<Vec<Variant>> = std::sync::OnceLock::new();
        VARIANTS.get_or_init(catalog::catalog)
    }

    /// The expert entry of `profile`'s local experts.
    pub fn variant(
        profile: Profile,
        projection: Projection,
        phase: Phase,
        tokens: u32,
    ) -> Result<&'static Variant, kernel_core::contract::Refusal> {
        variants()
            .iter()
            .find(|v| {
                v.profile == profile
                    && v.projection == projection
                    && v.phase == phase
                    && v.tokens_capacity == tokens
            })
            .ok_or(kernel_core::contract::Refusal(
                "no Qwen3.8 expert entry at this capacity",
            ))
    }

    /// The route entry `operation` of `profile`'s local experts; the router
    /// does not depend on them and is the TP4 profile's.
    #[cfg(test)]
    pub fn route(
        profile: Profile,
        operation: &str,
    ) -> Result<&'static RouteEntry, kernel_core::contract::Refusal> {
        static ENTRIES: std::sync::OnceLock<Vec<RouteEntry>> = std::sync::OnceLock::new();
        let profile = if operation == "router" {
            model_qwen38_contract::sm120::geometry::owner(profile)
                .ok_or(kernel_core::contract::Refusal(
                    "unknown Qwen geometry profile",
                ))?
                .model
                .at(4)
        } else {
            profile
        };
        let symbol = format!("chop_moe_{}_{operation}_v1", profile.tag());
        ENTRIES
            .get_or_init(route_catalog)
            .iter()
            .find(|entry| entry.symbol == symbol)
            .ok_or(kernel_core::contract::Refusal(
                "unknown Qwen3.8 route entry",
            ))
    }
}

pub mod dense {
    pub use kernels_sm120::gemm::GemmBuffers;

    /// The compiled Qwen3.8 dense entries on the shared GEMM families, with
    /// their profiles.
    pub fn profiled() -> &'static [(
        kernels_sm120::Profile,
        kernel_core::contract::InstantiationSpec,
    )] {
        static ENTRIES: std::sync::OnceLock<
            Vec<(
                kernels_sm120::Profile,
                kernel_core::contract::InstantiationSpec,
            )>,
        > = std::sync::OnceLock::new();
        ENTRIES.get_or_init(|| {
            model_qwen38_contract::sm120::gemm::catalog()
                .into_iter()
                .map(|entry| (entry.profile, entry.instantiation))
                .collect()
        })
    }

    /// The dense entries of `profiles`.
    pub fn entries(
        profiles: &[kernels_sm120::Profile],
    ) -> impl Iterator<Item = &'static kernel_core::contract::InstantiationSpec> {
        profiled()
            .iter()
            .filter(|(profile, _)| profiles.contains(profile))
            .map(|(_, entry)| entry)
    }

    pub fn entry(
        symbol: &str,
    ) -> Result<&'static kernel_core::contract::InstantiationSpec, kernel_core::contract::Refusal>
    {
        profiled()
            .iter()
            .map(|(_, entry)| entry)
            .find(|entry| entry.symbol == symbol)
            .ok_or(kernel_core::contract::Refusal(
                "unknown Qwen3.8 dense entry",
            ))
    }
}

/// A BF16 fragment-weight GEMM of the shared families (the shared expert
/// entries today) over `live` rows.
#[cfg(test)]
pub fn gemm(
    symbol: &str,
    live: u32,
    weight: DeviceBufferSpanV1,
    activation: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
    partial: DeviceBufferSpanV1,
    semaphores: DeviceBufferSpanV1,
) -> Result<PreparedLaunch, Refusal> {
    let entry = dense::entry(symbol)?;
    gemm_entry(entry, live, weight, activation, output, partial, semaphores)
}

#[cfg(test)]
pub fn gemm_entry(
    entry: &kernel_core::contract::InstantiationSpec,
    live: u32,
    weight: DeviceBufferSpanV1,
    activation: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
    partial: DeviceBufferSpanV1,
    semaphores: DeviceBufferSpanV1,
) -> Result<PreparedLaunch, Refusal> {
    let symbol = entry.symbol.as_str();
    let StaticQuery::Gemm(query) = entry.key.query else {
        return Err(Refusal("Qwen3.8 dense entry has no GEMM query"));
    };
    let rows = if query.epilogues.swiglu.is_some() {
        query.n / 2
    } else {
        query.n
    };
    // Every output holds the entry's full row capacity except a vocabulary
    // head's: logits buffers hold the rows a phase can carry, and fragment
    // stores are masked by the live row count.
    let output_bytes = if query.output == DType::F32 { 4 } else { 2 };
    let output_rows = if vocabulary_head(symbol) {
        live
    } else {
        query.m_capacity
    };
    if live == 0
        || live > query.m_capacity
        || weight.length < u64::from(query.n) * u64::from(query.k) * 2
        || activation.length < u64::from(query.m_capacity) * u64::from(query.k) * 2
        || output.length < u64::from(output_rows) * u64::from(rows) * output_bytes
        || semaphores.length < 4
    {
        return Err(Refusal("Qwen3.8 dense views do not cover the entry"));
    }
    let mut bindings = BufferBindings::default();
    let activation_id = bindings.bind(activation);
    let weight_id = bindings.bind(weight);
    let output_id = bindings.bind(output);
    let partial_id = bindings.bind(partial);
    let semaphores_id = bindings.bind(semaphores);
    let decode = query.m_capacity <= DECODE_ENTRY_ROWS;
    let maps =
        kernels_sm120::gemm::fragment_maps(&entry.key, &query, activation_id, weight_id, decode);
    let counts = kernels_sm120::gemm::row_counts(&entry.key, live);
    let buffers = dense::GemmBuffers {
        output: output_id,
        partial: partial_id,
        semaphores: semaphores_id,
        weight_scales: None,
        activation_scales: None,
        residual: None,
        weight_global_scale: None,
        activation_global_scale: None,
    };
    let plan = kernels_sm120::gemm::plan(&entry.key, counts)?;
    if partial.length < plan.workspace_bytes {
        return Err(Refusal("Qwen3.8 dense workspace is too small"));
    }
    let arguments = kernels_sm120::gemm::arguments(&entry.key, counts, maps, buffers)?;
    Ok(PreparedLaunch {
        symbol: entry.symbol.clone(),
        rows: live,
        plan,
        arguments,
        bindings,
    })
}

/// The Qwen-owned GEMM kernels (`kernels_sm120::gemm::fused`): cluster
/// decode, short K and skinny.
pub mod qwen {
    pub use kernels_sm120::gemm::fused::{Buffers, Entry, Kind};

    pub fn entries() -> &'static [Entry] {
        static ENTRIES: std::sync::OnceLock<Vec<Entry>> = std::sync::OnceLock::new();
        ENTRIES.get_or_init(model_qwen38_contract::sm120::gemm::fused_catalog)
    }

    pub fn entry(symbol: &str) -> Option<&'static Entry> {
        entries().iter().find(|entry| entry.symbol == symbol)
    }
}

/// A Qwen-owned GEMM over `live` rows: one fragment view per weight, the
/// activation, and `(buffer, row stride in elements)` outputs, one per
/// weight for the cluster decode and one otherwise (a skinny entry writes
/// weight s at columns `s * rows` of each row). `prefill` selects the
/// entry's eager prefill schedule (`Entry::prefill_plan`).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn qwen_gemm(
    entry: &qwen::Entry,
    live: u32,
    prefill: bool,
    weights: &[DeviceBufferSpanV1],
    activation: DeviceBufferSpanV1,
    outputs: &[(DeviceBufferSpanV1, u32)],
    partial: DeviceBufferSpanV1,
    semaphores: DeviceBufferSpanV1,
) -> Result<PreparedLaunch, Refusal> {
    let query = entry.query;
    let cluster = matches!(entry.kind, qwen::Kind::ClusterDecode { .. });
    let halves = 1 + u32::from(query.epilogues.swiglu.is_some());
    let element: u64 = if query.output == DType::F32 { 4 } else { 2 };
    let columns = |index: usize| {
        if cluster {
            entry.segments[index] / halves
        } else {
            query.n / halves
        }
    };
    if live == 0
        || live > query.m_capacity
        || weights.len() != entry.segments.len()
        || outputs.len() != if cluster { weights.len() } else { 1 }
        || weights.iter().zip(&entry.segments).any(|(weight, rows)| {
            weight.length < u64::from(rows.div_ceil(16) * 16) * u64::from(query.k) * 2
        })
        || activation.length < u64::from(live) * u64::from(query.k) * 2
        || outputs.iter().enumerate().any(|(index, (output, stride))| {
            *stride < columns(index)
                || output.length
                    < (u64::from(live - 1) * u64::from(*stride) + u64::from(columns(index)))
                        * element
        })
    {
        return Err(Refusal("Qwen3.8 GEMM views do not cover the entry"));
    }
    let plan = if prefill {
        entry.prefill_plan(live)?
    } else {
        entry.plan(live)?
    };
    if partial.length < plan.workspace_bytes
        || semaphores.length < u64::from(entry.semaphore_words()) * 4
    {
        return Err(Refusal("Qwen3.8 GEMM workspace is too small"));
    }
    let mut bindings = BufferBindings::default();
    let mut buffers = qwen::Buffers {
        activation: bindings.bind(activation),
        partial: bindings.bind(partial),
        semaphores: bindings.bind(semaphores),
        ..qwen::Buffers::default()
    };
    for (index, weight) in weights.iter().enumerate() {
        buffers.weights[index] = bindings.bind(*weight);
    }
    for (index, (output, stride)) in outputs.iter().enumerate() {
        buffers.outputs[index] = bindings.bind(*output);
        buffers.output_strides[index] = *stride;
    }
    let arguments = if prefill {
        entry.prefill_arguments(live, buffers)?
    } else {
        entry.arguments(live, buffers)?
    };
    Ok(PreparedLaunch {
        symbol: entry.symbol.clone(),
        rows: live,
        plan,
        arguments,
        bindings,
    })
}

pub mod sparse {
    pub use kernels_sm120::attention::sparse::{Entry, Kind};

    /// The compiled QSA entries of every topology.
    pub fn entries() -> &'static [Entry] {
        static ENTRIES: std::sync::OnceLock<Vec<Entry>> = std::sync::OnceLock::new();
        ENTRIES.get_or_init(|| {
            kernels_sm120::attention::sparse::catalog(model_qwen38_contract::sm120::catalogs())
                .into_iter()
                .filter(|entry| super::qwen(entry.profile))
                .collect()
        })
    }
}

/// The linear-attention engine (`kernels_sm120::linear`).
pub mod linear {
    pub use kernels_sm120::linear::{Bucket, Entry};

    pub fn entries() -> &'static [Entry] {
        static ENTRIES: std::sync::OnceLock<Vec<Entry>> = std::sync::OnceLock::new();
        ENTRIES.get_or_init(|| {
            model_qwen38_contract::sm120::linear::catalog()
                .into_iter()
                .filter(|entry| super::qwen(entry.profile))
                .collect()
        })
    }

    /// The entry `chop_linear_{tag}_{name}_v1`.
    pub fn entry(tag: &str, name: &str) -> Result<&'static Entry, kernel_core::contract::Refusal> {
        let symbol = format!("chop_linear_{tag}_{name}_v1");
        entries()
            .iter()
            .find(|entry| entry.symbol == symbol)
            .ok_or(kernel_core::contract::Refusal(
                "unknown Qwen3.8 linear-attention entry",
            ))
    }
}

/// The linear-attention prefill workspace at capacity: every row of a
/// prefill phase and one partial chunk per item.
pub fn linear_prefill_workspace(shape: Shape, served: Served) -> Result<u64, Refusal> {
    let bucket = linear::Bucket {
        rows: PREFILL_TOKENS,
        sequences: crate::launch::prefill_sequences(served),
    };
    Ok(linear::entry(shape.tag(), "delta_prefill_prepare")?
        .plan(bucket, bucket.rows, bucket.sequences)?
        .workspace_bytes)
}

/// Decode buckets of the dense decode entries.
pub const BUCKETS: [u32; 6] = DECODE_ENTRY_BUCKETS;

/// The largest workspace any stage of a rank of `shape` serving `served`
/// asks for: the rank-invariant entries and the shape's own.
pub fn workspace_bytes(shape: Shape, served: Served) -> Result<u64, Refusal> {
    let profiles = [geometry_owner(shape).model.at(4), shape.profile()];
    let mut maximum = linear_prefill_workspace(shape, served)?;
    // Graph buckets above the decode entries' run the QSA decode entries at
    // up to the largest token bucket.
    let wide = (DECODE_ENTRY_ROWS.trailing_zeros() + 1..=served.rows.trailing_zeros())
        .map(|shift| 1_u32 << shift);
    let buckets = BUCKETS.into_iter().chain(wide).collect::<Vec<_>>();
    for entry in sparse::entries()
        .iter()
        .filter(|entry| profiles.contains(&entry.profile))
    {
        for &rows in buckets.iter().chain(&[PREFILL_TOKENS]) {
            if rows <= entry.capacity() {
                maximum = maximum.max(entry.workspace_bytes(rows));
            }
        }
    }
    for entry in dense::entries(&profiles) {
        if matches!(entry.key.query, StaticQuery::Gemm(_)) {
            maximum = maximum.max(kernels_sm120::gemm::capacity_workspace_bytes(&entry.key)?);
        }
    }
    for entry in moe::route_catalog()
        .into_iter()
        .filter(|entry| profiles.contains(&entry.profile))
    {
        maximum = maximum.max(
            entry
                .plan(entry.tokens_capacity, entry.tokens_capacity)?
                .workspace_bytes,
        );
    }
    for entry in qwen::entries()
        .iter()
        .filter(|entry| profiles.contains(&entry.profile))
    {
        maximum = maximum.max(entry.workspace_capacity()?);
    }
    Ok(maximum)
}

/// Dynamic shared memory a compiled entry requests at its largest plan.
pub fn dynamic_shared_bytes(symbol: &str) -> Result<u32, Refusal> {
    // The HC precompute producers derive from the contract's catalog entries.
    if let Some(bytes) = kernels_sm120::gemm::fused::hc_precompute_shared(
        model_qwen38_contract::sm120::catalogs(),
        symbol,
    ) {
        return Ok(bytes);
    }
    for entry in sparse::entries() {
        if entry.symbol() == symbol {
            return Ok(entry.dynamic_shared_bytes());
        }
    }
    for entry in linear::entries() {
        if entry.symbol == symbol {
            return Ok(entry.dynamic_shared_bytes());
        }
    }
    // The dense attention variants share the cubin with the gated rows;
    // the program does not launch them.
    for variant in kernels_sm120::attention::catalog(model_qwen38_contract::sm120::catalogs()) {
        if qwen(variant.profile)
            && let Some(bytes) = variant.dynamic_shared_bytes(symbol)
        {
            return Ok(bytes);
        }
    }
    for variant in moe::variants() {
        if variant.symbol == symbol {
            return variant.capacity_shared_bytes();
        }
    }
    for (_, entry) in dense::profiled() {
        if entry.symbol == symbol && matches!(entry.key.query, StaticQuery::Gemm(_)) {
            return kernels_sm120::gemm::capacity_shared_bytes(&entry.key);
        }
    }
    if let Some(entry) = qwen::entry(symbol) {
        return Ok(entry.plan(entry.query.m_capacity)?.dynamic_shared_bytes);
    }
    // The vision encoder's GEMMs and pair attention; its rows need none.
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        if let Some(entry) = crate::encoder::kernels_for(geometry)
            .map_err(|_| Refusal("the Qwen3.8 vision declaration is invalid"))?
            .iter()
            .find(|entry| entry.symbol == symbol)
        {
            return Ok(entry.dynamic_shared_bytes().unwrap_or(0));
        }
    }
    // The context-parallel head-sum selection, loaded only above size 1;
    // the family's other entries need none.
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        let select = model_qwen38_contract::sm120::context_parallel::select_entry(Shape::new(
            geometry,
            Ranks::TP4,
        ));
        if select.symbol() == symbol {
            return Ok(select.dynamic_shared_bytes());
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use bundle_runtime::geometry::Items;
    use kernel_build::EntryParam;

    use super::*;

    /// Every graph bucket serving launches (powers of two up to the served
    /// items) plans the entry the bundle picks for it at each rank count, in
    /// both norm-weight forms: the decode entry up to 32 rows, the wide
    /// entry where graphs take it, the general entry above.
    #[test]
    fn served_graph_buckets_plan_every_all_reduce_form() {
        use kernels_sm120::comm::hyper_connection_decode as hcd;
        let items = Items::new(bundle_runtime::geometry::MAX_ITEMS).unwrap();
        for shape in [Shape::TP2, Shape::TP4] {
            let ranks = shape.ranks();
            for weight in [DType::Bf16, DType::F32] {
                let owner = geometry_owner(shape);
                let decode = owner.decode_for(weight).unwrap();
                let transport = owner.transport_for(weight).unwrap();
                for rows in items.buckets() {
                    let plan = if rows <= DECODE_ENTRY_ROWS {
                        hcd::plan_with(decode.shape, rows)
                    } else if hcd::graph_takes_wide(ranks, rows).unwrap() {
                        hcd::wide_plan_with(decode.shape, rows)
                    } else {
                        kernels_sm120::comm::hyper_connection::prefill_plan(
                            &transport,
                            hc_query(shape),
                            rows,
                        )
                    };
                    let plan =
                        plan.unwrap_or_else(|e| panic!("TP{ranks} {weight:?} {rows}: {e:?}"));
                    plan.validate().unwrap();
                    assert_eq!(plan.grid[0] % plan.cluster[0], 0, "TP{ranks} {rows}");
                }
            }
        }
    }

    /// Every declared norm-weight form lies in the embedded cubin it names,
    /// which loads at its rank count, and only F32 forms are F32 cubins.
    #[test]
    fn declared_norm_weight_forms_are_embedded() {
        let forms = model_qwen38_contract::sm120::comm::norm_weight_forms();
        let embedded = |cubin: &str, symbol: &str, ranks: u32| {
            crate::CUBINS.iter().any(|c| {
                c.name == cubin
                    && c.symbols.contains(&symbol)
                    && crate::GEOMETRY_CUBINS
                        .iter()
                        .any(|(name, _, n)| *name == cubin && (*n == 0 || *n == ranks))
            })
        };
        for row in &forms.rows {
            for (form, f32) in [(&row.bf16, false), (&row.f32, true)] {
                assert!(embedded(&form.cubin, &form.symbol, 4), "{form:?}");
                assert_eq!(crate::F32_NORM_CUBINS.contains(&form.cubin.as_str()), f32);
            }
        }
        for t in &forms.transports {
            for f32 in [0, 1] {
                let prefill = t.prefill[f32];
                for symbol in [
                    prefill.symbol,
                    prefill.scatter_symbol,
                    prefill.gather_symbol,
                ] {
                    assert!(embedded(prefill.cubin, symbol, t.ranks), "{symbol}");
                }
                let decode = t.decode[f32];
                let (cubin, wide) = (&t.decode_cubins[f32], &t.wide_cubins[f32]);
                for symbol in [
                    kernels_sm120::comm::hyper_connection_decode::symbol_v2(&decode, t.ranks),
                    kernels_sm120::comm::hyper_connection_decode::symbol_v3(&decode, t.ranks),
                ] {
                    assert!(embedded(cubin, symbol.unwrap(), t.ranks), "{cubin}");
                }
                let symbol =
                    kernels_sm120::comm::hyper_connection_decode::wide_symbol(&decode, t.ranks);
                assert!(embedded(wide, symbol.unwrap(), t.ranks), "{wide}");
                for name in [prefill.cubin, cubin, wide] {
                    assert_eq!(crate::F32_NORM_CUBINS.contains(&name), f32 == 1, "{name}");
                }
            }
        }
    }

    /// The F32 norm-weight forms share the BF16 forms' comm state, so calls
    /// of both may follow each other on it, and each topology's F32 entries
    /// lie in cubins that load at its rank count.
    #[test]
    fn f32_norm_forms_share_the_comm_state_and_load_per_topology() {
        use kernels_sm120::comm::hyper_connection as hc;
        use kernels_sm120::comm::hyper_connection_decode as hcd;
        for shape in [Shape::TP2, Shape::TP4] {
            let owner = geometry_owner(shape);
            let (bf16, f32) = (
                owner.transport_for(DType::Bf16).unwrap(),
                owner.transport_for(DType::F32).unwrap(),
            );
            assert_eq!(bf16, owner.transport);
            assert_eq!(f32.shape.epilogue.weight_pointer_type(), "const float");
            let query = hc_query(shape);
            let state = |buffers| format!("{buffers:?}");
            assert_eq!(
                state(hc::buffers(&bf16, query).unwrap()),
                state(hc::buffers(&f32, query).unwrap())
            );
            let (bf16, f32) = (
                owner.decode_for(DType::Bf16).unwrap(),
                owner.decode_for(DType::F32).unwrap(),
            );
            let ranks = shape.ranks();
            let state = |buffers| format!("{buffers:?}");
            assert_eq!(
                state(hcd::buffers_with(bf16.shape, ranks).unwrap()),
                state(hcd::buffers_with(f32.shape, ranks).unwrap())
            );
            assert_eq!(
                state(hcd::wide_buffers_with(bf16.shape, ranks).unwrap()),
                state(hcd::wide_buffers_with(f32.shape, ranks).unwrap())
            );
            assert_eq!(
                hcd::low_latency_rows(bf16.shape),
                hcd::low_latency_rows(f32.shape)
            );
            let loaded = |symbol: &str| {
                crate::CUBINS
                    .iter()
                    .filter(|cubin| cubin.symbols.contains(&symbol))
                    .filter(|cubin| {
                        crate::F32_NORM_CUBINS.contains(&cubin.name)
                            && crate::GEOMETRY_CUBINS.iter().any(|(name, id, n)| {
                                *name == cubin.name
                                    && *id == shape.geometry.id
                                    && (*n == 0 || *n == ranks)
                            })
                    })
                    .count()
            };
            for symbol in [
                f32.symbols_v2[(ranks / 2 - 1) as usize],
                hcd::symbol_v3(&f32, ranks).unwrap(),
                hcd::wide_symbol(&f32, ranks).unwrap(),
                owner.f32_norms.symbol,
                owner.f32_norms.scatter_symbol,
                owner.f32_norms.gather_symbol,
                owner.f32_norms.row_symbol,
            ] {
                assert_eq!(loaded(symbol), 1, "TP{ranks} {symbol}");
            }
            // Without F32 norm weights the residency check is unchanged.
            let bf16_only = spinning_entries(shape, 256, false, None).unwrap();
            assert!(bf16_only.iter().all(|e| !e.symbol.contains("_f32_")));
            let both = spinning_entries(shape, 256, true, None).unwrap();
            assert!(
                both.iter()
                    .any(|e| e.symbol == hcd::wide_symbol(&f32, ranks).unwrap())
            );
        }
    }

    #[test]
    fn every_topology_names_its_vocabulary_heads() {
        for g in model_qwen38_contract::geometry::GEOMETRIES {
            for ranks in [Ranks::TP2, Ranks::TP4] {
                let shape = Shape::of_geometry(g, ranks).unwrap();
                assert!(vocabulary_head(&format!(
                    "chop_gemm_{}_head_c32_v1",
                    shape.tag()
                )));
            }
        }
    }

    fn assert_abi(params: &[EntryParam], args: &[Argument], symbol: &str) {
        assert_eq!(params.len(), args.len(), "{symbol}");
        for (param, arg) in params.iter().zip(args) {
            assert!(
                matches!(
                    (param, arg),
                    (
                        EntryParam::DevicePointer { .. },
                        Argument::DeviceBuffer(_) | Argument::NullDevicePointer
                    ) | (EntryParam::TensorMap { .. }, Argument::TensorMap(_))
                        | (EntryParam::U32 { .. }, Argument::U32(_))
                        | (EntryParam::U64 { .. }, Argument::U64(_))
                        | (EntryParam::I32 { .. }, Argument::I32(_))
                        | (
                            EntryParam::PointerArray { .. },
                            Argument::DevicePointerArray(_)
                        )
                ),
                "{symbol}"
            );
        }
    }

    const SHAPES: [Shape; 2] = Shape::ALL;

    #[test]
    fn decode_comm_selection_follows_prepared_rows() {
        use model_qwen38_contract::sm120::comm::decode as hc;
        for shape in SHAPES {
            for rows in [1, 2, 4, 8, 16, 32] {
                let ranks = shape.ranks();
                let call = hc::Call {
                    input: 1,
                    residual: 2,
                    logits: 3,
                    weight: 4,
                    residual_out: 5,
                    normed: 6,
                    inboxes: (10..10 + u64::from(ranks)).collect(),
                    flags: (20..20 + u64::from(ranks)).collect(),
                    epochs: 7,
                    live_rows: 8,
                    rows,
                    rank: 0,
                };
                let actual =
                    allreduce_hc_decode(shape, DType::Bf16, call, BufferBindings::default())
                        .unwrap();
                let expected = if rows <= hc::LL_MAX_ROWS {
                    hc::symbol_v3(ranks)
                } else {
                    hc::symbol_v2(ranks)
                }
                .unwrap();
                assert_eq!(
                    actual.symbol, expected,
                    "{ranks} ranks, {rows} prepared rows"
                );
            }
        }
    }

    #[test]
    fn scored_head_requires_exact_live_f32_output_extent() {
        let buffer = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |length| DeviceBufferSpanV1::new(buffer, 0, length);
        for shape in SHAPES {
            let symbol = format!("chop_gemm_{}_head_f32_prefill_v1", shape.tag());
            assert!(vocabulary_head(&symbol));
            let vocab = u64::from(shape.vocab_per_rank);
            for rows in [1, 512] {
                let needed = u64::from(rows) * vocab * 4;
                for bytes in [needed - 1, needed] {
                    let result = gemm(
                        &symbol,
                        rows,
                        span(vocab * 2560 * 2),
                        span(8192 * 2560 * 2),
                        span(bytes),
                        span(2 << 30),
                        span(4096),
                    );
                    assert_eq!(
                        result.is_ok(),
                        bytes == needed,
                        "{symbol} rows={rows} bytes={bytes}"
                    );
                }
            }
        }
    }

    #[test]
    fn expert_and_route_entries_cover_every_bucket() {
        for shape in SHAPES {
            let profile = shape.profile();
            for tokens in BUCKETS {
                for projection in [moe::Projection::GateUp, moe::Projection::Down] {
                    moe::variant(profile, projection, moe::Phase::Decode, tokens).unwrap();
                }
                moe::route(profile, &format!("decode_route_t{tokens}")).unwrap();
            }
            for projection in [moe::Projection::GateUp, moe::Projection::Down] {
                moe::variant(profile, projection, moe::Phase::Prefill, PREFILL_TOKENS).unwrap();
            }
            for operation in [
                "router",
                "permute_count",
                "permute_prefix",
                "permute_scan",
                "permute_scatter",
                "combine",
            ] {
                moe::route(profile, operation).unwrap();
            }
            let tag = shape.tag();
            for m in BUCKETS {
                dense::entry(&format!("chop_gemm_{tag}_shared_gate_up_m{m}_v1")).unwrap();
                dense::entry(&format!("chop_gemm_{tag}_shared_down_m{m}_v1")).unwrap();
            }
            for items in [1, 32, 33, 256] {
                let served = Served::new(Items::new(items).unwrap());
                assert!(
                    workspace_bytes(shape, served).unwrap()
                        >= linear_prefill_workspace(shape, served).unwrap()
                );
            }
        }
        // 128 chunks of 64 rows plus one partial chunk per item: 32 items
        // up to 32 served sequences, as before the served count.
        let served = |items| Served::new(Items::new(items).unwrap());
        assert_eq!(
            linear_prefill_workspace(Shape::TP4, served(8)).unwrap(),
            160 * 557_056
        );
        assert_eq!(
            linear_prefill_workspace(Shape::TP4, served(32)).unwrap(),
            160 * 557_056
        );
        assert_eq!(
            linear_prefill_workspace(Shape::TP4, served(200)).unwrap(),
            384 * 557_056
        );
        assert_eq!(
            workspace_bytes(Shape::TP4, served(8)).unwrap(),
            workspace_bytes(Shape::TP4, served(32)).unwrap()
        );
        assert_eq!(
            moe::route(Shape::TP2.profile(), "router").unwrap().symbol,
            "chop_moe_qwen38_router_v1"
        );
    }

    #[test]
    fn qwen_gemm_launches_match_their_compiled_abi() {
        let buffer = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |length| DeviceBufferSpanV1::new(buffer, 0, length);
        let cubins =
            kernels_sm120::gemm::fused::cubin_specs(model_qwen38_contract::sm120::catalogs(), None);
        for entry in qwen::entries() {
            let query = entry.query;
            let live = query.m_capacity.min(3);
            let weights = entry
                .segments
                .iter()
                .map(|rows| span(u64::from(rows.div_ceil(16) * 16 * query.k) * 2))
                .collect::<Vec<_>>();
            let outputs = if matches!(entry.kind, qwen::Kind::ClusterDecode { .. }) {
                entry.segments.clone()
            } else {
                vec![query.n]
            }
            .into_iter()
            .map(|columns| (span(u64::from(query.m_capacity * columns) * 4), columns))
            .collect::<Vec<_>>();
            let launch = qwen_gemm(
                entry,
                live,
                false,
                &weights,
                span(u64::from(query.m_capacity * query.k) * 2),
                &outputs,
                span(1 << 30),
                span(4096),
            )
            .unwrap();
            assert!(launch.plan.pdl);
            let compiled = cubins
                .iter()
                .flat_map(|cubin| &cubin.entries)
                .find(|item| item.symbol == entry.symbol)
                .unwrap();
            assert_abi(&compiled.params, &launch.arguments.0, &entry.symbol);
        }
        assert!(
            workspace_bytes(Shape::TP4, Served::new(Items::new(32).unwrap())).unwrap()
                >= qwen::entries()[0].workspace_capacity().unwrap()
        );
        // The fused up mix, as `Sm120Own` prepares it.
        use kernel_core::backend::{Counts, Need, View};
        use model_qwen38_contract::own_ops::OwnOps;
        use model_qwen38_contract::own_ops::hc::{HcUpMix, HcUpMixBuffers, UpMixInput};
        let up_mix = crate::sm120::own_ops::Sm120Own::new(Shape::TP4, false)
            .hc_up_mix(&HcUpMix::of(Shape::TP4, UpMixInput::Down), Need::rows(32))
            .unwrap();
        let view = |handle, bytes| View { handle, bytes };
        let buffers = HcUpMixBuffers {
            weight: view(1, 1 << 30),
            input: view(2, 1 << 30),
            normed: view(3, 1 << 30),
            output: view(4, 1 << 30),
            live_rows: view(5, 4),
        };
        let [launch] = &up_mix
            .launches(&buffers, View::EMPTY, Counts::rows(32))
            .unwrap()[..]
        else {
            panic!("one up mix launch");
        };
        assert!(launch.plan.pdl);
        let compiled = cubins
            .iter()
            .flat_map(|cubin| &cubin.entries)
            .find(|item| item.symbol == launch.symbol)
            .unwrap();
        assert_abi(&compiled.params, &launch.arguments.0, &launch.symbol);
    }

    #[test]
    fn row_gathers_fit_their_input_and_target_output() {
        let buffer = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |length| DeviceBufferSpanV1::new(buffer, 0, length);
        // Scored logits: 512 F32 rows move as pairs of BF16 values.
        for shape in SHAPES {
            let ranks = shape.ranks();
            let shard = 2 * u64::from(shape.vocab_per_rank);
            let input = 512 * shard * 2;
            let output = input * u64::from(ranks);
            let check = |input, output, rows| {
                check_gather_extents(span(input), span(output), rows, shard, ranks)
            };
            assert!(check(input, output, 512).is_ok());
            assert!(check(input - 1, output, 512).is_err());
            assert!(check(input, output - 1, 512).is_err());
            assert!(check(input, output, 513).is_err());
        }
    }

    #[test]
    fn dense_outputs_other_than_the_head_need_their_full_row_capacity() {
        let buffer = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |length| DeviceBufferSpanV1::new(buffer, 0, length);
        let profiles = SHAPES.map(Shape::profile);
        for entry in dense::entries(&profiles).filter(|entry| {
            !vocabulary_head(&entry.symbol)
                && matches!(entry.key.query, StaticQuery::Gemm(query) if query.m_capacity > 512)
        }) {
            let StaticQuery::Gemm(query) = entry.key.query else {
                panic!("GEMM query")
            };
            let rows = u64::from(query.n / (1 + u32::from(query.epilogues.swiglu.is_some())));
            let bytes = if query.output == DType::F32 { 4 } else { 2 };
            let rows = rows * bytes / 2;
            for (bytes, accepted) in [
                (u64::from(query.m_capacity) * rows * 2, true),
                (u64::from(query.m_capacity) * rows * 2 - 1, false),
                (512 * rows * 2, false),
            ] {
                let result = gemm(
                    &entry.symbol,
                    512,
                    span(u64::from(query.n) * u64::from(query.k) * 2),
                    span(u64::from(query.m_capacity) * u64::from(query.k) * 2),
                    span(bytes),
                    span(1 << 30),
                    span(4096),
                );
                assert_eq!(result.is_ok(), accepted, "{} bytes={bytes}", entry.symbol);
            }
        }
    }

    #[test]
    fn shared_expert_launches_match_their_compiled_abi() {
        let buffer = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |length| DeviceBufferSpanV1::new(buffer, 0, length);
        let cubins =
            kernels_sm120::gemm::cubin_specs(model_qwen38_contract::sm120::catalogs(), None);
        for (_, entry) in dense::profiled() {
            let StaticQuery::Gemm(query) = entry.key.query else {
                panic!("GEMM query")
            };
            let launch = gemm(
                &entry.symbol,
                query.m_capacity.min(3),
                span(u64::from(query.n * query.k) * 2),
                span(u64::from(query.m_capacity * query.k) * 2),
                span(
                    u64::from(query.m_capacity * query.n)
                        * if query.output == DType::F32 { 4 } else { 2 },
                ),
                span(1 << 30),
                span(4096),
            )
            .unwrap();
            assert!(launch.plan.pdl);
            let compiled = cubins
                .iter()
                .flat_map(|cubin| &cubin.entries)
                .find(|item| item.symbol == entry.symbol)
                .unwrap();
            assert_abi(&compiled.params, &launch.arguments.0, &entry.symbol);
        }
    }

    /// Create refuses a comm lane grid past what the GPU can hold at once
    /// (the phased pair runs 256 threads per CTA, in clusters of eight),
    /// whatever the residency check would report; the defaults (TP4's 80,
    /// TP2's planned grid) and smaller overrides are checked at their size.
    #[test]
    fn create_bounds_the_comm_lanes_grid() {
        use super::super::overlap::{MAX_COMM_CTAS, config_with};
        assert_eq!(MAX_COMM_CTAS, 1128);
        for (shape, ranks) in [(Shape::TP2, 2), (Shape::TP4, 4)] {
            let read = |ctas: Option<&str>| {
                config_with(ranks, |name| {
                    (name == "CHOPINFER_QWEN38_OVERLAP_COMM_CTAS")
                        .then(|| ctas.map(str::to_owned))
                        .flatten()
                })
                .unwrap()
            };
            let phased_grids = |config: &bundle_runtime::lanes::Config| {
                spinning_entries(shape, 256, false, Some(config)).map(|entries| {
                    entries
                        .iter()
                        .filter(|entry| phased_comm(shape, entry.symbol))
                        .map(|entry| entry.plan.grid[0])
                        .max()
                })
            };
            let planned = phased_grids(&read(Some("0"))).unwrap();
            let default = read(None);
            assert_eq!(default.comm_ctas, [None, Some(80)][ranks as usize / 2 - 1]);
            phased_grids(&default).unwrap();
            for ctas in ["64", "80", "1128"] {
                let checked = phased_grids(&read(Some(ctas))).unwrap();
                assert_eq!(checked, planned.max(ctas.parse().ok()), "TP{ranks} {ctas}");
            }
            for ctas in [MAX_COMM_CTAS + 8, 536_870_920, u32::MAX - 7] {
                let config = read(Some(&ctas.to_string()));
                assert_eq!(config.comm_ctas, Some(ctas));
                let refusal = phased_grids(&config).unwrap_err();
                assert!(
                    refusal.0.contains("resident"),
                    "TP{ranks} {ctas}: {refusal}"
                );
            }
        }
    }
}
