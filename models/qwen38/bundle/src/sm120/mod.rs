//! Qwen3.8's SM120 selection: the `kernels_sm120` entries an instance of
//! one topology registers with the SM120 backend. The ops themselves are
//! stated in model math by `crate::ops`, which prepares them through
//! `&dyn Backend` and never names an SM120 entry.

pub mod catalog;
#[cfg(test)]
pub mod device;
pub mod entries;
pub mod overlap;
pub mod own_ops;
pub(crate) mod projection;
mod target;

use kernel_core::backend::{Backend, Refused};
use kernel_core::contract::{AttentionPhase, DType, NgramQuery, RouterKind, StaticQuery};
use kernel_core::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery};
use kernel_core::properties::Invariance;
use kernels_sm120::attention::sparse::Kind;
use kernels_sm120::backend::{GatedSparseEntries, NamedOffers, Sm120};
use kernels_sm120::moe;
use model_qwen38_contract::sm120::config::EXACT_VERIFY_ROWS;
use model_qwen38_contract::sm120::context_parallel as cp;
use model_qwen38_contract::sm120::moe::use_parallel_route;

use crate::contract::{DECODE_ENTRY_ROWS, PREFILL_TOKENS, Shape};
use crate::ops::Key;
use crate::scratch::{CommBytes, Served};
use crate::target::{EncoderKernels, Uses};

/// The SM120 backend of an instance of `shape`: the rank-invariant entries
/// of its geometry (the TP4 profile's) and the entries of its own profile,
/// so no two registrations of one op differ only by profile. Without
/// `wide_comm`, graphs keep off the wide all-reduce entries.
pub(crate) fn backend(shape: Shape, wide_comm: bool) -> Box<dyn Backend> {
    let owner = catalog::geometry_owner(shape);
    let context_parallel = !cp::sizes(shape).is_empty();
    let shared = owner.model.at(4);
    let mut backend = Sm120::new();
    // The hyper-connection rows and the PLE gate are the TP4 profile's at
    // every rank count.
    for entry in model_qwen38_contract::sm120::row::hyper_connection_rows()
        .into_iter()
        .filter(|entry| entry.profile == shared)
        .filter(|entry| {
            matches!(
                entry.query,
                StaticQuery::HyperConnection(_) | StaticQuery::Ngram(NgramQuery::PleGate { .. })
            )
        })
    {
        backend = backend.with_hyper_connection_row(entry);
    }
    // The per-sequence n-gram hash gather of the rank's local heads, with
    // the end-of-document token its sources compile in.
    backend = backend.with_ngram(
        model_qwen38_contract::sm120::row::ngram(shape.profile()),
        shape.ngram_eos(),
    );
    // The grouped norm over F32 weights, which only an image that stores
    // F32 norm weights loads.
    if let Ok(norm) = owner.f32_norm_row() {
        backend = backend.with_grouped_norm(norm, owner.f32_norms.row_symbol);
    }
    // The shared stream broadcast, which the bundle's own cubin compiles.
    backend = backend.with_row_copy(kernels_sm120::row::copy::Entry::new(
        kernels_sm120::row::copy::Pattern::BroadcastStreams,
        kernels_sm120::row::copy::BROADCAST_SYMBOL,
    ));
    // The embedding row, which also gathers the prefill head's rows.
    let embedding = crate::launch::tagged(entries::EMBEDDING, shape);
    for entry in model_qwen38_contract::sm120::row::rows()
        .into_iter()
        .filter(|entry| entry.symbol == embedding)
    {
        backend = backend.with_row(entry);
    }
    // The linear-attention entries of the rank's own heads.
    backend = backend.with_linear(
        catalog::linear::entries()
            .iter()
            .filter(|entry| entry.profile == shape.profile())
            .cloned(),
    );
    // The gated rows and the attention of the rank's own heads, and the
    // replicated indexer, which is the TP4 profile's at every rank count.
    backend = backend.with_gated_sparse(GatedSparseEntries {
        gated: model_qwen38_contract::sm120::attention::gated_catalog(shape.profile()),
        sparse: catalog::sparse::entries()
            .iter()
            .filter(|entry| match entry.kind {
                Kind::Attention => entry.profile == shape.profile(),
                _ => entry.profile == shared,
            })
            .cloned()
            .collect(),
        // The selected partials of context-parallel groups.
        partials: if context_parallel {
            cp::partial_entries(shape)
        } else {
            Vec::new()
        },
    });
    // The router is the TP4 profile's at every rank count; the decode
    // routes, permutation and combine are of the rank's own experts. Of a
    // bucket's plain and parallel decode routes, which tie, the one Qwen
    // qualified (`use_parallel_route`).
    for entry in catalog::moe::route_catalog() {
        // A decode route selects and builds the metadata in one launch.
        if entry.parallel.is_some() || entry.symbol.contains("_decode_route_") {
            if entry.profile == shape.profile()
                && entry.parallel.is_some() == use_parallel_route(entry.tokens_capacity)
            {
                backend =
                    backend.with_decode_route(entry, RouterKind::SoftmaxTopK { renormalize: true });
            }
        } else if entry.profile == shape.profile()
            || (entry.profile == shared && entry.symbol.ends_with("_router_v1"))
        {
            backend = backend.with_route_entry(entry);
        }
    }
    // The grouped experts of the rank's own experts.
    for variant in catalog::moe::variants()
        .iter()
        .filter(|variant| variant.profile == shape.profile())
    {
        backend = backend.with_experts(variant.clone());
    }
    // The hyper-connection all-reduce's general and decode transports over
    // BF16 norm weights, with what Qwen3.8 declares for them
    // (`sm120::comm::declared`), and their F32 forms, which declare
    // nothing.
    for weight in [DType::Bf16, DType::F32] {
        let (general, decode) = if weight == DType::Bf16 {
            (Invariance::PREFILL, Invariance::one_row(EXACT_VERIFY_ROWS))
        } else {
            (Invariance::NONE, Invariance::NONE)
        };
        if let (Ok(transport), Ok(decode_transport)) =
            (owner.transport_for(weight), owner.decode_for(weight))
        {
            backend = backend
                .with_hc_all_reduce(transport, general)
                .with_hc_decode_all_reduce(decode_transport, decode, Invariance::NONE);
        }
    }
    if !wide_comm {
        backend = backend.without_wide_collectives();
    }
    // Context-parallel groups: the owner-routed writes, the localization
    // and partial merge, the per-rank selection and the block-id merge,
    // with what Qwen3.8 declares for them (`cp::declared`); transfers run
    // the shared group all-gather.
    if context_parallel {
        backend = backend
            .with_owned_writes(cp::owned_entries(shape))
            .with_context_parallel(cp::group_entries(shape))
            .with_head_sum_indexer([cp::select_entry(shape)])
            .with_block_merges([cp::merge_entry(shape)])
            .with_context_parallel_claims(
                cp::declared(shape)
                    .into_iter()
                    .map(|declared| (declared.symbol, declared.invariance)),
            );
    }
    // The generated projection uses of a precision component.
    backend = backend.with_projections(Box::new(NamedOffers(projection::Offers::new(shape))));
    Box::new(backend)
}

/// The comm state of a rank of `shape` serving `served`: the decode
/// all-reduce's is the decode entry's, extended for the wide entry when the
/// instance serves graph buckets above 32 rows; the zeroed LL region is the
/// decode entry's either way.
// The served decode rows are the HC decode entries' rows, above which a
// graph takes the wide state (`buffers_for_rows`).
const _: () = assert!(DECODE_ENTRY_ROWS == kernels_sm120::comm::hyper_connection_decode::MAX_ROWS);

pub(crate) fn comm_bytes(shape: Shape, served: Served) -> CommBytes {
    let owner = catalog::geometry_owner(shape);
    let hc =
        kernels_sm120::comm::hyper_connection::buffers(&owner.transport, catalog::hc_query(shape))
            .expect("catalog comm geometry");
    let decode = kernels_sm120::comm::hyper_connection_decode::buffers_for_rows(
        owner.decode.shape,
        shape.ranks(),
        served.rows,
    )
    .expect("decode comm geometry");
    CommBytes {
        hc_inbox: hc.inbox_bytes,
        hc_flags: hc.flags_bytes,
        hc_epoch: hc.epoch_bytes,
        decode_inbox: decode.inbox_bytes,
        decode_flags: decode.flags_bytes,
        decode_epochs: decode.epoch_bytes,
        decode_zeroed_inbox: decode.zeroed_inbox_bytes,
        gather_flags: kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES,
        gather_epoch: kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES,
    }
}

/// What the registered dense entry `symbol` needs beyond its views: a
/// Qwen-owned GEMM's workspace at capacity, or a shared GEMM family entry's
/// needs.
pub(crate) fn registered_dense(symbol: &str) -> Result<bundle_runtime::projection::Needs, String> {
    use bundle_runtime::projection::Needs;
    if let Some(entry) = catalog::qwen::entry(symbol) {
        Ok(Needs {
            workspace_bytes: entry.workspace_capacity().map_err(|e| e.to_string())?,
            ..Default::default()
        })
    } else {
        let entry = catalog::dense::entry(symbol).map_err(|e| e.to_string())?;
        dense_entry_needs(&entry.key).map_err(|e| e.to_string())
    }
}

/// The workspace and counter storage of a dense GEMM entry launched at up
/// to its full row capacity.
fn dense_entry_needs(
    key: &kernel_core::contract::CompileKey,
) -> Result<bundle_runtime::projection::Needs, kernel_core::contract::Refusal> {
    use kernels_sm120::gemm::{GemmBuffers, plan, row_counts};
    let StaticQuery::Gemm(query) = key.query else {
        return Err(kernel_core::contract::Refusal(
            "dense entry has no GEMM query",
        ));
    };
    Ok(bundle_runtime::projection::Needs {
        workspace_bytes: plan(key, row_counts(key, query.m_capacity))?.workspace_bytes,
        semaphore_bytes: GemmBuffers::semaphore_capacity_bytes(key)?,
        ..Default::default()
    })
}

/// The contract the registered fragment expert entry `symbol` computes, and
/// the bytes of its BF16 output at its routed capacity.
pub(crate) fn registered_experts(symbol: &str) -> Result<(GatedMlpProjectionQuery, u64), String> {
    let variant = catalog::moe::variants()
        .iter()
        .find(|v| v.symbol == symbol)
        .ok_or("missing registered fragment expert entry")?;
    let query = variant.query;
    let columns = query.n / (1 + u32::from(query.epilogues.swiglu.is_some()));
    Ok((
        kernels_sm120::backend::grouped_contract(variant),
        u64::from(query.m_capacity) * u64::from(columns) * 2,
    ))
}

/// The vision encoder of `shape`'s geometry on the SM120 backend over the
/// generated catalog: each slot runs the query its entry was generated
/// from.
pub(crate) fn encoder(shape: Shape) -> Result<EncoderKernels, String> {
    use bundle_runtime::encoder::SlotOp;
    use kernels_sm120::encoder::{Shared, Slot};
    let [vision] = crate::encoder::declarations_for(shape.geometry).map_err(|error| error.0)?;
    let slots = kernels_sm120::encoder::slots(vision, &Shared).map_err(|error| error.0)?;
    let catalog = crate::encoder::kernels_for(shape.geometry).map_err(|error| error.0)?;
    Ok(EncoderKernels {
        backend: Box::new(Sm120::new().with_encoders(catalog.iter().cloned())),
        slots: Box::new(move |stage, part, block| {
            let slot = Slot { stage, part, block };
            let (_, entry) = slots
                .iter()
                .find(|(known, _)| *known == slot)
                .ok_or_else(|| format!("the vision catalog has no slot {slot:?}"))?;
            Ok(SlotOp::Query(entry.query))
        }),
    })
}

/// Whether an instance that runs `uses` loads `cubin`: the wide decode
/// all-reduce's cubins only for graph buckets above the decode entries'
/// rows (`hyper_connection_decode::MAX_ROWS`), which the wide entry may
/// take, the HC producers only with their bottleneck, the context-parallel
/// family only in groups.
pub(crate) fn loads(cubin: &str, uses: Uses) -> bool {
    let wide_graphs = uses.graph_rows > kernels_sm120::comm::hyper_connection_decode::MAX_ROWS;
    (wide_graphs || !cubin.ends_with("_decode_wide"))
        && (uses.hc_precompute || cubin != kernels_sm120::gemm::fused::HC_PRECOMPUTE_CUBIN)
        && (uses.context_parallel || !cp::is_cubin(cubin))
}

/// Whether an instance that runs `uses` loads `symbol`: the padded up mix
/// only with its layout.
pub(crate) fn loads_symbol(symbol: &str, uses: Uses) -> bool {
    uses.hc_padded_mix || symbol != kernels_sm120::gemm::fused::HC_PADDED_MIX
}

/// The phase of the selected partials a context-parallel step of `key`
/// runs: a graph's decode or verify entry where it holds the bucket, else
/// the prefill entry.
pub(crate) fn partials_phase(shape: Shape, key: Key) -> AttentionPhase {
    let graph = match key {
        Key::Prefill => return AttentionPhase::Prefill,
        Key::Decode(_) => AttentionPhase::Decode,
        Key::Verify(_) => AttentionPhase::Verify,
    };
    let holds = model_qwen38_contract::sm120::context_parallel::partial_entries(shape)
        .iter()
        .any(|entry| entry.phase == graph && key.capacity() <= entry.capacity);
    if holds {
        graph
    } else {
        AttentionPhase::Prefill
    }
}

/// The contract of the grouped expert entry a step of `key` runs
/// `projection` through: a graph bucket's decode entry where the decode
/// entries hold it, else the prefill entry, which the wide buckets run at
/// their rows. The entry's query states its routed capacity, so Qwen3.8
/// asks for the contract of the entry it runs.
pub(crate) fn expert_contract(
    shape: Shape,
    projection: GatedMlpProjection,
    key: Key,
) -> Result<GatedMlpProjectionQuery, Refused> {
    let (phase, tokens) = match key {
        Key::Decode(bucket) | Key::Verify(bucket) if bucket <= DECODE_ENTRY_ROWS => {
            (moe::Phase::Decode, bucket)
        }
        _ => (moe::Phase::Prefill, PREFILL_TOKENS),
    };
    let projection = match projection {
        GatedMlpProjection::GateUp => moe::Projection::GateUp,
        GatedMlpProjection::Down => moe::Projection::Down,
    };
    let variant = catalog::moe::variant(shape.profile(), projection, phase, tokens)?;
    Ok(kernels_sm120::backend::grouped_contract(variant))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_entry_needs_cover_the_full_capacity_plan() {
        use bundle_runtime::projection::Needs;
        use kernel_core::contract::{DenseKernel, RuntimeCounts, Target};
        use kernels_sm120::gemm;
        let spec = gemm::instantiate(gemm::DenseRequest {
            query: gemm::bf16_query(8, 256, 1024, DType::Bf16),
            kernel: DenseKernel::Tiled { splits: 2 },
            target: Target::Sm120f,
            symbol: "projection_split_v1".into(),
            knobs: None,
        })
        .unwrap();
        let plan = gemm::plan(
            &spec.key,
            RuntimeCounts {
                m: 8,
                batch: 1,
                context: 1,
                routed_rows: 0,
                split_k: 2,
            },
        )
        .unwrap();
        let needs = dense_entry_needs(&spec.key).unwrap();
        assert_eq!(
            needs,
            Needs {
                workspace_bytes: plan.workspace_bytes,
                semaphore_bytes: gemm::GemmBuffers::semaphore_capacity_bytes(&spec.key).unwrap(),
                ..Needs::default()
            }
        );
        assert!(needs.workspace_bytes > 0 && needs.semaphore_bytes > 0);
        let mut key = spec.key;
        key.query = StaticQuery::Row {
            family: "fixture",
            width_capacity: 1,
            dtype: DType::Bf16,
            traits_digest: 0,
        };
        assert!(dense_entry_needs(&key).is_err());
    }

    /// A graph's all-reduce binds the state its prepared op states: the
    /// decode state on the decode entries, the general state where a backend
    /// without decode transports prepares it on the general one.
    #[test]
    fn graph_all_reduces_bind_the_state_their_op_states() {
        use kernel_core::backend::Need;
        use kernel_core::precision::{ExecutionPhase, OperationUse};
        let shape = Shape::TP4;
        let served = Served::new(crate::contract::Items::new(32).unwrap());
        let comm = comm_bytes(shape, served);
        let general = Sm120::new().with_hc_all_reduce(
            catalog::geometry_owner(shape).transport,
            Invariance::PREFILL,
        );
        let prepare = |backend: &dyn Backend, rows| {
            let usage = OperationUse {
                phase: ExecutionPhase::Decode,
                capacity: rows,
                graph: true,
            };
            backend
                .hc_all_reduce(
                    &crate::ops::hc_reduce(shape),
                    DType::Bf16,
                    DType::Bf16,
                    usage,
                    0,
                    Need::rows(rows),
                )
                .unwrap()
        };
        let full = backend(shape, true);
        for rows in [1, 8, 32] {
            let decode = prepare(full.as_ref(), rows);
            assert!(
                crate::ops::state_binding([&decode], &comm).unwrap(),
                "{rows}"
            );
            let fallback = prepare(&general, rows);
            assert!(
                !crate::ops::state_binding([&fallback], &comm).unwrap(),
                "{rows}"
            );
            assert!(crate::ops::state_binding([&decode, &fallback], &comm).is_err());
        }
    }

    #[test]
    fn the_wide_all_reduce_loads_only_for_wide_graphs() {
        let uses = |wide_graphs| Uses {
            graph_rows: if wide_graphs { 64 } else { 32 },
            hc_precompute: false,
            hc_padded_mix: false,
            context_parallel: false,
        };
        assert!(loads("comm_qwen38_tp2_decode_wide", uses(true)));
        assert!(!loads("comm_qwen38_tp2_decode_wide", uses(false)));
        assert!(loads("comm_qwen38_tp2_decode", uses(false)));
    }

    /// The registered lookups the contract's backend states are the row
    /// entries the bundle's catalog compiles, which `legacy_lookup` checks
    /// its prepared ops against: the embedding row at the geometry's tag,
    /// the n-gram rows at the topology's.
    #[test]
    fn registered_lookups_name_the_bundle_row_entries() {
        use kernel_core::precision::ExecutionPhase;
        use model_qwen38_contract::precision::{legacy_lookup_capability, uses};
        let mut checked = 0;
        for geometry in model_qwen38_contract::geometry::GEOMETRIES.iter() {
            for topology in model_qwen38_contract::contract::TOPOLOGIES {
                let shape = Shape::new(geometry, topology.ranks);
                let per_rank = |name| crate::launch::per_rank(name, shape);
                for usage in uses().unwrap() {
                    let ngram = match usage.phase {
                        ExecutionPhase::Prefill => per_rank(entries::NGRAM),
                        ExecutionPhase::Decode => {
                            crate::launch::tagged(&per_rank(entries::NGRAM_DECODE), shape)
                        }
                        ExecutionPhase::Verify => {
                            per_rank(model_qwen38_contract::sm120::owned::NGRAM_VERIFY)
                        }
                    };
                    let embedding = crate::launch::tagged(entries::EMBEDDING, shape);
                    for (lookup, expected) in [(false, embedding), (true, ngram)] {
                        let capability = legacy_lookup_capability(shape, lookup, usage).unwrap();
                        assert_eq!(capability.stages, [expected], "{} {usage:?}", shape.tag());
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0);
    }
}
