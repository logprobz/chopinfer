//! The shared kernels' offers for glm5next's dense projections
//! (`kernels_sm120::precision::generated`): per declared projection
//! (`crate::precision`), weight format and use, its rank-local operation
//! over BF16 activations and the offer the shared generator gives it in the
//! kernel `tiled_kernel` chooses (at the balanced decoded-tile knobs for
//! the FP8 prefill projections `decoded_prefill` lists, else the default
//! knobs), the layout copies after them and the cubins that hold them.

use kernel_build::CubinSpec;
use kernel_core::contract::{ActivationFormat, DenseKernel, InstantiationSpec, WeightFormat};
use kernel_core::precision::capability::OperationCapability;
use kernel_core::precision::format::{A16Recipe, registry};
use kernel_core::precision::{
    ActivationSpec, ArithmeticSpec, ExecutionPhase, OperationSpec, OperationUse, Rounding,
};
use kernels_sm120::precision::Compute;
use kernels_sm120::precision::generated::{self as shared, GeneratedOffer};
use kernels_sm120::row::precision::{self as row, LayoutCopy};
use projection_spec::derive::{Topology, rank_selection};

use crate::backend::{Kernels, Selected};
use crate::contract::Shape;
use crate::precision::{checkpoint_formats, declared, uses};

/// The rank-local operation of projection `id` over `weight` at `usage`:
/// rank 0's selection of the declared matrix padded to the format's tile,
/// the declared composition's recipe for the format, BF16 activations and
/// F32 accumulation.
pub fn domain(
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<OperationSpec, String> {
    let (_, geometry, projection) = declared(id)?;
    let (_, output) = uses(id)?
        .into_iter()
        .find(|(used, _)| *used == usage)
        .ok_or_else(|| format!("projection {id} declares no use {usage:?}"))?;
    let plugin = registry::by_format(weight).map_err(|e| e.to_string())?;
    let recipe = projection
        .composition
        .as_ref()
        .and_then(|composition| composition.numerical_contracts.get(&plugin.id()));
    let (numerical_contract, a16) = match recipe {
        Some(recipe) => (recipe.numerical_contract.clone(), recipe.a16_recipe),
        None if weight == WeightFormat::Bf16 => (
            projection.numerical_contract.clone(),
            A16Recipe::ScaledWeight,
        ),
        None => return Err(format!("projection {id} composes no {weight:?} recipe")),
    };
    let selection = rank_selection(
        projection,
        geometry,
        Topology {
            ranks: Shape::TP4.ranks(),
            rank: 0,
        },
        0,
    )
    .map_err(|e| e.0)?;
    let family = projection.family();
    let operands = u8::try_from(projection.operand_count()).map_err(|e| e.to_string())?;
    // The derivation pads the rank's rows and K to the format's tile (BF16
    // dense rows to 64), as the packed weights store them.
    let padding = plugin.padding_tile(family);
    let padded = kernel_core::precision::MatrixGeometry {
        rows: selection
            .physical
            .rows
            .next_multiple_of(padding.rows as u64),
        k: selection.physical.k.next_multiple_of(padding.k as u64),
    };
    shared::domain(shared::Domain {
        kind: projection
            .operation_kind(geometry, &selection)
            .map_err(|e| e.0)?,
        target: Selected::TARGET,
        geometry: padded,
        operands,
        weight,
        local_scale: None,
        // NVFP4 rows carry one F32 global per tensor (`weight_scale_2`),
        // which each rank holds whole.
        global_scale: (weight == WeightFormat::Nvfp4)
            .then_some(kernel_core::precision::ScaleScope::Tensor),
        activations: vec![ActivationSpec::bf16(); usize::from(operands)],
        arithmetic: ArithmeticSpec {
            weight_operand: kernel_core::contract::DType::Bf16,
            accumulator: kernel_core::contract::DType::F32,
            output,
            decode_order: plugin
                .decode_order(family, ActivationFormat::Bf16, a16)
                .map_err(|e| e.to_string())?,
            rounding: Rounding::NearestEven,
            numerical_contract,
        },
        epilogues: projection.epilogues,
        usage,
    })
}

/// The capacity projection `id`'s kernel compiles for at `usage`, as its
/// use profile declares it: a decode or verify bucket of at most 32 rows
/// at its power of two, else the prefill chunk (as the shared default
/// schedule sizes it); the head's uses at their own rows, which its logits
/// buffers hold.
fn compiled_capacity(id: &str, usage: OperationUse) -> Result<u32, String> {
    let (declaration, geometry, projection) = declared(id)?;
    let profile = match &projection.use_profile {
        Some(projection_spec::profiles::UseProfileRef::Id(profile)) => profile,
        _ => return Err(format!("projection {id} names no use profile")),
    };
    let used = declaration
        .use_profiles
        .get(profile)
        .ok_or_else(|| format!("projection {id}: no use profile {profile}"))?
        .iter()
        .find(|used| {
            used.phase == usage.phase
                && used.graph == usage.graph
                && used.capacity.resolve(geometry).ok() == Some(u64::from(usage.capacity))
        })
        .ok_or_else(|| format!("projection {id} declares no use {usage:?}"))?;
    let contract = used
        .contract
        .as_ref()
        .ok_or_else(|| format!("projection {id}: a use without its contract"))?;
    u32::try_from(
        contract
            .compiled_capacity
            .resolve(geometry)
            .map_err(|e| e.0)?,
    )
    .map_err(|_| format!("projection {id}: compiled capacity exceeds u32"))
}

/// The shared kernels' offer for projection `id` over `weight` at
/// `usage`, under the generated symbols: the tiled GEMM, which reads the
/// rank image's checkpoint rows (`crate::direct`, `checkpoint_rows_v1`)
/// in every phase. The fragment kernels would need the image to store
/// their weight layout.
fn generate(id: &str, weight: WeightFormat, usage: OperationUse) -> Result<GeneratedOffer, String> {
    let operation = domain(id, weight, usage)?;
    let capacity = compiled_capacity(id, usage)?;
    let kernel = tiled_kernel(&operation, weight, capacity)?;
    shared::offer(
        shared::Request {
            operation: &operation,
            format: registry::by_format(weight).map_err(|e| e.to_string())?.id(),
            usage,
            implementation: shared::Implementation::Dense {
                kernel,
                capacity,
                knobs: if decoded_tiles(&operation, weight, kernel, usage, capacity)? {
                    Some(kernels_sm120::gemm::balanced_decoded_tile_knobs())
                } else if staged_tiles(&operation, weight, kernel, usage, capacity)? {
                    Some(kernels_sm120::gemm::staged_tile_knobs())
                } else {
                    None
                },
            },
        },
        &[],
    )
}

/// The FP8 128x128 prefill projections (`rows` x `k`) whose decoded weight
/// tiles (`gemm::decoded_tile_knobs`) kernel-lab measured faster than their
/// warp tiles at 2,048 and 8,192 rows (PERF-B18, `decoded_tile_latency`).
/// They run the balanced decoded tiles (`gemm::balanced_decoded_tile_knobs`,
/// the final partial wave split into 64-row halves when they fit the SMs),
/// which kernel-lab measured as fast as the decoded tiles at 64 to 8,192
/// rows and faster where the halves fit: 1.5 times at 512 to 2,048 rows for
/// the latent key-value input, 1.13 times at 2,048 for the query's low-rank
/// input, 1.13 to 1.15 times at 1,040 for the 4096-row ones (PERF-B23,
/// `balanced_decoded_latency`).
fn decoded_prefill(rows: u64, k: u64) -> bool {
    matches!(
        (rows, k),
        // The latent output (F32), the latent query's low-rank input, the
        // latent key-value input, the latent query's up projection, the
        // dense MLPs' down projection (F32) and the shared expert's down
        // projection (F32 onto the residual).
        (4096, 4096) | (1536, 4096) | (512, 4096) | (4096, 1536) | (4096, 3072) | (4096, 512)
    )
}

/// Whether `operation`'s `usage`, compiled at `capacity` rows, runs
/// balanced decoded weight tiles: a prefill use (more than 32 rows) of a
/// [`decoded_prefill`] shape over FP8 128x128 weights in
/// `DenseKernel::TiledChains`, whose bits they keep (every element's 128-K
/// chains and scale FMAs are the warp tiles'). Decode and verify uses keep
/// the warp tiles, including the verify buckets compiled at the prefill
/// capacity.
fn decoded_tiles(
    operation: &OperationSpec,
    weight: WeightFormat,
    kernel: DenseKernel,
    usage: OperationUse,
    capacity: u32,
) -> Result<bool, String> {
    if usage.phase != ExecutionPhase::Prefill
        || capacity <= 32
        || !matches!(kernel, DenseKernel::TiledChains { .. })
        || !matches!(weight, WeightFormat::Fp8Block128x128(_))
    {
        return Ok(false);
    }
    let geometry = operation.matrix_geometry().map_err(|e| e.to_string())?;
    let rows = geometry.rows * operation.weights.len() as u64;
    Ok(operation.epilogues.swiglu.is_none() && decoded_prefill(rows, geometry.k))
}

/// The BF16 prefill projections (`rows` x `k`) whose staged weight tiles
/// (`gemm::staged_tile_knobs`) kernel-lab measured faster than their warp
/// tiles at 960 to 8,192 rows (PERF-B23, `staged_tile_latency`).
fn staged_prefill(rows: u64, k: u64) -> bool {
    matches!(
        (rows, k),
        // The linear attention q, k and v; its output (F32); the gate and
        // forget gates' up projections; the indexer's query; the drafter's
        // input projection.
        (2048, 4096) | (4096, 2048) | (2048, 128) | (4096, 1536) | (4096, 8192)
    )
}

/// Whether `operation`'s `usage`, compiled at `capacity` rows, runs staged
/// weight tiles: a prefill use (more than 32 rows) of a [`staged_prefill`]
/// shape over BF16 weights in `DenseKernel::TiledChains`, whose bits they
/// keep (every element's 64-K chains and K-order sums are the warp tiles').
/// Decode and verify uses keep their kernels, including the verify buckets
/// compiled at the prefill capacity.
fn staged_tiles(
    operation: &OperationSpec,
    weight: WeightFormat,
    kernel: DenseKernel,
    usage: OperationUse,
    capacity: u32,
) -> Result<bool, String> {
    if usage.phase != ExecutionPhase::Prefill
        || capacity <= 32
        || !matches!(kernel, DenseKernel::TiledChains { .. })
        || weight != WeightFormat::Bf16
    {
        return Ok(false);
    }
    let geometry = operation.matrix_geometry().map_err(|e| e.to_string())?;
    let rows = geometry.rows * operation.weights.len() as u64;
    Ok(operation.epilogues.swiglu.is_none() && staged_prefill(rows, geometry.k))
}

/// The strip-chain schedule (`DenseKernel::StripChains`: weight strips
/// per CTA, K splits) of a BF16 decode or verify bucket of `capacity` rows
/// over a `rows` x `k` weight without SwiGLU, where kernel-lab's sweep
/// (family `strip_chains`) measured one faster than the tiled entry: per
/// shape and bucket the fastest candidate, bucket 2 taking bucket 1's.
fn strip_schedule(rows: u64, k: u64, capacity: u32) -> Option<(u8, u8)> {
    match (rows, k) {
        // KDA q, k and v.
        (2048, 4096) => Some((2, 1)),
        // KDA output.
        (4096, 2048) if matches!(capacity, 8 | 16) => Some((1, 1)),
        (4096, 2048) => Some((4, 1)),
        // KDA gate and forget low-rank inputs, the indexer's key.
        (128, 4096) if capacity <= 8 => Some((1, 23)),
        (128, 4096) => Some((1, 47)),
        // KDA beta, the indexer's head weights.
        (64, 4096) => Some((1, 64)),
        // The router's F32 logits.
        (320, 4096) => Some((1, 18)),
        // The vocabulary head's F32 logits shard (the target's and the
        // drafter's): four strips at one split measured fastest at 1, 4, 8
        // and 16 rows (PERF-B14b lab run 20261009T075837Z-665316).
        (38720, 4096) => Some((4, 1)),
        _ => None,
    }
}

/// [`strip_schedule`] over FP8 128x128 weights (kernel-lab family
/// `strip_chains_fp8`, 128-K chains with the block scale FMA).
fn fp8_strip_schedule(rows: u64, k: u64, capacity: u32) -> Option<(u8, u8)> {
    match (rows, k) {
        // The latent output (F32).
        (4096, 4096) if capacity <= 4 => Some((4, 1)),
        (4096, 4096) if capacity == 8 => Some((2, 2)),
        (4096, 4096) => Some((1, 1)),
        // The latent query's low-rank input.
        (1536, 4096) if capacity <= 8 => Some((2, 7)),
        (1536, 4096) => Some((1, 3)),
        // The latent key-value input.
        (512, 4096) => Some((1, 11)),
        // The latent query's up projection.
        (4096, 1536) if capacity <= 8 => Some((2, 1)),
        (4096, 1536) => Some((1, 1)),
        // The dense MLPs' down projection (F32).
        (4096, 3072) if capacity <= 4 => Some((4, 1)),
        (4096, 3072) if capacity == 8 => Some((2, 1)),
        (4096, 3072) => Some((1, 1)),
        // The shared expert's down projection (F32 onto the residual).
        (4096, 512) if capacity <= 2 => Some((4, 1)),
        (4096, 512) if capacity <= 8 => Some((2, 1)),
        (4096, 512) => Some((1, 1)),
        _ => None,
    }
}

/// The tiled kernel of `operation` at `capacity` rows, bitwise the
/// one-split tiled GEMM whichever it is (`DenseKernel::TiledChains` and
/// `StripChains` keep every MMA chain and add them in K order): over BF16
/// or FP8 128x128 weights, a prefill use runs one split in warp tiles; a
/// decode or verify bucket of a shape [`strip_schedule`] (BF16) or
/// [`fp8_strip_schedule`] (FP8) lists runs its strip chains; another
/// decode or verify bucket whose tiles leave most SMs idle spreads its K
/// chunks over them (`gemm::chain_splits`); NVFP4 weights and decode tiles
/// that nearly fill the SMs run the one-split tiled GEMM itself.
fn tiled_kernel(
    operation: &OperationSpec,
    weight: WeightFormat,
    capacity: u32,
) -> Result<DenseKernel, String> {
    if !matches!(
        weight,
        WeightFormat::Bf16 | WeightFormat::Fp8Block128x128(_)
    ) {
        return Ok(DenseKernel::Tiled { splits: 1 });
    }
    if capacity > 32 {
        return Ok(DenseKernel::TiledChains { splits: 1 });
    }
    let geometry = operation.matrix_geometry().map_err(|e| e.to_string())?;
    let rows = geometry.rows * operation.weights.len() as u64;
    let strips = match weight {
        WeightFormat::Bf16 => strip_schedule(rows, geometry.k, capacity),
        _ => fp8_strip_schedule(rows, geometry.k, capacity),
    };
    if operation.epilogues.swiglu.is_none()
        && let Some((strips, splits)) = strips
    {
        return Ok(DenseKernel::StripChains { strips, splits });
    }
    let mut query = kernels_sm120::gemm::bf16_query(
        capacity,
        u32::try_from(rows).map_err(|e| e.to_string())?,
        u32::try_from(geometry.k).map_err(|e| e.to_string())?,
        operation.arithmetic.output,
    );
    query.weight = weight;
    query.epilogues = operation.epilogues;
    // Tiles that need fewer than four splits to fill the SMs (n4096, the
    // head) measured slower with a chain workspace.
    Ok(match kernels_sm120::gemm::chain_splits(&query) {
        1 => DenseKernel::Tiled { splits: 1 },
        splits => DenseKernel::TiledChains { splits },
    })
}

/// [`generate`]d once per process for each (projection, format, use).
pub fn offer(
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<GeneratedOffer, String> {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (String, String, String);
    static OFFERS: OnceLock<Mutex<BTreeMap<Key, GeneratedOffer>>> = OnceLock::new();
    let key = (id.to_owned(), format!("{weight:?}"), format!("{usage:?}"));
    let offers = OFFERS.get_or_init(Mutex::default);
    if let Some(offer) = offers.lock().map_err(|e| e.to_string())?.get(&key) {
        return Ok(offer.clone());
    }
    let offer = generate(id, weight, usage)?;
    offers
        .lock()
        .map_err(|e| e.to_string())?
        .insert(key, offer.clone());
    Ok(offer)
}

/// The KDA projections whose rows fill a third of the convolution's
/// `[rows, 3 * width]` input (`KdaQkv`), and which third.
const KDA_THIRDS: [(&str, u32); 3] = [("linear.q", 0), ("linear.k", 1), ("linear.v", 2)];

/// The rows of projection `id`'s operation on a rank before the format
/// pads them (`domain`).
fn rank_rows(id: &str) -> Result<u64, String> {
    let (_, geometry, projection) = declared(id)?;
    Ok(rank_selection(
        projection,
        geometry,
        Topology {
            ranks: Shape::TP4.ranks(),
            rank: 0,
        },
        0,
    )
    .map_err(|e| e.0)?
    .physical
    .rows)
}

/// The exact copy of projection `id`'s rows over `weight` at `usage` into
/// the buffer its consumer reads, after the GEMM writes its padded rows
/// contiguous: linear.q, linear.k and linear.v into their third of the KDA
/// convolution's `[rows, 3 * width]` input, and a BF16 projection whose
/// rows the format pads (linear.beta's 16 of 64, the indexer's head
/// weights' 32 of 64) compacted to its rank's rows. `None` where the GEMM
/// writes the consumer's buffer itself, and for the router's padded F32
/// scores, which the BF16 copy does not move.
pub fn copy(
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<Option<LayoutCopy>, String> {
    let operation = domain(id, weight, usage)?;
    let padded = operation
        .geometry
        .ok_or_else(|| format!("projection {id} has no geometry"))?
        .rows;
    let padded = u32::try_from(padded).map_err(|e| e.to_string())?;
    let rows = u32::try_from(rank_rows(id)?).map_err(|e| e.to_string())?;
    let destination = match KDA_THIRDS.iter().find(|(kda, _)| *kda == id) {
        Some((_, third)) => row::CopyRows {
            row_stride: 3 * rows,
            column_stride: 1,
            column_offset: third * rows,
        },
        None if rows < padded
            && operation.arithmetic.output == kernel_core::contract::DType::Bf16 =>
        {
            row::CopyRows {
                row_stride: rows,
                column_stride: 1,
                column_offset: 0,
            }
        }
        None => return Ok(None),
    };
    let (copy, _) = shared::layout_copy(
        rows,
        compiled_capacity(id, usage)?,
        row::CopyRows {
            row_stride: padded,
            column_stride: 1,
            column_offset: 0,
        },
        destination,
        row::ShapeBinding::Compiled,
        &[],
    )?;
    Ok(Some(copy))
}

/// The exact copies that place the MTP drafter's two normalized `[rows,
/// hidden]` rows side by side as the `[rows, 2 * hidden]` input its input
/// projection (`mtp.eh`) reads at `usage`, `cat[enorm(embedding),
/// hnorm(hidden)]`: the embedding's half at column 0, the hidden row's at
/// `hidden`. The shared RMSNorm writes contiguous rows only. Each copy
/// runs at the projection's compiled capacity for the use.
pub fn eh_halves(usage: OperationUse) -> Result<[LayoutCopy; 2], String> {
    let hidden = Shape::TP4.profile.hidden;
    let capacity = compiled_capacity("mtp.eh", usage)?;
    let half = |offset| {
        shared::layout_copy(
            hidden,
            capacity,
            row::CopyRows {
                row_stride: hidden,
                column_stride: 1,
                column_offset: 0,
            },
            row::CopyRows {
                row_stride: 2 * hidden,
                column_stride: 1,
                column_offset: offset,
            },
            row::ShapeBinding::Compiled,
            &[],
        )
        .map(|(copy, _)| copy)
    };
    Ok([half(0)?, half(hidden)?])
}

/// The portable capability of projection `id`'s use: its offer's, then its
/// copy's stage where it has one (`copy`).
pub fn capability(
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<OperationCapability, String> {
    let mut capability = offer(id, weight, usage)?
        .offer
        .capability()
        .map_err(|e| e.to_string())?;
    if let Some(copy) = copy(id, weight, usage)? {
        capability.stages.push(copy.stage().entry().symbol.clone());
    }
    Ok(capability)
}

/// Every distinct copy use of the lowered projections and of the
/// drafter's input halves (`eh_halves`), in symbol then capacity order.
/// Uses of one copy at several capacities share its entry, so cubins take
/// each symbol once.
pub fn copies(shape: Shape) -> Result<Vec<LayoutCopy>, String> {
    let mut copies = std::collections::BTreeMap::new();
    for (id, weight) in checkpoint_formats(shape)? {
        for (usage, _) in uses(id)? {
            let mut used: Vec<_> = copy(id, weight, usage)?.into_iter().collect();
            if id == "mtp.eh" {
                used.extend(eh_halves(usage)?);
            }
            for copy in used {
                copies.insert(
                    (copy.stage().entry().symbol.clone(), copy.stage().capacity()),
                    copy,
                );
            }
        }
    }
    Ok(copies.into_values().collect())
}

/// Every lowered projection's offers at its checkpoint format and every
/// declared use, as (id, use, offer).
pub fn offers(shape: Shape) -> Result<Vec<(&'static str, OperationUse, GeneratedOffer)>, String> {
    let mut offers = Vec::new();
    for (id, weight) in checkpoint_formats(shape)? {
        for (usage, _) in uses(id)? {
            offers.push((id, usage, offer(id, weight, usage)?));
        }
    }
    Ok(offers)
}

/// Every distinct GEMM entry of the lowered projections' offers, in
/// symbol order. The indexer's pooling gate, a BF16 `[indexer_dim,
/// hidden]` side tensor, runs the indexer key projection's entries.
pub fn instantiations(shape: Shape) -> Result<Vec<InstantiationSpec>, String> {
    let mut instantiations = std::collections::BTreeMap::new();
    for (id, _, generated) in offers(shape)? {
        match &generated.offer.handle().compute {
            Compute::Dense(spec) => {
                instantiations.insert(spec.symbol.clone(), spec.clone());
            }
            _ => return Err(format!("projection {id} is not a dense GEMM")),
        }
    }
    Ok(instantiations.into_values().collect())
}

/// The lock family of the projections' cubins (`bundle/kernels.lock.d`).
pub const FAMILY: &str = "projections";
/// The projections' GEMM cubin.
pub const CUBIN: &str = "projections_glm5next_sm120";
/// The projections' layout copies' cubin.
pub const COPIES_CUBIN: &str = "projection_copies_glm5next_sm120";
/// The lock family of the projections' decoded-tile cubin.
pub const DECODED_FAMILY: &str = "projections_decoded";
/// The projections' decoded-tile GEMM cubin: the prefill entries at
/// `gemm::balanced_decoded_tile_knobs` ([`decoded_tiles`]), apart from
/// [`CUBIN`] so that its bytes and every other entry's stay as they were.
pub const DECODED_CUBIN: &str = "projections_decoded_glm5next_sm120";

/// Whether an instantiation runs (balanced) decoded weight tiles: its
/// knobs over FP8 128x128 weights (BF16 warp tiles take the same knobs by
/// default).
fn decoded(spec: &InstantiationSpec) -> bool {
    spec.key.knobs == kernels_sm120::gemm::balanced_decoded_tile_knobs()
        && matches!(
            spec.key.query,
            kernel_core::contract::StaticQuery::Gemm(query)
                if matches!(query.weight, WeightFormat::Fp8Block128x128(_))
        )
}

/// The lock family of the projections' staged-tile cubin.
pub const STAGED_FAMILY: &str = "projections_staged";
/// The projections' staged-tile GEMM cubin: the BF16 prefill entries at
/// `gemm::staged_tile_knobs` ([`staged_tiles`]), apart from [`CUBIN`] so
/// that its bytes and every other entry's stay as they were.
pub const STAGED_CUBIN: &str = "projections_staged_glm5next_sm120";

/// Whether an instantiation runs staged weight tiles: its knobs over BF16
/// weights (FP8 128x128 weights at the same knobs run decoded tiles).
fn staged(spec: &InstantiationSpec) -> bool {
    spec.key.knobs == kernels_sm120::gemm::staged_tile_knobs()
        && matches!(
            spec.key.query,
            kernel_core::contract::StaticQuery::Gemm(query) if query.weight == WeightFormat::Bf16
        )
}

/// The projections' cubins: the GEMM's distinct entries but the decoded
/// and the staged tiles', then the layout copies'.
pub fn cubin_specs(shape: Shape) -> Result<Vec<CubinSpec>, String> {
    let tiled: Vec<_> = instantiations(shape)?
        .into_iter()
        .filter(|spec| !decoded(spec) && !staged(spec))
        .collect();
    let mut gemm = kernels_sm120::gemm::cubin_spec(&tiled).map_err(|r| r.0.to_string())?;
    gemm.name = CUBIN.into();
    let stages: Vec<_> = copies(shape)?
        .into_iter()
        .map(|copy| (copy.stage().entry().symbol.clone(), copy.stage().clone()))
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_values()
        .collect();
    let copies = row::cubin_spec(COPIES_CUBIN, &stages).map_err(|r| r.0.to_string())?;
    Ok(vec![gemm, copies])
}

/// The projections' decoded-tile cubin ([`DECODED_CUBIN`]): the distinct
/// GEMM entries at `gemm::balanced_decoded_tile_knobs`.
pub fn decoded_cubin_spec(shape: Shape) -> Result<CubinSpec, String> {
    let decoded: Vec<_> = instantiations(shape)?.into_iter().filter(decoded).collect();
    let mut spec = kernels_sm120::gemm::cubin_spec(&decoded).map_err(|r| r.0.to_string())?;
    spec.name = DECODED_CUBIN.into();
    Ok(spec)
}

/// The projections' staged-tile cubin ([`STAGED_CUBIN`]): the distinct GEMM
/// entries at `gemm::staged_tile_knobs` over BF16 weights.
pub fn staged_cubin_spec(shape: Shape) -> Result<CubinSpec, String> {
    let staged: Vec<_> = instantiations(shape)?.into_iter().filter(staged).collect();
    let mut spec = kernels_sm120::gemm::cubin_spec(&staged).map_err(|r| r.0.to_string())?;
    spec.name = STAGED_CUBIN.into();
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::LOWERED;
    use kernel_core::contract::{DType, ScaleMode};

    const FP8: WeightFormat = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);

    /// Every lowered projection, over the FP8 checkpoint's format for it
    /// and over BF16, has a shared offer at each declared use, writing its
    /// declared output dtype with its declared residual.
    #[test]
    fn every_lowered_projection_use_has_a_shared_offer() {
        let mut refused = Vec::new();
        for &id in LOWERED {
            let (_, _, projection) = declared(id).unwrap();
            let fp8 = projection.composition.as_ref().is_some_and(|c| {
                c.numerical_contracts
                    .contains_key(&registry::by_format(FP8).unwrap().id())
            });
            let formats: &[WeightFormat] = if fp8 {
                &[FP8, WeightFormat::Bf16]
            } else {
                &[WeightFormat::Bf16]
            };
            for &weight in formats {
                for (usage, output) in uses(id).unwrap() {
                    match offer(id, weight, usage) {
                        Ok(generated) => {
                            let capability = generated.offer.capability().unwrap();
                            assert_eq!(capability.arithmetic.output, output, "{id}");
                            assert_eq!(
                                capability.epilogues.residual_add,
                                projection.epilogues.residual_add,
                                "{id}"
                            );
                        }
                        Err(error) => refused.push(format!("{id} {weight:?} {usage:?}: {error}")),
                    }
                }
            }
        }
        assert!(refused.is_empty(), "{refused:#?}");
        // The FP8 export's formats: one cubin of the distinct entries, and
        // the six decoded-tile prefill entries in their own cubin (the warp
        // tiles stay for the wide verify buckets).
        let cubins = cubin_specs(Shape::TP4).unwrap();
        assert_eq!(cubins[0].name, CUBIN);
        assert_eq!(cubins[0].entries.len(), 107);
        let decoded = decoded_cubin_spec(Shape::TP4).unwrap();
        assert_eq!(decoded.name, DECODED_CUBIN);
        assert_eq!(decoded.entries.len(), 6);
        assert!(
            decoded
                .entries
                .iter()
                .all(|e| cubins[0].entries.iter().all(|t| t.symbol != e.symbol))
        );
        // Each third of the KDA input, each half of the drafter's input and
        // the compaction of linear.beta's and the indexer's head weights'
        // padded rows (the drafter's alike), one entry each that serves
        // every compiled capacity.
        assert_eq!(cubins[1].name, COPIES_CUBIN);
        assert_eq!(cubins[1].entries.len(), 3 + 2 + 2);
        assert_eq!(copies(Shape::TP4).unwrap().len(), (3 + 2 + 2) * 7);
        assert!(
            uses("head")
                .unwrap()
                .iter()
                .all(|(_, output)| *output == DType::F32)
        );
    }

    /// The served FP8 export's BF16 and FP8 decode and verify buckets of
    /// the shapes kernel-lab's strip-chain sweeps measured run their strip
    /// chains; their prefill uses and the other shapes keep the tiled
    /// kernels.
    #[test]
    fn swept_decode_buckets_run_strip_chains() {
        let (mut bf16, mut fp8) = (0, 0);
        for (id, weight) in checkpoint_formats(Shape::TP4).unwrap() {
            for (usage, _) in uses(id).unwrap() {
                let generated = offer(id, weight, usage).unwrap();
                let Compute::Dense(spec) = &generated.offer.handle().compute else {
                    panic!("{id} is not a dense GEMM");
                };
                let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
                    panic!("{id} has no GEMM query");
                };
                let expected = match (weight, query.n, query.k, query.m_capacity) {
                    (_, _, _, 33..) => None,
                    (WeightFormat::Bf16, n, k, m) => match (n, k, m) {
                        (2048, 4096, _) => Some((2, 1)),
                        (4096, 2048, 8 | 16) => Some((1, 1)),
                        (4096, 2048, _) => Some((4, 1)),
                        (128, 4096, ..=8) => Some((1, 23)),
                        (128, 4096, _) => Some((1, 47)),
                        (64, 4096, _) => Some((1, 64)),
                        (320, 4096, _) => Some((1, 18)),
                        (38720, 4096, _) => Some((4, 1)),
                        _ => None,
                    },
                    (WeightFormat::Fp8Block128x128(_), n, k, m) => match (n, k, m) {
                        (4096, 4096, ..=4) => Some((4, 1)),
                        (4096, 4096, 8) => Some((2, 2)),
                        (4096, 4096, _) => Some((1, 1)),
                        (1536, 4096, ..=8) => Some((2, 7)),
                        (1536, 4096, _) => Some((1, 3)),
                        (512, 4096, _) => Some((1, 11)),
                        (4096, 1536, ..=8) => Some((2, 1)),
                        (4096, 1536, _) => Some((1, 1)),
                        (4096, 3072, ..=4) => Some((4, 1)),
                        (4096, 3072, 8) => Some((2, 1)),
                        (4096, 3072, _) => Some((1, 1)),
                        (4096, 512, ..=2) => Some((4, 1)),
                        (4096, 512, ..=8) => Some((2, 1)),
                        (4096, 512, _) => Some((1, 1)),
                        _ => None,
                    },
                    _ => None,
                };
                match (spec.key.dense, expected) {
                    (Some(DenseKernel::StripChains { strips: s, splits }), Some(e)) => {
                        assert_eq!((s, splits), e, "{id} {usage:?}");
                        if weight == WeightFormat::Bf16 {
                            bf16 += 1;
                        } else {
                            fp8 += 1;
                        }
                    }
                    (Some(DenseKernel::StripChains { .. }), None) => {
                        panic!("{id} {usage:?} runs strip chains")
                    }
                    (kernel, Some(_)) => panic!("{id} {usage:?} runs {kernel:?}"),
                    _ => {}
                }
            }
        }
        // Each at six decode and six verify buckets: BF16 q, k and v, the
        // KDA output, four n128 and three n64 projections, two routers and
        // the head; FP8 the latent q-a, kv-a, q-b and output of the target
        // and the drafter, the dense down projection and both shared
        // experts' down.
        assert_eq!(bf16, (3 + 1 + 4 + 3 + 2 + 1) * 12);
        assert_eq!(fp8, (8 + 1 + 2) * 12);
    }

    /// The served FP8 export's FP8 prefill projections run balanced decoded
    /// weight tiles (`gemm::balanced_decoded_tile_knobs`): the latent q-a,
    /// kv-a, q-b and output of the target and the drafter, the dense down
    /// projection and both shared experts' down projections; their decode
    /// and verify buckets keep their kernels and knobs.
    #[test]
    fn fp8_prefill_projections_run_decoded_tiles() {
        let mut decoded = std::collections::BTreeSet::new();
        let mut kept_verify = 0;
        for (id, weight) in checkpoint_formats(Shape::TP4).unwrap() {
            for (usage, _) in uses(id).unwrap() {
                let generated = offer(id, weight, usage).unwrap();
                let Compute::Dense(spec) = &generated.offer.handle().compute else {
                    panic!("{id} is not a dense GEMM");
                };
                let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
                    panic!("{id} has no GEMM query");
                };
                let fp8 = matches!(weight, WeightFormat::Fp8Block128x128(_));
                let knobs = fp8 && super::decoded(spec);
                let prefill = usage.phase == ExecutionPhase::Prefill;
                if fp8 && prefill && query.m_capacity > 32 {
                    assert!(knobs, "{id} {usage:?}");
                    assert_eq!(
                        spec.key.dense,
                        Some(DenseKernel::TiledChains { splits: 1 }),
                        "{id}"
                    );
                    assert!(decoded.insert(id), "{id} has two prefill uses");
                } else {
                    // Decode and verify uses, the verify buckets compiled at
                    // the prefill capacity included, keep their knobs.
                    assert!(!knobs, "{id} {usage:?}");
                    if fp8 && usage.phase == ExecutionPhase::Verify && query.m_capacity > 32 {
                        kept_verify += 1;
                    }
                }
            }
        }
        // The eleven projections' wide verify buckets (64, 128 and 192 rows,
        // compiled at the prefill capacity) keep their warp tiles.
        assert_eq!(kept_verify, 33);
        let want: std::collections::BTreeSet<&str> = [
            "latent.q-a",
            "latent.q-b",
            "latent.kv-a",
            "latent.output",
            "mtp.latent.q-a",
            "mtp.latent.q-b",
            "mtp.latent.kv-a",
            "mtp.latent.output",
            "dense.down",
            "moe.shared.down",
            "mtp.moe.shared.down",
        ]
        .into_iter()
        .collect();
        assert_eq!(decoded, want);
    }

    /// The served FP8 export's BF16 prefill projections of the shapes
    /// `staged_prefill` lists run staged weight tiles
    /// (`gemm::staged_tile_knobs`): the linear attention q, k, v and output,
    /// its gate and forget gates' up projections, the indexer's query (the
    /// drafter's too) and the drafter's input projection. Their decode and
    /// verify uses and every other projection keep their kernels and knobs.
    #[test]
    fn bf16_prefill_projections_run_staged_tiles() {
        let mut staged = std::collections::BTreeSet::new();
        for (id, weight) in checkpoint_formats(Shape::TP4).unwrap() {
            for (usage, _) in uses(id).unwrap() {
                let generated = offer(id, weight, usage).unwrap();
                let Compute::Dense(spec) = &generated.offer.handle().compute else {
                    panic!("{id} is not a dense GEMM");
                };
                let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
                    panic!("{id} has no GEMM query");
                };
                if !super::staged(spec) {
                    continue;
                }
                assert_eq!(weight, WeightFormat::Bf16, "{id}");
                assert_eq!(usage.phase, ExecutionPhase::Prefill, "{id} {usage:?}");
                assert!(query.m_capacity > 32, "{id} {usage:?}");
                assert_eq!(
                    spec.key.dense,
                    Some(DenseKernel::TiledChains { splits: 1 }),
                    "{id}"
                );
                assert!(staged.insert(id), "{id} has two prefill uses");
            }
        }
        let want: std::collections::BTreeSet<&str> = [
            "linear.q",
            "linear.k",
            "linear.v",
            "linear.output",
            "linear.gate-b",
            "linear.forget-b",
            "latent.indexer.q",
            "mtp.latent.indexer.q",
            "mtp.eh",
        ]
        .into_iter()
        .collect();
        assert_eq!(staged, want);
        let cubin = staged_cubin_spec(Shape::TP4).unwrap();
        assert_eq!(cubin.name, STAGED_CUBIN);
        assert_eq!(cubin.entries.len(), 5);
        let tiled = cubin_specs(Shape::TP4).unwrap();
        assert!(
            cubin
                .entries
                .iter()
                .all(|e| tiled[0].entries.iter().all(|t| t.symbol != e.symbol))
        );
    }

    /// The KDA q, k and v projections' uses copy their contiguous rows into
    /// their third of the convolution's `[rows, 3 * width]` input, at the
    /// use's compiled capacity; the padded BF16 projections compact their
    /// rows; the other projections write their consumer's buffer.
    #[test]
    fn kda_projections_copy_into_their_third_of_the_convolution_input() {
        for (id, third) in KDA_THIRDS {
            for (usage, _) in uses(id).unwrap() {
                let copy = copy(id, WeightFormat::Bf16, usage).unwrap().unwrap();
                let (width, source, destination) = copy.layout();
                let rows = domain(id, WeightFormat::Bf16, usage)
                    .unwrap()
                    .geometry
                    .unwrap()
                    .rows;
                assert_eq!(u64::from(width), rows);
                assert_eq!((source.row_stride, source.column_offset), (width, 0));
                assert_eq!(
                    (destination.row_stride, destination.column_offset),
                    (3 * width, third * width)
                );
                let capacity = u64::from(compiled_capacity(id, usage).unwrap());
                // Through the last row's third.
                assert_eq!(
                    copy.destination_bytes(),
                    ((capacity - 1) * 3 * rows + (u64::from(third) + 1) * rows) * 2
                );
                let capability = capability(id, WeightFormat::Bf16, usage).unwrap();
                assert_eq!(capability.stages.last(), Some(&copy.stage().entry().symbol));
            }
        }
        // The padded BF16 projections compact their rank's rows; the
        // router's padded F32 scores are not copied.
        let compacted = [
            ("linear.beta", 16),
            ("latent.indexer.weights", 32),
            ("mtp.latent.indexer.weights", 32),
        ];
        for id in LOWERED
            .iter()
            .filter(|id| KDA_THIRDS.iter().all(|(kda, _)| kda != *id))
        {
            for (usage, _) in uses(id).unwrap() {
                let copied = copy(id, WeightFormat::Bf16, usage).unwrap();
                match compacted.iter().find(|(padded, _)| padded == id) {
                    Some(&(_, rows)) => {
                        let (width, source, destination) = copied.unwrap().layout();
                        assert_eq!(
                            (width, source.row_stride, destination.row_stride),
                            (rows, 64, rows),
                            "{id}"
                        );
                    }
                    None => assert!(copied.is_none(), "{id}"),
                }
            }
        }
    }

    /// The drafter's input halves land side by side at each of its input
    /// projection's uses: the embedding's at column 0, the hidden row's at
    /// `hidden`, each row `2 * hidden` wide.
    #[test]
    fn the_drafter_input_halves_land_side_by_side() {
        let hidden = Shape::TP4.profile.hidden;
        for (usage, _) in uses("mtp.eh").unwrap() {
            let halves = eh_halves(usage).unwrap();
            for (half, offset) in halves.iter().zip([0, hidden]) {
                let (width, source, destination) = half.layout();
                assert_eq!((width, source.row_stride), (hidden, hidden));
                assert_eq!(
                    (destination.row_stride, destination.column_offset),
                    (2 * hidden, offset)
                );
                assert_eq!(
                    half.stage().capacity(),
                    compiled_capacity("mtp.eh", usage).unwrap()
                );
            }
        }
        // mtp.eh's offer reads the whole row.
        let (_, geometry, projection) = declared("mtp.eh").unwrap();
        let k = rank_selection(projection, geometry, Topology { ranks: 4, rank: 0 }, 0)
            .unwrap()
            .physical
            .k;
        assert_eq!(k, u64::from(2 * hidden));
    }

    /// The router's padded rows, which the compaction row kernel drops,
    /// are the ones its offers write, the drafter's alike.
    #[test]
    fn the_router_columns_are_its_padded_rows() {
        let (experts, padded) = crate::precision::router_columns(Shape::TP4);
        assert_eq!((experts, padded), (288, 320));
        for id in ["moe.router", "mtp.moe.router"] {
            assert_eq!(rank_rows(id).unwrap(), u64::from(experts), "{id}");
            for (usage, _) in uses(id).unwrap() {
                let rows = domain(id, WeightFormat::Bf16, usage)
                    .unwrap()
                    .geometry
                    .unwrap()
                    .rows;
                assert_eq!(rows, u64::from(padded), "{id}");
            }
        }
    }
}
