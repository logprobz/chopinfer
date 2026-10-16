//! Kernel time of the projections a decode step is made of, at the row
//! counts a step runs, to show where a single-row projection's time goes
//! before choosing a faster decode schedule.
//!
//! - `tiled_projection_latency`: every tiled dense GEMM entry
//!   (`DenseKernel::Tiled`, or `TiledChains` or `StripChains` at its
//!   declared splits) of the catalogs' generated projection inventory
//!   whose cubin a listed bundle report builds (`CHOPINFER_BUNDLE_REPORTS`,
//!   paths separated by `:`, each checked against its lock and sources), so
//!   the timed bytes are the ones a bundle loads. An entry runs at the rows
//!   its capacity serves: 1, 4 and 32 at those decode buckets, 55 and 2,048
//!   at a prefill capacity. Each one-split decode entry also runs the same
//!   compiled code at more K splits (`splits` is a launch argument; the grid
//!   and the workspace follow `gemm::plan`), each output compared with its
//!   one-split output bit for bit: with one split per K chunk each partial
//!   is one MMA chain, which shows whether the split reduction keeps the
//!   one-split bits. A chain-split entry's output is compared with its
//!   operation's one-split `Tiled` entry, launched from the listed reports
//!   or `CHOPINFER_REFERENCE_REPORTS` (optional); where no report builds
//!   that entry, the receipt records the comparison as not performed
//!   (`elements_differing_from_one_split` null). An inventory offer that
//!   no listed report builds is timed in the schedule a decode bucket ran
//!   before strip chains when a report builds that instead (a bundle
//!   before the change); the receipt names the inventory's symbol.
//! - `strip_chain_latency`: the strip-chain lab sweep
//!   (`strip_chain_shapes.rs`, kernel-lab's own cubins): each case's entry
//!   of today, its one-split entry and every strip-chain candidate at the
//!   case's capacity, each output compared with the one-split output bit for
//!   bit; then each case's fastest candidate against today's entry.
//!   `fp8_strip_chain_latency` does the same for the FP8 128x128 sweep
//!   (`strip_chain_fp8_shapes.rs`).
//! - `decoded_tile_latency`: the decoded-tile lab sweep
//!   (`decoded_tile_shapes.rs`, FP8 128x128 prefill projections): each
//!   case's one-split entry, its warp tiles of today and its decoded tiles
//!   at 64, 512, 2,048 and 8,192 live rows of an 8,192-row capacity, each
//!   output compared with the one-split output bit for bit.
//!   `balanced_decoded_latency` does the same for the balanced decoded-tile
//!   sweep (`decoded_balanced_shapes.rs`: the decoded tiles of today and
//!   the balanced ones) at its rows, or at the comma-separated
//!   `CHOPINFER_TIMED_ROWS`; `staged_tile_latency` for the staged-tile sweep
//!   (`staged_tile_shapes.rs`: BF16 prefill projections in the warp tiles of
//!   today, staged tiles and the other warp tiles).
//! - `chained_expert_latency`: the chained expert engine's served dense
//!   decode projections (16-row tiles) in FP8 128x128 blocks: the dense MLP
//!   and shared expert gate/up and down at 1, 4 and 32 tokens, each in the
//!   column schedule and in block warps (`chained::Schedule`).
//! - `split_expert_latency`: the routed FP8 projections with every expert
//!   split over the ranks (`chained::split`), on rank 0 of all the experts:
//!   uniform seeded routes at 2,048 and 8,192 tokens and a captured served
//!   prefill's routes (`CHOPINFER_MOE_ROUTE_CAPTURE`, one launch per MoE
//!   layer) at 2,048 and 4,096 tokens, 8,192 tokens over a layer and the
//!   next, and a layer's capture repeated to 8,192 tokens; then the served
//!   decode batches of the capture; with the F32 route sum. The served FP8
//!   split forms of each projection are compared with each other bit for
//!   bit.
//!
//! Each time is one captured graph of PDL launches over distinct weight
//! copies (at least `CHAIN_BYTES` of weights, so every launch reads its
//! weights from DRAM as a decode step does), divided by its launches: the
//! median of seven rounds, each after an L2 eviction. Receipts (`bench`)
//! hold the per-launch time, the grid, the bytes read and the bitwise
//! comparison.
//!
//! Run: `CHOPINFER_KERNEL_BENCH=1 CHOPINFER_KERNEL_GPU=<ordinal>
//! CHOPINFER_BUNDLE_REPORTS=<report> cargo test -p kernel-lab --release
//! --test projection_latency`. `CHOPINFER_KERNEL_BENCH_FILTER` keeps the
//! symbols containing it; `CHOPINFER_STRIP_FILTER` keeps the strip-chain
//! cases whose tag contains it.

use kernel_core::contract::InstantiationSpec;
use kernel_core::contract::{
    DType, DenseKernel, LaunchPlan, RuntimeCounts, StaticQuery, WeightFormat,
};
use kernel_core::gated_mlp_oracle::local_route_metadata;
use kernel_core::precision::format::registry;
use kernel_lab::families::decoded_balanced::shapes as balanced_shapes;
use kernel_lab::families::decoded_tiles::shapes as decoded_shapes;
use kernel_lab::families::moe_chained_decoded::shapes as chained_decoded_shapes;
use kernel_lab::families::moe_chained_split_tile_pairs as split_tile_pairs;
use kernel_lab::families::staged_tiles::shapes as staged_shapes;
use kernel_lab::families::strip_chains::shapes as strip_shapes;
use kernel_lab::families::strip_chains_fp8::shapes as fp8_strip_shapes;
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, gpu_test, poison, time};
use kernels_sm120::gemm::{self, GemmBuffers};
use kernels_sm120::moe::chained::{self, Projection, Schedule, Weights, split};
use kernels_sm120::precision::generated::{self, Implementation, Request};
use kernels_sm120::precision::{Compute, KernelOffer};
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Weight bytes one timed graph reads at least.
const CHAIN_BYTES: u64 = 256 << 20;
/// Most launches of one timed graph.
const CHAIN_MAX: u64 = 64;
/// Rows a timed entry runs at, where its capacity serves them.
const DECODE_ROWS: [u32; 3] = [1, 4, 32];
/// A short prompt's chunk and a full one.
const PREFILL_ROWS: [u32; 2] = [55, 2048];
/// K splits each decode entry also runs at, besides one per K chunk.
const SPLITS: [u8; 4] = [2, 4, 8, 16];

const WEIGHT: u64 = 1;
const ACTIVATION: u64 = 2;
const OUTPUT: u64 = 3;
const PARTIAL: u64 = 4;
const SEMAPHORES: u64 = 5;
const SCALES: u64 = 6;
const RESIDUAL: u64 = 7;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn benchmarks_enabled(what: &str) -> bool {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1") {
        return true;
    }
    kernel_lab::skip(&format!("{what}: set CHOPINFER_KERNEL_BENCH=1")).unwrap();
    false
}

fn selected(symbol: &str) -> bool {
    std::env::var("CHOPINFER_KERNEL_BENCH_FILTER")
        .map(|filter| symbol.contains(&filter))
        .unwrap_or(true)
}

/// Deterministic values.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// BF16 bit patterns of both signs with exponents -7 through 2.
    fn bf16_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|_| {
                let bits = self.next();
                ((bits as u16 & 0x807f) | ((120 + (bits >> 16) % 10) as u16) << 7).to_le_bytes()
            })
            .collect()
    }

    /// E4M3 codes other than the NaN codes.
    fn e4m3_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .map(|_| match self.next() as u8 {
                0x7f | 0xff => 0x38,
                code => code,
            })
            .collect()
    }

    /// Positive F32 scales between 2^-9 and 2^-7.
    fn scale_bytes(&mut self, count: usize) -> Vec<u8> {
        self.scale_bytes_over(count, 512.0)
    }

    /// Positive F32 scales between 4 / `divisor` and 1 / `divisor`.
    fn scale_bytes_over(&mut self, count: usize, divisor: f32) -> Vec<u8> {
        (0..count)
            .flat_map(|_| (f32::from(1 + self.next() as u16 % 4) / divisor).to_le_bytes())
            .collect()
    }
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(4))?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn zeroed(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.max(4))?;
    poison::fill_all(session, &buffer, &[0])?;
    Ok(buffer)
}

fn download(session: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(err)?;
    Ok(bytes)
}

/// Launches of one graph over `bytes` of weights each.
fn chain_launches(bytes: u64) -> usize {
    CHAIN_BYTES.div_ceil(bytes.max(1)).clamp(4, CHAIN_MAX) as usize
}

/// The per-launch time of `launches` launches captured in one graph: the
/// median of seven rounds after two warmups, each after an L2 eviction.
fn graph_us(
    session: &Session,
    launches: usize,
    mut launch: impl FnMut(usize, &runtime_cuda::Stream) -> LabResult<()>,
) -> LabResult<(f64, f64)> {
    let graph = session.capture(|stream| (0..launches).try_for_each(|i| launch(i, stream)))?;
    let timing = time(session, 2, 7, |stream| graph.launch(stream).map_err(err))?;
    let per = |ms: f32| f64::from(ms) * 1000.0 / launches as f64;
    Ok((per(timing.median_ms), per(timing.min_ms)))
}

// ------------------------------------------------------------ tiled ----

/// Every cubin the reports in `variable` build, by file, with the entry
/// symbols of its locked identity.
fn reported(variable: &str) -> LabResult<Vec<(PathBuf, Vec<String>)>> {
    let reports = std::env::var_os(variable)
        .ok_or_else(|| format!("set {variable} to the bundles' kernel-build reports"))?;
    let mut cubins = Vec::new();
    for path in std::env::split_paths(&reports) {
        let report = kernel_build::BuildReport::read(&path).map_err(err)?;
        report.verify_current().map_err(err)?;
        for cubin in report.cubins {
            let symbols = cubin.identity["entries"]
                .as_array()
                .ok_or_else(|| format!("{} identity lists no entries", cubin.name))?
                .iter()
                .map(|entry| {
                    entry["symbol"]
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| format!("{} has an entry without a symbol", cubin.name))
                })
                .collect::<LabResult<Vec<_>>>()?;
            cubins.push((cubin.path, symbols));
        }
    }
    Ok(cubins)
}

fn cubin_of(cubins: &[(PathBuf, Vec<String>)], symbol: &str) -> Option<PathBuf> {
    cubins
        .iter()
        .find(|(_, symbols)| symbols.iter().any(|s| s == symbol))
        .map(|(path, _)| path.clone())
}

/// The offer the shared generator gives `offer`'s operation and use with
/// `kernel` at its capacity and default knobs (as `tiled_chain_splits`
/// regenerates its references).
fn regenerate(offer: &KernelOffer, kernel: DenseKernel) -> LabResult<InstantiationSpec> {
    let Compute::Dense(spec) = &offer.handle().compute else {
        return Err("not a dense offer".into());
    };
    let StaticQuery::Gemm(query) = spec.key.query else {
        return Err("not a GEMM".into());
    };
    let capability = offer.capability().map_err(err)?;
    let operation = generated::comparison::operation(&capability)?;
    let generated = generated::offer(
        Request {
            operation: &operation,
            format: registry::by_format(capability.weight).map_err(err)?.id(),
            usage: offer.usage(),
            implementation: Implementation::Dense {
                kernel,
                capacity: query.m_capacity,
                knobs: None,
            },
        },
        &[],
    )?;
    match &generated.offer.handle().compute {
        Compute::Dense(spec) => Ok(spec.clone()),
        _ => Err("regenerated offer is not dense".into()),
    }
}

/// A tiled entry to time: its compile key and symbol, the reported cubin
/// that holds it, the rows it runs at, the inventory's symbol when the
/// reports build an earlier schedule of its operation instead, and the
/// one-split tiled entry its outputs are compared with, where a report
/// builds it.
struct Tiled {
    spec: InstantiationSpec,
    cubin: PathBuf,
    rows: Vec<u32>,
    inventory_symbol: Option<String>,
    reference: Option<(InstantiationSpec, PathBuf)>,
}

/// The reported tiled entries of the inventory over BF16 and FP8 128x128
/// weights, once per symbol, in symbol order.
///
/// An offer whose symbol no listed report builds is regenerated in the
/// schedules a decode bucket ran before strip chains (`TiledChains` at
/// `gemm::chain_splits`, else one-split `Tiled`), so reports of an earlier
/// bundle time the entries that bundle serves for the same operations. A
/// chain-split entry's one-split reference is the offer regenerated at
/// `Tiled { splits: 1 }`, from the reports in `CHOPINFER_REFERENCE_REPORTS`
/// (optional) or the timed ones.
fn tiled_entries() -> LabResult<Vec<Tiled>> {
    let cubins = reported("CHOPINFER_BUNDLE_REPORTS")?;
    let mut references = cubins.clone();
    if std::env::var_os("CHOPINFER_REFERENCE_REPORTS").is_some() {
        references.extend(reported("CHOPINFER_REFERENCE_REPORTS")?);
    }
    let (offers, _, _) = kernel_lab::models::projection_inventory()?;
    let mut entries = BTreeMap::new();
    for offer in offers {
        let Compute::Dense(served) = &offer.handle().compute else {
            continue;
        };
        if !matches!(
            served.key.dense,
            Some(
                DenseKernel::Tiled { .. }
                    | DenseKernel::TiledChains { .. }
                    | DenseKernel::StripChains { .. }
            )
        ) {
            continue;
        }
        let StaticQuery::Gemm(query) = served.key.query else {
            return Err(format!("{} has no GEMM query", served.symbol));
        };
        // Formats without a fixture here (NVFP4, MXFP8) are left out.
        if weight_layout(&query).is_err() {
            continue;
        }
        let mut inventory_symbol = None;
        let spec = if cubin_of(&cubins, &served.symbol).is_some() {
            served.clone()
        } else if query.m_capacity <= 32 {
            let earlier = match gemm::chain_splits(&query) {
                1 => DenseKernel::Tiled { splits: 1 },
                splits => DenseKernel::TiledChains { splits },
            };
            // An offer no report builds in either schedule is left out.
            let Ok(spec) = regenerate(&offer, earlier) else {
                continue;
            };
            inventory_symbol = Some(served.symbol.clone());
            spec
        } else {
            continue;
        };
        if entries.contains_key(&spec.symbol) || !selected(&spec.symbol) {
            continue;
        }
        let Some(cubin) = cubin_of(&cubins, &spec.symbol) else {
            continue;
        };
        let reference = match spec.key.dense {
            Some(DenseKernel::TiledChains { .. } | DenseKernel::StripChains { .. }) => {
                regenerate(&offer, DenseKernel::Tiled { splits: 1 })
                    .ok()
                    .and_then(|one| cubin_of(&references, &one.symbol).map(|path| (one, path)))
            }
            _ => None,
        };
        let capacity = query.m_capacity;
        let rows: Vec<u32> = if capacity <= 32 {
            DECODE_ROWS.into_iter().filter(|m| *m == capacity).collect()
        } else {
            PREFILL_ROWS
                .into_iter()
                .filter(|m| *m <= capacity)
                .collect()
        };
        if rows.is_empty() {
            continue;
        }
        entries.insert(
            spec.symbol.clone(),
            Tiled {
                spec,
                cubin,
                rows,
                inventory_symbol,
                reference,
            },
        );
    }
    Ok(entries.into_values().collect())
}

/// `key` with its kernel declaring `splits` K splits, and room for them.
fn with_splits(
    key: &kernel_core::contract::CompileKey,
    splits: u8,
) -> kernel_core::contract::CompileKey {
    let mut key = key.clone();
    key.dense = Some(DenseKernel::Tiled { splits });
    key.knobs.split_k_capacity = key.knobs.split_k_capacity.max(splits);
    key
}

/// Weight bytes and scale floats of one weight copy.
fn weight_layout(query: &kernel_core::contract::GemmQuery) -> LabResult<(usize, usize)> {
    let (n, k) = (query.n as usize, query.k as usize);
    match query.weight {
        WeightFormat::Bf16 => Ok((n * k * 2, 0)),
        WeightFormat::Fp8Block128x128(_) => Ok((n * k, n.div_ceil(128) * k.div_ceil(128))),
        other => Err(format!("no timing fixture for {other:?} weights")),
    }
}

/// Runs and times `entry` at `m` rows for each K split, from one split; a
/// chain-split entry at its declared splits only. Each output is compared
/// bit for bit with a one-split output: a one-split entry's own at one
/// split (that launch is the reference and records no comparison), a
/// chain-split entry's independent one-split reference entry (`reference`,
/// launched once untimed from its own cubin); a chain-split entry without
/// one records the comparison as not performed (null).
fn time_tiled(
    session: &Session,
    module: &Module,
    reference_module: Option<&Module>,
    entry: &Tiled,
    m: u32,
) -> LabResult<()> {
    let key = &entry.spec.key;
    let StaticQuery::Gemm(query) = key.query else {
        return Err("not a GEMM".into());
    };
    let symbol = &entry.spec.symbol;
    let (weight_bytes, scale_floats) = weight_layout(&query)?;
    let decode = query.m_capacity <= 32;
    let chunks = query.k.div_ceil(u32::from(key.knobs.tile_k));
    let chain_splits = match key.dense {
        Some(DenseKernel::TiledChains { splits } | DenseKernel::StripChains { splits, .. }) => {
            Some(splits)
        }
        _ => None,
    };
    let mut splits = vec![chain_splits.unwrap_or(1)];
    if decode && chain_splits.is_none() {
        splits.extend(SPLITS.into_iter().filter(|s| u32::from(*s) < chunks));
        if chunks <= u32::from(u8::MAX) {
            splits.push(chunks as u8);
        }
    }
    splits.dedup();
    let keys = splits
        .iter()
        .map(|&s| match chain_splits {
            Some(_) => key.clone(),
            None => with_splits(key, s),
        })
        .collect::<Vec<_>>();
    let counts_of = |k: &kernel_core::contract::CompileKey, s: u8| RuntimeCounts {
        split_k: s,
        ..gemm::row_counts(k, m)
    };
    let counts = |s: u8| counts_of(key, s);
    let plans = keys
        .iter()
        .zip(&splits)
        .map(|(k, &s)| gemm::plan(k, counts(s)).map_err(err))
        .collect::<LabResult<Vec<LaunchPlan>>>()?;
    let one_split = match (&entry.reference, reference_module) {
        (Some((spec, _)), Some(module)) => {
            let plan = gemm::plan(&spec.key, counts_of(&spec.key, 1)).map_err(err)?;
            Some((spec, plan, module.function(&spec.symbol).map_err(err)?))
        }
        _ => None,
    };
    let workspace = plans
        .iter()
        .chain(one_split.iter().map(|(_, plan, _)| plan))
        .map(|p| p.workspace_bytes)
        .max()
        .unwrap_or(0);
    let semaphore_bytes = keys
        .iter()
        .chain(one_split.iter().map(|(spec, _, _)| &spec.key))
        .map(|k| GemmBuffers::semaphore_capacity_bytes(k).map_err(err))
        .collect::<LabResult<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let mut rng = Rng(0x7f4a_7c15 ^ u64::from(query.n) << 20 ^ u64::from(query.k));
    let weights = match query.weight {
        WeightFormat::Bf16 => rng.bf16_bytes(weight_bytes / 2),
        _ => rng.e4m3_bytes(weight_bytes),
    };
    let scales = rng.scale_bytes(scale_floats);
    let launches = chain_launches((weight_bytes + scale_floats * 4) as u64);
    let copies = (0..launches)
        .map(|_| {
            Ok((
                upload(session, &weights)?,
                (scale_floats > 0)
                    .then(|| upload(session, &scales))
                    .transpose()?,
            ))
        })
        .collect::<LabResult<Vec<_>>>()?;
    let rows = query.m_capacity as usize;
    let activation = upload(session, &rng.bf16_bytes(rows * query.k as usize))?;
    let out_n = if query.epilogues.swiglu.is_some() {
        query.n / 2
    } else {
        query.n
    } as usize;
    let out_bytes = rows * out_n * if query.output == DType::F32 { 4 } else { 2 };
    let output = zeroed(session, out_bytes)?;
    let residual = if query.epilogues.residual_add {
        Some(upload(session, &vec![0; out_bytes])?)
    } else {
        None
    };
    let partial = zeroed(session, workspace as usize)?;
    let semaphores = zeroed(session, semaphore_bytes as usize)?;
    let function: Function = module.function(symbol).map_err(err)?;
    let shared = plans
        .iter()
        .map(|p| p.dynamic_shared_bytes)
        .max()
        .unwrap_or(0);
    if shared >= 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    if let Some((_, plan, function)) = &one_split
        && plan.dynamic_shared_bytes >= 48 * 1024
    {
        function
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(err)?;
    }
    let launch_one = |k: &kernel_core::contract::CompileKey,
                      plan: LaunchPlan,
                      function: &Function,
                      s: u8,
                      copy: usize,
                      stream: &runtime_cuda::Stream|
     -> LabResult<()> {
        let maps = gemm::tensor_maps(k, WEIGHT, ACTIVATION).map_err(err)?;
        let args = gemm::arguments(
            k,
            counts_of(k, s),
            maps,
            GemmBuffers {
                output: OUTPUT,
                partial: PARTIAL,
                semaphores: SEMAPHORES,
                weight_scales: (scale_floats > 0).then_some(SCALES),
                activation_scales: None,
                residual: residual.is_some().then_some(RESIDUAL),
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .map_err(err)?;
        let (weight, scale) = &copies[copy];
        let mut buffers = BTreeMap::from([
            (WEIGHT, weight),
            (ACTIVATION, &activation),
            (OUTPUT, &output),
            (PARTIAL, &partial),
            (SEMAPHORES, &semaphores),
        ]);
        if let Some(scale) = scale {
            buffers.insert(SCALES, scale);
        }
        if let Some(residual) = &residual {
            buffers.insert(RESIDUAL, residual);
        }
        launch::launch(
            &session.context,
            function,
            stream,
            plan,
            &args,
            &Bindings { buffers },
        )
    };
    let call = |index: usize, copy: usize, stream: &runtime_cuda::Stream| -> LabResult<()> {
        launch_one(
            &keys[index],
            plans[index],
            &function,
            splits[index],
            copy,
            stream,
        )
    };
    let width = if query.output == DType::F32 { 4 } else { 2 };
    let live = m as usize * out_n * width;
    // The one-split output and the entry that wrote it.
    let mut reference: Option<(Vec<u8>, String)> = None;
    if let Some((spec, plan, one)) = &one_split {
        poison::fill_all(session, &output, &[0xff])?;
        launch_one(&spec.key, *plan, one, 1, 0, &session.stream)?;
        reference = Some((download(session, &output)?, spec.symbol.clone()));
    }
    for (index, &s) in splits.iter().enumerate() {
        poison::fill_all(session, &output, &[0xff])?;
        call(index, 0, &session.stream)?;
        let bytes = download(session, &output)?;
        let differing = reference.as_ref().map(|(old, _)| {
            old[..live]
                .chunks(width)
                .zip(bytes[..live].chunks(width))
                .filter(|(a, b)| a != b)
                .count()
        });
        let compared_with = reference.as_ref().map(|(_, symbol)| symbol.clone());
        // A one-split entry's own one-split launch is the reference of its
        // other split counts.
        if reference.is_none() && chain_splits.is_none() && s == 1 {
            reference = Some((bytes, symbol.clone()));
        }
        let (median_us, min_us) = graph_us(session, launches, |i, stream| call(index, i, stream))?;
        let read = (weight_bytes + scale_floats * 4) as f64
            + f64::from(m) * f64::from(query.k) * 2.0
            + (m as usize * out_n * width) as f64;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("tiled_projection_latency"));
        receipt.insert("symbol".into(), json!(symbol));
        receipt.insert("weight".into(), json!(format!("{:?}", query.weight)));
        receipt.insert("n".into(), json!(query.n));
        receipt.insert("k".into(), json!(query.k));
        receipt.insert("capacity".into(), json!(query.m_capacity));
        receipt.insert("m".into(), json!(m));
        receipt.insert(
            "tile".into(),
            json!([key.knobs.tile_m, key.knobs.tile_n, key.knobs.tile_k]),
        );
        receipt.insert("stages".into(), json!(key.knobs.stages));
        receipt.insert("k_chunks".into(), json!(chunks));
        receipt.insert("splits".into(), json!(s));
        receipt.insert("kernel".into(), json!(format!("{:?}", keys[index].dense)));
        receipt.insert("grid".into(), json!(plans[index].grid[0]));
        receipt.insert("chain_launches".into(), json!(launches));
        receipt.insert("median_us".into(), json!(median_us));
        receipt.insert("min_us".into(), json!(min_us));
        receipt.insert("bytes_read".into(), json!(read));
        receipt.insert("gb_per_second".into(), json!(read / median_us / 1e3));
        receipt.insert("elements_differing_from_one_split".into(), json!(differing));
        receipt.insert("one_split_reference".into(), json!(compared_with));
        receipt.insert(
            "inventory_symbol".into(),
            json!(entry.inventory_symbol.as_ref().unwrap_or(symbol)),
        );
        session.receipt("bench", receipt)?;
        let compared = match (differing, &compared_with) {
            (Some(count), Some(other)) => format!("{count} elements differ from {other}"),
            _ if chain_splits.is_none() && s == 1 => "the one-split reference".to_owned(),
            _ => "no one-split comparison".to_owned(),
        };
        eprintln!(
            "{symbol} {:?} n{} k{} M={m} splits={s} grid={}: {median_us:.2} us ({:.0} GB/s), {compared}",
            query.weight,
            query.n,
            query.k,
            plans[index].grid[0],
            read / median_us / 1e3
        );
    }
    Ok(())
}

/// With reports listed, the inventory has reported tiled entries at every
/// row count the timing runs: the CPU check of a GPU job's inputs.
#[test]
fn reported_tiled_entries_cover_the_timed_rows() {
    if std::env::var_os("CHOPINFER_BUNDLE_REPORTS").is_none() {
        eprintln!("SKIP: set CHOPINFER_BUNDLE_REPORTS to check the reported cubins");
        return;
    }
    let entries = tiled_entries().unwrap();
    let rows = entries
        .iter()
        .flat_map(|e| e.rows.iter().copied())
        .collect::<BTreeSet<_>>();
    for entry in &entries {
        let StaticQuery::Gemm(query) = entry.spec.key.query else {
            panic!("not a GEMM");
        };
        weight_layout(&query).unwrap();
        eprintln!(
            "{} {:?} {:?} n{} k{} capacity {} rows {:?}, inventory {}, one-split reference {}",
            entry.spec.symbol,
            entry.spec.key.dense,
            query.weight,
            query.n,
            query.k,
            query.m_capacity,
            entry.rows,
            entry.inventory_symbol.as_deref().unwrap_or("the same"),
            entry
                .reference
                .as_ref()
                .map_or("none", |(spec, _)| spec.symbol.as_str())
        );
    }
    assert_eq!(
        rows,
        BTreeSet::from_iter(DECODE_ROWS.into_iter().chain(PREFILL_ROWS))
    );
}

#[test]
fn tiled_projection_latency() {
    if !benchmarks_enabled("tiled projection latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        let entries = tiled_entries()?;
        if entries.is_empty() {
            return Err("no reported tiled entry".into());
        }
        let mut modules: BTreeMap<PathBuf, Module> = BTreeMap::new();
        for entry in &entries {
            let cubins =
                std::iter::once(&entry.cubin).chain(entry.reference.as_ref().map(|(_, path)| path));
            for cubin in cubins {
                if !modules.contains_key(cubin) {
                    let bytes = std::fs::read(cubin).map_err(err)?;
                    modules.insert(cubin.clone(), session.load_cubin(&bytes)?);
                }
            }
            let reference = entry.reference.as_ref().map(|(_, path)| &modules[path]);
            for &m in &entry.rows {
                time_tiled(session, &modules[&entry.cubin], reference, entry, m)?;
            }
        }
        Ok(())
    })
    .expect("tiled projection latency");
}

// ----------------------------------------------------- strip chains ----

/// Times every entry of strip-chain case `case` at its capacity over one
/// set of weight copies (with their block scales over FP8 weights),
/// activation, residual, output and scratch, after one checked launch each
/// (compared with the one-split output; so is the output of the timed
/// launches, from a poisoned buffer); returns each entry's label and median
/// microseconds.
fn time_strip_case(
    session: &Session,
    module: &Module,
    case: &strip_shapes::Case,
    m: u32,
    receipt_case: &str,
) -> LabResult<Vec<(String, f64)>> {
    let query = case.query;
    let (weight_bytes, scale_floats) = weight_layout(&query)?;
    let entries: Vec<(String, &kernel_core::contract::InstantiationSpec)> = [
        ("one".to_owned(), &case.one),
        ("current".to_owned(), &case.current),
    ]
    .into_iter()
    .chain(case.candidates.iter().map(|spec| {
        let fp8 = matches!(query.weight, WeightFormat::Fp8Block128x128(_));
        let label = match spec.key.dense {
            Some(DenseKernel::StripChains { strips, splits }) => {
                format!("s{strips}x{splits}")
            }
            _ if fp8 && spec.key.knobs == gemm::decoded_tile_knobs() => "decoded".to_owned(),
            _ if fp8 && spec.key.knobs == gemm::balanced_decoded_tile_knobs() => {
                "balanced".to_owned()
            }
            _ if spec.key.knobs == gemm::staged_tile_knobs() => "staged".to_owned(),
            Some(DenseKernel::TiledChains { splits: 1 }) => {
                format!("t{}x{}", spec.key.knobs.tile_m, spec.key.knobs.tile_n)
            }
            other => format!("{other:?}"),
        };
        (label, spec)
    }))
    .collect();
    let counts = |key: &kernel_core::contract::CompileKey| gemm::row_counts(key, m);
    let plans = entries
        .iter()
        .map(|(_, spec)| gemm::plan(&spec.key, counts(&spec.key)).map_err(err))
        .collect::<LabResult<Vec<LaunchPlan>>>()?;
    let workspace = plans.iter().map(|p| p.workspace_bytes).max().unwrap_or(0);
    let semaphore_bytes = entries
        .iter()
        .map(|(_, spec)| GemmBuffers::semaphore_capacity_bytes(&spec.key).map_err(err))
        .collect::<LabResult<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let mut rng = Rng(0x7f4a_7c15 ^ u64::from(query.n) << 20 ^ u64::from(query.k));
    let weights = match query.weight {
        WeightFormat::Bf16 => rng.bf16_bytes(weight_bytes / 2),
        _ => rng.e4m3_bytes(weight_bytes),
    };
    let scales = rng.scale_bytes(scale_floats);
    let launches = chain_launches((weight_bytes + scale_floats * 4) as u64);
    let copies = (0..launches)
        .map(|_| {
            Ok((
                upload(session, &weights)?,
                (scale_floats > 0)
                    .then(|| upload(session, &scales))
                    .transpose()?,
            ))
        })
        .collect::<LabResult<Vec<_>>>()?;
    // Every row of the capacity: a tile past the live rows still loads its
    // whole activation box.
    let activation = upload(
        session,
        &rng.bf16_bytes(query.m_capacity as usize * query.k as usize),
    )?;
    let width = if query.output == DType::F32 { 4 } else { 2 };
    let out_bytes = m as usize * query.n as usize * width;
    let output = zeroed(session, out_bytes)?;
    let residual = if query.epilogues.residual_add {
        Some(upload(session, &vec![0; out_bytes])?)
    } else {
        None
    };
    let partial = zeroed(session, workspace as usize)?;
    let semaphores = zeroed(session, semaphore_bytes as usize)?;
    let mut reference: Option<Vec<u8>> = None;
    let mut times = Vec::new();
    for ((label, spec), plan) in entries.iter().zip(&plans) {
        let function: Function = module.function(&spec.symbol).map_err(err)?;
        if plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(err)?;
        }
        let call = |copy: usize, stream: &runtime_cuda::Stream| -> LabResult<()> {
            let maps = gemm::tensor_maps(&spec.key, WEIGHT, ACTIVATION).map_err(err)?;
            let args = gemm::arguments(
                &spec.key,
                counts(&spec.key),
                maps,
                GemmBuffers {
                    output: OUTPUT,
                    partial: PARTIAL,
                    semaphores: SEMAPHORES,
                    weight_scales: (scale_floats > 0).then_some(SCALES),
                    activation_scales: None,
                    residual: residual.is_some().then_some(RESIDUAL),
                    weight_global_scale: None,
                    activation_global_scale: None,
                },
            )
            .map_err(err)?;
            let (weight, scale) = &copies[copy];
            let mut buffers = BTreeMap::from([
                (WEIGHT, weight),
                (ACTIVATION, &activation),
                (OUTPUT, &output),
                (PARTIAL, &partial),
                (SEMAPHORES, &semaphores),
            ]);
            if let Some(scale) = scale {
                buffers.insert(SCALES, scale);
            }
            if let Some(residual) = &residual {
                buffers.insert(RESIDUAL, residual);
            }
            launch::launch(
                &session.context,
                &function,
                stream,
                *plan,
                &args,
                &Bindings { buffers },
            )
        };
        poison::fill_all(session, &output, &[0xff])?;
        call(0, &session.stream)?;
        let bytes = download(session, &output)?;
        let differing = match &reference {
            None => 0,
            Some(old) => old
                .chunks(width)
                .zip(bytes.chunks(width))
                .filter(|(a, b)| a != b)
                .count(),
        };
        if reference.is_none() {
            reference = Some(bytes);
        }
        // The timed launches write the same output from a poisoned buffer.
        poison::fill_all(session, &output, &[0xff])?;
        let (median_us, min_us) = graph_us(session, launches, call)?;
        let timed = download(session, &output)?;
        let timed_differing = reference.as_ref().map_or(0, |want| {
            want.chunks(width)
                .zip(timed.chunks(width))
                .filter(|(a, b)| a != b)
                .count()
        });
        let read = (weight_bytes + scale_floats * 4) as f64
            + f64::from(m) * f64::from(query.k) * 2.0
            + out_bytes as f64 * if residual.is_some() { 2.0 } else { 1.0 };
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(receipt_case));
        receipt.insert("tag".into(), json!(case.tag));
        receipt.insert("label".into(), json!(label));
        receipt.insert("symbol".into(), json!(spec.symbol));
        receipt.insert("kernel".into(), json!(format!("{:?}", spec.key.dense)));
        receipt.insert("n".into(), json!(query.n));
        receipt.insert("k".into(), json!(query.k));
        receipt.insert("weight".into(), json!(format!("{:?}", query.weight)));
        receipt.insert("output".into(), json!(format!("{:?}", query.output)));
        receipt.insert("residual".into(), json!(residual.is_some()));
        receipt.insert("m".into(), json!(m));
        receipt.insert(
            "tile".into(),
            json!([
                spec.key.knobs.tile_m,
                spec.key.knobs.tile_n,
                spec.key.knobs.tile_k
            ]),
        );
        receipt.insert("stages".into(), json!(spec.key.knobs.stages));
        receipt.insert("grid".into(), json!(plan.grid[0]));
        receipt.insert("block".into(), json!(plan.block[0]));
        receipt.insert("shared_bytes".into(), json!(plan.dynamic_shared_bytes));
        receipt.insert("workspace_bytes".into(), json!(plan.workspace_bytes));
        receipt.insert("chain_launches".into(), json!(launches));
        receipt.insert("median_us".into(), json!(median_us));
        receipt.insert("min_us".into(), json!(min_us));
        receipt.insert("bytes_read".into(), json!(read));
        receipt.insert("gb_per_second".into(), json!(read / median_us / 1e3));
        receipt.insert("elements_differing_from_one_split".into(), json!(differing));
        receipt.insert(
            "timed_elements_differing_from_one_split".into(),
            json!(timed_differing),
        );
        session.receipt("bench", receipt)?;
        eprintln!(
            "{} M={m} {label} {:?} grid={} stages={}: {median_us:.2} us ({:.0} GB/s), \
             {differing} elements differ from one split",
            case.tag,
            spec.key.dense,
            plan.grid[0],
            spec.key.knobs.stages,
            read / median_us / 1e3
        );
        if differing != 0 || timed_differing != 0 {
            return Err(format!(
                "{} M={m} {label}: {differing} elements differ from one split, \
                 {timed_differing} after the timed launches",
                case.tag
            ));
        }
        times.push((label.clone(), median_us));
    }
    Ok(times)
}

#[test]
fn strip_chain_latency() {
    if !benchmarks_enabled("strip chain latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        strip_latency_summary(session, strip_shapes::cases(), strip_shapes::cubin_name)
    })
    .expect("strip chain latency");
}

/// `strip_chain_latency` over the FP8 sweep (`strip_chain_fp8_shapes.rs`).
#[test]
fn fp8_strip_chain_latency() {
    if !benchmarks_enabled("FP8 strip chain latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        strip_latency_summary(
            session,
            fp8_strip_shapes::cases(),
            fp8_strip_shapes::cubin_name,
        )
    })
    .expect("FP8 strip chain latency");
}

/// Times each of `cases` the filter keeps (`time_strip_case`, from the
/// cubin `cubin_name` gives its shape) and prints each case's fastest
/// candidate against today's entry.
fn strip_latency_summary(
    session: &Session,
    cases: Vec<strip_shapes::Case>,
    cubin_name: fn(&str) -> String,
) -> LabResult<()> {
    let filter = std::env::var("CHOPINFER_STRIP_FILTER").ok();
    let mut summary = Vec::new();
    for case in cases {
        if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
            continue;
        }
        let module = session.module(&cubin_name(case.shape))?;
        let times = time_strip_case(
            session,
            &module,
            &case,
            case.query.m_capacity,
            "strip_chain_latency",
        )?;
        let current = times
            .iter()
            .find(|(label, _)| label == "current")
            .map(|(_, us)| *us)
            .ok_or("no current time")?;
        let (best, best_us) = times
            .iter()
            .filter(|(label, _)| label.starts_with('s'))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .cloned()
            .ok_or("no candidate time")?;
        summary.push(format!(
            "{}: current {current:.2} us, best {best} {best_us:.2} us ({:.2}x)",
            case.tag,
            current / best_us
        ));
    }
    if summary.is_empty() {
        return Err("no strip-chain case ran".into());
    }
    for line in &summary {
        eprintln!("summary {line}");
    }
    Ok(())
}

/// The decoded-tile sweep (`decoded_tile_shapes.rs`, kernel-lab's own
/// cubins): each case's one-split entry, its warp-tile entry of today and
/// its decoded tiles at each of `decoded_shapes::TIMED_ROWS` live rows of
/// the prefill capacity, each output compared with the one-split output bit
/// for bit (`time_strip_case`); then the decoded tiles against today's
/// entry per case and rows. `CHOPINFER_STRIP_FILTER` keeps the cases whose
/// tag contains it.
#[test]
fn decoded_tile_latency() {
    if !benchmarks_enabled("decoded tile latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        let filter = std::env::var("CHOPINFER_STRIP_FILTER").ok();
        let mut summary = Vec::new();
        for case in decoded_shapes::cases() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let module = session.module(&decoded_shapes::cubin_name(case.shape))?;
            for m in decoded_shapes::TIMED_ROWS {
                let times = time_strip_case(session, &module, &case, m, "decoded_tile_latency")?;
                let time = |label: &str| {
                    times
                        .iter()
                        .find(|(l, _)| l == label)
                        .map(|(_, us)| *us)
                        .ok_or(format!("no {label} time"))
                };
                let (current, decoded) = (time("current")?, time("decoded")?);
                summary.push(format!(
                    "{} M={m}: current {current:.2} us, decoded {decoded:.2} us ({:.2}x)",
                    case.tag,
                    current / decoded
                ));
            }
        }
        if summary.is_empty() {
            return Err("no decoded-tile case ran".into());
        }
        for line in &summary {
            eprintln!("summary {line}");
        }
        Ok(())
    })
    .expect("decoded tile latency");
}

#[test]
fn balanced_decoded_latency() {
    if !benchmarks_enabled("balanced decoded-tile latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        let rows = match std::env::var("CHOPINFER_TIMED_ROWS") {
            Ok(list) => list
                .split(',')
                .map(|m| m.trim().parse::<u32>().map_err(err))
                .collect::<LabResult<Vec<u32>>>()?,
            Err(_) => balanced_shapes::TIMED_ROWS.to_vec(),
        };
        let filter = std::env::var("CHOPINFER_STRIP_FILTER").ok();
        let mut summary = Vec::new();
        for case in balanced_shapes::cases() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let module = session.module(&balanced_shapes::cubin_name(case.shape))?;
            for &m in &rows {
                let times =
                    time_strip_case(session, &module, &case, m, "balanced_decoded_latency")?;
                let time = |label: &str| {
                    times
                        .iter()
                        .find(|(l, _)| l == label)
                        .map(|(_, us)| *us)
                        .ok_or(format!("no {label} time"))
                };
                let (current, balanced) = (time("current")?, time("balanced")?);
                let flops = 2.0 * f64::from(m) * f64::from(case.query.n) * f64::from(case.query.k);
                summary.push(format!(
                    "{} M={m}: decoded {current:.2} us ({:.0} TFLOP/s), balanced {balanced:.2} us \
                     ({:.0} TFLOP/s, {:.3}x)",
                    case.tag,
                    flops / current / 1e6,
                    flops / balanced / 1e6,
                    current / balanced
                ));
            }
        }
        if summary.is_empty() {
            return Err("no balanced decoded-tile case ran".into());
        }
        for line in &summary {
            eprintln!("summary {line}");
        }
        Ok(())
    })
    .expect("balanced decoded tile latency");
}

/// The staged-tile sweep (`staged_tile_shapes.rs`, BF16 prefill
/// projections): each case's warp tiles of today, its staged tiles and its
/// warp tiles at the other admitted tiles, at its rows or at the
/// comma-separated `CHOPINFER_TIMED_ROWS`, each output compared with the
/// one-split output bit for bit.
#[test]
fn staged_tile_latency() {
    if !benchmarks_enabled("staged tile latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        let rows = match std::env::var("CHOPINFER_TIMED_ROWS") {
            Ok(list) => list
                .split(',')
                .map(|m| m.trim().parse::<u32>().map_err(err))
                .collect::<LabResult<Vec<u32>>>()?,
            Err(_) => staged_shapes::TIMED_ROWS.to_vec(),
        };
        let filter = std::env::var("CHOPINFER_STRIP_FILTER").ok();
        let mut summary = Vec::new();
        for case in staged_shapes::cases() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let module = session.module(&staged_shapes::cubin_name(case.shape))?;
            for &m in &rows {
                let times = time_strip_case(session, &module, &case, m, "staged_tile_latency")?;
                let flops = 2.0 * f64::from(m) * f64::from(case.query.n) * f64::from(case.query.k);
                let line = times
                    .iter()
                    .filter(|(label, _)| label != "one")
                    .map(|(label, us)| {
                        format!("{label} {us:.2} us ({:.0} TFLOP/s)", flops / us / 1e6)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                summary.push(format!("{} M={m}: {line}", case.tag));
            }
        }
        if summary.is_empty() {
            return Err("no staged tile case ran".into());
        }
        for line in &summary {
            eprintln!("summary {line}");
        }
        Ok(())
    })
    .expect("staged tile latency");
}

// ---------------------------------------------------------- chained ----

const SHAPES: chained::ServedShapes = chained::TP4_288_TOP8;

/// The served dense decode projections in FP8 128x128 blocks, each in the
/// column schedule and then in block warps.
fn chained_decode() -> Vec<Projection> {
    let mut projections = chained::catalog()
        .into_iter()
        .chain(chained::block_warp_catalog())
        .filter(|p| p.row_tile == 16 && p.weights() == Weights::Fp8Block128 && selected(&p.symbol))
        .collect::<Vec<_>>();
    projections.sort_by_key(|p| {
        (
            !p.gate_up(),
            p.columns(),
            p.schedule == Schedule::BlockWarps,
        )
    });
    projections
}

/// The cubin of a served dense chained projection, or of a lab
/// decoded-tile one.
fn chained_cubin(projection: &Projection) -> String {
    let (suffix, _) = chained::schedule_cubin(projection.schedule);
    let prefix = match projection.schedule {
        Schedule::DecodedTiles => chained_decoded_shapes::PREFIX,
        _ => chained::PREFIX,
    };
    format!("{prefix}_{}{suffix}", projection.weights().tag())
}

/// The cubin of a routed projection of every expert split over the ranks
/// (`chained::split`): the split family's served cubin.
fn routed_cubin(projection: &Projection) -> String {
    let (suffix, _) = chained::schedule_cubin(projection.schedule);
    format!("{}_{}{suffix}", split::PREFIX, projection.weights().tag())
}

/// Seeded routes of `tokens` tokens: top-k distinct experts each.
fn routes(tokens: u32, seed: u64) -> Vec<usize> {
    let mut rng = Rng(seed);
    let mut ids = Vec::new();
    for _ in 0..tokens {
        let mut chosen = BTreeSet::new();
        while chosen.len() < SHAPES.top_k as usize {
            chosen.insert(rng.next() as usize % SHAPES.experts as usize);
        }
        // A token's routes in a seeded order, as a router gives them.
        let mut picked = chosen.into_iter().collect::<Vec<_>>();
        let rotate = rng.next() as usize % picked.len();
        picked.rotate_left(rotate);
        ids.extend(picked);
    }
    ids
}

fn words(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}

/// One launch's route metadata on rank 0, and the live local experts.
struct RouteSet {
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    experts: usize,
}

fn time_chained(session: &Session, projection: &Projection, tokens: u32) -> LabResult<()> {
    time_chained_entry(
        session,
        projection,
        &chained_cubin(projection),
        &projection.symbol,
        tokens,
        512.0,
    )
}

/// Block scales 64 times smaller than [`time_chained`]'s (2^-15 to
/// 2^-13): gate/up sums stay within a few units, as served activations and
/// weights give them, far from the SiLU tail below -80 that the epilogue
/// evaluates in F64 (the larger scales send many random sums there, which
/// then dominates a gate/up's time).
const QUIET_SCALE_DIVISOR: f32 = 32768.0;

/// [`time_chained`] of `projection`'s launch with the entry `symbol` of
/// `cubin` (a lab variant of its kernel), block scales from
/// `Rng::scale_bytes_over(_, scale_divisor)`.
fn time_chained_entry(
    session: &Session,
    projection: &Projection,
    cubin: &str,
    symbol: &str,
    tokens: u32,
    scale_divisor: f32,
) -> LabResult<()> {
    let gemm = projection.query.gemm;
    let (n, k) = (gemm.n as usize, gemm.k as usize);
    let mut rng = Rng(0x2545_f491 ^ (n as u64) << 16 ^ k as u64);
    // One set of weights, copied for every launch: the times do not depend
    // on the values, and each launch reads its own copy.
    let codes = rng.e4m3_bytes(n * k);
    let scales = rng.scale_bytes_over((n / 128) * (k / 128), scale_divisor);
    let launches = chain_launches((n * k + scales.len()) as u64);
    let banks = (0..launches)
        .map(|_| Ok((upload(session, &codes)?, upload(session, &scales)?)))
        .collect::<LabResult<Vec<_>>>()?;
    let capacity = projection.tokens_capacity as usize;
    let input = upload(session, &rng.bf16_bytes(capacity * k))?;
    let output = zeroed(session, capacity * projection.columns() as usize * 2)?;
    let live = upload(session, &words([tokens as i32]))?;
    let function = session.module(cubin)?.function(symbol).map_err(err)?;
    let plan = projection.plan(tokens).map_err(err)?;
    if plan.dynamic_shared_bytes > 0 {
        function
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(err)?;
    }
    let handles: [u64; 11] = std::array::from_fn(|i| i as u64 + 1);
    let mut used = handles;
    used[3] = 0; // local scales
    used[4] = 0; // globals
    used[5] = 0; // route offsets
    used[6] = 0; // sorted rows
    used[7] = 0; // padded rows
    used[9] = 0; // route weights
    let args = projection.arguments(used).map_err(err)?;
    let call = |i: usize, stream: &runtime_cuda::Stream| -> LabResult<()> {
        let (codes, scales) = &banks[i];
        let buffers = BTreeMap::from([
            (1, &input),
            (2, codes),
            (3, scales),
            (9, &live),
            (11, &output),
        ]);
        launch::launch(
            &session.context,
            &function,
            stream,
            plan,
            &args,
            &Bindings { buffers },
        )
    };
    call(0, &session.stream)?;
    session.stream.synchronize().map_err(err)?;
    let (median_us, min_us) = graph_us(session, launches, call)?;
    let read = (n * k + scales.len() * 4) as f64;
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("chained_expert_latency"));
    receipt.insert("symbol".into(), json!(symbol));
    receipt.insert(
        "schedule".into(),
        json!(format!("{:?}", projection.schedule)),
    );
    receipt.insert("gate_up".into(), json!(projection.gate_up()));
    receipt.insert("n".into(), json!(n));
    receipt.insert("k".into(), json!(k));
    receipt.insert("tokens".into(), json!(tokens));
    receipt.insert("scale_divisor".into(), json!(scale_divisor));
    receipt.insert("grid".into(), json!(plan.grid));
    receipt.insert("chain_launches".into(), json!(launches));
    receipt.insert("median_us".into(), json!(median_us));
    receipt.insert("min_us".into(), json!(min_us));
    receipt.insert("weight_bytes_read".into(), json!(read));
    receipt.insert("gb_per_second".into(), json!(read / median_us / 1e3));
    session.receipt("bench", receipt)?;
    eprintln!(
        "{symbol} n{n} k{k} tokens={tokens} scales/{scale_divisor} grid={:?}: {median_us:.2} us ({:.0} GB/s)",
        plan.grid,
        read / median_us / 1e3
    );
    Ok(())
}

/// Runs without a GPU: the timed chained projections are built.
#[test]
fn chained_decode_projections_are_built() {
    let projections = chained_decode();
    assert!(!projections.is_empty());
    assert!(projections.iter().all(|p| p.query.gemm.grouped.is_none()));
    for projection in projections {
        let cubin = chained_cubin(&projection);
        assert!(
            kernel_lab::cubin_entries(&cubin)
                .unwrap()
                .contains(&projection.symbol.as_str()),
            "{}",
            projection.symbol
        );
    }
}

#[test]
fn chained_expert_latency() {
    if !benchmarks_enabled("chained expert latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        for projection in chained_decode() {
            for tokens in DECODE_ROWS {
                time_chained(session, &projection, tokens)?;
            }
        }
        Ok(())
    })
    .expect("chained expert latency");
}

/// The prefill row counts the decoded-tile gate/up is timed at: a part
/// below one tile, a few tiles, a 2,048-token chunk and the capacity.
const DECODED_GATE_UP_ROWS: [u32; 4] = [64, 512, 2048, 8192];

/// Runs without a GPU: the timed decoded-tile gate/up projections and
/// their column references are built.
#[test]
fn chained_decoded_projections_are_built() {
    for projection in chained_decoded_shapes::references()
        .into_iter()
        .chain(chained_decoded_shapes::projections())
    {
        let cubin = chained_cubin(&projection);
        assert!(
            kernel_lab::cubin_entries(&cubin)
                .unwrap()
                .contains(&projection.symbol.as_str()),
            "{}",
            projection.symbol
        );
    }
}

/// The dense MLP's and the shared expert's FP8 prefill gate/up in the
/// column schedule, then in decoded tiles, at each of
/// `DECODED_GATE_UP_ROWS` live tokens (`time_chained_entry`), with quiet
/// block scales (`QUIET_SCALE_DIVISOR`, the served regime) and then with
/// `time_chained`'s.
#[test]
fn chained_decoded_latency() {
    if !benchmarks_enabled("chained decoded-tile latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        for (reference, candidate) in chained_decoded_shapes::references()
            .into_iter()
            .zip(chained_decoded_shapes::projections())
        {
            for tokens in DECODED_GATE_UP_ROWS {
                for divisor in [QUIET_SCALE_DIVISOR, 512.0] {
                    for projection in [&reference, &candidate] {
                        if selected(&projection.symbol) {
                            time_chained_entry(
                                session,
                                projection,
                                &chained_cubin(projection),
                                &projection.symbol,
                                tokens,
                                divisor,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
    .expect("chained decoded-tile latency");
}

// ------------------------------------------------- split experts ----

/// One timed routed case: per launch, the routes (global expert IDs,
/// `[tokens, top_k]`) and the rank whose local experts run, of
/// `local_experts` consecutive experts per rank: rank 0 of all the experts,
/// the split's.
#[derive(Clone)]
struct RoutedCase {
    name: String,
    tokens: u32,
    routes: Vec<(Vec<usize>, usize)>,
    local_experts: usize,
}

/// The split's rank-0 routes of each captured layer (or every `step`-th
/// layer) from `rows`.
fn captured_case(
    name: String,
    tokens: u32,
    layers: &[split_tile_pairs::CapturedLayer],
    step: usize,
    rows: impl Fn(usize) -> Vec<i32>,
) -> RoutedCase {
    RoutedCase {
        name,
        tokens,
        local_experts: split::SHAPES.local_experts as usize,
        routes: (0..layers.len())
            .step_by(step)
            .map(|i| (rows(i).into_iter().map(|id| id as usize).collect(), 0))
            .collect(),
    }
}

/// The prefill cases: uniform seeded routes on rank 0 at 2,048 and 8,192
/// tokens (eight launches), then, with a capture, every captured layer's
/// first 2,048 and 4,096 tokens, and 8,192 tokens of every other layer
/// over it and the next and of it repeated.
fn tile_pair_cases() -> LabResult<Vec<RoutedCase>> {
    let mut cases = Vec::from([2048, 8192].map(|tokens| {
        RoutedCase {
            name: format!("uniform {tokens}"),
            tokens,
            local_experts: split::SHAPES.local_experts as usize,
            routes: (0..8)
                .map(|i| (routes(tokens, 0x5eed_0000 + i), 0))
                .collect(),
        }
    }));
    if let Some(layers) = split_tile_pairs::captured_layers()? {
        for tokens in [2048, 4096] {
            cases.push(captured_case(
                format!("captured {tokens}"),
                tokens,
                &layers,
                1,
                |i| split_tile_pairs::adjacent_rows(&layers, i, tokens as usize),
            ));
        }
        cases.push(captured_case(
            "captured 8192 over two layers".into(),
            8192,
            &layers,
            2,
            |i| split_tile_pairs::adjacent_rows(&layers, i, 8192),
        ));
        cases.push(captured_case(
            "captured 8192 repeated".into(),
            8192,
            &layers,
            2,
            |i| split_tile_pairs::repeated_rows(&layers, i, 8192),
        ));
    } else {
        eprintln!(
            "{} unset: no captured prefill cases",
            split_tile_pairs::ROUTE_CAPTURE_ENV
        );
    }
    Ok(cases)
}

/// Decode cases shaped like served batches, with a capture: `requests`
/// requests of `rows` consecutive captured tokens each (request i from token
/// 256 + 224 i, all inside one layer's capture) in every captured layer, on
/// rank 0 of the split. AR at 1, 4, 8 and 16 requests; MTP verify (four rows
/// per request) at 1, 4 and 8.
fn served_decode_cases() -> LabResult<Vec<RoutedCase>> {
    let Some(layers) = split_tile_pairs::captured_layers()? else {
        eprintln!(
            "{} unset: no served decode cases",
            split_tile_pairs::ROUTE_CAPTURE_ENV
        );
        return Ok(Vec::new());
    };
    let top_k = SHAPES.top_k as usize;
    let shapes = [
        ("AR", 1, 1),
        ("AR", 4, 1),
        ("AR", 8, 1),
        ("AR", 16, 1),
        ("MTP", 1, 4),
        ("MTP", 4, 4),
        ("MTP", 8, 4),
    ];
    Ok(shapes
        .map(|(mode, requests, rows)| {
            captured_case(
                format!("{mode} C{requests} ({} rows)", requests * rows),
                (requests * rows) as u32,
                &layers,
                1,
                |i| {
                    (0..requests)
                        .flat_map(|r| {
                            let first = (256 + 224 * r) * top_k;
                            layers[i].ids[first..first + rows * top_k].to_vec()
                        })
                        .collect()
                },
            )
        })
        .into())
}

/// The buffers of one routed case for one column projection, shared by its
/// candidates: weight banks, input, output, live tokens, route weights and
/// each launch's route metadata.
struct RoutedSetup {
    banks: Vec<(DeviceBuffer, DeviceBuffer)>,
    input: DeviceBuffer,
    output: DeviceBuffer,
    live: DeviceBuffer,
    route_weights: DeviceBuffer,
    sets: Vec<RouteSet>,
    padded_rows: Vec<i32>,
}

fn routed_setup(
    session: &Session,
    reference: &Projection,
    case: &RoutedCase,
) -> LabResult<RoutedSetup> {
    let gemm = reference.query.gemm;
    let (n, k) = (gemm.n as usize, gemm.k as usize);
    let experts = case.local_experts;
    let top_k = SHAPES.top_k as usize;
    let mut rng = Rng(0x7f4a_7c15 ^ (n as u64) << 16 ^ k as u64);
    // One expert's weights, repeated over every expert: the times do not
    // depend on the values. Two banks hold enough distinct experts.
    let codes = rng.e4m3_bytes(n * k);
    let scales = rng.scale_bytes((n / 128) * (k / 128));
    let banks = (0..2)
        .map(|_| {
            Ok((
                upload(session, &codes.repeat(experts))?,
                upload(session, &scales.repeat(experts))?,
            ))
        })
        .collect::<LabResult<Vec<_>>>()?;
    let tokens = case.tokens as usize;
    let input_rows = if reference.gate_up() {
        tokens
    } else {
        tokens * top_k
    };
    let out_width = if reference.gate_up() { 2 } else { 4 };
    let mut sets = Vec::new();
    let mut padded_rows = Vec::new();
    for (ids, rank) in &case.routes {
        let metadata =
            local_route_metadata(ids, top_k, experts, *rank, reference.row_tile as usize)
                .map_err(err)?;
        padded_rows.push(metadata.num_padded_rows);
        sets.push(RouteSet {
            offsets: upload(session, &words(metadata.offsets.iter().copied()))?,
            sorted: upload(session, &words(metadata.sorted_rows.iter().copied()))?,
            padded: upload(session, &words([metadata.num_padded_rows]))?,
            experts: metadata.counts.iter().filter(|c| **c > 0).count(),
        });
    }
    Ok(RoutedSetup {
        banks,
        input: upload(session, &rng.bf16_bytes(input_rows * k))?,
        output: zeroed(
            session,
            tokens * top_k * reference.columns() as usize * out_width,
        )?,
        live: upload(session, &words([case.tokens as i32]))?,
        route_weights: upload(
            session,
            &(0..tokens * top_k)
                .flat_map(|_| 0.125_f32.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        sets,
        padded_rows,
    })
}

/// `projection`'s per-launch time over `setup` (the median of `graph_us`),
/// and its first launch's whole output from a poisoned buffer.
fn time_routed(
    session: &Session,
    projection: &Projection,
    case: &RoutedCase,
    setup: &RoutedSetup,
) -> LabResult<(f64, Vec<u8>)> {
    let gemm = projection.query.gemm;
    let plan = projection.plan(case.tokens).map_err(err)?;
    let function = session
        .module(&routed_cubin(projection))?
        .function(&projection.symbol)
        .map_err(err)?;
    let handles: [u64; 11] = std::array::from_fn(|i| i as u64 + 1);
    let mut used = handles;
    used[3] = 0; // local scales
    used[4] = 0; // globals
    used[8] = 0; // live tokens
    if projection.gate_up() {
        used[9] = 0;
    }
    let args = projection.arguments(used).map_err(err)?;
    let call = |i: usize, stream: &runtime_cuda::Stream| -> LabResult<()> {
        let (codes, scales) = &setup.banks[i % setup.banks.len()];
        let set = &setup.sets[i];
        let buffers = BTreeMap::from([
            (1, &setup.input),
            (2, codes),
            (3, scales),
            (6, &set.offsets),
            (7, &set.sorted),
            (8, &set.padded),
            (9, &setup.live),
            (10, &setup.route_weights),
            (11, &setup.output),
        ]);
        launch::launch(
            &session.context,
            &function,
            stream,
            plan,
            &args,
            &Bindings { buffers },
        )
    };
    poison::fill_all(session, &setup.output, &[poison::POISON])?;
    call(0, &session.stream)?;
    session.stream.synchronize().map_err(err)?;
    let first = download(session, &setup.output)?;
    let (median_us, min_us) = graph_us(session, setup.sets.len(), call)?;
    let (n, k) = (gemm.n as f64, gemm.k as f64);
    let launches = setup.sets.len() as f64;
    let rows = case
        .routes
        .iter()
        .map(|(ids, rank)| {
            ids.iter()
                .filter(|&&id| id / case.local_experts == *rank)
                .count()
        })
        .sum::<usize>() as f64
        / launches;
    let padded = setup.padded_rows.iter().map(|&r| f64::from(r)).sum::<f64>() / launches;
    let live_experts = setup.sets.iter().map(|s| s.experts).sum::<usize>() as f64 / launches;
    let useful_tflops = 2.0 * rows * n * k / median_us / 1e6;
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("split_expert_latency"));
    receipt.insert("routing".into(), json!(case.name));
    receipt.insert("local_experts".into(), json!(case.local_experts));
    receipt.insert("symbol".into(), json!(projection.symbol));
    receipt.insert(
        "schedule".into(),
        json!(format!("{:?}", projection.schedule)),
    );
    receipt.insert("gate_up".into(), json!(projection.gate_up()));
    receipt.insert("n".into(), json!(gemm.n));
    receipt.insert("k".into(), json!(gemm.k));
    receipt.insert("row_tile".into(), json!(projection.row_tile));
    receipt.insert("tokens".into(), json!(case.tokens));
    receipt.insert("grid".into(), json!(plan.grid));
    receipt.insert("block".into(), json!(plan.block));
    receipt.insert("launches".into(), json!(setup.sets.len()));
    receipt.insert("rows".into(), json!(rows));
    receipt.insert("padded_rows".into(), json!(padded));
    receipt.insert("live_local_experts".into(), json!(live_experts));
    receipt.insert("median_us".into(), json!(median_us));
    receipt.insert("min_us".into(), json!(min_us));
    receipt.insert("useful_tflops".into(), json!(useful_tflops));
    session.receipt("bench", receipt)?;
    eprintln!(
        "{} {} grid={:?}: {median_us:.1} us ({useful_tflops:.1} useful TFLOP/s, {rows:.0} rows, {padded:.0} padded, {live_experts:.1} experts)",
        case.name, projection.symbol, plan.grid
    );
    Ok((median_us, first))
}

/// The F32 route sum's per-launch time over `case` (one launch per layer)
/// from `prefix`'s route cubin, at `case`'s experts per rank, and the routed
/// F32 bytes it reads per launch.
fn time_route_sum(session: &Session, prefix: &str, case: &RoutedCase) -> LabResult<(f64, f64)> {
    use kernel_core::contract::{Argument, ArgumentList};
    let sum = chained::RoutedSum {
        local_experts: case.local_experts as u32,
        ..SHAPES.sums()[0]
    }
    .with_output(chained::RoutedSumOutput::F32);
    let function = session
        .module(&format!("{prefix}_route"))?
        .function(&sum.symbol(prefix))
        .map_err(err)?;
    let plan = sum.plan(case.tokens).map_err(err)?;
    let tokens = case.tokens as usize;
    let top_k = SHAPES.top_k as usize;
    let width = SHAPES.hidden as usize;
    let routed = zeroed(session, tokens * top_k * width * 4)?;
    let output = zeroed(session, tokens * width * 4)?;
    let live = upload(session, &words([case.tokens as i32]))?;
    let ids = case
        .routes
        .iter()
        .map(|(ids, _)| upload(session, &words(ids.iter().map(|&id| id as i32))))
        .collect::<LabResult<Vec<_>>>()?;
    let calls = case
        .routes
        .iter()
        .map(|(_, rank)| {
            ArgumentList(vec![
                Argument::DeviceBuffer(1),
                Argument::DeviceBuffer(2),
                Argument::DeviceBuffer(3),
                Argument::U32(*rank as u32),
                Argument::DeviceBuffer(4),
            ])
        })
        .collect::<Vec<_>>();
    let call = |i: usize, stream: &runtime_cuda::Stream| -> LabResult<()> {
        let buffers = BTreeMap::from([(1, &routed), (2, &ids[i]), (3, &live), (4, &output)]);
        launch::launch(
            &session.context,
            &function,
            stream,
            plan,
            &calls[i],
            &Bindings { buffers },
        )
    };
    let (median_us, _) = graph_us(session, case.routes.len(), call)?;
    let local_routes = case
        .routes
        .iter()
        .map(|(ids, rank)| {
            ids.iter()
                .filter(|&&id| id / case.local_experts == *rank)
                .count()
        })
        .sum::<usize>() as f64
        / case.routes.len() as f64;
    Ok((median_us, local_routes * (width * 4) as f64))
}

/// The served FP8 split forms of the gate/up or down at `row_tile`
/// (`split::forms`), each timed at every count of the phase's capacity,
/// whatever rows the served form takes.
fn split_forms(gate_up: bool, row_tile: u32) -> Vec<Projection> {
    let capacity = match row_tile {
        16 => split::SHAPES.decode_tokens,
        _ => split::SHAPES.prefill_tokens,
    };
    split::forms()
        .into_iter()
        .filter(|p| {
            p.weights() == Weights::Fp8Block128 && p.gate_up() == gate_up && p.row_tile == row_tile
        })
        .map(|mut p| {
            p.tokens_capacity = capacity;
            p
        })
        .collect()
}

/// Runs without a GPU: every served FP8 split form is in its cubin.
#[test]
fn split_expert_projections_are_built() {
    for gate_up in [true, false] {
        for row_tile in [16, 64] {
            let forms = split_forms(gate_up, row_tile);
            assert!(!forms.is_empty());
            for projection in forms {
                let cubin = routed_cubin(&projection);
                assert!(
                    kernel_lab::cubin_entries(&cubin)
                        .unwrap()
                        .contains(&projection.symbol.as_str()),
                    "{} in {cubin}",
                    projection.symbol
                );
            }
        }
    }
}

/// Every served FP8 split form of `split_forms` (`chained::split`) over
/// the uniform and captured prefill cases (`tile_pair_cases`) and, with a
/// capture, the served decode batches (`served_decode_cases`), each first
/// launch compared with the first form's bit for bit, and the F32 route
/// sum. A summary line per case gives each form's time and the sum's.
#[test]
fn split_expert_latency() {
    if !benchmarks_enabled("split expert latency") {
        return;
    }
    gpu_test(40 << 30, |session| {
        let mut cases = tile_pair_cases()?;
        let prefill = cases.len();
        cases.extend(served_decode_cases()?);
        let mut differing = Vec::new();
        for (index, case) in cases.iter().enumerate() {
            let row_tile = if index < prefill { 64 } else { 16 };
            let mut split_us = BTreeMap::new();
            for gate_up in [true, false] {
                let forms = split_forms(gate_up, row_tile);
                let setup = routed_setup(session, &forms[0], case)?;
                let (time, want) = time_routed(session, &forms[0], case, &setup)?;
                split_us.insert(forms[0].symbol.clone(), time);
                for form in &forms[1..] {
                    let (time, got) = time_routed(session, form, case, &setup)?;
                    if got != want {
                        let at = want.iter().zip(&got).position(|(a, b)| a != b);
                        differing.push(format!("{} {}: byte {at:?}", case.name, form.symbol));
                    }
                    split_us.insert(form.symbol.clone(), time);
                }
            }
            let (split_sum, split_sum_bytes) = time_route_sum(session, split::PREFIX, case)?;
            let mut receipt = BTreeMap::new();
            receipt.insert("case".into(), json!("split_expert_latency"));
            receipt.insert("routing".into(), json!(case.name));
            receipt.insert("tokens".into(), json!(case.tokens));
            receipt.insert("launches".into(), json!(case.routes.len()));
            receipt.insert("split_us".into(), json!(split_us));
            receipt.insert("split_sum_us".into(), json!(split_sum));
            receipt.insert("split_sum_bytes".into(), json!(split_sum_bytes));
            session.receipt("bench", receipt)?;
            eprintln!(
                "split summary {}: sum {split_sum:.1} us; {}",
                case.name,
                split_us
                    .iter()
                    .map(|(symbol, us)| format!("{symbol} {us:.1}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if differing.is_empty() {
            Ok(())
        } else {
            Err(format!("split forms differ from each other: {differing:?}"))
        }
    })
    .expect("split expert latency");
}
