//! Chain-split tiled GEMM entries (`DenseKernel::TiledChains` and
//! `StripChains`) against the one-split tiled entries they replace, bit for
//! bit, from the cubins their bundles build.
//!
//! Each chain-split offer of the catalogs' projection inventory is paired
//! with the offer the shared generator gives the same operation and use at
//! `Tiled { splits: 1 }` (`generated::comparison::operation` of its
//! capability). The candidate's cubin comes from a current bundle report
//! (`CHOPINFER_BUNDLE_REPORTS`), the reference's from a report of the
//! bundle before the change (`CHOPINFER_REFERENCE_REPORTS`), each checked
//! against its lock and sources; paths are separated by `:`.
//!
//! Above 32 rows a chain entry runs one split in warp tiles
//! (`tiled_warps.cuh`), or over FP8 128x128 weights at
//! `gemm::decoded_tile_knobs` in decoded weight tiles (`decoded_tiles.cuh`);
//! its reference is the one-split prefill entry.
//!
//! - CPU: regenerating every one-split tiled offer reproduces its symbol, so
//!   the pairing names the served reference; with reports listed, every
//!   pair resolves and every other tiled offer is in the reference report
//!   under its symbol (its bytes are unchanged).
//! - GPU: at every live count from 1 through a decode capacity (prefill:
//!   `bitwise_counts`, with the reference run twice and compared with
//!   itself), both entries read the same weights (BF16, or FP8
//!   128x128 codes with block scales over 24 binades), activations and
//!   residual (signed, about 30 binades, some zero rows), from outputs
//!   filled with poison, and the candidate's
//!   whole output must equal the reference's bit for bit. At
//!   `poison::live_counts`, the candidate runs eager, relaunched, from a
//!   captured graph and behind a PDL producer, with its output's write map
//!   (the live rows) checked and its chain workspace poisoned before every
//!   launch. The one-split reference's accumulation tests
//!   (`gemm::accumulation`) carry over through the bitwise equality.
//!
//! The strip-chain lab sweeps (`strip_chain_shapes.rs` over BF16 weights,
//! `strip_chain_fp8_shapes.rs` over FP8 128x128 weights) run the same GPU
//! checks from kernel-lab's own cubins: each case's entry of today and every
//! strip-chain candidate against the case's one-split entry, and their
//! mutants as the chain-split mutants.
//!
//! The decoded-tile lab sweep (`decoded_tile_shapes.rs`, FP8 128x128
//! prefill projections at a prefill capacity) runs the same checks: each
//! case's warp-tile entry of today and its decoded tiles against the
//! one-split entry at `bitwise_counts` of both tile heights, the one-split
//! entry run twice, then `qualify` at `poison::live_counts`, and the
//! mutants. At every count serving can launch them at, 1 through the
//! capacity (`decoded_serving_counts`), the decoded tiles pass `qualify`'s
//! phases with the comparisons on the device (`qualify_on_device`,
//! `kernel_lab::device_check`), which observe every mutant as well. The
//! balanced decoded-tile sweep (`decoded_balanced_shapes.rs`, the same
//! shapes, one item per CTA with the final partial wave split into 64-row
//! halves) runs every one of these checks too, beside the decoded tiles of
//! today.
//!
//! Run: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_BUNDLE_REPORTS=<current>
//! CHOPINFER_REFERENCE_REPORTS=<before> cargo test -p kernel-lab --release
//! --test tiled_chain_splits`. `CHOPINFER_CHAIN_PAIRS=strip` keeps the
//! pairs whose candidate is a strip-chain entry; `CHOPINFER_STRIP_FILTER`
//! keeps the sweep cases whose tag contains it.
//!
//! Bucket rows (`bucket_entries_keep_the_widest_rows`): a bundle may score
//! rows at a smaller decode or verify bucket than before only if the
//! bucket's tiled entry computes each row as the operation's widest graph
//! bucket entry does. Each such pair of the inventory (same capability but
//! for the bucket; both entries from the current reports, optionally only
//! the `NxK` GEMM shapes `CHOPINFER_BUCKET_SHAPES` lists, comma-separated)
//! runs the widest entry over all its rows, twice, and the bucket's entry at
//! every live count of its bucket (pairs of weights the fixture does not
//! build, NVFP4, are excluded and named; each listed shape must keep a
//! checked pair): each live row equals the widest entry's
//! row bit for bit, and the bucket entry writes nothing else. At
//! `poison::live_counts`, the bucket entry passes `qualify` against those
//! rows.

use kernel_core::contract::{
    CompileKnobs, DType, DenseKernel, GemmQuery, InstantiationSpec, StaticQuery,
};
use kernel_core::contract::{LaunchPlan, WeightFormat};
use kernel_core::precision::capability::OperationCapability;
use kernel_core::precision::format::registry;
use kernel_lab::device_check::{self, DeviceCheck};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Replay, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::gemm::{self, GemmBuffers};
use kernels_sm120::precision::generated::{self, Implementation, Request};
use kernels_sm120::precision::{Compute, KernelOffer};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[path = "../chain_mutant_shapes.rs"]
mod chain_mutant_shapes;

use kernel_lab::families::decoded_balanced::shapes as balanced_shapes;
use kernel_lab::families::decoded_tiles::shapes as decoded_shapes;
use kernel_lab::families::staged_tiles::shapes as staged_shapes;
use kernel_lab::families::strip_chains::shapes as strip_shapes;
use kernel_lab::families::strip_chains_fp8::shapes as fp8_strip_shapes;

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

/// Every cubin the reports in `variable` build, with its entry symbols.
fn reported(variable: &str) -> LabResult<Vec<(PathBuf, Vec<String>)>> {
    let reports = std::env::var_os(variable)
        .ok_or_else(|| format!("set {variable} to kernel-build reports"))?;
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

fn gemm_query(spec: &InstantiationSpec) -> LabResult<GemmQuery> {
    match spec.key.query {
        StaticQuery::Gemm(query) => Ok(query),
        _ => Err(format!("{} is not a GEMM", spec.symbol)),
    }
}

/// The rows `spec`'s GEMM is compiled for.
fn m_capacity(spec: &InstantiationSpec) -> LabResult<u32> {
    gemm_query(spec).map(|query| query.m_capacity)
}

fn dense(offer: &KernelOffer) -> Option<&InstantiationSpec> {
    match &offer.handle().compute {
        Compute::Dense(spec) => Some(spec),
        _ => None,
    }
}

/// The offer the shared generator gives `offer`'s operation and use with
/// `kernel` at its capacity and `knobs` (`None`: the defaults).
fn regenerate(
    offer: &KernelOffer,
    kernel: DenseKernel,
    knobs: Option<CompileKnobs>,
) -> LabResult<InstantiationSpec> {
    let spec = dense(offer).ok_or("not a dense offer")?;
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
                knobs,
            },
        },
        &[],
    )?;
    dense(&generated.offer)
        .cloned()
        .ok_or_else(|| "regenerated offer is not dense".into())
}

/// A chain-split candidate and its one-split reference.
struct Pair {
    candidate: InstantiationSpec,
    reference: InstantiationSpec,
}

/// Every chain-split and strip-chain offer of the inventory, once per
/// candidate symbol.
fn pairs() -> LabResult<Vec<Pair>> {
    let (offers, _, _) = kernel_lab::models::projection_inventory()?;
    let mut pairs = BTreeMap::new();
    for offer in &offers {
        let Some(spec) = dense(offer) else { continue };
        if !matches!(
            spec.key.dense,
            Some(DenseKernel::TiledChains { .. } | DenseKernel::StripChains { .. })
        ) || pairs.contains_key(&spec.symbol)
        {
            continue;
        }
        let reference = regenerate(offer, DenseKernel::Tiled { splits: 1 }, None)?;
        pairs.insert(
            spec.symbol.clone(),
            Pair {
                candidate: spec.clone(),
                reference,
            },
        );
    }
    Ok(pairs.into_values().collect())
}

/// Whether `spec` runs prefill tiles at their own knobs, which a regenerated
/// offer must request: decoded weight tiles (`gemm::decoded_tile_knobs`),
/// balanced decoded tiles (`gemm::balanced_decoded_tile_knobs`, over FP8
/// 128x128 weights; BF16 warp tiles take the same knobs by default) or
/// staged tiles (`gemm::staged_tile_knobs`, over BF16 weights).
fn own_prefill_tiles(spec: &InstantiationSpec) -> bool {
    let Ok(query) = gemm_query(spec) else {
        return false;
    };
    let fp8 = matches!(query.weight, WeightFormat::Fp8Block128x128(_));
    let knobs = spec.key.knobs;
    knobs == gemm::decoded_tile_knobs()
        || (fp8 && knobs == gemm::balanced_decoded_tile_knobs())
        || (query.weight == WeightFormat::Bf16 && knobs == gemm::staged_tile_knobs())
}

/// Runs without a GPU: the regeneration that names each reference
/// reproduces every served one-split and chain-split tiled offer under its
/// generated name (offers under an adopted legacy name keep their ledger's
/// symbol and are left out), at the default knobs or, for prefill tiles at
/// their own knobs (`own_prefill_tiles`), at those.
#[test]
fn regeneration_reproduces_served_tiled_offers() {
    let (offers, _, _) = kernel_lab::models::projection_inventory().unwrap();
    let mut checked = 0;
    for offer in &offers {
        let Some(spec) = dense(offer) else { continue };
        let Some(
            kernel @ (DenseKernel::Tiled { .. }
            | DenseKernel::TiledChains { .. }
            | DenseKernel::StripChains { .. }),
        ) = spec.key.dense
        else {
            continue;
        };
        if !spec.symbol.starts_with("chop_proj_") {
            continue;
        }
        let own = own_prefill_tiles(spec);
        let again = regenerate(offer, kernel, own.then_some(spec.key.knobs)).unwrap();
        assert_eq!(again.symbol, spec.symbol);
        assert_eq!(again.key, spec.key);
        checked += 1;
    }
    assert!(checked > 0);
    for pair in pairs().unwrap() {
        let own = own_prefill_tiles(&pair.candidate);
        let (StaticQuery::Gemm(a), StaticQuery::Gemm(b)) =
            (pair.candidate.key.query, pair.reference.key.query)
        else {
            panic!("not a GEMM");
        };
        assert_eq!(a, b, "{}", pair.candidate.symbol);
        let mut knobs = pair.reference.key.knobs;
        knobs.split_k_capacity = pair.candidate.key.knobs.split_k_capacity;
        if let Some(DenseKernel::StripChains { strips, .. }) = pair.candidate.key.dense {
            // Strip chains keep the token tile and 64-K chunks in their own
            // strips, warps and ring.
            let got = pair.candidate.key.knobs;
            assert_eq!(
                (got.tile_m, got.tile_n, got.tile_k),
                (16 * u16::from(strips), knobs.tile_n, knobs.tile_k),
                "{}",
                pair.candidate.symbol
            );
        } else if own {
            // Decoded, balanced decoded or staged tiles: FP8 128x128 or BF16
            // prefill chains in their own tiles and 64-K stages.
            assert!(
                matches!(
                    a.weight,
                    WeightFormat::Fp8Block128x128(_) | WeightFormat::Bf16
                ) && a.m_capacity > 32,
                "{}",
                pair.candidate.symbol
            );
        } else {
            assert_eq!(knobs, pair.candidate.key.knobs, "{}", pair.candidate.symbol);
        }
    }
}

/// With reports listed, every pair resolves, and every one-split tiled
/// offer is in the reference report under its symbol.
#[test]
fn reported_pairs_resolve() {
    if std::env::var_os("CHOPINFER_BUNDLE_REPORTS").is_none()
        || std::env::var_os("CHOPINFER_REFERENCE_REPORTS").is_none()
    {
        eprintln!("SKIP: set CHOPINFER_BUNDLE_REPORTS and CHOPINFER_REFERENCE_REPORTS");
        return;
    }
    let current = reported("CHOPINFER_BUNDLE_REPORTS").unwrap();
    let before = reported("CHOPINFER_REFERENCE_REPORTS").unwrap();
    let pairs = pairs().unwrap();
    assert!(!pairs.is_empty());
    for pair in &pairs {
        assert!(
            cubin_of(&current, &pair.candidate.symbol).is_some(),
            "{}",
            pair.candidate.symbol
        );
        assert!(
            cubin_of(&before, &pair.reference.symbol).is_some(),
            "{}",
            pair.reference.symbol
        );
    }
    let (offers, _, _) = kernel_lab::models::projection_inventory().unwrap();
    let mut unchanged = 0;
    for offer in &offers {
        let Some(spec) = dense(offer) else { continue };
        if matches!(spec.key.dense, Some(DenseKernel::Tiled { .. }))
            && cubin_of(&current, &spec.symbol).is_some()
        {
            assert!(cubin_of(&before, &spec.symbol).is_some(), "{}", spec.symbol);
            unchanged += 1;
        }
    }
    eprintln!(
        "{} chain-split pairs, {unchanged} unchanged one-split uses",
        pairs.len()
    );
}

// ---------------------------------------------------------------- GPU ----

/// Deterministic values.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// Signed BF16 values with exponents -15 through 14; `zero_rows` of
    /// every 7 rows of `width` are zero.
    fn bf16_bytes(&mut self, count: usize, width: usize, zero_rows: bool) -> Vec<u8> {
        (0..count)
            .flat_map(|index| {
                let bits = self.next();
                let value = if zero_rows && (index / width.max(1)) % 7 == 6 {
                    0
                } else {
                    (bits as u16 & 0x807f) | ((112 + (bits >> 16) % 30) as u16) << 7
                };
                value.to_le_bytes()
            })
            .collect()
    }

    /// E4M3 codes other than the NaN codes.
    fn e4m3_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .map(|_| match self.next() as u8 {
                0x7f | 0xff => 0,
                code => code,
            })
            .collect()
    }

    /// Positive F32 scales over 2^-20 through 2^3, mantissas included.
    fn scale_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|_| {
                let bits = self.next();
                let exponent = (127 - 20 + bits % 24) << 23;
                f32::from_bits(exponent | (bits >> 9 & 0x7f_ffff)).to_le_bytes()
            })
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

fn function(module: &Module, symbol: &str, shared: u32) -> LabResult<Function> {
    let function = module.function(symbol).map_err(err)?;
    if shared >= 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    Ok(function)
}

/// Activations of one entry's checks: the same weights read with three
/// different inputs.
const INPUTS: usize = 3;

/// The PDL producer probe (`pdl_prefetch_producer`): after its own wait it
/// saves the activation it is given, overwrites it with BF16 NaN pairs,
/// releases its dependents, waits `PRODUCER_TAIL_NS` and only then restores
/// the activation, so a consumer that reads its input before its own
/// dependency wait sees NaN.
struct Producer {
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    live: DeviceBuffer,
    /// CTAs of a launch: one, or [`WIDE_PRODUCER_CTAS`] for the device
    /// checks at every count, which poison and restore whole activations
    /// thousands of times.
    ctas: u32,
}

/// CTAs of a wide producer: one SM in six, so the consumer's CTAs (168
/// registers over 384 threads, which share no SM with a producer CTA) find
/// the other SMs free and start during the producer's tail.
const WIDE_PRODUCER_CTAS: u32 = 32;

/// Longer than any checked entry's run (at most about 40 us).
const PRODUCER_TAIL_NS: u64 = 100_000;

impl Producer {
    fn new(session: &Session) -> LabResult<Self> {
        Self::with_ctas(session, 1)
    }

    /// The producer over [`WIDE_PRODUCER_CTAS`] CTAs: the same poison,
    /// release, tail and restore, each CTA over its stride of the words, on
    /// few enough SMs that its dependents overlap it.
    fn wide(session: &Session) -> LabResult<Self> {
        Self::with_ctas(session, WIDE_PRODUCER_CTAS)
    }

    fn with_ctas(session: &Session, ctas: u32) -> LabResult<Self> {
        Ok(Self {
            function: session
                .module("probe_pdl_prefetch_sm120")?
                .function("chop_probe_pdl_prefetch_producer_v1")
                .map_err(err)?,
            eviction: upload(session, &[0; 16])?,
            checksum: session.allocate(ctas as usize * 128 * 4)?,
            live: session.allocate(4)?,
            ctas,
        })
    }

    /// The descriptor that holds `activation` poisoned, saving it to `fresh`
    /// (`PdlOperand`: fresh, input, words, flags preserve | floating poison).
    fn operand(
        session: &Session,
        activation: &DeviceBuffer,
        fresh: &DeviceBuffer,
    ) -> LabResult<DeviceBuffer> {
        let bytes = activation.byte_len();
        let mut descriptor = Vec::with_capacity(24);
        descriptor.extend_from_slice(
            &fresh
                .device_address_at(0, bytes)
                .map_err(err)?
                .to_ne_bytes(),
        );
        descriptor.extend_from_slice(
            &activation
                .device_address_at(0, bytes)
                .map_err(err)?
                .to_ne_bytes(),
        );
        descriptor.extend_from_slice(&u32::try_from(bytes / 4).map_err(err)?.to_ne_bytes());
        descriptor.extend_from_slice(&3_u32.to_ne_bytes());
        upload(session, &descriptor)
    }

    fn launch(&self, stream: &Stream, operand: &DeviceBuffer) -> LabResult<()> {
        self.function
            .launch(
                stream,
                launch::simple_config(
                    (self.ctas, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(operand),
                    KernelArgument::U32(1),
                    KernelArgument::Buffer(&self.live),
                    KernelArgument::U32(1),
                    KernelArgument::U64(PRODUCER_TAIL_NS),
                ],
            )
            .map_err(err)
    }
}

/// One entry's operands: weights (and FP8 block scales), `INPUTS`
/// activations with the producer's save areas and descriptors, a residual
/// where the epilogue adds one, one output per input, and one workspace and
/// ticket array that every launch shares, as a bundle's projection scratch
/// is shared.
struct Fixture {
    query: GemmQuery,
    weights: DeviceBuffer,
    scales: Option<DeviceBuffer>,
    activations: Vec<DeviceBuffer>,
    fresh: Vec<DeviceBuffer>,
    operands: Vec<DeviceBuffer>,
    residual: Option<DeviceBuffer>,
    outputs: Vec<DeviceBuffer>,
    workspace: DeviceBuffer,
    semaphores: DeviceBuffer,
    row_bytes: usize,
    element_bytes: usize,
}

impl Fixture {
    /// Operands for `query`, the workspace and tickets sized for every entry
    /// of `specs` (zeroed once).
    fn new(session: &Session, query: GemmQuery, specs: &[&InstantiationSpec]) -> LabResult<Self> {
        let (n, k, capacity) = (query.n as usize, query.k as usize, query.m_capacity);
        let mut rng = Rng(0x51ed_270b ^ (n as u64) << 24 ^ (k as u64) << 4 ^ u64::from(capacity));
        let (weights, scales) = match query.weight {
            WeightFormat::Bf16 => (rng.bf16_bytes(n * k, k, false), None),
            WeightFormat::Fp8Block128x128(_) => (
                rng.e4m3_bytes(n * k),
                Some(rng.scale_bytes(n.div_ceil(128) * k.div_ceil(128))),
            ),
            other => return Err(format!("no fixture for {other:?} weights")),
        };
        let element_bytes = if query.output == DType::F32 { 4 } else { 2 };
        let row_bytes = n * element_bytes;
        let out_bytes = capacity as usize * row_bytes;
        let mut workspace = 4;
        let mut semaphores = 4;
        for spec in specs {
            let rows = capacity.min(m_capacity(spec)?);
            let plan = gemm::plan(&spec.key, gemm::row_counts(&spec.key, rows)).map_err(err)?;
            workspace = workspace.max(plan.workspace_bytes as usize);
            semaphores = semaphores
                .max(GemmBuffers::semaphore_capacity_bytes(&spec.key).map_err(err)? as usize);
        }
        let activations = (0..INPUTS)
            .map(|_| upload(session, &rng.bf16_bytes(capacity as usize * k, k, true)))
            .collect::<LabResult<Vec<_>>>()?;
        let fresh = (0..INPUTS)
            .map(|_| session.allocate(capacity as usize * k * 2))
            .collect::<LabResult<Vec<_>>>()?;
        let operands = activations
            .iter()
            .zip(&fresh)
            .map(|(a, f)| Producer::operand(session, a, f))
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            query,
            weights: upload(session, &weights)?,
            scales: scales.map(|s| upload(session, &s)).transpose()?,
            activations,
            fresh,
            operands,
            residual: if query.epilogues.residual_add {
                let values = if element_bytes == 4 {
                    (0..out_bytes / 4)
                        .flat_map(|_| (rng.next() as f32 / 1e6).to_le_bytes())
                        .collect()
                } else {
                    rng.bf16_bytes(out_bytes / 2, n, false)
                };
                Some(upload(session, &values)?)
            } else {
                None
            },
            outputs: (0..INPUTS)
                .map(|_| session.allocate(out_bytes))
                .collect::<LabResult<Vec<_>>>()?,
            workspace: session.allocate(workspace)?,
            semaphores: zeroed(session, semaphores)?,
            row_bytes,
            element_bytes,
        })
    }

    /// The live rows of an output at `m` rows.
    fn map(&self, m: u32) -> LabResult<WriteMap> {
        WriteMap::of(self.query.output).byte_prefix(m as usize * self.row_bytes)
    }

    /// Launches `spec` (`function`) on input `input` into its output.
    fn launch(
        &self,
        session: &Session,
        spec: &InstantiationSpec,
        function: &Function,
        input: usize,
        m: u32,
        stream: &Stream,
    ) -> LabResult<()> {
        self.launch_into(
            session,
            spec,
            function,
            input,
            m,
            stream,
            &self.outputs[input],
        )
    }

    /// Launches `spec` (`function`) on input `input` into `output`.
    #[allow(clippy::too_many_arguments)]
    fn launch_into(
        &self,
        session: &Session,
        spec: &InstantiationSpec,
        function: &Function,
        input: usize,
        m: u32,
        stream: &Stream,
        output: &DeviceBuffer,
    ) -> LabResult<()> {
        let counts = gemm::row_counts(&spec.key, m);
        let plan: LaunchPlan = gemm::plan(&spec.key, counts).map_err(err)?;
        let maps = gemm::tensor_maps(&spec.key, WEIGHT, ACTIVATION).map_err(err)?;
        let args = gemm::arguments(
            &spec.key,
            counts,
            maps,
            GemmBuffers {
                output: OUTPUT,
                partial: PARTIAL,
                semaphores: SEMAPHORES,
                weight_scales: self.scales.is_some().then_some(SCALES),
                activation_scales: None,
                residual: self.residual.is_some().then_some(RESIDUAL),
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .map_err(err)?;
        let mut buffers = BTreeMap::from([
            (WEIGHT, &self.weights),
            (ACTIVATION, &self.activations[input]),
            (OUTPUT, output),
            (PARTIAL, &self.workspace),
            (SEMAPHORES, &self.semaphores),
        ]);
        if let Some(scales) = &self.scales {
            buffers.insert(SCALES, scales);
        }
        if let Some(residual) = &self.residual {
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
    }

    /// Each input's output of `spec` at `m` rows, each from one launch
    /// between host synchronizations, from a poisoned output and workspace.
    fn separate(
        &self,
        session: &Session,
        spec: &InstantiationSpec,
        function: &Function,
        m: u32,
    ) -> LabResult<Vec<Vec<u8>>> {
        (0..INPUTS)
            .map(|input| self.separate_input(session, spec, function, m, input))
            .collect()
    }

    /// Input `input`'s output of `spec` at `m` rows from one launch between
    /// host synchronizations, from a poisoned output and workspace.
    fn separate_input(
        &self,
        session: &Session,
        spec: &InstantiationSpec,
        function: &Function,
        m: u32,
        input: usize,
    ) -> LabResult<Vec<u8>> {
        poison::poison(session, &[&self.outputs[input], &self.workspace])?;
        self.launch(session, spec, function, input, m, &session.stream)?;
        session.stream.synchronize().map_err(err)?;
        let bytes = poison::read(session, &self.outputs[input])?;
        poison::check(
            &format!("{} M={m} input {input}", spec.symbol),
            &bytes,
            &self.map(m)?,
            poison::Fill::Pattern(&[poison::POISON]),
        )?;
        Ok(bytes)
    }
}

/// The first differing element of two output images, as text.
fn differs(label: &str, n: usize, element_bytes: usize, want: &[u8], got: &[u8]) -> Option<String> {
    let offset = want.iter().zip(got).position(|(a, b)| a != b)?;
    let element = offset / element_bytes;
    Some(format!(
        "{label}: row {} column {} differs",
        element / n,
        element % n
    ))
}

/// The checks every served chain-split entry passes and every mutant must
/// fail, against `expected` (each input's separate one-split output at `m`
/// rows):
///
/// - each input in its own launch, compared byte for byte;
/// - eager, relaunched, then behind the PDL producer, which holds input 0
///   at NaN until its tail ends, so a consumer that reads its input before
///   its dependency wait differs;
/// - one sequence with no host synchronization, every launch PDL and
///   sharing one workspace and ticket array: producer, input 0, producer,
///   input 1, then input 2 directly behind input 1's launch; run eager, then
///   from one captured graph replayed twice. Every output keeps its write
///   map and equals its expected bytes.
///
/// Outputs, the workspace and the activations' save areas are poisoned
/// before every phase; the tickets never are (each launch leaves them zero).
fn qualify(
    session: &Session,
    fixture: &Fixture,
    spec: &InstantiationSpec,
    function: &Function,
    producer: &Producer,
    m: u32,
    expected: &[Vec<u8>],
) -> LabResult<serde_json::Value> {
    let n = fixture.query.n as usize;
    let label = |what: &str| format!("{} M={m} {what}", spec.symbol);
    let separate = fixture.separate(session, spec, function, m)?;
    for (input, (want, got)) in expected.iter().zip(&separate).enumerate() {
        if let Some(e) = differs(
            &label(&format!("separate input {input}")),
            n,
            fixture.element_bytes,
            want,
            got,
        ) {
            return Err(e);
        }
    }
    let stream = &session.stream;
    let run =
        |input: usize, stream: &Stream| fixture.launch(session, spec, function, input, m, stream);
    // Behind the producer.
    let mut replay = Replay::new(session)
        .output("output", &fixture.outputs[0], fixture.map(m)?)
        .scratch("chains", &fixture.workspace)
        .scratch("saved input", &fixture.fresh[0]);
    replay.eager(|| run(0, stream))?;
    replay.relaunch(|| run(0, stream))?;
    replay.pdl(
        || producer.launch(stream, &fixture.operands[0]),
        || run(0, stream),
        true,
    )?;
    if let Some(e) = differs(
        &label("behind the producer"),
        n,
        fixture.element_bytes,
        &expected[0],
        replay.expected("output").ok_or("missing eager output")?,
    ) {
        return Err(e);
    }
    let mut pdl = BTreeMap::new();
    replay.record(&mut pdl);
    // Back to back on shared scratch.
    let sequence = |stream: &Stream| -> LabResult<()> {
        producer.launch(stream, &fixture.operands[0])?;
        run(0, stream)?;
        producer.launch(stream, &fixture.operands[1])?;
        run(1, stream)?;
        run(2, stream)
    };
    let mut replay = Replay::new(session);
    for (input, output) in fixture.outputs.iter().enumerate() {
        replay = replay.output(format!("output {input}"), output, fixture.map(m)?);
    }
    let mut replay = replay
        .scratch("chains", &fixture.workspace)
        .scratch("saved input 0", &fixture.fresh[0])
        .scratch("saved input 1", &fixture.fresh[1]);
    replay.eager(|| sequence(stream))?;
    for (input, want) in expected.iter().enumerate() {
        let got = replay
            .expected(&format!("output {input}"))
            .ok_or("missing eager output")?;
        if let Some(e) = differs(
            &label(&format!("sequence input {input}")),
            n,
            fixture.element_bytes,
            want,
            got,
        ) {
            return Err(e);
        }
    }
    let graph = session.capture(sequence)?;
    replay.graph(&graph)?;
    replay.graph(&graph)?;
    let mut back_to_back = BTreeMap::new();
    replay.record(&mut back_to_back);
    Ok(json!({"m": m, "behind_producer": pdl, "back_to_back": back_to_back}))
}

/// The live counts a pair is compared at: every count of a decode or
/// verify bucket; for a prefill capacity (warp tiles, `tile_m` rows by
/// `tiles_b` column tiles) every count through 65 and 127 to 129 (partial
/// row tiles), both sides of the row counts where the tiles first outnumber
/// one and two waves of `gemm::SM_COUNT` CTAs, 1,023 and 2,047, each power of
/// two from 1,024 to the capacity with one below and one above it, and the
/// capacity.
fn bitwise_counts(capacity: u32, tile_m: u32, tiles_b: u32) -> Vec<u32> {
    if capacity <= 32 {
        return (1..=capacity).collect();
    }
    let mut counts: Vec<u32> = (1..=65).chain(127..=129).chain([1023, 2047]).collect();
    for waves in [1, 2] {
        // The first row count whose tiles exceed `waves` full waves.
        let rows = (waves * gemm::SM_COUNT / tiles_b + 1) * tile_m - tile_m + 1;
        counts.extend([rows - 1, rows, rows + tile_m - 1, rows + tile_m]);
    }
    let mut power = 1024;
    while power <= capacity {
        counts.extend([power - 1, power, power + 1]);
        power *= 2;
    }
    counts.push(capacity);
    counts.retain(|&m| m >= 1 && m <= capacity);
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// How two output images differ: the differing element count and the first
/// few differing (row tile, column tile) pairs.
fn mismatch_summary(
    n: usize,
    element_bytes: usize,
    tile: (usize, usize),
    want: &[u8],
    got: &[u8],
) -> String {
    let mut count = 0;
    let mut tiles = std::collections::BTreeSet::new();
    for (element, (w, g)) in want
        .chunks_exact(element_bytes)
        .zip(got.chunks_exact(element_bytes))
        .enumerate()
    {
        if w != g {
            count += 1;
            tiles.insert((element / n / tile.0, element % n / tile.1));
        }
    }
    let first: Vec<_> = tiles.iter().take(8).collect();
    format!(
        "{count} elements differ in {} tiles (row tile, column tile), first {first:?}",
        tiles.len()
    )
}

/// The phases of `qualify` in which `spec`'s outputs differ from
/// `expected` (each input's separate unmodified output at `m` rows), each
/// launch from poisoned outputs, workspace and save areas, outputs read
/// back after a host synchronization and compared whole (bytes and write
/// maps): each input separately; input 0 behind the producer; the
/// back-to-back sequence eager, then from one captured graph replayed
/// twice. Preparation, launch, capture, synchronization and readback
/// errors propagate; only a compared output difference counts.
/// `inject_failure` returns an operational error before the first launch,
/// for the control that such an error fails the test.
#[allow(clippy::too_many_arguments)]
fn mutant_differences(
    session: &Session,
    fixture: &Fixture,
    spec: &InstantiationSpec,
    function: &Function,
    producer: &Producer,
    m: u32,
    expected: &[Vec<u8>],
    inject_failure: bool,
) -> LabResult<Vec<String>> {
    if inject_failure {
        return Err("injected operational failure before the first launch".into());
    }
    let stream = &session.stream;
    let mut differing = Vec::new();
    let mut compare = |phase: &str, inputs: &[usize]| -> LabResult<()> {
        stream.synchronize().map_err(err)?;
        for &input in inputs {
            if poison::read(session, &fixture.outputs[input])? != expected[input] {
                differing.push(format!("{phase}: output {input}"));
            }
        }
        Ok(())
    };
    for input in 0..INPUTS {
        poison::poison(session, &[&fixture.outputs[input], &fixture.workspace])?;
        fixture.launch(session, spec, function, input, m, stream)?;
        compare("separate", &[input])?;
    }
    poison::poison(
        session,
        &[&fixture.outputs[0], &fixture.workspace, &fixture.fresh[0]],
    )?;
    producer.launch(stream, &fixture.operands[0])?;
    fixture.launch(session, spec, function, 0, m, stream)?;
    compare("behind the producer", &[0])?;
    let sequence = |stream: &Stream| -> LabResult<()> {
        producer.launch(stream, &fixture.operands[0])?;
        fixture.launch(session, spec, function, 0, m, stream)?;
        producer.launch(stream, &fixture.operands[1])?;
        fixture.launch(session, spec, function, 1, m, stream)?;
        fixture.launch(session, spec, function, 2, m, stream)
    };
    let scratch: Vec<&DeviceBuffer> = fixture
        .outputs
        .iter()
        .chain([&fixture.workspace, &fixture.fresh[0], &fixture.fresh[1]])
        .collect();
    let all = [0, 1, 2];
    poison::poison(session, &scratch)?;
    sequence(stream)?;
    compare("sequence", &all)?;
    let graph = session.capture(sequence)?;
    for replay in 0..2 {
        poison::poison(session, &scratch)?;
        graph.launch(stream).map_err(err)?;
        compare(&format!("graph replay {replay}"), &all)?;
    }
    Ok(differing)
}

fn check_pair(
    session: &Session,
    pair: &Pair,
    candidate_module: &Module,
    reference_module: &Module,
    producer: &Producer,
) -> LabResult<()> {
    let StaticQuery::Gemm(query) = pair.candidate.key.query else {
        return Err("not a GEMM".into());
    };
    let (n, capacity) = (query.n as usize, query.m_capacity);
    let fixture = Fixture::new(session, query, &[&pair.reference, &pair.candidate])?;
    let shared = |spec: &InstantiationSpec| {
        gemm::plan(&spec.key, gemm::row_counts(&spec.key, capacity))
            .map(|plan| plan.dynamic_shared_bytes)
            .map_err(err)
    };
    let reference = function(
        reference_module,
        &pair.reference.symbol,
        shared(&pair.reference)?,
    )?;
    let candidate = function(
        candidate_module,
        &pair.candidate.symbol,
        shared(&pair.candidate)?,
    )?;
    // Every live count of a decode bucket (prefill: `bitwise_counts`), each
    // input separately, both entries.
    let knobs = pair.candidate.key.knobs;
    let tile = (usize::from(knobs.tile_m), usize::from(knobs.tile_n));
    let prefill = capacity > 32;
    let counts = bitwise_counts(
        capacity,
        u32::from(knobs.tile_m),
        query.n.div_ceil(u32::from(knobs.tile_n)),
    );
    for &m in &counts {
        let want = fixture.separate(session, &pair.reference, &reference, m)?;
        // A prefill reference runs twice, so a reference that varies between
        // runs is told apart from a candidate that differs from it.
        if prefill {
            let again = fixture.separate(session, &pair.reference, &reference, m)?;
            for (input, (first, second)) in want.iter().zip(&again).enumerate() {
                if first != second {
                    return Err(format!(
                        "reference {} varies between runs at M={m} input {input}: {}",
                        pair.reference.symbol,
                        mismatch_summary(n, fixture.element_bytes, tile, first, second)
                    ));
                }
            }
        }
        let got = fixture.separate(session, &pair.candidate, &candidate, m)?;
        for (input, (want, got)) in want.iter().zip(&got).enumerate() {
            if let Some(e) = differs(
                &format!(
                    "{} vs {} at M={m} input {input}",
                    pair.candidate.symbol, pair.reference.symbol
                ),
                n,
                fixture.element_bytes,
                want,
                got,
            ) {
                return Err(format!(
                    "{e}; {}",
                    mismatch_summary(n, fixture.element_bytes, tile, want, got)
                ));
            }
        }
    }
    let mut evidence = Vec::new();
    for m in poison::live_counts(capacity).into_iter().filter(|m| *m > 0) {
        let want = fixture.separate(session, &pair.reference, &reference, m)?;
        evidence.push(qualify(
            session,
            &fixture,
            &pair.candidate,
            &candidate,
            producer,
            m,
            &want,
        )?);
    }
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("tiled_chain_splits"));
    receipt.insert("candidate".into(), json!(pair.candidate.symbol));
    receipt.insert("reference".into(), json!(pair.reference.symbol));
    receipt.insert("weight".into(), json!(format!("{:?}", query.weight)));
    receipt.insert("n".into(), json!(n));
    receipt.insert("k".into(), json!(query.k));
    receipt.insert("capacity".into(), json!(capacity));
    receipt.insert(
        "kernel".into(),
        json!(format!("{:?}", pair.candidate.key.dense)),
    );
    receipt.insert("bitwise_live_counts".into(), json!(counts));
    receipt.insert("inputs".into(), json!(INPUTS));
    receipt.insert("qualified".into(), json!(evidence));
    receipt.insert("passed".into(), json!(true));
    session.receipt("tiled_chain_splits", receipt)?;
    eprintln!(
        "{} = {} bit for bit at {} live counts up to {capacity}; producer and back-to-back \
         checks passed",
        pair.candidate.symbol,
        pair.reference.symbol,
        counts.len()
    );
    Ok(())
}

#[test]
fn chain_splits_keep_one_split_bits() {
    if std::env::var_os("CHOPINFER_BUNDLE_REPORTS").is_none()
        || std::env::var_os("CHOPINFER_REFERENCE_REPORTS").is_none()
    {
        kernel_lab::skip(
            "chain-split bits: set CHOPINFER_BUNDLE_REPORTS and CHOPINFER_REFERENCE_REPORTS",
        )
        .unwrap();
        return;
    }
    gpu_test(8 << 30, |session| {
        let current = reported("CHOPINFER_BUNDLE_REPORTS")?;
        let before = reported("CHOPINFER_REFERENCE_REPORTS")?;
        let producer = Producer::new(session)?;
        let mut modules: BTreeMap<PathBuf, Module> = BTreeMap::new();
        let mut load = |path: PathBuf| -> LabResult<PathBuf> {
            if !modules.contains_key(&path) {
                let bytes = std::fs::read(&path).map_err(err)?;
                modules.insert(path.clone(), session.load_cubin(&bytes)?);
            }
            Ok(path)
        };
        let mut resolved = Vec::new();
        let strips_only = std::env::var("CHOPINFER_CHAIN_PAIRS").as_deref() == Ok("strip");
        for pair in pairs()? {
            if strips_only
                && !matches!(
                    pair.candidate.key.dense,
                    Some(DenseKernel::StripChains { .. })
                )
            {
                continue;
            }
            let candidate = cubin_of(&current, &pair.candidate.symbol)
                .ok_or_else(|| format!("no current report builds {}", pair.candidate.symbol))?;
            let reference = cubin_of(&before, &pair.reference.symbol)
                .ok_or_else(|| format!("no reference report builds {}", pair.reference.symbol))?;
            resolved.push((pair, load(candidate)?, load(reference)?));
        }
        if resolved.is_empty() {
            return Err("no chain-split offer in the inventory".into());
        }
        for (pair, candidate, reference) in &resolved {
            check_pair(
                session,
                pair,
                &modules[candidate],
                &modules[reference],
                &producer,
            )?;
        }
        Ok(())
    })
    .expect("chain splits keep one-split bits");
}

/// The mutant checks of one shape: the checks pass on its unmodified
/// instance (`instance(mutations[0])`: `qualify`, and `mutant_differences`
/// finds no difference), an injected operational failure propagates, and
/// each mutant's outputs differ from the unmodified instance's separate
/// launches in some phase of `mutant_differences`. One result per mutant.
#[allow(clippy::too_many_arguments)]
fn reject_mutants(
    session: &Session,
    module: &Module,
    producer: &Producer,
    tag: &str,
    query: GemmQuery,
    mutations: &[&str],
    instance: impl Fn(&str) -> InstantiationSpec,
) -> LabResult<Vec<serde_json::Value>> {
    let mut results = Vec::new();
    let capacity = query.m_capacity;
    let control = instance(mutations[0]);
    let shared = gemm::plan(&control.key, gemm::row_counts(&control.key, capacity))
        .map_err(err)?
        .dynamic_shared_bytes;
    let fixture = Fixture::new(session, query, &[&control])?;
    let control_fn = function(module, &control.symbol, shared)?;
    let expected = fixture.separate(session, &control, &control_fn, capacity)?;
    qualify(
        session,
        &fixture,
        &control,
        &control_fn,
        producer,
        capacity,
        &expected,
    )
    .map_err(|e| format!("the unmodified {tag} instance failed: {e}"))?;
    let control_differences = mutant_differences(
        session,
        &fixture,
        &control,
        &control_fn,
        producer,
        capacity,
        &expected,
        false,
    )?;
    if !control_differences.is_empty() {
        return Err(format!(
            "the unmodified {tag} instance differs: {control_differences:?}"
        ));
    }
    if mutant_differences(
        session,
        &fixture,
        &control,
        &control_fn,
        producer,
        capacity,
        &expected,
        true,
    )
    .is_ok()
    {
        return Err("an injected operational failure did not propagate".into());
    }
    for &mutation in &mutations[1..] {
        let mutant = instance(mutation);
        // A fresh fixture: a mutant may leave tickets set.
        let fixture = Fixture::new(session, query, &[&mutant])?;
        let mutant_fn = function(module, &mutant.symbol, shared)?;
        let differences = mutant_differences(
            session, &fixture, &mutant, &mutant_fn, producer, capacity, &expected, false,
        )?;
        if differences.is_empty() {
            return Err(format!("the {mutation} mutant of {tag} passed"));
        }
        eprintln!("{tag} {mutation}: outputs differ in {differences:?}");
        results.push(json!({"shape": tag, "mutation": mutation, "differs": differences}));
    }
    Ok(results)
}

/// Runs without a GPU: every lab instance is in the lab cubin, and the
/// mutants differ from the unmodified instance only in their mutation.
#[test]
fn chain_mutants_are_built() {
    let built = kernel_lab::cubin_entries(chain_mutant_shapes::CUBIN).unwrap();
    for (mutation, spec) in chain_mutant_shapes::instances() {
        assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        assert!(spec.symbol.contains(&mutation));
        let StaticQuery::Gemm(query) = spec.key.query else {
            panic!("{} is not a GEMM", spec.symbol);
        };
        let Some(DenseKernel::TiledChains { splits }) = spec.key.dense else {
            panic!("{} is not a chain entry", spec.symbol);
        };
        if query.m_capacity > 32 {
            assert_eq!(splits, 1, "{}", spec.symbol);
        } else {
            assert!(splits >= gemm::MIN_CHAIN_SPLITS, "{}", spec.symbol);
        }
    }
}

/// The checks above pass on the unmodified lab instance of each shape
/// (`qualify`, and `mutant_differences` finds no difference), and each
/// mutant's outputs differ from the unmodified instance's separate launches
/// in some phase of `mutant_differences`: no or a late dependency wait
/// (behind the producer), temporaries shared across tiles and a ticket
/// never reset (separate launches and the back-to-back sequence); for warp
/// tiles, no or a late wait. A mutant counts as caught only on an observed
/// output difference; operational errors fail the test, which an injected
/// failure before the first launch shows.
#[test]
fn chain_checks_reject_every_mutant() {
    gpu_test(4 << 30, |session| {
        let module = session.module(chain_mutant_shapes::CUBIN)?;
        let producer = Producer::new(session)?;
        let mut results = Vec::new();
        let shapes = chain_mutant_shapes::queries()
            .into_iter()
            .map(|(tag, query)| (tag, query, &chain_mutant_shapes::MUTATIONS[..]))
            .chain(
                chain_mutant_shapes::warp_queries()
                    .into_iter()
                    .map(|(tag, query)| (tag, query, &chain_mutant_shapes::WARP_MUTATIONS[..])),
            );
        for (tag, query, mutations) in shapes {
            results.extend(reject_mutants(
                session,
                &module,
                &producer,
                tag,
                query,
                mutations,
                |mutation| chain_mutant_shapes::instance(tag, query, mutation),
            )?);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("tiled_chain_mutants"));
        receipt.insert("mutants".into(), json!(results));
        receipt.insert("passed".into(), json!(true));
        session.receipt("tiled_chain_splits", receipt)?;
        Ok(())
    })
    .expect("chain checks reject every mutant");
}

// ------------------------------------------------------ strip chains ----

/// Runs without a GPU: every strip-chain lab instance is in its cubin, every
/// sweep case has candidates of each strip count, and the mutants differ
/// from the unmodified instance only in their mutation.
#[test]
fn strip_lab_instances_are_built() {
    for case in strip_shapes::cases() {
        let built = kernel_lab::cubin_entries(&strip_shapes::cubin_name(case.shape)).unwrap();
        for spec in [&case.current, &case.one]
            .into_iter()
            .chain(&case.candidates)
        {
            assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        }
        for strips in gemm::STRIP_COUNTS {
            assert!(
                case.candidates.iter().any(|spec| matches!(
                    spec.key.dense,
                    Some(DenseKernel::StripChains { strips: s, .. }) if s == strips
                )),
                "{} has no {strips}-strip candidate",
                case.tag
            );
        }
    }
    let built = kernel_lab::cubin_entries(strip_shapes::MUTANTS_CUBIN).unwrap();
    for (mutation, spec) in strip_shapes::mutants() {
        assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        assert!(spec.symbol.contains(&mutation));
    }
}

/// One sweep case: at every live count of `counts`, each input separately,
/// the entry of today and every candidate equal the one-split entry bit for
/// bit (a prefill reference also runs twice and must equal itself); at
/// every count of `extra_counts`, input 0 alone, every candidate equals the
/// one-split entry; at `poison::live_counts` and `extra_qualified`, every
/// candidate passes `qualify` (eager, relaunched, behind the PDL producer,
/// back to back on shared workspace and tickets, eager and from a graph
/// replayed twice). The receipt is `receipt_case`'s.
#[allow(clippy::too_many_arguments)]
fn check_strip_case(
    session: &Session,
    case: &strip_shapes::Case,
    module: &Module,
    producer: &Producer,
    counts: &[u32],
    extra_counts: &[u32],
    extra_qualified: &[u32],
    receipt_case: &str,
) -> LabResult<()> {
    let query = case.query;
    let (n, capacity) = (query.n as usize, query.m_capacity);
    let specs: Vec<&InstantiationSpec> = [&case.one, &case.current]
        .into_iter()
        .chain(&case.candidates)
        .collect();
    let fixture = Fixture::new(session, query, &specs)?;
    let functions = specs
        .iter()
        .map(|spec| {
            let shared = gemm::plan(&spec.key, gemm::row_counts(&spec.key, capacity))
                .map_err(err)?
                .dynamic_shared_bytes;
            function(module, &spec.symbol, shared)
        })
        .collect::<LabResult<Vec<_>>>()?;
    for &m in counts {
        let want = fixture.separate(session, &case.one, &functions[0], m)?;
        if capacity > 32 {
            let again = fixture.separate(session, &case.one, &functions[0], m)?;
            for (input, (first, second)) in want.iter().zip(&again).enumerate() {
                if first != second {
                    let tile = (
                        usize::from(case.one.key.knobs.tile_m),
                        usize::from(case.one.key.knobs.tile_n),
                    );
                    return Err(format!(
                        "reference {} varies between runs at M={m} input {input}: {}",
                        case.one.symbol,
                        mismatch_summary(n, fixture.element_bytes, tile, first, second)
                    ));
                }
            }
        }
        for (spec, f) in specs.iter().zip(&functions).skip(1) {
            let got = fixture.separate(session, spec, f, m)?;
            let tile = (
                usize::from(spec.key.knobs.tile_m),
                usize::from(spec.key.knobs.tile_n),
            );
            for (input, (want, got)) in want.iter().zip(&got).enumerate() {
                if let Some(e) = differs(
                    &format!(
                        "{} vs {} at M={m} input {input}",
                        spec.symbol, case.one.symbol
                    ),
                    n,
                    fixture.element_bytes,
                    want,
                    got,
                ) {
                    return Err(format!(
                        "{e}; {}",
                        mismatch_summary(n, fixture.element_bytes, tile, want, got)
                    ));
                }
            }
        }
    }
    for &m in extra_counts {
        let want = fixture.separate_input(session, &case.one, &functions[0], m, 0)?;
        for (spec, f) in specs.iter().zip(&functions).skip(2) {
            let got = fixture.separate_input(session, spec, f, m, 0)?;
            let tile = (
                usize::from(spec.key.knobs.tile_m),
                usize::from(spec.key.knobs.tile_n),
            );
            if let Some(e) = differs(
                &format!("{} vs {} at M={m} input 0", spec.symbol, case.one.symbol),
                n,
                fixture.element_bytes,
                &want,
                &got,
            ) {
                return Err(format!(
                    "{e}; {}",
                    mismatch_summary(n, fixture.element_bytes, tile, &want, &got)
                ));
            }
        }
    }
    let mut qualified = Vec::new();
    for m in poison::live_counts(capacity)
        .into_iter()
        .chain(extra_qualified.iter().copied())
    {
        if m > 0 && m <= capacity && !qualified.contains(&m) {
            qualified.push(m);
        }
    }
    let mut evidence = Vec::new();
    for &m in &qualified {
        let want = fixture.separate(session, &case.one, &functions[0], m)?;
        for (spec, f) in specs.iter().zip(&functions).skip(2) {
            evidence.push(json!({
                "symbol": spec.symbol,
                "checks": qualify(session, &fixture, spec, f, producer, m, &want)?,
            }));
        }
    }
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(receipt_case));
    receipt.insert("tag".into(), json!(case.tag));
    receipt.insert("reference".into(), json!(case.one.symbol));
    receipt.insert("current".into(), json!(case.current.symbol));
    receipt.insert(
        "candidates".into(),
        json!(
            case.candidates
                .iter()
                .map(|spec| json!({
                    "symbol": spec.symbol,
                    "kernel": format!("{:?}", spec.key.dense),
                    "stages": spec.key.knobs.stages,
                }))
                .collect::<Vec<_>>()
        ),
    );
    receipt.insert("n".into(), json!(n));
    receipt.insert("k".into(), json!(query.k));
    receipt.insert("output".into(), json!(format!("{:?}", query.output)));
    receipt.insert("capacity".into(), json!(capacity));
    receipt.insert("bitwise_live_counts".into(), json!(counts));
    receipt.insert("input_0_live_counts".into(), json!(extra_counts));
    receipt.insert("qualified_live_counts".into(), json!(qualified));
    receipt.insert("inputs".into(), json!(INPUTS));
    receipt.insert("qualified".into(), json!(evidence));
    receipt.insert("passed".into(), json!(true));
    session.receipt("tiled_chain_splits", receipt)?;
    eprintln!(
        "{}: today's entry and {} candidates = {} bit for bit at {} live counts through \
         {capacity}, the candidates at {} more; producer and back-to-back checks passed at \
         {qualified:?}",
        case.tag,
        case.candidates.len(),
        case.one.symbol,
        counts.len(),
        extra_counts.len()
    );
    Ok(())
}

#[test]
fn strip_chains_keep_one_split_bits() {
    gpu_test(8 << 30, |session| {
        let producer = Producer::new(session)?;
        let filter = std::env::var("CHOPINFER_STRIP_FILTER").ok();
        let mut checked = 0;
        for case in strip_shapes::cases() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let module = session.module(&strip_shapes::cubin_name(case.shape))?;
            let counts: Vec<u32> = (1..=case.query.m_capacity).collect();
            check_strip_case(
                session,
                &case,
                &module,
                &producer,
                &counts,
                &[],
                &[],
                "strip_chain_sweep",
            )?;
            checked += 1;
        }
        if checked == 0 {
            return Err("no strip-chain case ran".into());
        }
        Ok(())
    })
    .expect("strip chains keep one-split bits");
}

/// The chain checks reject every strip-chain mutant (`reject_mutants`):
/// no or a late dependency wait at one split and over K splits, temporaries
/// shared across tiles and a ticket never reset over K splits.
#[test]
fn strip_checks_reject_every_mutant() {
    gpu_test(4 << 30, |session| {
        let module = session.module(strip_shapes::MUTANTS_CUBIN)?;
        let producer = Producer::new(session)?;
        let mut results = Vec::new();
        for (tag, query, strips, splits, mutations) in strip_shapes::mutant_queries() {
            results.extend(reject_mutants(
                session,
                &module,
                &producer,
                tag,
                query,
                mutations,
                |mutation| strip_shapes::mutant(tag, query, strips, splits, mutation),
            )?);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("strip_chain_mutants"));
        receipt.insert("mutants".into(), json!(results));
        receipt.insert("passed".into(), json!(true));
        session.receipt("tiled_chain_splits", receipt)?;
        Ok(())
    })
    .expect("strip checks reject every mutant");
}

// -------------------------------------------------- FP8 strip chains ----

/// Runs without a GPU: every FP8 strip-chain lab instance is in its cubin,
/// every sweep case has candidates of each strip count, and the mutants
/// differ from the unmodified instance only in their mutation.
#[test]
fn fp8_strip_lab_instances_are_built() {
    for case in fp8_strip_shapes::cases() {
        assert!(matches!(
            case.query.weight,
            WeightFormat::Fp8Block128x128(_)
        ));
        let built = kernel_lab::cubin_entries(&fp8_strip_shapes::cubin_name(case.shape)).unwrap();
        for spec in [&case.current, &case.one]
            .into_iter()
            .chain(&case.candidates)
        {
            assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        }
        for strips in gemm::STRIP_COUNTS {
            assert!(
                case.candidates.iter().any(|spec| matches!(
                    spec.key.dense,
                    Some(DenseKernel::StripChains { strips: s, .. }) if s == strips
                )),
                "{} has no {strips}-strip candidate",
                case.tag
            );
        }
    }
    let built = kernel_lab::cubin_entries(fp8_strip_shapes::MUTANTS_CUBIN).unwrap();
    for (mutation, spec) in fp8_strip_shapes::mutants() {
        assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        assert!(spec.symbol.contains(&mutation));
    }
}

/// `check_strip_case` over every FP8 sweep case: at every live count each
/// candidate and the entry of today equal the one-split entry bit for bit,
/// and every candidate passes `qualify`.
#[test]
fn fp8_strip_chains_keep_one_split_bits() {
    gpu_test(8 << 30, |session| {
        let producer = Producer::new(session)?;
        let filter = std::env::var("CHOPINFER_STRIP_FILTER").ok();
        let mut checked = 0;
        for case in fp8_strip_shapes::cases() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let module = session.module(&fp8_strip_shapes::cubin_name(case.shape))?;
            let counts: Vec<u32> = (1..=case.query.m_capacity).collect();
            check_strip_case(
                session,
                &case,
                &module,
                &producer,
                &counts,
                &[],
                &[],
                "strip_chain_sweep",
            )?;
            checked += 1;
        }
        if checked == 0 {
            return Err("no FP8 strip-chain case ran".into());
        }
        Ok(())
    })
    .expect("FP8 strip chains keep one-split bits");
}

/// The chain checks reject every FP8 strip-chain mutant
/// (`reject_mutants`), as `strip_checks_reject_every_mutant` does for BF16.
#[test]
fn fp8_strip_checks_reject_every_mutant() {
    gpu_test(4 << 30, |session| {
        let module = session.module(fp8_strip_shapes::MUTANTS_CUBIN)?;
        let producer = Producer::new(session)?;
        let mut results = Vec::new();
        for (tag, query, strips, splits, mutations) in fp8_strip_shapes::mutant_queries() {
            results.extend(reject_mutants(
                session,
                &module,
                &producer,
                tag,
                query,
                mutations,
                |mutation| fp8_strip_shapes::mutant(tag, query, strips, splits, mutation),
            )?);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("fp8_strip_chain_mutants"));
        receipt.insert("mutants".into(), json!(results));
        receipt.insert("passed".into(), json!(true));
        session.receipt("tiled_chain_splits", receipt)?;
        Ok(())
    })
    .expect("FP8 strip checks reject every mutant");
}

// --------------------------------------------------- decoded tiles ----

/// Runs without a GPU: every decoded-tile lab instance is in its cubin,
/// each case's candidate takes the decoded knobs and its entry of today the
/// default (warp-tile) knobs of the same kernel, and the mutants differ from
/// the unmodified instance only in their mutation.
/// A decoded-tile lab sweep (or the staged tiles'): its cases, cubins,
/// mutant instances (of `mutant_query` with `decoded_shapes::MUTATIONS`)
/// and receipt names.
struct DecodedFamily {
    cases: fn() -> Vec<strip_shapes::Case>,
    cubin_name: fn(&str) -> String,
    mutants_cubin: &'static str,
    mutant: fn(&str) -> InstantiationSpec,
    mutant_query: fn() -> (&'static str, GemmQuery),
    /// The weights' format is BF16 (else FP8 128x128).
    bf16: bool,
    /// The first candidate's knobs (the one every-count check runs), the
    /// candidates per case, and `tile_k` of the entry each case runs today.
    knobs: fn() -> CompileKnobs,
    candidates: usize,
    current_tile_k: u16,
    /// Receipt `case` names: the sweep, the every-count checks, the
    /// mutants.
    receipts: [&'static str; 3],
}

/// The decoded tiles beside the warp tiles of today.
const DECODED: DecodedFamily = DecodedFamily {
    cases: decoded_shapes::cases,
    cubin_name: decoded_shapes::cubin_name,
    mutants_cubin: decoded_shapes::MUTANTS_CUBIN,
    mutant: decoded_shapes::mutant,
    mutant_query: decoded_shapes::mutant_query,
    bf16: false,
    knobs: gemm::decoded_tile_knobs,
    candidates: 1,
    current_tile_k: 128,
    receipts: [
        "decoded_tile_sweep",
        "decoded_tile_serving_counts",
        "decoded_tile_mutants",
    ],
};

/// The balanced decoded tiles beside the decoded tiles of today.
const BALANCED: DecodedFamily = DecodedFamily {
    cases: balanced_shapes::cases,
    cubin_name: balanced_shapes::cubin_name,
    mutants_cubin: balanced_shapes::MUTANTS_CUBIN,
    mutant: balanced_shapes::mutant,
    mutant_query: decoded_shapes::mutant_query,
    bf16: false,
    knobs: gemm::balanced_decoded_tile_knobs,
    candidates: 1,
    current_tile_k: 64,
    receipts: [
        "balanced_decoded_sweep",
        "balanced_decoded_serving_counts",
        "balanced_decoded_mutants",
    ],
};

/// The staged tiles (and the other warp tiles) beside the BF16 warp tiles
/// of today.
const STAGED: DecodedFamily = DecodedFamily {
    cases: staged_shapes::cases,
    cubin_name: staged_shapes::cubin_name,
    mutants_cubin: staged_shapes::MUTANTS_CUBIN,
    mutant: staged_shapes::mutant,
    mutant_query: staged_shapes::mutant_query,
    bf16: true,
    knobs: gemm::staged_tile_knobs,
    candidates: 1 + staged_shapes::WARP_TILES.len(),
    current_tile_k: 64,
    receipts: [
        "staged_tile_sweep",
        "staged_tile_serving_counts",
        "staged_tile_mutants",
    ],
};

fn lab_instances_are_built(family: &DecodedFamily) {
    assert_eq!(staged_shapes::MUTATIONS, decoded_shapes::MUTATIONS);
    for case in (family.cases)() {
        assert_eq!(
            family.bf16,
            case.query.weight == WeightFormat::Bf16,
            "{}",
            case.tag
        );
        if !family.bf16 {
            assert!(matches!(
                case.query.weight,
                WeightFormat::Fp8Block128x128(_)
            ));
        }
        let built = kernel_lab::cubin_entries(&(family.cubin_name)(case.shape)).unwrap();
        for spec in [&case.current, &case.one]
            .into_iter()
            .chain(&case.candidates)
        {
            assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        }
        assert_eq!(case.candidates.len(), family.candidates, "{}", case.tag);
        let candidate = &case.candidates[0];
        assert_eq!(candidate.key.knobs, (family.knobs)());
        assert_eq!(candidate.key.dense, case.current.key.dense);
        assert_eq!(case.current.key.knobs.tile_k, family.current_tile_k);
    }
    let built = kernel_lab::cubin_entries(family.mutants_cubin).unwrap();
    for &mutation in &decoded_shapes::MUTATIONS {
        let spec = (family.mutant)(mutation);
        assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        assert!(spec.symbol.contains(mutation));
        assert_eq!(spec.key.knobs, (family.knobs)());
    }
}

#[test]
fn decoded_tile_lab_instances_are_built() {
    lab_instances_are_built(&DECODED);
}

#[test]
fn staged_tile_lab_instances_are_built() {
    lab_instances_are_built(&STAGED);
}

/// The balanced sweep's current entries are the decoded tiles of the
/// decoded-tile sweep, under their own symbols.
#[test]
fn balanced_decoded_lab_instances_are_built() {
    lab_instances_are_built(&BALANCED);
    for (balanced, decoded) in balanced_shapes::cases()
        .into_iter()
        .zip(decoded_shapes::cases())
    {
        assert_eq!(balanced.query, decoded.query);
        assert_eq!(balanced.current.key, decoded.candidates[0].key);
        assert_eq!(balanced.one.key, decoded.one.key);
    }
}

/// The live counts a decoded-tile (or staged-tile) case is compared at:
/// `bitwise_counts` for 64-row and 128-row tiles (the warp tiles' and the
/// decoded tiles', and the halves and tiles of a split final wave) over
/// each entry's column tiles.
fn decoded_counts(case: &strip_shapes::Case) -> Vec<u32> {
    let capacity = case.query.m_capacity;
    let mut counts = Vec::new();
    for spec in [&case.current].into_iter().chain(&case.candidates) {
        let knobs = spec.key.knobs;
        for tile_m in [64, 128] {
            counts.extend(bitwise_counts(
                capacity,
                tile_m,
                case.query.n.div_ceil(u32::from(knobs.tile_n)),
            ));
        }
    }
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// The live counts serving can launch a decoded-tile projection at: a
/// prefill step of any count up to the capacity, run whole or as a B8 part
/// of up to 2,048 rows ending anywhere.
fn decoded_serving_counts(capacity: u32) -> std::ops::RangeInclusive<u32> {
    1..=capacity
}

/// Comparison slots one `qualify_on_device` count takes.
const DEVICE_SLOTS: usize = 3 * INPUTS + 3 + 3 * INPUTS;
/// Counts per receipt of the device checks.
const DEVICE_CHUNK: usize = 512;
/// The phases `qualify_on_device` runs at each count.
const DEVICE_PHASES: [&str; 7] = [
    "separate",
    "eager",
    "relaunch",
    "behind the producer",
    "sequence",
    "graph replay 0",
    "graph replay 1",
];

/// `qualify`'s phases at `m` live rows with every comparison on the device
/// (`DeviceCheck`) over the first `checked` rows (`m` but in the
/// unwritten-row control), against `reference`'s output for each input,
/// which it first writes into `references` from poisoned buffers (every
/// checked element written, every later byte still poison):
///
/// - each input in its own launch;
/// - eager, relaunched, then behind the PDL producer, which holds input 0 at
///   NaN until its tail ends;
/// - the sequence producer, input 0, producer, input 1, input 2, every
///   launch PDL and sharing one workspace, eager and from one graph
///   captured before any launch, replayed twice.
///
/// The outputs, the workspace and the producer's save areas are poisoned
/// before every phase; every output's checked rows must be written and
/// equal the reference's, and the rest keep the poison. Returns the
/// comparisons made and the failing ones; launch and capture errors
/// propagate.
#[allow(clippy::too_many_arguments)]
fn qualify_on_device(
    session: &Session,
    fixture: &Fixture,
    references: &[DeviceBuffer],
    reference: (&InstantiationSpec, &Function),
    candidate: (&InstantiationSpec, &Function),
    producer: &Producer,
    check: &mut DeviceCheck,
    m: u32,
    checked: u32,
) -> LabResult<(usize, Vec<String>)> {
    let stream = &session.stream;
    let live = checked as usize * fixture.row_bytes;
    let element = fixture.element_bytes;
    let (spec, function) = candidate;
    let run =
        |input: usize, stream: &Stream| fixture.launch(session, spec, function, input, m, stream);
    let sequence = |stream: &Stream| -> LabResult<()> {
        producer.launch(stream, &fixture.operands[0])?;
        run(0, stream)?;
        producer.launch(stream, &fixture.operands[1])?;
        run(1, stream)?;
        run(2, stream)
    };
    let graph = session.capture(sequence)?;
    let label = |what: &str| format!("{} M={m} {what}", spec.symbol);
    for (input, want) in references.iter().enumerate() {
        check.poison(stream, &[want, &fixture.workspace])?;
        fixture.launch_into(session, reference.0, reference.1, input, m, stream, want)?;
        check.compare(
            stream,
            format!("{} M={m} input {input}", reference.0.symbol),
            want,
            want,
            live,
            element,
        )?;
    }
    for (input, want) in references.iter().enumerate() {
        check.poison(stream, &[&fixture.outputs[input], &fixture.workspace])?;
        run(input, stream)?;
        check.compare(
            stream,
            label(&format!("separate input {input}")),
            want,
            &fixture.outputs[input],
            live,
            element,
        )?;
    }
    let first = [&fixture.outputs[0], &fixture.workspace, &fixture.fresh[0]];
    for phase in ["eager", "relaunch", "behind the producer"] {
        check.poison(stream, &first)?;
        if phase == "behind the producer" {
            producer.launch(stream, &fixture.operands[0])?;
        }
        run(0, stream)?;
        check.compare(
            stream,
            label(phase),
            &references[0],
            &fixture.outputs[0],
            live,
            element,
        )?;
    }
    let all: Vec<&DeviceBuffer> = fixture
        .outputs
        .iter()
        .chain([&fixture.workspace, &fixture.fresh[0], &fixture.fresh[1]])
        .collect();
    for phase in ["sequence", "graph replay 0", "graph replay 1"] {
        check.poison(stream, &all)?;
        if phase == "sequence" {
            sequence(stream)?;
        } else {
            graph.launch(stream).map_err(err)?;
        }
        for (input, want) in references.iter().enumerate() {
            check.compare(
                stream,
                label(&format!("{phase} input {input}")),
                want,
                &fixture.outputs[input],
                live,
                element,
            )?;
        }
    }
    check.finish(session)
}

/// The device checks on crafted 256-byte buffers whose first 64 bytes are
/// live (BF16 or F32 ones, the rest poison): a clean pair passes; a live
/// element that still holds the poison fails at its width (an F32 word, a
/// single BF16 halfword) even when the reference is compared with itself, as
/// the reference checks are; a candidate missing a write fails as differing
/// and unwritten; a changed tail byte fails as corrupt.
#[test]
fn device_checks_reject_unwritten_and_corrupt_elements() {
    gpu_test(1 << 30, |session| {
        let (total, live) = (256, 64);
        let image = |element: usize, unwritten: Option<usize>, tail: Option<usize>| {
            let one: &[u8] = if element == 2 {
                &[0x80, 0x3f]
            } else {
                &[0, 0, 0x80, 0x3f]
            };
            let mut bytes = vec![poison::POISON; total];
            for chunk in bytes[..live].chunks_mut(element) {
                chunk.copy_from_slice(one);
            }
            if let Some(at) = unwritten {
                bytes[at * element..(at + 1) * element].fill(poison::POISON);
            }
            if let Some(at) = tail {
                bytes[at] = 0;
            }
            bytes
        };
        // A control: its name, element bytes, reference, candidate (none: the
        // reference itself) and expected failure text.
        type Control<'a> = (&'a str, usize, Vec<u8>, Option<Vec<u8>>, Option<&'a str>);
        let cases: [Control; 7] = [
            (
                "clean BF16",
                2,
                image(2, None, None),
                Some(image(2, None, None)),
                None,
            ),
            ("clean F32", 4, image(4, None, None), None, None),
            (
                "reference F32 element unwritten",
                4,
                image(4, Some(3), None),
                None,
                Some("0 live words differ, 1 live elements unwritten, 0 poison words changed"),
            ),
            (
                "reference BF16 halfword unwritten",
                2,
                image(2, Some(5), None),
                None,
                Some("0 live words differ, 1 live elements unwritten, 0 poison words changed"),
            ),
            (
                "candidate F32 write missing",
                4,
                image(4, None, None),
                Some(image(4, Some(15), None)),
                Some("1 live words differ, 1 live elements unwritten, 0 poison words changed"),
            ),
            (
                "candidate BF16 write missing",
                2,
                image(2, None, None),
                Some(image(2, Some(0), None)),
                Some("1 live words differ, 1 live elements unwritten, 0 poison words changed"),
            ),
            (
                "tail corrupt",
                2,
                image(2, None, None),
                Some(image(2, None, Some(live + 5))),
                Some("0 live words differ, 0 live elements unwritten, 1 poison words changed"),
            ),
        ];
        let mut check = DeviceCheck::new(session, 1)?;
        let mut observed = Vec::new();
        for (case, element, reference, candidate, expected) in cases {
            let want = upload(session, &reference)?;
            let got = candidate.map(|bytes| upload(session, &bytes)).transpose()?;
            check.compare(
                &session.stream,
                case,
                &want,
                got.as_ref().unwrap_or(&want),
                live,
                element,
            )?;
            let (compared, failures) = check.finish(session)?;
            let wanted = expected.map(|text| format!("{case}: {text}"));
            if compared != 1 || failures != wanted.clone().into_iter().collect::<Vec<_>>() {
                return Err(format!(
                    "{case}: expected {wanted:?}, the check gave {failures:?}"
                ));
            }
            observed.push(json!({"case": case, "failures": failures}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("device_check_controls"));
        receipt.insert("controls".into(), json!(observed));
        receipt.insert("passed".into(), json!(true));
        session.receipt("tiled_chain_splits", receipt)?;
        Ok(())
    })
    .expect("device checks reject unwritten and corrupt elements");
}

/// One output buffer per input, the size of the fixture's outputs.
fn reference_outputs(session: &Session, fixture: &Fixture) -> LabResult<Vec<DeviceBuffer>> {
    fixture
        .outputs
        .iter()
        .map(|output| session.allocate(output.byte_len()))
        .collect()
}

/// Runs without a GPU: each decoded-tile case (of either sweep) is swept at
/// `decoded_counts` (both tile heights' edges, for the entry of today too)
/// and qualified on the device at every count of `decoded_serving_counts`,
/// which is every count from 1 through its capacity.
#[test]
fn decoded_tile_counts_cover_serving() {
    for case in decoded_shapes::cases()
        .into_iter()
        .chain(balanced_shapes::cases())
        .chain(staged_shapes::cases())
    {
        let capacity = case.query.m_capacity;
        assert_eq!(capacity, 8192, "{}", case.tag);
        let counts = decoded_serving_counts(capacity).collect::<Vec<_>>();
        assert_eq!(counts, (1..=capacity).collect::<Vec<_>>(), "{}", case.tag);
        assert!(
            decoded_counts(&case)
                .iter()
                .all(|m| decoded_serving_counts(capacity).contains(m))
        );
    }
}

/// `check_strip_case` over every decoded-tile sweep case at `decoded_counts`:
/// the decoded tiles and the warp tiles of today equal the one-split entry
/// bit for bit (the reference run twice), and the decoded tiles pass
/// `qualify` at `poison::live_counts`. `CHOPINFER_STRIP_FILTER` keeps the
/// cases whose tag contains it.
#[test]
fn decoded_tiles_keep_one_split_bits() {
    keep_one_split_bits(&DECODED, "decoded tiles keep one-split bits");
}

/// `decoded_tiles_keep_one_split_bits` over the balanced sweep: the
/// balanced and the decoded tiles equal the one-split entry bit for bit.
#[test]
fn balanced_decoded_tiles_keep_one_split_bits() {
    keep_one_split_bits(&BALANCED, "balanced decoded tiles keep one-split bits");
}

/// `decoded_tiles_keep_one_split_bits` over the staged-tile sweep: the
/// staged tiles and every warp tile equal the one-split entry bit for bit.
#[test]
fn staged_tiles_keep_one_split_bits() {
    keep_one_split_bits(&STAGED, "staged tiles keep one-split bits");
}

fn keep_one_split_bits(family: &DecodedFamily, what: &str) {
    gpu_test(8 << 30, |session| {
        let producer = Producer::new(session)?;
        let filter = std::env::var("CHOPINFER_STRIP_FILTER").ok();
        let mut checked = 0;
        for case in (family.cases)() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let module = session.module(&(family.cubin_name)(case.shape))?;
            let counts = decoded_counts(&case);
            check_strip_case(
                session,
                &case,
                &module,
                &producer,
                &counts,
                &[],
                &[],
                family.receipts[0],
            )?;
            checked += 1;
        }
        if checked == 0 {
            return Err("no decoded-tile case ran".into());
        }
        Ok(())
    })
    .expect(what);
}

/// Every decoded-tile sweep case passes `qualify_on_device` against the
/// one-split entry at every count of `decoded_serving_counts` (or of the
/// part `device_check::LIVE_COUNTS` names), behind the wide producer, with
/// a receipt per `DEVICE_CHUNK` counts. `CHOPINFER_STRIP_FILTER` keeps the
/// cases whose tag contains it.
#[test]
fn decoded_tiles_pass_at_every_serving_count() {
    pass_at_every_serving_count(&DECODED, "decoded tiles pass at every serving count");
}

/// `decoded_tiles_pass_at_every_serving_count` over the staged-tile sweep
/// (the staged tiles).
#[test]
fn staged_tiles_pass_at_every_serving_count() {
    pass_at_every_serving_count(&STAGED, "staged tiles pass at every serving count");
}

/// `decoded_tiles_pass_at_every_serving_count` over the balanced sweep.
#[test]
fn balanced_decoded_tiles_pass_at_every_serving_count() {
    pass_at_every_serving_count(
        &BALANCED,
        "balanced decoded tiles pass at every serving count",
    );
}

fn pass_at_every_serving_count(family: &DecodedFamily, what: &str) {
    gpu_test(16 << 30, |session| {
        let producer = Producer::wide(session)?;
        let mut check = DeviceCheck::new(session, DEVICE_SLOTS)?;
        let filter = std::env::var("CHOPINFER_STRIP_FILTER").ok();
        let mut checked = 0;
        for case in (family.cases)() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let capacity = case.query.m_capacity;
            let counts = device_check::live_counts(decoded_serving_counts(capacity))?;
            let module = session.module(&(family.cubin_name)(case.shape))?;
            let decoded = &case.candidates[0];
            let specs = [&case.one, decoded];
            let fixture = Fixture::new(session, case.query, &specs)?;
            let references = reference_outputs(session, &fixture)?;
            let [one, candidate] = specs.map(|spec| {
                let shared = gemm::plan(&spec.key, gemm::row_counts(&spec.key, capacity))
                    .map_err(err)?
                    .dynamic_shared_bytes;
                function(&module, &spec.symbol, shared)
            });
            let (one, candidate) = (one?, candidate?);
            let started = std::time::Instant::now();
            let mut comparisons = 0;
            let mut first = *counts.start();
            for m in counts.clone() {
                let (compared, failures) = qualify_on_device(
                    session,
                    &fixture,
                    &references,
                    (&case.one, &one),
                    (decoded, &candidate),
                    &producer,
                    &mut check,
                    m,
                    m,
                )?;
                if !failures.is_empty() {
                    return Err(failures.join("; "));
                }
                comparisons += compared;
                // A receipt per `DEVICE_CHUNK` counts, so a run cut short
                // keeps the counts it passed.
                if m == *counts.end() || (m - first + 1) as usize == DEVICE_CHUNK {
                    let mut receipt = BTreeMap::new();
                    receipt.insert("case".into(), json!(family.receipts[1]));
                    receipt.insert("tag".into(), json!(case.tag));
                    receipt.insert("reference".into(), json!(case.one.symbol));
                    receipt.insert("candidate".into(), json!(decoded.symbol));
                    receipt.insert("capacity".into(), json!(capacity));
                    receipt.insert("first".into(), json!(first));
                    receipt.insert("last".into(), json!(m));
                    receipt.insert("phases".into(), json!(DEVICE_PHASES));
                    receipt.insert("comparisons".into(), json!(comparisons));
                    receipt.insert("passed".into(), json!(true));
                    session.receipt("tiled_chain_splits", receipt)?;
                    eprintln!(
                        "{}: {} = {} at every count {first}..={m}, {comparisons} device \
                         comparisons, {:.1} s",
                        case.tag,
                        decoded.symbol,
                        case.one.symbol,
                        started.elapsed().as_secs_f64()
                    );
                    first = m + 1;
                    comparisons = 0;
                }
            }
            checked += 1;
        }
        if checked == 0 {
            return Err("no decoded-tile case ran".into());
        }
        Ok(())
    })
    .expect(what);
}

/// `qualify_on_device` (behind the producer of the checks at every count)
/// against the unmodified decoded-tile mutant instance at the mutant query's
/// capacity: it passes for that instance and gives an observed difference
/// for every mutant (each from a fresh fixture), the wait mutants' among
/// them behind the producer. A launch one row short of the checked rows (in
/// reference and candidate alike) fails every comparison on the unwritten
/// row. Lists the differences.
fn device_mutant_differences(
    session: &Session,
    module: &Module,
    producer: &Producer,
    family: &DecodedFamily,
) -> LabResult<Vec<serde_json::Value>> {
    let mutant = family.mutant;
    let (tag, query) = (family.mutant_query)();
    let m = query.m_capacity;
    let control = mutant(decoded_shapes::MUTATIONS[0]);
    let shared = gemm::plan(&control.key, gemm::row_counts(&control.key, m))
        .map_err(err)?
        .dynamic_shared_bytes;
    let control_fn = function(module, &control.symbol, shared)?;
    let mut check = DeviceCheck::new(session, DEVICE_SLOTS)?;
    let mut results = Vec::new();
    for &mutation in &decoded_shapes::MUTATIONS {
        let instance = mutant(mutation);
        let instance_fn = function(module, &instance.symbol, shared)?;
        let fixture = Fixture::new(session, query, &[&control])?;
        let references = reference_outputs(session, &fixture)?;
        let (_, failures) = qualify_on_device(
            session,
            &fixture,
            &references,
            (&control, &control_fn),
            (&instance, &instance_fn),
            producer,
            &mut check,
            m,
            m,
        )?;
        if mutation != decoded_shapes::MUTATIONS[0]
            && mutation != "long_chain"
            && !failures.iter().any(|f| f.contains("behind the producer"))
        {
            return Err(format!(
                "the {mutation} mutant shows no difference behind the producer: {failures:?}"
            ));
        }
        match (
            mutation == decoded_shapes::MUTATIONS[0],
            failures.is_empty(),
        ) {
            (true, false) => {
                return Err(format!(
                    "the unmodified {tag} instance differs: {failures:?}"
                ));
            }
            (false, true) => return Err(format!("the {mutation} mutant passed the device checks")),
            (true, true) => {}
            (false, false) => {
                eprintln!("{tag} {mutation} on the device: {failures:?}");
                results.push(json!({"mutation": mutation, "differs": failures}));
            }
        }
    }
    // The unwritten-row control: both entries launched at one row fewer
    // than checked leave that row's poison in the reference and the
    // candidate alike; every comparison must count it unwritten.
    let fixture = Fixture::new(session, query, &[&control])?;
    let references = reference_outputs(session, &fixture)?;
    let (compared, failures) = qualify_on_device(
        session,
        &fixture,
        &references,
        (&control, &control_fn),
        (&control, &control_fn),
        producer,
        &mut check,
        m - 1,
        m,
    )?;
    let row = query.n as usize;
    if failures.len() != compared
        || !failures
            .iter()
            .all(|f| f.contains(&format!(" {row} live elements unwritten")))
    {
        return Err(format!(
            "a launch one row short passed {} of {compared} comparisons: {failures:?}",
            compared - failures.len()
        ));
    }
    eprintln!("{tag} one row short on the device: {failures:?}");
    results.push(json!({"mutation": "one_row_short", "differs": failures}));
    Ok(results)
}

/// The chain checks reject every decoded-tile mutant (`reject_mutants`): no
/// dependency wait and the math warps' wait after the K loop (behind the
/// producer), and 256-K chains (every launch); so do the device checks
/// (`device_mutant_differences`).
#[test]
fn decoded_tile_checks_reject_every_mutant() {
    reject_every_mutant(&DECODED, "decoded tile checks reject every mutant");
}

/// `decoded_tile_checks_reject_every_mutant` over the staged-tile mutants
/// (128-K chains for the long-chain mutant).
#[test]
fn staged_tile_checks_reject_every_mutant() {
    reject_every_mutant(&STAGED, "staged tile checks reject every mutant");
}

/// `decoded_tile_checks_reject_every_mutant` over the balanced mutants.
#[test]
fn balanced_decoded_checks_reject_every_mutant() {
    reject_every_mutant(
        &BALANCED,
        "balanced decoded tile checks reject every mutant",
    );
}

fn reject_every_mutant(family: &DecodedFamily, what: &str) {
    gpu_test(4 << 30, |session| {
        let module = session.module(family.mutants_cubin)?;
        let producer = Producer::new(session)?;
        let (tag, query) = (family.mutant_query)();
        let results = reject_mutants(
            session,
            &module,
            &producer,
            tag,
            query,
            &decoded_shapes::MUTATIONS,
            family.mutant,
        )?;
        let device =
            device_mutant_differences(session, &module, &Producer::wide(session)?, family)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(family.receipts[2]));
        receipt.insert("mutants".into(), json!(results));
        receipt.insert("device_checks".into(), json!(device));
        receipt.insert("passed".into(), json!(true));
        session.receipt("tiled_chain_splits", receipt)?;
        Ok(())
    })
    .expect(what);
}

// ------------------------------------------------------- bucket rows ----

/// The widest decode and verify graph bucket.
const WIDEST: u32 = 32;

/// A decode or verify bucket's tiled entry and its operation's widest
/// graph bucket entry of the same use phase.
struct BucketPair {
    candidate: InstantiationSpec,
    reference: InstantiationSpec,
}

/// An offer's capability with its bucket, implementation and stages
/// cleared: equal for the buckets of one operation and use phase.
fn bucket_key(offer: &KernelOffer) -> LabResult<OperationCapability> {
    let mut capability = offer.capability().map_err(err)?;
    capability.usage.capacity = 0;
    capability.implementation.clear();
    capability.stages.clear();
    Ok(capability)
}

/// The GEMM shapes `CHOPINFER_BUCKET_SHAPES` lists (`NxK`), or none to
/// take every shape.
fn bucket_shapes() -> LabResult<Option<Vec<(u32, u32)>>> {
    let Ok(listed) = std::env::var("CHOPINFER_BUCKET_SHAPES") else {
        return Ok(None);
    };
    listed
        .split(',')
        .map(|shape| {
            let (n, k) = shape
                .split_once('x')
                .ok_or_else(|| format!("{shape} is not NxK"))?;
            Ok((n.parse().map_err(err)?, k.parse().map_err(err)?))
        })
        .collect::<LabResult<Vec<_>>>()
        .map(Some)
}

/// Every graph bucket below `WIDEST` of a tiled decode or verify offer in
/// the inventory, with the offer of its operation at `WIDEST`, once per
/// pair of symbols.
fn bucket_pairs() -> LabResult<Vec<BucketPair>> {
    use kernel_core::precision::ExecutionPhase;
    let (offers, _, _) = kernel_lab::models::projection_inventory()?;
    fn tiled(offer: &KernelOffer) -> Option<&InstantiationSpec> {
        dense(offer).filter(|spec| {
            matches!(
                spec.key.dense,
                Some(DenseKernel::Tiled { .. } | DenseKernel::TiledChains { .. })
            ) && offer.usage().graph
                && matches!(
                    offer.usage().phase,
                    ExecutionPhase::Decode | ExecutionPhase::Verify
                )
        })
    }
    let mut widest = Vec::new();
    for offer in &offers {
        if let Some(spec) = tiled(offer)
            && offer.usage().capacity == WIDEST
        {
            widest.push((bucket_key(offer)?, spec));
        }
    }
    let mut pairs = BTreeMap::new();
    for offer in &offers {
        let Some(spec) = tiled(offer) else { continue };
        if offer.usage().capacity >= WIDEST {
            continue;
        }
        let key = bucket_key(offer)?;
        let Some((_, reference)) = widest.iter().find(|(widest, _)| *widest == key) else {
            continue;
        };
        pairs
            .entry((spec.symbol.clone(), reference.symbol.clone()))
            .or_insert_with(|| BucketPair {
                candidate: spec.clone(),
                reference: (*reference).clone(),
            });
    }
    Ok(pairs.into_values().collect())
}

/// Whether `Fixture::new` builds operands for the pair's weights (BF16
/// and FP8 128x128); the GPU check excludes the others and names them.
fn fixture_builds(pair: &BucketPair) -> bool {
    gemm_query(&pair.reference).is_ok_and(|query| {
        matches!(
            query.weight,
            WeightFormat::Bf16 | WeightFormat::Fp8Block128x128(_)
        )
    })
}

/// Runs without a GPU: bucket pairs exist, some of weights the fixture
/// builds, each pair's GEMMs differ only in their row capacity, and with
/// reports listed both entries of every pair are built.
#[test]
fn bucket_pairs_differ_only_in_rows() {
    let pairs = bucket_pairs().unwrap();
    assert!(pairs.iter().any(fixture_builds));
    for pair in pairs.iter().filter(|pair| !fixture_builds(pair)) {
        eprintln!(
            "excluded (no fixture for its weights): {}",
            pair.candidate.symbol
        );
    }
    for pair in &pairs {
        let (Ok(mut a), Ok(b)) = (gemm_query(&pair.candidate), gemm_query(&pair.reference)) else {
            panic!("not a GEMM");
        };
        assert!(a.m_capacity < b.m_capacity, "{}", pair.candidate.symbol);
        eprintln!(
            "n{} k{} m{}: {} against {}",
            a.n, a.k, a.m_capacity, pair.candidate.symbol, pair.reference.symbol
        );
        a.m_capacity = b.m_capacity;
        assert_eq!(a, b, "{}", pair.candidate.symbol);
    }
    if std::env::var_os("CHOPINFER_BUNDLE_REPORTS").is_some() {
        let current = reported("CHOPINFER_BUNDLE_REPORTS").unwrap();
        let built = pairs
            .iter()
            .filter(|pair| cubin_of(&current, &pair.reference.symbol).is_some())
            .inspect(|pair| {
                assert!(
                    cubin_of(&current, &pair.candidate.symbol).is_some(),
                    "{}",
                    pair.candidate.symbol
                );
            })
            .count();
        eprintln!("{built} of {} bucket pairs built", pairs.len());
    }
}

/// One pair's bits: `BucketPair` at every live count of the candidate's
/// bucket, then its `qualify` checks at `poison::live_counts`.
fn check_bucket_pair(
    session: &Session,
    pair: &BucketPair,
    module: &Module,
    reference_module: &Module,
    producer: &Producer,
) -> LabResult<()> {
    let StaticQuery::Gemm(query) = pair.reference.key.query else {
        return Err("not a GEMM".into());
    };
    let capacity = m_capacity(&pair.candidate)?;
    let fixture = Fixture::new(session, query, &[&pair.reference, &pair.candidate])?;
    let shared = |spec: &InstantiationSpec| -> LabResult<u32> {
        let rows = m_capacity(spec)?;
        gemm::plan(&spec.key, gemm::row_counts(&spec.key, rows))
            .map(|plan| plan.dynamic_shared_bytes)
            .map_err(err)
    };
    let reference = function(
        reference_module,
        &pair.reference.symbol,
        shared(&pair.reference)?,
    )?;
    let candidate = function(module, &pair.candidate.symbol, shared(&pair.candidate)?)?;
    let n = query.n as usize;
    let want = fixture.separate(session, &pair.reference, &reference, WIDEST)?;
    let again = fixture.separate(session, &pair.reference, &reference, WIDEST)?;
    if want != again {
        return Err(format!(
            "reference {} varies between runs",
            pair.reference.symbol
        ));
    }
    let rows = |m: u32| m as usize * fixture.row_bytes;
    for m in 1..=capacity {
        let got = fixture.separate(session, &pair.candidate, &candidate, m)?;
        for (input, (want, got)) in want.iter().zip(&got).enumerate() {
            if let Some(e) = differs(
                &format!(
                    "{} vs {} at M={m} input {input}",
                    pair.candidate.symbol, pair.reference.symbol
                ),
                n,
                fixture.element_bytes,
                &want[..rows(m)],
                &got[..rows(m)],
            ) {
                return Err(e);
            }
        }
    }
    let mut evidence = Vec::new();
    for m in poison::live_counts(capacity).into_iter().filter(|m| *m > 0) {
        // The widest entry's live rows, the rest as poisoned.
        let expected = want
            .iter()
            .map(|output| {
                let mut bytes = output[..rows(m)].to_vec();
                bytes.resize(output.len(), poison::POISON);
                bytes
            })
            .collect::<Vec<_>>();
        evidence.push(qualify(
            session,
            &fixture,
            &pair.candidate,
            &candidate,
            producer,
            m,
            &expected,
        )?);
    }
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("tiled_bucket_rows"));
    receipt.insert("candidate".into(), json!(pair.candidate.symbol));
    receipt.insert("reference".into(), json!(pair.reference.symbol));
    receipt.insert("weight".into(), json!(format!("{:?}", query.weight)));
    receipt.insert("n".into(), json!(n));
    receipt.insert("k".into(), json!(query.k));
    receipt.insert("capacity".into(), json!(capacity));
    receipt.insert("reference_capacity".into(), json!(WIDEST));
    receipt.insert(
        "kernel".into(),
        json!(format!("{:?}", pair.candidate.key.dense)),
    );
    receipt.insert(
        "bitwise_live_counts".into(),
        json!((1..=capacity).collect::<Vec<_>>()),
    );
    receipt.insert("inputs".into(), json!(INPUTS));
    receipt.insert("qualified".into(), json!(evidence));
    receipt.insert("passed".into(), json!(true));
    session.receipt("tiled_bucket_rows", receipt)?;
    eprintln!(
        "{} = the first rows of {} bit for bit at every live count up to {capacity}; producer \
         and back-to-back checks passed",
        pair.candidate.symbol, pair.reference.symbol
    );
    Ok(())
}

#[test]
fn bucket_entries_keep_the_widest_rows() {
    if std::env::var_os("CHOPINFER_BUNDLE_REPORTS").is_none() {
        kernel_lab::skip("bucket rows: set CHOPINFER_BUNDLE_REPORTS").unwrap();
        return;
    }
    gpu_test(8 << 30, |session| {
        let current = reported("CHOPINFER_BUNDLE_REPORTS")?;
        let shapes = bucket_shapes()?;
        let producer = Producer::new(session)?;
        let mut modules: BTreeMap<PathBuf, Module> = BTreeMap::new();
        let mut load = |path: PathBuf| -> LabResult<PathBuf> {
            if !modules.contains_key(&path) {
                let bytes = std::fs::read(&path).map_err(err)?;
                modules.insert(path.clone(), session.load_cubin(&bytes)?);
            }
            Ok(path)
        };
        let mut resolved = Vec::new();
        let mut excluded = Vec::new();
        for pair in bucket_pairs()? {
            let query = gemm_query(&pair.reference)?;
            if shapes
                .as_ref()
                .is_some_and(|shapes| !shapes.contains(&(query.n, query.k)))
            {
                continue;
            }
            let Some(reference) = cubin_of(&current, &pair.reference.symbol) else {
                continue;
            };
            if !fixture_builds(&pair) {
                excluded.push(format!("{} ({:?})", pair.candidate.symbol, query.weight));
                continue;
            }
            let candidate = cubin_of(&current, &pair.candidate.symbol)
                .ok_or_else(|| format!("no current report builds {}", pair.candidate.symbol))?;
            resolved.push((pair, load(candidate)?, load(reference)?));
        }
        for symbol in &excluded {
            eprintln!("excluded (no fixture for its weights): {symbol}");
        }
        if resolved.is_empty() {
            return Err("no bucket pair of the listed shapes in the reports".into());
        }
        // Every listed shape has a checked pair.
        for &(n, k) in shapes.iter().flatten() {
            if !resolved.iter().any(|(pair, _, _)| {
                gemm_query(&pair.reference).is_ok_and(|query| (query.n, query.k) == (n, k))
            }) {
                return Err(format!(
                    "no checked bucket pair of the listed shape {n}x{k}"
                ));
            }
        }
        for (pair, candidate, reference) in &resolved {
            check_bucket_pair(
                session,
                pair,
                &modules[candidate],
                &modules[reference],
                &producer,
            )?;
        }
        eprintln!(
            "{} bucket pairs keep the widest rows; {} excluded for their weights",
            resolved.len(),
            excluded.len()
        );
        Ok(())
    })
    .expect("bucket entries keep the widest bucket's rows");
}
