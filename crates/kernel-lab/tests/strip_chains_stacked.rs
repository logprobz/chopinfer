//! Stacked strip-chain GEMMs (`gemm::stacked`) and the L2 weight prefetch
//! (`gemm::prefetch`), from kernel-lab's own cubins (family
//! `strip_chains_stacked`, `strip_chain_stacked_shapes.rs`).
//!
//! - CPU: every lab instance is built.
//! - GPU bits (`stacked_strips_keep_part_bits`): per stack and capacity, at
//!   every live count through the capacity, three inputs each in its own
//!   launch from poisoned outputs and workspace, every candidate's outputs
//!   (each part's leading columns at its place in a shared or compacted
//!   buffer, every other byte still poison) equal each part's one-split
//!   tiled output placed there, bit for bit; today's strip entry of each
//!   part shape equals the one-split entry too. At `poison::live_counts`,
//!   every candidate runs eager, behind the PDL producer (which holds input
//!   0 at NaN until its tail ends), and back to back on shared workspace
//!   and tickets, eager and from a graph replayed twice.
//! - Mutants (`stacked_checks_reject_every_mutant`): the same phases on the
//!   stacked kernel's deliberate defects; each must differ somewhere and the
//!   unmodified instance nowhere (only compared output differences count;
//!   launch or readback errors fail the test).
//! - Prefetch order (`prefetch_keeps_chain_order`): producer, prefetch over
//!   the parts' weights, stacked GEMM, eager and from a graph: the output is
//!   the expected one, and the weights are unchanged. The prefetch built
//!   without its trailing wait is run as a probe: whether its dependent then
//!   reads the producer's held input is recorded, not required.
//! - Timing (`stacked_prefetch_latency`, `CHOPINFER_KERNEL_BENCH=1`): cold
//!   L2 as served. Each round evicts L2, then runs one graph of `LAYERS`
//!   layers over distinct weight copies (more than twice the L2 in total),
//!   each layer two idle windows standing in for the collective and the
//!   mixer before the projections (released early, as those kernels release
//!   theirs), then the projections: today's entries with their layout
//!   copies, every stacked candidate, and either of those behind a
//!   prefetch launched between the two windows. The exposed time per layer
//!   is the graph's time minus the windows-only graph's, divided by the
//!   layers; every variant's outputs are compared with the expected bits.
//!
//! Run: `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release
//! --test strip_chains_stacked`; `CHOPINFER_STACKED_FILTER` keeps the cases
//! whose tag contains it.

use kernel_core::contract::{Argument, ArgumentList, DType, InstantiationSpec, LaunchPlan};
use kernel_lab::families::strip_chains_stacked::shapes;
use kernel_lab::launch;
use kernel_lab::poison::{self, Replay, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test, time};
use kernels_sm120::gemm::prefetch::Prefetch;
use kernels_sm120::gemm::stacked::{Part, Stacked};
use kernels_sm120::gemm::{self, GemmBuffers};
use kernels_sm120::row::precision::LayoutCopy;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;

const ACTIVATION: u64 = 2;
const PART_OUTPUT: u64 = 3;
const PARTIAL: u64 = 4;
const SEMAPHORES: u64 = 5;
const LIVE: u64 = 6;
/// Part p's weight is handle `WEIGHT + p`, its stacked output `OUTPUT + p`.
const WEIGHT: u64 = 100;
const OUTPUT: u64 = 200;

/// Activations of one case's checks.
const INPUTS: usize = 3;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn filtered(tag: &str) -> bool {
    std::env::var("CHOPINFER_STACKED_FILTER").map_or(true, |f| tag.contains(&f))
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

    /// Signed BF16 values with exponents -15 through 14; with `zero_rows`,
    /// every seventh row of `width` is zero.
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

/// What an argument handle resolves to: a whole buffer, or a byte range of
/// one (a part's place in a shared output).
#[derive(Clone, Copy)]
enum Bound<'a> {
    Whole(&'a DeviceBuffer),
    Range(&'a DeviceBuffer, usize),
}

type Handles<'a> = BTreeMap<u64, Bound<'a>>;

/// Launches `function` with `args`, tensor maps over whole buffers and
/// pointers resolved through `handles`.
fn run(
    session: &Session,
    function: &Function,
    stream: &Stream,
    plan: LaunchPlan,
    args: &ArgumentList,
    handles: &Handles<'_>,
) -> LabResult<()> {
    let resolve = |handle: u64| {
        handles
            .get(&handle)
            .copied()
            .ok_or_else(|| format!("unbound handle {handle}"))
    };
    let mut maps = Vec::new();
    for argument in &args.0 {
        if let Argument::TensorMap(spec) = argument {
            let Bound::Whole(buffer) = resolve(spec.base_buffer)? else {
                return Err("a tensor map spans a whole buffer".into());
            };
            maps.push(launch::tensor_map(&session.context, spec, buffer)?);
        }
    }
    let mut next_map = maps.iter();
    let mut packed = Vec::with_capacity(args.0.len());
    for argument in &args.0 {
        packed.push(match argument {
            Argument::DeviceBuffer(handle) => match resolve(*handle)? {
                Bound::Whole(buffer) => KernelArgument::Buffer(buffer),
                Bound::Range(buffer, offset) => KernelArgument::Range {
                    buffer,
                    offset,
                    byte_len: buffer.byte_len() - offset,
                },
            },
            Argument::U32(value) => KernelArgument::U32(*value),
            Argument::U64(value) => KernelArgument::U64(*value),
            Argument::I32(value) => KernelArgument::I32(*value),
            Argument::NullDevicePointer => KernelArgument::NullBuffer,
            Argument::TensorMap(_) => {
                KernelArgument::TensorMap(next_map.next().ok_or("missing tensor map")?)
            }
            Argument::DevicePointerArray(_) => return Err("no pointer arrays here".into()),
        });
    }
    launch::launch_with(function, stream, launch::config(plan)?, &packed)
}

/// The PDL producer probe (`chop_probe_pdl_prefetch_producer_v1`): after
/// its own wait it saves the activation it is given, overwrites it with
/// BF16 NaN pairs, releases its dependents, waits `PRODUCER_TAIL_NS` and
/// only then restores it, so a consumer that reads its input before its
/// own dependency wait sees NaN.
struct Producer {
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    live: DeviceBuffer,
}

/// Longer than any checked launch sequence.
const PRODUCER_TAIL_NS: u64 = 100_000;

impl Producer {
    fn new(session: &Session) -> LabResult<Self> {
        Ok(Self {
            function: session
                .module("probe_pdl_prefetch_sm120")?
                .function("chop_probe_pdl_prefetch_producer_v1")
                .map_err(err)?,
            eviction: upload(session, &[0; 16])?,
            checksum: session.allocate(128 * 4)?,
            live: session.allocate(4)?,
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
                    (1, 1, 1),
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

/// A one-part GEMM entry's launch over part `part`'s weight into `output`.
#[allow(clippy::too_many_arguments)]
fn launch_part(
    session: &Session,
    spec: &InstantiationSpec,
    function: &Function,
    weight: &DeviceBuffer,
    activation: &DeviceBuffer,
    output: Bound<'_>,
    workspace: &DeviceBuffer,
    semaphores: &DeviceBuffer,
    m: u32,
    stream: &Stream,
) -> LabResult<()> {
    let counts = gemm::row_counts(&spec.key, m);
    let plan = gemm::plan(&spec.key, counts).map_err(err)?;
    let maps = gemm::tensor_maps(&spec.key, WEIGHT, ACTIVATION).map_err(err)?;
    let args = gemm::arguments(
        &spec.key,
        counts,
        maps,
        GemmBuffers {
            output: PART_OUTPUT,
            partial: PARTIAL,
            semaphores: SEMAPHORES,
            weight_scales: None,
            activation_scales: None,
            residual: None,
            weight_global_scale: None,
            activation_global_scale: None,
        },
    )
    .map_err(err)?;
    let handles = Handles::from([
        (WEIGHT, Bound::Whole(weight)),
        (ACTIVATION, Bound::Whole(activation)),
        (PART_OUTPUT, output),
        (PARTIAL, Bound::Whole(workspace)),
        (SEMAPHORES, Bound::Whole(semaphores)),
    ]);
    run(session, function, stream, plan, &args, &handles)
}

/// The shared-memory size a GEMM entry is configured for at `capacity`.
fn gemm_shared(spec: &InstantiationSpec, capacity: u32) -> LabResult<u32> {
    Ok(gemm::plan(&spec.key, gemm::row_counts(&spec.key, capacity))
        .map_err(err)?
        .dynamic_shared_bytes)
}

/// One stack's operands at one capacity: each part's weights, `INPUTS`
/// activations with the producer's save areas and descriptors, one set of
/// output buffers per input (one per placement slot), a contiguous part
/// output for the references, and one workspace and ticket array every
/// launch shares.
struct Fixture {
    parts: Vec<Part>,
    capacity: u32,
    placed: Vec<(usize, u32)>,
    strides: Vec<u32>,
    weight_bytes: Vec<Vec<u8>>,
    weights: Vec<DeviceBuffer>,
    activations: Vec<DeviceBuffer>,
    fresh: Vec<DeviceBuffer>,
    operands: Vec<DeviceBuffer>,
    outputs: Vec<Vec<DeviceBuffer>>,
    part_output: DeviceBuffer,
    workspace: DeviceBuffer,
    semaphores: DeviceBuffer,
}

impl Fixture {
    fn new(
        session: &Session,
        parts: &[Part],
        capacity: u32,
        stacked: &[&Stacked],
        references: &[&InstantiationSpec],
    ) -> LabResult<Self> {
        let k = shapes::K as usize;
        let mut rng = Rng(0x5eed_51ac ^ u64::from(capacity) << 8 ^ parts.len() as u64);
        let placed = shapes::placements(parts);
        let slots = placed.iter().map(|&(slot, _)| slot + 1).max().unwrap_or(0);
        let strides: Vec<u32> = (0..slots)
            .map(|slot| {
                let part = placed
                    .iter()
                    .position(|&(s, _)| s == slot)
                    .expect("slot part");
                parts[part].row_stride
            })
            .collect();
        let weight_bytes: Vec<Vec<u8>> = parts
            .iter()
            .map(|p| rng.bf16_bytes(p.rows as usize * k, k, false))
            .collect();
        let mut workspace = 4u64;
        let mut semaphores = 4u64;
        for spec in stacked {
            workspace = workspace.max(spec.plan(capacity).map_err(err)?.workspace_bytes);
            semaphores = semaphores.max(spec.semaphore_bytes());
        }
        for spec in references {
            let plan = gemm::plan(&spec.key, gemm::row_counts(&spec.key, capacity)).map_err(err)?;
            workspace = workspace.max(plan.workspace_bytes);
            semaphores =
                semaphores.max(GemmBuffers::semaphore_capacity_bytes(&spec.key).map_err(err)?);
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
        let outputs = (0..INPUTS)
            .map(|_| {
                strides
                    .iter()
                    .map(|&stride| session.allocate(capacity as usize * stride as usize * 2))
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        let widest = parts.iter().map(|p| p.rows).max().unwrap_or(0) as usize;
        Ok(Self {
            parts: parts.to_vec(),
            capacity,
            placed,
            strides,
            weights: weight_bytes
                .iter()
                .map(|w| upload(session, w))
                .collect::<LabResult<Vec<_>>>()?,
            weight_bytes,
            activations,
            fresh,
            operands,
            outputs,
            part_output: session.allocate(capacity as usize * widest * 2)?,
            workspace: session.allocate(workspace as usize)?,
            semaphores: zeroed(session, semaphores as usize)?,
        })
    }

    /// The elements each slot's buffer holds written at `m` live rows.
    fn maps(&self, m: u32) -> LabResult<Vec<WriteMap>> {
        (0..self.strides.len())
            .map(|slot| {
                WriteMap::of(DType::Bf16).ranges(
                    self.parts
                        .iter()
                        .zip(&self.placed)
                        .filter(|(_, (s, _))| *s == slot)
                        .flat_map(|(part, &(_, offset))| {
                            let stride = part.row_stride as usize;
                            (0..m as usize).map(move |token| {
                                let start = token * stride + offset as usize;
                                start..start + part.width as usize
                            })
                        }),
                )
            })
            .collect()
    }

    /// Input `input`'s stacked handles: weights, activation, each part's
    /// place in its slot's buffer, workspace and tickets.
    fn handles(&self, input: usize) -> Handles<'_> {
        let mut handles = Handles::from([
            (ACTIVATION, Bound::Whole(&self.activations[input])),
            (PARTIAL, Bound::Whole(&self.workspace)),
            (SEMAPHORES, Bound::Whole(&self.semaphores)),
        ]);
        for (p, (weight, &(slot, offset))) in self.weights.iter().zip(&self.placed).enumerate() {
            handles.insert(WEIGHT + p as u64, Bound::Whole(weight));
            handles.insert(
                OUTPUT + p as u64,
                Bound::Range(&self.outputs[input][slot], offset as usize * 2),
            );
        }
        handles
    }

    fn launch(
        &self,
        session: &Session,
        spec: &Stacked,
        function: &Function,
        input: usize,
        m: u32,
        stream: &Stream,
    ) -> LabResult<()> {
        let parts = self.parts.len() as u64;
        let weights: Vec<u64> = (0..parts).map(|p| WEIGHT + p).collect();
        let outputs: Vec<u64> = (0..parts).map(|p| OUTPUT + p).collect();
        let args = spec
            .arguments(m, &weights, ACTIVATION, &outputs, PARTIAL, SEMAPHORES)
            .map_err(err)?;
        let plan = spec.plan(m).map_err(err)?;
        run(session, function, stream, plan, &args, &self.handles(input))
    }

    /// Part `part`'s contiguous `[m, rows]` output of `spec` on `input`,
    /// from a poisoned output and workspace.
    fn part_output(
        &self,
        session: &Session,
        spec: &InstantiationSpec,
        function: &Function,
        part: usize,
        input: usize,
        m: u32,
    ) -> LabResult<Vec<u8>> {
        poison::poison(session, &[&self.part_output, &self.workspace])?;
        launch_part(
            session,
            spec,
            function,
            &self.weights[part],
            &self.activations[input],
            Bound::Whole(&self.part_output),
            &self.workspace,
            &self.semaphores,
            m,
            &session.stream,
        )?;
        session.stream.synchronize().map_err(err)?;
        let bytes = poison::read(session, &self.part_output)?;
        Ok(bytes[..m as usize * self.parts[part].rows as usize * 2].to_vec())
    }

    /// Each input's slot images at `m` rows: poison, with each part's
    /// leading columns of its one-split output (`ones[p]`) at its place.
    fn expected(
        &self,
        session: &Session,
        ones: &[(&InstantiationSpec, &Function)],
        m: u32,
    ) -> LabResult<Vec<Vec<Vec<u8>>>> {
        (0..INPUTS)
            .map(|input| {
                let mut slots: Vec<Vec<u8>> = self
                    .strides
                    .iter()
                    .map(|&s| vec![poison::POISON; self.capacity as usize * s as usize * 2])
                    .collect();
                for (p, part) in self.parts.iter().enumerate() {
                    let (spec, function) = ones[p];
                    let rows = self.part_output(session, spec, function, p, input, m)?;
                    let (slot, offset) = self.placed[p];
                    for token in 0..m as usize {
                        let from = token * part.rows as usize * 2;
                        let to = (token * part.row_stride as usize + offset as usize) * 2;
                        let bytes = part.width as usize * 2;
                        slots[slot][to..to + bytes].copy_from_slice(&rows[from..from + bytes]);
                    }
                }
                Ok(slots)
            })
            .collect()
    }

    /// Each input's slot images of `spec` at `m` rows, each from one launch
    /// between host synchronizations, from poisoned outputs and workspace,
    /// with every write map checked.
    fn separate(
        &self,
        session: &Session,
        spec: &Stacked,
        function: &Function,
        m: u32,
    ) -> LabResult<Vec<Vec<Vec<u8>>>> {
        let maps = self.maps(m)?;
        (0..INPUTS)
            .map(|input| {
                let mut poisoned: Vec<&DeviceBuffer> = self.outputs[input].iter().collect();
                poisoned.push(&self.workspace);
                poison::poison(session, &poisoned)?;
                self.launch(session, spec, function, input, m, &session.stream)?;
                session.stream.synchronize().map_err(err)?;
                self.outputs[input]
                    .iter()
                    .zip(&maps)
                    .enumerate()
                    .map(|(slot, (buffer, map))| {
                        let bytes = poison::read(session, buffer)?;
                        poison::check(
                            &format!("{} M={m} input {input} slot {slot}", spec.symbol),
                            &bytes,
                            map,
                            poison::Fill::Pattern(&[poison::POISON]),
                        )?;
                        Ok(bytes)
                    })
                    .collect()
            })
            .collect()
    }

    /// The first differing element of two slot images, by part, row and
    /// column.
    fn differs(&self, label: &str, slot: usize, want: &[u8], got: &[u8]) -> Option<String> {
        let offset = want.iter().zip(got).position(|(a, b)| a != b)?;
        let stride = self.strides[slot] as usize;
        let element = offset / 2;
        let (token, column) = (element / stride, element % stride);
        let part = self
            .parts
            .iter()
            .zip(&self.placed)
            .position(|(p, &(s, o))| {
                s == slot && (o as usize..(o + p.width) as usize).contains(&column)
            });
        let count = want
            .chunks_exact(2)
            .zip(got.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        Some(format!(
            "{label}: slot {slot} row {token} column {column} (part {part:?}) differs; \
             {count} elements differ"
        ))
    }
}

/// The checks every stacked candidate passes and every mutant must fail,
/// against `expected` (each input's slot images at `m` rows): each input in
/// its own launch; eager, then behind the PDL producer; one sequence with no
/// host synchronization, every launch PDL and sharing one workspace and
/// ticket array (producer, input 0, producer, input 1, input 2), eager and
/// from one graph replayed twice. Outputs, the workspace and the save areas
/// are poisoned before every phase.
fn qualify(
    session: &Session,
    fixture: &Fixture,
    spec: &Stacked,
    function: &Function,
    producer: &Producer,
    m: u32,
    expected: &[Vec<Vec<u8>>],
) -> LabResult<serde_json::Value> {
    let label = |what: &str| format!("{} M={m} {what}", spec.symbol);
    let separate = fixture.separate(session, spec, function, m)?;
    for (input, (want, got)) in expected.iter().zip(&separate).enumerate() {
        for (slot, (want, got)) in want.iter().zip(got).enumerate() {
            if let Some(e) = fixture.differs(&label(&format!("input {input}")), slot, want, got) {
                return Err(e);
            }
        }
    }
    let maps = fixture.maps(m)?;
    let stream = &session.stream;
    let run =
        |input: usize, stream: &Stream| fixture.launch(session, spec, function, input, m, stream);
    let mut replay = Replay::new(session);
    for (slot, (buffer, map)) in fixture.outputs[0].iter().zip(&maps).enumerate() {
        replay = replay.output(format!("slot {slot}"), buffer, map.clone());
    }
    let mut replay = replay
        .scratch("chains", &fixture.workspace)
        .scratch("saved input", &fixture.fresh[0]);
    replay.eager(|| run(0, stream))?;
    replay.pdl(
        || producer.launch(stream, &fixture.operands[0]),
        || run(0, stream),
        true,
    )?;
    for (slot, want) in expected[0].iter().enumerate() {
        let got = replay
            .expected(&format!("slot {slot}"))
            .ok_or("missing eager output")?;
        if let Some(e) = fixture.differs(&label("behind the producer"), slot, want, got) {
            return Err(e);
        }
    }
    let mut pdl = BTreeMap::new();
    replay.record(&mut pdl);
    let sequence = |stream: &Stream| -> LabResult<()> {
        producer.launch(stream, &fixture.operands[0])?;
        run(0, stream)?;
        producer.launch(stream, &fixture.operands[1])?;
        run(1, stream)?;
        run(2, stream)
    };
    let mut replay = Replay::new(session);
    for (input, outputs) in fixture.outputs.iter().enumerate() {
        for (slot, (buffer, map)) in outputs.iter().zip(&maps).enumerate() {
            replay = replay.output(format!("input {input} slot {slot}"), buffer, map.clone());
        }
    }
    let mut replay = replay
        .scratch("chains", &fixture.workspace)
        .scratch("saved input 0", &fixture.fresh[0])
        .scratch("saved input 1", &fixture.fresh[1]);
    replay.eager(|| sequence(stream))?;
    for (input, want) in expected.iter().enumerate() {
        for (slot, want) in want.iter().enumerate() {
            let got = replay
                .expected(&format!("input {input} slot {slot}"))
                .ok_or("missing eager output")?;
            if let Some(e) =
                fixture.differs(&label(&format!("sequence input {input}")), slot, want, got)
            {
                return Err(e);
            }
        }
    }
    let graph = session.capture(sequence)?;
    replay.graph(&graph)?;
    replay.graph(&graph)?;
    let mut back_to_back = BTreeMap::new();
    replay.record(&mut back_to_back);
    Ok(json!({"m": m, "behind_producer": pdl, "back_to_back": back_to_back}))
}

/// The references of one case: per part its one-split and today's entry
/// with their functions, from the parts' cubin.
struct References {
    entries: Vec<shapes::PartEntries>,
    one: Vec<Function>,
    served: Vec<Function>,
}

impl References {
    fn new(module: &Module, parts: &[Part], capacity: u32) -> LabResult<Self> {
        let entries: Vec<_> = parts
            .iter()
            .map(|p| shapes::part_entries(p.rows, capacity))
            .collect();
        let load =
            |spec: &InstantiationSpec| function(module, &spec.symbol, gemm_shared(spec, capacity)?);
        Ok(Self {
            one: entries
                .iter()
                .map(|e| load(&e.one))
                .collect::<LabResult<_>>()?,
            served: entries
                .iter()
                .map(|e| load(&e.served))
                .collect::<LabResult<_>>()?,
            entries,
        })
    }

    fn ones(&self) -> Vec<(&InstantiationSpec, &Function)> {
        self.entries.iter().map(|e| &e.one).zip(&self.one).collect()
    }

    fn specs(&self) -> Vec<&InstantiationSpec> {
        self.entries
            .iter()
            .flat_map(|e| [&e.one, &e.served])
            .collect()
    }
}

fn stacked_function(module: &Module, spec: &Stacked) -> LabResult<Function> {
    let shared = spec.plan(spec.capacity).map_err(err)?.dynamic_shared_bytes;
    function(module, &spec.symbol, shared)
}

// ------------------------------------------------------------- CPU ----

/// Runs without a GPU: every lab instance is in its cubin, every case has
/// each schedule with and without prefetch, and the mutants differ from the
/// unmodified instance only in their mutation.
#[test]
fn stacked_lab_instances_are_built() {
    for case in shapes::cases() {
        let built = kernel_lab::cubin_entries(&shapes::cubin_name(case.stack)).unwrap();
        for spec in &case.candidates {
            assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        }
        assert_eq!(case.candidates.len(), 2 * shapes::SCHEDULES.len());
        let parts = kernel_lab::cubin_entries(shapes::PARTS_CUBIN).unwrap();
        for part in &case.parts {
            let entries = shapes::part_entries(part.rows, case.capacity);
            for spec in [&entries.one, &entries.served] {
                assert!(parts.contains(&spec.symbol.as_str()), "{}", spec.symbol);
            }
        }
        let copies = kernel_lab::cubin_entries(shapes::COPIES_CUBIN).unwrap();
        for (part, &(_, offset)) in case.parts.iter().zip(&shapes::placements(&case.parts)) {
            if let Some(copy) = shapes::layout_copy(part, offset, case.capacity) {
                let symbol = copy.stage().entry().symbol.as_str();
                assert!(copies.contains(&symbol), "{symbol}");
            }
        }
    }
    let prefetch = kernel_lab::cubin_entries(shapes::PREFETCH_CUBIN).unwrap();
    for entry in shapes::prefetches() {
        assert!(
            prefetch.contains(&entry.symbol.as_str()),
            "{}",
            entry.symbol
        );
    }
    let window = kernel_lab::cubin_entries(shapes::WINDOW_CUBIN).unwrap();
    assert!(window.contains(&shapes::IDLE_WINDOW));
    let built = kernel_lab::cubin_entries(shapes::MUTANTS_CUBIN).unwrap();
    for (mutation, spec) in shapes::mutants() {
        assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        assert!(spec.symbol.contains(&mutation));
    }
}

/// The three-part stack places its parts side by side in one buffer; the
/// compacted part gets its own.
#[test]
fn placements_share_strided_buffers() {
    let stacks = shapes::stacks();
    assert_eq!(
        shapes::placements(&stacks[0].1),
        [(0, 0), (0, 2048), (0, 4096)]
    );
    assert_eq!(
        shapes::placements(&stacks[1].1),
        [(0, 0), (0, 2048), (0, 4096), (1, 0), (2, 0), (3, 0)]
    );
}

// ------------------------------------------------------------- bits ----

/// One case: at every live count through its capacity, each input
/// separately, every candidate's slots equal the parts' one-split outputs
/// placed, and today's entry of each part equals its one-split entry; at
/// `poison::live_counts`, every candidate passes `qualify`.
fn check_case(
    session: &Session,
    case: &shapes::Case,
    module: &Module,
    parts_module: &Module,
    producer: &Producer,
) -> LabResult<()> {
    let capacity = case.capacity;
    let references = References::new(parts_module, &case.parts, capacity)?;
    let stacked: Vec<&Stacked> = case.candidates.iter().collect();
    let fixture = Fixture::new(
        session,
        &case.parts,
        capacity,
        &stacked,
        &references.specs(),
    )?;
    let functions = case
        .candidates
        .iter()
        .map(|spec| stacked_function(module, spec))
        .collect::<LabResult<Vec<_>>>()?;
    for m in 1..=capacity {
        let want = fixture.expected(session, &references.ones(), m)?;
        for (p, entries) in references.entries.iter().enumerate() {
            for input in 0..INPUTS {
                let one =
                    fixture.part_output(session, &entries.one, &references.one[p], p, input, m)?;
                let served = fixture.part_output(
                    session,
                    &entries.served,
                    &references.served[p],
                    p,
                    input,
                    m,
                )?;
                if one != served {
                    return Err(format!(
                        "{} differs from {} at M={m} input {input}",
                        entries.served.symbol, entries.one.symbol
                    ));
                }
            }
        }
        for (spec, f) in case.candidates.iter().zip(&functions) {
            let got = fixture.separate(session, spec, f, m)?;
            for (input, (want, got)) in want.iter().zip(&got).enumerate() {
                for (slot, (want, got)) in want.iter().zip(got).enumerate() {
                    if let Some(e) = fixture.differs(
                        &format!("{} at M={m} input {input}", spec.symbol),
                        slot,
                        want,
                        got,
                    ) {
                        return Err(e);
                    }
                }
            }
        }
    }
    let mut evidence = Vec::new();
    for m in poison::live_counts(capacity).into_iter().filter(|m| *m > 0) {
        let want = fixture.expected(session, &references.ones(), m)?;
        for (spec, f) in case.candidates.iter().zip(&functions) {
            evidence.push(json!({
                "symbol": spec.symbol,
                "checks": qualify(session, &fixture, spec, f, producer, m, &want)?,
            }));
        }
    }
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("stacked_strip_bits"));
    receipt.insert("tag".into(), json!(case.tag));
    receipt.insert(
        "parts".into(),
        json!(
            case.parts
                .iter()
                .map(|p| json!({"rows": p.rows, "width": p.width, "row_stride": p.row_stride}))
                .collect::<Vec<_>>()
        ),
    );
    receipt.insert(
        "references".into(),
        json!(
            references
                .entries
                .iter()
                .map(|e| json!({"one": e.one.symbol, "served": e.served.symbol}))
                .collect::<Vec<_>>()
        ),
    );
    receipt.insert(
        "candidates".into(),
        json!(
            case.candidates
                .iter()
                .map(
                    |s| json!({"symbol": s.symbol, "strips": s.strips, "splits": s.splits,
                    "prefetch": s.prefetch})
                )
                .collect::<Vec<_>>()
        ),
    );
    receipt.insert("k".into(), json!(shapes::K));
    receipt.insert("capacity".into(), json!(capacity));
    receipt.insert(
        "bitwise_live_counts".into(),
        json!((1..=capacity).collect::<Vec<_>>()),
    );
    receipt.insert("inputs".into(), json!(INPUTS));
    receipt.insert("qualified".into(), json!(evidence));
    receipt.insert("passed".into(), json!(true));
    session.receipt("strip_chains_stacked", receipt)?;
    eprintln!(
        "{}: {} stacked candidates = the parts' one-split entries placed, bit for bit at every \
         live count through {capacity}; today's part entries equal them; producer and \
         back-to-back checks passed",
        case.tag,
        case.candidates.len()
    );
    Ok(())
}

#[test]
fn stacked_strips_keep_part_bits() {
    gpu_test(8 << 30, |session| {
        let producer = Producer::new(session)?;
        let parts_module = session.module(shapes::PARTS_CUBIN)?;
        let mut checked = 0;
        for case in shapes::cases() {
            if !filtered(&case.tag) {
                continue;
            }
            let module = session.module(&shapes::cubin_name(case.stack))?;
            check_case(session, &case, &module, &parts_module, &producer)?;
            checked += 1;
        }
        if checked == 0 {
            return Err("no stacked case ran".into());
        }
        Ok(())
    })
    .expect("stacked strips keep the parts' bits");
}

// ---------------------------------------------------------- mutants ----

/// The phases of `qualify` in which `spec`'s outputs differ from
/// `expected`, each from poisoned outputs, workspace and save areas, read
/// back after a host synchronization. Launch, capture, synchronization and
/// readback errors propagate; only a compared difference counts.
fn mutant_differences(
    session: &Session,
    fixture: &Fixture,
    spec: &Stacked,
    function: &Function,
    producer: &Producer,
    m: u32,
    expected: &[Vec<Vec<u8>>],
) -> LabResult<Vec<String>> {
    let stream = &session.stream;
    let mut differing = Vec::new();
    let mut compare = |phase: &str, inputs: &[usize]| -> LabResult<()> {
        stream.synchronize().map_err(err)?;
        for &input in inputs {
            for (slot, buffer) in fixture.outputs[input].iter().enumerate() {
                if poison::read(session, buffer)? != expected[input][slot] {
                    differing.push(format!("{phase}: input {input} slot {slot}"));
                }
            }
        }
        Ok(())
    };
    let everything: Vec<&DeviceBuffer> = fixture
        .outputs
        .iter()
        .flatten()
        .chain([&fixture.workspace, &fixture.fresh[0], &fixture.fresh[1]])
        .collect();
    for input in 0..INPUTS {
        poison::poison(session, &everything)?;
        fixture.launch(session, spec, function, input, m, stream)?;
        compare("separate", &[input])?;
    }
    poison::poison(session, &everything)?;
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
    poison::poison(session, &everything)?;
    sequence(stream)?;
    compare("sequence", &[0, 1, 2])?;
    let graph = session.capture(sequence)?;
    for replay in 0..2 {
        poison::poison(session, &everything)?;
        graph.launch(stream).map_err(err)?;
        compare(&format!("graph replay {replay}"), &[0, 1, 2])?;
    }
    Ok(differing)
}

/// The stacked checks reject every mutant: no or a late dependency wait,
/// temporaries shared across tiles, a ticket never reset, and every part
/// reading the first part's weights; the unmodified instance passes.
#[test]
fn stacked_checks_reject_every_mutant() {
    gpu_test(4 << 30, |session| {
        let module = session.module(shapes::MUTANTS_CUBIN)?;
        let parts_module = session.module(shapes::PARTS_CUBIN)?;
        let producer = Producer::new(session)?;
        let mutants = shapes::mutants();
        let (_, first) = &mutants[0];
        let references = References::new(&parts_module, &first.parts, first.capacity)?;
        let specs: Vec<&Stacked> = mutants.iter().map(|(_, s)| s).collect();
        let fixture = Fixture::new(
            session,
            &first.parts,
            first.capacity,
            &specs,
            &references.specs(),
        )?;
        let m = first.capacity;
        let expected = fixture.expected(session, &references.ones(), m)?;
        let mut results = Vec::new();
        for (mutation, spec) in &mutants {
            // A kept ticket leaves its counts behind; every instance starts
            // from zeroed tickets.
            poison::fill_all(session, &fixture.semaphores, &[0])?;
            let f = stacked_function(&module, spec)?;
            let differing =
                mutant_differences(session, &fixture, spec, &f, &producer, m, &expected)?;
            eprintln!(
                "{} ({mutation}): {} differing outputs",
                spec.symbol,
                differing.len()
            );
            let rejected = !differing.is_empty();
            if (mutation == "none") == rejected {
                return Err(format!(
                    "{} ({mutation}): differing outputs {differing:?}",
                    spec.symbol
                ));
            }
            results.push(json!({"symbol": spec.symbol, "mutation": mutation,
                "splits": spec.splits, "differing": differing}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("stacked_strip_mutants"));
        receipt.insert("mutants".into(), json!(results));
        receipt.insert("passed".into(), json!(true));
        session.receipt("strip_chains_stacked", receipt)?;
        Ok(())
    })
    .expect("stacked checks reject every mutant");
}

// --------------------------------------------------------- prefetch ----

/// A prefetch launch configuration timed and checked.
struct PrefetchConfig {
    label: &'static str,
    symbol: &'static str,
    grid: u32,
    threads: u32,
    chunk: u32,
}

const PREFETCH_CONFIGS: [PrefetchConfig; 3] = [
    PrefetchConfig {
        label: "bulk256k_g16",
        symbol: shapes::PREFETCH_BULK,
        grid: 16,
        threads: 64,
        chunk: 256 << 10,
    },
    PrefetchConfig {
        label: "bulk64k_g64",
        symbol: shapes::PREFETCH_BULK,
        grid: 64,
        threads: 128,
        chunk: 64 << 10,
    },
    PrefetchConfig {
        label: "lines_g188",
        symbol: shapes::PREFETCH_LINES,
        grid: 188,
        threads: 256,
        chunk: 128,
    },
];

fn prefetch_entry(symbol: &str) -> LabResult<Prefetch> {
    shapes::prefetches()
        .into_iter()
        .find(|p| p.symbol == symbol)
        .ok_or_else(|| format!("no prefetch entry {symbol}"))
}

/// Launches a prefetch of `weights` (whole buffers, in order).
fn launch_prefetch(
    session: &Session,
    function: &Function,
    config: &PrefetchConfig,
    weights: &[&DeviceBuffer],
    stream: &Stream,
) -> LabResult<()> {
    let entry = prefetch_entry(config.symbol)?;
    let ranges: Vec<(u64, u64)> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| (WEIGHT + i as u64, w.byte_len() as u64))
        .collect();
    let args = entry.arguments(&ranges, config.chunk).map_err(err)?;
    let plan = entry.plan(config.grid, config.threads).map_err(err)?;
    let handles: Handles<'_> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| (WEIGHT + i as u64, Bound::Whole(w)))
        .collect();
    run(session, function, stream, plan, &args, &handles)
}

/// A prefetch between the PDL producer and the stacked GEMM keeps the
/// chain's order: the GEMM still reads its input only after the producer's
/// tail (eager and from a graph), and the weights are unchanged. The
/// prefetch without its trailing wait is a probe: whether the GEMM then
/// reads the held input is recorded.
#[test]
fn prefetch_keeps_chain_order() {
    gpu_test(4 << 30, |session| {
        let (stack, parts) = shapes::stacks().remove(0);
        let capacity = 4;
        let case = shapes::cases()
            .into_iter()
            .find(|c| c.stack == stack && c.capacity == capacity)
            .ok_or("no prefetch case")?;
        let spec = case
            .candidates
            .iter()
            .find(|s| s.strips == 2 && s.splits == 1 && !s.prefetch)
            .ok_or("no s2x1 candidate")?;
        let module = session.module(&shapes::cubin_name(stack))?;
        let parts_module = session.module(shapes::PARTS_CUBIN)?;
        let prefetch_module = session.module(shapes::PREFETCH_CUBIN)?;
        let producer = Producer::new(session)?;
        let references = References::new(&parts_module, &parts, capacity)?;
        let fixture = Fixture::new(session, &parts, capacity, &[spec], &references.specs())?;
        let gemm_function = stacked_function(&module, spec)?;
        let expected = fixture.expected(session, &references.ones(), capacity)?;
        let maps = fixture.maps(capacity)?;
        let weights: Vec<&DeviceBuffer> = fixture.weights.iter().collect();
        let stream = &session.stream;
        let mut results = Vec::new();
        let probe = PrefetchConfig {
            label: "bulk_no_wait",
            symbol: shapes::PREFETCH_NO_WAIT,
            ..PREFETCH_CONFIGS[0]
        };
        for config in PREFETCH_CONFIGS.iter().chain([&probe]) {
            let prefetch = prefetch_module.function(config.symbol).map_err(err)?;
            let chain = |stream: &Stream| -> LabResult<()> {
                launch_prefetch(session, &prefetch, config, &weights, stream)?;
                fixture.launch(session, spec, &gemm_function, 0, capacity, stream)
            };
            if config.symbol == shapes::PREFETCH_NO_WAIT {
                // Probe: count the phases whose output differs.
                let mut differing = 0;
                for _ in 0..3 {
                    let mut poisoned: Vec<&DeviceBuffer> = fixture.outputs[0].iter().collect();
                    poisoned.extend([&fixture.workspace, &fixture.fresh[0]]);
                    poison::poison(session, &poisoned)?;
                    producer.launch(stream, &fixture.operands[0])?;
                    chain(stream)?;
                    stream.synchronize().map_err(err)?;
                    for (slot, buffer) in fixture.outputs[0].iter().enumerate() {
                        if poison::read(session, buffer)? != expected[0][slot] {
                            differing += 1;
                        }
                    }
                }
                eprintln!(
                    "probe {}: the GEMM behind a prefetch without its wait differed in \
                     {differing} of 3 runs' slots",
                    config.label
                );
                results.push(json!({"prefetch": config.label, "symbol": config.symbol,
                    "probe": true, "differing_slots": differing}));
                continue;
            }
            let mut replay = Replay::new(session);
            for (slot, (buffer, map)) in fixture.outputs[0].iter().zip(&maps).enumerate() {
                replay = replay.output(format!("slot {slot}"), buffer, map.clone());
            }
            let mut replay = replay
                .scratch("chains", &fixture.workspace)
                .scratch("saved input", &fixture.fresh[0]);
            replay.eager(|| chain(stream))?;
            replay.pdl(
                || producer.launch(stream, &fixture.operands[0]),
                || chain(stream),
                true,
            )?;
            let graph = session.capture(|stream| {
                producer.launch(stream, &fixture.operands[0])?;
                chain(stream)
            })?;
            replay.graph(&graph)?;
            replay.graph(&graph)?;
            for (slot, want) in expected[0].iter().enumerate() {
                let got = replay
                    .expected(&format!("slot {slot}"))
                    .ok_or("missing eager output")?;
                if let Some(e) =
                    fixture.differs(&format!("behind {}", config.label), slot, want, got)
                {
                    return Err(e);
                }
            }
            for (part, (buffer, bytes)) in fixture
                .weights
                .iter()
                .zip(&fixture.weight_bytes)
                .enumerate()
            {
                if &poison::read(session, buffer)? != bytes {
                    return Err(format!("{} changed part {part}'s weights", config.label));
                }
            }
            let mut evidence = BTreeMap::new();
            replay.record(&mut evidence);
            eprintln!("{}: chain order and weights kept", config.label);
            results.push(json!({"prefetch": config.label, "symbol": config.symbol,
                "grid": config.grid, "threads": config.threads, "chunk": config.chunk,
                "checks": evidence}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("prefetch_chain_order"));
        receipt.insert("gemm".into(), json!(spec.symbol));
        receipt.insert("prefetches".into(), json!(results));
        receipt.insert("passed".into(), json!(true));
        session.receipt("strip_chains_stacked", receipt)?;
        Ok(())
    })
    .expect("prefetch keeps the chain order");
}

// ----------------------------------------------------------- timing ----

/// Layers of one timed graph: each over its own weight copies.
const LAYERS: usize = 8;
/// Capacities timed.
const TIMED_CAPACITIES: [u32; 4] = [1, 4, 16, 32];
/// (collective, mixer) idle windows in microseconds before each layer's
/// projections: none, a mixer alone, and a short and a long collective
/// before it (measured decode all-reduce and boundary mixer times).
const WINDOWS_US: [(u64, u64); 5] = [(0, 0), (0, 10), (0, 28), (15, 28), (36, 28)];

fn benchmarks_enabled(what: &str) -> bool {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1") {
        return true;
    }
    kernel_lab::skip(&format!("{what}: set CHOPINFER_KERNEL_BENCH=1")).unwrap();
    false
}

/// How a timed layer runs its projections.
#[derive(Clone)]
enum Projections<'a> {
    /// Today's entry per part, its contiguous output copied into place.
    Served,
    /// One stacked launch.
    Stacked(&'a Stacked),
}

/// The operands of a timed stack: `LAYERS` copies of every part's weights,
/// one activation, the slot buffers, each part's contiguous output for
/// today's entries, a live count, a workspace and tickets.
struct Timed<'a> {
    fixture: &'a Fixture,
    copies: Vec<Vec<DeviceBuffer>>,
    contiguous: Vec<DeviceBuffer>,
    live: DeviceBuffer,
}

/// Launches today's projections of `layer`: each part's served entry into
/// its contiguous output (directly into its slot where the stack's place is
/// contiguous), then its layout copy.
#[allow(clippy::too_many_arguments)]
fn launch_served(
    session: &Session,
    timed: &Timed<'_>,
    references: &References,
    copies: &[(Option<LayoutCopy>, Option<Function>)],
    layer: usize,
    m: u32,
    stream: &Stream,
) -> LabResult<()> {
    let fixture = timed.fixture;
    for (p, entries) in references.entries.iter().enumerate() {
        let (slot, _) = fixture.placed[p];
        let (copy, copy_function) = &copies[p];
        let output = match copy {
            Some(_) => &timed.contiguous[p],
            None => &fixture.outputs[0][slot],
        };
        launch_part(
            session,
            &entries.served,
            &references.served[p],
            &timed.copies[layer][p],
            &fixture.activations[0],
            Bound::Whole(output),
            &fixture.workspace,
            &fixture.semaphores,
            m,
            stream,
        )?;
        if let (Some(copy), Some(f)) = (copy, copy_function) {
            let args = copy.arguments(PART_OUTPUT, OUTPUT, LIVE).map_err(err)?;
            let plan = copy.stage().plan(m).map_err(err)?;
            let handles = Handles::from([
                (PART_OUTPUT, Bound::Whole(&timed.contiguous[p])),
                (OUTPUT, Bound::Whole(&fixture.outputs[0][slot])),
                (LIVE, Bound::Whole(&timed.live)),
            ]);
            run(session, f, stream, plan, &args, &handles)?;
        }
    }
    Ok(())
}

/// Launches one stacked layer over `layer`'s weight copies.
fn launch_stacked_layer(
    session: &Session,
    timed: &Timed<'_>,
    spec: &Stacked,
    function: &Function,
    layer: usize,
    m: u32,
    stream: &Stream,
) -> LabResult<()> {
    let fixture = timed.fixture;
    let mut handles = fixture.handles(0);
    for (p, weight) in timed.copies[layer].iter().enumerate() {
        handles.insert(WEIGHT + p as u64, Bound::Whole(weight));
    }
    let parts = fixture.parts.len() as u64;
    let weights: Vec<u64> = (0..parts).map(|p| WEIGHT + p).collect();
    let outputs: Vec<u64> = (0..parts).map(|p| OUTPUT + p).collect();
    let args = spec
        .arguments(m, &weights, ACTIVATION, &outputs, PARTIAL, SEMAPHORES)
        .map_err(err)?;
    run(
        session,
        function,
        stream,
        spec.plan(m).map_err(err)?,
        &args,
        &handles,
    )
}

fn launch_window(window: &Function, ns: u64, stream: &Stream) -> LabResult<()> {
    if ns == 0 {
        return Ok(());
    }
    launch::launch_with(
        window,
        stream,
        launch::simple_config(
            (1, 1, 1),
            (32, 1, 1),
            0,
            runtime_cuda::LaunchConfig::NO_CLUSTER,
            true,
        )?,
        &[KernelArgument::U64(ns)],
    )
}

/// A graph's time in microseconds: median and minimum of seven
/// rounds after two warmups, each after an L2 eviction.
fn graph_us(
    session: &Session,
    launch: impl FnOnce(&Stream) -> LabResult<()>,
) -> LabResult<(f64, f64)> {
    let graph = session.capture(launch)?;
    let timing = time(session, 2, 7, |stream| graph.launch(stream).map_err(err))?;
    Ok((
        f64::from(timing.median_ms) * 1000.0,
        f64::from(timing.min_ms) * 1000.0,
    ))
}

/// Times every variant of one stack at one capacity in every window and
/// returns summary lines.
fn time_case(session: &Session, case: &shapes::Case) -> LabResult<Vec<String>> {
    let capacity = case.capacity;
    let m = capacity;
    let module = session.module(&shapes::cubin_name(case.stack))?;
    let parts_module = session.module(shapes::PARTS_CUBIN)?;
    let copies_module = session.module(shapes::COPIES_CUBIN)?;
    let prefetch_module = session.module(shapes::PREFETCH_CUBIN)?;
    let window = session
        .module(shapes::WINDOW_CUBIN)?
        .function(shapes::IDLE_WINDOW)
        .map_err(err)?;
    let references = References::new(&parts_module, &case.parts, capacity)?;
    let stacked: Vec<&Stacked> = case.candidates.iter().collect();
    let fixture = Fixture::new(
        session,
        &case.parts,
        capacity,
        &stacked,
        &references.specs(),
    )?;
    let expected = fixture.expected(session, &references.ones(), m)?;
    let functions = case
        .candidates
        .iter()
        .map(|spec| stacked_function(&module, spec))
        .collect::<LabResult<Vec<_>>>()?;
    let copies = case
        .parts
        .iter()
        .zip(&fixture.placed)
        .map(|(part, &(_, offset))| {
            let copy = shapes::layout_copy(part, offset, capacity);
            let f = copy
                .as_ref()
                .map(|c| {
                    copies_module
                        .function(&c.stage().entry().symbol)
                        .map_err(err)
                })
                .transpose()?;
            Ok((copy, f))
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timed = Timed {
        fixture: &fixture,
        copies: (0..LAYERS)
            .map(|_| {
                fixture
                    .weight_bytes
                    .iter()
                    .map(|w| upload(session, w))
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?,
        contiguous: case
            .parts
            .iter()
            .map(|p| session.allocate(capacity as usize * p.rows as usize * 2))
            .collect::<LabResult<Vec<_>>>()?,
        live: upload(session, &m.to_le_bytes())?,
    };
    let weight_bytes: usize = fixture.weight_bytes.iter().map(Vec::len).sum();
    let mut variants: Vec<(
        String,
        Projections<'_>,
        Option<&PrefetchConfig>,
        Option<usize>,
    )> = vec![("served".into(), Projections::Served, None, None)];
    for (index, spec) in case.candidates.iter().enumerate() {
        variants.push((
            format!(
                "s{}x{}{}",
                spec.strips,
                spec.splits,
                if spec.prefetch { "_pf" } else { "" }
            ),
            Projections::Stacked(spec),
            None,
            Some(index),
        ));
    }
    for config in &PREFETCH_CONFIGS {
        variants.push((
            format!("{}+served", config.label),
            Projections::Served,
            Some(config),
            None,
        ));
        for (index, spec) in case.candidates.iter().enumerate() {
            if !spec.prefetch {
                variants.push((
                    format!("{}+s{}x{}", config.label, spec.strips, spec.splits),
                    Projections::Stacked(spec),
                    Some(config),
                    Some(index),
                ));
            }
        }
    }
    let mut summary = Vec::new();
    for (collective_us, mixer_us) in WINDOWS_US {
        let (collective, mixer) = (collective_us * 1000, mixer_us * 1000);
        // No windows: nothing to subtract (and no empty graph to time).
        let windows_us = if collective + mixer == 0 {
            0.0
        } else {
            graph_us(session, |stream| {
                for _ in 0..LAYERS {
                    launch_window(&window, collective, stream)?;
                    launch_window(&window, mixer, stream)?;
                }
                Ok(())
            })?
            .0
        };
        let mut times: Vec<(String, f64)> = Vec::new();
        for (label, projections, prefetch, index) in &variants {
            let prefetch_function = prefetch
                .map(|c| prefetch_module.function(c.symbol).map_err(err))
                .transpose()?;
            let layer = |layer: usize, stream: &Stream| -> LabResult<()> {
                launch_window(&window, collective, stream)?;
                if let (Some(config), Some(f)) = (prefetch, &prefetch_function) {
                    let weights: Vec<&DeviceBuffer> = timed.copies[layer].iter().collect();
                    launch_prefetch(session, f, config, &weights, stream)?;
                }
                launch_window(&window, mixer, stream)?;
                match projections {
                    Projections::Served => {
                        launch_served(session, &timed, &references, &copies, layer, m, stream)
                    }
                    Projections::Stacked(spec) => launch_stacked_layer(
                        session,
                        &timed,
                        spec,
                        &functions[index.expect("stacked index")],
                        layer,
                        m,
                        stream,
                    ),
                }
            };
            // One eager run of every layer, checked.
            let mut poisoned: Vec<&DeviceBuffer> = fixture.outputs[0].iter().collect();
            poisoned.push(&fixture.workspace);
            poison::poison(session, &poisoned)?;
            (0..LAYERS).try_for_each(|l| layer(l, &session.stream))?;
            session.stream.synchronize().map_err(err)?;
            for (slot, buffer) in fixture.outputs[0].iter().enumerate() {
                let got = poison::read(session, buffer)?;
                if let Some(e) = fixture.differs(label, slot, &expected[0][slot], &got) {
                    return Err(format!("{} {e}", case.tag));
                }
            }
            let (median_us, min_us) = graph_us(session, |stream| {
                (0..LAYERS).try_for_each(|l| layer(l, stream))
            })?;
            let exposed = (median_us - windows_us) / LAYERS as f64;
            let mut receipt = BTreeMap::new();
            receipt.insert("case".into(), json!("stacked_prefetch_latency"));
            receipt.insert("tag".into(), json!(case.tag));
            receipt.insert("variant".into(), json!(label));
            receipt.insert(
                "stacked".into(),
                json!(match projections {
                    Projections::Stacked(spec) => Some(&spec.symbol),
                    Projections::Served => None,
                }),
            );
            receipt.insert(
                "prefetch".into(),
                json!(
                    prefetch.map(|c| json!({"label": c.label, "symbol": c.symbol,
                    "grid": c.grid, "threads": c.threads, "chunk": c.chunk}))
                ),
            );
            receipt.insert("m".into(), json!(m));
            receipt.insert("layers".into(), json!(LAYERS));
            receipt.insert("weight_bytes_per_layer".into(), json!(weight_bytes));
            receipt.insert("collective_window_us".into(), json!(collective_us));
            receipt.insert("mixer_window_us".into(), json!(mixer_us));
            receipt.insert("windows_only_us".into(), json!(windows_us));
            receipt.insert("median_us".into(), json!(median_us));
            receipt.insert("min_us".into(), json!(min_us));
            receipt.insert("exposed_us_per_layer".into(), json!(exposed));
            receipt.insert("elements_differing".into(), json!(0));
            session.receipt("bench", receipt)?;
            eprintln!(
                "{} window {collective_us}+{mixer_us} us {label}: {exposed:.2} us exposed per \
                 layer ({median_us:.1} us graph, windows {windows_us:.1} us)",
                case.tag
            );
            times.push((label.clone(), exposed));
        }
        let served = times
            .iter()
            .find(|(l, _)| l == "served")
            .map(|(_, t)| *t)
            .ok_or("no served time")?;
        let (best, best_us) = times
            .iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .cloned()
            .ok_or("no time")?;
        let stacked_best = times
            .iter()
            .filter(|(l, _)| l.starts_with('s') && l != "served")
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .cloned()
            .ok_or("no stacked time")?;
        summary.push(format!(
            "{} window {collective_us}+{mixer_us}: served {served:.2} us, best {best} \
             {best_us:.2} us ({:.2}x), best without a prefetch launch {} {:.2} us",
            case.tag,
            served / best_us,
            stacked_best.0,
            stacked_best.1
        ));
    }
    Ok(summary)
}

#[test]
fn stacked_prefetch_latency() {
    if !benchmarks_enabled("stacked prefetch latency") {
        return;
    }
    gpu_test(16 << 30, |session| {
        let mut summary = Vec::new();
        for case in shapes::cases() {
            if !TIMED_CAPACITIES.contains(&case.capacity) || !filtered(&case.tag) {
                continue;
            }
            summary.extend(time_case(session, &case)?);
        }
        if summary.is_empty() {
            return Err("no stacked case was timed".into());
        }
        for line in &summary {
            eprintln!("summary {line}");
        }
        Ok(())
    })
    .expect("stacked prefetch latency");
}
