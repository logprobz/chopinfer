//! Each row stage's launch-shape entry against the compiled instance of its
//! shapes (`kernel_lab::row_shapes`): copies, lookups, activation quantizers
//! and gathers of every served use, every catalog probe and the lab shapes.
//!
//! Both arms read identical inputs and write their own outputs. Before every
//! eager launch, relaunch and graph replay, every output of both arms
//! (including the amax scratch and the dynamic global scale) is filled with
//! [`poison::POISON`] on the device, and afterwards the device counts the
//! bytes at which the two arms differ: whole outputs must be equal bit for
//! bit, at every live count from 0 through the capacity (a probe of a shape
//! nothing serves runs at `poison::live_counts`). At `poison::live_counts`,
//! each arm's write map is checked exactly: launched once over outputs filled
//! with POISON and once over outputs filled with its complement, a byte
//! inside the map must hold the same value both times and a byte outside it
//! its fill. Lookups run with the table in device and in mapped host memory,
//! with IDs below, inside and above the shard and beyond the vocabulary.
//!
//! The prepared selection (`examples/prepare_projections.rs`) holds every
//! entry; `CHOPINFER_PROJECTION_REPORT` names its report.
//! `CHOPINFER_ROW_SHAPE_SHARD=i/n` runs the pairs whose index is i modulo n.

use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, WeightFormat};
use kernel_lab::launch::{Bindings, launch};
use kernel_lab::projection_cases::Prepared;
use kernel_lab::row_shapes::{self, Origin, Pair, Stage, buffer};
use kernel_lab::{LabResult, Session, gpu_test, poison, time};
use kernels_sm120::row::precision::{LookupIds, Nvfp4Global, ShapeBinding};
use runtime_cuda::{DeviceBuffer, Function, GraphExec};
use serde_json::json;
use std::collections::BTreeMap;
use std::ops::Range;

/// Bytes past each output's extent that no launch may write.
const SLACK: usize = 256;
/// The helpers' own buffer handles.
const TARGET: u64 = 100;
const OTHER: u64 = 101;
const COUNT: u64 = 102;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn mix(x: u64) -> u64 {
    let mut x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// BF16 activations over about 60 binades, signed, with subnormals, an
/// all-zero row in seven and zero groups of 16 in some others.
fn activation(index: usize, width: usize) -> u16 {
    let (row, column) = (index / width.max(1), index % width.max(1));
    if row % 7 == 6 || (row % 3 == 1 && column / 16 % 5 == 2) {
        return 0;
    }
    let h = mix(index as u64);
    let exponent = if h % 23 == 0 { 0 } else { 80 + (h >> 8) % 60 };
    (((h >> 20) & 1) << 15) as u16 | (exponent << 7) as u16 | ((h >> 32) & 0x7f) as u16
}

fn bytes(seed: u64, len: usize) -> Vec<u8> {
    (0..len).map(|i| mix(seed ^ i as u64) as u8).collect()
}

fn le<T: Copy, const N: usize>(values: &[T], to: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|v| to(*v)).collect()
}

/// One output buffer of an arm and its write map at a live count.
struct Output {
    handle: u64,
    bytes: usize,
    written: Box<dyn Fn(u32) -> Vec<Range<usize>>>,
}

/// The outputs a stage writes, sized for its capacity plus [`SLACK`].
fn outputs(stage: &Stage) -> Vec<Output> {
    let prefix = |row_bytes: usize| -> Box<dyn Fn(u32) -> Vec<Range<usize>>> {
        Box::new(move |live| std::iter::once(0..live as usize * row_bytes).collect())
    };
    let capacity = stage.capacity() as usize;
    match stage {
        Stage::Copy(copy) => {
            let (columns, _, d) = copy.layout();
            vec![Output {
                handle: buffer::OUTPUT,
                bytes: copy.destination_bytes() as usize + SLACK,
                written: Box::new(move |live| {
                    let (row, column, offset) = (
                        d.row_stride as usize,
                        d.column_stride as usize,
                        d.column_offset as usize,
                    );
                    if column == 1 {
                        // Each row's columns are contiguous.
                        return (0..live as usize)
                            .map(|r| {
                                2 * (r * row + offset)..2 * (r * row + offset + columns as usize)
                            })
                            .collect();
                    }
                    (0..live as usize)
                        .flat_map(|r| {
                            (0..columns as usize).map(move |c| {
                                let at = 2 * (r * row + c * column + offset);
                                at..at + 2
                            })
                        })
                        .collect()
                }),
            }]
        }
        Stage::Lookup(lookup) => {
            let r = lookup.request();
            vec![Output {
                handle: buffer::OUTPUT,
                bytes: lookup.output_bytes() as usize + SLACK,
                written: prefix(r.heads as usize * r.width as usize * 2),
            }]
        }
        Stage::Quantizer(q) => {
            let mut outputs = vec![
                Output {
                    handle: buffer::CODES,
                    bytes: q.codes_bytes() as usize + SLACK,
                    written: prefix(q.codes_bytes() as usize / capacity),
                },
                Output {
                    handle: buffer::SCALES,
                    bytes: q.scales_bytes() as usize + SLACK,
                    written: prefix(q.scales_bytes() as usize / capacity),
                },
            ];
            if q.global() == Nvfp4Global::TensorAmax {
                outputs.push(Output {
                    handle: buffer::AMAX,
                    bytes: q.scratch_bytes() as usize + SLACK,
                    written: prefix(4),
                });
                // The global reduction always writes its scale.
                outputs.push(Output {
                    handle: buffer::GLOBAL,
                    bytes: 4 + SLACK,
                    written: Box::new(|_| std::iter::once(0..4).collect()),
                });
            }
            outputs
        }
        Stage::Gather(g) => {
            let width = g.stage().width() as usize;
            if g.quantized() {
                vec![
                    Output {
                        handle: buffer::CODES,
                        bytes: capacity * width + SLACK,
                        written: prefix(width),
                    },
                    Output {
                        handle: buffer::SCALES,
                        bytes: capacity * width / 32 + SLACK,
                        written: prefix(width / 32),
                    },
                ]
            } else {
                vec![Output {
                    handle: buffer::OUTPUT,
                    bytes: capacity * width * 2 + SLACK,
                    written: prefix(width * 2),
                }]
            }
        }
    }
}

/// The inputs both arms read, and the count buffer a sweep stores into.
struct Inputs {
    buffers: BTreeMap<u64, DeviceBuffer>,
    count: u64,
    row_offset: u64,
    shard_rows: u64,
}

fn upload(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = s.allocate(bytes.len().max(4))?;
    if !bytes.is_empty() {
        buffer.copy_from_bytes(bytes, &s.stream).map_err(err)?;
    }
    Ok(buffer)
}

fn inputs(s: &Session, stage: &Stage, mapped: bool, seed: u64) -> LabResult<Inputs> {
    let mut buffers = BTreeMap::new();
    let bf16 = |len: usize, width: usize| {
        le(
            &(0..len).map(|i| activation(i, width)).collect::<Vec<_>>(),
            u16::to_le_bytes,
        )
    };
    let (mut count, mut row_offset, mut shard_rows) = (buffer::LIVE, 0, 1);
    match stage {
        Stage::Copy(copy) => {
            let (columns, _, _) = copy.layout();
            buffers.insert(
                buffer::INPUT,
                upload(s, &bf16(copy.source_bytes() as usize / 2, columns as usize))?,
            );
        }
        Stage::Lookup(lookup) => {
            let r = lookup.request();
            let width = r.width as usize;
            shard_rows = (r.vocab.saturating_sub(2)).clamp(1, 61);
            row_offset = r.vocab - shard_rows - u64::from(r.vocab > shard_rows);
            let row_bytes = match r.weight {
                WeightFormat::Bf16 => width * 2,
                WeightFormat::Nvfp4 => width / 2,
                _ => width,
            };
            let table = bytes(seed, shard_rows as usize * row_bytes);
            let table = if mapped {
                let mut buffer =
                    DeviceBuffer::allocate_mapped_host(&s.stream, table.len()).map_err(err)?;
                buffer
                    .mapped_host_bytes_mut()
                    .map_err(err)?
                    .copy_from_slice(&table);
                buffer
            } else {
                upload(s, &table)?
            };
            buffers.insert(buffer::TABLE, table);
            if r.weight == WeightFormat::Nvfp4 {
                buffers.insert(
                    buffer::LOCAL_SCALES,
                    upload(s, &bytes(seed ^ 1, shard_rows as usize * width / 16))?,
                );
            }
            match r.global_scale {
                Some(kernel_core::contract::DType::Bf16) => {
                    buffers.insert(buffer::GLOBAL, upload(s, &0x3e80_u16.to_le_bytes())?);
                }
                Some(_) => {
                    buffers.insert(buffer::GLOBAL, upload(s, &0.0371_f32.to_le_bytes())?);
                }
                None => {}
            }
            let ids = (0..r.capacity as u64 * u64::from(r.heads))
                .map(|i| match i % 9 {
                    0 => row_offset,
                    1 => row_offset + shard_rows - 1,
                    2 => row_offset + shard_rows,
                    3 => row_offset.wrapping_sub(1),
                    4 => r.vocab,
                    5 => u64::MAX,
                    _ => row_offset + mix(seed ^ i) % shard_rows,
                })
                .collect::<Vec<_>>();
            let ids = match r.ids {
                LookupIds::U64 => le(&ids, u64::to_le_bytes),
                LookupIds::I32 => le(
                    &ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
                    i32::to_le_bytes,
                ),
            };
            buffers.insert(buffer::IDS, upload(s, &ids)?);
        }
        Stage::Quantizer(q) => {
            let capacity = q.stages()[0].capacity() as usize;
            let width = q.stages()[0].width() as usize;
            buffers.insert(buffer::INPUT, upload(s, &bf16(capacity * width, width))?);
            if q.global() == Nvfp4Global::Supplied {
                buffers.insert(
                    buffer::GLOBAL,
                    upload(s, &(0.25_f32 / 2688.0).to_le_bytes())?,
                );
            }
        }
        Stage::Gather(g) => {
            let width = g.stage().width() as usize;
            let rows = g.input_rows() as usize;
            buffers.insert(buffer::INPUT, upload(s, &bf16(rows * width, width))?);
            let sorted = (0..g.stage().capacity() as u64)
                .map(|j| {
                    if j % 5 == 4 {
                        -1
                    } else {
                        (mix(seed ^ j) % u64::from(g.routes())) as i32
                    }
                })
                .collect::<Vec<_>>();
            buffers.insert(buffer::SORTED, upload(s, &le(&sorted, i32::to_le_bytes))?);
            count = buffer::PADDED;
        }
    }
    buffers.insert(count, upload(s, &0_u32.to_le_bytes())?);
    Ok(Inputs {
        buffers,
        count,
        row_offset,
        shard_rows,
    })
}

struct Helpers {
    fill: Function,
    store: Function,
    count: Function,
}

impl Helpers {
    fn new(s: &Session, prepared: &Prepared) -> LabResult<Self> {
        Ok(Self {
            fill: prepared.function(s, row_shapes::FILL, 0)?,
            store: prepared.function(s, row_shapes::STORE, 0)?,
            count: prepared.function(s, row_shapes::COUNT, 0)?,
        })
    }
    fn fill(&self, s: &Session, target: &DeviceBuffer, value: u8) -> LabResult<()> {
        let bytes = target.byte_len() as u64;
        launch(
            &s.context,
            &self.fill,
            &s.stream,
            row_shapes::helper_plan(bytes),
            &ArgumentList(vec![
                Argument::DeviceBuffer(TARGET),
                Argument::U64(bytes),
                Argument::U32(value.into()),
            ]),
            &Bindings {
                buffers: BTreeMap::from([(TARGET, target)]),
            },
        )
    }
    fn store(&self, s: &Session, target: &DeviceBuffer, value: u32) -> LabResult<()> {
        launch(
            &s.context,
            &self.store,
            &s.stream,
            row_shapes::helper_plan(1),
            &ArgumentList(vec![Argument::DeviceBuffer(TARGET), Argument::U32(value)]),
            &Bindings {
                buffers: BTreeMap::from([(TARGET, target)]),
            },
        )
    }
    fn count(
        &self,
        s: &Session,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        counter: &DeviceBuffer,
    ) -> LabResult<()> {
        let bytes = a.byte_len() as u64;
        launch(
            &s.context,
            &self.count,
            &s.stream,
            row_shapes::helper_plan(bytes),
            &ArgumentList(vec![
                Argument::DeviceBuffer(TARGET),
                Argument::DeviceBuffer(OTHER),
                Argument::U64(bytes),
                Argument::DeviceBuffer(COUNT),
            ]),
            &Bindings {
                buffers: BTreeMap::from([(TARGET, a), (OTHER, b), (COUNT, counter)]),
            },
        )
    }
}

/// One arm: its stages' functions and launches, and its own outputs.
struct Arm {
    launches: Vec<(Function, LaunchPlan, ArgumentList)>,
    outputs: Vec<(Output, DeviceBuffer)>,
    symbols: Vec<String>,
}

impl Arm {
    fn new(s: &Session, prepared: &Prepared, stage: &Stage, inputs: &Inputs) -> LabResult<Self> {
        let launches = stage
            .launches(inputs.row_offset, inputs.shard_rows)?
            .into_iter()
            .map(|(stage, arguments)| {
                let plan = stage.plan(stage.capacity()).map_err(err)?;
                Ok((
                    prepared.function(s, &stage.entry().symbol, plan.dynamic_shared_bytes)?,
                    plan,
                    arguments,
                ))
            })
            .collect::<LabResult<Vec<_>>>()?;
        let outputs = outputs(stage)
            .into_iter()
            .map(|output| {
                let buffer = s.allocate(output.bytes)?;
                Ok((output, buffer))
            })
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            launches,
            outputs,
            symbols: stage
                .stages()
                .iter()
                .map(|s| s.entry().symbol.clone())
                .collect(),
        })
    }
    fn bindings<'a>(&'a self, inputs: &'a Inputs) -> Bindings<'a> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> =
            inputs.buffers.iter().map(|(h, b)| (*h, b)).collect();
        buffers.extend(self.outputs.iter().map(|(o, b)| (o.handle, b)));
        Bindings { buffers }
    }
    fn launch(&self, s: &Session, inputs: &Inputs, stream: &runtime_cuda::Stream) -> LabResult<()> {
        let bindings = self.bindings(inputs);
        for (function, plan, arguments) in &self.launches {
            launch(&s.context, function, stream, *plan, arguments, &bindings)?;
        }
        Ok(())
    }
    fn graph(&self, s: &Session, inputs: &Inputs) -> LabResult<GraphExec> {
        s.capture(|stream| self.launch(s, inputs, stream))
    }
    fn fill(&self, s: &Session, helpers: &Helpers, value: u8) -> LabResult<()> {
        for (_, buffer) in &self.outputs {
            helpers.fill(s, buffer, value)?;
        }
        Ok(())
    }
    /// Checks this arm's write map at `live` over two complementary fills.
    fn check_writes(
        &self,
        s: &Session,
        helpers: &Helpers,
        inputs: &Inputs,
        live: u32,
    ) -> LabResult<usize> {
        let mut images = Vec::new();
        for fill in [poison::POISON, !poison::POISON] {
            self.fill(s, helpers, fill)?;
            helpers.store(s, &inputs.buffers[&inputs.count], live)?;
            self.launch(s, inputs, &s.stream)?;
            images.push(
                self.outputs
                    .iter()
                    .map(|(_, buffer)| poison::read(s, buffer))
                    .collect::<LabResult<Vec<_>>>()?,
            );
        }
        let mut written = 0;
        for (index, (output, _)) in self.outputs.iter().enumerate() {
            let (a, b) = (&images[0][index], &images[1][index]);
            let mut map = vec![false; a.len()];
            for range in (output.written)(live) {
                if range.end > a.len() {
                    return Err(format!("write range {range:?} beyond {} bytes", a.len()));
                }
                map[range].fill(true);
            }
            for (at, inside) in map.iter().enumerate() {
                let ok = if *inside {
                    a[at] == b[at]
                } else {
                    a[at] == poison::POISON && b[at] == !poison::POISON
                };
                if !ok {
                    return Err(format!(
                        "{} output {} live {live}: byte {at} {} its write map ({:#04x}, {:#04x})",
                        self.symbols.last().unwrap(),
                        output.handle,
                        if *inside { "inside" } else { "outside" },
                        a[at],
                        b[at]
                    ));
                }
            }
            written += map.iter().filter(|w| **w).count();
        }
        Ok(written)
    }
}

fn selected(pairs: Vec<Pair>) -> LabResult<Vec<(usize, Pair)>> {
    let (index, count) = match std::env::var("CHOPINFER_ROW_SHAPE_SHARD") {
        Ok(value) => {
            let (i, n) = value.split_once('/').ok_or("shard is i/n")?;
            let (i, n) = (
                i.parse::<usize>().map_err(err)?,
                n.parse::<usize>().map_err(err)?,
            );
            if n == 0 || i >= n {
                return Err("shard i/n needs i < n".into());
            }
            (i, n)
        }
        Err(_) => (0, 1),
    };
    Ok(pairs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % count == index)
        .collect())
}

/// The table placements a pair runs with.
fn placements(pair: &Pair) -> Vec<bool> {
    if matches!(pair.reference, Stage::Lookup(_)) {
        vec![false, true]
    } else {
        vec![false]
    }
}

#[test]
fn launch_shape_row_stages_equal_their_compiled_instances() {
    gpu_test(4 << 30, |s| {
        let prepared = Prepared::read()?;
        let helpers = Helpers::new(s, &prepared)?;
        let pairs = selected(row_shapes::pairs()?)?;
        if pairs.is_empty() {
            return Err("no row-shape pair selected".into());
        }
        let mut failures = Vec::new();
        for (index, pair) in &pairs {
            for mapped in placements(pair) {
                let started = std::time::Instant::now();
                let inputs = inputs(s, &pair.reference, mapped, *index as u64)?;
                let reference = Arm::new(s, &prepared, &pair.reference, &inputs)?;
                let candidate = Arm::new(s, &prepared, &pair.candidate, &inputs)?;
                let capacity = pair.candidate.capacity();
                let lives = if pair.origin == Origin::Probe {
                    poison::live_counts(capacity)
                } else {
                    (0..=capacity).collect()
                };
                let graphs = [reference.graph(s, &inputs)?, candidate.graph(s, &inputs)?];
                let counter = upload(s, &[0; 8])?;
                let mut differing = Vec::new();
                for &live in &lives {
                    for phase in ["eager", "relaunch", "graph"] {
                        reference.fill(s, &helpers, poison::POISON)?;
                        candidate.fill(s, &helpers, poison::POISON)?;
                        helpers.store(s, &inputs.buffers[&inputs.count], live)?;
                        if phase == "graph" {
                            for graph in &graphs {
                                graph.launch(&s.stream).map_err(err)?;
                            }
                        } else {
                            reference.launch(s, &inputs, &s.stream)?;
                            candidate.launch(s, &inputs, &s.stream)?;
                        }
                        for ((_, a), (_, b)) in reference.outputs.iter().zip(&candidate.outputs) {
                            helpers.count(s, a, b, &counter)?;
                        }
                    }
                    let count = u64::from_le_bytes(
                        poison::read(s, &counter)?[..8].try_into().map_err(err)?,
                    );
                    if count != 0 {
                        differing.push(json!({"live": live, "differing_bytes": count}));
                        poison::write(s, &counter, 0, &[0; 8])?;
                    }
                }
                let mut written = BTreeMap::new();
                for live in poison::live_counts(capacity) {
                    for (name, arm) in [("reference", &reference), ("candidate", &candidate)] {
                        match arm.check_writes(s, &helpers, &inputs, live) {
                            Ok(bytes) => {
                                written.insert(format!("{name} live {live}"), bytes);
                            }
                            Err(e) => failures.push(format!("{}: {e}", pair.label)),
                        }
                    }
                }
                let equal = differing.is_empty();
                if !equal {
                    failures.push(format!(
                        "{} capacity {capacity} mapped {mapped}: arms differ at {} live counts, first {}",
                        pair.label,
                        differing.len(),
                        differing[0]
                    ));
                }
                eprintln!(
                    "{index:>4} {:<64} capacity {capacity:>5} mapped {mapped:<5} live counts {:>5} equal {equal} {:.1} s",
                    pair.label,
                    lives.len(),
                    started.elapsed().as_secs_f32()
                );
                s.receipt(
                    "projection-row-shape",
                    BTreeMap::from([
                        ("pair_index".into(), json!(index)),
                        ("label".into(), json!(pair.label)),
                        ("origin".into(), json!(pair.origin)),
                        ("capacity".into(), json!(capacity)),
                        ("mapped_host_table".into(), json!(mapped)),
                        ("reference".into(), json!(reference.symbols)),
                        ("candidate".into(), json!(candidate.symbols)),
                        ("live_counts".into(), json!(lives.len())),
                        ("every_live_count".into(), json!(pair.origin != Origin::Probe)),
                        ("phases".into(), json!(["eager", "relaunch", "graph"])),
                        ("poisoned_outputs".into(), json!(reference.outputs.len())),
                        ("write_map_bytes".into(), json!(written)),
                        ("differing".into(), json!(differing)),
                        ("bitwise".into(), json!(equal)),
                        ("seconds".into(), json!(started.elapsed().as_secs_f64())),
                    ]),
                )?;
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("launch-shape row stages equal their compiled instances");
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// Alternates the compiled instance (A) and the launch-shape entry (B),
/// A B A B per repetition, at each served or lab use's capacity, as
/// `dense_template_timing` does: each timing is the median over `ROUNDS` of a
/// batch of back-to-back launches after one L2 flush, divided by the batch;
/// the batch holds at least 32 launches and at least `MIN_BATCH_US` of work,
/// at most `MAX_BATCH`; an unrecorded repetition warms the clocks. A use
/// regresses when the median of its pair ratios B/A exceeds `1 + REGRESSION`;
/// above `NOISE_LIMIT` A-to-A noise it is inconclusive and asks for a rerun.
#[test]
#[ignore = "timing; run on an exclusive GPU with --ignored"]
fn launch_shape_row_stage_timing() {
    const REPETITIONS: usize = 7;
    const ROUNDS: usize = 15;
    const MIN_BATCH: usize = 32;
    const MAX_BATCH: usize = 1024;
    const MIN_BATCH_US: f32 = 400.0;
    const REGRESSION: f32 = 0.01;
    const NOISE_LIMIT: f32 = 0.02;
    gpu_test(4 << 30, |s| {
        let prepared = Prepared::read()?;
        let helpers = Helpers::new(s, &prepared)?;
        let pairs = selected(row_shapes::pairs()?)?;
        let (mut rows, mut slower, mut inconclusive) = (Vec::new(), Vec::new(), Vec::new());
        for (index, pair) in pairs.iter().filter(|(_, p)| p.origin != Origin::Probe) {
            let inputs = inputs(s, &pair.reference, false, *index as u64)?;
            let arms = [
                Arm::new(s, &prepared, &pair.reference, &inputs)?,
                Arm::new(s, &prepared, &pair.candidate, &inputs)?,
            ];
            let live = pair.candidate.capacity();
            helpers.store(s, &inputs.buffers[&inputs.count], live)?;
            let timed = |arm: usize, batch: usize| -> LabResult<f32> {
                let timing = time(s, 3, ROUNDS, |stream| {
                    for _ in 0..batch {
                        arms[arm].launch(s, &inputs, stream)?;
                    }
                    Ok(())
                })?;
                Ok(timing.median_ms * 1000.0 / batch as f32)
            };
            let batch =
                ((MIN_BATCH_US / timed(0, MIN_BATCH)?).ceil() as usize).clamp(MIN_BATCH, MAX_BATCH);
            for arm in [0, 1, 0, 1] {
                timed(arm, batch)?;
            }
            let (mut a, mut b, mut noise, mut ratios) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for _ in 0..REPETITIONS {
                let a1 = timed(0, batch)?;
                let b1 = timed(1, batch)?;
                let a2 = timed(0, batch)?;
                let b2 = timed(1, batch)?;
                noise.push((a1 - a2).abs() / a1.min(a2));
                ratios.extend([b1 / a1, b2 / a2]);
                a.extend([a1, a2]);
                b.extend([b1, b2]);
            }
            let range = |v: &[f32]| {
                (
                    v.iter().copied().fold(f32::MAX, f32::min),
                    v.iter().copied().fold(0.0, f32::max),
                )
            };
            let ((a_min, a_max), (b_min, b_max)) = (range(&a), range(&b));
            let (ratio_min, ratio_max) = range(&ratios);
            let band = range(&noise).1;
            let (a_median, b_median) = (median(&mut a), median(&mut b));
            let band_median = median(&mut noise);
            let ratio = median(&mut ratios);
            let label = format!("{} capacity {live}", pair.label);
            rows.push(json!({
                "pair_index": index,
                "label": pair.label,
                "origin": pair.origin,
                "reference": arms[0].symbols,
                "candidate": arms[1].symbols,
                "live": live,
                "launches_per_timing": batch,
                "reference_us": {"median": a_median, "min": a_min, "max": a_max},
                "candidate_us": {"median": b_median, "min": b_min, "max": b_max},
                "candidate_over_reference": ratio,
                "pair_ratio": {"median": ratio, "min": ratio_min, "max": ratio_max},
                "a_to_a_noise": {"median": band_median, "max": band},
            }));
            eprintln!(
                "{label:<80} A {a_median:8.2} us  B {b_median:8.2} us  B/A {ratio:.4}  A-A noise median {:.2}% max {:.2}%",
                100.0 * band_median,
                100.0 * band,
            );
            if band_median > NOISE_LIMIT {
                inconclusive.push(format!("{label} A-A noise median {:.2}%", 100.0 * band_median));
            } else if ratio > 1.0 + REGRESSION {
                slower.push(format!("{label} median pair B/A {ratio:.4}"));
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("repetitions".into(), json!(REPETITIONS));
        receipt.insert("rounds_per_timing".into(), json!(ROUNDS));
        receipt.insert("order".into(), json!("one unrecorded A B A B, then A B A B per repetition; L2 flushed before each timed batch; microseconds per launch"));
        receipt.insert(
            "batch".into(),
            json!({"min_launches": MIN_BATCH, "max_launches": MAX_BATCH, "min_batch_us": MIN_BATCH_US}),
        );
        receipt.insert("uses".into(), json!(rows));
        receipt.insert("executed_uses".into(), json!(rows.len()));
        receipt.insert(
            "gate".into(),
            json!({"median_pair_ratio_max": 1.0 + REGRESSION, "a_to_a_noise_median_max": NOISE_LIMIT}),
        );
        receipt.insert("slower".into(), json!(slower));
        receipt.insert("inconclusive_rerun".into(), json!(inconclusive));
        let path = s.receipt("projection_row_shape_timing", receipt)?;
        eprintln!("receipt {}", path.display());
        if rows.is_empty() {
            Err("no timing use ran".into())
        } else if !slower.is_empty() {
            Err(format!("launch-shape entry slower:\n{}", slower.join("\n")))
        } else if !inconclusive.is_empty() {
            Err(format!("inconclusive, rerun:\n{}", inconclusive.join("\n")))
        } else {
            Ok(())
        }
    })
    .expect("launch-shape row stage timing");
}

/// The lab shapes cover every row stage kind, both gather inputs (so both
/// divisors) and both gather encodings; every pair compares a compiled
/// instance with a launch-shape entry at one capacity.
#[test]
fn lab_shapes_cover_every_row_stage_kind_and_both_gather_inputs() {
    let pairs = row_shapes::pairs().unwrap();
    let lab = |kind: fn(&Stage) -> bool| {
        pairs
            .iter()
            .filter(|p| p.origin == Origin::Lab && kind(&p.candidate))
            .count()
    };
    assert!(lab(|s| matches!(s, Stage::Quantizer(_))) >= 7);
    assert!(lab(|s| matches!(s, Stage::Lookup(_))) >= 5);
    assert!(lab(|s| matches!(s, Stage::Copy(_))) >= 1);
    assert_eq!(lab(|s| matches!(s, Stage::Gather(_))), 6);
    for pair in &pairs {
        assert!(
            pair.reference
                .stages()
                .iter()
                .all(|s| s.shapes() == ShapeBinding::Compiled)
        );
        assert!(
            pair.candidate
                .stages()
                .iter()
                .all(|s| s.shapes() == ShapeBinding::Launch)
        );
        assert_eq!(pair.reference.capacity(), pair.candidate.capacity());
    }
}
