//! Exact old-cubin comparison, including state, with a delayed PDL producer.
//! The baseline hashes are the unmodified entries of 2026-09-26, CUDA 13.2.
use super::*;
use kernel_lab::poison::POISON;
use runtime_cuda::{GraphExec, KernelArgument, LaunchConfig};
use std::path::PathBuf;

const LAYERS: usize = 16;
const EVICTION_BYTES: usize = 256 << 20;
const TAIL_NS: u64 = 15_000;

fn baseline(s: &Session) -> LabResult<(Module, String)> {
    let ranks = geometry().ranks;
    let (name, hash) = match ranks {
        4 => (
            "linear_qwen38_flash_sm120",
            "d5d56229eec70e62a2ee8cd6cccb8c845d032eb8e9c5c4ab09879f1777e0a782",
        ),
        2 => (
            "linear_qwen38_tp2_sm120",
            "b2101bad3e885077fea33e80e8cb2dcf4984c6cdaa519b72ab032fb02f105ef4",
        ),
        _ => return Err("unsupported baseline topology".into()),
    };
    let path = PathBuf::from(std::env::var("CHOP_PDL_BASELINE_DIR").map_err(|e| e.to_string())?)
        .join(format!("{name}.cubin"));
    let digest = std::process::Command::new("sha256sum")
        .arg(&path)
        .output()
        .map_err(|e| e.to_string())?;
    if !digest.status.success()
        || String::from_utf8_lossy(&digest.stdout)
            .split_whitespace()
            .next()
            != Some(hash)
    {
        return Err(format!("baseline identity mismatch: {}", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    Ok((
        s.context.load_module(&bytes).map_err(|e| e.to_string())?,
        hash.into(),
    ))
}

struct Fixture {
    bytes: BTreeMap<&'static str, Vec<u8>>,
    scalars: BTreeMap<&'static str, u32>,
    input: &'static str,
    live: &'static str,
    capacity: usize,
    row_capacity: usize,
    rows: usize,
    sequences: usize,
    /// Output bytes of the live rows; the padding rows keep their poison.
    live_output_bytes: usize,
    /// Checkpoint entries of the live rows, which verify writes without
    /// reading and which start poisoned, and the state's element size.
    checkpoints: Vec<std::ops::Range<usize>>,
    state_element: usize,
}

impl Fixture {
    fn new(
        kind: &str,
        sequences: usize,
        capacity: usize,
        seed: usize,
        rows_per_sequence: usize,
    ) -> Self {
        let g = geometry();
        let verify = kind.ends_with("_verify");
        let norm = kind == "gated_norm";
        let lengths: Vec<_> = (0..capacity)
            .map(|i| {
                if rows_per_sequence == 0 {
                    1 + i % verify_row_limit() as usize
                } else {
                    rows_per_sequence
                }
            })
            .collect();
        let cu_rows = cu(&lengths);
        let rows = if verify {
            cu_rows[sequences] as usize
        } else {
            sequences
        };
        let row_capacity = if verify {
            cu_rows[capacity] as usize
        } else {
            capacity
        };
        let (slot_ids, base_slots) = slots(capacity, 2);
        // Checkpoints used by each sequence are distinct. Unused row-table
        // entries name spare slot zero, which must remain unchanged.
        let (checkpoint_ids, slots) = if verify {
            let ids = (0..capacity * verify_row_limit() as usize)
                .map(|i| {
                    let (seq, row) = (
                        i / verify_row_limit() as usize,
                        i % verify_row_limit() as usize,
                    );
                    if row < lengths[seq] {
                        (base_slots + cu_rows[seq] as usize + row) as i32
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>();
            (ids, base_slots + row_capacity)
        } else {
            (vec![], base_slots)
        };
        let mut f = Self {
            bytes: BTreeMap::new(),
            scalars: BTreeMap::new(),
            input: if kind.starts_with("delta") {
                "qkv"
            } else {
                "input"
            },
            live: if norm { "live_rows" } else { "live_sequences" },
            capacity,
            row_capacity,
            rows,
            sequences,
            live_output_bytes: 0,
            checkpoints: Vec::new(),
            state_element: 2,
        };
        // Cheap deterministic finite data, including signed zero and sigmoid
        // tails. Distinct layer seeds and allocations keep constants separate.
        let floats = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 71 + seed * 29) % 257) as f32 - 128.0) * scale)
                .collect()
        };
        let bf = |n: usize, scale: f32| {
            bytes_u16(
                &floats(n, scale)
                    .into_iter()
                    .map(f32_to_bf16)
                    .collect::<Vec<_>>(),
            )
        };
        let gates = |n: usize| {
            bytes_u16(
                &(0..n)
                    .map(|i| {
                        f32_to_bf16(
                            [-120.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 120.0][(i + seed) % 8],
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        };
        f.bytes
            .insert(f.live, (sequences as u32).to_ne_bytes().to_vec());
        if !norm {
            f.bytes.insert("slot_ids", bytes_i32(&slot_ids));
        }
        match kind {
            "conv_decode" | "conv_verify" => {
                f.bytes
                    .insert("input", bf(row_capacity * g.qkv_stride, 0.01));
                f.bytes.insert("weight", bf(g.channels * 4, 0.001));
                f.bytes
                    .insert("state", bf(slots * HISTORY * g.channels, 0.01));
                f.bytes
                    .insert("output", vec![POISON; row_capacity * g.channels * 2]);
                f.live_output_bytes = rows * g.channels * 2;
                f.scalars.insert("input_stride", g.qkv_stride as u32);
            }
            "delta_decode" | "delta_verify" => {
                f.bytes.insert("qkv", bf(row_capacity * g.qkv_stride, 0.01));
                f.bytes.insert("a", gates(row_capacity * g.gate_stride));
                f.bytes.insert("b", gates(row_capacity * g.gate_stride));
                f.bytes.insert(
                    "a_log",
                    bytes_f32(
                        &(0..g.value_heads)
                            .map(|h| -3.5 + h as f32 * 0.4)
                            .collect::<Vec<_>>(),
                    ),
                );
                f.bytes
                    .insert("dt_bias", bytes_f32(&floats(g.value_heads, 0.025)));
                f.bytes.insert("state", {
                    let n = slots * g.delta_state;
                    let pattern = bytes_f32(&floats(257, 0.001));
                    let mut bytes = pattern.repeat(n.div_ceil(257));
                    bytes.truncate(n * 4);
                    bytes
                });
                f.bytes
                    .insert("output", vec![POISON; row_capacity * g.norm_width * 2]);
                f.live_output_bytes = rows * g.norm_width * 2;
                f.state_element = 4;
                f.scalars.insert("qkv_stride", g.qkv_stride as u32);
                f.scalars.insert("gate_stride", g.gate_stride as u32);
            }
            "gated_norm" => {
                f.bytes
                    .insert("input", bf(row_capacity * g.norm_width, 0.01));
                f.bytes.insert("gate", gates(row_capacity * g.z_stride));
                f.bytes.insert("weight", bf(DIM as usize, 0.01));
                f.bytes
                    .insert("output", vec![POISON; row_capacity * g.norm_width * 2]);
                f.live_output_bytes = rows * g.norm_width * 2;
                f.scalars.insert("gate_stride", g.z_stride as u32);
            }
            _ => unreachable!(),
        }
        if verify {
            f.bytes.insert("cu_rows", bytes_i32(&cu_rows));
            f.bytes.insert("checkpoint_ids", bytes_i32(&checkpoint_ids));
            // The live rows' checkpoints start poisoned: every launch must
            // write them.
            let entry = f.bytes["state"].len() / slots;
            let state = f.bytes.get_mut("state").expect("verify state");
            for seq in 0..sequences {
                for row in 0..lengths[seq] {
                    let id = checkpoint_ids[seq * verify_row_limit() as usize + row] as usize;
                    state[id * entry..(id + 1) * entry].fill(POISON);
                    f.checkpoints.push(id * entry..(id + 1) * entry);
                }
            }
        }
        f
    }

    /// A launch on poisoned outputs and checkpoints wrote every live output
    /// row and every live checkpoint, and left the padding rows poisoned.
    fn written(&self, s: &Session, op: &Operands, label: &str) -> LabResult<()> {
        use kernel_lab::poison::{Fill, WriteMap, check, read};
        check(
            &format!("{label} output"),
            &read(s, &op.buffers["output"])?,
            &WriteMap::of(DType::Bf16).byte_prefix(self.live_output_bytes)?,
            Fill::Pattern(&[POISON]),
        )?;
        if self.checkpoints.is_empty() {
            return Ok(());
        }
        let state = read(s, &op.buffers["state"])?;
        for range in &self.checkpoints {
            if let Some(element) = state[range.clone()]
                .chunks_exact(self.state_element)
                .position(|element| element.iter().all(|&byte| byte == POISON))
            {
                return Err(format!(
                    "{label}: checkpoint element {element} at byte {} was not written",
                    range.start
                ));
            }
        }
        Ok(())
    }

    fn upload(&self, s: &Session) -> LabResult<Operands> {
        let mut op = Operands::new();
        for (&name, bytes) in &self.bytes {
            op.bytes(s, name, bytes)?;
        }
        op.scalars = self.scalars.clone();
        Ok(op)
    }

    fn reset(&self, s: &Session, op: &Operands) -> LabResult<()> {
        for name in [self.input, self.live, "output", "state"] {
            if let Some(bytes) = self.bytes.get(name) {
                op.buffers[name]
                    .copy_from_bytes(bytes, &s.stream)
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
}

struct Producer {
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    blocks: u32,
}

impl Producer {
    fn new(s: &Session) -> LabResult<Self> {
        let module = s.module("probe_pdl_prefetch_sm120")?;
        let blocks = s.context.device().multiprocessor_count() as u32;
        let eviction = s.allocate(EVICTION_BYTES)?;
        let flush = s
            .module("probe_sm120")?
            .function("chop_probe_flush_v1")
            .map_err(|e| e.to_string())?;
        flush
            .launch(
                &s.stream,
                launch::simple_config(
                    (1024, 1, 1),
                    (256, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    false,
                )?,
                &[
                    KernelArgument::Buffer(&eviction),
                    KernelArgument::U64((EVICTION_BYTES / 4) as u64),
                ],
            )
            .map_err(|e| e.to_string())?;
        Ok(Self {
            function: module
                .function("chop_probe_pdl_prefetch_producer_v1")
                .map_err(|e| e.to_string())?,
            eviction,
            checksum: s.allocate(blocks as usize * 128 * 4)?,
            blocks,
        })
    }

    fn launch(&self, stream: &Stream, case: &Case, arm: usize, live: u32) -> LabResult<()> {
        let op = &case.operands[arm];
        self.function
            .launch(
                stream,
                launch::simple_config(
                    (self.blocks, 1, 1),
                    (128, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64((EVICTION_BYTES / 16) as u64),
                    KernelArgument::Buffer(&case.staged[arm].descriptors),
                    KernelArgument::U32(case.staged[arm].count),
                    KernelArgument::Buffer(&op.buffers[case.live]),
                    KernelArgument::U32(live),
                    KernelArgument::U64(TAIL_NS),
                ],
            )
            .map_err(|e| e.to_string())
    }
}

struct StagedOperands {
    descriptors: DeviceBuffer,
    count: u32,
    // Keep every descriptor's source allocation alive through graph replay.
    _fresh: Vec<DeviceBuffer>,
}

impl StagedOperands {
    fn new(s: &Session, op: &Operands, f: &Fixture, include_state: bool) -> LabResult<Self> {
        let mut descriptors = Vec::new();
        let mut fresh_buffers = Vec::new();
        for name in [
            "input",
            "qkv",
            "a",
            "b",
            "gate",
            "slot_ids",
            "cu_rows",
            "checkpoint_ids",
            "state",
        ] {
            let Some(bytes) = f.bytes.get(name) else {
                continue;
            };
            if name == "state" && !include_state {
                continue;
            }
            let fresh = s.allocate(bytes.len())?;
            if name != "state" {
                fresh
                    .copy_from_bytes(bytes, &s.stream)
                    .map_err(|e| e.to_string())?;
            }
            // State is saved before poisoning and restored after the tail,
            // so replay preserves recurrence instead of resetting it.
            let flags = u32::from(name == "state")
                | if ["slot_ids", "cu_rows", "checkpoint_ids"].contains(&name) {
                    0
                } else {
                    2
                };
            for buffer in [&fresh, &op.buffers[name]] {
                descriptors.extend_from_slice(
                    &buffer
                        .device_address_at(0, bytes.len())
                        .map_err(|e| e.to_string())?
                        .to_ne_bytes(),
                );
            }
            descriptors.extend_from_slice(
                &(bytes.len() as u64 / 4)
                    .try_into()
                    .map(u32::to_ne_bytes)
                    .map_err(|_| "staged operand exceeds u32 words")?,
            );
            descriptors.extend_from_slice(&flags.to_ne_bytes());
            fresh_buffers.push(fresh);
        }
        let buffer = s.allocate(descriptors.len())?;
        buffer
            .copy_from_bytes(&descriptors, &s.stream)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            descriptors: buffer,
            count: fresh_buffers.len() as u32,
            _fresh: fresh_buffers,
        })
    }
}

struct Case {
    operands: [Operands; 2],
    kernels: [Kernel; 2],
    staged: [StagedOperands; 2],
    live: &'static str,
    rows: u32,
    sequences: u32,
}

impl Case {
    fn new(
        s: &Session,
        modules: [&Module; 2],
        kind: &str,
        f: &Fixture,
        include_state: bool,
    ) -> LabResult<Self> {
        let operands = [f.upload(s)?, f.upload(s)?];
        let staged = [
            StagedOperands::new(s, &operands[0], f, include_state)?,
            StagedOperands::new(s, &operands[1], f, include_state)?,
        ];
        let load = |module| {
            Kernel::load(module, &symbol(kind))
                .map(|k| k.with_bucket(f.row_capacity as u32, f.capacity as u32))
        };
        Ok(Self {
            operands,
            kernels: [load(modules[0])?, load(modules[1])?],
            staged,
            live: f.live,
            rows: f.rows as u32,
            sequences: f.sequences as u32,
        })
    }

    fn launch(&self, s: &Session, stream: &Stream, arm: usize, pdl: bool) -> LabResult<()> {
        self.kernels[arm].launch(
            s,
            stream,
            &self.operands[arm],
            self.rows,
            self.sequences,
            pdl,
        )
    }

    fn snapshot(&self, s: &Session, arm: usize) -> LabResult<BTreeMap<String, Vec<u8>>> {
        let op = &self.operands[arm];
        ["output", "state"]
            .into_iter()
            .filter(|name| op.buffers.contains_key(name))
            .map(|name| Ok((name.into(), op.read(s, name, op.buffers[name].byte_len())?)))
            .collect()
    }
}

fn graph(
    s: &Session,
    p: &Producer,
    cases: &[Case],
    arm: usize,
    zero: bool,
) -> LabResult<GraphExec> {
    s.capture(|stream| {
        for c in cases {
            p.launch(stream, c, arm, if zero { 0 } else { c.sequences })?;
            c.launch(s, stream, arm, true)?;
        }
        Ok(())
    })
}

fn exact(
    actual: &BTreeMap<String, Vec<u8>>,
    expected: &BTreeMap<String, Vec<u8>>,
    label: &str,
) -> LabResult<()> {
    for (name, a) in actual {
        let b = &expected[name];
        if a.len() != b.len() {
            return Err(format!("{label}: {name} length mismatch"));
        }
        // Byte-slice equality uses memcmp for wide recurrent pools. Scan for
        // the first differing byte only when a comparison actually fails.
        if a == b {
            continue;
        }
        if let Some(i) = a.iter().zip(b).position(|(a, b)| a != b) {
            return Err(format!("{label}: {name} byte {i}: {} != {}", a[i], b[i]));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an exclusive assigned GPU, pinned baseline cubins, and a receipt directory"]
fn linear_constants_pdl_bitwise_and_graph_bench() {
    assert!(
        kernel_lab::selected_ordinals()
            .expect("GPU selection")
            .is_some(),
        "the dedicated ignored test requires an explicitly selected GPU"
    );
    gpu_test(20_u64 << 30, |s| {
        if s.metadata["measurement_label"] != "exclusive-device" {
            return Err("PDL prefetch qualification requires an empty assigned GPU".into());
        }
        let (old, baseline_hash) = baseline(s)?;
        let new = module(s)?;
        let producer = Producer::new(s)?;
        let mut checks = Vec::new();
        for kind in ["conv_decode", "conv_verify", "delta_decode", "delta_verify", "gated_norm"] {
            let counts: &[usize] = if kind == "gated_norm" {
                &[1, 7, 16, 31, 32, 257, 2047, 2048]
            } else { &[1, 7, 16, 31, 32, 33, 64, 100, 128, 255, 256] };
            let mut cases = counts.iter().map(|&n| (n, 0)).collect::<Vec<_>>();
            if kind.ends_with("_verify") {
                cases.extend([1, 16, 32, 256].map(|n| (n, 1)));
                cases.push((256, 8));
            }
            for (live, row_mode) in cases {
                let capacity = if kind == "gated_norm" { live + 1 } else if live <= 32 { 32 } else { 256 };
                let f = Fixture::new(kind, live, capacity, live, row_mode);
                let c = Case::new(s, [&old, &new], kind, &f, true)?;
                let label = format!("{kind}/{live}/{row_mode}");
                // Both arms start from poisoned outputs and checkpoints.
                c.launch(s, &s.stream, 0, false)?;
                f.written(s, &c.operands[0], &format!("{label}/eager baseline"))?;
                let expected = c.snapshot(s, 0)?;
                c.launch(s, &s.stream, 1, false)?;
                f.written(s, &c.operands[1], &format!("{label}/eager"))?;
                exact(&c.snapshot(s, 1)?, &expected, &format!("{label}/eager"))?;
                let cases = [c];
                let graphs = [graph(s, &producer, &cases, 0, false)?, graph(s, &producer, &cases, 1, false)?];
                for replay in 0..3 {
                    // Every replay starts from the inputs and state with the
                    // outputs and checkpoints poisoned.
                    for arm in 0..2 { f.reset(s, &cases[0].operands[arm])?; }
                    for graph in &graphs { graph.launch(&s.stream).map_err(|e| e.to_string())?; }
                    for arm in 0..2 {
                        f.written(s, &cases[0].operands[arm], &format!("{label}/PDL/{replay}/arm{arm}"))?;
                    }
                    let a = cases[0].snapshot(s, 0)?;
                    let b = cases[0].snapshot(s, 1)?;
                    exact(&b, &a, &format!("{label}/PDL/{replay}"))?;
                    exact(&b, &expected, "PDL producer vs eager")?;
                }
                for arm in 0..2 {
                    f.reset(s, &cases[0].operands[arm])?;
                    let before = cases[0].snapshot(s, arm)?;
                    graph(s, &producer, &cases, arm, true)?.launch(&s.stream).map_err(|e| e.to_string())?;
                    exact(&cases[0].snapshot(s, arm)?, &before, "zero-live replay")?;
                }
                checks.push(json!({"kind":kind, "live":live, "capacity":capacity, "row_mode":row_mode,
                    "eager_bitwise":true,"pdl_replays":3,"whole_state_bitwise":true,"zero_live_unchanged":true,
                    "producer_restores_all_read_operands":true,"poisoned":["output","checkpoints"]}));
                eprintln!("PDL_BITWISE {kind} live={live} row_mode={row_mode}: passed");
            }
        }
        let mut timings = Vec::new();
        if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1") {
            for kind in ["conv_verify", "delta_verify", "gated_norm"] {
                for live in [1, 16, 32] {
                    for rows_per_sequence in [1, 8] {
                        let cases = (0..LAYERS).map(|layer| {
                            let count = if kind == "gated_norm" { live * rows_per_sequence } else { live };
                            let f = Fixture::new(kind, count, count, layer + 101, rows_per_sequence);
                            // State poisoning belongs to correctness. Avoid a
                            // synthetic state-copy cost in the timing producer.
                            Case::new(s, [&old, &new], kind, &f, false)
                        }).collect::<LabResult<Vec<_>>>()?;
                        let graphs = [graph(s, &producer, &cases, 0, false)?, graph(s, &producer, &cases, 1, false)?];
                        let mut passes = Vec::new();
                        for pass in 0..4 {
                            let mut us = [0.0;2];
                            let mut min_us = [0.0;2];
                            for arm in [pass % 2, 1 - pass % 2] {
                                let t = kernel_lab::time(s, 2, 11, |stream| graphs[arm].launch(stream).map_err(|e| e.to_string()))?;
                                us[arm] = f64::from(t.median_ms) * 1000.0;
                                min_us[arm] = f64::from(t.min_ms) * 1000.0;
                            }
                            passes.push(json!({"baseline_graph_us":us[0],"candidate_graph_us":us[1],
                                "baseline_min_us":min_us[0],"candidate_min_us":min_us[1],
                                "saving_per_layer_us":(us[0]-us[1])/LAYERS as f64,"first_arm":pass%2}));
                        }
                        // One more replay of each arm on poisoned outputs
                        // gives the compared state.
                        for c in &cases {
                            for op in &c.operands {
                                kernel_lab::poison::poison(s, &[&op.buffers["output"]])?;
                            }
                        }
                        for graph in &graphs { graph.launch(&s.stream).map_err(|e| e.to_string())?; }
                        for c in &cases {
                            exact(&c.snapshot(s, 1)?, &c.snapshot(s, 0)?, "timed graph state")?;
                            let written = c.snapshot(s, 1)?;
                            if written["output"].iter().all(|&byte| byte == POISON) {
                                return Err(format!("{kind} C{live}: the timed graph wrote no output"));
                            }
                        }
                        eprintln!("PDL_PREFETCH {kind} C{live} rows={rows_per_sequence}: {passes:?}");
                        timings.push(json!({"kind":kind,"live":live,"rows_per_sequence":rows_per_sequence,"passes":passes}));
                    }
                }
            }
        }
        let locks: Vec<Value> = serde_json::from_str(include_str!("../../kernels.lock")).map_err(|e| e.to_string())?;
        let candidate = locks.iter().find(|r| r["entries"].as_array().is_some_and(|es|
            es.iter().any(|e| e["symbol"] == symbol("conv_verify")))).ok_or("missing candidate identity")?;
        let receipt = json!({"metadata":s.metadata,"baseline_sha256":baseline_hash,
            "candidate":candidate,"checks":checks,"timings":timings,"layers":LAYERS,
            "eviction_bytes_per_layer":EVICTION_BYTES,"producer_tail_ns":TAIL_NS,
            "timing_includes_producer":true,"overlap_is_synthetic_upper_bound":true,"correct":true});
        let root = PathBuf::from(std::env::var("CHOP_PDL_OUTPUT_DIR").map_err(|e| e.to_string())?);
        std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        let path = root.join(format!("linear-pdl-tp{}-gpu{}.json", geometry().ranks, s.ordinal));
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(|e| e.to_string())?;
        use std::io::Write;
        file.write_all(&serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        eprintln!("PDL_PREFETCH receipt {}", path.display());
        Ok(())
    }).expect("PDL constant comparison").expect("GPU test must run");
}
