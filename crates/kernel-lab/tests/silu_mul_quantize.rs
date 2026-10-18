//! The clamped SwiGLU with BF16 rounding and ceil E4M3/32 quantization
//! (`kernels_sm120::row::silu_mul_quantize`) on the GPU. Run with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test
//! silu_mul_quantize -- --test-threads=1`; without a GPU only the CPU test
//! runs.
//!
//! Every lab entry (widths 32, 576 and 2304, 16 rows) runs over five input
//! sets: ordinary values past both clamps, products steered onto BF16
//! midpoints, group amaxes steered onto a scale-selection boundary,
//! extreme gates (saturated exponentials, the clamps, amaxes under the
//! floor), and signed zeros. Each launch's codes and scales pass the
//! boundary-aware gate (`kit::cases::silu_mul_quantize::Outcomes::check`):
//! bitwise wherever the inputs place the product clear of every boundary,
//! one of the two adjacent outcomes where `expf`'s error reaches one. The
//! counts of boundary elements and groups, and how many the kernel resolved
//! to the upper outcome, go into the receipt. The negative control moves
//! one exact element's code by one step and one exact group's scale by one
//! code, which must fail. Each entry also runs on gate and up views that
//! start 4 bytes into their allocations, which must give the aligned run's
//! bits. Poisoned eager, relaunch, graph and PDL replays run at the live
//! counts `poison::live_counts` lists.
//!
//! The gate's allowance assumes CUDA's `expf` is within 2 ulp of the exact
//! exponential, which NVIDIA documents as observed. The expf error probe
//! (`families::silu_mul_quantize::EXPF`) makes it a measured fact for this
//! toolkit: it evaluates the op's own exponential over every gate the op can
//! see (each finite F32 gate up to the clamp, so every input from minus the
//! clamp up), records the step distance from the correctly rounded F32 and
//! the error in ulps against exp in F64, and hands back every gate at two
//! steps or more (or with an infinity on one side only), which the host
//! checks against the gate's own enclosure (`exp_ends`). The receipt records the measured maxima against the
//! allowance and the toolkit the probe was built with.

use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan};
use kernel_lab::families::silu_mul_quantize::{EXPF, EXPF_CUBIN, EXPF_DISTANCES, EXPF_THREADS};
use kernel_lab::kit::cases::silu_mul_quantize::{
    Boundaries, EXP_ULPS, GROUP, Outcomes, exp_ends, outcomes, spread,
};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Replay, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::row::silu_mul_quantize::{Entry, LAB_CAPACITY, LAB_CUBIN, lab_entries};
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const ROWS: usize = LAB_CAPACITY as usize;
const GATE: u64 = 0;
const UP: u64 = 1;
const CODES: u64 = 2;
const SCALES: u64 = 3;
const LIVE: u64 = 4;
// The PDL producer's outputs.
const P_CODES: u64 = 5;
const P_SCALES: u64 = 6;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// One input set: its name and F32 gate and up `[rows, width]`.
struct Inputs {
    name: &'static str,
    gate: Vec<f32>,
    up: Vec<f32>,
}

/// The input sets of a width.
fn input_sets(width: usize) -> Vec<Inputs> {
    let len = ROWS * width;
    let silu = |g: f32| g / (1.0 + (-g).exp());
    let bf16 = |bits: u16| kernel_core::codec::bf16_to_f32(bits);
    let to_bf16 = kernel_core::codec::f32_to_bf16;
    // Midpoints between adjacent BF16 values spread over binades.
    let gate_steered = spread(11, len, 4.0)
        .into_iter()
        .map(|g| g.abs() + 0.25)
        .collect::<Vec<_>>();
    let up_steered = gate_steered
        .iter()
        .enumerate()
        .map(|(i, &g)| {
            let lower = to_bf16(0.25 + (i % 97) as f32 / 16.0);
            let midpoint = (bf16(lower) + bf16(lower + 1)) / 2.0;
            midpoint / silu(g)
        })
        .collect::<Vec<_>>();
    // Each group's largest product on the midpoint between 448 * 2^k and
    // the next BF16 value, where the ceil scale's exponent changes; the
    // rest small.
    let gate_scale = vec![1.5_f32; len];
    let up_scale = (0..len)
        .map(|i| {
            let k = ((i / GROUP) % 9) as i32 - 6;
            let target = if i % GROUP == 7 {
                449.0 * 2_f32.powi(k)
            } else {
                0.01 * ((i % 13) as f32 + 1.0)
            };
            target / silu(1.5)
        })
        .collect::<Vec<_>>();
    // Saturated exponentials, the overflow edge, the clamps and amaxes
    // under the floor.
    let edges = [
        -200.0, -104.0, -88.8, -88.7, -87.0, -20.0, 0.0, 1e-30, 9.99, 10.0, 10.01, 1e30,
    ];
    let gate_edge = (0..len).map(|i| edges[i % edges.len()]).collect::<Vec<_>>();
    let ups = [1.0, -1.0, 9.5, -10.0, 10.5, -1e30, 1e-6, 3.0];
    let up_edge = (0..len)
        .map(|i| ups[(i / 3) % ups.len()])
        .collect::<Vec<_>>();
    let zeros = (0..len)
        .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
        .collect::<Vec<_>>();
    vec![
        Inputs {
            name: "ordinary",
            gate: spread(1, len, 12.0),
            up: spread(2, len, 12.0),
        },
        Inputs {
            name: "bf16 midpoints",
            gate: gate_steered,
            up: up_steered,
        },
        Inputs {
            name: "scale boundaries",
            gate: gate_scale,
            up: up_scale,
        },
        Inputs {
            name: "extremes",
            gate: gate_edge,
            up: up_edge,
        },
        Inputs {
            name: "signed zeros",
            gate: zeros.clone(),
            up: zeros.iter().rev().copied().collect(),
        },
    ]
}

struct Fixture<'a> {
    entry: &'a Entry,
    function: Function,
    buffers: BTreeMap<u64, DeviceBuffer>,
}

impl<'a> Fixture<'a> {
    fn new(
        session: &Session,
        module: &Module,
        entry: &'a Entry,
        inputs: &Inputs,
    ) -> LabResult<Self> {
        let function = module.function(&entry.symbol).map_err(err)?;
        let width = entry.query.width as usize;
        let mut buffers = BTreeMap::new();
        buffers.insert(GATE, session.upload_f32(&inputs.gate)?);
        buffers.insert(UP, session.upload_f32(&inputs.up)?);
        for handle in [CODES, P_CODES] {
            buffers.insert(handle, session.allocate(ROWS * width)?);
        }
        for handle in [SCALES, P_SCALES] {
            buffers.insert(handle, session.allocate(ROWS * width / GROUP)?);
        }
        buffers.insert(LIVE, session.allocate(4)?);
        let fixture = Self {
            entry,
            function,
            buffers,
        };
        fixture.set_live(session, ROWS)?;
        Ok(fixture)
    }

    fn set_live(&self, session: &Session, live: usize) -> LabResult<()> {
        poison::write(
            session,
            &self.buffers[&LIVE],
            0,
            &(live as u32).to_ne_bytes(),
        )
    }

    fn launch(&self, session: &Session, outputs: [u64; 2], pdl: bool) -> LabResult<()> {
        let mut plan = self.entry.plan(ROWS as u32).map_err(err)?;
        plan.pdl = pdl;
        let args = self
            .entry
            .arguments(GATE, UP, outputs[0], outputs[1], LIVE)
            .map_err(err)?;
        let bindings = Bindings {
            buffers: self.buffers.iter().map(|(&k, v)| (k, v)).collect(),
        };
        launch::launch(
            &session.context,
            &self.function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    }

    fn bytes(&self, session: &Session, handle: u64, len: usize) -> LabResult<Vec<u8>> {
        let mut bytes = vec![0; len];
        self.buffers[&handle]
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(err)?;
        Ok(bytes)
    }
}

/// The negative control on a kernel's own output: one exact element's code
/// moved one step and one exact group's scale moved one code each fail.
fn control(admissible: &Outcomes, codes: &[u8], scales: &[u8]) -> LabResult<()> {
    let exact_group = |g: usize| admissible.scales[g].0 == admissible.scales[g].1;
    let element = (0..codes.len()).find(|&i| {
        let (lo, hi) = admissible.values[i];
        lo == hi && exact_group(i / GROUP) && codes[i] & 0x7f != 0 && codes[i] & 0x7f < 0x7e
    });
    if let Some(at) = element {
        let mut wrong = codes.to_vec();
        wrong[at] += 1;
        if admissible.check(&wrong, scales).is_ok() {
            return Err(format!("a code one step off at exact element {at} passed"));
        }
    }
    if let Some(group) = (0..scales.len()).find(|&g| exact_group(g) && scales[g] < 0xfe) {
        let mut wrong = scales.to_vec();
        wrong[group] += 1;
        if admissible.check(codes, &wrong).is_ok() {
            return Err(format!(
                "a scale one code off at exact group {group} passed"
            ));
        }
    }
    Ok(())
}

fn gate_entry(session: &Session, module: &Module, entry: &Entry) -> LabResult<Value> {
    let width = entry.query.width as usize;
    let mut sets = Vec::new();
    for inputs in input_sets(width) {
        let fixture = Fixture::new(session, module, entry, &inputs)?;
        let outputs = [&fixture.buffers[&CODES], &fixture.buffers[&SCALES]];
        poison::poison(session, &outputs)?;
        fixture.launch(session, [CODES, SCALES], false)?;
        session.stream.synchronize().map_err(err)?;
        let codes = fixture.bytes(session, CODES, ROWS * width)?;
        let scales = fixture.bytes(session, SCALES, ROWS * width / GROUP)?;
        let admissible = outcomes(entry.query, &inputs.gate, &inputs.up, ROWS)?;
        let counts: Boundaries = admissible
            .check(&codes, &scales)
            .map_err(|e| format!("{} {}: {e}", entry.symbol, inputs.name))?;
        control(&admissible, &codes, &scales)
            .map_err(|e| format!("{} {}: {e}", entry.symbol, inputs.name))?;
        if inputs.name == "bf16 midpoints" && counts.elements == 0 {
            return Err(format!(
                "{}: the steered midpoints reached no boundary",
                entry.symbol
            ));
        }
        if inputs.name == "scale boundaries" && counts.groups == 0 {
            return Err(format!(
                "{}: the steered amaxes reached no scale boundary",
                entry.symbol
            ));
        }
        sets.push(json!({
            "inputs": inputs.name,
            "elements": codes.len(),
            "groups": scales.len(),
            "boundary_elements": counts.elements,
            "boundary_groups": counts.groups,
            "boundary_elements_taken_high": counts.elements_taken_high,
        }));
    }
    Ok(json!({"symbol": entry.symbol, "width": width, "sets": sets}))
}

/// The entry over gate and up views 4 bytes into their allocations, which
/// must give the aligned launch's bits: the kernel needs only F32
/// alignment.
fn misaligned_entry(session: &Session, module: &Module, entry: &Entry) -> LabResult<Value> {
    let width = entry.query.width as usize;
    let inputs = input_sets(width).remove(0);
    let fixture = Fixture::new(session, module, entry, &inputs)?;
    let outputs = [&fixture.buffers[&CODES], &fixture.buffers[&SCALES]];
    poison::poison(session, &outputs)?;
    fixture.launch(session, [CODES, SCALES], false)?;
    let shifted = |values: &[f32]| {
        let mut padded = vec![0.0_f32; values.len() + 1];
        padded[1..].copy_from_slice(values);
        session.upload_f32(&padded)
    };
    let (gate, up) = (shifted(&inputs.gate)?, shifted(&inputs.up)?);
    let bytes = ROWS * width * 4;
    let address = |buffer: &DeviceBuffer| buffer.device_address_at(4, bytes).map_err(err);
    let mut args = entry
        .arguments(GATE, UP, P_CODES, P_SCALES, LIVE)
        .map_err(err)?;
    args.0[0] = Argument::U64(address(&gate)?);
    args.0[1] = Argument::U64(address(&up)?);
    if address(&gate)? % 8 != 4 || address(&up)? % 8 != 4 {
        return Err("the shifted views are 8-byte aligned".into());
    }
    let outputs = [&fixture.buffers[&P_CODES], &fixture.buffers[&P_SCALES]];
    poison::poison(session, &outputs)?;
    let bindings = Bindings {
        buffers: fixture.buffers.iter().map(|(&k, v)| (k, v)).collect(),
    };
    launch::launch(
        &session.context,
        &fixture.function,
        &session.stream,
        entry.plan(ROWS as u32).map_err(err)?,
        &args,
        &bindings,
    )?;
    session.stream.synchronize().map_err(err)?;
    for (aligned, shifted, len) in [
        (CODES, P_CODES, ROWS * width),
        (SCALES, P_SCALES, ROWS * width / GROUP),
    ] {
        if fixture.bytes(session, aligned, len)? != fixture.bytes(session, shifted, len)? {
            return Err(format!(
                "{}: the 4-byte shifted views changed the output bits",
                entry.symbol
            ));
        }
    }
    Ok(json!({"symbol": entry.symbol, "shifted_bytes": 4, "bits_equal": true}))
}

fn replay_entry(session: &Session, module: &Module, entry: &Entry) -> LabResult<Value> {
    let width = entry.query.width as usize;
    let inputs = input_sets(width).remove(0);
    let fixture = Fixture::new(session, module, entry, &inputs)?;
    let graph = session.capture(|_| fixture.launch(session, [CODES, SCALES], true))?;
    let mut evidence = Vec::new();
    for live in poison::live_counts(ROWS as u32) {
        let live = live as usize;
        fixture.set_live(session, live)?;
        let mut replay = Replay::new(session)
            .output(
                "codes",
                &fixture.buffers[&CODES],
                WriteMap::of(DType::Fp8E4M3).prefix(live * width)?,
            )
            .output(
                "scales",
                &fixture.buffers[&SCALES],
                WriteMap::of(DType::Ue8M0).prefix(live * width / GROUP)?,
            );
        replay.eager(|| fixture.launch(session, [CODES, SCALES], false))?;
        replay.relaunch(|| fixture.launch(session, [CODES, SCALES], false))?;
        replay.graph(&graph)?;
        replay.pdl(
            || fixture.launch(session, [P_CODES, P_SCALES], false),
            || fixture.launch(session, [CODES, SCALES], true),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        evidence.push(json!({"live": live, "replay": receipt}));
    }
    Ok(json!({"symbol": entry.symbol, "live_counts": evidence}))
}

#[test]
fn silu_mul_quantize_cpu_contract() {
    let symbols = kernel_lab::cubin_entries(LAB_CUBIN).unwrap();
    for entry in lab_entries() {
        assert!(symbols.contains(&entry.symbol.as_str()), "{}", entry.symbol);
        let width = entry.query.width as usize;
        for inputs in input_sets(width) {
            assert_eq!(inputs.gate.len(), ROWS * width);
            assert_eq!(inputs.up.len(), ROWS * width);
            let counts = outcomes(entry.query, &inputs.gate, &inputs.up, ROWS)
                .unwrap()
                .boundaries();
            match inputs.name {
                "bf16 midpoints" => assert!(counts.elements > 0, "{counts:?}"),
                "scale boundaries" => assert!(counts.groups > 0, "{counts:?}"),
                _ => {}
            }
        }
    }
    kernel_lab::cubin(LAB_CUBIN).unwrap();
}

#[test]
fn silu_mul_quantize_passes_the_boundary_aware_gate() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let module = session.module(LAB_CUBIN)?;
        let entries = lab_entries()
            .iter()
            .map(|entry| gate_entry(session, &module, entry))
            .collect::<LabResult<Vec<_>>>()?;
        let shifted = lab_entries()
            .iter()
            .map(|entry| misaligned_entry(session, &module, entry))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("silu_mul_quantize_gate"));
        receipt.insert("shifted_views".into(), json!(shifted));
        receipt.insert(
            "exp_ulps".into(),
            json!(kernel_lab::kit::cases::silu_mul_quantize::EXP_ULPS),
        );
        receipt.insert("entries".into(), json!(entries));
        session.receipt("silu_mul_quantize_gate", receipt)?;
        Ok(())
    })
    .expect("quantized SwiGLU gate");
}

#[test]
fn silu_mul_quantize_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let module = session.module(LAB_CUBIN)?;
        let entries = lab_entries()
            .iter()
            .map(|entry| replay_entry(session, &module, entry))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("silu_mul_quantize_poison"));
        receipt.insert("entries".into(), json!(entries));
        session.receipt("silu_mul_quantize_poison", receipt)?;
        Ok(())
    })
    .expect("quantized SwiGLU poisoned replays");
}

/// Words per class of the expf probe's results: the distance histogram,
/// then the packed largest error.
const EXPF_SLOTS: usize = EXPF_DISTANCES + 1;
/// Gates the probe hands back for the host to check.
const FLAG_CAPACITY: usize = 1 << 22;

/// One probe launch's results.
struct Measured {
    /// Per class (every gate, the gates within the clamp), the histogram
    /// and the largest error with its gate.
    classes: [([u64; EXPF_DISTANCES], f32, f32); 2],
    /// `(gate, t)` of the flagged gates, at most [`FLAG_CAPACITY`] of
    /// them.
    flagged: Vec<(f32, f32)>,
    /// Every flagged gate, including any past the capacity.
    flags: u64,
}

impl Measured {
    fn largest_steps(&self, class: usize) -> usize {
        let (histogram, _, _) = &self.classes[class];
        (0..EXPF_DISTANCES)
            .rev()
            .find(|&d| histogram[d] > 0)
            .unwrap_or(0)
    }

    fn json(&self, class: usize) -> Value {
        let (histogram, error, gate) = &self.classes[class];
        json!({
            "gates": histogram.iter().sum::<u64>(),
            "steps_histogram": histogram,
            "steps_last_bucket_means_at_least": EXPF_DISTANCES - 1,
            "largest_steps": self.largest_steps(class),
            "largest_error_ulps": error,
            "largest_error_gate": format!("{gate:e}"),
            "largest_error_gate_bits": format!("{:#010x}", gate.to_bits()),
        })
    }
}

/// Runs the expf probe over gates `start..start + count` (bits), flagging
/// gates at `flag_steps` or more, with `values` receiving each `t` when
/// given.
fn probe_expf(
    session: &Session,
    function: &Function,
    start: u32,
    count: u64,
    clamp_bits: u32,
    flag_steps: u32,
    values: Option<&DeviceBuffer>,
) -> LabResult<Measured> {
    let results = session.allocate((2 * EXPF_SLOTS + 1) * 8)?;
    poison::write(session, &results, 0, &[0; (2 * EXPF_SLOTS + 1) * 8])?;
    let flagged = session.allocate(FLAG_CAPACITY * 8)?;
    let (results_handle, flagged_handle, values_handle) = (0, 1, 2);
    let mut buffers = BTreeMap::from([(results_handle, &results), (flagged_handle, &flagged)]);
    if let Some(values) = values {
        buffers.insert(values_handle, values);
    }
    let args = ArgumentList(vec![
        Argument::U32(start),
        Argument::U64(count),
        Argument::U32(clamp_bits),
        Argument::U32(flag_steps),
        Argument::DeviceBuffer(results_handle),
        Argument::DeviceBuffer(flagged_handle),
        Argument::U32(FLAG_CAPACITY as u32),
        values.map_or(Argument::NullDevicePointer, |_| {
            Argument::DeviceBuffer(values_handle)
        }),
    ]);
    let blocks = count.div_ceil(u64::from(EXPF_THREADS)).min(4096) as u32;
    let plan = LaunchPlan {
        grid: [blocks.max(1), 1, 1],
        block: [EXPF_THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: false,
    };
    launch::launch(
        &session.context,
        function,
        &session.stream,
        plan,
        &args,
        &Bindings { buffers },
    )?;
    session.stream.synchronize().map_err(err)?;
    let mut words = vec![0u8; (2 * EXPF_SLOTS + 1) * 8];
    results
        .copy_to_bytes(&mut words, &session.stream)
        .map_err(err)?;
    let words = words
        .chunks_exact(8)
        .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
        .collect::<Vec<_>>();
    let class = |c: usize| {
        let base = c * EXPF_SLOTS;
        let histogram: [u64; EXPF_DISTANCES] =
            words[base..base + EXPF_DISTANCES].try_into().unwrap();
        let packed = words[base + EXPF_DISTANCES];
        (
            histogram,
            f32::from_bits((packed >> 32) as u32),
            f32::from_bits(packed as u32),
        )
    };
    let flags = words[2 * EXPF_SLOTS];
    let kept = (flags as usize).min(FLAG_CAPACITY);
    let mut bytes = vec![0u8; kept * 8];
    if kept > 0 {
        let mut all = vec![0u8; FLAG_CAPACITY * 8];
        flagged
            .copy_to_bytes(&mut all, &session.stream)
            .map_err(err)?;
        bytes.copy_from_slice(&all[..kept * 8]);
    }
    let flagged = bytes
        .chunks_exact(8)
        .map(|w| {
            let word = u64::from_le_bytes(w.try_into().unwrap());
            (
                f32::from_bits((word >> 32) as u32),
                f32::from_bits(word as u32),
            )
        })
        .collect();
    Ok(Measured {
        classes: [class(0), class(1)],
        flagged,
        flags,
    })
}

/// The F32 steps between two non-negative values, the infinity one past
/// the largest finite value; NaN is far.
fn steps(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

/// The host's check of flagged gates against the gate's enclosure.
#[derive(Default)]
struct HostCheck {
    gates: usize,
    /// Gates whose `t` lies outside `exp_ends`, and the first of them.
    outside: usize,
    first_outside: Option<String>,
    /// The largest step distance from the host's correctly rounded
    /// exponential.
    largest_steps: u32,
}

impl HostCheck {
    fn add(&mut self, flagged: &[(f32, f32)]) {
        for &(g, t) in flagged {
            self.gates += 1;
            let [lo, hi] = exp_ends(g);
            if t.is_nan() || t < lo || t > hi {
                self.outside += 1;
                self.first_outside.get_or_insert_with(|| {
                    format!(
                        "expf(-({g:e})) = {t:e} ({:#010x}) is outside the enclosure {lo:e}..={hi:e}",
                        t.to_bits()
                    )
                });
            }
            self.largest_steps = self
                .largest_steps
                .max(steps(t, (-f64::from(g)).exp() as f32));
        }
    }

    fn of(flagged: &[(f32, f32)]) -> Self {
        let mut check = Self::default();
        check.add(flagged);
        check
    }

    fn json(&self) -> Value {
        json!({
            "gates": self.gates,
            "outside_enclosure": self.outside,
            "first_outside": self.first_outside,
            "largest_steps": self.largest_steps,
        })
    }
}

/// The probe's own check on samples around 1 and across the overflow
/// threshold: with every gate flagged, the device's histogram matches the
/// host's distances from its own rounding (up to two gates whose double
/// exponentials round differently), every value the probe wrote is the
/// flagged one, and a value moved past the allowance leaves the enclosure.
fn probe_controls(session: &Session, function: &Function, clamp_bits: u32) -> LabResult<Value> {
    let mut samples = Vec::new();
    for (name, start) in [
        ("around 1", 1.0_f32.to_bits() - (1 << 15)),
        ("overflow", 88.5_f32.to_bits() | 0x8000_0000),
    ] {
        let count = 1 << 16;
        let values = session.allocate(count * 4)?;
        let measured = probe_expf(
            session,
            function,
            start,
            count as u64,
            clamp_bits,
            0,
            Some(&values),
        )?;
        let mut written = vec![0f32; count];
        values
            .copy_to_f32(&mut written, &session.stream)
            .map_err(err)?;
        if measured.flags != count as u64 || measured.flagged.len() != count {
            return Err(format!(
                "{name}: {} of {count} gates flagged at 0 steps",
                measured.flagged.len()
            ));
        }
        let mut host = [0u64; EXPF_DISTANCES];
        for &(g, t) in &measured.flagged {
            let i = (g.to_bits() - start) as usize;
            if written[i].to_bits() != t.to_bits() {
                return Err(format!(
                    "{name}: gate {g:e} flagged {t:e}, wrote {:e}",
                    written[i]
                ));
            }
            let d = steps(t, (-f64::from(g)).exp() as f32) as usize;
            host[d.min(EXPF_DISTANCES - 1)] += 1;
        }
        let (device, _, _) = measured.classes[0];
        let differ = host
            .iter()
            .zip(&device)
            .map(|(h, d)| h.abs_diff(*d))
            .sum::<u64>();
        if differ > 4 {
            return Err(format!(
                "{name}: device histogram {device:?}, host {host:?}"
            ));
        }
        let check = HostCheck::of(&measured.flagged);
        if let Some(outside) = check.first_outside {
            return Err(format!("{name}: {outside}"));
        }
        // A value moved one step past the allowance fails.
        let (g, _) = *measured
            .flagged
            .iter()
            .find(|(g, _)| exp_ends(*g)[1] < f32::MAX)
            .ok_or("no finite sample gate")?;
        let mut moved = (-f64::from(g)).exp() as f32;
        for _ in 0..=EXP_ULPS {
            moved = moved.next_up();
        }
        if HostCheck::of(&[(g, moved)]).outside != 1 {
            return Err(format!("{name}: a value {} steps off passed", EXP_ULPS + 1));
        }
        samples.push(
            json!({"sample": name, "gates": count, "histogram": device, "host_histogram": host}),
        );
    }
    Ok(json!(samples))
}

/// The toolkit line and nvcc digest the lock records for the probe cubin.
fn probe_toolkit() -> LabResult<Value> {
    let lock: Value =
        serde_json::from_str(include_str!("../kernels.lock.d/silu_mul_quantize.lock"))
            .map_err(err)?;
    let cubin = lock
        .as_array()
        .and_then(|cubins| cubins.iter().find(|c| c["name"] == EXPF_CUBIN))
        .ok_or("the lock has no expf probe cubin")?;
    Ok(json!({
        "toolchain_version": cubin["toolchain_version"],
        "nvcc_sha256": cubin["nvcc_sha256"],
        "flags": cubin["flags"],
    }))
}

#[test]
fn silu_mul_quantize_expf_error_is_within_the_allowance() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let module = session.module(EXPF_CUBIN)?;
        let function = module.function(EXPF).map_err(err)?;
        let clamps = lab_entries()
            .iter()
            .map(|e| e.query.clamp.gate_max().to_bits())
            .collect::<std::collections::BTreeSet<_>>();
        let [clamp_bits] = clamps.into_iter().collect::<Vec<_>>()[..] else {
            return Err("the lab entries clamp the gate differently".into());
        };
        let controls = probe_controls(session, &function, clamp_bits)?;
        // Every finite gate up to the clamp: the non-negative ones to the
        // clamp, every negative one.
        let flag_steps = EXP_ULPS - 1;
        let ranges = [
            (0, u64::from(clamp_bits) + 1),
            (0x8000_0000, 0x7f7f_ffff + 1),
        ];
        let mut runs = Vec::new();
        let mut totals = [[0u64; EXPF_DISTANCES]; 2];
        let mut worst = [(0f32, 0f32); 2];
        let mut host = HostCheck::default();
        let mut flags = 0;
        for (start, count) in ranges {
            let measured = probe_expf(
                session, &function, start, count, clamp_bits, flag_steps, None,
            )?;
            host.add(&measured.flagged);
            flags += measured.flags;
            for class in 0..2 {
                let (histogram, error, gate) = measured.classes[class];
                for (total, count) in totals[class].iter_mut().zip(histogram) {
                    *total += count;
                }
                if error > worst[class].0 {
                    worst[class] = (error, gate);
                }
            }
            runs.push(json!({
                "start_bits": format!("{start:#010x}"),
                "gates": count,
                "every": measured.json(0),
                "within_clamp": measured.json(1),
            }));
        }
        let gates = totals[0].iter().sum::<u64>();
        let expected = ranges.iter().map(|&(_, count)| count).sum::<u64>();
        if gates != expected {
            return Err(format!("the probe measured {gates} gates of {expected}"));
        }
        let largest = |class: usize| {
            (0..EXPF_DISTANCES)
                .rev()
                .find(|&d| totals[class][d] > 0)
                .unwrap_or(0)
        };
        // The host checks every flagged gate (two steps or more, or an
        // infinity on one side only) against the gate's own enclosure. A
        // finite gate the device saw at one step or fewer is within two of
        // the host's rounding, which differs from the device's by at most
        // one step, so the allowance covers every gate when no flagged one
        // leaves the enclosure.
        let unchecked = flags - host.gates as u64;
        let covered = host.outside == 0 && unchecked == 0;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("silu_mul_quantize_expf_error"));
        receipt.insert("exp_ulps_allowance".into(), json!(EXP_ULPS));
        receipt.insert("documented_expf_ulps".into(), json!(2));
        receipt.insert("clamp_bits".into(), json!(format!("{clamp_bits:#010x}")));
        receipt.insert("toolkit".into(), probe_toolkit()?);
        receipt.insert("controls".into(), controls);
        receipt.insert("runs".into(), json!(runs));
        for (class, name) in [(0, "every_gate"), (1, "within_clamp")] {
            receipt.insert(
                name.into(),
                json!({
                    "gates": totals[class].iter().sum::<u64>(),
                    "steps_histogram": totals[class],
                    "largest_steps": largest(class),
                    "largest_error_ulps": worst[class].0,
                    "largest_error_gate": format!("{:e}", worst[class].1),
                    "largest_error_gate_bits": format!("{:#010x}", worst[class].1.to_bits()),
                }),
            );
        }
        let mut host_json = host.json();
        host_json["flag_steps"] = json!(flag_steps);
        host_json["flagged"] = json!(flags);
        host_json["unchecked_past_capacity"] = json!(unchecked);
        receipt.insert("host_checked".into(), host_json);
        receipt.insert("allowance_covers".into(), json!(covered));
        session.receipt("silu_mul_quantize_expf_error", receipt)?;
        if !covered {
            return Err(format!(
                "{} gates leave the enclosure of {EXP_ULPS} steps and {unchecked} flagged gates went unchecked (device largest {} steps): {}",
                host.outside,
                largest(0),
                host.first_outside.unwrap_or_default()
            ));
        }
        Ok(())
    })
    .expect("expf within the gate's allowance");
}
