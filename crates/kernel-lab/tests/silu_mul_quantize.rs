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
//! code, which must fail. Poisoned eager, relaunch, graph and PDL replays
//! run at the live counts `poison::live_counts` lists.

use kernel_core::contract::DType;
use kernel_lab::kit::cases::silu_mul_quantize::{Boundaries, GROUP, Outcomes, outcomes, spread};
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
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("silu_mul_quantize_gate"));
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
