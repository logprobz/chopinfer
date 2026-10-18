//! The model's own ops on SM120: the own-op cubins rebuilt against the
//! bundle's `kernels.lock.d`, prepared as the bundle prepares them, checked
//! against the contract's oracles with poisoned outputs, relaunches and
//! graph replays.

use crate::own_cubin::{gate_cubin_spec, rope_cubin_spec};
use crate::own_entries::{FAMILY, ROPE_CUBIN, gate_cubin, gate_geometries, gates};
use crate::own_ops::Sm120Own;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_core::backend::{Counts, Launch, Need, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::poison::{Replay, WriteMap};
use kernel_harness::{LabResult, Session};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::engram::{
    EngramGate, EngramGateBuffers, GateRow, check_gate_row, gate_reference,
};
use model_deepseek41_contract::own_ops::rope::{
    MAX_POSITIONS, PAIRS, RopeCosSin, RopeCosSinBuffers, RopeRows, RopeTable, reference,
};
use runtime_cuda::{DeviceBuffer, Stream};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The own-op cubins by name, rebuilt from the bundle's sources and
/// verified against its lock family (an ordinary build: a mismatch fails).
fn cubins() -> &'static BTreeMap<String, Vec<u8>> {
    static BYTES: OnceLock<BTreeMap<String, Vec<u8>>> = OnceLock::new();
    BYTES.get_or_init(|| {
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle");
        let bundle = std::fs::canonicalize(bundle).expect("bundle directory");
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("deepseek41-own");
        std::fs::create_dir_all(&out).expect("cubin output directory");
        let mut specs = vec![rope_cubin_spec(&bundle)];
        specs.extend(gate_geometries().map(|g| gate_cubin_spec(&bundle, g)));
        let names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
        let lock = LockDir::new(bundle.join("kernels.lock.d")).family(FAMILY, &names);
        let built = kernel_build::build_in_families(
            &Toolchain::detect().expect("CUDA 13.2 toolchain"),
            &specs,
            &out,
            &lock,
            BuildOptions {
                lock_update: false,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
            None,
        )
        .expect("the own-op cubins match their lock");
        built
            .iter()
            .map(|c| (c.name.clone(), std::fs::read(&c.path).expect("built cubin")))
            .collect()
    })
}

fn cubin(name: &str) -> &'static [u8] {
    cubins()
        .get(name)
        .unwrap_or_else(|| panic!("no own-op cubin {name}"))
}

fn view(handle: u64, bytes: usize) -> View {
    View {
        handle,
        bytes: bytes as u64,
    }
}

fn single(launches: Vec<Launch>) -> LabResult<Launch> {
    match <[Launch; 1]>::try_from(launches) {
        Ok([launch]) => Ok(launch),
        Err(launches) => Err(format!("expected one launch, got {}", launches.len())),
    }
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(4))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_values(len: usize, seed: u64, scale: f32) -> Vec<u16> {
    kernel_harness::wide_values(len, seed)
        .into_iter()
        .map(|v| f32_to_bf16(v * scale))
        .collect()
}

#[test]
fn own_cubins_match_their_lock_and_hold_every_symbol() {
    let has = |cubin: &str, symbol: &str| {
        let bytes = self::cubin(cubin);
        bytes.windows(symbol.len()).any(|w| w == symbol.as_bytes())
    };
    let own = Sm120Own::new(&FLASH);
    let rope = own
        .rope_cos_sin(
            &RopeCosSin {
                table: RopeTable::Compressed,
                rows: RopeRows::Positions,
            },
            Need::rows(64),
        )
        .expect("prepare RoPE rows");
    for (symbol, _) in &rope.info().symbols {
        assert!(has(ROPE_CUBIN, symbol), "{symbol} is not in the RoPE cubin");
    }
    let mut count = 0;
    for geometry in gate_geometries() {
        let own = Sm120Own::new(geometry);
        for op in gates(geometry) {
            let gate = own
                .engram_gate(&op, Need::rows(64))
                .expect("prepare a compiled gate");
            for (symbol, _) in &gate.info().symbols {
                assert!(has(&gate_cubin(geometry), symbol), "{symbol}");
                count += 1;
            }
        }
        let other = EngramGate {
            width: geometry.shape.hidden * 2,
            ..EngramGate::of(geometry, true)
        };
        assert!(own.engram_gate(&other, Need::rows(8)).is_err());
    }
    assert!(count >= 2, "the flash gates are compiled");
}

#[test]
fn engram_gate_requires_f32_kv_storage_and_declares_float_pointers() {
    for g in gate_geometries() {
        let spec = gate_cubin_spec(Path::new("."), g);
        for entry in &spec.entries {
            assert!(
                matches!(&entry.params[1], kernel_build::EntryParam::DevicePointer { name, cxx_type } if name == "kv" && cxx_type == "const float")
            );
        }
        for op in gates(g) {
            let rows = 2;
            let row = op.row_elements() * 2;
            let kv_bytes = rows as usize * op.kv_elements() * 4;
            let prepared = Sm120Own::new(g).engram_gate(&op, Need::rows(rows)).unwrap();
            let mut buffers = EngramGateBuffers {
                x: view(1, rows as usize * row),
                kv: view(2, kv_bytes),
                q_weight: view(3, row),
                k_weight: view(4, row),
                text: op.masked.then(|| view(5, rows as usize)),
                live_rows: view(6, 4),
                output: view(1, rows as usize * row),
            };
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(launches.len(), 1);
            assert!(launches[0].plan.pdl);
            for bytes in [kv_bytes / 2, kv_bytes - 1] {
                buffers.kv.bytes = bytes as u64;
                assert!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .is_err()
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GateMutation {
    Bf16Read,
    RoundValue,
}

impl GateMutation {
    const ALL: [Self; 2] = [Self::Bf16Read, Self::RoundValue];

    fn symbol(self, serving: &str) -> String {
        format!("{}_{self:?}_v2", serving.strip_suffix("_v2").unwrap())
    }

    fn source(self, original: &str) -> String {
        let edits = match self {
            Self::Bf16Read => vec![
                (
                    "const float* __restrict__ kv,",
                    "const uint16_t* __restrict__ kv,",
                ),
                ("const float* kv_row", "const uint16_t* kv_row"),
                ("const float* key", "const uint16_t* key"),
                ("const float* value", "const uint16_t* value"),
                (
                    "const float k = key[i];",
                    "const float k = bf16_bits_to_f32(key[i]);",
                ),
                (
                    "gate, value[i], h[j]",
                    "gate, bf16_bits_to_f32(value[i]), h[j]",
                ),
            ],
            Self::RoundValue => vec![(
                "gate, value[i], h[j]",
                "gate, bf16_bits_to_f32(f32_to_bf16_bits(value[i])), h[j]",
            )],
        };
        let mut source = original.to_owned();
        for (from, to) in edits {
            assert_eq!(source.matches(from).count(), 1, "mutation anchor {from}");
            source = source.replace(from, to);
        }
        source
    }
}

/// Compile actual device controls from the serving source. Their generated
/// sources, cubins and test-only locks stay in the external target directory.
fn gate_mutants() -> &'static BTreeMap<String, Vec<u8>> {
    static BYTES: OnceLock<BTreeMap<String, Vec<u8>>> = OnceLock::new();
    BYTES.get_or_init(|| {
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle");
        let original = std::fs::read_to_string(bundle.join("cuda/deepseek41_own.cuh")).unwrap();
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("deepseek41-gate-mutants");
        let mut specs = Vec::new();
        for mutation in GateMutation::ALL {
            let root = out.join(format!("{mutation:?}"));
            std::fs::create_dir_all(root.join("cuda")).unwrap();
            let root = root.canonicalize().unwrap();
            std::fs::write(
                root.join("cuda/deepseek41_own.cuh"),
                mutation.source(&original),
            )
            .unwrap();
            for g in gate_geometries() {
                let mut spec = gate_cubin_spec(&root, g);
                spec.name = format!("{}_{mutation:?}", spec.name);
                for entry in &mut spec.entries {
                    entry.symbol = mutation.symbol(&entry.symbol);
                    if mutation == GateMutation::Bf16Read {
                        let kernel_build::EntryParam::DevicePointer { cxx_type, .. } =
                            &mut entry.params[1]
                        else {
                            panic!("kv is a pointer")
                        };
                        *cxx_type = "const uint16_t".into();
                    }
                }
                specs.push(spec);
            }
        }
        let built = kernel_build::build(
            &Toolchain::detect().expect("CUDA toolchain for device controls"),
            &specs,
            &out,
            &out.join("mutants.lock"),
            BuildOptions {
                lock_update: true,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
        )
        .expect("compile and audit gate device controls");
        built
            .iter()
            .map(|c| (c.name.clone(), std::fs::read(&c.path).unwrap()))
            .collect()
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GateInput {
    Random,
    NonFinite,
    Precision,
}

struct GateInputs {
    x: Vec<u16>,
    kv: Vec<f32>,
    q: Vec<u16>,
    k: Vec<u16>,
    text: Vec<u8>,
}

fn gate_inputs(op: EngramGate, seed: u64, case: GateInput) -> GateInputs {
    let rows = GATE_CAPACITY as usize;
    let (row, kv_row) = (op.row_elements(), op.kv_elements());
    let mut x = bf16_values(rows * row, seed, 8.0);
    // wide_values is already BF16. Fill the low F32 mantissa bits so each
    // random key and value carries information that BF16 cannot represent.
    let mut state = seed | 1;
    let mut kv: Vec<_> = kernel_harness::wide_values(rows * kv_row, seed + 1)
        .into_iter()
        .map(|v| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32::from_bits(v.to_bits() | (state as u32 & 0xffff) | 1)
        })
        .collect();
    for i in 0..row {
        kv[3 * kv_row + i] = bf16_to_f32(x[3 * row + i]);
        kv[6 * kv_row + i] = -bf16_to_f32(x[6 * row + i]);
    }
    let mut q = bf16_values(row, seed + 2, 1.0);
    let mut k = bf16_values(row, seed + 3, 1.0);
    match case {
        GateInput::Random => {}
        GateInput::Precision => {
            x.fill(f32_to_bf16(-1.0));
            for kv in kv.chunks_exact_mut(kv_row) {
                kv[..row].fill(-1.001);
                kv[row..].fill(1.002);
            }
            q.fill(f32_to_bf16(1.0));
            k.fill(f32_to_bf16(1.0));
        }
        GateInput::NonFinite => {
            // Only row 2 is poisoned. It is masked in the masked entry and
            // open in the other entry; other streams retain finite keys.
            kv[2 * kv_row] = f32::NAN;
            kv[2 * kv_row + 1] = f32::INFINITY;
            kv[2 * kv_row + 2] = f32::NEG_INFINITY;
            kv[2 * kv_row + row] = f32::NAN;
            kv[2 * kv_row + row + 1] = f32::INFINITY;
            kv[2 * kv_row + row + 2] = f32::NEG_INFINITY;
        }
    }
    GateInputs {
        x,
        kv,
        q,
        k,
        text: (0..rows).map(|r| u8::from(r % 4 != 2)).collect(),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn engram_gate_mutants_compile_and_precision_controls_reject() {
    let mutants = gate_mutants();
    assert_eq!(mutants.len(), 2 * gate_geometries().count());
    for g in gate_geometries() {
        for op in gates(g) {
            let data = gate_inputs(op, 11, GateInput::Precision);
            let row = GateRow {
                x: &data.x[..op.row_elements()],
                kv: &data.kv[..op.kv_elements()],
                q_weight: &data.q,
                k_weight: &data.k,
                open: true,
            };
            let expected = gate_reference(&op, row).unwrap();
            for mutation in GateMutation::ALL {
                let kv: Vec<_> = match mutation {
                    GateMutation::Bf16Read => f32_bytes(row.kv)
                        .chunks_exact(2)
                        .take(op.kv_elements())
                        .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                        .collect(),
                    GateMutation::RoundValue => row
                        .kv
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            if i >= op.row_elements() {
                                bf16_to_f32(f32_to_bf16(v))
                            } else {
                                v
                            }
                        })
                        .collect(),
                };
                let wrong = gate_reference(&op, GateRow { kv: &kv, ..row }).unwrap();
                let actual: Vec<_> = wrong
                    .values
                    .iter()
                    .map(|&v| f32_to_bf16(v as f32))
                    .collect();
                assert!(check_gate_row(&actual, &expected).is_err(), "{mutation:?}");
                eprintln!(
                    "CPU witness rejects {mutation:?}, width={}, masked={}",
                    op.width, op.masked
                );
            }
            assert!(
                gate_inputs(op, 11, GateInput::Random).kv[..op.kv_elements()]
                    .iter()
                    .all(|v| bf16_to_f32(f32_to_bf16(*v)) != *v)
            );
        }
    }
}

const GATE_CAPACITY: u32 = 8;

/// One gate configuration on the GPU: every live-row count, poisoned
/// outputs, a relaunch and graph replays, each live row against the oracle
/// and the rows past the live count untouched. `in_place` writes the
/// streams over `x`.
fn gate_case(
    session: &Session,
    geometry: &'static Geometry,
    op: EngramGate,
    in_place: bool,
    seed: u64,
    case: GateInput,
    mutation: Option<GateMutation>,
) -> LabResult<usize> {
    let name = gate_cubin(geometry);
    let bytes = match mutation {
        Some(mutation) => &gate_mutants()[&format!("{name}_{mutation:?}")],
        None => cubin(&name),
    };
    let module = session.load_cubin(bytes)?;
    let mut rejected = 0;
    let prepared = Sm120Own::new(geometry)
        .engram_gate(&op, Need::rows(GATE_CAPACITY))
        .map_err(|e| e.to_string())?;
    let rows = GATE_CAPACITY as usize;
    let (row, kv_row) = (op.row_elements(), op.kv_elements());
    let GateInputs { x, kv, q, k, text } = gate_inputs(op, seed, case);
    let x_buffer = upload(session, &u16_bytes(&x))?;
    let kv_buffer = upload(session, &f32_bytes(&kv))?;
    let q_buffer = upload(session, &u16_bytes(&q))?;
    let k_buffer = upload(session, &u16_bytes(&k))?;
    let text_buffer = upload(session, &text)?;
    let live = session.allocate(4)?;
    let out_buffer = session.allocate(rows * row * 2)?;
    let output = if in_place { &x_buffer } else { &out_buffer };
    let bindings = Bindings {
        buffers: BTreeMap::from([
            (1, &x_buffer),
            (2, &kv_buffer),
            (3, &q_buffer),
            (4, &k_buffer),
            (5, &text_buffer),
            (6, &live),
            (7, output),
        ]),
    };
    let buffers = EngramGateBuffers {
        x: view(1, rows * row * 2),
        kv: view(2, rows * kv_row * 4),
        q_weight: view(3, row * 2),
        k_weight: view(4, row * 2),
        text: op.masked.then(|| view(5, rows)),
        live_rows: view(6, 4),
        output: view(if in_place { 1 } else { 7 }, rows * row * 2),
    };
    let gate = single(
        prepared
            .launches(&buffers, View::EMPTY, Counts::rows(GATE_CAPACITY))
            .map_err(|e| e.to_string())?,
    )?;
    let symbol = mutation.map_or_else(|| gate.symbol.clone(), |m| m.symbol(&gate.symbol));
    let function = module.function(&symbol).map_err(|e| e.to_string())?;
    let run = |stream: &Stream| {
        launch(
            &session.context,
            &function,
            stream,
            gate.plan,
            &gate.arguments,
            &bindings,
        )
    };
    let graph = session.capture(run)?;
    let clean = if case == GateInput::NonFinite {
        let clean_kv = gate_inputs(op, seed, GateInput::Random).kv;
        kv_buffer
            .copy_from_bytes(&f32_bytes(&clean_kv), &session.stream)
            .map_err(|e| e.to_string())?;
        live.copy_from_bytes(&GATE_CAPACITY.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        run(&session.stream)?;
        let clean = session.download_bf16(output, rows * row)?;
        kv_buffer
            .copy_from_bytes(&f32_bytes(&kv), &session.stream)
            .map_err(|e| e.to_string())?;
        Some(clean)
    } else {
        None
    };
    for live_rows in 0..=GATE_CAPACITY {
        live.copy_from_bytes(&live_rows.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        let written = live_rows as usize * row;
        let mut replay = if in_place {
            Replay::new(session).state(
                "streams",
                &x_buffer,
                u16_bytes(&x),
                WriteMap::of(DType::Bf16).none()?,
                std::iter::once(0..written * 2).collect(),
            )
        } else {
            // A large finite sentinel distinguishes writes even when the
            // contract calls for NaN. These fixtures' finite outputs are small.
            Replay::new(session).output_with(
                "streams",
                output,
                WriteMap::of(DType::Bf16).prefix(written)?,
                &[0x7f, 0x7f],
            )
        };
        replay.eager(|| run(&session.stream))?;
        replay.relaunch(|| run(&session.stream))?;
        for _ in 0..3 {
            replay.graph(&graph)?;
        }
        let actual = session.download_bf16(output, rows * row)?;
        let mut worst = 0.0_f64;
        for r in 0..live_rows as usize {
            let reference = gate_reference(
                &op,
                GateRow {
                    x: &x[r * row..][..row],
                    kv: &kv[r * kv_row..][..kv_row],
                    q_weight: &q,
                    k_weight: &k,
                    open: !op.masked || text[r] != 0,
                },
            )
            .map_err(|e| e.to_string())?;
            let got = &actual[r * row..][..row];
            if r != 2
                && clean
                    .as_ref()
                    .is_some_and(|clean| got != &clean[r * row..][..row])
            {
                return Err(format!("non-finite kv changed unaffected row {r}"));
            }
            if let Err(i) = check_gate_row(got, &reference) {
                if mutation.is_none() {
                    return Err(format!(
                        "{} {case:?} row {r} element {i}: {:#06x} against {} within {}",
                        gate.symbol, got[i], reference.values[i], reference.bounds[i]
                    ));
                }
                rejected += 1;
            }
            for (i, &bits) in got.iter().enumerate() {
                if reference.bounds[i] > 0.0 {
                    let y = f64::from(kernel_core::codec::bf16_to_f32(bits));
                    worst = worst.max((y - reference.values[i]).abs() / reference.bounds[i]);
                }
            }
        }
        let mut receipt = BTreeMap::from([
            ("op".into(), json!(gate.symbol)),
            ("live_rows".into(), json!(live_rows)),
            ("in_place".into(), json!(in_place)),
            ("input".into(), json!(format!("{case:?}"))),
            ("mutation".into(), json!(format!("{mutation:?}"))),
            ("rejected_rows".into(), json!(rejected)),
            ("max_bound_use".into(), json!(worst)),
            ("graph_replays".into(), json!(3)),
        ]);
        replay.record(&mut receipt);
        session.receipt("deepseek41-engram-gate", receipt)?;
    }
    Ok(rejected)
}

#[test]
fn engram_gate_meets_its_bound_on_every_live_row_count() {
    let result = kernel_harness::gpu_test(64 << 20, |session| {
        let mut index = 0;
        for geometry in gate_geometries() {
            for op in gates(geometry) {
                for in_place in [false, true] {
                    for case in [
                        GateInput::Random,
                        GateInput::Precision,
                        GateInput::NonFinite,
                    ] {
                        assert_eq!(
                            gate_case(session, geometry, op, in_place, 11 + index * 7, case, None)?,
                            0
                        );
                    }
                }
                index += 1;
            }
        }
        Ok(())
    });
    result.unwrap();
}

#[test]
fn engram_gate_rejects_device_precision_mutants() {
    let result = kernel_harness::gpu_test(64 << 20, |session| {
        for geometry in gate_geometries() {
            for op in gates(geometry) {
                for mutation in GateMutation::ALL {
                    let rejected = gate_case(
                        session,
                        geometry,
                        op,
                        false,
                        11,
                        GateInput::Precision,
                        Some(mutation),
                    )?;
                    if rejected == 0 {
                        return Err(format!("device precision mutant {mutation:?} escaped"));
                    }
                }
            }
        }
        Ok(())
    });
    result.unwrap();
}

/// Every position of the domain, for both tables, bit for bit against the
/// oracle (a `Range` launch filling a position-indexed table).
#[test]
fn rope_rows_are_correctly_rounded_at_every_position() {
    let result = kernel_harness::gpu_test(512 << 20, |session| {
        let module = session.load_cubin(cubin(ROPE_CUBIN))?;
        let rows = MAX_POSITIONS;
        let output = session.allocate(RopeCosSin::output_bytes(rows) as usize)?;
        let bindings = Bindings {
            buffers: BTreeMap::from([(1, &output)]),
        };
        let positions: Vec<u32> = (0..rows).collect();
        for table in RopeTable::ALL {
            let op = RopeCosSin {
                table,
                rows: RopeRows::Range,
            };
            let prepared = Sm120Own::new(&FLASH)
                .rope_cos_sin(&op, Need::rows(rows))
                .map_err(|e| e.to_string())?;
            let buffers = RopeCosSinBuffers {
                positions: None,
                live_rows: None,
                first: 0,
                output: view(1, RopeCosSin::output_bytes(rows) as usize),
            };
            let fill = single(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .map_err(|e| e.to_string())?,
            )?;
            let function = module.function(&fill.symbol).map_err(|e| e.to_string())?;
            let mut replay = Replay::new(session).output(
                "cos_sin",
                &output,
                WriteMap::of(DType::F32).prefix(rows as usize * PAIRS * 2)?,
            );
            let run = || {
                launch(
                    &session.context,
                    &function,
                    &session.stream,
                    fill.plan,
                    &fill.arguments,
                    &bindings,
                )
            };
            replay.eager(run)?;
            replay.relaunch(run)?;
            let actual = session.download_f32(&output, rows as usize * PAIRS * 2)?;
            let expected = reference(table, &positions).map_err(|e| e.to_string())?;
            if let Some(i) = actual
                .iter()
                .zip(&expected)
                .position(|(a, e)| a.to_bits() != e.to_bits())
            {
                return Err(format!(
                    "{table:?} position {} pair {}: {} against {}",
                    i / (PAIRS * 2),
                    i / 2 % PAIRS,
                    actual[i],
                    expected[i]
                ));
            }
            let mut receipt = BTreeMap::from([
                ("op".into(), json!(fill.symbol)),
                ("positions".into(), json!(rows)),
                ("bitwise".into(), json!(true)),
            ]);
            replay.record(&mut receipt);
            session.receipt("deepseek41-rope", receipt)?;
        }
        Ok(())
    });
    result.unwrap();
}

/// A step's rows by position: live-row counts, poisoned rows past the live
/// count, graph replays, and a range launch from a nonzero first position.
#[test]
fn rope_rows_by_position_write_only_live_rows() {
    let result = kernel_harness::gpu_test(16 << 20, |session| {
        let module = session.load_cubin(cubin(ROPE_CUBIN))?;
        let capacity = 37_u32;
        // Spread positions, the last row at position 0.
        let mut positions: Vec<u32> = (0..capacity)
            .map(|r| (r * 30_011 + 7) % MAX_POSITIONS)
            .collect();
        positions[capacity as usize - 1] = 0;
        let positions_buffer = upload(session, &u32_bytes(&positions))?;
        let live = session.allocate(4)?;
        let output = session.allocate(RopeCosSin::output_bytes(capacity) as usize)?;
        let bindings = Bindings {
            buffers: BTreeMap::from([(1, &positions_buffer), (2, &live), (3, &output)]),
        };
        for table in RopeTable::ALL {
            let prepared = Sm120Own::new(&FLASH)
                .rope_cos_sin(
                    &RopeCosSin {
                        table,
                        rows: RopeRows::Positions,
                    },
                    Need::rows(capacity),
                )
                .map_err(|e| e.to_string())?;
            let step = single(
                prepared
                    .launches(
                        &RopeCosSinBuffers {
                            positions: Some(view(1, capacity as usize * 4)),
                            live_rows: Some(view(2, 4)),
                            first: 0,
                            output: view(3, RopeCosSin::output_bytes(capacity) as usize),
                        },
                        View::EMPTY,
                        Counts::rows(capacity),
                    )
                    .map_err(|e| e.to_string())?,
            )?;
            let function = module.function(&step.symbol).map_err(|e| e.to_string())?;
            let run = |stream: &Stream| {
                launch(
                    &session.context,
                    &function,
                    stream,
                    step.plan,
                    &step.arguments,
                    &bindings,
                )
            };
            let graph = session.capture(run)?;
            for live_rows in [1, 9, capacity] {
                live.copy_from_bytes(&live_rows.to_le_bytes(), &session.stream)
                    .map_err(|e| e.to_string())?;
                let written = live_rows as usize * PAIRS * 2;
                let mut replay = Replay::new(session).output(
                    "cos_sin",
                    &output,
                    WriteMap::of(DType::F32).prefix(written)?,
                );
                replay.eager(|| run(&session.stream))?;
                for _ in 0..3 {
                    replay.graph(&graph)?;
                }
                let actual = session.download_f32(&output, written)?;
                let expected = reference(table, &positions[..live_rows as usize])
                    .map_err(|e| e.to_string())?;
                if actual
                    .iter()
                    .zip(&expected)
                    .any(|(a, e)| a.to_bits() != e.to_bits())
                {
                    return Err(format!("{table:?} rows by position differ from the oracle"));
                }
                let mut receipt = BTreeMap::from([
                    ("op".into(), json!(step.symbol)),
                    ("live_rows".into(), json!(live_rows)),
                    ("graph_replays".into(), json!(3)),
                ]);
                replay.record(&mut receipt);
                session.receipt("deepseek41-rope", receipt)?;
            }
        }
        // Positions past the domain write NaN; the last position in it does not.
        let edge = [MAX_POSITIONS - 1, MAX_POSITIONS, u32::MAX];
        positions_buffer
            .copy_from_bytes(&u32_bytes(&edge), &session.stream)
            .map_err(|e| e.to_string())?;
        live.copy_from_bytes(&3_u32.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        let prepared = Sm120Own::new(&FLASH)
            .rope_cos_sin(
                &RopeCosSin {
                    table: RopeTable::Window,
                    rows: RopeRows::Positions,
                },
                Need::rows(capacity),
            )
            .map_err(|e| e.to_string())?;
        let step = single(
            prepared
                .launches(
                    &RopeCosSinBuffers {
                        positions: Some(view(1, capacity as usize * 4)),
                        live_rows: Some(view(2, 4)),
                        first: 0,
                        output: view(3, RopeCosSin::output_bytes(capacity) as usize),
                    },
                    View::EMPTY,
                    Counts::rows(capacity),
                )
                .map_err(|e| e.to_string())?,
        )?;
        let function = module.function(&step.symbol).map_err(|e| e.to_string())?;
        launch(
            &session.context,
            &function,
            &session.stream,
            step.plan,
            &step.arguments,
            &bindings,
        )?;
        let actual = session.download_f32(&output, 3 * PAIRS * 2)?;
        let inside = reference(RopeTable::Window, &edge[..1]).map_err(|e| e.to_string())?;
        if actual[..PAIRS * 2]
            .iter()
            .zip(&inside)
            .any(|(a, e)| a.to_bits() != e.to_bits())
            || actual[PAIRS * 2..].iter().any(|v| !v.is_nan())
        {
            return Err("RoPE rows at the domain's edge are not the oracle's, then NaN".into());
        }

        // A range from a nonzero first position equals those rows.
        let prepared = Sm120Own::new(&FLASH)
            .rope_cos_sin(
                &RopeCosSin {
                    table: RopeTable::Compressed,
                    rows: RopeRows::Range,
                },
                Need::rows(capacity),
            )
            .map_err(|e| e.to_string())?;
        let first = MAX_POSITIONS - capacity;
        let range = single(
            prepared
                .launches(
                    &RopeCosSinBuffers {
                        positions: None,
                        live_rows: None,
                        first,
                        output: view(3, RopeCosSin::output_bytes(capacity) as usize),
                    },
                    View::EMPTY,
                    Counts::rows(capacity),
                )
                .map_err(|e| e.to_string())?,
        )?;
        let function = module.function(&range.symbol).map_err(|e| e.to_string())?;
        launch(
            &session.context,
            &function,
            &session.stream,
            range.plan,
            &range.arguments,
            &bindings,
        )?;
        let actual = session.download_f32(&output, capacity as usize * PAIRS * 2)?;
        let expected = reference(
            RopeTable::Compressed,
            &(first..MAX_POSITIONS).collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
        if actual
            .iter()
            .zip(&expected)
            .any(|(a, e)| a.to_bits() != e.to_bits())
        {
            return Err("RoPE range rows from a nonzero first position differ".into());
        }
        Ok(())
    });
    result.unwrap();
}
