//! The model's own ops on SM120: the own-op cubin rebuilt against the
//! bundle's `kernels.lock.d`, prepared as the bundle prepares them, checked
//! against the contract's oracles with poisoned outputs, relaunches and
//! graph replays.

use crate::own_cubin::cubin_spec;
use crate::own_entries::{CUBIN, FAMILY};
use crate::own_ops::Sm120Own;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_core::backend::{Counts, Launch, Need, View};
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::DType;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::poison::{Replay, WriteMap};
use kernel_harness::{LabResult, Session};
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

/// The own-op cubin, rebuilt from the bundle's sources and verified
/// against its lock family (an ordinary build: a mismatch fails).
fn cubin() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle");
        let bundle = std::fs::canonicalize(bundle).expect("bundle directory");
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("deepseek41-own");
        std::fs::create_dir_all(&out).expect("cubin output directory");
        let lock = LockDir::new(bundle.join("kernels.lock.d")).family(FAMILY, [CUBIN]);
        let built = kernel_build::build_in_families(
            &Toolchain::detect().expect("CUDA 13.2 toolchain"),
            &[cubin_spec(&bundle)],
            &out,
            &lock,
            BuildOptions {
                lock_update: false,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
            None,
        )
        .expect("the own-op cubin matches its lock");
        std::fs::read(&built[0].path).expect("built cubin")
    })
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
fn own_cubin_matches_its_lock_and_holds_every_symbol() {
    let bytes = cubin();
    assert!(!bytes.is_empty());
    let own = Sm120Own::new();
    let gate = own
        .engram_gate(&EngramGate::FLASH, Need::rows(64))
        .expect("prepare the flash gate");
    let rope = own
        .rope_cos_sin(
            &RopeCosSin {
                table: RopeTable::Compressed,
                rows: RopeRows::Positions,
            },
            Need::rows(64),
        )
        .expect("prepare RoPE rows");
    for (symbol, _) in gate.info().symbols.iter().chain(&rope.info().symbols) {
        let needle = symbol.as_bytes();
        assert!(
            bytes.windows(needle.len()).any(|w| w == needle),
            "{symbol} is not in the cubin"
        );
    }
    let other = EngramGate {
        width: 4096,
        ..EngramGate::FLASH
    };
    assert!(own.engram_gate(&other, Need::rows(8)).is_err());
}

const GATE_CAPACITY: u32 = 8;

/// One gate configuration on the GPU: every live-row count, poisoned
/// outputs, a relaunch and graph replays, each live row against the oracle
/// and the rows past the live count untouched. `in_place` writes the
/// streams over `x`.
fn gate_case(session: &Session, op: EngramGate, in_place: bool, seed: u64) -> LabResult<()> {
    let module = session.load_cubin(cubin())?;
    let prepared = Sm120Own::new()
        .engram_gate(&op, Need::rows(GATE_CAPACITY))
        .map_err(|e| e.to_string())?;
    let rows = GATE_CAPACITY as usize;
    let (row, kv_row) = (op.row_elements(), op.kv_elements());
    let x = bf16_values(rows * row, seed, 8.0);
    let mut kv = bf16_values(rows * kv_row, seed + 1, 1.0);
    // Row 3's key equals its stream: a gate near 1. Row 6's key is the
    // stream negated: a gate near 0.
    for s in 0..op.streams as usize {
        let n = op.width as usize;
        for i in 0..n {
            kv[3 * kv_row + s * n + i] = x[3 * row + s * n + i];
            kv[6 * kv_row + s * n + i] = x[6 * row + s * n + i] ^ 0x8000;
        }
    }
    let q = bf16_values(row, seed + 2, 1.0);
    let k = bf16_values(row, seed + 3, 1.0);
    let text: Vec<u8> = (0..rows).map(|r| u8::from(r % 4 != 2)).collect();
    let x_buffer = upload(session, &u16_bytes(&x))?;
    let kv_buffer = upload(session, &u16_bytes(&kv))?;
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
        kv: view(2, rows * kv_row * 2),
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
    let function = module.function(&gate.symbol).map_err(|e| e.to_string())?;
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
    for live_rows in [1, 5, GATE_CAPACITY] {
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
            Replay::new(session).output(
                "streams",
                output,
                WriteMap::of(DType::Bf16).prefix(written)?,
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
            check_gate_row(got, &reference).map_err(|i| {
                format!(
                    "{} row {r} element {i}: {:#06x} against {} within {}",
                    gate.symbol, got[i], reference.values[i], reference.bounds[i]
                )
            })?;
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
            ("max_bound_use".into(), json!(worst)),
            ("graph_replays".into(), json!(3)),
        ]);
        replay.record(&mut receipt);
        session.receipt("deepseek41-engram-gate", receipt)?;
    }
    Ok(())
}

#[test]
fn engram_gate_meets_its_bound_on_every_live_row_count() {
    let result = kernel_harness::gpu_test(64 << 20, |session| {
        for (index, op) in crate::own_entries::gates().into_iter().enumerate() {
            for in_place in [false, true] {
                gate_case(session, op, in_place, 11 + index as u64 * 7)?;
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
        let module = session.load_cubin(cubin())?;
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
            let prepared = Sm120Own::new()
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
        let module = session.load_cubin(cubin())?;
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
            let prepared = Sm120Own::new()
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
        let prepared = Sm120Own::new()
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
        let prepared = Sm120Own::new()
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
