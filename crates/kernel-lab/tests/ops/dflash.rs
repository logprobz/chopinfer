use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, RuntimeCounts};
use kernel_core::oracle::{NormArithmetic, NormOffset, NormScale, RmsNorm};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{Replay, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::row::RowInstantiation;
use runtime_cuda::{DeviceBuffer, Module};
use std::collections::BTreeMap;

/// DFlash's Q/K RMSNorm with eps 1e-6: one BF16 weight vector for every
/// head, F32 arithmetic and one BF16 rounding of the output.
const QK_NORM: RmsNorm = RmsNorm {
    eps: 1e-6,
    offset: NormOffset::Plain,
    arithmetic: NormArithmetic::F32,
    scale: NormScale::MultiplyReciprocal,
    before_weight: None,
    output: Some(DType::Bf16),
};

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/row_engine.cubin"));

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(1))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|error| error.to_string())?;
    Ok(buffer)
}

fn bits_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

fn read(session: &Session, buffer: &DeviceBuffer, bytes: usize) -> LabResult<Vec<u8>> {
    let mut output = vec![0; bytes];
    buffer
        .copy_to_bytes(&mut output, &session.stream)
        .map_err(|error| error.to_string())?;
    Ok(output)
}

fn run(
    session: &Session,
    module: &Module,
    entry: &RowInstantiation,
    capacity: u32,
    live: u32,
    buffers: &BTreeMap<&str, DeviceBuffer>,
    scalars: &BTreeMap<&str, u32>,
) -> LabResult<()> {
    let bindings = buffers
        .iter()
        .enumerate()
        .map(|(index, (_, buffer))| (index as u64 + 1, buffer))
        .collect::<BTreeMap<_, _>>();
    let handles = buffers
        .keys()
        .enumerate()
        .map(|(index, name)| (*name, index as u64 + 1))
        .collect::<BTreeMap<_, _>>();
    let values = entry
        .params
        .iter()
        .map(|param| match param {
            EntryParam::DevicePointer { name, .. } => handles
                .get(name.as_str())
                .copied()
                .map(Argument::DeviceBuffer)
                .ok_or_else(|| format!("missing DFlash buffer {name}")),
            EntryParam::U32 { name } => scalars
                .get(name.as_str())
                .copied()
                .map(Argument::U32)
                .ok_or_else(|| format!("missing DFlash scalar {name}")),
            other => Err(format!("unexpected DFlash parameter {other:?}")),
        })
        .collect::<LabResult<Vec<_>>>()?;
    let args = entry
        .pack_arguments(values)
        .map_err(|error| error.to_string())?;
    let plan = entry
        .plan(
            capacity,
            RuntimeCounts {
                m: live,
                batch: live,
                context: 0,
                routed_rows: 0,
                split_k: 1,
            },
        )
        .map_err(|error| error.to_string())?;
    let function = module
        .function(&entry.symbol)
        .map_err(|error| error.to_string())?;
    launch::launch(
        &session.context,
        &function,
        &session.stream,
        plan,
        &args,
        &Bindings { buffers: bindings },
    )
}

fn catalog() -> Vec<RowInstantiation> {
    kernels_sm120::row::catalog(kernel_lab::catalogs())
}

#[test]
fn draft_row_entries_present_cpu() {
    let entries = catalog();
    for name in ["q_norm", "q_rope_pos32", "argmax_partial", "argmax_final"] {
        entry(&entries, name);
    }
    for name in ["argmax_partial", "argmax_final"] {
        assert!(
            entries
                .iter()
                .any(|item| item.symbol == format!("chop_row_dflash_tp2_{name}_v1"))
        );
    }
}

fn entry<'a>(entries: &'a [RowInstantiation], name: &str) -> &'a RowInstantiation {
    entries
        .iter()
        .find(|item| item.symbol == format!("chop_row_dflash_{name}_v1"))
        .expect("DFlash row entry")
}

#[test]
fn dflash_norm_rope_and_tp4_argmax_gpu() {
    gpu_test(64 * 1024 * 1024, |session| {
        let module = session
            .context
            .load_module(CUBIN)
            .map_err(|error| error.to_string())?;
        let entries = catalog();
        let values = (0..2 * 16 * 128)
            .map(|index| f32_to_bf16((index % 17 + 1) as f32 / 16.0))
            .collect::<Vec<_>>();
        let weights = (0..128)
            .map(|index| f32_to_bf16(0.5 + index as f32 / 256.0))
            .collect::<Vec<_>>();
        let mut strided = vec![f32_to_bf16(0.0); 2 * 2560];
        for token in 0..2 {
            strided[token * 2560..token * 2560 + 2048]
                .copy_from_slice(&values[token * 2048..(token + 1) * 2048]);
        }
        let mut buffers = BTreeMap::new();
        buffers.insert("input", upload(session, &bits_bytes(&strided))?);
        buffers.insert("weight", upload(session, &bits_bytes(&weights))?);
        buffers.insert("output", upload(session, &vec![0; values.len() * 2])?);
        buffers.insert("live_tokens", upload(session, &2_u32.to_ne_bytes())?);
        let mut scalars = BTreeMap::new();
        scalars.insert("eps_bits", 1e-6_f32.to_bits());
        scalars.insert("input_stride", 2560);
        scalars.insert("output_stride", 2048);
        let mut misses = Vec::new();
        let checked = (|| -> LabResult<()> {
            let norm = entry(&entries, "q_norm");
            // row norm/RoPE stores all live heads; input and output are distinct.
            let mut norm_replay = Replay::new(session).output(
                "Q norm",
                &buffers["output"],
                WriteMap::of(DType::Bf16).prefix(values.len())?,
            );
            norm_replay.eager(|| run(session, &module, norm, 2, 2, &buffers, &scalars))?;
            let norm_eager = read(session, &buffers["output"], values.len() * 2)?;
            let norm_graph =
                session.capture(|_| run(session, &module, norm, 2, 2, &buffers, &scalars))?;
            norm_replay.graph(&norm_graph)?;
            if read(session, &buffers["output"], values.len() * 2)? != norm_eager {
                return Err("DFlash Q norm graph replay changed output bits".into());
            }
            let normalized = norm_eager
                .chunks_exact(2)
                .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]))
                .collect::<Vec<_>>();
            let expected = QK_NORM
                .apply_bf16(&values, &weights, 128)
                .map_err(|error| error.to_string())?;
            for (actual, expected) in normalized.iter().zip(expected) {
                if (bf16_to_f32(*actual) - bf16_to_f32(expected)).abs() > 0.02 {
                    return Err("DFlash Q norm differs from BF16 oracle".into());
                }
            }
            let mut fields = BTreeMap::new();
            norm_replay.record(&mut fields);
            session.receipt("draft-row-norm-replay", fields)?;
            Ok(())
        })();
        if let Err(error) = checked {
            eprintln!("norm: {error}");
            misses.push(error);
        }
        let positions_bytes = [3_i32, 4_i32]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        buffers.insert("positions", upload(session, &positions_bytes)?);
        let checked = (|| -> LabResult<()> {
            let rope = entry(&entries, "q_rope_pos32");
            let mut rope_replay = Replay::new(session).output(
                "Q RoPE",
                &buffers["output"],
                WriteMap::of(DType::Bf16).prefix(values.len())?,
            );
            rope_replay.eager(|| run(session, &module, rope, 2, 2, &buffers, &scalars))?;
            let rope_eager = read(session, &buffers["output"], values.len() * 2)?;
            let rope_graph =
                session.capture(|_| run(session, &module, rope, 2, 2, &buffers, &scalars))?;
            rope_replay.graph(&rope_graph)?;
            if read(session, &buffers["output"], values.len() * 2)? != rope_eager {
                return Err("DFlash partial RoPE graph replay changed output bits".into());
            }
            let rotated = rope_eager
                .chunks_exact(2)
                .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]))
                .collect::<Vec<_>>();
            let mut positions = vec![3_usize; 16];
            positions.extend([4; 16]);
            let expected = crate::row::hf_rope_bf16(
                &values
                    .iter()
                    .map(|&x| f64::from(bf16_to_f32(x)))
                    .collect::<Vec<_>>(),
                &positions,
                128,
                1e4,
            )?;
            for (actual, expected) in rotated.iter().zip(expected) {
                if *actual != kernel_core::codec::f32_to_bf16(expected as f32) {
                    return Err("DFlash partial RoPE differs from oracle".into());
                }
            }
            let mut fields = BTreeMap::new();
            rope_replay.record(&mut fields);
            session.receipt("draft-row-rope-replay", fields)?;
            Ok(())
        })();
        if let Err(error) = checked {
            eprintln!("RoPE: {error}");
            misses.push(error);
        }
        let checked = (|| -> LabResult<()> {
            let low = f32_to_bf16(-1.0);
            let high = f32_to_bf16(9.0);
            let mut scores = [0_f32; 4];
            let mut ids = [0_i32; 4];
            for rank in 0..4 {
                let mut logits = vec![low; 38144];
                logits[if rank == 0 { 11 } else { 7 }] = high;
                let mut local = BTreeMap::new();
                local.insert("logits", upload(session, &bits_bytes(&logits))?);
                local.insert("scores_out", upload(session, &[0; 4])?);
                local.insert("ids_out", upload(session, &[0; 4])?);
                local.insert("live_rows", upload(session, &1_u32.to_ne_bytes())?);
                local.insert("live_width", upload(session, &38144_u32.to_ne_bytes())?);
                scalars.insert("vocab_offset", rank * 38144);
                let mut replay = Replay::new(session)
                    .output(
                        "scores",
                        &local["scores_out"],
                        WriteMap::of(DType::F32).prefix(1)?,
                    )
                    .output(
                        "IDs",
                        &local["ids_out"],
                        WriteMap::of(DType::I32).prefix(1)?,
                    );
                replay.eager(|| {
                    run(
                        session,
                        &module,
                        entry(&entries, "argmax_partial"),
                        1,
                        1,
                        &local,
                        &scalars,
                    )
                })?;
                scores[rank as usize] =
                    f32::from_ne_bytes(read(session, &local["scores_out"], 4)?.try_into().unwrap());
                ids[rank as usize] =
                    i32::from_ne_bytes(read(session, &local["ids_out"], 4)?.try_into().unwrap());
            }
            let mut gathered = BTreeMap::new();
            gathered.insert(
                "scores",
                upload(
                    session,
                    &scores
                        .iter()
                        .flat_map(|x| x.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?,
            );
            gathered.insert(
                "ids",
                upload(
                    session,
                    &ids.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
                )?,
            );
            gathered.insert("output", upload(session, &[0; 4])?);
            gathered.insert("live_rows", upload(session, &1_u32.to_ne_bytes())?);
            let mut replay = Replay::new(session).output(
                "selected ID",
                &gathered["output"],
                WriteMap::of(DType::I32).prefix(1)?,
            );
            replay.eager(|| {
                run(
                    session,
                    &module,
                    entry(&entries, "argmax_final"),
                    1,
                    1,
                    &gathered,
                    &scalars,
                )
            })?;
            let result =
                i32::from_ne_bytes(read(session, &gathered["output"], 4)?.try_into().unwrap());
            if result != 11 {
                return Err(format!("DFlash TP4 argmax returned {result}, expected 11"));
            }
            Ok(())
        })();
        if let Err(error) = checked {
            eprintln!("argmax: {error}");
            misses.push(error);
        }
        if misses.is_empty() {
            Ok(())
        } else {
            Err(misses.join("; "))
        }
    })
    .expect("DFlash row kernels match CPU oracles on GPU 1");
}

/// The vocabulary-sharded argmax at TP2: every rank's partial over
/// its shard, then the final over all ranks' pairs picks the global best,
/// the lowest ID on a tie across ranks. NaN ranks below every value: a
/// rank of only NaN loses to any other, and a row of only NaN on every
/// rank yields global ID 0, each rank's partial its lowest ID.
#[test]
fn dflash_argmax_at_tp2_gpu() {
    gpu_test(64 * 1024 * 1024, |session| {
        let module = session
            .context
            .load_module(CUBIN)
            .map_err(|error| error.to_string())?;
        let entries = catalog();
        let low = f32_to_bf16(-1.0);
        let high = f32_to_bf16(9.0);
        {
            let ranks = 2_u32;
            let shard = 152_576 / ranks;
            let named = |name: &str| {
                entries
                    .iter()
                    .find(|item| item.symbol == format!("chop_row_dflash_tp{ranks}_{name}_v1"))
                    .expect("DFlash rank argmax entry")
            };
            // Four rows: row 0's best is on the last rank, row 1 ties between
            // rank 0 and the last rank, so the lower global ID wins; row 2 is
            // only NaN on every rank; row 3 is only NaN on rank 0, and NaN
            // beside values on the last rank.
            let rows = 4_usize;
            let nan = 0xffc1_u16;
            let mut scores = vec![0_f32; rows * ranks as usize];
            let mut ids = vec![0_i32; rows * ranks as usize];
            for rank in 0..ranks {
                let width = shard as usize;
                let mut logits = vec![low; rows * width];
                logits[5] = f32_to_bf16(1.0);
                if rank + 1 == ranks {
                    logits[3] = high;
                }
                if rank == 0 || rank + 1 == ranks {
                    logits[width + 9] = high;
                }
                logits[2 * width..3 * width].fill(nan);
                if rank == 0 {
                    logits[3 * width..].fill(nan);
                } else {
                    logits[3 * width + 2] = nan;
                    logits[3 * width + 4] = high;
                }
                let mut local = BTreeMap::new();
                local.insert("logits", upload(session, &bits_bytes(&logits))?);
                local.insert("scores_out", upload(session, &[0; 16])?);
                local.insert("ids_out", upload(session, &[0; 16])?);
                local.insert("live_rows", upload(session, &4_u32.to_ne_bytes())?);
                local.insert("live_width", upload(session, &shard.to_ne_bytes())?);
                let mut scalars = BTreeMap::new();
                scalars.insert("vocab_offset", rank * shard);
                let mut replay = Replay::new(session)
                    .output(
                        "scores",
                        &local["scores_out"],
                        WriteMap::of(DType::F32).prefix(rows)?,
                    )
                    .output(
                        "IDs",
                        &local["ids_out"],
                        WriteMap::of(DType::I32).prefix(rows)?,
                    );
                replay.eager(|| {
                    run(
                        session,
                        &module,
                        named("argmax_partial"),
                        4,
                        4,
                        &local,
                        &scalars,
                    )
                })?;
                let row_scores = read(session, &local["scores_out"], 16)?;
                let row_ids = read(session, &local["ids_out"], 16)?;
                for row in 0..rows {
                    let at = row * ranks as usize + rank as usize;
                    scores[at] =
                        f32::from_ne_bytes(row_scores[row * 4..row * 4 + 4].try_into().unwrap());
                    ids[at] = i32::from_ne_bytes(row_ids[row * 4..row * 4 + 4].try_into().unwrap());
                }
                // A row of only NaN: the shard's lowest ID, with that
                // logit's bits.
                let (score, id) = (
                    scores[2 * ranks as usize + rank as usize],
                    ids[2 * ranks as usize + rank as usize],
                );
                if (score.to_bits(), id) != (u32::from(nan) << 16, (rank * shard) as i32) {
                    return Err(format!(
                        "DFlash TP{ranks} rank {rank}'s NaN row gave ({:#x}, {id})",
                        score.to_bits()
                    ));
                }
            }
            let mut gathered = BTreeMap::new();
            gathered.insert(
                "scores",
                upload(
                    session,
                    &scores
                        .iter()
                        .flat_map(|x| x.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?,
            );
            gathered.insert(
                "ids",
                upload(
                    session,
                    &ids.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
                )?,
            );
            gathered.insert("output", upload(session, &[0; 16])?);
            gathered.insert("live_rows", upload(session, &4_u32.to_ne_bytes())?);
            let mut replay = Replay::new(session).output(
                "selected ID",
                &gathered["output"],
                WriteMap::of(DType::I32).prefix(rows)?,
            );
            replay.eager(|| {
                run(
                    session,
                    &module,
                    named("argmax_final"),
                    4,
                    4,
                    &gathered,
                    &BTreeMap::new(),
                )
            })?;
            let output = read(session, &gathered["output"], 16)?;
            let result = [0, 1, 2, 3]
                .map(|row| i32::from_ne_bytes(output[row * 4..row * 4 + 4].try_into().unwrap()));
            let last = ((ranks - 1) * shard) as i32;
            let expected = [last + 3, 9, 0, last + 4];
            if result != expected {
                return Err(format!(
                    "DFlash TP{ranks} argmax returned {result:?}, expected {expected:?}"
                ));
            }
        }
        Ok(())
    })
    .expect("DFlash TP2 argmax on GPU");
}
