//! The linear-attention invariance adapter, at each entry's own profile:
//! the causal convolution and the chunked delta rule (prepare then scan) as
//! prefill operations over a state slot, their verify entries as exact
//! verify rows, and the gated norm as row-local rows. A chunk case carries
//! one sequence's state across the chunks, as the bundle's prefill does, and
//! also compares the final state. Prefill batch units are sequences beside
//! other sequences. The anchor case writes each sequence's state at its last
//! whole `CHUNK` into a spare entry, which must equal the slot state a
//! prefill of just those rows leaves.
//!
//! Verify entries: a rows unit is one row of one verify sequence, its output
//! followed by its checkpoint's state. The R-row launch runs the rows of one
//! sequence from a seeded slot state; alone, the same rows run through R
//! one-row decode launches (`conv_decode`, `delta_decode` of the profile)
//! that carry the slot state, each unit the row's output followed by the
//! slot's state after it. Batch units of a verify entry are whole sequences
//! (`Adapter::batch`): each sequence's rows and checkpoints, beside other
//! sequences with other rows and other slot states.
//!
//! Every compared launch starts with poisoned outputs, a poisoned delta-rule
//! workspace and poisoned write-only state entries (verify checkpoints,
//! anchor entries), and is checked afterwards (`written_output`,
//! `written_state`): every live output element and every write-only entry
//! the contract writes no longer holds the poison, every slot the launch
//! reads and rewrites changed, and every other byte kept its image. Input
//! state is never poisoned.

use super::*;
use crate::invariance::{Adapter, Compared, Units};
use kernel_core::contract::RecurrentPhase;
use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::linear::{CHUNK, Stage};
use model_qwen38_contract::sm120::config::EXACT_VERIFY_ROWS;

/// Per-rank linear geometry of `profile`, as `geometry` derives the
/// environment's.
struct Shape {
    profile: Profile,
    channels: usize,
    value_heads: usize,
    norm_width: usize,
    qkv_stride: usize,
    gate_stride: usize,
    z_stride: usize,
    delta_state: usize,
}

impl Shape {
    fn of(entry: &Declared) -> LabResult<Self> {
        let profile = entry.profile.ok_or("a linear entry has a profile")?;
        let shape = model_qwen38_contract::sm120::config::profile_shape(profile)
            .ok_or_else(|| format!("{}: not a Qwen3.8 profile", entry.symbol))?;
        let value_heads = shape.linear_value_heads() as usize;
        let channels = shape.linear_qkv() as usize;
        let norm_width = shape.linear_value() as usize;
        Ok(Self {
            profile,
            channels,
            value_heads,
            norm_width,
            qkv_stride: channels + 64,
            gate_stride: (value_heads + 4).next_multiple_of(8),
            z_stride: norm_width + 256,
            delta_state: value_heads * HEAD,
        })
    }

    fn symbol(&self, name: &str) -> String {
        format!("chop_linear_{}_{name}_v1", self.profile.tag())
    }

    fn module(&self, session: &Session) -> LabResult<Module> {
        let name = match self.profile {
            model_qwen38_contract::sm120::config::PROFILE => "linear_qwen38_flash_sm120".to_owned(),
            other => format!("linear_{}_sm120", other.tag()),
        };
        session.module(&name)
    }

    /// Output elements per row, bytes of one state entry and its element
    /// type: the convolution's (`conv`) or the delta rule's.
    fn recurrent(&self, conv: bool) -> (usize, usize, DType) {
        if conv {
            (self.channels, HISTORY * self.channels * 2, DType::Bf16)
        } else {
            (self.norm_width, self.delta_state * 4, DType::F32)
        }
    }
}

/// Which operation an entry belongs to.
#[derive(Clone, Copy, PartialEq)]
enum Op {
    /// The convolution's prefill.
    Conv,
    /// The delta rule's prefill (prepare, scan).
    Delta,
    Norm,
    /// A verify entry of the convolution (`conv`) or the delta rule, with
    /// its checkpoints per sequence.
    Verify {
        conv: bool,
        max_rows: usize,
    },
}

fn op(entry: &Declared) -> LabResult<Op> {
    let item = find(&entry.symbol);
    Ok(match (item.query, item.stage) {
        (
            RecurrentQuery::CausalConv1d {
                phase: RecurrentPhase::Verify { max_rows },
                ..
            },
            _,
        ) => Op::Verify {
            conv: true,
            max_rows: max_rows as usize,
        },
        (
            RecurrentQuery::GatedDeltaRule {
                phase: RecurrentPhase::Verify { max_rows },
                ..
            },
            _,
        ) => Op::Verify {
            conv: false,
            max_rows: max_rows as usize,
        },
        (
            RecurrentQuery::CausalConv1d {
                phase: RecurrentPhase::Prefill,
                ..
            },
            _,
        ) => Op::Conv,
        (RecurrentQuery::GatedDeltaRule { .. }, Stage::Prepare | Stage::Scan) => Op::Delta,
        (RecurrentQuery::GatedRmsNorm { .. }, _) => Op::Norm,
        _ => {
            return Err(format!(
                "{}: a decode entry, which claims nothing",
                entry.symbol
            ));
        }
    })
}

/// One run's inputs: `rows` rows of every operand, seeded, and a state pool
/// of `pool` entries.
struct Data {
    conv_weight: Vec<u16>,
    qkv: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    conv_pool: Vec<u16>,
    delta_pool: Vec<f32>,
}

impl Data {
    /// Rows seeded one by one, so a sequence's rows do not depend on where
    /// it sits in a launch.
    fn new(s: &Shape, row_seeds: &[u64], pool: usize) -> Self {
        let mut fixed = Rng::new(0x11ea);
        let heads = s.value_heads;
        let a_log = (0..heads)
            .map(|h| {
                let t = h as f64 / (heads - 1) as f64;
                (0.03_f64.ln() + t * (160.0_f64 / 0.03).ln()) as f32
            })
            .collect();
        let rows = |stride: usize, scale: f64, salt: u64| {
            row_seeds
                .iter()
                .flat_map(|&seed| Rng::new(seed ^ salt).bf16(stride, scale))
                .collect::<Vec<_>>()
        };
        Self {
            conv_weight: fixed.bf16(s.channels * 4, 0.1),
            dt_bias: (0..heads)
                .map(|_| (-8.0 + 10.5 * fixed.uniform()) as f32)
                .collect(),
            a_log,
            conv_pool: fixed.bf16(pool * HISTORY * s.channels, 1.0),
            delta_pool: fixed.f32(pool * s.delta_state, 0.3),
            qkv: rows(s.qkv_stride, 1.0, 1),
            a: rows(s.gate_stride, 1.5, 2),
            b: rows(s.gate_stride, 1.5, 3),
        }
    }

    /// The pool as bytes: the convolution's (`conv`) or the delta rule's.
    fn pool_bytes(&self, conv: bool) -> Vec<u8> {
        if conv {
            bytes_u16(&self.conv_pool)
        } else {
            bytes_f32(&self.delta_pool)
        }
    }
}

/// Rows `rows` of `stride`-element rows.
fn rows_of_data<'a>(values: &'a [u16], stride: usize, rows: &Range<usize>) -> &'a [u16] {
    &values[rows.start * stride..rows.end * stride]
}

/// Binds the convolution's (`conv`) or the delta rule's inputs: rows `rows`
/// of `data`, then its weights and decay parameters.
fn bind_inputs(
    session: &Session,
    s: &Shape,
    conv: bool,
    data: &Data,
    rows: Range<usize>,
    op: &mut Operands,
) -> LabResult<()> {
    if conv {
        op.bf16(
            session,
            "input",
            rows_of_data(&data.qkv, s.qkv_stride, &rows),
        )?;
        op.bf16(session, "weight", &data.conv_weight)?;
        op.scalars.insert("input_stride", s.qkv_stride as u32);
    } else {
        op.bf16(session, "qkv", rows_of_data(&data.qkv, s.qkv_stride, &rows))?;
        op.bf16(session, "a", rows_of_data(&data.a, s.gate_stride, &rows))?;
        op.bf16(session, "b", rows_of_data(&data.b, s.gate_stride, &rows))?;
        op.f32(session, "a_log", &data.a_log)?;
        op.f32(session, "dt_bias", &data.dt_bias)?;
        op.scalars.insert("qkv_stride", s.qkv_stride as u32);
        op.scalars.insert("gate_stride", s.gate_stride as u32);
    }
    Ok(())
}

/// Allocates buffer `name` of `bytes` filled with the poison.
fn poisoned(
    session: &Session,
    op: &mut Operands,
    name: &'static str,
    bytes: usize,
) -> LabResult<()> {
    let buffer = session.allocate(bytes.max(4))?;
    poison::poison(session, &[&buffer])?;
    op.buffers.insert(name, buffer);
    Ok(())
}

/// Checks a BF16 output buffer after a launch that found it poisoned: each
/// element of its first `written` bytes (the live rows) no longer holds the
/// poison and is finite, and every later byte still holds the poison. The
/// finite check also rejects an output computed from poisoned (unwritten)
/// workspace, which both arms of a comparison would share.
pub(crate) fn written_output(label: &str, bytes: &[u8], written: usize) -> Result<(), String> {
    poison::check(
        label,
        bytes,
        &WriteMap::of(DType::Bf16).byte_prefix(written)?,
        Fill::Pattern(&[POISON]),
    )?;
    let not_finite = bytes[..written]
        .chunks_exact(2)
        .position(|e| u16::from_ne_bytes([e[0], e[1]]) & 0x7f80 == 0x7f80);
    match not_finite {
        Some(i) => Err(format!("{label}: element {i} is not finite")),
        None => Ok(()),
    }
}

/// Checks a state pool of `entry_bytes` entries after a launch against its
/// image `before` it: each entry of `written` (write-only, so poisoned
/// before the launch) holds no element that is still poison; each entry of
/// `updated` (a slot the launch reads and rewrites, never poisoned) changed;
/// every other entry kept its image byte for byte.
pub(crate) fn written_state(
    label: &str,
    after: &[u8],
    before: &[u8],
    entry_bytes: usize,
    element: DType,
    written: &[usize],
    updated: &[usize],
) -> Result<(), String> {
    if entry_bytes == 0 || after.len() != before.len() || before.len() % entry_bytes != 0 {
        return Err(format!(
            "{label}: {} bytes after, {} before, in entries of {entry_bytes}",
            after.len(),
            before.len()
        ));
    }
    if let Some(entry) = written
        .iter()
        .chain(updated)
        .find(|&&entry| (entry + 1) * entry_bytes > before.len())
    {
        return Err(format!("{label}: entry {entry} lies outside the pool"));
    }
    let map = WriteMap::of(element).byte_prefix(entry_bytes)?;
    for (entry, (was, is)) in before
        .chunks(entry_bytes)
        .zip(after.chunks(entry_bytes))
        .enumerate()
    {
        if written.contains(&entry) {
            if was.iter().any(|&byte| byte != POISON) {
                return Err(format!(
                    "{label}: write-only entry {entry} was not poisoned before the launch"
                ));
            }
            poison::check(
                &format!("{label} entry {entry}"),
                is,
                &map,
                Fill::Pattern(&[POISON]),
            )?;
        } else if updated.contains(&entry) {
            if was == is {
                return Err(format!(
                    "{label}: slot entry {entry} kept its bytes; the launch did not write it"
                ));
            }
        } else if let Some(at) = was.iter().zip(is).position(|(a, b)| a != b) {
            return Err(format!(
                "{label}: entry {entry} byte {at} changed from {:#04x} to {:#04x}; the launch \
                 writes no state there",
                was[at], is[at]
            ));
        }
    }
    Ok(())
}

/// The prefill of `lengths` sequences (rows laid out in order) over
/// `slots`, starting from their slots, run as `calls` launches of row
/// ranges of one sequence each when given (chunked; ascending from row 0,
/// without anchors), else in one launch; anchors as the bundle sets them.
/// Returns the output and the pool, after checking every launch's writes:
/// anchor entries start poisoned, and the delta workspace is poisoned
/// before every call.
#[allow(clippy::too_many_arguments)]
fn prefill(
    session: &Session,
    s: &Shape,
    which: Op,
    data: &Data,
    lengths: &[usize],
    slot_ids: &[i32],
    anchors: &[i32],
    calls: Option<&[[i32; 2]]>,
) -> LabResult<(Vec<u8>, Vec<u8>)> {
    let conv = match which {
        Op::Conv => true,
        Op::Delta => false,
        _ => return Err("not a prefill operation".into()),
    };
    let module = s.module(session)?;
    let rows = lengths.iter().sum::<usize>();
    let (width, entry_bytes, element) = s.recurrent(conv);
    let mut op = Operands::new();
    bind_inputs(session, s, conv, data, 0..rows, &mut op)?;
    // Anchor entries are written without being read (conv.cuh,
    // delta_prefill.cuh): only by a sequence of at least one whole chunk
    // and a partial last one.
    let mut pool = data.pool_bytes(conv);
    let mut written = Vec::new();
    for (q, &anchor) in anchors.iter().enumerate() {
        if anchor < 0 {
            continue;
        }
        pool[anchor as usize * entry_bytes..][..entry_bytes].fill(POISON);
        let len = lengths[q];
        if len >= CHUNK as usize && len % CHUNK as usize != 0 {
            written.push(anchor as usize);
        }
    }
    if calls.is_some() && !written.is_empty() {
        return Err("chunked prefill calls take no anchors".into());
    }
    op.bytes(session, "state", &pool)?;
    poisoned(session, &mut op, "output", rows * width * 2)?;
    op.i32(session, "slot_ids", slot_ids)?;
    op.i32(session, "initial", &vec![1; lengths.len()])?;
    op.i32(session, "anchor_ids", anchors)?;
    let kernels = if conv {
        vec![Kernel::load(&module, &s.symbol("conv_prefill"))?]
    } else {
        vec![
            Kernel::load(&module, &s.symbol("delta_prefill_prepare"))?,
            Kernel::load(&module, &s.symbol("delta_prefill_scan"))?,
        ]
    };
    if !conv {
        let workspace = session.allocate(kernels[0].workspace_bytes().max(4))?;
        op.buffers.insert("workspace", workspace);
    }
    let whole = [[0, rows as i32]];
    // (cu_seqlens, live rows, live sequences, slots the launch rewrites).
    let launches: Vec<(Vec<i32>, u32, u32, Vec<usize>)> = match calls {
        Some(calls) => calls
            .iter()
            .map(|c| {
                (
                    c.to_vec(),
                    (c[1] - c[0]) as u32,
                    1,
                    vec![slot_ids[0] as usize],
                )
            })
            .collect(),
        None => {
            let cu_seqlens = if lengths.len() == 1 {
                whole[0].to_vec()
            } else {
                cu(lengths)
            };
            let slots = (0..lengths.len())
                .filter(|&q| lengths[q] > 0)
                .map(|q| slot_ids[q] as usize)
                .collect();
            vec![(cu_seqlens, rows as u32, lengths.len() as u32, slots)]
        }
    };
    let mut done = 0;
    for (cu_seqlens, live_rows, live_sequences, slots) in launches {
        if cu_seqlens[0] != done {
            return Err(format!("a prefill call starts at row {}", cu_seqlens[0]));
        }
        done = *cu_seqlens.last().expect("call bounds");
        op.i32(session, "cu_seqlens", &cu_seqlens)?;
        op.u32(session, "live_sequences", live_sequences)?;
        if !conv {
            poison::poison(session, &[&op.buffers["workspace"]])?;
        }
        let before = op.read(session, "state", pool.len())?;
        for kernel in &kernels {
            kernel.launch(
                session,
                &session.stream,
                &op,
                live_rows,
                live_sequences,
                true,
            )?;
        }
        let output = op.read(session, "output", rows * width * 2)?;
        written_output("prefill output", &output, done as usize * width * 2)?;
        let after = op.read(session, "state", pool.len())?;
        written_state(
            "prefill state",
            &after,
            &before,
            entry_bytes,
            element,
            &written,
            &slots,
        )?;
    }
    Ok((
        op.read(session, "output", rows * width * 2)?,
        op.read(session, "state", pool.len())?,
    ))
}

/// One sequence of a verify case: its rows' seeds and the seed of the slot
/// state it starts from.
struct Sequence {
    rows: Vec<u64>,
    state: u64,
}

/// The slot state seeded by `seed`, as pool bytes.
fn seeded_state(s: &Shape, conv: bool, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed ^ 0x57a7_e000);
    if conv {
        bytes_u16(&rng.bf16(HISTORY * s.channels, 1.0))
    } else {
        bytes_f32(&rng.f32(s.delta_state, 0.3))
    }
}

/// One verify launch of `sequences`, each from its own seeded slot state:
/// per sequence, per row, the row's output bytes followed by its
/// checkpoint's state bytes. The slots are pool entries `0..Q`; sequence
/// `q`'s `max_rows` checkpoints follow at `Q + q * max_rows..`. Outputs and
/// checkpoints start poisoned: the launch must write every row's output and
/// the checkpoint of every row, and leave the slots and the checkpoints past
/// a sequence's rows as they were.
fn verify(
    session: &Session,
    s: &Shape,
    entry: &Declared,
    conv: bool,
    max_rows: usize,
    sequences: &[Sequence],
) -> LabResult<Vec<Units>> {
    let module = s.module(session)?;
    let q = sequences.len();
    let lengths = sequences.iter().map(|x| x.rows.len()).collect::<Vec<_>>();
    if lengths.iter().any(|&n| n == 0 || n > max_rows) {
        return Err(format!(
            "verify sequences of {lengths:?} rows, at most {max_rows}"
        ));
    }
    let rows = lengths.iter().sum::<usize>();
    let row_seeds = sequences
        .iter()
        .flat_map(|x| x.rows.iter().copied())
        .collect::<Vec<_>>();
    let data = Data::new(s, &row_seeds, 0);
    let (width, entry_bytes, element) = s.recurrent(conv);
    let mut op = Operands::new();
    bind_inputs(session, s, conv, &data, 0..rows, &mut op)?;
    let mut pool = sequences
        .iter()
        .flat_map(|x| seeded_state(s, conv, x.state))
        .collect::<Vec<_>>();
    pool.resize((q + q * max_rows) * entry_bytes, POISON);
    op.bytes(session, "state", &pool)?;
    poisoned(session, &mut op, "output", rows * width * 2)?;
    let starts = cu(&lengths);
    op.i32(session, "cu_rows", &starts)?;
    op.i32(session, "slot_ids", &(0..q as i32).collect::<Vec<_>>())?;
    op.i32(
        session,
        "checkpoint_ids",
        &(0..q * max_rows)
            .map(|i| (q + i) as i32)
            .collect::<Vec<_>>(),
    )?;
    op.u32(session, "live_sequences", q as u32)?;
    Kernel::load(&module, &entry.symbol)?
        .with_bucket(rows as u32, q as u32)
        .launch(session, &session.stream, &op, rows as u32, q as u32, true)?;
    let output = op.read(session, "output", rows * width * 2)?;
    written_output("verify output", &output, output.len())?;
    let after = op.read(session, "state", pool.len())?;
    let checkpoint = |seq: usize, r: usize| q + seq * max_rows + r;
    let written = (0..q)
        .flat_map(|seq| (0..lengths[seq]).map(move |r| checkpoint(seq, r)))
        .collect::<Vec<_>>();
    written_state(
        "verify state",
        &after,
        &pool,
        entry_bytes,
        element,
        &written,
        &[],
    )?;
    let row_bytes = width * 2;
    Ok((0..q)
        .map(|seq| {
            (0..lengths[seq])
                .map(|r| {
                    let row = starts[seq] as usize + r;
                    let mut unit = output[row * row_bytes..(row + 1) * row_bytes].to_vec();
                    unit.extend_from_slice(
                        &after[checkpoint(seq, r) * entry_bytes..][..entry_bytes],
                    );
                    unit
                })
                .collect()
        })
        .collect())
}

/// The rows of `sequence` through one-row decode launches in turn (the
/// profile's `conv_decode` or `delta_decode`), each continuing the slot
/// state the one before left: per row, the row's output bytes followed by
/// the slot's state after it. The output starts poisoned before every
/// launch; the slot, read and rewritten in place, must change.
fn decode_rows(session: &Session, s: &Shape, conv: bool, sequence: &Sequence) -> LabResult<Units> {
    let module = s.module(session)?;
    let data = Data::new(s, &sequence.rows, 0);
    let (width, entry_bytes, element) = s.recurrent(conv);
    let name = if conv { "conv_decode" } else { "delta_decode" };
    let kernel = Kernel::load(&module, &s.symbol(name))?.with_bucket(1, 1);
    let mut op = Operands::new();
    op.bytes(session, "state", &seeded_state(s, conv, sequence.state))?;
    op.i32(session, "slot_ids", &[0])?;
    op.u32(session, "live_sequences", 1)?;
    poisoned(session, &mut op, "output", width * 2)?;
    let mut units = Vec::new();
    for r in 0..sequence.rows.len() {
        bind_inputs(session, s, conv, &data, r..r + 1, &mut op)?;
        poison::poison(session, &[&op.buffers["output"]])?;
        let before = op.read(session, "state", entry_bytes)?;
        kernel.launch(session, &session.stream, &op, 1, 1, true)?;
        let mut unit = op.read(session, "output", width * 2)?;
        written_output(&format!("decode row {r} output"), &unit, unit.len())?;
        let after = op.read(session, "state", entry_bytes)?;
        written_state(
            &format!("decode row {r} state"),
            &after,
            &before,
            entry_bytes,
            element,
            &[],
            &[0],
        )?;
        unit.extend_from_slice(&after);
        units.push(unit);
    }
    Ok(units)
}

/// The verify sequence of a rows case: one row per seed, from the slot
/// state of the first seed.
fn rows_sequence(seeds: &[u64]) -> LabResult<Sequence> {
    let state = *seeds.first().ok_or("a rows case of no rows")?;
    Ok(Sequence {
        rows: seeds.to_vec(),
        state,
    })
}

/// Row-local gated norm rows seeded by `seeds`, each row's output.
fn norm(session: &Session, s: &Shape, seeds: &[u64]) -> LabResult<Units> {
    let module = s.module(session)?;
    let rows = seeds.len();
    let per_row = |stride: usize, scale: f64, salt: u64| {
        seeds
            .iter()
            .flat_map(|&seed| Rng::new(seed ^ salt).bf16(stride, scale))
            .collect::<Vec<_>>()
    };
    let mut fixed = Rng::new(0x0a0b);
    let mut op = Operands::new();
    op.bf16(session, "input", &per_row(s.norm_width, 0.5, 4))?;
    op.bf16(session, "gate", &per_row(s.z_stride, 2.0, 5))?;
    op.bf16(
        session,
        "weight",
        &(0..DIM)
            .map(|_| f32_to_bf16((0.875 + 0.15 * fixed.uniform()) as f32))
            .collect::<Vec<_>>(),
    )?;
    poisoned(session, &mut op, "output", rows * s.norm_width * 2)?;
    op.u32(session, "live_rows", rows as u32)?;
    op.scalars.insert("gate_stride", s.z_stride as u32);
    Kernel::load(&module, &s.symbol("gated_norm"))?.launch(
        session,
        &session.stream,
        &op,
        rows as u32,
        rows as u32,
        true,
    )?;
    let output = op.read(session, "output", rows * s.norm_width * 2)?;
    written_output("gated norm output", &output, output.len())?;
    Ok(rows_of(&output, s.norm_width))
}

/// Splits `bytes` into rows of `width` BF16 elements.
fn rows_of(bytes: &[u8], width: usize) -> Units {
    bytes.chunks(width * 2).map(<[u8]>::to_vec).collect()
}

#[derive(Default)]
pub(crate) struct Linear;

impl Adapter for Linear {
    fn kind(&self) -> OpKind {
        OpKind::Linear
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let session = &sessions[0];
        let s = Shape::of(entry)?;
        let row_seeds = (0..u64::from(rows)).collect::<Vec<_>>();
        let which = op(entry)?;
        if which == Op::Norm {
            let mut out = Vec::new();
            let mut start = 0;
            for &end in ends {
                out.extend(norm(session, &s, &row_seeds[start..end as usize])?);
                start = end as usize;
            }
            return Ok(out);
        }
        let data = Data::new(&s, &row_seeds, 1);
        let mut start = 0;
        let calls = ends
            .iter()
            .map(|&end| {
                let call = [start, end as i32];
                start = end as i32;
                call
            })
            .collect::<Vec<_>>();
        let (output, state) = prefill(
            session,
            &s,
            which,
            &data,
            &[rows as usize],
            &[0],
            &[-1],
            Some(&calls),
        )?;
        let mut units = rows_of(&output, s.recurrent(which == Op::Conv).0);
        units.push(state);
        Ok(units)
    }

    /// Verify entries: the rows of one verify sequence (`verify`). Prefill
    /// entries: sequences, each its output rows and its slot's final state.
    /// The gated norm: rows.
    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let session = &sessions[0];
        let s = Shape::of(entry)?;
        let which = op(entry)?;
        match which {
            Op::Norm => return norm(session, &s, seeds),
            Op::Verify { conv, max_rows } => {
                let sequence = rows_sequence(seeds)?;
                return verify(session, &s, entry, conv, max_rows, &[sequence])?
                    .pop()
                    .ok_or_else(|| "no verify sequence".to_owned());
            }
            Op::Conv | Op::Delta => {}
        }
        // Each unit is a 70-row sequence (a whole chunk and a partial one)
        // in its own slot; its output rows and its slot's final state.
        const LENGTH: usize = 70;
        let row_seeds = seeds
            .iter()
            .flat_map(|&seed| (0..LENGTH as u64).map(move |r| seed.wrapping_mul(131) ^ r))
            .collect::<Vec<_>>();
        let sequences = seeds.len();
        let data = Data::new(&s, &row_seeds, sequences);
        let slot_ids = (0..sequences as i32).collect::<Vec<_>>();
        let (output, state) = prefill(
            session,
            &s,
            which,
            &data,
            &vec![LENGTH; sequences],
            &slot_ids,
            &vec![-1; sequences],
            None,
        )?;
        let width = s.recurrent(which == Op::Conv).0;
        let state_bytes = state.len() / sequences;
        Ok((0..sequences)
            .map(|q| {
                let mut unit =
                    output[q * LENGTH * width * 2..(q + 1) * LENGTH * width * 2].to_vec();
                // Each slot's state starts from its own seeded entry.
                unit.extend_from_slice(&state[q * state_bytes..(q + 1) * state_bytes]);
                unit
            })
            .collect())
    }

    /// Verify entries: the same rows through one-row decode launches that
    /// carry the slot state. Others: each seed's unit launched alone.
    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let s = Shape::of(entry)?;
        if let Op::Verify { conv, .. } = op(entry)? {
            return decode_rows(&sessions[0], &s, conv, &rows_sequence(seeds)?);
        }
        seeds
            .iter()
            .map(|&seed| {
                self.units(sessions, entry, &[seed])?
                    .pop()
                    .ok_or_else(|| format!("{}: no unit", entry.symbol))
            })
            .collect()
    }

    /// Verify entries: one launch of a sequence per seed, of
    /// `EXACT_VERIFY_ROWS - q % EXACT_VERIFY_ROWS` rows for the `q`th, each
    /// unit a sequence's rows and checkpoints. Others: `units`.
    fn batch(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let Op::Verify { conv, max_rows } = op(entry)? else {
            return self.units(sessions, entry, seeds);
        };
        let s = Shape::of(entry)?;
        let exact = EXACT_VERIFY_ROWS as usize;
        let sequences = seeds
            .iter()
            .enumerate()
            .map(|(q, &seed)| Sequence {
                rows: (0..(exact - q % exact) as u64)
                    .map(|r| seed.wrapping_mul(131) ^ r)
                    .collect(),
                state: seed,
            })
            .collect::<Vec<_>>();
        Ok(verify(&sessions[0], &s, entry, conv, max_rows, &sequences)?
            .into_iter()
            .map(|rows| rows.concat())
            .collect())
    }

    fn anchors(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        let session = &sessions[0];
        let s = Shape::of(entry)?;
        let which = op(entry)?;
        let chunk = CHUNK as usize;
        let lengths = [65_usize, 130, chunk * 5 + 63, 128, 200];
        let anchored = [true, true, true, true, false];
        let sequences = lengths.len();
        let (slot_ids, pool) = slots(sequences, 2 * sequences);
        let anchors = (0..sequences)
            .map(|q| if anchored[q] { q as i32 } else { -1 })
            .collect::<Vec<_>>();
        let rows = lengths.iter().sum::<usize>();
        let data = Data::new(&s, &(0..rows as u64).collect::<Vec<_>>(), pool);
        let run = |data: &Data, lengths: &[usize], anchors: &[i32]| {
            prefill(session, &s, which, data, lengths, &slot_ids, anchors, None)
        };
        let (with_out, with_pool) = run(&data, &lengths, &anchors)?;
        let (without_out, without_pool) = run(&data, &lengths, &[-1; 5])?;
        let whole = lengths.map(|len| len / chunk * chunk);
        let keep = |values: &[u16], stride: usize| leading_rows(values, stride, &lengths, &whole);
        let truncated = Data {
            qkv: keep(&data.qkv, s.qkv_stride),
            a: keep(&data.a, s.gate_stride),
            b: keep(&data.b, s.gate_stride),
            conv_weight: data.conv_weight.clone(),
            a_log: data.a_log.clone(),
            dt_bias: data.dt_bias.clone(),
            conv_pool: data.conv_pool.clone(),
            delta_pool: data.delta_pool.clone(),
        };
        let (_, ends) = run(&truncated, &whole, &[-1; 5])?;
        let initial = data.pool_bytes(which == Op::Conv);
        let n = with_pool.len() / pool;
        let entry_of = |bytes: &[u8], index: i32| bytes[index as usize * n..][..n].to_vec();
        let mut compared = vec![Compared {
            label: "output with anchors".into(),
            got: with_out,
            expected: without_out,
        }];
        for q in 0..sequences {
            let slot = slot_ids[q];
            compared.push(Compared {
                label: format!("sequence {q} slot with anchors"),
                got: entry_of(&with_pool, slot),
                expected: entry_of(&without_pool, slot),
            });
            // A sequence with a partial last chunk leaves its whole chunks'
            // state; an anchored one without keeps its poisoned anchor
            // entry, and an entry no sequence anchors keeps its bytes.
            let expected = if anchored[q] && lengths[q] % chunk != 0 {
                entry_of(&ends, slot)
            } else if anchored[q] {
                vec![POISON; n]
            } else {
                entry_of(&initial, q as i32)
            };
            compared.push(Compared {
                label: format!("sequence {q} anchor entry"),
                got: entry_of(&with_pool, q as i32),
                expected,
            });
        }
        Ok(compared)
    }
}

/// The write checks the adapter runs after every launch reject a live region
/// still holding the poison and accept one written in full (no GPU).
#[test]
fn linear_write_checks_reject_unwritten_live_regions() {
    let finite = 0x3f80_u16.to_ne_bytes();
    // Three rows of two BF16 elements; the first two are live.
    let mut output = vec![POISON; 12];
    output[..8].copy_from_slice(&finite.repeat(4));
    assert!(written_output("output", &output, 8).is_ok());
    // A live element left poisoned.
    let mut unwritten = output.clone();
    unwritten[6..8].fill(POISON);
    assert!(written_output("output", &unwritten, 8).is_err());
    // A live element written NaN, as from poisoned workspace.
    let mut nan = output.clone();
    nan[2..4].copy_from_slice(&0x7fc0_u16.to_ne_bytes());
    assert!(written_output("output", &nan, 8).is_err());
    // A row past the live ones written.
    let mut past = output.clone();
    past[8..10].copy_from_slice(&finite);
    assert!(written_output("output", &past, 8).is_err());

    // Four entries of two F32 elements: a slot read and rewritten, two
    // write-only checkpoints that start poisoned, and an untouched entry.
    let entry = 8;
    let mut before = vec![0x11_u8; 4 * entry];
    before[entry..3 * entry].fill(POISON);
    let mut after = before.clone();
    after[..entry].fill(0x22);
    after[entry..3 * entry].copy_from_slice(&1.5_f32.to_ne_bytes().repeat(4));
    let check = |after: &[u8], before: &[u8]| {
        written_state("state", after, before, entry, DType::F32, &[1, 2], &[0])
    };
    assert!(check(&after, &before).is_ok());
    // A checkpoint element left poisoned.
    let mut unwritten = after.clone();
    unwritten[2 * entry + 4..3 * entry].fill(POISON);
    assert!(check(&unwritten, &before).is_err());
    // A slot left as it was.
    let mut kept = after.clone();
    kept[..entry].fill(0x11);
    assert!(check(&kept, &before).is_err());
    // An untouched entry changed.
    let mut changed = after.clone();
    changed[3 * entry] = 0x33;
    assert!(check(&changed, &before).is_err());
    // A write-only entry that was never poisoned proves nothing.
    let mut unpoisoned = before.clone();
    unpoisoned[entry..2 * entry].fill(0x11);
    assert!(check(&after, &unpoisoned).is_err());
}
