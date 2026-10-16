//! Cases of the recurrent family (`kernel_core::recurrent`): the causal
//! convolution and the gated delta rule in prefill, decode and verify, and
//! the gated RMSNorm; with the SM120 selection the lab's linear catalog
//! gives and the family's required op set. The per-channel-decay rule and
//! its short convolution are not covered.
//!
//! A case row is one sequence: its rows (one in decode), strided in their
//! buffers, and its slot operands. States live in a pool of entries
//! addressed by id ([`kernel_core::recurrent`], "State"): each sequence's
//! slot is an entry from the top of the pool down and its verify
//! checkpoints entries from the bottom up, with a spare entry between, and
//! the pool starts from a nonzero image. Decode advances each live
//! sequence's slot in place, prefill writes it back (from the slot when
//! `initial` is set, every other sequence, from zeros otherwise), and
//! verify leaves it unchanged and writes one checkpoint per row; every
//! other entry must keep its image. A slot the launch reads and rewrites
//! keeps its image before each launch ([`Spec::state_updating`]); entries
//! it only writes (checkpoints, prefill slots without `initial`) are
//! poisoned first. The property checks move a sequence's slot and
//! checkpoints with its row and compare the state it writes. With several
//! rows per sequence the chunk check, which compares single rows, is
//! reported as not exercised, and so is the prefill batch check, whose
//! copies of a sequence would read and rewrite one slot together. A verify
//! op's one-row bits compare each verify row with a decode step over that
//! row alone ([`decoded`]).
//!
//! The oracles are `recurrent_oracle`'s, per sequence: the convolution's
//! output within its bound and its states as exact copies, the delta
//! rule's output and states within their bounds, the norm within its
//! bound.

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::decode::{Compared, DecodeLaunch, Verified};
use crate::kit::{Decoded, OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::poison::WriteMap;
use kernel_core::backend::{Counts, Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, RecurrentPhase, RecurrentQuery};
use kernel_core::oracle::OracleOutput;
use kernel_core::properties::Property;
use kernel_core::recurrent::{ConvBuffers, DeltaRuleBuffers, GatedNormBuffers, RecurrentSlots};
use kernel_core::recurrent_oracle::{self, GatedDeltaInputs};
use kernels_sm120::linear::Entry;

/// Sequences of a decode, verify and prefill case, and the rows of each
/// prefill sequence.
const DECODE_SEQUENCES: u32 = 8;
const STEP_SEQUENCES: u32 = 4;
const PREFILL_ROWS: u32 = 8;

/// Rows of a gated norm case.
const NORM_ROWS: u32 = super::ROW_CAPACITY;

/// Padding of each strided row past its columns.
const ROW_PAD: usize = 8;

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The sequences of a case and their pool entries.
#[derive(Clone)]
struct Sequences {
    phase: RecurrentPhase,
    /// Sequences, and rows of each.
    count: usize,
    rows: usize,
    /// Pool entries: the slots from the top down, checkpoints from the
    /// bottom up, a spare between.
    entries: usize,
    /// Elements of one entry.
    state: usize,
}

impl Sequences {
    fn new(phase: RecurrentPhase, state: u64) -> Self {
        let (count, rows) = match phase {
            RecurrentPhase::Decode => (DECODE_SEQUENCES, 1),
            RecurrentPhase::Verify { max_rows } => (STEP_SEQUENCES, max_rows),
            RecurrentPhase::Prefill => (STEP_SEQUENCES, PREFILL_ROWS),
        };
        let (count, rows) = (count as usize, rows as usize);
        let checkpoints = match phase {
            RecurrentPhase::Verify { .. } => count * rows,
            _ => 0,
        };
        Self {
            phase,
            count,
            rows,
            entries: count + checkpoints + 1,
            state: state as usize,
        }
    }

    /// The decode sequences of a verify case's pool: [`DECODE_SEQUENCES`]
    /// of one row each, whose slots may be any entry of `verify`'s pool.
    fn decoding(verify: &Sequences) -> Self {
        Self {
            phase: RecurrentPhase::Decode,
            count: DECODE_SEQUENCES as usize,
            rows: 1,
            entries: verify.entries,
            state: verify.state,
        }
    }

    fn slot(&self, sequence: usize) -> usize {
        self.entries - 1 - sequence
    }

    fn checkpoint(&self, sequence: usize, row: usize) -> usize {
        sequence * self.rows + row
    }

    /// Whether prefill reads sequence `s`'s slot: every other one.
    fn initial(&self, sequence: usize) -> bool {
        sequence % 2 == 0
    }

    /// The slot operands of the phase, over `Rows::Live` sequences.
    fn slots<'a>(&self) -> RecurrentSlots<Spec<'a>> {
        let per = |values: Vec<i32>| Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&values));
        let n = self.count;
        let prefill = self.phase == RecurrentPhase::Prefill;
        let multi = self.phase != RecurrentPhase::Decode;
        RecurrentSlots {
            row_offsets: multi.then(|| {
                Spec::input(
                    DType::I32,
                    (Rows::Fixed(1), n + 1),
                    i32_bytes(&(0..=n).map(|s| (s * self.rows) as i32).collect::<Vec<_>>()),
                )
            }),
            start_positions: prefill.then(|| {
                per((0..n)
                    .map(|s| if self.initial(s) { 17 } else { 0 })
                    .collect())
            }),
            slot_ids: per((0..n).map(|s| self.slot(s) as i32).collect()),
            initial: prefill.then(|| per((0..n).map(|s| i32::from(self.initial(s))).collect())),
            anchor_ids: prefill.then(|| per(vec![-1; n])),
            checkpoint_ids: matches!(self.phase, RecurrentPhase::Verify { .. }).then(|| {
                Spec::input(
                    DType::I32,
                    (Rows::Live, self.rows),
                    i32_bytes(
                        &(0..n * self.rows)
                            .map(|at| self.checkpoint(at / self.rows, at % self.rows) as i32)
                            .collect::<Vec<_>>(),
                    ),
                )
            }),
            live_sequences: Spec::live(),
        }
    }

    /// The global rows of sequence `sequence` by `row_offsets` in `images`
    /// (its own row in decode): at most the case's rows of each.
    fn span(&self, images: &Images, sequence: usize) -> LabResult<std::ops::Range<usize>> {
        if self.phase == RecurrentPhase::Decode {
            return Ok(sequence..sequence + 1);
        }
        let offsets = images.i32s("slots.row_offsets")?;
        let (start, end) = (offsets[sequence], offsets[sequence + 1]);
        if start < 0 || end < start || (end - start) as usize > self.rows {
            return Err(format!(
                "sequence {sequence}'s row offsets {start}..{end} do not fit"
            ));
        }
        Ok(start as usize..end as usize)
    }

    /// The pool entries the first `live` sequences write by the ids in
    /// `images`, with the state (`Some(row)` after that row, `None` after
    /// the last) each takes.
    fn written(&self, images: &Images, live: u32) -> LabResult<Vec<(usize, usize, Option<usize>)>> {
        let mut written = Vec::new();
        let entry = |id: i32| usize::try_from(id).map_err(|_| "a negative pool entry id");
        for sequence in 0..live as usize {
            match self.phase {
                RecurrentPhase::Verify { .. } => {
                    let ids = images.i32s("slots.checkpoint_ids")?;
                    for row in 0..self.span(images, sequence)?.len() {
                        written.push((
                            sequence,
                            entry(ids[sequence * self.rows + row])?,
                            Some(row),
                        ));
                    }
                }
                _ => {
                    let id = images.i32s("slots.slot_ids")?[sequence];
                    written.push((sequence, entry(id)?, None));
                }
            }
        }
        Ok(written)
    }

    /// Whether sequence `sequence` starts from the entry it writes back by
    /// the flags in `images`: a decode slot, a prefill slot with `initial`;
    /// never a verify checkpoint.
    fn rewrites(&self, images: &Images, sequence: usize) -> LabResult<bool> {
        Ok(match self.phase {
            RecurrentPhase::Decode => true,
            RecurrentPhase::Prefill => images.i32s("slots.initial")?[sequence] != 0,
            RecurrentPhase::Verify { .. } => false,
        })
    }

    /// The entries the first `live` sequences write, split into those
    /// written without being read (poisoned before a launch) and those
    /// read and rewritten in place (kept).
    fn entries(&self, images: &Images, live: u32) -> LabResult<(Vec<usize>, Vec<usize>)> {
        let (mut written, mut rewritten) = (Vec::new(), Vec::new());
        for (sequence, entry, _) in self.written(images, live)? {
            if self.rewrites(images, sequence)? {
                rewritten.push(entry);
            } else {
                written.push(entry);
            }
        }
        Ok((written, rewritten))
    }

    /// The state pool spec of `dtype` over the pool image: entries a
    /// launch only writes are poisoned first, and slots it reads and
    /// rewrites keep their image.
    fn pool<'a>(&self, dtype: DType, image: Vec<u8>) -> Spec<'a> {
        let (written, rewritten) = (self.clone(), self.clone());
        Spec::state_updating(
            dtype,
            (Rows::Fixed(self.entries), self.state),
            image,
            Write::Map(Box::new(move |images, live| {
                let (entries, _) = written.entries(images, live)?;
                WriteMap::of(dtype).rows(written.state, entries)
            })),
            move |images, live| {
                let entry_bytes = rewritten.state * WriteMap::of(dtype).bytes()?;
                let (_, entries) = rewritten.entries(images, live)?;
                Ok(entries
                    .into_iter()
                    .map(|entry| entry * entry_bytes..(entry + 1) * entry_bytes)
                    .collect())
            },
        )
    }

    /// The state a sequence starts from by the ids and flags in `images`:
    /// its slot, or zeros for a prefill sequence without `initial`.
    fn start<T: Copy + Default>(
        &self,
        images: &Images,
        pool: &[T],
        sequence: usize,
    ) -> LabResult<Vec<T>> {
        if self.phase == RecurrentPhase::Prefill && images.i32s("slots.initial")?[sequence] == 0 {
            return Ok(vec![T::default(); self.state]);
        }
        let slot = images.i32s("slots.slot_ids")?[sequence] as usize;
        Ok(pool[slot * self.state..][..self.state].to_vec())
    }

    /// The property checks a case of these sequences cannot run: with
    /// several rows per sequence, chunk (it compares single rows), while
    /// one-row bits compare decode steps ([`decoded`]); in prefill, batch
    /// too, since its copies of a sequence would read and rewrite one slot
    /// together.
    fn property_note<'a>(&self, case: OpCase<'a>) -> OpCase<'a> {
        let case = if self.rows > 1 {
            case.item_rows("a case row is a sequence of several rows, and the chunk check compares single rows")
        } else {
            case
        };
        if self.phase == RecurrentPhase::Prefill {
            case.without_properties(
                &[Property::Batch],
                "the batch check's copies of a sequence would read and rewrite its one slot \
                 together",
            )
        } else {
            case
        }
    }

    fn counts(&self) -> impl Fn(u32) -> Counts + use<> {
        let rows = self.rows as u32;
        move |sequences| Counts::rows(sequences * rows).with_items(sequences, 0)
    }
}

/// `rows` rows of `stride` elements, the first `columns` of each, from the
/// BF16 image `bits`.
fn columns(bits: &[u16], stride: usize, from: usize, columns: usize) -> Vec<f64> {
    bits.chunks_exact(stride)
        .flat_map(|row| widen(&row[from..from + columns]))
        .collect()
}

/// An oracle output of whole buffers: `values` element by element, with
/// their bounds where `parts` give them and none elsewhere.
fn joined(mut values: Vec<f64>, parts: &[(usize, OracleOutput)]) -> LabResult<OracleOutput> {
    let mut sums = vec![0.0; values.len()];
    let mut lengths = vec![1; values.len()];
    for (at, part) in parts {
        let range = *at..at + part.values.len();
        values[range.clone()].copy_from_slice(&part.values);
        sums[range.clone()].copy_from_slice(&part.sum_abs_products);
        lengths[range].copy_from_slice(&part.reduction_lengths);
    }
    Ok(OracleOutput {
        values,
        sum_abs_products: sums,
        reduction_lengths: lengths,
    })
}

/// The case of causal convolution `query` over its phase's sequences.
pub fn conv_case<'a>(
    query: RecurrentQuery,
    prepared: Prepared<ConvBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RecurrentQuery::CausalConv1d { phase, .. } = query else {
        return Err("a convolution case convolves by a convolution query".into());
    };
    conv_over(
        query,
        Sequences::new(phase, query.state_elements()),
        prepared,
    )
}

/// The case of causal convolution `query` over sequences `seq`.
fn conv_over<'a>(
    query: RecurrentQuery,
    seq: Sequences,
    prepared: Prepared<ConvBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RecurrentQuery::CausalConv1d {
        channels, kernel, ..
    } = query
    else {
        return Err("a convolution case convolves by a convolution query".into());
    };
    let (c, taps) = (channels as usize, kernel as usize);
    let stride = c + ROW_PAD;
    let (n, r) = (seq.count, seq.rows);
    let roles = ConvBuffers {
        input: Spec::input(
            DType::Bf16,
            (Rows::Live, r * stride),
            bf16_bytes(&bf16_values(41, n * r * stride)),
        ),
        input_stride: stride as u32,
        weights: vec![Spec::weight(bf16_bytes(&bf16_values(42, c * taps)))],
        state: seq.pool(
            DType::Bf16,
            bf16_bytes(&bf16_values(43, seq.entries * seq.state)),
        ),
        output: Spec::output(DType::Bf16, (Rows::Live, r * c)),
        slots: seq.slots(),
    };
    let counts = seq.counts();
    let oracle_seq = seq.clone();
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let seq = &oracle_seq;
        let input = images.u16s("input")?;
        let weight = widen(&images.u16s("weights[0]")?);
        let pool = images.u16s("state")?;
        let mut outputs = Vec::new();
        let mut states = pool.clone();
        let written = seq.written(images, live)?;
        for sequence in 0..n {
            let span = seq.span(images, sequence)?;
            let rows = &input[span.start * stride..span.end * stride];
            let start = widen(&seq.start(images, &pool, sequence)?);
            let result = recurrent_oracle::causal_conv1d(
                query,
                &columns(rows, stride, 0, c),
                &weight,
                &start,
            )
            .map_err(|e| e.0.to_string())?;
            outputs.push((span.start * c, result.output));
            for &(_, entry, row) in written.iter().filter(|(s, ..)| *s == sequence) {
                let state = match row {
                    Some(row) => &result.row_states[row],
                    None => &result.state,
                };
                let bits = state
                    .iter()
                    .map(|&v| f32_to_bf16(v as f32))
                    .collect::<Vec<_>>();
                states[entry * seq.state..][..seq.state].copy_from_slice(&bits);
            }
        }
        Ok(vec![
            (
                "output".to_owned(),
                Expected::Values(joined(vec![0.0; n * r * c], &outputs)?),
            ),
            ("state".to_owned(), Expected::Bits(bf16_bytes(&states))),
        ])
    };
    let case = OpCase::with_counts(prepared, n as u32, roles, counts)?.oracle(oracle);
    Ok(seq.property_note(case))
}

/// The case of gated delta rule `query` over its phase's sequences: q, k
/// and v columns of strided rows, decay and beta logits at one stride,
/// `A_log` from -1.5 to 0 and `dt_bias` from -0.5 to 0.5 per value head.
pub fn delta_case<'a>(
    query: RecurrentQuery,
    prepared: Prepared<DeltaRuleBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RecurrentQuery::GatedDeltaRule { phase, .. } = query else {
        return Err("a delta rule case runs a gated delta rule query".into());
    };
    delta_over(
        query,
        Sequences::new(phase, query.state_elements()),
        prepared,
    )
}

/// The case of gated delta rule `query` over sequences `seq`.
fn delta_over<'a>(
    query: RecurrentQuery,
    seq: Sequences,
    prepared: Prepared<DeltaRuleBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RecurrentQuery::GatedDeltaRule {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        ..
    } = query
    else {
        return Err("a delta rule case runs a gated delta rule query".into());
    };
    let (kh, vh, kd, vd) = (
        key_heads as usize,
        value_heads as usize,
        key_dim as usize,
        value_dim as usize,
    );
    let (n, r) = (seq.count, seq.rows);
    let qkv = 2 * kh * kd + vh * vd;
    let stride = qkv + ROW_PAD;
    let gates = vh + ROW_PAD;
    let workspace = prepared.info().bytes_of("workspace").unwrap_or(0) as usize;
    let a_log = (0..vh)
        .map(|h| -1.5 * h as f32 / vh as f32)
        .collect::<Vec<_>>();
    let dt_bias = (0..vh)
        .map(|h| (h % 5) as f32 / 4.0 - 0.5)
        .collect::<Vec<_>>();
    // States of magnitude below 0.25, as a decayed state holds.
    let pool = bf16_values(53, seq.entries * seq.state)
        .into_iter()
        .map(|bits| bf16_to_f32(bits) / 8.0)
        .collect::<Vec<_>>();
    let roles = DeltaRuleBuffers {
        qkv: Spec::input(
            DType::Bf16,
            (Rows::Live, r * stride),
            bf16_bytes(&bf16_values(51, n * r * stride)),
        ),
        qkv_stride: stride as u32,
        decay: Spec::input(
            DType::Bf16,
            (Rows::Live, r * gates),
            bf16_bytes(&bf16_values(52, n * r * gates)),
        ),
        decay_stride: gates as u32,
        beta: Spec::input(
            DType::Bf16,
            (Rows::Live, r * gates),
            bf16_bytes(&bf16_values(54, n * r * gates)),
        ),
        beta_stride: gates as u32,
        a_log: Spec::input(DType::F32, (Rows::Fixed(1), vh), f32_bytes(&a_log)),
        dt_bias: Spec::input(DType::F32, (Rows::Fixed(1), vh), f32_bytes(&dt_bias)),
        state: seq.pool(DType::F32, f32_bytes(&pool)),
        output: Spec::output(DType::Bf16, (Rows::Live, r * vh * vd)),
        workspace: (workspace > 0).then(|| Spec::scratch(workspace)),
        slots: seq.slots(),
    };
    let counts = seq.counts();
    let oracle_seq = seq.clone();
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let seq = &oracle_seq;
        let qkv_rows = images.u16s("qkv")?;
        let decay = images.u16s("decay")?;
        let beta = images.u16s("beta")?;
        let a_log = images
            .f32s("a_log")?
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>();
        let dt_bias = images
            .f32s("dt_bias")?
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>();
        let pool = images.f32s("state")?;
        let written = seq.written(images, live)?;
        let mut outputs = Vec::new();
        let mut states = Vec::new();
        for sequence in 0..n {
            let span = seq.span(images, sequence)?;
            let rows = &qkv_rows[span.start * stride..span.end * stride];
            let gate =
                |bits: &[u16]| columns(&bits[span.start * gates..span.end * gates], gates, 0, vh);
            let start = seq
                .start(images, &pool, sequence)?
                .iter()
                .map(|&v| f64::from(v))
                .collect::<Vec<_>>();
            let (q, k, v) = (
                columns(rows, stride, 0, kh * kd),
                columns(rows, stride, kh * kd, kh * kd),
                columns(rows, stride, 2 * kh * kd, vh * vd),
            );
            let result = recurrent_oracle::gated_delta_rule(
                query,
                GatedDeltaInputs {
                    q: &q,
                    k: &k,
                    v: &v,
                    a: &gate(&decay),
                    b: &gate(&beta),
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    state: &start,
                },
            )
            .map_err(|e| e.0.to_string())?;
            outputs.push((span.start * vh * vd, result.output));
            for &(_, entry, row) in written.iter().filter(|(s, ..)| *s == sequence) {
                let state = match row {
                    Some(row) => result.row_states[row].clone(),
                    None => result.state.clone(),
                };
                states.push((entry * seq.state, state));
            }
        }
        let initial = pool.iter().map(|&v| f64::from(v)).collect();
        Ok(vec![
            (
                "output".to_owned(),
                Expected::Values(joined(vec![0.0; n * r * vh * vd], &outputs)?),
            ),
            (
                "state".to_owned(),
                Expected::Values(joined(initial, &states)?),
            ),
        ])
    };
    let case = OpCase::with_counts(prepared, n as u32, roles, counts)?.oracle(oracle);
    Ok(seq.property_note(case))
}

/// How verify sequences `verify` map to launches of a decode op's `case`
/// over one row alone. `inputs` are the row inputs by path with the bytes
/// of one strided row, `output` the bytes of an output row and `entry` of
/// a pool entry.
///
/// - The verify step cuts each sequence to at most the claim's rows: row
///   `t` of sequence `s` moves to global row `s rows + t` of each row
///   input, as `row_offsets` then says.
/// - Its row `t` is a decode step of that row from the state decode would
///   hold before it: for row 0 the sequence's slot in the committed pool
///   the verify step read (not the pool it left, which a faulty step could
///   have changed alike), after that the verify step's checkpoint of row
///   `t - 1` in the pool it left.
/// - The decode step's output row equals the verify output row, and the
///   slot it advances in place equals the verify checkpoint of row `t`.
///   Rows that equal a decode step from the state before them equal the
///   chain of decode steps from the slot, one row at a time.
fn decoded<'a>(
    verify: Sequences,
    inputs: &[(&'static str, usize)],
    output: usize,
    entry: usize,
    case: OpCase<'a>,
) -> Decoded<'a> {
    let (n, r) = (verify.count, verify.rows);
    let (cut_inputs, inputs) = (inputs.to_vec(), inputs.to_vec());
    let cut = move |images: &Images, max: u32| -> LabResult<Images> {
        let rows = r.min(max as usize);
        let offsets = (0..=n).map(|s| (s * rows) as i32).collect::<Vec<_>>();
        let mut cut = images.with("slots.row_offsets", i32_bytes(&offsets));
        for &(path, row) in &cut_inputs {
            let old = images.bytes(path)?;
            let mut new = old.to_vec();
            for s in 0..n {
                for t in 0..rows {
                    new[(s * rows + t) * row..][..row]
                        .copy_from_slice(&old[(s * r + t) * row..][..row]);
                }
            }
            cut = cut.with(path, new);
        }
        Ok(cut)
    };
    let launches = move |verified: &Verified<'_>, base: &Images| -> LabResult<Vec<DecodeLaunch>> {
        let rows = r.min(verified.max as usize);
        let images = verified.images;
        let slots = images.i32s("slots.slot_ids")?;
        let checkpoints = images.i32s("slots.checkpoint_ids")?;
        // Row 0 starts from the committed pool the verify step read, never
        // from what it left: a verify step that corrupts its source slot
        // would hand the decode step the same wrong state.
        let committed = images.bytes("state")?.to_vec();
        let pool = verified
            .results
            .get("state")
            .ok_or("the verify step gave no state")?;
        let first = |path: &str, row: &[u8]| -> LabResult<Vec<u8>> {
            let mut bytes = base.bytes(path)?.to_vec();
            bytes[..row.len()].copy_from_slice(row);
            Ok(bytes)
        };
        let id = |id: i32| usize::try_from(id).map_err(|_| "a negative pool entry id".to_owned());
        let mut launches = Vec::new();
        for s in 0..verified.live as usize {
            for t in 0..rows {
                let global = s * rows + t;
                let start = id(if t == 0 {
                    slots[s]
                } else {
                    checkpoints[s * r + t - 1]
                })?;
                let checkpoint = id(checkpoints[s * r + t])?;
                let start_pool = if t == 0 { &committed } else { pool };
                let mut decode = base.with("state", start_pool.clone()).with(
                    "slots.slot_ids",
                    first("slots.slot_ids", &i32_bytes(&[start as i32]))?,
                );
                for &(path, row) in &inputs {
                    decode = decode.with(
                        path,
                        first(path, &images.bytes(path)?[global * row..][..row])?,
                    );
                }
                launches.push(DecodeLaunch {
                    label: format!("sequence {s} row {t}"),
                    images: decode,
                    compared: vec![
                        Compared {
                            verify: ("output".into(), global * output..(global + 1) * output),
                            decode: ("output".into(), 0..output),
                        },
                        Compared {
                            verify: ("state".into(), checkpoint * entry..(checkpoint + 1) * entry),
                            decode: ("state".into(), start * entry..(start + 1) * entry),
                        },
                    ],
                });
            }
        }
        Ok(launches)
    };
    Decoded::new(case, cut, launches)
}

/// How verify convolution `query`'s sequences map to one-row launches of
/// the decode convolution `prepared` ([`decoded`]).
fn conv_decoded<'a>(
    query: RecurrentQuery,
    decode: RecurrentQuery,
    prepared: Prepared<ConvBuffers<View>>,
) -> LabResult<Decoded<'a>> {
    let (
        RecurrentQuery::CausalConv1d {
            channels, phase, ..
        },
        Some((RecurrentPhase::Decode, _)),
    ) = (query, phase_capacity(decode))
    else {
        return Err("a convolution verify maps to a decode convolution".into());
    };
    let verify = Sequences::new(phase, query.state_elements());
    let case = conv_over(decode, Sequences::decoding(&verify), prepared)?;
    let c = channels as usize;
    Ok(decoded(
        verify.clone(),
        &[("input", (c + ROW_PAD) * 2)],
        c * 2,
        verify.state * 2,
        case,
    ))
}

/// How verify delta rule `query`'s sequences map to one-row launches of
/// the decode delta rule `prepared` ([`decoded`]).
fn delta_decoded<'a>(
    query: RecurrentQuery,
    decode: RecurrentQuery,
    prepared: Prepared<DeltaRuleBuffers<View>>,
) -> LabResult<Decoded<'a>> {
    let (
        RecurrentQuery::GatedDeltaRule {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            phase,
            ..
        },
        Some((RecurrentPhase::Decode, _)),
    ) = (query, phase_capacity(decode))
    else {
        return Err("a delta rule verify maps to a decode delta rule".into());
    };
    let verify = Sequences::new(phase, query.state_elements());
    let case = delta_over(decode, Sequences::decoding(&verify), prepared)?;
    let (kh, vh, kd, vd) = (
        key_heads as usize,
        value_heads as usize,
        key_dim as usize,
        value_dim as usize,
    );
    let gates = (vh + ROW_PAD) * 2;
    Ok(decoded(
        verify.clone(),
        &[
            ("qkv", (2 * kh * kd + vh * vd + ROW_PAD) * 2),
            ("decay", gates),
            ("beta", gates),
        ],
        vh * vd * 2,
        verify.state * 4,
        case,
    ))
}

/// The phase and capacity of a convolution or delta rule `query`.
fn phase_capacity(query: RecurrentQuery) -> Option<(RecurrentPhase, u32)> {
    match query {
        RecurrentQuery::CausalConv1d {
            phase, capacity, ..
        }
        | RecurrentQuery::GatedDeltaRule {
            phase, capacity, ..
        } => Some((phase, capacity)),
        RecurrentQuery::GatedRmsNorm { .. } => None,
    }
}

/// A convolution or delta rule `query` in `phase` at `to` capacity.
fn reshaped(query: RecurrentQuery, to: (RecurrentPhase, u32)) -> RecurrentQuery {
    match query {
        RecurrentQuery::CausalConv1d {
            channels,
            kernel,
            dilation,
            activation,
            ..
        } => RecurrentQuery::CausalConv1d {
            channels,
            kernel,
            dilation,
            activation,
            phase: to.0,
            capacity: to.1,
        },
        RecurrentQuery::GatedDeltaRule {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            ..
        } => RecurrentQuery::GatedDeltaRule {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            phase: to.0,
            capacity: to.1,
        },
        norm => norm,
    }
}

/// The decode query among `queries` whose one-row launches verify
/// `query`'s rows refer to: the same query in decode, at any capacity.
fn decode_of(query: RecurrentQuery, queries: &[RecurrentQuery]) -> Option<RecurrentQuery> {
    queries.iter().copied().find(|&other| {
        phase_capacity(other).is_some_and(|(phase, capacity)| {
            phase == RecurrentPhase::Decode
                && reshaped(query, (RecurrentPhase::Decode, capacity)) == other
        })
    })
}

/// The case of gated RMSNorm `query` over [`NORM_ROWS`] rows, the gate
/// strided past the input's width.
pub fn norm_case<'a>(
    query: RecurrentQuery,
    prepared: Prepared<GatedNormBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RecurrentQuery::GatedRmsNorm {
        width, head_dim, ..
    } = query
    else {
        return Err("a gated norm case normalizes by a gated norm query".into());
    };
    let (width, d, rows) = (width as usize, head_dim as usize, NORM_ROWS as usize);
    let stride = width + ROW_PAD;
    let roles = GatedNormBuffers {
        input: Spec::input(
            DType::Bf16,
            (Rows::Live, width),
            bf16_bytes(&bf16_values(61, rows * width)),
        ),
        gate: Spec::input(
            DType::Bf16,
            (Rows::Live, stride),
            bf16_bytes(&bf16_values(62, rows * stride)),
        ),
        gate_stride: stride as u32,
        weight: Spec::input(
            DType::Bf16,
            (Rows::Fixed(1), d),
            bf16_bytes(&bf16_values(63, d)),
        ),
        output: Spec::output(DType::Bf16, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let output = recurrent_oracle::gated_rmsnorm(
            query,
            &widen(&images.u16s("input")?),
            &columns(&images.u16s("gate")?, stride, 0, width),
            &widen(&images.u16s("weight")?),
        )
        .map_err(|e| e.0.to_string())?;
        Ok(vec![("output".to_owned(), Expected::Values(output))])
    };
    Ok(OpCase::new(prepared, NORM_ROWS, roles)?.oracle(oracle))
}

/// The linear entries of `catalogs`: SM120's selection, the first entry
/// of each query and stage (two of one are an SM120 tie).
pub fn sm120_entries(catalogs: &kernels_sm120::Catalogs) -> Vec<Entry> {
    let mut chosen: Vec<Entry> = Vec::new();
    for entry in kernels_sm120::linear::catalog(catalogs) {
        if !chosen
            .iter()
            .any(|other| other.query == entry.query && other.stage == entry.stage)
        {
            chosen.push(entry);
        }
    }
    chosen
}

fn phase_name(phase: RecurrentPhase) -> String {
    match phase {
        RecurrentPhase::Verify { max_rows } => format!("Verify {max_rows} rows"),
        other => format!("{other:?}"),
    }
}

/// The family's required ops: one per query of `entries`, over its
/// phase's sequences (rows for the norm).
pub fn family<'a>(entries: &[Entry]) -> Vec<Required<'a>> {
    let mut queries: Vec<RecurrentQuery> = Vec::new();
    for entry in entries {
        if !queries.contains(&entry.query) {
            queries.push(entry.query);
        }
    }
    let decodes = queries
        .iter()
        .map(|&query| match phase_capacity(query) {
            Some((RecurrentPhase::Verify { .. }, _)) => decode_of(query, &queries),
            _ => None,
        })
        .collect::<Vec<_>>();
    queries
        .into_iter()
        .zip(decodes)
        .map(|(query, decode)| match query {
            RecurrentQuery::CausalConv1d {
                channels,
                kernel,
                phase,
                ..
            } => {
                let seq = Sequences::new(phase, query.state_elements());
                let rows = (seq.count * seq.rows) as u32;
                let required = Required::new(
                    format!(
                        "causal conv {channels} channels {kernel} taps {} sequences {}",
                        phase_name(phase),
                        seq.count
                    ),
                    Need::rows(rows),
                    move |backend, need| backend.causal_conv1d(&query, seq.count as u32, need),
                    move |prepared| conv_case(query, prepared),
                );
                match decode {
                    Some(decode) => required.with_decode(move |backend| {
                        backend
                            .causal_conv1d(&decode, DECODE_SEQUENCES, Need::rows(DECODE_SEQUENCES))
                            .map(|prepared| conv_decoded(query, decode, prepared))
                    }),
                    None => required,
                }
            }
            RecurrentQuery::GatedDeltaRule {
                key_heads,
                value_heads,
                key_dim,
                phase,
                ..
            } => {
                let seq = Sequences::new(phase, query.state_elements());
                let rows = (seq.count * seq.rows) as u32;
                let required = Required::new(
                    format!(
                        "gated delta rule {key_heads}x{value_heads} heads {key_dim} {} \
                         sequences {}",
                        phase_name(phase),
                        seq.count
                    ),
                    Need::rows(rows),
                    move |backend, need| backend.gated_delta_rule(&query, seq.count as u32, need),
                    move |prepared| delta_case(query, prepared),
                );
                match decode {
                    Some(decode) => required.with_decode(move |backend| {
                        backend
                            .gated_delta_rule(
                                &decode,
                                DECODE_SEQUENCES,
                                Need::rows(DECODE_SEQUENCES),
                            )
                            .map(|prepared| delta_decoded(query, decode, prepared))
                    }),
                    None => required,
                }
            }
            RecurrentQuery::GatedRmsNorm { width, gate, .. } => Required::new(
                format!("gated rmsnorm {width} {gate:?} rows {NORM_ROWS}"),
                Need::rows(NORM_ROWS),
                move |backend, need| backend.gated_rmsnorm(&query, need),
                move |prepared| norm_case(query, prepared),
            ),
        })
        .collect()
}

/// Self-checks of the family's cases over a consumer's linear entries (as
/// [`sm120_entries`] selects them), for its tests: each panics naming the
/// first failure. kernel-lab runs them over the in-tree catalogs.
pub mod checks {
    use super::super::testing::{decode_gaps, prepared};
    use super::*;
    use crate::kit::Kind;

    /// Each verify row of a convolution and a delta rule maps to the decode
    /// step of its own row from the state before it: the decode oracle
    /// over a launch's images gives the verify oracle's output row and
    /// checkpoint within the F32 rounding of the state it starts from (the
    /// verify oracle chains in F64; 2e-6 at most), at four rows and two,
    /// and a launch of the next row's images does not (0.26 at least).
    /// Returns the verify ops checked.
    pub fn verify_rows_map_to_their_decode_steps(entries: &[Entry]) -> usize {
        let backend = kernels_sm120::backend::Sm120::new().with_linear(entries.to_vec());
        let mut mapped = 0;
        for required in family(entries) {
            if !required.name.contains("Verify") {
                continue;
            }
            let case = prepared(&backend, &required);
            for max in [4, 2] {
                let gaps = decode_gaps(&case, max, false, None);
                assert_eq!(gaps.len(), STEP_SEQUENCES as usize * max as usize);
                assert!(
                    gaps.iter().all(|&gap| gap < 1e-5),
                    "{} max {max}: {gaps:?}",
                    required.name
                );
                let shifted = decode_gaps(&case, max, true, None);
                assert!(
                    shifted.iter().all(|&gap| gap > 0.1),
                    "{} max {max}: {shifted:?}",
                    required.name
                );
            }
            mapped += 1;
        }
        mapped
    }

    /// A verify step that clears each sequence's source slot and computes
    /// from the cleared state (a fault only short verify steps might have)
    /// fails the mapping at every sequence's first row: its decode step
    /// starts from the committed slot, not from the pool the verify step
    /// left. Starting it from the pool the step left passed all 16
    /// comparisons (review R1). Returns the verify ops checked.
    pub fn a_verify_step_that_corrupts_its_slot_fails_the_mapping(entries: &[Entry]) -> usize {
        let backend = kernels_sm120::backend::Sm120::new().with_linear(entries.to_vec());
        let cleared = |images: &Images| {
            let slots = images.i32s("slots.slot_ids").unwrap();
            let checkpoints = images.i32s("slots.checkpoint_ids").unwrap().len();
            let mut pool = images.bytes("state").unwrap().to_vec();
            let entry = pool.len() / (slots.len() + checkpoints + 1);
            for &slot in &slots {
                pool[slot as usize * entry..][..entry].fill(0);
            }
            images.with("state", pool)
        };
        let mut faulted = 0;
        for required in family(entries) {
            if !required.name.contains("Verify") {
                continue;
            }
            let case = prepared(&backend, &required);
            let gaps = decode_gaps(&case, 4, false, Some(&cleared));
            for (at, gap) in gaps.iter().enumerate() {
                if at % 4 == 0 {
                    assert!(*gap > 0.1, "{} launch {at}: {gaps:?}", required.name);
                }
            }
            faulted += 1;
        }
        faulted
    }

    /// The state a launch starts from keeps its image: the slots decode
    /// and initialized prefill sequences read are rewritten in place, not
    /// poisoned, while verify checkpoints and uninitialized prefill slots
    /// are poisoned and must be written.
    pub fn read_slots_are_rewritten_not_poisoned(entries: &[Entry]) {
        let backend = kernels_sm120::backend::Sm120::new().with_linear(entries.to_vec());
        let ops = family(entries);
        for entry in entries {
            let (RecurrentQuery::CausalConv1d { phase, .. }
            | RecurrentQuery::GatedDeltaRule { phase, .. }) = entry.query
            else {
                continue;
            };
            let seq = Sequences::new(phase, entry.query.state_elements());
            let name = phase_name(phase);
            let required = ops
                .iter()
                .find(|op| op.name.contains(&name) && !op.name.contains("rmsnorm"))
                .unwrap();
            let images = prepared(&backend, required).case.images();
            let Kind::State(_, Write::Map(written), Some(updated)) =
                seq.pool(DType::F32, Vec::new()).kind
            else {
                panic!("the pool is a state with updated ranges");
            };
            let live = seq.count as u32;
            let poisoned = written(&images, live).unwrap().ranges().to_vec();
            let kept = updated(&images, live).unwrap();
            let bytes = seq.state * 4;
            let entry_of = |range: &std::ops::Range<usize>| range.start / bytes;
            let poisoned = poisoned
                .iter()
                .flat_map(|r| r.start / bytes..r.end / bytes)
                .collect::<std::collections::BTreeSet<_>>();
            let kept = kept
                .iter()
                .map(entry_of)
                .collect::<std::collections::BTreeSet<_>>();
            for sequence in 0..seq.count {
                let slot = seq.slot(sequence);
                let reads = match phase {
                    RecurrentPhase::Decode => true,
                    RecurrentPhase::Prefill => seq.initial(sequence),
                    RecurrentPhase::Verify { .. } => false,
                };
                let writes = !matches!(phase, RecurrentPhase::Verify { .. });
                assert_eq!(
                    kept.contains(&slot),
                    reads && writes,
                    "{name} sequence {sequence}"
                );
                assert_eq!(
                    poisoned.contains(&slot),
                    !reads && writes,
                    "{name} sequence {sequence}"
                );
            }
            if let RecurrentPhase::Verify { .. } = phase {
                assert_eq!(poisoned.len(), seq.count * seq.rows, "{name} checkpoints");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slots, checkpoints and the spare never share an entry.
    #[test]
    fn pool_entries_are_disjoint() {
        let seq = Sequences::new(RecurrentPhase::Verify { max_rows: 4 }, 8);
        let mut entries = (0..seq.count).map(|s| seq.slot(s)).collect::<Vec<_>>();
        entries.extend(
            (0..seq.count)
                .flat_map(|s| (0..4).map(move |r| (s, r)))
                .map(|(s, r)| seq.checkpoint(s, r)),
        );
        let distinct = entries.iter().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(distinct.len(), entries.len());
        assert!(entries.iter().all(|&e| e < seq.entries));
    }
}
