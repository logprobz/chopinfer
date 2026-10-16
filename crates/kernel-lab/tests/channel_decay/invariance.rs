//! The per-channel-decay invariance adapter (`OpKind::ChannelDecay`): the
//! served entries (`kernel_lab::declared::channel_decay`: the staged
//! catalog and the tiled and streamed prefill stages) launched as this
//! binary launches them, over its cases.
//!
//! - The short convolution's prefill entry, and the delta rule's prefill
//!   stages of each schedule (the prepare and its scan, launched together:
//!   the prepare's workspace is read only by its scan, so their cases
//!   compare the scan's outputs). A chunk case carries one sequence's
//!   state entry from call to call, each call at its absolute position,
//!   and compares every output row and the final state. Batch units are
//!   sequences (outputs and final state) beside other sequences of other
//!   operands and phases. The anchor case writes each sequence's state at
//!   the last multiple of 64 it crosses into a spare entry, which must equal
//!   the slot state the same call cut there leaves: cold prompts from
//!   position 0 and continuations from a slot at other positions.
//! - The verify entries: a rows unit is one row of a verify sequence, its
//!   output then its checkpoint entry; alone, the sequence's rows run as a
//!   chain of one-row decode launches from the committed slot, each unit
//!   the row's output then the slot after it. Batch units are sequences.
//! - The gated norm: rows of one source, launched together, alone and in
//!   chunks.
//!
//! Every compared launch starts with poisoned outputs, a poisoned
//! workspace and poisoned write-only state entries (fresh slots, anchors,
//! checkpoints), and is checked afterwards (`written`, `written_over`):
//! every live output element and written entry holds finite values, and
//! every other byte of a pool kept its image.
//!
//! The negative control (`a_split_off_the_granule_changes_the_delta_rule
//! _gpu`) splits a prompt at row 37: the delta rule's bits must change,
//! which the chunk case at multiples of 64 would catch. The continuation
//! case (`a_continuation_cut_off_the_absolute_granule_changes_the_delta
//! _rule_gpu`) shows the granule is absolute: an item continued from
//! position 17 keeps its bits cut at absolute 2048 (2031 rows into the
//! item) and changes them cut 2048 rows into the item, where a model
//! cutting row parts by item rows would.

use super::*;
use crate::invariance::{Adapter, Compared, Element, Units, written, written_over};
use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::linear::Stage;
use kernels_sm120::linear::channel_decay::PrefillSchedule;
use std::ops::Range;

/// Rows of a chunk case's prompt: five chunks of 64.
const PROMPT_ROWS: u32 = 320;
const CHUNK: usize = 64;

/// Which launches an entry's cases run.
#[derive(Clone, Debug)]
enum Role {
    /// The short convolution's prefill entry.
    Conv,
    /// The delta rule's prefill stages of one schedule: its prepare, its
    /// scan and their cubin.
    Rule(Box<(Entry, Entry)>, String),
    ConvVerify,
    DeltaVerify,
    Norm,
}

fn staged_cubin() -> String {
    channel_decay::cubin_specs().remove(0).name
}

/// The role of a declared entry, or why it has none (a decode entry
/// claims nothing).
fn role(declared: &Declared) -> LabResult<Role> {
    let stages = |catalog: Vec<Entry>, cubin: String| -> Option<Role> {
        let of = |stage| catalog.iter().find(|e| e.stage == stage).cloned();
        let (prepare, scan) = (of(Stage::Prepare)?, of(Stage::Scan)?);
        [&prepare, &scan]
            .iter()
            .any(|e| e.symbol == declared.symbol)
            .then(|| Role::Rule(Box::new((prepare.clone(), scan.clone())), cubin))
    };
    let staged = channel_decay::catalog();
    if let Some(role) = stages(staged.clone(), staged_cubin()) {
        return Ok(role);
    }
    if let Some(role) = stages(
        channel_decay::tiled_catalog(),
        channel_decay::tiled_cubin_specs().remove(0).name,
    ) {
        return Ok(role);
    }
    if let Some(role) = stages(
        channel_decay::streamed_catalog(),
        format!("channel_decay_streamed_{}_sm120", SERVED_PROFILE.tag()),
    ) {
        return Ok(role);
    }
    let entry = staged
        .into_iter()
        .find(|e| e.symbol == declared.symbol)
        .ok_or_else(|| format!("{}: not a served channel decay entry", declared.symbol))?;
    match entry.query {
        Query::Norm(_) => Ok(Role::Norm),
        Query::Rule(query) => match query {
            PerChannelDecayDeltaRuleQuery::ShortConv { phase, .. } => match phase {
                RecurrentPhase::Prefill => Ok(Role::Conv),
                RecurrentPhase::Verify { .. } => Ok(Role::ConvVerify),
                RecurrentPhase::Decode => Err(format!("{}: claims nothing", entry.symbol)),
            },
            PerChannelDecayDeltaRuleQuery::DeltaRule { phase, .. } => match phase {
                RecurrentPhase::Verify { .. } => Ok(Role::DeltaVerify),
                _ => Err(format!("{}: claims nothing", entry.symbol)),
            },
        },
    }
}

/// `entry` resolved in `module`, planned for a bucket of `rows` and
/// `sequences`.
fn kernel_of(module: &Module, entry: Entry, rows: usize, sequences: usize) -> LabResult<Kernel> {
    let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
    let shared = entry.dynamic_shared_bytes();
    if shared > 0 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(|e| e.to_string())?;
    }
    Ok(Kernel {
        entry,
        function,
        bucket: Bucket {
            rows: rows.max(1) as u32,
            sequences: sequences.max(1) as u32,
        },
    })
}

/// What a launch runs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Part {
    Conv,
    Delta,
    Norm,
}

/// One launch's sequences.
#[derive(Clone, Debug, Default)]
struct Plan {
    lengths: Vec<usize>,
    positions: Vec<usize>,
    slots: Vec<i32>,
    initial: Vec<bool>,
    anchors: Vec<i32>,
    checkpoints: Vec<i32>,
}

impl Plan {
    /// One sequence of `rows` rows at `position` in entry 0, read first
    /// when `initial`.
    fn one(rows: usize, position: usize, initial: bool) -> Self {
        Self {
            lengths: vec![rows],
            positions: vec![position],
            slots: vec![0],
            initial: vec![initial],
            anchors: vec![-1],
            checkpoints: vec![],
        }
    }

    fn rows(&self) -> usize {
        self.lengths.iter().sum()
    }

    /// The entries `phase` writes without reading (fresh prefill slots,
    /// anchors that a call crosses, verify checkpoints) and those it reads
    /// and rewrites (prefill slots read first, decode slots).
    fn entries(&self, phase: RecurrentPhase) -> (Vec<usize>, Vec<usize>) {
        let (mut fresh, mut updated) = (Vec::new(), Vec::new());
        for (s, &len) in self.lengths.iter().enumerate() {
            let slot = self.slots[s] as usize;
            match phase {
                RecurrentPhase::Prefill => {
                    if self.initial[s] {
                        updated.push(slot);
                    } else {
                        fresh.push(slot);
                    }
                    if self.anchors[s] >= 0 && anchor_row(self.positions[s], len).is_some() {
                        fresh.push(self.anchors[s] as usize);
                    }
                }
                RecurrentPhase::Decode => updated.push(slot),
                RecurrentPhase::Verify { .. } => {
                    fresh.extend((0..len).map(|r| self.checkpoints[s * VERIFY_ROWS + r] as usize))
                }
            }
        }
        (fresh, updated)
    }
}

/// A launch's results: every output and both pools.
struct Out {
    conv_out: Vec<u8>,
    core: Vec<u8>,
    normed: Vec<u8>,
    conv_pool: Vec<u8>,
    delta_pool: Vec<u8>,
}

/// The byte ranges of `entries` of `entry` bytes.
fn entry_ranges(entries: &[usize], entry: usize) -> Vec<Range<usize>> {
    let mut ranges = entries
        .iter()
        .map(|&e| e * entry..(e + 1) * entry)
        .collect::<Vec<_>>();
    ranges.sort_by_key(|r| r.start);
    ranges.dedup();
    ranges
}

/// Checks a pool after a launch that wrote entries `fresh` (poisoned
/// before) and rewrote `updated` (holding data before): every element of
/// both is written and finite, every other byte kept its image.
fn pool_written(
    label: &str,
    before: &[u8],
    after: &[u8],
    (fresh, updated): (&[usize], &[usize]),
    entry: usize,
    element: Element,
) -> LabResult<()> {
    let mut poisoned = before.to_vec();
    for range in entry_ranges(updated, entry) {
        poisoned[range].fill(POISON);
    }
    let all = [fresh, updated].concat();
    let map = element
        .write_map()?
        .byte_ranges(entry_ranges(&all, entry))?;
    written_over(label, &poisoned, after, &map, element)
}

/// Launches `parts` of `phase` (each its kernels, in order) over the rows
/// of `call` as `plan` lays them out, on poisoned outputs, workspace and
/// fresh entries, and checks every write. `norm_input` holds the norm's
/// input rows.
fn launch(
    session: &Session,
    parts: &[(Part, &Kernel)],
    phase: RecurrentPhase,
    call: &Case,
    plan: &Plan,
    norm_input: Option<&[u16]>,
) -> LabResult<Out> {
    let rows = plan.rows();
    let sequences = plan.lengths.len();
    let (fresh, updated) = plan.entries(phase);
    let mut call = Case {
        input: call.input.clone(),
        weights: call.weights.clone(),
        conv_pool: call.conv_pool.clone(),
        qkv: call.qkv.clone(),
        f: call.f.clone(),
        b: call.b.clone(),
        a_log: call.a_log.clone(),
        dt_bias: call.dt_bias.clone(),
        delta_pool: call.delta_pool.clone(),
        z: call.z.clone(),
        norm_weight: call.norm_weight.clone(),
    };
    let conv = parts.iter().any(|(part, _)| *part == Part::Conv);
    let delta = parts.iter().any(|(part, _)| *part == Part::Delta);
    let poison_u16 = u16::from_ne_bytes([POISON; 2]);
    if conv {
        for range in entry_ranges(&fresh, CONV_ENTRY) {
            call.conv_pool[range].fill(poison_u16);
        }
    }
    if delta {
        let poison_f32 = f32::from_ne_bytes([POISON; 4]);
        for range in entry_ranges(&fresh, STATE) {
            call.delta_pool[range].fill(poison_f32);
        }
    }
    let device = Device::new(session, &call, rows)?;
    poison::poison(session, &[&device.conv_out, &device.normed])?;
    match norm_input {
        Some(input) => device
            .core
            .copy_from_bytes(&bytes_u16(input), &session.stream)
            .map_err(|e| e.to_string())?,
        None => poison::poison(session, &[&device.core])?,
    }
    let workspace = parts
        .iter()
        .map(|(_, kernel)| kernel.workspace_bytes())
        .max()
        .unwrap_or(0);
    let meta = Metadata::new(
        session,
        &Batch {
            lengths: &plan.lengths,
            positions: &plan.positions,
            slots: &plan.slots,
            initial: &plan.initial,
            anchors: &plan.anchors,
            checkpoints: &plan.checkpoints,
        },
        workspace,
    )?;
    poison::poison(session, &[&meta.workspace])?;
    for (part, kernel) in parts {
        let binds = match part {
            Part::Conv => device.conv(),
            Part::Delta => device.delta(false),
            Part::Norm => device.norm(),
        };
        let (live_rows, live_sequences) = match part {
            Part::Norm => (rows, rows),
            _ => (rows, sequences),
        };
        kernel.launch(
            session,
            &session.stream,
            &meta.bind(binds, phase),
            live_rows,
            live_sequences,
            true,
        )?;
    }
    let out = Out {
        conv_out: download(session, &device.conv_out)?,
        core: download(session, &device.core)?,
        normed: download(session, &device.normed)?,
        conv_pool: download(session, &device.conv_state)?,
        delta_pool: download(session, &device.delta_state)?,
    };
    let bf16 = |n: usize| WriteMap::of(DType::Bf16).prefix(n);
    if conv {
        written(
            "conv output",
            &out.conv_out,
            &bf16(rows * CHANNELS)?,
            Element::Bf16,
        )?;
        pool_written(
            "conv pool",
            &bytes_u16(&call.conv_pool),
            &out.conv_pool,
            (&fresh, &updated),
            CONV_ENTRY * 2,
            Element::Bf16,
        )?;
    }
    if delta {
        written(
            "delta output",
            &out.core,
            &bf16(rows * WIDTH)?,
            Element::Bf16,
        )?;
        pool_written(
            "delta pool",
            &bytes_f32(&call.delta_pool),
            &out.delta_pool,
            (&fresh, &updated),
            STATE * 4,
            Element::F32,
        )?;
    }
    if parts.iter().any(|(part, _)| *part == Part::Norm) {
        written(
            "norm output",
            &out.normed,
            &bf16(rows * WIDTH)?,
            Element::Bf16,
        )?;
    }
    Ok(out)
}

/// A case over rows `rows` of `source` (each row's inputs, gates and
/// operands), with `source`'s weights and pools.
fn rows_of(source: &Case, rows: &[usize]) -> Case {
    let take = |data: &[u16], stride: usize| {
        rows.iter()
            .flat_map(|&r| data[r * stride..(r + 1) * stride].to_vec())
            .collect::<Vec<_>>()
    };
    Case {
        input: take(&source.input, INPUT_STRIDE),
        weights: source.weights.clone(),
        conv_pool: source.conv_pool.clone(),
        qkv: take(&source.qkv, INPUT_STRIDE),
        f: take(&source.f, F_STRIDE),
        b: take(&source.b, B_STRIDE),
        a_log: source.a_log.clone(),
        dt_bias: source.dt_bias.clone(),
        delta_pool: source.delta_pool.clone(),
        z: take(&source.z, Z_STRIDE),
        norm_weight: source.norm_weight.clone(),
    }
}

/// Rows of a launch's output of `row` bytes, one unit each.
fn row_units(bytes: &[u8], rows: Range<usize>, row: usize) -> Units {
    rows.map(|r| bytes[r * row..(r + 1) * row].to_vec())
        .collect()
}

/// Entry `entry` of a pool of `bytes` per entry.
fn entry_bytes(pool: &[u8], entry: usize, bytes: usize) -> Vec<u8> {
    pool[entry * bytes..(entry + 1) * bytes].to_vec()
}

/// A prefill part's launches and what they compare.
struct Prefill {
    module: Module,
    part: Part,
    /// The entries, in launch order.
    entries: Vec<Entry>,
}

impl Prefill {
    fn of(session: &Session, role: &Role) -> LabResult<Self> {
        match role {
            Role::Conv => Ok(Self {
                module: module(session)?,
                part: Part::Conv,
                entries: vec![find("conv_prefill")],
            }),
            Role::Rule(stages, cubin) => Ok(Self {
                module: session.module(cubin)?,
                part: Part::Delta,
                entries: vec![stages.0.clone(), stages.1.clone()],
            }),
            _ => Err("not a prefill role".into()),
        }
    }

    fn run(&self, session: &Session, call: &Case, plan: &Plan) -> LabResult<Out> {
        let kernels = self
            .entries
            .iter()
            .map(|e| kernel_of(&self.module, e.clone(), plan.rows(), plan.lengths.len()))
            .collect::<LabResult<Vec<_>>>()?;
        let parts = kernels.iter().map(|k| (self.part, k)).collect::<Vec<_>>();
        launch(session, &parts, RecurrentPhase::Prefill, call, plan, None)
    }

    /// The output row bytes, and the pool and entry bytes, it compares.
    fn output<'o>(&self, out: &'o Out) -> (&'o [u8], usize, &'o [u8], usize) {
        match self.part {
            Part::Conv => (&out.conv_out, CHANNELS * 2, &out.conv_pool, CONV_ENTRY * 2),
            _ => (&out.core, WIDTH * 2, &out.delta_pool, STATE * 4),
        }
    }
}

/// `pool`'s entry 0 replaced by `entry`, of `bytes` per entry.
fn with_entry0(case: &mut Case, part: Part, entry: &[u8]) {
    match part {
        Part::Conv => case.conv_pool[..CONV_ENTRY].copy_from_slice(&u16s(entry)),
        _ => case.delta_pool[..STATE].copy_from_slice(&f32s(entry)),
    }
}

/// A seed for a case of `seeds`, the same for the same seeds.
fn seed_of(seeds: &[u64]) -> u64 {
    seeds
        .iter()
        .fold(0x5eed_u64, |seed, &s| seed.rotate_left(7) ^ s)
}

/// The decode convolution or delta rule, launched one row at a time.
fn decode_kernel(session: &Session, part: Part) -> LabResult<(Module, Entry)> {
    let name = match part {
        Part::Conv => "conv_decode",
        _ => "delta_decode",
    };
    Ok((module(session)?, find(name)))
}

/// The anchor case's sequences, (rows, absolute position, read the slot
/// first): cold prompts from position 0, then eight continuations from a
/// slot at other positions that cross a multiple of 64, as chunked prefill
/// launches them; a cold prompt ending on a multiple of 64 and a
/// continuation crossing none write no anchor.
const ANCHOR_SEQUENCES: [(usize, usize, bool); 14] = [
    (65, 0, false),
    (130, 0, false),
    (CHUNK * 5 + 63, 0, false),
    (200, 0, false),
    (128, 0, false),
    (70, 64, true),
    (50, 100, true),
    (200, 1, true),
    (2, 63, true),
    (130, 130, true),
    (70, 192, true),
    (60, 17, true),
    (100, 256, true),
    (10, 70, true),
];

pub struct ChannelDecay;

impl ChannelDecay {
    /// A prompt of `rows` rows of one sequence from position 0, in calls
    /// ending at `ends` that carry its state entry: every output row, then
    /// the final state.
    fn prefill_chunks(
        &self,
        session: &Session,
        role: &Role,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        self.prefill_calls(session, role, 0, rows, ends)
    }

    /// [`Self::prefill_chunks`] for a sequence whose first row sits at
    /// absolute position `position`: past 0 the first call continues from
    /// the slot's state, as a later chunk of a prompt does.
    fn prefill_calls(
        &self,
        session: &Session,
        role: &Role,
        position: usize,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let prefill = Prefill::of(session, role)?;
        let source = Case::new(0xc0de, rows as usize, 2);
        let mut units = Vec::new();
        let mut state: Option<Vec<u8>> = None;
        let mut start = 0;
        for &end in ends {
            let range = start as usize..end as usize;
            let mut call = rows_of(&source, &range.clone().collect::<Vec<_>>());
            if let Some(state) = &state {
                with_entry0(&mut call, prefill.part, state);
            }
            let plan = Plan::one(
                range.len(),
                position + range.start,
                state.is_some() || position > 0,
            );
            let out = prefill.run(session, &call, &plan)?;
            let (output, row, pool, entry) = prefill.output(&out);
            units.extend(row_units(output, 0..range.len(), row));
            state = Some(entry_bytes(pool, 0, entry));
            start = end;
        }
        units.push(state.ok_or("no call")?);
        Ok(units)
    }

    /// Four prefill sequences of other lengths and phases, each of its own
    /// seed's operands and state: each sequence's output rows and final
    /// state, concatenated.
    fn prefill_batch(&self, session: &Session, role: &Role, seeds: &[u64]) -> LabResult<Units> {
        let prefill = Prefill::of(session, role)?;
        let lengths = [70, 64, 130, 5, 64, 33];
        let positions = [0, 64, 17, 130, 3, 128];
        let n = seeds.len();
        let (lengths, positions) = (&lengths[..n], &positions[..n]);
        let cases = seeds
            .iter()
            .zip(lengths)
            .map(|(&seed, &len)| Case::new(seed, len, 1))
            .collect::<Vec<_>>();
        let mut call = Case::new(seeds[0], lengths.iter().sum(), n);
        let mut at = 0;
        let mut all = rows_of(&cases[0], &[]);
        for (s, case) in cases.iter().enumerate() {
            let one = rows_of(case, &(0..lengths[s]).collect::<Vec<_>>());
            all.input.extend(one.input);
            all.qkv.extend(one.qkv);
            all.f.extend(one.f);
            all.b.extend(one.b);
            call.conv_pool[s * CONV_ENTRY..(s + 1) * CONV_ENTRY]
                .copy_from_slice(&case.conv_pool[..CONV_ENTRY]);
            call.delta_pool[s * STATE..(s + 1) * STATE].copy_from_slice(&case.delta_pool[..STATE]);
            at += lengths[s];
        }
        debug_assert_eq!(at, lengths.iter().sum::<usize>());
        call.input = all.input;
        call.qkv = all.qkv;
        call.f = all.f;
        call.b = all.b;
        let plan = Plan {
            lengths: lengths.to_vec(),
            positions: positions.to_vec(),
            slots: (0..n as i32).collect(),
            initial: (0..n).map(|s| s % 2 == 1).collect(),
            anchors: vec![-1; n],
            checkpoints: vec![],
        };
        let out = prefill.run(session, &call, &plan)?;
        let (output, row, pool, entry) = prefill.output(&out);
        let starts = cu(lengths);
        Ok((0..n)
            .map(|s| {
                let rows = starts[s] as usize..starts[s + 1] as usize;
                let mut unit = output[rows.start * row..rows.end * row].to_vec();
                unit.extend(entry_bytes(pool, s, entry));
                unit
            })
            .collect())
    }

    /// Sequences whose prompts cross multiples of 64 write their state at
    /// the last one into anchor entries: each against the slot a prefill of
    /// just those rows leaves.
    fn prefill_anchors(&self, session: &Session, role: &Role) -> LabResult<Vec<Compared>> {
        let prefill = Prefill::of(session, role)?;
        let sequences = ANCHOR_SEQUENCES;
        let n = sequences.len();
        let lengths = sequences.map(|(len, _, _)| len);
        let rows = lengths.iter().sum::<usize>();
        let source = Case::new(0xa9c, rows, 2 * n);
        let plan = Plan {
            lengths: lengths.to_vec(),
            positions: sequences.iter().map(|&(_, p, _)| p).collect(),
            slots: (0..n as i32).collect(),
            initial: sequences.iter().map(|&(_, _, i)| i).collect(),
            anchors: (n as i32..2 * n as i32).collect(),
            checkpoints: vec![],
        };
        let before = match prefill.part {
            Part::Conv => bytes_u16(&source.conv_pool),
            _ => bytes_f32(&source.delta_pool),
        };
        let out = prefill.run(session, &source, &plan)?;
        let (_, _, pool, entry) = prefill.output(&out);
        let starts = cu(&lengths);
        let mut compared = Vec::new();
        let mut continued = 0;
        for (s, &(len, position, initial)) in sequences.iter().enumerate() {
            // A call ending on a multiple of 64, or crossing none, writes no
            // anchor; the write check holds its entry unchanged.
            let Some(anchor) = anchor_row(position, len) else {
                continue;
            };
            continued += usize::from(initial);
            let first = starts[s] as usize;
            let mut call = rows_of(&source, &(first..first + anchor).collect::<Vec<_>>());
            // The same call cut at the anchor, from the same slot state.
            with_entry0(&mut call, prefill.part, &entry_bytes(&before, s, entry));
            let alone = prefill.run(session, &call, &Plan::one(anchor, position, initial))?;
            let (_, _, alone_pool, _) = prefill.output(&alone);
            compared.push(Compared {
                label: format!(
                    "sequence {s} of {len} rows at position {position}: anchor at row {anchor}"
                ),
                got: entry_bytes(pool, n + s, entry),
                expected: entry_bytes(alone_pool, 0, entry),
            });
        }
        if continued != 8 {
            return Err(format!(
                "the anchor case continues {continued} sequences across an anchor, not 8"
            ));
        }
        Ok(compared)
    }

    /// Verify sequences of `lengths` from slots holding each sequence's
    /// state: per row its output then its checkpoint, or, `alone`, per row
    /// its one-row decode output then the slot after it, the decode steps
    /// chained from the committed slot.
    fn verify(
        &self,
        session: &Session,
        part: Part,
        lengths: &[usize],
        cases: &[Case],
        alone: bool,
    ) -> LabResult<Vec<Units>> {
        let n = lengths.len();
        let (row, entry) = match part {
            Part::Conv => (CHANNELS * 2, CONV_ENTRY * 2),
            _ => (WIDTH * 2, STATE * 4),
        };
        let parts_of = |out: &Out| -> (Vec<u8>, Vec<u8>) {
            match part {
                Part::Conv => (out.conv_out.clone(), out.conv_pool.clone()),
                _ => (out.core.clone(), out.delta_pool.clone()),
            }
        };
        let entries = n + n * VERIFY_ROWS;
        let mut call = Case::new(seed_of(&[n as u64]), lengths.iter().sum(), entries);
        let mut all = rows_of(&cases[0], &[]);
        for (s, case) in cases.iter().enumerate() {
            let one = rows_of(case, &(0..lengths[s]).collect::<Vec<_>>());
            all.input.extend(one.input);
            all.qkv.extend(one.qkv);
            all.f.extend(one.f);
            all.b.extend(one.b);
            call.conv_pool[s * CONV_ENTRY..(s + 1) * CONV_ENTRY]
                .copy_from_slice(&case.conv_pool[..CONV_ENTRY]);
            call.delta_pool[s * STATE..(s + 1) * STATE].copy_from_slice(&case.delta_pool[..STATE]);
        }
        call.input = all.input;
        call.qkv = all.qkv;
        call.f = all.f;
        call.b = all.b;
        let starts = cu(lengths);
        if !alone {
            let verify = RecurrentPhase::Verify {
                max_rows: VERIFY_ROWS as u32,
            };
            let name = match part {
                Part::Conv => "conv_verify",
                _ => "delta_verify",
            };
            let kernel = kernel_of(&module(session)?, find(name), call.f.len() / F_STRIDE, n)?;
            let plan = Plan {
                lengths: lengths.to_vec(),
                positions: vec![0; n],
                slots: (0..n as i32).collect(),
                initial: vec![true; n],
                anchors: vec![-1; n],
                checkpoints: (0..(n * VERIFY_ROWS) as i32)
                    .map(|i| n as i32 + i)
                    .collect(),
            };
            let out = launch(session, &[(part, &kernel)], verify, &call, &plan, None)?;
            let (output, pool) = parts_of(&out);
            return Ok((0..n)
                .map(|s| {
                    (0..lengths[s])
                        .map(|r| {
                            let at = starts[s] as usize + r;
                            let mut unit = output[at * row..(at + 1) * row].to_vec();
                            let id = plan.checkpoints[s * VERIFY_ROWS + r] as usize;
                            unit.extend(entry_bytes(&pool, id, entry));
                            unit
                        })
                        .collect()
                })
                .collect());
        }
        let (module, decode) = decode_kernel(session, part)?;
        let kernel = kernel_of(&module, decode, 1, 1)?;
        let mut units = Vec::new();
        for s in 0..n {
            let mut state = entry_bytes(
                &match part {
                    Part::Conv => bytes_u16(&call.conv_pool),
                    _ => bytes_f32(&call.delta_pool),
                },
                s,
                entry,
            );
            let mut sequence = Vec::new();
            for r in 0..lengths[s] {
                // One state entry: the slot the step advances.
                let mut step = rows_of(&call, &[starts[s] as usize + r]);
                step.conv_pool.truncate(CONV_ENTRY);
                step.delta_pool.truncate(STATE);
                with_entry0(&mut step, part, &state);
                let out = launch(
                    session,
                    &[(part, &kernel)],
                    RecurrentPhase::Decode,
                    &step,
                    &Plan::one(1, 0, true),
                    None,
                )?;
                let (output, pool) = parts_of(&out);
                state = entry_bytes(&pool, 0, entry);
                let mut unit = output[..row].to_vec();
                unit.extend(&state);
                sequence.push(unit);
            }
            units.push(sequence);
        }
        Ok(units)
    }

    /// Verify sequences of up to `VERIFY_ROWS` rows holding `count` rows,
    /// each of its own seed's operands and state.
    fn verify_rows(
        &self,
        session: &Session,
        part: Part,
        seeds: &[u64],
        alone: bool,
    ) -> LabResult<Units> {
        let count = seeds.len();
        let lengths = (0..count.div_ceil(VERIFY_ROWS))
            .map(|s| VERIFY_ROWS.min(count - s * VERIFY_ROWS))
            .collect::<Vec<_>>();
        let seed = seed_of(seeds);
        let cases = lengths
            .iter()
            .enumerate()
            .map(|(s, &len)| Case::new(seed ^ s as u64, len, 1))
            .collect::<Vec<_>>();
        Ok(self
            .verify(session, part, &lengths, &cases, alone)?
            .into_iter()
            .flatten()
            .collect())
    }

    /// Rows `order` of the norm's source, in one launch.
    fn norm(&self, session: &Session, order: &[usize]) -> LabResult<Units> {
        let source = Case::new(0x0a11, 512, 1);
        let mut call = rows_of(&source, order);
        // The norm reads the delta rule's output rows: the source's qkv.
        let input = order
            .iter()
            .flat_map(|&r| source.qkv[r * INPUT_STRIDE..r * INPUT_STRIDE + WIDTH].to_vec())
            .collect::<Vec<_>>();
        call.qkv = vec![0; order.len() * INPUT_STRIDE];
        let kernel = kernel_of(
            &module(session)?,
            find("gated_norm"),
            order.len(),
            order.len(),
        )?;
        let plan = Plan {
            lengths: vec![1; order.len()],
            positions: vec![0; order.len()],
            slots: vec![0; order.len()],
            initial: vec![true; order.len()],
            anchors: vec![-1; order.len()],
            checkpoints: vec![],
        };
        let out = launch(
            session,
            &[(Part::Norm, &kernel)],
            RecurrentPhase::Decode,
            &call,
            &plan,
            Some(&input),
        )?;
        Ok(row_units(&out.normed, 0..order.len(), WIDTH * 2))
    }
}

impl Adapter for ChannelDecay {
    fn kind(&self) -> OpKind {
        OpKind::ChannelDecay
    }

    fn prompt_rows(&self, entry: &Declared) -> u32 {
        match role(entry) {
            Ok(Role::Norm) => 64,
            _ => PROMPT_ROWS,
        }
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let role = role(entry)?;
        match role {
            Role::Conv | Role::Rule(..) => self.prefill_chunks(&sessions[0], &role, rows, ends),
            Role::Norm => {
                let mut units = Vec::new();
                let mut start = 0;
                for &end in ends {
                    let order = (start as usize..end as usize).collect::<Vec<_>>();
                    units.extend(self.norm(&sessions[0], &order)?);
                    start = end;
                }
                Ok(units)
            }
            _ => Err(format!("{}: no chunk case", entry.symbol)),
        }
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        match role(entry)? {
            Role::Norm => self.norm(&sessions[0], &seed_rows(seeds)),
            Role::ConvVerify => self.verify_rows(&sessions[0], Part::Conv, seeds, false),
            Role::DeltaVerify => self.verify_rows(&sessions[0], Part::Delta, seeds, false),
            _ => Err(format!("{}: no rows case", entry.symbol)),
        }
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        match role(entry)? {
            Role::Norm => seed_rows(seeds)
                .into_iter()
                .map(|row| {
                    self.norm(&sessions[0], &[row])?
                        .pop()
                        .ok_or_else(|| "no unit".to_owned())
                })
                .collect(),
            Role::ConvVerify => self.verify_rows(&sessions[0], Part::Conv, seeds, true),
            Role::DeltaVerify => self.verify_rows(&sessions[0], Part::Delta, seeds, true),
            _ => Err(format!("{}: no rows case", entry.symbol)),
        }
    }

    fn batch_units(&self, _entry: &Declared) -> usize {
        4
    }

    fn batch(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let role = role(entry)?;
        match role {
            Role::Conv | Role::Rule(..) => self.prefill_batch(&sessions[0], &role, seeds),
            Role::ConvVerify | Role::DeltaVerify => {
                let part = if matches!(role, Role::ConvVerify) {
                    Part::Conv
                } else {
                    Part::Delta
                };
                let lengths = [VERIFY_ROWS, 3, VERIFY_ROWS, 1, 2, 5];
                let lengths = &lengths[..seeds.len()];
                let cases = seeds
                    .iter()
                    .zip(lengths)
                    .map(|(&seed, &len)| Case::new(seed, len, 1))
                    .collect::<Vec<_>>();
                Ok(self
                    .verify(&sessions[0], part, lengths, &cases, false)?
                    .into_iter()
                    .map(|rows| rows.concat())
                    .collect())
            }
            Role::Norm => self.norm(&sessions[0], &seed_rows(seeds)),
        }
    }

    fn anchors(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        let role = role(entry)?;
        match role {
            Role::Conv | Role::Rule(..) => self.prefill_anchors(&sessions[0], &role),
            _ => Err(format!("{}: no anchor case", entry.symbol)),
        }
    }
}

/// Distinct rows of the norm's source of 512 rows for `seeds`.
fn seed_rows(seeds: &[u64]) -> Vec<usize> {
    let mut taken = std::collections::BTreeSet::new();
    seeds
        .iter()
        .map(|&seed| {
            let mut row = (seed % 512) as usize;
            while !taken.insert(row) {
                row = (row + 1) % 512;
            }
            row
        })
        .collect()
}

/// Every declared entry that claims something has a role; the roles cover
/// each schedule's prefill stages, both verify entries, the convolution
/// and the norm; the anchor case's prompts write and skip anchors (no GPU).
#[test]
fn channel_decay_declarations_have_roles() {
    let declared = kernel_lab::declared::channel_decay();
    let mut rules = std::collections::BTreeSet::new();
    let mut others = Vec::new();
    for entry in &declared {
        match role(entry) {
            Ok(Role::Rule(stages, cubin)) => {
                let (prepare, scan) = *stages;
                assert!(
                    kernel_lab::cubin_entries(&cubin)
                        .unwrap()
                        .contains(&prepare.symbol.as_str())
                );
                assert!(
                    kernel_lab::cubin_entries(&cubin)
                        .unwrap()
                        .contains(&scan.symbol.as_str())
                );
                rules.insert(format!("{:?}", prepare.schedule));
            }
            Ok(role) => others.push(format!("{role:?}")),
            Err(_) => assert!(entry.invariance.claims().is_empty(), "{}", entry.symbol),
        }
    }
    for schedule in [
        PrefillSchedule::Staged,
        PrefillSchedule::Tiled,
        PrefillSchedule::Streamed,
    ] {
        assert!(rules.contains(&format!("{schedule:?}")), "{rules:?}");
    }
    for wanted in ["Conv", "ConvVerify", "DeltaVerify", "Norm"] {
        assert!(others.iter().any(|o| o == wanted), "{wanted}: {others:?}");
    }
    assert_eq!(anchor_row(0, 128), None);
    assert_eq!(anchor_row(0, 200), Some(192));
    // The anchor case crosses an anchor in eight continuations, and writes
    // none for one cold prompt and one continuation.
    let crossing = |continued: bool| {
        ANCHOR_SEQUENCES
            .iter()
            .filter(|&&(len, position, initial)| {
                initial == continued && anchor_row(position, len).is_some()
            })
            .count()
    };
    assert_eq!((crossing(false), crossing(true)), (4, 8));
    assert_eq!(
        ANCHOR_SEQUENCES
            .iter()
            .filter(|&&(len, position, _)| anchor_row(position, len).is_none())
            .count(),
        2
    );
}

/// A split at row 37, off the 64-row granule, changes the delta rule's
/// bits (one chunk regroups), which the chunk case would catch; the split
/// at 64 keeps them.
#[test]
fn a_split_off_the_granule_changes_the_delta_rule_gpu() {
    gpu_test(4 << 30, |session| {
        let role = Role::Rule(
            Box::new((find("delta_prefill_prepare"), find("delta_prefill_scan"))),
            staged_cubin(),
        );
        let adapter = ChannelDecay;
        let whole = adapter.prefill_chunks(session, &role, PROMPT_ROWS, &[PROMPT_ROWS])?;
        for (split, equal) in [(64, true), (37, false)] {
            let units =
                adapter.prefill_chunks(session, &role, PROMPT_ROWS, &[split, PROMPT_ROWS])?;
            let differing = units.iter().zip(&whole).filter(|(a, b)| a != b).count();
            eprintln!(
                "split at {split}: {differing} of {} units differ",
                whole.len()
            );
            if (differing == 0) != equal {
                return Err(format!("split at {split}: {differing} units differ"));
            }
        }
        Ok(())
    })
    .expect("the delta rule chunk negative control");
}

/// A model that cuts a long prefill into row parts of 2048 item rows cuts
/// at item-relative multiples of 64. The delta rule chunks at absolute
/// multiples of 64, so for an item continued from position 17 that cut
/// lands at absolute 2065, inside the chunk [2048, 2112): there the delta
/// rule's bits must change under every schedule, while the cut at the
/// chunk's start, absolute 2048 (2031 item rows), keeps them, and the
/// convolution keeps its bits at both. The declaration claims the absolute granule only; this case
/// shows such a cut off it is not covered.
#[test]
fn a_continuation_cut_off_the_absolute_granule_changes_the_delta_rule_gpu() {
    const POSITION: usize = 17;
    const ROWS: u32 = 3000;
    const PIECE: u32 = 2048;
    gpu_test(8 << 30, |session| {
        let mut roles = vec![(Role::Conv, "the convolution".to_owned())];
        for entry in kernel_lab::declared::channel_decay() {
            if let Ok(Role::Rule(stages, cubin)) = role(&entry)
                && entry.symbol == stages.0.symbol
            {
                let label = format!("the {:?} delta rule", stages.0.schedule);
                roles.push((Role::Rule(stages, cubin), label));
            }
        }
        if roles.len() != 4 {
            return Err(format!(
                "{} roles, not the convolution and three schedules",
                roles.len()
            ));
        }
        let adapter = ChannelDecay;
        let absolute = ((POSITION as u32 + PIECE) / CHUNK as u32 * CHUNK as u32) - POSITION as u32;
        for (role, label) in &roles {
            let whole = adapter.prefill_calls(session, role, POSITION, ROWS, &[ROWS])?;
            let item_relative = matches!(role, Role::Conv);
            for (cut, equal) in [(absolute, true), (PIECE, item_relative)] {
                let units = adapter.prefill_calls(session, role, POSITION, ROWS, &[cut, ROWS])?;
                let differing = units.iter().zip(&whole).filter(|(a, b)| a != b).count();
                eprintln!(
                    "{label} from position {POSITION}, cut after {cut} rows (absolute {}): \
                     {differing} of {} units differ",
                    POSITION as u32 + cut,
                    whole.len()
                );
                if (differing == 0) != equal {
                    return Err(format!(
                        "{label}: cut after {cut} rows (absolute {}), {differing} units differ",
                        POSITION as u32 + cut
                    ));
                }
            }
        }
        Ok(())
    })
    .expect("the delta rule continuation cut");
}
