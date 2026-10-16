use runtime_engine::{DraftSource, DraftWork, ModelBatchItem, RequestId, SamplingWork};
use runtime_sampling::{LogitBias, SamplingParameters, SamplingState};
use runtime_scheduler::BatchPhase;
use serde::{Deserialize, Serialize};

/// Trace version of a teacher-forced decode over prompts of mixed lengths.
/// Leading steps prefill whole prompts; every later step decodes one forced
/// token for each listed request. A request prefilled but absent from the
/// decode steps stays live and idle, holding its state slot and KV pages.
pub const MIXED_DECODE: u32 = 5;
/// Steps of a mixed decode trace, prefill and decode together.
const MIXED_STEPS: usize = 128;
/// Requests of a mixed decode trace, and rows of one of its steps: the most
/// live sequences a package may declare.
const MIXED_REQUESTS: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub version: u32,
    pub concurrency: usize,
    pub vocabulary: usize,
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub mode: Mode,
    pub items: Vec<Item>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Item {
    pub slot: usize,
    pub prefill: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub anchor: bool,
    pub inputs: Vec<u32>,
    pub drafts: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fixed_drafts: Vec<u32>,
    pub kept: usize,
    pub history: Vec<u32>,
    pub emitted: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Greedy,
    Seeded,
    Filtered,
    Grammar,
    Top,
    Mixed,
}

impl Trace {
    /// An explicit input is never regenerated, including for a baseline arm.
    pub fn select(
        path: &std::path::Path,
        supplied: bool,
        generate: impl FnOnce() -> super::diagnostic::Result<Self>,
    ) -> super::diagnostic::Result<Self> {
        super::diagnostic::external(path)?;
        let trace: Self = if supplied {
            serde_json::from_value(super::diagnostic::read_json(path, 16 << 20)?)?
        } else {
            generate()?
        };
        trace.validate()?;
        if !supplied {
            super::diagnostic::write_json(path, &serde_json::to_value(&trace)?, 16 << 20)?;
        }
        Ok(trace)
    }

    /// Fixed-depth wide verification over the same histories and sampling
    /// modes as the original replay. Every step rejects down to one row.
    pub fn construct_fixed_depth(prompts: Vec<Vec<u32>>, vocabulary: usize, depth: usize) -> Self {
        let mut trace = Self::construct(prompts, vocabulary, true);
        for step in trace.steps.iter_mut().skip(1) {
            for item in &mut step.items {
                item.drafts = depth;
            }
        }
        trace
    }

    pub fn construct(prompts: Vec<Vec<u32>>, vocabulary: usize, verify: bool) -> Self {
        let concurrency = prompts.len();
        let mut histories = prompts;
        let mut steps = vec![Step {
            mode: Mode::Greedy,
            items: histories
                .iter()
                .enumerate()
                .map(|(slot, prompt)| Item {
                    slot,
                    prefill: true,
                    anchor: false,
                    inputs: prompt.clone(),
                    drafts: 0,
                    fixed_drafts: Vec::new(),
                    kept: prompt.len(),
                    history: prompt.clone(),
                    emitted: 0,
                })
                .collect(),
        }];
        for (mode_index, mode) in [
            Mode::Greedy,
            Mode::Seeded,
            Mode::Filtered,
            Mode::Grammar,
            Mode::Top,
            Mode::Mixed,
        ]
        .into_iter()
        .enumerate()
        {
            for repeat in 0..2 {
                let emitted = mode_index * 2 + repeat + 1;
                let mut items = Vec::new();
                for position in 0..concurrency {
                    let slot = (position + emitted) % concurrency;
                    // Replay an existing prompt token, never a sampled output.
                    let token = histories[slot][(emitted * 7 + slot) % 16];
                    histories[slot].push(token);
                    let drafts = if !verify {
                        0
                    } else if repeat == 0 {
                        1
                    } else if position < concurrency / 2 {
                        2
                    } else {
                        0
                    };
                    items.push(Item {
                        slot,
                        prefill: false,
                        anchor: false,
                        inputs: vec![token],
                        drafts,
                        fixed_drafts: Vec::new(),
                        kept: 1,
                        history: histories[slot].clone(),
                        emitted,
                    });
                }
                steps.push(Step { mode, items });
            }
        }
        Self {
            version: 1,
            concurrency,
            vocabulary,
            steps,
        }
    }

    /// Full and partial verify steps whose bundle-generated proposals are recorded.
    pub fn construct_verify(
        prompts: Vec<Vec<u32>>,
        vocabulary: usize,
        rows_per_item: usize,
    ) -> Result<Self, String> {
        if !(2..=8).contains(&rows_per_item) {
            return Err("verify rows per item must be in 2..=8".into());
        }
        if !(1..=32).contains(&prompts.len()) || prompts.iter().any(|p| p.len() < 16) {
            return Err("verify trace requires 1..=32 prompts of at least 16 tokens".into());
        }
        let mut trace = Self::construct(prompts, vocabulary, false);
        trace.version = 3;
        for (index, step) in trace.steps.iter_mut().enumerate().skip(1) {
            for item in &mut step.items {
                item.drafts = rows_per_item - 1 - usize::from(index % 2 == 0);
            }
        }
        trace.validate()?;
        Ok(trace)
    }

    /// Upper bound, in bytes, of the frame lines a replay of this trace
    /// writes (`record::Frame` as JSON, one line per step; the identity line
    /// is counted by the caller). Row evidence dominates: every row of a
    /// grammar item carries its constraint mask of `vocabulary / 32` words,
    /// and a verify item has one row per draft plus one, so a drafter trace
    /// writes up to eight masks per request and grammar step. Every u32 (a
    /// mask word, token or page entry) serializes to at most 10 digits and a
    /// separator, and every request byte to at most three digits and a
    /// separator. The fixed allowances cover each row's sampling
    /// parameters, dispatch, logits geometry and sample bits with up to 20
    /// top logprobs, and each item's wire header and sampling fields; C16
    /// and C32 drafter replays with 152,576- and 248,320-token vocabularies
    /// measured about 1.2 KiB of row evidence outside masks and histories.
    pub fn metadata_bound(&self) -> u64 {
        const NUMBER: u64 = 11;
        const REQUEST_BYTE: u64 = 4;
        const ROW_FIXED: u64 = 4096;
        const ITEM_WIRE_FIXED: u64 = 1024;
        const FRAME_FIXED: u64 = 4096;
        let mask = NUMBER * (self.vocabulary as u64).div_ceil(32) + 16;
        self.steps
            .iter()
            .map(|step| {
                // Only grammar steps constrain their rows; mixed steps
                // sample seeded and filtered rows.
                let grammar = step.mode == Mode::Grammar;
                step.items
                    .iter()
                    .map(|item| {
                        let rows = 1 + item.drafts as u64;
                        // A row's history extends by at most the item's drafts.
                        let tokens = (item.history.len() + item.drafts + 1) as u64;
                        let row = ROW_FIXED + NUMBER * tokens + if grammar { mask } else { 0 };
                        // Inputs, drafts and at most one page entry per token.
                        let wire = ITEM_WIRE_FIXED
                            + 4 * (item.inputs.len() + item.drafts) as u64
                            + 4 * tokens;
                        rows * row
                            + REQUEST_BYTE * wire
                            + 2 * NUMBER * 3
                            + NUMBER * (item.drafts as u64 + 4)
                    })
                    .sum::<u64>()
                    + FRAME_FIXED
            })
            .sum()
    }

    /// Capacity for an instance may exceed the number of live trace requests.
    pub fn verify_capacity(&self, max_sequences: usize) -> Result<usize, String> {
        if max_sequences < self.concurrency || max_sequences > 32 {
            return Err("served capacity must cover the trace and be at most 32".into());
        }
        let width = self
            .steps
            .iter()
            .skip(1)
            .flat_map(|step| &step.items)
            .map(|item| item.drafts + 1)
            .max()
            .ok_or("empty verify trace")?;
        Ok(max_sequences * width)
    }

    pub fn matches_verify_rows(&self, rows_per_item: usize) -> bool {
        self.version == 3
            && (2..=8).contains(&rows_per_item)
            && self.steps.iter().enumerate().skip(1).all(|(index, step)| {
                step.items
                    .iter()
                    .all(|item| item.drafts == rows_per_item - 1 - usize::from(index % 2 == 0))
            })
    }

    /// Tokens each request holds after the trace's last step.
    pub fn final_tokens(&self) -> usize {
        self.steps
            .iter()
            .flat_map(|step| &step.items)
            .map(|item| item.history.len())
            .max()
            .unwrap_or(0)
    }

    /// The most prompt tokens one prefill step carries.
    pub fn prefill_tokens(&self) -> usize {
        self.steps
            .iter()
            .map(|step| {
                step.items
                    .iter()
                    .filter(|item| item.prefill)
                    .map(|item| item.inputs.len())
                    .sum()
            })
            .max()
            .unwrap_or(0)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version == MIXED_DECODE {
            return self.validate_mixed();
        }
        if ![1, 2, 3, 4].contains(&self.version)
            || !(1..=32).contains(&self.concurrency)
            || !(1024..=1 << 20).contains(&self.vocabulary)
            || self.steps.len() != if self.version == 4 { 1 } else { 13 }
        {
            return Err("unsupported or incomplete trace geometry".into());
        }
        if self.version == 4
            && (self.steps[0].mode != Mode::Greedy
                || self.steps[0]
                    .items
                    .iter()
                    .map(|i| i.inputs.len())
                    .sum::<usize>()
                    > 8192)
        {
            return Err("unsupported prefill-only trace".into());
        }
        let mut histories = vec![Vec::new(); self.concurrency];
        let mut prompt_lengths = vec![0; self.concurrency];
        for (step_index, step) in self.steps.iter().enumerate() {
            if step.rows() > if self.version == 1 { 128 } else { 256 }
                || step.items.len() != self.concurrency
            {
                return Err("trace lost a request".into());
            }
            let mut seen = std::collections::BTreeSet::new();
            for item in &step.items {
                if item.slot >= self.concurrency
                    || !seen.insert(item.slot)
                    || item.prefill != (step_index == 0)
                    || (item.anchor && (self.version != 4 || !item.prefill))
                    || item.drafts > if self.version == 1 { 3 } else { 7 }
                    || (self.version != 2 && !item.fixed_drafts.is_empty())
                    || (self.version == 2 && item.fixed_drafts.len() != item.drafts)
                    || (item.prefill && (item.inputs.len() < 16 || item.drafts != 0))
                    || (!item.prefill && item.inputs.len() != 1)
                    || item.kept != item.inputs.len()
                    || item
                        .history
                        .iter()
                        .chain(&item.inputs)
                        .chain(&item.fixed_drafts)
                        .any(|&t| t as usize >= self.vocabulary)
                {
                    return Err("invalid fixed trace item".into());
                }
                if item.prefill {
                    prompt_lengths[item.slot] = item.inputs.len();
                }
                histories[item.slot].extend_from_slice(&item.inputs);
                if histories[item.slot] != item.history
                    || item.emitted != item.history.len() - prompt_lengths[item.slot]
                {
                    return Err("trace history or prompt/generated split drifted".into());
                }
            }
        }
        Ok(())
    }

    fn validate_mixed(&self) -> Result<(), String> {
        if !(1..=MIXED_REQUESTS).contains(&self.concurrency)
            || !(1024..=1 << 20).contains(&self.vocabulary)
            || !(2..=MIXED_STEPS).contains(&self.steps.len())
        {
            return Err("unsupported mixed decode trace geometry".into());
        }
        let mut histories = vec![Vec::new(); self.concurrency];
        let mut prompt_lengths = vec![None; self.concurrency];
        let mut decoding = false;
        for step in &self.steps {
            let prefill = step.items.first().ok_or("empty trace step")?.prefill;
            if prefill && decoding {
                return Err("mixed decode trace prefills after decoding".into());
            }
            decoding |= !prefill;
            if step.rows() > MIXED_REQUESTS {
                return Err("mixed decode step exceeds the row limit".into());
            }
            let mut seen = std::collections::BTreeSet::new();
            for item in &step.items {
                if item.slot >= self.concurrency
                    || !seen.insert(item.slot)
                    || item.prefill != prefill
                    || item.anchor
                    || item.drafts != 0
                    || !item.fixed_drafts.is_empty()
                    || item.inputs.is_empty()
                    || (!prefill && item.inputs.len() != 1)
                    || prompt_lengths[item.slot].is_some() == prefill
                    || item.kept != item.inputs.len()
                    || item
                        .history
                        .iter()
                        .chain(&item.inputs)
                        .any(|&t| t as usize >= self.vocabulary)
                {
                    return Err("invalid mixed decode trace item".into());
                }
                if prefill {
                    prompt_lengths[item.slot] = Some(item.inputs.len());
                }
                histories[item.slot].extend_from_slice(&item.inputs);
                if histories[item.slot] != item.history
                    || item.history.len().checked_sub(item.emitted) != prompt_lengths[item.slot]
                {
                    return Err("trace history or prompt/generated split drifted".into());
                }
            }
        }
        if !decoding || prompt_lengths.contains(&None) {
            return Err("mixed decode trace must prefill every request, then decode".into());
        }
        Ok(())
    }
}

impl Step {
    pub fn rows(&self) -> usize {
        self.items.iter().map(|item| 1 + item.drafts).sum()
    }

    pub fn model_items(
        &self,
        constraint: Option<&runtime_engine::RowConstraint>,
    ) -> Vec<ModelBatchItem> {
        self.items
            .iter()
            .enumerate()
            .map(|(position, item)| {
                let mode = if self.mode == Mode::Mixed {
                    if position + 1 == self.items.len() {
                        Mode::Filtered
                    } else {
                        Mode::Seeded
                    }
                } else {
                    self.mode
                };
                let mut parameters =
                    SamplingParameters::new(if mode == Mode::Greedy { 0.0 } else { 0.8 }).unwrap();
                if matches!(mode, Mode::Filtered | Mode::Grammar) {
                    parameters = parameters
                        .with_top_p(0.95)
                        .unwrap()
                        .with_min_p(0.05)
                        .unwrap()
                        .with_top_k(32)
                        .unwrap()
                        .with_penalties(0.2, 0.1, 1.1)
                        .unwrap()
                        .with_logit_bias(vec![LogitBias::new(9, 0.5).unwrap()])
                        .unwrap();
                }
                if matches!(mode, Mode::Top | Mode::Grammar | Mode::Filtered) {
                    parameters = parameters.with_top_logprobs(5).unwrap();
                }
                let phase = if item.prefill {
                    BatchPhase::Prefill {
                        tokens: item.inputs.len(),
                    }
                } else if item.drafts > 0 {
                    BatchPhase::Verify {
                        draft_tokens: item.drafts,
                    }
                } else {
                    BatchPhase::Decode
                };
                let mut inputs = item.inputs.clone();
                inputs.extend_from_slice(&item.fixed_drafts);
                let mut model =
                    ModelBatchItem::new(RequestId::new(item.slot as u64 + 1), phase, inputs, true)
                        .with_anchor(item.anchor)
                        .with_sampling(SamplingWork {
                            parameters,
                            state: SamplingState::new(71 + item.slot as u64)
                                .after(item.emitted as u64),
                            history: item.history.clone().into(),
                            emitted_tokens: item.emitted,
                            stop_ids: Vec::new(),
                            constraint: if mode == Mode::Grammar {
                                Some(constraint.expect("compiled grammar").clone())
                            } else {
                                None
                            },
                        });
                if item.drafts > 0 {
                    model = model.with_draft(DraftWork::new(
                        item.drafts,
                        if item.fixed_drafts.is_empty() {
                            DraftSource::Bundle
                        } else {
                            DraftSource::Host
                        },
                    ));
                }
                model
            })
            .collect()
    }
}

/// Validate the live decode/verify logits owners for a replay arm.
pub(super) fn validate_row_owners(rows: &[u32], ranks: usize, owned: bool) -> Result<(), String> {
    if rows.is_empty() || ranks == 0 {
        return Err("row ownership requires live rows and ranks".into());
    }
    // Row ownership divides the padded graph bucket across ranks. Padding owns no
    // emitted logits, so K2 can leave the last rank without live rows.
    let block = rows.len().next_power_of_two().div_ceil(ranks);
    for (index, &observed) in rows.iter().enumerate() {
        let expected = if owned { (index / block) as u32 } else { 0 };
        if observed != expected {
            return Err(format!(
                "logits row {index}: observed owner {observed}, expected {expected} (live {}, block {block}, owned {owned})",
                rows.len()
            ));
        }
    }
    Ok(())
}
