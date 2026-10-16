//! The host's device sampler: what it computes for each logits row, the
//! records it reads and writes, and its buffers by role.
//!
//! The host samples a view of logits rows on the device after each step,
//! with per-row settings, and a backend must supply that sampler as well as
//! model ops. Rows are independent. The oracle is the host's CPU sampler,
//! `runtime_sampling::oracle::sample_row`, which takes a [`SamplingRow`] and
//! returns its [`RowOutcome`]; [`check`] judges a device's [`RowRecord`]
//! against it.
//!
//! # One row
//!
//! A row has `live` logits (`1..=MAX_WIDTH`), F32 or BF16, read as f64; a
//! BF16 logit widens exactly. Every step below computes in f64 with
//! round-to-nearest-even and separately rounded operations, in this order:
//!
//! 1. Validity. A non-finite live logit, admitted or not, fails the row
//!    ([`RowFailure::InvalidLogit`]).
//! 2. Admission. A constraint mask admits token `t` through bit `t % 32` of
//!    word `t / 32`; bits at or past `live` are ignored, and a row without a
//!    mask admits every token. Masked tokens take no part in any later step,
//!    the logprobs included.
//! 3. Logprobs. With `m` the largest admitted raw logit and
//!    `Z = sum(exp(raw - m))` over admitted tokens, a token's logprob is
//!    `raw - m - ln Z`: logprobs describe the unprocessed row renormalized
//!    over admitted tokens, never the processed values below. The top
//!    logprobs are the `min(top_logprobs, admitted)` admitted tokens with
//!    the largest raw logits, best first.
//! 4. Candidates. When `mask_eos` is set, the row's EOS tokens are removed
//!    from the candidates (they keep their part in step 3). No candidate
//!    left fails the row ([`RowFailure::NoCandidate`]). The EOS mask stands
//!    for a minimum length not yet reached, so its counts hold fewer than
//!    [`MAX_HISTORY`] output tokens.
//! 5. Processed values. A candidate's value starts at its raw logit, adds
//!    its logit bias (`v = raw + bias`), and, if it has a count entry, takes
//!    the penalties in this order: repetition (`v / r` when `v > 0`, else
//!    `v * r`), frequency (`v - f * output_count`, the product rounded
//!    before the subtraction) and presence (`v - p` when `output_count > 0`).
//! 6. Greedy rows (`temperature == 0`) sample the candidate with the largest
//!    processed value. Min-p, top-k and top-p do not apply.
//! 7. Other rows divide every value by the temperature (`s = v / T`) and
//!    filter, each step over the tokens the previous one kept:
//!    - min-p (`min_p > 0`): keep `s - max(s) >= ln(min_p)`;
//!    - top-k (`top_k > 0`): keep the `top_k` largest;
//!    - top-p (`top_p < 1`): rank by `s`, best first, take the terms
//!      `exp(s - s_best)` and their sum `S` added in rank order, and keep the
//!      shortest ranked prefix whose sum, added in rank order, reaches
//!      `top_p * S`.
//!
//!    Each filter keeps the best token, so the kept set is never empty.
//!    Among kept tokens the row samples the largest Gumbel score
//!    `(s - M) + g(t)`, where `M` is the largest kept `s` and
//!    `g(t) = -ln(-ln(u))`. `u = (bits + 0.5) * 2^-53` with
//!    `bits = (x0 << 21) | (x1 >> 11)` of `x = Philox4x32-10(key, counter)`:
//!    the counter words are the token ID, the low and high words of
//!    `generated`, and zero; the key is the low and high words of `seed`.
//! 8. The output is the sampled token, its logprob from step 3, and the top
//!    logprobs.
//!
//! Every ranking and every maximum (steps 3, 6 and 7) breaks ties by the
//! lower token ID; `-0.0` and `+0.0` are equal. A row samples the same token
//! for the same logits, settings and `(seed, generated)`; the host advances
//! `generated` by one per sampled token.
//!
//! # Exactness
//!
//! A backend samples the oracle's token and top-logprob IDs, and its
//! logprobs are within [`LOGPROB_TOLERANCE`] of the oracle's. Two
//! evaluations of one score may differ by a few ulps (an f64 `ln` within one
//! ulp, a tiled evaluation's per-tile `(raw - tile_max) / T +
//! (tile_max - m) / T`), which can change a selection only between scores
//! that tie within that rounding. Logprob normalizers may sum f32 terms.
//!
//! A nonzero temperature is at least [`MIN_TEMPERATURE`], so every processed
//! value `s` and every score a backend forms from them (`s - M`, per-tile
//! differences divided by `T`) is finite. Below it `v / T` can overflow: the
//! oracle's `s - M` becomes NaN while a tiled evaluation still ranks finite
//! differences. The floor is a host normalization: a request's
//! temperature below it becomes 0 (greedy) when its parameters are built, so
//! the host never writes such a row. [`RowSettings`] refuses one from other
//! callers.
//!
//! # The host's rows
//!
//! The host builds a row's inputs from its request: counts from the
//! request's history (one entry per seen token, prompt and output counted
//! apart) when any penalty is not the identity, the biases sorted by token,
//! the stop tokens as EOS tokens with `mask_eos` while fewer than the minimum
//! new tokens were emitted (unless the mask admits only stop tokens, so a
//! constrained row can stop), the constraint mask, and the temperature, 0
//! for any request temperature below [`MIN_TEMPERATURE`]. Rows without a mask
//! in a view that has masks carry all-ones words. After the device writes
//! the top logprobs of admitted tokens, the host appends masked tokens in
//! ID order with a logprob of negative infinity, up to
//! `min(top_logprobs, live)` entries.
//!
//! # The output record
//!
//! A backend writes one [`OUTPUT_BYTES`] record per row. A sampled row's
//! record holds [`STATUS_SAMPLED`], the token, its logprob and the first
//! [`SamplingRow::top_count`] top entries. A failing row's record holds
//! only its status, [`STATUS_INVALID`] for [`RowFailure::InvalidLogit`] or
//! [`STATUS_EMPTY`] for [`RowFailure::NoCandidate`]: its token, logprob
//! and top entries are unspecified, and the host reads none of them.
//!
//! The SM120 entries and how runtime-host launches them are described in
//! runtime-host's `sampling` module.

use crate::contract::{DType, Refusal, Validate};

/// The widest logits row.
pub const MAX_WIDTH: u32 = 1 << 20;
/// The most top logprobs a row reports.
pub const MAX_TOP_LOGPROBS: u32 = 20;
/// The most EOS tokens a row lists.
pub const MAX_EOS: usize = 256;
/// The most logit biases a row lists.
pub const MAX_LOGIT_BIASES: usize = 1 << 16;
/// The most history tokens a row's counts describe.
pub const MAX_HISTORY: u64 = 1 << 20;
/// The largest temperature.
pub const MAX_TEMPERATURE: f64 = 100.0;
/// The repetition penalty's range.
pub const REPETITION_PENALTY: (f64, f64) = (0.01, 10.0);
/// The largest magnitude of the frequency and presence penalties.
pub const MAX_PENALTY: f64 = 2.0;
/// The largest magnitude of a logit bias.
pub const MAX_LOGIT_BIAS: f64 = 100.0;
/// The largest factor the repetition penalty scales a value by: dividing a
/// positive value by the smallest penalty, or multiplying a negative one by
/// the largest.
const REPETITION_GAIN: f64 = {
    let divided = 1.0 / REPETITION_PENALTY.0;
    if divided > REPETITION_PENALTY.1 {
        divided
    } else {
        REPETITION_PENALTY.1
    }
};
/// A bound on the magnitude of a processed value before the temperature:
/// the largest finite logit with the largest bias, scaled by the largest
/// repetition gain (100), less the largest frequency penalty over the
/// longest history and the presence penalty.
pub const MAX_PROCESSED: f64 = (f32::MAX as f64 + MAX_LOGIT_BIAS) * REPETITION_GAIN
    + MAX_PENALTY * MAX_HISTORY as f64
    + MAX_PENALTY;
/// The smallest nonzero temperature (about 1.5e-267): dividing any processed
/// value by it stays within an eighth of the f64 range, so differences of
/// two such values and sums of two differences stay finite. The host samples
/// a request below it greedily.
pub const MIN_TEMPERATURE: f64 = 8.0 * MAX_PROCESSED / f64::MAX;
/// The largest absolute difference from the oracle's logprobs.
pub const LOGPROB_TOLERANCE: f64 = 1e-6;

/// The sampler's static shape: the logit type and the widest live row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingRequirement {
    pub dtype: DType,
    pub width: u32,
}

impl Validate for SamplingRequirement {
    fn validate(&self) -> Result<(), Refusal> {
        if !matches!(self.dtype, DType::F32 | DType::Bf16) {
            return Err(Refusal("sampling reads F32 or BF16 logits"));
        }
        if self.width == 0 || self.width > MAX_WIDTH {
            return Err(Refusal("sampling width must be 1..=1048576"));
        }
        Ok(())
    }
}

/// A row's settings, as its metadata record carries them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowSettings {
    pub seed: u64,
    /// Tokens the request sampled before this row: the Philox counter.
    pub generated: u64,
    /// Zero keeps every token.
    pub top_k: u32,
    pub top_logprobs: u32,
    /// Removes the row's EOS tokens from the candidates.
    pub mask_eos: bool,
    /// Zero samples greedily.
    pub temperature: f64,
    pub top_p: f64,
    pub min_p: f64,
    pub repetition_penalty: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
}

impl RowSettings {
    /// Plain sampling at `temperature`: no filter, penalty or EOS mask.
    pub const fn new(temperature: f64, seed: u64, generated: u64) -> RowSettings {
        RowSettings {
            seed,
            generated,
            top_k: 0,
            top_logprobs: 0,
            mask_eos: false,
            temperature,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// The metadata record, with the row's spans of the flat count, bias
    /// and EOS lists.
    pub fn encode(&self, counts: ListSpan, biases: ListSpan, eos: ListSpan) -> [u8; META_BYTES] {
        let mut record = [0; META_BYTES];
        let mut put = |offset: usize, bytes: &[u8]| {
            record[offset..offset + bytes.len()].copy_from_slice(bytes)
        };
        put(0, &self.seed.to_le_bytes());
        put(8, &self.generated.to_le_bytes());
        put(16, &counts.offset.to_le_bytes());
        put(24, &biases.offset.to_le_bytes());
        put(32, &eos.offset.to_le_bytes());
        put(40, &counts.len.to_le_bytes());
        put(44, &biases.len.to_le_bytes());
        put(48, &eos.len.to_le_bytes());
        put(52, &self.top_k.to_le_bytes());
        put(56, &self.top_logprobs.to_le_bytes());
        put(60, &u32::from(self.mask_eos).to_le_bytes());
        for (offset, value) in [
            (64, self.temperature),
            (72, self.top_p),
            (80, self.min_p),
            (88, self.repetition_penalty),
            (96, self.frequency_penalty),
            (104, self.presence_penalty),
        ] {
            put(offset, &value.to_le_bytes());
        }
        record
    }
}

impl Validate for RowSettings {
    fn validate(&self) -> Result<(), Refusal> {
        let within =
            |value: f64, low: f64, high: f64| value.is_finite() && low <= value && value <= high;
        if !within(self.temperature, 0.0, MAX_TEMPERATURE) {
            return Err(Refusal("sampling temperature must be 0..=100"));
        }
        if self.temperature != 0.0 && self.temperature < MIN_TEMPERATURE {
            return Err(Refusal(
                "a nonzero sampling temperature must be at least MIN_TEMPERATURE",
            ));
        }
        if !within(self.top_p, 0.0, 1.0) || self.top_p == 0.0 {
            return Err(Refusal("sampling top-p must be in (0, 1]"));
        }
        if !within(self.min_p, 0.0, 1.0) {
            return Err(Refusal("sampling min-p must be 0..=1"));
        }
        if self.top_k > MAX_WIDTH {
            return Err(Refusal("sampling top-k exceeds the widest row"));
        }
        if self.top_logprobs > MAX_TOP_LOGPROBS {
            return Err(Refusal("sampling reports at most 20 top logprobs"));
        }
        let (low, high) = REPETITION_PENALTY;
        if !within(self.repetition_penalty, low, high)
            || !within(self.frequency_penalty, -MAX_PENALTY, MAX_PENALTY)
            || !within(self.presence_penalty, -MAX_PENALTY, MAX_PENALTY)
        {
            return Err(Refusal("sampling penalty is out of range"));
        }
        Ok(())
    }
}

/// A row's span of a flat list: its first entry and its entry count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ListSpan {
    pub offset: u64,
    pub len: u32,
}

/// A token the request's history holds: whether the prompt holds it, and
/// how often the output does. Penalties apply to a token with an entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenCount {
    pub token_id: u32,
    pub prompt_seen: bool,
    pub output_count: u32,
}

impl TokenCount {
    pub fn encode(&self) -> [u8; COUNT_BYTES] {
        let mut record = [0; COUNT_BYTES];
        record[0..4].copy_from_slice(&self.token_id.to_le_bytes());
        record[4..8].copy_from_slice(&u32::from(self.prompt_seen).to_le_bytes());
        record[8..12].copy_from_slice(&self.output_count.to_le_bytes());
        record
    }
}

/// A logit bias.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenBias {
    pub token_id: u32,
    pub bias: f64,
}

impl TokenBias {
    pub fn encode(&self) -> [u8; BIAS_BYTES] {
        let mut record = [0; BIAS_BYTES];
        record[0..4].copy_from_slice(&self.token_id.to_le_bytes());
        record[8..16].copy_from_slice(&self.bias.to_le_bytes());
        record
    }
}

/// Bytes of one row's metadata record: seed and generated (u64), the
/// count, bias and EOS offsets (u64) and lengths (u32), top-k, top
/// logprobs and the EOS mask flag (u32), then temperature, top-p, min-p and
/// the repetition, frequency and presence penalties (f64, from byte 64), all
/// little-endian.
pub const META_BYTES: usize = 112;
/// Bytes of a count entry: token ID, prompt-seen flag, output count and a
/// reserved word (u32 each).
pub const COUNT_BYTES: usize = 16;
/// Bytes of a bias entry: token ID and a reserved word (u32), then the bias
/// (f64).
pub const BIAS_BYTES: usize = 16;
/// Bytes of one row's output record: token ID and status (u32), logprob
/// (f64), 20 top IDs (u32, from byte 16) and 20 top logprobs (f64, from
/// byte 96).
pub const OUTPUT_BYTES: usize = 256;

/// The row was sampled.
pub const STATUS_SAMPLED: u32 = 0;
/// A non-finite live logit. The host fails the whole view.
pub const STATUS_INVALID: u32 = 1;
/// No candidate. The host fails only the row's request and serves the
/// view's other rows.
pub const STATUS_EMPTY: u32 = 2;
/// Settings outside their ranges. The host fails the whole view.
pub const STATUS_MALFORMED: u32 = 3;

/// One row's inputs.
#[derive(Clone, Copy, Debug)]
pub struct SamplingRow<'a> {
    /// The live logits; BF16 logits widened.
    pub logits: &'a [f32],
    pub settings: RowSettings,
    /// One entry per token, by ascending token ID.
    pub counts: &'a [TokenCount],
    /// One entry per token, by ascending token ID.
    pub biases: &'a [TokenBias],
    /// Read only when `settings.mask_eos` is set.
    pub eos: &'a [u32],
    /// `live.div_ceil(32)` words, or none to admit every token.
    pub mask: Option<&'a [u32]>,
}

impl SamplingRow<'_> {
    pub fn admits(&self, token: u32) -> bool {
        self.mask
            .is_none_or(|mask| mask[token as usize / 32] & (1 << (token % 32)) != 0)
    }

    /// The admitted tokens.
    pub fn admitted(&self) -> usize {
        (0..self.logits.len() as u32)
            .filter(|&token| self.admits(token))
            .count()
    }

    /// The top logprobs a sampled row reports.
    pub fn top_count(&self) -> usize {
        (self.settings.top_logprobs as usize).min(self.admitted())
    }
}

impl Validate for SamplingRow<'_> {
    fn validate(&self) -> Result<(), Refusal> {
        self.settings.validate()?;
        let live = self.logits.len();
        if live == 0 || live > MAX_WIDTH as usize {
            return Err(Refusal("sampling width must be 1..=1048576"));
        }
        let ascending = |tokens: &[u32]| {
            tokens.windows(2).all(|pair| pair[0] < pair[1])
                && tokens.last().is_none_or(|&last| (last as usize) < live)
        };
        let tokens = self
            .counts
            .iter()
            .map(|count| count.token_id)
            .collect::<Vec<_>>();
        if !ascending(&tokens) {
            return Err(Refusal(
                "sampling counts must be unique live tokens by ascending ID",
            ));
        }
        if self
            .counts
            .iter()
            .any(|count| !count.prompt_seen && count.output_count == 0)
        {
            return Err(Refusal("a sampling count entry must name a seen token"));
        }
        let history = self
            .counts
            .iter()
            .map(|count| u64::from(count.prompt_seen) + u64::from(count.output_count))
            .sum::<u64>();
        if history > MAX_HISTORY {
            return Err(Refusal("sampling counts describe too long a history"));
        }
        // An EOS mask stands for a minimum length, at most MAX_HISTORY, that
        // the output has not reached.
        let output = self
            .counts
            .iter()
            .map(|count| u64::from(count.output_count))
            .sum::<u64>();
        if self.settings.mask_eos && output >= MAX_HISTORY {
            return Err(Refusal(
                "an EOS mask needs fewer than MAX_HISTORY output tokens",
            ));
        }
        if self.biases.len() > MAX_LOGIT_BIASES {
            return Err(Refusal("sampling lists at most 65,536 logit biases"));
        }
        let tokens = self
            .biases
            .iter()
            .map(|bias| bias.token_id)
            .collect::<Vec<_>>();
        if !ascending(&tokens) {
            return Err(Refusal(
                "sampling biases must be unique live tokens by ascending ID",
            ));
        }
        if self
            .biases
            .iter()
            .any(|bias| !(bias.bias.is_finite() && bias.bias.abs() <= MAX_LOGIT_BIAS))
        {
            return Err(Refusal("a sampling logit bias is out of range"));
        }
        if self.settings.mask_eos {
            if self.eos.len() > MAX_EOS {
                return Err(Refusal("sampling lists at most 256 EOS tokens"));
            }
            if self.eos.iter().any(|&token| token as usize >= live) {
                return Err(Refusal("a sampling EOS token is past the live width"));
            }
        }
        if let Some(mask) = self.mask {
            if mask.len() != live.div_ceil(32) {
                return Err(Refusal("a sampling mask holds one bit per live token"));
            }
            // The host keeps the EOS tokens of a mask that admits only them.
            let mut eos = self.eos.to_vec();
            eos.sort_unstable();
            if self.settings.mask_eos
                && (0..live as u32)
                    .all(|token| !self.admits(token) || eos.binary_search(&token).is_ok())
            {
                return Err(Refusal(
                    "a mask that admits only EOS tokens takes no EOS mask",
                ));
            }
        }
        Ok(())
    }
}

/// A sampled row.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampled {
    pub token_id: u32,
    pub logprob: f64,
    /// The admitted tokens with the largest raw logits and their logprobs,
    /// best first: `SamplingRow::top_count` entries.
    pub top: Vec<(u32, f64)>,
}

/// Why a row was not sampled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowFailure {
    /// A live logit is not finite.
    InvalidLogit,
    /// No admitted token is a candidate.
    NoCandidate,
}

impl RowFailure {
    /// The output statuses a backend may write for the failure.
    pub fn statuses(self) -> &'static [u32] {
        match self {
            RowFailure::InvalidLogit => &[STATUS_INVALID],
            RowFailure::NoCandidate => &[STATUS_EMPTY],
        }
    }
}

pub type RowOutcome = Result<Sampled, RowFailure>;

/// One row's output record as a device writes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowRecord {
    pub token_id: u32,
    pub status: u32,
    pub logprob: f64,
    pub top_ids: [u32; MAX_TOP_LOGPROBS as usize],
    pub top_logprobs: [f64; MAX_TOP_LOGPROBS as usize],
}

impl RowRecord {
    pub fn decode(bytes: &[u8; OUTPUT_BYTES]) -> RowRecord {
        let word = |offset: usize| {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
        };
        let double = |offset: usize| {
            f64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("eight bytes"))
        };
        RowRecord {
            token_id: word(0),
            status: word(4),
            logprob: double(8),
            top_ids: std::array::from_fn(|rank| word(16 + 4 * rank)),
            top_logprobs: std::array::from_fn(|rank| double(96 + 8 * rank)),
        }
    }

    /// The record a device writes for a sampled row; entries past its top
    /// logprobs are zero.
    pub fn sampled(sampled: &Sampled) -> RowRecord {
        let mut record = RowRecord {
            token_id: sampled.token_id,
            status: STATUS_SAMPLED,
            logprob: sampled.logprob,
            top_ids: [0; MAX_TOP_LOGPROBS as usize],
            top_logprobs: [0.0; MAX_TOP_LOGPROBS as usize],
        };
        for (rank, &(token, logprob)) in sampled.top.iter().enumerate() {
            record.top_ids[rank] = token;
            record.top_logprobs[rank] = logprob;
        }
        record
    }
}

/// Judges a device's record of `row` against the oracle's outcome: the same
/// failure, or the same token and top IDs with logprobs within
/// [`LOGPROB_TOLERANCE`]. A failing row's record is judged by its status
/// alone, and entries past a sampled row's top count are not results.
pub fn check(
    row: &SamplingRow<'_>,
    oracle: &RowOutcome,
    record: &RowRecord,
) -> Result<(), Refusal> {
    let sampled = match oracle {
        Err(failure) if failure.statuses().contains(&record.status) => return Ok(()),
        Err(_) => {
            return Err(Refusal(
                "the row's status differs from the oracle's failure",
            ));
        }
        Ok(_) if record.status != STATUS_SAMPLED => {
            return Err(Refusal("the oracle sampled a row the device failed"));
        }
        Ok(sampled) => sampled,
    };
    if record.token_id != sampled.token_id {
        return Err(Refusal("the sampled token differs from the oracle's"));
    }
    let close = |actual: f64, expected: f64| (actual - expected).abs() <= LOGPROB_TOLERANCE;
    if !close(record.logprob, sampled.logprob) {
        return Err(Refusal("the sampled logprob is outside the tolerance"));
    }
    if sampled.top.len() != row.top_count() {
        return Err(Refusal(
            "the oracle's top logprobs differ from the row's count",
        ));
    }
    for (rank, &(token, logprob)) in sampled.top.iter().enumerate() {
        if record.top_ids[rank] != token {
            return Err(Refusal("a top-logprob token differs from the oracle's"));
        }
        if !close(record.top_logprobs[rank], logprob) {
            return Err(Refusal("a top logprob is outside the tolerance"));
        }
    }
    Ok(())
}

crate::scratch_record! {
    /// A sampling launch's buffers by role, over `Counts::rows` rows:
    /// - `logits`: the rows, `row_stride` elements apart, of which the
    ///   first `live_width` are live;
    /// - `metadata`: one [`META_BYTES`] record per row;
    /// - `counts`, `biases` and `eos`: the flat lists the metadata records
    ///   index ([`COUNT_BYTES`] and [`BIAS_BYTES`] entries, u32 EOS
    ///   tokens), at least 16, 16 and 8 bytes even when empty;
    /// - `masks`: `live_width.div_ceil(32)` words per row, or none when no
    ///   row has a mask;
    /// - `output`: one [`OUTPUT_BYTES`] record per row: the status of
    ///   every row, and the token, logprob and first
    ///   `SamplingRow::top_count` top entries of a sampled row (see the
    ///   module's output record);
    /// - `scratch`: the backend's workspace.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SamplingBuffers {
        pub logits: value,
        pub metadata: value,
        pub counts: value,
        pub biases: value,
        pub eos: value,
        pub masks: option,
        pub output: value,
        pub scratch: record(SamplingScratch),
        pub live_width: scalar(u32),
        pub row_stride: scalar(u64),
    }
}

crate::scratch_record! {
    /// Scratch a backend uses as it chooses and lists in its op info. What
    /// the host provides for the SM120 entries is described in
    /// runtime-host's `sampling` module.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SamplingScratch {
        pub partials: value,
        pub filter: value,
        pub adjusted: value,
        pub order: value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation::Traversal;

    fn row<'a>(logits: &'a [f32], settings: RowSettings) -> SamplingRow<'a> {
        SamplingRow {
            logits,
            settings,
            counts: &[],
            biases: &[],
            eos: &[],
            mask: None,
        }
    }

    #[test]
    fn records_match_the_device_layout() {
        let settings = RowSettings {
            top_k: 7,
            top_logprobs: 3,
            mask_eos: true,
            top_p: 0.9,
            min_p: 0.05,
            repetition_penalty: 1.2,
            frequency_penalty: 0.5,
            presence_penalty: -0.25,
            ..RowSettings::new(0.7, 11, 12)
        };
        let span = |offset, len| ListSpan { offset, len };
        let meta = settings.encode(span(13, 14), span(15, 16), span(17, 18));
        let word = |offset: usize| u32::from_le_bytes(meta[offset..offset + 4].try_into().unwrap());
        let long = |offset: usize| u64::from_le_bytes(meta[offset..offset + 8].try_into().unwrap());
        let double =
            |offset: usize| f64::from_le_bytes(meta[offset..offset + 8].try_into().unwrap());
        assert_eq!([0, 8, 16, 24, 32].map(long), [11, 12, 13, 15, 17]);
        assert_eq!([40, 44, 48, 52, 56, 60].map(word), [14, 16, 18, 7, 3, 1]);
        assert_eq!(
            [64, 72, 80, 88, 96, 104].map(double),
            [0.7, 0.9, 0.05, 1.2, 0.5, -0.25]
        );
        let count = TokenCount {
            token_id: 5,
            prompt_seen: true,
            output_count: 9,
        };
        assert_eq!(&count.encode()[..12], &[5, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0]);
        let bias = TokenBias {
            token_id: 6,
            bias: -1.5,
        }
        .encode();
        assert_eq!(&bias[..8], &[6, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bias[8..], (-1.5_f64).to_le_bytes());

        let mut output = [0_u8; OUTPUT_BYTES];
        output[0..4].copy_from_slice(&9_u32.to_le_bytes());
        output[8..16].copy_from_slice(&(-0.5_f64).to_le_bytes());
        output[16 + 4 * 19..96].copy_from_slice(&4_u32.to_le_bytes());
        output[96 + 8 * 19..].copy_from_slice(&(-2.0_f64).to_le_bytes());
        let record = RowRecord::decode(&output);
        assert_eq!(
            (record.token_id, record.status, record.logprob),
            (9, 0, -0.5)
        );
        assert_eq!((record.top_ids[19], record.top_logprobs[19]), (4, -2.0));
    }

    #[test]
    fn settings_outside_their_ranges_are_refused() {
        let plain = RowSettings::new(1.0, 0, 0);
        assert_eq!(plain.validate(), Ok(()));
        for settings in [
            RowSettings::new(-0.5, 0, 0),
            RowSettings::new(f64::NAN, 0, 0),
            RowSettings::new(100.5, 0, 0),
            RowSettings {
                top_p: 0.0,
                ..plain
            },
            RowSettings {
                top_p: 1.5,
                ..plain
            },
            RowSettings {
                min_p: -0.1,
                ..plain
            },
            RowSettings {
                top_logprobs: 21,
                ..plain
            },
            RowSettings {
                top_k: MAX_WIDTH + 1,
                ..plain
            },
            RowSettings {
                repetition_penalty: 0.0,
                ..plain
            },
            RowSettings {
                frequency_penalty: 2.5,
                ..plain
            },
            RowSettings {
                presence_penalty: f64::INFINITY,
                ..plain
            },
        ] {
            assert!(settings.validate().is_err(), "{settings:?}");
        }
    }

    #[test]
    fn the_smallest_temperature_keeps_every_score_finite() {
        // Both repetition branches at their extremes, with every other
        // input at its extreme in the same direction, in the contract's order.
        let (smallest, largest_penalty) = REPETITION_PENALTY;
        let history = MAX_HISTORY as f64;
        let positive = (f64::from(f32::MAX) + MAX_LOGIT_BIAS) / smallest
            - -MAX_PENALTY * history
            - -MAX_PENALTY;
        let negative = (-f64::from(f32::MAX) - MAX_LOGIT_BIAS) * largest_penalty
            - MAX_PENALTY * history
            - MAX_PENALTY;
        assert!(positive <= MAX_PROCESSED && -negative <= MAX_PROCESSED);
        assert!(positive > (f64::from(f32::MAX) + MAX_LOGIT_BIAS) * largest_penalty);
        for value in [positive, negative] {
            let scaled = value / MIN_TEMPERATURE;
            assert!((scaled - -scaled).is_finite(), "{value}");
        }
        let largest = MAX_PROCESSED / MIN_TEMPERATURE;
        // s - M and a served sum of two scaled differences.
        assert!((2.0 * largest).is_finite() && (4.0 * largest).is_finite());
        assert!(MAX_PROCESSED > f64::from(f32::MAX));
        let at = |temperature| RowSettings::new(temperature, 0, 0).validate();
        assert_eq!(at(MIN_TEMPERATURE), Ok(()));
        assert_eq!(at(0.0), Ok(()));
        assert!(at(MIN_TEMPERATURE.next_down()).is_err());
        assert!(at(1e-320).is_err());
    }

    #[test]
    fn rows_must_list_their_inputs_as_the_device_reads_them() {
        let logits = [0.0_f32; 40];
        let plain = RowSettings::new(1.0, 0, 0);
        let count = |token_id, prompt_seen, output_count| TokenCount {
            token_id,
            prompt_seen,
            output_count,
        };
        assert_eq!(row(&logits, plain).validate(), Ok(()));
        assert!(row(&[], plain).validate().is_err());
        let counts = [count(3, true, 0), count(2, false, 1)];
        assert!(
            SamplingRow {
                counts: &counts,
                ..row(&logits, plain)
            }
            .validate()
            .is_err()
        );
        let counts = [count(3, false, 0)];
        assert!(
            SamplingRow {
                counts: &counts,
                ..row(&logits, plain)
            }
            .validate()
            .is_err()
        );
        let counts = [count(40, true, 0)];
        assert!(
            SamplingRow {
                counts: &counts,
                ..row(&logits, plain)
            }
            .validate()
            .is_err()
        );
        let biases = [TokenBias {
            token_id: 1,
            bias: 100.5,
        }];
        assert!(
            SamplingRow {
                biases: &biases,
                ..row(&logits, plain)
            }
            .validate()
            .is_err()
        );
        let biases = [
            TokenBias {
                token_id: 4,
                bias: 1.0,
            },
            TokenBias {
                token_id: 4,
                bias: 2.0,
            },
        ];
        assert!(
            SamplingRow {
                biases: &biases,
                ..row(&logits, plain)
            }
            .validate()
            .is_err()
        );

        let wide = vec![0.0_f32; MAX_LOGIT_BIASES + 1];
        let biases = (0..=MAX_LOGIT_BIASES as u32)
            .map(|token_id| TokenBias {
                token_id,
                bias: 1.0,
            })
            .collect::<Vec<_>>();
        let biased = |count: usize| {
            SamplingRow {
                biases: &biases[..count],
                ..row(&wide, plain)
            }
            .validate()
        };
        assert_eq!(biased(MAX_LOGIT_BIASES), Ok(()));
        assert!(biased(MAX_LOGIT_BIASES + 1).is_err());

        let masked = RowSettings {
            mask_eos: true,
            ..plain
        };
        // An EOS mask needs an output shorter than the longest history.
        let output = |output_count| [count(3, false, output_count)];
        let longest = MAX_HISTORY as u32;
        for (settings, output_count, valid) in [
            (masked, longest - 1, true),
            (masked, longest, false),
            (plain, longest, true),
        ] {
            let counts = output(output_count);
            let row = SamplingRow {
                counts: &counts,
                ..row(&logits, settings)
            };
            assert_eq!(row.validate().is_ok(), valid, "{output_count}");
        }
        // EOS tokens are read only when the row masks them.
        assert_eq!(
            SamplingRow {
                eos: &[45],
                ..row(&logits, plain)
            }
            .validate(),
            Ok(())
        );
        assert!(
            SamplingRow {
                eos: &[45],
                ..row(&logits, masked)
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplingRow {
                mask: Some(&[1]),
                ..row(&logits, plain)
            }
            .validate()
            .is_err()
        );
        // Bits past the live width are ignored.
        let mask = [0, 0x8000_0080];
        let only_eos = SamplingRow {
            eos: &[39],
            mask: Some(&mask),
            ..row(&logits, masked)
        };
        assert!(only_eos.validate().is_err());
        assert_eq!(
            (
                only_eos.admitted(),
                only_eos.admits(39),
                only_eos.admits(38)
            ),
            (1, true, false)
        );
        assert_eq!(
            SamplingRow {
                settings: plain,
                ..only_eos
            }
            .validate(),
            Ok(())
        );
        let mask = [1, 0x80];
        assert_eq!(
            SamplingRow {
                mask: Some(&mask),
                ..only_eos
            }
            .validate(),
            Ok(())
        );
    }

    #[test]
    fn check_compares_tokens_exactly_and_logprobs_within_the_tolerance() {
        let logits = [0.0_f32; 4];
        let row = row(
            &logits,
            RowSettings {
                top_logprobs: 2,
                ..RowSettings::new(1.0, 0, 0)
            },
        );
        let sampled = Sampled {
            token_id: 2,
            logprob: -1.0,
            top: vec![(0, -1.0), (1, -1.0)],
        };
        let oracle = Ok(sampled.clone());
        let record = RowRecord::sampled(&sampled);
        assert_eq!(check(&row, &oracle, &record), Ok(()));
        let near = RowRecord {
            logprob: -1.0 + 0.5e-6,
            ..record
        };
        assert_eq!(check(&row, &oracle, &near), Ok(()));
        for wrong in [
            RowRecord {
                token_id: 3,
                ..record
            },
            RowRecord {
                logprob: -1.0 + 2e-6,
                ..record
            },
            RowRecord {
                status: STATUS_INVALID,
                ..record
            },
            RowRecord {
                top_ids: [1; 20],
                ..record
            },
        ] {
            assert!(check(&row, &oracle, &wrong).is_err(), "{wrong:?}");
        }
        // A failing row's other fields are unspecified: the served entries
        // leave whatever bytes they held.
        let stale = RowRecord {
            token_id: u32::MAX,
            logprob: f64::NAN,
            top_ids: [u32::MAX; 20],
            top_logprobs: [f64::NAN; 20],
            ..record
        };
        let mut failed = [stale; 2];
        failed[0].status = STATUS_INVALID;
        failed[1].status = STATUS_EMPTY;
        // No candidate is status 2 only: status 1 would fail the whole view.
        let empty = Err(RowFailure::NoCandidate);
        assert!(check(&row, &empty, &failed[0]).is_err());
        assert_eq!(check(&row, &empty, &failed[1]), Ok(()));
        assert!(check(&row, &empty, &record).is_err());
        let invalid = Err(RowFailure::InvalidLogit);
        assert_eq!(check(&row, &invalid, &failed[0]), Ok(()));
        assert!(check(&row, &invalid, &failed[1]).is_err());
    }

    #[test]
    fn buffers_name_their_roles() {
        let buffers = SamplingBuffers {
            logits: 1,
            metadata: 2,
            counts: 3,
            biases: 4,
            eos: 5,
            masks: None,
            output: 6,
            scratch: SamplingScratch {
                partials: 7,
                filter: 8,
                adjusted: 9,
                order: 10,
            },
            live_width: 32,
            row_stride: 64,
        };
        let mut names = Vec::new();
        buffers.visit_named(&mut |name, _| names.push(name.to_owned()));
        assert_eq!(
            names,
            [
                "logits",
                "metadata",
                "counts",
                "biases",
                "eos",
                "output",
                "scratch.partials",
                "scratch.filter",
                "scratch.adjusted",
                "scratch.order"
            ]
        );
        assert!(
            SamplingRequirement {
                dtype: DType::Bf16,
                width: MAX_WIDTH
            }
            .validate()
            .is_ok()
        );
        assert!(
            SamplingRequirement {
                dtype: DType::F16,
                width: 8
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplingRequirement {
                dtype: DType::F32,
                width: 0
            }
            .validate()
            .is_err()
        );
    }
}
