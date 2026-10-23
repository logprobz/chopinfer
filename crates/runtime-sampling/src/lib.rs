#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

use runtime_model_api::ModelVocabulary;

pub mod oracle;

pub const MAX_VOCABULARY: usize = 1 << 20;
pub const MAX_HISTORY: usize = 1 << 20;
pub const MAX_LOGIT_BIASES: usize = 1 << 16;
pub const MAX_STOP_IDS: usize = 256;
pub const MAX_TOP_LOGPROBS: usize = 20;

/// Scores a known token against the unmodified live logits domain.
pub fn score_token_logprobs(
    logits: &[f32],
    token_id: u32,
    top_count: usize,
) -> Result<(f64, Vec<TokenLogprob>), SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyVocabulary);
    }
    if top_count > MAX_TOP_LOGPROBS {
        return Err(SamplingError::TooManyTopLogprobs);
    }
    if token_id as usize >= logits.len() {
        return Err(SamplingError::BiasOutOfRange {
            token_id,
            vocabulary: logits.len(),
        });
    }
    if let Some(index) = logits.iter().position(|value| !value.is_finite()) {
        return Err(SamplingError::InvalidLogit { token_id: index });
    }
    let maximum = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let normalizer = logits
        .iter()
        .map(|value| (f64::from(*value) - maximum).exp())
        .sum::<f64>()
        .ln();
    let score = |index: usize| f64::from(logits[index]) - maximum - normalizer;
    let mut ranked = (0..logits.len()).collect::<Vec<_>>();
    ranked.sort_unstable_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then(left.cmp(&right))
    });
    let top = ranked
        .into_iter()
        .take(top_count)
        .map(|index| TokenLogprob::new(index as u32, score(index)))
        .collect();
    Ok((score(token_id as usize), top))
}
/// The largest top-k width a request may declare, derived from the shared
/// vocabulary ceiling. A width above [`MAX_VOCABULARY`] can never truncate
/// anything because no declarable logits row is wider, so it is rejected here
/// instead of surfacing later as a silent no-op or a sampling-time error.
pub const MAX_TOP_K: usize = MAX_VOCABULARY;

/// Optional generation defaults. Later layers replace only fields they supply.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingDefaults {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
}

impl SamplingDefaults {
    pub const EMPTY: Self = Self {
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        repetition_penalty: None,
    };

    #[must_use]
    pub fn overridden_by(self, overrides: Self) -> Self {
        Self {
            temperature: overrides.temperature.or(self.temperature),
            top_p: overrides.top_p.or(self.top_p),
            top_k: overrides.top_k.or(self.top_k),
            min_p: overrides.min_p.or(self.min_p),
            repetition_penalty: overrides.repetition_penalty.or(self.repetition_penalty),
        }
    }

    pub fn validate(self) -> Result<(), SamplingError> {
        let temperature = self.temperature.unwrap_or(1.0);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(SamplingError::InvalidTemperature);
        }
        let top_k = match self.top_k.unwrap_or(0) {
            -1 | 0 => 0,
            value => usize::try_from(value).map_err(|_| SamplingError::TopKTooLarge)?,
        };
        SamplingParameters::new(temperature)?
            .with_top_p(self.top_p.unwrap_or(1.0))?
            .with_top_k(top_k)?
            .with_min_p(self.min_p.unwrap_or(0.0))?
            .with_penalties(0.0, 0.0, self.repetition_penalty.unwrap_or(1.0))?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogitBias {
    token_id: u32,
    bias: f64,
}

impl LogitBias {
    pub fn new(token_id: u32, bias: f64) -> Result<Self, SamplingError> {
        if !bias.is_finite() || !(-100.0..=100.0).contains(&bias) {
            return Err(SamplingError::InvalidLogitBias);
        }
        Ok(Self { token_id, bias })
    }

    #[must_use]
    pub const fn token_id(self) -> u32 {
        self.token_id
    }

    #[must_use]
    pub const fn bias(self) -> f64 {
        self.bias
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParameters {
    temperature: f64,
    top_k: usize,
    top_p: f64,
    min_p: f64,
    minimum_new_tokens: usize,
    presence_penalty: f64,
    frequency_penalty: f64,
    repetition_penalty: f64,
    logit_bias: Vec<LogitBias>,
    top_logprobs: usize,
}

pub const MIN_REPETITION_PENALTY: f64 = 0.01;
pub const MAX_REPETITION_PENALTY: f64 = 10.0;

impl SamplingParameters {
    /// A temperature below `kernel_core::sampling::MIN_TEMPERATURE` (about
    /// 1.5e-267), where dividing processed logits by it can overflow,
    /// samples greedily: it becomes 0, on the host's CPU and device samplers
    /// alike.
    pub fn new(temperature: f64) -> Result<Self, SamplingError> {
        if !temperature.is_finite() || !(0.0..=100.0).contains(&temperature) {
            return Err(SamplingError::InvalidTemperature);
        }
        let temperature = if temperature < kernel_core::sampling::MIN_TEMPERATURE {
            0.0
        } else {
            temperature
        };
        Ok(Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            minimum_new_tokens: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            logit_bias: Vec::new(),
            top_logprobs: 0,
        })
    }

    pub fn with_top_k(mut self, top_k: usize) -> Result<Self, SamplingError> {
        if top_k > MAX_TOP_K {
            return Err(SamplingError::TopKTooLarge);
        }
        self.top_k = top_k;
        Ok(self)
    }

    pub fn with_top_p(mut self, top_p: f64) -> Result<Self, SamplingError> {
        if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
            return Err(SamplingError::InvalidTopP);
        }
        self.top_p = top_p;
        Ok(self)
    }

    pub fn with_min_p(mut self, min_p: f64) -> Result<Self, SamplingError> {
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err(SamplingError::InvalidMinP);
        }
        self.min_p = min_p;
        Ok(self)
    }

    pub fn with_minimum_new_tokens(
        mut self,
        minimum_new_tokens: usize,
    ) -> Result<Self, SamplingError> {
        if minimum_new_tokens > MAX_HISTORY {
            return Err(SamplingError::MinimumNewTokensTooLarge);
        }
        self.minimum_new_tokens = minimum_new_tokens;
        Ok(self)
    }

    pub fn with_penalties(
        mut self,
        presence_penalty: f64,
        frequency_penalty: f64,
        repetition_penalty: f64,
    ) -> Result<Self, SamplingError> {
        if !presence_penalty.is_finite()
            || !frequency_penalty.is_finite()
            || !(-2.0..=2.0).contains(&presence_penalty)
            || !(-2.0..=2.0).contains(&frequency_penalty)
            || !repetition_penalty.is_finite()
            || !(MIN_REPETITION_PENALTY..=MAX_REPETITION_PENALTY).contains(&repetition_penalty)
        {
            return Err(SamplingError::InvalidPenalty);
        }
        self.presence_penalty = presence_penalty;
        self.frequency_penalty = frequency_penalty;
        self.repetition_penalty = repetition_penalty;
        Ok(self)
    }

    pub fn with_logit_bias(mut self, mut biases: Vec<LogitBias>) -> Result<Self, SamplingError> {
        if biases.len() > MAX_LOGIT_BIASES {
            return Err(SamplingError::TooManyLogitBiases);
        }
        biases.sort_by_key(|bias| bias.token_id);
        if let Some(duplicate) = biases
            .windows(2)
            .find(|pair| pair[0].token_id == pair[1].token_id)
        {
            return Err(SamplingError::DuplicateLogitBias(duplicate[0].token_id));
        }
        self.logit_bias = biases;
        Ok(self)
    }

    pub fn with_top_logprobs(mut self, count: usize) -> Result<Self, SamplingError> {
        if count > MAX_TOP_LOGPROBS {
            return Err(SamplingError::TooManyTopLogprobs);
        }
        self.top_logprobs = count;
        Ok(self)
    }

    #[must_use]
    pub const fn minimum_new_tokens(&self) -> usize {
        self.minimum_new_tokens
    }

    #[must_use]
    pub const fn temperature(&self) -> f64 {
        self.temperature
    }

    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    #[must_use]
    pub const fn top_p(&self) -> f64 {
        self.top_p
    }

    #[must_use]
    pub const fn min_p(&self) -> f64 {
        self.min_p
    }

    #[must_use]
    pub const fn presence_penalty(&self) -> f64 {
        self.presence_penalty
    }

    #[must_use]
    pub const fn frequency_penalty(&self) -> f64 {
        self.frequency_penalty
    }

    #[must_use]
    pub const fn repetition_penalty(&self) -> f64 {
        self.repetition_penalty
    }

    #[must_use]
    pub fn logit_bias(&self) -> &[LogitBias] {
        &self.logit_bias
    }

    #[must_use]
    pub const fn top_logprobs(&self) -> usize {
        self.top_logprobs
    }
}

/// Request-local deterministic RNG state.
///
/// Replaying the same seed, parameters, logits, and history is this runtime's
/// contract. The stream does not claim byte-for-byte PyTorch RNG compatibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingState {
    seed: u64,
    generated: u64,
}

impl SamplingState {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed, generated: 0 }
    }

    #[must_use]
    pub const fn generated(&self) -> u64 {
        self.generated
    }

    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The state after `count` more sampled tokens.
    #[must_use]
    pub const fn after(self, count: u64) -> Self {
        Self {
            seed: self.seed,
            generated: self.generated.wrapping_add(count),
        }
    }
}

/// Philox4x32-10. The counter words are token ID, generated index low and
/// high words, and zero. The key is the low and high words of the request
/// seed. Coupled drafts derive their noise from the same function.
pub use kernel_core::sampling::philox4x32_10;

fn philox_unit(seed: u64, generated: u64, token_id: u32) -> f64 {
    let words = philox4x32_10(seed, generated, token_id);
    let bits = (u64::from(words[0]) << 21) | (u64::from(words[1]) >> 11);
    (bits as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
}

#[derive(Clone, Copy, Debug)]
pub struct SampleContext<'a> {
    history: &'a [u32],
    /// Tokens after `history`, such as the drafts a verify row follows.
    history_tail: &'a [u32],
    emitted_tokens: usize,
    eos_token_ids: &'a [u32],
    allowed_tokens: Option<&'a [u32]>,
}

impl<'a> SampleContext<'a> {
    #[must_use]
    pub const fn new(history: &'a [u32], emitted_tokens: usize, eos_token_ids: &'a [u32]) -> Self {
        Self {
            history,
            history_tail: &[],
            emitted_tokens,
            eos_token_ids,
            allowed_tokens: None,
        }
    }

    /// Continues the history with `tail`, so a caller that holds the two
    /// parts apart need not join them.
    #[must_use]
    pub const fn with_history_tail(mut self, tail: &'a [u32]) -> Self {
        self.history_tail = tail;
        self
    }

    fn history_len(&self) -> usize {
        self.history.len() + self.history_tail.len()
    }

    /// Restricts the row to a constraint's allowed tokens: bit `t % 32` of
    /// word `t / 32` admits token `t`. Masked tokens are excluded before any
    /// other processing, and logprobs describe the row renormalized over the
    /// allowed tokens, as when a grammar mask is written into the logits.
    #[must_use]
    pub const fn with_allowed_tokens(mut self, mask: &'a [u32]) -> Self {
        self.allowed_tokens = Some(mask);
        self
    }
}

fn mask_allows(mask: &[u32], token: usize) -> bool {
    mask.get(token / 32)
        .is_some_and(|word| word & (1 << (token % 32)) != 0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenLogprob {
    token_id: u32,
    logprob: f64,
}

impl TokenLogprob {
    #[must_use]
    pub const fn new(token_id: u32, logprob: f64) -> Self {
        Self { token_id, logprob }
    }
    #[must_use]
    pub const fn token_id(self) -> u32 {
        self.token_id
    }

    #[must_use]
    pub const fn logprob(self) -> f64 {
        self.logprob
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    token_id: u32,
    logprob: f64,
    top_logprobs: Vec<TokenLogprob>,
}

impl Sample {
    #[must_use]
    pub const fn token_id(&self) -> u32 {
        self.token_id
    }

    #[must_use]
    pub const fn logprob(&self) -> f64 {
        self.logprob
    }

    #[must_use]
    pub fn top_logprobs(&self) -> &[TokenLogprob] {
        &self.top_logprobs
    }
}

/// Applies bias, repetition, frequency, and presence penalties in that order.
/// It then applies temperature, min-p, top-k, and top-p filtering. Returned
/// logprobs describe the original, unprocessed live logits row, restricted to
/// the context's allowed tokens when it carries a constraint mask.
pub fn sample(
    parameters: &SamplingParameters,
    logits: &[f32],
    context: SampleContext<'_>,
    state: &mut SamplingState,
) -> Result<Sample, SamplingError> {
    sample_in_vocabulary(parameters, logits, logits.len(), context, state)
}

/// Samples within an exact model-declared logits domain.
pub fn sample_for_vocabulary(
    parameters: &SamplingParameters,
    logits: &[f32],
    vocabulary: ModelVocabulary,
    context: SampleContext<'_>,
    state: &mut SamplingState,
) -> Result<Sample, SamplingError> {
    sample_in_vocabulary(parameters, logits, vocabulary.get(), context, state)
}

/// Samples the live vocabulary from a possibly padded logits row. Padding is
/// never validated as a logit and can never become a sampled token.
pub fn sample_padded_for_vocabulary(
    parameters: &SamplingParameters,
    logits: &[f32],
    vocabulary: ModelVocabulary,
    context: SampleContext<'_>,
    state: &mut SamplingState,
) -> Result<Sample, SamplingError> {
    let live = vocabulary.get();
    if logits.len() < live {
        return Err(SamplingError::VocabularyMismatch {
            declared: live,
            logits: logits.len(),
        });
    }
    sample_in_vocabulary(parameters, &logits[..live], live, context, state)
}

fn sample_in_vocabulary(
    parameters: &SamplingParameters,
    logits: &[f32],
    declared_vocabulary: usize,
    context: SampleContext<'_>,
    state: &mut SamplingState,
) -> Result<Sample, SamplingError> {
    validate_inputs(logits, declared_vocabulary, context)?;
    let vocabulary = logits.len();
    let raw = logits
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    // A constraint's mask bounds everything below, the reported logprobs
    // included; without one every live token takes part.
    let permitted = match context.allowed_tokens {
        Some(mask) => (0..vocabulary)
            .map(|token| mask_allows(mask, token))
            .collect::<Vec<_>>(),
        None => vec![true; vocabulary],
    };
    let raw_normalizer = log_normalizer(&raw, &permitted)?;
    let mut adjusted = logits
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let mut allowed = permitted.clone();

    for bias in &parameters.logit_bias {
        let token = usize::try_from(bias.token_id).map_err(|_| SamplingError::BiasOutOfRange {
            token_id: bias.token_id,
            vocabulary,
        })?;
        let Some(logit) = adjusted.get_mut(token) else {
            return Err(SamplingError::BiasOutOfRange {
                token_id: bias.token_id,
                vocabulary,
            });
        };
        *logit += bias.bias;
    }

    let mut prompt_seen = vec![false; vocabulary];
    let mut counts = vec![0u32; vocabulary];
    let prompt_length = context.history_len().saturating_sub(context.emitted_tokens);
    let history = context.history.iter().chain(context.history_tail);
    for (history_index, token_id) in history.enumerate() {
        let token = usize::try_from(*token_id).map_err(|_| SamplingError::HistoryOutOfRange {
            token_id: *token_id,
            vocabulary,
        })?;
        let Some(count) = counts.get_mut(token) else {
            return Err(SamplingError::HistoryOutOfRange {
                token_id: *token_id,
                vocabulary,
            });
        };
        if history_index < prompt_length {
            prompt_seen[token] = true;
        } else {
            *count = count
                .checked_add(1)
                .ok_or(SamplingError::HistoryCountOverflow)?;
        }
    }
    for (token, logit) in adjusted.iter_mut().enumerate() {
        let count = counts[token];
        if count == 0 && !prompt_seen[token] {
            continue;
        }
        if parameters.repetition_penalty != 1.0 {
            if *logit > 0.0 {
                *logit /= parameters.repetition_penalty;
            } else {
                *logit *= parameters.repetition_penalty;
            }
        }
        *logit -= parameters.frequency_penalty * f64::from(count);
        if count != 0 {
            *logit -= parameters.presence_penalty;
        }
    }

    if context.emitted_tokens < parameters.minimum_new_tokens {
        let before = allowed.clone();
        for token_id in context.eos_token_ids {
            let token = usize::try_from(*token_id).map_err(|_| SamplingError::EosOutOfRange {
                token_id: *token_id,
                vocabulary,
            })?;
            let Some(entry) = allowed.get_mut(token) else {
                return Err(SamplingError::EosOutOfRange {
                    token_id: *token_id,
                    vocabulary,
                });
            };
            *entry = false;
        }
        // A constraint that admits only a stop token ends the output even
        // before the minimum length.
        if context.allowed_tokens.is_some() && !allowed.iter().any(|entry| *entry) {
            allowed = before;
        }
    }
    ensure_allowed(&allowed)?;
    let generated = state.generated;

    if parameters.temperature == 0.0 {
        let token = greedy_token(&adjusted, &allowed);
        let top_logprobs =
            raw_top_logprobs(&raw, &permitted, raw_normalizer, parameters.top_logprobs);
        state.generated = state.generated.wrapping_add(1);
        return Ok(Sample {
            token_id: token as u32,
            logprob: raw_normalizer.logprob(raw[token]),
            top_logprobs,
        });
    }

    for value in &mut adjusted {
        *value /= parameters.temperature;
    }
    apply_min_p(&adjusted, &mut allowed, parameters.min_p)?;
    apply_top_k(&adjusted, &mut allowed, parameters.top_k);
    apply_top_p(&adjusted, &mut allowed, parameters.top_p)?;
    ensure_allowed(&allowed)?;

    // Keep order-one Gumbel noise even when the whole row has a large offset.
    let maximum = adjusted[greedy_token(&adjusted, &allowed)];
    let mut selected = None;
    let mut best_score = f64::NEG_INFINITY;
    for (token, is_allowed) in allowed.iter().copied().enumerate() {
        if !is_allowed {
            continue;
        }
        let unit = philox_unit(state.seed, generated, token as u32);
        let score = (adjusted[token] - maximum) - (-unit.ln()).ln();
        if score > best_score {
            best_score = score;
            selected = Some(token);
        }
    }
    let selected = selected.ok_or(SamplingError::AllTokensMasked)?;

    let top_logprobs = raw_top_logprobs(&raw, &permitted, raw_normalizer, parameters.top_logprobs);
    state.generated = state.generated.wrapping_add(1);
    Ok(Sample {
        token_id: selected as u32,
        logprob: raw_normalizer.logprob(raw[selected]),
        top_logprobs,
    })
}

fn raw_top_logprobs(
    raw: &[f64],
    permitted: &[bool],
    normalizer: LogNormalizer,
    count: usize,
) -> Vec<TokenLogprob> {
    let mut top = top_tokens(raw, permitted, count)
        .into_iter()
        .map(|token| TokenLogprob::new(token as u32, normalizer.logprob(raw[token])))
        .collect::<Vec<_>>();
    // Like topk over a grammar-masked row, retain the requested masked tail.
    let remaining = count.saturating_sub(top.len());
    top.extend(
        permitted
            .iter()
            .enumerate()
            .filter(|(_, allowed)| !**allowed)
            .take(remaining)
            .map(|(token, _)| TokenLogprob::new(token as u32, f64::NEG_INFINITY)),
    );
    top
}

fn validate_inputs(
    logits: &[f32],
    declared_vocabulary: usize,
    context: SampleContext<'_>,
) -> Result<(), SamplingError> {
    if logits.len() != declared_vocabulary {
        return Err(SamplingError::VocabularyMismatch {
            declared: declared_vocabulary,
            logits: logits.len(),
        });
    }
    if logits.is_empty() {
        return Err(SamplingError::EmptyVocabulary);
    }
    if logits.len() > MAX_VOCABULARY {
        return Err(SamplingError::VocabularyTooLarge);
    }
    if context.history_len() > MAX_HISTORY {
        return Err(SamplingError::HistoryTooLarge);
    }
    if context.eos_token_ids.len() > MAX_STOP_IDS {
        return Err(SamplingError::TooManyEosTokens);
    }
    for (token_id, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(SamplingError::InvalidLogit { token_id });
        }
    }
    Ok(())
}

fn greedy_token(logits: &[f64], allowed: &[bool]) -> usize {
    let mut best = allowed
        .iter()
        .position(|entry| *entry)
        .expect("allowed token was checked");
    for token in (best + 1)..logits.len() {
        if allowed[token] && logits[token] > logits[best] {
            best = token;
        }
    }
    best
}

fn ranked_tokens(logits: &[f64], allowed: &[bool]) -> Vec<usize> {
    let mut ranked = allowed
        .iter()
        .enumerate()
        .filter_map(|(token, entry)| entry.then_some(token))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .expect("validated adjusted logits are finite")
            .then_with(|| left.cmp(right))
    });
    ranked
}

fn top_tokens(logits: &[f64], allowed: &[bool], count: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    let mut ranked = allowed
        .iter()
        .enumerate()
        .filter_map(|(token, entry)| entry.then_some(token))
        .collect::<Vec<_>>();
    let compare = |left: &usize, right: &usize| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .expect("validated adjusted logits are finite")
            .then_with(|| left.cmp(right))
    };
    if count < ranked.len() {
        ranked.select_nth_unstable_by(count, compare);
        ranked.truncate(count);
    }
    ranked.sort_unstable_by(compare);
    ranked
}

fn apply_top_k(logits: &[f64], allowed: &mut [bool], top_k: usize) {
    // A disabled top-k and a width that covers the whole row truncate nothing;
    // skip the ranking allocation entirely.
    if top_k == 0 || top_k >= allowed.len() {
        return;
    }
    let ranked = top_tokens(logits, allowed, top_k);
    let mut kept = vec![false; allowed.len()];
    for token in ranked {
        kept[token] = true;
    }
    allowed.copy_from_slice(&kept);
}

fn apply_top_p(logits: &[f64], allowed: &mut [bool], top_p: f64) -> Result<(), SamplingError> {
    if top_p == 1.0 {
        return Ok(());
    }
    let ranked = ranked_tokens(logits, allowed);
    let maximum = logits[ranked[0]];
    let total = ranked
        .iter()
        .map(|&token| (logits[token] - maximum).exp())
        .sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::InvalidDistribution);
    }
    let mut cumulative = 0.0;
    let mut keep = 0usize;
    for token in &ranked {
        cumulative += (logits[*token] - maximum).exp();
        keep += 1;
        if cumulative >= top_p * total {
            break;
        }
    }
    for token in &ranked[keep..] {
        allowed[*token] = false;
    }
    Ok(())
}

fn apply_min_p(logits: &[f64], allowed: &mut [bool], min_p: f64) -> Result<(), SamplingError> {
    if min_p == 0.0 {
        return Ok(());
    }
    let maximum = logits
        .iter()
        .zip(allowed.iter())
        .filter_map(|(logit, keep)| keep.then_some(*logit))
        .max_by(f64::total_cmp)
        .ok_or(SamplingError::AllTokensMasked)?;
    let log_threshold = min_p.ln();
    for token in 0..allowed.len() {
        if allowed[token] && logits[token] - maximum < log_threshold {
            allowed[token] = false;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct LogNormalizer {
    maximum: f64,
    log_sum: f64,
}

impl LogNormalizer {
    fn logprob(self, logit: f64) -> f64 {
        (logit - self.maximum) - self.log_sum
    }
}

fn log_normalizer(logits: &[f64], allowed: &[bool]) -> Result<LogNormalizer, SamplingError> {
    ensure_allowed(allowed)?;
    let maximum = logits
        .iter()
        .zip(allowed)
        .filter_map(|(value, entry)| entry.then_some(*value))
        .max_by(f64::total_cmp)
        .ok_or(SamplingError::AllTokensMasked)?;
    let sum = logits
        .iter()
        .zip(allowed)
        .filter_map(|(value, entry)| entry.then_some((*value - maximum).exp()))
        .sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SamplingError::InvalidDistribution);
    }
    Ok(LogNormalizer {
        maximum,
        log_sum: sum.ln(),
    })
}

fn ensure_allowed(allowed: &[bool]) -> Result<(), SamplingError> {
    if allowed.iter().any(|entry| *entry) {
        Ok(())
    } else {
        Err(SamplingError::AllTokensMasked)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplingError {
    InvalidTemperature,
    TopKTooLarge,
    InvalidTopP,
    InvalidMinP,
    MinimumNewTokensTooLarge,
    InvalidPenalty,
    InvalidLogitBias,
    TooManyLogitBiases,
    DuplicateLogitBias(u32),
    TooManyTopLogprobs,
    EmptyVocabulary,
    VocabularyMismatch {
        declared: usize,
        logits: usize,
    },
    VocabularyTooLarge,
    HistoryTooLarge,
    TooManyEosTokens,
    InvalidLogit {
        token_id: usize,
    },
    BiasOutOfRange {
        token_id: u32,
        vocabulary: usize,
    },
    HistoryOutOfRange {
        token_id: u32,
        vocabulary: usize,
    },
    EosOutOfRange {
        token_id: u32,
        vocabulary: usize,
    },
    HistoryCountOverflow,
    AllTokensMasked,
    InvalidDistribution,
    /// A sampled token lies outside the row's constraint mask.
    DisallowedToken(u32),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "sampling failed: {self:?}")
    }
}

impl Error for SamplingError {}
