//! Real bounded generations before the process publishes readiness.
//!
//! Warmup runs waves of 1, 2, 4, ... concurrent requests up to the served
//! sequence count, so a bundle captures each of its decode graph buckets at
//! startup rather than stalling every live sequence the first time a batch
//! reaches a new bucket (docs/serving-profiles.md). The shared
//! bundle runtime logs each graph it captures ("graph capture rank R bucket B
//! items I"), so the capture lines between a wave's start and end lines are
//! the buckets that wave reached. A wave whose requests do not all decode in
//! one step leaves its bucket to be captured when serving first reaches it.
//! Waves need several output tokens to overlap in decode, so a profile that
//! admits fewer than [`WAVE_MINIMUM_OUTPUT_TOKENS`] warms up with one request,
//! as does a package without batch limits (the caller passes one sequence).
//!
//! With speculation, decode waves pin zero drafts so the policy cannot
//! turn their steps into verify steps, and verify waves follow. Each pins its requests' draft
//! tokens so one decode step reaches one (row bucket, item bucket) verify
//! graph shape; together they reach every shape the served sequences, draft
//! depths and verify rows allow. Single requests with prompts of growing
//! length then give a drafter that catches up on context a first step of
//! each size. Warmup outputs are discarded.
use runtime_bundle_sdk::warmup::waves;
use runtime_engine::warmup::{VerifyWarmup, context_ladder, verify_shapes};
use runtime_server::{
    CancellationSignal, EventSender, GenerationBackend, GenerationEvent, PreparedPrompt,
    PreparedRequest, RequestAcceptanceProfile, RequestedLogprobs,
};

/// Longest output of one warmup request in a wave of more than one, so the
/// wave's requests overlap in the decode steps after their prefills.
const WAVE_OUTPUT_TOKENS: usize = 16;

/// Fewest output tokens that let a wave's requests share decode steps.
const WAVE_MINIMUM_OUTPUT_TOKENS: usize = 4;

/// All decode, verify and context waves share this startup allowance.
const WARMUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

struct WarmupDeadline {
    at: tokio::time::Instant,
    unwarmed: Vec<String>,
}

impl WarmupDeadline {
    fn new(timeout: std::time::Duration) -> Self {
        Self {
            at: tokio::time::Instant::now() + timeout,
            unwarmed: Vec::new(),
        }
    }

    fn expired_key(&mut self, label: &str) -> String {
        self.unwarmed.push(label.to_owned());
        format!("key {label} not warmed: total warmup deadline expired")
    }
}

/// Parse the opt-in development switch without changing production defaults.
pub(crate) fn development_skip_warmup(value: Option<&std::ffi::OsStr>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err("CHOPINFER_DEV_SKIP_WARMUP must be 0 or 1".to_owned()),
    }
}

/// Cleanup is a worker barrier: readiness alone cannot prove a timed-out
/// request has retired or distinguish a pending failure from a healthy worker.
pub(crate) async fn warmup(
    backend: &dyn GenerationBackend,
    model: String,
    limit: usize,
    profile: RequestAcceptanceProfile,
    sequences: usize,
    skip_warmup: bool,
    verify: Option<VerifyWarmup>,
) -> Result<(), String> {
    if skip_warmup {
        eprintln!(
            "WARNING: CHOPINFER_DEV_SKIP_WARMUP=1: development startup skips generation validation \
             and warmup; first requests may capture graphs"
        );
        eprintln!("startup generation warmup: skipped (development)");
        return Ok(());
    }
    let mut deadline = WarmupDeadline::new(WARMUP_TIMEOUT);
    backend.startup_warmup(true).await?;
    let result = warmup_inner(
        backend,
        model,
        limit,
        profile,
        sequences,
        verify,
        &mut deadline,
    )
    .await;
    finish_warmup(backend, result).await
}

async fn finish_warmup(
    backend: &dyn GenerationBackend,
    result: Result<(), String>,
) -> Result<(), String> {
    backend.startup_warmup(false).await?;
    if let Err(error) = result {
        eprintln!(
            "WARNING: startup warmup incomplete: {error}; remaining keys were not warmed; continuing to listen"
        );
    }
    match backend.readiness() {
        runtime_server::BackendReadiness::Ready => Ok(()),
        runtime_server::BackendReadiness::NotReady(reason)
        | runtime_server::BackendReadiness::Terminated(reason) => Err(reason),
    }
}

async fn warmup_inner(
    backend: &dyn GenerationBackend,
    model: String,
    limit: usize,
    profile: RequestAcceptanceProfile,
    sequences: usize,
    verify: Option<VerifyWarmup>,
    deadline: &mut WarmupDeadline,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    if limit < 2 {
        return Err("warmup needs capacity for a prompt and one generated token".to_owned());
    }
    let (prompt, _) = backend
        .prepare_prompt(PreparedPrompt::Raw("Hello".to_owned()))
        .map_err(|error| format!("warmup tokenization failed: {error:?}"))?;
    let PreparedPrompt::TokenIds(mut tokens) = prompt else {
        return Err("warmup backend did not return token IDs".to_owned());
    };
    let first = *tokens.first().ok_or("warmup prompt encoded no tokens")?;
    // One request generates two tokens; a wave's requests generate more so
    // they decode together, within what the profile admits.
    let (prompt_tokens, single_output, wave_output) = match profile {
        RequestAcceptanceProfile::OneToken { prompt_tokens } => (prompt_tokens, 1, 1),
        RequestAcceptanceProfile::ExactShape {
            prompt_tokens,
            maximum_generated_tokens,
        } => (
            prompt_tokens,
            maximum_generated_tokens.min(2),
            maximum_generated_tokens.min(2),
        ),
        RequestAcceptanceProfile::Bounded {
            maximum_prompt_tokens,
            maximum_generated_tokens,
            ..
        } => (
            tokens.len().min(maximum_prompt_tokens).min(limit - 1),
            maximum_generated_tokens.min(2),
            maximum_generated_tokens.min(WAVE_OUTPUT_TOKENS),
        ),
        _ => (tokens.len().min(limit - 1), 2, WAVE_OUTPUT_TOKENS),
    };
    if prompt_tokens == 0 || prompt_tokens >= limit {
        return Err("warmup prompt does not fit the serving profile".to_owned());
    }
    tokens.resize(prompt_tokens, first);
    let sequences = if wave_output.min(limit - prompt_tokens) < WAVE_MINIMUM_OUTPUT_TOKENS {
        1
    } else {
        sequences
    };
    let mut result = Ok(());
    for wave in waves(sequences) {
        eprintln!("startup warmup wave: {wave} sequences begin");
        let wave_started = std::time::Instant::now();
        let output_tokens = if wave == 1 {
            single_output
        } else {
            wave_output
        };
        // With speculation, a decode wave pins no drafts, so the policy
        // cannot turn its steps into verify steps and miss the decode graph.
        let requests = vec![(tokens.clone(), verify.as_ref().map(|_| 0)); wave];
        result = run_wave(
            backend,
            &model,
            &format!("{wave}"),
            &requests,
            output_tokens.min(limit - prompt_tokens),
            deadline,
        )
        .await;
        eprintln!(
            "startup warmup wave: {wave} sequences end, {:.3}s",
            wave_started.elapsed().as_secs_f64()
        );
        if let Err(error) = &result {
            eprintln!("WARNING: startup warmup: {error}");
            if !matches!(backend.readiness(), runtime_server::BackendReadiness::Ready) {
                return result;
            }
        }
    }
    if let Some(verify) = verify {
        // Verify waves need output for two steps at the deepest pin, which
        // some request profiles do not admit.
        let admitted = match profile {
            RequestAcceptanceProfile::OneToken { .. } => 1,
            RequestAcceptanceProfile::ExactShape {
                maximum_generated_tokens,
                ..
            }
            | RequestAcceptanceProfile::Bounded {
                maximum_generated_tokens,
                ..
            } => maximum_generated_tokens,
            _ => usize::MAX,
        };
        if admitted < verify_output(&verify) {
            eprintln!("startup verify warmup: skipped, the request profile admits too few tokens");
        } else {
            result = verify_waves(
                backend, &model, &tokens, limit, sequences, &verify, deadline,
            )
            .await;
        }
    }
    eprintln!(
        "startup generation warmup: {:.3}s",
        started.elapsed().as_secs_f64()
    );
    if !deadline.unwarmed.is_empty() {
        return Err(format!(
            "total warmup deadline expired; keys not warmed: {}",
            deadline.unwarmed.join(", ")
        ));
    }
    result
}

/// Output tokens of a verify warmup request: one from prefill, then room for
/// two steps at the deepest pin with every draft accepted, so the wave's
/// requests, admitted together, share at least one step.
fn verify_output(verify: &VerifyWarmup) -> usize {
    2 * (verify.maximum_draft_tokens + 1) + 1
}

/// Runs one wave per verify shape, then the context ladder.
async fn verify_waves(
    backend: &dyn GenerationBackend,
    model: &str,
    tokens: &[u32],
    limit: usize,
    sequences: usize,
    verify: &VerifyWarmup,
    deadline: &mut WarmupDeadline,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    let shapes = verify_shapes(sequences, verify);
    let output = verify_output(verify);
    if tokens.len() + output > limit {
        eprintln!("startup verify warmup: skipped, the context is too short");
        return Ok(());
    }
    for shape in &shapes {
        eprintln!(
            "startup verify warmup: rows {} items {} begin",
            shape.row_bucket, shape.item_bucket
        );
        let shape_started = std::time::Instant::now();
        let requests = shape
            .drafts
            .iter()
            .map(|&drafts| (tokens.to_vec(), Some(drafts)))
            .collect::<Vec<_>>();
        let label = format!("verify-{}-{}", shape.row_bucket, shape.item_bucket);
        let result = run_wave(backend, model, &label, &requests, output, deadline).await;
        eprintln!(
            "startup verify warmup: rows {} items {} end, {:.3}s",
            shape.row_bucket,
            shape.item_bucket,
            shape_started.elapsed().as_secs_f64()
        );
        if let Err(error) = result {
            eprintln!(
                "WARNING: startup verify warmup: rows {} items {} not warmed: {error}",
                shape.row_bucket, shape.item_bucket
            );
            if !matches!(backend.readiness(), runtime_server::BackendReadiness::Ready) {
                return Err(error);
            }
        }
    }
    let pin = verify
        .item_rows
        .iter()
        .filter(|&&rows| rows > 1 && rows - 1 <= verify.maximum_draft_tokens)
        .max()
        .map(|rows| rows - 1);
    let ladder = context_ladder(verify.step_rows, limit.saturating_sub(output));
    if let Some(pin) = pin {
        let first = tokens.first().copied().unwrap_or(0);
        for &prompt in &ladder {
            eprintln!("startup verify warmup: context {prompt} begin");
            let context_started = std::time::Instant::now();
            let request = (vec![first; prompt], Some(pin));
            let result = run_wave(
                backend,
                model,
                &format!("context-{prompt}"),
                &[request],
                output,
                deadline,
            )
            .await;
            eprintln!(
                "startup verify warmup: context {prompt} end, {:.3}s",
                context_started.elapsed().as_secs_f64()
            );
            if let Err(error) = result {
                eprintln!("WARNING: startup verify warmup: context {prompt} not warmed: {error}");
                if !matches!(backend.readiness(), runtime_server::BackendReadiness::Ready) {
                    return Err(error);
                }
            }
        }
    }
    eprintln!(
        "startup verify warmup: {} verify shapes considered, {} context lengths, {:.3}s",
        shapes.len(),
        if pin.is_some() { ladder.len() } else { 0 },
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Submits the wave's requests, each a prompt and an optional draft pin,
/// back to back and waits for all of them to generate a token.
async fn run_wave(
    backend: &dyn GenerationBackend,
    model: &str,
    label: &str,
    requests: &[(Vec<u32>, Option<usize>)],
    max_new_tokens: usize,
    deadline: &mut WarmupDeadline,
) -> Result<(), String> {
    // An expired timeout can still poll a ready future once. Check before
    // submission so even an immediately ready backend cannot start more work.
    if tokio::time::Instant::now() >= deadline.at {
        return Err(deadline.expired_key(label));
    }
    let wave = requests.len();
    let reservation = requests
        .iter()
        .map(|(tokens, pin)| {
            tokens
                .len()
                .saturating_add(max_new_tokens)
                .saturating_add(pin.unwrap_or(0))
        })
        .max()
        .unwrap_or(1);
    let capacity = backend.startup_wave_capacity(reservation);
    if wave > capacity {
        eprintln!(
            "WARNING: startup warmup key {label} not warmed: {wave} sequences need more KV pages than capacity {capacity}"
        );
        return Ok(());
    }
    let cancellation = CancellationSignal::new();
    let run = async {
        let mut streams = Vec::with_capacity(wave);
        let mut batch = Vec::with_capacity(wave);
        for (index, (tokens, pin)) in requests.iter().enumerate() {
            let request = PreparedRequest {
                request_id: if label == "1" {
                    "startup-warmup".to_owned()
                } else {
                    format!("startup-warmup-{label}-{index}")
                },
                client_request_id: None,
                model: model.to_owned(),
                prompt: PreparedPrompt::TokenIds(tokens.to_vec()),
                max_new_tokens,
                stop: Vec::new(),
                // Every request of a wave runs to its length, so the wave
                // decodes together.
                decoding: runtime_server::DecodingOptions {
                    ignore_eos: wave > 1 || pin.is_some(),
                    warmup_draft_tokens: *pin,
                    ..runtime_server::DecodingOptions::default()
                },
                sampling: runtime_sampling::SamplingParameters::new(0.0)
                    .map_err(|error| error.to_string())?,
                seed: 0,
                logprobs: RequestedLogprobs {
                    enabled: false,
                    top: 0,
                    echo: false,
                    prompt_top: None,
                    return_token_ids: false,
                },
                chat: None,
                constraint: None,
                cache_salt: Some("startup-warmup-private".to_owned()),
                cancellation: cancellation.clone(),
            };
            // Room for every token event of the request and its end, so no
            // stream waits on another being read.
            let (sender, events) = tokio::sync::mpsc::channel(max_new_tokens + 32);
            batch.push((request, EventSender::new(sender)));
            streams.push(events);
        }
        // One group, so a worker admits the whole wave into the same step
        // rather than one request per step, which would stagger the wave.
        for admission in backend.submit_batch(batch) {
            admission.await.map_err(|error| format!("{error:?}"))?;
        }
        for events in &mut streams {
            finished(events).await?;
        }
        Ok::<(), String>(())
    };
    let result = tokio::time::timeout_at(deadline.at, run)
        .await
        .map_err(|_| deadline.expired_key(label))
        .and_then(|result| result.map_err(|error| format!("key {label} not warmed: {error}")));
    cancellation.cancel();
    result
}

async fn finished(events: &mut tokio::sync::mpsc::Receiver<GenerationEvent>) -> Result<(), String> {
    while let Some(event) = events.recv().await {
        match event {
            GenerationEvent::Finished { usage, .. } if usage.completion_tokens > 0 => {
                return Ok(());
            }
            GenerationEvent::Failed { message, .. } => return Err(message),
            _ => {}
        }
    }
    Err("warmup ended without a generated token".to_owned())
}

#[cfg(test)]
mod tests {
    use super::waves;

    #[test]
    fn waves_double_up_to_the_served_sequences() {
        assert_eq!(waves(1), [1]);
        assert_eq!(waves(2), [1, 2]);
        assert_eq!(waves(8), [1, 2, 4, 8]);
        assert_eq!(waves(32), [1, 2, 4, 8, 16, 32]);
        assert_eq!(waves(48), [1, 2, 4, 8, 16, 32, 48]);
        assert_eq!(waves(256), [1, 2, 4, 8, 16, 32, 64, 128, 256]);
        assert_eq!(waves(0), [1]);
    }
}

#[cfg(test)]
mod wave_tests {
    use std::sync::Mutex;

    use runtime_server::{
        AdmissionError, BackendReadiness, BoxFuture, EventSender, FinishReason, GenerationBackend,
        GenerationEvent, PreparedPrompt, PreparedRequest, RequestAcceptanceProfile, TokenUsage,
    };

    use super::{development_skip_warmup, warmup, waves};
    use runtime_engine::warmup::{VerifyWarmup, context_ladder, verify_shapes};

    /// The wave in progress, its held event streams, and every submitted
    /// request id with its wave size.
    type WaveState = (
        usize,
        Vec<EventSender>,
        Vec<(usize, String)>,
        Vec<(Option<usize>, usize)>,
    );

    /// Holds each request's events until the whole current wave is in flight,
    /// then finishes the wave, or fails it when it is `fail_wave`.
    struct WaveBackend {
        expected: Vec<usize>,
        fail_wave: Option<usize>,
        fatal: bool,
        state: Mutex<WaveState>,
        /// The size of every group submitted together.
        groups: Mutex<Vec<usize>>,
        lifecycle: Mutex<Vec<bool>>,
    }

    impl WaveBackend {
        fn new(sequences: usize, fail_wave: Option<usize>) -> Self {
            Self::with_waves(waves(sequences), fail_wave)
        }

        fn with_waves(expected: Vec<usize>, fail_wave: Option<usize>) -> Self {
            Self {
                expected,
                fail_wave,
                fatal: false,
                state: Mutex::new((0, Vec::new(), Vec::new(), Vec::new())),
                groups: Mutex::new(Vec::new()),
                lifecycle: Mutex::new(Vec::new()),
            }
        }

        /// Each request's draft pin and prompt length, in submission order.
        fn pins(&self) -> Vec<(Option<usize>, usize)> {
            self.state.lock().unwrap().3.clone()
        }

        /// Request ids by wave size, in submission order.
        fn submitted(&self) -> Vec<(usize, String)> {
            self.state.lock().unwrap().2.clone()
        }
    }

    impl GenerationBackend for WaveBackend {
        fn startup_warmup(&self, begin: bool) -> BoxFuture<'_, Result<(), String>> {
            self.lifecycle.lock().unwrap().push(begin);
            Box::pin(async { Ok(()) })
        }

        fn readiness(&self) -> BackendReadiness {
            if self.fatal && !self.submitted().is_empty() {
                BackendReadiness::Terminated("generation worker failed".to_owned())
            } else {
                BackendReadiness::Ready
            }
        }

        fn submit_batch(
            &self,
            requests: Vec<(PreparedRequest, EventSender)>,
        ) -> Vec<BoxFuture<'_, Result<(), AdmissionError>>> {
            self.groups.lock().unwrap().push(requests.len());
            requests
                .into_iter()
                .map(|(request, events)| self.submit(request, events))
                .collect()
        }

        fn maximum_sequence_tokens(&self) -> usize {
            1024
        }

        fn prompt_token_count(&self, _prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
            Ok(2)
        }

        fn prepare_prompt(
            &self,
            _prompt: PreparedPrompt,
        ) -> Result<(PreparedPrompt, usize), AdmissionError> {
            Ok((PreparedPrompt::TokenIds(vec![5, 6]), 2))
        }

        fn submit(
            &self,
            request: PreparedRequest,
            events: EventSender,
        ) -> BoxFuture<'_, Result<(), AdmissionError>> {
            let mut state = self.state.lock().unwrap();
            let wave = self.expected[state.0];
            let prompt = match &request.prompt {
                PreparedPrompt::TokenIds(tokens) => tokens.len(),
                _ => 0,
            };
            state.3.push((request.decoding.warmup_draft_tokens, prompt));
            state.2.push((wave, request.request_id));
            state.1.push(events);
            if state.1.len() == wave {
                // Every request of the wave is in flight at once.
                for events in state.1.drain(..) {
                    let event = if self.fail_wave == Some(wave) {
                        GenerationEvent::Failed {
                            message: format!("wave {wave} failed"),
                            code: None,
                        }
                    } else {
                        GenerationEvent::Finished {
                            reason: FinishReason::Length,
                            usage: TokenUsage {
                                prompt_tokens: 2,
                                completion_tokens: 1,
                                cached_tokens: None,
                            },
                        }
                    };
                    events.try_emit(event).unwrap();
                }
                state.0 += 1;
            }
            Box::pin(async { Ok(()) })
        }
    }

    fn run(
        backend: &WaveBackend,
        sequences: usize,
        profile: RequestAcceptanceProfile,
    ) -> Result<(), String> {
        run_with_mode(backend, sequences, profile, false, None)
    }

    fn run_with_mode(
        backend: &WaveBackend,
        sequences: usize,
        profile: RequestAcceptanceProfile,
        skip_warmup: bool,
        verify: Option<VerifyWarmup>,
    ) -> Result<(), String> {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(async {
                tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    warmup(
                        backend,
                        "model".to_owned(),
                        64,
                        profile,
                        sequences,
                        skip_warmup,
                        verify,
                    ),
                )
                .await
                .expect("each wave is submitted whole before it is awaited")
            })
    }

    #[test]
    fn development_warmup_skip_is_explicit_and_validated() {
        assert_eq!(development_skip_warmup(None), Ok(false));
        assert_eq!(development_skip_warmup(Some("0".as_ref())), Ok(false));
        assert_eq!(development_skip_warmup(Some("1".as_ref())), Ok(true));
        for value in ["", "true", "false", "2"] {
            assert!(development_skip_warmup(Some(value.as_ref())).is_err());
        }
    }

    #[test]
    fn skipping_warmup_submits_no_generation_requests() {
        let backend = WaveBackend::new(8, Some(1));
        run_with_mode(&backend, 8, RequestAcceptanceProfile::General, true, None).unwrap();
        assert!(backend.submitted().is_empty());
    }

    #[test]
    fn each_wave_submits_all_its_requests_before_waiting() {
        let backend = WaveBackend::new(8, None);
        run(&backend, 8, RequestAcceptanceProfile::General).unwrap();
        let submitted = backend.submitted();
        let sizes = submitted.iter().map(|(wave, _)| *wave).collect::<Vec<_>>();
        assert_eq!(sizes, [1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8]);
        // Each wave is one group, which a worker admits into one step.
        assert_eq!(*backend.groups.lock().unwrap(), [1, 2, 4, 8]);
        assert_eq!(submitted[0].1, "startup-warmup");
        assert_eq!(submitted[14].1, "startup-warmup-8-7");
    }

    #[test]
    fn a_failed_wave_is_nonfatal_for_a_healthy_worker() {
        let backend = WaveBackend::new(8, Some(2));
        assert_eq!(run(&backend, 8, RequestAcceptanceProfile::General), Ok(()));
        assert_eq!(backend.submitted().len(), 15);
    }

    #[test]
    fn a_timed_out_wave_cancels_and_can_leave_a_healthy_backend() {
        // This backend holds the first stream waiting for a second request.
        let backend = WaveBackend::with_waves(vec![2], None);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let error = runtime
            .block_on(async {
                super::run_wave(
                    &backend,
                    "test/model",
                    "timeout",
                    &[(vec![5, 6], None)],
                    2,
                    &mut super::WarmupDeadline::new(std::time::Duration::from_millis(1)),
                )
                .await
            })
            .unwrap_err();
        assert!(error.contains("key timeout not warmed"));
        assert!(error.contains("total warmup deadline expired"));
        assert_eq!(backend.readiness(), BackendReadiness::Ready);
        runtime
            .block_on(super::finish_warmup(&backend, Err(error)))
            .unwrap();
        assert!(
            backend
                .state
                .lock()
                .unwrap()
                .1
                .iter()
                .all(|events| events.is_closed())
        );
    }

    #[test]
    fn total_deadline_stops_all_remaining_waves_and_reports_their_keys() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let decode = waves(4);
        let shapes = verify_shapes(4, &dflash());
        let ladder = context_ladder(32, 64 - 17);
        let mut sizes = decode.clone();
        sizes.extend(shapes.iter().map(|shape| shape.drafts.len()));
        sizes.extend(ladder.iter().map(|_| 1));
        let mut keys = decode.iter().map(usize::to_string).collect::<Vec<_>>();
        keys.extend(
            shapes
                .iter()
                .map(|shape| format!("verify-{}-{}", shape.row_bucket, shape.item_bucket)),
        );
        keys.extend(ladder.iter().map(|prompt| format!("context-{prompt}")));

        // Exhaust the same budget in each phase, including after earlier
        // waves succeeded. The blocked wave waits for one extra request.
        for stalled in [0, decode.len(), decode.len() + shapes.len()] {
            let mut expected = sizes.clone();
            expected[stalled] += 1;
            let backend = WaveBackend::with_waves(expected, None);
            runtime.block_on(async {
                let mut deadline = super::WarmupDeadline::new(std::time::Duration::from_millis(10));
                backend.startup_warmup(true).await.unwrap();
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(1),
                    super::warmup_inner(
                        &backend,
                        "model".to_owned(),
                        64,
                        RequestAcceptanceProfile::General,
                        4,
                        Some(dflash()),
                        &mut deadline,
                    ),
                )
                .await
                .expect("the total deadline must bound all phases");
                assert_eq!(deadline.unwarmed, keys[stalled..]);
                assert_eq!(
                    result,
                    Err(format!(
                        "total warmup deadline expired; keys not warmed: {}",
                        keys[stalled..].join(", ")
                    ))
                );
                assert_eq!(*backend.groups.lock().unwrap(), sizes[..=stalled]);
                super::finish_warmup(&backend, result).await.unwrap();
                assert_eq!(*backend.lifecycle.lock().unwrap(), [true, false]);
                assert!(
                    backend
                        .state
                        .lock()
                        .unwrap()
                        .1
                        .iter()
                        .all(|events| events.is_closed())
                );
            });
        }
    }

    #[test]
    fn expired_deadline_does_not_submit_even_to_an_immediately_ready_backend() {
        let backend = WaveBackend::new(1, None);
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(async {
                let mut deadline = super::WarmupDeadline::new(std::time::Duration::ZERO);
                let result = super::run_wave(
                    &backend,
                    "model",
                    "1",
                    &[(vec![5, 6], None)],
                    2,
                    &mut deadline,
                )
                .await;
                assert!(result.unwrap_err().contains("key 1 not warmed"));
                assert!(backend.groups.lock().unwrap().is_empty());
            });
    }

    #[test]
    fn a_failed_worker_is_fatal() {
        let mut backend = WaveBackend::new(8, Some(1));
        backend.fatal = true;
        assert_eq!(
            run(&backend, 8, RequestAcceptanceProfile::General),
            Err("generation worker failed".to_owned())
        );
    }

    #[test]
    fn profiles_too_short_to_overlap_warm_up_with_one_request() {
        for profile in [
            RequestAcceptanceProfile::OneToken { prompt_tokens: 2 },
            RequestAcceptanceProfile::ExactShape {
                prompt_tokens: 2,
                maximum_generated_tokens: 8,
            },
        ] {
            let backend = WaveBackend::new(1, None);
            run(&backend, 8, profile).unwrap();
            assert_eq!(backend.submitted().len(), 1);
        }
    }

    fn dflash() -> VerifyWarmup {
        VerifyWarmup {
            item_rows: vec![2, 4, 8],
            maximum_draft_tokens: 7,
            step_rows: 32,
            drafting_items: None,
            exact_graphs: false,
        }
    }

    #[test]
    fn verify_waves_follow_the_decode_waves_with_pinned_drafts() {
        let shapes = verify_shapes(4, &dflash());
        let ladder = context_ladder(32, 64 - 17);
        let mut expected = waves(4);
        expected.extend(shapes.iter().map(|shape| shape.drafts.len()));
        expected.extend(ladder.iter().map(|_| 1));
        let backend = WaveBackend::with_waves(expected.clone(), None);
        run_with_mode(
            &backend,
            4,
            RequestAcceptanceProfile::General,
            false,
            Some(dflash()),
        )
        .unwrap();
        let pins = backend.pins();
        let decode = waves(4).iter().sum::<usize>();
        // Decode waves pin no drafts, so they reach the decode graphs.
        assert!(pins[..decode].iter().all(|(pin, _)| *pin == Some(0)));
        let pinned: Vec<Option<usize>> = shapes
            .iter()
            .flat_map(|shape| shape.drafts.iter().map(|&drafts| Some(drafts)))
            .collect();
        let verify_end = decode + pinned.len();
        assert_eq!(
            pins[decode..verify_end]
                .iter()
                .map(|(pin, _)| *pin)
                .collect::<Vec<_>>(),
            pinned
        );
        // The context ladder: single requests with the deepest pin and
        // growing prompts.
        assert_eq!(
            pins[verify_end..],
            ladder
                .iter()
                .map(|&prompt| (Some(7), prompt))
                .collect::<Vec<_>>()
        );
        let submitted = backend.submitted();
        assert_eq!(submitted[decode].1, "startup-warmup-verify-2-1-0");
    }

    #[test]
    fn verify_warmup_is_skipped_when_the_profile_admits_too_few_tokens() {
        let backend = WaveBackend::new(4, None);
        run_with_mode(
            &backend,
            4,
            RequestAcceptanceProfile::Bounded {
                maximum_prompt_tokens: 8,
                maximum_generated_tokens: 8,
                maximum_total_tokens: 16,
            },
            false,
            Some(dflash()),
        )
        .unwrap();
        // Only the decode waves ran.
        assert_eq!(backend.pins().len(), waves(4).iter().sum::<usize>());
        assert!(backend.pins().iter().all(|(pin, _)| *pin == Some(0)));
    }
}
