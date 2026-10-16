//! Debug-only teacher forcing of served requests onto known token sequences.
//!
//! When `serve --debug-reference-tokens FILE` names a JSON file of token
//! sequences, a served engine drafts and emits those sequences instead of its
//! own samples. Host drafts are the reference continuation of the request's
//! history, and every device-sampled row whose history continues a reference
//! reports the reference token with its raw logprob and the row's raw top
//! logprobs. The rows are computed exactly as without forcing, so the
//! logprobs a request returns score each reference position with the row
//! kind (prefill, decode or verify at some depth) that produced it.
//!
//! Qualification uses this to score verify rows against an exact reference
//! (`tools/qual/verify_exact.py`). Outputs are not model samples, so only an
//! explicit command-line flag enables it and serving announces it. The
//! environment variable that once did is refused at startup, because an
//! inherited variable must not silently change what a service returns.

use std::path::Path;
use std::rc::Rc;

use runtime_bundle_abi::{HostServiceResultV1, OwnedLogitsViewV1};
use runtime_engine::{ModelBatchItem, SampledRow};

use crate::executor::{DeviceLogitsSamplerV1, DevicePromptRowV1};

/// The retired variable that named the reference file; serving refuses it.
pub const REFERENCE_TOKENS_ENV_V1: &str = "CHOPINFER_DEBUG_REFERENCE_TOKENS";

/// Refuses to start while the retired forcing variable is set.
pub fn refuse_reference_tokens_variable_v1(
    variable: Option<std::ffi::OsString>,
) -> Result<(), String> {
    match variable {
        None => Ok(()),
        Some(_) => Err(format!(
            "{REFERENCE_TOKENS_ENV_V1} is set; serving refuses it because forced tokens are \
             not model samples. Unset it, and for a qualification run pass \
             --debug-reference-tokens FILE instead"
        )),
    }
}

/// Token sequences a debug run forces drafts and sampled tokens to follow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceTokensV1 {
    sequences: Vec<Vec<u32>>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceFileV1 {
    sequences: Vec<Vec<u32>>,
}

impl ReferenceTokensV1 {
    #[must_use]
    pub fn new(sequences: Vec<Vec<u32>>) -> Self {
        Self { sequences }
    }

    /// Parses `{"sequences": [[token, ...], ...]}`.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let file: ReferenceFileV1 = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid reference token file: {error}"))?;
        if file.sequences.is_empty() || file.sequences.iter().any(Vec::is_empty) {
            return Err("the reference token file needs nonempty sequences".to_owned());
        }
        Ok(Self::new(file.sequences))
    }

    /// Reads the reference file at `path`.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        Self::from_json(&bytes)
    }

    #[must_use]
    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    /// The tokens after `history` on which every sequence starting with
    /// `history` agrees; empty when none does.
    #[must_use]
    pub fn continuation(&self, history: &[u32]) -> &[u32] {
        let mut matches = self
            .sequences
            .iter()
            .filter(|sequence| sequence.len() > history.len() && sequence.starts_with(history))
            .map(|sequence| &sequence[history.len()..]);
        let Some(first) = matches.next() else {
            return &[];
        };
        let agreed = matches.fold(first.len(), |agreed, other| {
            first
                .iter()
                .zip(other)
                .take(agreed)
                .take_while(|(left, right)| left == right)
                .count()
        });
        &first[..agreed]
    }

    /// The reference token and top-logprob count a sampled row is forced to.
    fn forced(&self, item: &ModelBatchItem) -> Option<(u32, usize)> {
        let work = item.sampling()?;
        let token = *self.continuation(&work.history.to_vec()).first()?;
        Some((token, work.parameters.top_logprobs()))
    }
}

/// Wraps the device sampler so rows that continue a reference are scored
/// against the reference token instead of sampled.
pub struct ReferenceForcingSamplerV1 {
    inner: Rc<dyn DeviceLogitsSamplerV1>,
    reference: Rc<ReferenceTokensV1>,
}

impl ReferenceForcingSamplerV1 {
    #[must_use]
    pub fn new(inner: Rc<dyn DeviceLogitsSamplerV1>, reference: Rc<ReferenceTokensV1>) -> Self {
        Self { inner, reference }
    }
}

/// A row the inner sampler scores against `token`, reporting `top` raw
/// alternatives, like a prompt row.
fn scoring_item(item: &ModelBatchItem, token: u32, top: usize) -> ModelBatchItem {
    ModelBatchItem::new(item.request_id(), item.phase(), Vec::new(), true)
        .with_prompt_targets(vec![Some(token)], top)
}

impl ReferenceForcingSamplerV1 {
    fn sample_with_policy(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: Option<crate::executor::SamplingDispatch>,
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        let scoring = items
            .iter()
            .map(|item| {
                self.reference
                    .forced(item)
                    .map(|(token, top)| (token, scoring_item(item, token, top)))
            })
            .collect::<Vec<_>>();
        if scoring.iter().all(Option::is_none) {
            return match dispatch {
                Some(policy) => self.inner.sample_view_with_dispatch(view, items, policy),
                None => self.inner.sample_view(view, items),
            };
        }
        // A scored row with neither flag yields only its score; an unforced
        // output row is sampled as before.
        let rows = items
            .iter()
            .zip(&scoring)
            .map(|(item, scoring)| match scoring {
                Some((token, scoring)) => DevicePromptRowV1 {
                    item: scoring,
                    target: Some(*token),
                    prompt: false,
                    output: false,
                },
                None => DevicePromptRowV1 {
                    item,
                    target: None,
                    prompt: false,
                    output: true,
                },
            })
            .collect::<Vec<_>>();
        match dispatch {
            Some(policy) => self.inner.score_view_with_dispatch(view, &rows, policy),
            None => self.inner.score_view(view, &rows),
        }
        .map(|(scores, _)| scores)
    }

    fn score_with_policy(
        &self,
        view: OwnedLogitsViewV1,
        rows: &[DevicePromptRowV1<'_>],
        dispatch: Option<crate::executor::SamplingDispatch>,
    ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
        let scoring = rows
            .iter()
            .map(|row| {
                (row.output && row.target.is_none())
                    .then(|| self.reference.forced(row.item))
                    .flatten()
                    .map(|(token, top)| (token, scoring_item(row.item, token, top)))
            })
            .collect::<Vec<_>>();
        if scoring.iter().all(Option::is_none) {
            return match dispatch {
                Some(policy) => self.inner.score_view_with_dispatch(view, rows, policy),
                None => self.inner.score_view(view, rows),
            };
        }
        let replaced = rows
            .iter()
            .zip(&scoring)
            .map(|(row, scoring)| match scoring {
                Some((token, scoring)) => DevicePromptRowV1 {
                    item: scoring,
                    target: Some(*token),
                    prompt: false,
                    output: false,
                },
                None => DevicePromptRowV1 {
                    item: row.item,
                    target: row.target,
                    prompt: row.prompt,
                    output: row.output,
                },
            })
            .collect::<Vec<_>>();
        let (scores, mut generated) = match dispatch {
            Some(policy) => self.inner.score_view_with_dispatch(view, &replaced, policy),
            None => self.inner.score_view(view, &replaced),
        }?;
        // The last prefill row of a prompt-scored request emits its token
        // through `generated`; the executor reads output-only rows' scores.
        for ((row, scoring), score) in rows.iter().zip(&scoring).zip(&scores) {
            if scoring.is_some() && row.prompt {
                generated.push(score.clone());
            }
        }
        Ok((scores, generated))
    }
}

impl DeviceLogitsSamplerV1 for ReferenceForcingSamplerV1 {
    fn sampling_dispatch(
        &self,
        items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<crate::executor::SamplingDispatch> {
        let mut dispatch = crate::executor::SamplingDispatch {
            rows: items.len() as u32,
            simple: true,
            top: false,
            greedy: true,
        };
        for item in items {
            let row = if let Some((_, top)) = self.reference.forced(item) {
                crate::executor::SamplingDispatch {
                    rows: 1,
                    simple: true,
                    top: top != 0,
                    greedy: top <= 1,
                }
            } else {
                self.inner.sampling_dispatch(&[item])?
            };
            dispatch.simple &= row.simple;
            dispatch.top |= row.top;
            dispatch.greedy &= row.greedy;
        }
        Ok(dispatch)
    }

    fn sample_view_with_dispatch(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: crate::executor::SamplingDispatch,
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        self.sample_with_policy(view, items, Some(dispatch))
    }

    fn issue_view_with_dispatch(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: crate::executor::SamplingDispatch,
    ) -> HostServiceResultV1<bool> {
        if items
            .iter()
            .any(|item| self.reference.forced(item).is_some())
        {
            return Ok(false);
        }
        self.inner.issue_view_with_dispatch(view, items, dispatch)
    }

    fn score_view_with_dispatch(
        &self,
        view: OwnedLogitsViewV1,
        rows: &[DevicePromptRowV1<'_>],
        dispatch: crate::executor::SamplingDispatch,
    ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
        self.score_with_policy(view, rows, Some(dispatch))
    }

    fn sample_view(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        self.sample_with_policy(view, items, None)
    }

    /// Views with a forced row are scored synchronously by `sample_view`.
    fn issue_view(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<bool> {
        if items
            .iter()
            .any(|item| self.reference.forced(item).is_some())
        {
            return Ok(false);
        }
        self.inner.issue_view(view, items)
    }

    fn collect_view(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        self.inner.collect_view(view, items)
    }

    fn score_view(
        &self,
        view: OwnedLogitsViewV1,
        rows: &[DevicePromptRowV1<'_>],
    ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
        self.score_with_policy(view, rows, None)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use runtime_bundle_abi::{BundleLogitsViewV1, BundleStatusV1};
    use runtime_engine::{RequestId, SamplingWork};
    use runtime_sampling::{SamplingParameters, SamplingState};

    use super::*;

    #[test]
    fn continuation_follows_agreeing_references_only() {
        let reference = ReferenceTokensV1::new(vec![vec![1, 2, 3, 4], vec![1, 2, 5], vec![7, 8]]);
        assert_eq!(reference.continuation(&[1, 2, 3]), &[4]);
        assert_eq!(reference.continuation(&[7]), &[8]);
        // Two references continue [1, 2] differently, so nothing is forced.
        assert!(reference.continuation(&[1, 2]).is_empty());
        assert_eq!(reference.continuation(&[1]), &[2]);
        assert!(reference.continuation(&[1, 2, 3, 4]).is_empty());
        assert!(reference.continuation(&[9]).is_empty());
    }

    #[test]
    fn the_file_is_validated_and_the_variable_refused() {
        assert_eq!(
            ReferenceTokensV1::from_json(br#"{"sequences": [[1, 2], [3]]}"#).unwrap(),
            ReferenceTokensV1::new(vec![vec![1, 2], vec![3]])
        );
        for bad in [
            &br#"{"sequences": []}"#[..],
            br#"{"sequences": [[]]}"#,
            br#"{"sequences": [[1]], "other": 1}"#,
            br#"[[1]]"#,
        ] {
            assert!(ReferenceTokensV1::from_json(bad).is_err());
        }
        assert!(ReferenceTokensV1::from_file(Path::new("/nonexistent/reference.json")).is_err());
        assert_eq!(refuse_reference_tokens_variable_v1(None), Ok(()));
        assert!(refuse_reference_tokens_variable_v1(Some("".into())).is_err());
    }

    type ScoreCall = Vec<(u64, Option<u32>, bool, bool, usize)>;

    /// Returns each row's target (or 9 when sampled) and records the rows.
    #[derive(Default)]
    struct Recording {
        samples: RefCell<usize>,
        scores: RefCell<Vec<ScoreCall>>,
    }

    impl DeviceLogitsSamplerV1 for Recording {
        fn sample_view(
            &self,
            _view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            *self.samples.borrow_mut() += 1;
            Ok(items
                .iter()
                .map(|item| SampledRow::new(item.request_id(), 9, -0.5, Vec::new()))
                .collect())
        }

        fn score_view(
            &self,
            _view: OwnedLogitsViewV1,
            rows: &[DevicePromptRowV1<'_>],
        ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
            self.scores.borrow_mut().push(
                rows.iter()
                    .map(|row| {
                        (
                            row.item.request_id().get(),
                            row.target,
                            row.prompt,
                            row.output,
                            row.item.prompt_top_logprobs(),
                        )
                    })
                    .collect(),
            );
            let scores = rows
                .iter()
                .map(|row| {
                    SampledRow::new(
                        row.item.request_id(),
                        row.target.unwrap_or(9),
                        -1.0,
                        Vec::new(),
                    )
                })
                .collect::<Vec<_>>();
            let generated = rows
                .iter()
                .filter(|row| row.prompt && row.output && row.target.is_none())
                .map(|row| SampledRow::new(row.item.request_id(), 9, -0.5, Vec::new()))
                .collect();
            Ok((scores, generated))
        }

        fn issue_view(
            &self,
            _view: OwnedLogitsViewV1,
            _items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<bool> {
            *self.samples.borrow_mut() += 1;
            Ok(true)
        }
    }

    fn item(id: u64, history: Vec<u32>, top: usize) -> ModelBatchItem {
        ModelBatchItem::new(
            RequestId::new(id),
            runtime_scheduler::BatchPhase::Decode,
            vec![*history.last().unwrap()],
            true,
        )
        .with_sampling(SamplingWork {
            parameters: SamplingParameters::new(0.0)
                .unwrap()
                .with_top_logprobs(top)
                .unwrap(),
            state: SamplingState::new(0),
            history: history.into(),
            emitted_tokens: 0,
            stop_ids: Vec::new(),
            constraint: None,
        })
    }

    fn view() -> OwnedLogitsViewV1 {
        OwnedLogitsViewV1::new(BundleLogitsViewV1::empty())
    }

    fn sampler(inner: &Rc<Recording>) -> ReferenceForcingSamplerV1 {
        ReferenceForcingSamplerV1::new(
            inner.clone(),
            Rc::new(ReferenceTokensV1::new(vec![vec![1, 2, 3, 4]])),
        )
    }

    #[test]
    fn rows_off_the_reference_are_sampled_unchanged() {
        let inner = Rc::new(Recording::default());
        let other = item(1, vec![5, 6], 0);
        let sampled = sampler(&inner).sample_view(view(), &[&other]).unwrap();
        assert_eq!(sampled[0].token_id(), 9);
        assert_eq!(*inner.samples.borrow(), 1);
        assert!(inner.scores.borrow().is_empty());
    }

    #[test]
    fn only_views_without_forced_rows_are_deferred() {
        let inner = Rc::new(Recording::default());
        let forcing = sampler(&inner);
        let forced = item(1, vec![1, 2], 0);
        let other = item(2, vec![5], 0);
        assert!(!forcing.issue_view(view(), &[&forced, &other]).unwrap());
        assert_eq!(*inner.samples.borrow(), 0);
        assert!(forcing.issue_view(view(), &[&other]).unwrap());
        assert_eq!(*inner.samples.borrow(), 1);
    }

    #[test]
    fn rows_on_the_reference_report_the_reference_token() {
        let inner = Rc::new(Recording::default());
        let forced = item(1, vec![1, 2], 3);
        let other = item(2, vec![5], 0);
        let sampled = sampler(&inner)
            .sample_view(view(), &[&forced, &other])
            .unwrap();
        assert_eq!(
            sampled.iter().map(SampledRow::token_id).collect::<Vec<_>>(),
            vec![3, 9]
        );
        assert_eq!(*inner.samples.borrow(), 0);
        assert_eq!(
            inner.scores.borrow()[0],
            vec![(1, Some(3), false, false, 3), (2, None, false, true, 0)]
        );
    }

    #[test]
    fn prompt_scored_steps_force_output_rows_and_the_last_prompt_row() {
        let inner = Rc::new(Recording::default());
        let prompt = item(1, vec![1], 2);
        let decode = item(2, vec![1, 2, 3], 1);
        let rows = [
            DevicePromptRowV1 {
                item: &prompt,
                target: Some(2),
                prompt: true,
                output: false,
            },
            DevicePromptRowV1 {
                item: &prompt,
                target: None,
                prompt: true,
                output: true,
            },
            DevicePromptRowV1 {
                item: &decode,
                target: None,
                prompt: false,
                output: true,
            },
        ];
        let (scores, generated) = sampler(&inner).score_view(view(), &rows).unwrap();
        assert_eq!(
            inner.scores.borrow()[0],
            vec![
                (1, Some(2), true, false, 0),
                (1, Some(2), false, false, 2),
                (2, Some(4), false, false, 1),
            ]
        );
        assert_eq!(
            scores.iter().map(SampledRow::token_id).collect::<Vec<_>>(),
            vec![2, 2, 4]
        );
        assert_eq!(
            generated
                .iter()
                .map(|row| (row.request_id().get(), row.token_id()))
                .collect::<Vec<_>>(),
            vec![(1, 2)]
        );
    }

    #[test]
    fn inner_refusals_propagate() {
        struct Refusing;
        impl DeviceLogitsSamplerV1 for Refusing {
            fn sample_view(
                &self,
                _view: OwnedLogitsViewV1,
                _items: &[&ModelBatchItem],
            ) -> HostServiceResultV1<Vec<SampledRow>> {
                Err(BundleStatusV1::HostError)
            }
        }
        let forcing = ReferenceForcingSamplerV1::new(
            Rc::new(Refusing),
            Rc::new(ReferenceTokensV1::new(vec![vec![1, 2]])),
        );
        let forced = item(1, vec![1], 0);
        assert_eq!(
            forcing.sample_view(view(), &[&forced]).unwrap_err(),
            BundleStatusV1::Unsupported
        );
    }
}
