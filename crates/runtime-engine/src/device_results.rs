//! Checked device acceptance without reconstructing host draft IDs.
use super::*;

/// Whether a successor can use device feedback or needs collected host values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AsyncInputDependency {
    #[default]
    HostValues,
    DeviceFeedback,
}

/// Engine-owned identity and stop policy for one item's output cycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputCycle {
    pub epoch: u64,
    pub remaining_tokens: usize,
    pub termination_stop_ids: Vec<u32>,
}

/// A device reducer's decision after every target sampler row was validated.
/// Membership and counts are checked again against the scheduled item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptedPrefix {
    pub request_id: RequestId,
    pub cycle_epoch: u64,
    pub drafted: usize,
    pub accepted: usize,
    pub emitted: usize,
    pub terminal: bool,
    pub rejected: bool,
    /// A consumed EMPTY row fails the request and authorizes no publication.
    pub empty_row: Option<usize>,
}

impl SamplingWork {
    /// Verify rows can omit draft IDs only when their sampling mathematics
    /// does not depend on history or an evolving grammar.
    #[must_use]
    pub fn history_independent(&self) -> bool {
        self.constraint.is_none()
            && self.parameters.repetition_penalty() == 1.0
            && self.parameters.frequency_penalty() == 0.0
            && self.parameters.presence_penalty() == 0.0
    }

    pub fn for_device_verify_row(&self, row: usize) -> Result<Self, ExecutorError> {
        if !self.history_independent() {
            return Err(executor_error_fixed(
                "device verification needs history-independent sampling",
            ));
        }
        let emitted_tokens = self
            .emitted_tokens
            .checked_add(row)
            .ok_or_else(|| executor_error_fixed("device verify row count overflows"))?;
        Ok(Self {
            state: self.state.after(row as u64),
            emitted_tokens,
            ..self.clone()
        })
    }
}

impl AcceptedPrefix {
    pub(crate) fn matches(&self, item: &ModelBatchItem, samples: &[&SampledRow]) -> bool {
        let Some(cycle) = &item.output_cycle else {
            return false;
        };
        let drafted = item.draft.map_or(0, |draft| draft.tokens);
        if self.request_id != item.request_id
            || self.cycle_epoch == 0
            || self.cycle_epoch != cycle.epoch
            || self.drafted != drafted
            || item
                .draft
                .is_some_and(|draft| draft.source != DraftSource::Bundle)
            || item
                .sampling
                .as_ref()
                .is_none_or(|work| !work.history_independent())
            || drafted.checked_add(1) != Some(samples.len())
        {
            return false;
        }
        if let Some(row) = self.empty_row {
            return row <= drafted
                && self.emitted == 0
                && samples[row].failure() == Some(SamplingError::AllTokensMasked);
        }
        if self.emitted == 0
            || self.emitted > cycle.remaining_tokens
            || self.accepted > drafted
            || self.accepted > self.emitted
            || self.emitted > samples.len()
            || samples[..self.emitted]
                .iter()
                .any(|row| row.failure().is_some())
        {
            return false;
        }
        let last_matched = self.accepted == self.emitted;
        let last_unmatched = self.accepted.checked_add(1) == Some(self.emitted);
        if (!last_matched && !last_unmatched)
            || (last_matched
                && (self.rejected || (!self.terminal && self.emitted != cycle.remaining_tokens)))
            || (last_unmatched && self.rejected != (self.emitted <= drafted))
        {
            return false;
        }
        let stop = |sample: &&SampledRow| {
            cycle
                .termination_stop_ids
                .binary_search(&sample.token_id())
                .is_ok()
        };
        self.terminal == stop(&samples[self.emitted - 1])
            && !samples[..self.emitted - 1].iter().any(stop)
    }

    pub(crate) fn stage(
        &self,
        record: &RequestRecord,
        samples: &[&SampledRow],
    ) -> Result<StagedRun, SampleStageError> {
        if let Some(row) = self.empty_row {
            let error = samples[row]
                .failure()
                .expect("validated consumed empty row");
            return Err(SampleStageError::Request(self.request_id, error.into()));
        }
        // The host validates complete membership and every row before this
        // point. Only this prefix advances durable history and the RNG.
        let tokens = samples[..self.emitted]
            .iter()
            .map(|sample| StagedToken {
                token_id: sample.token_id(),
                logprob: sample.logprob(),
                top_logprobs: sample.top_logprobs().to_vec(),
            })
            .collect();
        Ok(StagedRun {
            tokens,
            state: record.sampling_state.after(self.emitted as u64),
            terminal: self.terminal,
            drafted: self.drafted,
            accepted: self.accepted,
            rejected: self.rejected,
            constraint: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NeverExecute;
    impl ModelExecutor for NeverExecute {
        type TransactionState<'a> = NoopModelTransactionState;
        fn execute<'a>(
            &'a mut self,
            _: &[ModelBatchItem],
            _: &PreparedAppendBatch,
        ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
            unreachable!("acceptance fixtures do not submit device work")
        }
    }

    fn engine(budget: usize) -> Engine<NeverExecute> {
        let vocabulary = ModelVocabulary::new(8).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(3, 4, 16, 16, 1).unwrap(),
            KvGeometry::new(4, 32).unwrap(),
            vocabulary,
            32,
            4,
        )
        .unwrap();
        let mut engine = Engine::new(config, NeverExecute).unwrap();
        for id in 1..=3 {
            engine
                .submit(
                    EngineRequest::new(
                        RequestId::new(id),
                        vec![0],
                        budget,
                        SamplingParameters::new(1.0)
                            .unwrap()
                            .with_top_logprobs(1)
                            .unwrap(),
                        17,
                        vec![3],
                        vocabulary,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        engine
    }

    fn item(engine: &Engine<NeverExecute>, id: u64, k: usize, device: bool) -> ModelBatchItem {
        let id = RequestId::new(id);
        let record = &engine.requests[&id];
        let mut item = ModelBatchItem::new(
            id,
            if k == 0 {
                BatchPhase::Decode
            } else {
                BatchPhase::Verify { draft_tokens: k }
            },
            vec![0],
            true,
        )
        .with_sampling(SamplingWork {
            parameters: record.sampling.clone(),
            state: record.sampling_state,
            history: SamplingHistory::new(Arc::clone(&record.history)),
            emitted_tokens: 0,
            stop_ids: vec![3],
            constraint: None,
        });
        if k != 0 {
            item.draft = Some(DraftWork::new(
                k,
                if device {
                    DraftSource::Bundle
                } else {
                    DraftSource::Host
                },
            ));
        }
        if device {
            item.output_cycle = Some(OutputCycle {
                epoch: 7,
                remaining_tokens: record.maximum_new_tokens,
                termination_stop_ids: record.termination_stop_ids.clone(),
            });
        }
        item
    }

    fn sample(id: u64, token: u32, row: usize) -> SampledRow {
        let logprob = -0.1234567891234567 - row as f64;
        SampledRow::new(
            RequestId::new(id),
            token,
            logprob,
            vec![TokenLogprob::new(token, logprob)],
        )
    }

    #[test]
    fn accepted_prefix_matches_host_walk_for_stops_limits_and_rejections() {
        let vocabulary = ModelVocabulary::new(8).unwrap();
        let mut cases = 0;
        for k in 0..=5 {
            for budget in 1..=6 {
                let engine = engine(budget);
                for reject_at in 0..=k {
                    for stop_at in 0..=k + 1 {
                        for matched_stop in [false, true] {
                            let mut proposals = vec![1; k];
                            let mut tokens = vec![1; k + 1];
                            tokens[reject_at] = 2;
                            if stop_at <= k {
                                tokens[stop_at] = 3;
                                if matched_stop && stop_at < k {
                                    proposals[stop_at] = 3;
                                }
                            }
                            let samples = tokens
                                .into_iter()
                                .enumerate()
                                .map(|(row, token)| sample(1, token, row))
                                .collect::<Vec<_>>();
                            let mut host_item = item(&engine, 1, k, false);
                            host_item.input_tokens.extend_from_slice(&proposals);
                            let host = stage_samples(
                                &engine.requests,
                                &[host_item],
                                &[],
                                &samples,
                                &[],
                                &[],
                                vocabulary,
                                &[],
                            )
                            .unwrap();
                            let expected = host[0].as_ref().unwrap();
                            let prefix = AcceptedPrefix {
                                request_id: RequestId::new(1),
                                cycle_epoch: 7,
                                drafted: k,
                                accepted: expected.accepted,
                                emitted: expected.tokens.len(),
                                terminal: expected.terminal,
                                rejected: expected.rejected,
                                empty_row: None,
                            };
                            let items = [item(&engine, 1, k, true)];
                            assert!(
                                outputs_match(
                                    &items,
                                    &[],
                                    &[],
                                    &samples,
                                    &[],
                                    &[prefix],
                                    vocabulary
                                ),
                                "{prefix:?}"
                            );
                            let actual = stage_samples(
                                &engine.requests,
                                &items,
                                &[],
                                &samples,
                                &[],
                                &[prefix],
                                vocabulary,
                                &[],
                            )
                            .unwrap();
                            let actual = actual[0].as_ref().unwrap();
                            assert_eq!(actual.state, expected.state);
                            assert_eq!(
                                (
                                    actual.drafted,
                                    actual.accepted,
                                    actual.terminal,
                                    actual.rejected
                                ),
                                (
                                    expected.drafted,
                                    expected.accepted,
                                    expected.terminal,
                                    expected.rejected
                                )
                            );
                            assert_eq!(actual.tokens.len(), expected.tokens.len());
                            for (a, b) in actual.tokens.iter().zip(&expected.tokens) {
                                assert_eq!(a.token_id, b.token_id);
                                assert_eq!(a.logprob.to_bits(), b.logprob.to_bits());
                                assert_eq!(a.top_logprobs, b.top_logprobs);
                            }
                            assert_eq!(actual.state.generated(), actual.tokens.len() as u64);
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 1344);
    }

    #[test]
    fn stale_or_ambiguous_results_and_invalid_suffixes_are_refused() {
        let engine = engine(6);
        let items = [item(&engine, 1, 5, true)];
        let mut samples = (0..6).map(|row| sample(1, 2, row)).collect::<Vec<_>>();
        let prefix = AcceptedPrefix {
            request_id: RequestId::new(1),
            cycle_epoch: 7,
            drafted: 5,
            accepted: 0,
            emitted: 1,
            terminal: false,
            rejected: true,
            empty_row: None,
        };
        let matches = |samples: &[SampledRow], prefixes: &[AcceptedPrefix], drafts: &[DraftRow]| {
            outputs_match(
                &items,
                &[],
                &[],
                samples,
                drafts,
                prefixes,
                engine.model_vocabulary,
            )
        };
        assert!(matches(&samples, &[prefix], &[]));
        assert!(!matches(
            &samples,
            &[AcceptedPrefix {
                cycle_epoch: 6,
                ..prefix
            }],
            &[]
        ));
        assert!(!matches(&samples, &[prefix, prefix], &[]));
        assert!(!matches(
            &samples,
            &[AcceptedPrefix {
                request_id: RequestId::new(2),
                ..prefix
            }],
            &[]
        ));
        samples[5].logprob = f64::NAN;
        assert!(
            !matches(&samples, &[prefix], &[]),
            "an unconsumed malformed suffix must still fail"
        );
        samples[5] = SampledRow::failed(RequestId::new(1), SamplingError::AllTokensMasked);
        assert!(
            matches(&samples, &[prefix], &[]),
            "an unconsumed EMPTY suffix is not read"
        );
        samples[0] = SampledRow::failed(RequestId::new(1), SamplingError::AllTokensMasked);
        assert!(!matches(&samples, &[prefix], &[]));
        let empty = AcceptedPrefix {
            emitted: 0,
            empty_row: Some(0),
            ..prefix
        };
        assert!(matches(&samples, &[empty], &[]));
        assert!(matches!(
            stage_samples(
                &engine.requests,
                &items,
                &[],
                &samples,
                &[],
                &[empty],
                engine.model_vocabulary,
                &[]
            ),
            Err(SampleStageError::Request(..))
        ));
    }

    #[test]
    fn mixed_prefill_decode_and_device_verify_keep_membership_and_bits() {
        let engine = engine(6);
        let mut prefill = item(&engine, 1, 0, false);
        prefill.phase = BatchPhase::Prefill { tokens: 1 };
        let items = [
            prefill,
            item(&engine, 2, 0, true),
            item(&engine, 3, 2, true),
        ];
        let samples = [
            sample(1, 1, 0),
            sample(2, 2, 0),
            sample(3, 1, 0),
            sample(3, 2, 1),
            sample(3, 2, 2),
        ];
        let prefixes = [
            AcceptedPrefix {
                request_id: RequestId::new(2),
                cycle_epoch: 7,
                drafted: 0,
                accepted: 0,
                emitted: 1,
                terminal: false,
                rejected: false,
                empty_row: None,
            },
            AcceptedPrefix {
                request_id: RequestId::new(3),
                cycle_epoch: 7,
                drafted: 2,
                accepted: 1,
                emitted: 2,
                terminal: false,
                rejected: true,
                empty_row: None,
            },
        ];
        assert!(outputs_match(
            &items,
            &[],
            &[],
            &samples,
            &[],
            &prefixes,
            engine.model_vocabulary
        ));
        let staged = stage_samples(
            &engine.requests,
            &items,
            &[],
            &samples,
            &[],
            &prefixes,
            engine.model_vocabulary,
            &[],
        )
        .unwrap();
        assert_eq!(
            staged
                .iter()
                .map(|run| run.as_ref().unwrap().tokens.len())
                .collect::<Vec<_>>(),
            [1, 1, 2]
        );
        assert_eq!(
            staged[2].as_ref().unwrap().tokens[1].logprob.to_bits(),
            samples[3].logprob.to_bits()
        );
    }

    #[test]
    fn history_independent_verify_keeps_offsets_and_refuses_penalties() {
        let engine = engine(6);
        let item = item(&engine, 1, 5, true);
        let work = item.sampling.as_ref().unwrap();
        for row in 0..=5 {
            let actual = work.for_device_verify_row(row).unwrap();
            let expected = work.for_verify_row(&[1; 5], row);
            assert_eq!(actual.state, expected.state);
            assert_eq!(actual.emitted_tokens, expected.emitted_tokens);
            assert_eq!(actual.stop_ids, expected.stop_ids);
        }
        let mut dependent = work.clone();
        dependent.parameters = dependent.parameters.with_penalties(0.0, 0.0, 1.1).unwrap();
        assert!(!dependent.history_independent());
        assert!(dependent.for_device_verify_row(1).is_err());
    }
}
