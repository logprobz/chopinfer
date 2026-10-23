use super::super::state::{Config, MAX_VERIFY_ROWS};
use super::*;
use std::collections::BTreeMap;

const VOCAB: usize = 16;

#[derive(Default)]
struct FakeHost {
    journal: Vec<String>,
    ids: Vec<[u32; WIDTH]>,
    scores: Vec<[f32; WIDTH]>,
    logits: Vec<[[f64; VOCAB]; WIDTH]>,
    target_rows: Vec<TargetRows>,
    target_prefixes: Vec<Vec<u64>>,
    committed: BTreeMap<u64, Vec<u32>>,
    records: Vec<Diagnostic>,
    fail: Option<&'static str>,
    bad_tail_id: bool,
    device_target: bool,
    confidence_override: Option<[f32; WIDTH]>,
    /// Journal `prefetch_engram` calls.
    prefetch: bool,
}

impl FakeHost {
    fn event(&mut self, kind: &'static str, detail: String) -> Result<(), &'static str> {
        self.journal.push(format!("{kind} {detail}"));
        if self.fail == Some(kind) {
            Err("injected host failure")
        } else {
            Ok(())
        }
    }
}

impl Host for FakeHost {
    type Error = &'static str;
    type Gathered = Vec<TargetRows>;

    fn project_committed(&mut self, pass: Projection) -> Result<(), Self::Error> {
        self.event(
            "project",
            format!(
                "{} {} {} {}",
                pass.request, pass.stage, pass.start, pass.count
            ),
        )
    }

    fn project_rows(&mut self, passes: &[RowProjection]) -> Result<(), Self::Error> {
        let detail = passes
            .iter()
            .map(|pass| format!("{} {} {}", pass.request, pass.start, pass.count))
            .collect::<Vec<_>>()
            .join(", ");
        self.event("project_rows", detail)
    }

    fn prefetch_engram(&mut self, inputs: &[Input]) -> Result<(), Self::Error> {
        if !self.prefetch {
            return Ok(());
        }
        let tokens = inputs
            .iter()
            .map(|input| format!("{}:{:?}", input.rows.request, input.tokens))
            .collect::<Vec<_>>()
            .join(" ");
        self.event("prefetch", tokens)
    }

    fn draft(&mut self, inputs: &[DraftInput]) -> Result<(), Self::Error> {
        self.event("draft", inputs.len().to_string())?;
        self.ids.clear();
        self.scores.clear();
        self.logits.clear();
        for input in inputs {
            assert_eq!(input.block.compute_rows, BLOCK_ROWS);
            // A small deterministic block with later noise inputs affecting
            // every query, followed by a sequential previous-token bias.
            // These synthetic logits test scheduling, not model numerics.
            let noise = [input.bonus, 7, 7, 7, 7];
            let block_sum = noise.iter().sum::<u32>();
            let mut previous = input.bonus;
            let mut ids = [0; WIDTH];
            let mut scores = [0.0; WIDTH];
            let mut logits = [[0.0; VOCAB]; WIDTH];
            for query in 0..WIDTH {
                for token in 0..VOCAB {
                    logits[query][token] = f64::from(
                        (block_sum
                            + previous * (token as u32 + 1)
                            + query as u32 * 7
                            + token as u32 * token as u32)
                            % 31,
                    );
                }
                let chosen = (0..VOCAB)
                    .max_by(|&left, &right| {
                        logits[query][left]
                            .total_cmp(&logits[query][right])
                            .then(right.cmp(&left))
                    })
                    .unwrap();
                ids[query] = chosen as u32;
                scores[query] = logits[query][chosen] as f32 / 7.0 - 2.0;
                previous = ids[query];
            }
            self.ids.push(ids);
            self.scores.push(scores);
            self.logits.push(logits);
        }
        Ok(())
    }

    fn device_target(
        &mut self,
        inputs: &[Input],
        drafts: &[DraftInput],
    ) -> Result<Option<Vec<[u32; WIDTH]>>, Self::Error> {
        if !self.device_target {
            return Ok(None);
        }
        self.event("device_hash_gather", drafts.len().to_string())?;
        let mut position = 0;
        let target = inputs
            .iter()
            .map(|input| {
                let mut tokens = input.tokens.clone();
                if input.rows.kind.drafts() > 0 {
                    tokens.extend_from_slice(
                        &self.ids[position][..input.rows.kind.drafts() as usize],
                    );
                    position += 1;
                }
                TargetRows {
                    rows: input.rows,
                    tokens,
                }
            })
            .collect::<Vec<_>>();
        self.target_rows = target.clone();
        self.target(&target)?;
        self.event("device_results", drafts.len().to_string())?;
        let mut ids = self.ids.clone();
        if self.bad_tail_id {
            ids[0][WIDTH - 1] = u32::MAX;
        }
        Ok(Some(ids))
    }

    fn read_draft_ids(&mut self, ids: &mut [[u32; WIDTH]]) -> Result<(), Self::Error> {
        self.event("read_ids", (ids.len() * WIDTH * 4).to_string())?;
        assert_eq!(ids.len(), self.ids.len());
        ids.copy_from_slice(&self.ids);
        if self.bad_tail_id {
            ids[0][WIDTH - 1] = u32::MAX;
        }
        Ok(())
    }

    fn read_confidence(&mut self, scores: &mut [[f32; WIDTH]]) -> Result<(), Self::Error> {
        self.event("read_confidence", (scores.len() * WIDTH * 4).to_string())?;
        scores.copy_from_slice(&self.scores);
        if let Some(replacement) = self.confidence_override {
            scores.fill(replacement);
        }
        Ok(())
    }

    fn hash_and_gather(&mut self, rows: &[TargetRows]) -> Result<Self::Gathered, Self::Error> {
        self.event(
            "hash_gather",
            rows.iter()
                .map(|row| row.tokens.len())
                .sum::<usize>()
                .to_string(),
        )?;
        for row in rows {
            assert_eq!(
                row.tokens.len(),
                row.rows.kind.target_rows().unwrap() as usize
            );
        }
        Ok(rows.to_vec())
    }

    fn upload_engram(&mut self, gathered: Self::Gathered) -> Result<(), Self::Error> {
        self.event("upload", gathered.len().to_string())?;
        self.target_rows = gathered;
        Ok(())
    }

    fn target(&mut self, rows: &[TargetRows]) -> Result<(), Self::Error> {
        self.event("target", rows.len().to_string())?;
        assert_eq!(rows, self.target_rows);
        // Row-local prefix fingerprints catch an omitted/reordered bonus or
        // draft. Gathering is provisional and never changes committed tokens.
        self.target_prefixes = rows
            .iter()
            .map(|row| {
                let mut fingerprint = self
                    .committed
                    .get(&row.rows.request)
                    .into_iter()
                    .flatten()
                    .fold(0_u64, |hash, &token| {
                        hash.wrapping_mul(31).wrapping_add(u64::from(token))
                    });
                row.tokens
                    .iter()
                    .map(|&token| {
                        fingerprint = fingerprint.wrapping_mul(31).wrapping_add(u64::from(token));
                        fingerprint
                    })
                    .collect()
            })
            .collect();
        Ok(())
    }

    fn commit_features(&mut self, plan: &CommitPlan) -> Result<(), Self::Error> {
        self.event("commit", plan.items().len().to_string())?;
        for item in plan.items() {
            let row = self
                .target_rows
                .iter()
                .find(|row| row.rows.request == item.rows.request)
                .unwrap();
            self.committed
                .entry(item.rows.request)
                .or_default()
                .extend_from_slice(&row.tokens[..item.kept as usize]);
        }
        Ok(())
    }

    fn record_diagnostics(&mut self, diagnostics: &[Diagnostic]) {
        self.journal
            .push(format!("diagnostics {}", diagnostics.len()));
        self.records.extend_from_slice(diagnostics);
    }
}

fn setup() -> (Runtime, FakeHost) {
    let mut state = State::new(Config {
        slots: 3,
        stages: 3,
        window: 4,
        max_prefill_rows: 64,
        context_tokens: 256,
        position_capacity: 256,
    })
    .unwrap()
    // The three restored rows are a bulk commit: they catch up in the
    // per-stage passes of an eager catch-up.
    .with_batch_rows(2)
    .unwrap();
    state
        .restore(
            41,
            0,
            Snapshot {
                start: 0,
                end: 3,
                window: 4,
                stages: 3,
            },
        )
        .unwrap();
    let mut host = FakeHost::default();
    host.committed.insert(41, vec![1, 2, 3]);
    (Runtime::new(state, VOCAB as u32, 2).unwrap(), host)
}

fn input(budget: u32) -> Input {
    Input {
        rows: Rows {
            request: 41,
            slot: 0,
            start: 3,
            kind: Kind::Decode { drafts: budget },
        },
        tokens: vec![4],
        noise: crate::dspark::execution::Noise::GREEDY,
    }
}

#[test]
fn host_verification_retains_each_accepted_prefix_without_redrafting() {
    for rows in 2..=MAX_VERIFY_ROWS {
        for kept in 1..=rows {
            let (mut runtime, mut host) = setup();
            let mut supplied = input(0);
            supplied.rows.kind = Kind::Verify { rows };
            supplied.tokens = (4..4 + rows).collect();
            assert!(
                runtime
                    .execute(&mut host, &[supplied.clone()])
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(host.target_rows[0].tokens, supplied.tokens);
            assert!(
                !host
                    .journal
                    .iter()
                    .any(|line| line.starts_with("draft ") || line.starts_with("read_"))
            );
            assert!(runtime.commit(&mut host, &[(41, kept)]).unwrap().is_empty());
            assert_eq!(
                host.committed[&41],
                [vec![1, 2, 3], supplied.tokens[..kept as usize].to_vec()].concat()
            );
            assert_eq!(
                runtime.state().context(41).unwrap().end,
                3 + u64::from(kept)
            );
            runtime.catch_up(&mut host, 41).unwrap();
            assert!(
                runtime
                    .state()
                    .context(41)
                    .unwrap()
                    .projected()
                    .iter()
                    .all(|&end| end == 3 + u64::from(kept))
            );
        }
    }
}

#[test]
fn mixed_host_and_bundle_verification_bind_diagnostics_by_request() {
    let (mut runtime, mut host) = setup();
    runtime
        .restore_after_copy(
            42,
            1,
            Snapshot {
                start: 0,
                end: 3,
                window: 4,
                stages: 3,
            },
        )
        .unwrap();
    host.committed.insert(42, vec![1, 2, 3]);
    let mut supplied = input(0);
    supplied.rows.kind = Kind::Verify { rows: 6 };
    supplied.tokens = vec![4, 5, 6, 7, 8, 9];
    let mut drafted = input(5);
    drafted.rows.request = 42;
    drafted.rows.slot = 1;
    let proposals = runtime.execute(&mut host, &[supplied, drafted]).unwrap();
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].request, 42);
    let diagnostics = runtime.commit(&mut host, &[(41, 1), (42, 6)]).unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].request, 42);
    assert_eq!(diagnostics[0].outcomes, [Outcome::Accepted; WIDTH]);
    assert_eq!(host.committed[&41], [1, 2, 3, 4]);
}

#[test]
fn journal_orders_catch_up_full_draft_readback_gather_upload_and_target() {
    let (mut runtime, mut host) = setup();
    let drafts = runtime.execute(&mut host, &[input(5)]).unwrap();
    assert_eq!(drafts[0].tokens, host.ids[0]);
    assert_eq!(host.committed[&41], vec![1, 2, 3]);
    assert_eq!(
        host.target_rows[0].tokens,
        [vec![4], host.ids[0].to_vec()].concat()
    );
    assert_eq!(
        host.journal,
        [
            "project 41 0 0 2",
            "project 41 0 2 1",
            "project 41 1 0 2",
            "project 41 1 2 1",
            "project 41 2 0 2",
            "project 41 2 2 1",
            "draft 1",
            "read_ids 20",
            "hash_gather 6",
            "upload 1",
            "target 1",
            "read_confidence 20",
        ]
    );
    runtime.commit(&mut host, &[(41, 6)]).unwrap();
    assert_eq!(host.committed[&41].len(), 9);
    assert_eq!(host.records[0].outcomes, [Outcome::Accepted; WIDTH]);
    assert_eq!(&host.journal[12..], ["commit 1", "diagnostics 1"]);
}

#[test]
fn budgets_one_through_four_match_full_block_ids_logits_and_committed_prefix() {
    let (mut full, mut full_host) = setup();
    let full_ids = full.execute(&mut full_host, &[input(5)]).unwrap();
    full.commit(&mut full_host, &[(41, 6)]).unwrap();
    for budget in 1..5 {
        let (mut runtime, mut host) = setup();
        let ids = runtime.execute(&mut host, &[input(budget)]).unwrap();
        assert_eq!(ids[0].tokens, full_ids[0].tokens[..budget as usize]);
        assert_eq!(host.logits, full_host.logits);
        assert_eq!(
            host.target_prefixes[0],
            full_host.target_prefixes[0][..budget as usize + 1]
        );
        let records = runtime.commit(&mut host, &[(41, budget + 1)]).unwrap();
        assert_eq!(
            host.committed[&41],
            full_host.committed[&41][..budget as usize + 4]
        );
        assert_eq!(records[0].proposed, full_host.records[0].proposed);
        assert_eq!(
            records[0].confidence_bits,
            full_host.records[0].confidence_bits
        );
        assert_eq!(
            &records[0].outcomes[..budget as usize],
            vec![Outcome::Accepted; budget as usize]
        );
        assert_eq!(
            &records[0].outcomes[budget as usize..],
            vec![Outcome::NotEmitted; WIDTH - budget as usize]
        );
        assert!(host.journal.contains(&"read_ids 20".to_owned()));
    }
}

#[test]
fn confidence_is_recorded_for_every_position_and_never_changes_serving() {
    let (mut baseline, mut baseline_host) = setup();
    let expected = baseline.execute(&mut baseline_host, &[input(3)]).unwrap();
    baseline.commit(&mut baseline_host, &[(41, 2)]).unwrap();
    let overrides = [
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::from_bits(0x7fc00123),
        -1e20,
        1e20,
    ];
    let (mut runtime, mut host) = setup();
    host.confidence_override = Some(overrides);
    assert_eq!(runtime.execute(&mut host, &[input(3)]).unwrap(), expected);
    let records = runtime.commit(&mut host, &[(41, 2)]).unwrap();
    assert_eq!(records[0].confidence_bits, overrides.map(f32::to_bits));
    assert_eq!(host.target_rows, baseline_host.target_rows);
    assert_eq!(host.committed, baseline_host.committed);
    assert_eq!(
        records[0].outcomes,
        [
            Outcome::Accepted,
            Outcome::Unknown,
            Outcome::Unknown,
            Outcome::NotEmitted,
            Outcome::NotEmitted
        ]
    );
}

#[test]
fn retained_row_counts_record_only_the_proven_acceptance_prefix() {
    for proven in 0..=5 {
        let (mut runtime, mut host) = setup();
        runtime.execute(&mut host, &[input(5)]).unwrap();
        let records = runtime.commit(&mut host, &[(41, proven + 1)]).unwrap();
        assert_eq!(
            runtime.state().context(41).unwrap().end,
            4 + u64::from(proven)
        );
        assert_eq!(host.committed[&41].len(), 4 + proven as usize);
        assert_eq!(
            records[0]
                .outcomes
                .iter()
                .filter(|&&outcome| outcome == Outcome::Accepted)
                .count(),
            proven as usize
        );
        assert_eq!(
            records[0]
                .outcomes
                .iter()
                .filter(|&&outcome| outcome == Outcome::Unknown)
                .count(),
            5 - proven as usize
        );
        assert_eq!(host.records, records);
    }
}

#[test]
fn matched_eos_is_not_recorded_as_a_rejected_proposal() {
    let (mut runtime, mut host) = setup();
    let drafts = runtime.execute(&mut host, &[input(5)]).unwrap();
    let eos = drafts[0].tokens[0];
    // The target matches the first draft and emits EOS. The engine counts
    // one accepted draft, then stops. Its commit callback carries the
    // emitted-token count (one), not accepted + one (two).
    let target_emitted = [eos];
    let accepted = target_emitted
        .iter()
        .zip(&drafts[0].tokens)
        .take_while(|(target, draft)| target == draft)
        .count();
    let kept_rows = target_emitted.len() as u32;
    assert_eq!((accepted, kept_rows), (1, 1));
    let records = runtime.commit(&mut host, &[(41, kept_rows)]).unwrap();
    assert_eq!(records[0].outcomes, [Outcome::Unknown; WIDTH]);
    assert_eq!(host.records, records);
    let encoded = super::super::calibration::encode(
        super::super::calibration::Context {
            instance_id: 1,
            step_id: 1,
        },
        &records[0],
    )
    .unwrap();
    let logged: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(logged["outcomes"], serde_json::json!(vec!["unknown"; 5]));
    // Only the target input/bonus row has features to retain. The accepted
    // EOS is an output token and has no target feature row in this commit.
    assert_eq!(runtime.state().context(41).unwrap().end, 4);
    assert_eq!(host.committed[&41], vec![1, 2, 3, 4]);
}

#[test]
fn token_limit_does_not_infer_rejection_from_retained_rows() {
    for limit in 1..=5 {
        let (mut runtime, mut host) = setup();
        let drafts = runtime.execute(&mut host, &[input(5)]).unwrap();
        // Every emitted target token matches its draft, but the token
        // budget ends verification before it produces a bonus token.
        let target_emitted = &drafts[0].tokens[..limit];
        let kept_rows = target_emitted.len() as u32;
        let records = runtime.commit(&mut host, &[(41, kept_rows)]).unwrap();
        assert_eq!(
            &records[0].outcomes[..limit - 1],
            vec![Outcome::Accepted; limit - 1]
        );
        assert_eq!(
            &records[0].outcomes[limit - 1..],
            vec![Outcome::Unknown; WIDTH - limit + 1]
        );
        assert!(!records[0].outcomes.contains(&Outcome::Rejected));
        assert_eq!(
            runtime.state().context(41).unwrap().end,
            3 + u64::from(kept_rows)
        );
    }
}

#[test]
fn cancellation_preserves_history_and_reports_only_emitted_positions_as_cancelled() {
    let (mut runtime, mut host) = setup();
    let first = runtime.execute(&mut host, &[input(2)]).unwrap();
    let records = runtime.cancel(&mut host).unwrap();
    assert_eq!(host.committed[&41], vec![1, 2, 3]);
    assert_eq!(runtime.state().context(41).unwrap().end, 3);
    assert_eq!(
        records[0].outcomes,
        [
            Outcome::Cancelled,
            Outcome::Cancelled,
            Outcome::NotEmitted,
            Outcome::NotEmitted,
            Outcome::NotEmitted
        ]
    );
    assert_eq!(runtime.execute(&mut host, &[input(2)]).unwrap(), first);
    assert_eq!(
        host.journal
            .iter()
            .filter(|entry| entry.starts_with("project "))
            .count(),
        6
    );
    runtime.commit(&mut host, &[(41, 3)]).unwrap();
    assert_eq!(host.committed[&41].len(), 6);
}

/// A step whose items all decode and some draft offers each item's first
/// target token before any catch-up or drafting; other steps offer nothing,
/// and the offer changes no result.
#[test]
fn drafted_decode_steps_offer_first_rows_before_catch_up_and_drafting() {
    let (mut runtime, mut host) = setup();
    host.prefetch = true;
    let (mut plain_runtime, mut plain) = setup();
    let drafts = runtime.execute(&mut host, &[input(5)]).unwrap();
    assert_eq!(
        drafts,
        plain_runtime.execute(&mut plain, &[input(5)]).unwrap()
    );
    assert_eq!(host.journal[0], "prefetch 41:[4]");
    assert_eq!(&host.journal[1..], plain.journal);
    runtime.commit(&mut host, &[(41, 6)]).unwrap();
    plain_runtime.commit(&mut plain, &[(41, 6)]).unwrap();
    assert_eq!(host.committed, plain.committed);
    assert_eq!(host.records, plain.records);
    // A decode step without drafts: no offer.
    let mut item = input(0);
    item.rows.start = 9;
    let before = host.journal.len();
    runtime.execute(&mut host, &[item]).unwrap();
    assert!(
        !host.journal[before..]
            .iter()
            .any(|entry| entry.starts_with("prefetch"))
    );
    // A failed offer poisons like any host failure, before drafting.
    let (mut runtime, mut host) = setup();
    host.prefetch = true;
    host.fail = Some("prefetch");
    assert!(runtime.execute(&mut host, &[input(5)]).is_err());
    assert_eq!(host.journal, ["prefetch 41:[4]"]);
    assert!(runtime.state().is_poisoned());
}

#[test]
fn zero_budget_defers_catch_up_without_drafting_or_id_readback() {
    let (mut runtime, mut host) = setup();
    for start in 3..11 {
        let mut item = input(0);
        item.rows.start = start;
        assert!(runtime.execute(&mut host, &[item]).unwrap().is_empty());
        assert!(runtime.commit(&mut host, &[]).unwrap().is_empty());
    }
    assert!(
        !host
            .journal
            .iter()
            .any(|entry| entry.starts_with("draft ") || entry.starts_with("read_"))
    );
    // Only the restored bulk projected, at the first step.
    assert_eq!(
        host.journal
            .iter()
            .filter(|entry| entry.starts_with("project"))
            .count(),
        6
    );
    assert_eq!(runtime.state().context(41).unwrap().projected(), &[3; 3]);
    assert_eq!(host.committed[&41].len(), 11);
    // A snapshot catch-up skips the rows the four-row window overwrites.
    runtime.catch_up(&mut host, 41).unwrap();
    assert_eq!(host.journal.last().unwrap(), "project_rows 41 7 4");
    assert_eq!(runtime.state().context(41).unwrap().projected(), &[11; 3]);
    runtime.state().snapshot(41).unwrap();
}

#[test]
fn deferred_rows_of_drafting_items_share_one_batched_projection() {
    let (mut runtime, mut host) = setup();
    runtime
        .restore_after_copy(
            42,
            1,
            Snapshot {
                start: 0,
                end: 2,
                window: 4,
                stages: 3,
            },
        )
        .unwrap();
    host.committed.insert(42, vec![1, 2]);
    let other = |start, budget| Input {
        rows: Rows {
            request: 42,
            slot: 1,
            start,
            kind: Kind::Decode { drafts: budget },
        },
        tokens: vec![3],
        noise: crate::dspark::execution::Noise::GREEDY,
    };
    // Request 41's restored rows are bulk and project eagerly at its first
    // step, before request 42's deferred rows.
    runtime
        .execute(&mut host, &[input(0), other(2, 0)])
        .unwrap();
    runtime.commit(&mut host, &[]).unwrap();
    assert_eq!(
        host.journal
            .iter()
            .filter(|entry| entry.starts_with("project"))
            .cloned()
            .collect::<Vec<_>>(),
        [
            "project 41 0 0 2",
            "project 41 0 2 1",
            "project 41 1 0 2",
            "project 41 1 2 1",
            "project 41 2 0 2",
            "project 41 2 2 1",
        ]
    );
    host.journal.clear();
    let mut first = input(1);
    first.rows.start = 4;
    runtime.execute(&mut host, &[first, other(3, 2)]).unwrap();
    assert_eq!(
        host.journal[0], "project_rows 41 3 1, 42 0 3",
        "{:?}",
        host.journal
    );
    assert_eq!(host.journal[1], "draft 2");
    runtime.commit(&mut host, &[(41, 2), (42, 2)]).unwrap();
    // Two-row verify commits stay within the batch bound: deferred again.
    host.journal.clear();
    let mut first = input(0);
    first.rows.start = 6;
    runtime.execute(&mut host, &[first, other(5, 0)]).unwrap();
    assert!(
        !host
            .journal
            .iter()
            .any(|entry| entry.starts_with("project"))
    );
    runtime.commit(&mut host, &[]).unwrap();
    assert_eq!(runtime.state().context(41).unwrap().projected(), &[4; 3]);
    assert_eq!(runtime.state().context(42).unwrap().projected(), &[3; 3]);
}

#[test]
fn a_failed_batched_projection_poisons_without_advancing_cursors() {
    let (mut runtime, mut host) = setup();
    runtime.execute(&mut host, &[input(0)]).unwrap();
    runtime.commit(&mut host, &[]).unwrap();
    host.fail = Some("project_rows");
    let mut item = input(5);
    item.rows.start = 4;
    assert_eq!(
        runtime.execute(&mut host, &[item.clone()]),
        Err(Error::Host("injected host failure"))
    );
    assert!(runtime.state().is_poisoned());
    assert!(runtime.execute(&mut host, &[item]).is_err());
    assert_eq!(runtime.state().context(41).unwrap().projected(), &[3; 3]);
    assert_eq!(host.committed[&41], vec![1, 2, 3, 4]);
}

#[test]
fn a_bulk_verify_commit_catches_up_eagerly_at_the_next_step() {
    let (mut runtime, mut host) = setup();
    runtime.execute(&mut host, &[input(2)]).unwrap();
    runtime.commit(&mut host, &[(41, 3)]).unwrap();
    assert_eq!(runtime.state().context(41).unwrap().bulk(), Some((3, 6)));
    host.journal.clear();
    let mut item = input(0);
    item.rows.start = 6;
    runtime.execute(&mut host, &[item]).unwrap();
    assert_eq!(
        host.journal
            .iter()
            .filter(|entry| entry.starts_with("project"))
            .cloned()
            .collect::<Vec<_>>(),
        [
            "project 41 0 3 2",
            "project 41 0 5 1",
            "project 41 1 3 2",
            "project 41 1 5 1",
            "project 41 2 3 2",
            "project 41 2 5 1",
        ]
    );
    runtime.commit(&mut host, &[]).unwrap();
    assert_eq!(runtime.state().context(41).unwrap().bulk(), None);
}

#[test]
fn mixed_batch_stages_correct_tokens_for_prefill_ar_and_verify_items() {
    let (mut runtime, mut host) = setup();
    runtime
        .restore_after_copy(
            42,
            1,
            Snapshot {
                start: 0,
                end: 2,
                window: 4,
                stages: 3,
            },
        )
        .unwrap();
    host.committed.insert(42, vec![6, 7]);
    let rows = vec![
        Input {
            rows: Rows {
                request: 43,
                slot: 2,
                start: 0,
                kind: Kind::Prefill { rows: 2 },
            },
            tokens: vec![8, 9],
            noise: crate::dspark::execution::Noise::GREEDY,
        },
        Input {
            rows: Rows {
                request: 42,
                slot: 1,
                start: 2,
                kind: Kind::Decode { drafts: 0 },
            },
            tokens: vec![10],
            noise: crate::dspark::execution::Noise::GREEDY,
        },
        input(2),
    ];
    let drafts = runtime.execute(&mut host, &rows).unwrap();
    assert_eq!(drafts.len(), 1);
    assert_eq!(host.target_rows[0].tokens, vec![8, 9]);
    assert_eq!(host.target_rows[1].tokens, vec![10]);
    assert_eq!(
        host.target_rows[2].tokens,
        [vec![4], drafts[0].tokens.clone()].concat()
    );
    let before = host.journal.clone();
    assert!(runtime.commit(&mut host, &[(42, 1), (41, 1)]).is_err());
    assert_eq!(host.journal, before);
    runtime.commit(&mut host, &[(41, 1)]).unwrap();
    assert_eq!(host.committed[&41], vec![1, 2, 3, 4]);
    assert_eq!(host.committed[&42], vec![6, 7, 10]);
    assert_eq!(host.committed[&43], vec![8, 9]);
}

#[test]
fn malformed_inputs_fail_before_host_calls_and_bad_device_ids_fail_before_gather() {
    let (mut runtime, mut host) = setup();
    for tokens in [vec![], vec![4, 5], vec![VOCAB as u32]] {
        let mut bad = input(1);
        bad.tokens = tokens;
        assert!(runtime.execute(&mut host, &[bad]).is_err());
        assert!(!runtime.state().is_pending());
        assert!(host.journal.is_empty());
    }
    // Even an un-emitted tail ID must come from a complete, valid block.
    host.bad_tail_id = true;
    assert!(runtime.execute(&mut host, &[input(1)]).is_err());
    assert!(runtime.state().is_poisoned());
    assert!(
        !host
            .journal
            .iter()
            .any(|entry| entry.starts_with("hash_gather "))
    );
    assert_eq!(host.committed[&41], vec![1, 2, 3]);
}

#[test]
fn host_failures_poison_the_instance_without_publishing_target_rows() {
    for operation in [
        "project",
        "draft",
        "read_ids",
        "read_confidence",
        "hash_gather",
        "upload",
        "target",
        "commit",
    ] {
        let (mut runtime, mut host) = setup();
        host.fail = Some(operation);
        let result = runtime.execute(&mut host, &[input(5)]);
        if operation == "commit" {
            result.unwrap();
            assert_eq!(
                runtime.commit(&mut host, &[(41, 6)]),
                Err(Error::Host("injected host failure"))
            );
        } else {
            assert_eq!(result, Err(Error::Host("injected host failure")));
        }
        assert!(runtime.state().is_poisoned());
        assert!(runtime.cancel(&mut host).is_err());
        assert!(runtime.execute(&mut host, &[input(5)]).is_err());
        assert_eq!(host.committed[&41], vec![1, 2, 3]);
        assert_eq!(runtime.state().context(41).unwrap().end, 3);
    }
}

#[test]
fn deferred_target_matches_host_rows_commits_and_diagnostics_at_every_budget_and_depth() {
    for budget in 1..=BLOCK_ROWS {
        for kept in 1..=budget + 1 {
            let (mut eager, mut a) = setup();
            let (mut deferred, mut b) = setup();
            b.device_target = true;
            let expected = eager.execute(&mut a, &[input(budget)]).unwrap();
            let actual = deferred.execute(&mut b, &[input(budget)]).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(b.target_rows, a.target_rows);
            assert_eq!(b.target_prefixes, a.target_prefixes);
            assert!(!b.journal.iter().any(|line| line.starts_with("read_ids ")
                || line.starts_with("hash_gather ")
                || line.starts_with("upload ")));
            let at = |prefix: &str| {
                b.journal
                    .iter()
                    .position(|line| line.starts_with(prefix))
                    .unwrap()
            };
            assert!(at("target ") < at("device_results "));
            assert!(at("device_results ") < at("read_confidence "));
            assert_eq!(
                deferred.commit(&mut b, &[(41, kept)]).unwrap(),
                eager.commit(&mut a, &[(41, kept)]).unwrap()
            );
            assert_eq!(b.committed, a.committed);
            assert_eq!(
                deferred.state().context(41).unwrap().end,
                eager.state().context(41).unwrap().end
            );
        }
    }
}

#[test]
fn deferred_failure_poison_and_cancellation_keep_committed_history() {
    for failure in ["device_hash_gather", "target", "device_results"] {
        let (mut runtime, mut host) = setup();
        host.device_target = true;
        host.fail = Some(failure);
        assert!(runtime.execute(&mut host, &[input(5)]).is_err());
        assert!(runtime.state().is_poisoned());
        assert_eq!(host.committed[&41], [1, 2, 3]);
    }
    let (mut runtime, mut host) = setup();
    host.device_target = true;
    runtime.execute(&mut host, &[input(2)]).unwrap();
    let diagnostics = runtime.cancel(&mut host).unwrap();
    assert_eq!(diagnostics[0].outcomes[..2], [Outcome::Cancelled; 2]);
    assert_eq!(host.committed[&41], [1, 2, 3]);
    let (mut runtime, mut host) = setup();
    host.device_target = true;
    host.bad_tail_id = true;
    assert!(runtime.execute(&mut host, &[input(1)]).is_err());
    assert!(runtime.state().is_poisoned());
}
