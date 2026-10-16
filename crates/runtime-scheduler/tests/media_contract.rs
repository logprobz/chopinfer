use runtime_kv::KvGeometry;
use runtime_scheduler::{
    Batch, BatchOutcome, BatchPhase, ChunkMedia, DraftLimits, InvalidRequest, RequestId,
    RequestSpec, RequestState, Scheduler, SchedulerConfig, SubmitError,
};

fn scheduler(budget: usize, chunk: usize, media_pages: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(4, 16, budget, chunk, 1)
            .unwrap()
            .with_media_pages(media_pages),
        KvGeometry::new(4, 64).unwrap(),
    )
    .unwrap()
}

fn spec(id: u64, prompt: usize, pages: usize) -> RequestSpec {
    RequestSpec::new(RequestId::new(id), prompt, 4)
        .unwrap()
        .with_media_pages(pages)
}

fn commit(scheduler: &mut Scheduler, batch: &Batch) {
    let outcomes: Vec<_> = batch
        .items()
        .iter()
        .map(|item| {
            BatchOutcome::success(
                item.request_id(),
                item.phase(),
                usize::from(item.expects_output()),
                false,
            )
        })
        .collect();
    scheduler.commit_batch(batch.id(), &outcomes).unwrap();
    scheduler.audit().unwrap();
}

fn ids(batch: &Batch) -> Vec<(u64, BatchPhase, usize)> {
    batch
        .items()
        .iter()
        .map(|item| (item.request_id().get(), item.phase(), item.token_cost()))
        .collect()
}

fn state(scheduler: &Scheduler, id: u64) -> RequestState {
    scheduler.request_state(RequestId::new(id)).unwrap()
}

/// A chunk's media demand with only an encode cost.
const fn encode(encode_cost: usize) -> ChunkMedia {
    ChunkMedia {
        encode_cost,
        pages: 0,
    }
}

/// A request's spec media pages are its largest single-step demand: only a
/// request whose smallest chunk cannot fit the arena is refused.
#[test]
fn a_request_whose_single_step_exceeds_the_arena_is_refused_at_submission() {
    let mut without_arena = scheduler(16, 8, 0);
    assert_eq!(
        without_arena.submit(spec(1, 4, 1)),
        Err(SubmitError::Invalid(InvalidRequest::ExceedsMediaArena {
            maximum: 0,
            requested: 1
        }))
    );
    let mut small = scheduler(16, 8, 3);
    assert_eq!(
        small.submit(spec(1, 4, 4)),
        Err(SubmitError::Invalid(InvalidRequest::ExceedsMediaArena {
            maximum: 3,
            requested: 4
        }))
    );
    small.submit(spec(2, 4, 3)).unwrap();
    small.submit(spec(3, 4, 0)).unwrap();
}

/// Admission reserves no arena pages: media requests whose step demands
/// exceed the arena together are admitted at once. A media prompt starts
/// only while the step demands of the started ones and its own fit the
/// arena, in arrival order, so started prompts keep their items between
/// chunks; text prompts are never held back.
#[test]
fn media_prompts_start_while_their_step_demands_fit_the_arena_together() {
    let mut scheduler = scheduler(16, 4, 4);
    scheduler.submit(spec(1, 8, 3)).unwrap();
    scheduler.submit(spec(2, 8, 3)).unwrap();
    scheduler.submit(spec(3, 8, 1)).unwrap();
    scheduler.submit(spec(4, 4, 0)).unwrap();
    let pages = |id: RequestId, _: usize, _: usize| ChunkMedia {
        encode_cost: 0,
        pages: [0, 3, 3, 1, 0][id.get() as usize],
    };
    let step = |scheduler: &mut Scheduler| {
        let batch = scheduler
            .next_batch_planned(DraftLimits::NONE, pages, |_, _| Vec::new())
            .unwrap()
            .unwrap();
        commit(scheduler, &batch);
        ids(&batch)
            .into_iter()
            .filter(|(_, phase, _)| matches!(phase, BatchPhase::Prefill { .. }))
            .map(|(id, _, _)| id)
            .collect::<Vec<_>>()
    };
    // Request 1 starts; request 2 would make six pages of four, and request
    // 3, which would fit, arrived after it. The text prompt runs.
    assert_eq!(step(&mut scheduler), [1, 4]);
    assert_eq!(state(&scheduler, 2), RequestState::Admitted);
    assert_eq!(state(&scheduler, 3), RequestState::Admitted);
    // Request 1 is still prefilling, so both keep waiting.
    assert_eq!(step(&mut scheduler), [1]);
    // Once it decodes, requests 2 and 3 fit the arena together.
    assert_eq!(step(&mut scheduler), [2, 3]);
    // Cancelling a started media request holds nothing back from the others.
    scheduler.cancel(RequestId::new(2)).unwrap();
    scheduler.audit().unwrap();
    assert_eq!(step(&mut scheduler), [3]);
}

/// A prompt's demand counts only the media ahead of its prefill. One with
/// none ahead, like a turn whose images all lie in its restored prefix,
/// starts though it arrived after a prompt the arena holds back; once the
/// first prompt's prefill passes its only item, the held one starts while
/// the first still prefills its text.
#[test]
fn only_the_media_ahead_of_a_prompt_count_toward_the_arena() {
    let mut scheduler = scheduler(16, 4, 4);
    scheduler.submit(spec(1, 8, 4)).unwrap();
    scheduler.submit(spec(2, 8, 4)).unwrap();
    scheduler.submit(spec(3, 8, 4)).unwrap();
    // Request 1's item lies in its first four rows, request 2's in all and
    // request 3's in none ahead of its prefill.
    let pages = |id: RequestId, offset: usize, _: usize| ChunkMedia {
        encode_cost: 0,
        pages: match id.get() {
            1 if offset >= 4 => 0,
            3 => 0,
            _ => 4,
        },
    };
    let step = |scheduler: &mut Scheduler| {
        let batch = scheduler
            .next_batch_planned(DraftLimits::NONE, pages, |_, _| Vec::new())
            .unwrap()
            .unwrap();
        commit(scheduler, &batch);
        let mut prefills: Vec<_> = ids(&batch)
            .into_iter()
            .filter(|(_, phase, _)| matches!(phase, BatchPhase::Prefill { .. }))
            .map(|(id, _, _)| id)
            .collect();
        prefills.sort_unstable();
        prefills
    };
    assert_eq!(step(&mut scheduler), [1, 3]);
    assert_eq!(state(&scheduler, 1), RequestState::Prefilling);
    assert_eq!(state(&scheduler, 2), RequestState::Admitted);
    assert_eq!(step(&mut scheduler), [1, 2, 3]);
}

/// Items spanning prompt rows, with their arena pages, for a chunk-media
/// callback: a chunk covers every item it overlaps.
fn covered(items: &[(usize, usize, usize)], offset: usize, tokens: usize) -> usize {
    items
        .iter()
        .filter(|(start, end, _)| *start < offset + tokens && *end > offset)
        .map(|(_, _, pages)| pages)
        .sum()
}

/// A chunk whose media exceed the whole arena is shortened to the longest
/// granule-aligned chunk whose media fit, and the rest follows in later
/// steps, so a request whose images total more than the arena completes.
#[test]
fn a_chunk_whose_media_exceed_the_arena_shortens_at_a_granule_boundary() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(4, 16, 64, 16, 1)
            .unwrap()
            .with_prefill_granule(4)
            .unwrap()
            .with_media_pages(5),
        KvGeometry::new(4, 64).unwrap(),
    )
    .unwrap();
    // Eight pages of media in all, five at most in any one granule.
    let items = [(0, 6, 2), (6, 12, 3), (12, 16, 3)];
    scheduler.submit(spec(1, 20, 5)).unwrap();
    let mut chunks = Vec::new();
    while let Some(batch) = scheduler
        .next_batch_planned(
            DraftLimits::NONE,
            |_, offset, tokens| ChunkMedia {
                encode_cost: 0,
                pages: covered(&items, offset, tokens),
            },
            |_, _| Vec::new(),
        )
        .unwrap()
    {
        for item in batch.items() {
            if let BatchPhase::Prefill { tokens } = item.phase() {
                chunks.push(tokens);
            }
        }
        commit(&mut scheduler, &batch);
        if state(&scheduler, 1) != RequestState::Prefilling
            && state(&scheduler, 1) != RequestState::Admitted
        {
            break;
        }
    }
    // Rows 0..12 cover the first two items (five pages); a 16-row chunk
    // would add the third. The rest then fits.
    assert_eq!(chunks, [12, 8]);
}

/// The chunks of one step never cover more pages than the arena, however
/// requests, chunk sizes and encodes mix.
#[test]
fn no_step_covers_more_media_pages_than_the_arena() {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move |bound: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % bound as u64) as usize
    };
    let pages =
        |id: RequestId, offset: usize, tokens: usize| (id.get() as usize * 7 + offset + tokens) % 6;
    for _ in 0..100 {
        let arena = 6 + next(8);
        let mut scheduler = scheduler(32, 1 + next(16), arena);
        let mut offsets = std::collections::BTreeMap::new();
        for id in 1..=6 {
            scheduler.submit(spec(id, 1 + next(40), next(6))).unwrap();
            offsets.insert(id, 0usize);
        }
        for _ in 0..200 {
            let Some(batch) = scheduler
                .next_batch_planned(
                    DraftLimits::NONE,
                    |id, offset, tokens| ChunkMedia {
                        encode_cost: next(3),
                        pages: pages(id, offset, tokens),
                    },
                    |_, _| Vec::new(),
                )
                .unwrap()
            else {
                break;
            };
            let covered: usize = batch
                .items()
                .iter()
                .filter_map(|item| match item.phase() {
                    BatchPhase::Prefill { tokens } => {
                        let offset = offsets[&item.request_id().get()];
                        Some(pages(item.request_id(), offset, tokens))
                    }
                    _ => None,
                })
                .sum();
            assert!(
                covered <= arena,
                "{covered} pages exceed the arena of {arena}"
            );
            for item in batch.items() {
                if let BatchPhase::Prefill { tokens } = item.phase() {
                    *offsets.get_mut(&item.request_id().get()).unwrap() += tokens;
                }
            }
            commit(&mut scheduler, &batch);
        }
    }
}

#[test]
fn encode_costs_are_charged_and_one_chunk_may_overrun_the_budget() {
    let mut scheduler = scheduler(8, 4, 16);
    scheduler.submit(spec(1, 2, 1)).unwrap();
    scheduler.submit(spec(2, 2, 1)).unwrap();
    scheduler.submit(spec(3, 2, 1)).unwrap();

    // Request 1's chunk fits with its encode; request 2's does not fit what
    // is left, but it is the first to overrun, so it is taken and the batch
    // closes; request 3 waits.
    let batch = scheduler
        .next_batch_planned(
            DraftLimits::NONE,
            |id, offset, tokens| {
                assert_eq!((offset, tokens), (0, 2));
                encode(match id.get() {
                    1 => 2,
                    _ => 100,
                })
            },
            |_, _| Vec::new(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        ids(&batch),
        [
            (1, BatchPhase::Prefill { tokens: 2 }, 4),
            (2, BatchPhase::Prefill { tokens: 2 }, 102)
        ]
    );
    commit(&mut scheduler, &batch);

    // A second overrunning chunk in one batch waits for the next batch.
    let batch = scheduler
        .next_batch_planned(
            DraftLimits::NONE,
            |id, _, _| encode(if id.get() == 3 { 100 } else { 0 }),
            |_, _| Vec::new(),
        )
        .unwrap()
        .unwrap();
    assert!(
        batch
            .items()
            .iter()
            .any(|item| item.request_id() == RequestId::new(3))
    );
    commit(&mut scheduler, &batch);
}

/// Only the encode may overrun. The overrunning chunk's prompt tokens are
/// held back from the items after it, so a batch never carries more tokens
/// than its budget, which a bundle sizes its prefill arena by.
#[test]
fn an_overrunning_encode_never_takes_the_batch_past_its_token_budget() {
    let mut scheduler = scheduler(8, 4, 16);
    scheduler.submit(spec(1, 4, 0)).unwrap();
    scheduler.submit(spec(2, 3, 1)).unwrap();
    scheduler.submit(spec(3, 4, 0)).unwrap();
    scheduler.submit(spec(4, 1, 0)).unwrap();
    let batch = scheduler
        .next_batch_planned(
            DraftLimits::NONE,
            |id, _, _| encode(if id.get() == 2 { 100 } else { 0 }),
            |_, _| Vec::new(),
        )
        .unwrap()
        .unwrap();
    // Request 2 overruns with 4 tokens left. Its 3 tokens stay reserved, so
    // request 3's full chunk waits and request 4's single token still fits.
    assert_eq!(
        ids(&batch),
        [
            (1, BatchPhase::Prefill { tokens: 4 }, 4),
            (4, BatchPhase::Prefill { tokens: 1 }, 1),
            (2, BatchPhase::Prefill { tokens: 3 }, 103)
        ]
    );
    let tokens: usize = batch
        .items()
        .iter()
        .map(|item| item.phase().appended_tokens())
        .sum();
    assert!(tokens <= 8, "{tokens} tokens exceed the budget of 8");
    commit(&mut scheduler, &batch);
}

#[test]
fn an_overrunning_encode_leaves_no_budget_for_drafts() {
    let mut scheduler = scheduler(8, 4, 16);
    scheduler.submit(spec(1, 2, 0)).unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    commit(&mut scheduler, &batch);
    scheduler.submit(spec(2, 4, 1)).unwrap();
    let mut planned = false;
    let batch = scheduler
        .next_batch_planned(
            DraftLimits::new(4, 32),
            |_, _, _| encode(100),
            |candidates, budget| {
                planned = true;
                assert_eq!(budget.spare_draft_tokens(), 0);
                vec![0; candidates.len()]
            },
        )
        .unwrap()
        .unwrap();
    assert!(planned);
    assert_eq!(batch.items().len(), 2);
    commit(&mut scheduler, &batch);
}

/// Across mixed prefill, media encodes, decodes and drafts, no batch carries
/// more tokens than its budget.
#[test]
fn batches_never_exceed_their_token_budget() {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move |bound: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % bound as u64) as usize
    };
    for round in 0..200 {
        let budget = 8 + next(24);
        let chunk = 1 + next(budget);
        let mut scheduler = Scheduler::new(
            SchedulerConfig::new(4, 16, budget, chunk, 1)
                .unwrap()
                .with_media_pages(64),
            KvGeometry::new(4, 256).unwrap(),
        )
        .unwrap();
        let mut emitted = std::collections::BTreeMap::new();
        for id in 1..=8 {
            let spec = RequestSpec::new(RequestId::new(id), 1 + next(40), 4)
                .unwrap()
                .with_media_pages(next(3));
            scheduler.submit(spec).unwrap();
            emitted.insert(id, 0usize);
        }
        for _ in 0..400 {
            let encode = next(3) * next(2 * budget);
            let Some(batch) = scheduler
                .next_batch_planned(
                    DraftLimits::new(3, budget),
                    |_, _, _| self::encode(encode),
                    |candidates, spare| {
                        let mut left = spare.spare_draft_tokens().min(spare.spare_rows());
                        candidates
                            .iter()
                            .map(|candidate| {
                                let drafts = candidate.maximum_draft_tokens().min(left);
                                left -= drafts;
                                drafts
                            })
                            .collect()
                    },
                )
                .unwrap()
            else {
                break;
            };
            let tokens: usize = batch
                .items()
                .iter()
                .map(|item| item.phase().appended_tokens())
                .sum();
            assert!(
                tokens <= budget,
                "round {round}: {tokens} tokens exceed the budget of {budget}: {:?}",
                ids(&batch)
            );
            let outcomes: Vec<_> = batch
                .items()
                .iter()
                .map(|item| {
                    let count = emitted.get_mut(&item.request_id().get()).unwrap();
                    *count += usize::from(item.expects_output());
                    BatchOutcome::success(
                        item.request_id(),
                        item.phase(),
                        usize::from(item.expects_output()),
                        *count == 4,
                    )
                })
                .collect();
            scheduler.commit_batch(batch.id(), &outcomes).unwrap();
        }
        assert!(
            (1..=8).all(|id| state(&scheduler, id) == RequestState::Finished),
            "round {round} left requests unfinished"
        );
    }
}
