use super::*;
use runtime_engine::SamplingWork;
use runtime_sampling::{SamplingParameters, SamplingState};

pub(super) fn work(temperature: f64) -> SamplingWork {
    SamplingWork {
        parameters: SamplingParameters::new(temperature).unwrap(),
        state: SamplingState::new(71),
        history: vec![3, 9, 9].into(),
        emitted_tokens: 1,
        stop_ids: Vec::new(),
        constraint: None,
    }
}

fn items(rows: usize, case: usize) -> Vec<ModelBatchItem> {
    (0..rows)
        .map(|row| {
            let mut sampling = work(if case == 0 { 0.0 } else { 1.0 });
            sampling.state = SamplingState::new(71 + row as u64);
            if row + 1 == rows {
                sampling.parameters = match case {
                    2 => sampling
                        .parameters
                        .with_top_p(0.9)
                        .unwrap()
                        .with_min_p(0.01)
                        .unwrap(),
                    3 => sampling.parameters.with_top_logprobs(5).unwrap(),
                    4 => SamplingParameters::new(0.0).unwrap(),
                    _ => sampling.parameters,
                };
            }
            ModelBatchItem::new(
                RequestId::new(row as u64 + 1),
                BatchPhase::Decode,
                vec![9],
                true,
            )
            .with_sampling(sampling)
        })
        .collect()
}

fn view(context: usize, first: usize, rows: usize, width: usize) -> OwnedLogitsViewV1 {
    let mut raw = runtime_bundle_abi::BundleLogitsViewV1::empty();
    raw.context = runtime_bundle_abi::BundleContextHandleV1::from_raw(context as u64 + 1);
    raw.element_type = runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32;
    raw.row_count = rows as u32;
    raw.first_item_index = first as u32;
    raw.live_width = width as u32;
    raw.row_stride_elements = width as u64;
    raw.buffer.offset = (first * width * 4) as u64;
    raw.buffer.length = (rows * width * 4) as u64;
    OwnedLogitsViewV1::new(raw)
}

fn sink<'a>(
    items: &'a [ModelBatchItem],
    sampler: Rc<dyn DeviceLogitsSamplerV1>,
    rows: usize,
    width: usize,
    defer: bool,
) -> CollectingSinkV1<'a> {
    CollectingSinkV1 {
        limits: ResultSinkLimitsV1::new(rows as u64, width as u64, (rows * width) as u64),
        rows: Vec::new(),
        sampled: Vec::new(),
        prompt_view_rows: Vec::new(),
        expected: items,
        sampled_indices: BTreeSet::new(),
        drafts: BTreeMap::new(),
        sampler: Some(sampler),
        defer,
        pending: Vec::new(),
        sampling_groups: BTreeMap::new(),
    }
}

#[test]
fn owner_partitions_keep_gathered_dispatch_for_mixed_settings_and_capacity_boundaries() {
    #[derive(Default)]
    struct Recorder(RefCell<Vec<(u32, u32, SamplingDispatch)>>);
    impl DeviceLogitsSamplerV1 for Recorder {
        fn sample_view(
            &self,
            _: OwnedLogitsViewV1,
            _: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            panic!("canonical dispatch must reach the sampler")
        }
        fn sample_view_with_dispatch(
            &self,
            view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
            dispatch: SamplingDispatch,
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            self.0
                .borrow_mut()
                .push((view.raw().first_item_index, view.raw().row_count, dispatch));
            Ok(items
                .iter()
                .map(|item| SampledRow::new(item.request_id(), 0, 0.0, Vec::new()))
                .collect())
        }
    }
    for rows in [1, 8, 16, 32, 96] {
        for case in 0..5 {
            let items = items(rows, case);
            for split in [false, true] {
                let sampler = Rc::new(Recorder::default());
                let mut sink = sink(&items, sampler.clone(), rows, 32, true);
                let block = if split { rows.div_ceil(4) } else { rows };
                for first in (0..rows).step_by(block) {
                    sink.emit_view(view(first / block, first, block.min(rows - first), 32))
                        .unwrap();
                }
                sink.collect_pending().unwrap();
                assert_eq!(sink.sampled.len(), rows);
                for &(first, count, dispatch) in sampler.0.borrow().iter() {
                    let group = gathered_ranges(&items)
                        .into_iter()
                        .find(|g| g.contains(&(first as usize)))
                        .unwrap();
                    assert!(first as usize + count as usize <= group.end);
                    let expected =
                        SamplingDispatch::for_items(&items[group].iter().collect::<Vec<_>>())
                            .unwrap();
                    assert_eq!(dispatch, expected);
                    if rows == 16 && case == 0 {
                        assert_eq!(dispatch.threads(), 64);
                        if split {
                            // The old per-view choice changes F32 accumulation.
                            let local = SamplingDispatch::for_items(
                                &items[first as usize..(first + count) as usize]
                                    .iter()
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap();
                            assert_eq!(local.threads(), 256);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn gathered_ranges_preserve_item_chunks_phase_changes_and_nonemitting_gaps() {
    use runtime_engine::DraftWork;
    let verify = |id| {
        ModelBatchItem::new(
            RequestId::new(id),
            BatchPhase::Verify { draft_tokens: 39 },
            vec![0; 40],
            true,
        )
        .with_draft(DraftWork::new(39, DraftSource::Host))
    };
    let mut batch = vec![verify(1), verify(2)];
    assert_eq!(gathered_ranges(&batch), vec![0..40, 40..80]);
    batch.push(ModelBatchItem::new(
        RequestId::new(3),
        BatchPhase::Prefill { tokens: 7 },
        vec![0; 7],
        true,
    ));
    batch.push(ModelBatchItem::new(
        RequestId::new(4),
        BatchPhase::Decode,
        vec![0],
        true,
    ));
    batch.push(ModelBatchItem::new(
        RequestId::new(5),
        BatchPhase::Decode,
        vec![0],
        false,
    ));
    batch.push(ModelBatchItem::new(
        RequestId::new(6),
        BatchPhase::Decode,
        vec![0],
        true,
    ));
    assert_eq!(
        gathered_ranges(&batch),
        vec![0..40, 40..80, 80..81, 81..82, 82..83]
    );
}

#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
#[test]
fn gathered_and_owned_sampling_match_gpu() {
    use crate::sampling::DeviceSamplingSession;
    struct NativeSampler {
        sessions: Vec<RefCell<DeviceSamplingSession>>,
        buffers: Vec<runtime_cuda::DeviceBuffer>,
    }
    impl DeviceLogitsSamplerV1 for NativeSampler {
        fn sample_view(
            &self,
            view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            let rank = view.raw().context.raw as usize - 1;
            self.sessions[rank]
                .borrow_mut()
                .sample(&self.buffers[rank], view.raw(), items)
        }
        fn sample_view_with_dispatch(
            &self,
            view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
            dispatch: SamplingDispatch,
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            let rank = view.raw().context.raw as usize - 1;
            self.sessions[rank].borrow_mut().sample_with_dispatch(
                &self.buffers[rank],
                view.raw(),
                items,
                Some(dispatch),
            )
        }
        fn issue_view_with_dispatch(
            &self,
            view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
            dispatch: SamplingDispatch,
        ) -> HostServiceResultV1<bool> {
            let rank = view.raw().context.raw as usize - 1;
            self.sessions[rank].borrow_mut().issue_with_dispatch(
                &self.buffers[rank],
                view.raw(),
                items,
                Some(dispatch),
            )?;
            Ok(true)
        }
        fn collect_view(
            &self,
            view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            let rank = view.raw().context.raw as usize - 1;
            self.sessions[rank].borrow_mut().collect(items)
        }
    }
    let Ok(devices) = std::env::var("CHOPINFER_KERNEL_GPUS") else {
        return;
    };
    let devices = devices
        .split(',')
        .map(|d| d.parse::<i32>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(devices.len(), 4, "four devices required");
    assert_eq!(devices.iter().collect::<BTreeSet<_>>().len(), 4);
    let driver = runtime_cuda::Driver::initialize().unwrap();
    let widths = std::env::var("CHOPINFER_SAMPLING_TEST_WIDTHS")
        .expect("the runner supplies vocabulary widths")
        .split(',')
        .map(|width| width.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    assert!(!widths.is_empty());
    for width in widths {
        assert!((1024..=1 << 20).contains(&width));
        let logits = (0..32)
            .flat_map(|row| {
                (0..width).map(move |token| {
                    if row % 2 == 0 {
                        match token {
                            0 => 0.0,
                            256 | 512 | 768 => -18.0,
                            _ => -1000.0,
                        }
                    } else {
                        ((token * 17 + row * 13) % 127) as f32 / 16.0 - 4.0
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut sessions = Vec::new();
        let mut buffers = Vec::new();
        for &device in &devices {
            let context = driver.retain_primary_context(device).unwrap();
            let stream = context.stream().unwrap();
            let buffer = context
                .default_memory_pool()
                .unwrap()
                .allocate(&stream, logits.len() * 4)
                .unwrap();
            buffer.copy_from_f32(&logits, &stream).unwrap();
            stream.synchronize().unwrap();
            buffers.push(buffer);
            sessions.push(RefCell::new(DeviceSamplingSession::new(context).unwrap()));
        }
        let sampler = Rc::new(NativeSampler { sessions, buffers });
        for rows in [16, 32] {
            for case in 0..5 {
                let items = items(rows, case);
                // No override: the compatibility reference uses the old gathered dispatch.
                let expected = sampler
                    .sample_view(view(0, 0, rows, width), &items.iter().collect::<Vec<_>>())
                    .unwrap();
                if rows == 16 && case == 0 {
                    let old_split = sampler
                        .sample_view(view(0, 0, 4, width), &items[..4].iter().collect::<Vec<_>>())
                        .unwrap();
                    assert_ne!(
                        expected[0].logprob().to_bits(),
                        old_split[0].logprob().to_bits(),
                        "adversarial row must expose the old reduction defect"
                    );
                }
                for defer in [false, true] {
                    let mut sink = sink(&items, sampler.clone(), rows, width, defer);
                    for rank in 0..4 {
                        sink.emit_view(view(rank, rank * (rows / 4), rows / 4, width))
                            .unwrap();
                    }
                    sink.collect_pending().unwrap();
                    assert_eq!(sink.sampled.len(), expected.len());
                    for (row, (actual, expected)) in sink.sampled.iter().zip(&expected).enumerate()
                    {
                        assert_eq!(actual.request_id(), expected.request_id());
                        assert_eq!(
                            actual.token_id(),
                            expected.token_id(),
                            "width {width} C{rows} case {case} row {row}"
                        );
                        assert_eq!(
                            actual.logprob().to_bits(),
                            expected.logprob().to_bits(),
                            "width {width} C{rows} case {case} row {row}"
                        );
                        assert_eq!(actual.top_logprobs().len(), expected.top_logprobs().len());
                        for (a, b) in actual.top_logprobs().iter().zip(expected.top_logprobs()) {
                            assert_eq!(a.token_id(), b.token_id());
                            assert_eq!(a.logprob().to_bits(), b.logprob().to_bits());
                        }
                    }
                }
                eprintln!(
                    "width {width} C{rows} sampling case {case}: gathered == synchronous owned == deferred owned, exact bits"
                );
            }
        }
    }
}

#[test]
fn reference_ranges_match_the_shared_bundle_gather_builder() {
    use bundle_runtime::{
        inputs::PhaseInputs,
        logits::{PhaseOutput, ordered_views},
    };
    use runtime_bundle_abi::{
        BUNDLE_LOGITS_ELEMENT_V1_F32, BundleContextHandleV1, BundleDeviceBufferHandleV1,
        DeviceBufferSpanV1,
    };
    let verify = |id, count| {
        ModelBatchItem::new(
            RequestId::new(id),
            BatchPhase::Verify {
                draft_tokens: count - 1,
            },
            vec![0; count],
            true,
        )
        .with_draft(runtime_engine::DraftWork::new(count - 1, DraftSource::Host))
    };
    let prefill = ModelBatchItem::new(
        RequestId::new(900),
        BatchPhase::Prefill { tokens: 2 },
        vec![0; 2],
        true,
    );
    let batches = vec![
        items(16, 0),
        items(96, 0),
        items(257, 0),
        vec![verify(1, 40), verify(2, 40)],
        vec![verify(1, 100), verify(2, 10)],
        vec![verify(1, 40), prefill.clone(), verify(2, 40)],
        vec![
            prefill.with_prompt_targets(vec![None, Some(0)], 0),
            verify(2, 11),
        ],
    ];
    for batch in batches {
        let mut phases = [PhaseInputs::default(), PhaseInputs::default()];
        let mut position = [0; 2];
        for (index, item) in batch.iter().enumerate() {
            let phase = usize::from(!matches!(item.phase(), BatchPhase::Prefill { .. }));
            let inputs = &mut phases[phase];
            inputs.item_order.push(index);
            inputs.emits_row.push(item.expects_logits());
            inputs.prompt_rows.push(!item.prompt_targets().is_empty());
            inputs.verify_rows.push(item.draft().is_some());
            inputs.q_start.push(position[phase]);
            let tokens = item.phase().appended_tokens() as i32;
            inputs.q_len.push(tokens);
            position[phase] += tokens;
        }
        let outputs = [0, 1].map(|phase| PhaseOutput {
            inputs: &phases[phase],
            buffer: DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, phase as u32 + 1),
                0,
                2048 * 32 * 4,
            ),
            element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
            compact: phase == 0,
        });
        let actual = ordered_views(
            bundle_runtime::ModelLimits::new(32, 2048, 64).unwrap(),
            BundleContextHandleV1::from_parts(1, 1),
            outputs[0],
            outputs[1],
        )
        .unwrap()
        .into_iter()
        .map(|v| v.first_item_index as usize..(v.first_item_index + v.row_count) as usize)
        .collect::<Vec<_>>();
        assert_eq!(gathered_ranges(&batch), actual);
    }
}

#[test]
fn scored_prompt_and_owned_rows_keep_item_dispatch_when_deferral_is_enabled() {
    struct Scorer(RefCell<Vec<(u32, u32, Option<SamplingDispatch>)>>);
    impl DeviceLogitsSamplerV1 for Scorer {
        fn sample_view(
            &self,
            _: OwnedLogitsViewV1,
            _: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            panic!("scored steps use scoring")
        }
        fn score_view(
            &self,
            view: OwnedLogitsViewV1,
            rows: &[DevicePromptRowV1<'_>],
        ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
            self.0
                .borrow_mut()
                .push((view.raw().first_item_index, view.raw().row_count, None));
            Ok((
                rows.iter()
                    .map(|r| {
                        SampledRow::new(
                            r.item.request_id(),
                            r.target.unwrap_or(0),
                            -1.0,
                            Vec::new(),
                        )
                    })
                    .collect(),
                Vec::new(),
            ))
        }
        fn score_view_with_dispatch(
            &self,
            view: OwnedLogitsViewV1,
            rows: &[DevicePromptRowV1<'_>],
            dispatch: SamplingDispatch,
        ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
            let result = self.score_view(view, rows)?;
            self.0.borrow_mut().last_mut().unwrap().2 = Some(dispatch);
            Ok(result)
        }
        fn issue_view(
            &self,
            _: OwnedLogitsViewV1,
            _: &[&ModelBatchItem],
        ) -> HostServiceResultV1<bool> {
            panic!("prompt-scored steps must stay synchronous")
        }
    }
    let mut batch = vec![
        ModelBatchItem::new(
            RequestId::new(1),
            BatchPhase::Prefill { tokens: 2 },
            vec![0; 2],
            false,
        )
        .with_prompt_targets(vec![None, Some(0)], 0),
        ModelBatchItem::new(
            RequestId::new(2),
            BatchPhase::Verify { draft_tokens: 15 },
            vec![9; 16],
            true,
        )
        .with_draft(runtime_engine::DraftWork::new(15, DraftSource::Host))
        .with_sampling(work(0.0)),
    ];
    batch.extend((0..4).map(|row| {
        ModelBatchItem::new(RequestId::new(100 + row), BatchPhase::Decode, vec![9], true)
            .with_sampling(work(0.0))
    }));
    let sampler = Rc::new(Scorer(RefCell::new(Vec::new())));
    let mut sink = sink(&batch, sampler.clone(), 22, 32, true);
    sink.emit_view(view(0, 0, 2, 32)).unwrap();
    for rank in 0..2 {
        sink.emit_view(view(rank, 2 + rank * 8, 8, 32)).unwrap();
    }
    for row in 0..4 {
        sink.emit_view(view(2, 18 + row, 1, 32)).unwrap();
    }
    sink.collect_pending().unwrap();
    assert_eq!(sink.prompt_view_rows.len(), 2);
    assert_eq!(sink.sampled.len(), 20);
    assert!(sink.pending.is_empty());
    assert!(sampler.0.borrow()[0].2.is_none());
    for (_, rows, dispatch) in &sampler.0.borrow()[1..3] {
        assert_eq!(*rows, 8);
        let dispatch = dispatch.unwrap();
        assert_eq!(dispatch.rows, 16);
        assert_eq!(dispatch.threads(), 64);
    }
    for (_, rows, dispatch) in &sampler.0.borrow()[3..] {
        assert_eq!(*rows, 1);
        let dispatch = dispatch.unwrap();
        assert_eq!(dispatch.rows, 1);
        assert_eq!(dispatch.threads(), 256);
    }
}

#[test]
fn reference_forcing_keeps_dispatch_for_forced_and_unforced_partitions() {
    use crate::reference_forcing::{ReferenceForcingSamplerV1, ReferenceTokensV1};
    struct Recorder(RefCell<Vec<SamplingDispatch>>);
    impl DeviceLogitsSamplerV1 for Recorder {
        fn sample_view(
            &self,
            _: OwnedLogitsViewV1,
            _: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            panic!("dispatch was lost")
        }
        fn sample_view_with_dispatch(
            &self,
            _: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
            dispatch: SamplingDispatch,
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            self.0.borrow_mut().push(dispatch);
            Ok(items
                .iter()
                .map(|item| SampledRow::new(item.request_id(), 0, 0.0, Vec::new()))
                .collect())
        }
        fn score_view_with_dispatch(
            &self,
            _: OwnedLogitsViewV1,
            rows: &[DevicePromptRowV1<'_>],
            dispatch: SamplingDispatch,
        ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
            self.0.borrow_mut().push(dispatch);
            assert_eq!(rows.last().unwrap().target, Some(7));
            Ok((
                rows.iter()
                    .map(|row| {
                        SampledRow::new(
                            row.item.request_id(),
                            row.target.unwrap_or(0),
                            0.0,
                            Vec::new(),
                        )
                    })
                    .collect(),
                Vec::new(),
            ))
        }
    }
    let mut batch = items(16, 2);
    let mut forced = batch[15].sampling().unwrap().clone();
    forced.history = vec![4, 9, 9].into();
    batch[15] = batch[15].clone().with_sampling(forced);
    assert!(
        !SamplingDispatch::for_items(&batch.iter().collect::<Vec<_>>())
            .unwrap()
            .simple
    );
    let inner = Rc::new(Recorder(RefCell::new(Vec::new())));
    let sampler = Rc::new(ReferenceForcingSamplerV1::new(
        inner.clone(),
        Rc::new(ReferenceTokensV1::new(vec![vec![4, 9, 9, 7]])),
    ));
    let mut sink = sink(&batch, sampler, 16, 32, true);
    for rank in 0..4 {
        sink.emit_view(view(rank, rank * 4, 4, 32)).unwrap();
    }
    sink.collect_pending().unwrap();
    assert_eq!(sink.sampled.len(), 16);
    assert_eq!(inner.0.borrow().len(), 4);
    for &dispatch in inner.0.borrow().iter() {
        assert!(
            dispatch.simple,
            "the only filtered row is scored as a forced raw token"
        );
        assert_eq!(dispatch.rows, 16);
        assert_eq!(dispatch.threads(), 64);
    }
}
