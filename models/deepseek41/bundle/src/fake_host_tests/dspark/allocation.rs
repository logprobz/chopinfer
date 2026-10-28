//! Allocated verify steps (`chopbatch-v7`): each DSpark instance prepares
//! their ops and storage. It runs them on the device results path, allocating on
//! every rank after the draft copies and before the hash, computing the
//! width eagerly, gathering the head and feature rows back into the
//! reserved layout and walking the verified drafts.
use super::results::{DeviceSink, metadata};
use super::*;
use runtime_bundle_sdk::{SamplingKindV1, VerifyAllocationV1, VerifyCalibrationV1};

/// `items` verify items reserving `k` drafts each from position `start`,
/// allocated to `width` rows.
fn allocated(items: u32, start: u64, k: u32, width: u32) -> BatchRequestV1 {
    let calibration = VerifyCalibrationV1::new(&[(1.0, 0.0); 5], &[(1.0, 0.0); 5]).unwrap();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..items)
            .map(|slot| BatchItemV1::verify(u64::from(slot + 1), k + 1, true, vec![7]))
            .collect(),
        (0..items)
            .map(|slot| {
                BatchAppendV1::new(
                    start,
                    u64::from(k + 1),
                    start + u64::from(k + 1),
                    vec![u64::from(slot)],
                    None,
                )
            })
            .collect(),
    )
    .with_state_slots((0..items).map(u64::from).collect())
    .with_verify_allocation(VerifyAllocationV1::new(
        width,
        1,
        calibration,
        vec![SamplingKindV1::Sampled; items as usize],
    ))
}

/// The position in `step` of the first launch of `symbol` on `stream`.
fn launch_of(step: &[String], stream: u32, symbol: &str) -> Option<usize> {
    step.iter()
        .position(|line| line.starts_with(&format!("launch {stream} {symbol} ")))
}

/// The buffer slot of every device buffer `line`'s launch binds.
fn bound_buffers(line: &str) -> Vec<u32> {
    line.split("BundleDeviceBufferHandleV1 { raw: ")
        .skip(1)
        .map(|rest| {
            let raw: u64 = rest.split(' ').next().unwrap().parse().unwrap();
            raw as u32
        })
        .collect()
}

/// Executes `request` with device results for `items` items of `k`
/// drafts each (cycle `epoch`), then commits each item's bonus row;
/// returns the step's journal, its hash on rank 0, and its acceptance of
/// `acceptance`.
fn device_step(
    host: &Rc<Host>,
    instance: &mut BundleInstanceV1,
    request: &[u8],
    (items, k, epoch): (u32, u32, u64),
    acceptance: &str,
) -> (Vec<String>, usize, usize) {
    let mut sink = DeviceSink {
        host: Rc::clone(host),
        metadata: Some(metadata(items, k, epoch)),
        legacy: Sink::default(),
        drafts: Vec::new(),
        samples: Vec::new(),
        completion: None,
    };
    let start = host.journal.borrow().len();
    instance
        .execute(request, &mut sink)
        .unwrap_or_else(|e| panic!("cycle {epoch}: {e:?} {:?}", host.logs.borrow()));
    assert_eq!(sink.drafts.len(), items as usize);
    assert_eq!(
        sink.samples
            .iter()
            .map(|s| s.logits.logits.row_count)
            .sum::<u32>(),
        items * (k + 1)
    );
    assert_eq!(sink.completion.unwrap().sample_rows, items * (k + 1));
    instance
        .commit_accepted(
            &(1..=items)
                .map(|id| super::kept(u64::from(id), 1))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let step = host.journal.borrow()[start..].to_vec();
    let rank0 = host.rank_streams.borrow()[&0];
    let hash = launch_of(&step, rank0, crate::sm120::engram_hash_rows::SYMBOL)
        .expect("rank 0 hashes the computed rows");
    let accepted = launch_of(&step, rank0, acceptance).expect("the step's acceptance runs");
    assert!(hash < accepted);
    (step, hash, accepted)
}

/// An allocated step runs on every topology: each rank allocates after its
/// draft copies and before rank 0 hashes the computed rows, the target
/// computes the width, every rank gathers its head and packed feature rows
/// back into the reserved rows by index, the sampler reads the reserved
/// rows, and the allocated acceptance walks the verified drafts. A host
/// without device results is refused before any work. The target's graph
/// is keyed by the width in the allocated bank: the first step of a width
/// captures it, a step of the same width that reserves other rows replays
/// it without lowering when its reservation rounds to the same tail rows
/// (a multiple of 16) and captures another when it does not, and an
/// unallocated step of the same key captures its own.
#[test]
fn allocated_steps_compute_the_width_and_keep_the_reserved_layout() {
    for shape in crate::shape::topologies() {
        let (items, width) = (8_u32, 16_u32);
        let (host, mut instance) = create_with_limits(shape, 8, 48);
        for slot in 0..items {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(u64::from(slot + 1), u64::from(slot), 0, 3, true),
            );
            instance.commit().unwrap();
        }
        let request = allocated(items, 3, 3, width).encode().unwrap();
        let mut legacy = Sink::default();
        assert_eq!(
            instance.execute(&request, &mut legacy).err(),
            Some(BundleAbiError::Status(BundleStatusV1::Unsupported))
        );
        let streams = host.rank_streams.borrow().clone();
        let allocated_acceptance = crate::sm120::allocated_acceptance_entries::SYMBOL;
        // Every rank allocates before rank 0 hashes, and no host
        // acceptance runs.
        let allocates = |step: &[String], hash: usize| {
            assert!(
                launch_of(
                    step,
                    streams[&0],
                    crate::sm120::host_acceptance_entries::SYMBOL
                )
                .is_none()
            );
            for rank in 0..shape.ranks() {
                let allocate =
                    launch_of(step, streams[&rank], crate::sm120::allocate_entries::SYMBOL)
                        .unwrap_or_else(|| panic!("rank {rank} allocates"));
                assert!(allocate < hash);
            }
        };
        let lines = |step: &[String], prefix: &str, stream: u32| {
            step.iter()
                .filter(|line| line.starts_with(&format!("{prefix} {stream} ")))
                .count()
        };

        // The first step of the width captures each rank's target graph:
        // the feature packing's gather and the packed rows' gather into the
        // reserved rows, then the reserved head rows' gather before the
        // head, then replays it.
        let (step, hash, accepted) = device_step(
            &host,
            &mut instance,
            &request,
            (items, 3, 1),
            allocated_acceptance,
        );
        allocates(&step, hash);
        // The allocation's launches bind only their own rank's
        // allocations, as a device context requires (collectives bind
        // peer-mapped buffers; weight images are not allocations).
        let owners = host.allocation_ranks.borrow().clone();
        let own = [
            crate::sm120::allocate_entries::SYMBOL,
            allocated_acceptance,
            "chop_row_gather_by_index",
        ];
        for rank in 0..shape.ranks() {
            let prefix = format!("launch {} ", streams[&rank]);
            for line in step.iter().filter(|line| {
                line.starts_with(&prefix) && own.iter().any(|symbol| line.contains(symbol))
            }) {
                for slot in bound_buffers(line) {
                    if let Some(&(owner, _)) = owners.get(&slot) {
                        assert_eq!(owner, rank, "rank {rank} binds buffer {slot}: {line}");
                    }
                }
            }
        }
        let target = &step[hash..accepted];
        for rank in 0..shape.ranks() {
            let stream = streams[&rank];
            assert!(lines(target, "captured", stream) > 0, "rank {rank}");
            assert!(lines(target, "replay", stream) > 0, "rank {rank}");
            let prefix = format!("launch {stream} ");
            let head = target
                .iter()
                .position(|line| {
                    line.starts_with(&prefix) && line.contains("chop_deepseek41_head_f32")
                })
                .unwrap_or_else(|| panic!("rank {rank} captures its head"));
            let gathers = target[..head]
                .iter()
                .filter(|line| {
                    line.starts_with(&prefix) && line.contains("chop_row_gather_by_index")
                })
                .count();
            assert_eq!(gathers, 3, "rank {rank}");
        }

        // The same width over 24 reserved rows, whose tail is still 32
        // rows, replays the graph: the allocation and the device counts
        // carry the reservation.
        let request = allocated(items, 4, 2, width).encode().unwrap();
        let (step, hash, accepted) = device_step(
            &host,
            &mut instance,
            &request,
            (items, 2, 2),
            allocated_acceptance,
        );
        allocates(&step, hash);
        let target = &step[hash..accepted];
        for rank in 0..shape.ranks() {
            let stream = streams[&rank];
            assert_eq!(lines(target, "captured", stream), 0, "rank {rank}");
            assert!(lines(target, "replay", stream) > 0, "rank {rank}");
            assert!(
                !target
                    .iter()
                    .any(|line| line.starts_with(&format!("launch {stream} "))
                        && line.contains("chop_deepseek41_head_f32")),
                "rank {rank} replays its head"
            );
        }

        // 40 reserved rows take a 48-row tail, another bank: the step
        // captures its graph.
        let request = allocated(items, 5, 4, width).encode().unwrap();
        let (step, hash, accepted) = device_step(
            &host,
            &mut instance,
            &request,
            (items, 4, 3),
            allocated_acceptance,
        );
        allocates(&step, hash);
        let target = &step[hash..accepted];
        for rank in 0..shape.ranks() {
            let stream = streams[&rank];
            assert!(lines(target, "captured", stream) > 0, "rank {rank}");
        }

        // An unallocated step of the same graph key captures its own graph.
        let plain = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            (0..items)
                .map(|slot| BatchItemV1::verify(u64::from(slot + 1), 2, true, vec![7]))
                .collect(),
            (0..items)
                .map(|slot| BatchAppendV1::new(6, 2, 8, vec![u64::from(slot)], None))
                .collect(),
        )
        .with_state_slots((0..items).map(u64::from).collect())
        .encode()
        .unwrap();
        let (step, hash, accepted) = device_step(
            &host,
            &mut instance,
            &plain,
            (items, 1, 4),
            crate::sm120::host_acceptance_entries::SYMBOL,
        );
        assert!(launch_of(&step, streams[&0], allocated_acceptance).is_none());
        let target = &step[hash..accepted];
        for rank in 0..shape.ranks() {
            let stream = streams[&rank];
            assert!(launch_of(&step, stream, crate::sm120::allocate_entries::SYMBOL).is_none());
            assert!(lines(target, "captured", stream) > 0, "rank {rank}");
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn allocated_heads_route_by_reserved_rows_at_narrow_target_widths() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let wide = crate::sm120::f32_verify::entries()
        .unwrap()
        .into_iter()
        .find(|entry| entry.entry().query().n == shape.vocab_rows())
        .unwrap()
        .symbols();
    for (items, drafts, width, reserved) in [(8, 3, 16, 32), (8, 5, 16, 48), (16, 3, 32, 64)] {
        let (host, mut instance) = create_with_limits(shape, items, reserved);
        for slot in 0..items {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(u64::from(slot + 1), u64::from(slot), 0, 3, true),
            );
            instance.commit().unwrap();
        }
        let request = allocated(items, 3, drafts, width).encode().unwrap();
        let (step, hash, accepted) = device_step(
            &host,
            &mut instance,
            &request,
            (items, drafts, 1),
            crate::sm120::allocated_acceptance_entries::SYMBOL,
        );
        let target = &step[hash..accepted];
        for stream in host.rank_streams.borrow().values() {
            for symbol in &wide {
                assert_eq!(
                    launch_of(target, *stream, symbol).is_some(),
                    reserved > 32,
                    "width {width}, reserved {reserved}, stream {stream}, symbol {symbol}"
                );
            }
            let old_head = target
                .iter()
                .any(|line| line.starts_with(&format!("launch {stream} chop_deepseek41_head_f32")));
            assert_eq!(
                old_head,
                reserved <= 32,
                "width {width}, reserved {reserved}"
            );
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn allocated_verify_steps_are_refused_without_a_drafter_before_io() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = super::super::create(&host, 1);
    let before = host.journal.borrow().len();
    assert_eq!(
        instance
            .execute(
                &allocated(1, 0, 3, 3).encode().unwrap(),
                &mut Sink::default()
            )
            .err(),
        Some(BundleAbiError::Status(BundleStatusV1::Unsupported))
    );
    assert_eq!(host.journal.borrow().len(), before);
    execute(
        &mut instance,
        &host,
        &super::super::prompt(1, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// DSpark prepares the allocation and allocated acceptance, holds their
/// storage on every rank and loads their cubins. An AR instance holds none.
#[test]
fn allocation_ops_and_storage_are_dspark_only() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let symbols = [
        crate::sm120::allocate_entries::SYMBOL,
        crate::sm120::allocated_acceptance_entries::SYMBOL,
    ];
    for dspark in [false, true] {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.drafter.set(dspark);
        host.speculation.set(dspark.then_some((5, 32)));
        let record = crate::config::tests::record(&shape, host.kv, 4, false);
        let mut instance = api()
            .create(
                HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
                &record,
            )
            .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
        assert_eq!(
            crate::instance::dspark_allocation_ranks(instance.handle()),
            dspark.then_some(4)
        );
        // Create launches the allocation once on every rank, then waits.
        let streams = host.rank_streams.borrow().clone();
        for rank in 0..4 {
            let launches = host
                .journal
                .borrow()
                .iter()
                .filter(|line| {
                    line.starts_with(&format!(
                        "launch {} {} ",
                        streams[&rank],
                        crate::sm120::allocate_entries::SYMBOL
                    ))
                })
                .count();
            assert_eq!(launches, usize::from(dspark), "rank {rank}");
        }
        for symbol in symbols {
            let resolved = host
                .symbol_names
                .borrow()
                .values()
                .filter(|name| name.as_str() == symbol)
                .count();
            assert_eq!(resolved, if dspark { 4 } else { 0 }, "{symbol}");
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

struct MappedSink {
    inner: DeviceSink,
    mapped: Vec<runtime_bundle_abi::device_results::BundleMappedDeviceSamplingRequestV1>,
}
impl LogitsRowSinkV1 for MappedSink {
    fn device_results_enabled(&self) -> bool {
        true
    }
    fn mapped_sampling_enabled(&self) -> bool {
        true
    }
    fn limits(&self) -> ResultSinkLimitsV1 {
        self.inner.limits()
    }
    fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        self.inner.emit_row(row)
    }
    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.inner.emit_view(view)
    }
    fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        self.inner.emit_draft_tokens(drafts)
    }
    fn device_cycle_metadata(
        &self,
    ) -> HostServiceResultV1<runtime_bundle_abi::device_results::DeviceCycleMetadataV1> {
        self.inner.device_cycle_metadata()
    }
    fn declare_input_dependency(&mut self, dependency: u32) -> HostServiceResultV1<()> {
        self.inner.declare_input_dependency(dependency)
    }
    fn emit_device_drafts(
        &mut self,
        drafts: runtime_bundle_abi::device_results::BundleDeviceDraftsV1,
    ) -> HostServiceResultV1<()> {
        self.inner.emit_device_drafts(drafts)
    }
    fn sample_device_logits(
        &mut self,
        request: runtime_bundle_abi::device_results::BundleDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        self.inner.sample_device_logits(request)
    }
    fn sample_mapped_device_logits(
        &mut self,
        request: runtime_bundle_abi::device_results::BundleMappedDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        request.validate().unwrap();
        for view in [
            request.row_indices,
            request.source.logits.buffer,
            request.output,
        ] {
            self.inner.host.check_span(DeviceBufferSpanV1::new(
                view.buffer,
                view.offset,
                view.length,
            ));
        }
        let stream = BundleStreamHandleV1::from_parts(
            1,
            self.inner.host.rank_streams.borrow()[&request.source.logits.rank],
        );
        self.inner.host.record_event(request.done, stream)?;
        self.mapped.push(request);
        Ok(())
    }
    fn emit_device_completion(
        &mut self,
        packet: runtime_bundle_abi::device_results::BundleDeviceCompletionV1,
    ) -> HostServiceResultV1<()> {
        for request in &self.mapped {
            assert!(
                self.inner
                    .host
                    .journal
                    .borrow()
                    .iter()
                    .any(|line| line.starts_with("wait ")
                        && line.ends_with(&format!(" {}", request.done.slot())))
            );
        }
        self.inner.emit_device_completion(packet)
    }
}

#[test]
fn compact_allocation_scores_only_computed_rows_and_keeps_reserved_samples() {
    use model_deepseek41_contract::dspark_options::COMPACT_ALLOCATION;
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    for (items, width) in [(4, 16), (8, 32)] {
        let (host, mut instance) =
            create_with_options(shape, 32, 64, &[(COMPACT_ALLOCATION, true.into())]);
        for slot in 0..items {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(u64::from(slot + 1), u64::from(slot), 0, 3, true),
            );
            instance.commit().unwrap();
        }
        for cycle in 1..=2 {
            let mut sink = MappedSink {
                inner: DeviceSink {
                    host: host.clone(),
                    metadata: Some(metadata(items, 5, cycle)),
                    legacy: Sink::default(),
                    drafts: vec![],
                    samples: vec![],
                    completion: None,
                },
                mapped: vec![],
            };
            if cycle == 1 {
                let before = host.journal.borrow().len();
                assert_eq!(
                    instance
                        .execute(
                            &allocated(items, cycle + 2, 5, width).encode().unwrap(),
                            &mut sink.inner,
                        )
                        .err(),
                    Some(BundleAbiError::Status(BundleStatusV1::Unsupported)),
                );
                assert_eq!(host.journal.borrow().len(), before);
            }
            let start = host.journal.borrow().len();
            instance
                .execute(
                    &allocated(items, cycle + 2, 5, width).encode().unwrap(),
                    &mut sink,
                )
                .unwrap_or_else(|e| panic!("{e:?} {:?}", host.logs.borrow()));
            assert_eq!(sink.mapped.len(), 1);
            assert!(sink.inner.samples.is_empty());
            let mapped = sink.mapped[0];
            assert_eq!(mapped.source.logits.row_count, width);
            assert_eq!(mapped.sample_rows, items * 6);
            assert_eq!(
                mapped.source.logits.buffer.length,
                u64::from(width) * u64::from(shape.dims().vocab) * 4
            );
            assert_eq!(mapped.row_indices.length, u64::from(items * 6) * 4);
            assert_eq!(sink.inner.completion.unwrap().sample_rows, items * 6);
            assert!(sink.inner.drafts.iter().all(|d| d.computed_tokens == 5));
            let journal = host.journal.borrow();
            if cycle == 1 {
                let hash = launch_of(
                    &journal[start..],
                    host.rank_streams.borrow()[&0],
                    crate::sm120::engram_hash_rows::SYMBOL,
                )
                .unwrap();
                for stream in host.rank_streams.borrow().values() {
                    let head: Vec<_> = journal[start + hash..]
                        .iter()
                        .filter(|line| {
                            line.starts_with(&format!("launch {stream} chop_deepseek41_head_f32"))
                        })
                        .collect();
                    assert!(!head.is_empty());
                    assert!(
                        head.iter()
                            .all(|line| line.contains(&format!("ScalarU32({width})"))),
                        "{head:#?}"
                    );
                }
            }
            drop(journal);
            instance
                .commit_accepted(
                    &(1..=items)
                        .map(|id| super::kept(u64::from(id), 1))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

mod heads;
