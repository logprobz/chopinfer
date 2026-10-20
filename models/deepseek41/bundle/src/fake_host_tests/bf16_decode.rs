//! Compare decode commands for equal inputs, independent of prefill numerics.
use super::*;
use runtime_bundle_abi::device_results::*;

struct Ready {
    rows: u32,
    count: u32,
}
impl LogitsRowSinkV1 for Ready {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }
    fn emit_row(&mut self, _: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        panic!("decode must emit ready views")
    }
    fn device_results_enabled(&self) -> bool {
        true
    }
    fn device_cycle_metadata(&self) -> HostServiceResultV1<DeviceCycleMetadataV1> {
        Ok(DeviceCycleMetadataV1 {
            cycle_epoch: 9,
            termination_stop_ids: Vec::new(),
            items: (0..self.rows)
                .map(|row| DeviceCycleItemV1 {
                    request_id: u64::from(row) + 1,
                    slot_generation: 1,
                    acceptance: BundleAcceptanceInputV1 {
                        cycle_epoch: 9,
                        remaining_tokens: 8,
                        first_sample_row: row,
                        draft_item_index: BUNDLE_ACCEPTANCE_NO_DRAFT_V1,
                        draft_tokens: 0,
                        stop_offset: 0,
                        stop_count: 0,
                        reserved: 0,
                    },
                })
                .collect(),
        })
    }
    fn declare_input_dependency(&mut self, dependency: u32) -> HostServiceResultV1<()> {
        assert_eq!(dependency, DEVICE_RESULTS_HOST_VALUES_V1);
        Ok(())
    }
    fn emit_ready_logits_view(&mut self, view: BundleReadyLogitsViewV1) -> HostServiceResultV1<()> {
        assert_eq!(view.cycle_epoch, 9);
        self.count += view.logits.row_count;
        Ok(())
    }
}

// Only opaque handles map. Allocation order and stream/event creation order
// establish identity before the compared operations. Scalars, launch grids,
// offsets, lengths, upload digests and every operation's order stay literal.
type Commands = (Vec<String>, BTreeMap<usize, (u32, u64)>);

fn commands(host: &Host, from: usize) -> Commands {
    let journal = host.device.journal.borrow();
    let buffers = host.device.buffer_order.borrow();
    let allocations = host.device.allocation_ranks.borrow();
    let mut streams = BTreeMap::new();
    let mut events = BTreeMap::new();
    let mut graphs = BTreeMap::new();
    for line in journal.iter() {
        let words = line.split_whitespace().collect::<Vec<_>>();
        match words[0] {
            "stream" => {
                streams.insert(words[1].to_owned(), streams.len());
            }
            "event" => {
                events.insert(words[1].to_owned(), events.len());
            }
            "captured" => {
                graphs.insert(words[2].to_owned(), graphs.len());
            }
            _ => {}
        }
    }
    let mut used = BTreeMap::new();
    let lines = journal[from..]
        .iter()
        .map(|line| {
            let mut parts = line.split(' ').map(str::to_owned).collect::<Vec<_>>();
            match parts[0].as_str() {
                "launch" | "upload" | "copy" | "readback" | "download" | "zero" | "synchronize"
                | "capture" => {
                    parts[1] = format!("stream{}", streams[&parts[1]]);
                }
                "record" | "wait" => {
                    parts[1] = format!("stream{}", streams[&parts[1]]);
                    parts[2] = format!("event{}", events[&parts[2]]);
                }
                "captured" | "replay" => {
                    parts[1] = format!("stream{}", streams[&parts[1]]);
                    parts[2] = format!("graph{}", graphs[&parts[2]]);
                }
                "event" => {
                    parts[1] = format!("event{}", events[&parts[1]]);
                }
                other => panic!("unmodelled decode operation {other}: {line}"),
            }
            let line = parts.join(" ");
            let mut out = String::new();
            let marker = "BundleDeviceBufferHandleV1 { raw: ";
            let mut rest = line.as_str();
            while let Some((before, after)) = rest.split_once(marker) {
                out.push_str(before);
                out.push_str(marker);
                let (raw, tail) = after.split_once(' ').unwrap();
                let raw: u64 = raw.parse().unwrap();
                let slot = raw as u32;
                if let Some(index) = buffers.iter().position(|&b| b == slot) {
                    used.insert(index, allocations[&slot]);
                    out.push_str(&format!("buffer{index}"));
                } else {
                    // Package-owned images, KV and request state have fixed IDs.
                    assert!(slot >= IMAGE_SLOT);
                    out.push_str(&raw.to_string());
                }
                out.push(' ');
                rest = tail;
            }
            out.push_str(rest);
            out
        })
        .collect();
    (lines, used)
}

#[test]
fn prefill_only_partials_preserve_decode_capture_and_replay_commands() {
    use bundle_runtime::entry::Bundle;
    for count in [8, 32] {
        let mut traces = Vec::new();
        for mode in ["0", "prefill"] {
            let host = super::bf16_partials::host(true);
            let mut instance = crate::device_results::with_override(None, || {
                crate::ops::with_partial_setting(Some(mode), || {
                    crate::ops::with_bf16_gather(true, || {
                        let record =
                            crate::config::tests::record(&host.shape, host.kv, count, false);
                        crate::instance::PARTIAL.with(|partial| partial.set(true));
                        let result = api()
                            .with_device_results_api(
                                crate::api::BUNDLE_DEEPSEEK41_DEVICE_RESULTS_API_V1,
                            )
                            .unwrap()
                            .create(HostServicesTableV1::new(host.clone()), &record);
                        crate::instance::PARTIAL.with(|partial| partial.set(false));
                        result.unwrap()
                    })
                })
            });
            for slot in 0..u64::from(count) {
                step(
                    &mut instance,
                    &prompt(slot + 1, slot, 0, 3, true),
                    &mut Views::default(),
                );
            }
            let mut phases = Vec::new();
            for repeat in 0..2 {
                // Even a due prefill reset must not change decode commands.
                <crate::instance::Instance as Bundle>::instances().with(|table| {
                    table
                        .borrow_mut()
                        .get_mut(instance.handle())
                        .unwrap()
                        .reset_prefill_reduce_next_run();
                });
                let from = host.device.journal.borrow().len();
                let entries = (0..u64::from(count))
                    .map(|slot| (slot + 1, slot, 3 + repeat, 7))
                    .collect::<Vec<_>>();
                let mut sink = Ready {
                    rows: count,
                    count: 0,
                };
                instance.execute(&decode(&entries), &mut sink).unwrap();
                assert_eq!(sink.count, count);
                instance.commit().unwrap();
                let trace = commands(&host, from);
                assert_eq!(
                    trace
                        .0
                        .iter()
                        .filter(|s| s.starts_with("captured "))
                        .count(),
                    if repeat == 0 { 12 } else { 0 }
                );
                assert_eq!(
                    trace.0.iter().filter(|s| s.starts_with("replay ")).count(),
                    12
                );
                phases.push(trace);
            }
            traces.push(phases);
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
        for (phase, (control, candidate)) in traces[0].iter().zip(&traces[1]).enumerate() {
            assert_eq!(
                control.1, candidate.1,
                "decode allocation roles differ at C{count}"
            );
            assert_eq!(
                control.0.len(),
                candidate.0.len(),
                "decode command count differs at C{count}"
            );
            println!(
                "decode-command-audit C{count} phase={phase}: {} operations, {} launches, {} replays",
                control.0.len(),
                control
                    .0
                    .iter()
                    .filter(|s| s.starts_with("launch "))
                    .count(),
                control
                    .0
                    .iter()
                    .filter(|s| s.starts_with("replay "))
                    .count()
            );
            for (line, (a, b)) in control.0.iter().zip(&candidate.0).enumerate() {
                assert_eq!(a, b, "decode C{count} command {line}");
            }
        }
    }
}
