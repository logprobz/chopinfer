//! Queue a private successor while the host validates the current acceptance.
use super::Instance;
#[cfg(test)]
use crate::dspark::device::DECODE_BASE;
use crate::dspark::device::slice;
use crate::dspark::execution::{DraftInput, Input};
use crate::dspark::state::{BLOCK_ROWS, DraftBlock, Kind};
#[cfg(test)]
use crate::dspark::transfers;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

impl Instance {
    fn prefetch_copy(&self, rank: usize, destination: Span, source: Span) -> Result<(), String> {
        if destination.length != source.length || destination.buffer == source.buffer {
            return Err("DSpark successor copy aliases or changes length".into());
        }
        self.services
            .copy_device_to_device(self.ranks[rank].2, destination, source)
            .map_err(|e| e.to_string())
    }

    /// Output descriptors already handed to the host keep their original
    /// allocations. Only the private successor graph binds these other views.
    fn swap_prefetch_outputs(&mut self) {
        let device = self.dspark.as_mut().expect("drafter");
        for (rank, memory) in device.ranks.iter_mut().zip(&mut device.prefetch.ranks) {
            std::mem::swap(&mut rank.draft_ids, &mut memory.draft_ids);
            std::mem::swap(&mut rank.confidence, &mut memory.confidence);
            std::mem::swap(&mut rank.status, &mut memory.status);
        }
        std::mem::swap(&mut self.draft_graphs, &mut device.prefetch.graphs);
        std::mem::swap(
            &mut device.computed_positions,
            &mut device.prefetch.computed_positions,
        );
    }

    fn prefetch_join(&self) -> Result<(), String> {
        let device = self.dspark.as_ref().expect("drafter");
        let joined = device
            .prefetch
            .joined
            .ok_or("DSpark successor join absent")?;
        for (&event, &(_, _, stream)) in device.prefetch.done.iter().zip(&self.ranks) {
            self.services
                .record_event(event, stream)
                .map_err(|e| e.to_string())?;
        }
        for &event in device.prefetch.done.iter().skip(1) {
            self.services
                .wait_event(self.ranks[0].2, event)
                .map_err(|e| e.to_string())?;
        }
        self.services
            .record_event(joined, self.ranks[0].2)
            .map_err(|e| e.to_string())?;
        for &(_, _, stream) in self.ranks.iter().skip(1) {
            self.services
                .wait_event(stream, joined)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub(super) fn queue_dspark_prefetch(&mut self) -> Result<(), String> {
        let device = self.dspark.as_mut().ok_or("DSpark absent")?;
        let Some(feedback) = device.prefetch.feedback.take() else {
            return Ok(());
        };
        let allocated = feedback.allocated;
        let state = device
            .runtime
            .as_ref()
            .ok_or("DSpark lifecycle borrowed")?
            .state();
        let Some(mut plan) = feedback.plan(
            state,
            device.shape.geometry.shape.vocab,
            device.prefetch.successor.items,
        )?
        else {
            return Ok(());
        };
        if device.prefetch.ranks.len() != self.ranks.len() || device.prefetch.plan.is_some() {
            return Err("DSpark successor storage or lifetime differs".into());
        }
        let started = std::time::Instant::now();
        let mut rings = Vec::new();
        // Bind every backup before temporarily touching committed buffers.
        for (rank, buffers) in self.buffers.iter().enumerate() {
            let memory = &device.prefetch.ranks[rank];
            let ring = buffers
                .state
                .named("dspark.ring")
                .ok_or("DSpark ring absent")?;
            if ring.length != memory.undo_ring.length || ring.length != memory.next_ring.length {
                return Err("DSpark successor ring capacity differs".into());
            }
            rings.push((rank, ring, memory.undo_ring, memory.next_ring));
            memory.status.discard();
        }
        for &(rank, ring, undo, _) in &rings {
            self.prefetch_copy(rank, undo, ring)?;
        }
        self.swap_prefetch_outputs();
        let queued = self
            .project_successor(&plan)
            .and_then(|()| self.dspark_draft(&plan.drafts, allocated));
        self.swap_prefetch_outputs();
        queued?;
        // No later rank may reuse collective scratch while another rank is
        // still reading it. Every restore and successor starts behind this join.
        self.prefetch_join()?;
        for &(rank, ring, undo, next) in &rings {
            self.prefetch_copy(rank, next, ring)?;
            self.prefetch_copy(rank, ring, undo)?;
        }
        let items = plan.drafts.len();
        let prefetch = &mut self.dspark.as_mut().expect("drafter").prefetch;
        plan.ring_writes = prefetch.ring_writes.get();
        prefetch.plan = Some(plan);
        self.prefetch_trace("queued", items, started.elapsed().as_nanos(), "");
        Ok(())
    }

    /// Zero every item's draft IDs and confidence past `depth` on every
    /// rank, as an inline draft of `depth` heads leaves them.
    fn cut_successor_heads(&self, items: usize, depth: u32) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark absent")?;
        let tail = u64::from(BLOCK_ROWS - depth) * 4;
        for (rank, stored) in device.ranks.iter().enumerate() {
            for output in [stored.draft_ids, stored.confidence] {
                for item in 0..items as u64 {
                    let at = (item * u64::from(BLOCK_ROWS) + u64::from(depth)) * 4;
                    self.services
                        .zero_device_buffer(self.ranks[rank].2, slice(output, at, tail)?)
                        .map_err(|e| format!("DSpark successor head cut: {e:?}"))?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn consume_dspark_prefetch(
        &mut self,
        inputs: &[Input],
        allocated: bool,
    ) -> Result<(), String> {
        let logs = self.config.execution.diagnostic_step_logs;
        let device = self.dspark.as_mut().ok_or("DSpark absent")?;
        device.prefetch.feedback = None;
        device.prefetch.reused = false;
        let Some(plan) = device.prefetch.plan.take() else {
            return Ok(());
        };
        let prefill = device.prefetch.successor.prefill;
        let restored: Vec<_> = inputs
            .iter()
            .filter(|input| matches!(input.rows.kind, Kind::Prefill { .. }))
            .map(|input| input.rows.request)
            .collect();
        let runtime = device.runtime.as_mut().ok_or("DSpark lifecycle borrowed")?;
        let mut probe = runtime.state().clone();
        let required_positions = if allocated {
            crate::dspark::state::BLOCK_ROWS
        } else {
            inputs
                .iter()
                .map(|input| input.rows.kind.drafts())
                .max()
                .unwrap_or(0)
        };
        let covered = device.prefetch.computed_positions >= required_positions
            && probe
                .prepare(inputs.iter().map(|input| input.rows).collect())
                .is_ok();
        // Beside prefill items the state may differ by restores, so the
        // committed ring must also be as the queue left it: the whole-ring
        // copy below rewrites every slot.
        let relaxed = || {
            plan.matches_with_prefill(runtime.state(), inputs)
                && plan.ring_writes == device.prefetch.ring_writes.get()
        };
        let matched = covered
            && if prefill {
                relaxed()
            } else {
                plan.matches(runtime.state(), inputs)
            };
        if !matched {
            // Without the option, whether its rule would have served the step.
            let would = logs && !prefill && covered && relaxed();
            let extra = format!(",\"prefill_items\":{},\"relaxed\":{would}", restored.len());
            self.prefetch_trace("discarded", plan.drafts.len(), 0, &extra);
            return Ok(());
        }
        let started = std::time::Instant::now();
        let bytes = plan.drafts.len() as u64 * 5 * 4;
        let mut copies = Vec::new();
        for (rank, buffers) in self.buffers.iter().enumerate() {
            let memory = &device.prefetch.ranks[rank];
            copies.push((
                rank,
                buffers
                    .state
                    .named("dspark.ring")
                    .ok_or("DSpark ring absent")?,
                memory.next_ring,
            ));
            for (destination, source) in [
                (device.ranks[rank].draft_ids, memory.draft_ids),
                (device.ranks[rank].confidence, memory.confidence),
            ] {
                copies.push((
                    rank,
                    slice(destination, 0, bytes)?,
                    slice(source, 0, bytes)?,
                ));
            }
        }
        for (rank, destination, source) in copies {
            self.prefetch_copy(rank, destination, source)?;
        }
        let mut computed = self
            .dspark
            .as_ref()
            .expect("drafter")
            .prefetch
            .computed_positions;
        if prefill {
            // The heads the step's own draft would compute: a five-head
            // successor serves a compact three-head step cut to its heads.
            let drafts: Vec<_> = inputs
                .iter()
                .filter(|input| input.rows.kind.drafts() > 0)
                .map(|input| DraftInput {
                    block: DraftBlock {
                        request: input.rows.request,
                        slot: input.rows.slot,
                        main_rows: input.rows.start,
                        compute_rows: BLOCK_ROWS,
                        emit_rows: input.rows.kind.drafts(),
                        verify_rows: input.rows.kind.drafts() + 1,
                    },
                    bonus: input.tokens[0],
                    noise: input.noise,
                })
                .collect();
            let depth = self.draft_positions(&drafts, allocated)?;
            if depth < computed {
                self.cut_successor_heads(drafts.len(), depth)?;
                computed = depth;
            }
        }
        let device = self.dspark.as_mut().expect("drafter");
        for (rank, memory) in device.prefetch.ranks.iter().enumerate() {
            device.ranks[rank]
                .status
                .append(&self.services, self.ranks[rank].2, &memory.status)?;
        }
        device
            .runtime
            .as_mut()
            .expect("lifecycle")
            .adopt_projections(
                &plan.committed,
                prefill.then_some(restored.as_slice()),
                &plan.projections,
            )?;
        device.computed_positions = computed;
        device.prefetch.reused = true;
        let extra = format!(",\"prefill_items\":{},\"heads\":{computed}", restored.len());
        self.prefetch_trace(
            "reused",
            plan.drafts.len(),
            started.elapsed().as_nanos(),
            &extra,
        );
        Ok(())
    }

    /// A diagnostic record of the successor: `extra` adds fields after
    /// the host time.
    fn prefetch_trace(&self, event: &str, items: usize, nanos: u128, extra: &str) {
        if self.config.execution.diagnostic_step_logs {
            let _ = self.services.log(
                format!(
                    "{{\"schema\":\"chopinfer.deepseek41.step-shape.v1\",\"phase\":\"dspark-prefetch\",\"event\":\"{event}\",\"items\":{items},\"host_ns\":{nanos}{extra}}}"
                ).as_bytes(),
            );
        }
    }
}

#[cfg(test)]
pub(crate) fn for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
    feedback: crate::dspark::prefetch::Feedback,
    queue: bool,
) -> (crate::dspark::state::State, Vec<Span>) {
    crate::instance::INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("instance");
        let device = instance.dspark.as_ref().expect("drafter");
        let state = device.runtime.as_ref().expect("lifecycle").state();
        let plan = feedback
            .plan(
                state,
                device.shape.geometry.shape.vocab,
                device.prefetch.successor.items,
            )
            .unwrap()
            .expect("predictable test");
        let mut spans = Vec::new();
        for (rank, buffers) in instance.buffers.iter().enumerate() {
            spans.push(buffers.state.named("dspark.ring").unwrap());
            spans.push(device.ranks[rank].draft_ids);
            spans.push(device.ranks[rank].confidence);
            let features = device
                .features
                .bind(
                    buffers.state.named("dspark.features").unwrap(),
                    device.ranks[rank].packed,
                )
                .unwrap();
            let mut offset = u64::from(DECODE_BASE);
            for item in plan.retain.items() {
                spans.extend(
                    features
                        .retain(&transfers::retain_copies(device.config, *item, offset).unwrap())
                        .unwrap()
                        .into_iter()
                        .map(|copy| copy.destination),
                );
                offset += u64::from(item.rows.kind.target_rows().unwrap());
            }
        }
        if queue {
            instance.dspark.as_mut().unwrap().prefetch.feedback = Some(feedback);
            instance.queue_dspark_prefetch().unwrap();
        }
        (
            instance
                .dspark
                .as_ref()
                .unwrap()
                .runtime
                .as_ref()
                .unwrap()
                .state()
                .clone(),
            spans,
        )
    })
}

/// The drafting items of the successor the instance holds, if any.
#[cfg(test)]
pub(crate) fn queued_for_test(handle: runtime_bundle_abi::BundleHandleV1) -> Option<usize> {
    crate::instance::INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let device = table.get_mut(handle)?.dspark.as_ref()?;
        device.prefetch.plan.as_ref().map(|plan| plan.drafts.len())
    })
}

/// Each rank's committed ring and the successor's undo and next rings.
#[cfg(test)]
pub(crate) fn rings_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Vec<(Span, Span, Span)> {
    crate::instance::INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("instance");
        let device = instance.dspark.as_ref().expect("drafter");
        instance
            .buffers
            .iter()
            .zip(&device.prefetch.ranks)
            .map(|(buffers, memory)| {
                (
                    buffers.state.named("dspark.ring").expect("ring"),
                    memory.undo_ring,
                    memory.next_ring,
                )
            })
            .collect()
    })
}
