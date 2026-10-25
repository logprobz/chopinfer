//! Queue a private successor while the host validates the current acceptance.
use super::Instance;
#[cfg(test)]
use crate::dspark::device::DECODE_BASE;
use crate::dspark::device::slice;
use crate::dspark::execution::Input;
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
        if !device.prefetch.enabled {
            return Ok(());
        }
        let state = device
            .runtime
            .as_ref()
            .ok_or("DSpark lifecycle borrowed")?
            .state();
        let Some(plan) = feedback.plan(state, device.shape.geometry.shape.vocab)? else {
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
            .and_then(|()| self.dspark_draft(&plan.drafts));
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
        self.dspark.as_mut().expect("drafter").prefetch.plan = Some(plan);
        self.prefetch_trace("queued", items, started.elapsed().as_nanos());
        Ok(())
    }

    pub(super) fn consume_dspark_prefetch(&mut self, inputs: &[Input]) -> Result<(), String> {
        let device = self.dspark.as_mut().ok_or("DSpark absent")?;
        device.prefetch.feedback = None;
        device.prefetch.reused = false;
        let Some(plan) = device.prefetch.plan.take() else {
            return Ok(());
        };
        let runtime = device.runtime.as_mut().ok_or("DSpark lifecycle borrowed")?;
        let mut probe = runtime.state().clone();
        if !plan.matches(runtime.state(), inputs)
            || probe
                .prepare(inputs.iter().map(|input| input.rows).collect())
                .is_err()
        {
            self.prefetch_trace("discarded", plan.drafts.len(), 0);
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
            .adopt_projections(&plan.committed, &plan.projections)?;
        device.computed_positions = device.prefetch.computed_positions;
        device.prefetch.reused = true;
        self.prefetch_trace("reused", plan.drafts.len(), started.elapsed().as_nanos());
        Ok(())
    }

    fn prefetch_trace(&self, event: &str, items: usize, nanos: u128) {
        if self.dspark.as_ref().is_some_and(|device| device.step_trace) {
            let _ = self.services.log(
                format!(
                    "{{\"schema\":\"chopinfer.deepseek41.step-shape.v1\",\"phase\":\"dspark-prefetch\",\"event\":\"{event}\",\"items\":{items},\"host_ns\":{nanos}}}"
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
            .plan(state, device.shape.geometry.shape.vocab)
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
