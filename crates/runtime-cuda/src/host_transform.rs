//! Bounded pinned staging for an event-ordered device/host/device transform.
use super::*;

/// A range uploaded from one output staging allocation after host work.
pub struct HostTransformTarget {
    pub buffer: DeviceBuffer,
    pub offset: usize,
    pub output: usize,
    pub source_offset: usize,
    pub length: usize,
}

#[derive(Debug)]
pub struct HostTransformStats {
    pub host_nanoseconds: u64,
    pub exposed_milliseconds: Vec<f32>,
    pub failure: Option<String>,
}

struct Output {
    pinned: PinnedHostBuffer,
    stream: Stream,
    owner: Stream,
    owner_ready: Event,
    ready: Event,
    before_wait: Event,
    after_wait: Event,
    waited: bool,
}

struct Pending {
    submitted: bool,
    result: ffi::host_work::HostResult,
    _buffers: Vec<DeviceBuffer>,
}

/// One reusable transfer slot. Its private pinned buffers cannot be reused
/// before completion; a failed enqueue also retains every in-flight allocation.
pub struct HostTransform {
    input: PinnedHostBuffer,
    stream: Stream,
    produced: Event,
    transformed: Event,
    outputs: Vec<Output>,
    pending: Option<Pending>,
}

impl HostTransform {
    pub fn new(
        input: &Context,
        input_capacity: usize,
        outputs: &[(&Stream, usize)],
    ) -> Result<Self, CudaError> {
        if outputs.is_empty() {
            return Err(CudaError::InvalidLaunch);
        }
        Ok(Self {
            input: PinnedHostBuffer::new(input, input_capacity)?,
            stream: input.stream()?,
            produced: input.event()?,
            transformed: input.event()?,
            outputs: outputs
                .iter()
                .map(|&(owner, capacity)| {
                    let context = owner.context();
                    Ok(Output {
                        pinned: PinnedHostBuffer::new(&context, capacity)?,
                        stream: context.stream()?,
                        owner: owner.clone(),
                        owner_ready: context.event()?,
                        ready: context.event()?,
                        before_wait: context.timed_event()?,
                        after_wait: context.timed_event()?,
                        waited: false,
                    })
                })
                .collect::<Result<_, CudaError>>()?,
            pending: None,
        })
    }

    pub fn pinned_bytes(&self) -> usize {
        self.input.len()
            + self
                .outputs
                .iter()
                .map(|output| output.pinned.len())
                .sum::<usize>()
    }

    /// The callback runs on CUDA's host worker after D2H completes. It must
    /// perform only CPU work, never call CUDA, and may report an error for the
    /// owner to check before publishing results. Uploads still complete after
    /// an error, so all output bytes are initialized before the callback runs.
    pub fn enqueue(
        &mut self,
        source: &DeviceBuffer,
        offset: usize,
        bytes: usize,
        producer: &Stream,
        targets: &[HostTransformTarget],
        call: impl FnOnce(&[u8], &mut [&mut [u8]]) -> Result<(), String> + Send + 'static,
    ) -> Result<(), CudaError> {
        if self.pending.is_some() {
            return Err(CudaError::ResourceShared("pending host transform"));
        }
        source.require_live()?;
        producer.require_live()?;
        require_same_context(&source.stream.context, &self.stream.inner.context)?;
        require_same_context(&source.stream.context, &producer.inner.context)?;
        checked_host_range(self.input.len(), 0, bytes)?;
        let pointer = source.pointer_at(offset, bytes)?;
        if bytes == 0 || targets.is_empty() {
            return Err(CudaError::InvalidLaunch);
        }
        let mut lengths = vec![0; self.outputs.len()];
        for target in targets {
            let output = self
                .outputs
                .get(target.output)
                .ok_or(CudaError::InvalidLaunch)?;
            target.buffer.require_live()?;
            target.buffer.pointer_at(target.offset, target.length)?;
            require_same_context(&target.buffer.stream.context, &output.stream.inner.context)?;
            if !Rc::ptr_eq(&target.buffer.stream, &output.owner.inner) {
                return Err(CudaError::InvalidLaunch);
            }
            checked_host_range(output.pinned.len(), target.source_offset, target.length)?;
            lengths[target.output] =
                lengths[target.output].max(target.source_offset + target.length);
        }
        if lengths.contains(&0) {
            return Err(CudaError::InvalidLaunch);
        }
        for output in &mut self.outputs {
            output.waited = false;
        }
        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        self.pending = Some(Pending {
            submitted: false,
            result,
            _buffers: std::iter::once(source.clone())
                .chain(targets.iter().map(|target| target.buffer.clone()))
                .collect(),
        });
        for output in &self.outputs {
            output.owner_ready.record(&output.owner)?;
            output.stream.wait_event(&output.owner_ready)?;
        }
        self.produced.record(producer)?;
        self.stream.wait_event(&self.produced)?;
        set_current_inner(&self.stream.inner.context)?;
        checked(
            "cuMemcpyDtoHAsync",
            ffi::copy_device_to_pinned(
                &mut self.input.allocation,
                0,
                pointer,
                bytes,
                self.stream.inner.handle,
            ),
        )?;
        let outputs = self
            .outputs
            .iter()
            .zip(&lengths)
            .map(|(output, &length)| (&output.pinned.allocation, length))
            .collect::<Vec<_>>();
        let (code, result) = ffi::host_work::enqueue(
            self.stream.inner.handle,
            &self.input.allocation,
            bytes,
            &outputs,
            Box::new(call),
        );
        self.pending.as_mut().expect("retained transfers").result = result;
        checked("cuLaunchHostFunc", code)?;
        self.transformed.record(&self.stream)?;
        for (index, output) in self.outputs.iter().enumerate() {
            output.stream.wait_event(&self.transformed)?;
            set_current_inner(&output.stream.inner.context)?;
            for target in targets.iter().filter(|target| target.output == index) {
                checked(
                    "cuMemcpyHtoDAsync",
                    ffi::copy_pinned_to_device(
                        target.buffer.pointer_at(target.offset, target.length)?,
                        &output.pinned.allocation,
                        target.source_offset,
                        target.length,
                        output.stream.inner.handle,
                    ),
                )?;
            }
            output.ready.record(&output.stream)?;
        }
        self.pending.as_mut().expect("retained transfers").submitted = true;
        Ok(())
    }

    /// Wait on the consumer stream immediately before its first use. Timing
    /// measures only the dependency exposed after preceding device work.
    pub fn wait_on(&mut self, output: usize, consumer: &Stream) -> Result<(), CudaError> {
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| !pending.submitted)
        {
            return Err(CudaError::InvalidLaunch);
        }
        let output = self
            .outputs
            .get_mut(output)
            .ok_or(CudaError::InvalidLaunch)?;
        require_same_context(&output.stream.inner.context, &consumer.inner.context)?;
        if output.waited {
            return Err(CudaError::InvalidLaunch);
        }
        output.before_wait.record(consumer)?;
        consumer.wait_event(&output.ready)?;
        output.after_wait.record(consumer)?;
        output.waited = true;
        Ok(())
    }

    /// A nonblocking completion check. Call after the consumer's ordinary
    /// result boundary; pending work cannot silently become a successful result.
    pub fn complete(&mut self) -> Result<Option<HostTransformStats>, CudaError> {
        let Some(pending) = &self.pending else {
            return Err(CudaError::InvalidLaunch);
        };
        if !pending.submitted {
            return Err(CudaError::InvalidLaunch);
        }
        if !self.transformed.query()? {
            return Ok(None);
        }
        for output in &self.outputs {
            if !output.ready.query()? || (output.waited && !output.after_wait.query()?) {
                return Ok(None);
            }
        }
        let (host_nanoseconds, result) = pending
            .result
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
            .ok_or(CudaError::InvalidLaunch)?;
        let exposed_milliseconds = self
            .outputs
            .iter()
            .map(|output| {
                if output.waited {
                    output.before_wait.elapsed_ms(&output.after_wait)
                } else {
                    Ok(0.0)
                }
            })
            .collect::<Result<_, _>>()?;
        self.pending = None;
        Ok(Some(HostTransformStats {
            host_nanoseconds,
            exposed_milliseconds,
            failure: result.err(),
        }))
    }
}

impl Drop for HostTransform {
    fn drop(&mut self) {
        if self.pending.is_none() {
            return;
        }
        // This is cleanup of interrupted/failed work, never the normal cycle.
        // Unconfirmed DMA cannot outlive its pinned storage.
        if self.stream.synchronize().is_err() {
            std::process::abort();
        }
        for output in &self.outputs {
            if output.stream.synchronize().is_err() {
                std::process::abort();
            }
        }
    }
}

#[cfg(all(test, feature = "fake-driver-tests"))]
mod tests;
