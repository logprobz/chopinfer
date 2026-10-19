//! A submitted execution keeps the instance and result consumers live until
//! collection or cancellation. The guard exposes no commit before collection.
use super::*;
use runtime_engine::AsyncInputDependency;

enum Emissions<'a> {
    Ready(BatchEmissionsV1),
    Submitted(Box<CollectingSinkV1<'a>>),
}

/// One submitted bundle execution, not yet collected or committed.
///
/// The exclusive instance borrow prevents another execution, retirement or
/// prefix snapshot while its result buffers remain borrowed. A successor
/// needs a separate provisional-state contract; DeviceFeedback alone does
/// not permit a second call to the existing execute entry.
#[must_use = "collect the execution or cancel it before reusing the instance"]
pub struct PendingBundleExecutionV1<'a, E: BundleExecutionV1> {
    instance: Option<&'a mut E>,
    emissions: Option<Emissions<'a>>,
    poison: Option<fn(&mut E)>,
}

impl<'a, E: BundleExecutionV1> PendingBundleExecutionV1<'a, E> {
    pub(super) fn ready(instance: &'a mut E, rows: BatchEmissionsV1) -> Self {
        Self {
            instance: Some(instance),
            emissions: Some(Emissions::Ready(rows)),
            poison: None,
        }
    }

    pub(super) fn submitted(
        instance: &'a mut E,
        sink: CollectingSinkV1<'a>,
        poison: fn(&mut E),
    ) -> Self {
        Self {
            instance: Some(instance),
            emissions: Some(Emissions::Submitted(Box::new(sink))),
            poison: Some(poison),
        }
    }

    #[must_use]
    pub fn input_dependency(&self) -> AsyncInputDependency {
        match &self.emissions {
            Some(Emissions::Submitted(sink)) => sink
                .device
                .as_ref()
                .map_or(AsyncInputDependency::HostValues, DeviceResults::dependency),
            _ => AsyncInputDependency::HostValues,
        }
    }

    #[must_use]
    pub fn cycle_epoch(&self) -> Option<u64> {
        match &self.emissions {
            Some(Emissions::Submitted(sink)) => sink.device.as_ref().map(DeviceResults::epoch),
            _ => None,
        }
    }

    /// Waits for all result records and leaves the model transaction pending.
    /// The ordinary engine validation and commit still decide publication.
    pub fn collect(self) -> Result<BatchEmissionsV1, BundleAbiError> {
        self.collect_parts().map(|(_, rows)| rows)
    }

    pub(super) fn collect_parts(mut self) -> Result<(&'a mut E, BatchEmissionsV1), BundleAbiError> {
        let _phase = matches!(self.emissions, Some(Emissions::Submitted(_)))
            .then(|| PhaseTimer::traced(StepPhase::ResultCollect));
        if let Some(Emissions::Submitted(sink)) = self.emissions.as_mut() {
            let rows = match sink.finish_emissions() {
                Ok(rows) => rows,
                Err(status) => {
                    self.cancel_inner()?;
                    return Err(BundleAbiError::Status(status));
                }
            };
            self.emissions = Some(Emissions::Ready(rows));
        }
        let Some(Emissions::Ready(rows)) = self.emissions.take() else {
            unreachable!("submission retained");
        };
        Ok((self.instance.take().expect("instance retained"), rows))
    }

    /// Discards every result after its consumers have stopped, then cancels
    /// the bundle. No row from this execution becomes commit authority.
    pub fn cancel(mut self) -> Result<(), BundleAbiError> {
        self.cancel_inner()
    }

    fn cancel_inner(&mut self) -> Result<(), BundleAbiError> {
        let Some(instance) = self.instance.take() else {
            return Ok(());
        };
        if let Some(Emissions::Submitted(sink)) = &mut self.emissions {
            // A legacy view uses the existing collection fence. Device result
            // cancellation drains private consumers, including partial enqueues.
            let legacy = sink.collect_pending();
            if let Err(error) = sink.discard_device() {
                if let Some(poison) = self.poison {
                    poison(instance);
                }
                return Err(BundleAbiError::Status(error));
            }
            // Legacy failures have the same cancel contract as synchronous execute.
            if let Err(error) = legacy {
                instance.cancel()?;
                return Err(BundleAbiError::Status(error));
            }
        }
        self.emissions = None;
        instance.cancel()
    }
}

impl<E: BundleExecutionV1> Drop for PendingBundleExecutionV1<'_, E> {
    fn drop(&mut self) {
        let _ = self.cancel_inner();
    }
}

impl CollectingSinkV1<'_> {
    pub(super) fn finish_emissions(&mut self) -> HostServiceResultV1<BatchEmissionsV1> {
        self.collect_pending()?;
        let accepted_prefixes = self.collect_device()?;
        Ok(BatchEmissionsV1 {
            accepted_prefixes,
            rows: std::mem::take(&mut self.rows),
            sampled: std::mem::take(&mut self.sampled),
            prompt_view_rows: std::mem::take(&mut self.prompt_view_rows),
            drafts: std::mem::take(&mut self.drafts)
                .into_iter()
                .map(|(id, tokens)| OwnedDraftTokensV1 {
                    request_id: id.get(),
                    tokens,
                })
                .collect(),
        })
    }
}
