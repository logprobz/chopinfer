//! A mixed step's verify-row Engram (`Instance::admits_mixed_engram`). The
//! host hashes the verify rows as for the synchronous gather; the cycle's
//! prepared layer gathers copy them once the step's prefill joined its
//! last Engram layer, so the host gathers them while the rest of the
//! prefill runs. The verify launch joins each layer before its consumer,
//! as an undrafted decode step's does. The ids, the rows' destinations and
//! the launches are the synchronous gather's.
use super::Instance;
use crate::dspark::cycle::{Deferred, Mixed};
use crate::engram_rows::{Hashed, PreparedGather};
use crate::instance::ar_engram;
use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::resources::upload_bytes;
use runtime_bundle_abi::{BundleAbiError, BundleStatusV1};

impl Instance {
    /// Takes a mixed step's verify-row ids where `admits_mixed_engram`
    /// admits the step (`media`: the step encodes media), for its prefill
    /// to gather after its last Engram join (`begin_mixed_dspark_engram`).
    /// False keeps the synchronous gather.
    pub(crate) fn defer_mixed_dspark_engram(
        &mut self,
        prefill: &PhaseInputs,
        verify: &PhaseInputs,
        media: bool,
        hashed: &Hashed,
    ) -> Result<bool, String> {
        if !self.admits_mixed_engram(prefill, verify, media)
            || verify.live_tokens() as usize != hashed.rows()
        {
            return Ok(false);
        }
        let Some(device) = self.dspark.as_mut() else {
            return Ok(false);
        };
        let cycle = &mut device.cycle;
        let stride = cycle.hash.shape.stride as usize;
        if cycle.layers.is_empty() || cycle.memory.is_none() || hashed.rows() > stride {
            return Ok(false);
        }
        if cycle.pending.is_some() || cycle.gathering || cycle.deferred.is_some() {
            return Err("DSpark device cycle is already pending".into());
        }
        let columns = cycle.hash.shape.columns() as usize;
        let ids = ar_engram::layer_major(hashed, cycle.layers.len(), columns, stride)
            .map_err(|error| error.to_string())?;
        cycle.deferred = Some(Deferred {
            ids,
            rows: hashed.rows() as u32,
            mixed: Some(Mixed::default()),
        });
        Ok(true)
    }

    /// Whether this step's verify rows wait for its prefill
    /// (`defer_mixed_dspark_engram`).
    pub(in crate::instance) fn mixed_engram_deferred(&self) -> bool {
        self.dspark
            .as_ref()
            .and_then(|device| device.cycle.deferred.as_ref())
            .is_some_and(|deferred| deferred.mixed.is_some())
    }

    /// Begins a mixed step's verify-row gathers once its prefill joined its
    /// last Engram layer on every stream. The ids go up on rank 0's stream,
    /// whose position each gather's id copy waits for, so their host
    /// callbacks follow the prefill's; each rank's upload waits for that
    /// rank's position, after every earlier reader of its rows.
    pub(in crate::instance) fn begin_mixed_dspark_engram(&self) -> Result<(), BundleAbiError> {
        let Some(cycle) = self.dspark.as_ref().map(|device| &device.cycle) else {
            return Ok(());
        };
        let Some((deferred, mixed)) = cycle
            .deferred
            .as_ref()
            .and_then(|deferred| deferred.mixed.as_ref().map(|mixed| (deferred, mixed)))
        else {
            return Ok(());
        };
        if cycle.gathering || mixed.begun.get() != 0 {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        let memory = cycle
            .memory
            .ok_or(BundleAbiError::InvalidRange("DSpark hash memory"))?;
        upload_bytes(&self.services, self.ranks[0].2, memory.ids, &deferred.ids)?;
        let rows = u64::from(deferred.rows) * cycle.hash.shape.columns();
        for layer in &cycle.layers {
            layer.begin(&self.services, rows, PreparedGather::DsparkMixedTarget)?;
            mixed.begun.set(mixed.begun.get() + 1);
        }
        Ok(())
    }

    /// Completes the verify-row gathers a failed mixed step left begun,
    /// before any later step writes their rows: every rank's stream makes
    /// the joins the verify launch did not, and the host waits for those
    /// streams. A deferral never begun is dropped. A poisoned instance
    /// keeps its gathers and refuses.
    pub(in crate::instance) fn settle_mixed_dspark_engram(&mut self) -> Result<(), BundleAbiError> {
        let Some(cycle) = self.dspark.as_ref().map(|device| &device.cycle) else {
            return Ok(());
        };
        let Some(mixed) = cycle
            .deferred
            .as_ref()
            .and_then(|deferred| deferred.mixed.as_ref())
        else {
            return Ok(());
        };
        let (begun, ranks) = (mixed.begun.get(), self.ranks.len());
        if begun > 0 {
            if self.collectives.is_poisoned() {
                return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
            }
            for join in mixed.joined.get()..begun * ranks {
                cycle.layers[join / ranks].join(&self.services, self.ranks[join % ranks].2)?;
                mixed.joined.set(join + 1);
            }
            for &(_, _, stream) in &self.ranks {
                self.services.synchronize_stream(stream)?;
            }
            for layer in &cycle.layers[..begun] {
                layer
                    .finish(&self.services)?
                    .ok_or(BundleAbiError::Status(BundleStatusV1::InvalidState))?;
            }
        }
        let cycle = &mut self.dspark.as_mut().expect("drafter retained").cycle;
        cycle.deferred = None;
        cycle.gathering = false;
        Ok(())
    }
}
