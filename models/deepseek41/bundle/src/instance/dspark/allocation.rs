//! Allocated verify steps on the instance (`crate::dspark::allocation`):
//! the allocation after the draft copies on every rank, the reserved head
//! rows and feature rows gathered back by `expand`, and the allocated
//! acceptance.
use super::Instance;
use crate::dspark::allocation::{Memory, Step};
use crate::dspark::device::{DECODE_BASE, slice};
use crate::lower::Lowering;
use crate::ops::{Bind, Lowered};
use crate::program::Phase;
use bundle_runtime::resources::upload_bytes;
use bundle_runtime::upload::DeviceInputs;
use kernel_core::backend::Counts;
use kernel_core::row_copy::GatherByIndexBuffers;
use model_deepseek41_contract::own_ops::acceptance::allocated::AllocatedAcceptanceBuffers;
use model_deepseek41_contract::own_ops::allocation::VerifyAllocateBuffers;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

impl Instance {
    /// The pending device target's allocation, when its step is allocated.
    pub(crate) fn allocated_step(&self) -> Option<&Step> {
        self.dspark
            .as_ref()?
            .cycle
            .pending
            .as_ref()?
            .allocation
            .as_ref()
    }

    /// The rows the pending allocated step's tail gathers into and scores:
    /// the prepared reserved rows, whatever the step reserves, so its
    /// programs depend only on its width. Rows past the step's reserved
    /// rows are never gathered, read or sampled.
    pub(crate) fn allocated_tail_rows(&self) -> Option<u32> {
        self.allocated_step()?;
        Some(self.dspark.as_ref()?.allocation.as_ref()?.op.reserved_rows)
    }

    /// Rank `rank`'s allocation storage.
    fn allocation_memory(&self, rank: usize) -> Result<Memory, String> {
        self.dspark
            .as_ref()
            .and_then(|device| device.allocation.as_ref())
            .and_then(|allocation| allocation.ranks.get(rank).copied())
            .ok_or_else(|| "DSpark verify allocation storage is absent".into())
    }

    /// Uploads `step`'s items and rule and allocates on every rank, in each
    /// rank's order after its reserved uploads and draft copies: the
    /// allocation compacts `target`'s per-row inputs (and rank 0's text
    /// mask `text`) to the computed rows, and the step's device counts
    /// become the width (`live_tokens`) and the reserved rows.
    pub(crate) fn allocate_dspark_step(
        &self,
        step: &Step,
        targets: &[DeviceInputs],
        text: Span,
    ) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let allocation = device
            .allocation
            .as_ref()
            .ok_or("DSpark verify allocation is not prepared")?;
        let words = |values: &[u32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let items = words(&step.item_words());
        let live = step.items.len() as u32;
        for (rank, (target, stored)) in targets.iter().zip(&device.ranks).enumerate() {
            let memory = self.allocation_memory(rank)?;
            let stream = self.ranks[rank].2;
            for (span, bytes) in [
                (memory.items, items.clone()),
                (memory.rule, words(&step.rule)),
                (memory.live_items, live.to_le_bytes().to_vec()),
                (memory.reserved, step.reserved_rows.to_le_bytes().to_vec()),
            ] {
                upload_bytes(
                    &self.services,
                    stream,
                    slice(span, 0, bytes.len() as u64)?,
                    &bytes,
                )
                .map_err(|e| e.to_string())?;
            }
            let scratch = &self.buffers[rank].scratch;
            let mut bind = Bind::default();
            let reserved = u64::from(allocation.op.reserved_rows);
            let buffers = VerifyAllocateBuffers {
                items: bind.view(memory.items),
                scores: bind.view(stored.confidence),
                rule: bind.view(memory.rule),
                live_items: bind.view(memory.live_items),
                status: bind.view(memory.status),
                verified: bind.view(memory.verified),
                starts: bind.view(memory.starts),
                rows: bind.view(memory.rows),
                expand: bind.view(memory.expand),
                token_ids: bind.view(target.token_ids),
                positions: bind.view(target.positions),
                token_batch: bind.view(target.token_batch),
                window_rows: bind.view(scratch.window_rows),
                index_visible: bind.view(scratch.index_visible),
                q_start: bind.view(target.q_start),
                text: bind.view(if rank == 0 {
                    slice(text, 0, reserved)?
                } else {
                    slice(text, 0, 0)?
                }),
            };
            let launches = bind
                .launch(&allocation.allocate, &buffers, Counts::rows(live))
                .map_err(|e| format!("DSpark verify allocation: {e}"))?;
            for lowered in launches {
                let launch = lowered.launch;
                bundle_runtime::dispatch::launch(
                    &self.services,
                    stream,
                    &self.symbols[rank],
                    &launch.symbol,
                    launch.plan,
                    &launch.arguments,
                    &launch.bindings,
                )
                .map_err(|e| format!("DSpark verify allocation launch: {e:?}"))?;
            }
            upload_bytes(
                &self.services,
                stream,
                target.live_tokens,
                &step.width.to_le_bytes(),
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// An allocated step's head on rank `rank`: its computed final-norm
    /// rows gathered by `expand` into the reserved rows, then the head over
    /// the prepared reserved rows (`base.head_rows`), the step's live.
    pub(crate) fn dspark_allocated_head(
        &self,
        base: &Lowering<'_>,
        phase: Phase,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let memory = self.allocation_memory(base.rank as usize)?;
        let Some(mut launches) =
            base.reserved_head_rows(phase, memory.expand, memory.reserved, memory.head)?
        else {
            return Ok(None);
        };
        let Some(head) = base.lm_head_over(memory.head, base.head_rows, memory.reserved)? else {
            return Ok(None);
        };
        launches.extend(head);
        Ok(Some(launches))
    }

    /// An allocated step's packed features on rank `rank`: its computed
    /// rows packed ahead of the decode rows, then gathered by `expand`
    /// into the reserved decode rows the commit reads. The gather's launch
    /// covers the `tail` prepared reserved rows; the device count bounds
    /// it to the step's.
    pub(crate) fn dspark_allocated_features(
        &self,
        rank: usize,
        phase: Phase,
        tail: u32,
    ) -> Result<Vec<Lowered>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let allocation = device
            .allocation
            .as_ref()
            .ok_or("DSpark verify allocation is not prepared")?;
        let memory = self.allocation_memory(rank)?;
        let stored = &device.ranks[rank];
        let width = device.features.row_bytes();
        let computed = slice(stored.packed, 0, u64::from(phase.rows()) * width)?;
        let counts = device.prepared.feature_pack_counts(phase.rows(), 0)?.len();
        let mut launches = device
            .prepared
            .pack_features(
                phase.rows(),
                stored.taps,
                stored.feature_indices,
                &stored.feature_counts[..counts],
                computed,
            )
            .map_err(|e| e.to_string())?;
        let mut bind = Bind::default();
        let buffers = GatherByIndexBuffers {
            source: bind.view(computed),
            row_indices: bind.view(memory.expand),
            count: bind.view(memory.reserved),
            destination: bind.view(slice(
                stored.packed,
                u64::from(DECODE_BASE) * width,
                u64::from(tail) * width,
            )?),
        };
        launches.extend(
            bind.launch(&allocation.features, &buffers, Counts::rows(tail))
                .map_err(|e| format!("DSpark reserved feature rows: {e}"))?,
        );
        Ok(launches)
    }

    /// The allocated acceptance of `count` items over rank 0's allocation.
    pub(crate) fn dspark_allocated_acceptance(
        &self,
        proposals: Span,
        count: u32,
    ) -> Result<Vec<Lowered>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let allocation = device
            .allocation
            .as_ref()
            .ok_or("DSpark verify allocation is not prepared")?;
        let memory = self.allocation_memory(0)?;
        let result = device
            .results
            .memory
            .ok_or("DSpark result storage absent")?;
        let mut bind = Bind::default();
        let buffers = AllocatedAcceptanceBuffers {
            items: bind.view(result.items),
            verified: bind.view(memory.verified),
            status: bind.view(memory.status),
            proposals: bind.view(proposals),
            samples: bind.view(slice(
                result.packet,
                crate::dspark::results::SAMPLE_OFFSET,
                u64::from(crate::dspark::results::ROWS) * 256,
            )?),
            stops: bind.view(result.stops),
            results: bind.view(slice(
                result.packet,
                0,
                u64::from(crate::dspark::results::ITEMS) * 32,
            )?),
        };
        bind.launch(&allocation.acceptance, &buffers, Counts::rows(count))
            .map_err(|e| e.to_string())
    }
}
