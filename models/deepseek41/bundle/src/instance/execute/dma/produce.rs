//! The producer before a BF16-payload DMA call, split by owner interval so
//! each destination can pull a source's interval as soon as that source has
//! produced and rounded it.
//!
//! Admitted only where every rank's launch immediately before the call is
//! the MoE route-row sum with its shared expert addend, sized to the call's
//! rows and writing exactly the call's partial rows. Each interval's launch
//! keeps the original's symbol and arguments, with its row views narrowed
//! to the interval, its live count read from the prefill count table and
//! one block per token as before. The sum computes every token from that
//! token's routes and addend alone, so each row's bytes are unchanged; the
//! owner sum still adds every rank's rows in rank order.
use super::Call;
use super::consume::{narrow, overlaps};
use crate::instance::Instance;
use crate::ops::RowLocalKind;
use bundle_runtime::dispatch::{self, DispatchError, Resolved};
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::contract::Argument;

pub(in crate::instance::execute) struct Producers {
    /// `[rank][slice]`: the producer over the interval the rank sends in
    /// scatter phase `slice`, then over its own interval.
    launches: Vec<Vec<PreparedLaunch>>,
}

impl Producers {
    /// `producers[rank]` is the launch before the call on that rank and
    /// `flags[rank]` its collective flag. `None` keeps it before the call.
    pub(in crate::instance::execute) fn prepare(
        instance: &Instance,
        call: &Call,
        producers: &[&PreparedLaunch],
        flags: &[bool],
    ) -> Result<Option<Self>, String> {
        let rows = call.rows;
        if !crate::ops::dma_produce_requested()
            || call.embedding
            || call.sends.len() != 4
            || !rows.is_multiple_of(32)
            || instance.prefill_counts.len() != 4
            || producers.len() != 4
            || flags.len() != 4
            || flags.iter().any(|&f| f)
        {
            return Ok(None);
        }
        let geometry = instance.config.shape.geometry;
        let owner_rows = rows / 4;
        let mut launches = Vec::new();
        for (rank, launch) in producers.iter().enumerate() {
            let Some(layout) = crate::Kernels::prefill_dma_producer(&launch.symbol, geometry)
            else {
                return Ok(None);
            };
            if layout.kind != RowLocalKind::Produce || launch.rows != rows {
                return Ok(None);
            }
            let partial = crate::buffers::row_range(
                instance.buffers[rank].scratch.partial,
                call.first,
                rows,
            )?;
            // The call reads and writes only its partial rows' intervals in
            // the producer's footprint, each after its interval's launch.
            for (index, argument) in launch.arguments.0.iter().enumerate() {
                let Argument::DeviceBuffer(id) = argument else {
                    continue;
                };
                if Some(index) == layout.call_rows {
                    continue;
                }
                let view = launch
                    .bindings
                    .get(*id)
                    .map_err(|e| format!("DMA producer argument: {e:?}"))?;
                if call.footprint.iter().any(|&span| overlaps(view, span)) {
                    return Ok(None);
                }
            }
            let mut slices = Vec::new();
            for owner in (1..=4).map(|step| (rank + step) % 4) {
                let begin = owner as u32 * owner_rows;
                let Some(part) =
                    narrow(instance, rank, launch, &layout, partial, begin, owner_rows)?
                else {
                    return Ok(None);
                };
                slices.push(part);
            }
            launches.push(slices);
        }
        Ok(Some(Self { launches }))
    }

    pub(super) fn resolve(&self, instance: &Instance) -> Result<Vec<Vec<Resolved>>, DispatchError> {
        self.launches
            .iter()
            .enumerate()
            .map(|(rank, slices)| {
                slices
                    .iter()
                    .map(|launch| {
                        dispatch::resolve(
                            &instance.symbols[rank],
                            &launch.symbol,
                            launch.plan,
                            &launch.arguments,
                            &launch.bindings,
                        )
                    })
                    .collect()
            })
            .collect()
    }
}
