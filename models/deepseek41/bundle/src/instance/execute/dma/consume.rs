//! The row-local launches after a DMA call, split by owner interval so a
//! rank can run them on each interval as it lands.
//!
//! Admitted only where every rank's next launches are exactly the HC expand
//! of the call's reduced rows, the r8 c4 prefill mix and the cached
//! norm-ceil (or its form that certifies each row's router operands), each
//! sized to the call's rows. Each interval's launch keeps
//! the original's symbol and arguments, with its row views narrowed to the
//! interval, its live count read from the prefill count table and its row
//! grid sized to the interval. These entries compute every row from that
//! row's inputs alone, so each row's bytes are unchanged. Owner intervals
//! are whole eight-row mix tiles.
use super::{Call, subspan};
use crate::instance::Instance;
use crate::ops::{RowLocal, RowLocalKind};
use bundle_runtime::dispatch::{self, DispatchError, Resolved};
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::contract::Argument;
use runtime_bundle_abi::DeviceBufferSpanV1;

const CHAIN: [RowLocalKind; 3] = [RowLocalKind::Expand, RowLocalKind::Mix, RowLocalKind::Norm];

pub(in crate::instance::execute) struct Consumers {
    /// `[rank][owner]`: the chain's launches over that owner's rows.
    launches: Vec<Vec<Vec<PreparedLaunch>>>,
}

fn overlaps(a: DeviceBufferSpanV1, b: DeviceBufferSpanV1) -> bool {
    a.buffer == b.buffer
        && a.length != 0
        && b.length != 0
        && a.offset < b.offset + b.length
        && b.offset < a.offset + a.length
}

impl Consumers {
    /// The number of launches each rank's chain takes from its program.
    pub(in crate::instance::execute) const LAUNCHES: usize = CHAIN.len();

    /// `following[rank]` are the launches after the call on that rank and
    /// `flags[rank]` their collective flags. `None` keeps them after the call.
    pub(in crate::instance::execute) fn prepare(
        instance: &Instance,
        call: &Call,
        following: &[&[PreparedLaunch]],
        flags: &[&[bool]],
    ) -> Result<Option<Self>, String> {
        let rows = call.rows;
        if !crate::ops::dma_consume_requested()
            || call.embedding
            || !rows.is_multiple_of(32)
            || instance.prefill_counts.len() != 4
            || following.len() != 4
            || flags.len() != 4
        {
            return Ok(None);
        }
        let dims = instance.config.shape.dims();
        let (hidden, streams) = (u64::from(dims.hidden), u64::from(dims.hc.streams));
        let owner_rows = rows / 4;
        let mut launches = Vec::new();
        for rank in 0..4 {
            if following[rank].len() < CHAIN.len()
                || flags[rank].iter().take(CHAIN.len()).any(|&f| f)
            {
                return Ok(None);
            }
            let mut layouts = Vec::new();
            for (launch, kind) in following[rank].iter().zip(CHAIN) {
                match crate::Kernels::prefill_dma_consumer(&launch.symbol, hidden, streams) {
                    Some(layout) if layout.kind == kind && launch.rows == rows => {
                        layouts.push(layout)
                    }
                    _ => return Ok(None),
                }
            }
            let reduced = crate::buffers::row_range(
                instance.buffers[rank].scratch.reduced,
                call.first,
                rows,
            )?;
            let mut per_owner = Vec::new();
            for owner in 0..4 {
                let begin = owner * owner_rows;
                let mut chain = Vec::new();
                for (launch, layout) in following[rank].iter().zip(&layouts) {
                    let Some(part) =
                        narrow(instance, rank, launch, layout, reduced, begin, owner_rows)?
                    else {
                        return Ok(None);
                    };
                    chain.push(part);
                }
                per_owner.push(chain);
            }
            // Nothing the chain touches may be in flight in the call, except
            // the reduced rows its expand reads after they land.
            for (launch, layout) in following[rank].iter().zip(&layouts) {
                for (index, argument) in launch.arguments.0.iter().enumerate() {
                    let Argument::DeviceBuffer(id) = argument else {
                        continue;
                    };
                    if Some(index) == layout.reduced {
                        continue;
                    }
                    let view = launch
                        .bindings
                        .get(*id)
                        .map_err(|e| format!("DMA consumer argument: {e:?}"))?;
                    if call.footprint.iter().any(|&span| overlaps(view, span)) {
                        return Ok(None);
                    }
                }
            }
            launches.push(per_owner);
        }
        Ok(Some(Self { launches }))
    }

    pub(super) fn resolve(
        &self,
        instance: &Instance,
    ) -> Result<Vec<Vec<Vec<Resolved>>>, DispatchError> {
        self.launches
            .iter()
            .enumerate()
            .map(|(rank, owners)| {
                owners
                    .iter()
                    .map(|chain| {
                        chain
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
            })
            .collect()
    }
}

/// `launch` over rows `begin..begin + count` of its rows, or `None` where
/// its arguments are not the row layout's.
fn narrow(
    instance: &Instance,
    rank: usize,
    launch: &PreparedLaunch,
    layout: &RowLocal,
    reduced: DeviceBufferSpanV1,
    begin: u32,
    count: u32,
) -> Result<Option<PreparedLaunch>, String> {
    let rows = u64::from(launch.rows);
    let mut arguments = launch.arguments.clone();
    let mut bindings = launch.bindings.clone();
    let view = |id: &u64| {
        launch
            .bindings
            .get(*id)
            .map_err(|e| format!("DMA consumer argument: {e:?}"))
    };
    let mut seen = Vec::new();
    let present = layout
        .optional
        .iter()
        .filter(|&&(index, _)| launch.arguments.0.get(index) != Some(&Argument::NullDevicePointer));
    for &(index, bytes) in layout.rows.iter().chain(present) {
        let Some(Argument::DeviceBuffer(id)) = launch.arguments.0.get(index) else {
            return Ok(None);
        };
        let span = view(id)?;
        if span.length != rows * bytes || (Some(index) == layout.reduced && span != reduced) {
            return Ok(None);
        }
        let part = subspan(span, u64::from(begin) * bytes, u64::from(count) * bytes)?;
        arguments.0[index] = Argument::DeviceBuffer(bindings.bind(part));
        seen.push(index);
    }
    if layout.reduced.is_some_and(|index| !seen.contains(&index)) {
        return Ok(None);
    }
    match launch.arguments.0.get(layout.live) {
        Some(Argument::DeviceBuffer(id)) if view(id)?.length == 4 => {}
        _ => return Ok(None),
    }
    let counts = instance.prefill_counts[rank];
    arguments.0[layout.live] =
        Argument::DeviceBuffer(bindings.bind(subspan(counts, u64::from(count) * 4, 4)?));
    if let Some(index) = layout.row_offset {
        let Some(Argument::U64(first)) = launch.arguments.0.get(index) else {
            return Ok(None);
        };
        arguments.0[index] = Argument::U64(first + u64::from(begin));
    }
    // Tensor maps and pointer arrays hold addresses a row view cannot narrow.
    if launch
        .arguments
        .0
        .iter()
        .any(|a| matches!(a, Argument::TensorMap(_) | Argument::DevicePointerArray(_)))
    {
        return Ok(None);
    }
    let (dimension, step) = layout.grid;
    let mut plan = launch.plan;
    if plan.grid[dimension] != launch.rows.div_ceil(step) || !count.is_multiple_of(step) {
        return Ok(None);
    }
    plan.grid[dimension] = count / step;
    Ok(Some(PreparedLaunch {
        symbol: launch.symbol.clone(),
        rows: count,
        plan,
        arguments,
        bindings,
    }))
}
