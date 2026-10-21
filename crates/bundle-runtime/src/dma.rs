//! Four-rank F32 scatter, local owner sum and BF16 gather.
//!
//! Arithmetic belongs to the caller's owner-sum entry. The transport copies
//! exact bytes into rank-indexed slots, waits for every partial producer,
//! and completes every gather before any rank can reuse the buffers or
//! events. Peer-visible buffers must have private allocation owners, outside
//! the active compute streams, so automatic peer-copy joins cannot order
//! unrelated compute. Drain active streams before releasing those owners.

use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleEventHandleV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1,
};

use crate::acquire::RankHandles;
use crate::dispatch::{DispatchError, Resolved};
use crate::pipeline::{Collectives, PoisonOnUnwind};
use crate::resources::{Held, allocate_span};

const RANKS: usize = 4;

pub struct Rank {
    /// Allocation and automatic peer-copy joins only, never compute.
    pub owner: BundleStreamHandleV1,
    context: BundleContextHandleV1,
    inbox: Span,
    events: [BundleEventHandleV1; 3],
}

/// Allocate the largest rank-indexed inbox and three reusable phase events.
pub fn acquire(
    services: &HostServicesV1,
    ranks: &[RankHandles],
    max_elements: u64,
    held: &mut Vec<Held>,
) -> Result<Vec<Rank>, BundleAbiError> {
    if ranks.len() != RANKS || max_elements == 0 || max_elements % 16 != 0 {
        return Err(BundleAbiError::InvalidRange("DMA rank or element capacity"));
    }
    let bytes = max_elements
        .checked_mul(4)
        .ok_or(BundleAbiError::InvalidRange("DMA inbox capacity overflow"))?;
    ranks
        .iter()
        .map(|&(context, pool, _)| {
            let owner = services.create_stream(context, 1)?;
            held.push(Held::Stream(owner));
            let inbox = allocate_span(services, pool, owner, bytes, held)?;
            let mut events = Vec::new();
            for _ in 0..3 {
                let event = services.create_event(context)?;
                held.push(Held::Event(event));
                events.push(event);
            }
            Ok(Rank {
                owner,
                context,
                inbox,
                events: events.try_into().expect("three phase events"),
            })
        })
        .collect()
}

/// A checked subview. Copy intervals never extend an allocation's view.
pub fn subspan(span: Span, offset: u64, length: u64) -> Result<Span, &'static str> {
    if offset
        .checked_add(length)
        .is_none_or(|end| end > span.length)
    {
        return Err("DMA interval exceeds its buffer view");
    }
    Ok(Span::new(
        span.buffer,
        span.offset
            .checked_add(offset)
            .ok_or("DMA offset overflow")?,
        length,
    ))
}

/// One owner's contiguous output interval and rank-indexed local inputs.
#[derive(Clone, Copy, Debug)]
pub struct Owner {
    pub input: Span,
    pub inbox: Span,
    pub output: Span,
    pub slot_elements: u64,
    pub rank: u32,
    pub rows: u32,
}

#[derive(Clone, Copy, Debug)]
struct Transfer {
    source: usize,
    destination: usize,
    input: Span,
    output: Span,
}

pub struct Call {
    scatter: [Vec<Transfer>; 3],
    gather: [Vec<Transfer>; 3],
    owners: Vec<Owner>,
}

impl Call {
    /// Partition complete rows among four owners. Every owner keeps its
    /// own F32 partial local; each peer contributes one unmodified slice.
    pub fn prepare(
        ranks: &[Rank],
        inputs: &[Span],
        outputs: &[Span],
        rows: u32,
        width: u32,
    ) -> Result<Self, &'static str> {
        if ranks.len() != RANKS || inputs.len() != RANKS || outputs.len() != RANKS {
            return Err("DMA requires four ranks");
        }
        Self::layout(
            &ranks.iter().map(|r| r.inbox).collect::<Vec<_>>(),
            inputs,
            outputs,
            rows,
            width,
        )
    }

    fn layout(
        inboxes: &[Span],
        inputs: &[Span],
        outputs: &[Span],
        rows: u32,
        width: u32,
    ) -> Result<Self, &'static str> {
        if rows < 4 || width == 0 || width % 8 != 0 {
            return Err("DMA needs four nonempty vector-aligned owner intervals");
        }
        let width = u64::from(width);
        let elements = u64::from(rows) * width;
        let input_bytes = elements.checked_mul(4).ok_or("DMA input size overflow")?;
        let output_bytes = elements.checked_mul(2).ok_or("DMA output size overflow")?;
        if inputs
            .iter()
            .any(|s| s.length != input_bytes || s.offset % 16 != 0)
            || outputs
                .iter()
                .any(|s| s.length != output_bytes || s.offset % 16 != 0)
            || inboxes.iter().any(|s| s.offset % 16 != 0)
        {
            return Err("DMA input or output extent differs");
        }
        let owner_rows = rows.div_ceil(RANKS as u32);
        let slot = u64::from(owner_rows) * width;
        let inbox_bytes = slot.checked_mul(16).ok_or("DMA inbox size overflow")?;
        let interval = |rank: usize| {
            let first = (rank as u32 * owner_rows).min(rows);
            (u64::from(first) * width, owner_rows.min(rows - first))
        };
        let mut owners = Vec::new();
        for rank in 0..RANKS {
            let (first, count) = interval(rank);
            if count == 0 {
                return Err("DMA owner interval is empty");
            }
            owners.push(Owner {
                input: subspan(inputs[rank], first * 4, u64::from(count) * width * 4)?,
                inbox: subspan(inboxes[rank], 0, inbox_bytes)?,
                output: subspan(outputs[rank], first * 2, u64::from(count) * width * 2)?,
                slot_elements: slot,
                rank: rank as u32,
                rows: count,
            });
        }
        let mut scatter = [Vec::new(), Vec::new(), Vec::new()];
        let mut gather = [Vec::new(), Vec::new(), Vec::new()];
        for phase in 0..3 {
            for destination in 0..RANKS {
                let source = (destination + 3 - phase) % RANKS;
                let (first, count) = interval(destination);
                scatter[phase].push(Transfer {
                    source,
                    destination,
                    input: subspan(inputs[source], first * 4, u64::from(count) * width * 4)?,
                    output: subspan(
                        inboxes[destination],
                        source as u64 * slot * 4,
                        u64::from(count) * width * 4,
                    )?,
                });
                let (first, count) = interval(source);
                gather[phase].push(Transfer {
                    source,
                    destination,
                    input: subspan(outputs[source], first * 2, u64::from(count) * width * 2)?,
                    output: subspan(
                        outputs[destination],
                        first * 2,
                        u64::from(count) * width * 2,
                    )?,
                });
            }
        }
        Ok(Self {
            scatter,
            gather,
            owners,
        })
    }

    /// Bind and resolve every arithmetic launch before executing any DMA.
    pub fn owners(&self) -> &[Owner] {
        &self.owners
    }

    /// Execute the qualified three-permutation schedule. Sums must implement
    /// the caller's rank-ordered F32 fold and one final BF16 rounding.
    pub fn execute(
        &self,
        services: &HostServicesV1,
        ranks: &[Rank],
        streams: &[BundleStreamHandleV1],
        sums: &[Resolved],
        collectives: &Collectives,
    ) -> Result<(), DispatchError> {
        if ranks.len() != RANKS
            || streams.len() != RANKS
            || sums.len() != RANKS
            || collectives.is_poisoned()
        {
            return Err(DispatchError::InvalidPlan);
        }
        let fence = |phase: usize| -> Result<(), BundleAbiError> {
            for (rank, &stream) in ranks.iter().zip(streams) {
                services.record_event(rank.events[phase], stream)?;
            }
            for &stream in streams {
                for rank in ranks {
                    services.wait_event(stream, rank.events[phase])?;
                }
            }
            Ok(())
        };
        let copy = |transfers: &[Transfer]| -> Result<(), BundleAbiError> {
            for t in transfers {
                services.copy_peer_to_peer(
                    streams[t.destination],
                    ranks[t.destination].context,
                    t.output,
                    ranks[t.source].context,
                    t.input,
                )?;
            }
            Ok(())
        };
        let _unwind = PoisonOnUnwind(collectives);
        let result = (|| {
            fence(0)?;
            for phase in &self.scatter {
                copy(phase)?;
            }
            for (sum, &stream) in sums.iter().zip(streams) {
                sum.submit(services, stream)?;
            }
            fence(1)?;
            for phase in &self.gather {
                copy(phase)?;
            }
            fence(2)?;
            Ok(())
        })();
        if result.is_err() {
            collectives.poison();
        }
        result
    }
}

#[cfg(test)]
mod tests;
