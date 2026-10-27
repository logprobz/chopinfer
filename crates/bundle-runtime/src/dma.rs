//! Four-rank ordered scatter, local owner sum and BF16 gather.
//!
//! Arithmetic belongs to the caller's owner-sum entry. The transport copies
//! exact bytes into rank-indexed slots, waits for every partial producer,
//! and completes every gather before any rank can reuse the buffers or
//! events. Peer-visible buffers must have private allocation owners, outside
//! the active compute streams, so automatic peer-copy joins cannot order
//! unrelated compute. Drain active streams before releasing those owners.

use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleEventHandleV1, BundleStatusV1,
    BundleStreamHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1,
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

impl Transfer {
    fn issuer(&self, push: bool) -> usize {
        if push { self.source } else { self.destination }
    }
}

/// The bytes supplied by each peer. Conversion and owner arithmetic belong
/// to the caller, including the owner's treatment of its local partial.
#[derive(Clone, Copy, Debug)]
pub enum Payload {
    F32,
    Bf16,
}

impl Payload {
    fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
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
        Self::with_payload(inboxes, inputs, outputs, rows, width, Payload::F32)
    }

    /// Prepare copies of already converted peer payloads. Local owner inputs
    /// may use another format; the caller binds those to its arithmetic.
    pub fn with_payload(
        inboxes: &[Span],
        inputs: &[Span],
        outputs: &[Span],
        rows: u32,
        width: u32,
        payload: Payload,
    ) -> Result<Self, &'static str> {
        if inboxes.len() != RANKS || inputs.len() != RANKS || outputs.len() != RANKS {
            return Err("DMA requires four ranks");
        }
        let bytes = payload.bytes();
        if rows < 4 || width == 0 || width % 8 != 0 {
            return Err("DMA needs four nonempty vector-aligned owner intervals");
        }
        let width = u64::from(width);
        let elements = u64::from(rows) * width;
        let input_bytes = elements
            .checked_mul(bytes)
            .ok_or("DMA input size overflow")?;
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
        let inbox_bytes = slot
            .checked_mul(RANKS as u64 * bytes)
            .ok_or("DMA inbox size overflow")?;
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
                input: subspan(
                    inputs[rank],
                    first * bytes,
                    u64::from(count) * width * bytes,
                )?,
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
                    input: subspan(
                        inputs[source],
                        first * bytes,
                        u64::from(count) * width * bytes,
                    )?,
                    output: subspan(
                        inboxes[destination],
                        source as u64 * slot * bytes,
                        u64::from(count) * width * bytes,
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

pub struct OrderedRank {
    /// Allocation/free and the host's automatic peer-copy joins only. Keeping
    /// this off both active lanes avoids serializing disjoint row parts.
    pub owner: BundleStreamHandleV1,
    /// Gather copies when the rank's main stream consumes each gathered
    /// interval as it lands, and scatter copies when it produces each sent
    /// interval before its copy; idle otherwise.
    lane: BundleStreamHandleV1,
    context: BundleContextHandleV1,
    inbox: Span,
    /// 0, 4 and 7 join the call's phases across ranks; 1 to 3, 5 and 6 order
    /// the per-interval producer form (`Call::execute_ordered`).
    events: [BundleEventHandleV1; 8],
    /// One per copy phase, recorded on the lane after the copies this rank
    /// issued in that phase.
    copied: [BundleEventHandleV1; 3],
}

pub fn acquire_ordered(
    services: &HostServicesV1,
    ranks: &[RankHandles],
    inbox_bytes: u64,
    held: &mut Vec<Held>,
) -> Result<Vec<OrderedRank>, BundleAbiError> {
    if ranks.len() != RANKS || inbox_bytes < 16 {
        return Err(BundleAbiError::InvalidRange("DMA rank or inbox capacity"));
    }
    ranks
        .iter()
        .map(|&(context, pool, _)| {
            let owner = services.create_stream(context, 1)?;
            held.push(Held::Stream(owner));
            let lane = services.create_stream(context, 1)?;
            held.push(Held::Stream(lane));
            let inbox = allocate_span(services, pool, owner, inbox_bytes, held)?;
            let mut events = Vec::new();
            for _ in 0..11 {
                let event = services.create_event(context)?;
                held.push(Held::Event(event));
                events.push(event);
            }
            let copied = events.split_off(8);
            Ok(OrderedRank {
                owner,
                context,
                lane,
                inbox,
                events: events.try_into().expect("eight DMA events"),
                copied: copied.try_into().expect("three copy events"),
            })
        })
        .collect()
}

/// Whether the host copies on a source's stream. One 16-byte push per
/// ordered pair of ranks, between their inboxes, before either holds data.
/// A host from before pushes refuses the first with `InvalidArgument`
/// before submitting anything, and the instance then pulls; any later
/// refusal fails creation, never a step.
pub fn pushes(services: &HostServicesV1, dma: &[OrderedRank]) -> Result<bool, BundleAbiError> {
    let probe = |span| subspan(span, 0, 16).map_err(|_| BundleAbiError::InvalidRange("DMA inbox"));
    for (source, from) in dma.iter().enumerate() {
        for (destination, to) in dma.iter().enumerate() {
            if source == destination {
                continue;
            }
            // The destination's inbox exists once its owner stream has
            // allocated it.
            services.record_event(to.events[0], to.owner)?;
            services.wait_event(from.owner, to.events[0])?;
            match services.copy_peer_to_peer(
                from.owner,
                to.context,
                probe(to.inbox)?,
                from.context,
                probe(from.inbox)?,
            ) {
                Ok(()) => {}
                Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
                    if (source, destination) == (0, 1) =>
                {
                    return Ok(false);
                }
                Err(error) => return Err(error),
            }
        }
    }
    Ok(true)
}

impl OrderedRank {
    /// The private allocation, including any caller-owned rounded outbox.
    pub fn inbox(&self) -> Span {
        self.inbox
    }
}

/// Complete compute reads, then interval-copy lanes, then caller side lanes
/// before releasing graphs or privately owned buffers. Stop on the first error.
pub fn drain(
    services: &HostServicesV1,
    streams: impl IntoIterator<Item = BundleStreamHandleV1>,
    ranks: &[OrderedRank],
    side_streams: impl IntoIterator<Item = BundleStreamHandleV1>,
) -> Result<(), BundleAbiError> {
    for stream in streams
        .into_iter()
        .chain(ranks.iter().map(|rank| rank.lane))
        .chain(side_streams)
    {
        services.synchronize_stream(stream)?;
    }
    Ok(())
}

/// Resolved per-rank, per-owner consumer chains.
pub type Consumers = Vec<Vec<Vec<Resolved>>>;

/// Per-rank producers in send-phase order, followed by the local interval.
/// Rounding entries are indexed by rank and destination owner.
pub struct Producers {
    pub launches: Vec<Vec<Resolved>>,
    pub rounds: Vec<Vec<Option<Resolved>>>,
}

/// Model-owned arithmetic resolved at the same boundaries as its transport.
/// Implementations keep payload conversion, row admission and sum order.
pub trait Arithmetic {
    fn rounds(&self) -> Result<Vec<Vec<Resolved>>, DispatchError>;
    fn sums(&self) -> Result<Vec<Resolved>, DispatchError>;
    fn producers(&self) -> Result<Option<Producers>, DispatchError>;
    fn consumers(&self) -> Result<Option<Consumers>, DispatchError>;
}

/// The rank that copies into `rank` in copy phase `phase`.
fn sender(rank: usize, phase: usize) -> usize {
    (rank + RANKS - 1 - phase) % RANKS
}

impl Call {
    /// Run the call. With `producers`, each rank's main stream produces and
    /// rounds the interval it sends in each scatter phase in phase order,
    /// then its own, and each copy of that interval runs on a lane once that
    /// source has rounded it. With `consumers`, the gather copies move to
    /// the lanes and each rank's main stream runs the consumers of its own
    /// interval after its sum and of every other interval after that
    /// interval's copy. With `push`, every copy runs on its
    /// source's stream or lane instead of its destination's, in the same
    /// permutation order, and each destination waits for its source's copy
    /// in that phase.
    pub fn execute_ordered(
        &self,
        services: &HostServicesV1,
        ranks: &[OrderedRank],
        streams: &[BundleStreamHandleV1],
        push: bool,
        arithmetic: &impl Arithmetic,
    ) -> Result<(), DispatchError> {
        if streams.len() != RANKS || ranks.len() != RANKS {
            return Err(DispatchError::InvalidPlan);
        }
        let rounds = arithmetic.rounds()?;
        let sums = arithmetic.sums()?;
        if rounds.len() != RANKS || sums.len() != RANKS {
            return Err(DispatchError::InvalidPlan);
        }
        let record = |stage: usize| -> Result<(), BundleAbiError> {
            for (rank, &stream) in streams.iter().enumerate() {
                services.record_event(ranks[rank].events[stage], stream)?;
            }
            Ok(())
        };
        let wait = |stage: usize| -> Result<(), BundleAbiError> {
            for &stream in streams {
                for rank in ranks {
                    services.wait_event(stream, rank.events[stage])?;
                }
            }
            Ok(())
        };
        let copies = |transfers: &[Transfer],
                      streams: &[BundleStreamHandleV1]|
         -> Result<(), BundleAbiError> {
            for transfer in transfers {
                services.copy_peer_to_peer(
                    streams[transfer.issuer(push)],
                    ranks[transfer.destination].context,
                    transfer.output,
                    ranks[transfer.source].context,
                    transfer.input,
                )?;
            }
            Ok(())
        };
        let lanes = ranks.iter().map(|rank| rank.lane).collect::<Vec<_>>();
        if let Some(producers) = arithmetic.producers()? {
            self.produce(services, ranks, streams, push, &producers)?;
        } else {
            // Every rank's rounded rows exist before any peer copies them.
            for (rank, rounds) in rounds.iter().enumerate() {
                for round in rounds {
                    round.submit(services, streams[rank])?;
                }
            }
            // Every partial is complete before any rank reads a peer's.
            record(0)?;
            wait(0)?;
            // A rank's three scatter pulls all land in its own inbox from
            // its own stream, so its sum follows them without a cross-rank
            // barrier between permutations; the permutation order is
            // unchanged. Pushes keep that order on each source's stream, and
            // each destination waits for the source that sent it each phase.
            for phase in 0..3 {
                copies(&self.scatter[phase], streams)?;
                if push {
                    for (rank, dma) in ranks.iter().enumerate() {
                        services.record_event(dma.copied[phase], streams[rank])?;
                    }
                }
            }
            if push {
                self.receive(services, ranks, streams)?;
            }
        }
        for (rank, sum) in sums.iter().enumerate() {
            sum.submit(services, streams[rank])?;
        }
        // Every owner interval is summed before any rank gathers it.
        record(4)?;
        let Some(resolved) = arithmetic.consumers()? else {
            wait(4)?;
            for phase in 0..3 {
                copies(&self.gather[phase], streams)?;
            }
            // All peer reads have completed before any rank can reuse
            // partials, rounded rows, outputs, inboxes, or this call's
            // event records.
            record(7)?;
            wait(7)?;
            return Ok(());
        };
        if resolved.len() != RANKS || resolved.iter().any(|owners| owners.len() != RANKS) {
            return Err(DispatchError::InvalidPlan);
        }
        for &lane in &lanes {
            for rank in ranks {
                services.wait_event(lane, rank.events[4])?;
            }
        }
        // A rank's own rows are final once its sum is, in its stream order.
        for (rank, &stream) in streams.iter().enumerate() {
            for launch in &resolved[rank][rank] {
                launch.submit(services, stream)?;
            }
        }
        for phase in 0..3 {
            copies(&self.gather[phase], &lanes)?;
            for (rank, dma) in ranks.iter().enumerate() {
                services.record_event(dma.copied[phase], lanes[rank])?;
            }
        }
        for phase in 0..3 {
            for (rank, &stream) in streams.iter().enumerate() {
                let issuer = if push { sender(rank, phase) } else { rank };
                services.wait_event(stream, ranks[issuer].copied[phase])?;
                for launch in &resolved[rank][(rank + 3 - phase) % 4] {
                    launch.submit(services, stream)?;
                }
            }
        }
        // As above, with the copies' completion recorded on the lanes.
        for (rank, &lane) in lanes.iter().enumerate() {
            services.record_event(ranks[rank].events[7], lane)?;
        }
        wait(7)?;
        Ok(())
    }

    /// Each rank's interval producers and roundings on its main stream, and
    /// the scatter copies on the lanes: each destination's pulls on its own,
    /// or each source's pushes on its own. Event 5 marks a rank's main
    /// stream at the call, events 1 to 3 its interval for each scatter
    /// phase, event 6 a destination's three pulls. Ends with every rank's
    /// main stream after its own interval's producer and the copies into
    /// its inbox.
    fn produce(
        &self,
        services: &HostServicesV1,
        ranks: &[OrderedRank],
        streams: &[BundleStreamHandleV1],
        push: bool,
        producers: &Producers,
    ) -> Result<(), DispatchError> {
        let produced = &producers.launches;
        let sends = &producers.rounds;
        if produced.len() != RANKS
            || sends.len() != RANKS
            || produced.iter().any(|parts| parts.len() != RANKS)
            || sends.iter().any(|parts| parts.len() != RANKS)
        {
            return Err(DispatchError::InvalidPlan);
        }
        let lanes = ranks.iter().map(|rank| rank.lane).collect::<Vec<_>>();
        for (rank, &stream) in streams.iter().enumerate() {
            let dma = &ranks[rank];
            // A lane writes this rank's inbox only after the rank's main
            // stream has reached the call: its previous sums have read it.
            services.record_event(dma.events[5], stream)?;
            if !push {
                services.wait_event(lanes[rank], dma.events[5])?;
            }
            for phase in 0..3 {
                let to = (rank + 1 + phase) % 4;
                let round = sends[rank][to].as_ref().ok_or(DispatchError::InvalidPlan)?;
                produced[rank][phase].submit(services, stream)?;
                round.submit(services, stream)?;
                services.record_event(dma.events[1 + phase], stream)?;
            }
            produced[rank][3].submit(services, stream)?;
        }
        // Each copy waits only for its own source's interval; a lane keeps
        // its copies in permutation order. A push also waits for its
        // destination to reach the call.
        for phase in 0..3 {
            for transfer in &self.scatter[phase] {
                let lane = lanes[transfer.issuer(push)];
                services.wait_event(lane, ranks[transfer.source].events[1 + phase])?;
                if push {
                    services.wait_event(lane, ranks[transfer.destination].events[5])?;
                }
                services.copy_peer_to_peer(
                    lane,
                    ranks[transfer.destination].context,
                    transfer.output,
                    ranks[transfer.source].context,
                    transfer.input,
                )?;
                if push {
                    services.record_event(ranks[transfer.source].copied[phase], lane)?;
                }
            }
        }
        if push {
            self.receive(services, ranks, streams)?;
            return Ok(());
        }
        for (rank, &stream) in streams.iter().enumerate() {
            let dma = &ranks[rank];
            services.record_event(dma.events[6], lanes[rank])?;
            services.wait_event(stream, dma.events[6])?;
        }
        Ok(())
    }

    /// Each rank's main stream waits for the push into its inbox in every
    /// scatter phase, recorded as that phase's copy event of its sender.
    fn receive(
        &self,
        services: &HostServicesV1,
        ranks: &[OrderedRank],
        streams: &[BundleStreamHandleV1],
    ) -> Result<(), BundleAbiError> {
        for phase in 0..3 {
            for (rank, &stream) in streams.iter().enumerate() {
                services.wait_event(stream, ranks[sender(rank, phase)].copied[phase])?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod payload_tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn payload_copy_coverage_keeps_rank_slots_and_owner_intervals() {
        for payload in [Payload::F32, Payload::Bf16] {
            for rows in [512_u32, 513, 1023, 2048, 8191, 8192] {
                let width = 16;
                let bytes = payload.bytes();
                let elements = u64::from(rows) * u64::from(width);
                let slot = u64::from(rows.div_ceil(4)) * u64::from(width);
                let spans = |first, length| {
                    (first..first + 4)
                        .map(|id| {
                            Span::new(BundleDeviceBufferHandleV1::from_parts(1, id), 256, length)
                        })
                        .collect::<Vec<_>>()
                };
                let inputs = spans(1, elements * bytes);
                let outputs = spans(5, elements * 2);
                let inboxes = spans(9, slot * 4 * bytes);
                let call =
                    Call::with_payload(&inboxes, &inputs, &outputs, rows, width, payload).unwrap();
                for (phase, transfers) in call.scatter.iter().enumerate() {
                    assert_eq!(transfers.len(), 4);
                    for (destination, transfer) in transfers.iter().enumerate() {
                        let source = sender(destination, phase);
                        assert_eq!(
                            (transfer.source, transfer.destination),
                            (source, destination)
                        );
                        assert_eq!(transfer.issuer(true), source);
                        assert_eq!(transfer.issuer(false), destination);
                        let owner = call.owners[destination];
                        assert_eq!(transfer.input.buffer, inputs[source].buffer);
                        assert_eq!(
                            (transfer.input.offset, transfer.input.length),
                            (owner.input.offset, owner.input.length)
                        );
                        assert_eq!(
                            transfer.output,
                            subspan(
                                inboxes[destination],
                                source as u64 * slot * bytes,
                                owner.input.length
                            )
                            .unwrap()
                        );
                    }
                }
                for (destination, output) in outputs.iter().enumerate() {
                    let mut intervals = vec![call.owners[destination].output];
                    for transfer in call
                        .gather
                        .iter()
                        .flatten()
                        .filter(|t| t.destination == destination)
                    {
                        assert_eq!(transfer.input, call.owners[transfer.source].output);
                        assert_eq!(transfer.output.buffer, output.buffer);
                        intervals.push(transfer.output);
                    }
                    intervals.sort_by_key(|s| s.offset);
                    let mut end = output.offset;
                    for interval in intervals {
                        assert_eq!(interval.offset, end);
                        end += interval.length;
                    }
                    assert_eq!(end, output.offset + output.length);
                }
                let moved: u64 = call
                    .scatter
                    .iter()
                    .chain(&call.gather)
                    .flatten()
                    .map(|t| t.input.length)
                    .sum();
                assert_eq!(moved, elements * 3 * (bytes + 2));
                let mut short = inputs.clone();
                short[0].length -= 1;
                assert!(
                    Call::with_payload(&inboxes, &short, &outputs, rows, width, payload).is_err()
                );
                let mut short = inboxes.clone();
                short[0].length -= 1;
                assert!(
                    Call::with_payload(&short, &inputs, &outputs, rows, width, payload).is_err()
                );
                assert!(
                    Call::with_payload(&inboxes[..3], &inputs, &outputs, rows, width, payload)
                        .is_err()
                );
            }
        }
    }
}
