//! Ordered DMA transport with an owner-local rank sum and BF16 gather. The
//! scatter carries the payload of the collective it replaces: F32 partials,
//! or each rank's partial rounded once to BF16 as the BF16-partial
//! contract's entries send it.
use super::{Run, Stage};
use crate::instance::Instance;
use bundle_runtime::acquire::RankHandles;
use bundle_runtime::dispatch::{self, DispatchError};
use bundle_runtime::pipeline::{PoisonOnUnwind, PreparedLaunch};
use bundle_runtime::resources::{Held, allocate_span};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{
    BundleAbiError, BundleEventHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1,
};

pub(crate) struct Rank {
    /// Allocation/free and the host's automatic peer-copy joins only. Keeping
    /// this off both active lanes avoids serializing disjoint row parts.
    pub(crate) owner: BundleStreamHandleV1,
    /// Gather copies when the rank's main stream consumes each gathered
    /// interval as it lands; idle otherwise.
    lane: BundleStreamHandleV1,
    inbox: DeviceBufferSpanV1,
    /// 0, 4 and 7 join the call's phases across ranks; 1 to 3, 5 and 6 order
    /// the per-interval producer form (`Call::produce`).
    events: [BundleEventHandleV1; 8],
    /// One per gather phase, recorded on the lane after its copy.
    gathered: [BundleEventHandleV1; 3],
}

pub(crate) fn acquire(
    services: &HostServicesV1,
    ranks: &[RankHandles],
    width: u32,
    held: &mut Vec<Held>,
) -> Result<Vec<Rank>, BundleAbiError> {
    ranks
        .iter()
        .map(|&(context, pool, _)| {
            let owner = services.create_stream(context, 1)?;
            held.push(Held::Stream(owner));
            let lane = services.create_stream(context, 1)?;
            held.push(Held::Stream(lane));
            let inbox = allocate_span(
                services,
                pool,
                owner,
                u64::from(PREFILL_TOKENS) * u64::from(width) * 4,
                held,
            )?;
            let mut events = Vec::new();
            for _ in 0..11 {
                let event = services.create_event(context)?;
                held.push(Held::Event(event));
                events.push(event);
            }
            let gathered = events.split_off(8);
            Ok(Rank {
                owner,
                lane,
                inbox,
                events: events.try_into().expect("eight DMA events"),
                gathered: gathered.try_into().expect("three gather events"),
            })
        })
        .collect()
}

pub(crate) fn drain(instance: &Instance) -> Result<(), BundleAbiError> {
    if instance.dma.is_empty() {
        return Ok(());
    }
    drain_streams(instance)
}

pub(crate) fn drain_streams(instance: &Instance) -> Result<(), BundleAbiError> {
    // Peer-copy owner joins cover DMA, not subsequent compute reads. Establish
    // completion even after a nonpoisoned step error before any graph or
    // privately owned buffer can be released. The shared entry layer refuses
    // destruction of poisoned instances before reaching this function.
    for stream in instance
        .ranks
        .iter()
        .map(|rank| rank.2)
        .chain(instance.dma.iter().map(|rank| rank.lane))
        .chain(instance.overlap.iter().map(|rank| rank.lane.stream))
        .chain(instance.side_lanes.iter().map(|lanes| lanes.side))
    {
        instance.services.synchronize_stream(stream)?;
    }
    Ok(())
}

fn subspan(
    span: DeviceBufferSpanV1,
    offset: u64,
    length: u64,
) -> Result<DeviceBufferSpanV1, String> {
    if offset
        .checked_add(length)
        .is_none_or(|end| end > span.length)
    {
        return Err("DMA interval exceeds its buffer view".into());
    }
    Ok(DeviceBufferSpanV1::new(
        span.buffer,
        span.offset
            .checked_add(offset)
            .ok_or("DMA offset overflow")?,
        length,
    ))
}

struct Transfer {
    source: usize,
    destination: usize,
    input: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
}

pub(super) struct Call {
    first: u32,
    rows: u32,
    embedding: bool,
    /// Each rank's rounding of the partial rows it sends (BF16 payload).
    rounds: Vec<Vec<PreparedLaunch>>,
    /// `[rank][owner]`: the same rounding of one owner's interval, for a
    /// call whose producer runs per interval (BF16 payload; empty for F32).
    sends: Vec<Vec<Option<PreparedLaunch>>>,
    scatter: [Vec<Transfer>; 3],
    sums: Vec<PreparedLaunch>,
    gather: [Vec<Transfer>; 3],
    footprint: Vec<DeviceBufferSpanV1>,
}

impl Call {
    /// The reduction of rows `first..first + rows` that replaces the four
    /// ranks' `collectives`, carrying the payload they send. `None` where
    /// the rows or the collectives are outside the transport; `composed`
    /// admits the composed form (`Kernels::prefill_dma_payload`).
    pub(super) fn prepare(
        instance: &Instance,
        first: u32,
        rows: u32,
        embedding: bool,
        collectives: &[&str],
        composed: bool,
    ) -> Result<Option<Self>, String> {
        if instance.dma.len() != 4 || !(512..=PREFILL_TOKENS).contains(&rows) {
            return Ok(None);
        }
        let payloads = collectives
            .iter()
            .map(|symbol| crate::Kernels::prefill_dma_payload(symbol, composed))
            .collect::<Vec<_>>();
        let Some(Some(payload)) = payloads.first().copied() else {
            return Ok(None);
        };
        if payloads.len() != 4 || payloads.iter().any(|p| *p != Some(payload)) {
            return Ok(None);
        }
        let bytes = payload.bytes();
        let width = u64::from(instance.config.shape.dims().hidden);
        let owner_rows = rows.div_ceil(4);
        let slot = u64::from(owner_rows) * width;
        let mut inputs = Vec::new();
        let mut sent = Vec::new();
        let mut outputs = Vec::new();
        let mut inboxes = Vec::new();
        let mut rounds = Vec::new();
        let mut sends = Vec::new();
        let mut sums = Vec::new();
        let mut footprint = Vec::new();
        for (rank, buffers) in instance.buffers.iter().enumerate() {
            let input = crate::buffers::row_range(buffers.scratch.partial, first, rows)?;
            let output = crate::buffers::row_range(
                if embedding {
                    buffers.scratch.embedded
                } else {
                    buffers.scratch.reduced
                },
                first,
                rows,
            )?;
            let inbox = subspan(instance.dma[rank].inbox, 0, slot * 4 * bytes)?;
            let begin = (rank as u32 * owner_rows).min(rows);
            let count = owner_rows.min(rows - begin);
            let own = |span: DeviceBufferSpanV1, bytes: u64| {
                subspan(
                    span,
                    u64::from(begin) * width * bytes,
                    u64::from(count) * width * bytes,
                )
            };
            // A BF16 payload's rounded rows follow the inbox in the same
            // allocation; the owner's own rows are rounded inside its sum.
            let outbox = match payload {
                crate::ops::DmaPayload::F32 => input,
                crate::ops::DmaPayload::Bf16 => subspan(
                    instance.dma[rank].inbox,
                    inbox.length,
                    u64::from(rows) * width * 2,
                )?,
            };
            let mut round = Vec::new();
            if payload == crate::ops::DmaPayload::Bf16 {
                for (from, to) in [(0, begin), (begin + count, rows)] {
                    if from < to {
                        let length = u64::from(to - from) * width;
                        round.push(crate::Kernels::prefill_dma_round(
                            subspan(input, u64::from(from) * width * 4, length * 4)?,
                            subspan(outbox, u64::from(from) * width * 2, length * 2)?,
                            to - from,
                        )?);
                    }
                }
                footprint.push(outbox);
                let mut by_owner = Vec::new();
                for owner in 0..4 {
                    let from = (owner * owner_rows).min(rows);
                    let to = (from + owner_rows).min(rows);
                    by_owner.push(if owner == rank as u32 || from == to {
                        None
                    } else {
                        let length = u64::from(to - from) * width;
                        Some(crate::Kernels::prefill_dma_round(
                            subspan(input, u64::from(from) * width * 4, length * 4)?,
                            subspan(outbox, u64::from(from) * width * 2, length * 2)?,
                            to - from,
                        )?)
                    });
                }
                sends.push(by_owner);
            }
            rounds.push(round);
            sums.push(crate::Kernels::prefill_dma_sum(
                payload,
                own(input, 4)?,
                inbox,
                own(output, 2)?,
                slot,
                rank as u32,
                count,
            )?);
            footprint.extend([input, output, inbox]);
            inputs.push(input);
            sent.push(outbox);
            outputs.push(output);
            inboxes.push(inbox);
        }
        let mut scatter = [Vec::new(), Vec::new(), Vec::new()];
        let mut gather = [Vec::new(), Vec::new(), Vec::new()];
        for phase in 0..3 {
            for destination in 0..4 {
                let source = (destination + 3 - phase) % 4;
                let first = (destination as u32 * owner_rows).min(rows);
                let count = owner_rows.min(rows - first);
                scatter[phase].push(Transfer {
                    source,
                    destination,
                    input: subspan(
                        sent[source],
                        u64::from(first) * width * bytes,
                        u64::from(count) * width * bytes,
                    )?,
                    output: subspan(
                        inboxes[destination],
                        source as u64 * slot * bytes,
                        u64::from(count) * width * bytes,
                    )?,
                });
                let first = (source as u32 * owner_rows).min(rows);
                let count = owner_rows.min(rows - first);
                gather[phase].push(Transfer {
                    source,
                    destination,
                    input: subspan(
                        outputs[source],
                        u64::from(first) * width * 2,
                        u64::from(count) * width * 2,
                    )?,
                    output: subspan(
                        outputs[destination],
                        u64::from(first) * width * 2,
                        u64::from(count) * width * 2,
                    )?,
                });
            }
        }
        Ok(Some(Self {
            first,
            rows,
            embedding,
            rounds,
            sends,
            scatter,
            sums,
            gather,
            footprint,
        }))
    }

    pub(super) fn audit(&self, compute: &[PreparedLaunch]) -> Result<(), String> {
        crate::shared_overlap::disjoint_copy_spans(&self.footprint, compute)
    }

    /// Run the call. With `producers`, each rank's main stream produces and
    /// rounds the interval it sends in each scatter phase in phase order,
    /// then its own, and each destination pulls on its lane once that
    /// source has rounded that interval. With `consumers`, each rank's
    /// gather copies move to its lane and its main stream runs the
    /// consumers of its own interval after its sum and of every other
    /// interval after that interval's copy.
    pub(super) fn execute(
        &self,
        instance: &Instance,
        streams: &[BundleStreamHandleV1],
        consumers: Option<&Consumers>,
        producers: Option<&Producers>,
    ) -> Result<(), DispatchError> {
        if streams.len() != 4 || instance.dma.len() != 4 {
            return Err(DispatchError::InvalidPlan);
        }
        let services = &instance.services;
        let resolve = |rank: usize, launch: &PreparedLaunch| {
            dispatch::resolve(
                &instance.symbols[rank],
                &launch.symbol,
                launch.plan,
                &launch.arguments,
                &launch.bindings,
            )
        };
        let rounds = self
            .rounds
            .iter()
            .enumerate()
            .map(|(rank, rounds)| rounds.iter().map(|r| resolve(rank, r)).collect())
            .collect::<Result<Vec<Vec<_>>, _>>()?;
        let sums = self
            .sums
            .iter()
            .enumerate()
            .map(|(rank, sum)| resolve(rank, sum))
            .collect::<Result<Vec<_>, _>>()?;
        let record = |stage: usize| -> Result<(), BundleAbiError> {
            for (rank, &stream) in streams.iter().enumerate() {
                services.record_event(instance.dma[rank].events[stage], stream)?;
            }
            Ok(())
        };
        let wait = |stage: usize| -> Result<(), BundleAbiError> {
            for &stream in streams {
                for rank in &instance.dma {
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
                    streams[transfer.destination],
                    instance.ranks[transfer.destination].0,
                    transfer.output,
                    instance.ranks[transfer.source].0,
                    transfer.input,
                )?;
            }
            Ok(())
        };
        let lanes = instance
            .dma
            .iter()
            .map(|rank| rank.lane)
            .collect::<Vec<_>>();
        if let Some(producers) = producers {
            self.produce(instance, streams, &lanes, producers)?;
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
            // A rank's three scatter copies all land in its own inbox from
            // its own stream, so its sum follows them without a cross-rank
            // barrier between permutations; the permutation order is
            // unchanged.
            for phase in 0..3 {
                copies(&self.scatter[phase], streams)?;
            }
        }
        for (rank, sum) in sums.iter().enumerate() {
            sum.submit(services, streams[rank])?;
        }
        // Every owner interval is summed before any rank gathers it.
        record(4)?;
        let Some(consumers) = consumers else {
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
        let resolved = consumers.resolve(instance)?;
        for &lane in &lanes {
            for rank in &instance.dma {
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
            for (rank, dma) in instance.dma.iter().enumerate() {
                services.record_event(dma.gathered[phase], lanes[rank])?;
            }
        }
        for phase in 0..3 {
            for (rank, &stream) in streams.iter().enumerate() {
                services.wait_event(stream, instance.dma[rank].gathered[phase])?;
                for launch in &resolved[rank][(rank + 3 - phase) % 4] {
                    launch.submit(services, stream)?;
                }
            }
        }
        // As above, with the copies' completion recorded on the lanes.
        for (rank, &lane) in lanes.iter().enumerate() {
            services.record_event(instance.dma[rank].events[7], lane)?;
        }
        wait(7)?;
        Ok(())
    }

    /// Each rank's interval producers and roundings on its main stream, and
    /// each destination's scatter pulls on its lane. Event 5 marks a rank's
    /// main stream at the call, events 1 to 3 its interval for each scatter
    /// phase, event 6 its three pulls. Ends with every rank's main stream
    /// after its own interval's producer and its pulls.
    fn produce(
        &self,
        instance: &Instance,
        streams: &[BundleStreamHandleV1],
        lanes: &[BundleStreamHandleV1],
        producers: &Producers,
    ) -> Result<(), DispatchError> {
        let services = &instance.services;
        let produced = producers.resolve(instance)?;
        let sends = self
            .sends
            .iter()
            .enumerate()
            .map(|(rank, owners)| {
                owners
                    .iter()
                    .map(|round| {
                        round
                            .as_ref()
                            .map(|round| {
                                dispatch::resolve(
                                    &instance.symbols[rank],
                                    &round.symbol,
                                    round.plan,
                                    &round.arguments,
                                    &round.bindings,
                                )
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        if produced.len() != 4 || sends.len() != 4 {
            return Err(DispatchError::InvalidPlan);
        }
        for (rank, &stream) in streams.iter().enumerate() {
            let dma = &instance.dma[rank];
            // A lane writes this rank's inbox only after the rank's main
            // stream has reached the call: its previous sums have read it.
            services.record_event(dma.events[5], stream)?;
            services.wait_event(lanes[rank], dma.events[5])?;
            for phase in 0..3 {
                let to = (rank + 1 + phase) % 4;
                let round = sends[rank][to].as_ref().ok_or(DispatchError::InvalidPlan)?;
                produced[rank][phase].submit(services, stream)?;
                round.submit(services, stream)?;
                services.record_event(dma.events[1 + phase], stream)?;
            }
            produced[rank][3].submit(services, stream)?;
        }
        // Each pull waits only for its own source's interval; a destination's
        // pulls keep the permutation order on its lane.
        for phase in 0..3 {
            for transfer in &self.scatter[phase] {
                let lane = lanes[transfer.destination];
                services.wait_event(lane, instance.dma[transfer.source].events[1 + phase])?;
                services.copy_peer_to_peer(
                    lane,
                    instance.ranks[transfer.destination].0,
                    transfer.output,
                    instance.ranks[transfer.source].0,
                    transfer.input,
                )?;
            }
        }
        for (rank, &stream) in streams.iter().enumerate() {
            let dma = &instance.dma[rank];
            services.record_event(dma.events[6], lanes[rank])?;
            services.wait_event(stream, dma.events[6])?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Calls {
    launches: Vec<PreparedLaunch>,
    flags: Vec<bool>,
}

fn copied(run: &Run, rank: usize, range: std::ops::Range<usize>) -> Calls {
    let mut out = Calls::default();
    for at in range {
        let launch = &run.programs[rank][at];
        out.launches.push(PreparedLaunch {
            symbol: launch.symbol.clone(),
            rows: launch.rows,
            plan: launch.plan,
            arguments: launch.arguments.clone(),
            bindings: launch.bindings.clone(),
        });
        out.flags.push(run.collectives[rank][at]);
    }
    out
}

pub(super) struct Plan {
    blocks: Vec<(Vec<Calls>, Call)>,
    tail: Vec<Calls>,
}

pub(super) fn eligible(instance: &Instance, run: &Run) -> bool {
    #[cfg(feature = "diagnostic-taps")]
    if instance.taps.text() {
        return false;
    }
    run.phase.prefill()
        && instance.dma.len() == 4
        && instance.ops.prefill_dma.enabled()
        && run.media.is_none()
        && run.inputs.request_ids.len() == 1
}

/// The full-schedule and two-part executors take DMA only when it was
/// selected explicitly, and never for a DSpark target; the default serves
/// the upper-skip executor alone.
pub(super) fn eligible_full_schedule(instance: &Instance, run: &Run) -> bool {
    eligible(instance, run)
        && instance.ops.prefill_dma.serves_full_schedule()
        && instance.dspark.is_none()
}

impl Plan {
    pub(super) fn prepare(instance: &Instance, run: &Run) -> Result<Option<Self>, String> {
        if !eligible_full_schedule(instance, run)
            || !instance.overlap.is_empty()
            || run.phase.rows() < 512
        {
            return Ok(None);
        }
        let stages = crate::program::stages(instance.config.shape.geometry, run.phase);
        let mut cursors = [0; 4];
        let mut blocks = Vec::new();
        for stage in stages
            .into_iter()
            .filter(|s| matches!(s, Stage::EmbeddingReduce | Stage::Reduce(..)))
        {
            let mut compute = Vec::new();
            let mut collectives = Vec::new();
            for (rank, cursor) in cursors.iter_mut().enumerate() {
                let begin = run.stages[rank]
                    .iter()
                    .position(|s| *s == stage)
                    .ok_or("DMA reduction stage missing")?;
                let end = begin
                    + run.stages[rank][begin..]
                        .iter()
                        .take_while(|s| **s == stage)
                        .count();
                let collective = (begin..end)
                    .find(|&at| run.collectives[rank][at])
                    .ok_or("DMA stage has no collective")?;
                if begin < *cursor || (stage != Stage::EmbeddingReduce && collective != begin) {
                    return Err("DMA reduction stage order changed".into());
                }
                compute.push(copied(run, rank, *cursor..collective));
                collectives.push(run.programs[rank][collective].symbol.as_str());
                *cursor = end;
            }
            // A collective the transport does not replace keeps the whole
            // step on the collectives.
            let Some(call) = Call::prepare(
                instance,
                0,
                run.phase.rows(),
                stage == Stage::EmbeddingReduce,
                &collectives,
                true,
            )?
            else {
                return Ok(None);
            };
            blocks.push((compute, call));
        }
        let tail = cursors
            .iter()
            .enumerate()
            .map(|(rank, &at)| copied(run, rank, at..run.programs[rank].len()))
            .collect();
        Ok(Some(Self { blocks, tail }))
    }

    pub(super) fn execute(&self, instance: &Instance) -> Result<(), DispatchError> {
        let streams = instance.ranks.iter().map(|r| r.2).collect::<Vec<_>>();
        let issue = |calls: &[Calls]| {
            instance.launch_eager(
                &streams,
                &calls
                    .iter()
                    .map(|c| c.launches.as_slice())
                    .collect::<Vec<_>>(),
                &calls.iter().map(|c| c.flags.as_slice()).collect::<Vec<_>>(),
            )
        };
        let _unwind = PoisonOnUnwind(&instance.collectives);
        let result = (|| {
            for (compute, reduction) in &self.blocks {
                issue(compute)?;
                reduction.execute(instance, &streams, None, None)?;
            }
            issue(&self.tail)
        })();
        if result.is_err() {
            instance.collectives.poison();
        }
        result
    }
}

mod consume;
mod produce;
pub(super) use consume::Consumers;
pub(super) use produce::Producers;

#[cfg(test)]
mod ownership_tests;
