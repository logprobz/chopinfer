//! Ordered DMA transport with an owner-local rank sum and BF16 gather. The
//! scatter carries the payload of the collective it replaces: F32 partials,
//! or each rank's partial rounded once to BF16 as the BF16-partial
//! contract's entries send it.
use crate::instance::Instance;
use bundle_runtime::acquire::RankHandles;
use bundle_runtime::dispatch::{self, DispatchError, Resolved};
pub(crate) use bundle_runtime::dma::OrderedRank as Rank;
use bundle_runtime::dma::{self as transport, subspan};
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::resources::Held;
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{
    BundleAbiError, BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1,
};

pub(crate) fn acquire(
    services: &HostServicesV1,
    ranks: &[RankHandles],
    width: u32,
    held: &mut Vec<Held>,
) -> Result<Vec<Rank>, BundleAbiError> {
    transport::acquire_ordered(
        services,
        ranks,
        u64::from(PREFILL_TOKENS) * u64::from(width) * 4,
        held,
    )
}

pub(crate) fn pushes(
    services: &HostServicesV1,
    _ranks: &[RankHandles],
    dma: &[Rank],
) -> Result<bool, BundleAbiError> {
    transport::pushes(services, dma)
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
    transport::drain(
        &instance.services,
        instance.ranks.iter().map(|rank| rank.2),
        &instance.dma,
        instance.side_lanes.iter().map(|lanes| lanes.side),
    )
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
    transport: transport::Call,
    sums: Vec<PreparedLaunch>,
    footprint: Vec<DeviceBufferSpanV1>,
}

/// Whether two launch lists issue the same calls
/// (`bundle_runtime::template::same_launch`).
pub(in crate::instance::execute) fn same_launches(
    a: &[PreparedLaunch],
    b: &[PreparedLaunch],
) -> Result<bool, String> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (a, b) in a.iter().zip(b) {
        if !bundle_runtime::template::same_launch(a, b)
            .map_err(|e| format!("{}: {e:?}", a.symbol))?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `launch` as one line of what `same_launch` compares: its symbol, rows,
/// plan, route and resolved arguments.
#[cfg(test)]
pub(in crate::instance::execute) fn describe_launch(launch: &PreparedLaunch) -> String {
    format!(
        "{} rows={} {:?} route={:?} {:?}",
        launch.symbol,
        launch.rows,
        launch.plan,
        launch.bindings.route(),
        dispatch::arguments(&launch.arguments, &launch.bindings),
    )
}

impl Call {
    /// The call as lines (`skip::Plan::describe`): its rows, footprint and
    /// owners, from which its transfers follow, then its roundings, sends
    /// and sums.
    #[cfg(test)]
    pub(super) fn describe(&self, lines: &mut Vec<String>) {
        lines.push(format!(
            "call first={} rows={} embedding={} footprint={:?} owners={:?}",
            self.first,
            self.rows,
            self.embedding,
            self.footprint,
            self.transport.owners()
        ));
        for (rank, rounds) in self.rounds.iter().enumerate() {
            for launch in rounds {
                lines.push(format!("round {rank} {}", describe_launch(launch)));
            }
        }
        for (rank, sends) in self.sends.iter().enumerate() {
            for (owner, send) in sends.iter().enumerate() {
                let send = send.as_ref().map(describe_launch);
                lines.push(format!("send {rank} {owner} {send:?}"));
            }
        }
        for (rank, sum) in self.sums.iter().enumerate() {
            lines.push(format!("sum {rank} {}", describe_launch(sum)));
        }
    }

    /// Whether `other` reduces the same rows with the same transfers,
    /// roundings, sends and sums.
    pub(super) fn same(&self, other: &Self) -> Result<bool, String> {
        if (self.first, self.rows, self.embedding) != (other.first, other.rows, other.embedding)
            || self.transport != other.transport
            || self.footprint != other.footprint
            || self.rounds.len() != other.rounds.len()
            || self.sends.len() != other.sends.len()
            || !same_launches(&self.sums, &other.sums)?
        {
            return Ok(false);
        }
        for (a, b) in self.rounds.iter().zip(&other.rounds) {
            if !same_launches(a, b)? {
                return Ok(false);
            }
        }
        for (a, b) in self.sends.iter().zip(&other.sends) {
            if a.len() != b.len() {
                return Ok(false);
            }
            for (a, b) in a.iter().zip(b) {
                let same = match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        same_launches(std::slice::from_ref(a), std::slice::from_ref(b))?
                    }
                    _ => false,
                };
                if !same {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Include all streams' transfer spans and generated arithmetic. The
    /// original program audit alone cannot see these owner-local launches.
    pub(in crate::instance::execute) fn audit_pending(
        &self,
        pending: &[DeviceBufferSpanV1],
        consumers: Option<&Consumers>,
        producers: Option<&Producers>,
    ) -> Result<(), String> {
        audit_spans(pending, &self.footprint)?;
        for rounds in &self.rounds {
            crate::shared_overlap::disjoint_copy_spans(pending, rounds)?;
        }
        for rounds in &self.sends {
            for launch in rounds.iter().flatten() {
                crate::shared_overlap::disjoint_copy_spans(pending, std::slice::from_ref(launch))?;
            }
        }
        crate::shared_overlap::disjoint_copy_spans(pending, &self.sums)?;
        if let Some(consumers) = consumers {
            consumers.audit_pending(pending)?;
        }
        if let Some(producers) = producers {
            producers.audit_pending(pending)?;
        }
        Ok(())
    }

    /// The reduction of rows `first..first + rows` that replaces the four
    /// ranks' `collectives`, carrying the payload they send. `None` where
    /// the rows or the collectives are outside the transport
    /// (`Kernels::prefill_dma_payload`).
    pub(super) fn prepare(
        instance: &Instance,
        first: u32,
        rows: u32,
        embedding: bool,
        collectives: &[&str],
    ) -> Result<Option<Self>, String> {
        if instance.dma.len() != 4 || !(512..=PREFILL_TOKENS).contains(&rows) {
            return Ok(None);
        }
        let payloads = collectives
            .iter()
            .map(|symbol| crate::Kernels::prefill_dma_payload(symbol))
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
            let inbox = subspan(instance.dma[rank].inbox(), 0, slot * 4 * bytes)?;
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
                    instance.dma[rank].inbox(),
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
            let mut sum = crate::Kernels::prefill_dma_sum(
                payload,
                own(input, 4)?,
                inbox,
                own(output, 2)?,
                slot,
                rank as u32,
                count,
            )?;
            // Generated DMA launches bypass normal lowering's route tags.
            // Attach metadata to the binding so only submitted sums count.
            if kernel_core::route::Metadata::enabled() {
                sum.bindings
                    .set_route(Some(kernel_core::route::RouteTag::new(
                        sum.symbol.clone(),
                        kernel_core::route::RouteClass::Fast,
                        "prefill_dma",
                        [512..=PREFILL_TOKENS],
                    )));
            }
            sums.push(sum);
            footprint.extend([input, output, inbox]);
            sent.push(outbox);
            outputs.push(output);
            inboxes.push(inbox);
        }
        let transport = transport::Call::with_payload(
            &inboxes,
            &sent,
            &outputs,
            rows,
            instance.config.shape.dims().hidden,
            match payload {
                crate::ops::DmaPayload::F32 => transport::Payload::F32,
                crate::ops::DmaPayload::Bf16 => transport::Payload::Bf16,
            },
        )?;
        Ok(Some(Self {
            first,
            rows,
            embedding,
            rounds,
            sends,
            transport,
            sums,
            footprint,
        }))
    }

    pub(super) fn execute(
        &self,
        instance: &Instance,
        streams: &[BundleStreamHandleV1],
        consumers: Option<&Consumers>,
        producers: Option<&Producers>,
    ) -> Result<(), DispatchError> {
        self.transport.execute_ordered(
            &instance.services,
            &instance.dma,
            streams,
            instance.dma_push,
            &Arithmetic {
                call: self,
                instance,
                consumers,
                producers,
            },
        )
    }
}

fn audit_spans(
    pending: &[DeviceBufferSpanV1],
    copies: &[DeviceBufferSpanV1],
) -> Result<(), String> {
    for &a in pending {
        let a_end = a
            .offset
            .checked_add(a.length)
            .ok_or("DMA Engram span overflow")?;
        for &b in copies {
            let b_end = b
                .offset
                .checked_add(b.length)
                .ok_or("DMA copy span overflow")?;
            if a.buffer == b.buffer
                && a.length != 0
                && b.length != 0
                && a.offset < b_end
                && b.offset < a_end
            {
                return Err("DMA transfer aliases pending Engram rows or IDs".into());
            }
        }
    }
    Ok(())
}

struct Arithmetic<'a> {
    call: &'a Call,
    instance: &'a Instance,
    consumers: Option<&'a Consumers>,
    producers: Option<&'a Producers>,
}

impl Arithmetic<'_> {
    fn resolve(&self, rank: usize, launch: &PreparedLaunch) -> Result<Resolved, DispatchError> {
        dispatch::resolve(
            &self.instance.symbols[rank],
            &launch.symbol,
            launch.plan,
            &launch.arguments,
            &launch.bindings,
        )
    }
}

impl transport::Arithmetic for Arithmetic<'_> {
    fn rounds(&self) -> Result<Vec<Vec<Resolved>>, DispatchError> {
        self.call
            .rounds
            .iter()
            .enumerate()
            .map(|(rank, rounds)| {
                rounds
                    .iter()
                    .map(|round| self.resolve(rank, round))
                    .collect()
            })
            .collect()
    }

    fn sums(&self) -> Result<Vec<Resolved>, DispatchError> {
        self.call
            .sums
            .iter()
            .enumerate()
            .map(|(rank, sum)| self.resolve(rank, sum))
            .collect()
    }

    fn producers(&self) -> Result<Option<transport::Producers>, DispatchError> {
        let Some(producers) = self.producers else {
            return Ok(None);
        };
        let launches = producers.resolve(self.instance)?;
        let rounds = self
            .call
            .sends
            .iter()
            .enumerate()
            .map(|(rank, owners)| {
                owners
                    .iter()
                    .map(|round| {
                        round
                            .as_ref()
                            .map(|round| self.resolve(rank, round))
                            .transpose()
                    })
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        Ok(Some(transport::Producers { launches, rounds }))
    }

    fn consumers(&self) -> Result<Option<transport::Consumers>, DispatchError> {
        self.consumers
            .map(|consumers| consumers.resolve(self.instance))
            .transpose()
    }
}

/// Whether the instance may send a text prefill of `requests` requests
/// through ordered DMA, before its reductions' own admission.
pub(super) fn admits(instance: &Instance, requests: usize) -> bool {
    #[cfg(feature = "diagnostic-taps")]
    if instance.taps.text() {
        return false;
    }
    instance.dma.len() == 4 && instance.ops.prefill_dma.enabled() && requests == 1
}

mod consume;
mod produce;
pub(super) use consume::Consumers;
pub(super) use produce::Producers;

#[cfg(test)]
mod ownership_tests;
