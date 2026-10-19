//! Deferred adapter for device acceptance. Serving keeps the existing path
//! until the shared executor supplies device drafts and ordered sampler rows.
use kernel_core::backend::{Need, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::sampling::{OUTPUT_BYTES, STATUS_EMPTY, STATUS_SAMPLED};
use model_deepseek41_contract::own_ops::{OwnOps, acceptance::*};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Identity {
    pub request: u64,
    pub slot: u32,
    pub lease: u64,
}
impl Identity {
    pub fn of(request: u64, context: &super::state::Context) -> Self {
        Self {
            request,
            slot: context.slot,
            lease: context.lease(),
        }
    }
}

pub struct Request {
    pub identity: Identity,
    pub k: u32,
    pub remaining: u64,
    pub terminal_stop_ids: Vec<u32>,
    pub top_logprobs: u32,
    pub constrained: bool,
    pub history_penalties: bool,
}

pub struct Batch {
    pub epoch: u64,
    pub op: Acceptance,
    pub items: Vec<Item>,
    pub stops: Vec<u32>,
    identities: Vec<Identity>,
    top_counts: Vec<u32>,
}
impl Batch {
    pub fn item_bytes(&self) -> Vec<u8> {
        self.items.iter().flat_map(|item| item.encode()).collect()
    }
    pub fn stop_bytes(&self) -> Vec<u8> {
        self.stops.iter().flat_map(|id| id.to_le_bytes()).collect()
    }
    pub fn prepare(&self, own: &dyn OwnOps) -> Result<Prepared<AcceptanceBuffers<View>>, Refused> {
        own.dspark_acceptance(&self.op, Need::rows(self.items.len() as u32))
    }
}

#[derive(Debug)]
pub struct Completed {
    pub identity: Identity,
    pub decision: Decision,
    /// Original sampler records for emitted tokens, with every bit retained.
    pub records: Vec<u8>,
}

#[derive(Default)]
pub struct Adapter {
    epoch: u64,
    pending: Option<Batch>,
}
impl Adapter {
    pub fn begin(&mut self, vocab: u32, requests: &[Request]) -> Result<&Batch, Refusal> {
        if self.pending.is_some() {
            return Err(Refusal("DSpark acceptance is already pending"));
        }
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(Refusal("DSpark acceptance epoch exhausted"))?;
        let mut batch = Batch {
            epoch,
            op: Acceptance {
                vocab,
                sample_rows: 0,
                draft_items: 0,
                stop_ids: 0,
            },
            items: Vec::new(),
            stops: Vec::new(),
            identities: Vec::new(),
            top_counts: Vec::new(),
        };
        if requests.is_empty() || requests.len() > 32 {
            return Err(Refusal("DSpark acceptance item count is invalid"));
        }
        for request in requests {
            if request.constrained || request.history_penalties {
                return Err(Refusal(
                    "DSpark device acceptance requires unconstrained identity-penalty sampling",
                ));
            }
            if request.k > 5
                || request.remaining == 0
                || request.top_logprobs > 20
                || request.terminal_stop_ids.iter().any(|&id| id >= vocab)
                || request
                    .terminal_stop_ids
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || batch.identities.iter().any(|id| {
                    id.request == request.identity.request || id.slot == request.identity.slot
                })
            {
                return Err(Refusal("DSpark acceptance request metadata is invalid"));
            }
            let stop_count = u32::try_from(request.terminal_stop_ids.len())
                .map_err(|_| Refusal("DSpark stop IDs exceed U32"))?;
            let next_stops = batch
                .op
                .stop_ids
                .checked_add(stop_count)
                .ok_or(Refusal("DSpark stop IDs overflow"))?;
            batch.items.push(Item {
                epoch,
                remaining: request.remaining,
                first_sample: batch.op.sample_rows,
                draft_item: if request.k == 0 {
                    NONE
                } else {
                    batch.op.draft_items
                },
                k: request.k,
                stop_offset: batch.op.stop_ids,
                stop_count,
                reserved: 0,
            });
            batch.op.sample_rows += request.k + 1;
            batch.op.draft_items += u32::from(request.k > 0);
            batch.op.stop_ids = next_stops;
            batch.stops.extend_from_slice(&request.terminal_stop_ids);
            batch.identities.push(request.identity);
            batch.top_counts.push(request.top_logprobs.min(vocab));
        }
        batch.op.validate(Need::rows(requests.len() as u32))?;
        self.epoch = epoch;
        self.pending = Some(batch);
        Ok(self.pending.as_ref().expect("stored batch"))
    }
    pub fn cancel(&mut self) {
        self.pending = None;
    }

    /// Consume one final bounded result packet. Full sampler records remain
    /// in this packet so rejected suffixes retain the shared host's validation.
    /// A stale completion cannot consume a newer cycle or a reused slot.
    pub fn finish(
        &mut self,
        epoch: u64,
        current: &[Identity],
        results: &[u8],
        samples: &[u8],
    ) -> Result<Vec<Completed>, Refusal> {
        let batch = self
            .pending
            .as_ref()
            .ok_or(Refusal("DSpark acceptance has no pending cycle"))?;
        if epoch != batch.epoch {
            return Err(Refusal("stale DSpark acceptance epoch"));
        }
        let batch = self.pending.take().expect("pending cycle");
        if current != batch.identities {
            return Err(Refusal("DSpark acceptance slot lease changed"));
        }
        if results.len() != batch.items.len() * RESULT_BYTES
            || samples.len() != batch.op.sample_rows as usize * OUTPUT_BYTES
        {
            return Err(Refusal("DSpark acceptance packet size differs"));
        }
        let mut completed = Vec::with_capacity(batch.items.len());
        for (index, item) in batch.items.iter().enumerate() {
            let decision = Decision::decode(&results[index * RESULT_BYTES..][..RESULT_BYTES])?;
            if decision.epoch != epoch
                || decision.reserved != 0
                || decision.flags & !(TERMINAL | REJECTED) != 0
            {
                return Err(Refusal("DSpark acceptance result identity is invalid"));
            }
            // Preserve full-view membership and finite-value checks before
            // returning any item's counts. EMPTY bytes are unspecified.
            for row in item.first_sample..=item.first_sample + item.k {
                let sample = &samples[row as usize * OUTPUT_BYTES..][..OUTPUT_BYTES];
                match u32_at(sample, 4) {
                    STATUS_EMPTY => continue,
                    STATUS_SAMPLED => {}
                    _ => return Err(Refusal("DSpark sampler view failed")),
                }
                if u32_at(sample, 0) >= batch.op.vocab
                    || !f64::from_bits(u64_at(sample, 8)).is_finite()
                    || (0..batch.top_counts[index] as usize).any(|top| {
                        u32_at(sample, 16 + top * 4) >= batch.op.vocab
                            || !f64::from_bits(u64_at(sample, 96 + top * 8)).is_finite()
                    })
                {
                    return Err(Refusal("DSpark sampler payload is invalid"));
                }
            }
            if decision.failure == EMPTY {
                if decision.accepted != 0
                    || decision.kept != 0
                    || decision.flags != 0
                    || decision.failure_row < item.first_sample
                    || decision.failure_row > item.first_sample + item.k
                    || u32_at(samples, decision.failure_row as usize * OUTPUT_BYTES + 4)
                        != STATUS_EMPTY
                {
                    return Err(Refusal("DSpark empty-distribution result is invalid"));
                }
            } else if decision.failure != OK {
                return Err(Refusal("DSpark acceptance device validation failed"));
            } else {
                let kept = decision.kept;
                if kept == 0
                    || kept > item.k + 1
                    || u64::from(kept) > item.remaining
                    || decision.accepted > item.k
                    || decision.failure_row != NONE
                    || kept < decision.accepted
                    || kept > decision.accepted + 1
                    || (kept == decision.accepted
                        && (decision.flags & REJECTED != 0
                            || (decision.flags & TERMINAL == 0
                                && u64::from(kept) != item.remaining)))
                    || (kept == decision.accepted + 1
                        && (decision.flags & REJECTED != 0) != (decision.accepted < item.k))
                {
                    return Err(Refusal("DSpark acceptance counts are invalid"));
                }
                for row in item.first_sample..item.first_sample + kept {
                    if u32_at(samples, row as usize * OUTPUT_BYTES + 4) != STATUS_SAMPLED {
                        return Err(Refusal("DSpark acceptance consumed an empty row"));
                    }
                }
                let last = u32_at(
                    samples,
                    (item.first_sample + kept - 1) as usize * OUTPUT_BYTES,
                );
                let stops = &batch.stops[item.stop_offset as usize..][..item.stop_count as usize];
                if (decision.flags & TERMINAL != 0) != stops.binary_search(&last).is_ok() {
                    return Err(Refusal("DSpark acceptance stop flag differs"));
                }
            }
            let first = item.first_sample as usize * OUTPUT_BYTES;
            completed.push(Completed {
                identity: batch.identities[index],
                decision,
                records: samples[first..first + decision.kept as usize * OUTPUT_BYTES].to_vec(),
            });
        }
        Ok(completed)
    }
}

#[cfg(test)]
mod tests;
