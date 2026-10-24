//! Prepared ops and storage of allocated verify steps
//! (`model_deepseek41_contract::verify_allocation`): an instance whose host
//! allocates prepares the verify allocation and the allocated acceptance
//! at create and allocates their storage on every rank before the KV pool.
//! An instance whose host does not allocate holds none of it.
use super::results::{ITEMS, ROWS, STOP_CAPACITY};
use bundle_runtime::resources::{Held, allocate_span};
use kernel_core::backend::{Need, Prepared, View};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::acceptance::Acceptance;
use model_deepseek41_contract::own_ops::acceptance::allocated::AllocatedAcceptanceBuffers;
use model_deepseek41_contract::own_ops::allocation::{
    ITEM_WORDS, RULE_WORDS, VerifyAllocate, VerifyAllocateBuffers,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleMemoryPoolHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1 as Span,
    HostServicesV1,
};

#[allow(
    dead_code,
    reason = "allocated steps launch them once the drafter wires them"
)]
pub(crate) struct Allocation {
    pub op: VerifyAllocate,
    pub allocate: Prepared<VerifyAllocateBuffers<View>>,
    pub acceptance: Prepared<AllocatedAcceptanceBuffers<View>>,
    pub ranks: Vec<Memory>,
}

/// One rank's allocation inputs and outputs (`VerifyAllocateBuffers`); the
/// confidence scores and the compacted rows are the drafter's and the
/// target's own storage.
#[derive(Clone, Copy)]
#[allow(
    dead_code,
    reason = "allocated steps launch them once the drafter wires them"
)]
pub(crate) struct Memory {
    pub items: Span,
    pub rule: Span,
    pub live_items: Span,
    pub status: Span,
    pub verified: Span,
    pub starts: Span,
    pub rows: Span,
    pub expand: Span,
}

impl Allocation {
    /// The ops for steps of up to `verify_rows` reserved rows, the
    /// instance's verify rows, and as wide.
    pub fn new(own: &dyn OwnOps, vocab: u32, verify_rows: u32) -> Result<Self, String> {
        let op = VerifyAllocate {
            items: ITEMS,
            reserved_rows: verify_rows,
            width: verify_rows,
        };
        let allocate = own
            .dspark_allocate(&op, Need::rows(ITEMS))
            .map_err(|e| format!("DSpark verify allocation: {e:?}"))?;
        let acceptance = own
            .dspark_allocated_acceptance(
                &Acceptance {
                    vocab,
                    sample_rows: ROWS,
                    draft_items: ITEMS,
                    stop_ids: STOP_CAPACITY,
                },
                Need::rows(ITEMS),
            )
            .map_err(|e| format!("DSpark allocated acceptance: {e:?}"))?;
        Ok(Self {
            op,
            allocate,
            acceptance,
            ranks: Vec::new(),
        })
    }

    /// One more rank's storage.
    pub fn allocate(
        &mut self,
        services: &HostServicesV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        held: &mut Vec<Held>,
    ) -> Result<(), BundleAbiError> {
        let mut alloc = |bytes| allocate_span(services, pool, stream, bytes, held);
        let items = u64::from(self.op.items);
        self.ranks.push(Memory {
            items: alloc(items * ITEM_WORDS as u64 * 4)?,
            rule: alloc(RULE_WORDS as u64 * 4)?,
            live_items: alloc(4)?,
            status: alloc(4)?,
            verified: alloc(items * 4)?,
            starts: alloc(items * 4)?,
            rows: alloc(u64::from(self.op.width) * 4)?,
            expand: alloc(u64::from(self.op.reserved_rows) * 4)?,
        });
        Ok(())
    }
}
