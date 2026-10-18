//! Authenticated F32 projection offers and checked buffer bindings.

use super::select::{Candidate, choose};
use crate::f32_projection::{self, Entry};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, Refusal};
use kernel_core::precision::buffers::ProjectionBuffers;
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::Binding;
use kernel_core::precision::{OperationSpec, OperationUse};
use kernel_core::properties::Invariance;
use std::collections::BTreeSet;

pub struct F32ProjectionOffers {
    entries: Vec<Entry>,
    inventory: CapabilityInventory,
}

impl F32ProjectionOffers {
    pub fn new(
        entries: Vec<Entry>,
        available: BTreeSet<String>,
        dependency: String,
    ) -> Result<Self, Refusal> {
        let inventory = CapabilityInventory::new(
            f32_projection::capabilities(&entries),
            available,
            dependency,
        )
        .map_err(|_| Refusal("invalid F32 projection inventory or dependency digest"))?;
        Ok(Self { entries, inventory })
    }
}

pub(super) fn prepare(
    offers: Option<&F32ProjectionOffers>,
    op: &OperationSpec,
    usage: OperationUse,
    need: Need,
) -> Result<Binding<Prepared<ProjectionBuffers<View>>>, Refused> {
    let offers = offers.ok_or(Refused::Unsupported)?;
    if need.capacity == 0 || need.capacity > usage.capacity {
        return Err(Refused::Rule("F32 projection need exceeds its use".into()));
    }
    let mut bound = Vec::new();
    loop {
        let next = offers.inventory.bind_matching(op, usage, |cap| {
            !bound.iter().any(|b: &Binding<usize>| {
                std::ptr::eq(cap, &offers.inventory.document().operations[b.handle])
            })
        });
        match next {
            Ok(b) => bound.push(b),
            Err(reason) if bound.is_empty() => return Err(Refused::Rule(reason)),
            Err(_) => break,
        }
    }
    let chosen = choose(
        "exact F32 projection",
        bound.iter().enumerate().map(|(index, b)| {
            let entry = &offers.entries[b.handle];
            Candidate::new(
                index,
                entry.symbol(),
                format!("{entry:?} {:?}", b.identity),
                Invariance::any_rows(entry.query().m_capacity),
            )
            .capacity(entry.query().m_capacity)
        }),
        need,
    )?;
    let binding = bound.swap_remove(chosen.entry);
    let entry = offers.entries[binding.handle].clone();
    let mut info = OpInfo::new(
        f32_projection::IMPLEMENTATION,
        chosen.guarantees,
        vec![(entry.symbol().into(), 0)],
    );
    info.weight_layouts = vec![(
        "weights".into(),
        kernel_core::precision::WeightLayout::row_major(),
    )];
    need.check(&info)?;
    Ok(Binding {
        identity: binding.identity,
        handle: Box::new(Projection {
            entry,
            capacity: need.capacity,
            info,
        }),
    })
}

struct Projection {
    entry: Entry,
    capacity: u32,
    info: OpInfo,
}
impl PreparedOp<ProjectionBuffers<View>> for Projection {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &ProjectionBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        if rows == 0
            || rows > self.capacity
            || b.residual.is_some()
            || b.routing.is_some()
            || b.activation_global.is_some()
            || b.weights.local_scales.is_some()
            || b.weights.global_scales.is_some()
            || b.row_offset != 0
            || b.shard_rows != 0
        {
            return Err(Refusal(
                "F32 projection requires bounded dense rows and unscaled operands",
            ));
        }
        let q = self.entry.query();
        b.activation.covers(u64::from(rows) * u64::from(q.k) * 4)?;
        b.output.covers(u64::from(rows) * u64::from(q.n) * 4)?;
        b.weights.codes.covers(self.entry.weight_bytes())?;
        b.live_rows.covers(4)?;
        Ok(vec![Launch {
            symbol: self.entry.symbol().into(),
            rows,
            plan: self.entry.plan(rows)?,
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(b.activation.handle),
                Argument::DeviceBuffer(b.weights.codes.handle),
                Argument::DeviceBuffer(b.output.handle),
                Argument::DeviceBuffer(b.live_rows.handle),
                Argument::U32(rows),
            ]),
            collective: false,
            persistent: false,
        }])
    }
}
