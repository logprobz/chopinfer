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
        entry.symbols().into_iter().map(|s| (s, 0)).collect(),
    );
    if entry.uses_cached_weights() {
        entry.decode_plan(need.capacity.min(32))?;
        entry.stage_plan([entry.query().n.div_ceil(16), 1, 1])?;
        let elements = need
            .capacity
            .min(32)
            .checked_mul(entry.query().n)
            .ok_or(Refused::Rule("F32 reduction grid overflows".into()))?;
        entry.stage_plan([elements.div_ceil(128), 1, 1])?;
        info.workspace = entry.decode_workspace_bytes(need.capacity);
        info.bytes
            .push(("scratch.weight_domain".into(), entry.weight_domain_bytes()));
    }
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
    fn initialize(&self, b: &ProjectionBuffers<View>) -> Result<Vec<Launch>, Refusal> {
        if !self.entry.uses_cached_weights() {
            return Ok(Vec::new());
        }
        b.weights.codes.covers(self.entry.weight_bytes())?;
        let flags = self.weight_domain(b)?;
        Ok(vec![
            self.stage(
                "_weights",
                0,
                self.entry
                    .stage_plan([self.entry.query().n.div_ceil(16), 1, 1])?,
                vec![
                    Argument::DeviceBuffer(b.weights.codes.handle),
                    Argument::DeviceBuffer(flags.handle),
                ],
            ),
        ])
    }
    fn launches(
        &self,
        b: &ProjectionBuffers<View>,
        workspace: View,
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
        if self.entry.uses_cached_weights() {
            let flags = self.weight_domain(b)?;
            let ptr = |v: View| Argument::DeviceBuffer(v.handle);
            if rows <= 32 {
                workspace.covers(self.entry.decode_workspace_bytes(rows))?;
                return Ok(vec![
                    self.stage(
                        "_rows",
                        rows,
                        self.entry.stage_plan([rows, 1, 1])?,
                        vec![
                            ptr(b.activation),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(workspace),
                        ],
                    ),
                    self.stage(
                        "_decode",
                        rows,
                        self.entry.decode_plan(rows)?,
                        vec![
                            ptr(b.activation),
                            ptr(b.weights.codes),
                            ptr(workspace),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(flags),
                        ],
                    ),
                    self.stage(
                        "_reduce",
                        rows,
                        self.entry.stage_plan([
                            rows.checked_mul(q.n)
                                .ok_or(Refusal("F32 reduction grid overflows"))?
                                .div_ceil(128),
                            1,
                            1,
                        ])?,
                        vec![
                            ptr(workspace),
                            ptr(b.output),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                        ],
                    ),
                ]);
            }
            return Ok(vec![self.stage(
                "",
                rows,
                self.entry.plan(rows)?,
                vec![
                    ptr(b.activation),
                    ptr(b.weights.codes),
                    ptr(b.output),
                    ptr(b.live_rows),
                    Argument::U32(rows),
                    ptr(flags),
                ],
            )]);
        }
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

impl Projection {
    fn weight_domain(&self, b: &ProjectionBuffers<View>) -> Result<View, Refusal> {
        let flags = b.scratch.weight_domain.ok_or(Refusal(
            "F32 projection requires initialized weight-domain flags",
        ))?;
        flags.covers(self.entry.weight_domain_bytes())?;
        Ok(flags)
    }

    fn stage(
        &self,
        suffix: &str,
        rows: u32,
        plan: kernel_core::contract::LaunchPlan,
        arguments: Vec<Argument>,
    ) -> Launch {
        Launch {
            symbol: self.entry.stage_symbol(suffix),
            rows,
            plan,
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        }
    }
}
