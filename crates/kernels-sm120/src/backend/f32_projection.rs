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
    for &(rows, _) in entry.prefill_buckets() {
        if rows <= need.capacity {
            entry.entry_for_rows(rows).prefill_plan(rows)?;
        }
    }
    let mut info = OpInfo::new(
        f32_projection::IMPLEMENTATION,
        chosen.guarantees,
        entry
            .runtime_symbols()
            .into_iter()
            .map(|s| {
                let bytes = entry.runtime_shared_bytes(&s);
                (s, bytes)
            })
            .collect(),
    );
    if entry.uses_cached_weights() {
        if entry.uses_prefill_tiles() {
            entry.stage_plan([need.capacity, 1, 1])?;
        } else {
            entry.decode_plan(need.capacity.min(entry.decode_max_rows()))?;
            if entry.uses_fused_split() {
                entry.prepare_plan(need.capacity.min(entry.decode_max_rows()))?;
            }
            entry.stage_plan([entry.query().n.div_ceil(16), 1, 1])?;
            let elements = need
                .capacity
                .min(entry.decode_max_rows())
                .checked_mul(entry.query().n)
                .ok_or(Refused::Rule("F32 reduction grid overflows".into()))?;
            entry.stage_plan([elements.div_ceil(128), 1, 1])?;
        }
        info.workspace = entry.runtime_workspace_bytes(need.capacity);
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
        Ok(vec![Self::stage(
            &self.entry,
            "_weights",
            0,
            self.entry
                .stage_plan([self.entry.query().n.div_ceil(16), 1, 1])?,
            vec![
                Argument::DeviceBuffer(b.weights.codes.handle),
                Argument::DeviceBuffer(flags.handle),
            ],
        )])
    }
    fn launches(
        &self,
        b: &ProjectionBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        let entry = self.entry.entry_for_rows(rows);
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
        let q = entry.query();
        b.activation.covers(u64::from(rows) * u64::from(q.k) * 4)?;
        b.output.covers(u64::from(rows) * u64::from(q.n) * 4)?;
        b.weights.codes.covers(entry.weight_bytes())?;
        b.live_rows.covers(4)?;
        if entry.uses_cached_weights() {
            let ptr = |v: View| Argument::DeviceBuffer(v.handle);
            if self.entry.uses_bounded_reuse(rows) {
                let candidate = self.entry.bounded_reuse_candidate().expect("selected rows");
                let cached = &candidate.entry;
                let plans = candidate.plans(rows)?;
                let flags = self.weight_domain(b)?;
                workspace.covers(candidate.workspace(rows))?;
                let mut calls = vec![
                    Self::stage(
                        cached,
                        "_rows",
                        rows,
                        plans[0],
                        vec![
                            ptr(b.activation),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(workspace),
                        ],
                    ),
                    Self::stage(
                        cached,
                        "_decode",
                        rows,
                        plans[1],
                        vec![
                            ptr(b.activation),
                            ptr(b.weights.codes),
                            ptr(workspace),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(flags),
                        ],
                    ),
                    Self::stage(
                        cached,
                        "_reduce",
                        rows,
                        plans[2],
                        vec![
                            ptr(workspace),
                            ptr(b.output),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                        ],
                    ),
                ];
                calls[1].symbol = candidate.symbol(false);
                calls[2].symbol = candidate.symbol(true);
                return Ok(calls);
            }
            if let Some(candidate) = entry.decode_last(rows) {
                let flags = self.weight_domain(b)?;
                workspace.covers(candidate.workspace(rows))?;
                if [b.activation, b.weights.codes, b.output, b.live_rows, flags]
                    .iter()
                    .any(|v| v.handle == workspace.handle)
                {
                    return Err(Refusal("last-producer workspace aliases an operand"));
                }
                let plans = candidate.plans(rows)?;
                let mut calls = vec![
                    Self::stage(
                        &entry,
                        "_rows",
                        rows,
                        plans[0],
                        vec![
                            ptr(b.activation),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(workspace),
                        ],
                    ),
                    Self::stage(
                        &entry,
                        "_decode",
                        rows,
                        plans[1],
                        vec![
                            ptr(b.activation),
                            ptr(b.weights.codes),
                            ptr(workspace),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(flags),
                            ptr(b.output),
                        ],
                    ),
                ];
                calls[0].symbol = candidate.prepare_symbol();
                calls[1].symbol = candidate.symbol(false);
                return Ok(calls);
            }
            if entry.uses_prefill_tiles() {
                let flags = self.weight_domain(b)?;
                workspace.covers(entry.prefill_workspace_bytes(rows))?;
                return Ok(vec![
                    Self::stage(
                        &entry,
                        "_prepare",
                        rows,
                        entry.stage_plan([rows, 1, 1])?,
                        vec![
                            ptr(b.activation),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(workspace),
                        ],
                    ),
                    Self::stage(
                        &entry,
                        "",
                        rows,
                        entry.prefill_plan(rows)?,
                        vec![
                            ptr(b.activation),
                            ptr(b.weights.codes),
                            ptr(workspace),
                            ptr(b.output),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(flags),
                        ],
                    ),
                ]);
            }
            if entry.uses_fused_split() && rows <= entry.decode_max_rows() {
                let flags = self.weight_domain(b)?;
                workspace.covers(entry.decode_workspace_bytes(rows))?;
                return Ok(vec![
                    Self::stage(
                        &entry,
                        "_prepare",
                        rows,
                        entry.prepare_plan(rows)?,
                        vec![
                            ptr(b.activation),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(workspace),
                        ],
                    ),
                    Self::stage(
                        &entry,
                        "",
                        rows,
                        entry.decode_plan(rows)?,
                        vec![
                            ptr(b.activation),
                            ptr(b.weights.codes),
                            ptr(workspace),
                            ptr(b.output),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(flags),
                        ],
                    ),
                ]);
            }
            // SM120 at 250 W, 2026-10-19: split decode won through 32
            // measured rows, but cached prefill at 64 rows was 1.15-1.17x
            // slower for small-N projections. Keep 32 as the last measured
            // winning split bucket for unfused entries; their larger rows,
            // and fused entries above their decode maximum, use the
            // original symbol, plan and arguments.
            if !entry.uses_fused_split() && rows <= 32 {
                let flags = self.weight_domain(b)?;
                workspace.covers(entry.decode_workspace_bytes(rows))?;
                let mut launches = vec![
                    Self::stage(
                        &entry,
                        "_rows",
                        rows,
                        entry.stage_plan([rows, 1, 1])?,
                        vec![
                            ptr(b.activation),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(workspace),
                        ],
                    ),
                    Self::stage(
                        &entry,
                        "_decode",
                        rows,
                        entry.decode_plan(rows)?,
                        vec![
                            ptr(b.activation),
                            ptr(b.weights.codes),
                            ptr(workspace),
                            ptr(b.live_rows),
                            Argument::U32(rows),
                            ptr(flags),
                        ],
                    ),
                    Self::stage(
                        &entry,
                        "_reduce",
                        rows,
                        entry.stage_plan([
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
                ];
                if entry.uses_decode_reuse(rows) {
                    launches[1].symbol = f32_projection::reuse::symbol(&entry);
                    launches[1].plan = f32_projection::reuse::plan(&entry, rows)?;
                }
                return Ok(launches);
            }
        }
        Ok(vec![Launch {
            symbol: entry.baseline_symbol().into(),
            rows,
            plan: entry.plan(rows)?,
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
        entry: &Entry,
        suffix: &str,
        rows: u32,
        plan: kernel_core::contract::LaunchPlan,
        arguments: Vec<Argument>,
    ) -> Launch {
        Launch {
            symbol: entry.stage_symbol(suffix),
            rows,
            plan,
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        }
    }
}
