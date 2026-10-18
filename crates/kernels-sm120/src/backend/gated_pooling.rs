//! Preparation and role validation for the F32 pooling family. This module
//! takes only the neutral pooling records; registration owns no arithmetic.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, RecurrentPhase, Refusal, Validate};
use kernel_core::gated_pooling::{GatedPoolingBuffers, GatedPoolingQuery};
use kernel_core::properties::{Anchors, Invariance};

use super::select::{Candidate, choose};
use crate::gated_pooling::Entry;
use crate::gated_pooling::F32Norm;
use kernel_core::rmsnorm::{RmsNormBuffers, RmsNormRequirement};

pub(super) fn norm(
    entries: &[(F32Norm, String)],
    query: &RmsNormRequirement,
    need: Need,
) -> Result<Prepared<RmsNormBuffers<View>>, Refused> {
    query.validate()?;
    let guarantees = Invariance::any_rows(need.capacity);
    let entry = choose(
        "F32 RMSNorm",
        entries
            .iter()
            .filter(|(norm, _)| norm.requirement() == query)
            .map(|entry| {
                Candidate::new(entry, &entry.1, entry.1.clone(), guarantees)
                    .capacity(entry.0.capacity)
            }),
        need,
    )?
    .entry;
    let info = OpInfo::new(
        "sm120.gated-pooling.f32-rmsnorm.tree128.v1",
        guarantees,
        vec![(entry.1.clone(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Norm {
        norm: entry.0,
        capacity: need.capacity,
        symbol: entry.1.clone(),
        info,
    }))
}

struct Norm {
    norm: F32Norm,
    capacity: u32,
    symbol: String,
    info: OpInfo,
}

impl PreparedOp<RmsNormBuffers<View>> for Norm {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &RmsNormBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let grid = counts.rows.max(counts.padded_rows);
        if counts.rows == 0 || grid > self.capacity || b.residual.is_some() {
            return Err(Refusal("F32 RMSNorm needs bounded rows and no residual"));
        }
        let q = self.norm.requirement();
        let row = u64::from(q.width);
        b.x.covers(u64::from(counts.rows) * row * 4)?;
        b.y.covers(u64::from(counts.rows) * row * 4)?;
        b.weight.covers(row * 2)?;
        b.live_rows.covers(4)?;
        Ok(vec![Launch {
            symbol: self.symbol.clone(),
            rows: counts.rows,
            plan: self.norm.plan(grid)?,
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(b.x.handle),
                Argument::DeviceBuffer(b.weight.handle),
                Argument::DeviceBuffer(b.y.handle),
                Argument::DeviceBuffer(b.live_rows.handle),
                Argument::U32(q.epsilon_bits),
            ]),
            collective: false,
            persistent: false,
        }])
    }
}

/// Counts read by the launches are `rows` and `items`. Device live rows
/// are at most `rows`, item IDs are below `items`, and each item occupies
/// one contiguous run. Its non-hole positions are consecutive. Source slots
/// are distinct across items; verify checkpoint slots are distinct from all
/// source slots and from each other. IDs are bounded by `tail.entries`.
pub(super) fn prepare(
    entries: &[Entry],
    query: GatedPoolingQuery,
    need: Need,
) -> Result<Prepared<GatedPoolingBuffers<View>>, Refused> {
    query.validate()?;
    if need.capacity == 0 || need.capacity > query.capacity {
        return Err(Refused::Rule(
            "pooling need exceeds the query capacity".into(),
        ));
    }
    let matches = |entry: &&Entry| {
        let offered = entry.query();
        offered.capacity >= need.capacity
            && GatedPoolingQuery {
                capacity: query.capacity,
                ..offered
            } == query
    };
    let guarantees = match query.phase {
        RecurrentPhase::Prefill => {
            Invariance::PREFILL.with_anchors(Anchors::Aligned { granule: 2 })
        }
        RecurrentPhase::Decode => Invariance::NONE.with_anchors(Anchors::None),
        RecurrentPhase::Verify { .. } => {
            Invariance::one_row(need.capacity).with_anchors(Anchors::None)
        }
    };
    let entry = choose(
        "F32 gated pooling",
        entries.iter().filter(matches).map(|entry| {
            Candidate::new(
                entry,
                entry.symbols()[0],
                format!("{:?}", entry.symbols()),
                guarantees,
            )
            .capacity(entry.query().capacity)
        }),
        need,
    )?
    .entry;
    let mut info = OpInfo::new(
        "sm120.gated-pooling.pair-f32-rmsnorm.v1",
        guarantees,
        entry.symbols().map(|s| (s.to_owned(), 0)).to_vec(),
    );
    info.workspace = entry.workspace_bytes(need.capacity)?;
    need.check(&info)?;
    Ok(Box::new(Pooling {
        entry: entry.clone(),
        capacity: need.capacity,
        info,
    }))
}

struct Pooling {
    entry: Entry,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<GatedPoolingBuffers<View>> for Pooling {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GatedPoolingBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity || counts.items == 0 {
            return Err(Refusal(
                "pooling needs rows within capacity and a nonzero item count",
            ));
        }
        let query = self.entry.query();
        let width = u64::from(query.width);
        let rows = u64::from(counts.rows);
        buffers.key.covers(rows * width * 4)?;
        buffers.gate.covers(rows * width * 4)?;
        buffers.norm_weight.covers(width * 2)?;
        buffers.positions.covers(rows * 4)?;
        buffers.row_batch.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        buffers.output.covers(rows * width * 4)?;
        buffers.output_positions.covers(rows * 4)?;
        workspace.covers(self.info.workspace)?;
        let tail = buffers.tail;
        if tail.entries == 0 {
            return Err(Refusal("pooling needs a nonempty F32 tail pool"));
        }
        tail.state
            .covers(u64::from(tail.entries) * query.tail_state_bytes())?;
        tail.slot_ids.covers(u64::from(counts.items) * 4)?;
        let checkpoints = if let RecurrentPhase::Verify { max_rows } = query.phase {
            let ids = tail
                .checkpoint_ids
                .ok_or(Refusal("pooling verify needs checkpoint IDs"))?;
            ids.covers(u64::from(counts.items) * u64::from(max_rows) * 4)?;
            Some(ids.handle)
        } else {
            None
        };
        let plan = self.entry.plan(counts.rows)?;
        let pointer = |view: View| Argument::DeviceBuffer(view.handle);
        let mut pool_arguments = vec![
            pointer(buffers.key),
            pointer(buffers.gate),
            pointer(buffers.positions),
            pointer(buffers.row_batch),
            pointer(buffers.live_rows),
            pointer(tail.slot_ids),
        ];
        pool_arguments.extend(checkpoints.map(Argument::DeviceBuffer));
        pool_arguments.extend([
            pointer(tail.state),
            pointer(workspace),
            pointer(buffers.output_positions),
            Argument::U32(counts.items),
            Argument::U32(tail.entries),
        ]);
        let arguments = [
            ArgumentList(pool_arguments),
            ArgumentList(vec![
                pointer(workspace),
                pointer(buffers.norm_weight),
                pointer(buffers.output_positions),
                pointer(buffers.output),
                pointer(buffers.live_rows),
                Argument::U32(query.norm_eps_bits),
            ]),
        ];
        arguments.iter().try_for_each(Validate::validate)?;
        Ok(self
            .entry
            .symbols()
            .into_iter()
            .zip(arguments)
            .map(|(symbol, arguments)| Launch {
                symbol: symbol.to_owned(),
                rows: counts.rows,
                plan,
                arguments,
                collective: false,
                persistent: false,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::gated_pooling::GatedPoolingState;

    fn query(phase: RecurrentPhase) -> GatedPoolingQuery {
        GatedPoolingQuery {
            group: 2,
            width: 512,
            norm_eps_bits: 1e-20f32.to_bits(),
            phase,
            capacity: 12,
        }
    }

    fn buffers() -> GatedPoolingBuffers<View> {
        let view = |handle| View {
            handle,
            bytes: 65536,
        };
        GatedPoolingBuffers {
            key: view(1),
            gate: view(2),
            norm_weight: view(3),
            positions: view(4),
            row_batch: view(5),
            live_rows: view(6),
            tail: GatedPoolingState {
                state: view(7),
                entries: 16,
                slot_ids: view(8),
                checkpoint_ids: Some(view(9)),
            },
            output: view(10),
            output_positions: view(11),
        }
    }

    #[test]
    fn prepares_every_phase_and_binds_f32_workspace_between_stages() {
        let mut prepared = 0;
        for phase in [
            RecurrentPhase::Prefill,
            RecurrentPhase::Decode,
            RecurrentPhase::Verify { max_rows: 6 },
        ] {
            let q = query(phase);
            let entries = [Entry::new(q, "chop_pool_test").unwrap()];
            let op = prepare(&entries, q, Need::rows(12)).unwrap();
            let workspace = View {
                handle: 99,
                bytes: op.info().workspace,
            };
            let launches = op
                .launches(&buffers(), workspace, Counts::rows(12).with_items(2, 0))
                .unwrap();
            assert_eq!(launches.len(), 2);
            let pooled_argument = if matches!(phase, RecurrentPhase::Verify { .. }) {
                8
            } else {
                7
            };
            assert_eq!(
                launches[0].arguments.0[pooled_argument],
                Argument::DeviceBuffer(99)
            );
            assert_eq!(launches[1].arguments.0[0], Argument::DeviceBuffer(99));
            assert_eq!(op.info().workspace, 12 * 512 * 4);
            assert!(launches.iter().all(|l| l.plan.pdl));
            prepared += 1;
        }
        assert_eq!(prepared, 3);
    }

    #[test]
    fn rejects_short_tail_and_missing_or_short_per_item_checkpoints() {
        let q = query(RecurrentPhase::Verify { max_rows: 6 });
        let entries = [Entry::new(q, "chop_pool_test").unwrap()];
        let op = prepare(&entries, q, Need::rows(12)).unwrap();
        let workspace = View {
            handle: 99,
            bytes: op.info().workspace,
        };
        let counts = Counts::rows(12).with_items(2, 0);
        let mut b = buffers();
        b.tail.state.bytes -= 1;
        assert!(op.launches(&b, workspace, counts).is_err());
        b = buffers();
        b.tail.checkpoint_ids = None;
        assert!(op.launches(&b, workspace, counts).is_err());
        b.tail.checkpoint_ids = Some(View {
            handle: 9,
            bytes: 47,
        });
        assert!(op.launches(&b, workspace, counts).is_err());
        b = buffers();
        b.tail.slot_ids.bytes = 7;
        assert!(op.launches(&b, workspace, counts).is_err());
    }

    #[test]
    fn selection_refuses_missing_geometry_and_ambiguous_symbols() {
        let q = query(RecurrentPhase::Prefill);
        let a = Entry::new(q, "chop_pool_a").unwrap();
        let b = Entry::new(q, "chop_pool_b").unwrap();
        assert!(prepare(&[a.clone(), b.clone()], q, Need::rows(12)).is_err());
        assert!(prepare(&[b, a], q, Need::rows(12)).is_err());
        assert!(Entry::new(GatedPoolingQuery { group: 4, ..q }, "chop_pool").is_err());
        assert!(Entry::new(GatedPoolingQuery { width: 256, ..q }, "chop_pool").is_err());
        assert!(prepare(&[], q, Need::rows(12)).is_err());
    }

    #[test]
    fn ratio_one_uses_plain_f32_norm_without_state_or_intermediate_rounding() {
        let norm = crate::gated_pooling::ratio_one_norm(512, 1e-20, 12).unwrap();
        let requirement = norm.requirement();
        assert_eq!(requirement.kind, kernel_core::contract::RmsNormKind::Plain);
        assert!(!requirement.fused_residual);
        assert_eq!(requirement.input, kernel_core::contract::DType::F32);
        assert_eq!(requirement.output, kernel_core::contract::DType::F32);
        assert!(!crate::row::rmsnorm::RmsNorm::numerics(requirement.kind).normalized_bf16);
        assert_eq!(norm.plan(12).unwrap().workspace_bytes, 0);
        assert!(crate::gated_pooling::ratio_one_norm(256, 1e-20, 12).is_err());
    }
}
