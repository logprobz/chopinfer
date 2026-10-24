//! The allocated acceptance kernel (`deepseek41_allocated_acceptance.cuh`):
//! one thread per item of an allocated verify step.
use super::allocated_acceptance_entries::{SYMBOL, THREADS};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;
use kernel_core::sampling::OUTPUT_BYTES;
use model_deepseek41_contract::own_ops::acceptance::allocated::{
    AllocatedAcceptanceBuffers, CONTRACT,
};
use model_deepseek41_contract::own_ops::acceptance::{Acceptance, BLOCK, ITEM_BYTES, RESULT_BYTES};

pub fn prepare(
    op: Acceptance,
    need: Need,
) -> Result<Prepared<AllocatedAcceptanceBuffers<View>>, Refused> {
    op.validate(need)?;
    let info = OpInfo::new(
        CONTRACT,
        Invariance::any_rows(need.capacity),
        vec![(SYMBOL.into(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Reduce {
        op,
        capacity: need.capacity,
        info,
    }))
}

struct Reduce {
    op: Acceptance,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<AllocatedAcceptanceBuffers<View>> for Reduce {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        b: &AllocatedAcceptanceBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal(
                "DSpark allocated acceptance item count exceeds capacity",
            ));
        }
        let op = self.op;
        b.items.covers(u64::from(counts.rows) * ITEM_BYTES as u64)?;
        b.verified.covers(u64::from(counts.rows) * 4)?;
        b.status.covers(4)?;
        b.proposals
            .covers(u64::from(op.draft_items) * BLOCK as u64 * 4)?;
        b.samples
            .covers(u64::from(op.sample_rows) * OUTPUT_BYTES as u64)?;
        b.stops.covers(u64::from(op.stop_ids) * 4)?;
        b.results
            .covers(u64::from(counts.rows) * RESULT_BYTES as u64)?;
        let inputs = [
            b.items,
            b.verified,
            b.status,
            b.proposals,
            b.samples,
            b.stops,
        ];
        if inputs.iter().any(|input| input.handle == b.results.handle) {
            return Err(Refusal(
                "DSpark allocated acceptance result aliases an input",
            ));
        }
        let mut arguments = inputs
            .into_iter()
            .chain([b.results])
            .map(|view| Argument::DeviceBuffer(view.handle))
            .collect::<Vec<_>>();
        arguments.extend(
            [
                counts.rows,
                op.vocab,
                op.sample_rows,
                op.draft_items,
                op.stop_ids,
            ]
            .into_iter()
            .map(Argument::U32),
        );
        Ok(vec![Launch {
            symbol: SYMBOL.into(),
            rows: counts.rows,
            plan: LaunchPlan {
                grid: [counts.rows.div_ceil(THREADS), 1, 1],
                block: [THREADS, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1; 3],
                pdl: true,
            },
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        }])
    }
}
