//! The verify allocation kernel (`deepseek41_allocate.cuh`): one block over
//! the step's decode-phase items.
use super::allocate_entries::{SYMBOL, THREADS};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;
use model_deepseek41_contract::own_ops::allocation::{
    BLOCK, CONTRACT, ITEM_WORDS, RULE_WORDS, VerifyAllocate, VerifyAllocateBuffers,
};

pub fn prepare(
    op: VerifyAllocate,
    need: Need,
) -> Result<Prepared<VerifyAllocateBuffers<View>>, Refused> {
    op.validate(need)?;
    let info = OpInfo::new(
        CONTRACT,
        Invariance::any_rows(need.capacity),
        vec![(SYMBOL.into(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Allocate { op, info }))
}

struct Allocate {
    op: VerifyAllocate,
    info: OpInfo,
}

impl PreparedOp<VerifyAllocateBuffers<View>> for Allocate {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    /// `Counts::rows` bounds the device live item count; the launch reads
    /// only the prepared capacities, so one graph serves every item count.
    fn launches(
        &self,
        b: &VerifyAllocateBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let op = self.op;
        if counts.rows == 0 || counts.rows > op.items {
            return Err(Refusal("verify allocation item count exceeds capacity"));
        }
        let items = u64::from(op.items);
        b.items.covers(items * ITEM_WORDS as u64 * 4)?;
        b.scores.covers(items * BLOCK as u64 * 4)?;
        b.rule.covers(RULE_WORDS as u64 * 4)?;
        b.live_items.covers(4)?;
        b.status.covers(4)?;
        b.verified.covers(items * 4)?;
        b.starts.covers(items * 4)?;
        b.rows.covers(u64::from(op.width) * 4)?;
        b.expand.covers(u64::from(op.reserved_rows) * 4)?;
        let reserved = u64::from(op.reserved_rows) * 4;
        for view in [
            b.token_ids,
            b.positions,
            b.token_batch,
            b.window_rows,
            b.index_visible,
        ] {
            view.covers(reserved)?;
        }
        b.q_start.covers(items * 4)?;
        // The text mask is optional: absent (zero bytes) or every reserved row.
        let text_rows = if b.text.bytes == 0 {
            0
        } else {
            b.text.covers(u64::from(op.reserved_rows))?;
            op.reserved_rows
        };
        let views = [
            b.items,
            b.scores,
            b.rule,
            b.live_items,
            b.status,
            b.verified,
            b.starts,
            b.rows,
            b.expand,
            b.token_ids,
            b.positions,
            b.token_batch,
            b.window_rows,
            b.index_visible,
            b.q_start,
            b.text,
        ];
        let bound: Vec<_> = views.iter().filter(|view| view.bytes > 0).collect();
        if bound
            .iter()
            .enumerate()
            .any(|(i, view)| bound[..i].iter().any(|other| other.handle == view.handle))
        {
            return Err(Refusal("verify allocation views alias"));
        }
        let mut arguments: Vec<_> = views
            .into_iter()
            .map(|view| Argument::DeviceBuffer(view.handle))
            .collect();
        arguments.extend(
            [op.items, op.items, op.reserved_rows, op.width, text_rows]
                .into_iter()
                .map(Argument::U32),
        );
        Ok(vec![Launch {
            symbol: SYMBOL.into(),
            rows: counts.rows,
            plan: LaunchPlan {
                grid: [1, 1, 1],
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
