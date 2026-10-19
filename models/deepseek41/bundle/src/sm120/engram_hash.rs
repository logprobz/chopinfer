use super::engram_hash_entries::{SYMBOL, THREADS};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;
use model_deepseek41_contract::own_ops::engram_hash::{CONTRACT, EngramHash, EngramHashBuffers};
use model_deepseek41_contract::serving::{DECODE_BATCH, MAX_VERIFY_ROWS};

pub fn prepare(op: EngramHash, need: Need) -> Result<Prepared<EngramHashBuffers<View>>, Refused> {
    op.validate(need)?;
    if op.stride > DECODE_BATCH * MAX_VERIFY_ROWS || op.items > DECODE_BATCH {
        return Err(Refused::Rule(
            "Engram device hash exceeds decode capacity".into(),
        ));
    }
    let info = OpInfo::new(
        CONTRACT,
        Invariance::any_rows(need.capacity),
        vec![(SYMBOL.into(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Hash {
        op,
        capacity: need.capacity,
        info,
    }))
}

struct Hash {
    op: EngramHash,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<EngramHashBuffers<View>> for Hash {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &EngramHashBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let op = self.op;
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal("Engram hash row count exceeds capacity"));
        }
        let rows = u64::from(counts.rows);
        b.tokens.covers(rows * 4)?;
        b.token_batch.covers(rows * 4)?;
        b.q_start.covers(u64::from(op.items) * 4)?;
        b.history
            .covers(u64::from(op.items) * u64::from(op.ngram - 1) * 8)?;
        b.text.covers(rows)?;
        b.constants.covers(op.constant_words() * 8)?;
        b.ids.covers(op.output_bytes())?;
        b.status.covers(rows * 4)?;
        let inputs = [
            b.tokens,
            b.token_batch,
            b.q_start,
            b.history,
            b.text,
            b.constants,
        ];
        if b.ids.handle == b.status.handle
            || inputs
                .iter()
                .any(|input| input.handle == b.ids.handle || input.handle == b.status.handle)
        {
            return Err(Refusal("Engram hash output aliases another operand"));
        }
        let mut arguments = inputs
            .into_iter()
            .chain([b.ids, b.status])
            .map(|view| Argument::DeviceBuffer(view.handle))
            .collect::<Vec<_>>();
        arguments.extend(
            [
                counts.rows,
                op.vocab,
                op.ngram,
                op.heads,
                op.layers,
                op.items,
                op.stride,
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
