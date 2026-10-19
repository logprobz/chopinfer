use super::ring_gather_entries::{SYMBOL, THREADS};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;
use model_deepseek41_contract::own_ops::ring_gather::{CONTRACT, RingGather, RingGatherBuffers};

pub fn prepare(op: RingGather, need: Need) -> Result<Prepared<RingGatherBuffers<View>>, Refused> {
    op.validate(need)?;
    if need.capacity > model_deepseek41_contract::serving::DECODE_BATCH
        || op.bytes > u32::MAX - THREADS
    {
        return Err(Refused::Rule(
            "ring gather exceeds the served item or byte-loop extent".into(),
        ));
    }
    let info = OpInfo::new(
        CONTRACT,
        Invariance::any_rows(need.capacity),
        vec![(SYMBOL.into(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Gather {
        op,
        capacity: need.capacity,
        info,
    }))
}

struct Gather {
    op: RingGather,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<RingGatherBuffers<View>> for Gather {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &RingGatherBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal("ring gather item count exceeds capacity"));
        }
        b.source.covers(self.op.source_bytes())?;
        b.destination
            .covers(self.op.destination_bytes(counts.rows))?;
        b.source_slots.covers(u64::from(counts.rows) * 4)?;
        b.live_items.covers(4)?;
        if [b.source, b.source_slots, b.live_items]
            .iter()
            .any(|v| v.handle == b.destination.handle)
        {
            return Err(Refusal("ring gather destination aliases an input"));
        }
        Ok(vec![Launch {
            symbol: SYMBOL.into(),
            rows: counts.rows,
            plan: LaunchPlan {
                grid: [counts.rows, 1, 1],
                block: [THREADS, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1; 3],
                pdl: true,
            },
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(b.source.handle),
                Argument::DeviceBuffer(b.source_slots.handle),
                Argument::DeviceBuffer(b.live_items.handle),
                Argument::DeviceBuffer(b.destination.handle),
                Argument::U32(self.op.bytes),
                Argument::U32(self.op.source_stride),
                Argument::U32(self.op.destination_stride),
                Argument::U32(self.op.slots),
            ]),
            collective: false,
            persistent: false,
        }])
    }
}
