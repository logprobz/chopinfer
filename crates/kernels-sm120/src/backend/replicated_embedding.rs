use super::select::{Candidate, choose};
use crate::row::replicated_embedding::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::replicated_embedding::{
    ReplicatedEmbedding as Query, ReplicatedEmbeddingBuffers as Buffers,
};

pub(super) fn prepare(
    entries: &[Entry],
    query: &Query,
    rank: u32,
    need: Need,
) -> Result<Prepared<Buffers<View>>, Refused> {
    query.validate()?;
    if rank >= query.ranks {
        return Err(Refusal("embedding rank exceeds topology").into());
    }
    let chosen = choose(
        "replicated embedding",
        entries.iter().filter(|e| e.query == *query).map(|e| {
            Candidate::new(
                e.clone(),
                &e.symbol,
                format!("{e:?}"),
                Invariance::any_rows(e.capacity),
            )
            .capacity(e.capacity)
        }),
        need,
    )?;
    chosen.entry.plan(need.capacity)?;
    let info = OpInfo::new(
        "sm120.embedding.peer-ordered-sum.v1",
        chosen.guarantees,
        vec![(chosen.entry.symbol.clone(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Op {
        entry: chosen.entry,
        rank,
        capacity: need.capacity,
        info,
    }))
}
struct Op {
    entry: Entry,
    rank: u32,
    capacity: u32,
    info: OpInfo,
}
impl PreparedOp<Buffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(&self, b: &Buffers<View>, _: View, counts: Counts) -> Result<Vec<Launch>, Refusal> {
        let q = self.entry.query;
        if counts.rows == 0
            || counts.rows > self.capacity
            || counts.padded_rows > counts.rows
            || b.tables.len() != q.ranks as usize
        {
            return Err(Refusal("embedding row or peer count differs"));
        }
        let n = u64::from(counts.rows) * u64::from(q.width);
        let roles = [b.ids, b.shard, b.wide, b.output, b.expanded, b.live_rows];
        for (v, bytes) in roles.into_iter().zip([
            u64::from(counts.rows) * 4,
            n * 2,
            n * 4,
            n * 2,
            n * 2 * u64::from(q.streams),
            4,
        ]) {
            v.covers(bytes)?;
        }
        for t in &b.tables {
            t.covers(u64::from(q.vocab / q.ranks) * u64::from(q.width) * 2)?;
        }
        let all: Vec<_> = roles.iter().chain(&b.tables).collect();
        for (i, a) in all.iter().enumerate() {
            if all[..i].iter().any(|v| v.handle == a.handle) {
                return Err(Refusal("embedding buffers alias"));
            }
        }
        let d = |v: View| Argument::DeviceBuffer(v.handle);
        let arguments = ArgumentList(vec![
            Argument::DevicePointerArray(b.tables.iter().map(|v| v.handle).collect()),
            d(b.ids),
            d(b.shard),
            d(b.wide),
            d(b.output),
            d(b.expanded),
            d(b.live_rows),
            Argument::U32(self.rank),
        ]);
        arguments.validate()?;
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            plan: self.entry.plan(counts.rows)?,
            rows: counts.rows,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}
