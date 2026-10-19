use super::select::{Candidate, choose};
use crate::sinkhorn_hc::expand_mix::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, DType, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_expand_mix::{
    SinkhornExpandMixQuery as Query, SinkhornRoundExpandMixBuffers as Buffers,
};

pub(super) fn prepare(
    entries: &[Entry],
    query: &Query,
    need: Need,
) -> Result<Prepared<Buffers<View>>, Refused> {
    query.validate()?;
    if !query.mix.width.is_multiple_of(4) {
        return Err(Refusal("rounded expansion needs whole four-element stores").into());
    }
    let chosen = choose(
        "lagged round-expand-mix",
        entries
            .iter()
            .filter(|e| e.query == *query && !e.probe)
            .map(|e| {
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
    let entry = chosen.entry;
    entry.validate()?;
    let mut info = OpInfo::new(
        "sm120.sinkhorn.round-expand-mix.chain128.v1",
        chosen.guarantees,
        entry.symbols().into_iter().map(|s| (s, 0)).collect(),
    );
    info.workspace = entry.workspace(need.capacity)?;
    let m = query.mix;
    let rows = u64::from(need.capacity);
    let n = u64::from(m.streams);
    let h = u64::from(m.width);
    info.bytes = vec![
        ("x".into(), rows * n * h * 2),
        ("sum".into(), rows * h * 4),
        ("post_in".into(), rows * n * 4),
        ("comb_in".into(), rows * n * n * 4),
        ("expanded".into(), rows * n * h * 2),
        (
            "fn_".into(),
            u64::from(m.mixes()) * n * h * if m.fn_dtype == DType::F32 { 4 } else { 2 },
        ),
        ("base".into(), u64::from(m.mixes()) * 4),
        ("scale".into(), 12),
        ("pre_in".into(), rows * n * 4),
        ("collapsed".into(), rows * h * 2),
        ("pre".into(), rows * n * 4),
        ("post".into(), rows * n * 4),
        ("comb".into(), rows * n * n * 4),
        ("rounded".into(), rows * h * 2),
        ("live_rows".into(), 4),
    ];
    need.check(&info)?;
    Ok(Box::new(Op {
        entry,
        capacity: need.capacity,
        info,
    }))
}
struct Op {
    entry: Entry,
    capacity: u32,
    info: OpInfo,
}
impl PreparedOp<Buffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &Buffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        if rows == 0 || rows > self.capacity || counts.padded_rows > rows {
            return Err(Refusal("expand-mix rows exceed prepared allocation"));
        }
        let m = self.entry.query.mix;
        let n = u64::from(m.streams);
        let h = u64::from(m.width);
        let r = u64::from(rows);
        let roles = [
            b.x,
            b.sum,
            b.post_in,
            b.comb_in,
            b.expanded,
            b.fn_,
            b.base,
            b.scale,
            b.pre_in,
            b.collapsed,
            b.pre,
            b.post,
            b.comb,
            b.rounded,
            b.live_rows,
        ];
        let sizes = [
            r * n * h * 2,
            r * h * 4,
            r * n * 4,
            r * n * n * 4,
            r * n * h * 2,
            u64::from(m.mixes()) * n * h * if m.fn_dtype == DType::F32 { 4 } else { 2 },
            u64::from(m.mixes()) * 4,
            12,
            r * n * 4,
            r * h * 2,
            r * n * 4,
            r * n * 4,
            r * n * n * 4,
            r * h * 2,
            4,
        ];
        for (view, size) in roles.into_iter().zip(sizes) {
            view.covers(size)?;
        }
        workspace.covers(self.entry.workspace(rows)?)?;
        // Previous post/comb are dead after the first kernel. Every other
        // input/output and scratch lifetime overlaps and must be distinct.
        for (i, a) in roles.iter().enumerate() {
            if a.handle == workspace.handle {
                return Err(Refusal("expand-mix workspace aliases a named role"));
            }
            for (j, b) in roles.iter().enumerate().take(i) {
                if a.handle == b.handle && !matches!((j, i), (2, 11) | (3, 12)) {
                    return Err(Refusal("expand-mix buffers alias incompatible roles"));
                }
            }
        }
        let d = |v: View| Argument::DeviceBuffer(v.handle);
        let arguments = [
            vec![
                d(b.x),
                d(b.sum),
                d(b.rounded),
                d(b.post_in),
                d(b.comb_in),
                d(b.expanded),
                d(b.fn_),
                d(workspace),
                d(b.live_rows),
            ],
            vec![
                d(b.expanded),
                d(b.base),
                d(b.scale),
                d(b.pre_in),
                d(b.collapsed),
                d(b.pre),
                d(b.post),
                d(b.comb),
                d(workspace),
                d(b.live_rows),
            ],
        ];
        Ok(self
            .entry
            .symbols()
            .into_iter()
            .zip(self.entry.plans(rows)?)
            .zip(arguments)
            .map(|((symbol, plan), arguments)| Launch {
                symbol,
                plan,
                rows,
                arguments: ArgumentList(arguments),
                collective: false,
                persistent: false,
            })
            .collect())
    }
}
