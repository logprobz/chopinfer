use super::select::{Candidate, choose};
use crate::sinkhorn_hc::expand_mix::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, DType, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_expand_mix::{
    SinkhornExpandMixQuery as Query, SinkhornRoundExpandMixBuffers as Buffers,
};

pub(super) type Chains = (
    crate::sinkhorn_hc::round_chain_partials::Entry,
    Vec<(u32, u32)>,
);

pub(super) fn prepare(
    entries: &[Entry],
    chains: &[Chains],
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
    let mut chain_entry = None;
    for (candidate, buckets) in chains.iter().filter(|(e, _)| e.reference.query == *query) {
        candidate.validate()?;
        if buckets.is_empty() || buckets.windows(2).any(|p| p[0].0 >= p[1].0) {
            return Err(Refusal("invalid qualified rounded chain buckets").into());
        }
        for &(rows, warps) in buckets {
            candidate.plan(rows, rows, warps)?;
        }
        if candidate.reference.symbol != entry.symbol
            || candidate.reference.capacity < need.capacity
        {
            continue;
        }
        if chain_entry.is_some() {
            return Err(Refusal("ambiguous rounded chain entries").into());
        }
        for &(_, warps) in buckets {
            let symbol = (candidate.symbols()[usize::from(warps == 8)].clone(), 0);
            if !info.symbols.contains(&symbol) {
                info.symbols.push(symbol);
            }
        }
        chain_entry = Some((candidate.clone(), buckets.clone()));
    }
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
        chains: chain_entry,
        capacity: need.capacity,
        info,
    }))
}
struct Op {
    entry: Entry,
    chains: Option<Chains>,
    capacity: u32,
    info: OpInfo,
}
impl PreparedOp<Buffers<View>> for Op {
    fn route(&self, counts: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        use kernel_core::route::{RouteClass, branch};
        // Chain tuning replaces only the producer. The standalone finisher is
        // unchanged; a composing norm wrapper classifies its own fallback tail.
        if (launch.symbol == self.entry.symbols()[0]
            || self
                .chains
                .as_ref()
                .is_some_and(|(entry, _)| entry.symbols().contains(&launch.symbol)))
            && let Some((_, buckets)) = &self.chains
        {
            let selected = buckets.iter().any(|&(rows, _)| rows == counts.rows);
            return Some(branch(
                &launch.symbol,
                if selected {
                    RouteClass::Fast
                } else {
                    RouteClass::Baseline
                },
                "sinkhorn_round_chain",
                if selected { "chain" } else { "chain-fallback" },
                buckets.iter().map(|&(rows, _)| rows..=rows),
            ));
        }
        Some(branch(
            &launch.symbol,
            RouteClass::Fast,
            "sinkhorn_round_chain",
            "standalone",
            [1..=self.capacity],
        ))
    }
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
        let mut symbols = self.entry.symbols();
        let mut plans = self.entry.plans(rows)?;
        if let Some((entry, warps)) = self
            .chains
            .as_ref()
            .and_then(|(e, b)| b.iter().find(|(r, _)| *r == rows).map(|&(_, w)| (e, w)))
        {
            symbols[0] = entry.symbols()[usize::from(warps == 8)].clone();
            plans[0] = entry.plan(rows, rows, warps)?;
        }
        Ok(symbols
            .into_iter()
            .zip(plans)
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
