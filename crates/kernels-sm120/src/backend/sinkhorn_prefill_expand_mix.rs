//! Measured prefill rows of the expansion followed by the lagged prefill
//! mixer in one launch (`sinkhorn_hc::prefill_expand_mix`). Only registered
//! row buckets launch; every other row count is refused, so a caller keeps
//! the separate expansion and mixer there.
use std::collections::BTreeMap;

use crate::sinkhorn_hc::prefill_expand_mix::{Entry, Mutation};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_expand_mix::{
    SinkhornExpandMixBuffers as Buffers, SinkhornExpandMixQuery as Query,
};

pub(super) type Registered = (Entry, Vec<u32>);

/// Whether any registered entry serves `query`.
pub(super) fn serves(entries: &[Registered], query: &Query) -> bool {
    entries.iter().any(|(e, _)| e.query == *query)
}

pub(super) fn prepare(
    entries: &[Registered],
    query: &Query,
    need: Need,
) -> Result<Prepared<Buffers<View>>, Refused> {
    query.validate()?;
    let mut choices = BTreeMap::new();
    let mut symbols = Vec::new();
    for (entry, buckets) in entries.iter().filter(|(e, _)| e.query == *query) {
        entry.validate()?;
        if entry.mutation != Mutation::None
            || buckets.is_empty()
            || buckets[0] == 0
            || buckets.windows(2).any(|pair| pair[0] >= pair[1])
            || *buckets.last().unwrap() > entry.capacity
            || entry.capacity < need.capacity
        {
            return Err(Refused::Rule(
                "invalid measured prefill expand-mix buckets".into(),
            ));
        }
        for &rows in buckets {
            let plan = entry.plan(rows, rows)?;
            if choices.insert(rows, (entry.symbol.clone(), plan)).is_some() {
                return Err(Refused::Rule(
                    "ambiguous measured prefill expand-mix bucket".into(),
                ));
            }
        }
        symbols.push((entry.symbol.clone(), 0));
    }
    if choices.is_empty() {
        return Err(Refused::Unsupported);
    }
    let mut info = OpInfo::new(
        "sm120.sinkhorn.prefill-expand-mix.pipeline.v1",
        Invariance::any_rows(need.capacity),
        symbols,
    );
    let m = query.mix;
    let rows = u64::from(need.capacity);
    let n = u64::from(m.streams);
    let h = u64::from(m.width);
    info.bytes = vec![
        ("x".into(), rows * n * h * 2),
        ("y".into(), rows * h * 2),
        ("post_in".into(), rows * n * 4),
        ("comb_in".into(), rows * n * n * 4),
        ("expanded".into(), rows * n * h * 2),
        ("fn_".into(), fn_bytes(query)),
        ("base".into(), u64::from(m.mixes()) * 4),
        ("scale".into(), 12),
        ("pre_in".into(), rows * n * 4),
        ("collapsed".into(), rows * h * 2),
        ("pre".into(), rows * n * 4),
        ("post".into(), rows * n * 4),
        ("comb".into(), rows * n * n * 4),
        ("live_rows".into(), 4),
    ];
    need.check(&info)?;
    Ok(Box::new(Op {
        query: *query,
        capacity: need.capacity,
        choices,
        info,
    }))
}

fn fn_bytes(query: &Query) -> u64 {
    let m = query.mix;
    u64::from(m.mixes())
        * u64::from(m.streams)
        * u64::from(m.width)
        * if m.fn_dtype == DType::F32 { 4 } else { 2 }
}

struct Op {
    query: Query,
    capacity: u32,
    choices: BTreeMap<u32, (String, LaunchPlan)>,
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
            return Err(Refusal(
                "prefill expand-mix rows exceed prepared allocation",
            ));
        }
        let Some((symbol, plan)) = self.choices.get(&rows) else {
            return Err(Refusal("unmeasured prefill expand-mix rows"));
        };
        let m = self.query.mix;
        let n = u64::from(m.streams);
        let h = u64::from(m.width);
        let r = u64::from(rows);
        let roles = [
            b.x,
            b.y,
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
            b.live_rows,
        ];
        let sizes = [
            r * n * h * 2,
            r * h * 2,
            r * n * 4,
            r * n * n * 4,
            r * n * h * 2,
            fn_bytes(&self.query),
            u64::from(m.mixes()) * 4,
            12,
            r * n * 4,
            r * h * 2,
            r * n * 4,
            r * n * 4,
            r * n * n * 4,
            4,
        ];
        for (view, size) in roles.into_iter().zip(sizes) {
            view.covers(size)?;
        }
        // Each CTA reads its rows' coefficients before it writes any mixer
        // output, so post and comb may alias post_in and comb_in. Every other
        // role is distinct, and the launch takes no workspace.
        for (i, a) in roles.iter().enumerate() {
            if workspace.bytes > 0 && a.handle == workspace.handle {
                return Err(Refusal("prefill expand-mix workspace aliases a named role"));
            }
            for (j, b) in roles.iter().enumerate().take(i) {
                if a.handle == b.handle && !matches!((j, i), (2, 11) | (3, 12)) {
                    return Err(Refusal(
                        "prefill expand-mix buffers alias incompatible roles",
                    ));
                }
            }
        }
        Ok(vec![Launch {
            symbol: symbol.clone(),
            plan: *plan,
            rows,
            arguments: ArgumentList(
                roles
                    .into_iter()
                    .map(|v| Argument::DeviceBuffer(v.handle))
                    .collect(),
            ),
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use crate::sinkhorn_hc::lagged::LAB;
    use kernel_core::backend::Backend;

    fn entry(tile_rows: u32, chains: u32) -> Entry {
        Entry {
            query: Query { mix: LAB.mix },
            capacity: 8192,
            tile_rows,
            chains,
            symbol: format!("expand_mix_r{tile_rows}_c{chains}_v1"),
            mutation: Mutation::None,
        }
    }
    fn backend(buckets: [Vec<u32>; 2]) -> Sm120 {
        let [small, large] = buckets;
        Sm120::new()
            .with_sinkhorn_prefill_expand_mix(entry(4, 4), small)
            .with_sinkhorn_prefill_expand_mix(entry(8, 8), large)
    }
    fn need() -> Need {
        Need::rows(8192).with_exact(Invariance::any_rows(8192))
    }
    fn buffers() -> Buffers<View> {
        Buffers {
            x: view(1, u64::MAX),
            y: view(2, u64::MAX),
            post_in: view(3, u64::MAX),
            comb_in: view(4, u64::MAX),
            expanded: view(5, u64::MAX),
            fn_: view(6, u64::MAX),
            base: view(7, u64::MAX),
            scale: view(8, u64::MAX),
            pre_in: view(9, u64::MAX),
            collapsed: view(10, u64::MAX),
            pre: view(11, u64::MAX),
            post: view(3, u64::MAX),
            comb: view(4, u64::MAX),
            live_rows: view(14, 4),
        }
    }

    #[test]
    fn measured_rows_launch_their_tile_and_others_are_refused() {
        let query = Query { mix: LAB.mix };
        let op = backend([vec![64, 320], vec![8192]])
            .sinkhorn_expand_mix(&query, need())
            .unwrap();
        assert_eq!(op.info().workspace, 0);
        assert_eq!(op.info().symbols.len(), 2);
        let b = buffers();
        for rows in [1, 63, 64, 65, 192, 320, 2496, 4096, 8191, 8192] {
            let launches = op.launches(&b, View::EMPTY, Counts::rows(rows));
            let tile = match rows {
                64 | 320 => (4, 4),
                8192 => (8, 8),
                _ => {
                    assert!(launches.is_err(), "{rows}");
                    continue;
                }
            };
            let launches = launches.unwrap();
            assert_eq!(launches.len(), 1);
            let e = entry(tile.0, tile.1);
            assert_eq!(launches[0].symbol, e.symbol);
            assert_eq!(launches[0].plan, e.plan(rows, rows).unwrap());
            assert_eq!(launches[0].arguments.0.len(), 14);
            assert_eq!(launches[0].arguments.0[2], launches[0].arguments.0[11]);
            let mut bad = b;
            bad.expanded = bad.x;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            bad = b;
            bad.collapsed = bad.y;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            bad = b;
            bad.pre = bad.pre_in;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            bad = b;
            bad.collapsed.bytes = 1;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
        }
        assert!(op.launches(&b, View::EMPTY, Counts::rows(8193)).is_err());
    }

    #[test]
    fn invalid_buckets_are_refused_and_decode_needs_keep_their_route() {
        let query = Query { mix: LAB.mix };
        for buckets in [
            [vec![], vec![8192]],
            [vec![0, 64], vec![8192]],
            [vec![320, 64], vec![8192]],
            [vec![64], vec![64]],
            [vec![64], vec![8193]],
        ] {
            assert!(
                backend(buckets)
                    .sinkhorn_expand_mix(&query, need())
                    .is_err()
            );
        }
        // A need no registered entry holds is unsupported, not a rule error.
        assert!(matches!(
            Sm120::new().sinkhorn_expand_mix(&query, need()),
            Err(Refused::Unsupported)
        ));
    }
}
