//! Lagged Sinkhorn hyper-connections through the model's entries
//! (`sinkhorn_hc::lagged`): the lagged mix-collapse, the F32 expand and the
//! sigmoid-weighted head. Each op runs the one entry [`super::sinkhorn`]'s
//! `fitting` chooses among the entries that run its query, whatever the
//! order the model registered them in, and plans every row count up to its
//! capacity; the kernels read the live row count on the device. Each op
//! claims what its entry declares (one-row bits up to the capacity, any
//! chunk split, row independence). Every op is refused past SM120's shared
//! Sinkhorn limits (`sinkhorn_hc::shape_limits`) before an entry is looked
//! up.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{LaunchPlan, Refusal};
use kernel_core::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, LaggedMixCollapseBuffers, SinkhornHyperConnectionQuery as Query,
    StreamExpandBuffers, StreamHeadMode, WeightedStreamHeadBuffers,
};

use super::sinkhorn::{
    Candidate, Selection, buffer, check_mixer, check_need, chosen, dense, fitting, planned, plans,
    uncapped,
};
use crate::sinkhorn_hc::{self, lagged, lagged_split, lagged_tiled, lagged_warp, split};

/// The lagged mix-collapse's numerics: the per-sublayer one-CTA entry's
/// projection, rstd and Sinkhorn, with the given collapse weights and F32
/// stores.
const LAGGED_MIX_COLLAPSE: &str = "sm120.sinkhorn.lagged-mix-collapse.v1[chain=128]";
const EXPAND_F32: &str = "sm120.sinkhorn.expand-f32.v1";
const WEIGHTED_HEAD: &str = "sm120.sinkhorn.weighted-head.v1";
const WEIGHTED_HEAD_F32: &str = "sm120.sinkhorn.weighted-head-f32.v1";

/// What a lagged entry launches: its symbol, ABI and the code its query's
/// shape and kernel compile; its capacity only bounds the rows.
fn launch_of(entry: &lagged::Entry) -> String {
    let lagged::Entry {
        profile: _, // filters catalogs; no launch reads it
        symbol,     // the launched symbol
        query,      // its shape selects the op and is the code's
        op,         // the kernel; selects the op
        params,     // the ABI
    } = entry;
    format!("{symbol} {:?} {op:?} {params:?}", uncapped(*query))
}

/// The lagged op of `op` as `kind`: its entry, plans and info.
struct LaggedOp<K> {
    entry: lagged::Entry,
    query: Query,
    plans: Vec<LaunchPlan>,
    info: OpInfo,
    split: Option<(lagged_split::Entry, Vec<LaunchPlan>)>,
    tiled: Option<(lagged_tiled::Entry, Vec<LaunchPlan>)>,
    warp: Option<lagged_warp::Entry>,
    kind: std::marker::PhantomData<fn() -> K>,
}

impl<K> LaggedOp<K> {
    fn new(
        selection: &Selection,
        op: Query,
        kind: lagged::Op,
        implementation: &str,
        need: Need,
    ) -> Result<Self, Refused> {
        sinkhorn_hc::shape_limits(op)?;
        let matches = match (op, kind) {
            (Query::MixCollapse { mix, .. }, lagged::Op::MixCollapse { probe: false }) => {
                mix.placement == HyperConnectionPlacement::LaggedSinglePass
            }
            (Query::ExpandF32 { .. }, lagged::Op::Expand) => true,
            (
                Query::Head {
                    mode: StreamHeadMode::SigmoidWeighted | StreamHeadMode::SigmoidWeightedF32,
                    ..
                },
                lagged::Op::Head,
            ) => true,
            _ => false,
        };
        if !matches {
            return Err(Refused::Rule(format!(
                "expected a lagged Sinkhorn {kind:?}"
            )));
        }
        check_need(op, need)?;
        let candidates = selection
            .lagged
            .iter()
            .filter(|entry| entry.op == kind && uncapped(entry.query) == uncapped(op))
            .map(|entry| {
                let capacity = entry.contract().and_then(|_| entry.capacity());
                Candidate::new(entry.clone(), &entry.symbol, launch_of(entry), capacity)
                    .claims(entry.invariance())
            })
            .collect();
        let name = format!("lagged {kind:?}");
        let entry = chosen(fitting(candidates, need, &name)?, &name)?;
        let plans = plans(need.capacity, |rows| entry.plan(rows, rows))?;
        let info = OpInfo::new(
            implementation,
            entry.invariance(),
            vec![(entry.symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Self {
            entry,
            query: op,
            plans,
            info,
            split: None,
            tiled: None,
            warp: None,
            kind: std::marker::PhantomData,
        })
    }

    fn launch(&self, rows: u32, plan: LaunchPlan, values: &[View]) -> Result<Vec<Launch>, Refusal> {
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan,
            arguments: self
                .entry
                .pack_arguments(values.iter().copied().map(buffer).collect())?,
            collective: false,
            persistent: false,
        }])
    }

    /// The streams and width of the op's rows.
    fn shape(&self) -> (u64, u64) {
        match self.query {
            Query::MixCollapse { mix, .. } => (u64::from(mix.streams), u64::from(mix.width)),
            Query::ExpandF32 { streams, width, .. } | Query::Head { streams, width, .. } => {
                (u64::from(streams), u64::from(width))
            }
            _ => unreachable!("refused when prepared"),
        }
    }
}

pub(super) fn mix_collapse(
    selection: &Selection,
    op: &Query,
    need: Need,
) -> Result<Prepared<LaggedMixCollapseBuffers<View>>, Refused> {
    let mut prepared = LaggedOp::new(
        selection,
        *op,
        lagged::Op::MixCollapse { probe: false },
        LAGGED_MIX_COLLAPSE,
        need,
    )?;
    let matching = |entry: &&lagged_split::Entry| {
        entry.reference.op == lagged::Op::MixCollapse { probe: false }
            && uncapped(entry.query()) == uncapped(*op)
    };
    let mut entries: Vec<_> = selection.lagged_split.iter().filter(matching).collect();
    let mut capacity = entries
        .iter()
        .filter_map(|entry| entry.reference.capacity().ok())
        .max()
        .unwrap_or(0)
        .min(need.capacity);
    if capacity < need.capacity {
        let bounded: Vec<_> = selection
            .lagged_split_bounded
            .iter()
            .filter(matching)
            .filter(|entry| entry.reference.capacity().is_ok_and(|c| c >= need.capacity))
            .collect();
        if !bounded.is_empty() {
            entries = bounded;
            capacity = need.capacity;
        }
    }
    if capacity > 0 {
        let candidates = entries
            .into_iter()
            .map(|entry| {
                let symbol = entry.symbol();
                Candidate::new(
                    entry.clone(),
                    &symbol,
                    format!("{} split {:?}", launch_of(&entry.reference), entry.params()),
                    entry.capacity(),
                )
                .claims(entry.invariance())
            })
            .collect();
        let split_need = Need::rows(capacity)
            .with_exact(kernel_core::properties::Invariance::any_rows(capacity));
        let entry = chosen(
            fitting(candidates, split_need, "lagged split")?,
            "lagged split",
        )?;
        let split_plans = plans(capacity, |rows| entry.plan(rows, rows))?;
        let Query::MixCollapse { mix, .. } = *op else {
            unreachable!("validated mix-collapse");
        };
        prepared.info.bytes = vec![
            (
                "scratch.partials".into(),
                split::partials_bytes(mix, capacity),
            ),
            ("scratch.tickets".into(), split::tickets_bytes(capacity)),
        ];
        prepared.info.symbols.push((entry.symbol(), 0));
        prepared.split = Some((entry, split_plans));
    }
    let candidates: Vec<_> = selection
        .lagged_tiled
        .iter()
        .filter(|entry| {
            entry.reference.op == lagged::Op::MixCollapse { probe: false }
                && uncapped(entry.reference.query) == uncapped(*op)
                && entry.from_rows <= need.capacity
        })
        .map(|entry| {
            Candidate::new(
                entry.clone(),
                &entry.reference.symbol,
                format!(
                    "{} tiled rows={} from={}",
                    launch_of(&entry.reference),
                    entry.rows,
                    entry.from_rows
                ),
                entry.capacity(),
            )
            .claims(entry.invariance())
        })
        .collect();
    if !candidates.is_empty() {
        let entry = chosen(fitting(candidates, need, "lagged tiled")?, "lagged tiled")?;
        let tiled_plans = plans(need.capacity, |rows| entry.plan(rows, rows))?;
        prepared
            .info
            .symbols
            .push((entry.reference.symbol.clone(), 0));
        prepared.tiled = Some((entry, tiled_plans));
    }
    let mut candidates = Vec::new();
    for entry in &selection.lagged_warp {
        if uncapped(entry.source.reference.query()) != uncapped(*op) {
            continue;
        }
        let capacity = entry.capacity()?;
        if need.capacity > capacity {
            continue;
        }
        let symbol = entry.symbol();
        candidates.push(
            Candidate::new(
                entry.clone(),
                &symbol,
                format!(
                    "{} warp-null {:?} {:?}",
                    launch_of(&entry.source.reference.reference),
                    entry.source.reference.params(),
                    entry.buckets
                ),
                Ok(capacity),
            )
            .claims(entry.source.reference.invariance()),
        );
    }
    if !candidates.is_empty() {
        let entry = chosen(fitting(candidates, need, "lagged warp")?, "lagged warp")?;
        let Query::MixCollapse { mix, .. } = *op else {
            unreachable!("validated mixer");
        };
        if prepared.split.is_none() {
            prepared.info.bytes = vec![
                (
                    "scratch.partials".into(),
                    split::partials_bytes(mix, need.capacity),
                ),
                (
                    "scratch.tickets".into(),
                    split::tickets_bytes(need.capacity),
                ),
            ];
        }
        prepared.info.symbols.push((entry.symbol(), 0));
        prepared.warp = Some(entry);
    }
    need.check(&prepared.info)?;
    Ok(Box::new(prepared))
}

impl PreparedOp<LaggedMixCollapseBuffers<View>> for LaggedOp<LaggedMixCollapseBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &LaggedMixCollapseBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let plan = planned(&self.plans, counts)?;
        let rows = counts.rows;
        let Query::MixCollapse { mix, .. } = self.query else {
            unreachable!("refused when prepared");
        };
        let (streams, width) = self.shape();
        buffers.x.covers(dense(rows, streams * width, 2)?)?;
        check_mixer(mix, &buffers.mixer)?;
        buffers.pre_in.covers(dense(rows, streams, 4)?)?;
        buffers.collapsed.covers(dense(rows, width, 2)?)?;
        buffers.pre.covers(dense(rows, streams, 4)?)?;
        buffers.post.covers(dense(rows, streams, 4)?)?;
        buffers.comb.covers(dense(rows, streams * streams, 4)?)?;
        buffers.live_rows.covers(4)?;
        let mut values = vec![
            buffers.x,
            buffers.mixer.fn_,
            buffers.mixer.base,
            buffers.mixer.scale,
            buffers.pre_in,
            buffers.collapsed,
            buffers.pre,
            buffers.post,
            buffers.comb,
        ];
        if let Some(entry) = &self.warp
            && entry.buckets.contains(&rows)
        {
            let partials = buffers
                .scratch
                .partials
                .ok_or(Refusal("warp mixer needs partials"))?;
            let tickets = buffers
                .scratch
                .tickets
                .ok_or(Refusal("warp mixer needs tickets"))?;
            partials.covers(split::partials_bytes(mix, rows))?;
            tickets.covers(split::tickets_bytes(rows))?;
            values.extend([partials, tickets, buffers.live_rows]);
            return Ok(vec![Launch {
                symbol: entry.symbol(),
                rows,
                plan: entry.source.reference.plan(rows, rows)?,
                arguments: entry.pack_arguments(values.into_iter().map(buffer).collect())?,
                collective: false,
                persistent: false,
            }]);
        }
        if let Some((entry, plans)) = &self.split
            && let Some(&plan) = plans.get(rows as usize - 1)
        {
            let partials = buffers
                .scratch
                .partials
                .ok_or(Refusal("lagged split needs partials"))?;
            let tickets = buffers
                .scratch
                .tickets
                .ok_or(Refusal("lagged split needs tickets"))?;
            partials.covers(split::partials_bytes(mix, rows))?;
            tickets.covers(split::tickets_bytes(rows))?;
            values.extend([partials, tickets, buffers.live_rows]);
            return Ok(vec![Launch {
                symbol: entry.symbol(),
                rows,
                plan,
                arguments: entry.pack_arguments(values.into_iter().map(buffer).collect())?,
                collective: false,
                persistent: false,
            }]);
        }
        values.push(buffers.live_rows);
        if let Some((entry, plans)) = &self.tiled
            && rows >= entry.from_rows
        {
            return Ok(vec![Launch {
                symbol: entry.reference.symbol.clone(),
                rows,
                plan: planned(plans, counts)?,
                arguments: entry
                    .reference
                    .pack_arguments(values.into_iter().map(buffer).collect())?,
                collective: false,
                persistent: false,
            }]);
        }
        self.launch(rows, plan, &values)
    }
}

pub(super) fn expand(
    selection: &Selection,
    op: &Query,
    need: Need,
) -> Result<Prepared<StreamExpandBuffers<View>>, Refused> {
    let op = LaggedOp::new(selection, *op, lagged::Op::Expand, EXPAND_F32, need)?;
    Ok(Box::new(op))
}

impl PreparedOp<StreamExpandBuffers<View>> for LaggedOp<StreamExpandBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &StreamExpandBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let plan = planned(&self.plans, counts)?;
        let rows = counts.rows;
        let (streams, width) = self.shape();
        let wide = dense(rows, streams * width, 2)?;
        buffers.x.covers(wide)?;
        buffers.y.covers(dense(rows, width, 2)?)?;
        buffers.post.covers(dense(rows, streams, 4)?)?;
        buffers.comb.covers(dense(rows, streams * streams, 4)?)?;
        buffers.output.covers(wide)?;
        buffers.live_rows.covers(4)?;
        self.launch(
            rows,
            plan,
            &[
                buffers.x,
                buffers.y,
                buffers.post,
                buffers.comb,
                buffers.output,
                buffers.live_rows,
            ],
        )
    }
}

pub(super) fn head(
    selection: &Selection,
    op: &Query,
    need: Need,
) -> Result<Prepared<WeightedStreamHeadBuffers<View>>, Refused> {
    let implementation = if matches!(
        op,
        Query::Head {
            mode: StreamHeadMode::SigmoidWeightedF32,
            ..
        }
    ) {
        WEIGHTED_HEAD_F32
    } else {
        WEIGHTED_HEAD
    };
    let op = LaggedOp::new(selection, *op, lagged::Op::Head, implementation, need)?;
    Ok(Box::new(op))
}

impl PreparedOp<WeightedStreamHeadBuffers<View>> for LaggedOp<WeightedStreamHeadBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &WeightedStreamHeadBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let plan = planned(&self.plans, counts)?;
        let rows = counts.rows;
        let (streams, width) = self.shape();
        buffers.x.covers(dense(rows, streams * width, 2)?)?;
        buffers.pre.covers(dense(rows, streams, 4)?)?;
        let output_bytes = if matches!(
            self.query,
            Query::Head {
                mode: StreamHeadMode::SigmoidWeightedF32,
                ..
            }
        ) {
            4
        } else {
            2
        };
        buffers.output.covers(dense(rows, width, output_bytes)?)?;
        buffers.live_rows.covers(4)?;
        self.launch(
            rows,
            plan,
            &[buffers.x, buffers.pre, buffers.output, buffers.live_rows],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use crate::sinkhorn_hc::lagged::{LAB, LAB_PROFILE, TWIN};
    use kernel_core::backend::Backend;
    use kernel_core::contract::Argument;
    use kernel_core::properties::Invariance;
    use kernel_core::sinkhorn_hyper_connection::{MixScratch, MixerWeights};

    fn backend() -> Sm120 {
        Sm120::new().with_sinkhorn_lagged(lagged::entries(LAB_PROFILE, LAB, true))
    }

    fn mix_collapse(capacity: u32) -> Query {
        Query::MixCollapse {
            mix: LAB.mix,
            capacity,
        }
    }

    fn expand(capacity: u32) -> Query {
        Query::ExpandF32 {
            streams: 4,
            width: 5120,
            capacity,
        }
    }

    fn head(capacity: u32) -> Query {
        Query::Head {
            streams: 4,
            width: 5120,
            mode: StreamHeadMode::SigmoidWeighted,
            capacity,
        }
    }

    fn handles(handles: impl IntoIterator<Item = u64>) -> Vec<Argument> {
        handles.into_iter().map(Argument::DeviceBuffer).collect()
    }

    #[test]
    fn measured_warp_buckets_keep_the_split_abi_and_other_preparations() {
        let geometry = sinkhorn_hc::Geometry {
            capacity: 32,
            ..LAB
        };
        let source = sinkhorn_hc::mixer_probe::Entry {
            reference: lagged_split::entry(LAB_PROFILE, geometry, false),
            prefix: "chop_hc_qualified".into(),
        };
        let warp = lagged_warp::Entry {
            source,
            buckets: vec![1, 8],
        };
        let backend = backend()
            .with_sinkhorn_lagged_split([warp.source.reference.clone()])
            .with_sinkhorn_lagged_warp(warp.clone());
        let buffers = LaggedMixCollapseBuffers {
            x: view(1, u64::MAX),
            mixer: MixerWeights {
                fn_: view(2, u64::MAX),
                base: view(3, u64::MAX),
                scale: view(4, u64::MAX),
            },
            pre_in: view(5, u64::MAX),
            collapsed: view(6, u64::MAX),
            pre: view(7, u64::MAX),
            post: view(8, u64::MAX),
            comb: view(9, u64::MAX),
            scratch: MixScratch {
                partials: Some(view(10, u64::MAX)),
                tickets: Some(view(11, u64::MAX)),
            },
            live_rows: view(12, 4),
        };
        for capacity in [8, 32, 8192] {
            let op = backend
                .sinkhorn_lagged_mix_collapse(&mix_collapse(capacity), Need::rows(capacity))
                .unwrap();
            assert_eq!(op.info().implementation, LAGGED_MIX_COLLAPSE);
            assert_eq!(
                op.info().symbols.iter().any(|(s, _)| s == &warp.symbol()),
                capacity == 8
            );
            for rows in [1, 2, 4, 8, 16, 32, 33, 8192]
                .into_iter()
                .filter(|r| *r <= capacity)
            {
                let launches = op
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let launch = &launches[0];
                let selected = capacity == 8 && [1, 8].contains(&rows);
                assert_eq!(launch.symbol == warp.symbol(), selected);
                if selected {
                    assert_eq!(launch.plan, warp.source.reference.plan(rows, rows).unwrap());
                    let mut args = handles(1..=12);
                    args.push(Argument::NullDevicePointer);
                    assert_eq!(launch.arguments.0, args);
                } else {
                    assert_eq!(launch.arguments.0.len(), if rows <= 32 { 12 } else { 10 });
                }
            }
        }
        assert!(warp.pack_arguments(handles(1..=11)).is_err());
        let mut bad = warp.clone();
        for buckets in [vec![], vec![0], vec![8, 1], vec![1, 1], vec![33]] {
            bad.buckets = buckets;
            assert!(bad.capacity().is_err());
        }
        let mut short = buffers;
        short.scratch.partials = Some(view(10, 15359));
        let op = backend
            .sinkhorn_lagged_mix_collapse(&mix_collapse(8), Need::rows(8))
            .unwrap();
        assert!(op.launches(&short, View::EMPTY, Counts::rows(1)).is_err());
    }

    /// A bounded split serves an op prepared for at most its capacity
    /// where the ordinary split holds fewer rows; every other preparation,
    /// including the prefill route over the same mixer, keeps its routes.
    #[test]
    fn bounded_splits_serve_only_ops_they_hold_whole() {
        let split = |capacity| {
            lagged_split::entry(
                LAB_PROFILE,
                sinkhorn_hc::Geometry { capacity, ..LAB },
                false,
            )
        };
        let backend = backend()
            .with_sinkhorn_lagged_split([split(32)])
            .with_sinkhorn_lagged_split_bounded([split(80)]);
        let symbol = split(32).symbol();
        let buffers = LaggedMixCollapseBuffers {
            x: view(1, u64::MAX),
            mixer: MixerWeights {
                fn_: view(2, u64::MAX),
                base: view(3, u64::MAX),
                scale: view(4, u64::MAX),
            },
            pre_in: view(5, u64::MAX),
            collapsed: view(6, u64::MAX),
            pre: view(7, u64::MAX),
            post: view(8, u64::MAX),
            comb: view(9, u64::MAX),
            scratch: MixScratch {
                partials: Some(view(10, u64::MAX)),
                tickets: Some(view(11, u64::MAX)),
            },
            live_rows: view(12, 4),
        };
        for (capacity, split_rows) in [(32, 32), (40, 40), (80, 80), (81, 32), (8192, 32)] {
            let op = backend
                .sinkhorn_lagged_mix_collapse(&mix_collapse(capacity), Need::rows(capacity))
                .unwrap();
            let partials = op
                .info()
                .bytes
                .iter()
                .find(|(path, _)| path == "scratch.partials")
                .map(|&(_, bytes)| bytes);
            assert_eq!(partials, Some(split::partials_bytes(LAB.mix, split_rows)));
            for rows in [1, 32, 33, 40, 80, 81]
                .into_iter()
                .filter(|r| *r <= capacity)
            {
                let launches = op
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                assert_eq!(
                    launches[0].symbol == symbol,
                    rows <= split_rows,
                    "{capacity} {rows}"
                );
            }
            let prefill = backend
                .sinkhorn_lagged_prefill(&mix_collapse(capacity), Need::rows(capacity))
                .unwrap();
            for rows in [32, 33, 40].into_iter().filter(|r| *r <= capacity) {
                let launches = prefill
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                assert_eq!(launches[0].symbol == symbol, rows <= split_rows);
            }
        }
    }

    #[test]
    fn each_op_launches_its_entry_in_abi_order() {
        let backend = backend();
        let catalog = lagged::catalog();
        let symbol = |op| {
            catalog
                .iter()
                .find(|entry| entry.op == op)
                .unwrap()
                .symbol
                .clone()
        };
        let need = Need::rows(8192).with_exact(Invariance::any_rows(8192));
        let mix = backend
            .sinkhorn_lagged_mix_collapse(&mix_collapse(8192), need)
            .unwrap();
        assert_eq!(mix.info().implementation, LAGGED_MIX_COLLAPSE);
        assert_eq!(mix.info().guarantees, Invariance::any_rows(8192));
        let buffers = LaggedMixCollapseBuffers {
            x: view(1, u64::MAX),
            mixer: MixerWeights {
                fn_: view(2, u64::MAX),
                base: view(3, u64::MAX),
                scale: view(4, u64::MAX),
            },
            pre_in: view(5, u64::MAX),
            collapsed: view(6, u64::MAX),
            pre: view(7, u64::MAX),
            post: view(8, u64::MAX),
            comb: view(9, u64::MAX),
            scratch: MixScratch {
                partials: None,
                tickets: None,
            },
            live_rows: view(10, 4),
        };
        for rows in [1, 32, 192, 8192] {
            let launches = mix
                .launches(&buffers, view(0, 0), Counts::rows(rows))
                .unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(
                launches[0].symbol,
                symbol(lagged::Op::MixCollapse { probe: false })
            );
            assert_eq!(launches[0].plan.grid, [rows, 1, 1]);
            assert_eq!(launches[0].arguments.0, handles(1..=10));
        }
        // An F32 pre_in, pre, post and comb of one row too few is refused.
        let short = LaggedMixCollapseBuffers {
            post: view(8, 4 * 4 * 2 - 1),
            ..buffers
        };
        assert!(mix.launches(&short, view(0, 0), Counts::rows(2)).is_err());
        let short = LaggedMixCollapseBuffers {
            pre_in: view(5, 4 * 4 - 1),
            ..buffers
        };
        assert!(mix.launches(&short, view(0, 0), Counts::rows(1)).is_err());

        let expand = backend.sinkhorn_expand(&expand(8192), need).unwrap();
        assert_eq!(expand.info().implementation, EXPAND_F32);
        let buffers = StreamExpandBuffers {
            x: view(1, u64::MAX),
            y: view(2, u64::MAX),
            post: view(3, u64::MAX),
            comb: view(4, u64::MAX),
            output: view(5, u64::MAX),
            live_rows: view(6, 4),
        };
        let launches = expand
            .launches(&buffers, view(0, 0), Counts::rows(32))
            .unwrap();
        assert_eq!(launches[0].symbol, symbol(lagged::Op::Expand));
        assert_eq!(launches[0].plan.grid, [10, 32, 1]);
        assert_eq!(launches[0].arguments.0, handles(1..=6));
        let short = StreamExpandBuffers {
            comb: view(4, 32 * 16 * 4 - 1),
            ..buffers
        };
        assert!(
            expand
                .launches(&short, view(0, 0), Counts::rows(32))
                .is_err()
        );

        let head = backend.sinkhorn_weighted_head(&head(8192), need).unwrap();
        assert_eq!(head.info().implementation, WEIGHTED_HEAD);
        let buffers = WeightedStreamHeadBuffers {
            x: view(1, u64::MAX),
            pre: view(2, u64::MAX),
            output: view(3, u64::MAX),
            live_rows: view(4, 4),
        };
        let launches = head
            .launches(&buffers, view(0, 0), Counts::rows(6))
            .unwrap();
        assert_eq!(launches[0].symbol, symbol(lagged::Op::Head));
        assert_eq!(launches[0].plan.grid, [10, 6, 1]);
        assert_eq!(launches[0].arguments.0, handles(1..=4));
        let short = WeightedStreamHeadBuffers {
            pre: view(2, 6 * 4 * 4 - 1),
            ..buffers
        };
        assert!(head.launches(&short, view(0, 0), Counts::rows(6)).is_err());
    }

    #[test]
    fn f32_head_selects_its_entry_and_requires_four_bytes_per_output() {
        let entry = lagged::head_f32_entry(LAB_PROFILE, LAB);
        let need = Need::rows(8).with_exact(Invariance::any_rows(8));
        let prepared = backend()
            .sinkhorn_weighted_head(&entry.query, need)
            .unwrap();
        assert_eq!(prepared.info().implementation, WEIGHTED_HEAD_F32);
        let buffers = WeightedStreamHeadBuffers {
            x: view(1, 8 * 4 * 5120 * 2),
            pre: view(2, 8 * 4 * 4),
            output: view(3, 8 * 5120 * 4),
            live_rows: view(4, 4),
        };
        let launches = prepared
            .launches(&buffers, view(0, 0), Counts::rows(8))
            .unwrap();
        assert_eq!(launches[0].symbol, entry.symbol);
        assert_eq!(launches[0].arguments.0, handles(1..=4));
        assert_eq!(launches[0].plan.grid, [10, 8, 1]);
        for bytes in [8 * 5120 * 2, 8 * 5120 * 4 - 1] {
            let short = WeightedStreamHeadBuffers {
                output: view(3, bytes),
                ..buffers
            };
            assert!(
                prepared
                    .launches(&short, view(0, 0), Counts::rows(8))
                    .is_err()
            );
        }
        let bf16 = Sm120::new().with_sinkhorn_lagged(
            lagged::catalog()
                .into_iter()
                .filter(|e| e.query != entry.query),
        );
        assert!(bf16.sinkhorn_weighted_head(&entry.query, need).is_err());
        let f32_only = Sm120::new().with_sinkhorn_lagged([entry.clone()]);
        assert!(f32_only.sinkhorn_weighted_head(&head(8), need).is_err());
        assert!(backend().sinkhorn_head(&entry.query, need).is_err());
        assert!(crate::sinkhorn_hc::limits(entry.query).is_err());
    }

    #[test]
    fn each_engine_refuses_the_other_engines_queries() {
        let lagged_backend = backend();
        let rule = |result: Result<(), Refused>| match result {
            Err(Refused::Rule(rule)) => rule,
            other => panic!("expected a rule refusal, got {other:?}"),
        };
        let need = Need::rows(8);
        // The per-sublayer twin of the lab mixer has no lagged entry, and the
        // per-sublayer mix-collapse, expand and head refuse the lagged ones.
        let twin = Query::MixCollapse {
            mix: TWIN.mix,
            capacity: 8,
        };
        assert!(
            rule(
                lagged_backend
                    .sinkhorn_lagged_mix_collapse(&twin, need)
                    .map(drop)
            )
            .contains("expected a lagged")
        );
        assert!(
            rule(
                lagged_backend
                    .sinkhorn_mix_collapse(&mix_collapse(8), need)
                    .map(drop)
            )
            .contains("lagged engine")
        );
        assert!(
            rule(lagged_backend.sinkhorn_head(&head(8), need).map(drop)).contains("lagged engine")
        );
        let bf16_expand = Query::Expand {
            streams: 4,
            width: 5120,
            capacity: 8,
        };
        assert!(
            rule(lagged_backend.sinkhorn_expand(&bf16_expand, need).map(drop))
                .contains("no SM120 Sinkhorn")
        );
        assert!(
            rule(
                lagged_backend
                    .sinkhorn_weighted_head(&expand(8), need)
                    .map(drop)
            )
            .contains("expected a lagged")
        );
        // Without the entries, nothing is served; past the query's rows or
        // the kernels' limits, nothing is prepared.
        assert!(matches!(
            Sm120::new().sinkhorn_lagged_mix_collapse(&mix_collapse(8), need),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            lagged_backend.sinkhorn_weighted_head(&head(8), Need::rows(9)),
            Err(Refused::Rule(_))
        ));
        let wide = Query::ExpandF32 {
            streams: 9,
            width: 5120,
            capacity: 8,
        };
        assert!(lagged_backend.sinkhorn_expand(&wide, need).is_err());
        // A narrower capacity than the need's rows is refused by the entry.
        let narrow = Sm120::new().with_sinkhorn_lagged(lagged::entries(
            LAB_PROFILE,
            crate::sinkhorn_hc::Geometry {
                capacity: 16,
                ..LAB
            },
            false,
        ));
        assert!(
            narrow
                .sinkhorn_expand(&expand(8192), Need::rows(17))
                .is_err()
        );
        narrow
            .sinkhorn_expand(&expand(8192), Need::rows(16))
            .unwrap();
    }
}
