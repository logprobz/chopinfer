//! Compose validated constituents, replacing only the qualified two-entry tail.
use super::Sm120;
use crate::{row::rmsnorm_quantize::resolved, sinkhorn_hc::finish_norm::Entry};
use kernel_core::{
    allocation::Traversal,
    backend::{Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View},
    contract::{ArgumentList, Refusal, Validate},
    properties::Invariance,
    rmsnorm_quantize::RmsNormQuantizeBuffers,
    sinkhorn_expand_mix::{
        SinkhornRoundExpandMixBuffers, SinkhornRoundNormBuffers as Buffers,
        SinkhornRoundNormQuery as Query,
    },
};

pub(super) fn prepare(
    backend: &Sm120,
    query: &Query,
    need: Need,
) -> Result<Prepared<Buffers<View>>, Refused> {
    query.validate()?;
    let mut choices = backend.finish_norm.iter().filter(|(e, _)| {
        e.mixer.query == query.mixer && resolved(e.norm.reference.query) == resolved(query.norm)
    });
    let (entry, buckets) = choices.next().ok_or(Refused::Unsupported)?;
    if choices.next().is_some() {
        return Err(Refusal("ambiguous HC finish norm entries").into());
    }
    entry.validate()?;
    if buckets.is_empty() || buckets.windows(2).any(|p| p[0] >= p[1]) {
        return Err(Refusal("invalid HC finish norm buckets").into());
    }
    for &rows in buckets {
        entry.plan(rows, rows, 32)?;
    }
    if entry.mixer.capacity < need.capacity {
        return Err(Refusal("HC finish norm capacity is too small").into());
    }
    let hc = backend.sinkhorn_round_expand_mix(&query.mixer, need)?;
    let norm = backend.rmsnorm_quantize(&query.norm, need)?;
    let guarantees = Invariance::any_rows(need.capacity);
    if !hc.info().guarantees.provides(guarantees)
        || !norm.info().guarantees.provides(guarantees)
        || norm.info().workspace != 0
        || !hc
            .info()
            .symbols
            .iter()
            .any(|(s, _)| s == &entry.mixer.symbols()[1])
        || !norm
            .info()
            .symbols
            .iter()
            .any(|(s, _)| s == &entry.norm.symbols()[2])
    {
        return Err(Refusal("HC finish norm constituents differ from qualification").into());
    }
    let mut symbols = hc.info().symbols.clone();
    symbols.extend(norm.info().symbols.clone());
    symbols.push((entry.symbols()[1].clone(), 0));
    let mut info = OpInfo::new(
        "sm120.sinkhorn.round-expand-mix.norm-tree256.ceil32.v1",
        guarantees,
        symbols,
    );
    info.workspace = hc.info().workspace;
    info.bytes = hc
        .info()
        .bytes
        .iter()
        .map(|(n, b)| (format!("hc.{n}"), *b))
        .chain(
            norm.info()
                .bytes
                .iter()
                .map(|(n, b)| (format!("norm.{n}"), *b)),
        )
        .collect();
    need.check(&info)?;
    Ok(Box::new(Op {
        hc,
        norm,
        entry: entry.clone(),
        buckets: buckets.clone(),
        info,
    }))
}
struct Op {
    hc: Prepared<SinkhornRoundExpandMixBuffers<View>>,
    norm: Prepared<RmsNormQuantizeBuffers<View>>,
    entry: Entry,
    buckets: Vec<u32>,
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
        if b.hc.collapsed != b.norm.input || b.hc.live_rows != b.norm.live_rows {
            return Err(Refusal("HC finish norm input/live views differ"));
        }
        let mut hc_roles = Vec::new();
        b.hc.visit(&mut |v| hc_roles.push(v.handle));
        let extra = [
            Some(b.norm.weight),
            Some(b.norm.normed),
            b.norm.widened,
            Some(b.norm.codes),
            Some(b.norm.scales),
            Some(b.norm.status),
        ];
        for v in extra.into_iter().flatten() {
            if hc_roles.contains(&v.handle) || v.handle == workspace.handle {
                return Err(Refusal("norm roles alias HC data or workspace"));
            }
        }
        // Both existing validators run before either launch can be returned.
        let mut hc = self.hc.launches(&b.hc, workspace, counts)?;
        let mut norm = self.norm.launches(&b.norm, View::EMPTY, counts)?;
        if !self.buckets.contains(&counts.rows) {
            hc.extend(norm);
            return Ok(hc);
        }
        if hc.len() != 2
            || norm.len() != 1
            || hc[1].symbol != self.entry.mixer.symbols()[1]
            || norm[0].symbol != self.entry.norm.symbols()[2]
        {
            return Err(Refusal("qualified HC finish norm launch chain differs"));
        }
        let finish = hc.pop().unwrap();
        let norm = norm.pop().unwrap();
        if finish.arguments.0.len() != 10 || norm.arguments.0.len() != 10 {
            return Err(Refusal("HC finish norm constituent ABI differs"));
        }
        let mut args = finish.arguments.0;
        args.extend_from_slice(&norm.arguments.0[1..6]);
        args.extend_from_slice(&norm.arguments.0[7..]);
        hc.push(Launch {
            symbol: self.entry.symbols()[1].clone(),
            plan: self.entry.plan(counts.rows, counts.rows, 32)?,
            rows: counts.rows,
            arguments: ArgumentList(args),
            collective: false,
            persistent: false,
        });
        Ok(hc)
    }
}
