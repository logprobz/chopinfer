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
    let chains = if backend
        .finish_norm_chains
        .iter()
        .any(|(e, _)| e.reference.query == query.mixer)
    {
        &backend.finish_norm_chains
    } else {
        &backend.round_chains
    };
    let hc = super::sinkhorn_round_expand_mix::prepare(
        &backend.round_expand_mix,
        chains,
        &query.mixer,
        need,
    )?;
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
    let mut row_tiles = None;
    for (candidate, tiles) in backend
        .finish_norm_rows
        .iter()
        .filter(|(e, _)| e.reference.reference.query == query.mixer)
    {
        candidate.validate()?;
        if tiles.is_empty()
            || tiles.windows(2).any(|p| p[0].0 >= p[1].0)
            || candidate.reference.reference.symbol != entry.mixer.symbol
            || candidate.reference.reference.capacity != entry.mixer.capacity
            || candidate.reference.reference.probe
        {
            return Err(Refusal("invalid HC finish norm row tiles").into());
        }
        if row_tiles.is_some() {
            return Err(Refusal("ambiguous HC finish norm row tiles").into());
        }
        for &(rows, tile) in tiles {
            if !buckets.contains(&rows) {
                return Err(Refusal("HC row tile lacks a qualified tail bucket").into());
            }
            candidate.plan(rows, rows, tile)?;
            let index = [2, 4, 8].iter().position(|&t| t == tile).unwrap();
            let symbol = (candidate.symbols()[index].clone(), 0);
            if !symbols.contains(&symbol) {
                symbols.push(symbol);
            }
        }
        row_tiles = Some((candidate.clone(), tiles.clone()));
    }
    symbols.extend(norm.info().symbols.clone());
    symbols.push((entry.symbols()[1].clone(), 0));
    let mut completion = None;
    for (candidate, tiles) in backend
        .finish_norm_completions
        .iter()
        .filter(|(e, _)| e.reference.mixer.query == query.mixer)
    {
        candidate.validate()?;
        if tiles.is_empty()
            || tiles.windows(2).any(|p| p[0].0 >= p[1].0)
            || candidate.reference.symbols() != entry.symbols()
            || candidate.reference.mixer.symbol != entry.mixer.symbol
            || candidate.reference.mixer.capacity != entry.mixer.capacity
            || candidate.mutation != crate::sinkhorn_hc::completion::Mutation::None
        {
            return Err(Refusal("invalid HC completion buckets").into());
        }
        if completion.is_some() {
            return Err(Refusal("ambiguous HC completion entries").into());
        }
        for &(rows, warps) in tiles {
            if !buckets.contains(&rows)
                || row_tiles
                    .as_ref()
                    .is_some_and(|(_, t)| t.iter().any(|&(r, _)| r == rows))
            {
                return Err(Refusal("HC completion bucket lacks a chain-producer tail").into());
            }
            candidate.plan(rows, rows, warps)?;
            let symbol = (candidate.symbol(warps), 0);
            if !symbols.contains(&symbol) {
                symbols.push(symbol);
            }
        }
        completion = Some((candidate.clone(), tiles.clone()));
    }
    let mut info = OpInfo::new(
        "sm120.sinkhorn.round-expand-mix.norm-tree256.ceil32.v1",
        guarantees,
        symbols,
    );
    info.workspace = hc.info().workspace;
    let tickets = completion
        .as_ref()
        .map(|_| ("scratch.tickets".to_owned(), u64::from(need.capacity) * 4));
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
        .chain(tickets)
        .collect();
    need.check(&info)?;
    Ok(Box::new(Op {
        hc,
        norm,
        entry: entry.clone(),
        buckets: buckets.clone(),
        row_tiles,
        completion,
        info,
    }))
}
struct Op {
    hc: Prepared<SinkhornRoundExpandMixBuffers<View>>,
    norm: Prepared<RmsNormQuantizeBuffers<View>>,
    entry: Entry,
    buckets: Vec<u32>,
    row_tiles: Option<(crate::sinkhorn_hc::round_rows::Entry, Vec<(u32, u32)>)>,
    completion: Option<(crate::sinkhorn_hc::completion::Entry, Vec<(u32, u32)>)>,
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
            || (norm[0].symbol != self.entry.norm.symbols()[2]
                && norm[0].symbol != self.entry.norm.reference.symbol)
        {
            return Err(Refusal("qualified HC finish norm launch chain differs"));
        }
        // Cached and original norm entries share this ABI and exact contract.
        // The composite validates either before replacing it with the tail;
        // standalone norm selection need not change for the fused row bucket.
        let finish = hc.pop().unwrap();
        let norm = norm.pop().unwrap();
        if finish.arguments.0.len() != 10 || norm.arguments.0.len() != 10 {
            return Err(Refusal("HC finish norm constituent ABI differs"));
        }
        if let Some((entry, tile)) = self.row_tiles.as_ref().and_then(|(e, tiles)| {
            tiles
                .iter()
                .find(|(rows, _)| *rows == counts.rows)
                .map(|&(_, tile)| (e, tile))
        }) {
            if hc[0].arguments.0.len() != 9 {
                return Err(Refusal("HC row tile producer ABI differs"));
            }
            let index = [2, 4, 8].iter().position(|&t| t == tile).unwrap();
            hc[0].symbol = entry.symbols()[index].clone();
            hc[0].plan = entry.plan(counts.rows, counts.rows, tile)?;
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
        // The one-launch boundary is composed from the validated pair; its
        // ticket must be distinct from every role and the workspace.
        if let Some((entry, warps)) = self.completion.as_ref().and_then(|(e, tiles)| {
            tiles
                .iter()
                .find(|(rows, _)| *rows == counts.rows)
                .map(|&(_, warps)| (e, warps))
        }) {
            let tickets = b
                .tickets
                .ok_or(Refusal("HC completion needs zeroed tickets"))?;
            if tickets.handle == workspace.handle {
                return Err(Refusal("HC completion tickets alias the workspace"));
            }
            let tail = hc.pop().unwrap();
            let producer = hc.pop().unwrap();
            hc.push(entry.compose(&producer, &tail, tickets, warps)?);
        }
        Ok(hc)
    }
}
