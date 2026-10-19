use super::select::{Candidate, choose};
use crate::row::rmsnorm_quantize::{Entry, resolved};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{DType, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::rmsnorm_quantize::{RmsNormQuantizeBuffers, RmsNormQuantizeQuery};

pub(super) fn prepare(
    entries: &[Entry],
    tiles: &[(crate::row::rmsnorm_quantize_tiles::Entry, Vec<u32>)],
    query: &RmsNormQuantizeQuery,
    need: Need,
) -> Result<Prepared<RmsNormQuantizeBuffers<View>>, Refused> {
    query.validate()?;
    let chosen = choose(
        "RMSNorm quantize",
        entries
            .iter()
            .filter(|e| resolved(e.query) == resolved(*query))
            .map(|e| {
                Candidate::new(
                    e.clone(),
                    &e.symbol,
                    format!("{:?}", e.spec()),
                    Invariance::any_rows(e.capacity),
                )
                .capacity(e.capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    entry.validate()?;
    entry.plan(need.capacity)?;
    let mut info = OpInfo::new(
        "sm120.rmsnorm-tree256.ceil32.v1",
        chosen.guarantees,
        vec![(entry.symbol.clone(), 0)],
    );
    let mut tiled = None;
    for (candidate, buckets) in tiles
        .iter()
        .filter(|(e, _)| resolved(e.reference.query) == resolved(*query))
    {
        candidate.validate()?;
        if buckets.is_empty()
            || buckets.windows(2).any(|p| p[0] >= p[1])
            || buckets
                .iter()
                .any(|&r| r == 0 || r > candidate.reference.capacity)
        {
            return Err(Refusal("invalid qualified norm tile buckets").into());
        }
        if candidate.reference.symbol != entry.symbol
            || candidate.reference.capacity < need.capacity
        {
            continue;
        }
        if tiled.is_some() {
            return Err(Refusal("ambiguous norm tile entries").into());
        }
        info.symbols.push((candidate.symbols()[1].clone(), 0));
        tiled = Some((candidate.clone(), buckets.clone()));
    }
    let values = u64::from(need.capacity) * u64::from(query.norm.width);
    let bytes = if query.norm.input == DType::F32 { 4 } else { 2 };
    info.bytes = vec![
        ("input".into(), values * bytes),
        ("weight".into(), u64::from(query.norm.width) * 2),
        ("normed".into(), values * bytes),
        ("codes".into(), values),
        ("scales".into(), values / 32),
        ("live_rows".into(), 4),
        ("status".into(), u64::from(need.capacity) * 4),
    ];
    need.check(&info)?;
    Ok(Box::new(Op {
        entry,
        tiled,
        capacity: need.capacity,
        info,
    }))
}

struct Op {
    entry: Entry,
    tiled: Option<(crate::row::rmsnorm_quantize_tiles::Entry, Vec<u32>)>,
    capacity: u32,
    info: OpInfo,
}
impl PreparedOp<RmsNormQuantizeBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &RmsNormQuantizeBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows.max(counts.padded_rows);
        if rows == 0 || rows > self.capacity {
            return Err(Refusal(
                "fused norm rows exceed prepared capacity or are zero",
            ));
        }
        let values = u64::from(counts.rows) * u64::from(self.entry.query.norm.width);
        let bytes = if self.entry.query.norm.input == DType::F32 {
            4
        } else {
            2
        };
        b.input.covers(values * bytes)?;
        b.weight
            .covers(u64::from(self.entry.query.norm.width) * 2)?;
        b.normed.covers(values * bytes)?;
        b.codes.covers(values)?;
        b.scales.covers(values / 32)?;
        b.live_rows.covers(4)?;
        b.status
            .covers(kernel_core::precision::quantizer::status_extent(
                b.status_row_offset,
                counts.rows,
            )?)?;
        if let Some(widened) = b.widened {
            if bytes != 2 {
                return Err(Refusal("only BF16 norm output can be widened"));
            }
            widened.covers(values * 4)?;
        }
        // All written roles are distinct, including the preserved norm output.
        let reads = [b.input.handle, b.weight.handle, b.live_rows.handle];
        let mut writes = vec![
            b.normed.handle,
            b.codes.handle,
            b.scales.handle,
            b.status.handle,
        ];
        writes.extend(b.widened.map(|v| v.handle));
        for (i, handle) in writes.iter().enumerate() {
            if reads.contains(handle) || writes[..i].contains(handle) {
                return Err(Refusal("fused norm buffers overlap incompatible roles"));
            }
        }
        let selected = self
            .tiled
            .as_ref()
            .filter(|(_, buckets)| buckets.contains(&rows));
        let (symbol, plan) = if let Some((entry, _)) = selected {
            (
                entry.symbols()[1].clone(),
                entry.plan(rows, counts.rows, true)?,
            )
        } else {
            (self.entry.symbol.clone(), self.entry.plan(rows)?)
        };
        Ok(vec![Launch {
            symbol,
            rows: counts.rows,
            plan,
            arguments: self.entry.arguments(RmsNormQuantizeBuffers {
                input: b.input.handle,
                weight: b.weight.handle,
                normed: b.normed.handle,
                widened: b.widened.map(|v| v.handle),
                codes: b.codes.handle,
                scales: b.scales.handle,
                live_rows: b.live_rows.handle,
                status: b.status.handle,
                status_row_offset: b.status_row_offset,
            })?,
            collective: false,
            persistent: false,
        }])
    }
}
