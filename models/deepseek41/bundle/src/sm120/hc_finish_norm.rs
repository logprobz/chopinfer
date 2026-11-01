//! Qualified HC finalization and cached input norm in an unchanged cubin.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::finish_norm::Entry;
use model_deepseek41_contract::geometry::Geometry;
pub const FAMILY: &str = "sinkhorn_hc_finish_norm";
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_finish_norm_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        mixer: super::hc_round_expand_mix::entry(g),
        norm: super::norm_quantize_cached::entry(g),
        prefix: format!("chop_{}_hc_finish_norm", g.kernel_tag),
    }
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                FAMILY,
                entry(g)
                    .cubin_spec(&cubin_name(g))
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

/// The qualified pair, with its tail on two CTAs per row at the measured
/// Flash buckets (`twin_buckets`), and both launches replaced by the
/// column-block producer and split finish at `hc_cols::buckets`. The
/// column pair replaced the one-launch boundary (`hc_completion`) at C1 and
/// C8, its last buckets. With `draft_rows` (the `dspark_hc_draft_rows`
/// candidate) the pair, its chain producer and the column pair also cover
/// the drafter table's rows (`draft_buckets`), which only a drafter's
/// finish admission reaches.
pub fn register(
    backend: kernels_sm120::backend::Sm120,
    g: &Geometry,
    draft_rows: bool,
) -> kernels_sm120::backend::Sm120 {
    let backend = register_pair_with(backend, g, draft_rows);
    if g.id == model_deepseek41_contract::geometry::FLASH.id {
        let mut cols = super::hc_cols::buckets();
        if draft_rows {
            cols.extend(super::hc_cols::draft_buckets());
            cols.sort_by_key(|b| b.rows);
        }
        backend
            .with_sinkhorn_finish_norm_twin(super::hc_finish_twin::served(), twin_buckets())
            .with_sinkhorn_finish_norm_cols(
                super::hc_cols::producer(),
                super::hc_cols::finish(),
                cols,
            )
    } else {
        backend
    }
}

/// The drafter table's rows of the finish pair and their chain producer's
/// warps (`hc.draft_admit.verify`, `hc.draft_chain.verify`).
pub(crate) fn draft_buckets() -> Vec<(u32, u32)> {
    counted_buckets("hc.draft_chain.verify", 64, "warps")
        .into_iter()
        .filter(|&(rows, _)| selected_rows("hc.draft_admit.verify", 64).contains(&rows))
        .collect()
}

/// Measured (row bucket, warp count) pairs of the two-CTA tail: DSpark's
/// Verify-6 bucket (five drafts and the root).
pub fn twin_buckets() -> Vec<(u32, u32)> {
    counted_buckets("hc.finish_twin.verify", 64, "warps")
}

/// Only measured Flash decode and verify buckets use the 1024-thread tail.
/// The lab compares one-launch boundaries against exactly this pair.
pub fn register_pair(
    backend: kernels_sm120::backend::Sm120,
    g: &Geometry,
) -> kernels_sm120::backend::Sm120 {
    register_pair_with(backend, g, false)
}

fn register_pair_with(
    backend: kernels_sm120::backend::Sm120,
    g: &Geometry,
    draft_rows: bool,
) -> kernels_sm120::backend::Sm120 {
    if g.id == model_deepseek41_contract::geometry::FLASH.id {
        let (mut pairs, mut chains) = (pair_buckets(), chain_buckets());
        if draft_rows {
            for (rows, warps) in draft_buckets() {
                pairs.push(rows);
                chains.push((rows, warps));
            }
            pairs.sort_unstable();
            chains.sort_unstable();
        }
        backend
            .with_sinkhorn_finish_norm(entry(g), pairs)
            .with_sinkhorn_finish_norm_chain(super::hc_round_chain::entry(g), chains)
            .with_sinkhorn_finish_norm_rows(super::hc_round_rows::entry(g), row_buckets())
    } else {
        backend
    }
}

pub(crate) fn pair_buckets() -> Vec<u32> {
    selected_rows("hc.finish_pair.verify", 64)
}

pub(crate) fn chain_buckets() -> Vec<(u32, u32)> {
    counted_buckets("hc.finish_chain.verify", 64, "warps")
}

pub(crate) fn row_buckets() -> Vec<(u32, u32)> {
    counted_buckets("hc.finish_rows.verify", 64, "tile")
}

pub(crate) fn choice(
    g: &Geometry,
    op: &str,
    bucket: u32,
    rows: u32,
) -> Option<&'static kernel_build::tuning::Row> {
    use model_deepseek41_contract::tuning::{self, Family};
    tuning::select(
        g,
        Family::Hc,
        op,
        "f32",
        &[("groups", g.shape.hc.streams), ("width", g.shape.hidden)],
        bucket,
        rows,
    )
}

pub(crate) fn selected_rows(op: &str, bucket: u32) -> Vec<u32> {
    (1..=bucket)
        .filter(|&rows| {
            choice(
                &model_deepseek41_contract::geometry::FLASH,
                op,
                bucket,
                rows,
            )
            .is_some()
        })
        .collect()
}

pub(crate) fn counted_buckets(op: &str, bucket: u32, knob: &str) -> Vec<(u32, u32)> {
    (1..=bucket)
        .filter_map(|rows| {
            let row = choice(
                &model_deepseek41_contract::geometry::FLASH,
                op,
                bucket,
                rows,
            )?;
            Some((rows, row.knobs[knob].count().expect("HC count knob")))
        })
        .collect()
}

pub(crate) fn prefill_buckets(op: &str) -> Vec<((u32, u32), Vec<u32>)> {
    let mut buckets: Vec<((u32, u32), Vec<u32>)> = Vec::new();
    for rows in 1..=model_deepseek41_contract::serving::PREFILL_TOKENS {
        let Some(row) = choice(&model_deepseek41_contract::geometry::FLASH, op, 8192, rows) else {
            continue;
        };
        let pair = (
            row.knobs["tile"].count().expect("HC tile"),
            row.knobs["chains"].count().expect("HC chains"),
        );
        if let Some((_, counts)) = buckets.iter_mut().find(|(p, _)| *p == pair) {
            counts.push(rows);
        } else {
            buckets.push((pair, vec![rows]));
        }
    }
    buckets
}
