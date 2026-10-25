//! Exact F32 prefill registrations with separate exact-row and band choices.
//! Exact rows retain precedence over the qualified packed-row bands; blocked
//! and wide overrides keep their existing order. Heads retain split decode.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, PrefillBand, cubin_spec};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "f32_prefill";
pub const TILE_FEATURES: [u32; 2] = [32, 64];

pub fn baseline_entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    let mut entries = super::f32_dense::entries(g)?;
    entries.extend(super::dspark_router::entries(g)?);
    Ok(entries)
}

pub fn cubin_name(g: &Geometry, features: u32) -> String {
    format!("f32_prefill_{}_n{features}_sm120", g.kernel_tag)
}

pub fn entries(g: &Geometry, features: u32) -> Result<Vec<Entry>, String> {
    baseline_entries(g)?
        .into_iter()
        .map(|entry| {
            entry
                .with_prefill_tiles(features)
                .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    geometries
        .iter()
        .flat_map(|g| TILE_FEATURES.map(move |features| (g, features)))
        .map(|(g, features)| {
            cubin_spec(cubin_name(g, features), &entries(g, features)?).map_err(|e| e.to_string())
        })
        .collect()
}

/// One measured shape and exact row count, with its winning feature tile.
#[derive(Clone, Copy, Debug)]
pub struct Bucket {
    pub n: u32,
    pub k: u32,
    pub rows: u32,
    pub features: u32,
}

/// Rows for one registered projection. The caller retains branch precedence.
pub fn rows_for(
    g: &Geometry,
    op: &str,
    q: kernel_core::contract::GemmQuery,
    ranks: Option<u32>,
) -> Vec<&'static kernel_build::tuning::Row> {
    use model_deepseek41_contract::tuning::{Family, table};
    let Some(table) = table(g, Family::F32) else {
        return Vec::new();
    };
    let mut shape: std::collections::BTreeMap<String, u32> =
        [("n".into(), q.n), ("k".into(), q.k)].into();
    if let Some(ranks) = ranks {
        shape.insert("ranks".into(), ranks);
    }
    table
        .rows()
        .iter()
        .filter(|row| {
            row.key.op == op
                && row.key.weights == "bf16"
                && row.key.shape == shape
                && row.key.bucket == q.m_capacity
        })
        .collect()
}

pub fn exact_buckets() -> Vec<Bucket> {
    use model_deepseek41_contract::tuning::{Family, table};
    table(&FLASH, Family::F32)
        .unwrap()
        .rows()
        .iter()
        .filter(|row| row.key.op == "f32.exact.prefill")
        .flat_map(|row| {
            let [lo, hi] = row.key.rows.unwrap_or([1, row.key.bucket]);
            (lo..=hi).map(move |rows| Bucket {
                n: row.key.shape["n"],
                k: row.key.shape["k"],
                rows,
                features: row.knobs["tile"].count().unwrap(),
            })
        })
        .collect()
}

pub fn serving_entry(g: &Geometry, entry: Entry) -> Result<Entry, String> {
    if g.id != model_deepseek41_contract::geometry::FLASH.id {
        return Ok(entry);
    }
    let q = entry.query();
    let mut buckets: Vec<_> = exact_buckets()
        .iter()
        .filter(|bucket| (bucket.n, bucket.k) == (q.n, q.k))
        .map(|bucket| (bucket.rows, bucket.features))
        .collect();
    for rows in super::f32_blocked::selected_rows(q.n, q.k) {
        if !buckets.iter().any(|&(r, _)| r == rows) {
            buckets.push((rows, 32));
        }
    }
    let entry = entry
        .with_prefill_buckets(buckets)
        .map_err(|e| e.to_string())?;
    let bands = rows_for(g, "f32.band.prefill", q, None)
        .into_iter()
        .map(|row| {
            assert_eq!(row.schedule, "tiled");
            let [lo, hi] = row.key.rows.unwrap_or([1, row.key.bucket]);
            PrefillBand {
                lo,
                hi,
                features: row.knobs["tile"].count().unwrap(),
            }
        })
        .collect();
    let entry = entry.with_prefill_bands(bands).map_err(|e| e.to_string())?;
    let entry = super::f32_blocked::serving_entry(g, entry)?;
    super::f32_wide::serving_entry(g, entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn bands_cover_packed_rows_and_preserve_every_exact_route_and_other_shape() {
        let mut changed = 0;
        for original in baseline_entries(&FLASH).unwrap() {
            let routed = serving_entry(&FLASH, original.with_split_decode()).unwrap();
            let exact = routed.clone().with_prefill_bands(vec![]).unwrap();
            let q = routed.query();
            for rows in 1..=8192 {
                let previous = exact.entry_for_rows(rows);
                let selected = routed.entry_for_rows(rows);
                let tile = match (q.n, q.k, rows) {
                    (512, 5120, 257..=1023) | (128, 512, 257..=4095) | (384, 5120, 257..=319) => 32,
                    (512, 5120, 1024..=8192) | (32, 5120, 257..=8192) | (128, 512, 4096..=8192) => {
                        64
                    }
                    _ => 0,
                };
                if previous.uses_prefill_tiles() {
                    assert_eq!(selected, previous, "exact {}x{} at {rows}", q.n, q.k);
                } else if tile != 0 {
                    assert_eq!(
                        selected.prefill_features(),
                        tile,
                        "{}x{} at {rows}",
                        q.n,
                        q.k
                    );
                    assert_eq!(selected.blocked_prefill_rows(), 0);
                    assert_eq!(selected.wide_prefill_features(), 0);
                    changed += 1;
                } else {
                    assert_eq!(
                        selected.symbol(),
                        previous.symbol(),
                        "{}x{} at {rows}",
                        q.n,
                        q.k
                    );
                    assert_eq!(selected.prefill_features(), previous.prefill_features());
                }
            }
        }
        assert!(changed > 23000);
    }

    #[test]
    fn bands_leave_other_geometries_unchanged() {
        let mut other = FLASH;
        other.id = "unqualified";
        for original in baseline_entries(&FLASH).unwrap() {
            let entry = original.with_split_decode();
            assert_eq!(serving_entry(&other, entry.clone()).unwrap(), entry);
        }
    }
}
