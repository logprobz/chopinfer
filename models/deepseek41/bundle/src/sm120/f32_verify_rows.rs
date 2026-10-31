//! Whole-K exact-F32 row tiles of Flash TP4 steps
//! (`kernels_sm120::f32_projection::rows`). Each tuning row of a site names
//! a projection, the rows it serves and its measured tile; the site's
//! family registers one candidate per row at the site's bound, in its own
//! cubin: `f32.rows.verify` above the decode bucket, and `f32.rows.decode`
//! at decode and verify widths of the decode bucket or fewer.
use kernel_build::CubinSpec;
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, Refusal, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::{
    Entry,
    rows::{self, Candidate, Tile},
};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    serving::DECODE_BATCH,
    tuning::{self, Family},
    verify_bounds::F32_ROWS,
};

/// A tuning site of row tiles: its family and cubin, the bound that is its
/// registered capacity and tuning bucket, and the phase that stems its
/// symbols.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Site {
    pub family: &'static str,
    pub cubin: &'static str,
    pub op: &'static str,
    pub bound: u32,
    phase: ExecutionPhase,
    pub stem: &'static str,
}

/// Verify steps above the decode bucket, to the package's verify rows.
pub const VERIFY: Site = Site {
    family: "f32_verify_rows",
    cubin: "f32_verify_rows_deepseek41_sm120",
    op: "f32.rows.verify",
    bound: F32_ROWS,
    phase: ExecutionPhase::Verify,
    stem: "verify",
};
/// Decode steps and verify steps of the decode bucket or fewer rows.
pub const DECODE: Site = Site {
    family: "f32_decode_rows",
    cubin: "f32_decode_rows_deepseek41_sm120",
    op: "f32.rows.decode",
    bound: DECODE_BATCH,
    phase: ExecutionPhase::Decode,
    stem: "decode",
};
pub const SITES: [Site; 2] = [VERIFY, DECODE];

/// A measured tile and the rows it serves.
#[derive(Clone, Debug)]
pub struct Served {
    pub rows: [u32; 2],
    pub candidate: Candidate,
}

/// The projection of `n` outputs from `k` inputs at `site`'s bound, which
/// stems its symbols.
pub fn entry(site: Site, n: u32, k: u32) -> Result<Entry, Refusal> {
    Entry::new(
        GemmQuery {
            m_capacity: site.bound,
            n,
            k,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: site.phase,
            capacity: site.bound,
            graph: true,
        },
        &format!("chop_{}_{}_f32_n{n}_k{k}", FLASH.kernel_tag, site.stem),
    )
}

fn count(row: &kernel_build::tuning::Row, name: &str) -> u32 {
    row.knobs
        .get(name)
        .and_then(|v| v.count())
        .unwrap_or_else(|| panic!("{}: missing count {name}", row.key))
}

/// Every measured tile of `site` for Flash TP4, in table order: by
/// projection, then by rows.
pub fn served(site: Site) -> Result<Vec<(u32, u32, Served)>, Refusal> {
    let Some(table) = tuning::table(&FLASH, Family::F32) else {
        return Ok(vec![]);
    };
    table
        .rows()
        .iter()
        .filter(|row| {
            row.key.op == site.op
                && row.key.weights == "bf16"
                && row.key.bucket == site.bound
                && row.key.shape.get("ranks") == Some(&4)
                && row.key.shape.len() == 3
        })
        .map(|row| {
            let (n, k) = (row.key.shape["n"], row.key.shape["k"]);
            let tile = Tile {
                features: count(row, "features"),
                rows: count(row, "rows"),
                warp_features: count(row, "warp_features"),
                warp_rows: count(row, "warp_rows"),
                stages: count(row, "stages"),
                plane_k: count(row, "plane_k"),
                slices: count(row, "slices"),
            };
            Ok((
                n,
                k,
                Served {
                    rows: row.key.rows.unwrap_or([1, site.bound]),
                    candidate: Candidate::new(entry(site, n, k)?, tile, site.bound)?,
                },
            ))
        })
        .collect()
}

/// The measured tiles of `site` for the projection of `n` outputs from `k`
/// inputs.
pub fn tiles(site: Site, n: u32, k: u32) -> Result<Vec<Served>, Refusal> {
    Ok(served(site)?
        .into_iter()
        .filter(|&(tn, tk, _)| (tn, tk) == (n, k))
        .map(|(_, _, served)| served)
        .collect())
}

/// One cubin per site that has measured tiles.
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(vec![]);
    }
    let mut cubins = Vec::new();
    for site in SITES {
        let candidates: Vec<_> = served(site)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|(_, _, served)| served.candidate)
            .collect();
        if !candidates.is_empty() {
            cubins.push((
                site.family,
                rows::cubin_spec(site.cubin, &candidates).map_err(|e| e.to_string())?,
            ));
        }
    }
    Ok(cubins)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiles the verify-row lab measured fastest: the head's 256 by 32
    /// tile over every chain, the compressor's eight K slices of 64 by 32
    /// at 33 to 64 rows, and ten K slices of 64 by 64 for the router and
    /// the compressor above.
    #[test]
    fn measured_tiles_plan_every_row_they_serve() {
        let served = served(VERIFY).unwrap();
        let stem = |n| format!("chop_deepseek41_verify_f32_n{n}_k5120_rows");
        assert_eq!(
            served
                .iter()
                .map(|(n, k, t)| (*n, *k, t.rows, t.candidate.symbols()))
                .collect::<Vec<_>>(),
            [
                (384, [65, 192], "f64_r64_w32x16_s2_k64_z10"),
                (512, [33, 64], "f64_r32_w32x16_s3_k64_z8"),
                (512, [65, 192], "f64_r64_w32x16_s2_k64_z10"),
                (32320, [33, 192], "f256_r32_w32x32_s2_k64"),
            ]
            .map(|(n, rows, tag)| {
                let mut symbols = vec![
                    format!("{}_prepare_v1", stem(n)),
                    format!("{}_{tag}_v1", stem(n)),
                ];
                if tag.contains("_z") {
                    symbols.push(format!("{}_reduce_v1", stem(n)));
                }
                (n, 5120, rows, symbols)
            })
        );
        for (_, _, t) in &served {
            assert!(t.rows[0] > 32 && t.rows[1] <= F32_ROWS);
            for rows in t.rows[0]..=t.rows[1] {
                t.candidate.plans(rows).unwrap();
            }
        }
        // One preparation and reducer per projection.
        let specs = cubins(&[FLASH]).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].0, VERIFY.family);
        assert_eq!(specs[0].1.name, VERIFY.cubin);
        assert_eq!(specs[0].1.entries.len(), 9);
        assert_eq!(specs[1].0, DECODE.family);
        assert_eq!(specs[1].1.name, DECODE.cubin);
        assert_eq!(specs[1].1.entries.len(), 4);
        assert!(cubins(&[]).unwrap().is_empty());
    }

    /// The head's decode tiles: 96 features by 8 rows through one decode
    /// tile, 96 by 16 through two, and the verify tile's 256 by 32 above,
    /// each over every chain.
    #[test]
    fn decode_tiles_cover_the_decode_bucket() {
        let served = served(DECODE).unwrap();
        let stem = "chop_deepseek41_decode_f32_n32320_k5120_rows";
        assert_eq!(
            served
                .iter()
                .map(|(n, k, t)| (*n, *k, t.rows, t.candidate.symbols()))
                .collect::<Vec<_>>(),
            [
                ([1, 8], "f96_r8_w32x8_s3_k64"),
                ([9, 16], "f96_r16_w32x16_s2_k64"),
                ([17, 32], "f256_r32_w32x32_s2_k64"),
            ]
            .map(|(rows, tag)| {
                let symbols = vec![format!("{stem}_prepare_v1"), format!("{stem}_{tag}_v1")];
                (32320, 5120, rows, symbols)
            })
        );
        let mut covered = 0;
        for (_, _, t) in &served {
            assert_eq!(t.rows[0], covered + 1);
            covered = t.rows[1];
            assert_eq!(t.candidate.capacity(), DECODE_BATCH);
            for rows in t.rows[0]..=t.rows[1] {
                t.candidate.plans(rows).unwrap();
            }
        }
        assert_eq!(covered, DECODE_BATCH);
        assert!(tiles(DECODE, 512, 5120).unwrap().is_empty());
    }
}
