//! Whole-K exact-F32 row tiles of Flash TP4 verify steps
//! (`kernels_sm120::f32_projection::rows`). Each `f32.rows.verify` tuning
//! row names a projection, the verify rows it serves and its measured tile;
//! the family registers one candidate per row at the package's verify rows.
use kernel_build::CubinSpec;
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, Refusal, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::{
    Entry,
    rows::{self, Candidate, Tile},
};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    tuning::{self, Family},
    verify_bounds::F32_ROWS,
};

pub const FAMILY: &str = "f32_verify_rows";
pub const CUBIN: &str = "f32_verify_rows_deepseek41_sm120";
/// The tuning site of the row tiles.
pub const OP: &str = "f32.rows.verify";

/// A measured tile and the verify rows it serves.
#[derive(Clone, Debug)]
pub struct Served {
    pub rows: [u32; 2],
    pub candidate: Candidate,
}

/// The projection of `n` outputs from `k` inputs at the package's verify
/// rows, which stems its symbols.
pub fn entry(n: u32, k: u32) -> Result<Entry, Refusal> {
    Entry::new(
        GemmQuery {
            m_capacity: F32_ROWS,
            n,
            k,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: ExecutionPhase::Verify,
            capacity: F32_ROWS,
            graph: true,
        },
        &format!("chop_{}_verify_f32_n{n}_k{k}", FLASH.kernel_tag),
    )
}

fn count(row: &kernel_build::tuning::Row, name: &str) -> u32 {
    row.knobs
        .get(name)
        .and_then(|v| v.count())
        .unwrap_or_else(|| panic!("{}: missing count {name}", row.key))
}

/// Every measured tile of Flash TP4, in table order: by projection, then
/// by rows.
pub fn served() -> Result<Vec<(u32, u32, Served)>, Refusal> {
    let Some(table) = tuning::table(&FLASH, Family::F32) else {
        return Ok(vec![]);
    };
    table
        .rows()
        .iter()
        .filter(|row| {
            row.key.op == OP
                && row.key.weights == "bf16"
                && row.key.bucket == F32_ROWS
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
                    rows: row.key.rows.unwrap_or([1, F32_ROWS]),
                    candidate: Candidate::new(entry(n, k)?, tile, F32_ROWS)?,
                },
            ))
        })
        .collect()
}

/// The measured tiles of the projection of `n` outputs from `k` inputs.
pub fn tiles(n: u32, k: u32) -> Result<Vec<Served>, Refusal> {
    Ok(served()?
        .into_iter()
        .filter(|&(tn, tk, _)| (tn, tk) == (n, k))
        .map(|(_, _, served)| served)
        .collect())
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(vec![]);
    }
    let candidates: Vec<_> = served()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|(_, _, served)| served.candidate)
        .collect();
    Ok(vec![(
        FAMILY,
        rows::cubin_spec(CUBIN, &candidates).map_err(|e| e.to_string())?,
    )])
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
        let served = served().unwrap();
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
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].1.entries.len(), 9);
        assert!(cubins(&[]).unwrap().is_empty());
    }
}
