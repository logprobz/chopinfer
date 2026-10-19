//! Exact-F32 projections with row expansion and a fused ordered merge.
//! Original and cached families remain locked for comparisons.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, cubin_spec};
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "f32_split";

pub fn cubin_name(g: &Geometry) -> String {
    format!("f32_split_{}_sm120", g.kernel_tag)
}

pub fn entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    let mut entries = super::f32_dense::entries(g)?;
    entries.extend(super::dspark_router::entries(g)?);
    let mut entries: Vec<_> = entries.into_iter().map(Entry::with_split_decode).collect();
    // TP1 N=129280, K=5120 at SM120, 250 W, 2026-10-19:
    // rows | old us     | split us  | paired new/old
    //   33 |  21065.727 |  7839.744 | 0.372053
    //   48 |  25672.705 |  8310.784 | 0.323743
    //   64 |  34449.410 | 10798.080 | 0.313288
    //  128 |  71147.520 | 25366.592 | 0.355800
    //  192 | 106509.310 | 38186.082 | 0.355776
    //  256 | 142274.560 | 50843.742 | 0.356169
    // Every measured bucket wins by more than the 3 percent margin, so
    // retain the 33-256 split route. Fivefold speedup is a reported goal,
    // not a qualification requirement. The prepared op uses the original
    // entry above 256; prefill promotion applies only to full-row shapes.
    entries.push(super::f32_head::entry(g, ranks)?.with_split_decode());
    Ok(entries)
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            let mut all: Vec<Entry> = Vec::new();
            for &ranks in SUPPORTED_RANKS {
                if let Ok(entries) = entries(g, ranks) {
                    for entry in entries {
                        if !all.iter().any(|e| e.symbol() == entry.symbol()) {
                            all.push(entry);
                        }
                    }
                }
            }
            Ok((
                FAMILY,
                cubin_spec(cubin_name(g), &all).map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
