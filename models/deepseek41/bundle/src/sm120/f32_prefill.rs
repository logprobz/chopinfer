//! Exact prefill routes qualified at SM120, 250 W, 2026-10-19.
//! Each row below passed eager and captured full-output bitwise comparison
//! with the old kernel. Admit only these measured row counts, with a fixed
//! 3 percent winning margin. No interval or 4096-row crossover is inferred.
//! Heads gather smaller row sets and retain their split decode routes.
//!
//! | N | K | Rows | Tile N | Old us | New us | Paired new/old |
//! |---:|---:|---:|---:|---:|---:|---:|
//! | 384 | 5120 | 257 | 32 | 1191.936 | 149.504 | 0.125430 |
//! | 384 | 5120 | 511 | 64 | 1204.224 | 235.520 | 0.195578 |
//! | 384 | 5120 | 512 | 64 | 1210.368 | 235.520 | 0.193929 |
//! | 384 | 5120 | 513 | 64 | 1208.320 | 235.520 | 0.194915 |
//! | 384 | 5120 | 2047 | 32 | 2398.208 | 679.936 | 0.282664 |
//! | 384 | 5120 | 2048 | 32 | 2399.232 | 675.872 | 0.281690 |
//! | 384 | 5120 | 2049 | 32 | 2398.208 | 698.368 | 0.291204 |
//! | 384 | 5120 | 8191 | 64 | 8082.432 | 2111.488 | 0.260859 |
//! | 384 | 5120 | 8192 | 64 | 8074.240 | 2115.584 | 0.261950 |
//! | 512 | 5120 | 257 | 32 | 1193.984 | 151.552 | 0.126496 |
//! | 512 | 5120 | 511 | 64 | 1226.752 | 235.520 | 0.191667 |
//! | 512 | 5120 | 512 | 64 | 1228.800 | 235.520 | 0.191348 |
//! | 512 | 5120 | 513 | 64 | 1228.800 | 235.520 | 0.192630 |
//! | 512 | 5120 | 2047 | 64 | 3386.368 | 690.176 | 0.203614 |
//! | 512 | 5120 | 2048 | 64 | 3389.440 | 692.224 | 0.204272 |
//! | 512 | 5120 | 2049 | 64 | 3375.104 | 708.608 | 0.209541 |
//! | 512 | 5120 | 8191 | 64 | 10659.809 | 2513.920 | 0.236120 |
//! | 512 | 5120 | 8192 | 64 | 10647.552 | 2521.088 | 0.235598 |
//! | 128 | 512 | 257 | 32 | 155.648 | 22.528 | 0.144737 |
//! | 128 | 512 | 511 | 32 | 157.696 | 22.528 | 0.142857 |
//! | 128 | 512 | 512 | 32 | 155.648 | 22.528 | 0.144737 |
//! | 128 | 512 | 513 | 32 | 155.648 | 22.528 | 0.144737 |
//! | 128 | 512 | 2047 | 64 | 165.888 | 34.848 | 0.210069 |
//! | 128 | 512 | 2048 | 64 | 165.888 | 34.848 | 0.210069 |
//! | 128 | 512 | 2049 | 64 | 165.888 | 36.864 | 0.222222 |
//! | 128 | 512 | 8191 | 64 | 450.560 | 102.400 | 0.227273 |
//! | 128 | 512 | 8192 | 64 | 448.512 | 102.432 | 0.228382 |
//! | 32 | 5120 | 257 | 64 | 1175.552 | 141.312 | 0.120209 |
//! | 32 | 5120 | 511 | 64 | 1175.552 | 147.456 | 0.125003 |
//! | 32 | 5120 | 512 | 64 | 1171.456 | 147.456 | 0.125654 |
//! | 32 | 5120 | 513 | 64 | 1169.408 | 147.456 | 0.126316 |
//! | 32 | 5120 | 2047 | 64 | 1193.984 | 202.752 | 0.170690 |
//! | 32 | 5120 | 2048 | 64 | 1189.888 | 204.800 | 0.172117 |
//! | 32 | 5120 | 2049 | 64 | 1198.080 | 219.136 | 0.182906 |
//! | 32 | 5120 | 8191 | 64 | 1814.560 | 565.248 | 0.311161 |
//! | 32 | 5120 | 8192 | 64 | 1821.696 | 563.200 | 0.309162 |
//! | 128 | 5120 | 257 | 32 | 1177.600 | 147.456 | 0.125217 |
//! | 128 | 5120 | 511 | 32 | 1173.504 | 153.600 | 0.130890 |
//! | 128 | 5120 | 512 | 32 | 1179.648 | 153.600 | 0.129758 |
//! | 128 | 5120 | 513 | 32 | 1181.696 | 153.600 | 0.129983 |
//! | 128 | 5120 | 2047 | 64 | 1243.136 | 292.864 | 0.235585 |
//! | 128 | 5120 | 2048 | 64 | 1249.280 | 292.864 | 0.234043 |
//! | 128 | 5120 | 2049 | 64 | 1251.328 | 307.200 | 0.245499 |
//! | 128 | 5120 | 8191 | 64 | 3495.936 | 931.840 | 0.266276 |
//! | 128 | 5120 | 8192 | 64 | 3504.128 | 931.840 | 0.266315 |
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, cubin_spec};
use model_deepseek41_contract::geometry::Geometry;

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

pub const BUCKETS: &[Bucket] = &[
    Bucket {
        n: 384,
        k: 5120,
        rows: 257,
        features: 32,
    },
    Bucket {
        n: 384,
        k: 5120,
        rows: 511,
        features: 64,
    },
    Bucket {
        n: 384,
        k: 5120,
        rows: 512,
        features: 64,
    },
    Bucket {
        n: 384,
        k: 5120,
        rows: 513,
        features: 64,
    },
    Bucket {
        n: 384,
        k: 5120,
        rows: 2047,
        features: 32,
    },
    Bucket {
        n: 384,
        k: 5120,
        rows: 2048,
        features: 32,
    },
    Bucket {
        n: 384,
        k: 5120,
        rows: 2049,
        features: 32,
    },
    Bucket {
        n: 384,
        k: 5120,
        rows: 8191,
        features: 64,
    },
    Bucket {
        n: 384,
        k: 5120,
        rows: 8192,
        features: 64,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 257,
        features: 32,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 511,
        features: 64,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 512,
        features: 64,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 513,
        features: 64,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 2047,
        features: 64,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 2048,
        features: 64,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 2049,
        features: 64,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 8191,
        features: 64,
    },
    Bucket {
        n: 512,
        k: 5120,
        rows: 8192,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 257,
        features: 32,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 511,
        features: 32,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 512,
        features: 32,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 513,
        features: 32,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 2047,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 2048,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 2049,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 8191,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 512,
        rows: 8192,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 257,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 511,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 512,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 513,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 2047,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 2048,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 2049,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 8191,
        features: 64,
    },
    Bucket {
        n: 32,
        k: 5120,
        rows: 8192,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 257,
        features: 32,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 511,
        features: 32,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 512,
        features: 32,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 513,
        features: 32,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 2047,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 2048,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 2049,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 8191,
        features: 64,
    },
    Bucket {
        n: 128,
        k: 5120,
        rows: 8192,
        features: 64,
    },
];

pub fn serving_entry(g: &Geometry, entry: Entry) -> Result<Entry, String> {
    if g.id != model_deepseek41_contract::geometry::FLASH.id {
        return Ok(entry);
    }
    let q = entry.query();
    let buckets = BUCKETS
        .iter()
        .filter(|bucket| (bucket.n, bucket.k) == (q.n, q.k))
        .map(|bucket| (bucket.rows, bucket.features))
        .collect();
    entry
        .with_prefill_buckets(buckets)
        .map_err(|e| e.to_string())
}
