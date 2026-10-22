//! Status-admitted cluster folds (`mxfp8_fold_cluster_fused::run`) on the
//! served cluster schedules of Flash TP4 at 16 and 32 rows, and the
//! row-sliced quantizer of the WO_a input: WO_a quantizing WO_b's input in
//! its merge, WO_b failing rows by status and WO_a's block flags, WQ_a
//! and WKV failing rows by status, and WQ_b's row-local fold on 48-column
//! tiles.
use kernel_build::{CubinSpec, EntrySpec};
use kernels_sm120::gemm::mxfp8_fold_cluster_fused::{Entry, Quantizer};
pub const FAMILY: &str = "fold_fused";
pub const CUBIN: &str = "fold_fused_sm120";
/// The served projections each row count fuses.
pub const PROJECTIONS: &[&str] = &[
    "attention.o-a",
    "attention.o-b",
    "attention.q-a",
    "attention.kv",
];
/// Slices and warps per row of the 8192-column WO_a input quantizer.
pub const SLICES: &[(u32, u32)] = &[(8, 8), (16, 8), (16, 4), (32, 8)];

/// The fused twin of the served cluster fold of `id` at `rows` rows.
pub fn fused(id: &str, rows: u32) -> Entry {
    let mut cluster =
        bundle_deepseek41::sm120::fold_cluster::served(id, rows).expect("a served cluster fold");
    let q = cluster.base.query;
    cluster.base.symbol = format!("chop_fold_fused_n{}_k{}_g{}_f32", q.n, q.k, q.groups);
    Entry {
        cluster,
        pending: if id == "attention.o-b" {
            q.activation_row_stride / 32
        } else {
            0
        },
        quantize: id == "attention.o-a",
    }
}
pub fn entries() -> Vec<Entry> {
    [16, 32]
        .into_iter()
        .flat_map(|rows| PROJECTIONS.iter().map(move |id| fused(id, rows)))
        .collect()
}
pub fn quantizers() -> Vec<Quantizer> {
    SLICES
        .iter()
        .map(|&(slices, warps)| Quantizer {
            width: 8192,
            slices,
            warps,
            f32_input: false,
            capacity: 32,
            symbol: format!("chop_fold_fused_sliced_w8192_s{slices}w{warps}_v1"),
        })
        .collect()
}
pub fn label(q: &Quantizer) -> String {
    format!("sliced{}w{}", q.slices, q.warps)
}
/// WQ_b's 48-column row-local tiles (`mxfp8_fold_local_wide.cuh`): folds
/// in flight, 0 for the pair loop (the deep rings swizzle).
pub const WIDE: &[u32] = &[0, 4];
pub const WIDE_TILE: u32 = 48;
pub fn wide_label(stages: u32) -> String {
    if stages == 0 {
        format!("wide{WIDE_TILE}")
    } else {
        format!("wide{WIDE_TILE}s{stages}x")
    }
}
pub fn wide_symbol(stages: u32) -> String {
    format!(
        "chop_fold_fused_{}_n8192_k1280_g1_f32_v1",
        wide_label(stages)
    )
}
/// The served WQ_b row-local fold's entry point with a 48-column tile.
pub fn wide_spec(stages: u32) -> EntrySpec {
    let served = bundle_deepseek41::sm120::fold_local_rows::served("attention.q-b", 32)
        .expect("the served WQ_b row-local fold");
    let q = served.base.query;
    let mut spec = served.specs().swap_remove(1);
    let shape = format!(
        "{},{},{},{},{},{WIDE_TILE}",
        q.n, q.k, q.groups, q.activation_row_stride, q.activation_group_stride
    );
    let args =
        "(activations,activation_scales,weights,weight_scales,output,workspace,live_rows,rows);";
    spec.symbol = wide_symbol(stages);
    spec.max_threads = WIDE_TILE * 4;
    spec.body = if stages == 0 {
        format!("chop::gemm::mxfp8_fold_local_wide::run<{shape},false>{args}")
    } else {
        format!("chop::gemm::mxfp8_fold_local_wide::run_deep<{shape},{stages},true,false>{args}")
    };
    spec
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut spec = kernels_sm120::gemm::mxfp8_fold_cluster_fused::cubin_spec(
        CUBIN,
        &entries(),
        &quantizers(),
        true,
    )
    .unwrap();
    spec.headers
        .push("#include \"gemm/mxfp8_fold_local_wide.cuh\"".into());
    spec.entries.extend(WIDE.iter().map(|&s| wide_spec(s)));
    vec![spec]
}
