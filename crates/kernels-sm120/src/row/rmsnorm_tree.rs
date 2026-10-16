//! The HF RMSNorm with its sum of squares in a fixed 256-lane F32 tree
//! (`cuda/row/rmsnorm_tree.cuh`, [`RmsNormNumerics::HF_TREE_256`]): the
//! order as a CPU emulation ([`tree_sum`], [`hf_norm`]), which the lab
//! compares with the row engine's and the fused mixer's outputs bit for bit
//! on finite rows, beside the serial order the HF kind used before
//! ([`serial_sum`]), and the lab's cubins of the row engine's tree entries.
//! [`super::rmsnorm::RmsNorm`] selects the tree for an HF requirement with
//! these numerics.

use kernel_build::CubinSpec;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::RmsNormKind;
use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};

use super::rmsnorm::{RmsNorm, RmsNormTarget};

/// Lab-only mutant: each warp adds its lanes at distances 1, 2, 4, 8, 16.
pub const DISTANCES_REVERSED_DEFINE: &str = "CHOP_RMSNORM_TREE_TEST_DISTANCES_REVERSED";
/// The tree's lanes, and the columns one pass of the lanes covers.
pub const LANES: usize = 256;
pub const SPAN: usize = LANES * 8;
const WARP: usize = 32;
/// The widths the lab runs: the hidden row, a width with a partial last
/// pass, the query-down and latent widths, and one vector.
pub const LAB_WIDTHS: [u32; 5] = [4096, 7168, 1536, 512, 8];
/// The lab entries' prepared capacity and eps (the mixer's norm eps).
pub const LAB_CAPACITY: u32 = 8192;
pub const LAB_EPS: f32 = 1e-5;

/// The HF RMSNorm of `width`-wide BF16 rows at `eps` with the tree sum.
pub fn requirement(width: u32, eps: f32) -> RmsNormRequirement {
    RmsNormRequirement::bf16(
        width,
        eps,
        RmsNormKind::HfBf16Intermediate,
        false,
        RmsNormNumerics::HF_TREE_256,
    )
}

/// The sum of `squares` in the tree's order, with IEEE F32 adds: lane `l`
/// adds columns `8 l + 2048 i + j` from +0, each warp of 32 lanes adds
/// pairwise at distances 16 to 1, the eight warps at distances 4, 2, 1.
pub fn tree_sum(squares: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; LANES];
    for (lane, partial) in lanes.iter_mut().enumerate() {
        let mut base = 0;
        while base < squares.len() {
            for j in 0..8 {
                if let Some(&square) = squares.get(base + 8 * lane + j) {
                    *partial += square;
                }
            }
            base += SPAN;
        }
    }
    // Lane m adds lane m + d at each distance; lane 0 ends with the sum.
    let halve = |values: &mut [f32], distances: &[usize]| {
        for &d in distances {
            for m in 0..d {
                values[m] += values[m + d];
            }
        }
        values[0]
    };
    let mut warps = [0.0_f32; LANES / WARP];
    for (warp, total) in warps.iter_mut().enumerate() {
        *total = halve(
            &mut lanes[warp * WARP..(warp + 1) * WARP],
            &[16, 8, 4, 2, 1],
        );
    }
    halve(&mut warps, &[4, 2, 1])
}

/// The serial sum the HF kind's ordered numerics add: from +0, one add per
/// square in column order.
pub fn serial_sum(squares: &[f32]) -> f32 {
    squares.iter().fold(0.0_f32, |s, &t| s + t)
}

/// The HF RMSNorm of BF16 row `x` with BF16 `weight` at `eps`, its sum of
/// squares `sum` (for example [`tree_sum`]): rounded squares,
/// `1 / sqrt(sum / width + eps)`, the normalized value rounded to BF16, then
/// the weight's product rounded to BF16. The device's bits for finite rows.
pub fn hf_norm(x: &[u16], weight: &[u16], eps: f32, sum: fn(&[f32]) -> f32) -> Vec<u16> {
    let squares: Vec<f32> = x
        .iter()
        .map(|&v| {
            let v = bf16_to_f32(v);
            v * v
        })
        .collect();
    let inv = 1.0 / (sum(&squares) / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(&v, &w)| {
            let n = bf16_to_f32(f32_to_bf16(bf16_to_f32(v) * inv));
            f32_to_bf16(n * bf16_to_f32(w))
        })
        .collect()
}

/// The exact sum of `squares` (F64 adds of F32 values, whose error is far
/// below an F32 ulp of the sum).
pub fn exact_sum(squares: &[f32]) -> f64 {
    squares.iter().map(|&t| f64::from(t)).sum()
}

/// The lab's row-engine tree entry of `width`, and its symbol
/// `chop_row_rmsnorm_tree_w{width}{suffix}_v1`.
pub fn lab_norm(width: u32, suffix: &str) -> (RmsNorm, String) {
    let norm = RmsNorm::new(
        requirement(width, LAB_EPS),
        LAB_CAPACITY,
        RmsNormTarget::SM120F,
    )
    .expect("the lab's tree widths are valid");
    (norm, format!("chop_row_rmsnorm_tree_w{width}{suffix}_v1"))
}

/// The serial-order HF RMSNorm of the hidden width (`row/row.cuh`), the
/// lab's baseline beside the tree, and its symbol
/// `chop_row_rmsnorm_ordered_w4096_v1`.
pub fn lab_ordered_norm() -> (RmsNorm, String) {
    let requirement = RmsNormRequirement {
        numerics: Some(RmsNormNumerics::HF_ORDERED),
        ..requirement(LAB_WIDTHS[0], LAB_EPS)
    };
    let norm = RmsNorm::new(requirement, LAB_CAPACITY, RmsNormTarget::SM120F)
        .expect("the ordered baseline is valid");
    (
        norm,
        format!("chop_row_rmsnorm_ordered_w{}_v1", LAB_WIDTHS[0]),
    )
}

/// The lab's cubins: the tree entry of every [`LAB_WIDTHS`] width and the
/// serial-order baseline ([`lab_ordered_norm`]) in `rmsnorm_tree_sm120`,
/// and the reversed-distances mutant of the hidden width
/// (`rmsnorm_tree_distances_reversed_sm120`).
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let (first, symbol) = lab_norm(LAB_WIDTHS[0], "");
    let mut served = first.cubin_spec("rmsnorm_tree_sm120", symbol);
    for &width in &LAB_WIDTHS[1..] {
        let (norm, symbol) = lab_norm(width, "");
        served.entries.push(norm.entry(symbol));
    }
    let (ordered, symbol) = lab_ordered_norm();
    served.entries.push(ordered.entry(symbol));
    let (norm, symbol) = lab_norm(LAB_WIDTHS[0], "_distances_reversed");
    let mut mutant = norm.cubin_spec("rmsnorm_tree_distances_reversed_sm120", symbol);
    mutant.defines = vec![(DISTANCES_REVERSED_DEFINE.to_owned(), "1".to_owned())];
    vec![served, mutant]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_sum_follows_the_lane_and_warp_order() {
        assert_eq!(tree_sum(&[1.0; 4096]), 4096.0);
        assert_eq!(tree_sum(&[]), 0.0);
        assert_eq!(tree_sum(&[3.0; 8]), 24.0);
        // One large square, then small ones: the serial adds round each
        // small square away; the tree adds 16 of them per lane first.
        let mut squares = vec![2.0_f32.powi(-25); 4096];
        squares[0] = 1.0;
        assert_eq!(serial_sum(&squares), 1.0);
        let tree = tree_sum(&squares);
        assert!(tree > 1.0, "{tree}");
        assert!((f64::from(tree) - exact_sum(&squares)).abs() < 2.0_f64.powi(-20));
        // Lane 1's first column is column 8; column 2048 is lane 0's ninth.
        let mut marked = vec![0.0_f32; 4096];
        marked[8] = 1.0;
        marked[2048] = 2.0;
        assert_eq!(tree_sum(&marked), 3.0);
        // A partial last pass: 7168 columns.
        assert_eq!(tree_sum(&[0.5; 7168]), 3584.0);
    }

    #[test]
    fn hf_norm_rounds_as_the_device() {
        let one = f32_to_bf16(1.0);
        let row = vec![one; 4096];
        let out = hf_norm(&row, &row, 0.0, tree_sum);
        assert!(out.iter().all(|&v| v == one));
        let half = f32_to_bf16(0.5);
        let two = f32_to_bf16(2.0);
        let out = hf_norm(&vec![half; 512], &vec![two; 512], 0.0, serial_sum);
        assert!(out.iter().all(|&v| v == two));
    }

    #[test]
    fn lab_cubins() {
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "rmsnorm_tree_sm120",
                "rmsnorm_tree_distances_reversed_sm120"
            ]
        );
        assert_eq!(specs[0].entries.len(), LAB_WIDTHS.len() + 1);
        assert_eq!(
            specs[0].entries[LAB_WIDTHS.len()].symbol,
            "chop_row_rmsnorm_ordered_w4096_v1"
        );
        assert!(
            specs[0].entries[LAB_WIDTHS.len()]
                .body
                .starts_with("chop::row::rmsnorm<4096,0,3,false>")
        );
        assert_eq!(specs[0].entries[0].symbol, "chop_row_rmsnorm_tree_w4096_v1");
        assert!(
            specs[0]
                .headers
                .iter()
                .any(|h| h.contains("rmsnorm_tree.cuh"))
        );
        assert_eq!(
            specs[1].defines,
            [(DISTANCES_REVERSED_DEFINE.into(), "1".into())]
        );
        assert_eq!(
            specs[1].entries[0].symbol,
            "chop_row_rmsnorm_tree_w4096_distances_reversed_v1"
        );
    }
}
