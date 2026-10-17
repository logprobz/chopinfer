//! A gated MLP of MXFP8 projections split over tensor-parallel ranks, as a
//! shared expert runs beside routed experts: gate and up split by columns
//! (each rank owns `inter / ranks` outputs), the clamped SwiGLU and its ceil
//! E4M3/32 quantization per rank (`SiluMulQuantizeQuery`), and down split
//! by rows of K (each rank's F32 partial over its `inter / ranks` slice,
//! folded at 128 K), added into the routed partial before the F32
//! all-reduce in rank order.
//!
//! The pinned geometry is 4 ranks of 576 columns (an inter width of 2304):
//! 576 is 18 whole quantization groups, so the column split keeps every
//! rank's codes and scales those of the unsplit row, and a rank's down K is
//! four 128-K folds and one of 64. The width is small (256) to keep the
//! oracle fast; the projections' K does not change the split.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::gated_mlp::SwigluClamp;
use kernel_core::gated_mlp::quantized::SiluMulQuantizeQuery;
use kernel_core::mxfp8_gemm::{self, Mxfp8FoldGemmQuery};
use kernel_core::precision::execution::BlockScaledExecution;
use kernel_core::precision::quantizer::BlockQuantization;
use kernel_core::quantize::{Quantized, Scales};

const RANKS: usize = 4;
const DIM: usize = 256;
const INTER: usize = 2304;
const LOCAL: usize = INTER / RANKS;
const ROWS: usize = 3;
/// The gate's per-term factor, 2^-22.
const GATE: f64 = 1.0 / 4_194_304.0;
/// The F32 unit roundoff.
const U: f64 = 1.0 / 16_777_216.0;

fn values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0) as f32 * scale
        })
        .collect()
}

fn quantized(values: &[f32], rows: usize, k: usize) -> Quantized {
    BlockQuantization::E4M3_CEIL32
        .apply(values, rows, k)
        .unwrap()
}

/// Columns `[start, start + width)` of every row, codes and 1x32 scales.
fn columns(matrix: &Quantized, start: usize, width: usize) -> Quantized {
    assert!(start.is_multiple_of(32) && width.is_multiple_of(32));
    let Scales::Ue8M0(scales) = &matrix.scales else {
        panic!("UE8M0 scales");
    };
    let (groups, from, count) = (matrix.k / 32, start / 32, width / 32);
    let codes = (0..matrix.rows)
        .flat_map(|r| matrix.codes[r * matrix.k + start..][..width].to_vec())
        .collect();
    let scales = (0..matrix.rows)
        .flat_map(|r| scales[r * groups + from..][..count].to_vec())
        .collect();
    Quantized::from_parts(
        matrix.rows,
        width,
        matrix.format,
        codes,
        Scales::Ue8M0(scales),
        1.0,
    )
    .unwrap()
}

/// Rows `[start, start + count)` of a matrix.
fn rows(matrix: &Quantized, start: usize, count: usize) -> Quantized {
    let Scales::Ue8M0(scales) = &matrix.scales else {
        panic!("UE8M0 scales");
    };
    let groups = matrix.k / 32;
    Quantized::from_parts(
        count,
        matrix.k,
        matrix.format,
        matrix.codes[start * matrix.k..][..count * matrix.k].to_vec(),
        Scales::Ue8M0(scales[start * groups..][..count * groups].to_vec()),
        1.0,
    )
    .unwrap()
}

fn fold_query(n: usize, k: usize, output: DType) -> Mxfp8FoldGemmQuery {
    Mxfp8FoldGemmQuery {
        m_capacity: ROWS as u32,
        n: n as u32,
        k: k as u32,
        groups: 1,
        activation_row_stride: k as u32,
        activation_group_stride: k as u32,
        output,
        execution: BlockScaledExecution::ExactDecodedBf16Mma,
    }
}

/// A fold kernel's F32 outputs `[rows, n]`: the CPU witness of the 128-K
/// folds over the exact decoded operands (`mxfp8_gemm::folded_dot`).
fn folded(activations: &Quantized, weights: &Quantized) -> Vec<f32> {
    let a = activations.decoded();
    let w = weights.decoded();
    let (k, n) = (activations.k, weights.rows);
    let narrow = |v: &[f64]| v.iter().map(|&v| v as f32).collect::<Vec<_>>();
    (0..activations.rows)
        .flat_map(|r| {
            let row = narrow(&a[r * k..][..k]);
            (0..n)
                .map(|c| mxfp8_gemm::folded_dot(&row, &narrow(&w[c * k..][..k])).unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn swiglu() -> SiluMulQuantizeQuery {
    SiluMulQuantizeQuery {
        width: LOCAL as u32,
        clamp: SwigluClamp::symmetric(10.0),
        amax_floor_bits: 1e-4_f32.to_bits(),
    }
}

struct Expert {
    w1: Quantized,
    w3: Quantized,
    w2: Quantized,
}

fn expert() -> Expert {
    Expert {
        w1: quantized(&values(INTER * DIM, 11, 0.25), INTER, DIM),
        w3: quantized(&values(INTER * DIM, 12, 0.25), INTER, DIM),
        w2: quantized(&values(DIM * INTER, 13, 0.05), DIM, INTER),
    }
}

/// The SwiGLU's quantized rows of the columns `[start, start + width)`:
/// gate and up from the fold kernel's F32 outputs of those weight rows.
fn hidden(x: &Quantized, expert: &Expert, start: usize, width: usize) -> Quantized {
    let gate = folded(x, &rows(&expert.w1, start, width));
    let up = folded(x, &rows(&expert.w3, start, width));
    SiluMulQuantizeQuery {
        width: width as u32,
        ..swiglu()
    }
    .apply(&gate, &up, x.rows)
    .unwrap()
}

#[test]
fn the_pinned_split_keeps_whole_groups_and_folds() {
    assert_eq!(LOCAL, 576);
    assert_eq!(LOCAL % 32, 0);
    assert_eq!(LOCAL % 64, 0);
    assert_eq!((LOCAL / 128, LOCAL % 128), (4, 64));
    swiglu().apply(&[0.0; LOCAL], &[0.0; LOCAL], 1).unwrap();
}

#[test]
fn column_split_swiglu_keeps_the_unsplit_codes_and_scales() {
    let x = quantized(&values(ROWS * DIM, 1, 3.0), ROWS, DIM);
    let expert = expert();
    let whole = hidden(&x, &expert, 0, INTER);
    for rank in 0..RANKS {
        let local = hidden(&x, &expert, rank * LOCAL, LOCAL);
        assert_eq!(local, columns(&whole, rank * LOCAL, LOCAL), "rank {rank}");
    }
    // A split off the group grid takes other amaxes: with columns 0 to 15
    // large, a slice starting at column 16 quantizes columns 16 to 31
    // against their own small amax, not the unsplit group's.
    let (gate, up): (Vec<f32>, Vec<f32>) = (0..64)
        .map(|c| if c < 16 { (3.0, 3.0) } else { (0.01, 0.0137) })
        .unzip();
    let at = |width: usize| SiluMulQuantizeQuery {
        width: width as u32,
        ..swiglu()
    };
    let unsplit = at(64).apply(&gate, &up, 1).unwrap().decoded();
    let shifted = at(32)
        .apply(&gate[16..48], &up[16..48], 1)
        .unwrap()
        .decoded();
    assert_ne!(&shifted[..16], &unsplit[16..32]);
}

/// The combine: each rank adds its F32 down partial to its routed partial,
/// and the all-reduce adds the ranks' sums in F32 from zero in rank order.
fn combine(routed: &[Vec<f32>], shared: &[Vec<f32>]) -> Vec<f32> {
    let len = routed[0].len();
    (0..len)
        .map(|i| {
            (0..RANKS).fold(0.0_f32, |total, rank| {
                total + (routed[rank][i] + shared[rank][i])
            })
        })
        .collect()
}

#[test]
fn row_split_down_partials_combine_within_the_unsplit_bound() {
    let x = quantized(&values(ROWS * DIM, 2, 3.0), ROWS, DIM);
    let expert = expert();
    let h = hidden(&x, &expert, 0, INTER);
    // The unsplit down projection: K = 2304, its f64 oracle and bound.
    let reference =
        mxfp8_gemm::oracle(fold_query(DIM, INTER, DType::F32), ROWS, &h, &expert.w2).unwrap();
    let routed = (0..RANKS as u64)
        .map(|rank| values(ROWS * DIM, 20 + rank, 0.5))
        .collect::<Vec<_>>();
    let partial = |rank: usize, offset: usize| {
        folded(
            &columns(&h, rank * LOCAL + offset, LOCAL),
            &columns(&expert.w2, rank * LOCAL + offset, LOCAL),
        )
    };
    let shared = (0..RANKS).map(|rank| partial(rank, 0)).collect::<Vec<_>>();
    let combined = combine(&routed, &shared);
    // Bound: the unsplit projection's gate (K = 2304), plus one rounding
    // per rank for each partial add and each all-reduce add.
    let check = |combined: &[f32], label: &str| -> Result<(), String> {
        for (i, &actual) in combined.iter().enumerate() {
            let routed_sum = (0..RANKS).map(|r| f64::from(routed[r][i])).sum::<f64>();
            let routed_abs = (0..RANKS)
                .map(|r| f64::from(routed[r][i]).abs())
                .sum::<f64>();
            let s = reference.sum_abs_products[i];
            let expected = routed_sum + reference.values[i];
            let bound = GATE * INTER as f64 * s + 2.0 * RANKS as f64 * U * (routed_abs + s);
            if (f64::from(actual) - expected).abs() > bound {
                return Err(format!(
                    "{label} {i}: {actual} vs {expected}, bound {bound}"
                ));
            }
        }
        Ok(())
    };
    check(&combined, "TP4 combine").unwrap();
    // Each rank's partial lies within its own K = 576 gate.
    for (rank, partial) in shared.iter().enumerate() {
        let local = mxfp8_gemm::oracle(
            fold_query(DIM, LOCAL, DType::F32),
            ROWS,
            &columns(&h, rank * LOCAL, LOCAL),
            &columns(&expert.w2, rank * LOCAL, LOCAL),
        )
        .unwrap();
        for (i, &actual) in partial.iter().enumerate() {
            let bound = GATE * LOCAL as f64 * local.sum_abs_products[i];
            assert!(
                (f64::from(actual) - local.values[i]).abs() <= bound,
                "rank {rank} partial {i}"
            );
        }
    }
    // Negative controls: a rank whose partial drops the last 64-K fold, or
    // whose slice is the neighbor's, is outside the bound.
    let mut short = shared.clone();
    short[1] = folded(
        &columns(&h, LOCAL, LOCAL - 64),
        &columns(&expert.w2, LOCAL, LOCAL - 64),
    );
    assert!(check(&combine(&routed, &short), "short").is_err());
    let mut swapped = shared.clone();
    swapped[2] = shared[3].clone();
    assert!(check(&combine(&routed, &swapped), "swapped").is_err());
}

#[test]
fn the_split_matches_the_vendor_order_within_one_bf16_rounding() {
    // The vendor runs the shared expert whole on every rank, rounds its
    // down output to BF16 and adds it after the routed all-reduce
    // (`model.py:904-905`, `:853`). The split adds F32 partials before the
    // all-reduce instead, so the two differ by that BF16 rounding and the
    // arithmetic's bound.
    let x = quantized(&values(ROWS * DIM, 3, 3.0), ROWS, DIM);
    let expert = expert();
    let h = hidden(&x, &expert, 0, INTER);
    let reference =
        mxfp8_gemm::oracle(fold_query(DIM, INTER, DType::F32), ROWS, &h, &expert.w2).unwrap();
    let routed = (0..RANKS as u64)
        .map(|rank| values(ROWS * DIM, 30 + rank, 0.5))
        .collect::<Vec<_>>();
    let shared = (0..RANKS)
        .map(|rank| {
            folded(
                &columns(&h, rank * LOCAL, LOCAL),
                &columns(&expert.w2, rank * LOCAL, LOCAL),
            )
        })
        .collect::<Vec<_>>();
    let split = combine(&routed, &shared);
    let whole = folded(&h, &expert.w2);
    let zeros = vec![vec![0.0_f32; ROWS * DIM]; RANKS];
    let routed_sum = combine(&routed, &zeros);
    let bf16 = |v: f32| bf16_to_f32(f32_to_bf16(v));
    let mut rounded = 0;
    for i in 0..ROWS * DIM {
        let vendor = routed_sum[i] + bf16(whole[i]);
        let half_ulp = (f64::from(bf16(whole[i])) - f64::from(whole[i])).abs();
        let s = reference.sum_abs_products[i];
        let bound = half_ulp
            + 2.0 * GATE * INTER as f64 * s
            + 4.0 * RANKS as f64 * U * (f64::from(routed_sum[i]).abs() + s);
        assert!(
            (f64::from(split[i]) - f64::from(vendor)).abs() <= bound,
            "{i}: split {} vendor {vendor}",
            split[i]
        );
        rounded += usize::from(half_ulp > 0.0);
    }
    assert!(rounded > 0, "the vendor's BF16 rounding never applied");
}
