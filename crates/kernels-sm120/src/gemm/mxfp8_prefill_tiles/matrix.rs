//! Matrix fetch catalog and its qualification cases.
pub use super::cubin_spec;
use super::{CubinSpec, DType, Refusal, mxfp8_fold};
pub type Entry = super::Entry<true>;

/// Capacity, N, K, groups, activation padding, output, row tile and stages
/// of one lab entry.
type LabShape = (u32, u32, u32, u32, u32, DType, u32, u32);

/// Every row tile (32, 64 and 128) and selected stage count,
/// final 64-K folds with K = 576 and 448, K divisible by 128, groups, padded
/// activation strides, both outputs, and the wide prefill shapes with
/// the tiles they serve at 448, 2496 and 8192 rows.
#[rustfmt::skip]
const LAB_SHAPES: &[LabShape] = &[
    (320, 72, 576, 1, 0, DType::F32, 32, 3),
    (320, 136, 1280, 1, 32, DType::Bf16, 64, 3),
    (320, 40, 448, 3, 32, DType::F32, 128, 2),
    (320, 256, 512, 2, 0, DType::Bf16, 64, 2),
    (8192, 1280, 5120, 1, 0, DType::F32, 32, 3),
    (8192, 1280, 5120, 1, 0, DType::F32, 64, 2),
    (8192, 1280, 5120, 1, 0, DType::F32, 128, 2),
    (8192, 8192, 1280, 1, 0, DType::F32, 128, 2),
    (8192, 1024, 4096, 2, 0, DType::F32, 64, 2),
    (8192, 1024, 4096, 2, 0, DType::F32, 128, 2),
    (8192, 5120, 2048, 1, 0, DType::F32, 128, 2),
];

/// The lab entries whose packed twins the lab compares bit for bit
/// ([`LAB_SHAPES`]).
pub fn packed_lab_entries() -> Vec<Entry> {
    LAB_SHAPES
        .iter()
        .map(|&(capacity, n, k, groups, pad, output, tile, stages)| {
            let q = kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
                m_capacity: capacity,
                n,
                k,
                groups,
                activation_group_stride: k + pad,
                activation_row_stride: groups * (k + pad) + pad,
                output,
                execution:
                    kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
            };
            let base = mxfp8_fold::Entry::new(
                q,
                format!(
                    "chop_mxfp8_prefill_matrix_packed_lab_n{n}_k{k}_g{groups}_{}",
                    if output == DType::F32 { "f32" } else { "bf16" }
                ),
            )
            .unwrap();
            Entry::new(base, stages, tile).unwrap()
        })
        .collect()
}

/// [`packed_lab_entries`] and their packed twins in one cubin.
pub fn packed_lab_cubin_spec(name: &str) -> Result<CubinSpec, Refusal> {
    let entries = packed_lab_entries();
    let mut spec = cubin_spec(name, &entries)?;
    for entry in &entries {
        spec.entries.push(entry.packed_spec()?);
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{mxfp8_gemm::Mxfp8FoldGemmQuery, precision::execution::BlockScaledExecution};
    #[test]
    fn tiles_cover_tails_and_refuse_oversized_rings() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 8192,
            n: 160,
            k: 576,
            groups: 2,
            activation_row_stride: 1216,
            activation_group_stride: 608,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let base = mxfp8_fold::Entry::new(q, "matrix_test").unwrap();
        for (m, s) in [(32, 3), (64, 2), (64, 3), (128, 2)] {
            let e = Entry::new(base.clone(), s, m).unwrap();
            for rows in [1, 32, 64, 65, 192, 2496, 8192] {
                let p = e.plan(rows, m).unwrap();
                assert_eq!(p.grid, [2, rows.div_ceil(m), 2]);
                assert!(p.dynamic_shared_bytes <= 101376);
                assert_eq!(e.prepare(rows).unwrap().info().workspace, 0);
            }
            assert!(e.prepare(0).is_err());
        }
        assert!(Entry::new(base.clone(), 3, 128).is_err());
        assert!(Entry::new(base.clone(), 4, 64).is_err());
        assert!(Entry::new(base, 2, 16).is_err());
    }
    #[test]
    fn packed_twins_change_only_the_weight_address() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 8192,
            n: 160,
            k: 576,
            groups: 2,
            activation_row_stride: 1216,
            activation_group_stride: 608,
            output: DType::Bf16,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let base = mxfp8_fold::Entry::new(q, "matrix_test").unwrap();
        for (m, s) in [(32, 3), (64, 2), (128, 2)] {
            let e = Entry::new(base.clone(), s, m).unwrap();
            let spec = e.specs().unwrap().remove(1);
            let packed = e.packed_spec().unwrap();
            assert_eq!(spec.symbol, e.symbol(m));
            assert_eq!(packed.symbol, format!("{}_packed_v1", spec.symbol));
            assert_eq!(packed.body, spec.body.replace(">(", ",true>("));
            assert_eq!(packed.params, spec.params);
            assert_eq!(packed.max_threads, spec.max_threads);
        }
        let mut odd = Entry::new(base, 2, 64).unwrap();
        odd.base.query.n = 164;
        assert!(odd.packed_spec().is_err());
        let lab = packed_lab_cubin_spec("lab").unwrap();
        assert_eq!(lab.entries.len(), 4 * packed_lab_entries().len());
    }
}
