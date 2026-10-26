//! Compact fetch catalog and its qualification cases.
pub use super::cubin_spec;
use super::{CubinSpec, DType, Refusal, mxfp8_fold};
pub type Entry = super::Entry<false>;

/// Capacity, N, K, groups, activation padding, output and stages of one lab
/// entry.
type LabShape = (u32, u32, u32, u32, u32, DType, u32);

/// Every row tile (the row counts the lab runs reach 16-, 32- and 64-row
/// tiles), both stage counts, final 64-K folds with K = 576 and 448, K
/// divisible by 128, groups, padded activation strides, both outputs, and
/// three wide projections at 512 to 8192 rows. Row-major
/// 64-row tiles spill below four folds, so every K here has at least four.
#[rustfmt::skip]
const LAB_SHAPES: &[LabShape] = &[
    (160, 72, 576, 1, 0, DType::F32, 2),
    (160, 136, 1280, 1, 32, DType::Bf16, 3),
    (160, 40, 448, 3, 32, DType::F32, 3),
    (160, 256, 512, 2, 0, DType::Bf16, 2),
    (8192, 5120, 2048, 1, 0, DType::F32, 2),
    (8192, 8192, 1280, 1, 0, DType::F32, 2),
    (8192, 5120, 576, 1, 0, DType::F32, 3),
];

/// The lab entries whose packed twins the lab compares bit for bit
/// ([`LAB_SHAPES`]).
pub fn packed_lab_entries() -> Vec<Entry> {
    LAB_SHAPES
        .iter()
        .map(|&(capacity, n, k, groups, pad, output, stages)| {
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
                    "chop_mxfp8_prefill_compact_packed_lab_n{n}_k{k}_g{groups}_{}",
                    if output == DType::F32 { "f32" } else { "bf16" }
                ),
            )
            .unwrap();
            Entry::new(base, stages).unwrap()
        })
        .collect()
}

/// [`packed_lab_entries`] and their packed twins in one cubin.
pub fn packed_lab_cubin_spec(name: &str) -> Result<CubinSpec, Refusal> {
    let entries = packed_lab_entries();
    let mut spec = cubin_spec(name, &entries)?;
    for entry in &entries {
        spec.entries.extend(entry.packed_specs()?);
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Counts, View};
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers as Buffers;
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
    use kernel_core::precision::execution::BlockScaledExecution;
    fn entry() -> Entry {
        Entry::new(
            mxfp8_fold::Entry::new(
                Mxfp8FoldGemmQuery {
                    m_capacity: 8192,
                    n: 160,
                    k: 576,
                    groups: 2,
                    activation_row_stride: 1216,
                    activation_group_stride: 608,
                    output: DType::F32,
                    execution: BlockScaledExecution::NativeBlockScaledMma,
                },
                "compact_test",
            )
            .unwrap(),
            2,
        )
        .unwrap()
    }
    #[test]
    fn exact_plans_keep_group_strides_tail_bounds_and_alias_refusals() {
        let e = entry();
        for rows in [1, 16, 17, 32, 33, 65, 257, 2048, 8192] {
            let op = e.prepare(rows).unwrap();
            let sizes = e.base.bytes(rows);
            let b = Buffers {
                activations: View {
                    handle: 1,
                    bytes: sizes.activations,
                },
                activation_scales: View {
                    handle: 2,
                    bytes: sizes.activation_scales,
                },
                weights: View {
                    handle: 3,
                    bytes: sizes.weights,
                },
                weight_scales: View {
                    handle: 4,
                    bytes: sizes.weight_scales,
                },
                output: View {
                    handle: 5,
                    bytes: sizes.output,
                },
                live_rows: View {
                    handle: 6,
                    bytes: 4,
                },
                status: View {
                    handle: 7,
                    bytes: sizes.status + 12,
                },
                status_row_offset: 3,
            };
            let calls = op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[1].plan.grid, [2, rows.div_ceil(Entry::tile(rows)), 2]);
            assert_eq!(calls[1].plan.workspace_bytes, 0);
            assert!(
                op.launches(&b, View::EMPTY, Counts::rows(rows - 1))
                    .is_err()
            );
            let mut bad = b;
            bad.status.bytes -= 1;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            for source in [
                b.activations,
                b.activation_scales,
                b.weights,
                b.weight_scales,
                b.live_rows,
                b.status,
            ] {
                let mut bad = b;
                bad.output.handle = source.handle;
                assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            }
            let mut bad = b;
            bad.status_row_offset = u64::MAX;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
        }
        assert!(e.prepare(0).is_err());
        assert!(e.prepare(8193).is_err());
        assert!(e.plan(1, 48).is_err());
        assert!(Entry::new(e.base, 1).is_err());
    }
    #[test]
    fn packed_twins_change_only_the_weight_address() {
        let e = entry();
        let row_major = e.specs().unwrap();
        for (tile, spec) in [16, 32, 64]
            .into_iter()
            .zip(row_major[1..].iter().step_by(2))
        {
            let packed = e.packed_spec(tile).unwrap();
            assert_eq!(spec.symbol, e.symbol(tile));
            assert_eq!(packed.symbol, format!("{}_packed_v1", spec.symbol));
            assert_eq!(packed.body, spec.body.replace(">(", ",true>("));
            assert_eq!(packed.params, spec.params);
            assert_eq!(packed.max_threads, spec.max_threads);
        }
        assert!(e.packed_spec(48).is_err());
        let mut odd = e.clone();
        odd.base.query.k = 544;
        assert!(odd.packed_spec(32).is_err());
        let lab = packed_lab_cubin_spec("lab").unwrap();
        assert_eq!(lab.entries.len(), 10 * packed_lab_entries().len());
    }
}
