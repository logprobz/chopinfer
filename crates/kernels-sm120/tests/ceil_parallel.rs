//! Keep the duplicated group arithmetic identical while legacy locks stay fixed.
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernels_sm120::row::{block_quantize, silu_mul_quantize};

#[test]
fn group_bodies_keep_the_legacy_arithmetic_and_reduction_order() {
    let new = include_str!("../cuda/row/ceil_parallel.cuh");
    for (old, start, end) in [
        (
            include_str!("../cuda/row/block_quantize.cuh"),
            "template<int Format, bool F32Input",
            "// One CTA validates",
        ),
        (
            include_str!("../cuda/row/silu_mul_quantize.cuh"),
            "template<uint32_t GateMaxBits, uint32_t UpLimitBits, uint32_t FloorBits",
            "// One warp per row",
        ),
    ] {
        let body = &old[old.find(start).unwrap()..old.find(end).unwrap()];
        assert!(new.contains(&body.replace(
            "const uint32_t lane = threadIdx.x;",
            "const uint32_t lane = threadIdx.x & 31;"
        )));
    }
}

#[test]
fn parallel_plans_are_opt_in_and_match_the_compiled_thread_bound() {
    for width in [32, 512, 576, 1280, 2048, 3456, 5120, 6144, 8192, 32768] {
        for input in [DType::Bf16, DType::F32] {
            let old = block_quantize::Entry::new(
                BlockQuantizeQuery {
                    width,
                    input,
                    format: BlockQuantization::E4M3_CEIL32,
                    output: QuantizeOutput::Codes,
                },
                32,
                "chop_ceil_test_v1",
            )
            .unwrap();
            assert_eq!(old.plan(1).unwrap().block, [32, 1, 1]);
            let new = old.clone().parallel().unwrap();
            assert_eq!(old.spec().params, new.spec().params);
            for rows in [1, 8, 32] {
                let plan = new.plan(rows).unwrap();
                assert_eq!(plan.grid, [rows, 1, 1]);
                assert_eq!(plan.block[0], (width / 32).min(32) * 32);
                assert_eq!(plan.block[0], new.spec().max_threads);
                assert!(plan.pdl);
            }
            assert!(new.plan(33).is_err());
            let mut fake = old;
            fake.query.output = QuantizeOutput::Bf16;
            assert!(fake.parallel().is_err());
        }
    }
    for old in silu_mul_quantize::lab_entries() {
        let new = old.clone().parallel().unwrap();
        assert_eq!(old.spec().params, new.spec().params);
        assert_eq!(new.plan(8).unwrap().block[0], new.spec().max_threads);
    }
}
