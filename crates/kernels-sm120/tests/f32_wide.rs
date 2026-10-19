use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::{Entry, wide};

fn entry(n: u32, k: u32) -> Entry {
    Entry::new(
        GemmQuery {
            m_capacity: 8192,
            n,
            k,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        },
        "test_wide",
    )
    .unwrap()
}

#[test]
fn wide_tiles_cover_tail_rows_and_features_in_both_arithmetic_paths() {
    for n in [32, 128, 384, 512] {
        let e = entry(n, 5120);
        for features in [64, 128] {
            let spec = wide::entry_spec(&e, features).unwrap();
            for rows in [1, 17, 129, 320, 8192] {
                let p = wide::plan(&e, features, rows).unwrap();
                assert_eq!(p.block, [spec.max_threads, 1, 1]);
                assert_eq!(
                    p.dynamic_shared_bytes,
                    if features == 64 { 40960 } else { 57344 }
                );
                assert!(p.pdl);
                for tensor in [false, true] {
                    let mut hits = vec![0_u8; (rows * n) as usize];
                    for y in 0..p.grid[1] {
                        for x in 0..p.grid[0] {
                            for thread in 0..features * 4 {
                                for r in 0..2 {
                                    for f in 0..4 {
                                        let (row, col) = if tensor {
                                            let warp = thread / 32;
                                            let lane = thread % 32;
                                            (
                                                y * 32
                                                    + (warp / (features / 16)) * 8
                                                    + r * 16
                                                    + (lane % 4) * 2
                                                    + f % 2,
                                                x * features
                                                    + (warp % (features / 16)) * 16
                                                    + lane / 4
                                                    + (f / 2) * 8,
                                            )
                                        } else {
                                            (
                                                y * 32 + thread / (features / 4) + r * 16,
                                                x * features
                                                    + thread % (features / 4)
                                                    + f * (features / 4),
                                            )
                                        };
                                        if row < rows && col < n {
                                            hits[(row * n + col) as usize] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    assert!(hits.iter().all(|&n| n == 1));
                }
            }
        }
    }
}

#[test]
fn wider_specs_keep_the_original_abi_and_reject_invalid_launches() {
    let e = entry(384, 5120);
    let before = e.clone().with_prefill_tiles(32).unwrap().entry_spec();
    for features in [64, 128] {
        let spec = wide::entry_spec(&e, features).unwrap();
        assert_eq!(spec.params, before.params);
        assert_eq!(spec.allowed_local_bytes, 0);
    }
    for (features, rows) in [(32, 1), (256, 1), (64, 0), (128, 8193)] {
        assert!(wide::plan(&e, features, rows).is_err());
    }
    assert!(wide::plan(&entry(384, 129), 64, 1).is_err());
}
