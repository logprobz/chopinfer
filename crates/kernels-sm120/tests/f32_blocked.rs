use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::{Entry, blocked};

fn entry(k: u32) -> Entry {
    Entry::new(
        GemmQuery {
            m_capacity: 8192,
            n: 384,
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
        "test_blocked",
    )
    .unwrap()
}
#[test]
fn blocked_tiles_cover_every_output_once_and_match_launch_bounds() {
    let e = entry(5120);
    for tile in [32, 64] {
        let spec = blocked::entry_spec(&e, tile).unwrap();
        for rows in [1, 17, 127, 129, 1000, 1024, 2048, 4095, 8150, 8191, 8192] {
            let p = blocked::plan(&e, tile, rows).unwrap();
            assert_eq!(p.block, [spec.max_threads, 1, 1]);
            assert_eq!(p.cluster, [1, 1, 1]);
            assert!(p.dynamic_shared_bytes <= 101376);
            let mut hits = vec![0_u8; (rows * 384) as usize];
            for y in 0..p.grid[1] {
                for x in 0..p.grid[0] {
                    for t in 0..128 {
                        for r in 0..tile / 16 {
                            for f in 0..4 {
                                let row = y * tile + t / 8 + r * 16;
                                let col = x * 32 + t % 8 + f * 8;
                                if row < rows && col < 384 {
                                    hits[(row * 384 + col) as usize] += 1;
                                }
                            }
                        }
                    }
                }
            }
            assert!(hits.iter().all(|&v| v == 1));
            hits.fill(0);
            for y in 0..p.grid[1] {
                for x in 0..p.grid[0] {
                    for thread in 0..128 {
                        let warp = thread / 32;
                        let lane = thread % 32;
                        for t in 0..tile / 16 {
                            for v in 0..4 {
                                let row =
                                    y * tile + (warp / 2) * 8 + t * 16 + (lane % 4) * 2 + v % 2;
                                let col = x * 32 + (warp % 2) * 16 + lane / 4 + (v / 2) * 8;
                                if row < rows && col < 384 {
                                    hits[(row * 384 + col) as usize] += 1;
                                }
                            }
                        }
                    }
                }
            }
            assert!(hits.iter().all(|&v| v == 1));
        }
    }
    for (tile, rows) in [(16, 1), (128, 1), (32, 0), (32, 8193)] {
        assert!(blocked::plan(&e, tile, rows).is_err());
    }
    assert!(blocked::plan(&entry(129), 32, 1).is_err());
}
#[test]
fn candidate_spec_keeps_existing_tiles_unchanged() {
    let e = entry(5120);
    let old = e.clone().with_prefill_tiles(32).unwrap();
    let before = old.entry_specs();
    let new = blocked::cubin_spec("blocked", &[e]).unwrap();
    assert_eq!(new.entries.len(), 4);
    assert_eq!(format!("{:?}", old.entry_specs()), format!("{before:?}"));
    assert!(new.entries.iter().all(|s| s.allowed_local_bytes == 0));
}
