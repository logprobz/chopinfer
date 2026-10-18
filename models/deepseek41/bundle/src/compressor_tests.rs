use super::*;
use bundle_runtime::fake_host::dump::{Memory, Origin};
use bundle_runtime::snapshot::{HeldBuffers, Pages, Table};
use bundle_runtime::state::Holder;
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::GEOMETRIES;
use runtime_bundle_abi::BundleDeviceBufferHandleV1;
use std::rc::Rc;

fn span(id: u32, offset: u64, length: u64) -> Span {
    Span::new(
        BundleDeviceBufferHandleV1::from_parts(id, 1),
        offset,
        length,
    )
}

#[test]
fn every_phase_writes_index_keys_before_rotating_compressed_latents() {
    use crate::program::{Phase, Stage, stages};
    for geometry in GEOMETRIES {
        for phase in [
            Phase::Prefill { rows: 71 },
            Phase::Decode(4),
            Phase::Verify { rows: 12, items: 2 },
        ] {
            let program = stages(geometry, phase);
            for &source in geometry.shape.csa.kv_sources {
                let at = |stage| program.iter().position(|&s| s == stage).unwrap();
                assert!(at(Stage::Compress(source)) < at(Stage::Project(source, "indexer.k")));
                assert!(at(Stage::Project(source, "indexer.k")) < at(Stage::IndexKeyWrite(source)));
                assert!(at(Stage::IndexKeyWrite(source)) < at(Stage::CompressedKvWrite(source)));
                assert!(at(Stage::CompressedKvWrite(source)) < at(Stage::LatentAttention(source)));
                assert_eq!(
                    program.contains(&Stage::Project(source, "compressor.gate")),
                    geometry.compress_ratio(source) == 2
                );
            }
        }
    }
}

#[test]
fn cache_sources_resolve_on_incomplete_steps_and_new_pool_bindings() {
    for g in GEOMETRIES {
        for storage in [KvStorage::Native, KvStorage::Bf16] {
            let state = state::request_state(g, storage, false);
            let pool = span(1, 256, 8 * state.page_bytes());
            for layer in 0..g.layers() {
                let result = source_binding(g, layer, storage, pool, 8, 71).unwrap();
                if g.layer_role(layer) == LayerRole::Window {
                    assert!(result.is_none());
                    continue;
                }
                let current = result.unwrap();
                let source = g
                    .shape
                    .csa
                    .kv_sources
                    .iter()
                    .rev()
                    .copied()
                    .find(|&l| l <= layer)
                    .unwrap();
                assert_eq!(current.layer, source);
                assert_eq!(current.completed_positions, 71 / u64::from(current.ratio));
                assert_eq!(
                    u64::from(current.latent.positions_per_page * current.ratio),
                    state.page_tokens
                );
                let before = source_binding(g, layer, storage, pool, 8, 70)
                    .unwrap()
                    .unwrap();
                if current.ratio == 2 {
                    // Token 70 opens a group and emits no latent. Both
                    // existing caches still resolve for every reuse layer.
                    assert_eq!(current, before);
                }
                let changed_pool = span(2, 1024, pool.length);
                let rebound = source_binding(g, layer, storage, changed_pool, 8, 71)
                    .unwrap()
                    .unwrap();
                assert_eq!(rebound.latent.values.buffer, changed_pool.buffer);
                assert_eq!(rebound.index.values.buffer, changed_pool.buffer);
                for plane in [current.latent, current.index] {
                    assert_eq!(plane.page_stride, state.page_bytes());
                    for view in [Some(plane.values), plane.scales].into_iter().flatten() {
                        assert!(view.offset >= pool.offset);
                        assert!(view.offset + view.length <= pool.offset + pool.length);
                    }
                }
            }
        }
    }
}

#[test]
fn tail_ids_keep_verify_history_disjoint_and_seed_off_grid_prefill() {
    for g in GEOMETRIES {
        let extent = Extent {
            served: 8,
            bucket: 8,
            verify_rows: 6,
        };
        let state = state::request_state(g, KvStorage::Native, false);
        let tail = state
            .buffers
            .iter()
            .find(|b| b.name == "compressor.tail")
            .unwrap();
        assert_eq!(tail.layers, state::sources_of(g, 2).count() as u64);
        let raw = span(3, 256, tail.bytes(extent));
        for &layer in g.shape.csa.kv_sources {
            let binding = tail_binding(g, layer, extent, raw).unwrap();
            if g.compress_ratio(layer) == 1 {
                assert!(binding.is_none());
                continue;
            }
            let binding = binding.unwrap();
            for first in 0..32 {
                let ids = binding
                    .ids(3, first, 6, RecurrentPhase::Verify { max_rows: 6 })
                    .unwrap();
                assert_eq!(ids.checkpoints.len(), 6);
                assert!(!ids.checkpoints.contains(&ids.source));
                let unique: std::collections::BTreeSet<_> =
                    ids.checkpoints.iter().copied().collect();
                assert_eq!(unique.len(), 6);
                for (row, id) in ids.checkpoints.iter().enumerate() {
                    assert_eq!(*id as u64, 3 * TAIL_ROWS + (first + row as u64) % TAIL_ROWS);
                }
            }
            let prefill = binding.ids(3, 69, 4, RecurrentPhase::Prefill).unwrap();
            let (from, to) = prefill
                .seed_copy
                .expect("incoming and final tails use different cells");
            assert_eq!(
                from,
                binding
                    .entry((3 * TAIL_ROWS + 68 % TAIL_ROWS) as i32)
                    .unwrap()
            );
            assert_eq!(
                to,
                binding
                    .entry((3 * TAIL_ROWS + 72 % TAIL_ROWS) as i32)
                    .unwrap()
            );
            assert_eq!(to.length, 2 * u64::from(g.shape.attention.head_dim) * 4);
            assert!(
                binding
                    .ids(3, 68, 5, RecurrentPhase::Prefill)
                    .unwrap()
                    .seed_copy
                    .is_none()
            );
            assert!(
                binding
                    .ids(3, 69, 3, RecurrentPhase::Prefill)
                    .unwrap()
                    .seed_copy
                    .is_none()
            );
            assert!(binding.ids(3, 69, 2, RecurrentPhase::Decode).is_err());
            assert!(
                binding
                    .ids(3, 69, 8, RecurrentPhase::Verify { max_rows: 8 })
                    .is_err()
            );
        }
    }
}

fn upload(memory: &mut Memory, to: Span, bytes: Vec<u8>) {
    assert_eq!(to.length as usize, bytes.len());
    memory.write(to, vec![(to.length, Origin::Uploaded(Rc::from(bytes), 0))]);
}

/// Seeded snapshot evidence: kernels do not run on this fake memory. The
/// exact F32 tails traverse the same state declaration and page-copy table
/// that instance capture/restore uses, including wrapped history cells.
#[test]
fn seeded_odd_prefix_tails_round_trip_across_pages_slots_and_ranks() {
    let extent = Extent {
        served: 8,
        bucket: 8,
        verify_rows: 6,
    };
    let mut checked = 0;
    for g in GEOMETRIES {
        for storage in [KvStorage::Native, KvStorage::Bf16] {
            let state = state::request_state(g, storage, false);
            let tail_index = state
                .buffers
                .iter()
                .position(|b| b.name == "compressor.tail")
                .unwrap();
            let capture = Table::new(&state, extent, Holder::Slot(2)).unwrap();
            let restore = Table::new(&state, extent, Holder::Slot(5)).unwrap();
            let ids: Vec<_> = (0..capture.pages()).rev().map(|n| 7 + 3 * n).collect();
            for &ranks in SUPPORTED_RANKS {
                for rank in 0..ranks {
                    for prefix in [1, 7, 9, 69, 71, 73] {
                        let held = HeldBuffers {
                            buffers: state
                                .buffers
                                .iter()
                                .enumerate()
                                .map(|(i, b)| Some(span(10 + i as u32, 256, b.bytes(extent))))
                                .collect(),
                            spares: vec![None; state.buffers.len()],
                        };
                        let tail_buffer = held.buffers[tail_index].unwrap();
                        let pool = span(99, 512, (ids[0] + 1) * state.page_bytes());
                        let pages = Pages {
                            pool,
                            ids: &ids,
                            page_bytes: state.page_bytes(),
                        };
                        let mut memory = Memory::default();
                        upload(
                            &mut memory,
                            tail_buffer,
                            vec![0xa5; tail_buffer.length as usize],
                        );
                        let mut expected = Vec::new();
                        for source in state::sources_of(g, 2) {
                            let binding = tail_binding(g, source, extent, tail_buffer)
                                .unwrap()
                                .unwrap();
                            let source_id = (2 * TAIL_ROWS + (prefix - 1) % TAIL_ROWS) as i32;
                            let target_id = (5 * TAIL_ROWS + (prefix - 1) % TAIL_ROWS) as i32;
                            let seed: Vec<u8> = (0..binding.entry_bytes / 4)
                                .flat_map(|d| {
                                    let bits = 0x3f80_0001u32
                                        + ((u64::from(rank) * 1009
                                            + u64::from(source) * 83
                                            + prefix * 17
                                            + d)
                                            % 0xffff)
                                            as u32;
                                    bits.to_le_bytes()
                                })
                                .collect();
                            assert!(seed.chunks_exact(4).any(|word| {
                                let bits = u32::from_le_bytes(word.try_into().unwrap());
                                kernel_core::codec::bf16_to_f32(kernel_core::codec::f32_to_bf16(
                                    f32::from_bits(bits),
                                ))
                                .to_bits()
                                    != bits
                            }));
                            upload(&mut memory, binding.entry(source_id).unwrap(), seed.clone());
                            expected.push((binding.entry(target_id).unwrap(), seed));
                        }
                        for (from, to) in capture
                            .copies(0..state.buffers.len(), &held, &pages, true)
                            .unwrap()
                        {
                            memory.copy(from, to);
                        }
                        for (from, to) in restore
                            .copies(0..state.buffers.len(), &held, &pages, false)
                            .unwrap()
                        {
                            memory.copy(from, to);
                        }
                        for (target, bytes) in expected {
                            assert_eq!(memory.bytes(target), bytes);
                            // A neighboring slot must retain its poison.
                            let neighbor = Span::new(
                                target.buffer,
                                target.offset + TAIL_ROWS * target.length,
                                target.length,
                            );
                            assert_eq!(
                                memory.bytes(neighbor),
                                vec![0xa5; neighbor.length as usize]
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0);
}
