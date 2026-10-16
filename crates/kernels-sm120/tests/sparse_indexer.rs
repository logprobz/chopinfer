//! CPU launch and admission checks for the indexer's head and rotary shapes.
use kernels_sm120::attention::sparse::{Geometry, Kind, indexer_entries};
use kernels_sm120::{Profile, ProfileNames};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "indexer-test",
    tag: "indexer_test",
});

fn geometry(heads: u32, rotary: u32) -> Geometry {
    Geometry {
        index_heads: heads,
        index_dim: 128,
        rotary_dim: rotary,
        theta: 1e7,
        mrope: if rotary == 32 {
            [6, 5, 5]
        } else {
            [11, 11, 10]
        },
        eps: 1e-6,
        block: 4,
        ring: 16,
        top_blocks: 512,
        max_context: 262144,
        q_heads: 6,
        kv_heads: 1,
        head_dim: 256,
        page_size: 64,
        softmax_scale: 0.0625,
    }
}

#[test]
fn indexer_shapes_fit_their_launch_and_score_tiles() {
    for heads in [4, 8, 16] {
        for rotary in [32, 64] {
            let entries = indexer_entries(PROFILE, geometry(heads, rotary), 2);
            for entry in entries {
                let spec = entry.contract().unwrap();
                let plan = entry.plan(entry.capacity(), 262144).unwrap();
                assert_eq!(plan.block[0], entry.threads());
                match entry.kind {
                    Kind::Query => {
                        assert_eq!(plan.block, [32 * heads, 1, 1]);
                        assert_eq!(spec.key.knobs.warps, heads as u8);
                        assert_eq!(spec.key.knobs.tile_n, 128);
                    }
                    Kind::Scores { rows, by_run } => {
                        assert_eq!(spec.key.knobs.tile_m as u32, (rows * heads).max(16));
                        assert!(rows * heads <= 256);
                        assert!(entry.dynamic_shared_bytes() <= 98_304);
                        if entry.capacity() == 2048 {
                            assert_eq!(rows, 256 / heads);
                            assert_eq!(plan.grid[0], 2048 / rows);
                        } else if by_run {
                            assert_eq!(rows, 8);
                        } else {
                            assert_eq!(rows, 1);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[test]
fn indexer_refuses_unsupported_heads_and_oversized_score_tiles() {
    for heads in [0, 1, 2, 3, 5, 32] {
        assert!(
            indexer_entries(PROFILE, geometry(heads, 64), 2)[0]
                .contract()
                .is_err()
        );
    }
    for heads in [8, 16] {
        let mut score = indexer_entries(PROFILE, geometry(heads, 32), 2)
            .into_iter()
            .find(|e| matches!(e.kind, Kind::Scores { .. }))
            .unwrap();
        score.kind = Kind::Scores {
            rows: 512 / heads,
            by_run: false,
        };
        assert!(score.contract().is_err());
    }
}

#[test]
fn indexer_refuses_unsupported_rotary_widths_and_sections() {
    for heads in [4, 8, 16] {
        for rotary in [16, 48, 128] {
            let mut g = geometry(heads, rotary);
            // Keep the section sum valid to isolate the rotary-width guard.
            g.mrope = [rotary / 2, 0, 0];
            for entry in indexer_entries(PROFILE, g, 2) {
                assert!(entry.contract().is_err(), "{} rotary {rotary}", entry.name);
            }
        }
        for (rotary, sections) in [(32, [11, 11, 10]), (64, [6, 5, 5])] {
            let g = Geometry {
                mrope: sections,
                ..geometry(heads, rotary)
            };
            for entry in indexer_entries(PROFILE, g, 2) {
                assert!(
                    entry.contract().is_err(),
                    "{} sections {sections:?}",
                    entry.name
                );
            }
        }
    }
}

/// The key write keeps its chunk claim at whole blocks and claims its spare
/// anchor ring past a ring of rows: needs whose anchors lie a ring or more
/// after their launch's start (a 64-row prefill granule) are provided,
/// needs of anchors at any span, with or without a chunk or rows claim,
/// are refused.
#[test]
fn key_write_anchors_are_claimed_past_the_ring() {
    use kernel_core::properties::{Anchors, Chunk, Invariance, Rows};
    let g = geometry(4, 64);
    let entry = indexer_entries(PROFILE, g, 2)
        .into_iter()
        .find(|entry| entry.kind == Kind::KeyWrite)
        .unwrap();
    assert_eq!(entry.invariance.chunk, Chunk::Granule(g.block));
    assert_eq!(
        entry.invariance.anchors,
        Anchors::SpareAfter { rows: g.ring }
    );
    let need = |chunk, rows, anchors| Invariance {
        chunk,
        rows,
        anchors,
        ..Invariance::NONE
    };
    let after = |rows| Anchors::SpareAfter { rows };
    assert!(
        entry
            .invariance
            .provides(need(Chunk::Granule(64), Rows::None, after(64)))
    );
    assert!(
        entry
            .invariance
            .provides(need(Chunk::None, Rows::None, after(g.ring)))
    );
    assert!(
        !entry
            .invariance
            .provides(need(Chunk::None, Rows::None, after(g.block)))
    );
    for chunk in [Chunk::None, Chunk::Granule(4), Chunk::Granule(64)] {
        for rows in [Rows::None, Rows::OneRow { max: 2 }] {
            assert!(!entry.invariance.provides(need(chunk, rows, Anchors::Spare)));
        }
    }
}
