//! Native cubin lookup over the shared topology contract.
use model_mimo26_contract::backend::{Kernels, Selected};
pub use model_mimo26_contract::topology::*;

pub fn all() -> impl Iterator<Item = Tp> {
    model_mimo26_contract::geometry::GEOMETRIES
        .iter()
        .flat_map(|g| {
            Tp::SUPPORTED.map(|ranks| Tp::of_geometry(g, ranks.ranks()).expect("listed geometry"))
        })
}

/// The topology an embedded cubin serves, or `None` when it holds kernels
/// every topology runs. `build.rs` records it as the cubin's rank count,
/// zero for the shared cubins.
pub fn cubin_topology(name: &str) -> Option<Tp> {
    let (_, geometry, ranks) = crate::GEOMETRY_CUBINS
        .iter()
        .find(|(cubin, _, _)| *cubin == name)?;
    if *ranks == 0 {
        return None;
    }
    let geometry = model_mimo26_contract::geometry::GEOMETRIES
        .iter()
        .find(|g| g.id == *geometry)?;
    Tp::of_geometry(geometry, *ranks).ok()
}

pub use crate::encoder_availability::{cubin_serves, encoders_available};

/// Retags a rank-invariant bundle entry using complete kernel tag tokens.
pub fn tagged(symbol: &str, tp: Tp) -> String {
    let base = model_mimo26_contract::geometry::V26_FLASH.kernel_tag;
    symbol.replacen(
        &format!("_{base}_"),
        &format!("_{}_", tp.geometry.kernel_tag),
        1,
    )
}

/// The bundle-owned kernel `name` (`ring_backup`, `dflash_drafts`, ...) as
/// the selected backend names its entry in the topology's cubins.
pub fn bundle_symbol(tp: Tp, name: &str) -> String {
    Selected::bundle_entry(tp, name)
}

/// The launch-bound threads `build.rs` declared for an embedded entry.
pub fn entry_threads(symbol: &str) -> Option<u32> {
    crate::CUBINS.iter().find_map(|cubin| {
        let index = cubin.symbols.iter().position(|s| *s == symbol)?;
        cubin.max_threads.get(index).copied()
    })
}

/// Threads of one ring backup or restore block: a thread per 16-byte vector
/// of a K and V entry, at least 96. The ring kernels return early on any
/// other block size, so this is the entry's declared bound, which
/// `build.rs` sets with `CHOP_MIMO_RING_THREADS`.
pub fn ring_threads(tp: Tp) -> u32 {
    entry_threads(&bundle_symbol(tp, "ring_backup")).expect("MiMo ring entry")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        GA_PAGE_BYTES_BF16, GA_PAGE_BYTES_FP8, RANKS, RING_SLOT_BYTES, RING_STATE_BYTES_PER_RANK,
        VOCAB_PER_RANK,
    };

    #[test]
    fn tp4_is_the_contract() {
        let tp = Tp::new(RANKS).unwrap();
        assert_eq!(topology_id(tp), crate::contract::TOPOLOGY_ID);
        assert_eq!(ga_page_bytes(tp, false), GA_PAGE_BYTES_BF16);
        assert_eq!(ga_page_bytes(tp, true), GA_PAGE_BYTES_FP8);
        assert_eq!(ring_slot_bytes(tp), RING_SLOT_BYTES);
        assert_eq!(ring_state_bytes_per_rank(tp), RING_STATE_BYTES_PER_RANK);
        assert_eq!(vocab_per_rank(tp), VOCAB_PER_RANK);
    }

    #[test]
    fn other_topologies_scale_the_per_rank_heads() {
        for tp in Tp::SUPPORTED {
            assert_eq!(from_topology_id(topology_id(tp)), Some(tp));
        }
        assert_eq!(from_topology_id("mimo-v26-flash-tp16"), None);
        assert_eq!(ga_page_bytes(Tp::TP2, false), 737_280);
        assert_eq!(ring_slot_bytes(Tp::TP2), 2 * RING_SLOT_BYTES);
        assert_eq!(vocab_per_rank(Tp::TP2), 76_288);
    }

    #[test]
    fn kernel_names_and_cubins_follow_the_topology() {
        assert_eq!(
            bundle_symbol(Tp::TP4, "ring_backup"),
            "chop_mimo_ring_backup_v1"
        );
        assert_eq!(
            bundle_symbol(Tp::TP2, "ring_backup"),
            "chop_mimo_tp2_ring_backup_v1"
        );
        assert_eq!(ring_threads(Tp::TP4), 96);
        assert_eq!(ring_threads(Tp::TP2), 160);
        let mut shared = 0;
        for cubin in crate::CUBINS {
            let own = cubin_topology(cubin.name);
            match own {
                None => shared += 1,
                Some(tp) => assert!(cubin.name.contains(kernel_tag(tp)), "{}", cubin.name),
            }
            // Every supported topology uses the shared comm entries.
            for symbol in cubin.symbols {
                assert!(!symbol.contains("_x2_v"), "{symbol} in {}", cubin.name);
                assert_eq!(
                    own.is_some_and(|tp| tp.ranks() != 4),
                    symbol.contains("_tp2"),
                    "{symbol} in {}",
                    cubin.name
                );
            }
        }
        assert!(shared > 0);
        // Every symbol an instance selects is in a cubin it loads.
        for tp in all() {
            let loaded = crate::CUBINS
                .iter()
                .filter(|cubin| cubin_serves(cubin.name, tp))
                .flat_map(|cubin| cubin.symbols.iter())
                .collect::<std::collections::BTreeSet<_>>();
            for stage in crate::program::stages_for(tp)
                .into_iter()
                .filter(|s| !s.projection() && !s.prepared())
            {
                for phase in [
                    crate::selection::Phase::Decode(1),
                    crate::selection::Phase::Verify(8),
                    crate::selection::Phase::Prefill,
                ] {
                    let symbol = match crate::sm120::selection::select(
                        stage,
                        phase,
                        kernel_core::contract::DType::Bf16,
                        tp,
                        crate::switches::Switches::configured(),
                    )
                    .unwrap()
                    {
                        crate::sm120::selection::Selected::Row(entry) => entry.symbol,
                        crate::sm120::selection::Selected::Attention(entry) => {
                            entry.instantiation().unwrap().symbol
                        }
                        _ => continue,
                    };
                    assert!(loaded.contains(&symbol.as_str()), "{symbol} at {tp:?}");
                }
            }
        }
    }

    /// The checkpoint's largest rank images: text, the DFlash and MTP
    /// drafters (TP4's measured, other counts' scaled by the rank count)
    /// and the encoders.
    const TEXT: [(Tp, u64); 2] = [(Tp::TP2, 85_552_652_288), (Tp::TP4, 43_450_937_344)];
    fn drafter_image(tp: Tp, drafter: Option<Drafter>) -> u64 {
        let tp4 = match drafter {
            Some(Drafter::Dflash) => 755_142_656,
            Some(Drafter::Mtp) => 297_455_616,
            None => 0,
        };
        tp4 * 4 / u64::from(tp.ranks())
    }
    use model_mimo26_contract::{encoders::Encoders, weight_plan::Drafter};

    /// The package maxima with the checkpoint's images: the largest graph
    /// bucket whose KV holds an 8K average context. The encoders take TP2
    /// from 64 to 32, and DFlash's per-slot cache halves TP2's.
    #[test]
    fn maximum_sequences_follow_the_kv_the_images_leave() {
        for (tp, text) in TEXT {
            for (drafter, capacities) in [
                (
                    None,
                    [
                        [
                            (64, 612_649),
                            (32, 495_989),
                            (32, 490_091),
                            (32, 490_091),
                            (32, 512_748),
                            (32, 394_122),
                            (32, 388_715),
                            (32, 388_715),
                        ],
                        [
                            (256, 7_635_592),
                            (256, 7_271_855),
                            (256, 7_260_058),
                            (256, 7_260_058),
                            (256, 7_305_374),
                            (256, 7_068_121),
                            (256, 7_057_308),
                            (256, 7_057_308),
                        ],
                    ],
                ),
                (
                    Some(Drafter::Mtp),
                    [
                        [
                            (32, 626_137),
                            (32, 444_269),
                            (32, 438_370),
                            (32, 438_370),
                            (32, 461_028),
                            (32, 342_402),
                            (32, 336_995),
                            (32, 336_995),
                        ],
                        [
                            (32, 8_044_262),
                            (32, 7_680_525),
                            (32, 7_668_728),
                            (32, 7_668_728),
                            (32, 7_714_044),
                            (32, 7_476_791),
                            (32, 7_465_978),
                            (32, 7_465_978),
                        ],
                    ],
                ),
                (
                    Some(Drafter::Dflash),
                    [
                        [
                            (32, 432_727),
                            (16, 335_891),
                            (16, 329_993),
                            (16, 329_993),
                            (32, 267_617),
                            (16, 234_024),
                            (16, 228_617),
                            (16, 228_617),
                        ],
                        [
                            (256, 6_647_283),
                            (256, 6_283_546),
                            (256, 6_271_749),
                            (256, 6_271_749),
                            (256, 6_317_064),
                            (256, 6_079_812),
                            (256, 6_068_999),
                            (256, 6_068_999),
                        ],
                    ],
                ),
            ] {
                let at = Tp::SUPPORTED.iter().position(|&at| at == tp).unwrap();
                for (with_media, (expected, tokens)) in
                    Encoders::SETS.into_iter().zip(capacities[at])
                {
                    let image = text + drafter_image(tp, drafter) + with_media.image_bytes_flash();
                    let admitted = maximum_sequences(tp, drafter, with_media, image);
                    assert_eq!(
                        admitted,
                        expected,
                        "TP{} {drafter:?} {with_media:?}",
                        tp.ranks()
                    );
                    let kv = |sequences| kv_tokens(tp, image, sequences, drafter, with_media);
                    assert_eq!(
                        kv(admitted),
                        tokens,
                        "TP{} {drafter:?} {with_media:?}",
                        tp.ranks()
                    );
                    assert!(kv(admitted) >= u64::from(admitted) * 8192);
                    if admitted < 256 && (drafter != Some(Drafter::Mtp) || admitted < 32) {
                        assert!(kv(2 * admitted) < u64::from(2 * admitted) * 8192);
                    }
                }
            }
        }
    }

    /// The manifest minimum follows the images at every rank count and, at
    /// each topology's package maximum with the checkpoint's images, fits a
    /// rank's device: one TP2 or TP4 rank on a
    /// 96 GB device's 94 GiB, the media encoders included.
    #[test]
    fn minimum_rank_bytes_follow_the_images_and_fit_the_device() {
        for (tp, text) in TEXT {
            for drafter in [None, Some(Drafter::Mtp), Some(Drafter::Dflash)] {
                for media in Encoders::SETS {
                    let image = text + drafter_image(tp, drafter) + media.image_bytes_flash();
                    let sequences = maximum_sequences(tp, drafter, media, image);
                    let floor = minimum_rank_bytes(tp, image, sequences, drafter, media);
                    assert_eq!(
                        minimum_rank_bytes(tp, image + (1 << 30), sequences, drafter, media),
                        floor + (1 << 30)
                    );
                    assert!(
                        floor <= rank_budget(tp),
                        "TP{} {drafter:?} media {media:?}",
                        tp.ranks()
                    );
                }
            }
        }
    }

    /// The ring-slots module an instance skips at 32 slots is its
    /// topology's own: among the cubins it loads, the one the backend names
    /// is the only ring-slots module.
    #[test]
    fn the_ring_slots_cubin_is_the_topologys_own() {
        use model_mimo26_contract::backend::{Kernels, Selected};
        for tp in all() {
            let loaded: Vec<_> = crate::CUBINS
                .iter()
                .map(|cubin| cubin.name)
                .filter(|name| cubin_serves(name, tp) && name.ends_with("_ring_slots"))
                .collect();
            assert_eq!(loaded, [Selected::bundle_cubin(tp, "ring_slots")], "{tp:?}");
        }
    }
}
