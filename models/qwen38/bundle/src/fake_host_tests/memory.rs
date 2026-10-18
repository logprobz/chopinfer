//! Planned per-rank memory of the mixed packages against the real device:
//! what create places on each rank (its device rank image, every buffer it
//! allocates and its ring state) at the package's served sequences with the
//! MTP drafter must fit the device floor the package declares, that floor
//! must fit an RTX PRO 6000's memory, and the pinned host memory a rank
//! takes for its n-gram rows must not exceed the legacy package's.
use super::mixed_mtp::component;
use super::speculation::drafter_host_at;
use super::*;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::execution::PrecisionDocument;
use model_qwen38_contract::package::{Encoders, maximum_sequences, minimum_device_memory};

/// The device memory an RTX PRO 6000 Blackwell reports: 97,887 MiB.
const REAL_DEVICE_BYTES: u64 = 97_887 << 20;

/// Per rank: device bytes create places (rank image on the device, buffers,
/// ring state) and pinned host bytes (the n-gram rows in host memory).
fn planned(
    shape: Shape,
    document: Option<PrecisionDocument>,
    norms: BTreeSet<String>,
) -> Vec<(u64, u64)> {
    let host = drafter_host_at(3, shape);
    host.served
        .set(Some(maximum_sequences(shape, true, Encoders::VISION)));
    *host.precision.borrow_mut() = document;
    *host.f32_norms.borrow_mut() = norms;
    let mut instance = create(&host);
    let allocations: Vec<(u32, u64)> = host
        .allocation_pools
        .borrow()
        .iter()
        .copied()
        .zip(host.allocation_bytes.borrow().iter().copied())
        .collect();
    let planned = (0..shape.ranks())
        .map(|rank| {
            let (index, image) = host.rank_index(rank);
            let table = crate::views::parse(&index, rank).unwrap().views
                [&crate::views::ngram_view_for(shape)]
                .bytes;
            let pinned = if host.ngram_host.get() { table } else { 0 };
            let device_image = image - pinned.next_multiple_of(4096);
            let buffers: u64 = allocations
                .iter()
                // Each rank allocates from its own pool, slot rank + 1.
                .filter(|(pool, _)| *pool == rank + 1)
                .map(|(_, bytes)| bytes)
                .sum();
            (device_image + buffers + host.layout().2, pinned)
        })
        .collect();
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
    planned
}

#[test]
fn mixed_packages_fit_their_declared_floor_and_the_real_device() {
    let fp8 = WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32);
    for shape in [Shape::TP2, Shape::TP4] {
        let declared = minimum_device_memory(shape, true, Encoders::VISION);
        assert!(
            declared <= REAL_DEVICE_BYTES,
            "TP{} declares {declared} device bytes per rank",
            shape.ranks()
        );
        let legacy = planned(shape, None, BTreeSet::new());
        for (layout, document, norms) in [
            (
                "nvfp4-routed-fp8-mtp",
                component(shape, false, Some(fp8)),
                BTreeSet::new(),
            ),
            (
                "mxfp8-dense-nvfp4",
                component(shape, true, Some(WeightFormat::Nvfp4)),
                super::norms::mxfp8_dense_norms(shape),
            ),
        ] {
            let label = format!("{layout} TP{}", shape.ranks());
            for (rank, ((device, pinned), (_, legacy_pinned))) in
                planned(shape, Some(document), norms)
                    .into_iter()
                    .zip(&legacy)
                    .enumerate()
            {
                // The declared floor also holds the encoder's media bytes.
                let floor = device + Encoders::VISION.media_bytes_for(shape);
                assert!(
                    floor <= declared,
                    "{label} rank {rank} places {floor} device bytes, over its declared {declared}"
                );
                assert!(
                    pinned <= *legacy_pinned,
                    "{label} rank {rank} pins {pinned} bytes"
                );
            }
        }
    }
}
