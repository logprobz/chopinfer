//! The capacity plan against what create holds when the host sizes the KV
//! pool: a package serving images holds exactly the plan's encoder term
//! more on each rank, so its planned KV is the pool the host leaves.

use kernel_core::latent_attention::LatentKvFormat;
use model_glm5next_contract::capacity::{DEVICE, Planner, Serving, encoder_bytes};
use model_glm5next_contract::geometry::FLASH;

use super::*;

/// Bytes create holds over every rank when it acquires the serving KV, with
/// or without the encoder.
fn held_at_kv(encoder: bool) -> u64 {
    let host = FakeHost::new((3, 32));
    let mut instance = if encoder {
        create_with_encoder(&host)
    } else {
        create(&host)
    };
    let held = host.held_at_kv.get();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    held
}

/// The host's sizing (`acquire_serving_kv`): the KV pool takes the KV
/// memory fraction of what a rank has free after the reserve and the state
/// pool, in whole pages.
fn host_pages(free: u64, page_bytes: u64) -> u64 {
    let fraction = u128::from(DEVICE.fraction_millionths);
    (u128::from(free - DEVICE.reserve) * fraction / 1_000_000) as u64 / page_bytes
}

/// A record naming the encoder holds the encoder's image and buffers on
/// every rank before the KV pool, `encoder_bytes` each. On a device where
/// the text package's create leaves its plan's KV pool, the host then sizes
/// the pool of a `text,image` package to its plan's pages: FP8 KV, three
/// draft tokens, 32 slots.
#[test]
fn a_text_image_package_plans_the_kv_pool_create_leaves() {
    let text_held = held_at_kv(false);
    let image_held = held_at_kv(true);
    assert_eq!(image_held - text_held, 4 * encoder_bytes());
    let planner = Planner::new(&FLASH, 4).unwrap();
    let serving = Serving {
        ranks: 4,
        context_parallel: 1,
        slots: 32,
        draft_tokens: 3,
        kv: LatentKvFormat::Fp8E4M3Block128,
        vision: false,
    };
    let text = planner.plan(serving, DEVICE).unwrap();
    let image = planner
        .plan(
            Serving {
                vision: true,
                ..serving
            },
            DEVICE,
        )
        .unwrap();
    // Free after the text package's create and its state pool, as planned.
    let text_free = DEVICE.bytes - text.weights - text.workspace.total() - text.state;
    assert_eq!(host_pages(text_free, text.page_bytes), text.pages);
    // The same device after the create naming the encoder.
    let image_free = text_free - (image_held - text_held) / 4;
    assert_eq!(host_pages(image_free, image.page_bytes), image.pages);
    assert_eq!(image.pages, 3_371);
}

/// At a KV memory fraction of 1.0 the host's pool takes every byte a rank
/// has free past the 1 GiB reserve and the recurrent state it reserves
/// (`runtime_host::recurrent::StateSlotPoolV1`: `rows + 1` copies of the
/// declared layouts per slot, none when the package declares its state
/// preallocated). Whatever create allocates after the pool beyond that
/// reservation comes out of the reserve. Create's allocations after the
/// pool, three drafts at 32 slots with host drafts and with the MTP
/// drafter, leave the whole reserve.
#[test]
fn create_leaves_the_host_reserve_at_a_kv_fraction_of_one() {
    use model_glm5next_contract::contract::CAPABILITIES;
    let preallocated =
        CAPABILITIES.contains(&runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1);
    let copy: u64 = Shape::TP4
        .recurrent_state_layouts()
        .iter()
        .map(|(_, bytes)| bytes)
        .sum();
    let reserved = if preallocated {
        0
    } else {
        u64::from(DECODE_BATCH) * (3 + 2) * copy
    };
    let mib = |bytes: u64| bytes as f64 / f64::from(1 << 20);
    for format in &Shape::TP4.kv_formats()[..2] {
        let host = FakeHost::with_format((3, 32), *format);
        let mut instance = create_over(&host, *format);
        let held: u64 = host.extents.borrow().values().sum();
        let kv = 4 * KV_PAGES * format.page_bytes;
        let after_kv = (held - host.held_at_kv.get() - kv) / 4;
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        let left = (DEVICE.reserve + reserved).saturating_sub(after_kv);
        assert!(
            left >= DEVICE.reserve,
            "{}: create allocates {after_kv} bytes per rank after the KV pool against {reserved} \
             reserved: {:.2} MiB of the {:.2} MiB reserve left",
            String::from_utf8_lossy(format.layout_id),
            mib(left),
            mib(DEVICE.reserve)
        );
    }
}
