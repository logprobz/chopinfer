//! The contract's per-rank device memory model against what `create`
//! allocates on the fake host.
use super::*;
use model_mimo26_contract::weight_plan::Drafter;

/// Bytes one rank of an instance serving `served` sequences allocates in
/// `create` when it verifies the most rows a step holds (with bundle or host
/// drafts, which needs more than serving without speculation), plus its ring
/// state; `None` where `create` refuses the count.
fn created_bytes(tp: Tp, served: u32, drafter: Option<Drafter>) -> Option<u64> {
    let host = Rc::new(FakeHost::at(tp));
    host.served.set(served);
    host.speculation
        .set(Some((7, model_mimo26_contract::serving::VERIFY_TOKENS)));
    host.dflash.set(drafter == Some(Drafter::Dflash));
    let mut instance = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .unwrap()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_serving(tp, &KV_FORMATS[0], drafter.is_some(), served),
        )
        .ok()?;
    let bundle: u64 = host.allocations.borrow().iter().sum();
    instance.destroy().unwrap();
    Some(
        bundle / u64::from(tp.ranks())
            + crate::topology::ring_state_bytes(tp, served.next_power_of_two()),
    )
}

/// The contract's media bytes bound the encoders' buffer plan within
/// 64 MiB, so they follow the plan if it changes.
#[test]
fn media_bytes_bound_the_encoder_allocations() {
    for set in model_mimo26_contract::encoders::Encoders::SETS {
        if !set.any() {
            assert_eq!(set.media_bytes_flash(), 0);
            continue;
        }
        let buffers = crate::encoders::runtime::plan(set).unwrap().bytes();
        let bound = set.media_bytes_flash();
        assert!(buffers <= bound, "{set:?}: {buffers} > {bound}");
        assert!(bound - buffers < 64 << 20, "{set:?}: {buffers} < {bound}");
    }
}

/// The contract's working set bounds what `create` allocates, and the ring
/// state, at every graph bucket from 16 to 256 (32 with MTP), with no drafter
/// (host drafts), the MTP drafter and DFlash, also with the media encoders'
/// buffers, and is within 0.75 GiB of it at the most
/// each drafter serves; MTP refuses a 33rd sequence.
#[test]
fn working_bytes_bound_create_at_every_bucket() {
    let gib = |bytes: u64| bytes as f64 / f64::from(1 << 30);
    for geometry in model_mimo26_contract::geometry::GEOMETRIES {
        for ranks in [2, 4] {
            let tp = Tp::of_geometry(geometry, ranks).unwrap();
            let buffers = crate::encoders::runtime::plan_for(
                geometry,
                model_mimo26_contract::encoders::Encoders::ALL,
            )
            .unwrap()
            .bytes();
            for drafter in [None, Some(Drafter::Mtp), Some(Drafter::Dflash)] {
                let most = if drafter == Some(Drafter::Mtp) {
                    model_mimo26_contract::serving::DRAFT_ITEMS
                } else {
                    crate::contract::DECODE_BATCH
                };
                let mut sequences = 1;
                while sequences <= most {
                    let created = created_bytes(tp, sequences, drafter)
                        .unwrap_or_else(|| panic!("TP{} C{sequences} {drafter:?}", tp.ranks()));
                    let bound = crate::topology::working_bytes(tp, sequences, drafter);
                    eprintln!(
                        "TP{} C{sequences} {drafter:?}: creates {:.3} GiB, bound {:.3} GiB",
                        tp.ranks(),
                        gib(created),
                        gib(bound)
                    );
                    assert!(
                        created <= bound,
                        "TP{} C{sequences} {drafter:?}",
                        tp.ranks()
                    );
                    assert!(
                        created + buffers
                            <= bound
                                + model_mimo26_contract::encoders::Encoders::ALL
                                    .media_bytes_for(geometry),
                        "TP{} C{sequences} {drafter:?} media",
                        tp.ranks()
                    );
                    if sequences == most {
                        assert!(bound - created < 3 << 28, "TP{} {drafter:?}", tp.ranks());
                    }
                    sequences *= 2;
                }
                if drafter == Some(Drafter::Mtp) {
                    assert!(created_bytes(tp, most + 1, drafter).is_none());
                }
            }
        }
    }
}
