//! Paired launch captures (`export_launches`): every launch of a scenario
//! with its rows, plan, grid and resolved arguments, for comparing two
//! trees. The scenarios cover the collectives' forms the journals do not:
//! the F32 norm-weight all-reduces (one-shot, phased, decode and wide, at
//! TP4 with 100 served sequences), the decode buckets of 256
//! served items, where graphs above the wide entry's cap take the phased
//! pair, and the padded HC mix.
use super::*;

/// A host of `shape` serving `served` sequences with the MXFP8-dense
/// layout's MXFP8 and NVFP4 package, whose rank views store its F32 norm
/// weights.
fn f32_host(shape: Shape, served: u32) -> Rc<FakeHost> {
    let host = Rc::new(FakeHost::at(shape));
    host.served.set(Some(served));
    *host.precision.borrow_mut() = Some(super::precision::document_for(shape, true, true, false));
    *host.f32_norms.borrow_mut() = super::norms::mxfp8_dense_norms(shape);
    host
}

/// A step of `items`, one KV page per 64 tokens numbered by its slot.
fn slots_step(items: &[Item<'_>]) -> BatchRequestV1 {
    let (batch, appends): (Vec<_>, Vec<_>) = items
        .iter()
        .map(|&(request, slot, previous, inputs)| {
            let count = inputs.len() as u64;
            (
                BatchItemV1::new(
                    request,
                    previous > 0 && count == 1,
                    count as u32,
                    true,
                    inputs.to_vec(),
                ),
                BatchAppendV1::new(previous, count, previous + count, vec![slot], None),
            )
        })
        .unzip();
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(items.iter().map(|item| item.1).collect())
}

/// Prefills `items` one-token-pair prompts, then decodes each of `live`.
fn prefill_and_decode(
    host: &Rc<FakeHost>,
    instance: &mut BundleInstanceV1,
    items: u32,
    lives: &[u32],
) {
    let prompts = (0..items)
        .map(|slot| [slot % 50 + 7, 9])
        .collect::<Vec<_>>();
    let prefill = (0..u64::from(items))
        .map(|slot| (1000 + slot, slot, 0, &prompts[slot as usize][..]))
        .collect::<Vec<_>>();
    run(host, instance, "prefill", slots_step(&prefill));
    instance.commit().unwrap();
    let mut lengths = vec![2; items as usize];
    for &live in lives {
        let decode = (0..u64::from(live))
            .map(|slot| (1000 + slot, slot, lengths[slot as usize], &[11][..]))
            .collect::<Vec<_>>();
        run(host, instance, "decode", slots_step(&decode));
        instance.commit().unwrap();
        lengths[..live as usize]
            .iter_mut()
            .for_each(|length| *length += 1);
    }
}

/// F32 norm-weight collectives of the MXFP8-dense TP4 package at 100 served
/// sequences: a short prompt (the one-shot all-reduce), a 40-row one (the
/// phased pair), a one-row decode (the LL decode entry) and a short
/// prefill; then, on a second instance, 40 live decode items (the wide
/// entry).
#[test]
fn f32_norm_collectives_export_their_launches() {
    crate::pipeline::take_recorded();
    let host = f32_host(Shape::TP4, 100);
    let mut instance = create(&host);
    let long = (0..40).map(|token| token % 50 + 7).collect::<Vec<u32>>();
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 2, 0, &PROMPT), (11, 3, 0, &long)]),
    );
    instance.commit().unwrap();
    run(&host, &mut instance, "decode", step(&[(10, 2, 3, &[7])]));
    instance.commit().unwrap();
    run(&host, &mut instance, "short", step(&[(12, 4, 0, &PROMPT)]));
    instance.commit().unwrap();
    instance.destroy().unwrap();
    export_launches("f32-tp4-c100-steps", &host);

    crate::pipeline::take_recorded();
    let host = f32_host(Shape::TP4, 100);
    let mut instance = create(&host);
    prefill_and_decode(&host, &mut instance, 40, &[40]);
    instance.destroy().unwrap();
    export_launches("f32-tp4-c100-live40", &host);
}

/// 256 served items at TP4: decode buckets of 256 items take the phased
/// pair above the wide entry's cap, 64 items the wide entry.
#[test]
fn wide_and_phased_decode_buckets_export_their_launches() {
    crate::pipeline::take_recorded();
    let host = Rc::new(FakeHost::default());
    host.served.set(Some(256));
    let mut instance = create(&host);
    prefill_and_decode(&host, &mut instance, 256, &[256, 33]);
    instance.destroy().unwrap();
    export_launches("tp4-c256-live256-33", &host);
}

/// The padded HC mix: precomputed decode buckets above 8 items, at each
/// rank count.
#[test]
fn padded_hc_mix_exports_its_launches() {
    for shape in Shape::ALL {
        crate::pipeline::take_recorded();
        let host = Rc::new(FakeHost::at(shape));
        let mut instance = crate::instance::with_hc_padded_mix(true, || {
            crate::instance::with_hc_precompute(true, || create(&host))
        });
        prefill_and_decode(&host, &mut instance, 17, &[8, 16, 17]);
        instance.destroy().unwrap();
        export_launches(&format!("padded-mix-tp{}", shape.ranks()), &host);
    }
}
