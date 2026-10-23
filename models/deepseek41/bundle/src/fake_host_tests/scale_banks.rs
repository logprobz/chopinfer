//! A rank image that stores the routed scale banks encoded: create admits
//! them, allocates the raw banks and loads the expansion, and every step
//! expands each layer's banks after its routes.
use super::*;
use model_deepseek41_contract::weight_plan::encoded_scale_layout;

#[test]
fn encoded_scale_banks_expand_in_every_step() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let kv = KV_LAYOUTS[0];
    let host = FakeHost::new(shape, kv);
    let layers = FLASH.layers() as usize;
    for layer in 0..FLASH.layers() {
        for (bank, width) in [("gate_up", 1), ("down", 2)] {
            host.layouts.borrow_mut().insert(
                format!("layers.{layer}.ffn.experts.{bank}.scale"),
                encoded_scale_layout(width),
            );
        }
    }
    let record = crate::config::tests::record(&shape, kv, 4, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
    let launches = |symbol: &str| {
        host.device
            .journal
            .borrow()
            .iter()
            .filter(|l| l.starts_with("launch ") && l.contains(&format!(" {symbol} ")))
            .count()
    };
    let mut views = Views::default();
    step(&mut instance, &prompt(7, 0, 0, 3, true), &mut views);
    // Each rank expands each layer's gate/up and down banks in one launch.
    let pair = "chop_moe_scale_expand_pair_v1";
    let expanded = launches(pair);
    assert_eq!(expanded, 4 * layers);
    for width in [1, 2, 4] {
        assert_eq!(launches(&format!("chop_moe_scale_expand_w{width}_v1")), 0);
    }
    step(&mut instance, &decode(&[(7, 0, 3, 1)]), &mut views);
    assert!(launches(pair) > expanded);
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// The scale expansion's cubin loads only for a rank image with an encoded
/// bank, so a raw image's create is unchanged.
#[test]
fn raw_scale_banks_load_no_expansion() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    let mut views = Views::default();
    step(&mut instance, &prompt(7, 0, 0, 3, true), &mut views);
    assert!(
        !host
            .device
            .journal
            .borrow()
            .iter()
            .any(|l| l.contains("scale_expand")),
        "a raw image loads or launches the expansion"
    );
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
}
