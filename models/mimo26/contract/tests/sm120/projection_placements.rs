use model_mimo26_contract::capabilities::sm120;

#[test]
fn every_locked_draft_gemm_is_adopted() {
    let lock: serde_json::Value =
        serde_json::from_str(include_str!("../../../bundle/kernels.lock")).unwrap();
    let records = sm120::adoptions().unwrap();
    let mut checked = 0;
    for cubin in lock.as_array().unwrap() {
        if !cubin["name"].as_str().unwrap().starts_with("gemm_") {
            continue;
        }
        for entry in cubin["entries"].as_array().unwrap() {
            let symbol = entry["symbol"].as_str().unwrap();
            if symbol.contains("_dflash_") {
                assert!(
                    records.iter().any(|a| a.symbol == symbol),
                    "missing draft projection {symbol}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn active_adoptions_match_current_hand_lists_and_locked_positions() {
    use kernels_sm120::precision::generated::placement;
    use std::collections::BTreeMap;

    let mut current = BTreeMap::new();
    for geometry in model_mimo26_contract::geometry::GEOMETRIES {
        for record in sm120::inventory(geometry).unwrap().adoptions {
            if let Some(previous) = current.insert(record.symbol.clone(), record.clone()) {
                assert_eq!(previous, record, "conflicting hand-list adoption");
            }
        }
    }
    let current = placement::capture(
        current.into_values(),
        include_str!("../../../bundle/kernels.lock"),
    )
    .unwrap();
    let current: BTreeMap<_, _> = current.into_iter().map(|a| (a.symbol.clone(), a)).collect();
    let active: BTreeMap<_, _> = sm120::ledger()
        .unwrap()
        .into_iter()
        .map(|a| (a.symbol.clone(), a))
        .collect();
    assert_eq!(
        active, current,
        "active adoptions must match current schedules and lock positions"
    );
}
