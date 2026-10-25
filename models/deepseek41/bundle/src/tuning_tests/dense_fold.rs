use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{Family, table},
};

#[test]
fn starts_with_baseline_only() {
    assert!(table(&FLASH, Family::DenseFold).unwrap().rows().is_empty());
}
