use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{Family, table},
};

#[test]
fn starts_with_baseline_only() {
    assert!(table(&FLASH, Family::Norm).unwrap().rows().is_empty());
}
