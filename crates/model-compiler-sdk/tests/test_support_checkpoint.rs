//! The checkpoint fixture parser of the test support sources
//! (test-support/checkpoint.rs), which models' projection tests include.

#[path = "../test-support/checkpoint.rs"]
mod checkpoint;

use checkpoint::fixture;

#[test]
fn a_fixture_expands_range_and_list_axes() {
    let metadata = fixture(
        r#"{"families": [
            {"tensor": "layers.{i}.experts.{e}.w", "dtype": "F8_E4M3", "shape": [4, 8],
             "axes": {"{i}": {"range": [0, 2]}, "{e}": {"indices": [3, 5]}}},
            {"tensor": "head.w", "dtype": "BF16", "shape": [2, 8], "axes": {}}],
         "quantization_config": {"quant_method": "fp8"}}"#,
    );
    let names: Vec<_> = metadata.tensors.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "head.w",
            "layers.0.experts.3.w",
            "layers.0.experts.5.w",
            "layers.1.experts.3.w",
            "layers.1.experts.5.w",
        ]
    );
    let expert = &metadata.tensors["layers.1.experts.5.w"];
    assert_eq!(
        (expert.dtype.as_str(), expert.shape.as_slice()),
        ("F8_E4M3", &[4, 8][..])
    );
    assert!(expert.file.is_none());
    assert_eq!(metadata.quantization_config["quant_method"], "fp8");
    assert!(metadata.validated_owners.is_empty());
}

#[test]
#[should_panic]
fn a_fixture_refuses_a_name_two_families_give() {
    fixture(
        r#"{"families": [
            {"tensor": "w.{i}", "dtype": "BF16", "shape": [1], "axes": {"{i}": {"range": [0, 2]}}},
            {"tensor": "w.1", "dtype": "BF16", "shape": [1], "axes": {}}],
         "quantization_config": null}"#,
    );
}

#[test]
#[should_panic]
fn a_fixture_refuses_an_unknown_field() {
    fixture(r#"{"families": [], "quantization_config": null, "extra": 1}"#);
}
