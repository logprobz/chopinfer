//! The official checkpoint's headers bind to the projection declaration
//! with no unknown and no missing tensor. A contract test of that revision
//! (`checkpoint-files.json`), not an admission list.
use kernel_core::contract::WeightFormat;
use model_deepseek41::checkpoint;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn the_official_headers_bind_every_tensor() {
    let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK_CHECKPOINT") else {
        eprintln!("CHOPINFER_DEEPSEEK_CHECKPOINT is unset: skipped");
        return;
    };
    let root = std::path::PathBuf::from(root);
    let config = checkpoint::document(&root, "config.json").unwrap();
    let mut metadata = checkpoint::metadata(&root).unwrap();
    assert_eq!(metadata.tensors.len(), 96_085);
    // The encoder declaration (contract/encoders) attests the vision and
    // aligner tensors; until it lands, their names stand in for it.
    let encoder: BTreeSet<_> = metadata
        .tensors
        .keys()
        .filter(|name| {
            name.starts_with("vision.")
                || name.starts_with("aligner.")
                || name.starts_with("image_")
        })
        .cloned()
        .collect();
    metadata
        .validated_owners
        .insert("encoder-spec".into(), encoder.clone());
    let bound = checkpoint::bind_metadata(&config, &metadata).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        checkpoint::bind(&root, encoder).unwrap(),
        bound,
        "reading the checkpoint binds as its metadata does"
    );

    // Every tensor is claimed once: a weight with its scale, a side tensor
    // or an encoder tensor.
    let mut claimed = BTreeSet::new();
    let mut formats = BTreeMap::<String, usize>::new();
    for operand in &bound.operands {
        let weight = &operand.imported.weight;
        assert!(claimed.insert(weight.codes.tensor.clone()));
        if let Some(scale) = &weight.local_scale {
            assert!(claimed.insert(scale.tensor.tensor.clone()));
        }
        *formats.entry(format!("{:?}", weight.format)).or_default() += 1;
    }
    for name in bound.side_tensors.iter().chain(&bound.external_tensors) {
        assert!(claimed.insert(name.clone()), "{name}");
    }
    assert_eq!(claimed, metadata.tensors.keys().cloned().collect());
    println!("bound weights by format: {formats:?}");
    // Routed experts (40 layers of 384, 3 DSpark stages of 128, three
    // matrices each) are MXFP4; the dense projections block-32 FP8; the
    // Engram tables MXFP8.
    assert_eq!(formats["Mxfp4"], (40 * 384 + 3 * 128) * 3);
    assert_eq!(formats["Mxfp8"], 2);
    let format_of = |tensor: &str| {
        bound
            .operands
            .iter()
            .find(|o| o.imported.weight.codes.tensor == tensor)
            .map(|o| o.imported.weight.format)
    };
    assert_eq!(
        format_of("layers.7.attn.wo_a.weight"),
        Some(WeightFormat::Fp8Block32x32)
    );
    assert_eq!(
        format_of("layers.20.attn.compressor.wkv.weight"),
        Some(WeightFormat::Bf16)
    );

    // Negative controls, each refused by name: a missing tensor, an unknown
    // one, a scale of another dtype and a weight of another shape.
    let refused = |change: &dyn Fn(&mut projection_spec::bind::CheckpointMetadata), name: &str| {
        let mut changed = metadata.clone();
        change(&mut changed);
        let error = checkpoint::bind_metadata(&config, &changed).unwrap_err();
        assert!(error.contains(name), "{name}: {error}");
    };
    refused(
        &|m| {
            m.tensors.remove("layers.3.ffn.experts.17.w2.scale");
        },
        "layers.3.ffn.experts.17.w2.scale",
    );
    refused(
        &|m| {
            let info = m.tensors["norm.weight"].clone();
            m.tensors
                .insert("layers.3.attn.unknown.weight".into(), info);
        },
        "layers.3.attn.unknown.weight",
    );
    refused(
        &|m| m.tensors.get_mut("layers.9.attn.wq_a.scale").unwrap().dtype = "F32".into(),
        "layers.9.attn.wq_a",
    );
    refused(
        &|m| m.tensors.get_mut("mtp.1.attn.wkv.weight").unwrap().shape = vec![512, 4096],
        "mtp.1.attn.wkv",
    );
}
