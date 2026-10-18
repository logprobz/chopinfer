//! Each validation rule refuses a one-edit change to a committed
//! declaration with no bespoke variants, with a message that names the
//! problem. Rules that need MiMo's provider are tested in its contract
//! crate.

use encoder_spec::{Environment, parse};
use serde_json::{Value, json};

const QWEN: &str = include_str!("../../../models/qwen38/contract/encoders/qwen38-vision.json");
const DSV41: &str = include_str!("fixtures/dsv41-vision.json");

fn edited(base: &str, edit: impl FnOnce(&mut Value)) -> String {
    let mut value: Value = serde_json::from_str(base).unwrap();
    edit(&mut value);
    value.to_string()
}

fn owner(base: &str) -> &'static str {
    if base == QWEN { "qwen38" } else { "dsv41" }
}

#[track_caller]
fn refuses(base: &str, edit: impl FnOnce(&mut Value), needle: &str) {
    let text = edited(base, edit);
    let error = parse(&text, &Environment::new(owner(base))).expect_err("accepted the edit");
    assert!(error.0.contains(needle), "expected `{needle}` in: {error}");
}

#[track_caller]
fn accepts(base: &str, edit: impl FnOnce(&mut Value)) {
    let text = edited(base, edit);
    parse(&text, &Environment::new(owner(base))).unwrap();
}

#[test]
fn unknown_component_ids() {
    refuses(
        QWEN,
        |d| d["stages"][0]["op"] = "linear.pairs.v9".into(),
        "unknown variant `linear.pairs.v9`",
    );
    refuses(
        QWEN,
        |d| d["stages"][0]["op"] = "kernel.pairs.v1".into(),
        "unknown kind",
    );
    refuses(
        QWEN,
        |d| d["stages"][0]["op"] = "linear.pairs".into(),
        "has no version",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["norm"]["op"] = "linear.pairs.v1".into(),
        "unknown variant `linear.pairs.v1`, expected one of `norm.layer.v1`",
    );
    refuses(
        QWEN,
        |d| d["stages"][3]["op"] = "attention.varlen.pairs.v1".into(),
        "a component of kind attention is not a stage",
    );
    refuses(
        QWEN,
        |d| d["stages"][3]["op"] = "quantize.rvq.v1".into(),
        "kind quantize has no shared variant",
    );
    refuses(
        QWEN,
        |d| d["output"]["op"] = "layout.tiles.v1".into(),
        "unknown variant `layout.tiles.v1`",
    );
    refuses(
        QWEN,
        |d| d["precision"] = "f32.v1".into(),
        "unknown variant `f32.v1`",
    );
    refuses(
        QWEN,
        |d| d["input"]["recipes"][0]["tag"] = "image.tiles.v1".into(),
        "unknown variant `image.tiles.v1`",
    );
}

#[test]
fn unknown_parameters() {
    refuses(
        QWEN,
        |d| d["stages"][0]["bais"] = true.into(),
        "unknown field `bais`",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["attention"]["window"] = 8.into(),
        "unknown field `window`",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["mlp"]["pad"] = 8.into(),
        "unknown field `pad`",
    );
    refuses(
        QWEN,
        |d| d["input"]["rows"] = 8.into(),
        "unknown field `rows`",
    );
    refuses(
        QWEN,
        |d| d["input"]["recipes"][0]["factor"] = 32.into(),
        "unknown field `factor`",
    );
    refuses(QWEN, |d| d["layout"] = 8.into(), "unknown field `layout`");
    refuses(
        QWEN,
        |d| d["stages"][0]["weights"]["weight"]["transform"] = json!([]),
        "unknown field `transform`",
    );
    refuses(
        QWEN,
        |d| d["stages"][0]["weights"]["weight"]["transforms"] = json!(["transpose"]),
        "unknown variant `transpose`",
    );
    refuses(
        QWEN,
        |d| {
            d["stages"][0].as_object_mut().unwrap().remove("bias");
        },
        "missing field `bias`",
    );
    refuses(
        QWEN,
        |d| d["sequences"] = json!({"kind": "item", "rows": 4}),
        "unknown field `rows`",
    );
}

#[test]
fn shape_mismatches() {
    refuses(
        QWEN,
        |d| d["stages"][2]["width"] = 1024.into(),
        "width 1024 does not match its input, f32 rows of width 1152",
    );
    refuses(
        QWEN,
        |d| d["output"]["hidden"] = 4096.into(),
        "reads bf16 rows of width 4096 but the last stage writes bf16 rows of width 2560",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["rotary"]["pairs_per_axis"] = 19.into(),
        "rotates 76 dimensions of a 72-dimension head",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["attention"]["kv_heads"] = 3.into(),
        "do not share 3 key/value heads evenly",
    );
    refuses(
        QWEN,
        |d| d["stages"][4] = json!({"op": "merge.space-to-depth.v1", "factor": 2, "pad": "zero"}),
        "space-to-depth gathers from a row-major",
    );
    refuses(
        QWEN,
        |d| d["stages"][4]["rows"] = 3.into(),
        "rows come in multiples of 4, not of 3",
    );
}

/// Review P2: role shapes that the declaration alone contradicts.
#[test]
fn weight_shapes() {
    refuses(
        QWEN,
        |d| d["stages"][2]["weights"]["qkv.weight"] = json!({"zeros": 1}),
        "1 zeros cannot give the role's shape [3456, 1152]",
    );
    refuses(
        QWEN,
        |d| d["stages"][0]["weights"]["bias"] = json!({"zeros": 1024}),
        "1024 zeros cannot give the role's shape [1152]",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["weights"]["up.weight"]["transforms"] = json!([{"pad_rows": 1}]),
        "PadRows(1) does not give the role's shape [4352, 1152]",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["weights"]["down.weight"]["transforms"] = json!([{"pad_cols": 4304}]),
        "PadCols(4304) does not give the role's shape [1152, 4352]",
    );
    refuses(
        QWEN,
        |d| {
            d["stages"][6]["weights"]["bias"] =
                json!({"concat_rows": ["merger.linear_fc2.bias", {"zeros": 2560}]})
        },
        "2560 zeros leave no room",
    );
    refuses(
        QWEN,
        |d| d["stages"][6]["weights"]["weight"] = json!({"concat_rows": ["a", {"zeros": 8}]}),
        "zeros stack only into a non-empty vector",
    );
}

#[test]
fn weight_mappings() {
    refuses(
        QWEN,
        |d| {
            d["stages"][2]["weights"]
                .as_object_mut()
                .unwrap()
                .remove("qkv.bias");
        },
        "role `qkv.bias` is not mapped",
    );
    refuses(
        QWEN,
        |d| d["stages"][4]["weights"] = json!({"weight": "x"}),
        "unknown role `weight`",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["weights"]["qkv.weight"] = "blocks.0.attn.qkv.weight".into(),
        "has no `{i}`",
    );
    refuses(
        QWEN,
        |d| d["stages"][0]["weights"]["bias"] = "blocks.{i}.bias".into(),
        "has `{i}` outside a transformer",
    );
    refuses(
        QWEN,
        |d| {
            d["stages"][2]["weights"]["qkv.bias"] =
                json!({"tensor": "blocks.{i}.attn.qkv.bias", "blocks": "1-5"})
        },
        "only a role some blocks lack names a block set",
    );
}

/// Review P2: recipe geometry against the declared encoder.
#[test]
fn recipe_geometry_and_capacity() {
    // Confirm review R5: the capacity is the derived run (9,189 rows: one
    // row of 1,021 cells of 3 x 3 patches) rounded up to 1,024 rows, 9,216.
    for capacity in [1, 9188, 9189, 9215, 9344] {
        refuses(
            DSV41,
            |d| d["input"]["capacity_rows"] = capacity.into(),
            &format!("capacity_rows {capacity} is not the derived 9216: the 9189-row run"),
        );
    }
    accepts(DSV41, |d| d["input"]["capacity_rows"] = 9216.into());
    refuses(
        DSV41,
        |d| d["input"]["chunking"] = "whole-sequences.v1".into(),
        "one item is one run; derive the capacity",
    );
    refuses(
        QWEN,
        |d| d["input"]["capacity_rows"] = 65535.into(),
        "capacity_rows 65535 is not the derived 65536",
    );
    refuses(
        QWEN,
        |d| d["input"]["recipes"][0]["patch"] = 14.into(),
        "emits 1176 columns, the input declares 1536",
    );
    refuses(
        DSV41,
        |d| d["input"]["recipes"][0]["maximum_tokens"] = 3.into(),
        "a token budget below 4",
    );
    refuses(
        QWEN,
        |d| d["input"]["recipes"][0]["maximum_pixels"] = 512.into(),
        "512 pixels hold no 2 x 2 merge block",
    );
    refuses(
        QWEN,
        |d| d["input"]["patchify"] = "row-major.v1".into(),
        "not the declared patchify",
    );
    refuses(
        QWEN,
        |d| d["sequences"] = json!({"kind": "groups", "rows": 4}),
        "groups are a stage's own sequences",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["sequences"] = json!({"kind": "groups", "rows": 8}),
        "groups of 8 rows need joined items whose rows come in whole groups",
    );
}

/// Review P2: derived dimensions use checked arithmetic.
#[test]
fn checked_arithmetic() {
    refuses(
        QWEN,
        |d| d["stages"][2]["attention"]["q_heads"] = 4294967280_u64.into(),
        "the q/k/v width does not fit 32 bits",
    );
    refuses(
        QWEN,
        |d| d["stages"][4]["rows"] = 4294967292_u64.into(),
        "rows come in multiples of 4",
    );
    refuses(
        DSV41,
        |d| d["input"]["recipes"][0]["downsample"] = 4294967295_u64.into(),
        "overflows",
    );
}

/// Review P2: scalars are checked in the F32 the kernels and tables use.
#[test]
fn f32_effective_scalars() {
    refuses(
        QWEN,
        |d| d["stages"][2]["attention"]["scale"] = 1e300.into(),
        "scale 1e300 is inf in F32",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["norm"]["eps"] = 1e-300.into(),
        "eps 1e-300 is 0.0 in F32",
    );
    refuses(
        QWEN,
        |d| d["stages"][3]["eps"] = 1e-46.into(),
        "is 0.0 in F32",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["rotary"]["theta"] = 1e39.into(),
        "theta 1e39 is inf in F32",
    );
    accepts(QWEN, |d| {
        d["stages"][2]["attention"]["scale"] = 0.125.into()
    });
}

/// Review P2: the precision gate follows the rows' element, whatever the
/// last op is.
#[test]
fn precision_gate_follows_elements() {
    // A second ending: the last GEMM stays F32, a norm follows, and a cast
    // rounds once into the language model.
    accepts(QWEN, |d| {
        let stages = d["stages"].as_array_mut().unwrap();
        stages[6].as_object_mut().unwrap().remove("output");
        stages.push(json!({"op": "norm.layer-f32.v1", "eps": 1e-6,
                           "weights": {"weight": "post.weight", "bias": "post.bias"}}));
        stages.push(json!({"op": "cast.bf16.v1"}));
    });
    refuses(
        QWEN,
        |d| {
            d["stages"]
                .as_array_mut()
                .unwrap()
                .insert(4, json!({"op": "cast.bf16.v1"}));
        },
        "reads rows already rounded to BF16",
    );
    refuses(
        QWEN,
        |d| d["stages"][5]["output"] = "bf16".into(),
        "reads rows already rounded to BF16",
    );
    refuses(
        QWEN,
        |d| {
            d["stages"][6].as_object_mut().unwrap().remove("output");
        },
        "the last stage writes f32 rows of width 2560",
    );
    refuses(
        QWEN,
        |d| d["stages"][6]["output"] = "f32".into(),
        "`output: f32` is the default",
    );
}

#[test]
fn precision_policies() {
    refuses(
        QWEN,
        |d| d["precision"] = "bf16-legacy.v1".into(),
        "precision bf16-legacy.v1 is retired; use f32-pairs.v1",
    );
    refuses(
        QWEN,
        |d| d["stages"][3]["op"] = "norm.layer.v1".into(),
        "`norm.layer.v1` is not defined under precision policy f32-pairs.v1",
    );
    refuses(
        QWEN,
        |d| d["stages"][0]["weights"]["weight"]["transforms"] = json!(["flatten", "narrow_bf16"]),
        "allowed only under bf16-legacy.v1",
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["mlp"]["round_linear"] = true.into(),
        "round_linear is allowed only",
    );
    refuses(
        QWEN,
        |d| d["input"]["payload"] = "bf16".into(),
        "f32-pairs.v1 takes an F32 payload",
    );
}

#[test]
fn owners() {
    let error = parse(QWEN, &Environment::new("mimo26")).unwrap_err();
    assert!(
        error
            .0
            .contains("declares model qwen38 but is loaded for model mimo26"),
        "{error}"
    );
    refuses(
        QWEN,
        |d| d["stages"][2]["rotary"] = json!({"op": "mimo26:rotary.axial-2d-f32-powf.v1", "pairs_per_axis": 18, "theta": 10000, "style": "neox"}),
        "`mimo26:rotary.axial-2d-f32-powf.v1` belongs to model mimo26; model qwen38 cannot use it",
    );
}

#[test]
fn parallel_policies() {
    refuses(
        QWEN,
        |d| d["parallel"] = json!(["record-parallel", "replicated"]),
        "`replicated` comes first",
    );
    refuses(
        QWEN,
        |d| d["parallel"] = json!(["replicated", "replicated"]),
        "is repeated",
    );
    refuses(
        QWEN,
        |d| {
            d["stages"][2]["reorder"] =
                json!([{"blocks": "1-3", "reorder": {"op": "reorder.merge-group-columns.v1"}}])
        },
        "sequence-split needs",
    );
}

/// Review P3: each meaning has one spelling where the digest would
/// otherwise split it, and pins are compared and hashed as written.
#[test]
fn one_spelling_per_meaning() {
    refuses(
        QWEN,
        |d| d["stages"][2]["attention"]["scale"] = (72_f64).powf(-0.5).into(),
        "is the default for head_dim 72; remove it",
    );
    let evens = (0..27)
        .filter(|b| b % 2 == 0)
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",");
    refuses(
        QWEN,
        |d| {
            d["stages"][2]["overrides"] =
                json!([{"blocks": evens, "attention": {"mask": "causal"}}])
        },
        "is `even`; write it so",
    );
    refuses(
        QWEN,
        |d| {
            d["stages"][2]["overrides"] =
                json!([{"blocks": "0-26", "attention": {"mask": "causal"}}])
        },
        "names every block of 27",
    );
    refuses(
        QWEN,
        |d| {
            d["stages"][2]["overrides"] = json!([{"blocks": "even", "attention": {"mask": "full"}}])
        },
        "changes nothing",
    );
    // A pin keeps its JSON number type: 27.0 is not the config's 27.
    let config: Value = serde_json::from_str(include_str!(
        "../../../models/qwen38/compiler/fixtures/config.json"
    ))
    .unwrap();
    let respelled = edited(QWEN, |d| {
        d["checkpoint"]["config"][0]["values"]["depth"] = json!(27.0)
    });
    let respelled = parse(&respelled, &Environment::new("qwen38")).unwrap();
    let error = respelled.check_config("config.json", &config).unwrap_err();
    assert!(
        error
            .0
            .contains("/vision_config/depth is 27, the declaration pins 27.0"),
        "{error}"
    );
    let original = parse(QWEN, &Environment::new("qwen38")).unwrap();
    assert_ne!(respelled.digest(), original.digest());
    // Typed numbers and key order do not enter the digest.
    let typed = edited(QWEN, |d| d["stages"][2]["rotary"]["theta"] = json!(10000.0));
    assert_eq!(
        parse(&typed, &Environment::new("qwen38")).unwrap().digest(),
        original.digest()
    );
}
