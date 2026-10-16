mod common;

use common::*;
use projection_spec::registry::Registry;
use projection_spec::{Dim, DimExpr, LayerIndices, ProjectionKind, SplitClass};

fn refuses(change: impl FnOnce(&mut projection_spec::Declaration), expected: &str) {
    let mut d = declaration();
    change(&mut d);
    let error = d.validate(&geometry(), &Registry::default()).unwrap_err();
    assert!(error.0.contains(expected), "expected {expected}: {error}");
}

#[test]
fn synthetic_declaration_covers_every_split_and_kind() {
    declaration()
        .validate(&geometry(), &Registry::default())
        .unwrap();
}

#[test]
fn reject_wrong_schema_namespace_empty_inventory_and_duplicate_ids() {
    refuses(|d| d.schema = "other".into(), "schema");
    refuses(|d| d.namespace = "Bad".into(), "namespace");
    refuses(|d| d.projections.clear(), "no projections");
    refuses(|d| d.projections[1].id = "q".into(), "duplicate id");
    refuses(|d| d.side_tensors[0].id = "q".into(), "duplicate id");
    refuses(|d| d.fusion_groups[0].id = "q".into(), "duplicate id");
    refuses(|d| d.projections[0].id = "".into(), "invalid id");
}

#[test]
fn reject_unordered_empty_or_malformed_stems() {
    refuses(|d| d.layer_stems[0].template = "layers".into(), "needs {i}");
    refuses(
        |d| d.layer_stems[0].template = "layers.{j}".into(),
        "unknown placeholder",
    );
    refuses(
        |d| {
            d.layer_stems[0].indices = LayerIndices::List {
                indices: vec![Dim::Literal(2), Dim::Literal(1)],
            }
        },
        "strictly increasing",
    );
    refuses(
        |d| d.layer_stems[0].indices = LayerIndices::List { indices: vec![] },
        "nonempty",
    );
    refuses(
        |d| {
            d.layer_stems[0].indices = LayerIndices::Range {
                start: Dim::Literal(2),
                end: Dim::Literal(2),
            }
        },
        "nonempty",
    );
    refuses(
        |d| {
            d.layer_stems[0].indices = LayerIndices::Range {
                start: Dim::Literal(0),
                end: Dim::Literal(u64::MAX),
            }
        },
        "bounded",
    );
}

#[test]
fn reject_bad_source_patterns_and_missing_stems() {
    refuses(
        |d| d.projections[0].source.as_mut().unwrap().stem = Some("missing".into()),
        "missing stem",
    );
    refuses(
        |d| d.projections[0].source.as_mut().unwrap().tensor = "q.weight".into(),
        "needs {stem}",
    );
    refuses(
        |d| d.projections[0].source.as_mut().unwrap().stem = None,
        "without a stem",
    );
    refuses(
        |d| d.projections[0].source.as_mut().unwrap().tensor = "{stem}.{expert}.weight".into(),
        "unknown placeholder",
    );
    refuses(
        |d| d.projections[0].source.as_mut().unwrap().tensor = "{stem}.{stem}.weight".into(),
        "repeats",
    );
    refuses(
        |d| d.projections[0].source.as_mut().unwrap().tensor = "".into(),
        "invalid tensor pattern",
    );
    refuses(
        |d| d.side_tensors[0].source.tensor = "{stem}.{operand}.norm".into(),
        "unknown placeholder",
    );
}

#[test]
fn reject_invalid_operands_parts_contracts_and_formats() {
    refuses(|d| d.projections[0].operands.clear(), "operand_count");
    refuses(
        |d| d.projections[6].operands = vec!["a".into(), "b".into()],
        "lookup table requires one",
    );
    refuses(
        |d| d.projections[0].parts[0].rows = Dim::Literal(3),
        "part rows total",
    );
    refuses(
        |d| d.projections[0].parts[1].id = "query".into(),
        "duplicate id",
    );
    refuses(
        |d| d.projections[0].parts[1].rows = Dim::Literal(u64::MAX),
        "part rows overflow",
    );
    refuses(
        |d| d.projections[0].numerical_contract.clear(),
        "invalid numerical contract",
    );
    let mut malformed: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    malformed["projections"][0]["format"] = serde_json::json!("unknown");
    assert!(
        projection_spec::parse(&malformed.to_string(), &geometry(), &Registry::default()).is_err()
    );
}

#[test]
fn reject_illegal_splits_and_head_geometry() {
    refuses(
        |d| {
            d.projections[0].split = SplitClass::ExpertOwned {
                tile: Dim::Literal(4),
            }
        },
        "illegal",
    );
    refuses(
        |d| {
            d.projections[4].split = SplitClass::Column {
                tile: Dim::Literal(4),
            }
        },
        "illegal",
    );
    refuses(
        |d| {
            d.projections[6].split = SplitClass::Row {
                tile: Dim::Literal(4),
            }
        },
        "illegal",
    );
    refuses(
        |d| {
            d.projections[0].split = SplitClass::VocabSharded {
                tile: Dim::Literal(4),
            }
        },
        "illegal",
    );
    refuses(
        |d| {
            d.projections[0].split = SplitClass::HeadGrouped {
                heads: Dim::Literal(2),
                head_dim: Dim::Literal(3),
                tile: Dim::Literal(4),
            }
        },
        "head count times head dimension",
    );
    refuses(
        |d| {
            d.projections[1].split = SplitClass::Column {
                tile: Dim::Literal(0),
            }
        },
        "positive",
    );
}

#[test]
fn reject_invalid_expert_top_k() {
    refuses(
        |d| {
            d.projections[4].kind = ProjectionKind::ExpertBank {
                experts: Dim::Literal(4),
                top_k: Dim::Literal(5),
            }
        },
        "top_k",
    );
    refuses(
        |d| {
            d.projections[4].kind = ProjectionKind::ExpertBank {
                experts: Dim::Literal(u64::MAX),
                top_k: Dim::Literal(1),
            }
        },
        "u32 expert count",
    );
    refuses(
        |d| {
            d.projections[4].kind = ProjectionKind::ExpertBank {
                experts: Dim::Literal(4),
                top_k: Dim::Literal(0),
            }
        },
        "positive",
    );
}

#[test]
fn reject_bad_epilogues() {
    refuses(
        |d| d.projections[3].epilogues.gelu = Some(kernel_core::contract::GeluMode::Erf),
        "cannot coexist",
    );
    refuses(
        |d| {
            d.projections[0].epilogues.swiglu =
                Some(kernel_core::contract::SwigluColumns::ConcatenatedHalves)
        },
        "two operands",
    );
    refuses(
        |d| d.projections[0].epilogues.weighted_moe_combine = true,
        "expert bank",
    );
    refuses(
        |d| d.projections[6].epilogues.bias = true,
        "lookup table does not admit",
    );
}

#[test]
fn reject_empty_duplicate_or_overflowing_uses() {
    refuses(|d| d.projections[0].uses.clear(), "no uses");
    refuses(
        |d| {
            let duplicate = d.projections[0].uses[0].clone();
            d.projections[0].uses.push(duplicate);
        },
        "duplicate projection use",
    );
    refuses(
        |d| d.projections[0].uses[0].capacity = Dim::Literal(u64::MAX),
        "exceeds u32",
    );
    refuses(
        |d| d.projections[0].uses[0].capacity = Dim::Literal(0),
        "positive",
    );
}

#[test]
fn reject_bad_fusion_members_and_input_contracts() {
    refuses(
        |d| d.fusion_groups[0].members = vec!["q".into()],
        "at least two",
    );
    refuses(|d| d.fusion_groups[0].input.clear(), "needs an input");
    refuses(
        |d| d.fusion_groups[0].members[1] = "missing".into(),
        "missing member",
    );
    refuses(
        |d| d.fusion_groups[0].members[1] = "q".into(),
        "repeats member",
    );
    refuses(
        |d| d.projections[1].shape[1] = Dim::Literal(8),
        "input contract",
    );
    refuses(
        |d| d.projections[1].source.as_mut().unwrap().stem = Some("drafter".into()),
        "input contract",
    );
    refuses(|d| d.projections[1].uses[1].graph = false, "input contract");
}

#[test]
fn reject_bad_side_roles_shapes_and_consumers() {
    refuses(
        |d| d.side_tensors[0].consumer = "missing".into(),
        "missing consumer",
    );
    refuses(
        |d| {
            d.side_tensors[0].norm = None;
            d.side_tensors[0].dtypes.clear();
        },
        "admitted dtypes",
    );
    refuses(
        |d| {
            let duplicate = d.side_tensors[0].dtypes[0];
            d.side_tensors[0].dtypes.push(duplicate);
        },
        "distinct",
    );
    refuses(
        |d| d.side_tensors[0].shape = vec![Dim::Literal(0)],
        "positive",
    );
    let mut arbitrary_rank = declaration();
    arbitrary_rank.side_tensors[0].norm = None;
    arbitrary_rank.side_tensors[0].shape = vec![Dim::Literal(1); 6];
    arbitrary_rank
        .validate(&geometry(), &Registry::default())
        .unwrap();
    refuses(
        |d| d.side_tensors[0].source.stem = Some("drafter".into()),
        "different layer stems",
    );
}

#[test]
fn dimensions_bind_checked_geometry_arithmetic() {
    let parse = |s| serde_json::from_str::<Dim>(s).unwrap().resolve(&geometry());
    assert_eq!(
        parse(r#"{"op":"product","factors":[2,"hidden"]}"#).unwrap(),
        32
    );
    assert_eq!(parse(r#"{"op":"sum","terms":["hidden",4]}"#).unwrap(), 20);
    assert_eq!(
        parse(r#"{"op":"quotient","numerator":"hidden","denominator":4}"#).unwrap(),
        4
    );
    for (text, why) in [
        ("0", "positive"),
        (r#""absent""#, "missing geometry key"),
        (r#""bad key""#, "invalid geometry key"),
        (
            r#"{"op":"sum","terms":[18446744073709551615,1]}"#,
            "overflows",
        ),
        (
            r#"{"op":"product","factors":[18446744073709551615,2]}"#,
            "overflows",
        ),
        (r#"{"op":"product","factors":[2]}"#, "at least two"),
        (r#"{"op":"sum","terms":[]}"#, "at least two"),
        (
            r#"{"op":"quotient","numerator":16,"denominator":3}"#,
            "not exact",
        ),
        (
            r#"{"op":"quotient","numerator":16,"denominator":0}"#,
            "positive",
        ),
        (
            r#"{"op":"quotient","numerator":1,"denominator":2}"#,
            "not exact",
        ),
    ] {
        let error = parse(text).unwrap_err();
        assert!(error.0.contains(why), "{text}: {error}");
    }
    let mut dim = Dim::Literal(1);
    for _ in 0..34 {
        dim = Dim::Expr(Box::new(DimExpr::Product {
            factors: vec![Dim::Literal(1), dim],
        }));
    }
    assert!(dim.resolve(&geometry()).unwrap_err().0.contains("depth"));
}

#[test]
fn chunks_can_have_different_head_counts() {
    let mut d = declaration();
    d.projections[0].shape[0] = Dim::Literal(12);
    d.projections[0].parts[0].rows = Dim::Literal(8);
    d.projections[0].parts[0].range.as_mut().unwrap().length = Dim::Literal(8);
    d.projections[0].parts[1].range.as_mut().unwrap().start = Dim::Literal(8);
    d.projections[0].parts[0].heads = Some(Dim::Literal(2));
    d.validate(&geometry(), &Registry::default()).unwrap();
    refuses(
        |d| d.projections[0].parts[0].heads = Some(Dim::Literal(3)),
        "head count times head dimension",
    );
    refuses(
        |d| d.projections[0].parts[0].heads = Some(Dim::Literal(0)),
        "positive",
    );
    refuses(
        |d| {
            d.projections[0].split = SplitClass::Column {
                tile: Dim::Literal(4),
            }
        },
        "part heads require",
    );
}

#[test]
fn fusion_refuses_different_rank_local_input_splits() {
    refuses(
        |d| {
            d.projections[1].split = SplitClass::Row {
                tile: Dim::Literal(4),
            }
        },
        "input split",
    );
}

#[test]
fn validate_resolves_packed_dimensions_and_fusion_use_sets() {
    refuses(
        |d| {
            d.projections[3].source.as_mut().unwrap().tensor = "{stem}.packed.weight".into();
            d.projections[3].shape = [Dim::Literal(1 << 41), Dim::Literal(1)];
        },
        "element count exceeds the supported bound",
    );
    refuses(
        |d| d.projections[7].shape[0] = Dim::Literal(u64::MAX),
        "element count exceeds the supported bound",
    );
    let mut d = declaration();
    d.projections[1].uses.reverse();
    d.projections[1].uses[3].capacity = Dim::Literal(128);
    d.validate(&geometry(), &Registry::default()).unwrap();
}
