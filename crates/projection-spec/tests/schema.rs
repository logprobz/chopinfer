use projection_spec::{Dim, DimExpr, ProjectionKind, SplitClass};

#[test]
fn dimensions_round_trip() {
    for text in [
        "128",
        "\"hidden\"",
        r#"{"op":"product","factors":[2,"heads"]}"#,
    ] {
        let dim: Dim = serde_json::from_str(text).unwrap();
        assert_eq!(
            serde_json::from_str::<Dim>(&serde_json::to_string(&dim).unwrap()).unwrap(),
            dim
        );
    }
    assert!(matches!(
        serde_json::from_str::<Dim>("128").unwrap(),
        Dim::Literal(128)
    ));
    assert!(
        matches!(serde_json::from_str::<Dim>(r#"{"op":"sum","terms":[1,2]}"#).unwrap(), Dim::Expr(expr) if matches!(*expr, DimExpr::Sum { .. }))
    );
}

#[test]
fn tagged_empty_variants_refuse_unknown_fields() {
    assert!(serde_json::from_str::<ProjectionKind>(r#"{"kind":"dense","unknown":1}"#).is_err());
    assert!(serde_json::from_str::<SplitClass>(r#"{"class":"replicate","tile":1}"#).is_err());
    assert!(
        serde_json::from_str::<Dim>(r#"{"op":"product","factors":[1,2],"unknown":1}"#).is_err()
    );
}
