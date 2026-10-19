//! `frontend_goldens.v1`: a served capture's chat-render and tokenizer token
//! IDs compared exactly with the declared goldens
//! (docs/check-kinds.md#frontend_goldens-frontend_goldensv1). An offline
//! kind outside the portable eval subset: `eval run` does not capture it,
//! `eval rescore` judges its reports.

use serde_json::Value;

use crate::declaration::Check;
use crate::json::python_eq;
use crate::kind::{Kind, KindError, KindInfo, natural};
use crate::kinds::{applied_constants, get, require};

pub struct FrontendGoldens;

pub const INFO: KindInfo = KindInfo {
    name: "frontend_goldens",
    version: "frontend_goldens.v1",
    modules: &["frontend_goldens"],
    reference: None,
    metrics: &[
        "chat-renders.json",
        "tokenizer-golden.json",
        "needles.json",
        "api-samples.json",
        "coverage",
        "status",
        "reference",
    ],
};

/// The golden files a report compares.
const FILES: [&str; 2] = ["chat-renders.json", "tokenizer-golden.json"];

impl Kind for FrontendGoldens {
    fn info(&self) -> &'static KindInfo {
        &INFO
    }

    /// The report compared the goldens the reference names, each with rows,
    /// its reference row count and consistent equality evidence
    /// (`kinds.FrontendGoldens.verify`).
    fn verify(
        &self,
        value: &Value,
        reference: Option<&Value>,
        check: &Check,
    ) -> Result<Vec<String>, KindError> {
        applied_constants(check, &[])?;
        let reference = reference.filter(|reference| {
            get(value, "reference").is_some_and(|stated| python_eq(stated, reference))
        });
        require(reference.is_some(), || {
            "the receipt compared other frontend goldens".into()
        })?;
        let reference = reference.expect("checked");
        for name in FILES {
            let report = get(value, name).filter(|report| report.is_object());
            let positive = |key: &str| {
                report
                    .and_then(|report| report.get(key))
                    .is_some_and(|count| natural(count) && count.as_u64() > Some(0))
            };
            require(positive("rows"), || format!("{name}: no compared rows"))?;
            require(positive("reference_rows"), || {
                format!("{name}: no reference rows")
            })?;
            let report = report.expect("checked");
            require(
                python_eq(&report["reference_rows"], &reference["files"][name]["rows"]),
                || format!("{name}: reference row count differs"),
            )?;
            let agrees = report.get("agrees");
            require(
                !agrees.is_some_and(crate::json::truthy)
                    || python_eq(&report["rows"], &report["reference_rows"]),
                || format!("{name}: unequal row counts"),
            )?;
            let mismatches = report.get("mismatches").and_then(Value::as_array);
            require(
                agrees.is_some_and(Value::is_boolean) && mismatches.is_some(),
                || format!("{name}: missing equality evidence"),
            )?;
            let rows = report["rows"].as_u64().expect("checked");
            let mismatches = mismatches.expect("checked");
            require(
                mismatches
                    .iter()
                    .all(|index| natural(index) && index.as_u64().is_some_and(|at| at < rows)),
                || format!("{name}: mismatch outside the compared rows"),
            )?;
            require(
                report["agrees"] != Value::Bool(true) || mismatches.is_empty(),
                || format!("{name}: agrees with mismatched IDs"),
            )?;
            if name == "tokenizer-golden.json" {
                verify_refusals(report, &check.declared)?;
            }
        }
        Ok(vec![
            "chat-renders.json".into(),
            "tokenizer-golden.json".into(),
            "coverage".into(),
        ])
    }
}

fn verify_refusals(report: &Value, check: &Value) -> Result<(), KindError> {
    let empty = Vec::new();
    let declared = check["tokenizer_refusals"].as_array().unwrap_or(&empty);
    let observed = report.get("expected_refusals");
    require(observed.is_none_or(Value::is_array), || {
        "tokenizer-golden.json: invalid refusal evidence".into()
    })?;
    let observed = observed.and_then(Value::as_array).unwrap_or(&empty);
    let agrees = report["agrees"] == true;
    let mut digests = std::collections::BTreeSet::new();
    let mut rows = std::collections::BTreeSet::new();
    for refusal in observed {
        let declaration = declared
            .iter()
            .find(|item| item["sha256"] == refusal["sha256"]);
        require(declaration.is_some(), || {
            "tokenizer-golden.json: undeclared refusal".into()
        })?;
        let declaration = declaration.expect("checked");
        require(
            ["sha256", "special_token", "reason"]
                .iter()
                .all(|key| refusal[*key] == declaration[*key])
                && refusal["sha256"]
                    .as_str()
                    .is_some_and(|sha| digests.insert(sha))
                && refusal["row"]
                    .as_u64()
                    .is_some_and(|row| Some(row) < report["rows"].as_u64() && rows.insert(row)),
            || "tokenizer-golden.json: refusal identity differs".into(),
        )?;
        let message = format!(
            "prompt contains a special token the model does not admit in prompts: {}",
            declaration["special_token"].as_str().unwrap_or_default()
        );
        require(
            refusal["agrees"].is_boolean()
                && (!agrees || refusal["agrees"] == true)
                && (refusal["agrees"] != true
                    || (refusal["status"] == 400 && refusal["message"] == message)),
            || "tokenizer-golden.json: expected prompt-special-token refusal differs".into(),
        )?;
    }
    if !declared.is_empty() && agrees {
        require(
            observed.len() == declared.len()
                && report["matched_rows"]
                    .as_u64()
                    .and_then(|count| count.checked_add(observed.len() as u64))
                    == report["rows"].as_u64(),
            || "tokenizer-golden.json: missing declared refusal evidence".into(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn declared_refusals_require_complete_specific_evidence() {
        let declaration = json!({"sha256": "a".repeat(64), "special_token": "<special>", "reason": "Unserved task."});
        let check = json!({"tokenizer_refusals": [declaration.clone()]});
        let mut refusal = declaration;
        refusal["row"] = json!(1);
        refusal["status"] = json!(400);
        refusal["agrees"] = json!(true);
        refusal["message"] =
            json!("prompt contains a special token the model does not admit in prompts: <special>");
        let report = json!({"rows": 2, "matched_rows": 1, "agrees": true, "expected_refusals": [refusal.clone()]});
        verify_refusals(&report, &check).unwrap();
        for (pointer, value) in [
            ("/expected_refusals", json!([])),
            ("/expected_refusals", json!([refusal.clone(), refusal])),
            ("/expected_refusals/0/status", json!(200)),
            ("/expected_refusals/0/status", json!(500)),
            (
                "/expected_refusals/0/message",
                json!("prompt tokenization failed"),
            ),
            ("/expected_refusals/0/sha256", json!("b".repeat(64))),
            ("/expected_refusals/0/reason", json!("Other reason.")),
            ("/expected_refusals/0/row", json!(2)),
            ("/expected_refusals/0/agrees", json!(false)),
            ("/matched_rows", json!(2)),
        ] {
            let mut changed = report.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(verify_refusals(&changed, &check).is_err(), "{changed}");
        }
        assert!(verify_refusals(&report, &json!({})).is_err());
        verify_refusals(&json!({"agrees": true, "rows": 2}), &json!({})).unwrap();
    }
}
