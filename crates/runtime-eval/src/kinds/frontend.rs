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
        }
        Ok(vec![
            "chat-renders.json".into(),
            "tokenizer-golden.json".into(),
            "coverage".into(),
        ])
    }
}
