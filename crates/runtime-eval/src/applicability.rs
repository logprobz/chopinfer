//! Validate unserved media evidence before treating a check as not applicable.

use serde_json::Value;

use crate::declaration::Check;
use crate::json::truthy;
use crate::kind::KindError;
use crate::kinds::require;

const MODALITIES: [&str; 4] = ["text", "image", "video", "audio"];

fn served(value: &Value) -> Option<Vec<&str>> {
    value["server"]["health"]["modalities"]
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect()
}

fn reason(value: &Value, modality: &str) -> Option<String> {
    let served = served(value)?;
    if served.contains(&modality) {
        return None;
    }
    let listed = MODALITIES
        .iter()
        .filter(|name| served.contains(name))
        .copied()
        .collect::<Vec<_>>()
        .join(",");
    Some(format!(
        "the server does not serve {modality} input (--modalities {listed})"
    ))
}

/// A skipped row must agree with the server's reported modalities.
pub(crate) fn unserved_row(value: &Value, row: &Value, modality: &str) -> Result<bool, KindError> {
    require(
        value
            .get("server")
            .is_none_or(|server| server.is_null() || server.is_object()),
        || "invalid recorded server".into(),
    )?;
    let reason = reason(value, modality);
    let skipped = row["status"] == "not_applicable";
    require(skipped == reason.is_some(), || {
        "media row does not match the served modalities".into()
    })?;
    if let Some(reason) = reason {
        require(
            row["kind"] == modality
                && row.get("passed") == Some(&Value::Null)
                && row["reason"] == reason
                && value["not_applicable"]
                    .as_array()
                    .is_some_and(|reasons| reasons.iter().any(|entry| entry == &reason)),
            || "invalid unserved media row".into(),
        )?;
    }
    Ok(skipped)
}

/// Validate receipt reasons and their binding to the policy's checkpoint.
pub(crate) fn inapplicable_reason(
    value: &Value,
    check: &Check,
) -> Result<Option<String>, KindError> {
    let Some(reasons) = value
        .get("not_applicable")
        .filter(|reasons| truthy(reasons))
    else {
        return Ok(None);
    };
    let reasons: Option<Vec<&str>> = reasons
        .as_array()
        .and_then(|items| items.iter().map(Value::as_str).collect());
    require(reasons.is_some(), || {
        "invalid not_applicable reasons".into()
    })?;
    let reasons = reasons.expect("validated");
    require(
        value["schema"] == "chopinfer-qualification/v1" && !truthy(&value["mechanics_only"]),
        || "not_applicable needs a qualification receipt".into(),
    )?;
    require(value["server"].is_object(), || {
        "no server for not_applicable evidence".into()
    })?;
    require(
        served(value).is_some_and(|names| {
            names.contains(&"text") && names.iter().all(|name| MODALITIES.contains(name))
        }),
        || "no valid served modalities for not_applicable evidence".into(),
    )?;
    let serving = &value["server"]["health"]["serving"];
    require(
        crate::http::startup_nonce(&serving["startup_nonce"])
            && check.declared["checkpoint_identity"]
                .as_str()
                .is_some_and(|pin| !pin.is_empty() && serving["checkpoint_identity"] == pin),
        || "not_applicable evidence names another checkpoint or lacks its startup nonce".into(),
    )?;
    require(check.module == "multimodal", || {
        "this check has no optional media".into()
    })?;
    let rows = value["rows"].as_array();
    require(
        rows.is_some_and(|rows| rows.iter().all(Value::is_object)),
        || "no media rows".into(),
    )?;
    let mut expected: Vec<&str> = rows
        .expect("validated")
        .iter()
        .filter(|row| row["status"] == "not_applicable")
        .filter_map(|row| row["reason"].as_str())
        .collect();
    expected.sort_unstable();
    expected.dedup();
    require(!expected.is_empty() && reasons == expected, || {
        "not_applicable reasons do not match the media evidence".into()
    })?;
    Ok(Some(reasons.join("; ")))
}

pub(crate) fn all_unserved(metrics: &serde_json::Map<String, Value>) -> bool {
    metrics
        .get("rows")
        .and_then(Value::as_array)
        .is_some_and(|rows| {
            !rows.is_empty() && rows.iter().all(|row| row["status"] == "not_applicable")
        })
}

pub(crate) fn passed(rows: &[Value]) -> bool {
    let applicable: Vec<&Value> = rows
        .iter()
        .filter(|row| row["status"] != "not_applicable")
        .collect();
    !applicable.is_empty() && applicable.iter().all(|row| truthy(&row["passed"]))
}
