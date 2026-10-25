//! Diagnostic comparison with complete identity pins and no release eligibility.

use serde_json::{Map, Value, json};

use crate::json::{canonical_sha256, truthy};
use crate::kind::{KindError, admit};
use crate::kinds::invalid;

const MARKER: &str = "experimental_determinism";

fn hex(value: &Value, length: usize) -> bool {
    value.as_str().is_some_and(|value| {
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn context(record: &Value) -> Result<&Value, KindError> {
    if truthy(&record["errors"]) || truthy(&record["skipped"]) {
        return Err(invalid(
            "diagnostic determinism contains errors or skipped coverage",
        ));
    }
    let serving = record
        .pointer("/server/health/serving")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid("diagnostic determinism needs a complete serving identity"))?;
    if !hex(&serving["checkpoint_identity"], 64)
        || !hex(&serving["startup_nonce"], 32)
        || ["checkpoint_identity", "startup_nonce"]
            .iter()
            .any(|key| record["server"]["checkpoint"][key] != serving[key])
    {
        return Err(invalid(
            "diagnostic determinism needs a bound checkpoint and startup nonce",
        ));
    }
    let Some(marker) = record.get(MARKER) else {
        admit(record, false)?;
        return Ok(serving);
    };
    let pin = &marker["pin"];
    let complete_pin = pin.as_object().is_some_and(|pin| {
        pin.len() == 3
            && ["model_options", "checkpoint_identity", "startup_nonce"]
                .iter()
                .all(|key| pin.contains_key(*key))
    });
    let options = pin["model_options"].as_object().filter(|options| {
        !options.is_empty()
            && options.iter().all(|(key, value)| {
                !key.is_empty()
                    && match value {
                        Value::Bool(_) | Value::String(_) => true,
                        Value::Number(number) => !number.is_f64(),
                        _ => false,
                    }
            })
    });
    if !complete_pin
        || options.is_none()
        || !hex(&pin["checkpoint_identity"], 64)
        || !hex(&pin["startup_nonce"], 32)
        || !hex(&marker["file_sha256"], 64)
        || marker["pin_sha256"] != canonical_sha256(pin)
        || marker["qualified"] != false
        || record["qualified"] != false
        || record["server"][MARKER] != *marker
        || serving["model_options_qualified"] != false
        || ["model_options", "checkpoint_identity", "startup_nonce"]
            .iter()
            .any(|key| serving[key] != pin[key])
        || ["checkpoint_identity", "startup_nonce"]
            .iter()
            .any(|key| record["server"]["checkpoint"][key] != pin[key])
    {
        return Err(invalid(
            "diagnostic determinism options, checkpoint or startup nonce differ from the sealed pin",
        ));
    }
    Ok(serving)
}

pub(crate) fn score_capture(left: &Value, right: &Value) -> Result<Map<String, Value>, KindError> {
    if left.get(MARKER).is_none() && right.get(MARKER).is_none() {
        return Err(invalid(
            "diagnostic determinism requires an explicitly experimental capture",
        ));
    }
    let (a, b) = (context(left)?, context(right)?);
    if a["model_options"] != b["model_options"]
        || a["checkpoint_identity"] != b["checkpoint_identity"]
    {
        return Err(invalid(
            "diagnostic determinism arms use different options or checkpoints",
        ));
    }
    let mut metrics = super::compare(left, right)?;
    metrics.insert("qualified".into(), json!(false));
    metrics.insert(MARKER.into(), json!(true));
    Ok(metrics)
}
