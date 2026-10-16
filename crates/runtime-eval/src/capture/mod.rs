//! What the kinds' live captures share: the receipt envelope
//! (`common.receipt`), the declared inputs and arguments they read, and the
//! frontend's thinking switch (`common.thinking_kwargs`).

pub mod host_load;

use std::io::Read as _;
use std::path::PathBuf;
use std::time::{Instant, SystemTime};

use serde_json::{Map, Value, json};

use crate::declaration::Check;
use crate::http::HttpError;
use crate::json;
use crate::kind::{Capture, KindError};
use crate::references::{input_path, process_environment};
use crate::report::utc_stamp;

pub const SCHEMA: &str = "chopinfer-qualification/v1";

/// The receipt a capture writes (`common.receipt`): the envelope, with the
/// host load from the capture's start to now, then the kind's fields.
/// `server` is the identity bound to the pinned checkpoint.
pub fn receipt(check: &str, capture: &Capture<'_>, result: Map<String, Value>) -> Value {
    let mut server = capture.server.to_json();
    server["checkpoint"] = capture.bound.to_json();
    let mut value = Map::new();
    value.insert("schema".into(), json!(SCHEMA));
    value.insert("check".into(), json!(check));
    value.insert("at".into(), json!(utc_stamp(SystemTime::now())));
    value.insert("server".into(), server);
    value.insert("mechanics_only".into(), json!(capture.mechanics));
    value.insert(
        "host_load".into(),
        host_load::judge(
            &[capture.host_load_start.clone(), host_load::sample()],
            None,
        ),
    );
    value.extend(result);
    Value::Object(value)
}

/// The check's declared arguments.
pub fn arguments(check: &Check) -> Vec<String> {
    check.declared["args"]
        .as_array()
        .map(|args| {
            args.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

/// Whether the check's arguments hold `flag`.
pub fn flag(check: &Check, flag: &str) -> bool {
    arguments(check).iter().any(|arg| arg == flag)
}

/// The value after `option` in the check's arguments.
pub fn option(check: &Check, option: &str) -> Option<String> {
    let args = arguments(check);
    args.iter()
        .position(|arg| arg == option)
        .and_then(|index| args.get(index + 1).cloned())
}

/// A declared input file read as JSON, authenticated by the reference
/// lock (`common.input_bytes`), with its path.
pub fn input_json(capture: &Capture<'_>, name: &str) -> Result<(PathBuf, Value), KindError> {
    let path = input_path(capture.declaration, name, &process_environment).ok_or_else(|| {
        KindError::Insufficient(format!(
            "{} declares no usable input {name}",
            capture.declaration.declaration.model
        ))
    })?;
    let bytes =
        crate::lock::read(capture.declaration, name, &path).map_err(KindError::Insufficient)?;
    let value = json::parse(&bytes)
        .map_err(|error| KindError::Insufficient(format!("{}: {error}", path.display())))?;
    Ok((path, value))
}

/// The chat template arguments that turn the model's thinking on or off:
/// the frontend declaration's first thinking keyword.
pub fn thinking_kwargs(capture: &Capture<'_>, enabled: bool) -> Result<Value, KindError> {
    let (path, frontend) = input_json(capture, "frontend")?;
    let keyword = frontend["reasoning"]["thinking_keywords"][0]
        .as_str()
        .ok_or_else(|| {
            KindError::Insufficient(format!(
                "{}: the frontend has no thinking switch keyword",
                path.display()
            ))
        })?;
    Ok(json!({keyword: enabled}))
}

/// Whether the frontend admits a request that disables thinking
/// (`reasoning.allow_disabled`, true when absent;
/// `common.thinking_can_be_disabled`).
pub fn thinking_can_be_disabled(capture: &Capture<'_>) -> Result<bool, KindError> {
    let (_, frontend) = input_json(capture, "frontend")?;
    Ok(frontend["reasoning"]["allow_disabled"] != Value::Bool(false))
}

/// The switches of a request that wants an answer rather than reasoning
/// (`common.answer_kwargs`): thinking disabled, or none where the frontend
/// refuses that.
pub fn answer_kwargs(capture: &Capture<'_>) -> Result<Value, KindError> {
    if thinking_can_be_disabled(capture)? {
        thinking_kwargs(capture, false)
    } else {
        Ok(json!({}))
    }
}

/// The time left before the check's deadline, or an environment error.
pub fn remaining(capture: &mut Capture<'_>) -> Result<std::time::Duration, KindError> {
    capture.client.set_deadline(Some(capture.deadline));
    capture
        .deadline
        .checked_duration_since(Instant::now())
        .filter(|left| !left.is_zero())
        .ok_or_else(|| KindError::Environment("the check's time budget is exhausted".into()))
}

/// A transport failure is the environment's; anything else the server
/// answered fails the check.
pub fn http_error(error: HttpError) -> KindError {
    match error {
        HttpError::Transport(message) => KindError::Environment(message),
        other => KindError::Invalid(other.to_string()),
    }
}

/// A random version 4 UUID as 32 hex digits (Python's `uuid.uuid4().hex`),
/// the fresh part of a cache salt.
pub fn salt() -> Result<String, KindError> {
    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut source| source.read_exact(&mut bytes))
        .map_err(|error| KindError::Environment(format!("/dev/urandom: {error}")))?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}
