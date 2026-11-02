use axum::Json;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use std::borrow::Cow;

use crate::OutputRefusal;
use crate::request_log::{FailureDetail, JsonErrorPosition};

pub(crate) const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    /// What the client reads. Only a literal is logged: a formatted message
    /// can quote what the request sent.
    pub(crate) message: Cow<'static, str>,
    pub(crate) kind: &'static str,
    pub(crate) parameter: Option<String>,
    pub(crate) code: Option<String>,
    /// What the request-failure line may say beyond the status.
    pub(crate) log: ErrorLog,
}

/// The log-safe facts of an error, beside the message its client reads.
#[derive(Debug, Default)]
pub(crate) struct ErrorLog {
    /// Logged in place of a message that is not a literal.
    pub(crate) summary: &'static str,
    /// What only a few errors carry, boxed so every error stays small.
    pub(crate) detail: Option<Box<ErrorLogDetail>>,
}

#[derive(Debug, Default)]
pub(crate) struct ErrorLogDetail {
    pub(crate) json_error: Option<JsonErrorPosition>,
    pub(crate) output_refusal: Option<OutputRefusal>,
    /// The model name an unknown-model refusal was asked for, bounded.
    pub(crate) requested_model: Option<String>,
    /// The refusal ended a request whose client had already left.
    pub(crate) disconnected: bool,
    /// The refusal or failure came from drain or shutdown.
    pub(crate) shutdown: bool,
}

impl ErrorLog {
    const fn summary(summary: &'static str) -> Self {
        Self {
            summary,
            detail: None,
        }
    }

    pub(crate) fn detail_mut(&mut self) -> &mut ErrorLogDetail {
        self.detail.get_or_insert_default()
    }
}

impl ApiError {
    pub(crate) fn invalid(message: impl Into<Cow<'static, str>>, parameter: Option<&str>) -> Self {
        Self::invalid_owned(message, parameter.map(str::to_owned))
    }

    /// The same refusal, when the parameter name is already owned.
    pub(crate) fn invalid_owned(
        message: impl Into<Cow<'static, str>>,
        parameter: Option<String>,
    ) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "BadRequestError",
            code: None,
            parameter,
            log: ErrorLog::summary("invalid request"),
        }
    }

    pub(crate) fn capacity(message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: message.into(),
            kind: "TooManyRequestsError",
            code: None,
            parameter: None,
            log: ErrorLog::summary("capacity exhausted"),
        }
    }

    pub(crate) fn context_length(message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "BadRequestError",
            code: None,
            parameter: None,
            log: ErrorLog::summary("context length exceeded"),
        }
    }

    pub(crate) fn unavailable(message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            kind: "InternalServerError",
            code: None,
            parameter: None,
            log: ErrorLog::summary("unavailable"),
        }
    }

    pub(crate) fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!("model '{model}' is not available").into(),
            kind: "NotFoundError",
            code: None,
            parameter: Some("model".to_owned()),
            log: ErrorLog {
                detail: Some(Box::new(ErrorLogDetail {
                    requested_model: Some(bounded_log_text(model, 128)),
                    ..ErrorLogDetail::default()
                })),
                ..ErrorLog::summary("model is not available")
            },
        }
    }

    pub(crate) fn backend(message: impl Into<Cow<'static, str>>, code: Option<String>) -> Self {
        let status = match code.as_deref() {
            Some("shutdown_timeout" | "worker_unavailable") => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            message: message.into(),
            kind: "InternalServerError",
            log: ErrorLog {
                detail: (code.as_deref() == Some("shutdown_timeout")).then(|| {
                    Box::new(ErrorLogDetail {
                        shutdown: true,
                        ..ErrorLogDetail::default()
                    })
                }),
                ..ErrorLog::summary("backend failure")
            },
            code,
            parameter: None,
        }
    }

    /// A refusal for a JSON body that did not parse, logged with where it
    /// stopped instead of serde's message, which can quote the body.
    pub(crate) fn invalid_json(error: &serde_json::Error, parameter: Option<&str>) -> Self {
        Self::invalid(format!("invalid JSON request: {error}"), parameter).json_error(error)
    }

    /// The same refusal, logged with where `error` stopped instead of its
    /// message.
    pub(crate) fn json_error(mut self, error: &serde_json::Error) -> Self {
        self.log.summary = "invalid JSON request";
        self.log.detail_mut().json_error = Some(JsonErrorPosition::of(error));
        self
    }

    /// Adds log-safe output facts without changing the client's error.
    pub(crate) fn with_output_refusal(mut self, mut refusal: OutputRefusal) -> Self {
        refusal.reason = bounded_log_text(&refusal.reason, 256);
        self.log.detail_mut().output_refusal = Some(refusal);
        self
    }

    /// What the request-failure line reports: the literal message, or the
    /// constructor's summary for a formatted one.
    pub(crate) fn failure_detail(&self) -> FailureDetail {
        let detail = self.log.detail.as_deref();
        FailureDetail {
            status: self.status.as_u16(),
            error_type: self.kind,
            code: self.code.as_deref().map(|code| bounded_log_text(code, 64)),
            param: self
                .parameter
                .as_deref()
                .map(|parameter| bounded_log_text(parameter, 64)),
            message: match &self.message {
                Cow::Borrowed(message) => message,
                Cow::Owned(_) => self.log.summary,
            },
            json_error: detail.and_then(|detail| detail.json_error),
            output_refusal: detail.and_then(|detail| detail.output_refusal.clone()),
            requested_model: detail.and_then(|detail| detail.requested_model.clone()),
            disconnected: detail.is_some_and(|detail| detail.disconnected),
            routine: detail.is_some_and(|detail| detail.shutdown),
        }
    }

    pub(crate) fn value(&self) -> serde_json::Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.parameter,
                "code": self.code.as_ref().map_or_else(|| json!(self.status.as_u16()), |code| json!(code)),
            }
        })
    }

    pub(crate) fn into_response(self, request_id: &str) -> Response {
        let status = self.status;
        let mut response = (status, Json(self.value())).into_response();
        if status == StatusCode::SERVICE_UNAVAILABLE {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                HeaderValue::from_static("1"),
            );
        }
        insert_request_id(&mut response, request_id);
        response.extensions_mut().insert(self.failure_detail());
        response
    }
}

/// `value` cut to at most `max_bytes` at a character boundary, on one line:
/// control and line-separator characters become `?`. Every field a client
/// controls passes through here before a log line holds it.
pub(crate) fn bounded_log_text(value: &str, max_bytes: usize) -> String {
    let mut result = String::with_capacity(value.len().min(max_bytes));
    for ch in value.chars() {
        let ch = if ch.is_control() || matches!(ch, '\u{2028}' | '\u{2029}') {
            '?'
        } else {
            ch
        };
        if result.len() + ch.len_utf8() > max_bytes {
            break;
        }
        result.push(ch);
    }
    result
}

pub(crate) fn insert_request_id(response: &mut Response, request_id: &str) {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert(X_REQUEST_ID, value);
    }
}

#[cfg(test)]
mod tests {
    use super::{ApiError, bounded_log_text};

    #[test]
    fn log_text_is_bounded_in_bytes_and_single_line() {
        assert_eq!(
            bounded_log_text("a\n\r\t\0\u{1b}\u{7f}\u{2028}\u{2029}b", 64),
            "a????????b"
        );
        assert_eq!(bounded_log_text(&"x".repeat(200), 128).len(), 128);
        // A multibyte character that would cross the limit is left out whole.
        assert_eq!(bounded_log_text("ab\u{00e9}", 3), "ab");
        assert_eq!(bounded_log_text("ab\u{00e9}", 4), "ab\u{00e9}");
    }

    #[test]
    fn output_refusals_are_bounded_and_only_reach_the_log() {
        let error = ApiError::backend(
            "model output could not be decoded".to_owned(),
            Some("invalid_model_output".to_owned()),
        );
        let client = error.value();
        let error = error.with_output_refusal(crate::OutputRefusal {
            reason: format!("fixed\n\r\u{2028}{}canary-7f3a", "\u{00e9}".repeat(200)),
            token_ordinal: Some(7),
            token_id: Some(42),
        });
        assert_eq!(error.value(), client);
        let detail = error.failure_detail();
        assert_eq!(detail.message, "backend failure");
        assert_eq!(detail.code.as_deref(), Some("invalid_model_output"));
        let refusal = detail.output_refusal.unwrap();
        assert_eq!(refusal.reason.len(), 256);
        assert!(refusal.reason.starts_with("fixed???"));
        assert!(!refusal.reason.contains(['\n', '\r', '\u{2028}']));
        assert!(!refusal.reason.contains("canary"));
        assert_eq!(refusal.token_ordinal, Some(7));
        assert_eq!(refusal.token_id, Some(42));
        assert!(
            ApiError::backend("other failure", None)
                .failure_detail()
                .output_refusal
                .is_none()
        );
    }

    #[test]
    fn only_a_literal_message_reaches_the_failure_line() {
        let literal = ApiError::invalid("prompt must not be empty", Some("prompt"));
        assert_eq!(literal.failure_detail().message, "prompt must not be empty");
        assert_eq!(literal.failure_detail().param.as_deref(), Some("prompt"));
        let canary = "canary-7f3a";
        let formatted = ApiError::invalid(format!("value {canary} is invalid"), None);
        assert_eq!(formatted.message, "value canary-7f3a is invalid");
        assert_eq!(formatted.failure_detail().message, "invalid request");
        let backend = ApiError::backend(format!("bundle said {canary}"), None);
        assert_eq!(backend.failure_detail().message, "backend failure");
        assert!(!backend.failure_detail().routine);
        assert!(
            ApiError::backend(
                "server drain deadline expired",
                Some("shutdown_timeout".into())
            )
            .failure_detail()
            .routine
        );
    }

    #[test]
    fn json_refusals_log_where_parsing_stopped_not_what_it_read() {
        let error =
            serde_json::from_str::<serde_json::Value>("{\"prompt\": \"canary-7f3a").unwrap_err();
        let refusal = ApiError::invalid_json(&error, None);
        let detail = refusal.failure_detail();
        assert_eq!(detail.message, "invalid JSON request");
        let position = detail.json_error.unwrap();
        assert_eq!((position.category, position.line), ("eof", 1));
        assert!(!format!("{detail:?}").contains("canary"));
    }

    #[test]
    fn an_unknown_model_keeps_its_requested_name_bounded() {
        let refusal = ApiError::model_not_found(&format!("wrong\nname{}", "x".repeat(300)));
        let detail = refusal.failure_detail();
        assert_eq!(detail.status, 404);
        assert_eq!(detail.message, "model is not available");
        let requested = detail.requested_model.unwrap();
        assert!(requested.starts_with("wrong?name"));
        assert_eq!(requested.len(), 128);
    }
}
