//! The `chop.model-options.v1` options a package declares beside its
//! checkpoint layout (`crate::quantization`) and its overridable runtime
//! fields (`crate::geometry`): the Sinkhorn mHC comm epilogue switch.

/// The option that runs each sublayer boundary's all-reduce, round,
/// Sinkhorn expand, next mix-collapse and next input norm as the all-reduce
/// followed by the Sinkhorn mHC comm epilogue split over CTAs.
/// Boolean and off by default: the unfused launches are the qualified path
/// until the epilogue's GPU bitwise check passes, so the serving identity
/// reports a server with it on as unqualified.
pub const COMM_EPILOGUE_OPTION: &str = "runtime.sinkhorn_comm_epilogue";

/// The [`COMM_EPILOGUE_OPTION`] a package declares.
pub fn comm_epilogue_option() -> runtime_bundle::model_options::ModelOptionV1 {
    runtime_bundle::model_options::ModelOptionV1 {
        name: COMM_EPILOGUE_OPTION.into(),
        definition: runtime_bundle::model_options::OptionDefinitionV1::Boolean { default: false },
        default_is_qualified: true,
    }
}

/// A resolved [`COMM_EPILOGUE_OPTION`] value: a Boolean, off when absent.
pub fn comm_epilogue_value(value: Option<&serde_json::Value>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some(serde_json::Value::Bool(on)) => Ok(*on),
        Some(other) => Err(format!("unsupported {COMM_EPILOGUE_OPTION} {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch is a Boolean, off and qualified by default.
    #[test]
    fn the_comm_epilogue_option_is_an_unqualified_switch_off_by_default() {
        let option = comm_epilogue_option();
        assert_eq!(option.definition.default_value(), false);
        assert!(option.default_is_qualified);
        assert_eq!(comm_epilogue_value(None), Ok(false));
        assert_eq!(comm_epilogue_value(Some(&true.into())), Ok(true));
        assert_eq!(
            comm_epilogue_value(Some(&1.into())).unwrap_err(),
            "unsupported runtime.sinkhorn_comm_epilogue 1"
        );
    }
}
