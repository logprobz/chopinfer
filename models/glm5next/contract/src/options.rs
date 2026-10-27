//! The `chop.model-options.v1` options a package declares beside its
//! checkpoint layout (`crate::quantization`) and its overridable runtime
//! fields (`crate::geometry`): the Sinkhorn mHC comm epilogue switch and the
//! MTP drafter's coupled sampled drafts.

/// The option that runs each sublayer boundary's all-reduce, round,
/// Sinkhorn expand, next mix-collapse and next input norm as the all-reduce
/// followed by the Sinkhorn mHC comm epilogue split over CTAs.
/// Boolean and off by default: the unfused launches are the qualified path
/// until the epilogue's GPU bitwise check passes, so the serving identity
/// reports a server with it on as unqualified.
pub const COMM_EPILOGUE_OPTION: &str = "runtime.sinkhorn_comm_epilogue";

/// The option that drafts each sampled item's MTP proposals from the target
/// sampler's own noise (`kernel_core::draft_noise`): draft position `j`
/// proposes the argmax of the drafter's logits divided by the item's
/// temperature plus the Gumbel noise of the generated index the target
/// samples that position at (`crate::draft_argmax`). Greedy items draft the
/// same tokens either way, and every emitted token stays the target's
/// sample: a draft is kept only when it equals the target's sample at its
/// row. Boolean and off by default, so the serving identity reports a
/// server with it on as unqualified, until served timings resolve a gain.
pub const COUPLED_DRAFTS_OPTION: &str = "runtime.mtp_coupled_drafts";

/// The [`COMM_EPILOGUE_OPTION`] a package declares.
pub fn comm_epilogue_option() -> runtime_bundle::model_options::ModelOptionV1 {
    switch(COMM_EPILOGUE_OPTION)
}

/// The [`COUPLED_DRAFTS_OPTION`] a package declares.
pub fn coupled_drafts_option() -> runtime_bundle::model_options::ModelOptionV1 {
    switch(COUPLED_DRAFTS_OPTION)
}

/// A Boolean switch `name`, off and qualified by default.
fn switch(name: &str) -> runtime_bundle::model_options::ModelOptionV1 {
    runtime_bundle::model_options::ModelOptionV1 {
        name: name.into(),
        definition: runtime_bundle::model_options::OptionDefinitionV1::Boolean { default: false },
        default_is_qualified: true,
    }
}

/// A resolved [`COMM_EPILOGUE_OPTION`] value: a Boolean, off when absent.
pub fn comm_epilogue_value(value: Option<&serde_json::Value>) -> Result<bool, String> {
    switch_value(COMM_EPILOGUE_OPTION, value)
}

/// A resolved [`COUPLED_DRAFTS_OPTION`] value: a Boolean, off when absent.
pub fn coupled_drafts_value(value: Option<&serde_json::Value>) -> Result<bool, String> {
    switch_value(COUPLED_DRAFTS_OPTION, value)
}

fn switch_value(name: &str, value: Option<&serde_json::Value>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some(serde_json::Value::Bool(on)) => Ok(*on),
        Some(other) => Err(format!("unsupported {name} {other}")),
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

    /// The coupled drafts switch is a Boolean, off and qualified by default.
    #[test]
    fn the_coupled_drafts_option_is_a_switch_off_by_default() {
        let option = coupled_drafts_option();
        assert_eq!(option.name, COUPLED_DRAFTS_OPTION);
        assert_eq!(option.definition.default_value(), false);
        assert!(option.default_is_qualified);
        assert_eq!(coupled_drafts_value(None), Ok(false));
        assert_eq!(coupled_drafts_value(Some(&false.into())), Ok(false));
        assert_eq!(coupled_drafts_value(Some(&true.into())), Ok(true));
        assert_eq!(
            coupled_drafts_value(Some(&"true".into())).unwrap_err(),
            "unsupported runtime.mtp_coupled_drafts \"true\""
        );
    }
}
