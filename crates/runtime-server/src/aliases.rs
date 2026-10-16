//! A deployment can rename its API identities without changing its package.
use crate::frontend::*;
use std::sync::Arc;

pub(crate) fn wrap(
    frontend: Arc<dyn ServerFrontendV1>,
    names: &[String],
) -> Arc<dyn ServerFrontendV1> {
    if names.is_empty() {
        return frontend;
    }
    let owner = frontend
        .models()
        .first()
        .map(|model| model.owned_by.clone())
        .unwrap_or_default();
    Arc::new(AliasedFrontend {
        models: names
            .iter()
            .map(|name| FrontendModelV1::new(name, &owner))
            .collect(),
        inner: frontend,
    })
}

pub(crate) struct AliasedFrontend {
    pub(crate) inner: Arc<dyn ServerFrontendV1>,
    pub(crate) models: Vec<FrontendModelV1>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe;
    impl ServerFrontendV1 for Probe {
        fn sampling_defaults(&self) -> runtime_sampling::SamplingDefaults {
            runtime_sampling::SamplingDefaults {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: Some(23),
                min_p: Some(0.1),
                repetition_penalty: Some(1.1),
            }
        }
        fn delegation_probe(&self) -> usize {
            42
        }
        fn models(&self) -> &[FrontendModelV1] {
            &[]
        }
        fn maximum_total_tokens(&self) -> usize {
            99
        }
        fn maximum_tool_calls(&self) -> usize {
            3
        }
        fn allows_remote_media(&self) -> bool {
            true
        }
        fn supports_media(&self) -> bool {
            true
        }
        fn has_selected_modalities(&self) -> bool {
            true
        }
        fn served_media(&self) -> Vec<crate::FrontendMediaPartKindV1> {
            vec![crate::FrontendMediaPartKindV1::Audio]
        }
        fn allowed_local_media_path(&self) -> Option<&std::path::Path> {
            Some(std::path::Path::new("/data/media"))
        }
        fn reasoning_field(&self) -> &'static str {
            "reasoning"
        }
        fn tool_call_reasoning_fallback(&self) -> bool {
            true
        }
        fn extra_roles(&self) -> &[String] {
            const ROLES: &[String] = &[String::new()];
            ROLES
        }
        fn tool_references(&self) -> bool {
            true
        }
        fn render_chat(
            &self,
            _: FrontendChatRequestV1,
        ) -> Result<FrontendRenderedChatV1, FrontendErrorV1> {
            Err(FrontendErrorV1::new("render delegated", None))
        }
        fn validate_tool_call(&self, name: &str, arguments: &str) -> Result<(), FrontendErrorV1> {
            Err(FrontendErrorV1::new(format!("{name}:{arguments}"), None))
        }
        fn tool_constraint(
            &self,
            _: &[FrontendToolV1],
            _: &FrontendToolChoiceV1,
        ) -> Result<Option<runtime_grammar::Grammar>, FrontendErrorV1> {
            Err(FrontendErrorV1::new("constraint delegated", None))
        }
    }

    #[test]
    fn new_default_methods_are_forwarded_without_wrapper_edits() {
        let inner: Arc<dyn ServerFrontendV1> = Arc::new(Probe);
        let wrapped = wrap(inner.clone(), &["alias".into()]);
        let request = FrontendChatRequestV1 {
            messages: vec![],
            continue_final_message: false,
            add_generation_prompt: true,
            tools: vec![],
            tool_choice: FrontendToolChoiceV1::Auto,
            response_format: None,
            reasoning_effort: None,
            enable_thinking: None,
            thinking: None,
            nested_reasoning_effort: None,
            preserve_thinking: None,
            clear_thinking: None,
            add_vision_id: None,
            tool_objects: Vec::new(),
            chat_template_kwargs: None,
        };
        assert_eq!(
            wrapped.render_chat(request).unwrap_err().message,
            "render delegated"
        );
        assert_eq!(wrapped.delegation_probe(), 42);
        assert_eq!(wrapped.sampling_defaults(), inner.sampling_defaults());
        assert_eq!(wrapped.models()[0].id, "alias");
        assert_eq!(wrapped.supports_media(), inner.supports_media());
        assert!(wrapped.has_selected_modalities());
        assert_eq!(
            wrapped.served_media(),
            [crate::FrontendMediaPartKindV1::Audio]
        );
        assert!(wrapped.allows_remote_media());
        assert_eq!(
            wrapped.allowed_local_media_path(),
            inner.allowed_local_media_path()
        );
        assert_eq!(wrapped.reasoning_field(), inner.reasoning_field());
        assert_eq!(
            wrapped.tool_call_reasoning_fallback(),
            inner.tool_call_reasoning_fallback()
        );
        assert_eq!(wrapped.extra_roles(), inner.extra_roles());
        assert!(wrapped.tool_references());
        assert_eq!(wrapped.maximum_total_tokens(), inner.maximum_total_tokens());
        assert_eq!(wrapped.maximum_tool_calls(), inner.maximum_tool_calls());
        assert_eq!(
            wrapped.validate_tool_call("tool", "{}"),
            inner.validate_tool_call("tool", "{}")
        );
        assert_eq!(
            wrapped
                .tool_constraint(&[], &FrontendToolChoiceV1::Auto)
                .unwrap_err(),
            inner
                .tool_constraint(&[], &FrontendToolChoiceV1::Auto)
                .unwrap_err()
        );
    }
}
