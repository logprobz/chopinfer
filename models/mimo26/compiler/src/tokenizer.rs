//! The tokenizer contract declared by the pinned family frontend.
use runtime_tokenizer::{Tokenizer, TokenizerContract};
use std::path::Path;

pub fn special_tokens() -> Vec<String> {
    crate::frontend::descriptor()
        .vocabulary
        .prompt_special_tokens
}

pub fn tokenizer_contract() -> runtime_tokenizer::Result<TokenizerContract> {
    tokenizer_contract_for_flash(crate::checkpoint::default_identity())
}

pub fn tokenizer_contract_for_flash(
    identity: &crate::checkpoint::Identity,
) -> runtime_tokenizer::Result<TokenizerContract> {
    crate::checkpoint::registry().tokenizer_contract_for_flash(identity)
}

impl crate::checkpoint::Registry {
    pub fn tokenizer_contract_for_flash(
        &self,
        identity: &crate::checkpoint::Identity,
    ) -> runtime_tokenizer::Result<TokenizerContract> {
        self.tokenizer_contract_for_geometry(identity, &model_mimo26_contract::geometry::V26_FLASH)
    }

    pub fn tokenizer_contract_for_geometry(
        &self,
        identity: &crate::checkpoint::Identity,
        geometry: &'static model_mimo26_contract::geometry::Geometry,
    ) -> runtime_tokenizer::Result<TokenizerContract> {
        let build = || -> crate::MimoResult<TokenizerContract> {
            let bytes = self.component_for_geometry(
                identity,
                model_mimo26_contract::encoders::Encoders::NONE,
                geometry,
            )?;
            let descriptor =
                runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(&bytes)?;
            Ok(runtime_frontend_contract::tokenizer_contract_v1(
                &descriptor,
            )?)
        };
        build().map_err(|error| runtime_tokenizer::TokenizerError::validation(error.to_string()))
    }
}

pub fn load_tokenizer(checkpoint: &Path) -> runtime_tokenizer::Result<Tokenizer> {
    Tokenizer::load(checkpoint, &tokenizer_contract()?)
}
