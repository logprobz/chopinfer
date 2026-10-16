//! Everything a deployment needs, read out of the bundle it selected.
//!
//! The frontend declaration and the two tokenizer artifacts are components of
//! one installed bundle, resolved by role through the package that verified
//! them. Reading them together is what keeps them consistent: the tokenizer a
//! prompt is encoded with is the one the declaration that rendered the prompt
//! pinned, because both came out of the same manifest in one pass.
//!
//! The frontend itself is the package's: its native module exports the chat
//! format (`chop.frontend.v1`), and the host serves that export over the same
//! authenticated component.
//!
//! There is no discovery and no path lookup here. A component that is missing,
//! that fails its digest, or that does not match what the declaration pinned
//! is a refusal, not a fallback, and so is a package without a frontend
//! export: the host has no format of its own to serve instead.

use std::fmt;
use std::sync::Arc;

use runtime_bundle::{BundleManifestV1, BundlePackageV1};
use runtime_bundle_abi::BundleLibraryV1;
use runtime_bundle_abi::frontend::{FRONTEND_CAPABILITY_V1, FrontendExportV1};
use runtime_frontend::{
    BundleFrontendV1, BundleParserSourceV1, FRONTEND_COMPONENT_ROLE_V1, FrontendComponentErrorV1,
    FrontendDescriptorV1, TOKENIZER_CONFIGURATION_ROLE_V1, TOKENIZER_DEFINITION_ROLE_V1,
    tokenizer_from_components_v1,
};
use runtime_server::{ServerConfig, ServerFrontendV1};
use runtime_tokenizer::Tokenizer;

use crate::generation::EngineModelPolicyV1;
use crate::install::{HostLoadErrorV1, InstalledBundleHostV1, verified_component_by_role_v1};

/// Why one bundle could not be served.
#[derive(Debug)]
pub enum BundleDeploymentErrorV1 {
    Component(HostLoadErrorV1),
    Declaration(FrontendComponentErrorV1),
    Tokenizer(String),
    /// The package serves no frontend export this host can open.
    Frontend(String),
}

impl fmt::Display for BundleDeploymentErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Component(error) => error.fmt(formatter),
            Self::Declaration(error) => error.fmt(formatter),
            Self::Tokenizer(message) | Self::Frontend(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for BundleDeploymentErrorV1 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Component(error) => Some(error),
            Self::Declaration(error) => Some(error),
            Self::Tokenizer(_) | Self::Frontend(_) => None,
        }
    }
}

/// One deployment, built from one bundle.
pub struct BundleDeploymentV1 {
    frontend: Arc<dyn ServerFrontendV1>,
    declaration: FrontendDescriptorV1,
    /// The authenticated frontend component the declaration was read from,
    /// which the frontend's format opens over.
    frontend_component: Arc<[u8]>,
    tokenizer: Tokenizer,
    policy: EngineModelPolicyV1,
}

impl BundleDeploymentV1 {
    /// The frontend the transport serves.
    #[must_use]
    pub fn frontend(&self) -> Arc<dyn ServerFrontendV1> {
        Arc::clone(&self.frontend)
    }

    /// The tokenizer that declaration pinned.
    #[must_use]
    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The tokenizer, moved out for a generation backend to own.
    #[must_use]
    pub fn into_tokenizer(self) -> Tokenizer {
        self.tokenizer
    }

    /// What the generation side reads from the same declaration.
    #[must_use]
    pub fn policy(&self) -> EngineModelPolicyV1 {
        self.policy.clone()
    }

    #[must_use]
    pub const fn declaration(&self) -> &FrontendDescriptorV1 {
        &self.declaration
    }

    /// The authenticated frontend component bytes the declaration was read
    /// from, for a frontend built over the package's own component.
    #[must_use]
    pub fn frontend_component(&self) -> &[u8] {
        &self.frontend_component
    }
}

/// Reads one deployment out of the bundle this process selected, served by
/// its loaded module's frontend export.
pub fn deployment_from_bundle_v1(
    host: &InstalledBundleHostV1,
    config: ServerConfig,
) -> Result<BundleDeploymentV1, BundleDeploymentErrorV1> {
    let frontend = frontend_export_v1(host.manifest(), host.library())?;
    deployment_from_package_v1(host.package(), &frontend, config)
}

/// The chat frontend a package's opened native module exports.
///
/// The package must require `chop.frontend.v1` and its module must export
/// the frontend table; without either it is refused. The export holds the
/// module, so the module stays mapped while any frontend opened from it is
/// in use.
pub fn frontend_export_v1(
    manifest: &BundleManifestV1,
    library: &BundleLibraryV1,
) -> Result<FrontendExportV1, BundleDeploymentErrorV1> {
    require_frontend_capability_v1(manifest)?;
    exported_frontend_v1(library.frontend_v1())
}

/// The frontend table's export name, for refusals.
const FRONTEND_SYMBOL_V1: &str = "chop_bundle_frontend_api_v1";

/// What a module answered for its frontend export, or why it cannot serve.
fn exported_frontend_v1(
    answer: Result<Option<FrontendExportV1>, runtime_bundle_abi::BundleAbiError>,
) -> Result<FrontendExportV1, BundleDeploymentErrorV1> {
    answer
        .map_err(|error| {
            BundleDeploymentErrorV1::Frontend(format!(
                "the package's native module exports an invalid {FRONTEND_SYMBOL_V1} table: \
                 {error}"
            ))
        })?
        .ok_or_else(|| {
            BundleDeploymentErrorV1::Frontend(format!(
                "the package requires {FRONTEND_CAPABILITY_V1} but its native module does not \
                 export {FRONTEND_SYMBOL_V1}"
            ))
        })
}

/// Refuses a package that declares no frontend export, before any of its
/// native code runs.
pub(crate) fn require_frontend_capability_v1(
    manifest: &BundleManifestV1,
) -> Result<(), BundleDeploymentErrorV1> {
    let declared = manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| capability.as_str() == FRONTEND_CAPABILITY_V1);
    if declared {
        Ok(())
    } else {
        Err(BundleDeploymentErrorV1::Frontend(format!(
            "the package does not require {FRONTEND_CAPABILITY_V1}, so it serves no chat \
             frontend; rebuild it with a compiler that exports one"
        )))
    }
}

/// Reads one deployment out of a verified package, served by `frontend`,
/// the export of the package's own native module.
///
/// This is the whole of it: three components by role, one declaration, a
/// tokenizer built from bytes that were authenticated twice, and the
/// package's frontend opened over the same declaration bytes.
pub fn deployment_from_package_v1(
    package: &BundlePackageV1,
    frontend: &FrontendExportV1,
    config: ServerConfig,
) -> Result<BundleDeploymentV1, BundleDeploymentErrorV1> {
    require_frontend_capability_v1(package.manifest())?;
    let component = |role: &str| {
        verified_component_by_role_v1(package, role).map_err(BundleDeploymentErrorV1::Component)
    };
    let frontend_component: Arc<[u8]> = component(FRONTEND_COMPONENT_ROLE_V1)?.into();
    let declaration = FrontendDescriptorV1::from_component_bytes(&frontend_component)
        .map_err(BundleDeploymentErrorV1::Declaration)?;
    let tokenizer = tokenizer_from_components_v1(
        &declaration,
        &component(TOKENIZER_DEFINITION_ROLE_V1)?,
        &component(TOKENIZER_CONFIGURATION_ROLE_V1)?,
    )
    .map_err(BundleDeploymentErrorV1::Tokenizer)?;
    // A completion without a rendered chat parses through the package's
    // frontend, as the transport's frontend parses a chat. Both open the
    // export over the package's own component bytes.
    let policy = EngineModelPolicyV1::from_declaration(&declaration).with_parser_source(
        BundleParserSourceV1::from_export(frontend, &frontend_component)
            .map_err(BundleDeploymentErrorV1::Declaration)?,
    );
    let frontend = frontend_from_component_v1(&frontend_component, frontend, config)?;
    Ok(BundleDeploymentV1 {
        frontend,
        declaration,
        frontend_component,
        tokenizer,
        policy,
    })
}

/// Reads the frontend declaration one package ships, by the role it serves.
///
/// This is the only place a declaration is read out of a package, so a caller
/// that wants only the served identities takes the same path as a caller that
/// wants a whole deployment.
pub fn declaration_from_package_v1(
    package: &BundlePackageV1,
) -> Result<FrontendDescriptorV1, BundleDeploymentErrorV1> {
    let bytes = verified_component_by_role_v1(package, FRONTEND_COMPONENT_ROLE_V1)
        .map_err(BundleDeploymentErrorV1::Component)?;
    FrontendDescriptorV1::from_component_bytes(&bytes).map_err(BundleDeploymentErrorV1::Declaration)
}

/// Builds the frontend the transport serves from one authenticated frontend
/// component, whose bytes the package's frontend export opens over.
///
/// A deployment may serve less context than the bundle declared, never more.
pub fn frontend_from_component_v1(
    component: &[u8],
    export: &FrontendExportV1,
    config: ServerConfig,
) -> Result<Arc<dyn ServerFrontendV1>, BundleDeploymentErrorV1> {
    let frontend = BundleFrontendV1::from_export(export, component)
        .map_err(BundleDeploymentErrorV1::Declaration)?;
    Ok(Arc::new(match config.maximum_context_tokens() {
        Some(maximum) => frontend.with_maximum_total_tokens(maximum),
        None => frontend,
    }))
}

/// Builds the frontend from one declaration assembled in code, as tests do;
/// `export` opens over the declaration written back as JSON. A deployment
/// serves [`frontend_from_component_v1`] instead.
///
/// A deployment may serve less context than the bundle declared, never more.
pub fn frontend_from_declaration_v1(
    declaration: &FrontendDescriptorV1,
    export: &FrontendExportV1,
    config: ServerConfig,
) -> Result<Arc<dyn ServerFrontendV1>, BundleDeploymentErrorV1> {
    let frontend = BundleFrontendV1::with_export(declaration.clone(), export)
        .map_err(BundleDeploymentErrorV1::Declaration)?;
    Ok(Arc::new(match config.maximum_context_tokens() {
        Some(maximum) => frontend.with_maximum_total_tokens(maximum),
        None => frontend,
    }))
}

/// Applies `--max-images-per-request`: the most image parts one request may
/// carry, above or below the package's declared count, since images stream
/// through the media arena chunk by chunk. A package that declares no image
/// modality refuses the option rather than ignoring it.
pub fn configure_served_images_v1(
    frontend: BundleFrontendV1,
    maximum: Option<usize>,
) -> Result<BundleFrontendV1, String> {
    let Some(maximum) = maximum else {
        return Ok(frontend);
    };
    let declares_images = frontend
        .descriptor()
        .multimodal
        .iter()
        .flat_map(|multimodal| &multimodal.modalities)
        .any(|modality| modality.kind == runtime_frontend::FrontendMediaKindV1::Image);
    if !declares_images {
        return Err("--max-images-per-request needs a package that accepts images".to_owned());
    }
    Ok(frontend.with_maximum_images(maximum))
}

/// Body bytes allowed for each image a media request may carry: base64 of a
/// 3 MiB file, which holds a PNG screenshot up to 1440p or a JPEG photo of
/// about 12 megapixels. The declared per-image maximum (64 MiB) is far above
/// what clients send, and 64 of them would make a 5 GiB body.
const IMAGE_BODY_BYTES_V1: usize = 4 << 20;

/// The media endpoints' default body limit: the text default, plus
/// `IMAGE_BODY_BYTES_V1` for each of `images`, plus each other served kind's
/// declared items at their declared size in base64, up to the largest body
/// any endpoint accepts.
fn media_body_bytes(images: usize, other: impl IntoIterator<Item = (u16, usize)>) -> usize {
    other
        .into_iter()
        .map(|(items, bytes)| {
            usize::from(items).saturating_mul(bytes.div_ceil(3).saturating_mul(4))
        })
        .fold(
            runtime_server::DEFAULT_REQUEST_BODY_BYTES
                .saturating_add(images.saturating_mul(IMAGE_BODY_BYTES_V1)),
            usize::saturating_add,
        )
        .min(runtime_server::MAX_REQUEST_BODY_BYTES)
}

/// The body limit of chat completions and `/tokenize`, the endpoints that
/// carry media, sized from what `frontend` serves: the images it accepts per
/// request and the video and audio items it serves.
pub fn media_body_bytes_v1(frontend: &BundleFrontendV1) -> usize {
    use runtime_frontend::FrontendMediaKindV1;
    use runtime_server::FrontendMediaPartKindV1;
    let served = frontend.served_media();
    let other = frontend
        .descriptor()
        .multimodal
        .iter()
        .flat_map(|multimodal| &multimodal.modalities)
        .filter(|modality| match modality.kind {
            FrontendMediaKindV1::Image => false,
            FrontendMediaKindV1::Video => served.contains(&FrontendMediaPartKindV1::Video),
            FrontendMediaKindV1::Audio => served.contains(&FrontendMediaPartKindV1::Audio),
        })
        .map(|modality| (modality.maximum_items, modality.maximum_bytes));
    media_body_bytes(frontend.maximum_images_per_request().unwrap_or(0), other)
}

/// Sizes the media endpoints' body limit from what `frontend` serves, unless
/// the deployment set a body limit (`--max-request-body-bytes`), which then
/// applies to every endpoint as it is.
pub fn configure_media_body_limit_v1(
    config: ServerConfig,
    frontend: &BundleFrontendV1,
    explicit: bool,
) -> Result<ServerConfig, String> {
    if explicit {
        return Ok(config);
    }
    config
        .with_max_media_body_bytes(media_body_bytes_v1(frontend))
        .map_err(|error| format!("the media request body limit is unusable: {error}"))
}

/// Applies one remote-media policy to both halves of the serving path.
pub fn configure_remote_media_v1(
    frontend: BundleFrontendV1,
    backend: crate::EngineGenerationBackend,
    policy: &runtime_media::remote::RemoteMediaPolicy,
) -> Result<(BundleFrontendV1, crate::EngineGenerationBackend), runtime_media::MediaError> {
    policy.validate()?;
    Ok((
        frontend.with_remote_media(!policy.allowed_domains.is_empty()),
        backend.with_remote_media(policy.clone()),
    ))
}

#[cfg(test)]
mod tests {
    use super::exported_frontend_v1;

    /// Text only keeps the text default; each accepted image adds 4 MiB, and
    /// a served video or audio item its declared size in base64.
    #[test]
    fn media_body_limits_grow_with_the_media_a_request_may_carry() {
        const MIB: usize = 1 << 20;
        assert_eq!(super::media_body_bytes(0, []), 64 * MIB);
        assert_eq!(super::media_body_bytes(64, []), 320 * MIB);
        // Every modality: 64 images, one video and one audio item of 64 MiB
        // each.
        let every = super::media_body_bytes(64, [(1, 64 * MIB), (1, 64 * MIB)]);
        assert_eq!(every, 320 * MIB + 2 * (64 * MIB).div_ceil(3) * 4);
        assert_eq!(every / MIB, 490);
        assert_eq!(
            super::media_body_bytes(1000, []),
            runtime_server::MAX_REQUEST_BODY_BYTES
        );
    }

    /// A module without the export, or with a table the loader refused,
    /// serves no frontend: there is no built-in format to fall back to.
    #[test]
    fn a_module_without_a_frontend_export_is_refused() {
        let missing = exported_frontend_v1(Ok(None)).err().unwrap().to_string();
        assert_eq!(
            missing,
            "the package requires chop.frontend.v1 but its native module does not export \
             chop_bundle_frontend_api_v1"
        );
        let invalid = exported_frontend_v1(Err(
            runtime_bundle_abi::BundleAbiError::MissingApiCallback("frontend render"),
        ))
        .err()
        .unwrap()
        .to_string();
        assert!(
            invalid.starts_with(
                "the package's native module exports an invalid chop_bundle_frontend_api_v1 table"
            ),
            "{invalid}"
        );
        assert!(
            exported_frontend_v1(Ok(
                Some(runtime_frontend::test_support::fixture_export_v1())
            ))
            .is_ok()
        );
    }
}
