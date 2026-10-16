//! Model-neutral rank index parsing shared with package admission.
pub use runtime_bundle_sdk::rank_index::*;

pub mod encoder;

/// Which compiled image a rank index describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageKind {
    /// The text rank image.
    Text,
    /// The encoder image.
    Encoder,
    /// The drafter image.
    Drafter,
}

/// The authenticated bundle-metadata role of rank `rank`'s index of a
/// `kind` image: `chop.bundle-metadata.<model>.<kind>rank-index.r<rank>.v1`,
/// with `model` the model's role token and `<kind>` empty for the text image
/// or `encoder-`/`drafter-`.
pub fn index_role(model: &str, kind: ImageKind, rank: u32) -> String {
    let kind = match kind {
        ImageKind::Text => "",
        ImageKind::Encoder => "encoder-",
        ImageKind::Drafter => "drafter-",
    };
    format!("chop.bundle-metadata.{model}.{kind}rank-index.r{rank}.v1")
}

/// Binds a rank index to the rank the host serves: the physical rank file's
/// SHA-256 (`sha256`), or, for a rank planned from a checkpoint, the plan's
/// identity (`direct_identity`). The index names exactly one, and the host's
/// identity must name the same kind.
pub fn authenticate_rank_identity(
    services: &runtime_bundle_abi::HostServicesV1,
    image_role: &str,
    rank: u32,
    image_bytes: u64,
    sha256: Option<&str>,
    direct_identity: Option<&str>,
) -> Result<(), runtime_bundle_abi::BundleAbiError> {
    if sha256.is_some() == direct_identity.is_some() {
        return Err(runtime_bundle_abi::BundleAbiError::InvalidRange(
            "rank index needs one identity",
        ));
    }
    let role = format!("chop.compiled-image-identity.v1/{image_role}/{rank}");
    let component = services.open_authenticated_component(role.as_bytes())?;
    let result = (|| {
        let metadata = services.inspect_authenticated_component(component)?;
        if metadata.role_length != role.len() as u64 || metadata.byte_length > 1024 {
            return Err(runtime_bundle_abi::BundleAbiError::InvalidRange(
                "rank identity metadata",
            ));
        }
        let mut role_bytes = vec![0; role.len()];
        let mut bytes = vec![0; metadata.byte_length as usize];
        services.read_authenticated_component(component, &mut role_bytes, &mut bytes)?;
        let identity: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|_| runtime_bundle_abi::BundleAbiError::InvalidRange("rank identity JSON"))?;
        if role_bytes != role.as_bytes()
            || identity["byte_length"].as_u64() != Some(image_bytes)
            || identity["sha256"].as_str() != sha256
            || identity["direct_identity"].as_str() != direct_identity
        {
            return Err(runtime_bundle_abi::BundleAbiError::InvalidRange(
                "rank image identity mismatch",
            ));
        }
        Ok(())
    })();
    services.release_authenticated_component(component)?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roles_name_the_model_image_kind_and_rank() {
        assert_eq!(
            index_role("m", ImageKind::Text, 3),
            "chop.bundle-metadata.m.rank-index.r3.v1"
        );
        assert_eq!(
            index_role("m", ImageKind::Encoder, 0),
            "chop.bundle-metadata.m.encoder-rank-index.r0.v1"
        );
        assert_eq!(
            index_role("m", ImageKind::Drafter, 1),
            "chop.bundle-metadata.m.drafter-rank-index.r1.v1"
        );
    }
}
