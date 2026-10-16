//! Whole components a deployment reads as data before it loads a package.
//!
//! A package carries its model's serving profiles, from which a bare
//! `serve --bundle DIR --profile NAME` resolves NAME, and its deployment
//! policy, against which the container entrypoint validates its
//! environment. The manifest authenticates each like any other component.

/// The serving profiles: the model's `chopinfer-serve-profiles-v2` file.
pub const SERVE_PROFILES_COMPONENT_V1: &str = "serve-profiles";
pub const SERVE_PROFILES_ROLE_V1: &str = "chop.serve-profiles.v1";
pub const SERVE_PROFILES_FORMAT_V1: &str = "chop.serve-profiles.json.v2";
pub const SERVE_PROFILES_PATH_V1: &str = "components/serve-profiles.json";

/// The deployment policy: the model's `chopinfer-deployment-policy-v1` file.
pub const DEPLOYMENT_POLICY_COMPONENT_V1: &str = "deployment-policy";
pub const DEPLOYMENT_POLICY_ROLE_V1: &str = "chop.deployment-policy.v1";
pub const DEPLOYMENT_POLICY_FORMAT_V1: &str = "chop.deployment-policy.json.v1";
pub const DEPLOYMENT_POLICY_PATH_V1: &str = "components/deployment-policy.json";

/// The most bytes a deployment component may hold: a small JSON file read
/// whole before anything loads.
pub const DEPLOYMENT_COMPONENT_MAX_BYTES: u64 = 256 * 1024;

/// Whether `role` is one of these components, which a deployment reads and
/// the bundle's module never opens.
#[must_use]
pub fn is_deployment_role(role: &str) -> bool {
    role == SERVE_PROFILES_ROLE_V1 || role == DEPLOYMENT_POLICY_ROLE_V1
}
