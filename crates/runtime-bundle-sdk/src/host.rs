use std::error::Error;
use std::fmt;

use runtime_bundle_abi::{BundleAbiError, BundleStatusV1};

/// A host-service refusal with the service and optional rank retained.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostServiceErrorV1 {
    service: &'static str,
    rank: Option<u32>,
    source: BundleAbiError,
}

impl HostServiceErrorV1 {
    #[must_use]
    pub const fn new(service: &'static str, rank: Option<u32>, source: BundleAbiError) -> Self {
        Self {
            service,
            rank,
            source,
        }
    }

    #[must_use]
    pub const fn service(&self) -> &'static str {
        self.service
    }

    #[must_use]
    pub const fn rank(&self) -> Option<u32> {
        self.rank
    }

    #[must_use]
    pub const fn source_error(&self) -> &BundleAbiError {
        &self.source
    }

    #[must_use]
    pub const fn status(&self) -> BundleStatusV1 {
        BundleStatusV1::HostError
    }
}

impl fmt::Display for HostServiceErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.rank {
            Some(rank) => write!(
                formatter,
                "host service {} failed for rank {rank}: {}",
                self.service, self.source
            ),
            None => write!(
                formatter,
                "host service {} failed: {}",
                self.service, self.source
            ),
        }
    }
}

impl Error for HostServiceErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// One exact package-authenticated component copied into bundle-owned memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedComponentV1 {
    role: Vec<u8>,
    sha256: [u8; 64],
    bytes: Vec<u8>,
}

impl AuthenticatedComponentV1 {
    #[must_use]
    pub fn role(&self) -> &[u8] {
        &self.role
    }

    #[must_use]
    pub const fn sha256(&self) -> &[u8; 64] {
        &self.sha256
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Why an authenticated component could not be copied and released.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthenticatedComponentErrorV1 {
    Host(HostServiceErrorV1),
    Allocation { surface: &'static str },
    RoleMismatch,
    Release(HostServiceErrorV1),
}

impl fmt::Display for AuthenticatedComponentErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host(error) => error.fmt(formatter),
            Self::Allocation { surface } => {
                write!(
                    formatter,
                    "cannot allocate authenticated component {surface}"
                )
            }
            Self::RoleMismatch => formatter.write_str(
                "authenticated component returned a role different from the requested role",
            ),
            Self::Release(error) => write!(
                formatter,
                "authenticated component could not be released: {error}"
            ),
        }
    }
}

impl Error for AuthenticatedComponentErrorV1 {}

/// Opens, copies, verifies the role of, and releases one authenticated component.
pub fn authenticated_component_v1(
    services: &runtime_bundle_abi::HostServicesV1,
    role: &[u8],
) -> Result<AuthenticatedComponentV1, AuthenticatedComponentErrorV1> {
    let handle = host_service_v1(
        "open_authenticated_component",
        None,
        services.open_authenticated_component(role),
    )
    .map_err(AuthenticatedComponentErrorV1::Host)?;
    let copied = (|| {
        let metadata = host_service_v1(
            "inspect_authenticated_component",
            None,
            services.inspect_authenticated_component(handle),
        )
        .map_err(AuthenticatedComponentErrorV1::Host)?;
        let role_length = usize::try_from(metadata.role_length)
            .map_err(|_| AuthenticatedComponentErrorV1::Allocation { surface: "role" })?;
        let byte_length = usize::try_from(metadata.byte_length)
            .map_err(|_| AuthenticatedComponentErrorV1::Allocation { surface: "payload" })?;
        let mut returned_role = Vec::new();
        returned_role
            .try_reserve_exact(role_length)
            .map_err(|_| AuthenticatedComponentErrorV1::Allocation { surface: "role" })?;
        returned_role.resize(role_length, 0);
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_length)
            .map_err(|_| AuthenticatedComponentErrorV1::Allocation { surface: "payload" })?;
        bytes.resize(byte_length, 0);
        host_service_v1(
            "read_authenticated_component",
            None,
            services.read_authenticated_component(handle, &mut returned_role, &mut bytes),
        )
        .map_err(AuthenticatedComponentErrorV1::Host)?;
        if returned_role != role {
            return Err(AuthenticatedComponentErrorV1::RoleMismatch);
        }
        Ok(AuthenticatedComponentV1 {
            role: returned_role,
            sha256: metadata.sha256,
            bytes,
        })
    })();
    let release = host_service_v1(
        "release_authenticated_component",
        None,
        services.release_authenticated_component(handle),
    );
    match (copied, release) {
        (_, Err(error)) => Err(AuthenticatedComponentErrorV1::Release(error)),
        (result, Ok(())) => result,
    }
}

pub fn host_service_v1<T>(
    service: &'static str,
    rank: Option<u32>,
    result: Result<T, BundleAbiError>,
) -> Result<T, HostServiceErrorV1> {
    result.map_err(|source| HostServiceErrorV1::new(service, rank, source))
}
