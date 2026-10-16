use std::error::Error;
use std::fmt;

const MAX_OPAQUE_ID_BYTES: usize = 160;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelOpaqueIdV1(String);

impl KernelOpaqueIdV1 {
    pub fn new(value: impl Into<String>) -> Result<Self, KernelIdentityErrorV1> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_OPAQUE_ID_BYTES
            || !value.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err(KernelIdentityErrorV1::InvalidOpaqueId(value));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelOpaqueIdV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSha256V1(String);

impl KernelSha256V1 {
    pub fn new(value: impl Into<String>) -> Result<Self, KernelIdentityErrorV1> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(KernelIdentityErrorV1::InvalidSha256(value));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelSha256V1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelArtifactIdentityV1 {
    format_id: KernelOpaqueIdV1,
    byte_length: u64,
    sha256: KernelSha256V1,
}

impl KernelArtifactIdentityV1 {
    pub fn new(
        format_id: KernelOpaqueIdV1,
        byte_length: u64,
        sha256: KernelSha256V1,
    ) -> Result<Self, KernelIdentityErrorV1> {
        if byte_length == 0 {
            return Err(KernelIdentityErrorV1::ZeroByteLength("artifact"));
        }
        Ok(Self {
            format_id,
            byte_length,
            sha256,
        })
    }

    #[must_use]
    pub const fn format_id(&self) -> &KernelOpaqueIdV1 {
        &self.format_id
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    #[must_use]
    pub const fn sha256(&self) -> &KernelSha256V1 {
        &self.sha256
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSourceIdentityV1 {
    source_id: KernelOpaqueIdV1,
    revision: KernelOpaqueIdV1,
    path: KernelOpaqueIdV1,
    byte_length: u64,
    sha256: KernelSha256V1,
}

impl KernelSourceIdentityV1 {
    pub fn new(
        source_id: KernelOpaqueIdV1,
        revision: KernelOpaqueIdV1,
        path: KernelOpaqueIdV1,
        byte_length: u64,
        sha256: KernelSha256V1,
    ) -> Result<Self, KernelIdentityErrorV1> {
        if byte_length == 0 {
            return Err(KernelIdentityErrorV1::ZeroByteLength("source"));
        }
        Ok(Self {
            source_id,
            revision,
            path,
            byte_length,
            sha256,
        })
    }

    #[must_use]
    pub const fn source_id(&self) -> &KernelOpaqueIdV1 {
        &self.source_id
    }

    #[must_use]
    pub const fn revision(&self) -> &KernelOpaqueIdV1 {
        &self.revision
    }

    #[must_use]
    pub const fn path(&self) -> &KernelOpaqueIdV1 {
        &self.path
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    #[must_use]
    pub const fn sha256(&self) -> &KernelSha256V1 {
        &self.sha256
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelBuildIdentityV1 {
    toolchain_id: KernelOpaqueIdV1,
    toolchain_version: KernelOpaqueIdV1,
    build_id: KernelOpaqueIdV1,
    recipe_sha256: KernelSha256V1,
}

impl KernelBuildIdentityV1 {
    #[must_use]
    pub const fn new(
        toolchain_id: KernelOpaqueIdV1,
        toolchain_version: KernelOpaqueIdV1,
        build_id: KernelOpaqueIdV1,
        recipe_sha256: KernelSha256V1,
    ) -> Self {
        Self {
            toolchain_id,
            toolchain_version,
            build_id,
            recipe_sha256,
        }
    }

    #[must_use]
    pub const fn toolchain_id(&self) -> &KernelOpaqueIdV1 {
        &self.toolchain_id
    }

    #[must_use]
    pub const fn toolchain_version(&self) -> &KernelOpaqueIdV1 {
        &self.toolchain_version
    }

    #[must_use]
    pub const fn build_id(&self) -> &KernelOpaqueIdV1 {
        &self.build_id
    }

    #[must_use]
    pub const fn recipe_sha256(&self) -> &KernelSha256V1 {
        &self.recipe_sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelIdentityErrorV1 {
    InvalidOpaqueId(String),
    InvalidSha256(String),
    ZeroByteLength(&'static str),
}

impl fmt::Display for KernelIdentityErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOpaqueId(value) => {
                write!(formatter, "invalid opaque kernel ID: {value}")
            }
            Self::InvalidSha256(value) => {
                write!(formatter, "invalid lowercase SHA-256: {value}")
            }
            Self::ZeroByteLength(kind) => write!(formatter, "{kind} identity has zero bytes"),
        }
    }
}

impl Error for KernelIdentityErrorV1 {}
