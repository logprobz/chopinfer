//! Contract fragments shared by semantic keys and implementation details.

use std::error::Error;
use std::fmt;

use crate::artifact::KernelOpaqueIdV1;

/// The scale format one operand carries, with its block geometry.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelScaleContractV1 {
    format_id: KernelOpaqueIdV1,
    block_dimensions: Vec<u64>,
}

impl KernelScaleContractV1 {
    pub fn new(
        format_id: KernelOpaqueIdV1,
        block_dimensions: Vec<u64>,
    ) -> Result<Self, KernelContractErrorV1> {
        if block_dimensions.contains(&0) {
            return Err(KernelContractErrorV1::ZeroDimension("scale block"));
        }
        Ok(Self {
            format_id,
            block_dimensions,
        })
    }

    #[must_use]
    pub const fn format_id(&self) -> &KernelOpaqueIdV1 {
        &self.format_id
    }

    #[must_use]
    pub fn block_dimensions(&self) -> &[u64] {
        &self.block_dimensions
    }
}

/// How an operation touches one operand.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelTensorAccessV1 {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Whether an operand may share storage with another.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelAliasContractV1 {
    Disjoint,
    AliasGroup(KernelOpaqueIdV1),
}

/// An ordering constraint from one operand onto an earlier one.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelTensorDependencyV1 {
    tensor_index: u16,
    dependency_id: KernelOpaqueIdV1,
}

impl KernelTensorDependencyV1 {
    #[must_use]
    pub const fn new(tensor_index: u16, dependency_id: KernelOpaqueIdV1) -> Self {
        Self {
            tensor_index,
            dependency_id,
        }
    }

    #[must_use]
    pub const fn tensor_index(&self) -> u16 {
        self.tensor_index
    }

    #[must_use]
    pub const fn dependency_id(&self) -> &KernelOpaqueIdV1 {
        &self.dependency_id
    }
}

/// The exact launch geometry of one implementation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelLaunchContractV1 {
    grid: [u32; 3],
    block: [u32; 3],
    dynamic_shared_memory_bytes: u32,
}

impl KernelLaunchContractV1 {
    pub fn new(
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_shared_memory_bytes: u32,
    ) -> Result<Self, KernelContractErrorV1> {
        if grid.contains(&0) {
            return Err(KernelContractErrorV1::ZeroLaunchDimension("grid"));
        }
        if block.contains(&0) {
            return Err(KernelContractErrorV1::ZeroLaunchDimension("block"));
        }
        Ok(Self {
            grid,
            block,
            dynamic_shared_memory_bytes,
        })
    }

    #[must_use]
    pub const fn grid(&self) -> [u32; 3] {
        self.grid
    }

    #[must_use]
    pub const fn block(&self) -> [u32; 3] {
        self.block
    }

    #[must_use]
    pub const fn dynamic_shared_memory_bytes(&self) -> u32 {
        self.dynamic_shared_memory_bytes
    }
}

/// The scratch storage one implementation needs.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelWorkspaceContractV1 {
    byte_length: u64,
    alignment_bytes: u64,
    layout_id: KernelOpaqueIdV1,
    access: KernelTensorAccessV1,
    alias: KernelAliasContractV1,
}

impl KernelWorkspaceContractV1 {
    pub fn new(
        byte_length: u64,
        alignment_bytes: u64,
        layout_id: KernelOpaqueIdV1,
        access: KernelTensorAccessV1,
        alias: KernelAliasContractV1,
    ) -> Result<Self, KernelContractErrorV1> {
        if byte_length == 0 {
            return Err(KernelContractErrorV1::ZeroWorkspaceBytes);
        }
        validate_alignment(alignment_bytes, "workspace")?;
        Ok(Self {
            byte_length,
            alignment_bytes,
            layout_id,
            access,
            alias,
        })
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    #[must_use]
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    #[must_use]
    pub const fn layout_id(&self) -> &KernelOpaqueIdV1 {
        &self.layout_id
    }

    #[must_use]
    pub const fn access(&self) -> KernelTensorAccessV1 {
        self.access
    }

    #[must_use]
    pub const fn alias(&self) -> &KernelAliasContractV1 {
        &self.alias
    }
}

/// One compile-time choice baked into an implementation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSpecializationV1 {
    id: KernelOpaqueIdV1,
    value: KernelOpaqueIdV1,
}

impl KernelSpecializationV1 {
    #[must_use]
    pub const fn new(id: KernelOpaqueIdV1, value: KernelOpaqueIdV1) -> Self {
        Self { id, value }
    }

    #[must_use]
    pub const fn id(&self) -> &KernelOpaqueIdV1 {
        &self.id
    }

    #[must_use]
    pub const fn value(&self) -> &KernelOpaqueIdV1 {
        &self.value
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelContractErrorV1 {
    ZeroDimension(&'static str),
    InvalidAlignment { kind: &'static str, value: u64 },
    ZeroLaunchDimension(&'static str),
    ZeroWorkspaceBytes,
}

impl fmt::Display for KernelContractErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(kind) => write!(formatter, "{kind} has a zero dimension"),
            Self::InvalidAlignment { kind, value } => {
                write!(
                    formatter,
                    "{kind} alignment {value} is not a nonzero power of two"
                )
            }
            Self::ZeroLaunchDimension(kind) => {
                write!(formatter, "kernel launch {kind} has a zero dimension")
            }
            Self::ZeroWorkspaceBytes => formatter.write_str("kernel workspace has zero bytes"),
        }
    }
}

impl Error for KernelContractErrorV1 {}

fn validate_alignment(value: u64, kind: &'static str) -> Result<(), KernelContractErrorV1> {
    if value == 0 || !value.is_power_of_two() {
        return Err(KernelContractErrorV1::InvalidAlignment { kind, value });
    }
    Ok(())
}
