//! The semantic half of a kernel contract.
//!
//! A semantic key states what an operation must compute and how its operands
//! are shaped. It carries no artifact, symbol, launch, source, build, or
//! provider identity: those belong to `crate::implementation`.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::artifact::KernelOpaqueIdV1;
use crate::contract::{
    KernelAliasContractV1, KernelScaleContractV1, KernelTensorAccessV1, KernelTensorDependencyV1,
};

/// The largest number of tensor operands one semantic key may declare.
pub const MAX_SEMANTIC_TENSORS: usize = 64;
/// The largest number of scalar arguments one semantic key may declare.
pub const MAX_SEMANTIC_SCALARS: usize = 64;
/// The largest number of device capabilities one semantic key may require.
pub const MAX_SEMANTIC_CAPABILITIES: usize = 64;
/// The largest tensor rank one semantic key may declare.
pub const MAX_SEMANTIC_DIMENSIONS: usize = 8;

/// The exact operation ABI a semantic key speaks.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelOperationAbiV1 {
    id: KernelOpaqueIdV1,
    version: u32,
}

impl KernelOperationAbiV1 {
    pub fn new(id: KernelOpaqueIdV1, version: u32) -> Result<Self, KernelSemanticErrorV1> {
        if version == 0 {
            return Err(KernelSemanticErrorV1::ZeroOperationAbiVersion);
        }
        Ok(Self { id, version })
    }

    #[must_use]
    pub const fn id(&self) -> &KernelOpaqueIdV1 {
        &self.id
    }

    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }
}

/// One tensor operand requirement: dimensions, dtype, layout, scale, alignment.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelTensorRequirementV1 {
    role_id: KernelOpaqueIdV1,
    dimensions: Vec<u64>,
    dtype_id: KernelOpaqueIdV1,
    layout_id: KernelOpaqueIdV1,
    scale: KernelScaleContractV1,
    alignment_bytes: u64,
    access: KernelTensorAccessV1,
    alias: KernelAliasContractV1,
    dependencies: Vec<KernelTensorDependencyV1>,
}

impl KernelTensorRequirementV1 {
    pub fn new(
        role_id: KernelOpaqueIdV1,
        dimensions: Vec<u64>,
        dtype_id: KernelOpaqueIdV1,
        layout_id: KernelOpaqueIdV1,
        scale_id: KernelOpaqueIdV1,
        alignment_bytes: u64,
    ) -> Result<Self, KernelSemanticErrorV1> {
        if dimensions.is_empty() || dimensions.len() > MAX_SEMANTIC_DIMENSIONS {
            return Err(KernelSemanticErrorV1::TensorRank {
                role_id: role_id.clone(),
                rank: dimensions.len(),
            });
        }
        if dimensions.contains(&0) {
            return Err(KernelSemanticErrorV1::ZeroTensorDimension {
                role_id: role_id.clone(),
            });
        }
        if alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(KernelSemanticErrorV1::InvalidAlignment {
                role_id: role_id.clone(),
                alignment_bytes,
            });
        }
        let scale = KernelScaleContractV1::new(scale_id, Vec::new()).map_err(|_| {
            KernelSemanticErrorV1::InvalidScaleContract {
                role_id: role_id.clone(),
            }
        })?;
        Ok(Self {
            role_id,
            dimensions,
            dtype_id,
            layout_id,
            scale,
            alignment_bytes,
            access: KernelTensorAccessV1::ReadOnly,
            alias: KernelAliasContractV1::Disjoint,
            dependencies: Vec::new(),
        })
    }

    /// Declares the block geometry of a blocked scale format.
    pub fn with_scale_block_dimensions(
        mut self,
        block_dimensions: Vec<u64>,
    ) -> Result<Self, KernelSemanticErrorV1> {
        self.scale = KernelScaleContractV1::new(self.scale.format_id().clone(), block_dimensions)
            .map_err(|_| KernelSemanticErrorV1::InvalidScaleContract {
            role_id: self.role_id.clone(),
        })?;
        Ok(self)
    }

    /// Declares how the operation reads, writes, and aliases this operand.
    #[must_use]
    pub fn with_access(
        mut self,
        access: KernelTensorAccessV1,
        alias: KernelAliasContractV1,
    ) -> Self {
        self.access = access;
        self.alias = alias;
        self
    }

    /// Declares the earlier operands this one is ordered against.
    #[must_use]
    pub fn with_dependencies(mut self, dependencies: Vec<KernelTensorDependencyV1>) -> Self {
        self.dependencies = dependencies;
        self
    }

    #[must_use]
    pub const fn role_id(&self) -> &KernelOpaqueIdV1 {
        &self.role_id
    }

    #[must_use]
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    #[must_use]
    pub const fn dtype_id(&self) -> &KernelOpaqueIdV1 {
        &self.dtype_id
    }

    #[must_use]
    pub const fn layout_id(&self) -> &KernelOpaqueIdV1 {
        &self.layout_id
    }

    #[must_use]
    pub const fn scale(&self) -> &KernelScaleContractV1 {
        &self.scale
    }

    #[must_use]
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    #[must_use]
    pub const fn access(&self) -> KernelTensorAccessV1 {
        self.access
    }

    #[must_use]
    pub const fn alias(&self) -> &KernelAliasContractV1 {
        &self.alias
    }

    #[must_use]
    pub fn dependencies(&self) -> &[KernelTensorDependencyV1] {
        &self.dependencies
    }
}

/// One scalar argument in the operation ABI.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelScalarArgumentV1 {
    argument_id: KernelOpaqueIdV1,
    dtype_id: KernelOpaqueIdV1,
    width_bits: u16,
}

impl KernelScalarArgumentV1 {
    pub fn new(
        argument_id: KernelOpaqueIdV1,
        dtype_id: KernelOpaqueIdV1,
        width_bits: u16,
    ) -> Result<Self, KernelSemanticErrorV1> {
        if width_bits == 0 || !width_bits.is_multiple_of(8) {
            return Err(KernelSemanticErrorV1::InvalidScalarWidth {
                argument_id: argument_id.clone(),
                width_bits,
            });
        }
        Ok(Self {
            argument_id,
            dtype_id,
            width_bits,
        })
    }

    #[must_use]
    pub const fn argument_id(&self) -> &KernelOpaqueIdV1 {
        &self.argument_id
    }

    #[must_use]
    pub const fn dtype_id(&self) -> &KernelOpaqueIdV1 {
        &self.dtype_id
    }

    #[must_use]
    pub const fn width_bits(&self) -> u16 {
        self.width_bits
    }
}

/// The open form of a semantic key, used to build and to inspect one.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSemanticKeyPartsV1 {
    pub operation_abi: KernelOperationAbiV1,
    pub tensors: Vec<KernelTensorRequirementV1>,
    pub scalar_arguments: Vec<KernelScalarArgumentV1>,
    pub numerical_contract: KernelOpaqueIdV1,
    pub required_capabilities: BTreeSet<KernelOpaqueIdV1>,
}

/// What an operation must compute, independent of who implements it.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSemanticKeyV1 {
    operation_abi: KernelOperationAbiV1,
    tensors: Vec<KernelTensorRequirementV1>,
    scalar_arguments: Vec<KernelScalarArgumentV1>,
    numerical_contract: KernelOpaqueIdV1,
    required_capabilities: BTreeSet<KernelOpaqueIdV1>,
}

impl KernelSemanticKeyV1 {
    pub fn new(parts: KernelSemanticKeyPartsV1) -> Result<Self, KernelSemanticErrorV1> {
        if parts.tensors.is_empty() || parts.tensors.len() > MAX_SEMANTIC_TENSORS {
            return Err(KernelSemanticErrorV1::TensorCount {
                count: parts.tensors.len(),
            });
        }
        if parts.scalar_arguments.len() > MAX_SEMANTIC_SCALARS {
            return Err(KernelSemanticErrorV1::ScalarCount {
                count: parts.scalar_arguments.len(),
            });
        }
        if parts.required_capabilities.len() > MAX_SEMANTIC_CAPABILITIES {
            return Err(KernelSemanticErrorV1::CapabilityCount {
                count: parts.required_capabilities.len(),
            });
        }
        let mut roles = BTreeSet::new();
        for (index, tensor) in parts.tensors.iter().enumerate() {
            if !roles.insert(tensor.role_id()) {
                return Err(KernelSemanticErrorV1::DuplicateTensorRole(
                    tensor.role_id().clone(),
                ));
            }
            // A dependency names an earlier operand, so the ordering is a
            // strict prefix and no cycle can be declared.
            for dependency in tensor.dependencies() {
                if usize::from(dependency.tensor_index()) >= index {
                    return Err(KernelSemanticErrorV1::InvalidTensorDependency {
                        tensor_index: index,
                        depends_on: dependency.tensor_index(),
                    });
                }
            }
        }
        let mut argument_ids = BTreeSet::new();
        for argument in &parts.scalar_arguments {
            if !argument_ids.insert(argument.argument_id()) {
                return Err(KernelSemanticErrorV1::DuplicateScalarArgument(
                    argument.argument_id().clone(),
                ));
            }
        }
        Ok(Self {
            operation_abi: parts.operation_abi,
            tensors: parts.tensors,
            scalar_arguments: parts.scalar_arguments,
            numerical_contract: parts.numerical_contract,
            required_capabilities: parts.required_capabilities,
        })
    }

    #[must_use]
    pub const fn operation_abi(&self) -> &KernelOperationAbiV1 {
        &self.operation_abi
    }

    #[must_use]
    pub fn tensors(&self) -> &[KernelTensorRequirementV1] {
        &self.tensors
    }

    #[must_use]
    pub fn scalar_arguments(&self) -> &[KernelScalarArgumentV1] {
        &self.scalar_arguments
    }

    #[must_use]
    pub const fn numerical_contract(&self) -> &KernelOpaqueIdV1 {
        &self.numerical_contract
    }

    #[must_use]
    pub const fn required_capabilities(&self) -> &BTreeSet<KernelOpaqueIdV1> {
        &self.required_capabilities
    }

    #[must_use]
    pub fn parts(&self) -> KernelSemanticKeyPartsV1 {
        KernelSemanticKeyPartsV1 {
            operation_abi: self.operation_abi.clone(),
            tensors: self.tensors.clone(),
            scalar_arguments: self.scalar_arguments.clone(),
            numerical_contract: self.numerical_contract.clone(),
            required_capabilities: self.required_capabilities.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelSemanticErrorV1 {
    ZeroOperationAbiVersion,
    TensorRank {
        role_id: KernelOpaqueIdV1,
        rank: usize,
    },
    ZeroTensorDimension {
        role_id: KernelOpaqueIdV1,
    },
    InvalidAlignment {
        role_id: KernelOpaqueIdV1,
        alignment_bytes: u64,
    },
    InvalidScaleContract {
        role_id: KernelOpaqueIdV1,
    },
    InvalidScalarWidth {
        argument_id: KernelOpaqueIdV1,
        width_bits: u16,
    },
    TensorCount {
        count: usize,
    },
    ScalarCount {
        count: usize,
    },
    CapabilityCount {
        count: usize,
    },
    DuplicateTensorRole(KernelOpaqueIdV1),
    DuplicateScalarArgument(KernelOpaqueIdV1),
    InvalidTensorDependency {
        tensor_index: usize,
        depends_on: u16,
    },
}

impl fmt::Display for KernelSemanticErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroOperationAbiVersion => {
                formatter.write_str("kernel operation ABI version is zero")
            }
            Self::TensorRank { role_id, rank } => write!(
                formatter,
                "tensor {role_id} rank {rank} is outside 1..={MAX_SEMANTIC_DIMENSIONS}"
            ),
            Self::ZeroTensorDimension { role_id } => {
                write!(formatter, "tensor {role_id} has a zero dimension")
            }
            Self::InvalidAlignment {
                role_id,
                alignment_bytes,
            } => write!(
                formatter,
                "tensor {role_id} alignment {alignment_bytes} is not a nonzero power of two"
            ),
            Self::InvalidScaleContract { role_id } => {
                write!(formatter, "tensor {role_id} has an invalid scale contract")
            }
            Self::InvalidScalarWidth {
                argument_id,
                width_bits,
            } => write!(
                formatter,
                "scalar {argument_id} width {width_bits} is not whole nonzero bytes"
            ),
            Self::TensorCount { count } => write!(
                formatter,
                "semantic key declares {count} tensors, expected 1..={MAX_SEMANTIC_TENSORS}"
            ),
            Self::ScalarCount { count } => write!(
                formatter,
                "semantic key declares {count} scalars, expected at most {MAX_SEMANTIC_SCALARS}"
            ),
            Self::CapabilityCount { count } => write!(
                formatter,
                "semantic key requires {count} capabilities, expected at most {MAX_SEMANTIC_CAPABILITIES}"
            ),
            Self::DuplicateTensorRole(role_id) => {
                write!(formatter, "semantic key repeats tensor role {role_id}")
            }
            Self::DuplicateScalarArgument(argument_id) => {
                write!(
                    formatter,
                    "semantic key repeats scalar argument {argument_id}"
                )
            }
            Self::InvalidTensorDependency {
                tensor_index,
                depends_on,
            } => write!(
                formatter,
                "tensor {tensor_index} depends on operand {depends_on}, which is not earlier"
            ),
        }
    }
}

impl Error for KernelSemanticErrorV1 {}
