//! Source operations use declared operands, rank ownership and format recipes.
use super::*;
use kernel_core::precision::format::{A16Recipe, KernelFamily, registry};
use model_compiler_sdk::quantization::import::ImportedOperand;
use model_qwen38_contract::backend::{Kernels, Selected};
use model_qwen38_contract::projections::ports::Ports;
use projection_spec::derive::{Topology, rank_selection};

pub struct Definitions {
    pub ports: &'static Ports,
}

impl Definitions {
    pub fn new(shape: Shape) -> Result<Self> {
        Ok(Self {
            ports: model_qwen38_contract::projections::ports::get(shape)?,
        })
    }

    pub fn projection(&self, id: &str) -> Result<&projection_spec::Projection> {
        Ok(self.ports.projection(id)?)
    }

    /// The checkpoint names of one projection instance's operands.
    pub fn names(
        &self,
        p: &projection_spec::Projection,
        layer: Option<u32>,
    ) -> Result<Vec<String>> {
        Ok(self.ports.names(&p.id, layer.map(u64::from))?)
    }

    pub fn dense_id(role: DenseRole, shape: Shape, layer: u32) -> &'static str {
        role.projection_id(shape, Some(layer))
    }

    pub fn legacy_dense(
        &self,
        source: &SourcePrecision,
        p: &projection_spec::Projection,
        layer: u32,
    ) -> Result<bool> {
        let members = self
            .ports
            .declaration
            .fusion_groups
            .iter()
            .find(|f| f.members.contains(&p.id))
            .map_or_else(
                || vec![p.id.as_str()],
                |f| f.members.iter().map(String::as_str).collect(),
            );
        // Fetch every peer operand before judging the fusion, so a missing
        // peer fails first.
        let mut legacy = true;
        for id in members {
            for name in self.names(self.projection(id)?, Some(layer))? {
                let operand = source
                    .operands
                    .get(&name)
                    .ok_or_else(|| format!("missing dense operand {name}"))?;
                legacy &= operand.weight.format == WeightFormat::Bf16
                    && operand.activation.as_ref().is_some_and(|a| a.is_bf16());
            }
        }
        Ok(legacy)
    }

    pub fn operation(
        &self,
        p: &projection_spec::Projection,
        id: String,
        operands: &[ImportedOperand],
        legacy: bool,
    ) -> Result<OperationSpec> {
        let selection = rank_selection(
            p,
            &self.ports.geometry,
            Topology {
                ranks: self.ports.shape.ranks(),
                rank: 0,
            },
            0,
        )?;
        let kind = p.operation_kind(&self.ports.geometry, &selection)?;
        let family = p.family();
        let weights = operands
            .iter()
            .map(|o| o.weight.clone())
            .collect::<Vec<_>>();
        let activations = if family == KernelFamily::Lookup {
            vec![]
        } else {
            operands
                .iter()
                .map(|o| {
                    o.activation
                        .clone()
                        .ok_or("projection requires activation operands")
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let activation = activations
            .first()
            .map_or(ActivationFormat::Bf16, |a| a.format);
        let plugin = registry::by_weight(&weights[0])?;
        let recipe = p
            .composition
            .as_ref()
            .and_then(|c| c.numerical_contracts.get(&plugin.id()))
            .map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe);
        let fused = self
            .ports
            .declaration
            .fusion_groups
            .iter()
            .any(|f| f.members.contains(&p.id));
        let mut geometry = if legacy && fused {
            selection.logical
        } else {
            selection.physical
        };
        if !legacy && family == KernelFamily::Dense {
            let padding = plugin.padding_tile(family);
            geometry.rows = geometry.rows.next_multiple_of(u64::try_from(padding.rows)?);
            geometry.k = geometry.k.next_multiple_of(u64::try_from(padding.k)?);
        }
        let uses = p
            .uses
            .iter()
            .map(|u| {
                Ok(kernel_core::precision::OperationUse {
                    phase: u.phase,
                    capacity: u32::try_from(u.capacity.resolve(&self.ports.geometry)?)?,
                    graph: u.graph,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let operation = OperationSpec {
            id,
            kind,
            target: Selected::TARGET,
            weights,
            geometry: (family != KernelFamily::Lookup).then_some(geometry),
            activations,
            arithmetic: ArithmeticSpec {
                weight_operand: match activation {
                    ActivationFormat::F32 => DType::F32,
                    ActivationFormat::Bf16 => DType::Bf16,
                    ActivationFormat::Fp8Group128(_) | ActivationFormat::Mxfp8 => DType::Fp8E4M3,
                    ActivationFormat::Nvfp4 => DType::Fp4E2M1,
                },
                accumulator: DType::F32,
                output: DType::Bf16,
                decode_order: plugin.decode_order(family, activation, recipe)?,
                rounding: Rounding::NearestEven,
                numerical_contract: p.numerical_contract.clone(),
            },
            epilogues: p.epilogues,
            uses,
        };
        operation.validate()?;
        Ok(operation)
    }
}
