//! Checkpoint-wide refusals come from the declaration's support predicates.
use crate::contract::Shape;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::ExecutionPhase;
use kernel_core::precision::format::{FormatId, registry};
use projection_spec::Error;
use projection_spec::bind::BoundDeclaration;
use projection_spec::support::Admission;

pub struct Support {
    ports: &'static super::ports::Ports,
    bound: BoundDeclaration,
    /// [`supplied`] at this topology, once per checkpoint.
    supplied: std::sync::OnceLock<Result<(), Error>>,
}

impl Support {
    /// Resolve the declared checkpoint conditions from each imported
    /// operand's projection and format.
    pub fn new<'a>(
        shape: Shape,
        operands: impl IntoIterator<Item = (&'a str, WeightFormat)>,
    ) -> Result<Self, Error> {
        Self::in_ports(super::ports::get(shape)?, operands)
    }

    /// [`Self::new`] over one topology's ports.
    pub fn in_ports<'a>(
        ports: &'static super::ports::Ports,
        operands: impl IntoIterator<Item = (&'a str, WeightFormat)>,
    ) -> Result<Self, Error> {
        let operands = operands
            .into_iter()
            .map(|(projection, format)| {
                registry::by_format(format)
                    .map(|plugin| (projection, plugin.id()))
                    .map_err(|e| Error(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bound = BoundDeclaration {
            conditions: ports.declaration.checkpoint_conditions(operands),
            ..BoundDeclaration::default()
        };
        Ok(Self {
            ports,
            bound,
            supplied: std::sync::OnceLock::new(),
        })
    }

    /// Conditions over checkpoint members, by their source tensor names.
    /// Members of no declared projection select no condition.
    pub fn sources<'a>(
        shape: Shape,
        operands: impl IntoIterator<Item = (&'a String, WeightFormat)>,
    ) -> Result<Self, Error> {
        let bindings = super::ports::get(shape)?.bindings()?;
        Self::new(
            shape,
            operands.into_iter().filter_map(|(name, format)| {
                bindings
                    .get(name)
                    .map(|binding| (binding.projection.as_str(), format))
            }),
        )
    }

    pub fn admit(&self, projection: &str, phase: ExecutionPhase) -> Result<(), Error> {
        self.admit_format(projection, phase, FormatId::Bf16)
    }

    pub fn admit_format(
        &self,
        projection: &str,
        phase: ExecutionPhase,
        format: FormatId,
    ) -> Result<(), Error> {
        let declaration = &self.ports.declaration;
        let role = declaration
            .projections
            .iter()
            .find(|p| p.id == projection)
            .ok_or_else(|| Error("unknown projection".into()))?;
        let layer = role
            .stem()
            .map(|id| {
                self.ports
                    .first_executed(id)?
                    .ok_or_else(|| Error("projection has no executed layer".into()))
            })
            .transpose()?;
        declaration.admit_expanded(
            &self.ports.geometry,
            Admission {
                projection,
                layer,
                ranks: self.ports.shape.ranks(),
                phase,
                format,
            },
            Some(&self.bound),
        )
    }
}

impl Support {
    /// [`Self::admit_format`] for a port served by a mixed composition,
    /// which a topology also refuses when the declaration supplies it no
    /// mixed composition at all ([`supplied`]).
    pub fn admit_mixed(
        &self,
        projection: &str,
        phase: ExecutionPhase,
        format: FormatId,
    ) -> Result<(), Error> {
        self.admit_format(projection, phase, format)?;
        self.supplied
            .get_or_init(|| supplied(self.ports.shape))
            .clone()
    }
}

/// Whether the declaration supplies any mixed composition at this topology:
/// with every port format the shared kernels publish present, its conditions
/// admit at least one of them. The registered legacy compositions serve
/// every topology; a mixed one is refused where none is supplied, whatever
/// its format. The declaration supplies mixed compositions at TP2 and TP4.
pub fn supplied(shape: Shape) -> Result<(), Error> {
    let ports = <crate::backend::Selected as crate::backend::Kernels>::published_ports(shape)
        .map_err(Error)?;
    let support = Support::new(shape, ports.iter().copied())?;
    let mut refusal = None;
    for (projection, weight) in ports {
        let format = registry::by_format(weight)
            .map_err(|e| Error(e.to_string()))?
            .id();
        match support.admit_format(projection, ExecutionPhase::Decode, format) {
            Ok(()) => return Ok(()),
            Err(error) => {
                refusal.get_or_insert(error);
            }
        }
    }
    Err(refusal.unwrap_or_else(|| Error("no mixed projection port is published".into())))
}

/// The refusal of a mixed component without MTP records: it was prepared
/// before the MTP layer was recorded, or from an export whose MTP layer the
/// kernels do not compose (preparation names that tensor).
pub const UNRECORDED_MTP: &str = "the precision component records no MTP layer operations; \
     regenerate it from the checkpoint to serve the MTP drafter";

/// Whether a prepared component runs the MTP drafter: one that records the
/// MTP layer's operations, or one of the registered legacy compositions
/// alone, whose drafter keeps its fixed program. The declaration's
/// refusals apply first.
pub fn mtp(
    shape: Shape,
    precision: &kernel_core::precision::execution::PrecisionDocument,
) -> Result<(), Error> {
    document(shape, precision)?.admit("mtp.q", ExecutionPhase::Decode)?;
    if crate::prepared::records_drafter(precision, shape).map_err(Error)? {
        return Ok(());
    }
    let legacy = crate::precision::legacy_formats(shape).map_err(|e| Error(e.0.into()))?;
    let id = |weight| registry::by_format(weight).map(|plugin| plugin.id());
    for selection in &precision.operations {
        let effective = &selection.effective;
        for weight in &effective.weights {
            let format = id(weight.format).map_err(|e| Error(e.to_string()))?;
            if !legacy.iter().any(|&(kind, legacy)| {
                std::mem::discriminant(&kind) == std::mem::discriminant(&effective.kind)
                    && id(legacy).is_ok_and(|id| id == format)
            }) {
                return Err(Error(format!(
                    "operation {}: {UNRECORDED_MTP}",
                    effective.id
                )));
            }
        }
    }
    Ok(())
}

/// Conditions over a prepared document's source operands. An operation of no
/// declared projection selects no condition; binding refuses it later.
pub fn document(
    shape: Shape,
    document: &kernel_core::precision::execution::PrecisionDocument,
) -> Result<Support, Error> {
    let ports = super::ports::get(shape)?;
    let mut operands = Vec::new();
    for selection in &document.operations {
        if let Some(projection) = ports.operation_projection(&selection.source.id)? {
            operands.extend(
                selection
                    .source
                    .weights
                    .iter()
                    .map(|w| (projection, w.format)),
            );
        }
    }
    Support::new(shape, operands)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The declaration supplies mixed compositions at TP2 and TP4, so a
    /// mixed port of any format is admitted at both rank counts.
    #[test]
    fn mixed_ports_are_supplied_at_every_topology() {
        for shape in [Shape::TP2, Shape::TP4] {
            assert!(supplied(shape).is_ok(), "TP{}", shape.ranks());
            for format in [FormatId::Mxfp8, FormatId::Mxfp4, FormatId::Fp8Block128] {
                let support = Support::new(shape, []).unwrap();
                assert!(
                    support
                        .admit_mixed("linear.qkv", ExecutionPhase::Decode, format)
                        .is_ok(),
                    "{format:?} at TP{}",
                    shape.ranks()
                );
            }
            // A quantized text checkpoint no longer selects a declared MTP
            // refusal; the MTP layer's own compositions decide.
            let support = Support::new(shape, [("linear.z", WeightFormat::Mxfp8)]).unwrap();
            assert!(
                support
                    .admit_mixed("mtp.q", ExecutionPhase::Decode, FormatId::Mxfp8)
                    .is_ok()
            );
        }
    }

    /// No checkpoint format of any port selects a declared refusal of TP2,
    /// MTP or the NVFP4 lookup.
    #[test]
    fn mixed_checkpoints_select_no_declared_refusal() {
        for shape in [Shape::TP2, Shape::TP4] {
            for (projection, format) in [
                ("linear.z", WeightFormat::Mxfp8),
                ("full.q", WeightFormat::Mxfp8),
                ("main.experts.gate-up", WeightFormat::Nvfp4),
                ("ngram", WeightFormat::Nvfp4),
                ("ple.key", WeightFormat::Mxfp8),
                ("linear.z", WeightFormat::Bf16),
                ("main.attn.input_mix_weight_up", WeightFormat::Mxfp8),
            ] {
                let support = Support::new(shape, [(projection, format)]).unwrap();
                for admitted in ["linear.qkv", "full.q", "mtp.q"] {
                    assert!(
                        support.admit(admitted, ExecutionPhase::Decode).is_ok(),
                        "{projection} {format:?}: {admitted} at TP{}",
                        shape.ranks()
                    );
                }
            }
            assert!(
                Support::new(shape, [])
                    .unwrap()
                    .admit_format("ngram", ExecutionPhase::Decode, FormatId::Nvfp4)
                    .is_ok(),
                "NVFP4 lookup at TP{}",
                shape.ranks()
            );
        }
    }
}
