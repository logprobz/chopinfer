//! Qwen's finite mixed projection geometry and shared implementation requests.
//! Source interpretation and activation policy remain in the compiler SDK.
use crate::backend::{Kernels, Selected};
use crate::contract::*;
use kernel_core::contract::{Epilogues, Refusal, WeightFormat};
/// An expert-bank projection: the gate/up halves or the down projection.
pub use kernel_core::gated_mlp::GatedMlpProjection as Projection;
use kernel_core::precision::{MatrixGeometry, OperationUse};

pub use crate::projections::registered::{
    legacy_capabilities, legacy_capabilities_for, legacy_formats, legacy_lookup_capability,
};

/// Every phase/capacity the text package can execute, including wide graph
/// buckets that use the prefill arithmetic. KV and recurrent state are separate.
fn declared_uses() -> Result<&'static [(OperationUse, u32)], Refusal> {
    type Uses = Result<Vec<(OperationUse, u32)>, Refusal>;
    static USES: std::sync::OnceLock<Uses> = std::sync::OnceLock::new();
    USES.get_or_init(|| {
        let invalid = Refusal("invalid Qwen projection use profile");
        let geometry = crate::projections::geometry(TOPOLOGY.ranks.get()).map_err(|_| invalid)?;
        let declaration =
            crate::projections::declaration(TOPOLOGY.ranks.get()).map_err(|_| invalid)?;
        let capacity = |dim: &projection_spec::Dim| {
            dim.resolve(&geometry)
                .ok()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(invalid)
        };
        declaration
            .use_profiles
            .get("dense")
            .ok_or(invalid)?
            .iter()
            .map(|usage| {
                let contract = usage.contract.as_ref().ok_or(invalid)?;
                Ok((
                    OperationUse {
                        phase: usage.phase,
                        capacity: capacity(&usage.capacity)?,
                        graph: usage.graph,
                    },
                    capacity(&contract.compiled_capacity)?,
                ))
            })
            .collect()
    })
    .as_ref()
    .map(Vec::as_slice)
    .map_err(|e| *e)
}

pub fn uses() -> Result<Vec<OperationUse>, Refusal> {
    Ok(declared_uses()?.iter().map(|(usage, _)| *usage).collect())
}

pub fn projection_capacity(usage: OperationUse) -> Result<u32, Refusal> {
    declared_uses()?
        .iter()
        .find(|(u, _)| *u == usage)
        .map(|(_, capacity)| *capacity)
        .ok_or(Refusal(
            "Qwen precision use is outside the declared phase/capacity set",
        ))
}

/// Roles that can use separate projections when an input fusion contains a
/// quantized operand. The all-BF16 composition retains the legacy fused entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum DenseRole {
    LinearQkv,
    LinearZ,
    LinearA,
    LinearB,
    AttentionQ,
    AttentionK,
    AttentionV,
    IndexQk,
    Output,
    AttentionOutput,
    SharedGateUp,
    SharedDown,
}

/// The topology's declaration ports. A declaration that does not expand is
/// refused, never unwound.
pub fn ports(shape: Shape) -> Result<&'static crate::projections::ports::Ports, Refusal> {
    crate::projections::ports::get(shape)
        .map_err(|_| Refusal("invalid Qwen projection declaration"))
}

impl DenseRole {
    /// The dense port of a declared projection. Both output projections are
    /// one port unless their geometries differ.
    pub fn declared(id: &str, attention_output: bool) -> Option<Self> {
        use DenseRole::*;
        Some(match id {
            "linear.qkv" => LinearQkv,
            "linear.z" => LinearZ,
            "linear.a" => LinearA,
            "linear.b" => LinearB,
            "full.q" => AttentionQ,
            "full.k" => AttentionK,
            "full.v" => AttentionV,
            "full.index" => IndexQk,
            "linear.output" => Output,
            "full.output" if !attention_output => Output,
            "full.output" => AttentionOutput,
            "main.shared.gate-up" => SharedGateUp,
            "main.shared.down" => SharedDown,
            _ => return None,
        })
    }

    /// The dense ports of a main layer or of the MTP layer.
    pub fn roles_for(shape: Shape, layer: u32) -> Result<Vec<Self>, Refusal> {
        if layer > shape.layers() || (layer == shape.layers() && shape.mtp_layers() == 0) {
            return Err(Refusal("Qwen dense operation requires a main or MTP layer"));
        }
        Ok(ports(shape)?
            .layer(layer)
            .iter()
            .map(|(role, _)| *role)
            .collect())
    }

    /// The declared projection of this port at `layer`: at a drafter layer
    /// the declaration executes after the main layers, the drafter
    /// projection that publishes as it; else the main one.
    pub fn projection_id(self, shape: Shape, layer: Option<u32>) -> &'static str {
        if let Some(layer) = layer.filter(|&i| i >= shape.layers())
            && let Some(id) = ports(shape)
                .ok()
                .and_then(|ports| ports.layer_projection(layer, self))
        {
            return id;
        }
        match self {
            Self::LinearQkv => "linear.qkv",
            Self::LinearZ => "linear.z",
            Self::LinearA => "linear.a",
            Self::LinearB => "linear.b",
            Self::AttentionQ => "full.q",
            Self::AttentionK => "full.k",
            Self::AttentionV => "full.v",
            Self::IndexQk => "full.index",
            Self::AttentionOutput => "full.output",
            Self::Output if layer.is_some_and(|i| shape.is_full(i)) => "full.output",
            Self::Output => "linear.output",
            Self::SharedGateUp => "main.shared.gate-up",
            Self::SharedDown => "main.shared.down",
        }
    }

    /// The checkpoint names of this port's operands at a main layer.
    pub fn operand_names(self, shape: Shape, layer: u32) -> Result<Vec<String>, Refusal> {
        ports(shape)?
            .names(
                self.projection_id(shape, Some(layer)),
                Some(u64::from(layer)),
            )
            .map_err(|_| Refusal("dense port is absent from this Qwen layer"))
    }

    pub fn operation_id_for(self, shape: Shape, layer: u32) -> Result<String, Refusal> {
        ports(shape)?
            .layer(layer)
            .iter()
            .find(|(role, _)| *role == self)
            .map(|(_, id)| id.clone())
            .ok_or(Refusal("dense port is absent from this Qwen layer"))
    }

    pub fn all_for(shape: Shape) -> Result<Vec<Self>, Refusal> {
        Ok(ports(shape)?.roles())
    }

    pub fn logical_geometry(self, shape: Shape) -> Result<MatrixGeometry, Refusal> {
        let ports = ports(shape)?;
        let id = self.projection_id(shape, None);
        let fused = ports
            .declaration
            .fusion_groups
            .iter()
            .any(|f| f.members.iter().any(|p| p == id));
        ports
            .geometry(id, fused)
            .map_err(|_| Refusal("Qwen dense port has no rank geometry"))
    }

    pub fn epilogues(self, shape: Shape) -> Result<Epilogues, Refusal> {
        Ok(ports(shape)?
            .projection(self.projection_id(shape, None))
            .map_err(|_| Refusal("undeclared Qwen dense port"))?
            .epilogues)
    }

    /// The exact copy of this port's columns into its fused family's output
    /// row: (columns, source stride, destination stride, column offset).
    pub fn copy_layout_in(
        self,
        ports: &crate::projections::ports::Ports,
        weight: WeightFormat,
    ) -> Result<Option<(u32, u32, u32, u32)>, projection_spec::Error> {
        let id = self.projection_id(ports.shape, None);
        let Some(group) = ports
            .declaration
            .fusion_groups
            .iter()
            .find(|f| f.members.iter().any(|p| p == id))
        else {
            return Ok(None);
        };
        let Some(output) = group.outputs.iter().find(|o| o.member == id) else {
            return Ok(None);
        };
        let peers = group
            .outputs
            .iter()
            .filter(|o| o.buffer == output.buffer)
            .collect::<Vec<_>>();
        if peers.len() <= 1 {
            return Ok(None);
        }
        let rows = |id: &str| ports.geometry(id, true).map(|g| g.rows);
        let mut destination_stride = 0;
        let mut offset = None;
        for peer in &peers {
            if peer.member == id {
                offset = Some(destination_stride);
            }
            destination_stride += rows(&peer.member)?;
        }
        if output.buffer == "ab" {
            destination_stride = ports.geometry["ab_stride"];
        }
        let narrow = |v: u64| u32::try_from(v).map_err(|e| projection_spec::Error(e.to_string()));
        Ok(Some((
            narrow(rows(id)?)?,
            narrow(ports.weight_geometry(id, weight)?.rows)?,
            narrow(destination_stride)?,
            narrow(offset.unwrap_or_default())?,
        )))
    }

    #[cfg(test)]
    pub fn mixed_implementation(self, weight: WeightFormat) -> String {
        self.mixed_implementation_for(Shape::flash(Ranks::TP4), weight)
    }

    /// The mixed composition of this port at one topology. Ranks with equal
    /// port geometry can differ in their copy into the fused row, so a rank
    /// count other than TP4 names its own.
    pub fn mixed_implementation_for(self, shape: Shape, weight: WeightFormat) -> String {
        let topology = if shape.ranks == crate::contract::TOPOLOGY.ranks {
            String::new()
        } else {
            format!(".tp{}", shape.ranks())
        };
        if shape.geometry.id == crate::geometry::FLASH.id {
            format!("qwen38.dense-composition.{self:?}.{weight:?}{topology}.v1")
        } else {
            format!(
                "qwen38.{}.dense-composition.{self:?}.{weight:?}{topology}.v1",
                shape.geometry.id
            )
        }
    }
}

/// The registered fragment expert adapter of an expert-bank port: the
/// projection of the FP8 block engine that serves its weights.
pub fn legacy_expert_projection(port: &str) -> Option<Projection> {
    [
        ("main.experts.gate-up", Projection::GateUp),
        ("main.experts.down", Projection::Down),
    ]
    .into_iter()
    .find(|(p, _)| *p == port)
    .map(|(_, projection)| projection)
}

/// The capability of the compiled grouped entry that serves a registered
/// fragment expert `projection` at `usage`, as the backend states it.
pub fn legacy_expert_capability(
    shape: Shape,
    projection: Projection,
    usage: OperationUse,
) -> Result<kernel_core::precision::capability::OperationCapability, Refusal> {
    <crate::backend::Selected as crate::backend::Kernels>::registered_expert(
        shape, projection, usage,
    )
}

/// The packed n-gram lookup composition with its registered hash prefix.
/// The table keeps its whole geometry at every rank count, so each topology
/// names its own composition.
pub fn lookup_implementation(shape: Shape) -> String {
    let ranks = shape.ranks();
    if shape.geometry.id == crate::geometry::FLASH.id {
        format!("qwen38.ngram-lookup.nvfp4.tp{ranks}.v1")
    } else {
        format!(
            "qwen38.{}.ngram-lookup.nvfp4.tp{ranks}.v1",
            shape.geometry.id
        )
    }
}

pub fn capabilities() -> Result<kernel_core::precision::capability::CapabilityDocument, Refusal> {
    let mut operations = Selected::generated_capabilities(None)
        .map_err(|_| Refusal("invalid generated projection inventory"))?;
    operations.extend(legacy_capabilities()?);
    Ok(kernel_core::precision::capability::CapabilityDocument {
        version: 1,
        operations,
    })
}

pub fn capabilities_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<kernel_core::precision::capability::CapabilityDocument, Refusal> {
    let mut operations = Selected::generated_capabilities(Some(geometry))
        .map_err(|_| Refusal("invalid generated projection inventory"))?;
    operations.extend(legacy_capabilities_for(geometry)?);
    Ok(kernel_core::precision::capability::CapabilityDocument {
        version: 1,
        operations,
    })
}
