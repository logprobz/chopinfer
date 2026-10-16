//! Projection ports of Qwen's existing BF16 fusions. Family preparation and
//! lowering keep the peer operands together; portable capability selection
//! never infers a fusion from matching matrix dimensions.
use crate::backend::{Kernels, Selected};
use crate::contract::*;
use crate::precision::*;
use kernel_core::contract::{Refusal, WeightFormat};
use kernel_core::precision::capability::OperationCapability;
use kernel_core::precision::{MatrixGeometry, OperationKind, OperationUse};

impl DenseRole {
    /// Stable composition identity, independent of a particular phase entry.
    pub fn legacy_implementation(self, shape: Shape) -> String {
        format!("qwen38.legacy-dense.{self:?}.tp{}.v1", shape.ranks())
    }

    /// The output rows of each weight the registered fusion of this role
    /// declares at a decode or prefill use: its schedule group's members,
    /// summed where a prefill group writes one output, else the role's own
    /// rows (`geometry`, its logical geometry) per operand.
    pub fn legacy_segments(
        self,
        shape: Shape,
        decode: bool,
        geometry: MatrixGeometry,
    ) -> Result<Vec<u32>, Refusal> {
        let ports = ports(shape)?;
        let id = self.projection_id(shape, None);
        let rows = |p: &str| {
            ports
                .geometry(p, true)
                .map(|g| g.rows as u32)
                .map_err(|_| Refusal("fused legacy member has no rank geometry"))
        };
        let group = ports
            .declaration
            .fusion_groups
            .iter()
            .find(|f| f.members.iter().any(|p| p == id));
        Ok(if let Some(group) = group {
            let parameters: crate::projections::InputFusionParameters = serde_json::from_value(
                group
                    .composition
                    .as_ref()
                    .and_then(|c| c.implementation.as_ref())
                    .ok_or(Refusal("legacy fusion has no registered schedule"))?
                    .parameters
                    .clone(),
            )
            .map_err(|_| Refusal("invalid registered fusion schedule"))?;
            let members = if decode {
                parameters.decode_group
            } else {
                parameters
                    .prefill_groups
                    .into_iter()
                    .find(|g| g.iter().any(|p| p == id))
                    .ok_or(Refusal("projection is absent from its fusion schedule"))?
            };
            if !decode && members.len() > 1 && self != Self::LinearA && self != Self::LinearB {
                vec![members.iter().map(|p| rows(p)).sum::<Result<_, _>>()?]
            } else {
                members.iter().map(|p| rows(p)).collect::<Result<_, _>>()?
            }
        } else {
            let operands = ports
                .projection(id)
                .map_err(|_| Refusal("undeclared Qwen dense port"))?
                .operand_count();
            vec![geometry.rows as u32 * operands]
        })
    }

    /// The capability of the compiled BF16 fusion that serves this role at
    /// `usage`, as the backend states it.
    pub fn legacy_capability(
        self,
        shape: Shape,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal> {
        Selected::registered_dense(shape, self, usage)
    }
}

pub fn legacy_capabilities() -> Result<Vec<OperationCapability>, Refusal> {
    let mut result = Vec::new();
    for geometry in crate::geometry::GEOMETRIES {
        result.extend(legacy_capabilities_for(geometry)?);
    }
    Ok(result)
}

pub fn legacy_capabilities_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<Vec<OperationCapability>, Refusal> {
    let mut result = Vec::new();
    for shape in TOPOLOGIES.map(|t| Shape::new(geometry, t.ranks)) {
        for usage in uses()? {
            for role in DenseRole::all_for(shape)? {
                result.push(role.legacy_capability(shape, usage)?);
            }
            for projection in [Projection::GateUp, Projection::Down] {
                result.push(legacy_expert_capability(shape, projection, usage)?);
            }
            for ngram in [false, true] {
                result.push(legacy_lookup_capability(shape, ngram, usage)?);
            }
        }
    }
    Ok(result)
}

/// The operation kinds and weight formats the registered legacy
/// compositions of one topology serve.
pub fn legacy_formats(shape: Shape) -> Result<Vec<(OperationKind, WeightFormat)>, Refusal> {
    let usage = *uses()?
        .first()
        .ok_or(Refusal("Qwen declares no projection uses"))?;
    let mut result = Vec::new();
    for role in DenseRole::all_for(shape)? {
        result.push(role.legacy_capability(shape, usage)?);
    }
    for projection in [Projection::GateUp, Projection::Down] {
        result.push(legacy_expert_capability(shape, projection, usage)?);
    }
    for ngram in [false, true] {
        result.push(legacy_lookup_capability(shape, ngram, usage)?);
    }
    Ok(result.into_iter().map(|c| (c.kind, c.weight)).collect())
}

/// The capability of the compiled row entry that serves an embedding or
/// n-gram lookup at `usage`, as the backend states it.
pub fn legacy_lookup_capability(
    shape: Shape,
    ngram: bool,
    usage: OperationUse,
) -> Result<OperationCapability, Refusal> {
    Selected::registered_lookup(shape, ngram, usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_projection_ports_keep_the_fused_entries_and_all_buckets() {
        let capabilities = legacy_capabilities_for(&crate::geometry::FLASH).unwrap();
        assert_eq!(
            capabilities.len(),
            2 * uses().unwrap().len() * (DenseRole::all_for(Shape::TP4).unwrap().len() + 4)
        );
        for shape in Shape::ALL {
            for usage in uses().unwrap() {
                let a = DenseRole::LinearA.legacy_capability(shape, usage).unwrap();
                let b = DenseRole::LinearB.legacy_capability(shape, usage).unwrap();
                assert_eq!(a.geometry.rows, u64::from(shape.value_heads));
                assert_eq!(a.stages, b.stages);
                assert_ne!(a.implementation, b.implementation);
                let q = DenseRole::AttentionQ
                    .legacy_capability(shape, usage)
                    .unwrap();
                let k = DenseRole::AttentionK
                    .legacy_capability(shape, usage)
                    .unwrap();
                assert_eq!(q.stages, k.stages);
                if projection_capacity(usage).unwrap() == PREFILL_TOKENS {
                    assert!(q.stages[0].contains("attn_qkv_prefill"));
                    assert!(a.stages[0].ends_with("lin_ab_v1"));
                }
            }
        }
    }
}
