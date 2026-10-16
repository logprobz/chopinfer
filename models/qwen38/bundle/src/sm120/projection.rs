//! Qwen3.8's generated projection uses on the SM120 backend: each use the
//! precision component binds to a generated composition takes the offer its
//! declared port publishes for that use (`sm120::generated::publish`),
//! with the exact copy into its fused family's output row, under the
//! identity the component authenticates. Registered compositions (legacy
//! dense, FP8 experts, FP8 lookups) keep Qwen's registered launchers
//! (`crate::projections`); a lookup's registered hash stays a prefix of the
//! prepared op.

use std::sync::OnceLock;

use kernel_core::precision::OperationSpec;
use kernel_core::precision::OperationUse;
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::{Binding, ImplementationProvider};
use kernels_sm120::backend::{DenseForm, ProjectionOffer};
use model_qwen38_contract::sm120::generated;

use crate::contract::Shape;

/// The offer of each generated use of an instance of one topology.
pub(crate) struct Offers {
    shape: Shape,
    inventory: OnceLock<Result<CapabilityInventory, String>>,
}

impl Offers {
    pub(crate) fn new(shape: Shape) -> Self {
        Self {
            shape,
            inventory: OnceLock::new(),
        }
    }
}

impl ImplementationProvider for Offers {
    type Handle = ProjectionOffer;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<ProjectionOffer>, String> {
        let shape = self.shape;
        let id = &operation.id;
        let ports =
            model_qwen38_contract::projections::ports::get(shape).map_err(|e| e.to_string())?;
        let projection = ports
            .operation_projection(id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("operation {id}: no declared projection"))?;
        let port = ports
            .role(projection)
            .map_or(ports.published(projection), |role| {
                role.projection_id(shape, None)
            });
        let published = generated::publish(shape, port, operation.weights[0].format, usage)?;
        let handle = published.offer.handle();
        if handle.gather.is_some() || handle.quantizer.is_some() {
            return Err(format!(
                "operation {id}: BF16 projection cannot bind an activation conversion"
            ));
        }
        // The identity the component authenticates for this composition.
        let inventory = self
            .inventory
            .get_or_init(crate::precision::inventory)
            .as_ref()
            .map_err(Clone::clone)?;
        let identity = inventory
            .bind_matching(operation, usage, |capability| {
                capability.implementation == published.capability.implementation
                    && capability.stages == published.capability.stages
            })?
            .identity;
        Ok(Binding {
            identity,
            handle: ProjectionOffer {
                offer: published.offer.clone(),
                copy: published.copy.clone(),
                dense: DenseForm::Tensor,
                live_output: false,
            },
        })
    }
}
