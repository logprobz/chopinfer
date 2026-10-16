//! Prepare the exercise's operations once, through a backend, before any
//! device allocations.
use super::precision_contract as model;
use kernel_core::backend::{Backend, Need, Prepared, PreparedOp, View};
use kernel_core::precision::OperationUse;
use kernel_core::precision::buffers::ProjectionBuffers;
use kernel_core::precision::execution::ActivationPolicy;

pub struct PreparedExercise {
    pub metadata: model::Exercise,
    /// Each operation's prepared op at each of its uses, in the order of
    /// `model::uses()`.
    pub operations: Vec<Vec<Prepared<ProjectionBuffers<View>>>>,
}

impl PreparedExercise {
    /// Prepares every binding of the authenticated component `bytes` on
    /// `backend`, which must bind the implementation the package recorded.
    pub fn bind(
        bytes: &[u8],
        policy: ActivationPolicy,
        backend: &dyn Backend,
    ) -> Result<Self, String> {
        if bytes.len() > kernel_core::precision::execution::MAX_PRECISION_DOCUMENT_BYTES {
            return Err("precision exercise component exceeds its bound".into());
        }
        let metadata: model::Exercise = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        metadata.validate()?;
        let mut operations = Vec::new();
        for selected in &metadata.precision.operations {
            selected.require_policy(policy).map_err(|e| e.to_string())?;
            let mut prepared = Vec::new();
            for binding in &selected.bindings {
                let current = backend
                    .projection(
                        &selected.effective,
                        binding.usage,
                        Need::rows(binding.usage.capacity),
                    )
                    .map_err(|e| format!("operation {}: {e}", selected.source.id))?;
                if current.identity != binding.implementation {
                    return Err(format!(
                        "operation {}: package binding differs from supplied module",
                        selected.source.id
                    ));
                }
                prepared.push(current.handle);
            }
            operations.push(prepared);
        }
        Ok(Self {
            metadata,
            operations,
        })
    }

    pub fn handle(
        &self,
        operation: usize,
        usage: OperationUse,
    ) -> Result<&dyn PreparedOp<ProjectionBuffers<View>>, String> {
        let index = model::uses()
            .iter()
            .position(|u| *u == usage)
            .ok_or("unsupported exercise capacity")?;
        self.operations
            .get(operation)
            .and_then(|h| h.get(index))
            .map(|prepared| prepared.as_ref())
            .ok_or_else(|| "unknown exercise operation".into())
    }
}
