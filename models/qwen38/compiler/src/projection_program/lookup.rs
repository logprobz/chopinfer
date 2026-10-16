use super::*;
use kernel_core::precision::OperationUse;
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::{Binding, ImplementationProvider};
use model_compiler_sdk::quantization::bank::ImportedTable;
use model_compiler_sdk::quantization::table::concatenate_rows;
use model_compiler_sdk::quantization::{ActivationPolicy, PreparedOperation, resolve_operation};

pub struct LookupOperation {
    pub source: OperationSpec,
    pub table: ImportedTable,
    pub shape: Shape,
    pub ngram: bool,
}

/// Integer lookup operands stay independent of projection activation policy.
/// The family supplies table shard order, hashing and rank ownership.
pub fn lookup(source: &SourcePrecision, ngram: bool, shape: Shape) -> Result<LookupOperation> {
    let definitions = super::declaration::Definitions::new(shape)?;
    let declared = definitions.projection(if ngram { "ngram" } else { "embedding" })?;
    let table = if let Some(lookup) = &declared.lookup {
        let shards = lookup
            .shards
            .resolve(&definitions.ports.geometry)?
            .into_iter()
            .map(|shard| {
                let name = definitions.ports.source_name(&shard.source, None)?;
                source
                    .operands
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| format!("missing table shard {name}").into())
            })
            .collect::<Result<Vec<_>>>()?;
        let name = shards[0]
            .weight
            .codes
            .tensor
            .split_once(".shard_")
            .ok_or("missing declared shard identity")?
            .0;
        concatenate_rows(name, &shards)?
    } else {
        let name = definitions.names(declared, None)?.remove(0);
        ImportedTable {
            operand: source
                .operands
                .get(&name)
                .cloned()
                .ok_or("missing token embedding")?,
            assemblies: vec![],
        }
    };
    let name = table
        .operand
        .weight
        .codes
        .tensor
        .trim_end_matches(".weight")
        .to_owned();
    if table.operand.weight.format == WeightFormat::Nvfp4
        && model_qwen38_contract::projections::support::Support::sources(
            shape,
            source
                .operands
                .iter()
                .map(|(name, o)| (name, o.weight.format)),
        )?
        .admit_mixed(
            &declared.id,
            kernel_core::precision::ExecutionPhase::Decode,
            kernel_core::precision::format::registry::by_weight(&table.operand.weight)?.id(),
        )
        .is_err()
    {
        return Err(UnsuppliedTopology(format!("operation {name}: packed NVFP4 lookup has no supplied TP{} composition; TP4 is supported", shape.ranks())).into());
    }
    let operation =
        definitions.operation(declared, name, std::slice::from_ref(&table.operand), true)?;
    operation.validate()?;
    Ok(LookupOperation {
        source: operation,
        table,
        shape,
        ngram,
    })
}

struct LookupProvider<'a> {
    inventory: &'a CapabilityInventory,
    implementation: String,
}

impl ImplementationProvider for LookupProvider<'_> {
    type Handle = usize;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> std::result::Result<Binding<usize>, String> {
        self.inventory.bind_matching(operation, usage, |c| {
            c.implementation == self.implementation
        })
    }
}

impl LookupOperation {
    pub fn resolve(
        &self,
        policy: ActivationPolicy,
        inventory: &CapabilityInventory,
    ) -> Result<PreparedOperation<usize>> {
        let usage = self.source.uses[0];
        let implementation = if self.ngram && self.source.weights[0].format == WeightFormat::Nvfp4 {
            precision::lookup_implementation(self.shape)
        } else {
            precision::legacy_lookup_capability(self.shape, self.ngram, usage)?.implementation
        };
        Ok(resolve_operation(
            &self.source,
            policy,
            &LookupProvider {
                inventory,
                implementation,
            },
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::precision::capability::CapabilityDocument;

    #[test]
    fn real_checkpoint_lookups_bind_without_projection_activation_recipes() {
        let Ok(root) = std::env::var("CHOPINFER_QWEN38_CHECKPOINT") else {
            eprintln!("UNRUN: set CHOPINFER_QWEN38_CHECKPOINT for real lookup recipe checks");
            return;
        };
        let catalog = crate::catalog::Catalog::load(std::path::Path::new(&root)).unwrap();
        let mut operations = model_qwen38_contract::sm120::generated::all_capabilities().unwrap();
        operations.extend(precision::legacy_capabilities().unwrap());
        let document = CapabilityDocument {
            version: 1,
            operations,
        };
        let entries = document.entries();
        let inventory = CapabilityInventory::new(document, entries, "0".repeat(64)).unwrap();
        for ngram in [false, true] {
            let op = lookup(catalog.precision.as_ref().unwrap(), ngram, Shape::TP4).unwrap();
            let auto = op.resolve(ActivationPolicy::Auto, &inventory).unwrap();
            assert!(auto.selection().bindings.iter().all(|b| {
                !b.implementation
                    .stages
                    .iter()
                    .any(|s| s.contains("qwen38_tp2"))
            }));
            for policy in [ActivationPolicy::Checkpoint, ActivationPolicy::Bf16] {
                let selected = op.resolve(policy, &inventory).unwrap();
                assert_eq!(selected.selection().effective, auto.selection().effective);
                assert!(selected.selection().effective.activations.is_empty());
                assert!(selected.selection().reuse.is_none());
            }
            if ngram {
                assert_eq!(op.table.assemblies[0].parts.len(), 128);
                assert_eq!(op.source.weights[0].logical_shape, [320001536, 160]);
            }
        }
    }
}
