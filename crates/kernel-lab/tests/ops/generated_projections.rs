use crate::invariance::Adapter;
use kernel_core::precision::OperationKind;
use kernel_lab::{gpu_test, projection_cases};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn generated_catalog_offers_have_oracles_and_replay_invariance() {
    gpu_test(8 << 30, |session| {
        let (offers, capabilities, _) = kernel_lab::models::projection_inventory()?;
        let variants = kernels_sm120::moe::catalog(kernel_lab::catalogs());
        let mut tested = 0;
        let mut invariant_entries = BTreeSet::new();
        let selection = projection_cases::Selection::read()?;
        for (offset, capability) in capabilities.into_iter().enumerate() {
            let case_index = offers.len() + offset;
            if !selection.contains(case_index) {
                continue;
            }
            let case = projection_cases::catalog(capability.clone())?;
            let symbol = &capability.stages[0];
            {
                match capability.kind {
                    OperationKind::Grouped { .. } => {
                        let variant = variants
                            .iter()
                            .find(|v| &v.symbol == symbol)
                            .ok_or("missing generated catalog variant")?;
                        crate::moe::generated_catalog_oracle(session, variant.clone())?;
                    }
                    OperationKind::Lookup => {
                        crate::row::generated_lookup_oracle(session, &capability)?
                    }
                    _ => return Err("unexpected retained catalog operation".into()),
                }
            }
            if invariant_entries.insert(symbol.clone()) {
                let declared = kernels_sm120::invariance::declarations(kernel_lab::catalogs());
                let entry = declared
                    .iter()
                    .find(|e| &e.symbol == symbol)
                    .ok_or("missing generated invariance entry")?;
                let mut adapter: Box<dyn Adapter> = match capability.kind {
                    OperationKind::Grouped { .. } => {
                        Box::new(crate::moe::invariance::Moe::default())
                    }
                    OperationKind::Lookup => Box::new(crate::row::invariance::Row),
                    _ => return Err("unsupported generated invariance adapter".into()),
                };
                let sessions = std::slice::from_ref(session);
                if let Some(reason) = adapter.unqualified(sessions, entry) {
                    return Err(reason);
                }
                for property in entry.invariance.claims() {
                    kernel_lab::take_launched();
                    let comparisons =
                        crate::invariance::compare(adapter.as_mut(), sessions, entry, property)?;
                    if comparisons.is_empty() || comparisons.iter().any(|c| c.got != c.expected) {
                        return Err(format!("generated catalog {property:?} invariance differs"));
                    }
                    if !kernel_lab::take_launched().contains(symbol) {
                        return Err("generated invariant did not launch its entry".into());
                    }
                }
            }
            tested += 1;
            session.receipt(
                "generated-projection",
                BTreeMap::from([
                    ("case_index".into(), json!(case_index)),
                    ("case".into(), json!(case)),
                    ("oracle".into(), json!(true)),
                    ("replay".into(), json!(true)),
                ]),
            )?;
        }
        if tested == 0 {
            return Err("no retained generated offers ran".into());
        }
        Ok(())
    })
    .expect("generated catalog oracle and invariance");
}
