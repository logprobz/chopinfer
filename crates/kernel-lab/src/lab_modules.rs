//! SM120's symbols through the lab's plumbing ([`LabModules`]).

use crate::kit::Modules;
use crate::{LabResult, Session};
use kernel_harness::conformance::Probes;
use kernel_harness::kit::modules::Functions;
use runtime_cuda::Module;
use std::cell::RefCell;
use std::collections::BTreeMap;

/// SM120's symbols through the lab's plumbing: the cubins this crate built
/// from the catalogs ([`crate::cubin`]), then the prepared projection
/// report (`CHOPINFER_PROJECTION_REPORT`) for typed projection entries.
pub struct LabModules;

impl Modules for LabModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let loaded = RefCell::new(BTreeMap::<&'static str, Module>::new());
        Ok(Box::new(move |symbol| {
            let Some(name) = crate::LAB_CUBINS.holding(symbol) else {
                return crate::projection_function(session, symbol, 0)?
                    .ok_or_else(|| format!("no lab cubin or prepared projection holds {symbol}"));
            };
            let mut loaded = loaded.borrow_mut();
            if !loaded.contains_key(name) {
                loaded.insert(name, session.module(name)?);
            }
            loaded[name].function(symbol).map_err(|e| e.to_string())
        }))
    }

    /// SM120's probes: the live-count producer and the latent lab module's
    /// copy producer.
    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let lab = session.module(&crate::latent_shapes::cubin_name())?;
        Probes::sm120(session, Some(&lab)).map(Some)
    }
}

impl LabModules {
    /// Whether a lab cubin holds `symbol`, so a GPU run can load it
    /// without a prepared projection report.
    pub fn holds(symbol: &str) -> bool {
        crate::LAB_CUBINS.holding(symbol).is_some()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn lab_symbols_resolve_to_their_cubin() {
        let (name, entries) = crate::CUBIN_ENTRIES
            .iter()
            .find(|(_, entries)| !entries.is_empty())
            .expect("a lab cubin with entries");
        assert_eq!(crate::LAB_CUBINS.holding(entries[0]), Some(*name));
        assert_eq!(crate::LAB_CUBINS.holding("chop_no_such_entry_v1"), None);
    }
}
