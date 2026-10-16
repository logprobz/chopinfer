//! Formats a projection publishes through shared generated kernel offers.
use std::collections::BTreeSet;

use kernel_core::contract::WeightFormat;
use kernel_core::precision::format::{FormatId, registry};

use crate::{Declaration, Error, Projection, ProjectionKind};

/// Generated offers take a parameterless weight format.
const OFFERED: [WeightFormat; 4] = [
    WeightFormat::Bf16,
    WeightFormat::Mxfp8,
    WeightFormat::Mxfp4,
    WeightFormat::Nvfp4,
];

/// The weight format a generated offer of `format` takes: the
/// parameterless format its plug-in accepts.
pub fn offered_weight(format: FormatId) -> Result<WeightFormat, Error> {
    let plugin = registry::by_id(format).map_err(|e| Error(e.to_string()))?;
    OFFERED
        .into_iter()
        .find(|weight| plugin.accepts(*weight))
        .ok_or_else(|| Error(format!("{format:?} has no generated offer weight format")))
}

impl Projection {
    /// The weight formats of this projection's generated offers, in order.
    pub fn offered_weights(&self) -> Result<Vec<WeightFormat>, Error> {
        self.offers.iter().map(|f| offered_weight(*f)).collect()
    }
}

impl Declaration {
    pub(crate) fn validate_offers(&self) -> Result<(), Error> {
        for p in self.projections.iter().filter(|p| !p.offers.is_empty()) {
            let refuse = |reason: String| Err(Error(format!("projection `{}` {reason}", p.id)));
            if p.publishes_as.is_some() {
                return refuse("publishes its target's offers and declares none".into());
            }
            let mut weights = BTreeSet::new();
            for (format, weight) in p.offers.iter().zip(p.offered_weights()?) {
                if !weights.insert(format!("{weight:?}")) {
                    return refuse(format!("offers {weight:?} twice"));
                }
                // A lookup's recipe is its own numerical id; any other
                // offer needs the composition's recipe for its format.
                let recipe = p
                    .composition
                    .as_ref()
                    .is_some_and(|c| c.numerical_contracts.contains_key(format));
                if !recipe && !matches!(p.kind, ProjectionKind::LookupTable {}) {
                    return refuse(format!("offers {format:?} without a numerical recipe"));
                }
            }
        }
        Ok(())
    }
}
