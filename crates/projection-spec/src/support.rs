//! Checkpoint precision predicates are resolved once by binding, before offers.
use crate::bind::BoundDeclaration;
use crate::contracts::SupportRule;
use crate::{Declaration, Error, Geometry};
use kernel_core::precision::{ExecutionPhase, format::FormatId};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCondition {
    pub any_format_in: Vec<String>,
    pub formats: Vec<FormatId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundCondition {
    pub predicate: CheckpointCondition,
    pub matches: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Admission<'a> {
    pub projection: &'a str,
    pub layer: Option<u64>,
    pub ranks: u32,
    pub phase: ExecutionPhase,
    pub format: FormatId,
}

impl Declaration {
    pub(crate) fn support_rules(&self) -> impl Iterator<Item = &SupportRule> {
        self.layer_stems
            .iter()
            .flat_map(|s| &s.support)
            .chain(
                self.projections
                    .iter()
                    .flat_map(|p| &p.uses)
                    .filter_map(|u| u.contract.as_ref())
                    .flat_map(|c| &c.support),
            )
            .chain(
                self.fusion_groups
                    .iter()
                    .flat_map(|p| &p.uses)
                    .filter_map(|u| u.contract.as_ref())
                    .flat_map(|c| &c.support),
            )
    }
    pub(crate) fn validate_conditions(&self) -> Result<(), Error> {
        for condition in self.support_rules().filter_map(|r| r.when.as_ref()) {
            if condition.any_format_in.is_empty() || condition.formats.is_empty() {
                return Err(Error(
                    "checkpoint conditions need scopes and formats".into(),
                ));
            }
            let scopes = condition
                .any_format_in
                .iter()
                .collect::<std::collections::BTreeSet<_>>();
            let formats = condition
                .formats
                .iter()
                .collect::<std::collections::BTreeSet<_>>();
            if scopes.len() != condition.any_format_in.len()
                || formats.len() != condition.formats.len()
            {
                return Err(Error("checkpoint condition has duplicate selectors".into()));
            }
            for scope in scopes {
                if !self.layer_stems.iter().any(|s| s.id == *scope)
                    && !self.projections.iter().any(|p| p.id == *scope)
                {
                    return Err(Error(format!(
                        "checkpoint condition names unknown scope `{scope}`"
                    )));
                }
            }
            for format in formats {
                kernel_core::precision::format::registry::by_id(*format)
                    .map_err(|e| Error(e.to_string()))?;
            }
        }
        Ok(())
    }
    pub(crate) fn bind_conditions(&self, bound: &BoundDeclaration) -> Vec<BoundCondition> {
        self.checkpoint_conditions(
            bound
                .operands
                .iter()
                .map(|operand| (operand.projection.as_str(), operand.format)),
        )
    }
    /// Resolve every checkpoint condition from the imported format of each
    /// projection operand. A scope matches an operand's projection or the
    /// layer stem that projection belongs to.
    pub fn checkpoint_conditions<'a>(
        &self,
        operands: impl IntoIterator<Item = (&'a str, FormatId)>,
    ) -> Vec<BoundCondition> {
        let operands: Vec<_> = operands.into_iter().collect();
        let mut result = Vec::new();
        for predicate in self.support_rules().filter_map(|r| r.when.as_ref()) {
            if result
                .iter()
                .any(|c: &BoundCondition| c.predicate == *predicate)
            {
                continue;
            }
            let matches = operands.iter().any(|&(projection, format)| {
                predicate.formats.contains(&format)
                    && predicate.any_format_in.iter().any(|scope| {
                        projection == scope
                            || self
                                .projections
                                .iter()
                                .find(|p| p.id == projection)
                                .and_then(|p| p.stem())
                                == Some(scope.as_str())
                    })
            });
            result.push(BoundCondition {
                predicate: predicate.clone(),
                matches,
            });
        }
        result
    }
    pub fn admit_checkpoint(
        &self,
        geometry: &Geometry,
        bound: &BoundDeclaration,
        request: Admission<'_>,
    ) -> Result<(), Error> {
        self.expanded_profiles(geometry)?
            .admit_expanded(geometry, request, Some(bound))
    }
}

impl BoundDeclaration {
    pub fn condition(&self, predicate: &CheckpointCondition) -> Result<bool, Error> {
        self.conditions
            .iter()
            .find(|c| c.predicate == *predicate)
            .map(|c| c.matches)
            .ok_or_else(|| Error("checkpoint condition needs a matching bound snapshot".into()))
    }
}

pub(crate) fn active<'a>(
    rules: &'a [SupportRule],
    bound: Option<&BoundDeclaration>,
) -> Result<Vec<&'a SupportRule>, Error> {
    let mut result = Vec::new();
    for rule in rules {
        let enabled = match &rule.when {
            None => true,
            Some(predicate) => bound
                .ok_or_else(|| Error("conditional support needs checkpoint binding".into()))?
                .condition(predicate)?,
        };
        if enabled {
            result.push(rule);
        }
    }
    Ok(result)
}
