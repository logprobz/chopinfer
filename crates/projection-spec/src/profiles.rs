//! Shared requested-use schedules and scoped overrides.

use crate::contracts::UseContract;
use crate::registry::valid_name;
use crate::{DType, Declaration, Dim, Error, Geometry, Use, Variant};
use kernel_core::precision::ExecutionPhase;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UseProfileRef {
    Id(String),
    Overrides(UseProfileOverride),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UseProfileOverride {
    pub profile: String,
    pub overrides: Vec<UseOverride>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UseSelector {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<ExecutionPhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity: Option<Dim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UseOverride {
    #[serde(default)]
    pub when: UseSelector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dtype: Option<DType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation: Option<Variant>,
}

impl UseProfileRef {
    pub fn id(&self) -> &str {
        match self {
            Self::Id(id) => id,
            Self::Overrides(reference) => &reference.profile,
        }
    }
    pub fn overrides(&self) -> &[UseOverride] {
        match self {
            Self::Id(_) => &[],
            Self::Overrides(reference) => &reference.overrides,
        }
    }
    pub(crate) fn overrides_mut(&mut self) -> &mut [UseOverride] {
        match self {
            Self::Id(_) => &mut [],
            Self::Overrides(reference) => &mut reference.overrides,
        }
    }
}

impl UseSelector {
    fn matches(&self, usage: &Use, geometry: &Geometry) -> Result<bool, Error> {
        Ok(self.phase.is_none_or(|phase| phase == usage.phase)
            && self.graph.is_none_or(|graph| graph == usage.graph)
            && self
                .capacity
                .as_ref()
                .map(|c| c.resolve(geometry))
                .transpose()?
                .is_none_or(|n| usage.capacity.resolve(geometry).ok() == Some(n)))
    }
}

impl Declaration {
    /// Expand references for validation, binding and derivation. Serialization
    /// and the digest retain the compact declaration's typed wire representation.
    pub fn expanded_profiles(&self, geometry: &Geometry) -> Result<Self, Error> {
        for (id, uses) in &self.use_profiles {
            if !valid_name(id) || uses.is_empty() || uses.len() > crate::validate::MAX_INSTANCES {
                return Err(Error(
                    "use profiles need valid ids and bounded nonempty schedules".into(),
                ));
            }
        }
        let expand = |inline: &[Use],
                      reference: &Option<UseProfileRef>|
         -> Result<Vec<Use>, Error> {
            let Some(reference) = reference else {
                return Ok(inline.to_vec());
            };
            if !inline.is_empty() {
                return Err(Error("uses and use_profile are mutually exclusive".into()));
            }
            let mut uses = self
                .use_profiles
                .get(reference.id())
                .cloned()
                .ok_or_else(|| Error(format!("unknown use profile `{}`", reference.id())))?;
            let mut changed = std::collections::BTreeSet::new();
            for override_ in reference.overrides() {
                if override_.output_dtype.is_none() && override_.implementation.is_none() {
                    return Err(Error("empty use profile override".into()));
                }
                let mut matches = 0;
                for (index, usage) in uses.iter_mut().enumerate() {
                    if !override_.when.matches(usage, geometry)? {
                        continue;
                    }
                    matches += 1;
                    let contract: &mut UseContract = usage.contract.as_mut().ok_or_else(|| {
                        Error("use overrides require a profile arithmetic contract".into())
                    })?;
                    if let Some(dtype) = override_.output_dtype {
                        if !changed.insert((index, "output")) {
                            return Err(Error("overlapping output dtype overrides".into()));
                        }
                        contract.output_dtype = dtype;
                    }
                    if let Some(implementation) = &override_.implementation {
                        if !changed.insert((index, "implementation")) {
                            return Err(Error("overlapping implementation overrides".into()));
                        }
                        contract.numerical_contract = implementation.numerical_contract.clone();
                        contract.implementation = Some(implementation.clone());
                    }
                }
                if matches == 0 {
                    return Err(Error(
                        "use profile override selects no requested bucket".into(),
                    ));
                }
            }
            Ok(uses)
        };
        let mut expanded = self.clone();
        for projection in &mut expanded.projections {
            projection.uses = expand(&projection.uses, &projection.use_profile)?;
            projection.use_profile = None;
        }
        for group in &mut expanded.fusion_groups {
            group.uses = expand(&group.uses, &group.use_profile)?;
            group.use_profile = None;
        }
        Ok(expanded)
    }
}
