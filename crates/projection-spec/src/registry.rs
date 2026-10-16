//! Namespaced provider contracts. Variant math stays with the provider.

use serde_json::Value;

use crate::{Error, Geometry};

/// Where a registered variant may occur.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Target {
    Projection,
    FusionGroup,
    Stage,
    SourceTransform,
}

/// Decode, validate, and serialize typed parameters to their canonical value.
pub type Decoder = fn(&crate::Variant, &Geometry) -> Result<Value, Error>;

/// One immutable variant id and its parameter contract.
#[derive(Clone, Copy, Debug)]
pub struct Registration {
    pub id: &'static str,
    pub numerical_contract: &'static str,
    pub target: Target,
    pub decoder: Decoder,
}

/// A caller-owned namespace and its projection or group variants.
#[derive(Debug)]
pub struct Provider {
    pub namespace: &'static str,
    pub registrations: &'static [Registration],
}

/// The providers explicitly admitted by a caller.
#[derive(Default, Debug)]
pub struct Registry {
    pub(crate) providers: Vec<&'static Provider>,
}

/// A namespace or local id uses lowercase words separated by `.`, `-`, `_`.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'-' | b'_')
        })
        && name
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphanumeric())
        && name
            .bytes()
            .last()
            .is_some_and(|b| b.is_ascii_alphanumeric())
}

/// Parse the required `<namespace>/<name>` variant id grammar.
pub fn split_id(id: &str) -> Result<(&str, &str), Error> {
    let (namespace, name) = id
        .split_once('/')
        .filter(|(namespace, name)| valid_name(namespace) && valid_name(name))
        .ok_or_else(|| Error(format!("variant `{id}` must be <namespace>/<name>")))?;
    Ok((namespace, name))
}

impl Registry {
    /// Register atomically, refusing foreign ids and duplicate namespaces/ids.
    pub fn register(&mut self, provider: &'static Provider) -> Result<(), Error> {
        if !valid_name(provider.namespace) {
            return Err(Error("invalid provider namespace".into()));
        }
        if self
            .providers
            .iter()
            .any(|p| p.namespace == provider.namespace)
        {
            return Err(Error(format!(
                "namespace `{}` is already registered",
                provider.namespace
            )));
        }
        let mut ids = std::collections::BTreeSet::new();
        for registration in provider.registrations {
            if !valid_name(registration.numerical_contract) {
                return Err(Error("registration needs a numerical contract id".into()));
            }
            if split_id(registration.id)?.0 != provider.namespace {
                return Err(Error(format!(
                    "variant `{}` is outside provider namespace",
                    registration.id
                )));
            }
            if !ids.insert(registration.id) {
                return Err(Error(format!(
                    "variant `{}` is registered twice",
                    registration.id
                )));
            }
        }
        self.providers.push(provider);
        Ok(())
    }

    /// Every admitted variant id, in provider registration order.
    pub fn ids(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.providers
            .iter()
            .flat_map(|p| p.registrations.iter().map(|r| r.id))
    }

    /// Resolve one owned variant and check its typed parameter contract.
    pub fn resolve(
        &self,
        namespace: &str,
        variant: &crate::Variant,
        target: Target,
        geometry: &Geometry,
    ) -> Result<Value, Error> {
        if split_id(&variant.id)?.0 != namespace {
            return Err(Error(format!(
                "variant `{}` is outside declaration namespace `{namespace}`",
                variant.id
            )));
        }
        let registration = self
            .providers
            .iter()
            .filter(|p| p.namespace == namespace)
            .flat_map(|p| p.registrations)
            .find(|r| r.id == variant.id)
            .ok_or_else(|| Error(format!("variant `{}` is not registered", variant.id)))?;
        if registration.target != target {
            return Err(Error(format!(
                "variant `{}` is registered for another target",
                variant.id
            )));
        }
        if variant.inputs.is_empty() || variant.outputs.is_empty() {
            return Err(Error(format!(
                "variant `{}` needs typed inputs/outputs",
                variant.id
            )));
        }
        if registration.numerical_contract != variant.numerical_contract {
            return Err(Error(format!(
                "variant `{}` numerical contract differs from registration",
                variant.id
            )));
        }
        (registration.decoder)(variant, geometry)
            .map_err(|e| Error(format!("variant `{}`: {e}", variant.id)))
    }
}

/// Decoder for strict serde parameter structs. Providers can supply their own
/// decoder when parameters also have geometry-dependent constraints.
pub fn parameters<T: serde::de::DeserializeOwned + serde::Serialize>(
    variant: &crate::Variant,
    _geometry: &Geometry,
) -> Result<Value, Error> {
    canonical(&typed::<T>(variant)?)
}

/// A variant's parameters as a provider's typed struct.
pub fn typed<T: serde::de::DeserializeOwned>(variant: &crate::Variant) -> Result<T, Error> {
    serde_json::from_value(variant.parameters.clone()).map_err(|e| Error(e.to_string()))
}

/// Typed parameters as the canonical value a decoder returns.
pub fn canonical<T: serde::Serialize>(parameters: &T) -> Result<Value, Error> {
    serde_json::to_value(parameters).map_err(|e| Error(e.to_string()))
}

/// A decoder refusal with `reason` unless `condition` holds.
pub fn require(condition: bool, reason: &str) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error(reason.into()))
    }
}

/// Refuse a variant whose ports differ from the registered contract: the
/// same count, and each port, in order, bound under its own id with the
/// listed dtype.
pub fn require_ports(
    variant: &crate::Variant,
    inputs: &[(&str, crate::DType)],
    outputs: &[(&str, crate::DType)],
) -> Result<(), Error> {
    for (ports, expected) in [(&variant.inputs, inputs), (&variant.outputs, outputs)] {
        require(
            ports.len() == expected.len(),
            "port count differs from the registered contract",
        )?;
        for (port, &(id, dtype)) in ports.iter().zip(expected) {
            require(
                port.id == id && port.binding == id && port.dtype == dtype,
                "port identity or dtype differs from the registered contract",
            )?;
        }
    }
    Ok(())
}
