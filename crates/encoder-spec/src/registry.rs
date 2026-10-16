//! Model providers. A model's bespoke variants (ids `<model>:<kind>...`)
//! live in the model's own crate and reach validation only through a
//! `Provider` registered in the caller's `Registry`. Each registration
//! names one immutable versioned id and the decoder of its typed
//! parameters; the decoded value states its shape and weight contracts and
//! its policy and parallel capabilities through `Component` or
//! `RotaryComponent`. Shared code never names a provider's variants, and an
//! id no provider registers is refused.

use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::Error;
use crate::components::{Component, Registered, RotaryComponent, Unresolved};
use crate::id::{self, Kind};

pub type StageDecoder = fn(Value) -> Result<Arc<dyn Component>, String>;
pub type RotaryDecoder = fn(Value) -> Result<Arc<dyn RotaryComponent>, String>;

/// What a registered id decodes into.
#[derive(Clone, Copy, Debug)]
pub enum Decoder {
    /// A stage (any stage kind).
    Stage(StageDecoder),
    /// A transformer's rotary slot.
    Rotary(RotaryDecoder),
}

/// One registered id. Its numerics never change; a change is a new id.
#[derive(Clone, Copy, Debug)]
pub struct Registration {
    pub id: &'static str,
    pub decoder: Decoder,
}

/// A model's bespoke variants.
#[derive(Debug)]
pub struct Provider {
    /// The owning model; every id carries it as its namespace.
    pub namespace: &'static str,
    pub registrations: &'static [Registration],
}

/// Decodes a stage variant `T`, a serde enum tagged by `op` whose fields
/// refuse unknown names.
pub fn stage<T: Component + DeserializeOwned + 'static>(
    mut value: Value,
) -> Result<Arc<dyn Component>, String> {
    normalize_json_floats(&mut value)?;
    serde_json::from_value::<T>(value)
        .map(|component| Arc::new(component) as Arc<dyn Component>)
        .map_err(|error| error.to_string())
}

/// Decodes a rotary variant `T`.
pub fn rotary<T: RotaryComponent + DeserializeOwned + 'static>(
    mut value: Value,
) -> Result<Arc<dyn RotaryComponent>, String> {
    normalize_json_floats(&mut value)?;
    serde_json::from_value::<T>(value)
        .map(|component| Arc::new(component) as Arc<dyn RotaryComponent>)
        .map_err(|error| error.to_string())
}

// Tagged serde enums buffer generic numbers before choosing a variant. With
// arbitrary precision, normalize float spellings to their binary64 value so
// the buffer sees a float rather than serde_json's private number map. Integer
// tokens keep their exact digits and remain subject to each typed field's bounds.
pub(crate) fn normalize_json_floats(value: &mut Value) -> Result<(), String> {
    match value {
        Value::Number(number) if number.to_string().contains(['.', 'e', 'E']) => {
            *number = number
                .as_f64()
                .filter(|number| number.is_finite())
                .and_then(serde_json::Number::from_f64)
                .ok_or_else(|| "non-finite encoder number".to_owned())?;
        }
        Value::Array(items) => {
            for item in items {
                normalize_json_floats(item)?;
            }
        }
        Value::Object(fields) => {
            for item in fields.values_mut() {
                normalize_json_floats(item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// The providers a caller admits.
#[derive(Default)]
pub struct Registry {
    providers: Vec<&'static Provider>,
}

impl Registry {
    /// Adds `provider`, refusing malformed or duplicate ids, a decoder of
    /// the wrong kind, and a namespace another provider already owns.
    pub fn register(&mut self, provider: &'static Provider) -> Result<(), Error> {
        let fail = |why: String| Err(Error(format!("provider {}: {why}", provider.namespace)));
        if self
            .providers
            .iter()
            .any(|p| p.namespace == provider.namespace)
        {
            return fail("the namespace is already registered".into());
        }
        for (index, registration) in provider.registrations.iter().enumerate() {
            let parsed = id::parse(registration.id).map_err(Error)?;
            if parsed.namespace != Some(provider.namespace) {
                return fail(format!("`{}` is outside its namespace", registration.id));
            }
            let fits = match registration.decoder {
                Decoder::Stage(_) => parsed.kind.is_stage(),
                Decoder::Rotary(_) => parsed.kind == Kind::Rotary,
            };
            if !fits {
                return fail(format!(
                    "`{}` has a decoder for another kind than {}",
                    registration.id,
                    parsed.kind.name()
                ));
            }
            if provider.registrations[..index]
                .iter()
                .any(|other| other.id == registration.id)
            {
                return fail(format!("`{}` is registered twice", registration.id));
            }
        }
        self.providers.push(provider);
        Ok(())
    }

    /// Every registered id.
    pub fn ids(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.providers
            .iter()
            .flat_map(|provider| provider.registrations.iter().map(|r| r.id))
    }

    /// Whether this registry holds `provider` itself, and it registers `id`.
    pub fn admits(&self, provider: &'static Provider, id: &str) -> bool {
        self.providers
            .iter()
            .any(|held| std::ptr::eq(*held, provider))
            && provider.registrations.iter().any(|r| r.id == id)
    }

    fn find(&self, owner: &str, id: &str) -> Result<(&'static Provider, Registration), String> {
        let namespace = id::parse(id)?.namespace;
        if namespace != Some(owner) {
            return Err(format!(
                "`{id}` belongs to model {}; model {owner} cannot use it",
                namespace.unwrap_or("(shared)")
            ));
        }
        self.providers
            .iter()
            .filter(|provider| provider.namespace == owner)
            .find_map(|provider| {
                let registration = provider.registrations.iter().find(|r| r.id == id)?;
                Some((*provider, *registration))
            })
            .ok_or_else(|| format!("`{id}` is not registered by a provider of model {owner}"))
    }

    pub(crate) fn stage(
        &self,
        owner: &str,
        unresolved: &Unresolved,
    ) -> Result<Registered<dyn Component>, String> {
        let id = &unresolved.id;
        let (provider, registration) = self.find(owner, id)?;
        let Decoder::Stage(decode) = registration.decoder else {
            return Err(format!(
                "`{id}` is registered as a rotary slot, not a stage"
            ));
        };
        let component = decode(Value::Object(unresolved.value.clone()))
            .map_err(|error| format!("`{id}`: {error}"))?;
        if component.id() != *id {
            return Err(format!("`{id}` decodes as `{}`", component.id()));
        }
        Ok(Registered {
            component,
            provider,
        })
    }

    pub(crate) fn rotary(
        &self,
        owner: &str,
        unresolved: &Unresolved,
    ) -> Result<Registered<dyn RotaryComponent>, String> {
        let id = &unresolved.id;
        let (provider, registration) = self.find(owner, id)?;
        let Decoder::Rotary(decode) = registration.decoder else {
            return Err(format!(
                "`{id}` is registered as a stage, not a rotary slot"
            ));
        };
        let component = decode(Value::Object(unresolved.value.clone()))
            .map_err(|error| format!("`{id}`: {error}"))?;
        if component.id() != *id {
            return Err(format!("`{id}` decodes as `{}`", component.id()));
        }
        Ok(Registered {
            component,
            provider,
        })
    }
}
