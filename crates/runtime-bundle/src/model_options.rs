//! Package-declared, server-wide options. The host validates values without
//! assigning meaning to checkpoint configuration fields.
use crate::BundleError;
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use std::collections::BTreeMap;

pub const CAPABILITY: &str = "chop.model-options.v1";
pub const MAX_OPTIONS: usize = 64;
pub const MAX_VALUE_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOptionsV1 {
    pub schema: String,
    pub options: Vec<ModelOptionV1>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOptionV1 {
    /// The checkpoint configuration field, including its path when nested.
    pub name: String,
    pub definition: OptionDefinitionV1,
    pub default_is_qualified: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum OptionDefinitionV1 {
    Integer {
        default: i64,
        minimum: i64,
        maximum: i64,
    },
    Number {
        default: Number,
        minimum: Number,
        maximum: Number,
    },
    Boolean {
        default: bool,
    },
    String {
        default: String,
        allowed_values: Vec<String>,
    },
}

impl OptionDefinitionV1 {
    pub fn default_value(&self) -> Value {
        match self {
            Self::Integer { default, .. } => Value::from(*default),
            Self::Number { default, .. } => Value::Number(default.clone()),
            Self::Boolean { default } => Value::Bool(*default),
            Self::String { default, .. } => Value::String(default.clone()),
        }
    }

    pub fn accepts(&self, value: &Value) -> bool {
        match self {
            Self::Integer {
                minimum, maximum, ..
            } => value
                .as_i64()
                .is_some_and(|v| (*minimum..=*maximum).contains(&v)),
            Self::Number {
                minimum, maximum, ..
            } => value.as_f64().is_some_and(|v| {
                v.is_finite()
                    && v >= minimum.as_f64().unwrap_or(f64::NAN)
                    && v <= maximum.as_f64().unwrap_or(f64::NAN)
            }),
            Self::Boolean { .. } => value.is_boolean(),
            Self::String { allowed_values, .. } => value
                .as_str()
                .is_some_and(|v| allowed_values.iter().any(|allowed| allowed == v)),
        }
    }

    fn parse(&self, text: &str) -> Result<Value, BundleError> {
        if matches!(self, Self::String { .. }) {
            Ok(Value::String(text.to_owned()))
        } else {
            serde_json::from_str(text)
                .map_err(|_| BundleError::validation("invalid model option value"))
        }
    }
}

pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.')
}

impl ModelOptionsV1 {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.schema != CAPABILITY || self.options.is_empty() || self.options.len() > MAX_OPTIONS
        {
            return Err(BundleError::validation(
                "invalid model options schema or count",
            ));
        }
        let mut previous: Option<&str> = None;
        for option in &self.options {
            if !valid_name(&option.name)
                || previous.is_some_and(|name| name >= option.name.as_str())
                || !option
                    .definition
                    .accepts(&option.definition.default_value())
            {
                return Err(BundleError::validation(
                    "model options require unique sorted names and defaults within their ranges",
                ));
            }
            if let OptionDefinitionV1::String { allowed_values, .. } = &option.definition
                && (allowed_values.len() > 64
                    || allowed_values.iter().any(|v| v.len() > 1024)
                    || allowed_values.windows(2).any(|pair| pair[0] >= pair[1]))
            {
                return Err(BundleError::validation(
                    "invalid model option string choices",
                ));
            }
            previous = Some(&option.name);
        }
        let defaults: BTreeMap<_, _> = self
            .options
            .iter()
            .map(|option| (&option.name, option.definition.default_value()))
            .collect();
        check_payload(&defaults)?;
        Ok(())
    }

    pub fn resolve(
        &self,
        overrides: &BTreeMap<String, String>,
    ) -> Result<BTreeMap<String, Value>, BundleError> {
        self.validate()?;
        for name in overrides.keys() {
            if !self.options.iter().any(|option| &option.name == name) {
                return Err(BundleError::validation(format!(
                    "unknown model option '{name}'"
                )));
            }
        }
        let values: BTreeMap<String, Value> = self
            .options
            .iter()
            .map(|option| {
                let value = match overrides.get(&option.name) {
                    Some(text) => option.definition.parse(text)?,
                    None => option.definition.default_value(),
                };
                if !option.definition.accepts(&value) {
                    return Err(BundleError::validation(format!(
                        "model option '{}' has the wrong type or is outside its allowed range",
                        option.name
                    )));
                }
                Ok((option.name.clone(), value))
            })
            .collect::<Result<_, BundleError>>()?;
        check_payload(&values)?;
        Ok(values)
    }

    pub fn is_qualified(&self, values: &BTreeMap<String, Value>) -> bool {
        values.len() == self.options.len()
            && self.options.iter().all(|option| {
                option.default_is_qualified
                    && values.get(&option.name) == Some(&option.definition.default_value())
            })
    }
}

fn check_payload(values: &impl Serialize) -> Result<(), BundleError> {
    if serde_json::to_vec(values)
        .map_err(|e| BundleError::validation(e.to_string()))?
        .len()
        > MAX_VALUE_BYTES
    {
        return Err(BundleError::validation(
            "model option values exceed the 64 KiB create payload limit",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn declaration() -> ModelOptionsV1 {
        ModelOptionsV1 {
            schema: CAPABILITY.into(),
            options: vec![ModelOptionV1 {
                name: "budget".into(),
                definition: OptionDefinitionV1::Integer {
                    default: 8,
                    minimum: 4,
                    maximum: 16,
                },
                default_is_qualified: true,
            }],
        }
    }
    #[test]
    fn refuses_unknown_types_ranges_and_invalid_defaults() {
        let declaration = declaration();
        for (name, value) in [
            ("unknown", "8"),
            ("budget", "true"),
            ("budget", "8.0"),
            ("budget", "3"),
            ("budget", "17"),
        ] {
            assert!(
                declaration
                    .resolve(&BTreeMap::from([(name.into(), value.into())]))
                    .is_err()
            );
        }
        let defaults = declaration.resolve(&BTreeMap::new()).unwrap();
        assert!(declaration.is_qualified(&defaults));
        let changed = declaration
            .resolve(&BTreeMap::from([("budget".into(), "4".into())]))
            .unwrap();
        assert!(!declaration.is_qualified(&changed));
        let mut invalid = declaration.clone();
        invalid.options.push(invalid.options[0].clone());
        assert!(invalid.validate().is_err());
        invalid = declaration.clone();
        invalid.options[0].definition = OptionDefinitionV1::Integer {
            default: 3,
            minimum: 4,
            maximum: 16,
        };
        assert!(invalid.validate().is_err());
        invalid.schema = "chop.model-options.v2".into();
        assert!(invalid.validate().is_err());
    }
}
