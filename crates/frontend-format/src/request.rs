//! The chat request as the declared format reads it.
//!
//! A host sends a request across the `chop.frontend.v1` documents, where
//! every JSON value crosses as its text. These are the values the format
//! reads back. They mirror the transport's request types field for field,
//! so the library links no transport, and the JSON reader is the
//! transport's: object keys keep their order, numbers keep their digits,
//! and a duplicate key is refused.

use std::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

/// The most entries one JSON object may hold, as the transport bounds it.
const MAX_OBJECT_ENTRIES: usize = 4096;

/// One JSON value from a request, with object keys in their arrival order.
///
/// Order is preserved because a rendered prompt can depend on it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendJsonV1 {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl FrontendJsonV1 {
    /// The same value as `serde_json` holds it, keys still in arrival order.
    #[must_use]
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::Null => serde_json::Value::Null,
            Self::Bool(value) => serde_json::Value::Bool(*value),
            Self::Number(number) => serde_json::Value::Number(number.clone()),
            Self::String(text) => serde_json::Value::String(text.clone()),
            Self::Array(values) => {
                serde_json::Value::Array(values.iter().map(Self::to_json_value).collect())
            }
            Self::Object(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(name, value)| (name.clone(), value.to_json_value()))
                    .collect(),
            ),
        }
    }
}

impl<'de> Deserialize<'de> for FrontendJsonV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(OrderedVisitorV1)
    }
}

struct OrderedVisitorV1;

impl<'de> Visitor<'de> for OrderedVisitorV1 {
    type Value = FrontendJsonV1;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::Number(value.into()))
    }

    fn visit_i128<E: de::Error>(self, value: i128) -> Result<Self::Value, E> {
        serde_json::Number::from_i128(value)
            .map(FrontendJsonV1::Number)
            .ok_or_else(|| E::custom("integer out of range"))
    }

    fn visit_u128<E: de::Error>(self, value: u128) -> Result<Self::Value, E> {
        serde_json::Number::from_u128(value)
            .map(FrontendJsonV1::Number)
            .ok_or_else(|| E::custom("integer out of range"))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(FrontendJsonV1::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(FrontendJsonV1::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut items = Vec::new();
        while let Some(item) = sequence.next_element()? {
            items.push(item);
        }
        Ok(FrontendJsonV1::Array(items))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        // serde_json carries numbers outside its primitive visitor callbacks
        // through this single-field representation when arbitrary precision is on.
        let sized_map = map.size_hint().is_some();
        let first = map.next_key::<OrderedKeyV1>()?;
        if !sized_map && first.as_ref().is_some_and(|key| key.number) {
            let text = map.next_value::<String>()?;
            if map.next_key::<String>()?.is_some() {
                return Err(de::Error::custom("invalid JSON number representation"));
            }
            let number: serde_json::Number = text.parse().map_err(de::Error::custom)?;
            if text.contains(['.', 'e', 'E']) && number.as_f64().is_none_or(|v| !v.is_finite()) {
                return Err(de::Error::custom("non-finite JSON number"));
            }
            return Ok(FrontendJsonV1::Number(number));
        }
        let mut entries: Vec<(String, FrontendJsonV1)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut next = first.map(|key| key.text);
        while let Some(key) = next {
            if entries.len() >= MAX_OBJECT_ENTRIES {
                return Err(de::Error::custom("JSON object entry limit exceeded"));
            }
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate field `{key}`")));
            }
            let value = map.next_value_seed(OrderedSeedV1)?;
            entries.push((key, value));
            next = map.next_key::<String>()?;
        }
        Ok(FrontendJsonV1::Object(entries))
    }
}

// The number wrapper forwards option deserialization directly to a borrowed
// string. Real JSON map keys visit_some instead. Owned Value keys visit_string,
// and borrowed Value maps expose a size hint. Keep ordinary keys ordinary even
// when they spell serde_json's private marker.
struct OrderedKeyV1 {
    text: String,
    number: bool,
}

impl<'de> Deserialize<'de> for OrderedKeyV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = OrderedKeyV1;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object key or internal number marker")
            }
            fn visit_some<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                String::deserialize(deserializer).map(|text| OrderedKeyV1 {
                    text,
                    number: false,
                })
            }
            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                Ok(OrderedKeyV1 {
                    text: text.to_owned(),
                    number: text == "$serde_json::private::Number",
                })
            }
            fn visit_string<E: de::Error>(self, text: String) -> Result<Self::Value, E> {
                Ok(OrderedKeyV1 {
                    text,
                    number: false,
                })
            }
        }
        deserializer.deserialize_option(KeyVisitor)
    }
}

struct OrderedSeedV1;

impl<'de> DeserializeSeed<'de> for OrderedSeedV1 {
    type Value = FrontendJsonV1;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(OrderedVisitorV1)
    }
}

/// The role one conversation message carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontendRoleV1 {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// One tool call a conversation already contains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendToolCallV1 {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// One message of a conversation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendMessageV1 {
    pub role: FrontendRoleV1,
    pub content: Option<String>,
    /// Ordered content parts when the client supplied an array.
    pub parts: Option<Vec<FrontendContentPartV1>>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<FrontendToolCallV1>,
    pub tool_call_id: Option<String>,
    pub tools: Vec<FrontendJsonV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendContentPartV1 {
    Text(String),
    Media(FrontendMediaPartV1),
    /// A tool message's `tool_reference` part, by the tool name it carries.
    /// A format that reads none refuses one
    /// ([`TOOL_REFERENCES_UNSUPPORTED_V1`]).
    ToolReference {
        name: String,
    },
}

/// The refusal of a `tool_reference` part by a format that does not read
/// them, with the parameter `messages`. Every such format refuses with this
/// message, so a client sees one wording whichever model it reached.
pub const TOOL_REFERENCES_UNSUPPORTED_V1: &str =
    "tool_reference content parts are not supported by this model";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontendMediaPartKindV1 {
    Image,
    Audio,
    Video,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendMediaPartV1 {
    pub kind: FrontendMediaPartKindV1,
    pub part_type: String,
    /// Where the media comes from. The format never reads it; a request
    /// across the documents carries the part's request index here.
    pub source: String,
    pub format: Option<String>,
}

/// One piece of a rendered prompt: text, or the media part at an index of
/// the rendered media.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptSegmentV1 {
    Text(String),
    Media(usize),
}

/// One function tool a request offered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendToolV1 {
    pub name: String,
    pub description: Option<String>,
    pub parameters: FrontendJsonV1,
    pub parameters_present: bool,
    pub strict: Option<bool>,
}

/// Whether a request permitted tool calls, or forced them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendToolChoiceV1 {
    Auto,
    None,
    /// At least one call to an offered function.
    Required,
    /// Exactly one call to the named function.
    Named(String),
}

/// One format refusal, with the request parameter it blames if there is one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendErrorV1 {
    pub message: String,
    pub parameter: Option<String>,
}

impl FrontendErrorV1 {
    #[must_use]
    pub fn new(message: impl Into<String>, parameter: Option<&str>) -> Self {
        Self {
            message: message.into(),
            parameter: parameter.map(str::to_owned),
        }
    }
}

impl fmt::Display for FrontendErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for FrontendErrorV1 {}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::FrontendJsonV1;

    fn tagged(value: &FrontendJsonV1) -> Value {
        match value {
            FrontendJsonV1::Null => Value::Null,
            FrontendJsonV1::Bool(flag) => Value::Bool(*flag),
            FrontendJsonV1::Number(number) => json!({"number": number.to_string()}),
            FrontendJsonV1::String(text) => json!({"string": text}),
            FrontendJsonV1::Array(items) => {
                json!({"array": items.iter().map(tagged).collect::<Vec<_>>()})
            }
            FrontendJsonV1::Object(entries) => json!({"object": entries
                .iter()
                .map(|(name, value)| json!([name, tagged(value)]))
                .collect::<Vec<_>>()}),
        }
    }

    #[test]
    fn the_shared_json_reader_vectors_read_alike() {
        crate::test_support::json_reader::assert_json_reader_cases(
            tagged,
            FrontendJsonV1::to_json_value,
        );
    }
}
