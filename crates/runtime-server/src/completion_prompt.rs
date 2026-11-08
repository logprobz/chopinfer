//! A completion's `prompt` as read: token IDs straight into `u32`s.
//!
//! [`PromptInput`] reads what [`FrontendJsonV1`] reads and refuses what it
//! refuses, at the same place in the body: only how a token ID is held
//! differs. An integer from 0 to `u32::MAX` in the prompt array, or in an
//! array inside it, is held as a `u32` instead of a [`FrontendJsonV1`]
//! number, which holds its digits in a string of its own. Every other value
//! is read by the same visitor [`FrontendJsonV1`] reads with, and the
//! completion checks each element as it checked the [`FrontendJsonV1`] one.

use std::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::frontend::{FrontendJsonV1, OrderedVisitorV1};

/// A completion's `prompt`.
pub(crate) enum PromptInput {
    /// An array whose every element is a token ID, or an empty one.
    Ids(Vec<u32>),
    /// An array with an element that is not a token ID.
    Elements(Vec<PromptElement>),
    /// Any other value.
    Value(FrontendJsonV1),
}

/// One element of a prompt array.
pub(crate) enum PromptElement {
    /// An integer from 0 to `u32::MAX`.
    Id(u32),
    /// An array whose every element is a token ID, or an empty one.
    Ids(Vec<u32>),
    /// Any other value. An array here holds an element that is not a token
    /// ID, and its token IDs are [`FrontendJsonV1`] numbers again.
    Value(FrontendJsonV1),
}

impl PromptElement {
    pub(crate) const fn is_string(&self) -> bool {
        matches!(self, Self::Value(FrontendJsonV1::String(_)))
    }

    pub(crate) const fn is_array(&self) -> bool {
        matches!(self, Self::Ids(_) | Self::Value(FrontendJsonV1::Array(_)))
    }
}

impl<'de> Deserialize<'de> for PromptInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(PromptVisitor)
    }
}

/// Which array an element stands in: the prompt, or an array in it.
#[derive(Clone, Copy)]
enum Depth {
    Prompt,
    Element,
}

/// Reads the prompt: an array by its elements, any other value as
/// [`FrontendJsonV1`] does.
struct PromptVisitor;

/// Reads an element of the array at its depth: a token ID as an ID, an
/// array in the prompt by its elements, and any other value, an array
/// deeper included, as [`FrontendJsonV1`] does.
#[derive(Clone, Copy)]
struct ElementSeed(Depth);

struct ElementVisitor(Depth);

/// An array's elements: IDs while every element is one, then each element.
enum Elements {
    Ids(Vec<u32>),
    Mixed(Vec<PromptElement>),
}

fn elements<'de, A: SeqAccess<'de>>(mut sequence: A, depth: Depth) -> Result<Elements, A::Error> {
    let mut ids = Vec::new();
    while let Some(element) = sequence.next_element_seed(ElementSeed(depth))? {
        let PromptElement::Id(id) = element else {
            let mut mixed: Vec<PromptElement> = ids.into_iter().map(PromptElement::Id).collect();
            mixed.push(element);
            while let Some(element) = sequence.next_element_seed(ElementSeed(depth))? {
                mixed.push(element);
            }
            return Ok(Elements::Mixed(mixed));
        };
        ids.push(id);
    }
    Ok(Elements::Ids(ids))
}

/// An array of `elements` as [`FrontendJsonV1`] holds it.
fn json_array(elements: Vec<PromptElement>) -> FrontendJsonV1 {
    let id = |id: u32| FrontendJsonV1::Number(u64::from(id).into());
    FrontendJsonV1::Array(
        elements
            .into_iter()
            .map(|element| match element {
                PromptElement::Id(value) => id(value),
                PromptElement::Ids(ids) => FrontendJsonV1::Array(ids.into_iter().map(id).collect()),
                PromptElement::Value(value) => value,
            })
            .collect(),
    )
}

impl<'de> DeserializeSeed<'de> for ElementSeed {
    type Value = PromptElement;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        let Self(depth) = self;
        deserializer.deserialize_any(ElementVisitor(depth))
    }
}

/// The visitor [`FrontendJsonV1`] reads with.
const JSON: OrderedVisitorV1 = OrderedVisitorV1::REFUSING_DUPLICATES;

impl<'de> Visitor<'de> for PromptVisitor {
    type Value = PromptInput;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        JSON.expecting(formatter)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, sequence: A) -> Result<Self::Value, A::Error> {
        Ok(match elements(sequence, Depth::Prompt)? {
            Elements::Ids(ids) => PromptInput::Ids(ids),
            Elements::Mixed(elements) => PromptInput::Elements(elements),
        })
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        JSON.visit_unit().map(PromptInput::Value)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        JSON.visit_none().map(PromptInput::Value)
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
        JSON.visit_bool(value).map(PromptInput::Value)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        JSON.visit_i64(value).map(PromptInput::Value)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        JSON.visit_u64(value).map(PromptInput::Value)
    }

    fn visit_i128<E: de::Error>(self, value: i128) -> Result<Self::Value, E> {
        JSON.visit_i128(value).map(PromptInput::Value)
    }

    fn visit_u128<E: de::Error>(self, value: u128) -> Result<Self::Value, E> {
        JSON.visit_u128(value).map(PromptInput::Value)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        JSON.visit_f64(value).map(PromptInput::Value)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        JSON.visit_str(value).map(PromptInput::Value)
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        JSON.visit_string(value).map(PromptInput::Value)
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        JSON.visit_map(map).map(PromptInput::Value)
    }
}

impl<'de> Visitor<'de> for ElementVisitor {
    type Value = PromptElement;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        JSON.expecting(formatter)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, sequence: A) -> Result<Self::Value, A::Error> {
        let Self(depth) = self;
        match depth {
            Depth::Prompt => Ok(match elements(sequence, Depth::Element)? {
                Elements::Ids(ids) => PromptElement::Ids(ids),
                Elements::Mixed(elements) => PromptElement::Value(json_array(elements)),
            }),
            Depth::Element => JSON.visit_seq(sequence).map(PromptElement::Value),
        }
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        JSON.visit_unit().map(PromptElement::Value)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        JSON.visit_none().map(PromptElement::Value)
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
        JSON.visit_bool(value).map(PromptElement::Value)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        match u32::try_from(value) {
            Ok(id) => Ok(PromptElement::Id(id)),
            Err(_) => JSON.visit_i64(value).map(PromptElement::Value),
        }
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        match u32::try_from(value) {
            Ok(id) => Ok(PromptElement::Id(id)),
            Err(_) => JSON.visit_u64(value).map(PromptElement::Value),
        }
    }

    fn visit_i128<E: de::Error>(self, value: i128) -> Result<Self::Value, E> {
        JSON.visit_i128(value).map(PromptElement::Value)
    }

    fn visit_u128<E: de::Error>(self, value: u128) -> Result<Self::Value, E> {
        JSON.visit_u128(value).map(PromptElement::Value)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        JSON.visit_f64(value).map(PromptElement::Value)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        JSON.visit_str(value).map(PromptElement::Value)
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        JSON.visit_string(value).map(PromptElement::Value)
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        JSON.visit_map(map).map(PromptElement::Value)
    }
}
