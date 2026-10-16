//! Bounded request shape validation before typed deserialization.
use serde::Deserialize;
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

pub(crate) const MAX_OBJECT_ENTRIES: usize = 4096;
const MAX_NODES: usize = 2 * 1024 * 1024;

pub(crate) fn validate(body: &[u8]) -> Result<(), serde_json::Error> {
    let mut remaining = MAX_NODES;
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    Budget(&mut remaining).deserialize(&mut deserializer)?;
    deserializer.end()
}

struct Budget<'a>(&'a mut usize);

impl<'de> DeserializeSeed<'de> for Budget<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        *self.0 = self
            .0
            .checked_sub(1)
            .ok_or_else(|| de::Error::custom("JSON node limit exceeded"))?;
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Budget<'_> {
    type Value = ();
    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("bounded JSON")
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }
    fn visit_bool<E>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
    fn visit_str<E>(self, _: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
        while sequence.next_element_seed(Budget(self.0))?.is_some() {}
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let first = map.next_key::<NumberKey>()?;
        if first.is_some_and(|key| key.0) {
            let text = map.next_value::<String>()?;
            if text.contains(['.', 'e', 'E']) && !text.parse::<f64>().is_ok_and(f64::is_finite) {
                return Err(de::Error::custom("non-finite JSON number"));
            }
            if map.next_key::<IgnoredAny>()?.is_some() {
                return Err(de::Error::custom("invalid JSON number representation"));
            }
            return Ok(());
        }
        let mut entries = 0;
        let mut present = first.is_some();
        while present {
            entries += 1;
            if entries > MAX_OBJECT_ENTRIES {
                return Err(de::Error::custom("JSON object entry limit exceeded"));
            }
            map.next_value_seed(Budget(self.0))?;
            present = map.next_key::<IgnoredAny>()?.is_some();
        }
        Ok(())
    }
}

// Real JSON map keys dispatch through visit_some. Only serde_json's internal
// arbitrary-precision number map forwards directly to the string callback.
#[derive(Clone, Copy)]
struct NumberKey(bool);
impl<'de> Deserialize<'de> for NumberKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = NumberKey;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an object key or internal number marker")
            }
            fn visit_some<D: serde::Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                IgnoredAny::deserialize(deserializer)?;
                Ok(NumberKey(false))
            }
            fn visit_str<E: de::Error>(self, key: &str) -> Result<Self::Value, E> {
                Ok(NumberKey(key == "$serde_json::private::Number"))
            }
        }
        deserializer.deserialize_option(KeyVisitor)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn arbitrary_numbers_count_as_one_node_and_floats_stay_finite() {
        use serde::de::DeserializeSeed;
        for text in [
            "18446744073709551617",
            "1234567890123456789012345678901234567890",
            "1.0",
        ] {
            let mut nodes = 1;
            let mut reader = serde_json::Deserializer::from_str(text);
            super::Budget(&mut nodes).deserialize(&mut reader).unwrap();
            assert_eq!(nodes, 0);
        }
        assert!(super::validate(b"1e400").is_err());
    }

    #[test]
    fn aggregate_budget_includes_nested_arrays() {
        let body = format!("[{}]", "[0,0],".repeat(super::MAX_NODES / 3) + "[0,0]");
        assert!(
            super::validate(body.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("node limit")
        );
    }
}
