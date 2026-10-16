//! The instance configuration this fixture decodes at the C ABI boundary.
//!
//! Nothing Rust-owned crosses the boundary, so the wire forms are declared here
//! rather than shared through a crate with the host. Every field is fixed width
//! and little endian, and every length is checked before it is used. The batch record uses the shared SDK reader.

use std::error::Error;
use std::fmt;

/// The fixed prefix of one encoded instance configuration.
pub const BUNDLE_CONFIGURATION_MAGIC_V1: &[u8; 20] = b"chopbundle-config-v1";

/// The largest repeated field the configuration reader allocates.
pub const MAXIMUM_WIRE_ELEMENTS_V1: u32 = 1 << 20;

/// Why one host record was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WireErrorV1 {
    Magic {
        surface: &'static str,
    },
    Truncated {
        surface: &'static str,
        needed: usize,
    },
    TrailingBytes {
        surface: &'static str,
        remaining: usize,
    },
    CountOutOfRange {
        surface: &'static str,
        actual: u32,
    },
}

impl fmt::Display for WireErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Magic { surface } => {
                write!(formatter, "{surface} does not carry its declared prefix")
            }
            Self::Truncated { surface, needed } => {
                write!(formatter, "{surface} needs {needed} more bytes than it has")
            }
            Self::TrailingBytes { surface, remaining } => {
                write!(formatter, "{surface} has {remaining} unread trailing bytes")
            }
            Self::CountOutOfRange { surface, actual } => {
                write!(formatter, "{surface} declares {actual} elements")
            }
        }
    }
}

impl Error for WireErrorV1 {}

/// A bounds-checked cursor over one host record.
struct WireReaderV1<'a> {
    surface: &'static str,
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> WireReaderV1<'a> {
    const fn new(surface: &'static str, bytes: &'a [u8]) -> Self {
        Self {
            surface,
            bytes,
            offset: 0,
        }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], WireErrorV1> {
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(WireErrorV1::Truncated {
                surface: self.surface,
                needed: length,
            })?;
        let read = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(read)
    }

    fn expect_magic(&mut self, magic: &[u8]) -> Result<(), WireErrorV1> {
        if self.take(magic.len())? == magic {
            Ok(())
        } else {
            Err(WireErrorV1::Magic {
                surface: self.surface,
            })
        }
    }

    fn u32(&mut self) -> Result<u32, WireErrorV1> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32, WireErrorV1> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64, WireErrorV1> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Reads one bounded element count and reserves nothing before checking it.
    fn count(&mut self, surface: &'static str) -> Result<usize, WireErrorV1> {
        let count = self.u32()?;
        if count > MAXIMUM_WIRE_ELEMENTS_V1 {
            return Err(WireErrorV1::CountOutOfRange {
                surface,
                actual: count,
            });
        }
        Ok(count as usize)
    }

    /// Reads one length-prefixed opaque identifier.
    fn identifier(&mut self, surface: &'static str) -> Result<Vec<u8>, WireErrorV1> {
        let length = self.count(surface)?;
        Ok(self.take(length)?.to_vec())
    }

    fn finish(self) -> Result<(), WireErrorV1> {
        let remaining = self.bytes.len() - self.offset;
        if remaining == 0 {
            Ok(())
        } else {
            Err(WireErrorV1::TrailingBytes {
                surface: self.surface,
                remaining,
            })
        }
    }
}

/// The declared KV page layout this instance asks the host to allocate.
///
/// This fixture allocates a physical pool from exactly this record, so a
/// configuration it cannot read whole is one it refuses rather than guesses at.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfiguredKvLayoutV1 {
    pub layout_id: Vec<u8>,
    pub page_bytes: u64,
    pub page_alignment: u64,
    pub page_tokens: u64,
    pub rank_count: u64,
    pub state_lifetime: u64,
}

/// The declared persistent state this instance asks the host to allocate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfiguredStateV1 {
    pub state_id: Vec<u8>,
    pub byte_length: u64,
    pub alignment: u64,
}

/// Everything the host told this instance about the topology it serves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleConfigurationV1 {
    pub topology_id: Vec<u8>,
    pub device_ordinals: Vec<i32>,
    pub kv: ConfiguredKvLayoutV1,
    pub state: ConfiguredStateV1,
    pub component_roles: Vec<Vec<u8>>,
    pub scratch_multiplier: u64,
}

impl BundleConfigurationV1 {
    pub fn decode(bytes: &[u8]) -> Result<Self, WireErrorV1> {
        let mut reader = WireReaderV1::new("instance configuration", bytes);
        reader.expect_magic(BUNDLE_CONFIGURATION_MAGIC_V1)?;
        let topology_id = reader.identifier("topology identifier")?;
        let rank_count = reader.count("topology rank count")?;
        let mut device_ordinals = Vec::with_capacity(rank_count);
        for _ in 0..rank_count {
            device_ordinals.push(reader.i32()?);
        }
        let kv = ConfiguredKvLayoutV1 {
            layout_id: reader.identifier("KV layout identifier")?,
            page_bytes: reader.u64()?,
            page_alignment: reader.u64()?,
            page_tokens: reader.u64()?,
            rank_count: reader.u64()?,
            state_lifetime: reader.u64()?,
        };
        let state = ConfiguredStateV1 {
            state_id: reader.identifier("state identifier")?,
            byte_length: reader.u64()?,
            alignment: reader.u64()?,
        };
        let role_count = reader.count("component role count")?;
        let mut component_roles = Vec::with_capacity(role_count);
        for _ in 0..role_count {
            component_roles.push(reader.identifier("component role")?);
        }
        let scratch_multiplier = if reader.offset == bytes.len() {
            1
        } else if bytes[reader.offset..].starts_with(runtime_bundle_abi::MODEL_OPTIONS_TAG_V1) {
            let mut options_reader =
                runtime_bundle_sdk::BatchReaderV1::new("model options", &bytes[reader.offset..]);
            let values =
                runtime_bundle_sdk::model_options::read(&mut options_reader).map_err(|_| {
                    WireErrorV1::Magic {
                        surface: "model options",
                    }
                })?;
            options_reader
                .finish()
                .map_err(|_| WireErrorV1::TrailingBytes {
                    surface: "model options",
                    remaining: bytes.len() - reader.offset,
                })?;
            if values.len() != 1 {
                return Err(WireErrorV1::Magic {
                    surface: "model options names",
                });
            }
            let multiplier = values
                .get("scratch_multiplier")
                .and_then(serde_json::Value::as_u64)
                .filter(|n| (1..=4).contains(n))
                .ok_or(WireErrorV1::Magic {
                    surface: "model options compiled limits",
                })?;
            reader.offset = bytes.len();
            multiplier
        } else {
            1
        };
        reader.finish()?;
        Ok(Self {
            topology_id,
            device_ordinals,
            kv,
            state,
            component_roles,
            scratch_multiplier,
        })
    }

    #[must_use]
    pub fn rank_count(&self) -> usize {
        self.device_ordinals.len()
    }

    #[must_use]
    pub fn declares_component_role(&self, role: &[u8]) -> bool {
        self.component_roles.iter().any(|declared| declared == role)
    }
}

#[cfg(test)]
mod tests {
    use super::{BUNDLE_CONFIGURATION_MAGIC_V1, BundleConfigurationV1, WireErrorV1, WireReaderV1};

    fn configuration_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BUNDLE_CONFIGURATION_MAGIC_V1);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(b"tp4");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        for ordinal in 0..4_i32 {
            bytes.extend_from_slice(&ordinal.to_le_bytes());
        }
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(b"kv");
        for value in [128_u64, 128, 32, 4, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&5_u32.to_le_bytes());
        bytes.extend_from_slice(b"state");
        bytes.extend_from_slice(&8_u64.to_le_bytes());
        bytes.extend_from_slice(&64_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&7_u32.to_le_bytes());
        bytes.extend_from_slice(b"program");
        bytes
    }

    #[test]
    fn historical_configuration_magic_is_rejected() {
        let mut bytes = configuration_bytes();
        assert!(BundleConfigurationV1::decode(&bytes).is_ok());
        bytes[..20].copy_from_slice(b"\x72\x74\x78-bundle-config-v1");
        assert!(BundleConfigurationV1::decode(&bytes).is_err());
    }

    #[test]
    fn a_configuration_round_trips_every_declared_field() {
        let configuration =
            BundleConfigurationV1::decode(&configuration_bytes()).expect("the record decodes");

        assert_eq!(configuration.topology_id, b"tp4");
        assert_eq!(configuration.device_ordinals, [0, 1, 2, 3]);
        assert_eq!(configuration.rank_count(), 4);
        assert_eq!(configuration.kv.rank_count, 4);
        assert_eq!(configuration.kv.page_tokens, 32);
        assert_eq!(configuration.kv.page_bytes, 128);
        assert_eq!(configuration.state.byte_length, 8);
        assert!(configuration.declares_component_role(b"program"));
        assert!(!configuration.declares_component_role(b"absent"));
    }

    #[test]
    fn a_truncated_or_padded_configuration_is_refused() {
        let bytes = configuration_bytes();

        assert!(matches!(
            BundleConfigurationV1::decode(&bytes[..bytes.len() - 1]),
            Err(WireErrorV1::Truncated { .. })
        ));
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(matches!(
            BundleConfigurationV1::decode(&padded),
            Err(WireErrorV1::TrailingBytes { remaining: 1, .. })
        ));
        let mut wrong = bytes;
        wrong[0] = b'x';
        assert!(matches!(
            BundleConfigurationV1::decode(&wrong),
            Err(WireErrorV1::Magic { .. })
        ));
    }

    #[test]
    fn an_unbounded_element_count_is_refused_before_any_reservation() {
        let mut reader = WireReaderV1::new("test", &[0xff, 0xff, 0xff, 0xff]);

        assert!(matches!(
            reader.count("test count"),
            Err(WireErrorV1::CountOutOfRange { .. })
        ));
    }
}
