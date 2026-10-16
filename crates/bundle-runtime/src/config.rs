//! The host's `chopbundle-config-v1` create record, read without judging it.
//!
//! The record names the topology and its device ordinals, the serving KV
//! layout, the persistent state layout, the component roles the package
//! holds and, last, the items the instance serves. This reader checks the
//! topology and rank count against the bundle's supported topologies, and the
//! item count against its package maximum, as it reads, then returns the rest
//! for the bundle to check against its own layouts.

use runtime_bundle_abi::BundleStatusV1;
use runtime_bundle_sdk::BatchReaderV1;
use runtime_bundle_sdk::context_parallel::{self, ContextParallelV1};

use crate::geometry::{BundleGeometry, Items, Ranks};

pub const MAGIC: &[u8] = b"chopbundle-config-v1";

/// The serving KV layout the host prepared.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvLayout {
    pub layout_id: Vec<u8>,
    pub page_bytes: u64,
    pub alignment: u64,
    pub page_tokens: u64,
    pub ranks: u64,
    pub lifetime: u64,
}

/// The persistent state layout the host allocates per rank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateLayout {
    pub state_id: Vec<u8>,
    pub bytes: u64,
    pub alignment: u64,
}

/// One topology a bundle serves: the package topology id and its rank count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Topology<'a> {
    pub id: &'a str,
    pub ranks: Ranks,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CreateConfig {
    /// The rank count of the record's topology.
    pub ranks: Ranks,
    /// One device ordinal per rank.
    pub ordinals: Vec<i32>,
    pub kv: KvLayout,
    pub state: StateLayout,
    /// The package's component roles, in record order.
    pub roles: Vec<Vec<u8>>,
    /// The items the instance serves: the host's `--max-sequences`, or the
    /// package maximum from a record that names none.
    pub items: Items,
    /// Served context when the package requires chop.served-context.v1.
    pub context_tokens: Option<u64>,
    /// The context-parallel groups the ranks form; one rank per group when
    /// the record names no size.
    pub context_parallel: ContextParallelV1,
    /// The serving process's startup nonce, the record's optional last field
    /// (`runtime_bundle_sdk::taps::PRODUCER_TAG`), which a host sends when
    /// taps are requested; a diagnostic bundle's taps record it.
    pub tap_producer: Option<String>,
}

impl CreateConfig {
    /// Refuses a served context above the model's validated maximum.
    pub fn validate_context(&self, limits: crate::ModelLimits) -> Result<(), BundleStatusV1> {
        if self
            .context_tokens
            .is_some_and(|tokens| tokens == 0 || tokens > limits.context_tokens())
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(())
    }
}

fn id(reader: &mut BatchReaderV1<'_>) -> Result<Vec<u8>, BundleStatusV1> {
    let len = reader
        .count("identifier")
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    Ok(reader
        .take(len)
        .map_err(|_| BundleStatusV1::InvalidArgument)?
        .to_vec())
}

fn u64_value(reader: &mut BatchReaderV1<'_>) -> Result<u64, BundleStatusV1> {
    reader.u64().map_err(|_| BundleStatusV1::InvalidArgument)
}

/// Read the topology identifier before choosing model-owned admission rules.
pub fn topology_id(bytes: &[u8]) -> Result<&str, BundleStatusV1> {
    let mut reader = BatchReaderV1::new("bundle configuration", bytes);
    reader
        .expect_magic(MAGIC)
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let length = reader
        .count("topology identifier")
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let bytes = reader
        .take(length)
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    std::str::from_utf8(bytes).map_err(|_| BundleStatusV1::InvalidArgument)
}

/// Reads one create record for one of `topologies`, the bundle's supported
/// list, for a bundle of geometry `G`. A malformed record or repeated device
/// ordinal is `InvalidArgument`: each rank needs its own device ordinal.
/// A topology outside the list, a rank count other than the
/// listed topology's, or more items than `G::DECODE_BATCH` is `Unsupported`.
///
/// The item count is optional; a context-aware package can also receive a
/// trailing u64 context limit. A record without the item count
/// (from a host that predates it) serves `G`'s package maximum. Last, a
/// record may end with the tap producer field: `PRODUCER_TAG` and the
/// serving process's startup nonce. A record that names a context-parallel
/// size is `Unsupported` here; see [`decode_with_context_parallel`].
pub fn decode<G: BundleGeometry>(
    bytes: &[u8],
    topologies: &[Topology<'_>],
) -> Result<CreateConfig, BundleStatusV1> {
    decode_with_log::<G>(bytes, topologies, |_| {})
}

/// Reads a create record like [`decode`], logging the reason when repeated
/// device ordinals are refused. Bundles use this to explain a stale host's
/// create refusal without changing the status returned through the ABI.
pub fn decode_with_log<G: BundleGeometry>(
    bytes: &[u8],
    topologies: &[Topology<'_>],
    log: impl Fn(&str),
) -> Result<CreateConfig, BundleStatusV1> {
    decode_with_context_parallel::<G>(bytes, topologies, &[], log)
}

/// Reads a create record like [`decode_with_log`] for a bundle whose
/// topologies serve the context-parallel group sizes `context_parallel_sizes`
/// (1 may be listed or not). A record naming another size is `Unsupported`.
pub fn decode_with_context_parallel<G: BundleGeometry>(
    bytes: &[u8],
    topologies: &[Topology<'_>],
    context_parallel_sizes: &[u32],
    log: impl Fn(&str),
) -> Result<CreateConfig, BundleStatusV1> {
    let mut reader = BatchReaderV1::new("bundle configuration", bytes);
    reader
        .expect_magic(MAGIC)
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let topology_id = id(&mut reader)?;
    let ranks = topologies
        .iter()
        .find(|topology| topology.id.as_bytes() == topology_id)
        .ok_or(BundleStatusV1::Unsupported)?
        .ranks;
    reader
        .exact_count("rank count", ranks.get().into())
        .map_err(|_| BundleStatusV1::Unsupported)?;
    let ordinals = (0..ranks.get())
        .map(|_| {
            reader
                .u32()
                .map(|ordinal| ordinal as i32)
                .map_err(|_| BundleStatusV1::InvalidArgument)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ordinals
        .iter()
        .enumerate()
        .any(|(rank, ordinal)| ordinals[..rank].contains(ordinal))
    {
        log("each rank needs its own device ordinal");
        return Err(BundleStatusV1::InvalidArgument);
    }
    let kv = KvLayout {
        layout_id: id(&mut reader)?,
        page_bytes: u64_value(&mut reader)?,
        alignment: u64_value(&mut reader)?,
        page_tokens: u64_value(&mut reader)?,
        ranks: u64_value(&mut reader)?,
        lifetime: u64_value(&mut reader)?,
    };
    let state = StateLayout {
        state_id: id(&mut reader)?,
        bytes: u64_value(&mut reader)?,
        alignment: u64_value(&mut reader)?,
    };
    let count = reader
        .count("component roles")
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let roles = (0..count)
        .map(|_| id(&mut reader))
        .collect::<Result<Vec<_>, _>>()?;
    let items = if reader.remaining() == 0 {
        Items::package::<G>()
    } else {
        let count = reader.u32().map_err(|_| BundleStatusV1::InvalidArgument)?;
        if count == 0 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Items::within::<G>(count).ok_or(BundleStatusV1::Unsupported)?
    };
    let rest = |reader: &BatchReaderV1<'_>| &bytes[bytes.len() - reader.remaining()..];
    let tagged_next = |reader: &BatchReaderV1<'_>| {
        rest(reader).starts_with(runtime_bundle_sdk::taps::PRODUCER_TAG)
            || context_parallel::is_next(rest(reader))
    };
    let context_tokens = if reader.remaining() == 0 || tagged_next(&reader) {
        None
    } else {
        Some({
            let tokens = u64_value(&mut reader)?;
            if tokens == 0 {
                return Err(BundleStatusV1::InvalidArgument);
            }
            tokens
        })
    };
    let context_parallel = if context_parallel::is_next(rest(&reader)) {
        context_parallel::read(&mut reader, ranks.get(), context_parallel_sizes)?
    } else {
        ContextParallelV1::NONE
    };
    let tap_producer = if reader.remaining() == 0 {
        None
    } else {
        reader
            .expect_magic(runtime_bundle_sdk::taps::PRODUCER_TAG)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let nonce = id(&mut reader)?;
        Some(String::from_utf8(nonce).map_err(|_| BundleStatusV1::InvalidArgument)?)
    };
    reader
        .finish()
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    Ok(CreateConfig {
        ranks,
        ordinals,
        kv,
        state,
        roles,
        items,
        context_tokens,
        context_parallel,
        tap_producer,
    })
}

/// Refuses, at create, a host that schedules more scored logits rows per
/// prefill phase than the bundle's `capacity`, the rows its buffers hold.
///
/// The host reports its manifest's limit; a host that reports none (zero)
/// is accepted, and execute still refuses an oversized phase.
pub fn check_scored_rows(
    services: &runtime_bundle_abi::HostServicesV1,
    bundle: &str,
    capacity: u32,
) -> Result<(), BundleStatusV1> {
    let limits = services
        .query_batch_limits()
        .map_err(crate::resources::status)?;
    if limits.maximum_scored_rows > capacity {
        let _ = services.log(
            format!(
                "{bundle}: the manifest's maximum_scored_rows is {}, but this bundle holds {capacity} scored rows per prefill phase; package the bundle with a limit of at most {capacity}",
                limits.maximum_scored_rows
            )
            .as_bytes(),
        );
        return Err(BundleStatusV1::Unsupported);
    }
    Ok(())
}

/// Test-only encoder for the host create wire record, including malformed values.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::{KvLayout, MAGIC, StateLayout};

    pub fn encode(
        topology: &[u8],
        ordinals: &[i32],
        kv: &KvLayout,
        state: &StateLayout,
        roles: &[Vec<u8>],
        items: Option<u32>,
    ) -> Vec<u8> {
        fn field(bytes: &mut Vec<u8>, value: &[u8]) {
            bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value);
        }
        let mut bytes = MAGIC.to_vec();
        field(&mut bytes, topology);
        bytes.extend_from_slice(&(ordinals.len() as u32).to_le_bytes());
        for ordinal in ordinals {
            bytes.extend_from_slice(&ordinal.to_le_bytes());
        }
        field(&mut bytes, &kv.layout_id);
        for value in [
            kv.page_bytes,
            kv.alignment,
            kv.page_tokens,
            kv.ranks,
            kv.lifetime,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        field(&mut bytes, &state.state_id);
        for value in [state.bytes, state.alignment] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&(roles.len() as u32).to_le_bytes());
        for role in roles {
            field(&mut bytes, role);
        }
        if let Some(items) = items {
            bytes.extend_from_slice(&items.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::SUPPORTED_RANKS;
    use crate::geometry::tests::{Max, Tp4, every_item_count};

    #[test]
    fn served_context_respects_model_limits() {
        let base = record(b"example-tp2", 2);
        let limits = crate::ModelLimits::new(43, 128, 64).unwrap();
        assert_eq!(
            decode::<Tp4>(&base, &TP2).unwrap().validate_context(limits),
            Ok(())
        );
        for (tokens, expected) in [
            (128_u64, Ok(())),
            (129, Err(BundleStatusV1::InvalidArgument)),
        ] {
            let mut bytes = base.clone();
            bytes.extend_from_slice(&2_u32.to_le_bytes());
            bytes.extend_from_slice(&tokens.to_le_bytes());
            assert_eq!(
                decode::<Tp4>(&bytes, &TP2)
                    .unwrap()
                    .validate_context(limits),
                expected
            );
        }
    }

    fn record(topology: &[u8], ranks: u32) -> Vec<u8> {
        test_support::encode(
            topology,
            &(4..4 + ranks as i32).collect::<Vec<_>>(),
            &KvLayout {
                layout_id: b"kv.v1".to_vec(),
                page_bytes: 4096,
                alignment: 256,
                page_tokens: 64,
                ranks: ranks.into(),
                lifetime: 0,
            },
            &StateLayout {
                state_id: b"state.v1".to_vec(),
                bytes: 1024,
                alignment: 256,
            },
            &[b"images.v1".to_vec(), b"index.r0.v1".to_vec()],
            None,
        )
    }

    fn with_items(mut bytes: Vec<u8>, items: u32) -> Vec<u8> {
        bytes.extend_from_slice(&items.to_le_bytes());
        bytes
    }

    const TP2: [Topology<'static>; 1] = [Topology {
        id: "example-tp2",
        ranks: Ranks::TP2,
    }];

    #[test]
    fn topology_reader_checks_the_prefix_and_utf8() {
        let bytes = record(b"example-tp2", 2);
        assert_eq!(topology_id(&bytes).unwrap(), "example-tp2");
        for length in 0..MAGIC.len() + 4 + b"example-tp2".len() {
            assert_eq!(
                topology_id(&bytes[..length]),
                Err(BundleStatusV1::InvalidArgument)
            );
        }
        assert_eq!(
            topology_id(&record(&[255], 2)),
            Err(BundleStatusV1::InvalidArgument)
        );
    }

    #[test]
    fn repeated_device_ordinals_are_refused() {
        let mut bytes = record(b"example-tp2", 2);
        let first = MAGIC.len() + 4 + b"example-tp2".len() + 4;
        bytes[first + 4..first + 8].copy_from_slice(&4_i32.to_le_bytes());
        assert_eq!(
            decode::<Tp4>(&bytes, &TP2),
            Err(BundleStatusV1::InvalidArgument)
        );
    }

    #[test]
    fn served_context_is_optional_nonzero_and_exactly_sized() {
        let base = with_items(record(b"example-tp2", 2), 16);
        assert_eq!(decode::<Tp4>(&base, &TP2).unwrap().context_tokens, None);
        let mut bytes = base.clone();
        bytes.extend_from_slice(&65536_u64.to_le_bytes());
        assert_eq!(
            decode::<Tp4>(&bytes, &TP2).unwrap().context_tokens,
            Some(65536)
        );
        bytes.push(0);
        assert!(decode::<Tp4>(&bytes, &TP2).is_err());
        let mut zero = base;
        zero.extend_from_slice(&0_u64.to_le_bytes());
        assert!(decode::<Tp4>(&zero, &TP2).is_err());
    }

    #[test]
    fn the_tap_producer_ends_a_record_with_or_without_served_context() {
        let producer = |mut bytes: Vec<u8>, nonce: &[u8]| {
            bytes.extend_from_slice(runtime_bundle_sdk::taps::PRODUCER_TAG);
            bytes.extend_from_slice(&(nonce.len() as u32).to_le_bytes());
            bytes.extend_from_slice(nonce);
            bytes
        };
        let base = with_items(record(b"example-tp2", 2), 16);
        assert_eq!(decode::<Tp4>(&base, &TP2).unwrap().tap_producer, None);
        let bare = decode::<Tp4>(&producer(base.clone(), b"ab12"), &TP2).unwrap();
        assert_eq!(
            (bare.context_tokens, bare.tap_producer.as_deref()),
            (None, Some("ab12"))
        );
        let mut context = base;
        context.extend_from_slice(&65536_u64.to_le_bytes());
        let both = decode::<Tp4>(&producer(context.clone(), b"ab12"), &TP2).unwrap();
        assert_eq!(
            (both.context_tokens, both.tap_producer.as_deref()),
            (Some(65536), Some("ab12"))
        );
        let mut trailing = producer(context.clone(), b"ab12");
        trailing.push(0);
        assert!(decode::<Tp4>(&trailing, &TP2).is_err());
        assert!(decode::<Tp4>(&producer(context, &[0xff]), &TP2).is_err());
    }

    #[test]
    fn reads_every_field_and_refuses_other_topologies() {
        let config = decode::<Tp4>(&record(b"example-tp2", 2), &TP2).unwrap();
        assert_eq!(config.ranks, Ranks::TP2);
        assert_eq!(config.ordinals, [4, 5]);
        assert_eq!(config.kv.layout_id, b"kv.v1");
        assert_eq!(
            (config.kv.page_bytes, config.kv.page_tokens, config.kv.ranks),
            (4096, 64, 2)
        );
        assert_eq!(config.state.state_id, b"state.v1");
        assert_eq!((config.state.bytes, config.state.alignment), (1024, 256));
        assert_eq!(
            config.roles,
            [b"images.v1".to_vec(), b"index.r0.v1".to_vec()]
        );
        assert_eq!(
            decode::<Tp4>(&record(b"other-tp2", 2), &TP2),
            Err(BundleStatusV1::Unsupported)
        );
        assert_eq!(
            decode::<Tp4>(&record(b"example-tp2", 4), &TP2),
            Err(BundleStatusV1::Unsupported)
        );
        assert_eq!(
            decode::<Tp4>(&record(b"example-tp2", 2), &[]),
            Err(BundleStatusV1::Unsupported)
        );
        let mut trailing = record(b"example-tp2", 2);
        trailing.push(0);
        assert_eq!(
            decode::<Tp4>(&trailing, &TP2),
            Err(BundleStatusV1::InvalidArgument)
        );
        let truncated = record(b"example-tp2", 2);
        assert_eq!(
            decode::<Tp4>(&truncated[..truncated.len() - 1], &TP2),
            Err(BundleStatusV1::InvalidArgument)
        );
    }

    #[test]
    fn reads_the_served_item_count_within_the_package_maximum() {
        let base = record(b"example-tp2", 2);
        // A record without the count serves the package maximum.
        assert_eq!(decode::<Tp4>(&base, &TP2).unwrap().items.get(), 32);
        assert_eq!(decode::<Max>(&base, &TP2).unwrap().items.get(), 256);
        for items in every_item_count() {
            let bytes = with_items(base.clone(), items.get());
            assert_eq!(decode::<Max>(&bytes, &TP2).unwrap().items, items);
            let package = decode::<Tp4>(&bytes, &TP2).map(|config| config.items);
            if items.get() <= 32 {
                assert_eq!(package, Ok(items));
            } else {
                assert_eq!(package, Err(BundleStatusV1::Unsupported));
            }
        }
        assert_eq!(
            decode::<Max>(&with_items(base.clone(), 257), &TP2),
            Err(BundleStatusV1::Unsupported)
        );
        assert_eq!(
            decode::<Max>(&with_items(base.clone(), 0), &TP2),
            Err(BundleStatusV1::InvalidArgument)
        );
        let mut truncated = with_items(base.clone(), 8);
        truncated.pop();
        assert_eq!(
            decode::<Max>(&truncated, &TP2),
            Err(BundleStatusV1::InvalidArgument)
        );
        let mut trailing = with_items(base, 8);
        trailing.push(0);
        assert_eq!(
            decode::<Max>(&trailing, &TP2),
            Err(BundleStatusV1::InvalidArgument)
        );
    }

    #[test]
    fn reads_each_supported_rank_count_from_the_bundle_list() {
        let topologies = [
            Topology {
                id: "example-tp1",
                ranks: Ranks::TP1,
            },
            Topology {
                id: "example-tp2",
                ranks: Ranks::TP2,
            },
            Topology {
                id: "example-tp4",
                ranks: Ranks::TP4,
            },
            Topology {
                id: "example-tp8",
                ranks: Ranks::TP8,
            },
        ];
        for topology in topologies {
            let count = topology.ranks.get();
            let config =
                decode::<Tp4>(&record(topology.id.as_bytes(), count), &topologies).unwrap();
            assert_eq!(config.ranks, topology.ranks);
            assert_eq!(config.ordinals, (4..4 + count as i32).collect::<Vec<_>>());
            assert_eq!(config.kv.ranks, u64::from(count));
            // Another listed topology's rank count under this id is refused.
            for other in SUPPORTED_RANKS.into_iter().filter(|&other| other != count) {
                assert_eq!(
                    decode::<Tp4>(&record(topology.id.as_bytes(), other), &topologies),
                    Err(BundleStatusV1::Unsupported),
                    "{} with {other} ranks",
                    topology.id
                );
            }
        }
        // A bundle that lists only TP4 refuses the others.
        assert_eq!(
            decode::<Tp4>(&record(b"example-tp8", 8), &topologies[2..3]),
            Err(BundleStatusV1::Unsupported)
        );
    }
}
