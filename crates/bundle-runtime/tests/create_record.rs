//! The host's create record against the bundles that read it: bundles that
//! require `chop.served-items.v1` receive the served item count, and bundles
//! built before the field existed receive the record they always read.

use bundle_runtime::BundleGeometry;
use bundle_runtime::config::{self, Topology};
use bundle_runtime::geometry::Ranks;
use runtime_bundle::{BundleBatchLimitsV1, BundleOpaqueId};
use runtime_bundle_abi::{BundleStatusV1, SERVED_ITEMS_CAPABILITY_V1};
use runtime_bundle_sdk::BatchReaderV1;
use runtime_host::{HostInstanceConfigV1, HostTopologyV1, served_sequences_v1};

/// Step limits at MiMo's numbers.
struct Steps;

impl BundleGeometry for Steps {
    const PAGE_TOKENS: u64 = 64;
    const PREFILL_TOKENS: u32 = 8192;
    const DECODE_BATCH: u32 = 32;
    const MAX_VERIFY_ROWS: u32 = 8;
    const TOKEN_GRANULE: u32 = 256;
}

const TP2: [Topology<'static>; 1] = [Topology {
    id: "tp2",
    ranks: Ranks::TP2,
}];

/// The record the host sends a package with these capabilities and limits
/// when serving `requested` sequences.
fn record(
    capabilities: &[&str],
    limits: Option<BundleBatchLimitsV1>,
    requested: Option<u32>,
) -> Result<Vec<u8>, String> {
    let capabilities = capabilities
        .iter()
        .map(|capability| BundleOpaqueId::new(*capability).unwrap())
        .collect::<Vec<_>>();
    let mut config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap());
    if let Some(sequences) =
        served_sequences_v1(&capabilities, limits, requested).map_err(|error| error.to_string())?
    {
        config = config.with_maximum_sequences(sequences);
    }
    Ok(config.encode_v1(b"tp2"))
}

/// The record reader of bundles built before the served item count: every
/// field up to the component roles, then nothing.
fn old_bundle_reads(bytes: &[u8]) -> bool {
    let mut reader = BatchReaderV1::new("bundle configuration", bytes);
    let id = |reader: &mut BatchReaderV1<'_>| {
        let len = reader.count("identifier")?;
        reader.take(len).map(|_| ())
    };
    (|| {
        reader.expect_magic(config::MAGIC)?;
        id(&mut reader)?;
        let ranks = reader.count("ranks")?;
        for _ in 0..ranks {
            reader.u32()?;
        }
        id(&mut reader)?;
        for _ in 0..5 {
            reader.u64()?;
        }
        id(&mut reader)?;
        reader.u64()?;
        reader.u64()?;
        for _ in 0..reader.count("roles")? {
            id(&mut reader)?;
        }
        reader.finish()
    })()
    .is_ok()
}

#[test]
fn only_bundles_that_read_the_served_items_receive_them() {
    let limits = Some(BundleBatchLimitsV1::new(32, 8192).unwrap());
    let base =
        HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap()).encode_v1(b"tp2");

    // An old package (no capability) gets the record it always read, with or
    // without batch limits and whatever --max-sequences says.
    for (capabilities, limits) in [
        (&[][..], limits),
        (&[][..], None),
        (&[SERVED_ITEMS_CAPABILITY_V1][..], None),
    ] {
        for requested in [None, Some(8), Some(32)] {
            let bytes = record(capabilities, limits, requested).unwrap();
            assert_eq!(bytes, base);
            assert!(old_bundle_reads(&bytes));
            assert_eq!(
                config::decode::<Steps>(&bytes, &TP2).unwrap().items.get(),
                32
            );
        }
    }

    // A new package gets the served count, which an old reader would refuse.
    let new = &[SERVED_ITEMS_CAPABILITY_V1][..];
    for (requested, served) in [(None, 32), (Some(1), 1), (Some(8), 8), (Some(32), 32)] {
        let bytes = record(new, limits, requested).unwrap();
        assert_eq!(bytes[..base.len()], base[..]);
        assert_eq!(bytes[base.len()..], u32::to_le_bytes(served));
        assert!(!old_bundle_reads(&bytes));
        assert_eq!(
            config::decode::<Steps>(&bytes, &TP2).unwrap().items.get(),
            served
        );
    }

    // The host refuses counts outside the declaration before any record.
    for requested in [0, 33] {
        assert!(record(new, limits, Some(requested)).is_err());
    }
    // A bundle refuses a count above its package even from a lax host.
    let mut oversized = base.clone();
    oversized.extend_from_slice(&33_u32.to_le_bytes());
    assert_eq!(
        config::decode::<Steps>(&oversized, &TP2).map(|config| config.items),
        Err(BundleStatusV1::Unsupported)
    );
}

/// A group size above 1 reaches only a bundle that declares it; a record at
/// size 1 is the record every bundle already reads.
#[test]
fn only_bundles_that_declare_a_context_parallel_size_accept_it() {
    use runtime_bundle_sdk::context_parallel::{ContextParallelV1, TAG};
    assert_eq!(runtime_host::CONTEXT_PARALLEL_TAG_V1, TAG);
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_maximum_sequences(8)
        .with_maximum_context_tokens(4096);
    let single = config
        .clone()
        .with_context_parallel_size(1)
        .encode_v1(b"tp2");
    assert_eq!(single, config.encode_v1(b"tp2"));
    let decoded = config::decode::<Steps>(&single, &TP2).unwrap();
    assert_eq!(decoded.context_parallel, ContextParallelV1::NONE);
    assert_eq!(decoded.context_tokens, Some(4096));

    let pairs = config.with_context_parallel_size(2).encode_v1(b"tp2");
    let declared =
        config::decode_with_context_parallel::<Steps>(&pairs, &TP2, &[1, 2], |_| {}).unwrap();
    assert_eq!(declared.context_parallel.size(), 2);
    assert_eq!(declared.context_parallel.group_rank(1), 1);
    assert_eq!(declared.items.get(), 8);
    assert_eq!(declared.context_tokens, Some(4096));
    // A bundle that declares no size above 1, as every bundle reading
    // through `decode` does, refuses the group.
    assert_eq!(
        config::decode::<Steps>(&pairs, &TP2).map(|config| config.context_parallel),
        Err(BundleStatusV1::Unsupported)
    );
    assert_eq!(
        config::decode_with_context_parallel::<Steps>(&pairs, &TP2, &[1], |_| {})
            .map(|config| config.context_parallel),
        Err(BundleStatusV1::Unsupported)
    );
}
