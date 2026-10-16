//! What the retired rank slicer produced for the served comparisons, as the
//! digests in `fixtures/projections/retired-slicer.sha256`. A comparison
//! adds canonical lines under labels (comparison, sweep, topology, rank,
//! use); each label's SHA-256 over its lines must equal the recorded one.
use kernel_core::precision::OperationUse;
use kernel_core::weight_transform::{WeightRecord, WeightTransform};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const RECORDED: &str = include_str!("../fixtures/projections/retired-slicer.sha256");

fn recorded() -> &'static BTreeMap<&'static str, &'static str> {
    static RECORDED_DIGESTS: std::sync::OnceLock<BTreeMap<&str, &str>> = std::sync::OnceLock::new();
    RECORDED_DIGESTS.get_or_init(|| {
        RECORDED
            .lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(|line| {
                let (digest, label) = line.split_once("  ").expect("digest  label");
                (label, digest)
            })
            .collect()
    })
}

/// One comparison's lines, by label.
#[derive(Default)]
pub struct Lines(BTreeMap<String, Sha256>);

impl Lines {
    pub fn add(&mut self, label: impl Into<String>, line: impl AsRef<str>) {
        let digest = self.0.entry(label.into()).or_default();
        digest.update(line.as_ref().as_bytes());
        digest.update(b"\n");
    }

    /// Every label under `scope` (equal to it, or it and a space) is a
    /// recorded label with the recorded digest, and every recorded label
    /// under `scope` is produced.
    pub fn check(self, scope: &str) {
        let within = |label: &str| {
            label == scope
                || label
                    .strip_prefix(scope)
                    .is_some_and(|rest| rest.starts_with(' '))
        };
        let produced: BTreeMap<String, String> = self
            .0
            .into_iter()
            .map(|(label, digest)| (label, format!("{:x}", digest.finalize())))
            .collect();
        assert!(
            produced.keys().all(|label| within(label)),
            "labels outside {scope}"
        );
        let expected: BTreeMap<String, String> = recorded()
            .iter()
            .filter(|(label, _)| within(label))
            .map(|(label, digest)| (label.to_string(), digest.to_string()))
            .collect();
        assert!(!expected.is_empty(), "no recorded digests for {scope}");
        let differing: Vec<_> = expected
            .keys()
            .chain(produced.keys())
            .filter(|label| expected.get(*label) != produced.get(*label))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert!(
            differing.is_empty(),
            "outputs differ from the retired slicer's: {differing:?}"
        );
    }
}

/// A record's transform, its encoded parameters, sources, selection and
/// stored shape. A checked scale copy is the retired slicer's plain copy: it
/// stores the same bytes and only refuses scales outside their domain.
pub fn record(record: &WeightRecord) -> String {
    let transform = match record.transform {
        WeightTransform::CopyScales => WeightTransform::Copy,
        ref transform => transform.clone(),
    };
    let sources: Vec<String> = record
        .sources
        .iter()
        .map(|s| format!("{}{:?}:{}", s.dtype, s.shape, s.byte_length))
        .collect();
    format!(
        "transform={} params={} sources={} axis={:?} ranges={:?} stored={:?} bytes={}",
        transform.id(),
        hex(&transform.encode_params().unwrap()),
        sources.join(","),
        record.axis,
        record.ranges,
        record.stored_shape,
        record.output_bytes().unwrap()
    )
}

/// The SHA-256 of payload bytes.
pub fn bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A use's phase, capacity and graph capture, as a label.
pub fn usage(usage: &OperationUse) -> String {
    format!(
        "{:?}-{}{}",
        usage.phase,
        usage.capacity,
        if usage.graph { "-graph" } else { "" }
    )
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
