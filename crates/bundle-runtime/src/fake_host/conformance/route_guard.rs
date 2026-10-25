//! CPU coverage of a model's declared served shapes through its real bundle ABI.

mod allowlist;
mod audit;
mod budget;
mod pieces;
mod rotation;
mod sweep;

use super::Serving;
use kernel_core::route::RouteTag;
use runtime_bundle_sdk::warmup::{VerifyShape, verify_shapes, waves};
pub use sweep::{memory_regression, sweep};

#[derive(Clone, Debug)]
pub struct Envelope {
    pub prefill_tokens: u32,
    pub chunk_offsets: Vec<u32>,
    pub packed_prompts: u32,
    pub decode_buckets: Vec<u32>,
    pub verify: Vec<VerifyShape>,
    pub probes: Vec<u32>,
}

impl Envelope {
    pub fn served(serving: &Serving, prefill_tokens: u32) -> Self {
        assert!(prefill_tokens > 0 && serving.sequences > 0);
        Self {
            prefill_tokens,
            chunk_offsets: Vec::new(),
            packed_prompts: serving.sequences.min(prefill_tokens),
            decode_buckets: waves(serving.sequences as usize)
                .into_iter()
                .map(|n| n as u32)
                .collect(),
            verify: serving
                .verify_warmup()
                .map_or_else(Vec::new, |v| verify_shapes(serving.sequences as usize, &v)),
            probes: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    Quick,
    Cpu,
    Full,
    Exhaustive,
}

impl Tier {
    fn configured() -> Self {
        match std::env::var("CHOPINFER_ROUTE_SWEEP").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("quick") => Self::Quick,
            Ok("cpu") => Self::Cpu,
            Ok("full") => Self::Full,
            Ok("exhaustive") => Self::Exhaustive,
            other => panic!("invalid CHOPINFER_ROUTE_SWEEP: {other:?}"),
        }
    }
}

fn probes(envelope: &Envelope, tags: &[RouteTag], tier: Tier) -> Vec<u32> {
    use std::collections::BTreeSet;
    let n = envelope.prefill_tokens;
    if matches!(tier, Tier::Full | Tier::Exhaustive) {
        return (1..=n).collect();
    }
    let mut rows: BTreeSet<u32> = (1..=64.min(n)).collect();
    rows.extend((1..=n).filter(|r| r % if tier == Tier::Quick { 512 } else { 16 } == 0));
    rows.extend(&envelope.probes);
    rows.insert(n);
    let boundaries = (0..32).map(|i| 1u32 << i).chain(
        tags.iter()
            .flat_map(|t| t.admitted.iter().flat_map(|r| [*r.start(), *r.end()])),
    );
    for boundary in boundaries {
        rows.extend([
            boundary.saturating_sub(1),
            boundary,
            boundary.saturating_add(1),
        ]);
    }
    rows.into_iter().filter(|&r| r > 0 && r <= n).collect()
}

fn boundary_probes(envelope: &Envelope, tags: &[RouteTag]) -> std::collections::BTreeSet<u32> {
    let mut rows: std::collections::BTreeSet<_> = envelope.probes.iter().copied().collect();
    for point in boundary_points(envelope, tags) {
        rows.extend([point.saturating_sub(1), point, point.saturating_add(1)]);
    }
    rows.retain(|&r| r > 0 && r <= envelope.prefill_tokens);
    rows
}

fn boundary_points(envelope: &Envelope, tags: &[RouteTag]) -> std::collections::BTreeSet<u32> {
    [1, envelope.prefill_tokens]
        .into_iter()
        .chain((0..32).map(|n| 1u32 << n))
        .chain(
            tags.iter()
                .flat_map(|t| t.admitted.iter().flat_map(|r| [*r.start(), *r.end()])),
        )
        .collect()
}

fn seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(|| match std::env::var("CHOPINFER_ROUTE_SEED") {
        Ok(value) => value
            .parse()
            .expect("CHOPINFER_ROUTE_SEED must be an unsigned integer"),
        Err(std::env::VarError::NotPresent) => {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                / 86400
        }
        Err(error) => panic!("invalid CHOPINFER_ROUTE_SEED: {error}"),
    })
}

fn distributed(row: u32, seed: u64, group: usize, groups: usize) -> bool {
    let mut value = seed.wrapping_add(u64::from(row).wrapping_mul(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((value ^ (value >> 31)) % groups as u64) as usize == group
}
