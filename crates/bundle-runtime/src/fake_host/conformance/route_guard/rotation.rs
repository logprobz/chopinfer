//! Bounded CPU coverage. Every boundary triple visits every eligible secondary
//! profile/mode combination within one printed, deterministic rotation period.

use super::{Envelope, Tier, boundary_probes, probes, sweep::modes};
use kernel_core::route::RouteTag;
use std::collections::{BTreeMap, BTreeSet};

type Key = (String, String);

pub(super) struct Profile {
    pub name: String,
    pub envelope: Envelope,
    pub tags: Vec<RouteTag>,
}

pub(super) struct Rotation {
    rows: BTreeMap<Key, BTreeSet<u32>>,
    centers: BTreeMap<Key, Vec<u32>>,
    pub period: usize,
    seed: u64,
}

fn point_offset(point: u32) -> u64 {
    u64::from(point)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .rotate_right(17)
}

fn pick(point: u32, phase: u64, count: usize) -> usize {
    ((point_offset(point) % count as u64 + phase) % count as u64) as usize
}

impl Rotation {
    pub fn new(profiles: &[Profile], seed: u64) -> Self {
        assert!(!profiles.is_empty());
        let mut rows = BTreeMap::<Key, BTreeSet<u32>>::new();
        let mut boundaries = BTreeSet::new();
        for profile in profiles {
            let envelope = &profile.envelope;
            boundaries.extend(
                super::boundary_points(envelope, &profile.tags)
                    .into_iter()
                    .filter(|&row| row > 0 && row <= envelope.prefill_tokens),
            );
            rows.insert(
                (profile.name.clone(), "single".into()),
                boundary_probes(envelope, &profile.tags),
            );
            for (mode, _, prompts) in modes(envelope).into_iter().skip(1) {
                let mut fixed: BTreeSet<_> = envelope.probes.iter().copied().collect();
                fixed.extend([
                    prompts,
                    prompts.saturating_add(1),
                    envelope.prefill_tokens.saturating_sub(1),
                    envelope.prefill_tokens,
                ]);
                fixed.retain(|&row| row >= prompts && row <= envelope.prefill_tokens);
                rows.insert((profile.name.clone(), mode), fixed);
            }
        }
        let choices: Vec<_> = boundaries
            .into_iter()
            .map(|point| {
                let eligible: Vec<_> = profiles
                    .iter()
                    .flat_map(|profile| {
                        modes(&profile.envelope).into_iter().skip(1).filter_map(
                            move |(mode, _, prompts)| {
                                (point >= prompts && point <= profile.envelope.prefill_tokens).then(
                                    || {
                                        (
                                            (profile.name.clone(), mode),
                                            prompts,
                                            profile.envelope.prefill_tokens,
                                        )
                                    },
                                )
                            },
                        )
                    })
                    .collect();
                (point, eligible)
            })
            .collect();
        // A common phase makes the entire plan repeat after `period` runs.
        // Every eligible list is no longer than the phase's domain, so each
        // boundary visits all its combinations within any full period.
        let period = choices
            .iter()
            .map(|(_, eligible)| eligible.len())
            .max()
            .unwrap_or(0)
            .max(profiles.len());
        let phase = seed % period as u64;
        let mut centers = BTreeMap::<Key, Vec<u32>>::new();
        for (point, eligible) in choices {
            if eligible.is_empty() {
                continue;
            }
            let (key, prompts, maximum) = &eligible[pick(point, phase, eligible.len())];
            rows.get_mut(key).unwrap().extend(
                [point.saturating_sub(1), point, point.saturating_add(1)]
                    .into_iter()
                    .filter(|row| row >= prompts && row <= maximum),
            );
            centers.entry(key.clone()).or_default().push(point);
        }
        // Dense probes need no repeated long-prefix reconstruction. Rotate
        // them among the eligible single-prompt profiles instead.
        let dense: BTreeSet<_> = profiles
            .iter()
            .flat_map(|profile| probes(&profile.envelope, &profile.tags, Tier::Cpu))
            .collect();
        for point in dense {
            let eligible: Vec<_> = profiles
                .iter()
                .filter(|profile| point <= profile.envelope.prefill_tokens)
                .collect();
            let profile = eligible[pick(point, phase, eligible.len())];
            rows.get_mut(&(profile.name.clone(), "single".into()))
                .unwrap()
                .insert(point);
        }
        Self {
            rows,
            centers,
            period,
            seed,
        }
    }

    pub fn rows(&self, profile: &str, mode: &str) -> Vec<u32> {
        self.rows[&(profile.into(), mode.into())]
            .iter()
            .copied()
            .collect()
    }

    pub fn report(&self, bundle: &str, profile: &str) {
        let selected: Vec<_> = self
            .rows
            .iter()
            .filter(|((name, mode), _)| name == profile && mode != "single")
            .map(|(key, rows)| {
                serde_json::json!({
                    "mode": key.1,
                    "rows": rows,
                    "boundary_centers": self.centers.get(key).cloned().unwrap_or_default(),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({"schema": "chopinfer.route-rotation.v1", "bundle": bundle,
            "profile": profile, "seed": self.seed, "rotation_period": self.period, "selected_combinations": selected})
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::route::RouteClass;

    fn profiles() -> Vec<Profile> {
        [64, 128]
            .into_iter()
            .enumerate()
            .map(|(index, maximum)| Profile {
                name: format!("profile-{index}"),
                envelope: Envelope {
                    prefill_tokens: maximum,
                    chunk_offsets: vec![8192, 40960],
                    packed_prompts: 8,
                    decode_buckets: vec![1, 8],
                    verify: vec![],
                    probes: vec![17, 47],
                },
                tags: vec![RouteTag::new(
                    "base",
                    RouteClass::Baseline,
                    "projection",
                    [32..=32, 96..=96],
                )],
            })
            .collect()
    }

    #[test]
    fn every_eligible_triple_visits_every_secondary_combination_in_one_period() {
        let profiles = profiles();
        let first = Rotation::new(&profiles, 0);
        let mut seen = BTreeSet::new();
        for seed in 0..first.period as u64 {
            let rotation = Rotation::new(&profiles, seed);
            assert_eq!(rotation.period, first.period);
            for (key, points) in &rotation.centers {
                let profile = profiles.iter().find(|p| p.name == key.0).unwrap();
                let (_, _, minimum) = modes(&profile.envelope)
                    .into_iter()
                    .find(|m| m.0 == key.1)
                    .unwrap();
                for &point in points {
                    for row in [point.saturating_sub(1), point, point.saturating_add(1)] {
                        if row >= minimum && row <= profile.envelope.prefill_tokens {
                            assert!(rotation.rows[key].contains(&row));
                        }
                    }
                    seen.insert((key.clone(), point));
                }
            }
            for profile in &profiles {
                let single = rotation.rows(&profile.name, "single");
                assert!(
                    boundary_probes(&profile.envelope, &profile.tags)
                        .iter()
                        .all(|r| single.contains(r))
                );
                for (mode, _, minimum) in modes(&profile.envelope).into_iter().skip(1) {
                    let rows = rotation.rows(&profile.name, &mode);
                    assert!(
                        rows.contains(&minimum) && rows.contains(&profile.envelope.prefill_tokens)
                    );
                    assert!(profile.envelope.probes.iter().all(|r| rows.contains(r)));
                }
            }
        }
        let points: BTreeSet<_> = profiles
            .iter()
            .flat_map(|p| super::super::boundary_points(&p.envelope, &p.tags))
            .filter(|&p| p > 0 && p <= 128)
            .collect();
        for profile in &profiles {
            for (mode, _, minimum) in modes(&profile.envelope).into_iter().skip(1) {
                for &point in points
                    .iter()
                    .filter(|&&p| p >= minimum && p <= profile.envelope.prefill_tokens)
                {
                    assert!(seen.contains(&((profile.name.clone(), mode.clone()), point)));
                }
            }
        }
        let repeated = Rotation::new(&profiles, first.period as u64);
        assert_eq!(first.rows, repeated.rows);
        assert_eq!(first.centers, repeated.centers);
    }
}
