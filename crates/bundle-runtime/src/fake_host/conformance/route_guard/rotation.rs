//! Every CPU run covers all boundaries; only extra dense probes rotate.

use super::{Envelope, Tier, boundary_probes, probes, sweep::modes};
use kernel_core::route::RouteTag;
use std::collections::{BTreeMap, BTreeSet};

type Key = (String, String);

pub(super) struct Profile {
    pub name: String,
    pub envelope: Envelope,
    pub tags: Vec<RouteTag>,
}

pub(super) struct CpuPlan {
    rows: BTreeMap<Key, BTreeSet<u32>>,
    dense: BTreeMap<String, BTreeSet<u32>>,
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

impl CpuPlan {
    pub fn new(profiles: &[Profile], seed: u64) -> Self {
        assert!(!profiles.is_empty());
        // Secondary modes retain boundaries from every profile, including
        // points the current profile did not declare itself.
        let boundaries: BTreeSet<_> = profiles
            .iter()
            .flat_map(|profile| {
                super::boundary_points(&profile.envelope, &profile.tags)
                    .into_iter()
                    .filter(|&row| row > 0 && row <= profile.envelope.prefill_tokens)
            })
            .collect();
        let mut rows = BTreeMap::new();
        for profile in profiles {
            let envelope = &profile.envelope;
            let maximum = envelope.prefill_tokens;
            let single = boundary_probes(envelope, &profile.tags);
            rows.insert((profile.name.clone(), "single".into()), single);
            for (mode, _, prompts) in modes(envelope).into_iter().skip(1) {
                let mut selected: BTreeSet<_> = envelope.probes.iter().copied().collect();
                selected.extend([
                    prompts,
                    prompts.saturating_add(1),
                    maximum.saturating_sub(1),
                    maximum,
                ]);
                for &point in boundaries.range(prompts..=maximum) {
                    selected.extend([point.saturating_sub(1), point, point.saturating_add(1)]);
                }
                selected.retain(|&row| row >= prompts && row <= maximum);
                rows.insert((profile.name.clone(), mode), selected);
            }
        }
        // Preserve the old phase domain so the dense assignment is identical
        // for every seed, even when profiles have different row limits.
        let period = boundaries
            .iter()
            .map(|&point| {
                profiles
                    .iter()
                    .map(|profile| {
                        modes(&profile.envelope)
                            .into_iter()
                            .skip(1)
                            .filter(|(_, _, prompts)| {
                                point >= *prompts && point <= profile.envelope.prefill_tokens
                            })
                            .count()
                    })
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
            .max(profiles.len());
        let phase = seed % period as u64;
        let dense: BTreeSet<_> = profiles
            .iter()
            .flat_map(|profile| probes(&profile.envelope, &profile.tags, Tier::Cpu))
            .collect();
        let mut assigned = BTreeMap::<String, BTreeSet<u32>>::new();
        for point in dense {
            let eligible: Vec<_> = profiles
                .iter()
                .filter(|profile| point <= profile.envelope.prefill_tokens)
                .collect();
            let profile = eligible[pick(point, phase, eligible.len())];
            assigned
                .entry(profile.name.clone())
                .or_default()
                .insert(point);
        }
        Self {
            rows,
            dense: assigned,
            period,
            seed,
        }
    }

    pub fn rows(&self, profile: &str, mode: &str) -> Vec<u32> {
        let mut rows = self.rows[&(profile.into(), mode.into())].clone();
        if mode == "single" {
            rows.extend(self.dense.get(profile).into_iter().flatten());
        }
        rows.into_iter().collect()
    }

    pub fn report(&self, bundle: &str, profile: &str) {
        let combinations: Vec<_> = self
            .rows
            .iter()
            .filter(|((name, _), _)| name == profile)
            .map(|((_, mode), rows)| serde_json::json!({"mode": mode, "boundary_rows": rows,
                "dense_rows": if mode == "single" { self.dense.get(profile).cloned().unwrap_or_default() } else { BTreeSet::new() }}))
            .collect();
        println!(
            "{}",
            serde_json::json!({"schema": "chopinfer.route-plan.v1", "bundle": bundle,
                "profile": profile, "tier": "cpu", "seed": self.seed,
                "dense_rotation_period": self.period, "combinations": combinations})
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
                    allocated: vec![],
                    probes: vec![17, 47],
                    cuts: vec![],
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
    fn one_plan_covers_every_eligible_boundary_triple() {
        let profiles = profiles();
        let plan = CpuPlan::new(&profiles, 0);
        for profile in &profiles {
            let maximum = profile.envelope.prefill_tokens;
            let single = plan.rows(&profile.name, "single");
            for row in [
                1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 95, 96, 97, 127,
                128,
            ] {
                if row <= maximum {
                    assert!(single.contains(&row), "{} single {row}", profile.name);
                }
            }
            for (mode, _, minimum) in modes(&profile.envelope).into_iter().skip(1) {
                let rows = plan.rows(&profile.name, &mode);
                for point in [1_u32, 2, 4, 8, 16, 32, 64, 96, 128] {
                    if point < minimum || point > maximum {
                        continue;
                    }
                    for row in [point.saturating_sub(1), point, point + 1] {
                        if row >= minimum && row <= maximum {
                            assert!(rows.contains(&row), "{} {mode} {row}", profile.name);
                        }
                    }
                }
                for row in [minimum, minimum + 1, 17, 47, maximum - 1, maximum] {
                    assert!(rows.contains(&row), "{} {mode} {row}", profile.name);
                }
                assert!(rows.iter().all(|&row| row >= minimum && row <= maximum));
            }
        }
    }

    #[test]
    fn boundaries_are_fixed_while_dense_probes_keep_the_old_rotation() {
        let profiles = profiles();
        let first = CpuPlan::new(&profiles, 0);
        assert_eq!(first.period, 8);
        assert!(!first.rows("profile-0", "single").contains(&6));
        assert!(first.rows("profile-1", "single").contains(&6));
        let next = CpuPlan::new(&profiles, 1);
        assert!(next.rows("profile-0", "single").contains(&6));
        assert!(!next.rows("profile-1", "single").contains(&6));
        for seed in 0..first.period as u64 {
            let plan = CpuPlan::new(&profiles, seed);
            assert_eq!(first.rows, plan.rows);
            assert_eq!(
                plan.dense,
                CpuPlan::new(&profiles, seed + first.period as u64).dense
            );
        }
    }

    #[test]
    fn boundaries_reach_other_eligible_secondary_profiles() {
        let mut profiles = profiles();
        profiles[1].tags.push(RouteTag::new(
            "other",
            RouteClass::Baseline,
            "projection",
            [53..=53],
        ));
        for seed in 0..8 {
            let plan = CpuPlan::new(&profiles, seed);
            for (mode, _, _) in modes(&profiles[0].envelope).into_iter().skip(1) {
                for row in [52, 53, 54] {
                    assert!(plan.rows("profile-0", &mode).contains(&row));
                }
            }
        }
    }

    #[test]
    fn profile_order_does_not_change_the_deterministic_boundaries() {
        let mut profiles = profiles();
        let plan = CpuPlan::new(&profiles, 0);
        profiles.reverse();
        assert_eq!(plan.rows, CpuPlan::new(&profiles, 0).rows);
    }
}
