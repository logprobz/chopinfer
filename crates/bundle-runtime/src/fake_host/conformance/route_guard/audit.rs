use super::{Tier, allowlist::Allow};
use kernel_core::route::RouteTag;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Sample {
    pub mode: String,
    pub phase: String,
    pub rows: u32,
    pub symbols: BTreeMap<String, u64>,
}

// Owned metadata lets bounded sweep pieces be audited together without keeping
// model instances, launch arguments or tensor backing from earlier shapes.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub(super) struct Tag {
    pub entry: String,
    pub symbol: String,
    pub class: String,
    pub family: String,
    pub admitted: Vec<[u32; 2]>,
    pub wrappers: Vec<String>,
}

impl From<RouteTag> for Tag {
    fn from(tag: RouteTag) -> Self {
        Self {
            entry: tag.entry,
            symbol: tag.symbol,
            class: tag.class.name().into(),
            family: tag.family.into(),
            admitted: tag
                .admitted
                .iter()
                .map(|r| [*r.start(), *r.end()])
                .collect(),
            wrappers: tag.wrappers.into_iter().map(str::to_owned).collect(),
        }
    }
}

pub(super) fn remember(tags: &mut BTreeMap<String, Tag>, tag: Tag) {
    assert!(matches!(
        tag.class.as_str(),
        "Fast" | "Baseline" | "Reference"
    ));
    assert!(!tag.entry.is_empty() && !tag.symbol.is_empty() && !tag.family.is_empty());
    assert!(tag.admitted.iter().all(|r| r[0] > 0 && r[0] <= r[1]));
    if let Some(previous) = tags.get_mut(&tag.entry) {
        assert_eq!(
            previous.class, tag.class,
            "conflicting selected entry class"
        );
        assert_eq!(
            previous.family, tag.family,
            "conflicting selected entry family"
        );
        assert_eq!(
            previous.symbol, tag.symbol,
            "conflicting selected entry symbol"
        );
        if previous == &tag {
            return;
        }
        previous.admitted.extend(tag.admitted);
        previous.admitted.sort_unstable();
        let mut ranges: Vec<[u32; 2]> = Vec::new();
        for range in previous.admitted.drain(..) {
            if let Some(last) = ranges.last_mut()
                && range[0] <= last[1].saturating_add(1)
            {
                last[1] = last[1].max(range[1]);
            } else {
                ranges.push(range);
            }
        }
        previous.admitted = ranges;
        for wrapper in tag.wrappers {
            if !previous.wrappers.contains(&wrapper) {
                previous.wrappers.push(wrapper);
            }
        }
    } else {
        tags.insert(tag.entry.clone(), tag);
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct Failure {
    pub entry: String,
    pub mode: String,
    pub phase: String,
    pub family: String,
    pub symbol: String,
    pub rows: u32,
}

pub(super) fn failures(samples: &[Sample], tags: &[Tag], tier: Tier) -> Vec<Failure> {
    let mut groups = BTreeMap::<(&str, &str), Vec<&Sample>>::new();
    for sample in samples {
        groups
            .entry((&sample.mode, &sample.phase))
            .or_default()
            .push(sample);
    }
    let tagged: BTreeSet<&str> = tags
        .iter()
        .flat_map(|t| [t.symbol.as_str(), t.entry.as_str()])
        .collect();
    let fallbacks: Vec<_> = tags.iter().filter(|t| t.class != "Fast").collect();
    let mut failed = BTreeSet::new();
    for group in groups.values_mut() {
        group.sort_by_key(|s| s.rows);
        let totals: Vec<u64> = group
            .iter()
            .map(|s| {
                fallbacks
                    .iter()
                    .map(|t| s.symbols.get(&t.entry).copied().unwrap_or(0))
                    .sum()
            })
            .collect();
        for tag in &fallbacks {
            let counts: Vec<u64> = group
                .iter()
                .map(|s| s.symbols.get(&tag.entry).copied().unwrap_or(0))
                .collect();
            let mut left = Vec::with_capacity(counts.len());
            let mut best = u64::MAX;
            for &count in &counts {
                left.push(best);
                best = best.min(count);
            }
            let mut right = vec![u64::MAX; counts.len()];
            best = u64::MAX;
            for i in (0..counts.len()).rev() {
                right[i] = best;
                best = best.min(counts[i]);
            }
            for (i, sample) in group.iter().enumerate().filter(|(i, _)| counts[*i] > 0) {
                let less = |j: usize| counts[j] < counts[i];
                let interior = left[i] < counts[i] && right[i] < counts[i];
                let radius = 16.max(sample.rows / 8);
                let lo = group.partition_point(|s| s.rows < sample.rows.saturating_sub(radius));
                let hi = group.partition_point(|s| s.rows <= sample.rows.saturating_add(radius));
                let neighbour = !interior && (lo..hi).any(|j| less(j) && totals[j] < totals[i]);
                if neighbour || interior {
                    failed.insert(Failure {
                        entry: tag.entry.clone(),
                        mode: sample.mode.clone(),
                        phase: sample.phase.clone(),
                        family: tag.family.clone(),
                        symbol: tag.symbol.clone(),
                        rows: sample.rows,
                    });
                }
            }
        }
        if (tier == Tier::Exhaustive || (tier == Tier::Full && group[0].mode == "single"))
            && group[0].phase == "prefill"
        {
            let symbols: BTreeSet<_> = group
                .iter()
                .flat_map(|s| s.symbols.keys())
                .filter(|s| !tagged.contains(s.as_str()))
                .collect();
            for symbol in symbols {
                let mut runs: Vec<Vec<&Sample>> = Vec::new();
                let mut run = Vec::new();
                for sample in group.iter().copied() {
                    let present = sample.symbols.get(symbol).copied().unwrap_or(0) > 0;
                    if (!present
                        || run
                            .last()
                            .is_some_and(|previous: &&Sample| previous.rows + 1 != sample.rows))
                        && !run.is_empty()
                    {
                        runs.push(std::mem::take(&mut run));
                    }
                    if present {
                        run.push(sample);
                    }
                }
                if !run.is_empty() {
                    runs.push(run);
                }
                // A broad route with short boundary fragments is not an island.
                if !runs.is_empty() && runs.iter().all(|run| run.len() <= 3) {
                    for s in runs.into_iter().flatten() {
                        failed.insert(Failure {
                            entry: symbol.clone(),
                            mode: s.mode.clone(),
                            phase: s.phase.clone(),
                            family: "untagged".into(),
                            symbol: symbol.clone(),
                            rows: s.rows,
                        });
                    }
                }
            }
        }
    }
    failed.into_iter().collect()
}

pub(super) fn check(
    profile: &str,
    failures: &[Failure],
    allow: &[Allow],
    tier: Tier,
    tags: &[Tag],
) -> Result<(), String> {
    let mut used = BTreeSet::new();
    let mut missing = BTreeMap::<(&str, &str, &str, &str, &str), Vec<u32>>::new();
    for f in failures {
        let matches: Vec<_> = allow
            .iter()
            .enumerate()
            .filter(|(_, a)| a.matches(profile, &f.phase, &f.family, &f.entry, &f.symbol, f.rows))
            .map(|(i, _)| i)
            .collect();
        if matches.is_empty() {
            missing
                .entry((&f.mode, &f.phase, &f.family, &f.entry, &f.symbol))
                .or_default()
                .push(f.rows);
        }
        used.extend(matches);
    }
    for &index in &used {
        let entry = &allow[index];
        let matched: BTreeSet<_> = failures
            .iter()
            .filter(|f| entry.matches(profile, &f.phase, &f.family, &f.entry, &f.symbol, f.rows))
            .map(|f| f.rows)
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "schema": "chopinfer.route-exception.v1", "profile": profile,
                "phase": entry.phase, "family": entry.family, "entry": entry.entry,
                "symbol": entry.symbol, "allowed_rows": entry.rows,
                "observed_rows": ranges(&matched.into_iter().collect::<Vec<_>>()),
                "reason": entry.reason, "task": entry.task,
            })
        );
    }
    let mut errors = Vec::new();
    for ((mode, phase, family, entry, symbol), rows) in missing {
        let tag = tags.iter().find(|t| t.entry == entry);
        let admitted = tag.map(|t| t.admitted.as_slice()).unwrap_or_default();
        let wrappers = tag.map(|t| t.wrappers.as_slice()).unwrap_or_default();
        errors.push(format!("{profile} mode={mode} phase={phase} family={family} entry={entry} symbol={symbol} rows={} admitted={admitted:?} wrappers={wrappers:?}", ranges(&rows)));
    }
    if matches!(tier, Tier::Full | Tier::Exhaustive) {
        for (i, entry) in allow.iter().enumerate().filter(|(_, a)| a.applies(profile)) {
            if !used.contains(&i) {
                errors.push(format!(
                    "{profile}: stale allowlist entry {} ({}/{})",
                    i + 1,
                    entry.family,
                    entry.symbol
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

fn ranges(rows: &[u32]) -> String {
    let rows: BTreeSet<_> = rows.iter().copied().collect();
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for row in rows {
        if let Some((_, end)) = ranges.last_mut()
            && *end + 1 == row
        {
            *end = row;
        } else {
            ranges.push((row, row));
        }
    }
    ranges
        .into_iter()
        .map(|(lo, hi)| {
            if lo == hi {
                lo.to_string()
            } else {
                format!("{lo}..{hi}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::super::allowlist::parse;
    use super::*;
    use kernel_core::route::RouteClass;

    #[test]
    fn repeated_selected_entries_have_one_bounded_metadata_record() {
        let mut tags = BTreeMap::new();
        for row in 1..=8192 {
            remember(
                &mut tags,
                RouteTag::new("base", RouteClass::Baseline, "gemm", [row..=row]).into(),
            );
        }
        assert_eq!(tags.len(), 1);
        assert_eq!(tags["gemm/base"].admitted, [[1, 8192]]);
        let mut conflict = tags["gemm/base"].clone();
        conflict.class = "Fast".into();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| remember(
                &mut tags, conflict
            )))
            .is_err()
        );
    }

    #[test]
    fn point_admission_holes_require_justification_and_stale_entries_fail() {
        let tag = RouteTag::new(
            "base",
            RouteClass::Baseline,
            "gemm",
            [512..=512, 2048..=2048],
        );
        let samples: Vec<_> = [511, 512, 513, 1000, 2048]
            .into_iter()
            .map(|r| Sample {
                mode: "single".into(),
                phase: "prefill".into(),
                rows: r,
                symbols: BTreeMap::from([(
                    (if r == 512 || r == 2048 {
                        "fast"
                    } else {
                        "gemm/base"
                    })
                    .into(),
                    1,
                )]),
            })
            .collect();
        let failed = failures(&samples, &[tag.into()], Tier::Quick);
        assert_eq!(
            failed.iter().map(|f| f.rows).collect::<Vec<_>>(),
            [511, 513, 1000]
        );
        assert!(check("tp4-ar", &failed, &[], Tier::Quick, &[]).is_err());
        let allow = parse("[[allow]]\nprofiles = [\"tp4-*\"]\nphase = \"prefill\"\nfamily = \"gemm\"\nsymbol = \"base\"\nrows = [[1, 2048]]\nreason = \"Point routes await qualification\"\ntask = \"range-admission\"\n").unwrap();
        assert!(check("tp4-ar", &failed, &allow, Tier::Full, &[]).is_ok());
        assert!(
            check("tp4-ar", &[], &allow, Tier::Full, &[])
                .unwrap_err()
                .contains("stale")
        );
    }

    #[test]
    fn full_tier_detects_untagged_islands_but_constant_routes_pass() {
        let samples: Vec<_> = (1..=20)
            .map(|r| Sample {
                mode: "single".into(),
                phase: "prefill".into(),
                rows: r,
                symbols: if (7..=9).contains(&r) {
                    BTreeMap::from([("island".into(), 1), ("constant".into(), 1)])
                } else {
                    BTreeMap::from([("constant".into(), 1)])
                },
            })
            .collect();
        assert!(failures(&samples, &[], Tier::Cpu).is_empty());
        let failed = failures(&samples, &[], Tier::Full);
        assert_eq!(failed.iter().map(|f| f.rows).collect::<Vec<_>>(), [7, 8, 9]);
        assert!(failed.iter().all(|f| f.symbol == "island"));
        let mut broad = samples;
        for sample in broad.iter_mut().filter(|s| s.rows >= 15) {
            sample.symbols.insert("island".into(), 1);
        }
        assert!(failures(&broad, &[], Tier::Full).is_empty());
    }
}
