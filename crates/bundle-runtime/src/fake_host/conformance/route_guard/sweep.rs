use super::super::{Item, Phase, Serving, Startup};
use super::{
    Envelope, Tier, allowlist, audit, boundary_probes, budget, distributed, pieces, probes,
    rotation, seed,
};
use crate::fake_host::RecordingSink;
use kernel_core::route::{RouteTag, merge};
use runtime_bundle_abi::BundleInstanceV1;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

// Keep the total shard count bounded when libtest runs profiles concurrently.
static SWEEP: Mutex<()> = Mutex::new(());

#[derive(Clone)]
struct Case {
    mode: String,
    phase: Phase,
    rows: u32,
    offset: u32,
    lengths: Vec<u32>,
}

pub(super) fn modes(envelope: &Envelope) -> Vec<(String, u32, u32)> {
    let mut modes = vec![("single".into(), 0, 1)];
    modes.extend(
        envelope
            .chunk_offsets
            .iter()
            .map(|&offset| (format!("chunk-{offset}"), offset, 1)),
    );
    for prompts in [2, envelope.packed_prompts] {
        if prompts > envelope.packed_prompts
            || prompts < 2
            || modes.iter().any(|(_, _, p)| *p == prompts)
        {
            continue;
        }
        modes.push((format!("packed-{prompts}"), 0, prompts));
    }
    modes
}

fn cases(
    envelope: &Envelope,
    tags: &[RouteTag],
    tier: Tier,
    group: usize,
    groups: usize,
    seed: u64,
    rotation: Option<(&rotation::Rotation, &str)>,
) -> Vec<Case> {
    let mut cases = Vec::new();
    let mandatory = boundary_probes(envelope, tags);
    for (index, (mode, offset, prompts)) in modes(envelope).into_iter().enumerate() {
        if tier == Tier::Quick && index != 0 {
            continue;
        }
        let full = tier == Tier::Exhaustive || (tier == Tier::Full && index == 0);
        let rows = if tier == Tier::Cpu {
            let (rotation, profile) = rotation.expect("CPU rotation plan");
            rotation.rows(profile, &mode)
        } else {
            probes(
                envelope,
                tags,
                if full {
                    Tier::Full
                } else if tier == Tier::Quick {
                    Tier::Quick
                } else {
                    Tier::Cpu
                },
            )
        };
        for r in rows {
            if r < prompts {
                continue;
            }
            if !full
                && !matches!(tier, Tier::Quick | Tier::Cpu)
                && !mandatory.contains(&r)
                && !distributed(r, seed, group + index, groups)
            {
                continue;
            }
            let mut lengths: Vec<u32> = (0..prompts)
                .map(|i| r / prompts + u32::from(i < r % prompts))
                .collect();
            if prompts == 2 && lengths[1] > 3 {
                lengths[0] += 3;
                lengths[1] -= 3;
            }
            cases.push(Case {
                mode: mode.clone(),
                phase: Phase::Prefill,
                rows: r,
                offset,
                lengths,
            });
        }
    }
    for &r in &envelope.decode_buckets {
        cases.push(Case {
            mode: "decode".into(),
            phase: Phase::Decode,
            rows: r,
            offset: 5,
            lengths: vec![1; r as usize],
        });
    }
    for shape in &envelope.verify {
        cases.push(Case {
            mode: format!("verify-items-{}", shape.item_bucket),
            phase: Phase::Verify,
            rows: shape.row_bucket as u32,
            offset: 5,
            lengths: shape.drafts.iter().map(|n| *n as u32 + 1).collect(),
        });
    }
    cases
}

fn execute<S: Startup>(
    adapter: &S,
    instance: &mut BundleInstanceV1,
    items: &[Item],
    prompt_ends: &[u64],
    label: &str,
) {
    adapter
        .execute_route_step(
            instance,
            items,
            prompt_ends,
            &mut RecordingSink::<S::Limits>::default(),
        )
        .unwrap_or_else(|e| panic!("{label}: execute: {e}"));
    let kept: Vec<_> = items
        .iter()
        .filter(|i| i.phase == Phase::Verify)
        .map(|i| super::super::kept(i.request, i.tokens.len() as u32))
        .collect();
    if kept.is_empty() {
        instance.commit()
    } else {
        instance.commit_accepted(&kept)
    }
    .unwrap_or_else(|e| panic!("{label}: commit: {e}"));
}

fn shard<S: Startup>(
    serving: &Serving,
    envelope: &Envelope,
    cases: &[Case],
    full: bool,
) -> (Vec<audit::Sample>, Vec<audit::Tag>, f64) {
    let mut budget = budget::Budget::begin(&serving.profile, full);
    let setup = Instant::now();
    let _metadata = kernel_core::route::Metadata::enable();
    let adapter = S::new(serving);
    adapter.route_device().symbols_only.set(true);
    let create_counts = crate::routes::RouteCounter::begin(true);
    let mut instance = adapter
        .create()
        .unwrap_or_else(|e| panic!("{}: sweep create: {e}", serving.profile));
    drop(create_counts);
    let setup = setup.elapsed().as_secs_f64();
    let mut result = Vec::new();
    let mut observed_tags = BTreeMap::new();
    for (case_index, case) in cases.iter().enumerate() {
        budget.check();
        let label = format!("{} {} rows={}", serving.profile, case.mode, case.rows);
        let item = |slot: usize, start: u32, rows: u32, phase| Item {
            request: slot as u64 + 1,
            slot: slot as u64,
            start: u64::from(start),
            tokens: (start..start + rows)
                .map(|n| adapter.token(u64::from(n)))
                .collect(),
            phase,
        };
        // Establish real committed state before a final chunk, decode or verify.
        let prompt_ends: Vec<_> = case
            .lengths
            .iter()
            .map(|&rows| {
                u64::from(if case.phase == Phase::Prefill {
                    case.offset + rows
                } else {
                    case.offset
                })
            })
            .collect();
        let mut start = 0;
        while start < case.offset {
            let count =
                (case.offset - start).min(envelope.prefill_tokens / case.lengths.len() as u32);
            assert!(count > 0, "{label}: envelope cannot hold setup requests");
            let setup: Vec<_> = (0..case.lengths.len())
                .map(|slot| item(slot, start, count, Phase::Prefill))
                .collect();
            execute(&adapter, &mut instance, &setup, &prompt_ends, &label);
            start += count;
            adapter.route_device().clear_route_journal();
            budget.check();
        }
        adapter.route_device().clear_route_journal();
        let items: Vec<_> = case
            .lengths
            .iter()
            .enumerate()
            .map(|(slot, &rows)| {
                item(
                    slot,
                    case.offset,
                    rows,
                    if case.phase == Phase::Verify && rows == 1 {
                        Phase::Decode
                    } else {
                        case.phase
                    },
                )
            })
            .collect();
        let counter = crate::routes::RouteCounter::begin(true);
        execute(&adapter, &mut instance, &items, &prompt_ends, &label);
        let selected = counter.finish().expect("sweep counter");
        let mut symbols = BTreeMap::new();
        for symbol in adapter.launched(0) {
            *symbols.entry(symbol).or_default() += 1;
        }
        for (entry, (tag, count)) in selected.entries {
            symbols.insert(entry, count);
            audit::remember(&mut observed_tags, tag.into());
        }
        assert!(!symbols.is_empty(), "{label}: zero kernel launches");
        result.push(audit::Sample {
            mode: case.mode.clone(),
            phase: match case.phase {
                Phase::Prefill => "prefill",
                Phase::Decode => "decode",
                Phase::Verify => "verify",
            }
            .into(),
            rows: case.rows,
            symbols,
        });
        for item in items {
            instance
                .retire_request(item.request)
                .unwrap_or_else(|e| panic!("{label}: retire: {e}"));
        }
        adapter.finish_route_case();
        budget.check();
        if full && (case_index + 1) % 64 == 0 {
            println!(
                "{}: completed {}/{} sequential shapes",
                serving.profile,
                case_index + 1,
                cases.len()
            );
        }
    }
    instance.destroy().expect("sweep destroy");
    adapter.assert_released();
    (result, observed_tags.into_values().collect(), setup)
}

fn export(bundle: &str, profile: &str, tags: &[audit::Tag]) {
    let Some(directory) = std::env::var_os("CHOPINFER_ROUTE_TAGS_OUT") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory)
        .canonicalize()
        .expect("existing external CHOPINFER_ROUTE_TAGS_OUT directory");
    // Never emit receipts into the checkout, including through a symlink.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    assert!(
        !directory.starts_with(root),
        "route tags must be outside the repository"
    );
    let document = serde_json::json!({"schema": "chopinfer.route-tags.v1", "bundle": bundle, "profile": profile, "tags": tags});
    let path = directory.join(format!("{bundle}-{profile}.json"));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("new route tags export");
    serde_json::to_writer_pretty(&mut file, &document).unwrap();
}

pub fn sweep<S: Startup>(profile: &str, allowlist: &str, bundle: &str) {
    let tier = Tier::configured();
    let servings = S::servings();
    let serving = servings
        .iter()
        .find(|s| s.profile == profile)
        .expect("declared profile");
    if tier == Tier::Quick {
        let first = servings
            .iter()
            .find(|s| {
                s.tensor_parallel == serving.tensor_parallel
                    && (s.draft_tokens > 0) == (serving.draft_tokens > 0)
            })
            .unwrap();
        if first.profile != profile {
            println!("{profile}: quick tier represented by {}", first.profile);
            return;
        }
    }
    // Serialize CPU profiles that own internal workers. Full-tier pieces
    // execute one shape at a time in a required memory-limited scope.
    let _serial = (tier != Tier::Quick).then(|| SWEEP.lock().unwrap_or_else(|e| e.into_inner()));
    let started = Instant::now();
    let full = matches!(tier, Tier::Full | Tier::Exhaustive);
    let mut budget = budget::Budget::begin(profile, full);
    let envelope = S::envelope(serving);
    assert!(
        envelope.prefill_tokens > 0,
        "the served prefill envelope is empty"
    );
    assert!(
        envelope.packed_prompts > 0
            && envelope.packed_prompts <= serving.sequences
            && envelope.packed_prompts <= envelope.prefill_tokens
    );
    assert!(
        envelope
            .chunk_offsets
            .iter()
            .all(|&n| n > 0 && n.checked_add(envelope.prefill_tokens).is_some()),
        "invalid chunk offset"
    );
    let host = Envelope::served(serving, envelope.prefill_tokens);
    assert!(
        host.decode_buckets
            .iter()
            .all(|b| envelope.decode_buckets.contains(b)),
        "missing host decode bucket"
    );
    assert!(
        host.verify
            .iter()
            .all(|shape| envelope.verify.contains(shape)),
        "missing host verify shape"
    );
    let tags = merge(S::new(serving).route_tags()).expect("consistent route registrations");
    let seed = seed();
    let mut group = 0;
    let mut groups = 0;
    for s in &servings {
        if s.profile == profile {
            group = groups;
        }
        groups += modes(&S::envelope(s)).len();
    }
    let rotation = (tier == Tier::Cpu).then(|| {
        let profiles: Vec<_> = servings
            .iter()
            .map(|s| rotation::Profile {
                name: s.profile.clone(),
                envelope: S::envelope(s),
                tags: if s.profile == profile {
                    tags.clone()
                } else {
                    S::new(s).route_tags()
                },
            })
            .collect();
        rotation::Rotation::new(&profiles, seed)
    });
    if let Some(rotation) = &rotation {
        rotation.report(bundle, profile);
    }
    let cases = cases(
        &envelope,
        &tags,
        tier,
        group,
        groups,
        seed,
        rotation.as_ref().map(|r| (r, profile)),
    );
    assert!(!cases.is_empty());
    let count = cases.len();
    let pieces = pieces::Pieces::configured(tier, bundle, profile, seed, count);
    let mut catalogue = BTreeMap::new();
    for tag in tags {
        audit::remember(&mut catalogue, tag.into());
    }
    let threads = if full { 1 } else { S::route_threads(tier) };
    assert!((1..=8).contains(&threads), "route workers must be in 1..=8");
    let shards = threads.min(count);
    let mut setup = 0.0_f64;
    let mut samples = if let Some(pieces) = pieces.as_ref() {
        if pieces.audit() {
            let (samples, observed) = pieces.load();
            for tag in observed {
                audit::remember(&mut catalogue, tag);
            }
            // Compare the complete case plan, so duplicated or omitted pieces
            // cannot turn a partial sweep into a passing coverage verdict.
            let mut expected: Vec<_> = cases.iter().map(case_key).collect();
            let mut actual: Vec<_> = samples.iter().map(sample_key).collect();
            expected.sort();
            actual.sort();
            assert_eq!(
                actual, expected,
                "route piece coverage differs from the envelope"
            );
            println!("route audit: case_plan_matches=true cases={count}");
            samples
        } else {
            let cases: Vec<_> = cases
                .into_iter()
                .enumerate()
                .filter(|(index, _)| pieces.includes(*index))
                .map(|(_, case)| case)
                .collect();
            let (samples, observed, _) = shard::<S>(serving, &envelope, &cases, true);
            for tag in observed {
                audit::remember(&mut catalogue, tag);
            }
            budget.check();
            pieces.save(samples, catalogue.into_values().collect());
            return;
        }
    } else {
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..shards)
                .map(|i| {
                    let cases: Vec<_> = cases.iter().skip(i).step_by(shards).cloned().collect();
                    let envelope = &envelope;
                    scope.spawn(move || shard::<S>(serving, envelope, &cases, false))
                })
                .collect();
            let mut samples = Vec::new();
            for worker in workers {
                let (part, observed, created) = worker.join().expect("sweep shard");
                samples.extend(part);
                for tag in observed {
                    audit::remember(&mut catalogue, tag);
                }
                setup = setup.max(created);
            }
            samples
        })
    };
    samples.sort_by(|a, b| (&a.mode, &a.phase, a.rows).cmp(&(&b.mode, &b.phase, b.rows)));
    let tags: Vec<_> = catalogue.into_values().collect();
    budget.check();
    let failures = audit::failures(&samples, &tags, tier);
    let allow = allowlist::parse(allowlist).expect("justified route allowlist");
    export(bundle, profile, &tags);
    for tag in tags.iter().filter(|t| t.class != "Fast") {
        for phase in ["prefill", "decode", "verify"] {
            let launches: u64 = samples
                .iter()
                .filter(|s| s.phase == phase)
                .map(|s| s.symbols.get(&tag.entry).copied().unwrap_or(0))
                .sum();
            if launches > 0 {
                println!(
                    "{profile} phase={phase} family={} entry={} symbol={} class={} launches={launches} wrappers={:?} admitted={:?}",
                    tag.family, tag.entry, tag.symbol, tag.class, tag.wrappers, tag.admitted
                );
            }
        }
    }
    println!(
        "{profile}: route sweep {tier:?}, seed={seed}, {count} cases, {:.3}s, workers={shards}, max_setup={setup:.3}s",
        started.elapsed().as_secs_f64()
    );
    if let Err(error) = audit::check(profile, &failures, &allow, tier, &tags) {
        panic!("{error}");
    }
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Prefill => "prefill",
        Phase::Decode => "decode",
        Phase::Verify => "verify",
    }
}
fn case_key(case: &Case) -> (String, String, u32) {
    (case.mode.clone(), phase_name(case.phase).into(), case.rows)
}
fn sample_key(sample: &audit::Sample) -> (String, String, u32) {
    (sample.mode.clone(), sample.phase.clone(), sample.rows)
}

/// Exercise repeated varying shapes with the same fake instance. Model hooks
/// assert that diagnostics do not retain buffers after request retirement.
pub fn memory_regression<S: Startup>() {
    let servings = S::servings();
    let serving = servings
        .iter()
        .find(|s| s.draft_tokens > 0)
        .unwrap_or(&servings[0]);
    let envelope = S::envelope(serving);
    let cases: Vec<_> = (0..8)
        .flat_map(|_| [1, 2, 3, 7, 16, 31, 32, 63, 64])
        .map(|rows| Case {
            mode: "single".into(),
            phase: Phase::Prefill,
            rows,
            offset: 0,
            lengths: vec![rows],
        })
        .collect();
    let (samples, _, _) = shard::<S>(serving, &envelope, &cases, false);
    assert_eq!(samples.len(), 72);
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::route::RouteClass;

    #[test]
    fn full_keeps_boundaries_in_every_mode_and_every_single_row() {
        let envelope = Envelope {
            prefill_tokens: 2048,
            chunk_offsets: vec![8192, 40960],
            packed_prompts: 8,
            decode_buckets: vec![1, 8],
            verify: vec![],
            probes: vec![446, 789],
        };
        let tags = vec![RouteTag::new(
            "base",
            RouteClass::Baseline,
            "projection",
            [512..=512, 2048..=2048],
        )];
        let mandatory = boundary_probes(&envelope, &tags);
        let count = modes(&envelope).len();
        let full = cases(&envelope, &tags, Tier::Full, 0, count, 123, None);
        for (mode, _, prompts) in modes(&envelope) {
            for &rows in mandatory.iter().filter(|&&r| r >= prompts) {
                assert!(
                    full.iter().any(|c| c.mode == mode && c.rows == rows),
                    "{mode} {rows}"
                );
            }
        }
        let full = cases(&envelope, &tags, Tier::Full, 0, count, 123, None);
        assert_eq!(full.iter().filter(|c| c.mode == "single").count(), 2048);
        assert!(cases(&envelope, &tags, Tier::Exhaustive, 0, count, 123, None).len() > full.len());
        assert_eq!(
            full.len(),
            cases(&envelope, &tags, Tier::Full, 0, count, 123, None).len()
        );
        for row in [100, 333, 1000] {
            assert_eq!(
                (0..count)
                    .filter(|&g| distributed(row, 123, g, count))
                    .count(),
                1
            );
            let groups: std::collections::BTreeSet<_> = (0..30)
                .map(|seed| {
                    (0..count)
                        .find(|&g| distributed(row, seed, g, count))
                        .unwrap()
                })
                .collect();
            assert!(groups.len() > 1);
        }
    }
}
