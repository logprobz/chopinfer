use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn recipes_match_locked_names_and_entry_symbols() {
    let mut checked = 0;
    for recipe in all() {
        for step in (recipe.steps)().unwrap() {
            if matches!(step.root, Root::Output(_)) {
                continue;
            }
            let mut families: BTreeMap<&str, Vec<&CubinSpec>> = BTreeMap::new();
            for (name, spec) in &step.specs {
                families.entry(name).or_default().push(spec);
            }
            for (name, specs) in families {
                let path = step.root.path(Path::new("")).join(format!("{name}.lock"));
                let locked: Vec<serde_json::Value> = serde_json::from_slice(
                    &std::fs::read(&path)
                        .unwrap_or_else(|error| panic!("{} {name}: {error}", recipe.key())),
                )
                .unwrap();
                let actual: BTreeSet<_> = specs.iter().map(|s| s.name.as_str()).collect();
                let expected: BTreeSet<_> =
                    locked.iter().map(|r| r["name"].as_str().unwrap()).collect();
                assert_eq!(actual, expected, "{} {name}", recipe.key());
                for spec in specs {
                    let row = locked.iter().find(|r| r["name"] == spec.name).unwrap();
                    let actual: BTreeSet<_> =
                        spec.entries.iter().map(|e| e.symbol.as_str()).collect();
                    let expected: BTreeSet<_> = row["entries"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|e| e["symbol"].as_str().unwrap())
                        .collect();
                    assert_eq!(actual, expected, "{} {}", recipe.key(), spec.name);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn cpu_model_query_candidate_keeps_prefill_and_serving_registration_separate() {
    use kernel_core::backend::{Backend, Need};
    use model_deepseek41_contract::{indexer, serving::PREFILL_TOKENS};
    for g in GEOMETRIES {
        let backend = bundle::indexer_query_parallel::register_parallel(
            kernels_sm120::backend::Sm120::new(),
            g,
        );
        let query = indexer::selection(g, g.shape.csa.index_sources[0], PREFILL_TOKENS).unwrap();
        for rows in [1, 2, 4, 6, 8, 16, 32, PREFILL_TOKENS] {
            let op = backend
                .position_indexer_query(&query, Need::rows(rows))
                .unwrap();
            assert_eq!(op.info().symbols.len(), if rows <= 32 { 1 } else { 3 });
            assert!(
                op.info()
                    .symbols
                    .iter()
                    .any(|(name, _)| name.ends_with("parallel_v1"))
            );
        }
    }
}
