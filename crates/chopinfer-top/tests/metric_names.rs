//! Every series chopinfer-top reads is a published contract: the vLLM
//! fixture names it, or the observability doc does. A renamed series fails
//! here instead of silently showing `-`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use chopinfer_top::stats::SERIES;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Series names in the fixture: its histogram, counter and gauge names.
fn fixture_names() -> BTreeSet<String> {
    let text =
        std::fs::read_to_string(root().join("crates/runtime-engine/tests/vllm_metrics.json"))
            .expect("the vLLM metrics fixture is readable");
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("the fixture is JSON");
    let mut names = BTreeSet::new();
    for section in ["histograms", "counters"] {
        names.extend(
            fixture[section]
                .as_object()
                .expect("a section")
                .keys()
                .cloned(),
        );
    }
    names.extend(
        fixture["gauges"]
            .as_array()
            .expect("gauges")
            .iter()
            .filter_map(|name| name.as_str().map(str::to_owned)),
    );
    names
}

/// Expands `a_{x,y}_b{labels}` into `a_x_b` and `a_y_b`: braces after `_`
/// are alternatives within the name, braces after a complete name are its
/// labels.
fn expand(token: &str) -> Vec<String> {
    let Some(open) = token.find('{') else {
        return vec![token.to_owned()];
    };
    let Some(close) = token[open..].find('}').map(|close| open + close) else {
        return vec![token[..open].to_owned()];
    };
    let (head, inner, tail) = (&token[..open], &token[open + 1..close], &token[close + 1..]);
    if !head.ends_with('_') {
        return vec![head.to_owned()];
    }
    inner
        .split(',')
        .flat_map(|choice| expand(&format!("{head}{choice}{tail}")))
        .collect()
}

/// Series names the observability doc mentions in code spans.
fn documented_names() -> BTreeSet<String> {
    let text = std::fs::read_to_string(root().join("docs/operations-observability.md"))
        .expect("the observability doc is readable");
    text.split('`')
        .skip(1)
        .step_by(2)
        .filter(|span| span.starts_with("chop_") || span.starts_with("vllm:"))
        .flat_map(expand)
        .collect()
}

#[test]
fn every_series_read_is_published_or_documented() {
    let known = fixture_names()
        .into_iter()
        .chain(documented_names())
        .collect::<BTreeSet<_>>();
    let missing = SERIES
        .iter()
        .filter(|series| !known.contains(**series))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "undocumented series read by chopinfer-top: {missing:?}"
    );
}

#[test]
fn brace_forms_expand_to_names() {
    assert_eq!(
        expand("chop_{prefill_only,decode_only,mixed}_batches_total"),
        [
            "chop_prefill_only_batches_total",
            "chop_decode_only_batches_total",
            "chop_mixed_batches_total"
        ]
    );
    assert_eq!(
        expand("chop_instance_poisoned{model_name,engine}"),
        ["chop_instance_poisoned"]
    );
    assert_eq!(
        expand("chop_prefix_cache_{anchors,blocks}"),
        ["chop_prefix_cache_anchors", "chop_prefix_cache_blocks"]
    );
}
