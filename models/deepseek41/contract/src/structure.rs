//! The relations between execution fields that no single field states:
//! the layer schedule of the shared compressed KV and index keys, the
//! candidate pool, Engram tables and DSpark targets. Shared with `build.rs`,
//! so a geometry that breaks one fails the build, and admission refuses a
//! config that breaks one naming the field.
use serde_json::Value;

fn number(config: &Value, path: &str) -> Result<u64, String> {
    let mut value = config;
    for key in path.split('.') {
        value = value.get(key).unwrap_or(&Value::Null);
    }
    value.as_u64().ok_or_else(|| {
        format!("unsupported {path}: expected an unsigned integer, received {value}")
    })
}

fn list(config: &Value, path: &str) -> Result<Vec<u64>, String> {
    let mut value = config;
    for key in path.split('.') {
        value = value.get(key).unwrap_or(&Value::Null);
    }
    value
        .as_array()
        .and_then(|items| items.iter().map(Value::as_u64).collect::<Option<Vec<_>>>())
        .ok_or_else(|| {
            format!("unsupported {path}: expected a list of unsigned integers, received {value}")
        })
}

fn refuse(path: &str, expected: impl std::fmt::Display, received: impl std::fmt::Debug) -> String {
    format!("unsupported {path}: expected {expected}, received {received:?}")
}

/// Layer ids in increasing order, each a backbone layer (below `layers`).
fn layer_ids(config: &Value, path: &str, layers: u64) -> Result<Vec<u64>, String> {
    let ids = list(config, path)?;
    if !ids.windows(2).all(|pair| pair[0] < pair[1]) || ids.iter().any(|&id| id >= layers) {
        return Err(refuse(
            path,
            format!("increasing backbone layer ids below {layers}"),
            &ids,
        ));
    }
    Ok(ids)
}

/// The smallest prime above `start` not yet handed out (the vendor's
/// `find_next_prime`).
fn next_prime(start: u64, seen: &[u64]) -> u64 {
    let prime = |n: u64| n >= 2 && (2..).take_while(|i| i * i <= n).all(|i| n % i != 0);
    let mut candidate = start + 1;
    while !prime(candidate) || seen.contains(&candidate) {
        candidate += 1;
    }
    candidate
}

/// Rows of each Engram layer's table: one prime-sized bucket range per
/// (n-gram size, head), drawn in order from `vocab` up and never reused
/// (the vendor's `EngramLayout.from_args`).
pub fn engram_rows(layers: usize, vocab: u64, max_ngram: u64, heads: u64) -> Vec<u64> {
    let mut seen = Vec::new();
    (0..layers)
        .map(|_| {
            let mut rows = 0;
            for _ in 1..max_ngram {
                let mut current = vocab.saturating_sub(1);
                for _ in 0..heads {
                    current = next_prime(current, &seen);
                    seen.push(current);
                    rows += current;
                }
            }
            rows
        })
        .collect()
}

/// Check the cross-field relations of an execution config.
pub fn check(config: &Value) -> Result<(), String> {
    let t = |key: &str| format!("text_config.{key}");
    let layers = number(config, &t("num_hidden_layers"))?;
    let stages = number(config, &t("num_nextn_predict_layers"))?;

    // One compression ratio per backbone layer, then one per DSpark stage,
    // which attends over its window only.
    let ratios_path = t("compress_ratios");
    let ratios = list(config, &ratios_path)?;
    if ratios.len() as u64 != layers + stages
        || ratios.iter().any(|ratio| *ratio > 2)
        || ratios[layers as usize..].iter().any(|ratio| *ratio != 0)
    {
        return Err(refuse(
            &ratios_path,
            format!(
                "{} ratios of 0, 1 or 2, with 0 for the {stages} DSpark stages",
                layers + stages
            ),
            &ratios,
        ));
    }

    // Every compressing layer reads the compressed KV of the latest KV
    // source, which compresses at its own ratio, and the top-k of the
    // latest index source; every KV source owns the index keys, so it is
    // also an index source.
    let kv_path = t("kv_source_layer_ids");
    let index_path = t("index_source_layer_ids");
    let kv = layer_ids(config, &kv_path, layers)?;
    let index = layer_ids(config, &index_path, layers)?;
    if let Some(id) = kv.iter().find(|&&id| ratios[id as usize] == 0) {
        return Err(refuse(&kv_path, "layers with a nonzero compress ratio", id));
    }
    if let Some(id) = index.iter().find(|&&id| ratios[id as usize] == 0) {
        return Err(refuse(
            &index_path,
            "layers with a nonzero compress ratio",
            id,
        ));
    }
    if let Some(id) = kv.iter().find(|id| !index.contains(id)) {
        return Err(refuse(&index_path, "every KV source layer", id));
    }
    for layer in 0..layers {
        let ratio = ratios[layer as usize];
        if ratio == 0 {
            continue;
        }
        for (path, sources) in [(&kv_path, &kv), (&index_path, &index)] {
            let source = sources.iter().rev().find(|&&id| id <= layer);
            if source.is_none_or(|&id| ratios[id as usize] != ratio) {
                return Err(refuse(
                    path,
                    format!("a source at or before layer {layer} compressing at its ratio {ratio}"),
                    sources,
                ));
            }
        }
    }

    // The candidate pool is built from a KV source's own index keys and
    // read by the index sources after it.
    let candidate_path = t("candidate_source_layer_id");
    let candidate = number(config, &candidate_path)?;
    if !kv.contains(&candidate) {
        return Err(refuse(&candidate_path, "a KV source layer", candidate));
    }
    for key in [
        "candidate_topk_blocks",
        "candidate_block_size",
        "index_topk",
    ] {
        if number(config, &t(key))? == 0 {
            return Err(refuse(&t(key), "a positive value", 0));
        }
    }

    // The projection declaration names the CSA2 roles by position: three
    // pooling (ratio 2) KV sources, then the ratio-1 candidate source, then
    // four reindexing index sources reading its candidates.
    let pooled = kv.iter().filter(|&&id| ratios[id as usize] == 2).count();
    if kv.len() != 4 || pooled != 3 || kv[3] != candidate {
        return Err(refuse(
            &kv_path,
            "three ratio-2 sources, then the ratio-1 candidate source",
            &kv,
        ));
    }
    let reindex = index
        .iter()
        .filter(|id| !kv.contains(id))
        .collect::<Vec<_>>();
    if reindex.len() != 4 || reindex.iter().any(|&&id| id <= candidate) {
        return Err(refuse(
            &index_path,
            format!("the KV sources and four more layers after layer {candidate}"),
            &index,
        ));
    }

    // Attention: whole output groups of whole heads, and a RoPE tail inside
    // the latent.
    let heads = number(config, &t("num_attention_heads"))?;
    let groups = number(config, &t("o_groups"))?;
    if groups == 0 || !heads.is_multiple_of(groups) {
        return Err(refuse(
            &t("o_groups"),
            format!("a divisor of {heads} heads"),
            groups,
        ));
    }
    let head_dim = number(config, &t("head_dim"))?;
    let rope = number(config, &t("qk_rope_head_dim"))?;
    if rope == 0 || !rope.is_multiple_of(2) || rope >= head_dim {
        return Err(refuse(
            &t("qk_rope_head_dim"),
            format!("an even width below head_dim {head_dim}"),
            rope,
        ));
    }

    // Routing picks at most every expert.
    for (experts, top_k) in [
        ("n_routed_experts", "num_experts_per_tok"),
        ("dspark_n_routed_experts", "dspark_num_experts_per_tok"),
    ] {
        let (experts, chosen) = (number(config, &t(experts))?, number(config, &t(top_k))?);
        if chosen == 0 || chosen > experts {
            return Err(refuse(&t(top_k), format!("1..={experts}"), chosen));
        }
    }

    // Engram tables: one per Engram layer, sized as the vendor lays out
    // its hash buckets.
    let engram_path = t("engram_layer_ids");
    let engram = layer_ids(config, &engram_path, layers)?;
    if engram.len() != 2 {
        return Err(refuse(
            &engram_path,
            "two layers (the declared tables)",
            &engram,
        ));
    }
    let rows_path = t("engram_num_embeddings");
    let rows = list(config, &rows_path)?;
    let expected = engram_rows(
        engram.len(),
        number(config, &t("engram_vocab_size"))?,
        number(config, &t("engram_max_ngram_size"))?,
        number(config, &t("engram_n_heads"))?,
    );
    if rows != expected {
        return Err(refuse(
            &rows_path,
            format!("{expected:?} (one prime bucket range per n-gram size and head)"),
            &rows,
        ));
    }

    // DSpark reads the attention inputs of backbone target layers.
    let targets_path = t("dspark_target_layer_ids");
    let targets = layer_ids(config, &targets_path, layers)?;
    if stages == 0 {
        return Err(refuse(
            &t("num_nextn_predict_layers"),
            "at least one DSpark stage",
            0,
        ));
    }
    if targets.is_empty() {
        return Err(refuse(&targets_path, "at least one target layer", &targets));
    }
    if number(config, &t("dspark_block_size"))? == 0 {
        return Err(refuse(&t("dspark_block_size"), "a positive block", 0));
    }
    Ok(())
}
