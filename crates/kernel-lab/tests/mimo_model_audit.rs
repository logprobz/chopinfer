//! Full-model MiMo GPU capture on the frozen teacher-forced texts.
//!
//! `gpu_hidden_states` runs the embedding, all 48 layers and the vocabulary
//! head on four GPUs, one layer at a time with bundle-mimo26's launch choices,
//! feeding each layer the GPU output of the layer before it. It writes hidden
//! states, router choices and logits under `CHOPINFER_MIMO_AUDIT_DIR`.
//! `crates/kernel-lab/tools/moe_routing_locality.py` reads the router choices.
//!
//! References come from the GPU reference runner (tools/refrun); its taps
//! and `tools/qual/tapdiff.py` locate the first layer that leaves the
//! reference's spread. `mimo_layers.rs` checks single layers against the f64
//! layer oracle.

#[path = "mimo_layers.rs"]
mod mimo_layers;

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kernel_lab::{LabResult, Session};
use mimo_layers::execution::{self, LAYERS, Rank, VOCAB_PER_RANK};
use serde_json::{Value, json};

const H: usize = 4096;
const VOCAB: usize = 4 * VOCAB_PER_RANK;
const PREFILL_CAPACITY: usize = 64;

/// Prefill rows per layer launch: `CHOPINFER_MIMO_AUDIT_CAPACITY`, else 64.
fn prefill_capacity() -> usize {
    std::env::var("CHOPINFER_MIMO_AUDIT_CAPACITY")
        .ok()
        .map_or(PREFILL_CAPACITY, |value| {
            value
                .parse::<usize>()
                .expect("CHOPINFER_MIMO_AUDIT_CAPACITY")
        })
}

/// Routing capture only: `CHOPINFER_MIMO_AUDIT_ROUTES_ONLY=1` writes the
/// router choices and the manifest, and skips hidden states and the head.
fn routes_only() -> bool {
    std::env::var("CHOPINFER_MIMO_AUDIT_ROUTES_ONLY").as_deref() == Ok("1")
}

/// `CHOPINFER_MIMO_AUDIT_GSM8K=N` adds the first N frozen GSM8K examples as
/// whole-text prefill streams: the fixture's instruction, the question and
/// the reference solution as plain text, cut to the prefill capacity. The
/// solutions stand in for generated reasoning when measuring how
/// consecutive tokens share experts.
fn gsm8k_streams(first: usize) -> Vec<Stream> {
    let Some(count) = std::env::var("CHOPINFER_MIMO_AUDIT_GSM8K")
        .ok()
        .map(|value| value.parse::<usize>().expect("CHOPINFER_MIMO_AUDIT_GSM8K"))
    else {
        return Vec::new();
    };
    let Ok(fixtures) = std::env::var("CHOPINFER_MIMO_FIXTURES") else {
        kernel_lab::skip("GSM8K audit streams: set CHOPINFER_MIMO_FIXTURES").unwrap();
        return Vec::new();
    };
    let Ok(checkpoint) = std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_CHECKPOINT"))
    else {
        kernel_lab::skip(
            "GSM8K audit streams: set CHOPINFER_MIMO_TOKENIZER_DIR or CHOPINFER_MIMO_CHECKPOINT",
        )
        .unwrap();
        return Vec::new();
    };
    let dir = Path::new(&fixtures).join("gsm8k");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("GSM8K manifest"))
            .expect("GSM8K manifest JSON");
    let examples: Vec<Value> =
        serde_json::from_slice(&std::fs::read(dir.join("examples.json")).expect("GSM8K examples"))
            .expect("GSM8K examples JSON");
    let tokenizer = model_mimo26::tokenizer::load_tokenizer(Path::new(&checkpoint))
        .expect("MiMo checkpoint tokenizer");
    let instruction = manifest["instruction"].as_str().expect("GSM8K instruction");
    let capacity = prefill_capacity();
    examples
        .iter()
        .take(count)
        .enumerate()
        .map(|(index, example)| {
            let text = format!(
                "{instruction}\n\n{}\n\n{}",
                example["question"].as_str().expect("GSM8K question"),
                example["expected"].as_str().expect("GSM8K solution"),
            );
            let mut ids = tokenizer.encode(&text, &[]).expect("GSM8K tokens");
            ids.truncate(capacity);
            Stream {
                name: format!("gsm8k{index:03}"),
                text: first + index,
                chunks: std::iter::once(0..ids.len()).collect(),
                ids,
                scored: false,
            }
        })
        .collect()
}
/// `CHOPINFER_MIMO_AUDIT_PROMPTS=FILE` adds whole-text prefill streams from a
/// JSON list of `{"name", "text"}` objects, tokenized with the MiMo tokenizer
/// and cut to the prefill capacity. It reproduces the router choices of a
/// served prompt's first chunk, for example a synthetic benchmark prompt.
fn prompt_streams(first: usize) -> Vec<Stream> {
    let Ok(file) = std::env::var("CHOPINFER_MIMO_AUDIT_PROMPTS") else {
        return Vec::new();
    };
    let checkpoint = std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_CHECKPOINT"))
        .expect("CHOPINFER_MIMO_AUDIT_PROMPTS needs CHOPINFER_MIMO_TOKENIZER_DIR or CHOPINFER_MIMO_CHECKPOINT");
    let prompts: Vec<Value> =
        serde_json::from_slice(&std::fs::read(&file).expect("audit prompts file"))
            .expect("audit prompts JSON");
    let tokenizer = model_mimo26::tokenizer::load_tokenizer(Path::new(&checkpoint))
        .expect("MiMo checkpoint tokenizer");
    let capacity = prefill_capacity();
    prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let text = prompt["text"].as_str().expect("prompt text");
            let mut ids = tokenizer.encode(text, &[]).expect("prompt tokens");
            ids.truncate(capacity);
            Stream {
                name: prompt["name"].as_str().expect("prompt name").to_owned(),
                text: first + index,
                chunks: std::iter::once(0..ids.len()).collect(),
                ids,
                scored: false,
            }
        })
        .collect()
}

const DECODE_STEPS: usize = 4;

/// One token sequence: a prefill chunk, then single-token decode steps.
struct Stream {
    name: String,
    text: usize,
    ids: Vec<u32>,
    chunks: Vec<Range<usize>>,
    /// A whole teacher text in one prefill, the streams `routes_only` keeps.
    scored: bool,
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn teacher() -> Vec<Value> {
    let path = repository().join("fixtures/mimo26/teacher-exact-reference.json");
    let reference: Value =
        serde_json::from_slice(&std::fs::read(path).expect("HF teacher reference"))
            .expect("teacher JSON");
    reference["texts"]
        .as_array()
        .expect("teacher texts")
        .iter()
        .filter(|row| {
            matches!(
                row["name"].as_str(),
                Some("frozen0" | "frozen1" | "frozen2")
            )
        })
        .cloned()
        .collect()
}

fn token_ids(text: &Value) -> Vec<u32> {
    text["prompt_token_ids"]
        .as_array()
        .expect("prompt token IDs")
        .iter()
        .map(|id| id.as_u64().expect("token ID") as u32)
        .collect()
}

fn streams() -> Vec<Stream> {
    let texts = teacher();
    let mut streams = texts
        .iter()
        .enumerate()
        .map(|(text, value)| {
            let ids = token_ids(value);
            Stream {
                name: format!("text{text}"),
                text,
                chunks: std::iter::once(0..ids.len()).collect(),
                ids,
                scored: true,
            }
        })
        .collect::<Vec<_>>();
    let ids = token_ids(&texts[0]);
    let prefill = ids.len() - DECODE_STEPS;
    streams.push(Stream {
        name: "text0_decode".into(),
        text: 0,
        chunks: std::iter::once(0..prefill)
            .chain((prefill..ids.len()).map(|position| position..position + 1))
            .collect(),
        ids,
        scored: false,
    });
    streams
}

fn layer_count() -> usize {
    std::env::var("CHOPINFER_MIMO_AUDIT_LAYERS")
        .ok()
        .map_or(LAYERS, |value| {
            value.parse::<usize>().expect("CHOPINFER_MIMO_AUDIT_LAYERS")
        })
        .min(LAYERS)
}

fn write<T: Copy>(path: &Path, values: &[T], bytes: impl Fn(T) -> Vec<u8>) -> LabResult<()> {
    std::fs::create_dir_all(path.parent().expect("parent")).map_err(|e| e.to_string())?;
    let data = values.iter().flat_map(|&v| bytes(v)).collect::<Vec<_>>();
    std::fs::write(path, data).map_err(|e| format!("{}: {e}", path.display()))
}

fn layer_file(dir: &Path, stream: &Stream, layer: usize, suffix: &str) -> PathBuf {
    dir.join(&stream.name)
        .join(format!("layer{layer:02}.{suffix}"))
}

/// Rank 0's output rows, and how many elements other ranks disagree on.
fn gather_equal(sessions: &[Session], ranks: &[Rank], rows: usize) -> LabResult<(Vec<u16>, usize)> {
    let outputs = ranks
        .iter()
        .zip(sessions)
        .map(|(rank, session)| execution::output(session, rank, rows))
        .collect::<LabResult<Vec<_>>>()?;
    let mismatched = outputs[1..]
        .iter()
        .map(|other| {
            other
                .iter()
                .zip(&outputs[0])
                .filter(|(a, b)| a != b)
                .count()
        })
        .sum();
    Ok((outputs[0].clone(), mismatched))
}

#[test]
fn gpu_hidden_states() {
    let (Ok(checkpoint), Ok(out)) = (
        std::env::var("CHOPINFER_MIMO_CHECKPOINT"),
        std::env::var("CHOPINFER_MIMO_AUDIT_DIR"),
    ) else {
        kernel_lab::skip(
            "MiMo model audit: set CHOPINFER_MIMO_CHECKPOINT and CHOPINFER_MIMO_AUDIT_DIR",
        )
        .unwrap();
        return;
    };
    let Some(ordinals) = kernel_lab::selected_ordinals().expect("GPU gate") else {
        kernel_lab::skip("MiMo model audit: set CHOPINFER_KERNEL_GPUS to idle GPUs").unwrap();
        return;
    };
    assert_eq!(ordinals.len(), 4, "MiMo audit requires four GPU ordinals");
    let out = PathBuf::from(out);
    // Every layer, planned from the checkpoint and read as each rank uploads.
    let images = execution::layer_images(Path::new(&checkpoint), None).expect("TP4 model image");
    let mut streams = streams();
    if routes_only() {
        streams.retain(|stream| stream.scored);
    }
    let first = streams.len();
    streams.extend(gsm8k_streams(first));
    let first = streams.len();
    streams.extend(prompt_streams(first));
    let layers = layer_count();
    let capacity = prefill_capacity();
    kernel_lab::gpu_test_multi(8 << 30, |sessions| {
        let started = Instant::now();
        let mut rank_mismatches = BTreeMap::new();
        let mut hidden = {
            let ranks = (0..4)
                .map(|rank| Rank::embedding(&sessions[rank], &images, rank, capacity))
                .collect::<LabResult<Vec<_>>>()?;
            let mut hidden = Vec::new();
            for stream in &streams {
                let mut rows = Vec::new();
                for (index, chunk) in stream.chunks.iter().enumerate() {
                    for (rank, session) in ranks.iter().zip(sessions) {
                        rank.set_end_inputs(session, Some(&stream.ids[chunk.clone()]), None)?;
                    }
                    execution::launch_all(sessions, &ranks, index == 0, true)?;
                    execution::synchronize(sessions)?;
                    let (output, mismatched) = gather_equal(sessions, &ranks, chunk.len())?;
                    *rank_mismatches.entry("embedding".to_owned()).or_insert(0) += mismatched;
                    rows.extend(output);
                }
                if !routes_only() {
                    write(
                        &out.join(&stream.name).join("embed.bf16"),
                        &rows,
                        |v: u16| v.to_le_bytes().to_vec(),
                    )?;
                }
                hidden.push(rows);
            }
            hidden
        };
        let mut layer_seconds = Vec::new();
        for layer in 0..layers {
            let layer_start = Instant::now();
            let ranks = (0..4)
                .map(|rank| {
                    Rank::build(
                        &sessions[rank],
                        &images,
                        rank,
                        layer,
                        1,
                        true,
                        capacity,
                        true,
                    )
                })
                .collect::<LabResult<Vec<_>>>()?;
            let mut next = Vec::new();
            for (stream, input) in streams.iter().zip(&hidden) {
                let mut rows = Vec::new();
                let mut routes = Vec::new();
                let mut residual = Vec::new();
                let mut attention = Vec::new();
                for (index, chunk) in stream.chunks.iter().enumerate() {
                    let prefill = index == 0;
                    let chunk_input = &input[chunk.start * H..chunk.end * H];
                    for (rank, session) in ranks.iter().zip(sessions) {
                        rank.set_inputs(session, chunk_input, chunk.start, prefill)?;
                    }
                    execution::launch_all(sessions, &ranks, prefill, true)?;
                    execution::synchronize(sessions)?;
                    let (output, mismatched) = gather_equal(sessions, &ranks, chunk.len())?;
                    *rank_mismatches
                        .entry(format!("layer{layer:02}"))
                        .or_insert(0) += mismatched;
                    if layer > 0 {
                        routes.extend(
                            execution::router_choices(&sessions[0], &ranks[0], chunk.len())?
                                .into_iter()
                                .flatten()
                                .map(|id| id as i32),
                        );
                    }
                    residual.extend(execution::intermediate(
                        &sessions[0],
                        &ranks[0],
                        "attention_residual",
                        chunk.len(),
                    )?);
                    // Rank r holds query heads 16r..16r+16 of every row.
                    let heads = ranks
                        .iter()
                        .zip(sessions)
                        .map(|(rank, session)| {
                            execution::intermediate(session, rank, "attention", chunk.len())
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    for row in 0..chunk.len() {
                        for rank in &heads {
                            attention.extend_from_slice(&rank[row * 2048..(row + 1) * 2048]);
                        }
                    }
                    rows.extend(output);
                }
                if !routes_only() {
                    write(&layer_file(&out, stream, layer, "bf16"), &rows, |v: u16| {
                        v.to_le_bytes().to_vec()
                    })?;
                    write(
                        &layer_file(&out, stream, layer, "resid.bf16"),
                        &residual,
                        |v: u16| v.to_le_bytes().to_vec(),
                    )?;
                    write(
                        &layer_file(&out, stream, layer, "attn.bf16"),
                        &attention,
                        |v: u16| v.to_le_bytes().to_vec(),
                    )?;
                }
                if layer > 0 {
                    write(
                        &layer_file(&out, stream, layer, "ids"),
                        &routes,
                        |v: i32| v.to_le_bytes().to_vec(),
                    )?;
                }
                next.push(rows);
            }
            drop(ranks);
            hidden = next;
            layer_seconds.push(layer_start.elapsed().as_secs_f64());
            eprintln!(
                "MiMo audit: layer {layer} done in {:.1} s",
                layer_seconds[layer]
            );
        }
        if layers == LAYERS && !routes_only() {
            let ranks = (0..4)
                .map(|rank| Rank::head(&sessions[rank], &images, rank, capacity))
                .collect::<LabResult<Vec<_>>>()?;
            for (stream, input) in streams.iter().zip(&hidden) {
                let mut logits = Vec::with_capacity(stream.ids.len() * VOCAB);
                for (index, chunk) in stream.chunks.iter().enumerate() {
                    let prefill = index == 0;
                    for (rank, session) in ranks.iter().zip(sessions) {
                        rank.set_end_inputs(
                            session,
                            None,
                            Some(&input[chunk.start * H..chunk.end * H]),
                        )?;
                    }
                    execution::launch_all(sessions, &ranks, prefill, true)?;
                    execution::synchronize(sessions)?;
                    let slices = ranks
                        .iter()
                        .zip(sessions)
                        .map(|(rank, session)| rank.head_logits(session, chunk.len(), prefill))
                        .collect::<LabResult<Vec<_>>>()?;
                    for row in 0..chunk.len() {
                        for slice in &slices {
                            logits.extend_from_slice(
                                &slice[row * VOCAB_PER_RANK..(row + 1) * VOCAB_PER_RANK],
                            );
                        }
                    }
                }
                write(
                    &out.join(&stream.name).join("logits.f32"),
                    &logits,
                    |v: f32| v.to_le_bytes().to_vec(),
                )?;
            }
        }
        let manifest = json!({
            "schema": "chopinfer.mimo26.numerics-audit.v1",
            "layers": layers,
            "prefill_capacity": capacity,
            "routes_only": routes_only(),
            "streams": streams.iter().map(|stream| json!({
                "name": stream.name,
                "text": stream.text,
                "ids": stream.ids,
                "chunks": stream.chunks.iter().map(|c| [c.start, c.end]).collect::<Vec<_>>(),
                "scored": stream.scored,
            })).collect::<Vec<_>>(),
            "rank_mismatched_elements": rank_mismatches,
            "layer_seconds": layer_seconds,
            "total_seconds": started.elapsed().as_secs_f64(),
            "devices": sessions.iter().map(|session| json!({
                "ordinal": session.ordinal,
                "measurement_label": session.metadata.get("measurement_label"),
                "compute_processes": session.metadata.get("compute_processes"),
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            out.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    })
    .expect("MiMo GPU audit");
}
