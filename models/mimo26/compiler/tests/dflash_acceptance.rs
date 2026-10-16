//! Offline DFlash acceptance on the spec_bench GSM8K prompts, on the CPU.
//!
//! The f64 layer oracle generates each prompt's greedy continuation and keeps
//! the target taps of every position. The target is causal and greedy
//! verification accepts exactly the drafts that equal the greedy tokens, so a
//! draft block anchored at any generated position can be scored against the
//! recorded continuation without running the target again.
//!
//! Every anchor drafts with two-, four- and eight-row blocks of the vLLM
//! fork's semantics (`model_mimo26::oracle::dflash`), and with an eight-row
//! block whose K and V pass through the FP8 E4M3 cache the deployed vLLM
//! drafter uses. The receipt lists each anchor's accepted draft count per
//! block shape and replays two verify schedules over them:
//!
//! - vLLM K7: every block has eight rows and every step verifies seven
//!   drafts, and the counters include drafts accepted past the output limit;
//! - Chopinfer fixed K: a block of K + 1 rows, with K clamped to the tokens
//!   left after the bonus and rounded down to a row bucket of 1, 2, 4 or 8.
//!
//! Environment: `CHOPINFER_MIMO_CHECKPOINT`, `CHOPINFER_MIMO_FIXTURES`,
//! `CHOPINFER_DFLASH_ACCEPTANCE_RECEIPT` (output JSON), optionally
//! `CHOPINFER_DFLASH_ACCEPTANCE_PROMPTS` (comma-separated `C:item` pairs of
//! spec_bench's GSM8K cohorts, default `4:0,4:1,4:2,4:3`) and
//! `CHOPINFER_DFLASH_ACCEPTANCE_TOKENS` (output tokens, default 64).

use std::path::Path;
use std::time::Instant;

use frontend_format::{
    ChatFormatV1, FormatDescriptorV1, FormatRenderRequestV1, FrontendMessageV1, FrontendRoleV1,
};
use kernel_core::select::{NanKeys, Ties, topk};
use model_mimo26::frontend::descriptor;
use model_mimo26::oracle::dflash::{ContextLayer, DflashOracle};
use model_mimo26::oracle::{KvHistory, LayerOracle, ModelEnds};
use model_mimo26::tokenizer::{load_tokenizer, special_tokens};
use serde_json::{Value, json};

const HIDDEN: usize = 4096;
const LAYERS: usize = 48;
const TAP_LAYERS: [usize; 5] = [0, 11, 23, 35, 47];
/// Entries of one context layer per position: 8 KV heads of 128.
const ENTRY: usize = 8 * 128;
const BLOCKS: [usize; 3] = [2, 4, 8];
const RUN_ID: &str = "spec-001";

/// The first strict maximum's index (+0 equal to -0), as torch.argmax picks.
fn argmax(values: &[f64]) -> u32 {
    let best = topk(values, 1, Ties::IeeeLowerIndex, NanKeys::Refuse, None);
    best.expect("finite logits")[0] as u32
}

/// spec_bench's GSM8K request body for cohort `concurrency`, item `index`.
fn prompt(fixtures: &Path, concurrency: usize, index: usize) -> (String, String) {
    let read = |name: &str| -> Value {
        serde_json::from_slice(&std::fs::read(fixtures.join("gsm8k").join(name)).expect(name))
            .expect(name)
    };
    let manifest = read("manifest.json");
    let examples = read("examples.json");
    let instruction = format!(
        "{} Explain your reasoning in at least ten numbered steps before the final answer.",
        manifest["instruction"].as_str().expect("instruction")
    );
    (
        format!("Spec qualification {RUN_ID} gsm8k C{concurrency} item {index}. {instruction}"),
        examples[index]["question"]
            .as_str()
            .expect("question")
            .to_owned(),
    )
}

/// The greedy continuation of `prompt` (`extra` tokens) and the context
/// feature of every position through `features` - 1, [row][5 x 4096].
fn generate(
    ends: &ModelEnds,
    layers: &mut [LayerOracle],
    prompt: &[u32],
    extra: usize,
    features: usize,
) -> (Vec<u32>, Vec<u16>) {
    let mut histories = vec![KvHistory::default(); LAYERS];
    let mut ids = prompt.to_vec();
    let mut taps = Vec::new();
    let mut input = prompt.to_vec();
    while ids.len() < prompt.len() + extra {
        let mut hidden = ends.embed(&input).expect("embedding");
        let mut rows = vec![Vec::with_capacity(5 * HIDDEN); input.len()];
        for (layer, oracle) in layers.iter_mut().enumerate() {
            hidden = oracle
                .run(&hidden, &mut histories[layer])
                .expect("target layer");
            oracle.release_experts();
            if TAP_LAYERS.contains(&layer) {
                for (row, values) in rows.iter_mut().zip(hidden.chunks_exact(HIDDEN)) {
                    row.extend_from_slice(values);
                }
            }
        }
        taps.extend(rows.into_iter().flatten());
        let last = ends
            .final_norm(&hidden[hidden.len() - HIDDEN..])
            .expect("final norm");
        let next = argmax(&ends.logits(&last)[0]);
        ids.push(next);
        input = vec![next];
    }
    taps.truncate(features * 5 * HIDDEN);
    (ids, taps)
}

fn truncated(context: &[ContextLayer], positions: usize) -> Vec<ContextLayer> {
    context
        .iter()
        .map(|layer| ContextLayer {
            keys: layer.keys[..positions * ENTRY].to_vec(),
            values: layer.values[..positions * ENTRY].to_vec(),
        })
        .collect()
}

/// Drafts of a block of `rows` rows whose bonus token sits at `anchor`.
fn drafts(
    oracle: &DflashOracle,
    ends: &ModelEnds,
    context: &[ContextLayer],
    ids: &[u32],
    anchor: usize,
    rows: usize,
    fp8_kv: bool,
) -> Vec<u32> {
    let mut embeddings = ends.embed(&ids[anchor..=anchor]).expect("bonus embedding");
    for _ in 1..rows {
        embeddings.extend_from_slice(oracle.mask_embedding());
    }
    let normalized = oracle
        .block_emulated(&embeddings, anchor, context, fp8_kv)
        .expect("draft block");
    ends.logits(&normalized[HIDDEN..])
        .iter()
        .map(|row| argmax(row))
        .collect()
}

fn matched(drafts: &[u32], truth: &[u32]) -> usize {
    drafts
        .iter()
        .zip(truth)
        .take_while(|(draft, token)| draft == token)
        .count()
}

/// Verify steps from the first generated token until `output` tokens exist.
/// `depth` gives the step's draft count from the tokens left and `accepted`
/// the drafts it accepts at an anchor. Returns the receipt entry.
fn replay(
    output: usize,
    depth: impl Fn(usize) -> usize,
    accepted: impl Fn(usize, usize) -> usize,
) -> Value {
    let (mut emitted, mut anchor) = (1, 0);
    let (mut items, mut decodes, mut drafted, mut counted) = (0, 0, 0, 0);
    let mut kept = 0;
    let mut positions = [0_usize; 7];
    while emitted < output {
        let k = depth(output - emitted);
        if k == 0 {
            decodes += 1;
            emitted += 1;
            anchor += 1;
            continue;
        }
        let n = accepted(anchor, k).min(k);
        items += 1;
        drafted += k;
        counted += n;
        kept += n.min(output - emitted);
        for position in positions.iter_mut().take(n) {
            *position += 1;
        }
        emitted += n + 1;
        anchor += n + 1;
    }
    json!({
        "verify_items": items, "zero_draft_steps": decodes, "draft_tokens": drafted,
        "accepted_tokens": counted, "accepted_within_output": kept,
        "accepted_positions": positions,
        "acceptance_length": 1.0 + counted as f64 / items as f64,
        "acceptance_length_within_output": 1.0 + kept as f64 / items as f64,
        "tokens_per_verify_item": (output - 1 - decodes) as f64 / items as f64,
    })
}

/// Largest row bucket at or below `rows`, minus the bonus row.
fn bucketed(rows: usize) -> usize {
    [8, 4, 2, 1]
        .into_iter()
        .find(|&bucket| bucket <= rows)
        .unwrap_or(1)
        - 1
}

#[test]
#[ignore = "CPU only, but runs the f64 target over each prompt; needs the MiMo checkpoint"]
fn dflash_acceptance_on_greedy_gsm8k_continuations() {
    let (Some(checkpoint), Some(fixtures), Some(receipt)) = (
        std::env::var_os("CHOPINFER_MIMO_CHECKPOINT"),
        std::env::var_os("CHOPINFER_MIMO_FIXTURES"),
        std::env::var_os("CHOPINFER_DFLASH_ACCEPTANCE_RECEIPT"),
    ) else {
        eprintln!(
            "skipping: set CHOPINFER_MIMO_CHECKPOINT, CHOPINFER_MIMO_FIXTURES and CHOPINFER_DFLASH_ACCEPTANCE_RECEIPT"
        );
        return;
    };
    let (checkpoint, fixtures) = (Path::new(&checkpoint), Path::new(&fixtures));
    let output = std::env::var("CHOPINFER_DFLASH_ACCEPTANCE_TOKENS")
        .map_or(64, |value| value.parse::<usize>().expect("output tokens"));
    let cohorts = std::env::var("CHOPINFER_DFLASH_ACCEPTANCE_PROMPTS")
        .unwrap_or_else(|_| "4:0,4:1,4:2,4:3".into())
        .split(',')
        .map(|pair| {
            let (concurrency, item) = pair.split_once(':').expect("C:item");
            (
                concurrency.parse::<usize>().expect("concurrency"),
                item.parse::<usize>().expect("item"),
            )
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    let tokenizer = load_tokenizer(checkpoint).expect("tokenizer");
    let ends = ModelEnds::load(checkpoint).expect("embedding and head");
    let mut layers = (0..LAYERS)
        .map(|layer| LayerOracle::load(checkpoint, layer).expect("layer oracle"))
        .collect::<Vec<_>>();
    let oracle = DflashOracle::load(checkpoint).expect("DFlash oracle");
    let format = model_mimo26_contract::frontend::Mimo26Format::open(
        &FormatDescriptorV1::new(descriptor()).unwrap(),
    )
    .unwrap();
    let message = |role, content| FrontendMessageV1 {
        role,
        content: Some(content),
        parts: None,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    };
    let mut prompts = Vec::new();
    for (concurrency, index) in cohorts {
        let (system, user) = prompt(fixtures, concurrency, index);
        let text = format
            .render(FormatRenderRequestV1 {
                messages: vec![
                    message(FrontendRoleV1::System, system),
                    message(FrontendRoleV1::User, user),
                ],
                continue_final_message: false,
                add_generation_prompt: true,
                tools: &[],
                tool_objects: Vec::new(),
                response_format: None,
                reasoning_effort: None,
                enable_thinking: None,
                thinking: None,
                nested_reasoning_effort: None,
                preserve_thinking: None,
                clear_thinking: None,
                add_vision_id: None,
            })
            .expect("chat render")
            .prompt;
        let ids = tokenizer
            .encode(
                &text,
                &special_tokens()
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
            .expect("tokenize");
        let length = ids.len();
        eprintln!("C{concurrency} item {index}: {length} prompt tokens");
        // An anchor at the last output position looks seven tokens ahead.
        let (ids, features) = generate(&ends, &mut layers, &ids, output + 7, length + output - 1);
        eprintln!(
            "C{concurrency} item {index}: generated after {:.0} s",
            started.elapsed().as_secs_f64()
        );
        let positions = (0..length + output - 1).collect::<Vec<_>>();
        let context = oracle.context(&features, &positions).expect("context");
        let mut table = Vec::new();
        for anchor in length..length + output {
            let context = truncated(&context, anchor);
            let truth = &ids[anchor + 1..];
            let mut row = serde_json::Map::new();
            for (rows, fp8_kv) in BLOCKS
                .map(|rows| (rows, false))
                .into_iter()
                .chain([(8, true)])
            {
                let tokens = drafts(&oracle, &ends, &context, &ids, anchor, rows, fp8_kv);
                let name = if fp8_kv {
                    "8_fp8_kv".to_owned()
                } else {
                    rows.to_string()
                };
                row.insert(format!("accepted_{name}"), json!(matched(&tokens, truth)));
                row.insert(format!("drafts_{name}"), json!(tokens));
            }
            row.insert("truth".into(), json!(&truth[..7]));
            table.push(Value::Object(row));
        }
        eprintln!(
            "C{concurrency} item {index}: drafted after {:.0} s",
            started.elapsed().as_secs_f64()
        );
        let column = |key: &str| {
            table
                .iter()
                .map(|row| row[key].as_u64().expect("accepted count") as usize)
                .collect::<Vec<_>>()
        };
        let (two, four, eight, fp8) = (
            column("accepted_2"),
            column("accepted_4"),
            column("accepted_8"),
            column("accepted_8_fp8_kv"),
        );
        // Chopinfer's block has K + 1 rows; K is 1, 3 or 7 after bucketing.
        let own = |anchor: usize, k: usize| match k {
            1 => two[anchor],
            3 => four[anchor],
            _ => eight[anchor],
        };
        let clamped = |limit: usize| move |left: usize| bucketed(limit.min(left - 1) + 1);
        let walks = json!({
            "vllm_k7": replay(output, |_| 7, |anchor, _| eight[anchor]),
            "vllm_k7_fp8_kv": replay(output, |_| 7, |anchor, _| fp8[anchor]),
            "chopinfer_k7": replay(output, clamped(7), own),
            "chopinfer_k7_eight_row_blocks": replay(output, clamped(7), |anchor, _| eight[anchor]),
            "chopinfer_k3": replay(output, clamped(3), own),
            "chopinfer_k3_eight_row_blocks": replay(output, clamped(3), |anchor, _| eight[anchor]),
            "chopinfer_k1": replay(output, clamped(1), own),
            "chopinfer_k1_eight_row_blocks": replay(output, clamped(1), |anchor, _| eight[anchor]),
        });
        prompts.push(json!({
            "cohort": format!("C{concurrency}"), "item": index, "prompt_tokens": length,
            "continuation": tokenizer.decode_utf8(&ids[length..length + output]).ok(),
            "tokens": &ids[length..], "anchors": table, "walks": walks,
        }));
        std::fs::write(
            &receipt,
            serde_json::to_vec_pretty(&json!({
                "semantics": "model_mimo26::oracle::dflash (vLLM fork: window 1023 behind each \
                              query, bidirectional block, sink, V x 0.612); f64 target oracle",
                "output_tokens": output, "prompts": prompts,
                "seconds": started.elapsed().as_secs_f64(),
            }))
            .expect("receipt JSON"),
        )
        .expect("write receipt");
    }
}
