//! Image inputs through the frontend a package with the encoder images
//! declares, against the HF processor goldens (`fixtures/qwen38/frontend/images.json`):
//! the rendered and expanded prompt token IDs and the 3-axis positions.
use std::path::Path;

use model_qwen38::tokenizer::{PROMPT_TOKENS, load_tokenizer};
use model_qwen38_contract::frontend::Qwen38Format;
use runtime_engine::{PositionAxesV1, PositionSpanV1};
use runtime_frontend::test_support::documents::PairedFrontendV1;
use runtime_host::image_recipe_v1;
use runtime_media::{MediaLimits, RgbImage, recipe};
use runtime_server::{
    FrontendChatRequestV1, FrontendContentPartV1, FrontendMediaPartKindV1, FrontendMediaPartV1,
    FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1, PromptSegmentV1, ServerFrontendV1,
};
use serde_json::Value;

/// The frontend of a package with the encoder images, through its chat
/// format called directly and through its export.
fn frontend() -> PairedFrontendV1 {
    PairedFrontendV1::chat_format::<Qwen38Format>(model_qwen38::frontend::multimodal_descriptor())
        .unwrap()
}

fn fixture(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/qwen38")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 1 << 26,
        maximum_pixels: 1 << 25,
        maximum_audio_frames: 0,
        maximum_frames: 2,
        maximum_tokens: 1 << 16,
        maximum_aspect_ratio: 200.0,
    }
}

/// `fixtures/qwen38/capture/images.py`, as in `media_recipes.rs`.
fn gradient(width: u32, height: u32) -> RgbImage {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            data.extend([
                (x * 255 / (width - 1).max(1)) as u8,
                (y * 255 / (height - 1).max(1)) as u8,
                if (x * 7 + y * 3) % 97 < 48 { 255 } else { 0 },
            ]);
        }
    }
    RgbImage::new(width, height, data, limits()).unwrap()
}

fn solid(width: u32, height: u32, rgb: [u8; 3]) -> RgbImage {
    let data = rgb.repeat(width as usize * height as usize);
    RgbImage::new(width, height, data, limits()).unwrap()
}

/// The functional MiMo fixture `frames/00.png` (256 x 256) when
/// `CHOPINFER_MIMO_FIXTURES` is set; otherwise an image of its size, which
/// is all the grid reads.
fn frame_00() -> RgbImage {
    if let Some(root) = std::env::var_os("CHOPINFER_MIMO_FIXTURES") {
        let path = Path::new(&root).join("functional/frames/00.png");
        if path.exists() {
            let image =
                runtime_media::png::decode(&std::fs::read(path).unwrap(), limits()).unwrap();
            assert_eq!((image.width, image.height), (256, 256));
            return image;
        }
    }
    solid(256, 256, [255, 0, 0])
}

/// The merged patch grid (t, h, w) of a probe, as the host computes it.
fn probe_grid(name: &str, recipe: recipe::ImageRecipe) -> [u32; 3] {
    let image = match name {
        "blue-square" => solid(256, 256, [0, 0, 255]),
        "frame-00" => frame_00(),
        // 18 MP exceeds the host's 16 MP decode bound; its grid is the
        // resize geometry alone.
        "large-4500x4000" => {
            let (height, width) = recipe::smart_resize(4000, 4500, recipe, limits()).unwrap();
            return [1, height / recipe.patch, width / recipe.patch];
        }
        _ => {
            let sizes = &fixture("image-processor.json")["images"][name];
            let size = |key: &str| sizes[key].as_u64().unwrap() as u32;
            gradient(size("width"), size("height"))
        }
    };
    let grid = recipe::image_grid(&image, recipe, limits()).unwrap();
    let [t, h, w] = grid.grid;
    assert_eq!(u64::from(t * h * w / 4), grid.tokens, "{name}");
    grid.grid
}

#[test]
fn the_merged_descriptor_declares_the_pinned_image_recipe() {
    let frontend = frontend();
    let media = frontend
        .document()
        .descriptor()
        .multimodal
        .as_ref()
        .unwrap();
    // The processor sees decoded RGB, so its capture pins every parameter
    // but the decode's alpha policy, which image-decode.json pins
    // (media_declaration.rs).
    let mut parameters = media.modalities[0].recipe.parameters.clone();
    let alpha = parameters
        .as_object_mut()
        .unwrap()
        .remove("alpha_background");
    assert_eq!(alpha, Some(Value::from("none")));
    let parameters = &parameters;
    let pinned = &fixture("image-processor.json")["recipe"];
    assert_eq!(parameters, pinned);
    assert_eq!(
        format!("{:?}", image_recipe_v1(parameters).unwrap()),
        format!("{:?}", image_recipe_v1(pinned).unwrap())
    );
    for (name, grid) in fixture("image-processor.json")["images"]
        .as_object()
        .unwrap()
    {
        let want: Vec<u32> = serde_json::from_value(grid["grid"].clone()).unwrap();
        let recipe = image_recipe_v1(parameters).unwrap();
        assert_eq!(probe_grid(name, recipe).to_vec(), want, "{name}");
    }
}

struct Expanded {
    /// Token IDs when the tokenizer is available.
    tokens: Option<Vec<u32>>,
    pads: usize,
    spans: Vec<PositionSpanV1>,
}

/// Renders a probe as the capture sent it (one user message with its image
/// parts then the text, `enable_thinking: false`, a generation prompt) and
/// splices the markers' text, then `pad x tokens` per image.
fn expand(probe: &Value, tokenizer: Option<&runtime_tokenizer::Tokenizer>) -> Expanded {
    let frontend = frontend();
    let descriptor = frontend.document().descriptor();
    let modality = &descriptor.multimodal.as_ref().unwrap().modalities[0];
    let recipe = image_recipe_v1(&modality.recipe.parameters).unwrap();
    let merge = recipe.merge;
    let pad = descriptor
        .vocabulary
        .boundary_tokens
        .iter()
        .find(|token| token.text == modality.render.pad)
        .unwrap()
        .id;
    let names: Vec<&str> = probe["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| name.as_str().unwrap())
        .collect();
    let mut parts: Vec<_> = names
        .iter()
        .map(|name| {
            FrontendContentPartV1::Media(FrontendMediaPartV1 {
                kind: FrontendMediaPartKindV1::Image,
                part_type: "image_url".into(),
                source: format!("sha256:{name}"),
                format: None,
            })
        })
        .collect();
    parts.push(FrontendContentPartV1::Text(
        probe["text"].as_str().unwrap().into(),
    ));
    let rendered = frontend
        .render_chat(FrontendChatRequestV1 {
            messages: vec![FrontendMessageV1 {
                role: FrontendRoleV1::User,
                content: None,
                parts: Some(parts),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tools: Vec::new(),
            }],
            continue_final_message: false,
            add_generation_prompt: true,
            tools: Vec::new(),
            tool_choice: FrontendToolChoiceV1::None,
            response_format: None,
            reasoning_effort: None,
            enable_thinking: Some(false),
            thinking: None,
            nested_reasoning_effort: None,
            preserve_thinking: None,
            clear_thinking: None,
            add_vision_id: None,
            tool_objects: Vec::new(),
            chat_template_kwargs: None,
        })
        .unwrap();
    let (segments, media) = (rendered.segments, rendered.media);
    assert_eq!(media.len(), names.len());
    let mut tokens = Vec::new();
    let mut spans = Vec::new();
    let mut pads = 0;
    for segment in &segments {
        match segment {
            PromptSegmentV1::Text(text) => {
                if let Some(tokenizer) = tokenizer {
                    tokens.extend(tokenizer.encode(text, PROMPT_TOKENS).unwrap());
                }
            }
            PromptSegmentV1::Media(index) => {
                let [t, h, w] = probe_grid(names[*index], recipe);
                let rows = t * (h / merge) * (w / merge);
                spans.push(PositionSpanV1 {
                    prompt_row: tokens.len() as u32,
                    rows,
                    grid: [t, h / merge, w / merge],
                });
                tokens.extend(std::iter::repeat_n(pad, rows as usize));
                pads += rows as usize;
            }
        }
    }
    Expanded {
        tokens: tokenizer.map(|_| tokens),
        pads,
        spans,
    }
}

fn tokenizer() -> Option<runtime_tokenizer::Tokenizer> {
    let tokenizer = std::env::var("CHOPINFER_QWEN38_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_QWEN38_CHECKPOINT"))
        .ok()
        .map(|root| load_tokenizer(Path::new(&root)).unwrap());
    if tokenizer.is_none() {
        eprintln!("CHOPINFER_QWEN38_CHECKPOINT unset: comparing pad counts only");
    }
    tokenizer
}

fn check_probe(name: &str) {
    let probes = fixture("frontend/images.json");
    let probe = probes
        .as_array()
        .unwrap()
        .iter()
        .find(|probe| probe["name"] == name)
        .unwrap();
    let tokenizer = tokenizer();
    let expanded = expand(probe, tokenizer.as_ref());
    let want: Vec<u32> = serde_json::from_value(probe["token_ids"].clone()).unwrap();
    assert_eq!(
        expanded.pads,
        probe["image_pad_tokens"].as_u64().unwrap() as usize,
        "{name} pad tokens"
    );
    if let Some(tokens) = expanded.tokens {
        assert_eq!(tokens, want, "{name} token IDs");
        for span in &expanded.spans {
            let start = span.prompt_row as usize;
            // <|vision_start|>, the pad run, <|vision_end|>.
            assert_eq!(want[start - 1], 248053);
            assert!(
                want[start..start + span.rows as usize]
                    .iter()
                    .all(|&id| id == 248056)
            );
            assert_eq!(want[start + span.rows as usize], 248054);
        }
    }
}

#[test]
fn blue_square_expands_like_hf() {
    check_probe("blue-square");
}

#[test]
fn frame_00_expands_like_hf() {
    check_probe("frame-00");
}

#[test]
fn four_images_expand_like_hf() {
    check_probe("four-images");
}

#[test]
fn gradient_333x517_expands_like_hf() {
    check_probe("gradient-333x517");
}

#[test]
fn small_100x60_expands_like_hf() {
    check_probe("small-100x60");
}

#[test]
fn wide_1000x40_expands_like_hf() {
    check_probe("wide-1000x40");
}

#[test]
fn large_4500x4000_expands_like_hf() {
    check_probe("large-4500x4000");
}

/// A transcription of the reference `get_rope_index` for image-only prompts
/// (`Qwen3VLModel`, as the pinned transformers has it): the position IDs of
/// every prompt row and the decode delta.
fn reference_rope_index(ids: &[u32], grids: &[[u32; 3]], merge: u32) -> (Vec<[i64; 3]>, i64) {
    let mut positions: Vec<[i64; 3]> = Vec::with_capacity(ids.len());
    let next = |positions: &Vec<[i64; 3]>| {
        positions
            .iter()
            .flat_map(|p| p.iter().copied())
            .max()
            .map_or(0, |max| max + 1)
    };
    let mut start = 0;
    let mut images = grids.iter();
    // Each image's pad run begins after a <|vision_start|>.
    for (vision_start, _) in ids.iter().enumerate().filter(|&(_, &id)| id == 248053) {
        let end = vision_start + 1;
        let [t, h, w] = images.next().unwrap().map(i64::from);
        let (h, w) = (h / i64::from(merge), w / i64::from(merge));
        let base = next(&positions);
        for k in 0..(end - start) as i64 {
            positions.push([base + k; 3]);
        }
        let base = next(&positions);
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    positions.push([base + ti, base + hi, base + wi]);
                }
            }
        }
        start = end + (t * h * w) as usize;
    }
    assert!(images.next().is_none());
    let base = next(&positions);
    for k in 0..(ids.len() - start) as i64 {
        positions.push([base + k; 3]);
    }
    let delta = next(&positions) - ids.len() as i64;
    (positions, delta)
}

#[test]
fn four_image_positions_follow_the_reference_rope_index() {
    let probes = fixture("frontend/images.json");
    let probe = probes
        .as_array()
        .unwrap()
        .iter()
        .find(|probe| probe["name"] == "four-images")
        .unwrap();
    // The spans' rows do not depend on the tokenizer: the captured IDs
    // locate the text around them.
    let expanded = expand(probe, None);
    let ids: Vec<u32> = serde_json::from_value(probe["token_ids"].clone()).unwrap();
    let recipe = image_recipe_v1(&fixture("image-processor.json")["recipe"]).unwrap();
    let mut grids = Vec::new();
    let mut spans = Vec::new();
    let mut row = 0;
    for span in &expanded.spans {
        let start = row + ids[row..].iter().position(|&id| id == 248056).unwrap();
        let [t, h, w] = span.grid;
        grids.push([t, h * recipe.merge, w * recipe.merge]);
        spans.push(PositionSpanV1 {
            prompt_row: start as u32,
            ..*span
        });
        row = start + span.rows as usize;
    }
    assert_eq!(spans.len(), 4);
    let (reference, delta) = reference_rope_index(&ids, &grids, recipe.merge);
    let axes = PositionAxesV1::new(ids.len(), ids.len() + 64, &spans).unwrap();
    assert_eq!(axes.delta(), delta);
    assert!(delta < 0);
    for (row, want) in reference.iter().enumerate() {
        assert_eq!(axes.position(row).map(i64::from), *want, "row {row}");
    }
    for row in ids.len()..ids.len() + 64 {
        assert_eq!(
            axes.position(row).map(i64::from),
            [row as i64 + delta; 3],
            "row {row}"
        );
    }
}
