//! Media payload memory through admission and retirement: valid images of a
//! 16 MiP recipe (4 per request) and of a 1 MiP recipe (32 per request) are
//! prepared by the backend, their payloads stay leased while the requests
//! wait and run, a request that finds the process budget taken waits, and
//! retiring requests leaves only the prepared-image cache's idle payloads,
//! which the pool reclaims when a reservation needs their room.

use super::*;
use crate::media_memory::MediaMemoryV1;
use runtime_engine::EngineConfig;
use runtime_kv::KvGeometry;
use runtime_scheduler::SchedulerConfig;

struct Modality {
    maximum_pixels: u64,
    minimum_pixels: u64,
    interpolation: &'static str,
    items: usize,
}

/// 4 images per request under a 16 MiP bicubic recipe; 32 under a 1 MiP
/// bilinear one.
const MODALITIES: [Modality; 2] = [
    Modality {
        maximum_pixels: 16_777_216,
        minimum_pixels: 65_536,
        interpolation: "bicubic-antialias",
        items: 4,
    },
    Modality {
        maximum_pixels: 1_048_576,
        minimum_pixels: 8_192,
        interpolation: "bilinear",
        items: 32,
    },
];

fn backend(
    modality: &Modality,
    root: &std::path::Path,
    pool: Arc<MediaMemoryV1>,
    request_bytes: u64,
) -> EngineGenerationBackend {
    backend_for(
        serde_json::json!({
            "arena": {"rows": 65536, "page_rows": 64, "hidden": 8, "element": "bf16"},
            "encoders": [{"name": "image", "payload_element": "bf16", "payload_columns": 1536,
                "maximum_payload_bytes": 201_326_592, "cost_per_row_milli": 1}],
            "modalities": [{"kind": "image", "part_types": ["image_url"],
                "maximum_items": modality.items, "maximum_bytes": 64 << 20,
                "maximum_tokens": 16_384, "encoder": 0,
                "recipe": {"tag": "image.patch-grid.v1", "parameters": {
                    "factor": 32, "minimum_pixels": modality.minimum_pixels,
                    "maximum_pixels": modality.maximum_pixels,
                    "interpolation": modality.interpolation, "upscale_small_side": false,
                    "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 1.0,
                    "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}},
                "render": {"pad": "[U]"}}]
        }),
        Some(root),
        pool,
        request_bytes,
    )
}

/// A backend serving `multimodal` with a scripted engine.
fn backend_for(
    multimodal: serde_json::Value,
    root: Option<&std::path::Path>,
    pool: Arc<MediaMemoryV1>,
    request_bytes: u64,
) -> EngineGenerationBackend {
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 16, 32, 32, 1).unwrap(),
        KvGeometry::new(4, 12).unwrap(),
        ModelVocabulary::new(259).unwrap(),
        48,
        16,
    )
    .unwrap();
    let engine = Engine::new(config, super::stop_tests::ScriptedModel::default()).unwrap();
    let mut descriptor = FrontendDescriptorV1::from_component_bytes(
        super::stop_tests::declarations::KEY_VALUE.as_bytes(),
    )
    .unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = 1;
    descriptor.vocabulary.stop_token_ids = vec![1];
    descriptor.vocabulary.prompt_special_tokens = vec!["[BOS]".to_owned()];
    descriptor.multimodal = Some(serde_json::from_value(multimodal).unwrap());
    EngineGenerationBackend::new(
        super::stop_tests::tokenizer(),
        engine,
        EngineBackendConfig::new(4).unwrap(),
        EngineModelPolicyV1::from_declaration(&descriptor).with_media_inputs(true),
    )
    .unwrap()
    .with_media_processing(root.map(std::path::Path::to_owned), 0)
    .with_media_memory(pool, request_bytes)
}

/// A request of `items` images, each a distinct valid 256 x 256 PNG.
fn request(root: &std::path::Path, items: usize, first: usize) -> PreparedPrompt {
    let mut segments = vec![PromptSegmentV1::Text("look".to_owned())];
    let media = (first..first + items)
        .map(|image| {
            let path = root.join(format!("{image}.png"));
            if !path.exists() {
                let color = [(image % 251) as u8, (image / 251) as u8, 7];
                std::fs::write(&path, super::tests::solid_png(256, 256, color)).unwrap();
            }
            segments.push(PromptSegmentV1::Media(segments.len() - 1));
            FrontendMediaPartV1 {
                kind: FrontendMediaPartKindV1::Image,
                part_type: "image_url".to_owned(),
                source: format!("file://{}", path.display()),
                format: None,
            }
        })
        .collect();
    PreparedPrompt::MediaChat { segments, media }
}

fn payload_bytes(prompt: &PreparedPrompt) -> u64 {
    match prompt {
        PreparedPrompt::MediaTokens { media, .. } => {
            media.iter().map(|item| item.payload.len() as u64).sum()
        }
        _ => panic!("a prepared media prompt"),
    }
}

#[test]
fn queued_requests_keep_their_payloads_leased_until_they_retire() {
    for modality in &MODALITIES {
        let root = std::env::temp_dir().join(format!(
            "chop-media-lease-{}-{}",
            std::process::id(),
            modality.items
        ));
        std::fs::create_dir_all(&root).unwrap();
        let root = std::fs::canonicalize(&root).unwrap();
        // One request's payloads, measured on an unbounded pool.
        let unbounded = Arc::new(MediaMemoryV1::new(u64::MAX));
        let probe = backend(modality, &root, Arc::clone(&unbounded), u64::MAX);
        let (prepared, _) = probe
            .prepare_prompt(request(&root, modality.items, 0))
            .unwrap();
        // A bicubic image keeps its 8-bit canvas and tables leased, far less
        // than its payload's bytes; a bilinear one keeps its payload.
        let per_request = unbounded.in_use();
        if modality.interpolation == "bicubic-antialias" {
            assert_eq!(
                per_request,
                modality.items as u64 * (256 * 256 * 3 + 3 * 256 * 4)
            );
            // BF16 payloads take 12 bytes per pixel, the canvas 3.
            assert!(3 * per_request < payload_bytes(&prepared));
        } else {
            assert_eq!(per_request, payload_bytes(&prepared));
        }
        drop(prepared);
        // The cache keeps the payloads, idle, until they are reclaimed.
        assert_eq!(unbounded.in_use(), per_request);
        probe.image_cache.reclaim_idle();
        assert_eq!(unbounded.in_use(), 0);

        // Room for three requests' payloads plus one image's preparation.
        let preparation = {
            let source = std::fs::read(root.join("0.png")).unwrap();
            let modality = &probe
                .policy
                .declaration()
                .multimodal
                .as_ref()
                .unwrap()
                .modalities[0];
            let limits = part_limits_v1(modality);
            let probed = runtime_media::probe(&source, limits).unwrap();
            let recipe = declared_image_recipe_v1(&modality.recipe).unwrap();
            image_reservation_v1(&source, probed, (256, 256), recipe, limits).unwrap()
        };
        let allowance = per_request + preparation;
        let pool = Arc::new(MediaMemoryV1::new(3 * per_request + preparation));
        let backend = Arc::new(backend(modality, &root, Arc::clone(&pool), allowance));
        let mut queued = Vec::new();
        for next in 0..3 {
            queued.push(
                backend
                    .prepare_prompt(request(&root, modality.items, next * modality.items))
                    .unwrap()
                    .0,
            );
            assert_eq!(pool.in_use(), queued.len() as u64 * per_request);
        }
        let waiting = request(&root, modality.items, 3 * modality.items);
        let worker_backend = Arc::clone(&backend);
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            tx.send(worker_backend.prepare_prompt(waiting)).unwrap();
        });
        let blocked = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert!(matches!(
            blocked,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
        assert_eq!(pool.in_use(), 3 * per_request);

        // The engine's copy keeps a payload leased after the prepared
        // request is gone; its retirement releases it.
        let running = match queued.remove(0) {
            PreparedPrompt::MediaTokens { media, .. } => media
                .into_iter()
                .map(|item| engine_payload_v1(item.payload))
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        assert_eq!(pool.in_use(), 3 * per_request);
        drop(running);
        queued.push(
            rx.recv_timeout(std::time::Duration::from_secs(10))
                .unwrap()
                .unwrap()
                .0,
        );
        worker.join().unwrap();
        assert_eq!(pool.in_use(), 3 * per_request);
        drop(queued);
        backend.image_cache.reclaim_idle();
        assert_eq!(pool.in_use(), 0);

        // A request whose own images exceed its budget is a client error.
        let tight = backend_with_request_limit(modality, &root, per_request);
        assert!(matches!(
            tight.prepare_prompt(request(&root, modality.items, 0)),
            Err(AdmissionError::Rejected(_))
        ));
        std::fs::remove_dir_all(&root).unwrap();
    }
}

fn backend_with_request_limit(
    modality: &Modality,
    root: &std::path::Path,
    request_bytes: u64,
) -> EngineGenerationBackend {
    backend(
        modality,
        root,
        Arc::new(MediaMemoryV1::new(u64::MAX)),
        request_bytes,
    )
}

#[test]
fn saturated_media_memory_waits_before_preparing_and_honors_cancellation() {
    let modality = &MODALITIES[0];
    let root = std::env::temp_dir().join(format!("media-wait-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let pool = Arc::new(MediaMemoryV1::new(64 << 20));
    let backend = Arc::new(backend(modality, &root, Arc::clone(&pool), 64 << 20));
    for cancel in [false, true] {
        let mut blocker =
            crate::media_memory::RequestMediaMemoryV1::new(Arc::clone(&pool), 64 << 20);
        blocker.reserve(64 << 20).unwrap();
        let prompt = request(&root, modality.items, 0);
        let worker_backend = Arc::clone(&backend);
        let cancellation = CancellationSignal::new();
        let worker_cancel = cancellation.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            tx.send(worker_backend.prepare_prompt_cancellable(prompt, &worker_cancel))
                .unwrap();
        });
        let early = rx.recv_timeout(std::time::Duration::from_millis(100));
        if cancel {
            cancellation.cancel();
        }
        drop(blocker);
        let result = match early {
            Ok(result) => {
                worker.join().unwrap();
                panic!("returned instead of waiting: {result:?}");
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap()
            }
            Err(error) => panic!("{error}"),
        };
        worker.join().unwrap();
        if cancel {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok(), "{result:?}");
        }
        drop(result);
        backend.image_cache.reclaim_idle();
        assert_eq!(pool.in_use(), 0);
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// Standard padded base64, for inline sources.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::new();
    for chunk in bytes.chunks(3) {
        let value = chunk.iter().enumerate().fold(0u32, |value, (index, byte)| {
            value | u32::from(*byte) << (16 - 8 * index)
        });
        for index in 0..4 {
            text.push(if index <= chunk.len() {
                char::from(ALPHABET[(value >> (18 - 6 * index) & 63) as usize])
            } else {
                '='
            });
        }
    }
    text
}

/// A request of `items` copies of one inline image, labelled `mime`.
fn inline_request(items: usize, mime: &str) -> PreparedPrompt {
    let png = super::tests::solid_png(256, 256, [9, 8, 7]);
    let data = base64(&png);
    let mut segments = vec![PromptSegmentV1::Text("look".to_owned())];
    let media = (0..items)
        .map(|_| {
            segments.push(PromptSegmentV1::Media(segments.len() - 1));
            FrontendMediaPartV1 {
                kind: FrontendMediaPartKindV1::Image,
                part_type: "image_url".to_owned(),
                source: format!("data:{mime};base64,{data}"),
                format: None,
            }
        })
        .collect();
    PreparedPrompt::MediaChat { segments, media }
}

/// A request of one inline PNG of each color.
fn inline_colors(colors: &[[u8; 3]]) -> PreparedPrompt {
    let mut segments = vec![PromptSegmentV1::Text("look".to_owned())];
    let media = colors
        .iter()
        .map(|color| {
            segments.push(PromptSegmentV1::Media(segments.len() - 1));
            let png = super::tests::solid_png(256, 256, *color);
            FrontendMediaPartV1 {
                kind: FrontendMediaPartKindV1::Image,
                part_type: "image_url".to_owned(),
                source: format!("data:image/png;base64,{}", base64(&png)),
                format: None,
            }
        })
        .collect();
    PreparedPrompt::MediaChat { segments, media }
}

fn media_inputs(prompt: &PreparedPrompt) -> &[MediaInputV1] {
    match prompt {
        PreparedPrompt::MediaTokens { media, .. } => media,
        _ => panic!("a prepared media prompt"),
    }
}

#[test]
fn media_preparation_preserves_the_frontends_prompt_order() {
    let modality = &MODALITIES[1];
    let root = std::env::temp_dir().join(format!("media-prompt-order-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let root = std::fs::canonicalize(root).unwrap();
    let pool = Arc::new(MediaMemoryV1::new(u64::MAX));
    let backend = backend(modality, &root, pool, 1 << 30);
    let (original, _) = backend.prepare_prompt(request(&root, 3, 0)).unwrap();
    let PreparedPrompt::MediaChat { segments, media } = request(&root, 3, 0) else {
        panic!("image request");
    };
    // The frontend has sorted results by call order and numbered segments
    // against the returned media list, including intervening user images.
    let order = [2, 0, 1];
    let (sorted, _) = backend
        .prepare_prompt(PreparedPrompt::MediaChat {
            segments,
            media: order.iter().map(|index| media[*index].clone()).collect(),
        })
        .unwrap();
    assert_eq!(
        media_inputs(&sorted),
        order.map(|index| media_inputs(&original)[index].clone())
    );
    let PreparedPrompt::MediaTokens { spans, .. } = &sorted else {
        panic!("prepared images have spans");
    };
    assert_eq!(
        spans.iter().map(|span| span.media).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// An image sent again is the item its source prepared before: equal to a
/// fresh preparation and backed by the same payload, so nothing is decoded
/// or leased again.
#[test]
fn a_repeated_image_is_the_item_its_source_prepared_before() {
    let modality = &MODALITIES[1];
    let root = std::env::temp_dir().join(format!("media-cache-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let root = std::fs::canonicalize(&root).unwrap();
    let pool = Arc::new(MediaMemoryV1::new(u64::MAX));
    let backend = backend(modality, &root, Arc::clone(&pool), 1 << 30);
    let (first, _) = backend.prepare_prompt(request(&root, 2, 0)).unwrap();
    let leased = pool.in_use();
    let (again, _) = backend.prepare_prompt(request(&root, 2, 0)).unwrap();
    assert_eq!(pool.in_use(), leased, "a hit leases nothing more");
    assert_eq!(media_inputs(&again), media_inputs(&first));
    for (hit, prepared) in media_inputs(&again).iter().zip(media_inputs(&first)) {
        assert!(hit.payload.ptr_eq(&prepared.payload));
    }
    let fresh = backend_with_request_limit(modality, &root, 1 << 30);
    let (expected, _) = fresh.prepare_prompt(request(&root, 2, 0)).unwrap();
    assert_eq!(media_inputs(&again), media_inputs(&expected));
    // Inline sources hit the entries their bytes made, whatever named them.
    let (inline, _) = backend
        .prepare_prompt(inline_request(2, "image/png"))
        .unwrap();
    assert_eq!(backend.image_cache.len(), 3);
    let (inline_again, _) = backend
        .prepare_prompt(inline_request(2, "image/png"))
        .unwrap();
    assert_eq!(media_inputs(&inline_again), media_inputs(&inline));
    assert_eq!(backend.image_cache.len(), 3);
    std::fs::remove_dir_all(root).unwrap();
}

/// The MIME type picks the decoder, so it is part of the key: PNG bytes
/// cached as PNG and sent again as JPEG still meet the JPEG decoder.
#[test]
fn a_cached_source_under_another_mime_type_is_decoded_as_that_type() {
    let modality = &MODALITIES[1];
    let root = std::env::temp_dir();
    let backend = backend(
        modality,
        &root,
        Arc::new(MediaMemoryV1::new(u64::MAX)),
        1 << 30,
    );
    backend
        .prepare_prompt(inline_request(1, "image/png"))
        .unwrap();
    assert!(matches!(
        backend.prepare_prompt(inline_request(1, "image/jpeg")),
        Err(AdmissionError::Rejected(_))
    ));
}

/// A request whose inline images all hit needs no preparation memory, so it
/// does not wait for an allowance the pool cannot give now.
#[test]
fn a_request_whose_images_all_hit_needs_no_preparation_allowance() {
    let modality = &MODALITIES[1];
    let root = std::env::temp_dir();
    let pool = Arc::new(MediaMemoryV1::new(64 << 20));
    let backend = Arc::new(backend(modality, &root, Arc::clone(&pool), 64 << 20));
    let (first, _) = backend
        .prepare_prompt(inline_request(3, "image/png"))
        .unwrap();
    // The pool's remaining room is taken, so an allowance would wait until
    // the blocker drops, which is after this request must have returned.
    let room = (64 << 20) - pool.in_use();
    let mut blocker = crate::media_memory::RequestMediaMemoryV1::new(Arc::clone(&pool), room);
    blocker.reserve(room).unwrap();
    let worker_backend = Arc::clone(&backend);
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        tx.send(worker_backend.prepare_prompt(inline_request(3, "image/png")))
            .unwrap();
    });
    let again = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("a request whose images all hit does not wait")
        .unwrap()
        .0;
    worker.join().unwrap();
    assert_eq!(media_inputs(&again), media_inputs(&first));
    drop(blocker);
}

/// Idle cached payloads hold room in the pool only until a reservation
/// needs it: the pool reclaims them, least recently used first.
#[test]
fn a_short_pool_reclaims_idle_cached_payloads() {
    let modality = &MODALITIES[1];
    let root = std::env::temp_dir();
    let pool = Arc::new(MediaMemoryV1::new(64 << 20));
    let backend = backend(modality, &root, Arc::clone(&pool), 64 << 20);
    let (prepared, _) = backend
        .prepare_prompt(inline_request(1, "image/png"))
        .unwrap();
    let cached = pool.in_use();
    assert!(cached > 0);
    // A payload a request still holds is not idle.
    let mut full = crate::media_memory::RequestMediaMemoryV1::new(Arc::clone(&pool), 64 << 20);
    assert!(matches!(
        full.reserve(64 << 20),
        Err(AdmissionError::Unavailable(_))
    ));
    assert_eq!(backend.image_cache.len(), 1);
    drop(prepared);
    full.reserve(64 << 20).unwrap();
    assert_eq!(backend.image_cache.len(), 0);
    drop(full);
    assert_eq!(pool.in_use(), 0);
}

/// A request with a miss waits for its allowance holding none of its hits:
/// a held hit is not idle, so the pool could not reclaim it, and here the
/// request would wait for room only that hit takes. Preparation finds the
/// hit again, or prepares it again once the pool has reclaimed it.
#[test]
fn a_request_with_a_miss_waits_holding_none_of_its_hits() {
    let modality = &MODALITIES[1];
    let root = std::env::temp_dir();
    let pool = Arc::new(MediaMemoryV1::new(64 << 20));
    let backend = Arc::new(backend(modality, &root, Arc::clone(&pool), 64 << 20));
    let (first, _) = backend.prepare_prompt(inline_colors(&[[9, 8, 7]])).unwrap();
    drop(first);
    assert!(pool.in_use() > 0);
    let worker_backend = Arc::clone(&backend);
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        tx.send(worker_backend.prepare_prompt(inline_colors(&[[9, 8, 7], [1, 2, 3]])))
            .unwrap();
    });
    let (prepared, _) = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("a request waiting for its allowance holds none of its hits")
        .unwrap();
    worker.join().unwrap();
    let fresh = backend_with_request_limit(modality, &root, 64 << 20);
    let (expected, _) = fresh
        .prepare_prompt(inline_colors(&[[9, 8, 7], [1, 2, 3]]))
        .unwrap();
    assert_eq!(media_inputs(&prepared), media_inputs(&expected));
}

/// One image modality of `recipe` whose encoder stores `element` values in
/// rows of `columns`.
fn image_modality(
    recipe: serde_json::Value,
    element: &str,
    columns: u32,
    tokens: u32,
    layout: bool,
) -> serde_json::Value {
    let mut modality = serde_json::json!({"kind": "image", "part_types": ["image_url"],
        "maximum_items": 4, "maximum_bytes": 64 << 20, "maximum_tokens": tokens, "encoder": 0,
        "recipe": recipe, "render": {"pad": "[U]"}});
    if layout {
        modality["layout"] = "learned-delimiters.v1".into();
    }
    serde_json::json!({
        "arena": {"rows": 65536, "page_rows": 64, "hidden": 8, "element": "bf16"},
        "encoders": [{"name": "image", "payload_element": element, "payload_columns": columns,
            "maximum_payload_bytes": 402_653_184_u64, "cost_per_row_milli": 1}],
        "modalities": [modality]
    })
}

/// A prepared image kept as its 8-bit canvas has the content key, geometry,
/// rows and payload bytes of the F32 rows its recipe makes, for the recipes
/// of each in-tree image model and both payload elements.
#[test]
fn canvas_payloads_keep_the_keys_rows_and_bytes_of_their_rows() {
    let qwen = serde_json::json!({"tag": "image.patch-grid.v1", "parameters": {
        "factor": 32, "minimum_pixels": 65536, "maximum_pixels": 16_777_216,
        "upscale_small_side": false, "interpolation": "bicubic-antialias",
        "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 0.00392156862745098,
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}});
    let mimo = serde_json::json!({"tag": "image.patch-grid.v1", "parameters": {
        "factor": 32, "minimum_pixels": 8192, "maximum_pixels": 1_048_576,
        "upscale_small_side": false, "interpolation": "bicubic-antialias",
        "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 1.0,
        "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]}});
    let glm = serde_json::json!({"tag": TOKEN_BUDGET_TAG_V1, "parameters": {
        "patch": 14, "temporal_patch": 2, "merge": 2, "minimum_tokens": 16,
        "maximum_tokens": 8000, "value_scale": 0.00392156862745098,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711], "alpha_background": "none"}});
    let deepseek = serde_json::json!({"tag": CENTERED_PAD_TAG_V1, "parameters": {
        "patch": 14, "downsample": 3, "minimum_pixels": 295_936, "maximum_tokens": 1024,
        "pad": [127, 127, 127], "value_divisor": 255.0, "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5], "alpha_background": "none", "value_rounding": "bf16"}});
    let declarations = [
        (
            "qwen38",
            image_modality(qwen.clone(), "f32", 1536, 16_384, false),
        ),
        (
            "qwen38-bf16",
            image_modality(qwen, "bf16", 1536, 16_384, false),
        ),
        ("mimo26", image_modality(mimo, "f32", 1536, 1024, false)),
        ("glm5next", image_modality(glm, "f32", 1176, 8000, false)),
        (
            "deepseek41",
            image_modality(deepseek, "f32", 588, 1024, true),
        ),
    ];
    let fixtures = [
        (
            "image/jpeg",
            &include_bytes!(
                "../../../runtime-media/tests/fixtures/centered-pad-jpeg-444-301x199.jpg"
            )[..],
        ),
        (
            "image/jpeg",
            &include_bytes!(
                "../../../runtime-media/tests/fixtures/centered-pad-jpeg-exif6-64x40.jpg"
            )[..],
        ),
        (
            "image/png",
            &include_bytes!(
                "../../../runtime-media/tests/fixtures/zero-padded-token-budget-rgba-translucent-64x50.png"
            )[..],
        ),
    ];
    for (name, multimodal) in declarations {
        let declared: FrontendMultimodalV1 = serde_json::from_value(multimodal.clone()).unwrap();
        let modality = &declared.modalities[0];
        let pool = Arc::new(MediaMemoryV1::new(1 << 32));
        let backend = backend_for(multimodal, None, Arc::clone(&pool), 1 << 32);
        for (mime, bytes) in fixtures {
            let source = runtime_media::MediaSource {
                mime: mime.to_owned(),
                bytes: bytes.to_vec(),
            };
            let request = PreparedPrompt::MediaChat {
                segments: vec![
                    PromptSegmentV1::Text("look".to_owned()),
                    PromptSegmentV1::Media(0),
                ],
                media: vec![FrontendMediaPartV1 {
                    kind: FrontendMediaPartKindV1::Image,
                    part_type: "image_url".to_owned(),
                    source: format!("data:{mime};base64,{}", base64(bytes)),
                    format: None,
                }],
            };
            let (prepared, _) = backend.prepare_prompt(request).unwrap();
            let item = &media_inputs(&prepared)[0];

            // The rows the recipe makes, as the flat payload carried them.
            let limits = part_limits_v1(modality);
            let recipe = declared_image_recipe_v1(&modality.recipe).unwrap();
            let alpha = match recipe {
                ImageRecipeV1::PatchGrid(_) => runtime_media::Alpha::CompositeWhite,
                ImageRecipeV1::TokenBudget(_, alpha) | ImageRecipeV1::CenteredPad(_, alpha) => {
                    alpha
                }
            };
            let decoded =
                decode_image_v1(&source, limits, alpha, image_orientation_v1(recipe)).unwrap();
            let grid = match recipe {
                ImageRecipeV1::PatchGrid(recipe) => runtime_media::recipe::patch_grid(
                    &[decoded.clone(), decoded.clone()],
                    recipe,
                    limits,
                ),
                ImageRecipeV1::TokenBudget(recipe, _) => {
                    runtime_media::recipe::padded_token_budget_grid(&decoded, recipe, limits)
                }
                ImageRecipeV1::CenteredPad(recipe, _) => {
                    runtime_media::recipe::centered_pad_grid(&decoded, recipe, limits)
                }
            }
            .unwrap();
            let span = match modality.layout {
                None => grid.tokens,
                Some(_) => {
                    image_span_rows_v1(modality, recipe_merge_v1(recipe), grid.grid).unwrap()
                }
            };
            let encoder = &declared.encoders[0];
            let flat = media_values_to_bytes_v1(grid.rows, encoder.payload_element).unwrap();
            let key = runtime_media::content_key(
                &modality.recipe.tag,
                &serde_json::to_vec(&modality.recipe.parameters).unwrap(),
                &flat,
            );
            assert_eq!(item.payload.len(), flat.len(), "{name} {mime}");
            assert!(
                item.payload.to_vec() == flat,
                "{name} {mime}: payload bytes"
            );
            let mut written = vec![1, 2, 3];
            item.payload.write_to(&mut written);
            assert!(written[3..] == flat[..], "{name} {mime}: written bytes");
            assert_eq!(item.key, key, "{name} {mime}: content key");
            assert_eq!(
                (item.geometry, item.output_rows),
                ([grid.grid[0], grid.grid[1], grid.grid[2], 0], span as u32),
                "{name} {mime}"
            );
            // The lease is the canvas, not the payload.
            assert!(pool.in_use() < flat.len() as u64, "{name} {mime}");
            drop(prepared);
            backend.image_cache.reclaim_idle();
            assert_eq!(pool.in_use(), 0, "{name} {mime}");
        }
    }
}

/// The Qwen38 recipe with an F32 encoder.
fn qwen38_declaration() -> serde_json::Value {
    image_modality(
        serde_json::json!({"tag": "image.patch-grid.v1", "parameters": {
            "factor": 32, "minimum_pixels": 65536, "maximum_pixels": 16_777_216,
            "upscale_small_side": false, "interpolation": "bicubic-antialias",
            "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 0.00392156862745098,
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}),
        "f32",
        1536,
        16_384,
        false,
    )
}

/// A 301 x 199 JPEG, whole or cut halfway through its scan: the header
/// still plans it, but it cannot decode.
fn jpeg(truncated: bool) -> Vec<u8> {
    let bytes =
        include_bytes!("../../../runtime-media/tests/fixtures/centered-pad-jpeg-444-301x199.jpg");
    if truncated {
        bytes[..bytes.len() / 2].to_vec()
    } else {
        bytes.to_vec()
    }
}

/// A request of inline JPEGs, each a distinct source: the same image with
/// trailing bytes after its end marker.
fn inline_jpegs(images: &[Vec<u8>]) -> PreparedPrompt {
    let mut segments = vec![PromptSegmentV1::Text("look at these".to_owned())];
    let media = images
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            let mut bytes = bytes.clone();
            bytes.extend(std::iter::repeat_n(0, index));
            segments.push(PromptSegmentV1::Media(index));
            segments.push(PromptSegmentV1::Text(" and".to_owned()));
            FrontendMediaPartV1 {
                kind: FrontendMediaPartKindV1::Image,
                part_type: "image_url".to_owned(),
                source: format!("data:image/jpeg;base64,{}", base64(&bytes)),
                format: None,
            }
        })
        .collect();
    PreparedPrompt::MediaChat { segments, media }
}

/// A request of inline images that cannot fit its context or its memory is
/// refused from the images' headers, before any image decodes: a truncated
/// JPEG reports its decode error only when the request fits.
#[test]
fn inline_images_are_refused_from_their_headers_before_decoding() {
    let pool = Arc::new(MediaMemoryV1::new(1 << 32));
    let backend = backend_for(qwen38_declaration(), None, Arc::clone(&pool), 1 << 32);
    let whole = inline_jpegs(&[jpeg(false), jpeg(false)]);
    let cancellation = CancellationSignal::new();
    let (prepared, count) = backend
        .prepare_prompt_within(whole.clone(), &cancellation, Some(1 << 20))
        .unwrap();
    // The plan's count is the prompt's: a ceiling at it admits the prompt,
    // one below refuses it with that count.
    drop(prepared);
    backend.image_cache.reclaim_idle();
    assert!(
        backend
            .prepare_prompt_within(whole.clone(), &cancellation, Some(count))
            .is_ok()
    );
    backend.image_cache.reclaim_idle();
    assert!(matches!(
        backend.prepare_prompt_within(whole, &cancellation, Some(count - 1)),
        Err(AdmissionError::PromptTooLong { prompt_tokens }) if prompt_tokens == count
    ));

    let broken = inline_jpegs(&[jpeg(false), jpeg(true)]);
    assert!(matches!(
        backend.prepare_prompt_within(broken.clone(), &cancellation, Some(count - 1)),
        Err(AdmissionError::PromptTooLong { prompt_tokens }) if prompt_tokens == count
    ));
    assert_eq!(pool.in_use(), 0, "a refused plan prepares nothing");
    // Within the context, the truncated image decodes and fails.
    let error = backend
        .prepare_prompt_within(broken.clone(), &cancellation, Some(count))
        .unwrap_err();
    assert!(
        matches!(&error, AdmissionError::Rejected(message) if !message.contains("host memory")),
        "{error:?}"
    );
    backend.image_cache.reclaim_idle();
    assert_eq!(pool.in_use(), 0);

    // A request limit below the two images' need refuses the second by
    // position, before the first decodes.
    let PreparedPrompt::MediaChat { segments, media } = &broken else {
        unreachable!()
    };
    let plan = backend
        .plan_inline_media_v1(
            backend.policy.declaration().multimodal.as_ref().unwrap(),
            segments,
            media,
            None,
            |text| backend.tokenize(text, false),
        )
        .unwrap();
    let (allowance, first) = (plan.allowance, plan.images[0].reserved);
    drop(plan);
    let tight = backend_for(qwen38_declaration(), None, Arc::clone(&pool), allowance - 1);
    let error = tight
        .prepare_prompt_within(broken, &cancellation, None)
        .unwrap_err();
    assert!(
        matches!(&error, AdmissionError::Rejected(message)
            if message.starts_with("image 2 of 2: ") && message.contains("host memory")),
        "{error:?}"
    );
    assert!(allowance > first && allowance < 1 << 30);
    assert_eq!(pool.in_use(), 0);
}

/// A request of inline images waits for exactly what its plan needs, not a
/// request's whole allowance, honors cancellation while it waits, and holds
/// no decoded source then.
#[test]
fn inline_images_wait_for_their_planned_need() {
    let images = [jpeg(false), jpeg(false), jpeg(false)];
    let probe = backend_for(
        qwen38_declaration(),
        None,
        Arc::new(MediaMemoryV1::new(1 << 32)),
        1 << 32,
    );
    let request = inline_jpegs(&images);
    let PreparedPrompt::MediaChat { segments, media } = &request else {
        unreachable!()
    };
    let need = probe
        .plan_inline_media_v1(
            probe.policy.declaration().multimodal.as_ref().unwrap(),
            segments,
            media,
            None,
            |text| probe.tokenize(text, false),
        )
        .unwrap()
        .allowance;
    assert!(need < REQUEST_MEDIA_PREPARATION_BYTES_V1 / 16, "{need}");

    // Others hold all but the need: the request prepares without waiting.
    let pool = Arc::new(MediaMemoryV1::new(2 * need));
    let backend = Arc::new(backend_for(
        qwen38_declaration(),
        None,
        Arc::clone(&pool),
        REQUEST_MEDIA_PREPARATION_BYTES_V1,
    ));
    let mut others = crate::media_memory::RequestMediaMemoryV1::new(Arc::clone(&pool), u64::MAX);
    others.reserve(need).unwrap();
    let (prepared, _) = backend.prepare_prompt(request.clone()).unwrap();
    drop(prepared);
    backend.image_cache.reclaim_idle();
    assert_eq!(pool.in_use(), need);

    // One byte more held elsewhere and it waits, holding no source, until
    // its client leaves.
    others.reserve(1).unwrap();
    let cancellation = CancellationSignal::new();
    let worker = {
        let (backend, request, cancellation) =
            (Arc::clone(&backend), request.clone(), cancellation.clone());
        std::thread::spawn(move || backend.prepare_prompt_cancellable(request, &cancellation))
    };
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(!worker.is_finished(), "the request waits for its need");
    assert_eq!(
        backend.planning_sources.load(Ordering::SeqCst),
        0,
        "a waiting request holds no decoded source"
    );
    cancellation.cancel();
    assert!(matches!(
        worker.join().unwrap(),
        Err(AdmissionError::Rejected(message)) if message.contains("cancelled")
    ));
    drop(others);
    assert_eq!(pool.in_use(), 0);
}

/// An image that fails to decode partway through a request releases the
/// request's reservation; the images before it stay cached and idle, and
/// the pool reclaims them.
#[test]
fn a_failed_image_releases_its_request_and_leaves_earlier_images_reclaimable() {
    let pool = Arc::new(MediaMemoryV1::new(1 << 32));
    let backend = backend_for(qwen38_declaration(), None, Arc::clone(&pool), 1 << 32);
    let mut good = jpeg(false);
    good.push(7);
    let error = backend
        .prepare_prompt(inline_jpegs(&[jpeg(false), good.clone(), jpeg(true)]))
        .unwrap_err();
    assert!(matches!(error, AdmissionError::Rejected(_)), "{error:?}");
    assert_eq!(backend.image_cache.len(), 2);
    assert!(pool.in_use() > 0, "the prepared images stay cached");
    backend.image_cache.reclaim_idle();
    assert_eq!(pool.in_use(), 0);
    // The cached images are hits for the next request.
    let (prepared, _) = backend
        .prepare_prompt(inline_jpegs(&[jpeg(false), good]))
        .unwrap();
    assert_eq!(media_inputs(&prepared).len(), 2);
}
