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
    descriptor.multimodal = Some(
        serde_json::from_value(serde_json::json!({
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
        }))
        .unwrap(),
    );
    EngineGenerationBackend::new(
        super::stop_tests::tokenizer(),
        engine,
        EngineBackendConfig::new(4).unwrap(),
        EngineModelPolicyV1::from_declaration(&descriptor).with_media_inputs(true),
    )
    .unwrap()
    .with_media_processing(Some(root.to_owned()), 0)
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
        let per_request = payload_bytes(&prepared);
        assert_eq!(unbounded.in_use(), per_request);
        drop(prepared);
        // The cache keeps the payloads, idle, until they are reclaimed.
        assert_eq!(unbounded.in_use(), per_request);
        probe.image_cache.reclaim_idle();
        assert_eq!(unbounded.in_use(), 0);

        // Room for three requests' payloads plus one image's preparation.
        let limit = 3 * per_request + per_request / modality.items as u64 * 8;
        let pool = Arc::new(MediaMemoryV1::new(limit));
        let backend = Arc::new(backend(
            modality,
            &root,
            Arc::clone(&pool),
            per_request + per_request / modality.items as u64 * 8,
        ));
        let mut queued = Vec::new();
        for next in 0..3 {
            queued.push(
                backend
                    .prepare_prompt(request(&root, modality.items, next * modality.items))
                    .unwrap()
                    .0,
            );
            assert_eq!(pool.in_use(), queued.iter().map(payload_bytes).sum::<u64>());
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
