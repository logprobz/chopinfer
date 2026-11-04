//! KV-state taps on the CPU fake host: the taps of each KV layout read the
//! bytes the fake device stores, and the bitwise comparison of two
//! layouts' captures by layer, plane, position and rank (the comparison
//! `tools/qual/kv_layout_equality.py taps` performs) finds one corrupted
//! stored byte of either layout at exactly its layer, plane and position.
//!
//! Every rank's window rings of slot 0 and the pages of request 7 hold
//! fake-quantized rows of the trained formats (native codes and scales, or
//! their BF16 products), distinct per rank, in the tapped layers: the
//! first and last, a window-only layer, a ratio-2 KV source and the ratio-1
//! source. Request 7 prefills 9 tokens and decodes one; its state is
//! captured to snapshot pages and restored into slot 2 as request 9, which
//! prefills 5 more tokens on the same pages.
use super::*;

use std::path::{Path, PathBuf};

use bundle_runtime::fake_host::dump::Origin;
use kernel_core::quantize::Scales;
use model_deepseek41_contract::state::{self, Widths};

use crate::taps::kv::{self, Plane};

const PROMPT: u64 = 9;
const CONTINUATION: u64 = 5;
/// The positions whose state the fake device stores.
const STORED: u64 = PROMPT + 1 + CONTINUATION;
const ITEMS: u32 = 4;
/// The tapped layers (`CHOPINFER_TAPS_LAYERS`).
const LAYERS: [u32; 5] = [0, 5, 8, 20, 39];
/// The test's child process reads its layout and corruption here.
const LAYOUT: &str = "DEEPSEEK41_KV_TAPS_LAYOUT";
const CORRUPT: &str = "DEEPSEEK41_KV_TAPS_CORRUPT";

/// One stored row: a layer's plane at a position on a rank.
type Row = (u32, Plane, u64, u32);

/// A stored byte to flip: its row, whether a scale byte (else a code or
/// BF16 value byte), its byte in the row and the bits flipped.
#[derive(Clone, Copy, Debug)]
struct Corruption {
    row: Row,
    scale: bool,
    byte: usize,
    bits: u8,
}

fn plane_name(plane: Plane) -> &'static str {
    match plane {
        Plane::Window => "window",
        Plane::Compressed => "compressed",
        Plane::Index => "index",
    }
}

fn parse_plane(name: &str) -> Plane {
    match name {
        "window" => Plane::Window,
        "compressed" => Plane::Compressed,
        "index" => Plane::Index,
        _ => panic!("no plane {name}"),
    }
}

/// The corruptions the test applies: a native index key's code byte, a
/// native compressed latent's E4M3 scale byte and a BF16 window value's
/// low byte.
fn corruption(name: &str) -> Corruption {
    match name {
        "native-index-code" => Corruption {
            row: (8, Plane::Index, 2, 1),
            scale: false,
            byte: 3,
            bits: 0x10,
        },
        "native-compressed-scale" => Corruption {
            row: (20, Plane::Compressed, 6, 3),
            scale: true,
            byte: 5,
            bits: 0x01,
        },
        "bf16-window-value" => Corruption {
            row: (5, Plane::Window, 3, 2),
            scale: false,
            byte: 64,
            bits: 0x01,
        },
        _ => panic!("no corruption {name}"),
    }
}

/// A row's F32 inputs: pseudo-random values whose blocks span a range of
/// magnitudes, distinct per rank.
fn inputs((layer, plane, position, rank): Row, width: usize) -> Vec<f32> {
    let seed = (u64::from(layer) << 40)
        ^ ((plane as u64) << 36)
        ^ (position << 8)
        ^ u64::from(rank)
        ^ 0x9e37_79b9_7f4a_7c15;
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut values = Vec::with_capacity(width);
    let mut scale = 1.0;
    for column in 0..width {
        if column % 16 == 0 {
            scale = 2_f32.powi((next() % 9) as i32 - 4);
        }
        let unit = (next() >> 11) as f32 / (1_u64 << 53) as f32;
        values.push((unit * 2.0 - 1.0) * 6.0 * scale);
    }
    values
}

/// A row as each layout stores it: native codes and scales, and the BF16
/// products, the logical values, little-endian.
struct Encoded {
    codes: Vec<u8>,
    scales: Vec<u8>,
    values: Vec<u8>,
}

fn encode(row: Row, w: Widths) -> Encoded {
    let plane = row.1;
    let width = plane.width(w) as usize;
    let inputs = inputs(row, width);
    let quantized = plane.format().apply(&inputs, 1, width).unwrap();
    let (Scales::Ue8M0(scales) | Scales::Ue4M3(scales)) = quantized.scales else {
        panic!("byte scales")
    };
    let values = plane
        .format()
        .fake_quant(&inputs, 1, width)
        .unwrap()
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    Encoded {
        codes: quantized.codes,
        scales,
        values,
    }
}

/// Every stored row: each tapped layer's window at every stored position,
/// and each tapped KV source's compressed latents and index keys at every
/// completed compressed position, on every rank.
fn rows(ranks: u32) -> Vec<Row> {
    let g = &FLASH;
    let mut rows = Vec::new();
    for rank in 0..ranks {
        for layer in LAYERS {
            for position in 0..STORED {
                rows.push((layer, Plane::Window, position, rank));
            }
        }
        for &layer in g.shape.csa.kv_sources {
            if !LAYERS.contains(&layer) {
                continue;
            }
            let ratio = u64::from(g.compress_ratio(layer));
            for plane in [Plane::Compressed, Plane::Index] {
                for position in 0..STORED / ratio {
                    rows.push((layer, plane, position, rank));
                }
            }
        }
    }
    rows
}

/// Where `row`'s codes (or BF16 values) and scales lie in `storage`, from
/// the state declaration alone: slot 0's ring cell in each rank's
/// `window.ring` (`states`), or the compressed position's entry in page
/// `position * ratio / 64` (request 7's page table is 0, 1, ...).
fn spans(
    row: Row,
    storage: KvStorage,
    states: &[crate::buffers::StateSpans],
    page_bytes: u64,
) -> (DeviceBufferSpanV1, Option<DeviceBufferSpanV1>) {
    let g = &FLASH;
    let (layer, plane, position, rank) = row;
    let w = Widths::of(g);
    let (code_bytes, scale_bytes) = plane.row_bytes(w, storage);
    let declaration = state::request_state(g, storage, false);
    let at = |span: DeviceBufferSpanV1, offset: u64, length: u64| {
        assert!(offset + length <= span.length);
        DeviceBufferSpanV1::new(span.buffer, span.offset + offset, length)
    };
    if plane == Plane::Window {
        let ring = states[rank as usize].named("window.ring").unwrap();
        let buffer = declaration
            .buffers
            .iter()
            .find(|buffer| buffer.name == "window.ring")
            .unwrap();
        let stride = ring.length / buffer.layers;
        let entries = stride / buffer.bytes_per_layer_rank();
        assert_eq!(entries * buffer.bytes_per_layer_rank(), stride);
        let cells = state::ring_cells(w);
        let cell = position % cells;
        let base = u64::from(layer) * stride;
        let codes = at(ring, base + cell * code_bytes, code_bytes);
        let scales = (scale_bytes > 0).then(|| {
            at(
                ring,
                base + entries * cells * code_bytes + cell * scale_bytes,
                scale_bytes,
            )
        });
        return (codes, scales);
    }
    let ratio = g.compress_ratio(layer);
    let group = if ratio == 2 {
        "csa2.ratio2"
    } else {
        "csa2.ratio1"
    };
    let local = state::sources_of(g, ratio)
        .position(|source| source == layer)
        .unwrap() as u64;
    let stream = if plane == Plane::Compressed {
        "latent"
    } else {
        "index-key"
    };
    let layout = declaration
        .page_planes()
        .into_iter()
        .find(|p| p.layers == group && p.layer == local && p.stream == stream)
        .unwrap();
    let per_page = declaration.page_tokens / u64::from(ratio);
    let pool = DeviceBufferSpanV1::new(
        BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank),
        0,
        KV_PAGES * page_bytes,
    );
    let (page, entry) = (position / per_page, position % per_page);
    let base = page * page_bytes + layout.offset;
    let codes = at(pool, base + entry * code_bytes, code_bytes);
    let scales = (scale_bytes > 0).then(|| {
        at(
            pool,
            base + per_page * code_bytes + entry * scale_bytes,
            scale_bytes,
        )
    });
    (codes, scales)
}

/// Stores every row in `storage` on `host`, the fake device's memory, with
/// `corrupt` applied.
fn store(
    host: &FakeHost,
    instance: &BundleInstanceV1,
    storage: KvStorage,
    corrupt: Option<Corruption>,
) {
    let (_, states) = crate::instance::anchor_spans(instance.handle());
    let w = Widths::of(&FLASH);
    let write = |span: DeviceBufferSpanV1, bytes: Vec<u8>| {
        assert_eq!(span.length as usize, bytes.len());
        host.memory
            .borrow_mut()
            .write(span, vec![(span.length, Origin::Uploaded(bytes.into(), 0))]);
    };
    for row in rows(host.shape.ranks()) {
        let Encoded {
            mut codes,
            mut scales,
            mut values,
        } = encode(row, w);
        if let Some(c) = corrupt.filter(|c| c.row == row) {
            let bytes = match (storage, c.scale) {
                (KvStorage::Native, true) => &mut scales,
                (KvStorage::Native, false) => &mut codes,
                (KvStorage::Bf16, _) => &mut values,
            };
            bytes[c.byte] ^= c.bits;
        }
        let (code_span, scale_span) = spans(row, storage, &states, host.page_bytes());
        match storage {
            KvStorage::Native => {
                write(code_span, codes);
                write(scale_span.unwrap(), scales);
            }
            KvStorage::Bf16 => write(code_span, values),
        }
    }
    host.read_memory.set(true);
}

/// A prompt chunk of `count` tokens after `previous` of `request` in
/// `slot` on `pages`.
fn chunk(request: u64, slot: u64, previous: u64, count: u64, pages: Vec<u64>) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + count;
    let inputs = (previous..end).map(|token| token as u32).collect();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, false, count as u32, true, inputs)],
        vec![BatchAppendV1::new(previous, count, end, pages, None)],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

/// The child: one layout's lifecycle with the fake device's stored rows.
fn lifecycle(shape: Shape, layout: KvLayout, corrupt: Option<Corruption>) {
    let host = FakeHost::new(shape, layout);
    let mut instance = create(&host, ITEMS);
    store(&host, &instance, layout.storage, corrupt);
    let mut views = Views::default();
    step(&mut instance, &prompt(7, 0, 0, PROMPT, true), &mut views);
    step(&mut instance, &decode(&[(7, 0, PROMPT, 3)]), &mut views);
    let captured = PROMPT + 1;
    let count = instance.describe_prefix_state().unwrap();
    let snapshot: Vec<u64> = (100..100 + count).collect();
    instance
        .capture_prefix_state(7, captured, &snapshot)
        .unwrap();
    instance.retire_request(7).unwrap();
    let ids: Vec<u32> = (0..captured as u32).collect();
    instance
        .restore_prefix_state_in_slot(9, &ids, &snapshot, 2)
        .unwrap();
    // The restored request shares request 7's pages, as a prefix hit does.
    let pages = (0..STORED.div_ceil(64)).collect();
    step(
        &mut instance,
        &chunk(9, 2, captured, CONTINUATION, pages),
        &mut views,
    );
    // A cancelled step's KV steps are removed.
    instance
        .execute(&decode(&[(9, 2, STORED, 3)]), &mut views)
        .unwrap();
    instance.cancel().unwrap();
    instance.retire_request(9).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    assert!(
        host.logs
            .borrow()
            .iter()
            .all(|line| !line.contains("KV state")),
        "{:?}",
        host.logs.borrow()
    );
}

/// A readback: (request, phase, the state's end).
type Event = (u64, String, u64);
/// One row of a capture: its readback, layer, plane, position and rank.
type Key = (Event, u32, &'static str, u64, u32);

/// A capture's rows: the logical values, and the native codes and scales.
#[derive(Default)]
struct Capture {
    values: BTreeMap<Key, Vec<u8>>,
    codes: BTreeMap<Key, Vec<u8>>,
    scales: BTreeMap<Key, Vec<u8>>,
    /// Each step's (program, phase, axis, ratio, rows written).
    steps: Vec<(String, String, String, u64, u64)>,
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// Reads the KV steps under `root` in the representation `taps.json`
/// documents: each index entry's rows are the step's rows from
/// `first_row`, whose positions `items.json` `kv.positions` lists, of the
/// layer and plane its name gives, in its rank's file.
fn read(root: &Path) -> Capture {
    let mut capture = Capture::default();
    let mut steps: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect();
    steps.sort();
    for step in steps {
        let header = json(&step.join("capture.json"));
        let items = json(&step.join("items.json"));
        assert_eq!(json(&step.join("completion.json"))["completion"], "full");
        let kv = &items["kv"];
        let event: Event = (
            items["items"][0]["request"].as_u64().unwrap(),
            kv["phase"].as_str().unwrap().to_owned(),
            kv["end"].as_u64().unwrap(),
        );
        let positions: Vec<u64> = kv["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_u64().unwrap())
            .collect();
        let mut written = 0;
        for line in std::fs::read_to_string(step.join("index.jsonl"))
            .unwrap()
            .lines()
        {
            let entry: serde_json::Value = serde_json::from_str(line).unwrap();
            let name = entry["name"].as_str().unwrap();
            let rank = entry["rank"].as_u64().unwrap() as u32;
            let (first, rows) = (
                entry["first_row"].as_u64().unwrap(),
                entry["rows"].as_u64().unwrap(),
            );
            let row_bytes = (entry["cols"].as_u64().unwrap()
                * match entry["dtype"].as_str().unwrap() {
                    "bf16" => 2,
                    "u32" => 4,
                    dtype => panic!("{dtype}"),
                }) as usize;
            let bytes = std::fs::read(step.join(format!("{name}.r{rank}.bin"))).unwrap();
            assert_eq!(bytes.len(), rows as usize * row_bytes, "{name}");
            // `L<nn>.x.kv.<plane>[.codes|.scales]`
            let mut parts = name.split('.');
            let layer = parts.next().unwrap()[1..].parse().unwrap();
            assert_eq!((parts.next(), parts.next()), (Some("x"), Some("kv")));
            let plane = plane_name(parse_plane(parts.next().unwrap()));
            let part = parts.next();
            for (index, row) in bytes.chunks_exact(row_bytes).enumerate() {
                let position = positions[first as usize + index];
                let key = (event.clone(), layer, plane, position, rank);
                let map = match part {
                    None => &mut capture.values,
                    Some("codes") => &mut capture.codes,
                    Some("scales") => &mut capture.scales,
                    Some(other) => panic!("{other}"),
                };
                assert!(map.insert(key, row.to_vec()).is_none(), "{name}");
            }
            written += rows;
        }
        capture.steps.push((
            header["program"].as_str().unwrap().to_owned(),
            event.1.clone(),
            kv["axis"].as_str().unwrap().to_owned(),
            kv["ratio"].as_u64().unwrap(),
            written,
        ));
    }
    capture
}

/// The bitwise comparison of two captures' logical values: every key
/// either holds but not both, or whose bytes differ.
fn compare(a: &Capture, b: &Capture) -> Vec<Key> {
    let keys: std::collections::BTreeSet<&Key> = a.values.keys().chain(b.values.keys()).collect();
    keys.into_iter()
        .filter(|key| a.values.get(*key) != b.values.get(*key))
        .cloned()
        .collect()
}

/// Runs the lifecycle of `layout` with `corrupt` in a child process and
/// reads its capture.
fn capture(name: &str, layout: usize, corrupt: Option<&str>) -> Capture {
    let directory =
        std::env::temp_dir().join(format!("deepseek41-kv-taps-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let layout = layout.to_string();
    let layers = LAYERS.map(|layer| layer.to_string()).join(",");
    let mut env = vec![
        ("CHOPINFER_TAPS_SITES", "x.kv.*"),
        ("CHOPINFER_TAPS_LAYERS", layers.as_str()),
        (LAYOUT, layout.as_str()),
    ];
    if let Some(corrupt) = corrupt {
        env.push((CORRUPT, corrupt));
    }
    in_child(
        "fake_host_tests::kv_state::kv_state_taps_read_the_stored_rows_and_find_a_corrupted_byte",
        &directory,
        &env,
    );
    let capture = read(&directory);
    std::fs::remove_dir_all(&directory).unwrap();
    capture
}

/// The rows of `capture` that differ from `other` lie at exactly `row`, in
/// every readback that reads its position.
fn assert_found(differences: &[Key], row: Row, readbacks: usize) {
    let (layer, plane, position, rank) = row;
    assert_eq!(differences.len(), readbacks, "{differences:?}");
    for (_, l, p, at, r) in differences {
        assert_eq!(
            (*l, *p, *at, *r),
            (layer, plane_name(plane), position, rank),
            "{differences:?}"
        );
    }
}

/// The KV-state taps of both layouts read exactly what the fake device
/// stores, on every rank, through prefill, decode and a prefix restore;
/// the native codes and scales decode to the BF16 layout's values; the two
/// layouts' captures are bitwise equal; and one corrupted stored byte of
/// either layout makes the comparison fail at exactly its layer, plane,
/// position and rank, in every readback that reads it.
#[test]
fn kv_state_taps_read_the_stored_rows_and_find_a_corrupted_byte() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    if std::env::var_os(TAPS_CHILD).is_some() {
        let layout = KV_LAYOUTS[std::env::var(LAYOUT).unwrap().parse::<usize>().unwrap()];
        let corrupt = std::env::var(CORRUPT).ok().map(|name| corruption(&name));
        lifecycle(shape, layout, corrupt);
        return;
    }
    let native = capture("native", 0, None);
    let bf16 = capture("bf16", 1, None);
    // Request 7's prefill (end 9) and decode (end 10), request 9's restore
    // (end 10) and prefill (end 15): a window step and one step per page
    // group each; the cancelled decode left none.
    let mut expected_steps = Vec::new();
    for (phase, end) in [
        ("prefill", 9),
        ("decode", 10),
        ("restore", 10),
        ("prefill", 15),
    ] {
        for (axis, ratio) in [("window", 1), ("compressed", 2), ("compressed", 1)] {
            let rows: u64 = if axis == "window" {
                LAYERS.len() as u64 * end
            } else {
                // Two planes of each tapped source of the ratio.
                let sources = state::sources_of(&FLASH, ratio as u32)
                    .filter(|layer| LAYERS.contains(layer))
                    .count() as u64;
                2 * sources * (end / ratio)
            };
            expected_steps.push((phase.to_owned(), axis.to_owned(), ratio, rows));
        }
    }
    let ranks = u64::from(shape.ranks());
    for (capture, parts) in [(&native, 3), (&bf16, 1)] {
        let steps: Vec<_> = capture
            .steps
            .iter()
            .map(|(program, phase, axis, ratio, written)| {
                assert_eq!(program, kv::PROGRAM);
                (phase.clone(), axis.clone(), *ratio, *written)
            })
            .collect();
        let expected: Vec<_> = expected_steps
            .iter()
            .map(|(phase, axis, ratio, rows)| {
                (phase.clone(), axis.clone(), *ratio, rows * ranks * parts)
            })
            .collect();
        assert_eq!(steps, expected);
    }
    // Each layout read the stored rows, its raw bytes the native ones.
    let w = Widths::of(&FLASH);
    let stored: BTreeMap<_, _> = rows(shape.ranks())
        .into_iter()
        .map(|row| ((row.0, plane_name(row.1), row.2, row.3), encode(row, w)))
        .collect();
    for (capture, storage) in [(&native, KvStorage::Native), (&bf16, KvStorage::Bf16)] {
        for (key, values) in &capture.values {
            let (_, layer, plane, position, rank) = key;
            let row = (*layer, parse_plane(plane), *position, *rank);
            let encoded = &stored[&(*layer, *plane, *position, *rank)];
            assert_eq!(values, &encoded.values, "{key:?}");
            if storage == KvStorage::Native {
                assert_eq!(capture.codes[key], encoded.codes, "{key:?}");
                assert_eq!(capture.scales[key], encoded.scales, "{key:?}");
                let stored = kv::Stored {
                    codes: capture.codes[key].clone(),
                    scales: capture.scales[key].clone(),
                };
                assert_eq!(&kv::decode(row.1, w, &stored).unwrap(), values);
            }
        }
        if storage == KvStorage::Bf16 {
            assert!(capture.codes.is_empty() && capture.scales.is_empty());
        }
    }
    assert!(!native.values.is_empty());
    assert_eq!(compare(&native, &bf16), Vec::<Key>::new());
    // A corrupted byte of either layout differs at its row in every
    // readback holding its position: the index key at compressed position
    // 2 and the window at position 3 in all four, the ratio-1 compressed
    // latent at position 6 in all four.
    for (name, layout, clean) in [
        ("native-index-code", 0, &bf16),
        ("native-compressed-scale", 0, &bf16),
        ("bf16-window-value", 1, &native),
    ] {
        let corrupted = capture(name, layout, Some(name));
        let differences = compare(&corrupted, clean);
        assert_found(&differences, corruption(name).row, 4);
    }
}
