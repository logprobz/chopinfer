//! KV-state taps on the CPU fake host: the taps of each KV layout read the
//! bytes the fake device stores at the item's own state slot, page table
//! and ring cells, and the bitwise comparison of two layouts' captures by
//! layer, plane, position and rank (the comparison
//! `tools/qual/kv_layout_equality.py taps` performs) finds one corrupted
//! stored byte of either layout at exactly its layer, plane and position.
//!
//! The fake device's kernels write nothing, so before each step the test
//! stores the rows the step writes: fake-quantized rows of the trained
//! formats (native codes and scales, or their BF16 products), distinct per
//! position and rank, in the step's slot and pages, for the tapped layers
//! (the first and last, a window-only layer, a ratio-2 KV source and the
//! ratio-1 source). Every other slot's rings and every other page hold
//! decoy rows that decode exactly, so a readback from the wrong slot, page
//! or cell returns other bytes.
//!
//! Request 7 prefills 150 tokens in slot 1 in two chunks on pages 11, 4
//! and 9: its rows cross two page boundaries, and positions 136 on wrap
//! the 136-cell rings over the first chunk's rows. It decodes one token,
//! and its state is captured to snapshot pages and restored into slot 3 as
//! request 9. As a prefix hit's page table does, request 9's shares the
//! two full pages and holds a copy of the last in page 13. Slot 1 and page
//! 9 then hold decoys, and request 9 prefills 5 more tokens.
use super::*;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use bundle_runtime::fake_host::dump::Origin;
use kernel_core::quantize::Scales;
use model_deepseek41_contract::state::{self, Widths};

use crate::taps::kv::{self, Axis, Plane};

/// Request 7's prompt chunks: two whole pages, then the rest.
const CHUNKS: [u64; 2] = [128, 22];
const PROMPT: u64 = CHUNKS[0] + CHUNKS[1];
/// Request 7's committed tokens when its state is captured.
const CAPTURED: u64 = PROMPT + 1;
const CONTINUATION: u64 = 5;
const ITEMS: u32 = 4;
/// Request 7's state slot and page table.
const SLOT_7: u64 = 1;
const PAGES_7: [u64; 3] = [11, 4, 9];
/// Request 9's: the two full pages shared, the last copied.
const SLOT_9: u64 = 3;
const PAGES_9: [u64; 3] = [11, 4, 13];
/// Pages that start with decoy rows.
const DECOY_PAGES: std::ops::Range<u64> = 0..16;
/// The tapped layers (`CHOPINFER_TAPS_LAYERS`).
const LAYERS: [u32; 5] = [0, 5, 8, 20, 39];
/// The readbacks: (request, phase, the state's end). The cancelled decode
/// leaves none.
const EVENTS: [(u64, &str, u64); 5] = [
    (7, "prefill", CHUNKS[0]),
    (7, "prefill", PROMPT),
    (7, "decode", CAPTURED),
    (9, "restore", CAPTURED),
    (9, "prefill", CAPTURED + CONTINUATION),
];
/// The test's child process reads its layout and corruption here.
const LAYOUT: &str = "DEEPSEEK41_KV_TAPS_LAYOUT";
const CORRUPT: &str = "DEEPSEEK41_KV_TAPS_CORRUPT";
/// Set in a child that runs its lifecycle with the snapshot copy kernel off.
const COPIES: &str = "DEEPSEEK41_KV_TAPS_DEVICE_COPIES";

/// One stored row: a layer's plane at a position (or a cell or page entry,
/// for a decoy) on a rank.
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

/// The corruptions the test applies, each with the readbacks that hold its
/// row: a native index key's code byte on request 7's second page, in all
/// five; a native compressed latent's E4M3 scale byte on the last page,
/// which request 9's copy keeps, in the four from the second chunk on; a
/// BF16 window value at a position that wrapped the ring, in the same
/// four; and a native window scale byte at a position the second chunk
/// overwrites in its cell, in the first readback alone.
fn corruption(name: &str) -> (Corruption, usize) {
    match name {
        "native-index-code" => (
            Corruption {
                row: (8, Plane::Index, 40, 1),
                scale: false,
                byte: 3,
                bits: 0x10,
            },
            5,
        ),
        "native-compressed-scale" => (
            Corruption {
                row: (20, Plane::Compressed, 140, 3),
                scale: true,
                byte: 5,
                bits: 0x01,
            },
            4,
        ),
        "bf16-window-value" => (
            Corruption {
                row: (5, Plane::Window, 145, 2),
                scale: false,
                byte: 64,
                bits: 0x01,
            },
            4,
        ),
        "native-window-scale" => (
            Corruption {
                row: (39, Plane::Window, 5, 0),
                scale: true,
                byte: 7,
                bits: 0x01,
            },
            1,
        ),
        _ => panic!("no corruption {name}"),
    }
}

/// The corruptions the test applies, each to its layout (an index of
/// `KV_LAYOUTS`: 0 native, 1 BF16).
const CORRUPTIONS: [(&str, usize); 4] = [
    ("native-index-code", 0),
    ("native-compressed-scale", 0),
    ("bf16-window-value", 1),
    ("native-window-scale", 0),
];

/// A pseudo-random stream from `seed`.
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// A row's seed: its layer, plane, position and rank, and `tag`, which is
/// 0 for a request's rows and distinct for each decoy fill.
fn seed(tag: u64, (layer, plane, position, rank): Row) -> u64 {
    (tag << 48)
        ^ (u64::from(layer) << 40)
        ^ ((plane as u64) << 36)
        ^ (position << 8)
        ^ u64::from(rank)
}

/// A request's row's F32 inputs: pseudo-random values whose blocks span a
/// range of magnitudes, distinct per position and rank.
fn inputs(row: Row, width: usize) -> Vec<f32> {
    let mut next = stream(seed(0, row));
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

/// A request's row as each layout stores it: native codes and scales, and
/// the BF16 products, the logical values, little-endian.
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

/// A decoy row in `storage`, its codes (or BF16 values) and scales:
/// pseudo-random bytes that decode exactly. E4M3 codes keep their
/// exponent's top bit clear, so none is a NaN; UE8M0 scales lie in
/// 2^-4 .. 2^3 and E4M3 scales in 0.5 .. 1.875; BF16 values keep their
/// exponent's top bit clear, so none is infinite or a NaN.
fn decoy(tag: u64, row: Row, w: Widths, storage: KvStorage) -> (Vec<u8>, Vec<u8>) {
    let plane = row.1;
    let (code_bytes, scale_bytes) = plane.row_bytes(w, storage);
    let mut next = stream(seed(tag, row));
    let mut bytes = |count: u64, byte: fn(u64, u8) -> u8| -> Vec<u8> {
        (0..count).map(|at| byte(at, next() as u8)).collect()
    };
    let ue8m0 = |_, r: u8| 123 + r % 8;
    match (storage, plane) {
        // Little-endian: each value's high byte holds the exponent's top bit.
        (KvStorage::Bf16, _) => (
            bytes(code_bytes, |at, r| if at % 2 == 1 { r & 0xbf } else { r }),
            Vec::new(),
        ),
        (KvStorage::Native, Plane::Window) => (
            bytes(code_bytes, |_, r| r & 0xbf),
            bytes(scale_bytes, ue8m0),
        ),
        (KvStorage::Native, Plane::Index) => {
            (bytes(code_bytes, |_, r| r), bytes(scale_bytes, ue8m0))
        }
        (KvStorage::Native, Plane::Compressed) => (
            bytes(code_bytes, |_, r| r),
            bytes(scale_bytes, |_, r| 0x30 + r % 16),
        ),
    }
}

/// Every row of the requests: each tapped layer's window at every token
/// position, and each tapped KV source's compressed latents and index keys
/// at every compressed position, on every rank.
fn rows(ranks: u32) -> Vec<Row> {
    let g = &FLASH;
    let end = CAPTURED + CONTINUATION;
    let mut rows = Vec::new();
    for rank in 0..ranks {
        for layer in LAYERS {
            for position in 0..end {
                rows.push((layer, Plane::Window, position, rank));
            }
        }
        for layer in sources() {
            let ratio = u64::from(g.compress_ratio(layer));
            for plane in [Plane::Compressed, Plane::Index] {
                for position in 0..end / ratio {
                    rows.push((layer, plane, position, rank));
                }
            }
        }
    }
    rows
}

/// The tapped KV sources.
fn sources() -> impl Iterator<Item = u32> {
    FLASH
        .shape
        .csa
        .kv_sources
        .iter()
        .copied()
        .filter(|layer| LAYERS.contains(layer))
}

/// The row axis of `layer`'s `plane`.
fn axis(layer: u32, plane: Plane) -> Axis {
    match plane {
        Plane::Window => Axis::Window,
        Plane::Compressed | Plane::Index => Axis::Sources {
            ratio: FLASH.compress_ratio(layer),
        },
    }
}

/// Where a row's codes (or BF16 values) and scales lie.
type Spans = (DeviceBufferSpanV1, Option<DeviceBufferSpanV1>);

/// The fake device's KV state, addressed from the state declaration alone,
/// and the rows the test stores in it.
struct Device<'a> {
    host: &'a FakeHost,
    storage: KvStorage,
    corrupt: Option<Corruption>,
    /// Each rank's `window.ring` state buffer.
    rings: Vec<DeviceBufferSpanV1>,
    /// Bytes of one layer's rings, and its ring entries (state slots).
    layer_bytes: u64,
    entries: u64,
    page_bytes: u64,
    page_tokens: u64,
}

impl<'a> Device<'a> {
    fn new(
        host: &'a FakeHost,
        instance: &BundleInstanceV1,
        storage: KvStorage,
        corrupt: Option<Corruption>,
    ) -> Self {
        let (_, states) = crate::instance::anchor_spans(instance.handle());
        let rings: Vec<_> = states
            .iter()
            .map(|state| state.named("window.ring").unwrap())
            .collect();
        let declaration = state::request_state(&FLASH, storage, false);
        let buffer = declaration
            .buffers
            .iter()
            .find(|buffer| buffer.name == "window.ring")
            .unwrap();
        let layer_bytes = rings[0].length / buffer.layers;
        let entries = layer_bytes / buffer.bytes_per_layer_rank();
        assert_eq!(entries * buffer.bytes_per_layer_rank(), layer_bytes);
        assert!(entries > SLOT_9);
        Self {
            host,
            storage,
            corrupt,
            rings,
            layer_bytes,
            entries,
            page_bytes: host.page_bytes(),
            page_tokens: declaration.page_tokens,
        }
    }

    fn ranks(&self) -> u32 {
        self.host.shape.ranks()
    }

    /// Layer `layer`'s ring cell `cell` of `slot` on `rank`: a layer's
    /// rings hold every slot's cells' codes (or BF16 values), then every
    /// slot's cells' scales.
    fn cell(&self, layer: u32, slot: u64, cell: u64, rank: u32) -> Spans {
        let w = Widths::of(&FLASH);
        let (code_bytes, scale_bytes) = Plane::Window.row_bytes(w, self.storage);
        let cells = state::ring_cells(w);
        assert!(slot < self.entries && cell < cells);
        let ring = self.rings[rank as usize];
        let base = u64::from(layer) * self.layer_bytes;
        let at = slot * cells + cell;
        let codes = within(ring, base + at * code_bytes, code_bytes);
        let scales = (scale_bytes > 0).then(|| {
            within(
                ring,
                base + self.entries * cells * code_bytes + at * scale_bytes,
                scale_bytes,
            )
        });
        (codes, scales)
    }

    /// Each rank's KV pool.
    fn pool(&self, rank: u32) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank),
            0,
            KV_PAGES * self.page_bytes,
        )
    }

    /// KV source `layer`'s entry `entry` of `plane` in page `page` on
    /// `rank`: the plane's stream in a page holds its entries' codes (or
    /// BF16 values), then their scales.
    fn entry(&self, layer: u32, plane: Plane, page: u64, entry: u64, rank: u32) -> Spans {
        let g = &FLASH;
        let (code_bytes, scale_bytes) = plane.row_bytes(Widths::of(g), self.storage);
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
        let declaration = state::request_state(g, self.storage, false);
        let layout = declaration
            .page_planes()
            .into_iter()
            .find(|p| p.layers == group && p.layer == local && p.stream == stream)
            .unwrap();
        let per_page = declaration.page_tokens / u64::from(ratio);
        assert!(entry < per_page);
        let pool = self.pool(rank);
        let base = page * self.page_bytes + layout.offset;
        let codes = within(pool, base + entry * code_bytes, code_bytes);
        let scales = (scale_bytes > 0).then(|| {
            within(
                pool,
                base + per_page * code_bytes + entry * scale_bytes,
                scale_bytes,
            )
        });
        (codes, scales)
    }

    /// Stores `codes` and `scales` at `spans`.
    fn put(&self, (code_span, scale_span): Spans, (codes, scales): (Vec<u8>, Vec<u8>)) {
        let mut memory = self.host.memory.borrow_mut();
        let mut write = |span: DeviceBufferSpanV1, bytes: Vec<u8>| {
            assert_eq!(span.length as usize, bytes.len());
            memory.write(span, vec![(span.length, Origin::Uploaded(bytes.into(), 0))]);
        };
        write(code_span, codes);
        if let Some(span) = scale_span {
            write(span, scales);
        }
    }

    /// A request's `row` as this layout stores it, corrupted if it is the
    /// corrupted row.
    fn row(&self, row: Row) -> (Vec<u8>, Vec<u8>) {
        let encoded = encode(row, Widths::of(&FLASH));
        let (mut codes, mut scales) = match self.storage {
            KvStorage::Native => (encoded.codes, encoded.scales),
            KvStorage::Bf16 => (encoded.values, Vec::new()),
        };
        if let Some(c) = self.corrupt.filter(|c| c.row == row) {
            let bytes = if c.scale { &mut scales } else { &mut codes };
            bytes[c.byte] ^= c.bits;
        }
        (codes, scales)
    }

    /// Fills every tapped layer's ring cells of `slot` with decoys.
    fn decoy_slot(&self, slot: u64, tag: u64) {
        let w = Widths::of(&FLASH);
        for rank in 0..self.ranks() {
            for layer in LAYERS {
                for cell in 0..state::ring_cells(w) {
                    let row = (layer, Plane::Window, cell, rank);
                    self.put(
                        self.cell(layer, slot, cell, rank),
                        decoy(tag, row, w, self.storage),
                    );
                }
            }
        }
    }

    /// Fills every tapped KV source's entries in `page` with decoys.
    fn decoy_page(&self, page: u64, tag: u64) {
        let w = Widths::of(&FLASH);
        for rank in 0..self.ranks() {
            for layer in sources() {
                let per_page = self.page_tokens / u64::from(FLASH.compress_ratio(layer));
                for plane in [Plane::Compressed, Plane::Index] {
                    for entry in 0..per_page {
                        let row = (layer, plane, entry, rank);
                        self.put(
                            self.entry(layer, plane, page, entry, rank),
                            decoy(tag, row, w, self.storage),
                        );
                    }
                }
            }
        }
    }

    /// Stores what a step of `tokens` in `slot` on `pages` writes: each
    /// token's window row in its ring cell, and each compressed position
    /// the step completes in its page's entry.
    fn step(&self, slot: u64, pages: &[u64], tokens: std::ops::Range<u64>) {
        let cells = state::ring_cells(Widths::of(&FLASH));
        for rank in 0..self.ranks() {
            for layer in LAYERS {
                for position in tokens.clone() {
                    let row = (layer, Plane::Window, position, rank);
                    self.put(
                        self.cell(layer, slot, position % cells, rank),
                        self.row(row),
                    );
                }
            }
            for layer in sources() {
                let ratio = u64::from(FLASH.compress_ratio(layer));
                let per_page = self.page_tokens / ratio;
                for position in tokens.start / ratio..tokens.end / ratio {
                    let page = pages[(position / per_page) as usize];
                    for plane in [Plane::Compressed, Plane::Index] {
                        let row = (layer, plane, position, rank);
                        self.put(
                            self.entry(layer, plane, page, position % per_page, rank),
                            self.row(row),
                        );
                    }
                }
            }
        }
    }

    /// Copies page `from` to page `to` on every rank.
    fn copy_page(&self, from: u64, to: u64) {
        for rank in 0..self.ranks() {
            let pool = self.pool(rank);
            let page = |page: u64| within(pool, page * self.page_bytes, self.page_bytes);
            self.host.memory.borrow_mut().copy(page(from), page(to));
        }
    }
}

/// Bytes `[offset, offset + length)` of `span`.
fn within(span: DeviceBufferSpanV1, offset: u64, length: u64) -> DeviceBufferSpanV1 {
    assert!(offset + length <= span.length);
    DeviceBufferSpanV1::new(span.buffer, span.offset + offset, length)
}

/// A prompt chunk of `count` tokens after `previous` of `request` in
/// `slot` on `pages`, scoring its last row when `last`.
fn chunk(request: u64, slot: u64, previous: u64, count: u64, last: bool, pages: &[u64]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + count;
    let inputs = (previous..end).map(|token| token as u32).collect();
    let pages = pages[..end.div_ceil(64) as usize].to_vec();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, false, count as u32, last, inputs)],
        vec![BatchAppendV1::new(previous, count, end, pages, None)],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

/// One decode step of `request` in `slot` on `pages` after `previous`
/// tokens.
fn decode_one(request: u64, slot: u64, previous: u64, pages: &[u64]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + 1;
    let pages = pages[..end.div_ceil(64) as usize].to_vec();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, true, 1, true, vec![3])],
        vec![BatchAppendV1::new(previous, 1, end, pages, None)],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

/// The child: one layout's lifecycle with the fake device's stored rows.
fn lifecycle(shape: Shape, layout: KvLayout, corrupt: Option<Corruption>) {
    let host = FakeHost::new(shape, layout);
    let mut instance = create(&host, ITEMS);
    let device = Device::new(&host, &instance, layout.storage, corrupt);
    for slot in 0..device.entries {
        device.decoy_slot(slot, 1 + slot);
    }
    for page in DECOY_PAGES {
        device.decoy_page(page, 1000 + page);
    }
    host.read_memory.set(true);
    let mut views = Views::default();
    let mut previous = 0;
    for count in CHUNKS {
        device.step(SLOT_7, &PAGES_7, previous..previous + count);
        let last = previous + count == PROMPT;
        let batch = chunk(7, SLOT_7, previous, count, last, &PAGES_7);
        step(&mut instance, &batch, &mut views);
        previous += count;
    }
    device.step(SLOT_7, &PAGES_7, PROMPT..CAPTURED);
    step(
        &mut instance,
        &decode_one(7, SLOT_7, PROMPT, &PAGES_7),
        &mut views,
    );
    let count = instance.describe_prefix_state().unwrap();
    let snapshot: Vec<u64> = (100..100 + count).collect();
    assert!(snapshot[0] >= DECOY_PAGES.end);
    instance
        .capture_prefix_state(7, CAPTURED, &snapshot)
        .unwrap();
    instance.retire_request(7).unwrap();
    let ids: Vec<u32> = (0..CAPTURED as u32).collect();
    instance
        .restore_prefix_state_in_slot(9, &ids, &snapshot, SLOT_9)
        .unwrap();
    // Request 9 shares request 7's full pages and holds a copy of its last;
    // request 7's slot and last page then hold other rows.
    device.copy_page(PAGES_7[2], PAGES_9[2]);
    device.decoy_slot(SLOT_7, 100);
    device.decoy_page(PAGES_7[2], 2000);
    let end = CAPTURED + CONTINUATION;
    device.step(SLOT_9, &PAGES_9, CAPTURED..end);
    let batch = chunk(9, SLOT_9, CAPTURED, CONTINUATION, true, &PAGES_9);
    step(&mut instance, &batch, &mut views);
    // A cancelled step's KV steps are removed.
    instance
        .execute(&decode_one(9, SLOT_9, end, &PAGES_9), &mut views)
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

/// The keys each readback should hold: every tapped layer's plane at each
/// position of its axis's rows for the readback's end, on every rank.
fn expected_keys(ranks: u32) -> BTreeSet<Key> {
    let mut planes: Vec<(u32, Plane)> = LAYERS.map(|layer| (layer, Plane::Window)).to_vec();
    for layer in sources() {
        planes.extend([(layer, Plane::Compressed), (layer, Plane::Index)]);
    }
    let mut keys = BTreeSet::new();
    for (request, phase, end) in EVENTS {
        for &(layer, plane) in &planes {
            for position in axis(layer, plane).rows(&FLASH, end) {
                for rank in 0..ranks {
                    let event = (request, phase.to_owned(), end);
                    keys.insert((event, layer, plane_name(plane), position, rank));
                }
            }
        }
    }
    keys
}

/// Fails unless `found` is exactly `keys`, naming the first keys missing
/// and unexpected.
fn assert_keys<'a>(found: impl Iterator<Item = &'a Key>, keys: &BTreeSet<Key>) {
    let found: BTreeSet<Key> = found.cloned().collect();
    let missing: Vec<_> = keys.difference(&found).take(3).collect();
    let unexpected: Vec<_> = found.difference(keys).take(3).collect();
    assert!(
        missing.is_empty() && unexpected.is_empty(),
        "missing {missing:?}, unexpected {unexpected:?}"
    );
}

/// The bitwise comparison of two captures' logical values: every key
/// either holds but not both, or whose bytes differ.
fn compare(a: &Capture, b: &Capture) -> BTreeSet<Key> {
    let keys: BTreeSet<&Key> = a.values.keys().chain(b.values.keys()).collect();
    keys.into_iter()
        .filter(|key| a.values.get(*key) != b.values.get(*key))
        .cloned()
        .collect()
}

/// Runs the lifecycle of `layout` with `corrupt` in a child process and
/// reads its capture. The directory names the snapshot copy mode, so the
/// tests with the kernel on and off may run at once in one process.
fn capture(name: &str, layout: usize, corrupt: Option<&str>, copies: bool) -> Capture {
    let mode = if copies { "copies" } else { "snapshot-kernel" };
    let directory = std::env::temp_dir().join(format!(
        "deepseek41-kv-taps-{name}-{mode}-{}",
        std::process::id()
    ));
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
    if copies {
        env.push((COPIES, "1"));
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

/// The KV-state taps of both layouts read exactly what the fake device
/// stores in each item's own slot, pages and ring cells, on every rank,
/// through two prefill chunks that wrap the rings, a decode, a prefix
/// restore into another slot and a prefill there; the native codes and
/// scales decode to the BF16 layout's values; the two layouts' captures
/// are bitwise equal; and one corrupted stored byte of either layout makes
/// the comparison fail at exactly its layer, plane, position and rank, in
/// exactly the readbacks that hold it.
#[test]
fn kv_state_taps_read_the_stored_rows_and_find_a_corrupted_byte() {
    if std::env::var_os(TAPS_CHILD).is_some() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let layout = KV_LAYOUTS[std::env::var(LAYOUT).unwrap().parse::<usize>().unwrap()];
        let corrupt = std::env::var(CORRUPT).ok().map(|name| corruption(&name).0);
        if std::env::var_os(COPIES).is_some() {
            super::snapshot_kernel::with_copies(|| lifecycle(shape, layout, corrupt));
        } else {
            lifecycle(shape, layout, corrupt);
        }
        return;
    }
    taps(false);
}

/// The same lifecycles with the `prefix_snapshot_copy_kernel` candidate
/// off: the restore's snapshot moves in device copies, and the taps read
/// the same stored rows and find the same corrupted bytes.
#[test]
fn kv_state_taps_read_the_stored_rows_with_device_copies() {
    taps(true);
}

/// Runs each layout's clean lifecycle and each corrupted one in a child,
/// with the snapshot copy kernel or with device `copies`, and checks their
/// captures.
fn taps(copies: bool) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    // Each layout's clean lifecycle and each corrupted one, in child
    // processes at once.
    let mut captures = std::thread::scope(|scope| {
        let clean = [("native", 0, None), ("bf16", 1, None)];
        let corrupted = CORRUPTIONS.map(|(name, layout)| (name, layout, Some(name)));
        let runs: Vec<_> = clean
            .into_iter()
            .chain(corrupted)
            .map(|(name, layout, corrupt)| {
                scope.spawn(move || capture(name, layout, corrupt, copies))
            })
            .collect();
        runs.into_iter()
            .map(|run| run.join().expect("a capture"))
            .collect::<Vec<_>>()
    })
    .into_iter();
    let (native, bf16) = (captures.next().unwrap(), captures.next().unwrap());
    // Each readback is a window step and one step per page group.
    let mut expected_steps = Vec::new();
    for (_, phase, end) in EVENTS {
        for (axis, ratio) in [("window", 1), ("compressed", 2), ("compressed", 1)] {
            let rows: u64 = if axis == "window" {
                LAYERS.len() as u64 * Axis::Window.rows(&FLASH, end).count() as u64
            } else {
                // Two planes of each tapped source of the ratio.
                let sources = sources()
                    .filter(|&layer| FLASH.compress_ratio(layer) == ratio as u32)
                    .count() as u64;
                2 * sources * (end / ratio)
            };
            expected_steps.push((phase.to_owned(), axis.to_owned(), ratio, rows));
        }
    }
    let ranks = u64::from(shape.ranks());
    let keys = expected_keys(shape.ranks());
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
        assert_keys(capture.values.keys(), &keys);
    }
    assert_keys(native.codes.keys(), &keys);
    assert_keys(native.scales.keys(), &keys);
    // Each layout read the stored rows, its raw bytes the native ones.
    let w = Widths::of(&FLASH);
    let stored: BTreeMap<_, _> = rows(shape.ranks())
        .into_iter()
        .map(|row| ((row.0, plane_name(row.1), row.2, row.3), encode(row, w)))
        .collect();
    for (capture, storage) in [(&native, KvStorage::Native), (&bf16, KvStorage::Bf16)] {
        let mut wrong = Vec::new();
        for (key, values) in &capture.values {
            let (_, layer, plane, position, rank) = key;
            let encoded = &stored[&(*layer, *plane, *position, *rank)];
            let mut right = values == &encoded.values;
            if storage == KvStorage::Native {
                let stored = kv::Stored {
                    codes: capture.codes[key].clone(),
                    scales: capture.scales[key].clone(),
                };
                right &= stored.codes == encoded.codes
                    && stored.scales == encoded.scales
                    && kv::decode(parse_plane(plane), w, &stored).as_ref() == Ok(values);
            }
            if !right {
                wrong.push(key);
            }
        }
        assert!(
            wrong.is_empty(),
            "{storage:?}: {} of {} rows are not the stored rows, the first {:?}",
            wrong.len(),
            capture.values.len(),
            &wrong[..wrong.len().min(3)]
        );
        if storage == KvStorage::Bf16 {
            assert!(capture.codes.is_empty() && capture.scales.is_empty());
        }
    }
    assert_eq!(compare(&native, &bf16), BTreeSet::new());
    // A corrupted byte of either layout differs at its row in exactly the
    // readbacks that hold its position.
    for ((name, layout), corrupted) in CORRUPTIONS.into_iter().zip(captures) {
        let clean = if layout == 0 { &bf16 } else { &native };
        let (corrupt, readbacks) = corruption(name);
        let (layer, plane, position, rank) = corrupt.row;
        let expected: BTreeSet<Key> = keys
            .iter()
            .filter(|(_, l, p, at, r)| {
                (*l, *p, *at, *r) == (layer, plane_name(plane), position, rank)
            })
            .cloned()
            .collect();
        assert_eq!(expected.len(), readbacks, "{name}");
        assert_eq!(compare(&corrupted, clean), expected, "{name}");
    }
}
