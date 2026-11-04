//! KV-state taps: a request's stored KV, read back on every rank and
//! written by layer and position as the logical values the attention reads
//! (`models/deepseek41/taps.json`, "KV-state taps").
//!
//! Three planes: each layer's window ring (`L<nn>.x.kv.window`, the last
//! `sliding_window` positions), and each KV source's compressed latents
//! (`L<nn>.x.kv.compressed`) and index keys (`L<nn>.x.kv.index`) in the serving
//! pages, one row per compressed position. The BF16 layout's values are
//! its stored bytes. The native layout's codes and scales decode to their
//! BF16 products with the trained quantizers' shared decoder
//! (`kernel_core::quantize::Quantized` of each plane's
//! `BlockQuantization`), and the native layout also writes the raw bytes
//! (`.codes`, `.scales`), so a check can confirm that they decode to
//! exactly the BF16 layout's values.
//!
//! A readback is one tap step per row axis (`Axis`): the window, then the
//! KV sources of each compress ratio in the pages' declaration order. Its
//! `items.json` names the request and, under `kv`, the phase, axis, ratio,
//! storage, the state's end and each row's position.

use std::ops::Range;

use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::state::{Extent, PagedLayers};
use kernel_core::codec::{bf16_to_f32, f64_to_bf16};
use kernel_core::precision::quantizer::BlockQuantization;
use kernel_core::quantize::{Quantized, ScaleFormat, Scales};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::state::{self, KvStorage, Widths};
use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1};
use runtime_bundle_sdk::taps::{TapCompare, TapDType, TapError, TapSharding, TapSpec, TapStep};

/// The request state buffer that holds the window rings.
const RING_BUFFER: &str = "window.ring";

/// The program each KV step's `capture.json` records.
pub const PROGRAM: &str = "KvState";

/// A plane of a request's KV state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Plane {
    /// A layer's window ring: E4M3 codes with a UE8M0 scale per 32.
    Window,
    /// A KV source's compressed latents: E2M1 codes with an E4M3 scale
    /// per 16.
    Compressed,
    /// A KV source's index keys: E2M1 codes with a UE8M0 scale per 32.
    Index,
}

impl Plane {
    /// The site of the plane's logical values.
    pub const fn site(self) -> &'static str {
        match self {
            Self::Window => "x.kv.window",
            Self::Compressed => "x.kv.compressed",
            Self::Index => "x.kv.index",
        }
    }

    /// The trained quantizer whose codes and scales the native layout
    /// stores.
    pub const fn format(self) -> BlockQuantization {
        match self {
            Self::Window => BlockQuantization::E4M3_CEIL32,
            Self::Compressed => BlockQuantization::E2M1E4M3Scale16,
            Self::Index => BlockQuantization::E2M1Ceil32,
        }
    }

    /// Values in one row.
    pub const fn width(self, w: Widths) -> u64 {
        match self {
            Self::Window | Self::Compressed => w.latent,
            Self::Index => w.index,
        }
    }

    /// The page stream that holds a source's plane
    /// (`state::request_state`).
    const fn stream(self) -> &'static str {
        match self {
            Self::Window => RING_BUFFER,
            Self::Compressed => "latent",
            Self::Index => "index-key",
        }
    }

    /// Bytes of one row's codes (or BF16 values) and of its scales in
    /// `storage`.
    pub fn row_bytes(self, w: Widths, storage: KvStorage) -> (u64, u64) {
        let width = self.width(w);
        match storage {
            KvStorage::Bf16 => (2 * width, 0),
            KvStorage::Native => {
                let format = self.quantizer();
                (
                    format.code_bytes(width as usize) as u64,
                    format.scale_cols(width as usize) as u64,
                )
            }
        }
    }

    fn quantizer(self) -> kernel_core::quantize::QuantFormat {
        self.format()
            .quantizer()
            .expect("the trained KV formats validate")
            .format
    }
}

/// What a tap holds of its plane's rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Part {
    /// The native layout's stored codes.
    Codes,
    /// The native layout's stored scales.
    Scales,
    /// The logical BF16 values.
    Values,
}

/// The rows of a KV step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis {
    /// Token positions: each layer's window.
    Window,
    /// Compressed positions: the KV sources of compress ratio `ratio`.
    Sources { ratio: u32 },
}

impl Axis {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Window => "window",
            Self::Sources { .. } => "compressed",
        }
    }

    /// Tokens per row.
    pub const fn ratio(self) -> u32 {
        match self {
            Self::Window => 1,
            Self::Sources { ratio } => ratio,
        }
    }

    /// The rows of a request whose state ends before token `end`: the
    /// window's positions, or every completed compressed position.
    pub fn rows(self, g: &Geometry, end: u64) -> Range<u64> {
        match self {
            Self::Window => end.saturating_sub(u64::from(g.shape.attention.window))..end,
            Self::Sources { ratio } => 0..end / u64::from(ratio),
        }
    }
}

/// The axes of `g`: the window, then the KV sources of each page group in
/// the declaration's order.
pub fn axes(g: &Geometry) -> Vec<Axis> {
    std::iter::once(Axis::Window)
        .chain(
            state::request_state(g, KvStorage::Native, false)
                .pages
                .iter()
                .map(|group| Axis::Sources {
                    ratio: ratio(group),
                }),
        )
        .collect()
}

/// The compress ratio of a page group's sources.
fn ratio(group: &PagedLayers) -> u32 {
    group
        .streams
        .first()
        .map_or(1, |stream| stream.tokens_per_entry) as u32
}

/// One KV-state tap.
#[derive(Clone, Debug)]
pub struct KvTap {
    pub spec: TapSpec,
    pub layer: u32,
    pub plane: Plane,
    pub part: Part,
}

fn spec(layer: u32, plane: Plane, part: Part, w: Widths, storage: KvStorage) -> TapSpec {
    let (codes, scales) = plane.row_bytes(w, storage);
    // Raw bytes are written as u32 words, the shared taps' narrowest
    // integer: four stored bytes each, in memory order.
    let (suffix, cols, dtype, compare) = match part {
        Part::Values => ("", plane.width(w), TapDType::Bf16, TapCompare::Dense),
        Part::Codes => (".codes", codes / 4, TapDType::U32, TapCompare::Codes),
        Part::Scales => (".scales", scales / 4, TapDType::U32, TapCompare::Codes),
    };
    TapSpec {
        name: format!("L{layer:02}.{}{suffix}", plane.site()),
        layer: Some(layer),
        detail: true,
        cols: cols as u32,
        dtype,
        sharding: TapSharding::RankCopies,
        compare,
    }
}

/// The taps of `axis` in `storage`, in the order a step writes them: by
/// layer, then plane, the native layout's codes and scales before its
/// values.
pub fn taps(g: &Geometry, storage: KvStorage, axis: Axis) -> Vec<KvTap> {
    let w = Widths::of(g);
    let (layers, planes): (Vec<u32>, &[Plane]) = match axis {
        Axis::Window => ((0..g.layers()).collect(), &[Plane::Window]),
        Axis::Sources { ratio } => (
            state::sources_of(g, ratio).collect(),
            &[Plane::Compressed, Plane::Index],
        ),
    };
    let parts: &[Part] = match storage {
        KvStorage::Native => &[Part::Codes, Part::Scales, Part::Values],
        KvStorage::Bf16 => &[Part::Values],
    };
    let mut taps = Vec::new();
    for layer in layers {
        for &plane in planes {
            for &part in parts {
                taps.push(KvTap {
                    spec: spec(layer, plane, part, w, storage),
                    layer,
                    plane,
                    part,
                });
            }
        }
    }
    taps
}

/// Every KV-state tap of `g` in `storage`.
pub fn specs(g: &Geometry, storage: KvStorage) -> Vec<TapSpec> {
    axes(g)
        .into_iter()
        .flat_map(|axis| taps(g, storage, axis))
        .map(|tap| tap.spec)
        .collect()
}

/// When a readback runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// After a prefill chunk's launches: the state through its last row.
    Prefill,
    /// After a decode step's launches: the state through its row.
    Decode,
    /// Before the first step of a request restored from a prefix hit: the
    /// restored state.
    Restore,
}

impl Phase {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Restore => "restore",
        }
    }
}

/// Where one rank holds the KV state: its stream, its `window.ring` state
/// buffer and its serving KV pool.
#[derive(Clone, Copy, Debug)]
pub struct RankState {
    pub stream: BundleStreamHandleV1,
    pub ring: DeviceBufferSpanV1,
    pub pool: DeviceBufferSpanV1,
}

/// The instance's KV layout.
#[derive(Clone, Copy, Debug)]
pub struct Layout<'a> {
    pub geometry: &'a Geometry,
    pub storage: KvStorage,
    /// What the state buffers are sized for.
    pub extent: Extent,
    pub page_bytes: u64,
    /// Pages in the pool.
    pub pages: u64,
}

/// One item's state to read: its request, state slot, page table and the
/// token its state ends before.
#[derive(Clone, Debug)]
pub struct Item<'a> {
    pub request: u64,
    pub slot: u32,
    pub pages: &'a [i32],
    pub end: u64,
}

/// The items of `inputs` a readback in `phase` reads, `restored` naming
/// the requests a restore readback still owes.
pub fn items<'a>(
    inputs: &'a PhaseInputs,
    phase: Phase,
    mut restored: impl FnMut(u64) -> bool,
) -> Result<Vec<Item<'a>>, String> {
    let mut items = Vec::new();
    for (index, &request) in inputs.request_ids.iter().enumerate() {
        let (start, rows) = (inputs.q_start[index], inputs.q_len[index]);
        if rows <= 0 || (phase == Phase::Restore && !restored(request)) {
            continue;
        }
        let row = |at: i32| {
            usize::try_from(at)
                .ok()
                .and_then(|at| inputs.positions.get(at))
                .and_then(|&position| u64::try_from(position).ok())
                .ok_or_else(|| format!("request {request} has no position at row {at}"))
        };
        let end = match phase {
            Phase::Restore => row(start)?,
            Phase::Prefill | Phase::Decode => row(start + rows - 1)? + 1,
        };
        let slot = inputs
            .slot_ids
            .get(index)
            .and_then(|&slot| u32::try_from(slot).ok())
            .ok_or_else(|| format!("request {request} has no state slot"))?;
        let pages = inputs
            .block_tables
            .get(index)
            .ok_or_else(|| format!("request {request} has no page table"))?;
        items.push(Item {
            request,
            slot,
            pages,
            end,
        });
    }
    Ok(items)
}

/// One plane's rows on one rank as stored: codes (or BF16 values) row
/// after row, then the native layout's scales.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Stored {
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
}

fn invalid(message: String) -> TapError {
    TapError::Io(std::io::Error::other(message))
}

/// `span`'s bytes, once the work queued on `stream` is done.
fn read(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    span: DeviceBufferSpanV1,
) -> Result<Vec<u8>, TapError> {
    services
        .synchronize_stream(stream)
        .map_err(TapError::Host)?;
    let mut bytes = vec![0; span.length as usize];
    services
        .copy_device_to_host(stream, &mut bytes, span)
        .map_err(TapError::Host)?;
    services
        .synchronize_stream(stream)
        .map_err(TapError::Host)?;
    Ok(bytes)
}

/// Bytes `[offset, offset + length)` of `span`, refused past its end.
fn part(
    span: DeviceBufferSpanV1,
    offset: u64,
    length: u64,
) -> Result<DeviceBufferSpanV1, TapError> {
    if offset
        .checked_add(length)
        .is_none_or(|end| end > span.length)
    {
        return Err(invalid("a KV-state read lies outside its buffer".into()));
    }
    Ok(DeviceBufferSpanV1::new(
        span.buffer,
        span.offset + offset,
        length,
    ))
}

/// Layer `layer`'s window rows `rows` (token positions) of `item` on
/// `rank`: each position's ring cell (`position % cells`) of the item's
/// slot, read as the lowering addresses the ring (`lower::ring`).
pub fn window(
    services: &HostServicesV1,
    rank: &RankState,
    layout: &Layout,
    layer: u32,
    item: &Item,
    rows: Range<u64>,
) -> Result<Stored, TapError> {
    let ring = crate::lower::ring(
        layout.geometry,
        layout.storage,
        layout.extent,
        rank.ring,
        layer,
    )
    .map_err(invalid)?;
    if item.slot >= ring.slots {
        return Err(invalid(format!("slot {} has no window ring", item.slot)));
    }
    let cells = u64::from(ring.cells);
    let slot = u64::from(item.slot);
    let entry = |plane: DeviceBufferSpanV1, stride: u64| -> Result<Vec<u8>, TapError> {
        read(services, rank.stream, part(plane, slot * stride, stride)?)
    };
    let codes = entry(ring.codes, ring.code_stride)?;
    let scales = match ring.scales {
        Some(scales) => entry(scales, ring.scale_stride)?,
        None => Vec::new(),
    };
    let (code_cell, scale_cell) = (ring.code_stride / cells, ring.scale_stride / cells);
    let mut stored = Stored::default();
    for position in rows {
        let cell = (position % cells) as usize;
        let at = |bytes: &[u8], width: u64| {
            bytes[cell * width as usize..(cell + 1) * width as usize].to_vec()
        };
        stored.codes.extend(at(&codes, code_cell));
        stored.scales.extend(at(&scales, scale_cell));
    }
    Ok(stored)
}

/// KV source `layer`'s rows `rows` (compressed positions) of `plane` of
/// `item` on `rank`, from the item's pages: in each page, the plane's
/// stream holds the codes (or BF16 values) of its positions, then their
/// scales (`state::request_state`, `page_planes`).
pub fn source(
    services: &HostServicesV1,
    rank: &RankState,
    layout: &Layout,
    plane: Plane,
    layer: u32,
    item: &Item,
    rows: Range<u64>,
) -> Result<Stored, TapError> {
    let g = layout.geometry;
    let declaration = state::request_state(g, layout.storage, false);
    let ratio = g.compress_ratio(layer);
    let group = declaration
        .pages
        .iter()
        .find(|group| self::ratio(group) == ratio)
        .ok_or_else(|| invalid(format!("layer {layer} is no KV source")))?;
    let local = state::sources_of(g, ratio)
        .position(|source| source == layer)
        .ok_or_else(|| invalid(format!("layer {layer} is no KV source")))? as u64;
    let stored = declaration
        .page_planes()
        .into_iter()
        .find(|p| p.layers == group.name && p.layer == local && p.stream == plane.stream())
        .ok_or_else(|| invalid(format!("layer {layer} has no {} pages", plane.site())))?;
    let per_page = declaration.page_tokens / u64::from(ratio);
    let (code_bytes, scale_bytes) = plane.row_bytes(Widths::of(g), layout.storage);
    if per_page * (code_bytes + scale_bytes) != stored.bytes {
        return Err(invalid(format!(
            "layer {layer}'s {} pages disagree with the state layout",
            plane.site()
        )));
    }
    let mut out = Stored::default();
    let mut position = rows.start;
    while position < rows.end {
        let index = position / per_page;
        let first = position % per_page;
        let count = (rows.end - position).min(per_page - first);
        let page = item
            .pages
            .get(index as usize)
            .and_then(|&page| u64::try_from(page).ok())
            .filter(|&page| page < layout.pages)
            .ok_or_else(|| {
                invalid(format!(
                    "request {} has no page {index} for compressed position {position}",
                    item.request
                ))
            })?;
        let base = page * layout.page_bytes + stored.offset;
        let span = |offset: u64, length: u64| part(rank.pool, base + offset, length);
        out.codes.extend(read(
            services,
            rank.stream,
            span(first * code_bytes, count * code_bytes)?,
        )?);
        if scale_bytes > 0 {
            out.scales.extend(read(
                services,
                rank.stream,
                span(
                    per_page * code_bytes + first * scale_bytes,
                    count * scale_bytes,
                )?,
            )?);
        }
        position += count;
    }
    Ok(out)
}

/// Writes `taps`' rows `rows` of `item` into `step` in their declared
/// order, each layer's plane read once from every rank of `ranks`. A
/// native row whose codes and scales do not decode stops the step there,
/// incomplete, and is the note returned.
pub fn write(
    services: &HostServicesV1,
    ranks: &[RankState],
    layout: &Layout,
    step: &mut TapStep,
    taps: &[KvTap],
    item: &Item,
    rows: Range<u64>,
) -> Result<Result<(), String>, TapError> {
    let w = Widths::of(layout.geometry);
    let mut at = 0;
    while at < taps.len() {
        let (layer, plane) = (taps[at].layer, taps[at].plane);
        let group = taps[at..]
            .iter()
            .take_while(|tap| (tap.layer, tap.plane) == (layer, plane))
            .count();
        let stored = ranks
            .iter()
            .map(|rank| match plane {
                Plane::Window => window(services, rank, layout, layer, item, rows.clone()),
                Plane::Compressed | Plane::Index => {
                    source(services, rank, layout, plane, layer, item, rows.clone())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        for tap in &taps[at..at + group] {
            for (rank, stored) in stored.iter().enumerate() {
                let decoded;
                let bytes = match (tap.part, layout.storage) {
                    (Part::Codes, _) | (Part::Values, KvStorage::Bf16) => &stored.codes,
                    (Part::Scales, _) => &stored.scales,
                    (Part::Values, KvStorage::Native) => match decode(plane, w, stored) {
                        Ok(values) => {
                            decoded = values;
                            &decoded
                        }
                        Err(error) => {
                            return Ok(Err(format!(
                                "{} rank {rank} (rows from position {}): {error}",
                                tap.spec.name, rows.start
                            )));
                        }
                    },
                };
                step.write(&tap.spec, rank as u32, bytes)?;
            }
        }
        at += group;
    }
    Ok(Ok(()))
}

/// The little-endian BF16 products of `stored`'s native rows of `plane`,
/// each code times its block scale decoded exactly
/// (`Quantized::decode`); refuses, naming the row, a product that is not
/// exactly a finite BF16, which no valid store holds.
pub fn decode(plane: Plane, w: Widths, stored: &Stored) -> Result<Vec<u8>, String> {
    let width = plane.width(w) as usize;
    let (code_bytes, _) = plane.row_bytes(w, KvStorage::Native);
    let rows = stored.codes.len() / code_bytes as usize;
    if rows == 0 {
        return Ok(Vec::new());
    }
    let format = plane.quantizer();
    let scales = match format.scale {
        ScaleFormat::Ue8M0 => Scales::Ue8M0(stored.scales.clone()),
        ScaleFormat::E4M3 => Scales::Ue4M3(stored.scales.clone()),
        ScaleFormat::F32 | ScaleFormat::Bf16 => {
            return Err(format!("{} stores no byte scales", plane.site()));
        }
    };
    let quantized = Quantized::from_parts(rows, width, format, stored.codes.clone(), scales, 1.0)
        .map_err(|refusal| format!("{}: {}", plane.site(), refusal.0))?;
    let mut values = Vec::with_capacity(rows * width * 2);
    for row in 0..rows {
        for column in 0..width {
            let value = quantized.decode(row, column);
            let bits = f64_to_bf16(value);
            if !value.is_finite() || f64::from(bf16_to_f32(bits)) != value {
                return Err(format!(
                    "{} row {row} column {column}: the native code and scale decode to {value}, \
                     not a finite BF16",
                    plane.site()
                ));
            }
            values.extend(bits.to_le_bytes());
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};

    /// Each plane's native row is the declared entry: the window ring's
    /// cell (`ring_cell`), a compressed latent and an index key, and every
    /// raw part is whole u32 words.
    #[test]
    fn native_rows_are_the_declared_entries() {
        for g in GEOMETRIES {
            let w = Widths::of(g);
            for storage in [KvStorage::Native, KvStorage::Bf16] {
                let [cell_codes, cell_scales] = storage.ring_cell(w);
                assert_eq!(
                    Plane::Window.row_bytes(w, storage),
                    (cell_codes, cell_scales)
                );
                let entry = |plane: Plane| {
                    let (codes, scales) = plane.row_bytes(w, storage);
                    codes + scales
                };
                assert_eq!(entry(Plane::Compressed), storage.latent_bytes(w));
                assert_eq!(entry(Plane::Index), storage.index_bytes(w));
                for plane in [Plane::Window, Plane::Compressed, Plane::Index] {
                    let (codes, scales) = plane.row_bytes(w, storage);
                    assert_eq!((codes % 4, scales % 4), (0, 0), "{plane:?} {storage:?}");
                }
            }
        }
    }

    /// Flash's axes are the window and its two page groups, ratio 2 (KV
    /// sources 2, 8 and 14) then ratio 1 (source 20); the native layout
    /// writes codes, scales and values of every plane, the BF16 layout
    /// values alone, all detail taps on every rank.
    #[test]
    fn flash_taps_every_layer_window_and_every_source() {
        assert_eq!(
            axes(&FLASH),
            [
                Axis::Window,
                Axis::Sources { ratio: 2 },
                Axis::Sources { ratio: 1 }
            ]
        );
        let names = |storage, axis| -> Vec<String> {
            taps(&FLASH, storage, axis)
                .into_iter()
                .map(|tap| tap.spec.name)
                .collect()
        };
        let window = names(KvStorage::Bf16, Axis::Window);
        assert_eq!(window.len(), FLASH.layers() as usize);
        assert_eq!(window[3], "L03.x.kv.window");
        assert_eq!(
            names(KvStorage::Native, Axis::Sources { ratio: 1 }),
            [
                "L20.x.kv.compressed.codes",
                "L20.x.kv.compressed.scales",
                "L20.x.kv.compressed",
                "L20.x.kv.index.codes",
                "L20.x.kv.index.scales",
                "L20.x.kv.index",
            ]
        );
        assert_eq!(names(KvStorage::Bf16, Axis::Sources { ratio: 2 }).len(), 6);
        for storage in [KvStorage::Native, KvStorage::Bf16] {
            for spec in specs(&FLASH, storage) {
                assert!(spec.detail, "{}", spec.name);
                assert_eq!(spec.sharding, TapSharding::RankCopies);
            }
        }
        let rows = |axis: Axis, end| axis.rows(&FLASH, end);
        assert_eq!(rows(Axis::Window, 5), 0..5);
        assert_eq!(rows(Axis::Window, 300), 172..300);
        assert_eq!(rows(Axis::Sources { ratio: 2 }, 9), 0..4);
        assert_eq!(rows(Axis::Sources { ratio: 1 }, 9), 0..9);
    }

    /// The shared decoder's products of the trained quantizers' codes are
    /// their fake-quantized BF16 values, and a product no BF16 holds is
    /// refused with its row.
    #[test]
    fn native_rows_decode_to_the_fake_quantized_values() {
        let w = Widths::of(&FLASH);
        for plane in [Plane::Window, Plane::Compressed, Plane::Index] {
            let width = plane.width(w) as usize;
            let values: Vec<f32> = (0..2 * width)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.173)
                .collect();
            let quantized = plane.format().apply(&values, 2, width).unwrap();
            let (Scales::Ue8M0(scales) | Scales::Ue4M3(scales)) = quantized.scales else {
                panic!("byte scales")
            };
            let stored = Stored {
                codes: quantized.codes,
                scales,
            };
            let expected: Vec<u8> = plane
                .format()
                .fake_quant(&values, 2, width)
                .unwrap()
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect();
            assert_eq!(decode(plane, w, &stored).unwrap(), expected, "{plane:?}");
            // A NaN scale (UE8M0 0xff, or a UE4M3 byte with its high bit).
            let mut bad = stored.clone();
            let row = bad.scales.len() / 2;
            bad.scales[row + 1] = 0xff;
            let error = decode(plane, w, &bad).unwrap_err();
            assert!(error.contains("row 1 "), "{error}");
        }
    }
}
