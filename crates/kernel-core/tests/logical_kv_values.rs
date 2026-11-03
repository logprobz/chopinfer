//! The native DeepSeek KV layout and its BF16 comparison layout hold the
//! same logical values. For each plane, the BF16 layout's store of the rows
//! the native layout quantizes equals the native decode bit for bit, at the
//! edge scale bytes the format reaches, and one flipped stored bit, native
//! or BF16, changes the oracle that reads the plane.
//!
//! - Window ring, E4M3 codes and a UE8M0 scale per 32: the native writer
//!   encodes with `encode_latent_row`, `store_window_rows` writes either
//!   layout, and latent attention decodes rows as `decode_native_page` does.
//! - Compressed latent, E2M1 codes and an E4M3 scale per 16, and index key,
//!   E2M1 codes and a UE8M0 scale per 32: the writers take the shared
//!   quantizer's row status, then `apply` for codes or `fake_quant` for
//!   BF16. A flagged row stores zero codes with the invalid scale byte, or
//!   BF16 zeros. Latent attention reads compressed pages; the pooled indexer
//!   oracle scores decoded index keys.

use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
use kernel_core::contract::{AttentionPhase, DType, Refusal};
use kernel_core::latent_attention::{
    LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat, LatentWindowStorage,
};
use kernel_core::latent_attention_oracle::{
    EncodedLatent, LatentAttentionResult, LatentCompressedInput, LatentKeyLists, LatentPaging,
    LatentSourcesInput, LatentWindowInput, decode_native_page, encode_latent_row,
    latent_attention_with_sources, latent_kv_write,
};
use kernel_core::latent_glue::{LatentWindowWrite, WindowAddressing};
use kernel_core::latent_glue_oracle::{exact_window_bf16, store_window_rows};
use kernel_core::pooled_indexer::{IndexerCandidates, PooledIndexerQuery, PositionIndexerQuery};
use kernel_core::pooled_indexer_oracle::{decode_bf16, pool_scores};
use kernel_core::precision::quantizer::{
    BlockQuantization, BlockQuantizeQuery, NONFINITE_INPUT, QuantizeOutput, SCALE_OVERFLOW,
    UNREPRESENTABLE_OUTPUT,
};
use kernel_core::quantize::Scales;

const LATENT: usize = 512;
const INDEX: usize = 128;
/// Window cells: the 128-token window and eight verify rows of headroom.
const RING: usize = 136;
/// Compressed positions per page.
const PAGE: usize = 8;
const WINDOW: LatentKvFormat = LatentKvFormat::Fp8E4M3Ue8M0Block32;
const COMPRESSED: LatentKvFormat = LatentKvFormat::Fp4E2M1E4M3Block16;

fn widen(bits: u16) -> f64 {
    f64::from(bf16_to_f32(bits))
}

/// BF16 bits of a value BF16 holds exactly.
fn exact_bf16(value: f64) -> u16 {
    let bits = f32_to_bf16(value as f32);
    assert_eq!(
        widen(bits).to_bits(),
        value.to_bits(),
        "{value:e} is not BF16"
    );
    bits
}

fn e4m3(code: u8) -> f64 {
    SmallFloat::E4M3.decode_f64(code)
}

fn e2m1(code: u8) -> f64 {
    SmallFloat::E2M1.decode_f64(code)
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

/// Each stored BF16 value widens to the native decode, signed zero included.
fn assert_holds(stored: &[u16], native: &[f64], what: &str) {
    assert_eq!(stored.len(), native.len(), "{what}");
    for (column, (&bits, &value)) in stored.iter().zip(native).enumerate() {
        assert_eq!(
            widen(bits).to_bits(),
            value.to_bits(),
            "{what} column {column}: BF16 {bits:#06x}, native {value:e}"
        );
    }
}

/// A deterministic value in [-1, 1].
fn spread(seed: usize, i: usize) -> f32 {
    ((seed * 7919 + i * 104_729) % 2001) as f32 / 1000.0 - 1.0
}

/// Flip each target bit of `stored` in turn. Every flip must change the
/// oracle's output bits or be refused; the refused flips are returned.
fn flips(
    stored: &[u8],
    targets: impl IntoIterator<Item = (usize, u32)>,
    baseline: &[u64],
    mut oracle: impl FnMut(&[u8]) -> Result<Vec<u64>, Refusal>,
    what: &str,
) -> Vec<(usize, u32)> {
    let mut flipped = stored.to_vec();
    let mut refused = Vec::new();
    let mut count = 0;
    for (byte, bit) in targets {
        flipped[byte] ^= 1 << bit;
        match oracle(&flipped) {
            Ok(output) => assert!(
                output != baseline,
                "{what}: flipping byte {byte} bit {bit} left the oracle output unchanged"
            ),
            Err(_) => refused.push((byte, bit)),
        }
        flipped[byte] ^= 1 << bit;
        count += 1;
    }
    assert!(count > 0, "{what}: no flips");
    refused
}

/// Every bit of each listed byte.
fn every_bit(bytes: impl IntoIterator<Item = usize>) -> Vec<(usize, u32)> {
    bytes
        .into_iter()
        .flat_map(|byte| (0..8).map(move |bit| (byte, bit)))
        .collect()
}

fn window_write(kv: LatentKvFormat) -> LatentWindowWrite {
    LatentWindowWrite {
        kv,
        latent: LATENT as u32,
        rotary: 64,
        epsilon_bits: 1e-20_f32.to_bits(),
        addressing: WindowAddressing::Circular { slots: RING as u32 },
    }
}

/// One sequence's window ring in both layouts.
struct Ring {
    codes: Vec<u8>,
    scales: Vec<u8>,
    bf16: Vec<u8>,
}

impl Ring {
    /// Rows of each layout's writer at ring `positions`.
    fn store(rows: &[EncodedLatent], positions: &[i32]) -> Ring {
        let batch = vec![0; rows.len()];
        let last = [*positions.iter().max().unwrap()];
        let mut ring = Ring {
            codes: vec![0xa5; RING * LATENT],
            scales: vec![0xa5; RING * LATENT / 32],
            bf16: vec![0xa5; RING * LATENT * 2],
        };
        store_window_rows(
            window_write(WINDOW),
            rows,
            positions,
            &batch,
            &[0],
            &last,
            &mut ring.codes,
            &mut ring.scales,
            RING * LATENT,
            RING * LATENT / 32,
        )
        .unwrap();
        store_window_rows(
            window_write(LatentKvFormat::Bf16),
            rows,
            positions,
            &batch,
            &[0],
            &last,
            &mut ring.bf16,
            &mut [],
            RING * LATENT * 2,
            0,
        )
        .unwrap();
        ring
    }

    fn native(&self, cell: usize) -> Vec<f64> {
        let mut page = self.codes[cell * LATENT..(cell + 1) * LATENT].to_vec();
        page.extend_from_slice(&self.scales[cell * LATENT / 32..(cell + 1) * LATENT / 32]);
        decode_native_page(WINDOW, LATENT, 1, &page, 0).unwrap()
    }

    fn stored(&self, cell: usize) -> Vec<u16> {
        u16s(&self.bf16[cell * LATENT * 2..(cell + 1) * LATENT * 2])
    }
}

/// One compressed or index row as its writer stores it.
struct Written {
    codes: Vec<u8>,
    scales: Vec<u8>,
    bf16: Vec<u16>,
    status: u32,
}

/// The shared quantizer's status, then codes and scales or the BF16 store;
/// a flagged row writes zero codes with the invalid scale byte, or zeros.
fn write(format: BlockQuantization, row: &[f32], prior: u32) -> Written {
    let width = row.len();
    let query = BlockQuantizeQuery {
        width: width as u32,
        input: DType::F32,
        format,
        output: QuantizeOutput::Codes,
    };
    let status = query.row_status(row, &[prior]).unwrap()[0];
    if status != 0 {
        return Written {
            codes: vec![0; width / 2],
            scales: vec![query.invalid_scale_code(); width / format.group_size() as usize],
            bf16: vec![0; width],
            status,
        };
    }
    let quantized = format.apply(row, 1, width).unwrap();
    let scales = match quantized.scales {
        Scales::Ue8M0(scales) | Scales::Ue4M3(scales) => scales,
        Scales::F32(_) => panic!("FP4 scales are bytes"),
    };
    Written {
        codes: quantized.codes,
        scales,
        bf16: format.fake_quant(row, 1, width).unwrap(),
        status,
    }
}

/// One page of `PAGE` FP4 rows: code rows, scale rows and the BF16 layout.
struct Pages {
    width: usize,
    block: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    bf16: Vec<u16>,
}

impl Pages {
    fn store(width: usize, block: usize, rows: &[Written]) -> Pages {
        let mut pages = Pages {
            width,
            block,
            codes: vec![0xa5; PAGE * width / 2],
            scales: vec![0xa5; PAGE * width / block],
            bf16: vec![0xa5a5; PAGE * width],
        };
        for (slot, row) in rows.iter().enumerate() {
            pages.codes[slot * width / 2..(slot + 1) * width / 2].copy_from_slice(&row.codes);
            pages.scales[slot * width / block..(slot + 1) * width / block]
                .copy_from_slice(&row.scales);
            pages.bf16[slot * width..(slot + 1) * width].copy_from_slice(&row.bf16);
        }
        pages
    }

    /// The latent page: every code row, then every scale row.
    fn pool(&self) -> Vec<u8> {
        [&self.codes[..], &self.scales[..]].concat()
    }

    fn row_codes(&self, slot: usize) -> &[u8] {
        &self.codes[slot * self.width / 2..(slot + 1) * self.width / 2]
    }

    fn row_scales(&self, slot: usize) -> &[u8] {
        &self.scales[slot * self.width / self.block..(slot + 1) * self.width / self.block]
    }

    fn stored(&self, slot: usize) -> &[u16] {
        &self.bf16[slot * self.width..(slot + 1) * self.width]
    }
}

/// The native index reader's decode, code times scale.
fn index_decode(codes: &[u8], scales: &[u8]) -> Vec<f64> {
    (0..codes.len() * 2)
        .map(|c| e2m1((codes[c / 2] >> (4 * (c % 2))) & 15) * decode_ue8m0_f64(scales[c / 32]))
        .collect()
}

/// Latent planes as a launch reads them: native or BF16 window and pool.
#[derive(Clone, Copy)]
struct Stored<'a> {
    bf16: bool,
    window: &'a [u8],
    window_scales: &'a [u8],
    pool: &'a [u8],
}

fn attention(kv: LatentKvFormat, output: LatentAttentionOutput) -> LatentAttentionQuery {
    LatentAttentionQuery::Attention {
        heads: 2,
        latent: LATENT as u32,
        max_keys: 8,
        softmax_scale_bits: (LATENT as f32).powf(-0.5).to_bits(),
        page_size: PAGE as u32,
        kv,
        phase: AttentionPhase::Decode,
        output,
        sink: true,
        tail_rope_dims: 64,
        second_fp4_keys: true,
        capacity: 1,
    }
}

/// One query row over window cells and compressed positions of item 0.
fn attend(
    stored: Stored<'_>,
    output: LatentAttentionOutput,
    q: &[f64],
    window: &[i32],
    compressed: &[i32],
) -> Result<Vec<u64>, Refusal> {
    let row = if stored.bf16 { 2 * LATENT } else { LATENT };
    let result = latent_attention_with_sources(
        attention(
            if stored.bf16 {
                LatentKvFormat::Bf16
            } else {
                WINDOW
            },
            output,
        ),
        LatentSourcesInput {
            q,
            row_batch: &[0],
            window: LatentWindowInput {
                codes: stored.window,
                scales: stored.window_scales,
                item_slots: &[0],
                sequences: 1,
                code_stride: RING * row,
                scale_stride: if stored.bf16 { 0 } else { RING * LATENT / 32 },
                storage: LatentWindowStorage::Staged { slots: RING as u32 },
                lists: LatentKeyLists {
                    indices: window,
                    stride: window.len(),
                    budget: window.len(),
                },
            },
            compressed: Some(LatentCompressedInput {
                pool: stored.pool,
                pages: 1,
                paging: LatentPaging {
                    block_tables: &[0],
                    max_pages: 1,
                    page_stride: stored.pool.len(),
                },
                lists: LatentKeyLists {
                    indices: compressed,
                    stride: compressed.len(),
                    budget: compressed.len(),
                },
            }),
            sinks: Some(&[0.25, -0.5]),
        },
    )?;
    Ok(match result {
        LatentAttentionResult::Normalized(out) => out.values,
        LatentAttentionResult::Partial(out) => [
            out.maximum.values,
            out.denominator.values,
            out.numerator.values,
        ]
        .concat(),
    }
    .into_iter()
    .map(f64::to_bits)
    .collect())
}

/// Both layouts give latent attention the same output bits in both modes.
fn assert_attention_agrees(
    native: Stored<'_>,
    bf16: Stored<'_>,
    q: &[f64],
    window: &[i32],
    compressed: &[i32],
) {
    for output in [
        LatentAttentionOutput::Normalized,
        LatentAttentionOutput::Partial,
    ] {
        assert_eq!(
            attend(native, output, q, window, compressed).unwrap(),
            attend(bf16, output, q, window, compressed).unwrap(),
            "{output:?}"
        );
    }
}

fn index_query() -> PooledIndexerQuery {
    PositionIndexerQuery {
        heads: 2,
        head_dim: INDEX as u32,
        compression_ratio: 4,
        tail_rope_dims: 64,
        key_norm_eps_bits: 1e-20_f32.to_bits(),
        max_top_k: 512,
        page_size: 64,
        candidates: IndexerCandidates::All,
        capacity: 1,
    }
    .selection()
}

fn scores(q: &[f64], keys: &[f64]) -> Result<Vec<u64>, Refusal> {
    Ok(pool_scores(index_query(), q, &[1.0, 0.75], keys)?
        .values
        .into_iter()
        .map(f64::to_bits)
        .collect())
}

/// Window blocks and the UE8M0 byte the native encoder gives each: grids
/// of E4M3 codes from the amax floor (105) to the largest admitted scale
/// (247), then rounded BF16 blocks: signed zeros, below the floor, E4M3
/// ties at 448, the ceil step at 450 and the largest input that encodes
/// inside BF16.
fn window_blocks() -> Vec<(Vec<u16>, u8)> {
    // Both signs of the largest finite code, subnormals, the smallest
    // normal, one, signed zeros and codes with every exponent.
    const CODES: [u8; 32] = [
        0x7e, 0xfe, 0x01, 0x81, 0x02, 0x07, 0x87, 0x08, 0x88, 0x09, 0x38, 0xb8, 0x00, 0x80, 0x3f,
        0x40, 0x44, 0x4a, 0x52, 0x5a, 0x62, 0x6a, 0x72, 0x77, 0xf7, 0x7d, 0x7a, 0x30, 0x20, 0x10,
        0x0f, 0x8f,
    ];
    let grid = |exponent: i32, cap: u8| {
        CODES
            .iter()
            .map(|&code| {
                let code = if code & 0x7f > cap {
                    code & 0x80 | cap
                } else {
                    code
                };
                exact_bf16(e4m3(code) * 2_f64.powi(exponent))
            })
            .collect::<Vec<_>>()
    };
    let rounded = |seed: usize, amax: f32, scale: f32| {
        (0..32)
            .map(|i| {
                f32_to_bf16(if i == 0 {
                    amax
                } else {
                    spread(seed, i) * scale
                })
            })
            .collect::<Vec<_>>()
    };
    let mut blocks = [-22, -21, -2, -1, 0, 1, 64, 118, 119]
        .map(|exponent| (grid(exponent, 0x7e), (127 + exponent) as u8))
        .to_vec();
    // 240 * 2^120: no larger code stays inside BF16 at this scale.
    blocks.push((grid(120, 0x77), 247));
    blocks.push(((0..32).map(|i| (i % 2) << 15).collect(), 105));
    blocks.push((rounded(1, 9.5e-5, 9e-5), 105));
    // Ties round to even: 1, -16, 2^-8 and zero.
    let mut ties = rounded(2, 448.0, 400.0);
    ties[1..5].copy_from_slice(
        &[1.0625_f32, -17.0, 1.5 * 2_f32.powi(-9), 2_f32.powi(-10)].map(f32_to_bf16),
    );
    blocks.push((ties, 127));
    blocks.push((rounded(3, 450.0, 400.0), 128));
    // 0x7f77 is 247 * 2^120 and rounds to the code 240.
    blocks.push((rounded(4, bf16_to_f32(0x7f77), 2_f32.powi(127)), 247));
    blocks.push((rounded(5, 3.75, 3.5), 121));
    blocks
}

#[test]
fn window_bf16_ring_holds_the_native_decode_at_edge_scales() {
    let blocks = window_blocks();
    assert_eq!(blocks.len() * 32, LATENT);
    let forward = blocks.iter().flat_map(|b| b.0.clone()).collect::<Vec<_>>();
    let reverse = blocks.iter().rev().flat_map(|b| b.0.clone()).collect();
    let mut encoded = Vec::new();
    for (row, order) in [(forward.clone(), false), (reverse, true)] {
        let row_encoded = encode_latent_row(WINDOW, &row).unwrap();
        let mut expected = blocks.iter().map(|b| b.1).collect::<Vec<_>>();
        if order {
            expected.reverse();
        }
        assert_eq!(row_encoded.scale_bytes, expected);
        // The device writer follows the shared quantizer, which agrees.
        let shared = BlockQuantization::E4M3_CEIL32
            .apply(
                &row.iter().map(|&b| bf16_to_f32(b)).collect::<Vec<_>>(),
                1,
                LATENT,
            )
            .unwrap();
        assert_eq!(shared.codes, row_encoded.values);
        assert_eq!(
            shared.scales,
            Scales::Ue8M0(row_encoded.scale_bytes.clone())
        );
        encoded.push(row_encoded);
    }
    // Positions 135 and 136 of a 136-cell ring are cells 135 and 0.
    let ring = Ring::store(&encoded, &[135, 136]);
    for cell in [135, 0] {
        assert_holds(
            &ring.stored(cell),
            &ring.native(cell),
            &format!("window cell {cell}"),
        );
    }
    assert!(
        ring.bf16[2 * LATENT..135 * 2 * LATENT]
            .iter()
            .all(|&b| b == 0xa5)
    );
    // A zero query reads the edge values from both layouts alike.
    let pool = vec![0; COMPRESSED.page_bytes(LATENT as u32, PAGE as u32) as usize];
    let pool_bf16 = vec![0; LatentKvFormat::Bf16.page_bytes(LATENT as u32, PAGE as u32) as usize];
    assert_attention_agrees(
        Stored {
            bf16: false,
            window: &ring.codes,
            window_scales: &ring.scales,
            pool: &pool,
        },
        Stored {
            bf16: true,
            window: &ring.bf16,
            window_scales: &[],
            pool: &pool_bf16,
        },
        &[0.0; 2 * LATENT],
        &[135, 0],
        &[-1],
    );
    // 0x7f78 is 248 * 2^120, a tie that rounds to the code 256, past BF16.
    // Neither layout can store the row: the status flags it as a hole.
    let mut overflow = forward;
    overflow[0] = 0x7f78;
    assert!(encode_latent_row(WINDOW, &overflow).is_err());
    let overflow = overflow.iter().map(|&b| bf16_to_f32(b)).collect::<Vec<_>>();
    assert!(
        BlockQuantization::E4M3_CEIL32
            .fake_quant(&overflow, 1, LATENT)
            .is_err()
    );
    let status = BlockQuantizeQuery {
        width: LATENT as u32,
        input: DType::F32,
        format: BlockQuantization::E4M3_CEIL32,
        output: QuantizeOutput::Codes,
    }
    .row_status(&overflow, &[0])
    .unwrap();
    assert_eq!(status, [UNREPRESENTABLE_OUTPUT]);
}

#[test]
fn window_bf16_store_of_raw_native_bytes_admits_what_the_native_decoder_admits() {
    // UE8M0 bytes below and above the encoder's 105..=247, the NaN byte,
    // and E4M3 codes at zero, subnormal, normal, largest and NaN points.
    for scale in [0_u8, 1, 2, 3, 104, 105, 127, 247, 248, 252, 253, 254, 255] {
        for code in [
            0x00_u8, 0x80, 0x01, 0x81, 0x07, 0x08, 0x38, 0x40, 0x77, 0x7e, 0xfe, 0x7f, 0xff,
        ] {
            let row = EncodedLatent {
                values: vec![code; LATENT],
                scales: vec![],
                scale_bytes: vec![scale; LATENT / 32],
            };
            let mut page = row.values.clone();
            page.extend_from_slice(&row.scale_bytes);
            let product = e4m3(code) * decode_ue8m0_f64(scale);
            let admitted = product.is_finite()
                && widen(f32_to_bf16(product as f32)).to_bits() == product.to_bits();
            let native = decode_native_page(WINDOW, LATENT, 1, &page, 0);
            let stored = exact_window_bf16(&row, LATENT);
            let what = format!("code {code:#04x} scale {scale}");
            assert_eq!(native.is_ok(), admitted, "{what}");
            assert_eq!(stored.is_ok(), admitted, "{what}");
            if admitted {
                assert_holds(&u16s(&stored.unwrap()), &native.unwrap(), &what);
            }
        }
    }
}

/// Compressed blocks and the E4M3 scale byte the shared quantizer gives
/// each: all sixteen E2M1 codes on grids from the amax floor (0x01, the
/// smallest subnormal) through every subnormal and the smallest normal to
/// the largest finite scale (0x7e), then rounded blocks: signed zeros,
/// below and just above the floor, scale and code ties, saturation, the
/// largest BF16 input and an F32 input that rounds to BF16 before amax.
fn compressed_blocks() -> Vec<(Vec<f32>, u8)> {
    let grid = |scale: u8| {
        (0..16)
            .map(|code| (e2m1(code) * e4m3(scale)) as f32)
            .collect::<Vec<_>>()
    };
    let rounded = |seed: usize, amax: f32, scale: f32| {
        (0..16)
            .map(|i| {
                if i == 0 {
                    amax
                } else {
                    spread(seed, i) * scale
                }
            })
            .collect::<Vec<_>>()
    };
    let mut blocks = [
        0x01_u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x20, 0x30, 0x37, 0x38,
        0x40, 0x48, 0x50, 0x60, 0x70, 0x77, 0x7c, 0x7d, 0x7e,
    ]
    .map(|scale| (grid(scale), scale))
    .to_vec();
    blocks.push((
        (0..16)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect(),
        0x01,
    ));
    blocks.push((
        (0..16)
            .map(|i| (i as f32 - 7.5) * 2_f32.powi(-13))
            .collect(),
        0x01,
    ));
    // One BF16 step above the floor 6 * 2^-9 keeps the scale 2^-9.
    let floor = 6.0 * 2_f32.powi(-9);
    blocks.push((
        rounded(
            1,
            f32::from_bits(floor.to_bits() + 0x1_0000),
            2_f32.powi(-8),
        ),
        0x01,
    ));
    // 7 / 6 rounds to the scale 1.125.
    blocks.push((rounded(2, 7.0, 6.5), 0x39));
    // Code ties at scale 1 round to even: 1, 2, 4, 0, 1, 2 and 4.
    blocks.push((
        vec![
            6.0, 1.25, -1.25, 2.5, -2.5, 5.0, -5.0, 0.25, -0.25, 0.75, -0.75, 1.75, -1.75, 3.5,
            -3.5, 0.5,
        ],
        0x38,
    ));
    blocks.push((rounded(3, 3000.0, 2900.0), 0x7e));
    blocks.push((rounded(4, bf16_to_f32(0x7f7f), 1e38), 0x7e));
    // In F32 the scale would round down to 1.125; BF16 rounding makes 7.125
    // an E4M3 tie, which rounds to 1.25.
    blocks.push((rounded(5, 7.125 - 2_f32.powi(-20), 6.5), 0x3a));
    blocks.push((rounded(6, 5.25, 5.0), 0x36));
    blocks
}

#[test]
fn compressed_bf16_pages_hold_the_native_decode_at_edge_scales() {
    let format = BlockQuantization::E2M1E4M3Scale16;
    let blocks = compressed_blocks();
    assert_eq!(blocks.len() * 16, LATENT);
    let forward = blocks.iter().flat_map(|b| b.0.clone()).collect::<Vec<_>>();
    let reverse = blocks
        .iter()
        .rev()
        .flat_map(|b| b.0.clone())
        .collect::<Vec<_>>();
    let mut nan = forward.clone();
    nan[5] = f32::NAN;
    // F32 above the largest BF16 rounds to infinity.
    let mut infinite = forward.clone();
    infinite[0] = f32::MAX;
    let written = [
        write(format, &forward, 0),
        write(format, &reverse, 0),
        write(format, &nan, 0),
        write(format, &infinite, 0),
        write(format, &forward, SCALE_OVERFLOW),
    ];
    assert_eq!(
        written.iter().map(|w| w.status).collect::<Vec<_>>(),
        [0, 0, NONFINITE_INPUT, NONFINITE_INPUT, SCALE_OVERFLOW]
    );
    let expected = blocks.iter().map(|b| b.1).collect::<Vec<_>>();
    assert_eq!(written[0].scales, expected);
    assert_eq!(
        written[1].scales,
        expected.iter().rev().copied().collect::<Vec<_>>()
    );
    for flagged in &written[2..] {
        assert!(flagged.scales.iter().all(|&s| s == 0x38));
    }
    let pages = Pages::store(LATENT, 16, &written);
    let pool = pages.pool();
    for slot in 0..written.len() {
        let native = decode_native_page(COMPRESSED, LATENT, PAGE, &pool, slot).unwrap();
        assert_holds(
            pages.stored(slot),
            &native,
            &format!("compressed slot {slot}"),
        );
    }
    // The latent oracle's own writer encodes the same pages.
    let rows = [&forward, &reverse]
        .iter()
        .flat_map(|row| row.iter().map(|&v| f32_to_bf16(v)))
        .collect::<Vec<_>>();
    let mut latent_pool = vec![0xa5; pool.len()];
    latent_kv_write(
        LatentAttentionQuery::KvWrite {
            latent: LATENT as u32,
            page_size: PAGE as u32,
            kv: COMPRESSED,
            capacity: 2,
        },
        &rows,
        &[0, 0],
        &[0, 1],
        LatentPaging {
            block_tables: &[0],
            max_pages: 1,
            page_stride: pool.len(),
        },
        &mut latent_pool,
    )
    .unwrap();
    for slot in 0..2 {
        let page = PAGE * LATENT / 2;
        assert_eq!(
            &latent_pool[slot * LATENT / 2..(slot + 1) * LATENT / 2],
            pages.row_codes(slot)
        );
        assert_eq!(
            &latent_pool[page + slot * LATENT / 16..page + (slot + 1) * LATENT / 16],
            pages.row_scales(slot)
        );
    }
    // A zero query reads the edge values and the flagged rows alike.
    let ring = vec![0; RING * LATENT];
    let ring_scales = vec![127; RING * LATENT / 32];
    let ring_bf16 = vec![0; RING * LATENT * 2];
    let pool_bf16 = bytes(&pages.bf16);
    assert_attention_agrees(
        Stored {
            bf16: false,
            window: &ring,
            window_scales: &ring_scales,
            pool: &pool,
        },
        Stored {
            bf16: true,
            window: &ring_bf16,
            window_scales: &[],
            pool: &pool_bf16,
        },
        &[0.0; 2 * LATENT],
        &[-1],
        &[0, 1, 2, 3, 4],
    );
    // No write produces a zero, NaN or negative scale byte, and the native
    // decoder refuses each, even under zero codes.
    for scale in [0x00_u8, 0x7f, 0x80, 0xff] {
        for code in [0x00_u8, 0x77] {
            let mut page = vec![code; LATENT / 2];
            page.extend(vec![scale; LATENT / 16]);
            assert!(
                decode_native_page(COMPRESSED, LATENT, 1, &page, 0).is_err(),
                "scale {scale:#04x} code {code:#04x}"
            );
        }
    }
}

/// Index blocks and the UE8M0 byte the shared quantizer gives each: every
/// E2M1 code twice on grids from the amax floor (1) to the largest scale
/// that admits every code (252), then rounded blocks: signed zeros, BF16
/// subnormals, the largest admitted scale (253, codes up to 3), the ceil
/// step above 6 and an F32 input that rounds to BF16 before amax.
fn index_blocks() -> Vec<(Vec<f32>, u8)> {
    let grid = |exponent: i32| {
        (0..32)
            .map(|i| (e2m1(i % 16) * 2_f64.powi(exponent)) as f32)
            .collect::<Vec<_>>()
    };
    let rounded = |seed: usize, amax: f32, scale: f32| {
        (0..32)
            .map(|i| {
                if i == 0 {
                    amax
                } else {
                    spread(seed, i) * scale
                }
            })
            .collect::<Vec<_>>()
    };
    let mut blocks = [-126, -125, -100, -1, 0, 1, 64, 100, 124, 125]
        .map(|exponent| (grid(exponent), (127 + exponent) as u8))
        .to_vec();
    blocks.push((
        (0..32)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect(),
        1,
    ));
    blocks.push((
        (0..32)
            .map(|i: i32| (2 * i - 31) as f32 * 2_f32.powi(-132))
            .collect(),
        1,
    ));
    // 0x7f41 is 1.5078125 * 2^127: its scale is 2^126 and its code 3.
    blocks.push((rounded(1, bf16_to_f32(0x7f41), 2_f32.powi(127)), 253));
    // 6.03125 is the BF16 after 6.
    blocks.push((rounded(2, 6.03125, 5.0), 128));
    blocks.push((rounded(3, 3.0, 2.9), 126));
    // Unrounded, 6 + 2^-18 would take the scale 2.
    blocks.push((rounded(4, 6.0 + 2_f32.powi(-18), 5.9), 127));
    blocks
}

#[test]
fn index_bf16_pages_hold_the_native_decode_at_edge_scales() {
    let format = BlockQuantization::E2M1Ceil32;
    let blocks = index_blocks();
    assert_eq!(blocks.len() * 32, 4 * INDEX);
    let mut written = blocks
        .chunks(4)
        .map(|row| {
            write(
                format,
                &row.iter().flat_map(|b| b.0.clone()).collect::<Vec<_>>(),
                0,
            )
        })
        .collect::<Vec<_>>();
    for (row, chunk) in written.iter().zip(blocks.chunks(4)) {
        assert_eq!(row.scales, chunk.iter().map(|b| b.1).collect::<Vec<_>>());
    }
    // 0x7f7f and 0x7f60 (1.75 * 2^127, a code tie at 3.5) round to the
    // code 4 at scale 2^126, past BF16.
    let first = blocks[..4]
        .iter()
        .flat_map(|b| b.0.clone())
        .collect::<Vec<_>>();
    for (value, status) in [
        (bf16_to_f32(0x7f7f), UNREPRESENTABLE_OUTPUT),
        (bf16_to_f32(0x7f60), UNREPRESENTABLE_OUTPUT),
        (f32::NAN, NONFINITE_INPUT),
    ] {
        let mut row = first.clone();
        row[0] = value;
        let flagged = write(format, &row, 0);
        assert_eq!(flagged.status, status);
        assert!(flagged.scales.iter().all(|&s| s == 127));
        written.push(flagged);
    }
    let pages = Pages::store(INDEX, 32, &written);
    let mut native = Vec::new();
    for slot in 0..written.len() {
        let decoded = index_decode(pages.row_codes(slot), pages.row_scales(slot));
        assert_holds(pages.stored(slot), &decoded, &format!("index slot {slot}"));
        native.extend(decoded);
    }
    let q = (0..2 * INDEX)
        .map(|i| ((i % 5) as f64 - 2.0) * 0.25)
        .collect::<Vec<_>>();
    assert_eq!(
        scores(&q, &native).unwrap(),
        scores(&q, &decode_bf16(&pages.bf16[..written.len() * INDEX])).unwrap()
    );
}

/// Window values of E4M3 magnitudes with at most five of seven bits set,
/// so no single flip reaches the NaN code, all at the scale 2^-9 (118), so
/// every flipped scale keeps exact BF16 products.
fn window_flip_row(seed: usize) -> Vec<u16> {
    const MAGNITUDES: [f64; 10] = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0];
    (0..LATENT)
        .map(|i| {
            let magnitude = if i % 32 == 0 {
                320.0
            } else {
                MAGNITUDES[(i * 7 + seed * 3) % 10]
            };
            let sign = if (i * 5 + seed) % 3 == 0 { -1.0 } else { 1.0 };
            exact_bf16(sign * magnitude * 2_f64.powi(-9))
        })
        .collect()
}

/// Nonzero E2M1 values at `scale`, each block holding 6, so the shared
/// quantizer keeps the scale.
fn fp4_flip_row(width: usize, block: usize, scale: f64, seed: usize) -> Vec<f32> {
    const MAGNITUDES: [f64; 7] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    (0..width)
        .map(|i| {
            let magnitude = if i % block == 0 {
                6.0
            } else {
                MAGNITUDES[(i * 3 + seed) % 7]
            };
            let sign = if (i * 11 + seed) % 4 == 0 { -1.0 } else { 1.0 };
            (sign * magnitude * scale) as f32
        })
        .collect()
}

#[test]
fn one_flipped_window_or_compressed_bit_changes_latent_attention() {
    let window = [window_flip_row(0), window_flip_row(1)].map(|row| {
        let encoded = encode_latent_row(WINDOW, &row).unwrap();
        assert!(encoded.scale_bytes.iter().all(|&s| s == 118));
        assert!(
            encoded
                .values
                .iter()
                .all(|&c| c & 0x7f != 0 && (c & 0x7f).count_ones() <= 5)
        );
        encoded
    });
    let ring = Ring::store(&window, &[0, 1]);
    let format = BlockQuantization::E2M1E4M3Scale16;
    // 0x1c is 0.09375: only its sign bit flips to an invalid scale, and
    // every value stays below 1, so no flipped BF16 exponent is infinite.
    let rows = [0, 1].map(|seed| write(format, &fp4_flip_row(LATENT, 16, 0.09375, seed), 0));
    assert!(rows.iter().all(|r| r.scales.iter().all(|&s| s == 0x1c)));
    let pages = Pages::store(LATENT, 16, &rows);
    let pool = pages.pool();
    let pool_bf16 = bytes(&pages.bf16);
    let native = Stored {
        bf16: false,
        window: &ring.codes,
        window_scales: &ring.scales,
        pool: &pool,
    };
    let bf16 = Stored {
        bf16: true,
        window: &ring.bf16,
        window_scales: &[],
        pool: &pool_bf16,
    };
    let q = (0..2 * LATENT)
        .map(|i| ((i * 5 % 9) as f64 - 4.0) * 2_f64.powi(-5))
        .collect::<Vec<_>>();
    let (lists, positions) = ([1, 0], [0, 1]);
    assert_attention_agrees(native, bf16, &q, &lists, &positions);
    let output = LatentAttentionOutput::Normalized;
    let baseline = attend(native, output, &q, &lists, &positions).unwrap();
    // Every scale byte, and in each scale block one value at a varying
    // offset, of window cell 1 and compressed position 0.
    let window_columns = (0..LATENT / 32).map(|b| LATENT + b * 32 + (5 * b + 3) % 32);
    let compressed_columns = (0..LATENT / 16).map(|b| b * 16 + (3 * b + 1) % 16);
    let refused = flips(
        &ring.codes,
        every_bit(window_columns.clone()),
        &baseline,
        |codes| {
            attend(
                Stored {
                    window: codes,
                    ..native
                },
                output,
                &q,
                &lists,
                &positions,
            )
        },
        "window code",
    );
    assert!(refused.is_empty(), "{refused:?}");
    let refused = flips(
        &ring.scales,
        every_bit(LATENT / 32..LATENT / 16),
        &baseline,
        |scales| {
            let stored = Stored {
                window_scales: scales,
                ..native
            };
            attend(stored, output, &q, &lists, &positions)
        },
        "window scale",
    );
    assert!(refused.is_empty(), "{refused:?}");
    let refused = flips(
        &ring.bf16,
        every_bit(window_columns.flat_map(|c| [2 * c, 2 * c + 1])),
        &baseline,
        |values| {
            attend(
                Stored {
                    window: values,
                    ..bf16
                },
                output,
                &q,
                &lists,
                &positions,
            )
        },
        "window BF16",
    );
    assert!(refused.is_empty(), "{refused:?}");
    let refused = flips(
        &pool,
        every_bit(compressed_columns.clone().map(|c| c / 2)),
        &baseline,
        |pool| attend(Stored { pool, ..native }, output, &q, &lists, &positions),
        "compressed code",
    );
    assert!(refused.is_empty(), "{refused:?}");
    // Flipping an E4M3 scale's sign bit makes it negative, which the native
    // decoder refuses; every other flip changes the output.
    let scale_plane = PAGE * LATENT / 2;
    let refused = flips(
        &pool,
        every_bit(scale_plane..scale_plane + LATENT / 16),
        &baseline,
        |pool| attend(Stored { pool, ..native }, output, &q, &lists, &positions),
        "compressed scale",
    );
    assert_eq!(
        refused,
        (scale_plane..scale_plane + LATENT / 16)
            .map(|byte| (byte, 7))
            .collect::<Vec<_>>()
    );
    let refused = flips(
        &pool_bf16,
        every_bit(compressed_columns.flat_map(|c| [2 * c, 2 * c + 1])),
        &baseline,
        |pool| attend(Stored { pool, ..bf16 }, output, &q, &lists, &positions),
        "compressed BF16",
    );
    assert!(refused.is_empty(), "{refused:?}");
}

#[test]
fn one_flipped_index_key_bit_changes_its_score() {
    let format = BlockQuantization::E2M1Ceil32;
    let rows = [0, 1].map(|seed| write(format, &fp4_flip_row(INDEX, 32, 2_f64.powi(-7), seed), 0));
    // 120 is 2^-7: every flipped scale keeps exact BF16 products.
    assert!(rows.iter().all(|r| r.scales.iter().all(|&s| s == 120)));
    let pages = Pages::store(INDEX, 32, &rows);
    let native = |codes: &[u8], scales: &[u8]| {
        let keys = (0..2)
            .flat_map(|slot| {
                index_decode(
                    &codes[slot * INDEX / 2..(slot + 1) * INDEX / 2],
                    &scales[slot * INDEX / 32..(slot + 1) * INDEX / 32],
                )
            })
            .collect::<Vec<_>>();
        for &key in &keys {
            assert_eq!(widen(f32_to_bf16(key as f32)).to_bits(), key.to_bits());
        }
        keys
    };
    let keys = native(&pages.codes, &pages.scales);
    // Head 0 agrees in sign with key 0 everywhere, so its dot stays positive.
    let q = (0..2 * INDEX)
        .map(|i| {
            let sign = keys[i % INDEX].signum() * if i < INDEX { 1.0 } else { -1.0 };
            sign * (0.25 + (i % 4) as f64 * 0.125)
        })
        .collect::<Vec<_>>();
    let bf16 = &pages.bf16[..2 * INDEX];
    let baseline = scores(&q, &keys).unwrap();
    assert_eq!(baseline, scores(&q, &decode_bf16(bf16)).unwrap());
    // Every bit of key 0's codes, scales and BF16 values.
    let refused = flips(
        &pages.codes,
        every_bit(0..INDEX / 2),
        &baseline,
        |codes| scores(&q, &native(codes, &pages.scales)),
        "index code",
    );
    assert!(refused.is_empty(), "{refused:?}");
    let refused = flips(
        &pages.scales,
        every_bit(0..INDEX / 32),
        &baseline,
        |scales| scores(&q, &native(&pages.codes, scales)),
        "index scale",
    );
    assert!(refused.is_empty(), "{refused:?}");
    let refused = flips(
        &bytes(bf16),
        every_bit(0..2 * INDEX),
        &baseline,
        |values| scores(&q, &decode_bf16(&u16s(values))),
        "index BF16",
    );
    assert!(refused.is_empty(), "{refused:?}");
}
