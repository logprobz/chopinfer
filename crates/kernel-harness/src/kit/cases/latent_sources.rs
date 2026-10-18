//! Native latent source conformance. The initial constant-query cases
//! isolate exact storage decoding, one softmax, sink ownership and the
//! inverse tail permutation from probability-quantization error.

use crate::LabResult;
use crate::conformance::{Expected, Holes, Images, Rows};
use crate::kit::{OpCase, Required, Spec, bf16_bytes};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{AttentionPhase, DType};
use kernel_core::latent_attention::{
    BlockTables, LatentAttentionOutput, LatentAttentionQuery, LatentCompressedBuffers,
    LatentKvCache, LatentKvFormat, LatentPartial, LatentSourcesBuffers, LatentWindowBuffers,
    LatentWindowStorage,
};
use kernel_core::latent_attention_oracle::{
    LatentAttentionResult, LatentCompressedInput, LatentKeyLists, LatentPaging, LatentSourcesInput,
    LatentWindowInput, decode_native_page, latent_attention_with_sources,
};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernels_sm120::latent::sources::Entry;

pub const CAPACITY: u32 = 4;
const PAD: usize = 64;
const SEQUENCES: usize = 2;
const ITEMS: usize = 3;

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// Empty partials carry an exact -inf sentinel. The numeric gate covers
// finite arithmetic only, so keep that sentinel out of its reduction.
fn maximum_expected(reference: OracleOutput, live_elements: usize) -> Expected {
    Expected::Judge(Box::new(move |bytes| {
        if bytes.len() != reference.values.len() * 4 || live_elements > reference.values.len() {
            return Err("partial maximum shape mismatch".into());
        }
        let mut actual = Vec::new();
        let mut finite = OracleOutput {
            values: Vec::new(),
            sum_abs_products: Vec::new(),
            reduction_lengths: Vec::new(),
        };
        for (at, bytes) in bytes.chunks_exact(4).take(live_elements).enumerate() {
            let bits = u32::from_le_bytes(bytes.try_into().expect("F32 bytes"));
            if reference.values[at] == f64::NEG_INFINITY {
                if bits != f32::NEG_INFINITY.to_bits() {
                    return Err(format!("empty partial maximum {at} is not -inf: {bits:#x}"));
                }
            } else {
                actual.push(f64::from(f32::from_bits(bits)));
                finite.values.push(reference.values[at]);
                finite.sum_abs_products.push(reference.sum_abs_products[at]);
                finite
                    .reduction_lengths
                    .push(reference.reduction_lengths[at]);
            }
        }
        check_numeric(&actual, &finite, DType::F32)
            .map_err(|failure| format!("finite partial maximum: {failure:?}"))
    }))
}

/// Gate inverse rotation against the BF16 attention values a device run
/// actually stored. The prefix must stay bit-identical; the tail uses the
/// f64 rotation oracle and the same BF16 rounding interval as conformance.
pub fn check_inverse_tail(
    before: &[u16],
    after: &[u16],
    width: usize,
    heads: usize,
    tail: usize,
    cos_sin: &[f32],
) -> LabResult<()> {
    if before.len() != after.len() {
        return Err("inverse rotation output shape mismatch".into());
    }
    let values = before
        .iter()
        .map(|&b| f64::from(bf16_to_f32(b)))
        .collect::<Vec<_>>();
    let reference = kernel_core::latent_attention_oracle::latent_tail_rope(
        &values, width, heads, tail, cos_sin, true,
    )
    .map_err(|e| e.to_string())?;
    for (a, b) in before.chunks_exact(width).zip(after.chunks_exact(width)) {
        if a[..width - tail] != b[..width - tail] {
            return Err("inverse rotation changed the latent prefix".into());
        }
    }
    let actual = after.iter().map(|&b| bf16_to_f32(b)).collect::<Vec<_>>();
    crate::conformance::rounding_admissible(DType::Bf16, &actual, &reference)
}

/// Raw trained-format values with independent code and scale variation.
/// Positive and negative codes exercise cancellation, nibble order and
/// nonuniform scale blocks. No payload quantizer runs in this fixture.
fn pool(format: LatentKvFormat, width: usize, slots: usize, pages: usize) -> (Vec<u8>, usize) {
    let bytes = format.page_bytes(width as u32, slots as u32) as usize;
    let stride = bytes + PAD;
    let mut out = vec![0xa5; pages * stride];
    let fp8 = format == LatentKvFormat::Fp8E4M3Ue8M0Block32;
    let codes = if fp8 { width } else { width / 2 };
    let scales = if fp8 { width / 32 } else { width / 16 };
    for p in 0..pages {
        for row in 0..slots {
            for col in 0..codes {
                let code = if fp8 {
                    (0x38 + ((p + row + col) % 7) as u8)
                        | if (row + col) % 3 == 0 { 128 } else { 0 }
                } else {
                    ((p + row + col * 2) % 16) as u8 | (((p + row + col * 2 + 1) % 16) as u8) << 4
                };
                out[p * stride + row * codes + col] = code;
            }
            for block in 0..scales {
                out[p * stride + slots * codes + row * scales + block] = if fp8 {
                    126 + ((p + row + block) % 4) as u8
                } else {
                    0x30 + ((p + row + block) % 8) as u8
                };
            }
        }
    }
    (out, stride)
}

// Expand the same trained codes into their exact BF16 operands. This is
// fixture storage conversion, not another quantization of the input rows.
fn expanded_pool(
    format: LatentKvFormat,
    width: usize,
    slots: usize,
    pages: usize,
    native: &[u8],
    native_stride: usize,
) -> LabResult<(Vec<u8>, usize)> {
    let stride = slots * width * 2 + PAD + 2;
    let mut out = vec![0xa5; pages * stride];
    for page in 0..pages {
        for slot in 0..slots {
            let values = decode_native_page(
                format,
                width,
                slots,
                &native[page * native_stride..(page + 1) * native_stride],
                slot,
            )
            .map_err(|e| e.to_string())?;
            let bits = values
                .iter()
                .map(|&v| f32_to_bf16(v as f32))
                .collect::<Vec<_>>();
            let at = page * stride + slot * width * 2;
            out[at..at + width * 2].copy_from_slice(&bf16_bytes(&bits));
        }
    }
    Ok((out, stride))
}

pub fn case<'a>(
    query: LatentAttentionQuery,
    capacity: u32,
    prepared: Prepared<LatentSourcesBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    with_queries(query, capacity, prepared, false, false)
}

/// Signed nonzero BF16 queries span several exponents. Every finite QK
/// maximum and partial stays under the native f64 oracle's original gate.
pub fn numerical_case<'a>(
    query: LatentAttentionQuery,
    capacity: u32,
    prepared: Prepared<LatentSourcesBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    with_queries(query, capacity, prepared, true, false)
}

fn with_queries<'a>(
    query: LatentAttentionQuery,
    capacity: u32,
    prepared: Prepared<LatentSourcesBuffers<View>>,
    nonzero: bool,
    page_edges: bool,
) -> LabResult<OpCase<'a>> {
    let LatentAttentionQuery::Attention {
        heads,
        latent,
        max_keys,
        page_size,
        kv,
        sink,
        tail_rope_dims,
        second_fp4_keys,
        output,
        phase,
        ..
    } = query
    else {
        return Err("native latent case requires attention".into());
    };
    let (heads, width, rows, page) = (
        heads as usize,
        latent as usize,
        capacity as usize,
        page_size as usize,
    );
    let bf16 = kv == LatentKvFormat::Bf16;
    let window_budget = if second_fp4_keys {
        max_keys.min(128)
    } else {
        max_keys
    } as usize;
    let compressed_budget = if second_fp4_keys {
        max_keys as usize - window_budget
    } else {
        0
    };
    if !nonzero && window_budget + compressed_budget > 4096 {
        return Err("exact native storage fixture supports at most 4096 keys".into());
    }
    let slots = if nonzero && phase == AttentionPhase::Prefill {
        window_budget + 17
    } else if nonzero {
        window_budget.max(136)
    } else {
        window_budget.max(128)
    };
    let storage = if slots == 128 {
        LatentWindowStorage::Ring128
    } else {
        LatentWindowStorage::Staged {
            slots: slots as u32,
        }
    };
    let pages_per_sequence = if page_edges {
        4
    } else {
        compressed_budget.max(1).div_ceil(page)
    };
    let visible_pages = pages_per_sequence * ITEMS;
    let pages = visible_pages + 1;
    let (window, packed_stride) =
        pool(LatentKvFormat::Fp8E4M3Ue8M0Block32, width, slots, SEQUENCES);
    let window_code_stride =
        slots * width * if bf16 { 2 } else { 1 } + PAD + if bf16 { 2 } else { 0 };
    let window_scale_stride = if bf16 {
        1
    } else {
        slots * (width / 32) + PAD + 1
    };
    let mut window_codes = vec![0xa5; SEQUENCES * window_code_stride];
    let mut window_scales = vec![0xa5; SEQUENCES * window_scale_stride];
    for seq in 0..if bf16 { 0 } else { SEQUENCES } {
        window_codes[seq * window_code_stride..seq * window_code_stride + slots * width]
            .copy_from_slice(&window[seq * packed_stride..seq * packed_stride + slots * width]);
        window_scales[seq * window_scale_stride..seq * window_scale_stride + slots * (width / 32)]
            .copy_from_slice(
                &window[seq * packed_stride + slots * width
                    ..seq * packed_stride + slots * (width + width / 32)],
            );
    }
    if bf16 {
        window_codes = expanded_pool(
            LatentKvFormat::Fp8E4M3Ue8M0Block32,
            width,
            slots,
            SEQUENCES,
            &window,
            packed_stride,
        )?
        .0;
    }
    let (compressed, page_stride) = pool(LatentKvFormat::Fp4E2M1E4M3Block16, width, page, pages);
    let (compressed, page_stride) = if bf16 {
        expanded_pool(
            LatentKvFormat::Fp4E2M1E4M3Block16,
            width,
            page,
            pages,
            &compressed,
            page_stride,
        )?
    } else {
        (compressed, page_stride)
    };
    let mut tables = (0..visible_pages as i32).rev().collect::<Vec<_>>();
    if page_edges {
        for item in 0..ITEMS {
            tables[item * pages_per_sequence + 2] = -1;
        }
    }
    let mut window_keys = vec![-1; rows * window_budget];
    let mut compressed_keys = vec![-1; rows * compressed_budget];
    for row in 0..rows {
        for key in 0..window_budget {
            if (row != 1 || nonzero) && key % 7 != 6 {
                window_keys[row * window_budget + key] = ((slots - 1 + key) % slots) as i32;
            }
        }
        for key in 0..compressed_budget {
            if (row != 1 || nonzero) && key % 5 != 4 {
                compressed_keys[row * compressed_budget + key] =
                    ((key + row * 5) % compressed_budget) as i32;
            }
        }
    }
    if page_edges {
        compressed_keys.fill(-1);
        for row in 0..rows {
            compressed_keys[row * compressed_budget..row * compressed_budget + 4].copy_from_slice(
                &[
                    (page - 1) as i32,
                    page as i32,
                    (2 * page) as i32,
                    (3 * page) as i32,
                ],
            );
        }
    }
    let partial = output == LatentAttentionOutput::Partial;
    let sinks = (0..heads)
        .map(|h| if nonzero { h as f32 * 0.125 } else { 0.0 })
        .collect::<Vec<_>>();
    // Exact quarter turn isolates the inverse operation without an extra
    // rounding error between the attention bound and this first smoke.
    let cos_sin = [0.0_f32, 1.0].repeat(rows * tail_rope_dims as usize / 2);
    let workspace = prepared
        .info()
        .bytes_of("workspace")
        .ok_or("native latent workspace missing")? as usize;
    let queries = if nonzero {
        crate::wide_values(rows * heads * width, 0x128c)
            .into_iter()
            .enumerate()
            .map(|(at, q)| f32_to_bf16(q * [0.015625, 1.0, 8.0, 64.0][(at / width) % 4]))
            .collect()
    } else {
        vec![0; rows * heads * width]
    };
    let roles = LatentSourcesBuffers {
        q: Spec::input(
            DType::Bf16,
            (Rows::Live, heads * width),
            bf16_bytes(&queries),
        ),
        row_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&(0..rows).map(|r| (r % ITEMS) as i32).collect::<Vec<_>>()),
        ),
        window: LatentWindowBuffers {
            codes: Spec::input(
                DType::U8,
                (Rows::Fixed(SEQUENCES), window_code_stride),
                window_codes,
            ),
            scales: Spec::input(
                DType::U8,
                (Rows::Fixed(SEQUENCES), window_scale_stride),
                window_scales,
            ),
            item_slots: Spec::input(DType::I32, (Rows::Fixed(ITEMS), 1), i32_bytes(&[1, -1, 0])),
            items: ITEMS as u32,
            sequences: SEQUENCES as u32,
            code_stride: window_code_stride as u64,
            scale_stride: window_scale_stride as u64,
            storage,
            key_lists: Spec::input(
                DType::I32,
                (Rows::Live, window_budget),
                i32_bytes(&window_keys),
            ),
            key_stride: window_budget as u32,
            budget: window_budget as u32,
        },
        compressed: second_fp4_keys.then(|| LatentCompressedBuffers {
            cache: LatentKvCache {
                pool: Spec::input(DType::U8, (Rows::Fixed(pages), page_stride), compressed),
                pages: pages as u32,
                page_stride: page_stride as u64,
            },
            block_tables: BlockTables {
                tables: if nonzero || page_edges {
                    Spec::input(
                        DType::I32,
                        (Rows::Fixed(ITEMS), pages_per_sequence),
                        i32_bytes(&tables),
                    )
                } else {
                    Spec::ids(
                        (Rows::Fixed(ITEMS), pages_per_sequence),
                        tables,
                        &[("compressed.cache.pool", page_stride)],
                        Holes::Never,
                    )
                },
                sequences: ITEMS as u32,
                max_pages: pages_per_sequence as u32,
            },
            key_lists: Spec::input(
                DType::I32,
                (Rows::Live, compressed_budget),
                i32_bytes(&compressed_keys),
            ),
            key_stride: compressed_budget as u32,
            budget: compressed_budget as u32,
        }),
        sinks: sink.then(|| Spec::input(DType::F32, (Rows::Fixed(1), heads), f32_bytes(&sinks))),
        cos_sin: (!partial && tail_rope_dims != 0).then(|| {
            Spec::input(
                DType::F32,
                (Rows::Live, tail_rope_dims as usize),
                f32_bytes(&cos_sin),
            )
        }),
        output: (!partial).then(|| Spec::output(DType::Bf16, (Rows::Live, heads * width))),
        partial: partial.then(|| LatentPartial {
            max: Spec::output(DType::F32, (Rows::Live, heads)).nonfinite_sentinels(),
            den: Spec::output(DType::F32, (Rows::Live, heads)),
            num: Spec::output(DType::F32, (Rows::Live, heads * width)),
        }),
        workspace: Spec::scratch(workspace),
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let q = images
            .u16s("q")?
            .iter()
            .map(|&v| f64::from(bf16_to_f32(v)))
            .collect::<Vec<_>>();
        let batch = images.i32s("row_batch")?;
        let window_list = images.i32s("window.key_lists")?;
        let item_slots = images.i32s("window.item_slots")?;
        let compressed_list = if second_fp4_keys {
            images.i32s("compressed.key_lists")?
        } else {
            vec![]
        };
        let table = if second_fp4_keys {
            images.i32s("compressed.block_tables.tables")?
        } else {
            vec![]
        };
        let sinks = if sink { images.f32s("sinks")? } else { vec![] };
        let compressed = if second_fp4_keys {
            Some(LatentCompressedInput {
                pool: images.bytes("compressed.cache.pool")?,
                pages,
                paging: LatentPaging {
                    block_tables: &table,
                    max_pages: pages_per_sequence,
                    page_stride,
                },
                lists: LatentKeyLists {
                    indices: &compressed_list,
                    stride: compressed_budget,
                    budget: compressed_budget,
                },
            })
        } else {
            None
        };
        let result = latent_attention_with_sources(
            query,
            LatentSourcesInput {
                q: &q,
                row_batch: &batch,
                window: LatentWindowInput {
                    codes: images.bytes("window.codes")?,
                    scales: images.bytes("window.scales")?,
                    item_slots: &item_slots,
                    sequences: SEQUENCES,
                    code_stride: window_code_stride,
                    scale_stride: window_scale_stride,
                    storage,
                    lists: LatentKeyLists {
                        indices: &window_list,
                        stride: window_budget,
                        budget: window_budget,
                    },
                },
                compressed,
                sinks: sink.then_some(sinks.as_slice()),
            },
        )
        .map_err(|e| e.0.to_string())?;
        Ok(match result {
            LatentAttentionResult::Normalized(mut out) => {
                if tail_rope_dims != 0 {
                    let frequencies = images.f32s("cos_sin")?;
                    for vector in 0..batch.len() * heads {
                        for pair in 0..tail_rope_dims as usize / 2 {
                            let frequency = (vector / heads) * tail_rope_dims as usize + 2 * pair;
                            let (c, s) = (frequencies[frequency], frequencies[frequency + 1]);
                            if c == 1.0 && s == 0.0 {
                                continue;
                            }
                            if c != 0.0 || s != 1.0 {
                                return Err("use the observed BF16 boundary to gate an arbitrary inverse rotation".into());
                            }
                            let at = vector * width + width - tail_rope_dims as usize + pair * 2;
                            out.values.swap(at, at + 1);
                            out.values[at + 1] = if out.values[at + 1] == 0.0 {
                                0.0
                            } else {
                                -out.values[at + 1]
                            };
                            out.sum_abs_products.swap(at, at + 1);
                            out.reduction_lengths.swap(at, at + 1);
                        }
                    }
                }
                let expected = if nonzero {
                    Expected::Values(out)
                } else {
                    // P is exactly one, the sink contributes one, and these
                    // bounded fixed-point values sum exactly in F32. The
                    // division and BF16 conversion are the declared stores.
                    Expected::Bits(bf16_bytes(
                        &out.values
                            .iter()
                            .map(|&v| f32_to_bf16(v as f32))
                            .collect::<Vec<_>>(),
                    ))
                };
                vec![("output".into(), expected)]
            }
            LatentAttentionResult::Partial(p) => {
                if nonzero {
                    vec![
                        (
                            "partial.max".into(),
                            maximum_expected(p.maximum, live as usize * heads),
                        ),
                        ("partial.den".into(), Expected::Values(p.denominator)),
                        ("partial.num".into(), Expected::Values(p.numerator)),
                    ]
                } else {
                    [
                        ("partial.max", p.maximum),
                        ("partial.den", p.denominator),
                        ("partial.num", p.numerator),
                    ]
                    .into_iter()
                    .map(|(name, values)| {
                        (
                            name.into(),
                            Expected::Bits(f32_bytes(
                                &values.values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                            )),
                        )
                    })
                    .collect()
                }
            }
        })
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

pub fn family<'a>(entries: &[Entry]) -> Vec<Required<'a>> {
    family_with_queries(entries, false, false)
}

pub fn numerical_family<'a>(entries: &[Entry]) -> Vec<Required<'a>> {
    family_with_queries(entries, true, false)
}

/// Last row, next page, a missing page and a later valid page, for each item.
pub fn paged_family<'a>(entries: &[Entry], nonzero: bool) -> Vec<Required<'a>> {
    family_with_queries(entries, nonzero, true)
}

pub fn paged_case<'a>(
    query: LatentAttentionQuery,
    capacity: u32,
    prepared: Prepared<LatentSourcesBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    with_queries(query, capacity, prepared, true, true)
}

fn family_with_queries<'a>(
    entries: &[Entry],
    nonzero: bool,
    page_edges: bool,
) -> Vec<Required<'a>> {
    entries
        .iter()
        .map(|entry| {
            let query = entry.query;
            Required::new(
                format!(
                    "native latent {} {}",
                    if nonzero {
                        "nonzero-query"
                    } else {
                        "constant-query"
                    },
                    entry.name
                ),
                Need::rows(CAPACITY),
                move |backend, need| backend.latent_attention_sources(&query, need),
                move |prepared| with_queries(query, CAPACITY, prepared, nonzero, page_edges),
            )
            .required()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernel_core::contract::AttentionPhase;
    use kernels_sm120::backend::Sm120;
    use kernels_sm120::{Profile, ProfileNames};

    #[test]
    fn inverse_tail_gate_requires_the_stored_bf16_boundary() {
        let (width, tail) = (128, 64);
        let mut before = vec![0; width];
        before[width - tail] = f32_to_bf16(1.0 / 3.0);
        before[width - tail + 1] = f32_to_bf16(1.0);
        let mut table = [1.0, 0.0].repeat(tail / 2);
        table[..2].copy_from_slice(&[0.6, 0.8]);
        let mut correct = before.clone();
        correct[width - tail] = f32_to_bf16(1.0);
        correct[width - tail + 1] = f32_to_bf16(0.33203125);
        check_inverse_tail(&before, &correct, width, 1, tail, &table).unwrap();
        let mut skipped = correct.clone();
        skipped[width - tail + 1] = f32_to_bf16(1.0 / 3.0);
        assert!(check_inverse_tail(&before, &skipped, width, 1, tail, &table).is_err());
        correct[0] = f32_to_bf16(1.0);
        assert!(check_inverse_tail(&before, &correct, width, 1, tail, &table).is_err());
    }

    #[test]
    fn native_partial_gate_covers_maximum_score_cancellation() {
        let (width, heads, budget) = (512, 4, 128);
        let query = LatentAttentionQuery::Attention {
            heads: heads as u32,
            latent: width as u32,
            max_keys: budget as u32,
            softmax_scale_bits: (width as f32).powf(-0.5).to_bits(),
            page_size: 16,
            kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
            phase: AttentionPhase::Prefill,
            output: LatentAttentionOutput::Partial,
            sink: true,
            tail_rope_dims: 64,
            second_fp4_keys: false,
            capacity: CAPACITY,
        };
        let profile = Profile::synthetic(ProfileNames {
            name: "native-max-bound",
            tag: "native_max_bound",
        });
        let backend = kernels_sm120::latent::sources::entries(profile, query)
            .unwrap()
            .into_iter()
            .fold(Sm120::new(), Sm120::with_latent_sources);
        let prepared = backend
            .latent_attention_sources(&query, Need::rows(CAPACITY))
            .unwrap();
        let case = numerical_case(query, CAPACITY, prepared).unwrap();
        let mut images = case.images();
        let code_stride = images.bytes("window.codes").unwrap().len() / SEQUENCES;
        let scale_stride = images.bytes("window.scales").unwrap().len() / SEQUENCES;
        let slots = (code_stride - PAD) / width;
        let mut codes = vec![0; code_stride * SEQUENCES];
        let mut scales = vec![0; scale_stride * SEQUENCES];
        for seq in 0..SEQUENCES {
            let start = seq * scale_stride;
            scales[start..start + slots * width / 32].fill(127);
        }
        for at in [0, 128, 256, width + 1] {
            codes[at] = 0x38;
        }
        let mut q = vec![0; CAPACITY as usize * heads * width];
        for head in 0..heads {
            for (d, value) in [(0, 2_f32.powi(24)), (128, 1.0), (256, -2_f32.powi(24))] {
                q[head * width + d] = f32_to_bf16(value);
            }
        }
        let mut ids = vec![-1; CAPACITY as usize * budget];
        ids[..2].copy_from_slice(&[0, 1]);
        images = images
            .with("window.codes", codes)
            .with("window.scales", scales)
            .with("q", bf16_bytes(&q))
            .with("row_batch", i32_bytes(&[0; CAPACITY as usize]))
            .with("window.item_slots", i32_bytes(&[0, 1, -1]))
            .with("window.key_lists", i32_bytes(&ids));
        let mut den = vec![0.0; CAPACITY as usize * heads];
        den[..heads].fill(2.0);
        let mut num = vec![0.0; CAPACITY as usize * heads * width];
        for head in 0..heads {
            for d in [0, 1, 128, 256] {
                num[head * width + d] = 1.0;
            }
        }
        let outputs = std::collections::BTreeMap::from([
            (
                "partial.max".into(),
                f32_bytes(&vec![0.0; CAPACITY as usize * heads]),
            ),
            ("partial.den".into(), f32_bytes(&den)),
            ("partial.num".into(), f32_bytes(&num)),
        ]);
        case.judge_outputs(&images, 1, &outputs).unwrap();
    }

    #[test]
    fn native_partial_oracle_accepts_empty_sentinels_and_rejects_corruption() {
        for kv in [LatentKvFormat::Fp8E4M3Ue8M0Block32, LatentKvFormat::Bf16] {
            for phase in [
                AttentionPhase::Prefill,
                AttentionPhase::Decode,
                AttentionPhase::Verify,
            ] {
                check_partial_sentinels(kv, phase);
            }
        }
    }

    fn check_partial_sentinels(kv: LatentKvFormat, phase: AttentionPhase) {
        let (heads, width) = (16, 512);
        let query = LatentAttentionQuery::Attention {
            heads: heads as u32,
            latent: width as u32,
            max_keys: 128,
            softmax_scale_bits: (width as f32).powf(-0.5).to_bits(),
            page_size: 16,
            kv,
            phase,
            output: LatentAttentionOutput::Partial,
            sink: true,
            tail_rope_dims: 64,
            second_fp4_keys: false,
            capacity: CAPACITY,
        };
        let profile = Profile::synthetic(ProfileNames {
            name: "native-latent-oracle-test",
            tag: "native_latent_oracle_test",
        });
        let entries = kernels_sm120::latent::sources::entries(profile, query).unwrap();
        let backend = entries
            .into_iter()
            .fold(Sm120::new(), Sm120::with_latent_sources);
        let prepared = backend
            .latent_attention_sources(&query, Need::rows(CAPACITY))
            .unwrap();
        let workspace = prepared.info().bytes_of("workspace").unwrap() as usize;
        let op_case = numerical_case(query, CAPACITY, prepared).unwrap();
        let case = &op_case.case;
        let images = case
            .images()
            .with("q", vec![0; CAPACITY as usize * heads * width * 2]);
        let mut result = std::collections::BTreeMap::new();
        for (name, expected) in case.expected(&images, CAPACITY).unwrap() {
            let bytes = if name == "partial.max" {
                let mut maxima = vec![0.0_f32; CAPACITY as usize * heads];
                maxima[heads..2 * heads].fill(f32::NEG_INFINITY);
                f32_bytes(&maxima)
            } else {
                let Expected::Values(values) = expected else {
                    panic!("finite partial oracle")
                };
                f32_bytes(&values.values.iter().map(|&x| x as f32).collect::<Vec<_>>())
            };
            result.insert(name, bytes);
        }
        result.insert("workspace".into(), vec![0; workspace]);
        let judge = |result: &_| {
            case.judge(
                case.expected(&images, CAPACITY).unwrap(),
                &images,
                CAPACITY,
                result,
            )
        };
        case.check_unit_writes(&images, CAPACITY, &result).unwrap();
        judge(&result).unwrap();
        let mut outputs = result.clone();
        outputs.remove("workspace");
        op_case.judge_outputs(&images, CAPACITY, &outputs).unwrap();
        for value in [0.0_f32, f32::INFINITY, f32::NAN] {
            let mut bad = result.clone();
            bad.get_mut("partial.max").unwrap()[heads * 4..heads * 4 + 4]
                .copy_from_slice(&value.to_le_bytes());
            assert!(judge(&bad).is_err());
            assert!(op_case.judge_outputs(&images, CAPACITY, &bad).is_err());
            if !value.is_finite() {
                assert!(case.check_unit_writes(&images, CAPACITY, &bad).is_err());
            }
        }
        let mut bad = result.clone();
        bad.get_mut("partial.max").unwrap()[0..4].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
        assert!(case.check_unit_writes(&images, CAPACITY, &bad).is_err());
        for name in ["partial.den", "partial.num"] {
            let mut bad = result.clone();
            bad.get_mut(name).unwrap()[0..4].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
            assert!(case.check_unit_writes(&images, CAPACITY, &bad).is_err());
        }
        let mut bad = result;
        bad.get_mut("partial.max").unwrap()[0..4].copy_from_slice(&0.25_f32.to_le_bytes());
        assert!(judge(&bad).is_err());
    }
}
