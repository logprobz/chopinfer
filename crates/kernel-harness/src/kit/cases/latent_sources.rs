//! Native latent source conformance. The initial constant-query cases
//! isolate exact storage decoding, one softmax, sink ownership and the
//! inverse tail permutation from probability-quantization error.

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows};
use crate::kit::{OpCase, Required, Spec, bf16_bytes};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::DType;
use kernel_core::latent_attention::{
    BlockTables, LatentAttentionOutput, LatentAttentionQuery, LatentCompressedBuffers,
    LatentKvCache, LatentKvFormat, LatentPartial, LatentSourcesBuffers, LatentWindowBuffers,
    LatentWindowStorage,
};
use kernel_core::latent_attention_oracle::{
    LatentAttentionResult, LatentCompressedInput, LatentKeyLists, LatentPaging, LatentSourcesInput,
    LatentWindowInput, latent_attention_with_sources,
};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernels_sm120::latent::sources::Entry;

pub const CAPACITY: u32 = 4;
const PAD: usize = 64;
const SEQUENCES: usize = 2;

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

pub fn case<'a>(
    query: LatentAttentionQuery,
    capacity: u32,
    prepared: Prepared<LatentSourcesBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let LatentAttentionQuery::Attention {
        heads,
        latent,
        max_keys,
        page_size,
        sink,
        tail_rope_dims,
        second_fp4_keys,
        output,
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
    let window_budget = max_keys.min(128) as usize;
    let compressed_budget = if second_fp4_keys {
        (max_keys as usize - window_budget).min(128)
    } else {
        0
    };
    let pages_per_sequence = 128_usize.div_ceil(page);
    let pages = pages_per_sequence * SEQUENCES;
    let (window, window_stride) = pool(LatentKvFormat::Fp8E4M3Ue8M0Block32, width, 128, SEQUENCES);
    let (compressed, page_stride) = pool(LatentKvFormat::Fp4E2M1E4M3Block16, width, page, pages);
    let tables = (0..pages as i32).rev().collect::<Vec<_>>();
    let mut window_keys = vec![-1; rows * window_budget];
    let mut compressed_keys = vec![-1; rows * compressed_budget];
    for row in 0..rows {
        for key in 0..window_budget {
            if row != 1 && key % 7 != 6 {
                window_keys[row * window_budget + key] = ((127 + key) % 128) as i32;
            }
        }
        for key in 0..compressed_budget {
            if row != 1 && key % 5 != 4 {
                compressed_keys[row * compressed_budget + key] =
                    ((key * 37 + row * 5) % 128) as i32;
            }
        }
    }
    let partial = output == LatentAttentionOutput::Partial;
    let sinks = (0..heads).map(|h| h as f32 * 0.125).collect::<Vec<_>>();
    // Exact quarter turn isolates the inverse operation without an extra
    // rounding error between the attention bound and this first smoke.
    let cos_sin = [0.0_f32, 1.0].repeat(rows * tail_rope_dims as usize / 2);
    let workspace = prepared
        .info()
        .bytes_of("workspace")
        .ok_or("native latent workspace missing")? as usize;
    let roles = LatentSourcesBuffers {
        q: Spec::input(
            DType::Bf16,
            (Rows::Live, heads * width),
            bf16_bytes(&vec![0; rows * heads * width]),
        ),
        row_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&(0..rows).map(|r| (r % 2) as i32).collect::<Vec<_>>()),
        ),
        window: LatentWindowBuffers {
            pool: Spec::input(DType::U8, (Rows::Fixed(SEQUENCES), window_stride), window),
            sequences: SEQUENCES as u32,
            sequence_stride: window_stride as u64,
            storage: LatentWindowStorage::Ring128,
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
                tables: Spec::input(
                    DType::I32,
                    (Rows::Fixed(SEQUENCES), pages_per_sequence),
                    i32_bytes(&tables),
                ),
                sequences: SEQUENCES as u32,
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
            max: Spec::output(DType::F32, (Rows::Live, heads)),
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
                    pool: images.bytes("window.pool")?,
                    sequence_stride: window_stride,
                    storage: LatentWindowStorage::Ring128,
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
                    for vector in 0..batch.len() * heads {
                        for pair in 0..tail_rope_dims as usize / 2 {
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
                vec![("output".into(), Expected::Values(out))]
            }
            LatentAttentionResult::Partial(p) => vec![
                (
                    "partial.max".into(),
                    maximum_expected(p.maximum, live as usize * heads),
                ),
                ("partial.den".into(), Expected::Values(p.denominator)),
                ("partial.num".into(), Expected::Values(p.numerator)),
            ],
        })
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

pub fn family<'a>(entries: &[Entry]) -> Vec<Required<'a>> {
    entries
        .iter()
        .map(|entry| {
            let query = entry.query;
            Required::new(
                format!("native latent constant-query {}", entry.name),
                Need::rows(CAPACITY),
                move |backend, need| backend.latent_attention_sources(&query, need),
                move |prepared| case(query, CAPACITY, prepared),
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
    fn native_partial_oracle_accepts_empty_sentinels_and_rejects_corruption() {
        let query = LatentAttentionQuery::Attention {
            heads: 4,
            latent: 128,
            max_keys: 128,
            softmax_scale_bits: (128_f32).powf(-0.5).to_bits(),
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
        let case = case(query, CAPACITY, prepared).unwrap().case;
        let images = case.images();
        let mut result = std::collections::BTreeMap::new();
        for (name, expected) in case.expected(&images, CAPACITY).unwrap() {
            let bytes = if name == "partial.max" {
                let mut maxima = vec![0.0_f32; 16];
                maxima[4..8].fill(f32::NEG_INFINITY);
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
        judge(&result).unwrap();
        for value in [0.0_f32, f32::INFINITY, f32::NAN] {
            let mut bad = result.clone();
            bad.get_mut("partial.max").unwrap()[16..20].copy_from_slice(&value.to_le_bytes());
            assert!(judge(&bad).is_err());
        }
        let mut bad = result;
        bad.get_mut("partial.max").unwrap()[0..4].copy_from_slice(&0.25_f32.to_le_bytes());
        assert!(judge(&bad).is_err());
    }
}
