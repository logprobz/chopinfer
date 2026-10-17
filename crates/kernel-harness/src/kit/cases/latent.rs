//! Cases of the latent attention family (`kernel_core::latent_attention`):
//! the latent KV write (BF16 and FP8 block-128 caches), latent attention in
//! prefill, decode and verify with a normalized or partial output, and the
//! head-batched GEMMs of the absorbed projections; with the SM120 selection
//! the lab's latent catalog gives and the family's required op set.
//!
//! Rows alternate between two sequences whose pages run in reverse across
//! a pool of padded pages (a stride past the page bytes). The KV write's
//! rows cross a page boundary and its block tables are ID inputs over the
//! pool, so the run also reruns them moved to the top of the pool; its
//! last live row reruns with a hole position, which writes nothing. The
//! attention reads key lists of distinct positions from both pages of its
//! sequence, every seventh entry a hole, through a pool of encoded latent
//! rows. The oracles are `latent_attention_oracle`'s: the KV write as
//! exact bits, the attention and GEMMs within their bounds.
//!
//! The forms SM120 runs in an attention entry's place (the slice-widened
//! FP8 one-split kernel and the column merge of a splitting entry) run the
//! same attention cases ([`form_family`]).

use crate::LabResult;
use crate::conformance::{Expected, Holes, Images, Rows, Write};
use crate::kit::decode::{Compared, DecodeLaunch, Verified};
use crate::kit::{Decoded, OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::poison::WriteMap;
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{AttentionPhase, DType};
use kernel_core::latent_attention::{
    BlockTables, HeadBatchedGemmQuery, HeadGemmBuffers, LATENT_FP8_SCALE_BLOCK,
    LatentAttentionBuffers, LatentAttentionOutput, LatentAttentionQuery, LatentKvCache,
    LatentKvFormat, LatentKvWriteBuffers, LatentPartial,
};
use kernel_core::latent_attention_oracle::{
    self as latent_oracle, LatentAttentionInput, LatentAttentionResult, LatentPaging,
};
use kernels_sm120::latent::{Entry, Query, merge_columns, slices};
use std::collections::HashMap;

/// The rows every latent op serves.
pub const CAPACITY: u32 = super::ROW_CAPACITY;

/// Padding of each pool page past its bytes, so a launch that assumes
/// packed pages reads the wrong tokens.
const PAGE_PAD: u64 = 256;

/// Entries of each key list a launch reads.
const BUDGET: u32 = 48;

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// A pool of two sequences of `max_pages` pages each, reversed across it,
/// and a spare page.
struct Pool {
    max_pages: usize,
    pages: usize,
    stride: usize,
    tables: Vec<i32>,
}

impl Pool {
    fn new(kv: LatentKvFormat, latent: u32, page_size: u32, max_pages: usize) -> Self {
        let pages = 2 * max_pages + 1;
        let stride = (kv.page_bytes(latent, page_size) + PAGE_PAD) as usize;
        Self {
            max_pages,
            pages,
            stride,
            tables: (0..2 * max_pages)
                .map(|at| (2 * max_pages - 1 - at) as i32)
                .collect(),
        }
    }

    fn block_tables<'a>(&self, tables: Spec<'a>) -> BlockTables<Spec<'a>> {
        BlockTables {
            tables,
            sequences: 2,
            max_pages: self.max_pages as u32,
        }
    }

    fn paging<'t>(&self, tables: &'t [i32]) -> LatentPaging<'t> {
        LatentPaging {
            block_tables: tables,
            max_pages: self.max_pages,
            page_stride: self.stride,
        }
    }
}

/// The byte spans the KV write stores for a token at `slot` of a page
/// starting at byte `base`: its row and, for FP8, its scales.
fn spans(
    kv: LatentKvFormat,
    latent: usize,
    page_size: usize,
    base: usize,
    slot: usize,
) -> Vec<std::ops::Range<usize>> {
    match kv {
        LatentKvFormat::Fp8E4M3Ue8M0Block32 | LatentKvFormat::Fp4E2M1E4M3Block16 => {
            unreachable!("native latent formats need their own conformance cases")
        }
        LatentKvFormat::Bf16 => {
            std::iter::once(base + slot * latent * 2..base + (slot + 1) * latent * 2).collect()
        }
        LatentKvFormat::Fp8E4M3Block128 => {
            let blocks = latent / LATENT_FP8_SCALE_BLOCK as usize;
            let scales = base + page_size * latent + slot * blocks * 4;
            vec![
                base + slot * latent..base + (slot + 1) * latent,
                scales..scales + blocks * 4,
            ]
        }
    }
}

/// The element a KV write's pool is checked in: an element the write
/// leaves all 0xff bytes, the poison, is unwritten, so it must be one no
/// written value can be. A BF16 row's elements are BF16 (0xffff is NaN);
/// an FP8 page's codes and F32 scales are checked four bytes at a time
/// (four NaN codes, or a NaN scale). Single bytes would read a written
/// BF16 or scale byte of 0xff as unwritten.
fn pool_element(kv: LatentKvFormat) -> DType {
    match kv {
        LatentKvFormat::Bf16 => DType::Bf16,
        LatentKvFormat::Fp8E4M3Block128 => DType::F32,
        LatentKvFormat::Fp8E4M3Ue8M0Block32 | LatentKvFormat::Fp4E2M1E4M3Block16 => {
            unreachable!("native latent formats need their own conformance cases")
        }
    }
}

/// The case of a latent KV write over `capacity` rows of two sequences,
/// the first crossing a page.
pub fn kv_write_case<'a>(
    query: LatentAttentionQuery,
    capacity: u32,
    prepared: Prepared<LatentKvWriteBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let LatentAttentionQuery::KvWrite {
        latent,
        page_size,
        kv,
        ..
    } = query
    else {
        return Err("a latent KV write case writes by a KV write query".into());
    };
    if !matches!(kv, LatentKvFormat::Bf16 | LatentKvFormat::Fp8E4M3Block128) {
        return Err("native latent formats require the source-aware conformance case".into());
    }
    let (width, page) = (latent as usize, page_size as usize);
    let rows = capacity as usize;
    let bases = [page.saturating_sub(3), 2 * page + 1];
    let placed = (0..rows)
        .map(|row| (row % 2, bases[row % 2] + row / 2))
        .collect::<Vec<_>>();
    let last = placed.iter().map(|&(_, p)| p).max().unwrap_or(0);
    let pool = Pool::new(kv, latent, page_size, last / page + 2);
    let stride = pool.stride;
    let element = pool_element(kv);
    let element_bytes = WriteMap::of(element).bytes()?;
    if stride % element_bytes != 0 {
        return Err(format!(
            "the pool stride {stride} is not whole {element:?} elements"
        ));
    }
    let map_pool = Pool {
        tables: pool.tables.clone(),
        ..pool
    };
    let written = move |images: &Images, live: u32| -> LabResult<Vec<std::ops::Range<usize>>> {
        let tables = images.i32s("block_tables.tables")?;
        let batch = images.i32s("row_batch")?;
        let positions = images.i32s("positions")?;
        let mut ranges = Vec::new();
        for row in 0..live as usize {
            let Ok(position) = usize::try_from(positions[row]) else {
                continue;
            };
            let page_of = tables[batch[row] as usize * map_pool.max_pages + position / page];
            ranges.extend(spans(
                kv,
                width,
                page,
                page_of as usize * map_pool.stride,
                position % page,
            ));
        }
        Ok(ranges)
    };
    let roles = LatentKvWriteBuffers {
        latent: Spec::input(
            DType::Bf16,
            (Rows::Live, width),
            bf16_bytes(&bf16_values(31, rows * width)),
        ),
        cache: LatentKvCache {
            pool: Spec::state(
                element,
                (Rows::Fixed(pool.pages), stride / element_bytes),
                (0..pool.pages * stride)
                    .map(|at| (at * 7 + 3) as u8)
                    .collect(),
                Write::Map(Box::new(move |images, live| {
                    WriteMap::of(element).byte_ranges(written(images, live)?)
                })),
            ),
            pages: pool.pages as u32,
            page_stride: stride as u64,
        },
        block_tables: pool.block_tables(Spec::ids(
            (Rows::Fixed(2), pool.max_pages),
            pool.tables.clone(),
            &[("cache.pool", stride)],
            Holes::Never,
        )),
        row_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&placed.iter().map(|&(s, _)| s as i32).collect::<Vec<_>>()),
        ),
        positions: Spec::ids(
            (Rows::Live, 1),
            placed.iter().map(|&(_, p)| p as i32).collect(),
            &[],
            Holes::LastLive,
        ),
        live_rows: Spec::live(),
    };
    let max_pages = pool.max_pages;
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let mut bytes = images.bytes("cache.pool")?.to_vec();
        if live > 0 {
            let live = live as usize;
            let tables = images.i32s("block_tables.tables")?;
            latent_oracle::latent_kv_write(
                query,
                &images.u16s("latent")?[..live * width],
                &images.i32s("row_batch")?[..live],
                &images.i32s("positions")?[..live],
                LatentPaging {
                    block_tables: &tables,
                    max_pages,
                    page_stride: stride,
                },
                &mut bytes,
            )
            .map_err(|e| e.0.to_string())?;
        }
        Ok(vec![("cache.pool".to_owned(), Expected::Bits(bytes))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The encoded pool of an attention case over sequences of `tokens`:
/// every page of both holds encoded latent rows of its tokens; the spare
/// page and the padding hold a pattern. Pools are kept by their shape, as
/// FP8 rows take milliseconds each to encode and every case of a budget
/// shares one.
fn pool_image(
    pool: &Pool,
    kv: LatentKvFormat,
    latent: u32,
    page_size: u32,
    tokens: [usize; 2],
) -> LabResult<Vec<u8>> {
    type Key = (LatentKvFormat, u32, u32, [usize; 2]);
    static POOLS: std::sync::OnceLock<std::sync::Mutex<HashMap<Key, Vec<u8>>>> =
        std::sync::OnceLock::new();
    let key = (kv, latent, page_size, tokens);
    let pools = POOLS.get_or_init(Default::default);
    if let Some(image) = pools.lock().map_err(|e| e.to_string())?.get(&key) {
        return Ok(image.clone());
    }
    let width = latent as usize;
    let mut image = (0..pool.pages * pool.stride)
        .map(|at| (at * 11 + 5) as u8)
        .collect::<Vec<_>>();
    for (sequence, &count) in tokens.iter().enumerate() {
        let values = bf16_values(32 + sequence as u64, count * width);
        let positions = (0..count as i32).collect::<Vec<_>>();
        latent_oracle::latent_kv_write(
            LatentAttentionQuery::KvWrite {
                latent,
                page_size,
                kv,
                capacity: count as u32,
            },
            &values,
            &vec![sequence as i32; count],
            &positions,
            pool.paging(&pool.tables),
            &mut image,
        )
        .map_err(|e| e.0.to_string())?;
    }
    pools
        .lock()
        .map_err(|e| e.to_string())?
        .insert(key, image.clone());
    Ok(image)
}

/// A stride coprime to `tokens` for listing distinct positions.
fn coprime_stride(tokens: usize) -> usize {
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    (37..)
        .find(|&s| gcd(s, tokens) == 1)
        .expect("a coprime stride")
}

/// The case of a latent attention over `capacity` rows of two sequences of
/// `2 page + 11` and `3 page - 13` tokens, each row listing [`BUDGET`]
/// distinct positions of its sequence (every seventh a hole) in a shuffled
/// order, with the rest of its list holes the launch does not read.
pub fn attention_case<'a>(
    query: LatentAttentionQuery,
    capacity: u32,
    prepared: Prepared<LatentAttentionBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    attention_case_with(query, capacity, BUDGET, prepared)
}

/// [`attention_case`] with each row listing `budget` entries (at most the
/// query's keys): the second sequence holds at least `budget` tokens, and a
/// list longer than the first sequence's tokens repeats its positions.
pub fn attention_case_with<'a>(
    query: LatentAttentionQuery,
    capacity: u32,
    budget: u32,
    prepared: Prepared<LatentAttentionBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let LatentAttentionQuery::Attention {
        heads,
        latent,
        max_keys,
        page_size,
        kv,
        output,
        ..
    } = query
    else {
        return Err("a latent attention case attends by an attention query".into());
    };
    let (width, page, heads) = (latent as usize, page_size as usize, heads as usize);
    let rows = capacity as usize;
    let budget = budget.min(max_keys);
    let tokens = [
        2 * page + 11,
        (3 * page).saturating_sub(13).max(budget as usize),
    ];
    let pool = Pool::new(kv, latent, page_size, tokens[1].div_ceil(page));
    let image = pool_image(&pool, kv, latent, page_size, tokens)?;
    let mut lists = vec![-1_i32; rows * max_keys as usize];
    for row in 0..rows {
        let count = tokens[row % 2];
        let stride = coprime_stride(count);
        for entry in 0..budget as usize {
            lists[row * max_keys as usize + entry] = if entry % 7 == 6 {
                -1
            } else {
                ((entry * stride + row * 5) % count) as i32
            };
        }
    }
    let workspace = prepared.info().bytes_of("workspace").unwrap_or(0) as usize;
    let partial = output == LatentAttentionOutput::Partial;
    let roles = LatentAttentionBuffers {
        q: Spec::input(
            DType::Bf16,
            (Rows::Live, heads * width),
            bf16_bytes(&bf16_values(33, rows * heads * width)),
        ),
        cache: LatentKvCache {
            pool: Spec::input(DType::U8, (Rows::Fixed(pool.pages), pool.stride), image),
            pages: pool.pages as u32,
            page_stride: pool.stride as u64,
        },
        block_tables: pool.block_tables(Spec::input(
            DType::I32,
            (Rows::Fixed(2), pool.max_pages),
            i32_bytes(&pool.tables),
        )),
        row_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&(0..rows).map(|row| (row % 2) as i32).collect::<Vec<_>>()),
        ),
        key_lists: Spec::input(
            DType::I32,
            (Rows::Live, max_keys as usize),
            i32_bytes(&lists),
        ),
        budget,
        output: (!partial).then(|| Spec::output(DType::Bf16, (Rows::Live, heads * width))),
        partial: partial.then(|| LatentPartial {
            max: Spec::output(DType::F32, (Rows::Live, heads)),
            den: Spec::output(DType::F32, (Rows::Live, heads)),
            num: Spec::output(DType::F32, (Rows::Live, heads * width)),
        }),
        workspace: if workspace > 0 {
            Spec::scratch(workspace)
        } else {
            Spec::untouched(16)
        },
        live_rows: Spec::live(),
    };
    let (max_pages, stride) = (pool.max_pages, pool.stride);
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let tables = images.i32s("block_tables.tables")?;
        let result = latent_oracle::latent_attention(
            query,
            LatentAttentionInput {
                q: &widen(&images.u16s("q")?),
                row_batch: &images.i32s("row_batch")?,
                key_lists: &images.i32s("key_lists")?,
                budget: budget as usize,
                paging: LatentPaging {
                    block_tables: &tables,
                    max_pages,
                    page_stride: stride,
                },
                pool: images.bytes("cache.pool")?,
            },
        )
        .map_err(|e| e.0.to_string())?;
        Ok(match result {
            LatentAttentionResult::Normalized(output) => {
                vec![("output".to_owned(), Expected::Values(output))]
            }
            LatentAttentionResult::Partial(partials) => vec![
                ("partial.max".to_owned(), Expected::Values(partials.maximum)),
                (
                    "partial.den".to_owned(),
                    Expected::Values(partials.denominator),
                ),
                (
                    "partial.num".to_owned(),
                    Expected::Values(partials.numerator),
                ),
            ],
        })
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// Padding of each head's weight past its `n * k` elements.
const HEAD_PAD: usize = 64;

/// The case of a head-batched GEMM over `capacity` rows: each head's
/// weight padded past its elements, so a launch that assumes packed heads
/// reads the wrong weights.
pub fn head_gemm_case<'a>(
    query: HeadBatchedGemmQuery,
    capacity: u32,
    prepared: Prepared<HeadGemmBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let (heads, n, k) = (query.heads as usize, query.n as usize, query.k as usize);
    let rows = capacity as usize;
    let stride = n * k + HEAD_PAD;
    let weights = bf16_values(34, heads * stride);
    let decoded = widen(&weights);
    let roles = HeadGemmBuffers {
        x: Spec::input(
            DType::Bf16,
            (Rows::Live, heads * k),
            bf16_bytes(&bf16_values(35, rows * heads * k)),
        ),
        weight: Spec::weight(bf16_bytes(&weights)),
        head_stride: stride as u64,
        y: Spec::output(query.output, (Rows::Live, heads * n)),
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let x = widen(&images.u16s("x")?);
        let output = latent_oracle::head_batched_gemm(query, &x, &decoded, stride, rows)
            .map_err(|e| e.0.to_string())?;
        Ok(vec![("y".to_owned(), Expected::Values(output))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// SM120's selection of the latent entries `catalog`, one per query.
pub fn sm120_entries(catalog: Vec<Entry>) -> Vec<Entry> {
    let mut chosen: Vec<Entry> = Vec::new();
    for entry in catalog {
        if !chosen.iter().any(|other| other.query == entry.query) {
            chosen.push(entry);
        }
    }
    chosen
}

/// The family's required ops: one per entry of `entries`, over
/// [`CAPACITY`] rows.
pub fn family<'a>(entries: &[Entry]) -> Vec<Required<'a>> {
    let need = Need::rows(CAPACITY);
    entries
        .iter()
        .map(|entry| match entry.query {
            Query::Latent(query @ LatentAttentionQuery::KvWrite { latent, kv, .. }) => {
                Required::new(
                    format!("latent kv write {kv:?} latent {latent} rows {CAPACITY}"),
                    need,
                    move |backend, need| backend.latent_kv_write(&query, need),
                    move |prepared| kv_write_case(query, CAPACITY, prepared),
                )
            }
            Query::Latent(
                query @ LatentAttentionQuery::Attention {
                    kv, phase, output, ..
                },
            ) => with_decode(
                Required::new(
                    format!("latent attention {kv:?} {phase:?} {output:?} rows {CAPACITY}"),
                    need,
                    move |backend, need| backend.latent_attention(&query, need),
                    move |prepared| verify_rows(query, attention_case(query, CAPACITY, prepared)),
                ),
                query,
                entries,
            ),
            Query::HeadGemm(query) => Required::new(
                format!(
                    "head gemm heads {} {}x{} {:?} {:?} rows {CAPACITY}",
                    query.heads, query.n, query.k, query.layout, query.output
                ),
                need,
                move |backend, need| backend.head_batched_gemm(&query, need),
                move |prepared| head_gemm_case(query, CAPACITY, prepared),
            ),
        })
        .collect()
}

/// SM120 over `entries` with the forms it runs in their places: the
/// slice-widened forms of the FP8 one-split attention entries and the
/// column merges of the splitting ones (`latent::slices`,
/// `latent::merge_columns`), which the lab builds for its catalog.
pub fn sm120_with_forms(entries: &[Entry]) -> kernels_sm120::backend::Sm120 {
    let backend = entries
        .iter()
        .cloned()
        .fold(kernels_sm120::backend::Sm120::new(), |b, e| {
            b.with_latent(e)
        });
    let backend = slices::entries(entries)
        .into_iter()
        .fold(backend, |b, form| b.with_latent_slices(form));
    merge_columns::entries(entries)
        .into_iter()
        .fold(backend, |b, form| b.with_latent_merge_columns(form))
}

/// The attention ops of `entries` that SM120 runs through a form (a
/// slice-widened kernel or a column merge), over [`CAPACITY`] rows, each
/// required. A case is declared only when the prepared op launches the
/// form, so a backend that ran the entry in its place fails the run.
pub fn form_family<'a>(entries: &[Entry]) -> Vec<Required<'a>> {
    let need = Need::rows(CAPACITY);
    let mut ops = Vec::new();
    for entry in entries {
        let Query::Latent(
            query @ LatentAttentionQuery::Attention {
                kv, phase, output, ..
            },
        ) = entry.query
        else {
            continue;
        };
        let one = std::slice::from_ref(entry);
        let form = match (
            slices::entries(one).pop(),
            merge_columns::entries(one).pop(),
        ) {
            (Some(form), _) => ("slices", form.symbol()),
            (None, Some(form)) => ("column merge", form.symbol()),
            (None, None) => continue,
        };
        let (name, symbol) = form;
        let required = Required::new(
            format!("latent attention {kv:?} {phase:?} {output:?} rows {CAPACITY} {name}"),
            need,
            move |backend, need| backend.latent_attention(&query, need),
            move |prepared| {
                if !prepared
                    .info()
                    .symbols
                    .iter()
                    .any(|(launched, _)| *launched == symbol)
                {
                    return Err(format!("the op does not launch {symbol}"));
                }
                verify_rows(query, attention_case(query, CAPACITY, prepared))
            },
        )
        .required();
        ops.push(with_decode(required, query, entries));
    }
    ops
}

/// Why a verify attention case's rows check against decode launches: a
/// verify row's one-row launch is the decode op's of the same query.
const VERIFY_ROWS: &str =
    "a verify row's one-row launch is the decode op's of the same query, not the verify op's";

/// A verify attention `case` with its rows checked against decode launches
/// ([`attention_decoded`]); any other case unchanged.
fn verify_rows<'a>(
    query: LatentAttentionQuery,
    case: LabResult<OpCase<'a>>,
) -> LabResult<OpCase<'a>> {
    match query {
        LatentAttentionQuery::Attention {
            phase: AttentionPhase::Verify,
            ..
        } => Ok(case?.item_rows(VERIFY_ROWS)),
        _ => case,
    }
}

/// `required` with the decode mapping of a verify attention `query`: the
/// decode entry among `entries` of the same cache, output and heads.
fn with_decode<'a>(
    required: Required<'a>,
    query: LatentAttentionQuery,
    entries: &[Entry],
) -> Required<'a> {
    let LatentAttentionQuery::Attention {
        phase: AttentionPhase::Verify,
        kv,
        output,
        ..
    } = query
    else {
        return required;
    };
    let decode = entries.iter().find_map(|entry| match entry.query {
        Query::Latent(
            decode @ LatentAttentionQuery::Attention {
                phase: AttentionPhase::Decode,
                kv: k,
                output: o,
                ..
            },
        ) if k == kv && o == output => Some(decode),
        _ => None,
    });
    required.with_decode(move |backend| {
        let Some(decode) = decode else {
            return Ok(Err("no decode entry of the verify query".into()));
        };
        backend
            .latent_attention(&decode, Need::rows(CAPACITY))
            .map(|prepared| attention_decoded(query, decode, prepared))
    })
}

/// How a verify attention `query`'s rows map to one-row launches of the
/// decode attention `decode` over `prepared` (`kit::decode`): each verify
/// row is its own query row (its sequence, key list and query), so the
/// verify step runs every case row, and row `r` runs alone as the decode
/// op's row 0 over the same pool, whose output row must equal the verify
/// step's row `r`.
fn attention_decoded<'a>(
    query: LatentAttentionQuery,
    decode: LatentAttentionQuery,
    prepared: Prepared<LatentAttentionBuffers<View>>,
) -> LabResult<Decoded<'a>> {
    let (
        LatentAttentionQuery::Attention {
            heads,
            latent,
            max_keys,
            output,
            ..
        },
        LatentAttentionQuery::Attention {
            max_keys: decode_keys,
            ..
        },
    ) = (query, decode)
    else {
        return Err("a verify attention maps to a decode attention".into());
    };
    if max_keys != decode_keys {
        return Err("the decode attention lists keys of another width".into());
    }
    let case = attention_case(decode, CAPACITY, prepared)?;
    let (heads, width) = (heads as usize, latent as usize);
    let inputs = [
        ("q", heads * width * 2),
        ("row_batch", 4),
        ("key_lists", max_keys as usize * 4),
    ];
    let outputs = match output {
        LatentAttentionOutput::Normalized => vec![("output", heads * width * 2)],
        LatentAttentionOutput::Partial => vec![
            ("partial.max", heads * 4),
            ("partial.den", heads * 4),
            ("partial.num", heads * width * 4),
        ],
    };
    let launches = move |verified: &Verified<'_>, base: &Images| -> LabResult<Vec<DecodeLaunch>> {
        let images = verified.images;
        (0..verified.live as usize)
            .map(|row| {
                let mut decode = base.clone();
                for (path, bytes) in inputs {
                    let mut image = base.bytes(path)?.to_vec();
                    image[..bytes].copy_from_slice(&images.bytes(path)?[row * bytes..][..bytes]);
                    decode = decode.with(path, image);
                }
                Ok(DecodeLaunch {
                    label: format!("verify row {row}"),
                    images: decode,
                    compared: outputs
                        .iter()
                        .map(|&(path, bytes)| Compared {
                            verify: (path.into(), row * bytes..(row + 1) * bytes),
                            decode: (path.into(), 0..bytes),
                        })
                        .collect(),
                })
            })
            .collect()
    };
    Ok(Decoded::new(
        case,
        |images: &Images, _| Ok(images.clone()),
        launches,
    ))
}

/// Self-checks of the family's cases over a consumer's latent entries (as
/// [`sm120_entries`] selects them), for its tests: each panics naming the
/// first failure. kernel-lab runs them over its latent lab catalog.
pub mod checks {
    use super::*;

    /// The pool's written elements never read as poison: the oracle's
    /// written BF16 rows hold 0xff bytes, which a byte-wise check takes for
    /// unwritten (GPU: every BF16 write failed so), but no element of the
    /// pool's element size is all 0xff.
    pub fn written_pool_elements_never_read_as_poison(entries: &[Entry]) {
        let backend = entries
            .iter()
            .cloned()
            .fold(kernels_sm120::backend::Sm120::new(), |b, e| {
                b.with_latent(e)
            });
        for required in family(entries) {
            let Some(kv) = [LatentKvFormat::Bf16, LatentKvFormat::Fp8E4M3Block128]
                .into_iter()
                .find(|kv| {
                    required
                        .name
                        .starts_with(&format!("latent kv write {kv:?} "))
                })
            else {
                continue;
            };
            let case = super::super::testing::prepared(&backend, &required);
            let images = case.case.images();
            let Some(Query::Latent(query)) =
                entries
                    .iter()
                    .map(|entry| entry.query)
                    .find(|query| match query {
                        Query::Latent(LatentAttentionQuery::KvWrite { latent, kv: k, .. }) => {
                            *k == kv && required.name.contains(&format!("latent {latent} "))
                        }
                        _ => false,
                    })
            else {
                panic!("{}", required.name);
            };
            let LatentAttentionQuery::KvWrite { latent, .. } = query else {
                unreachable!()
            };
            let mut pool = images.bytes("cache.pool").unwrap().to_vec();
            let tables = images.i32s("block_tables.tables").unwrap();
            let max_pages = tables.len() / 2;
            let page_stride = case
                .leaves
                .iter()
                .find(|leaf| leaf.path == "cache.pool")
                .unwrap()
                .row_bytes;
            latent_oracle::latent_kv_write(
                query,
                &images.u16s("latent").unwrap(),
                &images.i32s("row_batch").unwrap(),
                &images.i32s("positions").unwrap(),
                LatentPaging {
                    block_tables: &tables,
                    max_pages,
                    page_stride,
                },
                &mut pool,
            )
            .unwrap();
            let written = case
                .case
                .state_writes("cache.pool", &images, CAPACITY)
                .unwrap();
            let element = WriteMap::of(pool_element(kv)).bytes().unwrap();
            let bytes = written.iter().cloned().flatten().collect::<Vec<_>>();
            // BF16 rows hold written 0xff bytes; the FP8 data happens not
            // to, which is why its byte-wise check passed.
            assert_eq!(
                bytes.iter().any(|&at| pool[at] == 0xff),
                kv == LatentKvFormat::Bf16,
                "{}",
                required.name
            );
            for range in &written {
                for at in range.clone().step_by(element) {
                    assert!(
                        pool[at..at + element].iter().any(|&byte| byte != 0xff),
                        "{} latent {latent}: written element at byte {at} reads as poison",
                        required.name
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_lists_hold_distinct_positions() {
        for count in [139, 179, 75] {
            let stride = coprime_stride(count);
            let listed = (0..count)
                .map(|entry| (entry * stride + 3) % count)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(listed.len(), count);
        }
    }
}
