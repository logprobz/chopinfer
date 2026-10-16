//! Cases of the gated and sparse attention family, with the SM120 selection
//! the lab's catalogs give (`attention::gated::catalog`,
//! `attention::sparse::catalog`), whose entries the lab's attention cubins
//! hold.
//!
//! The output gate compares within the contract oracle's tolerance
//! (`gated_attention_oracle::attention_output_gate`: one F32 sigmoid and
//! product per element, rounded to BF16 once). The gated KV write compares
//! its Q rows and both cache planes, BF16 or FP8 E4M3 in the cache's code
//! units, within the tolerance of `qk_norm_rope_kv_write` over the table the
//! case binds: two items whose rows cross a page, through permuted block
//! tables, with text and M-RoPE positions. The sparse indexer's query norm
//! and M-RoPE compares within the tolerance of
//! `sparse_attention_oracle::indexer_query` over the bound table.

use super::super::{OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::poison::WriteMap;
use kernel_core::backend::{Counts, Need, Prepared, View};
use kernel_core::codec::{SmallFloat, bf16_to_f32};
use kernel_core::contract::{DType, GatedAttentionQuery, SparseAttentionQuery};
use kernel_core::gated_attention_oracle::{
    GatedAttentionInputs, PagedKvPair, attention_output_gate, mrope_axis, qk_norm_rope_kv_write,
};
use kernel_core::gated_sparse::{
    FoldInputs, GatedKvWriteBuffers, GatedKvWriteOp, IndexQueryBuffers, KvCache, OutputGateBuffers,
    OutputGateOp,
};
use kernel_core::oracle::OracleOutput;
use kernel_core::sparse_attention_oracle::indexer_query;
use kernels_sm120::Catalogs;
use kernels_sm120::attention::{gated, sparse};
use kernels_sm120::backend::{GatedSparseEntries, Sm120};

/// The token rows every gated and sparse case serves.
pub const CAPACITY: u32 = 16;

fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// The case of an output gate over `capacity` rows: BF16 attention rows
/// and packed rows (each Q head's query then its gate logits, then the K
/// and V heads) from the kit's generator, and the contract's
/// `attention * sigmoid(gate)` within tolerance.
pub fn output_gate_case<'a>(
    op: OutputGateOp,
    capacity: u32,
    prepared: Prepared<OutputGateBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let GatedAttentionQuery::OutputGate {
        heads, head_dim, ..
    } = op.query
    else {
        return Err("an output gate case needs an output gate".into());
    };
    let (heads, dim, kv) = (heads as usize, head_dim as usize, op.kv_heads as usize);
    let rows = capacity as usize;
    let packed = 2 * (heads + kv) * dim;
    let roles = OutputGateBuffers {
        attention: Spec::input(
            DType::Bf16,
            (Rows::Live, heads * dim),
            bf16_bytes(&bf16_values(41, rows * heads * dim)),
        ),
        qkv: Spec::input(
            DType::Bf16,
            (Rows::Live, packed),
            bf16_bytes(&bf16_values(42, rows * packed)),
        ),
        output: Spec::output(DType::Bf16, (Rows::Live, heads * dim)),
        live_rows: Spec::live(),
    };
    let query = GatedAttentionQuery::OutputGate {
        heads: heads as u32,
        head_dim: dim as u32,
        capacity,
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let attention = widen(&images.u16s("attention")?);
        let qkv = widen(&images.u16s("qkv")?);
        // Each row's gates: the second half of each Q head.
        let gate = qkv
            .chunks_exact(packed)
            .flat_map(|row| {
                row[..2 * heads * dim]
                    .chunks_exact(2 * dim)
                    .flat_map(|head| head[dim..].to_vec())
            })
            .collect::<Vec<_>>();
        let output =
            attention_output_gate(query, &attention, &gate).map_err(|e| e.0.to_string())?;
        Ok(vec![("output".into(), Expected::Values(output))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required output gate `op`.
pub fn output_gate<'a>(name: String, op: OutputGateOp, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.output_gate(&op, need),
        move |prepared| output_gate_case(op, need.capacity, prepared),
    )
}

/// The items of a gated KV write case.
const ITEMS: usize = 2;

/// The pages of one item's block table row in a gated KV write case.
const ITEM_PAGES: usize = 4;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// An F32 cos/sin table of `rows` positions of `rotary / 2` pairs:
/// distinct angles per position and pair.
fn cos_sin_table(rows: usize, rotary: usize) -> Vec<f32> {
    (0..rows * rotary / 2)
        .flat_map(|at| {
            let angle = (at % (rotary / 2)) as f32 * 0.37 + (at / (rotary / 2)) as f32 * 0.11;
            [angle.cos(), angle.sin()]
        })
        .collect()
}

/// Each row's cos and sin per rotary pair from `table`, at the position of
/// the pair's M-RoPE axis (`mrope` sections) among the row's `axes`.
fn pair_tables(
    table: &[f32],
    rotary: usize,
    mrope: [u32; 3],
    axes: &[[usize; 3]],
) -> Vec<Vec<(f64, f64)>> {
    axes.iter()
        .map(|axes| {
            (0..rotary / 2)
                .map(|pair| {
                    let at = (axes[mrope_axis(pair, mrope)] * rotary / 2 + pair) * 2;
                    (f64::from(table[at]), f64::from(table[at + 1]))
                })
                .collect()
        })
        .collect()
}

/// A gated KV write's rows: row `r` belongs to item `r % 2` at position
/// three before item 0's second page, or one into item 1's third page,
/// plus `r / 2`; even rows are text (equal T, H and W), odd rows M-RoPE
/// positions `(p + 5, p + 1, p + 2)`, whose T differs from the cache
/// position, so a write placed by a RoPE axis lands elsewhere.
fn kv_rows(rows: usize, page: usize) -> Vec<(usize, usize, [usize; 3])> {
    (0..rows)
        .map(|row| {
            let item = row % ITEMS;
            let position = [page - 3, 2 * page + 1][item] + row / ITEMS;
            let axes = if row % 2 == 0 {
                [position; 3]
            } else {
                [position + 5, position + 1, position + 2]
            };
            (item, position, axes)
        })
        .collect()
}

/// The items' block table rows of `max_pages` entries: each item's pages
/// on distinct slots, none at its own index among the items' pages
/// (`(i * 5 + 3) % slots` for page index `i`), so a write that ignores the
/// table or the row's item lands elsewhere; unused entries -1.
fn kv_tables(max_pages: usize) -> Vec<i32> {
    let slots = ITEMS * ITEM_PAGES;
    let mut tables = vec![-1; ITEMS * max_pages];
    for item in 0..ITEMS {
        for page in 0..ITEM_PAGES {
            tables[item * max_pages + page] = (((item * ITEM_PAGES + page) * 5 + 3) % slots) as i32;
        }
    }
    tables
}

/// The first cache element of a row at `position` of `item` through
/// `tables`, in pages of `page` token rows of `row` elements.
fn kv_slot(
    tables: &[i32],
    max_pages: usize,
    (page, row): (usize, usize),
    (item, position): (usize, usize),
) -> usize {
    tables[item * max_pages + position / page] as usize * page * row + position % page * row
}

/// The case of a gated KV write over `capacity` rows (see the module
/// docs): BF16 packed rows and Gemma norm weights from the kit's generator,
/// an F32 cos/sin table of `max_context + 6` positions, and the cache planes
/// in place, each written at its live rows' slots and kept elsewhere.
pub fn gated_kv_write_case<'a>(
    op: GatedKvWriteOp,
    capacity: u32,
    prepared: Prepared<GatedKvWriteBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let GatedAttentionQuery::QkNormRopeKvWrite {
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        mrope,
        page_size,
        kv_dtype,
        ..
    } = op.query
    else {
        return Err("a gated KV write case needs a gated KV write".into());
    };
    let (qh, kh, d, rd, page) = (
        q_heads as usize,
        kv_heads as usize,
        head_dim as usize,
        rotary_dim as usize,
        page_size as usize,
    );
    let max_pages = op.max_context.div_ceil(page_size) as usize;
    if max_pages < ITEM_PAGES || page < 4 {
        return Err("the kit's gated KV write case needs four pages of four tokens".into());
    }
    let rows = capacity as usize;
    let packed = 2 * (qh + kh) * d;
    let table_rows = op.max_context as usize + 6;
    let slots = ITEMS * ITEM_PAGES;
    let page_elements = page * kh * d;
    let fp8 = kv_dtype == DType::Fp8E4M3;
    let element = if fp8 { 1 } else { 2 };
    let tables = kv_tables(max_pages);
    let placed = kv_rows(rows, page);
    let cos_sin = cos_sin_table(table_rows, rd);
    let plane = |seed| -> Vec<u8> {
        if fp8 {
            // E4M3 codes in [-2, 2], none of them NaN.
            bf16_values(seed, slots * page_elements)
                .into_iter()
                .map(|b| SmallFloat::E4M3.encode_f32(bf16_to_f32(b)))
                .collect()
        } else {
            bf16_bytes(&bf16_values(seed, slots * page_elements))
        }
    };
    // The cache elements each live row writes: its slot's token row.
    let slot_of = {
        let tables = tables.clone();
        let placed = placed.clone();
        move |row: usize| {
            let (item, position, _) = placed[row];
            kv_slot(&tables, max_pages, (page, kh * d), (item, position))
        }
    };
    // The cache rows the live rows write, from the images' items,
    // positions and tables, so a property check's moved rows map where they
    // write; rows writing one cache row (copies) map it once.
    let written = move || {
        Write::Map(Box::new(move |images: &Images, live: u32| {
            let tables = images.i32s("block_tables")?;
            let items = images.i32s("token_batch")?;
            let positions = images.i32s("positions")?;
            let starts = (0..live as usize)
                .map(|row| {
                    let at = (items[row] as usize, positions[row] as usize);
                    kv_slot(&tables, max_pages, (page, kh * d), at)
                })
                .collect::<std::collections::BTreeSet<_>>();
            WriteMap::of(kv_dtype).ranges(starts.into_iter().map(|start| start..start + kh * d))
        }))
    };
    let state = |seed| {
        Spec::state(
            kv_dtype,
            (Rows::Fixed(slots), page_elements),
            plane(seed),
            written(),
        )
    };
    let roles = GatedKvWriteBuffers {
        qkv: Spec::input(
            DType::Bf16,
            (Rows::Live, packed),
            bf16_bytes(&bf16_values(43, rows * packed)),
        ),
        q: Spec::output(DType::Bf16, (Rows::Live, qh * d)),
        cache: KvCache {
            k: state(44),
            v: state(45),
            slots: slots as u32,
            k_slot_stride: (page_elements * element) as u64,
            v_slot_stride: (page_elements * element) as u64,
            k_scale: 0.5,
            v_scale: 0.25,
        },
        block_tables: Spec::input(
            DType::I32,
            (Rows::Fixed(ITEMS), max_pages),
            i32_bytes(&tables),
        ),
        token_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&placed.iter().map(|p| p.0 as i32).collect::<Vec<_>>()),
        ),
        positions: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&placed.iter().map(|p| p.1 as i32).collect::<Vec<_>>()),
        ),
        live_rows: Spec::live(),
        rope: FoldInputs {
            cos_sin: Some(Spec::input(
                DType::F32,
                (Rows::Fixed(table_rows), rd),
                f32_bytes(&cos_sin),
            )),
            q_norm: Some(Spec::input(
                DType::Bf16,
                (Rows::Fixed(1), d),
                bf16_bytes(&bf16_values(46, d)),
            )),
            k_norm: Some(Spec::input(
                DType::Bf16,
                (Rows::Fixed(1), d),
                bf16_bytes(&bf16_values(47, d)),
            )),
            rope_positions: Some(Spec::input(
                DType::I32,
                (Rows::Live, 3),
                i32_bytes(
                    &placed
                        .iter()
                        .flat_map(|p| p.2.map(|axis| axis as i32))
                        .collect::<Vec<_>>(),
                ),
            )),
        },
    };
    let query = op.query;
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let live = live as usize;
        let widen = |bits: Vec<u16>| bits.into_iter().map(|b| f64::from(bf16_to_f32(b)));
        let qkv = widen(images.u16s("qkv")?).collect::<Vec<_>>();
        let q_norm = widen(images.u16s("rope.q_norm")?).collect::<Vec<_>>();
        let k_norm = widen(images.u16s("rope.k_norm")?).collect::<Vec<_>>();
        let cos_sin = images.f32s("rope.cos_sin")?;
        let decode = |path: &str| -> LabResult<Vec<f64>> {
            Ok(if fp8 {
                images
                    .bytes(path)?
                    .iter()
                    .map(|&code| SmallFloat::E4M3.decode_f64(code))
                    .collect()
            } else {
                widen(images.u16s(path)?).collect()
            })
        };
        let exact = |values: Vec<f64>| {
            let bounds = values.iter().map(|v| v.abs()).collect();
            OracleOutput::new(values, bounds, 1).map_err(|e| e.0.to_string())
        };
        let mut planes = [exact(decode("cache.k")?)?, exact(decode("cache.v")?)?];
        let mut q = OracleOutput {
            values: vec![0.0; rows * qh * d],
            sum_abs_products: vec![0.0; rows * qh * d],
            reduction_lengths: vec![1; rows * qh * d],
        };
        for item in 0..ITEMS {
            let mine = (0..live)
                .filter(|&row| placed[row].0 == item)
                .collect::<Vec<_>>();
            if mine.is_empty() {
                continue;
            }
            let gather = |from: usize, width: usize| {
                mine.iter()
                    .flat_map(|&row| qkv[row * packed + from..][..width].to_vec())
                    .collect::<Vec<_>>()
            };
            let axes = mine.iter().map(|&row| placed[row].2).collect::<Vec<_>>();
            let pair_tables = pair_tables(&cos_sin, rd, mrope, &axes);
            let (v, k_raw) = (
                gather((2 * qh + kh) * d, kh * d),
                gather(2 * qh * d, kh * d),
            );
            let block_table = (0..ITEM_PAGES)
                .map(|page| tables[item * max_pages + page] as usize)
                .collect::<Vec<_>>();
            // The oracle's own cache, of which only Q and K's bounds are
            // used; the planes are placed below.
            let mut scratch_k = vec![0.0; slots * page_elements];
            let mut scratch_v = vec![0.0; slots * page_elements];
            let result = qk_norm_rope_kv_write(
                query,
                GatedAttentionInputs {
                    q_gate: &gather(0, 2 * qh * d),
                    k: &k_raw,
                    v: &v,
                    q_norm: &q_norm,
                    k_norm: &k_norm,
                    rope_positions: &axes,
                    token_positions: &mine.iter().map(|&row| placed[row].1).collect::<Vec<_>>(),
                    tables: Some(&pair_tables),
                },
                PagedKvPair {
                    keys: &mut scratch_k,
                    values: &mut scratch_v,
                    block_table: &block_table,
                },
            )
            .map_err(|e| e.0.to_string())?;
            for (at, &row) in mine.iter().enumerate() {
                let span = |width: usize| at * width..(at + 1) * width;
                let (from, to) = (span(qh * d), row * qh * d..(row + 1) * qh * d);
                q.values[to.clone()].copy_from_slice(&result.q.values[from.clone()]);
                q.sum_abs_products[to.clone()]
                    .copy_from_slice(&result.q.sum_abs_products[from.clone()]);
                q.reduction_lengths[to].copy_from_slice(&result.q.reduction_lengths[from]);
                let slot = slot_of(row);
                let from = span(kh * d);
                for (plane, (values, bounds, lengths), scale) in [
                    (
                        0,
                        (
                            &result.k.values[from.clone()],
                            result.k.sum_abs_products[from.clone()].to_vec(),
                            result.k.reduction_lengths[from.clone()].to_vec(),
                        ),
                        0.5,
                    ),
                    (
                        1,
                        (
                            &v[from.clone()],
                            v[from.clone()].iter().map(|v| v.abs()).collect(),
                            vec![1; kh * d],
                        ),
                        0.25,
                    ),
                ] {
                    // An FP8 plane holds values over its scale.
                    let scale = if fp8 { scale } else { 1.0 };
                    let target = &mut planes[plane];
                    for i in 0..kh * d {
                        target.values[slot + i] = values[i] / scale;
                        target.sum_abs_products[slot + i] = bounds[i] / scale;
                        target.reduction_lengths[slot + i] = lengths[i];
                    }
                }
            }
        }
        let [k, v] = planes;
        Ok(vec![
            ("q".into(), Expected::Values(q)),
            ("cache.k".into(), Expected::Values(k)),
            ("cache.v".into(), Expected::Values(v)),
        ])
    };
    Ok(OpCase::with_counts(prepared, capacity, roles, move |rows| {
        Counts::rows(rows)
            .with_items(ITEMS as u32, op.max_context)
            .with_positions(table_rows as u32)
    })?
    .oracle(oracle))
}

/// The required gated KV write `op`.
pub fn gated_kv_write<'a>(name: String, op: GatedKvWriteOp, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.gated_kv_write(&op, need),
        move |prepared| gated_kv_write_case(op, need.capacity, prepared),
    )
}

/// The case of an indexer query norm and M-RoPE over `capacity` rows of a
/// context of `context` tokens: BF16 packed projection rows (the query
/// heads, then the raw key) and Gemma norm weight from the kit's generator,
/// text and M-RoPE positions below the context with an F32 table two
/// positions past it, and `sparse_attention_oracle::indexer_query` over
/// the bound table within its tolerance.
pub fn index_query_case<'a>(
    op: SparseAttentionQuery,
    context: u32,
    capacity: u32,
    prepared: Prepared<IndexQueryBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let SparseAttentionQuery::IndexerQueryGemmaNormMrope {
        heads,
        head_dim,
        rotary_dim,
        mrope,
        ..
    } = op
    else {
        return Err("an index query case needs an index query".into());
    };
    let (heads, dim, rd, rows) = (
        heads as usize,
        head_dim as usize,
        rotary_dim as usize,
        capacity as usize,
    );
    let packed = (heads + 1) * dim;
    let table_rows = context as usize + 2;
    let axes = (0..rows)
        .map(|row| {
            let position = (row * 7 + 3) % context as usize;
            if row % 2 == 0 {
                [position; 3]
            } else {
                [position, position + 1, position + 2]
            }
        })
        .collect::<Vec<_>>();
    let cos_sin = cos_sin_table(table_rows, rd);
    let roles = IndexQueryBuffers {
        projection: Spec::input(
            DType::Bf16,
            (Rows::Live, packed),
            bf16_bytes(&bf16_values(48, rows * packed)),
        ),
        weight: Spec::input(
            DType::Bf16,
            (Rows::Fixed(1), dim),
            bf16_bytes(&bf16_values(49, dim)),
        ),
        rope_positions: Spec::input(
            DType::I32,
            (Rows::Live, 3),
            i32_bytes(
                &axes
                    .iter()
                    .flat_map(|axes| axes.map(|axis| axis as i32))
                    .collect::<Vec<_>>(),
            ),
        ),
        cos_sin: Spec::input(
            DType::F32,
            (Rows::Fixed(table_rows), rd),
            f32_bytes(&cos_sin),
        ),
        live_rows: Spec::live(),
        q: Spec::output(DType::Bf16, (Rows::Live, heads * dim)),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let widen = |bits: Vec<u16>| {
            bits.into_iter()
                .map(|b| f64::from(bf16_to_f32(b)))
                .collect::<Vec<_>>()
        };
        let projection = widen(images.u16s("projection")?);
        let q = projection
            .chunks_exact(packed)
            .flat_map(|row| row[..heads * dim].to_vec())
            .collect::<Vec<_>>();
        let axes = images
            .i32s("rope_positions")?
            .chunks_exact(3)
            .map(|axes| [axes[0] as usize, axes[1] as usize, axes[2] as usize])
            .collect::<Vec<_>>();
        let tables = pair_tables(&images.f32s("cos_sin")?, rd, mrope, &axes);
        let output = indexer_query(op, &q, &widen(images.u16s("weight")?), &axes, Some(&tables))
            .map_err(|e| e.0.to_string())?;
        Ok(vec![("q".into(), Expected::Values(output))])
    };
    Ok(OpCase::with_counts(prepared, capacity, roles, move |rows| {
        Counts::rows(rows)
            .with_items(1, context)
            .with_positions(table_rows as u32)
    })?
    .oracle(oracle))
}

/// The catalogs' gated rows, the first of each body: entries of one body
/// (the same code at another capacity or profile) are an SM120 tie.
fn gated_rows(catalogs: &Catalogs) -> Vec<gated::Entry> {
    let mut chosen: Vec<gated::Entry> = Vec::new();
    for entry in gated::catalog(catalogs) {
        if !chosen.iter().any(|other| other.body == entry.body) {
            chosen.push(entry);
        }
    }
    chosen
}

/// The catalogs' sparse index query entries, the first of each query
/// shape (its capacity aside): entries of one shape are an SM120 tie. The
/// kit runs no other sparse op.
fn index_queries(catalogs: &Catalogs) -> Vec<sparse::Entry> {
    let shape = |query: SparseAttentionQuery| match query {
        SparseAttentionQuery::IndexerQueryGemmaNormMrope {
            heads,
            head_dim,
            rotary_dim,
            theta_bits,
            mrope,
            eps_bits,
            ..
        } => Some((heads, head_dim, rotary_dim, theta_bits, mrope, eps_bits)),
        _ => None,
    };
    let mut chosen: Vec<sparse::Entry> = Vec::new();
    for entry in sparse::catalog(catalogs) {
        if shape(entry.query).is_some()
            && !chosen
                .iter()
                .any(|other| shape(other.query) == shape(entry.query))
        {
            chosen.push(entry);
        }
    }
    chosen
}

/// SM120's gated and sparse attention over `catalogs`: the first catalog
/// gated row of each body and sparse index query of each shape.
pub fn sm120(catalogs: &Catalogs) -> Sm120 {
    Sm120::new().with_gated_sparse(GatedSparseEntries {
        gated: gated_rows(catalogs),
        sparse: index_queries(catalogs),
        ..GatedSparseEntries::default()
    })
}

/// The first catalog gated KV write of each head width and cache dtype
/// (over a context of four pages) and the first output gate of each head
/// width (over the packed rows of its profile's gated KV write) at
/// [`CAPACITY`] rows, each required. The catalogs' other shapes run the
/// same code at other head counts.
pub fn required<'a>(catalogs: &Catalogs) -> Vec<Required<'a>> {
    let entries = gated_rows(catalogs);
    let mut ops = Vec::new();
    let mut seen = Vec::new();
    for entry in &entries {
        let GatedAttentionQuery::QkNormRopeKvWrite {
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            kv_dtype,
            ..
        } = entry.query
        else {
            continue;
        };
        if seen.contains(&(head_dim, kv_dtype)) {
            continue;
        }
        seen.push((head_dim, kv_dtype));
        let op = GatedKvWriteOp {
            query: entry.query,
            max_context: ITEM_PAGES as u32 * page_size,
        };
        ops.push(
            gated_kv_write(
                format!(
                    "gated KV write {q_heads} q heads {kv_heads} KV heads dim {head_dim} page \
                     {page_size} {kv_dtype:?}"
                ),
                op,
                Need::rows(CAPACITY),
            )
            .required(),
        );
    }
    let mut seen = Vec::new();
    for entry in index_queries(catalogs) {
        let op = entry.query;
        let SparseAttentionQuery::IndexerQueryGemmaNormMrope {
            heads, head_dim, ..
        } = op
        else {
            continue;
        };
        if seen.contains(&(heads, head_dim)) {
            continue;
        }
        seen.push((heads, head_dim));
        let context = entry.geometry.max_context.min(256);
        ops.push(
            Required::new(
                format!("index query {heads} heads dim {head_dim} context {context}"),
                Need::rows(CAPACITY),
                move |backend, need| backend.sparse_index_query(&op, need),
                move |prepared| index_query_case(op, context, CAPACITY, prepared),
            )
            .required(),
        );
    }
    let mut seen = Vec::new();
    for entry in &entries {
        let GatedAttentionQuery::OutputGate {
            heads, head_dim, ..
        } = entry.query
        else {
            continue;
        };
        // The packed rows are the profile's gated KV write's.
        let Some(kv_heads) = entries.iter().find_map(|other| match other.query {
            GatedAttentionQuery::QkNormRopeKvWrite { kv_heads, .. }
                if other.profile == entry.profile =>
            {
                Some(kv_heads)
            }
            _ => None,
        }) else {
            continue;
        };
        let op = OutputGateOp {
            query: entry.query,
            kv_heads,
        };
        if seen.contains(&head_dim) {
            continue;
        }
        seen.push(head_dim);
        ops.push(
            output_gate(
                format!("output gate {heads} heads dim {head_dim} over {kv_heads} KV heads"),
                op,
                Need::rows(CAPACITY),
            )
            .required(),
        );
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A write that ignores the block table, the row's item or its cache
    /// position (taking the RoPE T axis) lands on other cache rows than the
    /// case's at a partial live count, so the cache comparison fails it.
    #[test]
    fn gated_kv_write_fixtures_tell_ignored_addressing_apart() {
        let (page, row, max_pages, live) = (64, 256, ITEM_PAGES, 8);
        let tables = kv_tables(max_pages);
        let placed = kv_rows(live, page);
        let slots = |slot: &dyn Fn(usize, usize, usize) -> usize| {
            placed
                .iter()
                .map(|&(item, position, axes)| slot(item, position, axes[0]))
                .collect::<Vec<_>>()
        };
        let truth =
            slots(&|item, position, _| kv_slot(&tables, max_pages, (page, row), (item, position)));
        let ignoring = [
            // The logical page index as the slot.
            slots(&|item, position, _| {
                (item * ITEM_PAGES + position / page) * page * row + position % page * row
            }),
            // Item 0's table for every row.
            slots(&|_, position, _| kv_slot(&tables, max_pages, (page, row), (0, position))),
            // The T axis as the cache position.
            slots(&|item, _, t| kv_slot(&tables, max_pages, (page, row), (item, t))),
        ];
        for wrong in ignoring {
            assert_ne!(wrong, truth);
        }
        // No page sits at its own index, so no row's slot agrees by chance.
        for item in 0..ITEMS {
            for page in 0..ITEM_PAGES {
                assert_ne!(
                    tables[item * max_pages + page] as usize,
                    item * ITEM_PAGES + page
                );
            }
        }
        let mut used = tables.iter().filter(|&&slot| slot >= 0).collect::<Vec<_>>();
        used.sort();
        used.dedup();
        assert_eq!(used.len(), ITEMS * ITEM_PAGES);
    }
}
