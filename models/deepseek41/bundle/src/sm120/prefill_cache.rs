//! Decode native keys once for qualified single-item prefill chunks.
use kernel_build::{CubinSpec, EntryParam};
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{Argument as A, ArgumentList, Refusal};
use kernel_core::latent_attention::{
    LatentAttentionQuery, LatentSourcesBuffers, LatentWindowStorage,
};
use kernels_sm120::latent::{
    sources, sources_prefill, sources_prefill::heads, sources_prefill_cache as cache,
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use model_deepseek41_contract::own_ops::prefill_cache::Buffers;
use model_deepseek41_contract::state::KvStorage;

pub const FAMILY: &str = "latent_prefill_cache";
pub const CUBIN: &str = "latent_sources_prefill_cache_deepseek41_sm120";
/// Bound the decoded physical-page span, independent of the full serving pool.
pub const PAGE_LIMIT: u32 = 1024;

pub fn eligible(geometry: &Geometry, ranks: u32, storage: KvStorage) -> bool {
    geometry.id == FLASH.id && ranks == 4 && storage == KvStorage::Native
}

pub fn cubin() -> Result<CubinSpec, String> {
    cache::cubin_spec(
        super::latent::profile(&FLASH),
        &super::prefill_attention::entries()?,
    )
    .map_err(|e| e.to_string())
}

/// A dedicated arena is shared only by successive attention layers on one stream.
pub fn layout(slots: u32, page: u32, pages: u32) -> Result<cache::Layout, Refusal> {
    cache::Layout::new(512, slots, 1, page, pages)
}

pub fn prepare(
    backend: &dyn Backend,
    query: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<Buffers>, Refused> {
    let entry = super::prefill_attention::entries()
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|e| e.query == *query)
        .ok_or_else(|| Refused::Rule("unqualified decoded prefill query".into()))?;
    let reference = backend.latent_attention_sources(query, need)?;
    let decoded = cache::decoded_entry(&entry)?;
    let LatentAttentionQuery::Attention {
        latent,
        page_size,
        second_fp4_keys,
        ..
    } = entry.query
    else {
        unreachable!()
    };
    let mut info = reference.info().clone();
    info.symbols
        .push((sources_prefill::reuse_symbol(&decoded, 2), 85504));
    if second_fp4_keys {
        info.symbols.push((
            heads::asynchronous::symbol(&decoded, false),
            heads::SHARED_BYTES,
        ));
        info.symbols.push((
            super::prefill_heads::finish_symbol(&decoded),
            heads::SHARED_BYTES,
        ));
    }
    info.symbols.push((cache::CLEAR.into(), 0));
    info.symbols.push((cache::window_symbol(latent), 0));
    if second_fp4_keys {
        info.symbols.push((cache::mark_symbol(page_size), 0));
        info.symbols
            .push((cache::pages_symbol(latent, page_size), 0));
    }
    need.check(&info)?;
    Ok(Box::new(Cached {
        reference,
        entry,
        decoded,
        info,
    }))
}

struct Cached {
    reference: Prepared<LatentSourcesBuffers<View>>,
    entry: sources::Entry,
    decoded: sources::Entry,
    info: OpInfo,
}

fn buffer(view: View) -> A {
    A::DeviceBuffer(view.handle)
}

fn producer(symbol: String, items: u64, rows: u32, arguments: Vec<A>) -> Result<Launch, Refusal> {
    Ok(Launch {
        symbol,
        rows,
        plan: cache::producer_plan(items)?,
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    })
}

impl PreparedOp<Buffers> for Cached {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        b: &Buffers,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let n = &b.native;
        // The native validator still checks every source and partial/output span.
        let mut attention = self.reference.launches(n, workspace, counts)?;
        let LatentWindowStorage::Staged { slots } = n.window.storage else {
            return Err(Refusal("decoded prefill requires staged keys"));
        };
        if counts.rows != super::prefill_attention::ROWS || n.window.sequences != 1 {
            return Err(Refusal("decoded prefill requires one full staged chunk"));
        }
        let LatentAttentionQuery::Attention {
            latent,
            page_size,
            second_fp4_keys,
            ..
        } = self.entry.query
        else {
            unreachable!()
        };
        let pages = n.compressed.map_or(1, |c| c.cache.pages);
        let regions = layout(slots, page_size, pages)?;
        b.window.covers(regions.window.end - regions.window.start)?;
        b.compressed
            .covers(regions.compressed.end - regions.compressed.start)?;
        b.marks.covers(regions.marks.end - regions.marks.start)?;
        let rows = counts.rows;
        let mut launches = vec![producer(
            cache::CLEAR.into(),
            u64::from(pages),
            rows,
            vec![buffer(b.marks), A::U32(pages)],
        )?];
        if let Some(c) = n.compressed {
            launches.push(producer(
                cache::mark_symbol(page_size),
                u64::from(rows) * u64::from(c.budget),
                rows,
                vec![
                    buffer(n.row_batch),
                    buffer(n.live_rows),
                    buffer(c.key_lists),
                    buffer(c.block_tables.tables),
                    buffer(b.marks),
                    A::U32(rows),
                    A::U32(c.key_stride),
                    A::U32(c.budget),
                    A::U32(c.block_tables.max_pages),
                    A::U32(pages),
                    A::U32(c.block_tables.sequences),
                ],
            )?);
        }
        launches.push(producer(
            cache::window_symbol(latent),
            u64::from(slots) * u64::from(latent / 8),
            rows,
            vec![
                buffer(n.window.codes),
                buffer(n.window.scales),
                buffer(b.window),
                A::U32(slots),
                A::U32(1),
                A::U64(n.window.code_stride),
                A::U64(n.window.scale_stride),
            ],
        )?);
        if let Some(c) = n.compressed {
            launches.push(producer(
                cache::pages_symbol(latent, page_size),
                u64::from(pages) * u64::from(page_size) * u64::from(latent / 8),
                rows,
                vec![
                    buffer(c.cache.pool),
                    buffer(b.marks),
                    buffer(b.compressed),
                    A::U32(pages),
                    A::U64(c.cache.page_stride),
                ],
            )?);
        }
        let first = attention
            .first_mut()
            .ok_or(Refusal("missing native attention launch"))?;
        if first.symbol != self.entry.symbol() {
            return Err(Refusal("unexpected native attention producer"));
        }
        for (param, argument) in self.entry.params()?.iter().zip(&mut first.arguments.0) {
            let name = match param {
                EntryParam::DevicePointer { name, .. }
                | EntryParam::U32 { name }
                | EntryParam::U64 { name } => name.as_str(),
                _ => return Err(Refusal("unsupported decoded prefill parameter")),
            };
            match name {
                "window_codes" => *argument = buffer(b.window),
                "window_scales" => *argument = A::NullDevicePointer,
                "window_code_stride" => {
                    *argument = A::U64(u64::from(slots) * u64::from(latent) * 2)
                }
                "window_scale_stride" => *argument = A::U64(0),
                "compressed_pool" => *argument = buffer(b.compressed),
                "page_stride" => *argument = A::U64(u64::from(page_size) * u64::from(latent) * 2),
                _ => (),
            }
        }
        if second_fp4_keys {
            launches.extend(super::prefill_heads::finish(
                &self.entry,
                &self.decoded,
                n,
                rows,
                attention,
            )?);
        } else {
            first.symbol = sources_prefill::reuse_symbol(&self.decoded, 2);
            first.plan = sources_prefill::reuse_plan(&self.decoded, rows, 2)?;
            launches.extend(attention);
        }
        Ok(launches)
    }
}
