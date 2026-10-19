//! Exact temporary BF16 operands for paired native prefill attention.
use super::{sources, sources_prefill};
use crate::Profile;
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use std::ops::Range;

pub const HEADER: &str = "latent/sources_prefill_cache.cuh";
pub const CLEAR: &str = "chop_latent_prefill_clear_pages_v1";
pub const GUARD: u64 = 256;

#[derive(Clone, Debug)]
pub struct Layout {
    pub window: Range<u64>,
    pub compressed: Range<u64>,
    pub marks: Range<u64>,
    pub bytes: u64,
}
impl Layout {
    pub fn new(
        width: u32,
        slots: u32,
        sequences: u32,
        page: u32,
        pages: u32,
    ) -> Result<Self, Refusal> {
        if width == 0
            || width > 512
            || width % 128 != 0
            || slots == 0
            || sequences == 0
            || !page.is_power_of_two()
            || pages == 0
        {
            return Err(Refusal("invalid prefill decoded-cache geometry"));
        }
        let product = |dims: &[u64]| {
            dims.iter()
                .try_fold(1_u64, |n, d| n.checked_mul(*d))
                .ok_or(Refusal("decoded cache size overflows"))
        };
        let mut cursor = GUARD;
        let mut region = |bytes: u64| -> Result<Range<u64>, Refusal> {
            let start = cursor;
            let end = start
                .checked_add(bytes)
                .ok_or(Refusal("decoded cache range overflows"))?;
            cursor = end
                .checked_add(GUARD * 2 - 1)
                .ok_or(Refusal("decoded cache guard overflows"))?
                & !(GUARD - 1);
            Ok(start..end)
        };
        let window = region(product(&[
            u64::from(width),
            u64::from(slots),
            u64::from(sequences),
            2,
        ])?)?;
        let compressed = region(product(&[
            u64::from(width),
            u64::from(page),
            u64::from(pages),
            2,
        ])?)?;
        let marks = region(product(&[u64::from(pages), 4])?)?;
        Ok(Self {
            window,
            compressed,
            marks,
            bytes: cursor,
        })
    }
    pub fn guards(&self) -> [Range<u64>; 4] {
        [
            0..self.window.start,
            self.window.end..self.compressed.start,
            self.compressed.end..self.marks.start,
            self.marks.end..self.bytes,
        ]
    }
}

pub fn decoded_entry(reference: &sources::Entry) -> Result<sources::Entry, Refusal> {
    sources_prefill::reuse_plan(reference, 1, 2)?;
    let mut entry = reference.clone();
    let LatentAttentionQuery::Attention { kv, .. } = &mut entry.query else {
        unreachable!()
    };
    if *kv != LatentKvFormat::Fp8E4M3Ue8M0Block32 {
        return Err(Refusal("decoded prefill requires native source storage"));
    }
    *kv = LatentKvFormat::Bf16;
    entry.name.push_str("_decoded");
    Ok(entry)
}
pub fn window_symbol(width: u32) -> String {
    format!("chop_latent_prefill_decode_window_d{width}_v1")
}
pub fn pages_symbol(width: u32, page: u32) -> String {
    format!("chop_latent_prefill_decode_pages_d{width}_p{page}_v1")
}
pub fn mark_symbol(page: u32) -> String {
    format!("chop_latent_prefill_mark_pages_p{page}_v1")
}

/// Producers serialize with their consumers. No PDL or cross-stream reuse.
pub fn producer_plan(items: u64) -> Result<LaunchPlan, Refusal> {
    if items == 0 {
        return Err(Refusal("empty decoded-cache producer"));
    }
    Ok(LaunchPlan {
        grid: [items.div_ceil(256).min(65535) as u32, 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: false,
    })
}
fn ptr(name: &str, ty: &str) -> P {
    P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}
fn u32p(name: &str) -> P {
    P::U32 { name: name.into() }
}
fn u64p(name: &str) -> P {
    P::U64 { name: name.into() }
}
pub fn cubin_spec(profile: Profile, references: &[sources::Entry]) -> Result<CubinSpec, Refusal> {
    let entries = references
        .iter()
        .map(decoded_entry)
        .collect::<Result<Vec<_>, _>>()?;
    let mut spec = sources_prefill::reuse_cubin_spec(profile, &entries, 2)?;
    spec.name = format!("latent_sources_prefill_cache_{}_sm120", profile.tag());
    spec.headers = vec![format!("#include \"{HEADER}\"")];
    spec.entries.push(EntrySpec::new(
        CLEAR.into(),
        vec![ptr("marked", "uint32_t"), u32p("pages")],
        "chop::latent::sources_prefill::clear_pages(marked,pages);".into(),
        256,
        1,
    ));
    let mut widths = std::collections::BTreeSet::new();
    let mut shapes = std::collections::BTreeSet::new();
    let mut pages = std::collections::BTreeSet::new();
    for reference in references {
        let shape = sources::shape(reference.query)?;
        widths.insert(shape.latent);
        if shape.second {
            shapes.insert((shape.latent, shape.page));
            pages.insert(shape.page);
        }
    }
    for width in widths {
        spec.entries.push(EntrySpec::new(window_symbol(width),vec![ptr("codes","const uint8_t"),ptr("scales","const uint8_t"),ptr("decoded","uint16_t"),u32p("slots"),u32p("sequences"),u64p("code_stride"),u64p("scale_stride")],
            format!("chop::latent::sources_prefill::decode_window<{width}>(codes,scales,decoded,slots,sequences,code_stride,scale_stride);"),256,1));
    }
    for page in pages {
        spec.entries.push(EntrySpec::new(mark_symbol(page),vec![ptr("row_batch","const int32_t"),ptr("live_rows","const uint32_t"),ptr("keys","const int32_t"),ptr("tables","const int32_t"),ptr("marked","uint32_t"),u32p("capacity"),u32p("key_stride"),u32p("budget"),u32p("max_pages"),u32p("pages"),u32p("sequences")],
            format!("chop::latent::sources_prefill::mark_pages<{page}>(row_batch,live_rows,keys,tables,marked,capacity,key_stride,budget,max_pages,pages,sequences);"),256,1));
    }
    for (width, page) in shapes {
        spec.entries.push(EntrySpec::new(pages_symbol(width,page),vec![ptr("pool","const uint8_t"),ptr("marked","const uint32_t"),ptr("decoded","uint16_t"),u32p("pages"),u64p("page_stride")],
            format!("chop::latent::sources_prefill::decode_pages<{width},{page}>(pool,marked,decoded,pages,page_stride);"),256,1));
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decoded_cache_ranges_keep_guards_and_reject_overflow() {
        let p = Layout::new(512, 8319, 1, 64, 512).unwrap();
        assert_eq!(p.window.end - p.window.start, 8319 * 512 * 2);
        assert_eq!(p.compressed.end - p.compressed.start, 32768 * 512 * 2);
        assert_eq!(p.marks.end - p.marks.start, 512 * 4);
        for r in [&p.window, &p.compressed, &p.marks] {
            assert_eq!(r.start % 256, 0);
            assert!(r.end <= p.bytes);
        }
        assert!(p.guards().iter().all(|g| g.end - g.start >= GUARD));
        assert!(Layout::new(512, u32::MAX, u32::MAX, 64, 512).is_err());
        assert!(Layout::new(512, 1, 1, 63, 1).is_err());
        assert!(producer_plan(0).is_err());
        assert_eq!(producer_plan(u64::MAX).unwrap().grid, [65535, 1, 1]);
        assert!(!producer_plan(1).unwrap().pdl);
    }
}
