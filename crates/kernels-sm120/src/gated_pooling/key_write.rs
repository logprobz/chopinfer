//! Paged index-key and compressed-KV writers. Quantization is fused after
//! the trained BF16 rounding points; cache plane strides remain 64-bit.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::compressed_kv::{CompressedKvStorage, CompressedKvWriteQuery};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::pooled_indexer::PositionIndexerQuery;

use super::{THREADS, pointer, scalar};

pub const HEADER: &str = "gated_pooling/key_write.cuh";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Format {
    Index,
    Compressed(CompressedKvStorage),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    width: u32,
    rotary: u32,
    ratio: u32,
    page_size: u32,
    eps_bits: u32,
    format: Format,
}

impl Geometry {
    fn index(query: &PositionIndexerQuery) -> Result<Self, Refusal> {
        query.validate()?;
        Ok(Self {
            width: query.head_dim,
            rotary: query.tail_rope_dims,
            ratio: query.compression_ratio,
            page_size: query.page_size,
            eps_bits: query.key_norm_eps_bits,
            format: Format::Index,
        })
    }

    fn compressed(query: &CompressedKvWriteQuery) -> Result<Self, Refusal> {
        query.validate()?;
        Ok(Self {
            width: query.width,
            rotary: query.tail_rope_dims,
            ratio: query.compression_ratio,
            page_size: query.page_size,
            eps_bits: 0,
            format: Format::Compressed(query.storage),
        })
    }

    fn validate(self) -> Result<(), Refusal> {
        if !matches!(self.width, 128 | 512)
            || self.rotary != 64
            || !matches!(self.ratio, 1 | 2)
            || self.page_size > i32::MAX as u32
        {
            return Err(Refusal(
                "FP4 key writes support widths 128/512, tail 64 and ratios 1/2",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    geometry: Geometry,
    capacity: u32,
    symbol: String,
}

impl Entry {
    pub fn index(query: &PositionIndexerQuery, symbol: &str) -> Result<Self, Refusal> {
        Self::new(Geometry::index(query)?, query.capacity, symbol)
    }

    pub fn compressed(query: &CompressedKvWriteQuery, symbol: &str) -> Result<Self, Refusal> {
        Self::new(Geometry::compressed(query)?, query.capacity, symbol)
    }

    fn new(geometry: Geometry, capacity: u32, symbol: &str) -> Result<Self, Refusal> {
        geometry.validate()?;
        if capacity == 0 || capacity > crate::backend::MAX_CAPACITY {
            return Err(Refusal("FP4 key-write capacity exceeds the SM120 limit"));
        }
        if symbol.is_empty()
            || !symbol
                .bytes()
                .enumerate()
                .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            return Err(Refusal(
                "FP4 key writes need a C identifier for their symbol",
            ));
        }
        Ok(Self {
            geometry,
            capacity,
            symbol: symbol.into(),
        })
    }

    pub fn matches_index(&self, query: &PositionIndexerQuery) -> bool {
        Geometry::index(query) == Ok(self.geometry)
    }

    pub fn matches_compressed(&self, query: &CompressedKvWriteQuery) -> bool {
        Geometry::compressed(query) == Ok(self.geometry)
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
    pub fn width(&self) -> u32 {
        self.geometry.width
    }
    pub fn page_size(&self) -> u32 {
        self.geometry.page_size
    }
    pub fn rotary(&self) -> u32 {
        self.geometry.rotary
    }
    pub fn ratio(&self) -> u32 {
        self.geometry.ratio
    }
    pub fn eps_bits(&self) -> u32 {
        self.geometry.eps_bits
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("FP4 key-write rows exceed the prepared capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn entry_spec(&self) -> EntrySpec {
        let g = self.geometry;
        let normalize = g.format == Format::Index;
        let e4m3 = !normalize;
        let packed = g.format != Format::Compressed(CompressedKvStorage::Bf16);
        let mut params = vec![
            pointer("key", "const float"),
            pointer("norm_weight", "const uint16_t"),
            pointer("cos_sin", "const float"),
            pointer("positions", "const int32_t"),
            pointer("row_batch", "const int32_t"),
            pointer("live_rows", "const uint32_t"),
            pointer("page_tables", "const int32_t"),
            pointer("codes", "uint8_t"),
            pointer("scales", "uint8_t"),
            scalar("items"),
            scalar("pages"),
            scalar("max_pages"),
            EntryParam::U64 {
                name: "code_stride".into(),
            },
            EntryParam::U64 {
                name: "scale_stride".into(),
            },
            scalar("table_rows"),
            scalar("eps_bits"),
        ];
        params.retain(|p| {
            !matches!(p, EntryParam::DevicePointer { name, .. }
            if (!normalize && name == "norm_weight") || (!packed && name == "scales"))
        });
        let norm = if normalize { "norm_weight" } else { "nullptr" };
        let scales = if packed { "scales" } else { "nullptr" };
        EntrySpec::new(
            self.symbol.clone(),
            params,
            format!(
                "chop::gated_pooling::key_write<{},{},{},{},{normalize},{e4m3},{packed}>(\
             key,{norm},cos_sin,positions,row_batch,live_rows,page_tables,codes,\
             {scales},items,pages,max_pages,code_stride,scale_stride,table_rows,__uint_as_float(eps_bits));",
                g.width, g.rotary, g.ratio, g.page_size,
            ),
            THREADS,
            1,
        )
    }
}

pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("an FP4 key-write cubin needs entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: entries.iter().map(Entry::entry_spec).collect(),
    })
}
