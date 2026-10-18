//! F32 tail-rotation entries used before the shared MXFP4 query quantizer.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::pooled_indexer::PositionIndexerQuery;
use kernel_core::properties::Invariance;

pub const HEADER: &str = "pooled_indexer/position_query.cuh";
pub const THREADS: u32 = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub name: String,
    pub heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub capacity: u32,
}

impl Entry {
    pub fn new(name: impl Into<String>, query: PositionIndexerQuery) -> Result<Self, Refusal> {
        query.validate()?;
        let entry = Self {
            name: name.into(),
            heads: query.heads,
            head_dim: query.head_dim,
            rotary_dim: query.tail_rope_dims,
            capacity: query.capacity,
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        if self.name.is_empty()
            || !self
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || self.heads == 0
            || self.heads > 64
            || self.head_dim == 0
            || self.head_dim > 512
            || !self.head_dim.is_multiple_of(32)
            || self.rotary_dim == 0
            || self.rotary_dim > self.head_dim
            || !self.rotary_dim.is_multiple_of(2)
            || self.capacity == 0
            || self.capacity > super::position::MAX_ROWS
        {
            return Err(Refusal(
                "position query rotation is outside the SM120 engine",
            ));
        }
        Ok(())
    }

    pub fn matches(&self, query: &PositionIndexerQuery) -> bool {
        (self.heads, self.head_dim, self.rotary_dim)
            == (query.heads, query.head_dim, query.tail_rope_dims)
    }

    pub fn symbol(&self) -> String {
        format!("chop_position_query_{}_rotate_v2", self.name)
    }

    pub fn guarantees(&self) -> Invariance {
        Invariance::any_rows(self.capacity)
    }

    pub fn spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: cxx_type.into(),
        };
        Ok(EntrySpec::new(
            self.symbol(),
            vec![
                ptr("input", "const float"),
                ptr("cos_sin", "const float2"),
                ptr("positions", "const int32_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("rotated", "float"),
            ],
            format!(
                "chop::position_query::rotate<{},{},{}>(input,cos_sin,positions,live_rows,rotated);",
                self.heads, self.head_dim, self.rotary_dim
            ),
            THREADS,
            1,
        ))
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("position query rows exceed the prepared capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows, self.heads, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn row_bytes(&self) -> u64 {
        u64::from(self.heads) * u64::from(self.head_dim) * 4
    }
}

pub fn cubin_spec(
    name: impl Into<String>,
    entries: &[Entry],
    quantizers: &[crate::row::block_quantize::Entry],
) -> Result<CubinSpec, Refusal> {
    let mut specs = entries
        .iter()
        .map(Entry::spec)
        .collect::<Result<Vec<_>, _>>()?;
    specs.extend(
        quantizers
            .iter()
            .map(crate::row::block_quantize::Entry::spec),
    );
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec![
            format!("#include \"{HEADER}\""),
            format!("#include \"{}\"", crate::row::block_quantize::HEADER),
        ],
        defines: vec![],
        entries: specs,
    })
}
