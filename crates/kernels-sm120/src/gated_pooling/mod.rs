//! F32 pair pooling, RMSNorm and raw F32 tail state. The two launches
//! retain the F32 intermediate for separate oracle checks. Ratio one uses
//! the same F32 RMSNorm without a state or completion mask.

pub mod key_write;
mod norm;
pub use norm::{F32Norm, ratio_one_norm};

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, RecurrentPhase, Refusal, Validate};
use kernel_core::gated_pooling::GatedPoolingQuery;

pub const HEADER: &str = "gated_pooling/pool.cuh";
pub const THREADS: u32 = 128;

/// One immutable pairing of a pool entry and its masked RMSNorm entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    query: GatedPoolingQuery,
    pool_symbol: String,
    norm_symbol: String,
}

impl Entry {
    pub fn new(query: GatedPoolingQuery, stem: &str) -> Result<Self, Refusal> {
        query.validate()?;
        if query.group != 2 || !matches!(query.width, 128 | 512) {
            return Err(Refusal("F32 pooling supports pairs at widths 128 and 512"));
        }
        if query.capacity > crate::backend::MAX_CAPACITY {
            return Err(Refusal("F32 pooling capacity exceeds the SM120 limit"));
        }
        if matches!(query.phase, RecurrentPhase::Verify { max_rows } if max_rows > 16) {
            return Err(Refusal("F32 pooling verifies at most 16 rows per item"));
        }
        if stem.is_empty()
            || !stem
                .bytes()
                .enumerate()
                .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            return Err(Refusal(
                "F32 pooling needs a C identifier for its symbol stem",
            ));
        }
        Ok(Self {
            query,
            pool_symbol: format!("{stem}_pool_v1"),
            norm_symbol: format!("{stem}_norm_v1"),
        })
    }

    pub fn query(&self) -> GatedPoolingQuery {
        self.query
    }

    pub fn symbols(&self) -> [&str; 2] {
        [&self.pool_symbol, &self.norm_symbol]
    }

    pub fn workspace_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.plan(rows)?;
        Ok(u64::from(rows) * u64::from(self.query.width) * 4)
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows == 0 || rows > self.query.capacity {
            return Err(Refusal("F32 pooling rows exceed the prepared capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: u64::from(rows) * u64::from(self.query.width) * 4,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn entry_specs(&self) -> [EntrySpec; 2] {
        let max_rows = match self.query.phase {
            RecurrentPhase::Verify { max_rows } => max_rows,
            _ => 0,
        };
        let width = self.query.width;
        let mut pool_params = vec![
            pointer("key", "const float"),
            pointer("gate", "const float"),
            pointer("positions", "const int32_t"),
            pointer("row_batch", "const int32_t"),
            pointer("live_rows", "const uint32_t"),
            pointer("slot_ids", "const int32_t"),
            pointer("checkpoint_ids", "const int32_t"),
            pointer("state", "float"),
            pointer("pooled", "float"),
            pointer("output_positions", "int32_t"),
            scalar("items"),
            scalar("entries"),
        ];
        if max_rows == 0 {
            pool_params.retain(|p| !matches!(p, EntryParam::DevicePointer { name, .. } if name == "checkpoint_ids"));
        }
        let checkpoints = if max_rows == 0 {
            "nullptr"
        } else {
            "checkpoint_ids"
        };
        let pool = EntrySpec::new(
            self.pool_symbol.clone(),
            pool_params,
            format!(
                "chop::gated_pooling::pool2<{width},{max_rows}>(key,gate,positions,\
                 row_batch,live_rows,slot_ids,{checkpoints},state,pooled,\
                 output_positions,items,entries);"
            ),
            THREADS,
            1,
        );
        let norm = EntrySpec::new(
            self.norm_symbol.clone(),
            vec![
                pointer("pooled", "const float"),
                pointer("norm_weight", "const uint16_t"),
                pointer("output_positions", "const int32_t"),
                pointer("output", "float"),
                pointer("live_rows", "const uint32_t"),
                scalar("eps_bits"),
            ],
            format!(
                "chop::gated_pooling::norm<{width}>(pooled,norm_weight,\
                 output_positions,output,live_rows,__uint_as_float(eps_bits));"
            ),
            THREADS,
            1,
        );
        [pool, norm]
    }
}

pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("a pooling cubin needs at least one entry"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: entries.iter().flat_map(Entry::entry_specs).collect(),
    })
}

pub(super) fn pointer(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

pub(super) fn scalar(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}
