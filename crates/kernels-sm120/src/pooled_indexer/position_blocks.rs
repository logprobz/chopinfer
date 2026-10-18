//! Compact Position v3 score/select and block-ID producer entries.

use super::position;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::pooled_indexer::CandidateBlocksQuery;
use kernel_core::position_blocks::PositionBlockIndexerQuery;
use kernel_core::properties::Invariance;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub name: String,
    pub query: PositionBlockIndexerQuery,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateEntry {
    pub name: String,
    pub query: CandidateBlocksQuery,
}

impl Entry {
    fn dense(&self) -> position::Entry {
        position::Entry {
            name: self.name.clone(),
            query: self.query.indexer,
        }
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        self.dense().validate()?;
        // Only the consumed block geometry is implemented.
        if self.query.block_size != 8 || self.query.top_k_blocks > 2048 {
            return Err(Refusal(
                "compact position block geometry is outside the SM120 engine",
            ));
        }
        Ok(())
    }
    pub fn guarantees(&self) -> Invariance {
        self.dense().guarantees()
    }
    pub fn score_symbol(&self) -> String {
        format!("chop_position_blocks_{}_scores_v3", self.name)
    }
    pub fn select_symbol(&self) -> String {
        format!("chop_position_blocks_{}_select_v3", self.name)
    }
    pub fn score_plan(&self, rows: u32, width: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        self.dense().score_plan(rows, width)?;
        self.dense().score_plan(rows, self.query.score_slots()?)
    }
    pub fn select_plan(&self, rows: u32, width: u32, top_k: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        self.dense().select_plan(rows, width, top_k)
    }
    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.control_specs(0)
    }
    /// Nonzero controls are lab-only, with distinct symbols.
    pub fn control_specs(&self, control: u32) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let q = self.query.indexer;
        let block = self.query.block_size;
        let kept = self.query.top_k_blocks;
        let scale = kernel_core::position_indexer_oracle::weight_scale(q.selection())?.to_bits();
        let common = [
            ("tables", "const int32_t"),
            ("batches", "const int32_t"),
            ("positions", "const int32_t"),
            ("live_rows", "const uint32_t"),
            ("ids", "const int32_t"),
            ("tags", "const uint64_t"),
            ("stored_tags", "const uint64_t"),
            ("stored_visible", "const int32_t"),
            ("source_status", "const int32_t"),
        ];
        let mut score = [
            ("q", "const uint16_t"),
            ("weights", "const float"),
            ("codes", "const uint8_t"),
            ("scales", "const uint8_t"),
        ]
        .into_iter()
        .chain(common)
        .map(|(n, t)| ptr(n, t))
        .collect::<Vec<_>>();
        score.extend([
            ptr("scores", "float"),
            u32_arg("width"),
            u32_arg("max_pages"),
            EntryParam::U64 {
                name: "code_stride".into(),
            },
            EntryParam::U64 {
                name: "scale_stride".into(),
            },
        ]);
        let mut select = vec![ptr("scores", "float")];
        select.extend(common.into_iter().map(|(n, t)| ptr(n, t)));
        select.extend([
            ptr("output", "int32_t"),
            ptr("status", "int32_t"),
            u32_arg("width"),
            u32_arg("max_pages"),
            u32_arg("top_k"),
        ]);
        Ok(vec![
            EntrySpec::new(
                control_symbol(self.score_symbol(), control)?,
                score,
                format!(
                    "chop::position_blocks::scores<{},{},{},{},{block},{kept},{control}>(q,weights,codes,scales,tables,batches,positions,live_rows,ids,tags,stored_tags,stored_visible,source_status,scores,width,max_pages,code_stride,scale_stride,__uint_as_float({scale}u));",
                    q.heads, q.head_dim, q.page_size, q.compression_ratio
                ),
                position::SCORE_THREADS,
                1,
            ),
            EntrySpec::new(
                control_symbol(self.select_symbol(), control)?,
                select,
                format!(
                    "chop::position_blocks::select<{},{},{},{block},{kept},{control}>(scores,tables,batches,positions,live_rows,ids,tags,stored_tags,stored_visible,source_status,output,status,width,max_pages,top_k);",
                    q.page_size, q.compression_ratio, q.max_top_k
                ),
                position::SELECT_THREADS,
                1,
            ),
        ])
    }
}

impl CandidateEntry {
    fn dense(&self) -> position::CandidateEntry {
        position::CandidateEntry {
            name: self.name.clone(),
            query: self.query,
        }
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.dense().validate()?;
        if self.query.block_size != 8 || self.query.top_k_blocks > 2048 {
            return Err(Refusal(
                "compact candidate geometry is outside the SM120 engine",
            ));
        }
        Ok(())
    }
    pub fn guarantees(&self) -> Invariance {
        self.dense().guarantees()
    }
    pub fn score_symbol(&self) -> String {
        format!("chop_candidate_block_ids_{}_scores_v3", self.name)
    }
    pub fn gather_symbol(&self) -> String {
        format!("chop_candidate_block_ids_{}_gather_v3", self.name)
    }
    pub fn workspace_bytes(&self, rows: u32, width: u32) -> Result<u64, Refusal> {
        self.validate()?;
        self.dense().workspace_bytes(rows, width)
    }
    pub fn score_plan(&self, rows: u32, width: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        self.dense().score_plan(rows, width)
    }
    pub fn gather_plan(&self, rows: u32, width: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        self.dense().mask_plan(rows, width)
    }
    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let score = EntrySpec::new(
            self.score_symbol(),
            vec![
                ptr("scores", "const float"),
                ptr("visible", "const int32_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("blocks", "float"),
                u32_arg("width"),
            ],
            format!(
                "chop::position_blocks::block_scores<{}>(scores,visible,live_rows,blocks,width);",
                self.query.block_size
            ),
            position::SELECT_THREADS,
            1,
        );
        let mut params = [
            ("blocks", "const float"),
            ("live_rows", "const uint32_t"),
            ("tags", "const uint64_t"),
            ("visible", "const int32_t"),
            ("source_status", "const int32_t"),
            ("ids", "int32_t"),
            ("out_tags", "uint64_t"),
            ("out_visible", "int32_t"),
            ("status", "int32_t"),
        ]
        .into_iter()
        .map(|(n, t)| ptr(n, t))
        .collect::<Vec<_>>();
        params.push(u32_arg("width"));
        Ok(vec![
            score,
            EntrySpec::new(
                self.gather_symbol(),
                params,
                format!(
                    "chop::position_blocks::gather<{},{}>(blocks,live_rows,tags,visible,source_status,ids,out_tags,out_visible,status,width);",
                    self.query.block_size, self.query.top_k_blocks
                ),
                position::SELECT_THREADS,
                1,
            ),
        ])
    }
}

pub fn control_symbol(symbol: String, control: u32) -> Result<String, Refusal> {
    let suffix = match control {
        0 => return Ok(symbol),
        1 => "ignore_order",
        2 => "ignore_tag",
        3 => "ignore_pin",
        _ => return Err(Refusal("unknown compact position control")),
    };
    let base = symbol
        .strip_suffix("_v3")
        .expect("versioned compact symbol");
    Ok(format!("{base}_{suffix}_v3"))
}
fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}
fn u32_arg(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}
pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"pooled_indexer/position_blocks.cuh\"".into()],
        defines: vec![],
        entries,
    }
}
