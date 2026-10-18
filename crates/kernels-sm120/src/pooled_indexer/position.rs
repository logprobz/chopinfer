//! MXFP4 position score/select and candidate-block filter builders. Their
//! cubins include only the position header and its immutable helpers.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use kernel_core::properties::{Invariance, Ties};

pub const HEADER: &str = "pooled_indexer/position.cuh";
pub const SCORE_THREADS: u32 = 128;
pub const SELECT_THREADS: u32 = 256;
/// The scoring grid uses its second dimension for rows.
pub const MAX_ROWS: u32 = 65_535;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub name: String,
    pub query: PositionIndexerQuery,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateEntry {
    pub name: String,
    pub query: CandidateBlocksQuery,
}

fn name_valid(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !name_valid(&self.name)
            || self.query.heads > 64
            || self.query.head_dim > 512
            || self.query.capacity > MAX_ROWS
        {
            return Err(Refusal(
                "position indexer geometry is outside the SM120 engine",
            ));
        }
        Ok(())
    }

    pub fn score_symbol(&self) -> String {
        format!("chop_position_indexer_{}_scores_v3", self.name)
    }
    pub fn select_symbol(&self) -> String {
        format!("chop_position_indexer_{}_select_v3", self.name)
    }

    pub fn guarantees(&self) -> Invariance {
        Invariance {
            ties: Ties::LowerIndex,
            ..Invariance::any_rows(self.query.capacity)
        }
    }

    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.specs_with_control(0, "")
    }

    /// An ignored-mask negative control. Only lab families build it; it
    /// has distinct symbols and cannot be registered as a served entry.
    pub fn ignored_mask_specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        if self.query.candidates != IndexerCandidates::Mask {
            return Err(Refusal("ignored-mask control requires a masked entry"));
        }
        self.specs_with_control(1, "_ignore_mask")
    }

    /// Lab-only mutations of the score or exact selection. None can be
    /// registered as a served entry because all have distinct symbols.
    pub fn control_specs(&self, control: u32) -> Result<Vec<EntrySpec>, Refusal> {
        let suffix = match control {
            2 => "_reverse_ties",
            3 => "_drop_candidate",
            4 => "_wrong_weight",
            5 => "_swap_nibbles",
            _ => return Err(Refusal("unknown position indexer control")),
        };
        self.specs_with_control(control, suffix)
    }

    fn specs_with_control(&self, control: u32, suffix: &str) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let q = self.query;
        let masked = q.candidates == IndexerCandidates::Mask;
        let scale = kernel_core::position_indexer_oracle::weight_scale(q.selection())?.to_bits();
        let score = EntrySpec::new(
            control_symbol(self.score_symbol(), suffix),
            vec![
                ptr("q", "const uint16_t"),
                ptr("weights", "const float"),
                ptr("codes", "const uint8_t"),
                ptr("scales", "const uint8_t"),
                ptr("tables", "const int32_t"),
                ptr("batches", "const int32_t"),
                ptr("positions", "const int32_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("mask", "const uint8_t"),
                ptr("scores", "float"),
                u32_arg("width"),
                u32_arg("max_pages"),
                u64_arg("code_stride"),
                u64_arg("scale_stride"),
            ],
            format!(
                "chop::position_indexer::scores<{},{},{},{},{masked},{control}>(q,weights,codes,scales,tables,batches,positions,live_rows,mask,scores,width,max_pages,code_stride,scale_stride,__uint_as_float({scale}u));",
                q.heads, q.head_dim, q.page_size, q.compression_ratio
            ),
            SCORE_THREADS,
            1,
        );
        let select = EntrySpec::new(
            control_symbol(self.select_symbol(), suffix),
            vec![
                ptr("scores", "float"),
                ptr("tables", "const int32_t"),
                ptr("batches", "const int32_t"),
                ptr("positions", "const int32_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("mask", "const uint8_t"),
                ptr("output", "int32_t"),
                ptr("status", "int32_t"),
                u32_arg("width"),
                u32_arg("max_pages"),
                u32_arg("top_k"),
            ],
            format!(
                "chop::position_indexer::select<{},{},{},{masked},{control}>(scores,tables,batches,positions,live_rows,mask,output,status,width,max_pages,top_k);",
                q.page_size, q.compression_ratio, q.max_top_k
            ),
            SELECT_THREADS,
            1,
        );
        Ok(vec![score, select])
    }

    pub fn score_plan(&self, rows: u32, width: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        dimensions(rows, self.query.capacity, width)?;
        plan([width.div_ceil(4), rows, 1], SCORE_THREADS)
    }

    pub fn select_plan(&self, rows: u32, width: u32, top_k: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        dimensions(rows, self.query.capacity, width)?;
        if top_k == 0 || top_k > self.query.max_top_k {
            return Err(Refusal("position indexer top-k exceeds its entry"));
        }
        plan([rows, 1, 1], SELECT_THREADS)
    }
}

impl CandidateEntry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !name_valid(&self.name) || self.query.block_size > 128 || self.query.capacity > MAX_ROWS
        {
            return Err(Refusal(
                "candidate block geometry is outside the SM120 engine",
            ));
        }
        Ok(())
    }

    pub fn score_symbol(&self) -> String {
        format!("chop_candidate_blocks_{}_scores_v1", self.name)
    }
    pub fn mask_symbol(&self) -> String {
        format!("chop_candidate_blocks_{}_mask_v1", self.name)
    }
    pub fn guarantees(&self) -> Invariance {
        Invariance {
            ties: Ties::LowerIndex,
            ..Invariance::any_rows(self.query.capacity)
        }
    }
    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.specs_with_pin_control(false)
    }

    /// A lab control that pins count / block instead of (count - 1) / block.
    pub fn wrong_pin_specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.specs_with_pin_control(true)
    }

    fn specs_with_pin_control(&self, wrong_pin: bool) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let q = self.query;
        let suffix = if wrong_pin { "_wrong_pin" } else { "" };
        Ok(vec![
            EntrySpec::new(
                control_symbol(self.score_symbol(), suffix),
                vec![
                    ptr("scores", "const float"),
                    ptr("visible", "const int32_t"),
                    ptr("live_rows", "const uint32_t"),
                    ptr("blocks", "float"),
                    u32_arg("width"),
                ],
                format!(
                    "chop::position_indexer::block_scores<{}, {wrong_pin}>(scores,visible,live_rows,blocks,width);",
                    q.block_size
                ),
                SELECT_THREADS,
                1,
            ),
            EntrySpec::new(
                control_symbol(self.mask_symbol(), suffix),
                vec![
                    ptr("blocks", "const float"),
                    ptr("live_rows", "const uint32_t"),
                    ptr("output", "uint8_t"),
                    u32_arg("width"),
                ],
                format!(
                    "chop::position_indexer::block_mask<{},{}>(blocks,live_rows,output,width);",
                    q.block_size, q.top_k_blocks
                ),
                SELECT_THREADS,
                1,
            ),
        ])
    }

    pub fn workspace_bytes(&self, rows: u32, width: u32) -> Result<u64, Refusal> {
        self.validate()?;
        dimensions(rows, self.query.capacity, width)?;
        Ok(u64::from(rows) * u64::from(width.div_ceil(self.query.block_size)) * 4)
    }

    pub fn score_plan(&self, rows: u32, width: u32) -> Result<LaunchPlan, Refusal> {
        self.workspace_bytes(rows, width)?;
        plan(
            [
                width
                    .div_ceil(self.query.block_size)
                    .div_ceil(SELECT_THREADS),
                rows,
                1,
            ],
            SELECT_THREADS,
        )
    }

    pub fn mask_plan(&self, rows: u32, width: u32) -> Result<LaunchPlan, Refusal> {
        self.workspace_bytes(rows, width)?;
        plan([rows, 1, 1], SELECT_THREADS)
    }
}

fn dimensions(rows: u32, capacity: u32, width: u32) -> Result<(), Refusal> {
    if rows == 0 || rows > capacity || width == 0 || width > i32::MAX as u32 {
        return Err(Refusal(
            "position indexer rows or position capacity invalid",
        ));
    }
    Ok(())
}

fn plan(grid: [u32; 3], threads: u32) -> Result<LaunchPlan, Refusal> {
    let plan = LaunchPlan {
        grid,
        block: [threads, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

fn control_symbol(symbol: String, suffix: &str) -> String {
    let (base, version) = symbol.rsplit_once("_v").expect("versioned symbol");
    format!("{base}{suffix}_v{version}")
}
fn u32_arg(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}
fn u64_arg(name: &str) -> EntryParam {
    EntryParam::U64 { name: name.into() }
}

pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries,
    }
}
