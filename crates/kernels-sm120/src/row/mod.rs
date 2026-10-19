//! SM120 row kernels. Static geometry is compiled once per operation; launch
//! grids select prepared token capacity buckets without changing the cubin.

pub mod argmax;
pub mod block_quantize;
pub mod builders;
pub mod copy;
pub mod gather;
pub mod hyper_connection;
pub mod mxfp8_decode;
pub mod ngram_hash;
pub mod ordered_sum;
pub mod precision;
pub mod replicated_embedding;
pub mod rmsnorm;
pub mod rmsnorm_prefill;
pub mod rmsnorm_quantize;
pub mod rmsnorm_quantize_cached;
pub mod rmsnorm_quantize_tiles;
pub mod rmsnorm_tree;
pub mod rounding;
pub mod silu_mul_quantize;

use crate::invariance::Invariance;
use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    OpFamily, OutputQuantization, Refusal, RmsNormKind, RotarySource, RouterKind, RowQuery,
    RuntimeCounts, ScaleMode, StaticQuery, SwigluColumns, Target, Validate,
};
use std::path::PathBuf;
use std::sync::OnceLock;

const TUNING_JSON: &str = include_str!("../../tuning/row.json");

#[derive(Debug)]
struct RowTuning {
    buckets: Vec<u32>,
    row_threads: u32,
    quant_threads: u32,
    pdl: bool,
}

fn number_after(key: &str) -> u32 {
    TUNING_JSON
        .split_once(&format!("\"{key}\""))
        .and_then(|(_, rest)| rest.split_once(':'))
        .and_then(|(_, rest)| {
            rest.trim_start()
                .split(|c: char| !c.is_ascii_digit())
                .next()
        })
        .and_then(|digits| digits.parse().ok())
        .unwrap_or_else(|| panic!("invalid row tuning number: {key}"))
}

fn bucket_values(key: &str) -> Vec<u32> {
    let (_, tail) = TUNING_JSON
        .split_once(&format!("\"{key}\""))
        .unwrap_or_else(|| panic!("missing row tuning bucket: {key}"));
    let (_, tail) = tail
        .split_once('[')
        .expect("row tuning bucket opening bracket");
    let (values, _) = tail
        .split_once(']')
        .expect("row tuning bucket closing bracket");
    values
        .split(',')
        .map(|value| value.trim().parse().expect("row tuning bucket integer"))
        .collect()
}

fn tuning() -> &'static RowTuning {
    static TABLE: OnceLock<RowTuning> = OnceLock::new();
    TABLE.get_or_init(|| {
        assert!(TUNING_JSON.contains("\"compute_capability\": \"12.0\""));
        assert_eq!(number_after("sm_count"), 188);
        let mut buckets = bucket_values("decode");
        buckets.extend(bucket_values("prefill"));
        let row_threads = number_after("row_threads");
        let quant_threads = number_after("quant_threads");
        assert!(row_threads.is_multiple_of(32) && row_threads <= 256);
        assert!(quant_threads.is_multiple_of(32) && quant_threads <= 256);
        assert!(TUNING_JSON.contains("\"pdl\": true"));
        RowTuning {
            buckets,
            row_threads,
            quant_threads,
            pdl: true,
        }
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Grid {
    Rows,
    RowsHeads(u32),
    RowsGroups(u32),
    Experts(u32),
    PaddedRows {
        experts: u32,
        top_k: u32,
        block: u32,
    },
    RouteTiles {
        top_k: u32,
        tile: u32,
    },
    RingTail {
        ring: u32,
        heads: u32,
    },
    One,
}

#[derive(Clone, Debug)]
pub struct RowInstantiation {
    pub profile: Profile,
    pub symbol: String,
    pub query: RowQuery,
    pub width: u32,
    pub grid: Grid,
    pub threads: u32,
    pub params: Vec<EntryParam>,
    pub body: String,
    /// The selected entry's declared invariance.
    pub invariance: Invariance,
}

/// The declaration of a row-local entry: every row has the bits it has
/// alone, at any row count up to the largest capacity bucket.
pub fn row_local() -> Invariance {
    Invariance::any_rows(
        tuning()
            .buckets
            .iter()
            .copied()
            .max()
            .expect("row capacity buckets"),
    )
}

impl RowInstantiation {
    /// Instantiate an explicitly selected row with the shared row tuning.
    pub fn new(
        profile: Profile,
        name: &str,
        query: RowQuery,
        width: u32,
        launch: (Grid, u32),
        params: Vec<EntryParam>,
        body: String,
    ) -> Self {
        Self {
            profile,
            symbol: format!("chop_row_{name}_v1"),
            query,
            width,
            grid: launch.0,
            threads: if matches!(query, RowQuery::Quantize { .. }) && launch.1 == 32 {
                tuning().quant_threads
            } else if launch.1 == 256 {
                tuning().row_threads
            } else {
                launch.1
            },
            params,
            body,
            invariance: Invariance::NONE,
        }
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.query.validate()?;
        let dtype = match self.query {
            RowQuery::RingWrite { kv_dtype, .. } => kv_dtype,
            RowQuery::Router { .. }
            | RowQuery::LogitsArgmax { .. }
            | RowQuery::ShardedArgmaxFinal { .. }
            | RowQuery::LogitsTopK { .. } => DType::F32,
            RowQuery::ShardedArgmaxPartial { logits, .. } => logits,
            RowQuery::ExpertPermute { .. } => DType::I32,
            _ => DType::Bf16,
        };
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::Row {
                    family: "row",
                    width_capacity: self.width,
                    dtype,
                    traits_digest: static_traits_digest(&self.body),
                },
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: self.threads as u16,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (self.threads / 32) as u8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol.clone(),
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn plan(&self, capacity: u32, live: RuntimeCounts) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if !capacity_bucket(capacity) || live.m > capacity {
            return Err(Refusal("row token count exceeds prepared capacity bucket"));
        }
        let x = match self.grid {
            Grid::Rows | Grid::RowsHeads(_) | Grid::RowsGroups(_) => capacity,
            Grid::Experts(experts) => experts,
            Grid::RouteTiles { top_k, tile } => capacity
                .checked_mul(top_k)
                .ok_or(Refusal("route tile capacity overflows"))?
                .div_ceil(tile),
            Grid::RingTail { ring, .. } => capacity.min(ring),
            Grid::PaddedRows {
                experts,
                top_k,
                block,
            } => capacity
                .checked_mul(top_k)
                .and_then(|n| n.checked_add(experts.saturating_mul(block.checked_sub(1)?)))
                .ok_or(Refusal("padded row capacity overflows"))?,
            Grid::One => 1,
        };
        let y = match self.grid {
            Grid::RowsHeads(heads) | Grid::RowsGroups(heads) => heads,
            Grid::RingTail { heads, .. } => heads,
            _ => 1,
        };
        // Below 512 rows the RoPE rows run a block per Q head; their grid
        // names the heads in groups of four.
        let y = if self.symbol.starts_with("chop_row_rope_") && capacity < 512 {
            y.checked_mul(4)
                .ok_or(Refusal("RoPE head grid overflows"))?
        } else {
            y
        };
        let workspace_bytes = match self.query {
            RowQuery::Quantize {
                format: OutputQuantization::Nvfp4,
                ..
            }
            | RowQuery::RmsNorm {
                output_quantization: Some(OutputQuantization::Nvfp4),
                ..
            } => u64::from(capacity) * 4 + 4,
            RowQuery::ExpertPermute { experts, .. }
                if self.symbol.starts_with("chop_row_permute_tile_") =>
            {
                u64::from(experts) * 256 * 4 * 2
            }
            RowQuery::ExpertPermute { experts, .. } => u64::from(experts) * 4,
            _ => 0,
        };
        let plan = LaunchPlan {
            grid: [x, y, 1],
            block: [
                if self.symbol.starts_with("chop_row_permute_count_") && capacity < 512 {
                    256
                } else {
                    self.threads
                },
                1,
                1,
            ],
            dynamic_shared_bytes: 0,
            workspace_bytes,
            cluster: [1, 1, 1],
            pdl: tuning().pdl,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Keep the launch argument order identical to the generated CUDA entry.
    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        pack_arguments(&self.params, values)
    }
}

fn pack_arguments(params: &[EntryParam], values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
    if values.len() != params.len() {
        return Err(Refusal("row argument count mismatch"));
    }
    for (param, value) in params.iter().zip(&values) {
        let matches = matches!(
            (param, value),
            (
                EntryParam::DevicePointer { .. },
                Argument::DeviceBuffer(_) | Argument::NullDevicePointer
            ) | (EntryParam::U32 { .. }, Argument::U32(_))
                | (EntryParam::U64 { .. }, Argument::U64(_))
                | (EntryParam::I32 { .. }, Argument::I32(_))
                | (EntryParam::TensorMap { .. }, Argument::TensorMap(_))
                | (
                    EntryParam::PointerArray { .. },
                    Argument::DevicePointerArray(_)
                )
        );
        if !matches {
            return Err(Refusal("row argument ABI kind mismatch"));
        }
    }
    let args = ArgumentList(values);
    args.validate()?;
    Ok(args)
}

pub fn capacity_bucket(capacity: u32) -> bool {
    tuning().buckets.contains(&capacity)
}

/// The smallest capacity bucket that holds `rows` live rows. Row kernels
/// return for rows at or past their device live count, so a launch at this
/// bucket writes what a launch at any larger bucket writes, without the
/// larger grid's empty CTAs.
pub fn smallest_bucket(rows: u32) -> Option<u32> {
    tuning()
        .buckets
        .iter()
        .copied()
        .filter(|&bucket| bucket >= rows)
        .min()
}

fn static_traits_digest(body: &str) -> u64 {
    body.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}
fn u32_arg(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}
fn u64_arg(name: &str) -> EntryParam {
    EntryParam::U64 { name: name.into() }
}
#[expect(
    clippy::too_many_arguments,
    reason = "selected rows carry their explicit ABI and geometry"
)]
fn add(
    rows: &mut Vec<RowInstantiation>,
    profile: Profile,
    name: &str,
    query: RowQuery,
    width: u32,
    launch: (Grid, u32),
    params: Vec<EntryParam>,
    body: String,
) {
    rows.push(RowInstantiation::new(
        profile, name, query, width, launch, params, body,
    ));
}

/// Why the standard catalog's legacy rows build.
const STANDARD: &str = "the standard catalog's geometry fits its row templates";

/// The standard profile's rows.
fn standard_catalog() -> Vec<RowInstantiation> {
    use builders::*;
    let profile = Profile::StdDenseMoe;
    let mut rows = Vec::new();
    for (name, width, head, kind) in [
        ("plain4096", 4096, 0, 0),
        ("plain8192", 8192, 0, 0),
        ("gemma4096", 4096, 0, 1),
        ("qk128", 128, 128, 2),
    ] {
        rows.push(rmsnorm_entry(profile, name, width, head, kind, false).expect(STANDARD));
    }
    rows.extend(quantize_entries(profile, 8192, "8192").expect(STANDARD));
    for (name, kind, grouped) in [("std_softmax128", 0, false), ("std_grouped128", 2, true)] {
        rows.push(router_entry(profile, name, 128, kind, grouped).expect(STANDARD));
    }
    for (name, block) in [
        ("std_b16", 16),
        ("std_b32", 32),
        ("std", 64),
        ("std_b128", 128),
    ] {
        rows.extend(expert_permute_entries(profile, name, 128, block).expect(STANDARD));
    }
    for (name, width, preweighted) in [("std4096", 4096, false), ("std8192", 8192, true)] {
        rows.push(expert_combine_entry(profile, name, width, 128, preweighted).expect(STANDARD));
    }
    let mut shared = quantize_entries(profile, 4096, "4096").expect(STANDARD);
    shared.push(nvfp4_global_scale(profile).expect(STANDARD));
    for (name, sharded) in [("full", false), ("shard", true)] {
        shared.push(embedding(
            profile,
            &format!("embedding_{name}"),
            152576,
            4096,
            sharded,
        ));
    }
    for width in [512, 1536, 3584, 4096, 7168] {
        shared.extend(swiglu_entries(profile, width).expect(STANDARD));
    }
    for (name, vocab, k) in [
        ("argmax_full", 152576, 1),
        ("top20_full", 152576, 20),
        ("argmax_shard", 38144, 1),
        ("top20_shard", 38144, 20),
    ] {
        shared.push(logits(profile, name, vocab, k));
    }
    for row in &mut shared {
        row.symbol = row.symbol.replace("_v1", "_std_v1");
    }
    rows.extend(shared);
    rows
}

/// The kernel base's model-neutral rows.
pub fn own_catalog() -> Vec<RowInstantiation> {
    standard_catalog()
}

/// Every row of `catalogs`: the kernel base's, then each model's.
pub fn catalog(catalogs: &Catalogs) -> Vec<RowInstantiation> {
    let mut rows = standard_catalog();
    for model in catalogs.models() {
        rows.extend((model.rows)());
    }
    rows
}

/// The row cubin of `profile`'s rows in `catalogs`, or of every profile's
/// for `None`.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda");
    vec![CubinSpec {
        name: "row_engine".into(),
        sources_root: root,
        include_dirs: vec![],
        headers: vec![
            "#include \"row/row.cuh\"".into(),
            "#include \"row/router.cuh\"".into(),
            "#include \"row/rope_kv.cuh\"".into(),
            "#include \"row/quantize.cuh\"".into(),
            "#include \"row/logits.cuh\"".into(),
            "#include \"row/dflash.cuh\"".into(),
        ],
        defines: vec![],
        entries: catalog(catalogs)
            .into_iter()
            .filter(|item| profile.is_none_or(|profile| item.profile == profile))
            .map(|item| {
                let accurate_rope = matches!(
                    item.query,
                    RowQuery::Rope {
                        rotary_dim: 64,
                        source: RotarySource::Computed,
                        ..
                    }
                );
                let mut entry =
                    EntrySpec::new(item.symbol, item.params, item.body, item.threads, 1);
                // Double sincos uses a stack frame without register spills.
                if accurate_rope {
                    entry.allowed_local_bytes = 40;
                }
                entry
            })
            .collect(),
    }]
}

/// The embedding gather of a `vocab` x `width` table into `width`-wide
/// rows, named `chop_row_{name}_v1`; a `sharded` table holds one rank's
/// vocabulary shard.
pub fn embedding(
    profile: Profile,
    name: &str,
    vocab: u32,
    width: u32,
    sharded: bool,
) -> RowInstantiation {
    let mut rows = Vec::new();
    add(
        &mut rows,
        profile,
        name,
        RowQuery::EmbeddingGather { vocab, width },
        width,
        (Grid::Rows, 256),
        vec![
            ptr("table", "const uint16_t"),
            ptr("ids", "const int32_t"),
            ptr("output", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
            u32_arg("vocab_offset"),
            u32_arg("shard_vocab"),
        ],
        format!(
            "chop::row::embedding<{width},{sharded}>(table,ids,output,live_rows,vocab_offset,shard_vocab);"
        ),
    );
    rows.remove(0)
}

/// The argmax (`k` = 1) or top-`k` F32 log-probabilities over `vocab`
/// logits per row, named `chop_row_logits_{name}_v1`.
pub fn logits(profile: Profile, name: &str, vocab: u32, k: u32) -> RowInstantiation {
    let mut rows = Vec::new();
    add(
        &mut rows,
        profile,
        &format!("logits_{name}"),
        if k == 1 {
            RowQuery::LogitsArgmax { vocab }
        } else {
            RowQuery::LogitsTopK { vocab, k }
        },
        vocab,
        (Grid::Rows, 1024),
        vec![
            ptr("logits", "const float"),
            ptr("ids", "int32_t"),
            ptr("log_probs", "float"),
            ptr("live_rows", "const uint32_t"),
            ptr("live_width", "const uint32_t"),
        ],
        format!("chop::row::logits_topk<{vocab},{k}>(logits,ids,log_probs,live_rows,live_width);"),
    );
    rows.remove(0)
}
