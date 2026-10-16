//! Linear-attention engine: depthwise causal convolution with
//! slot state, the gated delta rule for prefill, decode and verify, and the
//! per-head gated RMSNorm. Recurrent states live in pools addressed by slot
//! id, never in batch order; live row and sequence counts stay on the device.
//! See `cuda/linear/README.md` for the ABI and the numerics.

pub mod channel_decay;

use std::path::PathBuf;

use crate::invariance::Invariance;
use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, ConvActivation, GateActivation,
    InstantiationSpec, LaunchPlan, NumericalContract, OpFamily, RecurrentPhase, RecurrentQuery,
    Refusal, StaticQuery, Target, Validate,
};

/// Rows per prefill chunk of the delta rule and per convolution prefill CTA.
pub const CHUNK: u32 = 64;
/// Sequences one prefill launch may carry: every item a step of an instance
/// serving the package maximum may hold (docs/serving-profiles.md,
/// CONC-QWEN). A launch's grid and workspace hold one partial chunk per
/// sequence of its bucket on top of the row chunks, so a planner sizes its
/// workspace for the sequences it runs, not for this bound.
pub const PREFILL_SEQUENCES: u32 = 256;
/// Value rows of one head per delta-rule scan CTA.
pub const SCAN_ROWS: u32 = 16;
/// Value rows of one head per decode or verify CTA.
pub const RECURRENT_ROWS: u32 = 16;
/// F32 elements of a packed lower-triangular 64 x 64 matrix whose rows are
/// padded to multiples of four.
pub const TRIANGLE_FLOATS: u32 = 2176;
/// F32 workspace elements per (prefill chunk, value head): packed T' and P and
/// four 64-element scale vectors.
pub const WORK_FLOATS: u64 = 2 * TRIANGLE_FLOATS as u64 + 4 * 64;

const CONV_THREADS: u32 = 128;
const RECURRENT_THREADS: u32 = 128;
const PREFILL_THREADS: u32 = 256;
const NORM_THREADS: u32 = 128;

/// Prepared rows and sequences a launch plan is built for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bucket {
    pub rows: u32,
    pub sequences: u32,
}

impl Bucket {
    /// Prefill chunks: whole chunks of the rows plus one partial chunk per
    /// sequence.
    pub fn chunks(self) -> u32 {
        self.rows.div_ceil(CHUNK) + self.sequences
    }
}

/// Which kernel of an op an entry is. The chunked delta-rule prefill runs
/// `Prepare` then `Scan` over one workspace; every other op is `Single`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Stage {
    Single,
    Prepare,
    Scan,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    pub query: RecurrentQuery,
    pub stage: Stage,
    pub params: Vec<EntryParam>,
    /// Prefill entries: the convolution carries its window exactly, so any
    /// split keeps a row's bits; the delta rule keeps them over splits at
    /// whole `CHUNK`s. Both leave their state at anchor rows in spare
    /// entries. Verify entries are one-row exact over the exact verify
    /// steps (`verify_invariance`); decode entries claim nothing. The gated
    /// norm is row-local.
    pub invariance: Invariance,
}

/// The most taps of an SM120 convolution (the CUDA templates keep the
/// window in registers). The causal convolution's contract still refuses
/// more taps itself until its model migrates.
pub const MAX_CONV_TAPS: u32 = 8;

/// The most value heads of an SM120 gated delta rule: its scan, decode and
/// verify grids run one CTA row per value head on the grid's y dimension.
pub const MAX_VALUE_HEADS: u32 = 65_535;

/// The most rows per sequence of an SM120 verify launch, in either engine:
/// the verify templates hold every verified row's inputs in registers. The
/// contract takes any count.
pub const MAX_VERIFY_ROWS: u32 = 16;

/// Refuses a verify phase of more than `MAX_VERIFY_ROWS` rows.
pub fn check_phase(phase: RecurrentPhase) -> Result<(), Refusal> {
    match phase {
        RecurrentPhase::Verify { max_rows } if max_rows > MAX_VERIFY_ROWS => Err(Refusal(
            "SM120 recurrent verify launches take at most 16 rows per sequence",
        )),
        _ => Ok(()),
    }
}

/// Validates `query` against its contract, then against SM120's limits: a
/// convolution of at most `MAX_CONV_TAPS` taps, a verify phase of at most
/// `MAX_VERIFY_ROWS` rows, delta-rule key and value dims of 64, 128 or 256
/// (the CUDA templates' head dims; the launch plan's shared memory refuses
/// the dims a phase cannot fit) and at most `MAX_VALUE_HEADS` value heads.
pub fn limits(query: RecurrentQuery) -> Result<(), Refusal> {
    query.validate()?;
    if let Some(phase) = phase(query) {
        check_phase(phase)?;
    }
    match query {
        RecurrentQuery::CausalConv1d { kernel, .. } if kernel > MAX_CONV_TAPS => {
            Err(Refusal("SM120 causal convolutions take at most 8 taps"))
        }
        RecurrentQuery::GatedDeltaRule {
            key_dim, value_dim, ..
        } if !matches!(key_dim, 64 | 128 | 256) || !matches!(value_dim, 64 | 128 | 256) => Err(
            Refusal("SM120 gated delta rules take key and value dims of 64, 128 or 256"),
        ),
        RecurrentQuery::GatedDeltaRule { value_heads, .. } if value_heads > MAX_VALUE_HEADS => Err(
            Refusal("SM120 gated delta rules take at most 65535 value heads"),
        ),
        _ => Ok(()),
    }
}

fn capacity(query: RecurrentQuery) -> u32 {
    match query {
        RecurrentQuery::CausalConv1d { capacity, .. }
        | RecurrentQuery::GatedDeltaRule { capacity, .. }
        | RecurrentQuery::GatedRmsNorm { capacity, .. } => capacity,
    }
}

fn phase(query: RecurrentQuery) -> Option<RecurrentPhase> {
    match query {
        RecurrentQuery::CausalConv1d { phase, .. }
        | RecurrentQuery::GatedDeltaRule { phase, .. } => Some(phase),
        RecurrentQuery::GatedRmsNorm { .. } => None,
    }
}

impl Entry {
    fn threads(&self) -> u32 {
        match (self.query, self.stage) {
            (RecurrentQuery::CausalConv1d { .. }, _) => CONV_THREADS,
            (RecurrentQuery::GatedDeltaRule { .. }, Stage::Single) => RECURRENT_THREADS,
            (RecurrentQuery::GatedDeltaRule { .. }, _) => PREFILL_THREADS,
            (RecurrentQuery::GatedRmsNorm { .. }, _) => NORM_THREADS,
        }
    }

    /// SM120's refusals on top of the contract's math (`limits`).
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        limits(self.query)?;
        let family = match self.query {
            RecurrentQuery::GatedRmsNorm { .. } => OpFamily::Row,
            _ => OpFamily::Recurrent,
        };
        let (tile_m, tile_n) = match self.stage {
            Stage::Single => (1, 1),
            Stage::Prepare => (CHUNK as u16, CHUNK as u16),
            Stage::Scan => (CHUNK as u16, SCAN_ROWS as u16),
        };
        let threads = self.threads();
        let spec = InstantiationSpec {
            family,
            key: CompileKey {
                query: StaticQuery::Recurrent(self.query),
                knobs: CompileKnobs {
                    tile_m,
                    tile_n,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (threads / 32) as u8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol.clone(),
        };
        spec.validate()?;
        self.query.numerical_contract_id()?;
        Ok(spec)
    }

    /// The catalog's prepared maximum: rows, and sequences (the row capacity
    /// for decode, rows over `max_rows` for verify, `PREFILL_SEQUENCES` for
    /// prefill).
    pub fn capacity(&self) -> Bucket {
        let rows = capacity(self.query);
        let sequences = match phase(self.query) {
            Some(RecurrentPhase::Prefill) => PREFILL_SEQUENCES,
            Some(RecurrentPhase::Decode) | None => rows,
            Some(RecurrentPhase::Verify { max_rows }) => rows / max_rows,
        };
        Bucket { rows, sequences }
    }

    /// Workspace bytes shared by the prefill `Prepare` and `Scan` stages of a
    /// bucket: whole chunks of its rows plus one partial chunk per sequence.
    /// Each chunk holds, per value head, packed T' and P and the scales; per
    /// key head, the K and Q rows in the scan's padded layout; and per value
    /// head, the value columns.
    pub fn workspace_bytes(&self, bucket: Bucket) -> u64 {
        match (self.query, self.stage) {
            (
                RecurrentQuery::GatedDeltaRule {
                    key_heads,
                    value_heads,
                    key_dim,
                    value_dim,
                    ..
                },
                Stage::Prepare | Stage::Scan,
            ) => {
                // In u64: head counts up to `MAX_VALUE_HEADS` overflow u32.
                let per_chunk = u64::from(value_heads) * WORK_FLOATS * 4
                    + u64::from(key_heads) * u64::from(2 * CHUNK * (key_dim + 8) * 2)
                    + u64::from(value_heads) * u64::from(CHUNK * value_dim * 2);
                u64::from(bucket.chunks()) * per_chunk
            }
            _ => 0,
        }
    }

    pub fn dynamic_shared_bytes(&self) -> u32 {
        let (key_heads, value_heads, key_dim) = match self.query {
            RecurrentQuery::GatedDeltaRule {
                key_heads,
                value_heads,
                key_dim,
                ..
            } => (key_heads, value_heads, key_dim),
            _ => return 0,
        };
        let kq = 2 * CHUNK * (key_dim + 8) * 2;
        let matrix = CHUNK * (CHUNK + 4) * 4;
        let group = value_heads / key_heads;
        match self.stage {
            Stage::Single => 0,
            // K/Q rows, two Gram matrices, inverse norms, and per value head
            // beta, the F64 cumulative decay and its F32 hi and lo parts.
            Stage::Prepare => kq + 2 * matrix + 2 * CHUNK * 4 + group * CHUNK * 20 + 32,
            // Double-buffered K rows and scales, Q rows, the value slice,
            // packed T' and P, BF16 hi and lo splits of S and U, F32 Z/X and
            // U, and five transaction barriers.
            Stage::Scan => {
                3 * kq / 2
                    + CHUNK * SCAN_ROWS * 2
                    + 2 * TRIANGLE_FLOATS * 4
                    + 2 * 4 * CHUNK * 4
                    + 2 * SCAN_ROWS * (key_dim + 8) * 2
                    + 2 * SCAN_ROWS * (CHUNK + 8) * 2
                    + 3 * CHUNK * (SCAN_ROWS + 1) * 4
                    + 5 * 8
            }
        }
    }

    /// Launch plan for a prepared bucket at most the catalog capacity. The
    /// grid depends only on the bucket, never on live counts, which the
    /// kernels read on the device, so one plan serves every graph replay of
    /// the bucket; `live_rows` and `live_sequences` are only checked. Live
    /// work sits on the slowest grid axis so empty CTAs dispatch last.
    pub fn plan(
        &self,
        bucket: Bucket,
        live_rows: u32,
        live_sequences: u32,
    ) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let limit = self.capacity();
        if bucket.rows == 0
            || bucket.sequences == 0
            || bucket.rows > limit.rows
            || bucket.sequences > limit.sequences
        {
            return Err(Refusal("recurrent bucket exceeds the catalog capacity"));
        }
        if live_rows > bucket.rows || live_sequences > bucket.sequences {
            return Err(Refusal(
                "recurrent rows or sequences exceed the prepared bucket",
            ));
        }
        match phase(self.query) {
            Some(RecurrentPhase::Decode)
                if bucket.rows != bucket.sequences || live_rows != live_sequences =>
            {
                return Err(Refusal("decode takes one row per sequence"));
            }
            Some(RecurrentPhase::Verify { max_rows })
                if live_rows > live_sequences.saturating_mul(max_rows) =>
            {
                return Err(Refusal("verify rows exceed max_rows per sequence"));
            }
            _ => {}
        }
        let grid = match (self.query, self.stage) {
            (RecurrentQuery::CausalConv1d { channels, .. }, _) => {
                let blocks = channels.div_ceil(2 * CONV_THREADS);
                match phase(self.query) {
                    Some(RecurrentPhase::Prefill) => [blocks, bucket.chunks(), 1],
                    _ => [blocks, bucket.sequences, 1],
                }
            }
            (
                RecurrentQuery::GatedDeltaRule {
                    value_heads,
                    value_dim,
                    ..
                },
                Stage::Single,
            ) => [value_dim / RECURRENT_ROWS, value_heads, bucket.sequences],
            (RecurrentQuery::GatedDeltaRule { key_heads, .. }, Stage::Prepare) => {
                [key_heads, bucket.chunks(), 1]
            }
            (
                RecurrentQuery::GatedDeltaRule {
                    value_heads,
                    value_dim,
                    ..
                },
                Stage::Scan,
            ) => [value_dim / SCAN_ROWS, value_heads, bucket.sequences],
            (RecurrentQuery::GatedRmsNorm { .. }, _) => [bucket.rows, 1, 1],
        };
        let plan = LaunchPlan {
            grid,
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: self.dynamic_shared_bytes(),
            workspace_bytes: self.workspace_bytes(bucket),
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        if values.len() != self.params.len() {
            return Err(Refusal("linear-attention argument count mismatch"));
        }
        for (param, value) in self.params.iter().zip(&values) {
            if !matches!(
                (param, value),
                (
                    EntryParam::DevicePointer { .. },
                    Argument::DeviceBuffer(_) | Argument::NullDevicePointer
                ) | (EntryParam::U32 { .. }, Argument::U32(_))
            ) {
                return Err(Refusal("linear-attention argument ABI kind mismatch"));
            }
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }

    fn body(&self) -> String {
        match self.query {
            RecurrentQuery::CausalConv1d {
                channels,
                kernel,
                dilation,
                activation,
                phase,
                ..
            } => {
                let silu = activation == ConvActivation::Silu;
                let t = format!("{channels},{kernel},{dilation},{silu}");
                match phase {
                    RecurrentPhase::Prefill => format!(
                        "chop::linear::conv_prefill<{t}>(input,input_stride,weight,state,output,cu_seqlens,slot_ids,initial,anchor_ids,live_sequences);"
                    ),
                    RecurrentPhase::Decode => format!(
                        "chop::linear::conv_decode<{t}>(input,input_stride,weight,state,output,slot_ids,live_sequences);"
                    ),
                    RecurrentPhase::Verify { max_rows } => format!(
                        "chop::linear::conv_verify<{t},{max_rows}>(input,input_stride,weight,state,output,cu_rows,slot_ids,checkpoint_ids,live_sequences);"
                    ),
                }
            }
            RecurrentQuery::GatedDeltaRule {
                key_heads,
                value_heads,
                key_dim,
                phase,
                ..
            } => {
                let t = format!("{key_heads},{value_heads},{key_dim}");
                match (phase, self.stage) {
                    (RecurrentPhase::Prefill, Stage::Prepare) => format!(
                        "chop::linear::gated_delta_prepare<{t}>(qkv,qkv_stride,a,b,gate_stride,a_log,dt_bias,workspace,cu_seqlens,live_sequences);"
                    ),
                    (RecurrentPhase::Prefill, _) => format!(
                        "chop::linear::gated_delta_scan<{t}>(qkv,qkv_stride,workspace,state,output,cu_seqlens,slot_ids,initial,anchor_ids,live_sequences);"
                    ),
                    (RecurrentPhase::Decode, _) => format!(
                        "chop::linear::gated_delta_recurrent<{t},0>(qkv,qkv_stride,a,b,gate_stride,a_log,dt_bias,state,output,nullptr,slot_ids,nullptr,live_sequences);"
                    ),
                    (RecurrentPhase::Verify { max_rows }, _) => format!(
                        "chop::linear::gated_delta_recurrent<{t},{max_rows}>(qkv,qkv_stride,a,b,gate_stride,a_log,dt_bias,state,output,cu_rows,slot_ids,checkpoint_ids,live_sequences);"
                    ),
                }
            }
            RecurrentQuery::GatedRmsNorm {
                width,
                head_dim,
                eps_bits,
                gate,
                ..
            } => format!(
                "chop::linear::gated_rmsnorm<{width},{head_dim},{}>(input,gate,gate_stride,weight,output,live_rows,__uint_as_float({eps_bits}u));",
                gate == GateActivation::Sigmoid
            ),
        }
    }
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

/// The causal convolution entry ABI for a recurrent phase.
pub fn conv_params(phase: RecurrentPhase) -> Vec<EntryParam> {
    let mut params = vec![
        ptr("input", "const uint16_t"),
        u32_arg("input_stride"),
        ptr("weight", "const uint16_t"),
        ptr("state", "uint16_t"),
        ptr("output", "uint16_t"),
    ];
    match phase {
        RecurrentPhase::Prefill => params.extend([
            ptr("cu_seqlens", "const int32_t"),
            ptr("slot_ids", "const int32_t"),
            ptr("initial", "const int32_t"),
            ptr("anchor_ids", "const int32_t"),
        ]),
        RecurrentPhase::Decode => params.push(ptr("slot_ids", "const int32_t")),
        RecurrentPhase::Verify { .. } => params.extend([
            ptr("cu_rows", "const int32_t"),
            ptr("slot_ids", "const int32_t"),
            ptr("checkpoint_ids", "const int32_t"),
        ]),
    }
    params.push(ptr("live_sequences", "const uint32_t"));
    params
}

/// The gated delta-rule entry ABI for a recurrent phase and stage.
pub fn delta_params(phase: RecurrentPhase, stage: Stage) -> Vec<EntryParam> {
    let qkv = [ptr("qkv", "const uint16_t"), u32_arg("qkv_stride")];
    let gates = [
        ptr("a", "const uint16_t"),
        ptr("b", "const uint16_t"),
        u32_arg("gate_stride"),
        ptr("a_log", "const float"),
        ptr("dt_bias", "const float"),
    ];
    let mut params = qkv.to_vec();
    match (phase, stage) {
        (RecurrentPhase::Prefill, Stage::Prepare) => {
            params.extend(gates);
            params.extend([
                ptr("workspace", "float"),
                ptr("cu_seqlens", "const int32_t"),
            ]);
        }
        (RecurrentPhase::Prefill, _) => params.extend([
            ptr("workspace", "const float"),
            ptr("state", "float"),
            ptr("output", "uint16_t"),
            ptr("cu_seqlens", "const int32_t"),
            ptr("slot_ids", "const int32_t"),
            ptr("initial", "const int32_t"),
            ptr("anchor_ids", "const int32_t"),
        ]),
        (RecurrentPhase::Decode, _) => {
            params.extend(gates);
            params.extend([
                ptr("state", "float"),
                ptr("output", "uint16_t"),
                ptr("slot_ids", "const int32_t"),
            ]);
        }
        (RecurrentPhase::Verify { .. }, _) => {
            params.extend(gates);
            params.extend([
                ptr("state", "float"),
                ptr("output", "uint16_t"),
                ptr("cu_rows", "const int32_t"),
                ptr("slot_ids", "const int32_t"),
                ptr("checkpoint_ids", "const int32_t"),
            ]);
        }
    }
    params.push(ptr("live_sequences", "const uint32_t"));
    params
}

/// Every linear-attention entry contributed by the models in `catalogs`.
pub fn catalog(catalogs: &Catalogs) -> Vec<Entry> {
    catalogs
        .models()
        .iter()
        .flat_map(|model| (model.linear)())
        .collect()
}

/// Dynamic shared memory of a catalog symbol, for function configuration at
/// load time. Unknown symbols return `None`.
pub fn dynamic_shared_bytes(catalogs: &Catalogs, symbol: &str) -> Option<u32> {
    catalog(catalogs)
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .map(|entry| entry.dynamic_shared_bytes())
}

pub(crate) fn default_cubin_name(owner: Profile) -> String {
    if owner.ranks().is_some() && !owner.is_base() {
        format!("linear_{}_sm120", owner.tag())
    } else {
        format!("linear_{}_sm120", owner.as_str().replace('-', "_"))
    }
}

/// One cubin per profile with linear-attention entries.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    catalogs
        .profiles()
        .into_iter()
        .filter(|&owner| profile.is_none_or(|selected| selected == owner))
        .filter_map(|owner| {
            let entries = catalog(catalogs)
                .into_iter()
                .filter(|entry| entry.profile == owner)
                .map(|entry| {
                    let body = entry.body();
                    let threads = entry.threads();
                    EntrySpec::new(entry.symbol, entry.params, body, threads, 1)
                })
                .collect::<Vec<_>>();
            (!entries.is_empty()).then(|| CubinSpec {
                // `linear_qwen38_flash_sm120` at TP4, `linear_{tag}_sm120`
                // at other rank counts.
                name: catalogs
                    .models()
                    .iter()
                    .find(|model| model.profiles.contains(&owner))
                    .map_or_else(
                        || default_cubin_name(owner),
                        |model| (model.linear_cubin_name)(owner),
                    ),
                sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
                include_dirs: vec![],
                headers: vec![
                    "#include \"linear/conv.cuh\"".into(),
                    "#include \"linear/delta.cuh\"".into(),
                    "#include \"linear/delta_prefill.cuh\"".into(),
                    "#include \"linear/norm.cuh\"".into(),
                ],
                defines: vec![],
                entries,
            })
        })
        .collect()
}
