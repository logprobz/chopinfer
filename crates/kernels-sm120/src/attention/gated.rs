//! Gated attention rows (`cuda/attention/gated.cuh`): the per-head Gemma
//! Q/K norm with interleaved M-RoPE and the paged KV write, which the dense
//! and QSA sparse attention both read, and the sigmoid output gate. Paged
//! decode folds the first into its `_rope_v1` entries
//! (`Variant::gated_fold`). A model declares its geometry ([`Config`]) and
//! lists the rows of each profile ([`entries`]) in its catalog.

use crate::invariance::Invariance;
use crate::{Catalogs, Profile};
use kernel_build::{EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, GatedAttentionQuery,
    InstantiationSpec, LaunchPlan, OpFamily, Refusal, StaticQuery, Target, Validate,
};

use super::GatedFold;

const THREADS: u32 = 256;

/// One profile's gated attention geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Q heads per rank.
    pub q_heads: u32,
    /// KV heads per rank.
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary: u32,
    pub theta: f32,
    /// Interleaved M-RoPE sections (T, H, W).
    pub sections: [u32; 3],
    pub eps: f32,
    pub page: u32,
    /// Rows of one launch: a prefill chunk.
    pub max_rows: u32,
}

impl Config {
    /// The decode fold of the same arithmetic.
    pub fn fold(&self) -> GatedFold {
        GatedFold {
            rotary: self.rotary,
            sections: self.sections,
            eps: self.eps,
        }
    }

    /// Elements of one packed projection row: each Q head's query and
    /// gate, then each KV head's key and value.
    pub const fn row_width(&self) -> u32 {
        self.q_heads * 2 * self.head_dim + self.kv_heads * 2 * self.head_dim
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub query: GatedAttentionQuery,
    pub symbol: String,
    pub params: Vec<EntryParam>,
    pub body: String,
    /// Row-local: each row's norm, RoPE, KV write and gate read only that
    /// row, so prefill chunks, other rows and the launch's row count leave
    /// its bits unchanged.
    pub invariance: Invariance,
}

impl Entry {
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.query.validate()?;
        check_query(&self.query)?;
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::GatedAttention(self.query),
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: self.head_dim() as u16,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (THREADS / 32) as u8,
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

    fn head_dim(&self) -> u32 {
        match self.query {
            GatedAttentionQuery::QkNormRopeKvWrite { head_dim, .. }
            | GatedAttentionQuery::OutputGate { head_dim, .. } => head_dim,
        }
    }

    pub(crate) fn capacity(&self) -> u32 {
        match self.query {
            GatedAttentionQuery::QkNormRopeKvWrite { capacity, .. }
            | GatedAttentionQuery::OutputGate { capacity, .. } => capacity,
        }
    }

    /// One CTA per prepared row; rows past the device live count return.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        self.plan_rows(prepared_rows)
    }

    /// [`Entry::plan`] of an entry whose contract was checked.
    pub(crate) fn plan_rows(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        if prepared_rows == 0 || prepared_rows > self.capacity() {
            return Err(Refusal("gated attention rows exceed prepared capacity"));
        }
        let plan = LaunchPlan {
            grid: [prepared_rows, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        if values.len() != self.params.len()
            || !self.params.iter().zip(&values).all(|(p, a)| {
                matches!(
                    (p, a),
                    (EntryParam::DevicePointer { .. }, Argument::DeviceBuffer(_))
                        | (EntryParam::U32 { .. }, Argument::U32(_))
                        | (EntryParam::U64 { .. }, Argument::U64(_))
                )
            })
        {
            return Err(Refusal("gated attention row argument ABI mismatch"));
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }

    pub fn entry_spec(&self) -> EntrySpec {
        EntrySpec::new(
            self.symbol.clone(),
            self.params.clone(),
            self.body.clone(),
            THREADS,
            1,
        )
    }
}

/// Refuses gated rows the SM120 templates do not instantiate: head widths
/// other than 128 and 256 (a warp holds one K or V head), and for the KV
/// write rotary widths other than 32 and 64, M-RoPE sections that do not
/// cover them, and more heads than eight warps hold.
pub fn check_query(query: &GatedAttentionQuery) -> Result<(), Refusal> {
    let head_dim = match *query {
        GatedAttentionQuery::QkNormRopeKvWrite { head_dim, .. }
        | GatedAttentionQuery::OutputGate { head_dim, .. } => head_dim,
    };
    if ![128, 256].contains(&head_dim) {
        return Err(Refusal("gated attention requires head dim 128 or 256"));
    }
    if let GatedAttentionQuery::QkNormRopeKvWrite {
        q_heads,
        kv_heads,
        rotary_dim,
        mrope,
        ..
    } = *query
        && (![32, 64].contains(&rotary_dim)
            || mrope.iter().sum::<u32>() != rotary_dim / 2
            || q_heads.div_ceil(8) + kv_heads > 8)
    {
        return Err(Refusal(
            "gated attention requires rotary dim 32 or 64 in M-RoPE sections and heads within eight warps",
        ));
    }
    Ok(())
}

fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}

fn u32_param(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}

fn u64_param(name: &str) -> EntryParam {
    EntryParam::U64 { name: name.into() }
}

/// Every model's gated rows in `catalogs`, in order.
pub fn catalog(catalogs: &Catalogs) -> Vec<Entry> {
    catalogs
        .models()
        .iter()
        .flat_map(|model| (model.gated)())
        .collect()
}

/// The gated rows of `profile` with geometry `config`: the norm, RoPE and KV
/// write for BF16 and FP8 caches, and the output gate, symbols
/// `chop_attn_{tag}_...`.
pub fn entries(profile: Profile, config: Config) -> Vec<Entry> {
    let Config {
        q_heads,
        kv_heads,
        head_dim,
        rotary,
        theta,
        sections,
        eps,
        page,
        max_rows,
    } = config;
    let tag = profile.tag();
    let mut entries = Vec::new();
    for (dtype, name) in [(DType::Bf16, "bf16"), (DType::Fp8E4M3, "fp8")] {
        entries.push(Entry {
            profile,
            query: GatedAttentionQuery::QkNormRopeKvWrite {
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim: rotary,
                theta_bits: theta.to_bits(),
                mrope: sections,
                eps_bits: eps.to_bits(),
                page_size: page,
                kv_dtype: dtype,
                capacity: max_rows,
            },
            symbol: format!("chop_attn_{tag}_qk_norm_rope_kv_{name}_v1"),
            params: vec![
                ptr("qkv", "const uint16_t"),
                ptr("q_norm", "const uint16_t"),
                ptr("k_norm", "const uint16_t"),
                ptr("rope_positions", "const int32_t"),
                ptr("cos_sin", "const float"),
                ptr("row_batch", "const int32_t"),
                ptr("positions", "const int32_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("block_tables", "const int32_t"),
                ptr("k_cache", "void"),
                ptr("v_cache", "void"),
                ptr("q", "uint16_t"),
                u32_param("max_pages"),
                u64_param("k_page_stride_bytes"),
                u64_param("v_page_stride_bytes"),
                u32_param("eps_bits"),
                u32_param("k_scale_bits"),
                u32_param("v_scale_bits"),
            ],
            body: format!(
                "chop::attention::gated::qk_norm_rope_kv<{q_heads},{kv_heads},{head_dim},{head_dim},{rotary},{},{},{page},{}>(qkv,q_norm,k_norm,rope_positions,cos_sin,row_batch,positions,live_rows,block_tables,k_cache,v_cache,q,max_pages,k_page_stride_bytes,v_page_stride_bytes,eps_bits,k_scale_bits,v_scale_bits);",
                sections[1],
                sections[2],
                dtype == DType::Fp8E4M3,
            ),
            invariance: Invariance::any_rows(max_rows),
        });
    }
    entries.push(Entry {
        profile,
        query: GatedAttentionQuery::OutputGate {
            heads: q_heads,
            head_dim,
            capacity: max_rows,
        },
        symbol: format!("chop_attn_{tag}_output_gate_v1"),
        params: vec![
            ptr("attention", "const uint16_t"),
            ptr("qkv", "const uint16_t"),
            ptr("out", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
        ],
        body: format!(
            "chop::attention::gated::output_gate<{q_heads},{kv_heads},{head_dim},{head_dim}>(attention,qkv,out,live_rows);"
        ),
        invariance: Invariance::any_rows(max_rows),
    });
    entries
}
