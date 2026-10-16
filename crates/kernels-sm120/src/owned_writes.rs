//! Owner-routed cache writes under context parallelism
//! (`cuda/owned_writes/owned_writes.cuh`, `cuda/owned_writes/README.md`):
//! the gated Q/K norm, M-RoPE and KV write
//! (`ContextParallelQuery::OwnedQkNormRopeKvWrite`) and the block-sparse
//! indexer's key write (`ContextParallelQuery::OwnedIndexerKeyWrite`), each
//! storing a K/V row or block key only on the group rank that owns its page.
//! They run the served kernels' arithmetic and live in their own cubin, so
//! the served entries' bits and locks stay as they are at context-parallel
//! size 1.

use std::path::PathBuf;

use crate::Profile;
use crate::attention::{gated, sparse};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    CompileKey, CompileKnobs, DType, GatedAttentionQuery, InstantiationSpec, LaunchPlan,
    NumericalContract, OpFamily, Refusal, SparseAttentionQuery, StaticQuery, Target, Validate,
};

const KV_THREADS: u32 = 256;
/// Rows (warps) per key-write CTA.
const KEY_WRITE_WARPS: u32 = 4;
/// Rows of one indexer key write launch.
pub const KEY_WRITE_ROWS: u32 = 8192;

/// Which write an entry routes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// The gated norm, RoPE and K/V write over group-local positions.
    KvWrite { kv: DType },
    /// The indexer key write with owner-routed block keys.
    KeyWrite,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    /// `{tag}_{kernel}`.
    pub name: String,
    pub kind: Kind,
    pub gated: gated::Config,
    pub indexer: sparse::Geometry,
    /// The group size the entry serves.
    pub ranks: u32,
}

fn pointer(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}
fn u32_param(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}
fn u64_param(name: &str) -> EntryParam {
    EntryParam::U64 { name: name.into() }
}

impl Entry {
    pub fn symbol(&self) -> String {
        format!("chop_owned_writes_{}_v1", self.name)
    }

    /// Rows a launch may carry: a gated prefill chunk, or a key write's.
    pub fn capacity(&self) -> u32 {
        match self.kind {
            Kind::KvWrite { .. } => self.gated.max_rows,
            Kind::KeyWrite => KEY_WRITE_ROWS,
        }
    }

    /// The served write the entry routes.
    fn write(&self) -> StaticQueryWrite {
        let (g, i) = (self.gated, self.indexer);
        match self.kind {
            Kind::KvWrite { kv } => StaticQueryWrite::Kv(GatedAttentionQuery::QkNormRopeKvWrite {
                q_heads: g.q_heads,
                kv_heads: g.kv_heads,
                head_dim: g.head_dim,
                rotary_dim: g.rotary,
                theta_bits: g.theta.to_bits(),
                mrope: g.sections,
                eps_bits: g.eps.to_bits(),
                page_size: g.page,
                kv_dtype: kv,
                capacity: g.max_rows,
            }),
            Kind::KeyWrite => StaticQueryWrite::Key(SparseAttentionQuery::IndexerKeyWrite {
                head_dim: i.index_dim,
                rotary_dim: i.rotary_dim,
                theta_bits: i.theta.to_bits(),
                mrope: i.mrope,
                eps_bits: i.eps.to_bits(),
                block: i.block,
                ring: i.ring,
                capacity: KEY_WRITE_ROWS,
            }),
        }
    }

    pub fn query(&self) -> ContextParallelQuery {
        match self.write() {
            StaticQueryWrite::Kv(write) => ContextParallelQuery::OwnedQkNormRopeKvWrite {
                write,
                ranks: self.ranks,
            },
            StaticQueryWrite::Key(write) => ContextParallelQuery::OwnedIndexerKeyWrite {
                write,
                page: self.indexer.page_size,
                ranks: self.ranks,
            },
        }
    }

    /// Checks the query and the shapes the templates instantiate.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let (g, i) = (self.gated, self.indexer);
        let valid = match self.kind {
            Kind::KvWrite { kv } => {
                matches!(g.head_dim, 128 | 256)
                    && matches!(g.rotary, 32 | 64)
                    && g.sections.iter().sum::<u32>() == g.rotary / 2
                    && g.q_heads.div_ceil(8) + g.kv_heads <= 8
                    && matches!(kv, DType::Bf16 | DType::Fp8E4M3)
            }
            Kind::KeyWrite => {
                matches!(i.index_heads, 4 | 8 | 16)
                    && i.index_dim == 128
                    && matches!(i.rotary_dim, 32 | 64)
                    && i.block > 1
                    && i.ring >= i.block
                    && i.page_size.is_multiple_of(i.block)
                    // The `Indexer` traits the body names require a top-k.
                    && i.top_blocks > 0
            }
        };
        if !valid {
            return Err(Refusal(
                "owner-routed write geometry is outside the SM120 templates",
            ));
        }
        let query = self.query();
        query.validate()?;
        crate::context_parallel::check_query(&query)?;
        query.numerical_contract_id()?;
        let (tile_m, warps) = match self.kind {
            Kind::KvWrite { .. } => (1, KV_THREADS / 32),
            Kind::KeyWrite => (KEY_WRITE_WARPS as u16, KEY_WRITE_WARPS),
        };
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::ContextParallel(query),
                knobs: CompileKnobs {
                    tile_m,
                    tile_n: 128,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: warps as u8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol(),
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn threads(&self) -> u32 {
        match self.kind {
            Kind::KvWrite { .. } => KV_THREADS,
            Kind::KeyWrite => 32 * KEY_WRITE_WARPS,
        }
    }

    /// The launch for `prepared_rows` rows; CTAs past the live count return
    /// after the PDL wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if prepared_rows == 0 || prepared_rows > self.capacity() {
            return Err(Refusal(
                "owner-routed write rows exceed the prepared capacity",
            ));
        }
        let grid = match self.kind {
            Kind::KvWrite { .. } => [prepared_rows, 1, 1],
            Kind::KeyWrite => [prepared_rows.div_ceil(KEY_WRITE_WARPS), 1, 1],
        };
        let plan = LaunchPlan {
            grid,
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Refuses a key write's group rank and size: a rank of the entry's
    /// group size.
    pub fn check_rank(&self, cp_rank: u32, cp_size: u32) -> Result<(), Refusal> {
        if cp_size != self.ranks || cp_rank >= cp_size {
            return Err(Refusal(
                "owner-routed write group rank is outside its group",
            ));
        }
        Ok(())
    }

    /// The K/V write takes the served write's arguments with `positions`
    /// replaced by the group-local positions; the key write takes the served
    /// key write's and the group rank and size.
    pub fn params(&self) -> Vec<EntryParam> {
        match self.kind {
            Kind::KvWrite { .. } => vec![
                pointer("qkv", "const uint16_t"),
                pointer("q_norm", "const uint16_t"),
                pointer("k_norm", "const uint16_t"),
                pointer("rope_positions", "const int32_t"),
                pointer("cos_sin", "const float"),
                pointer("row_batch", "const int32_t"),
                pointer("local_positions", "const int32_t"),
                pointer("live_rows", "const uint32_t"),
                pointer("block_tables", "const int32_t"),
                pointer("k_cache", "void"),
                pointer("v_cache", "void"),
                pointer("q", "uint16_t"),
                u32_param("max_pages"),
                u64_param("k_page_stride_bytes"),
                u64_param("v_page_stride_bytes"),
                u32_param("eps_bits"),
                u32_param("k_scale_bits"),
                u32_param("v_scale_bits"),
            ],
            Kind::KeyWrite => vec![
                pointer("projection", "const uint16_t"),
                pointer("weight", "const uint16_t"),
                pointer("rope_positions", "const int32_t"),
                pointer("cos_sin", "const float"),
                pointer("row_batch", "const int32_t"),
                pointer("positions", "const int32_t"),
                pointer("live_rows", "const uint32_t"),
                pointer("slot_ids", "const int32_t"),
                pointer("ring_keys", "uint16_t"),
                pointer("ring_positions", "int32_t"),
                pointer("anchor_slots", "const int32_t"),
                pointer("anchor_ends", "const int32_t"),
                pointer("anchor_keys", "uint16_t"),
                pointer("anchor_positions", "int32_t"),
                pointer("block_tables", "const int32_t"),
                pointer("block_keys", "uint8_t"),
                u32_param("max_pages"),
                u64_param("block_page_stride_bytes"),
                u32_param("eps_bits"),
                u32_param("cp_rank"),
                u32_param("cp_size"),
            ],
        }
    }

    fn body(&self) -> String {
        let (g, i) = (self.gated, self.indexer);
        match self.kind {
            Kind::KvWrite { kv } => format!(
                "chop::owned_writes::owned_qk_norm_rope_kv<{},{},{},{},{},{},{},{},{}>(qkv,q_norm,\
                 k_norm,rope_positions,cos_sin,row_batch,local_positions,live_rows,block_tables,\
                 k_cache,v_cache,q,max_pages,k_page_stride_bytes,v_page_stride_bytes,eps_bits,\
                 k_scale_bits,v_scale_bits);",
                g.q_heads,
                g.kv_heads,
                g.head_dim,
                g.head_dim,
                g.rotary,
                g.sections[1],
                g.sections[2],
                g.page,
                kv == DType::Fp8E4M3,
            ),
            Kind::KeyWrite => format!(
                "chop::owned_writes::owned_index_key_write<chop::attention::sparse::Indexer<{},{},\
                 {},{},{},{},{},{},{}>>(projection,weight,rope_positions,cos_sin,row_batch,\
                 positions,live_rows,slot_ids,ring_keys,ring_positions,anchor_slots,anchor_ends,\
                 anchor_keys,anchor_positions,block_tables,block_keys,max_pages,\
                 block_page_stride_bytes,eps_bits,cp_rank,cp_size);",
                i.index_heads,
                i.index_dim,
                i.rotary_dim,
                i.mrope[1],
                i.mrope[2],
                i.block,
                i.ring,
                i.page_size,
                i.top_blocks,
            ),
        }
    }

    pub fn entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        Ok(EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            self.threads(),
            1,
        ))
    }
}

/// The served write an entry routes.
enum StaticQueryWrite {
    Kv(GatedAttentionQuery),
    Key(SparseAttentionQuery),
}

/// The owner-routed writes of one profile at group size `ranks`, named
/// `{tag}_kv_write_{bf16,fp8}` and `{tag}_index_key_write`.
pub fn entries(
    profile: Profile,
    gated: gated::Config,
    indexer: sparse::Geometry,
    ranks: u32,
) -> Vec<Entry> {
    let entry = |name: &str, kind| Entry {
        profile,
        name: format!("{}_{name}", profile.tag()),
        kind,
        gated,
        indexer,
        ranks,
    };
    vec![
        entry("kv_write_bf16", Kind::KvWrite { kv: DType::Bf16 }),
        entry("kv_write_fp8", Kind::KvWrite { kv: DType::Fp8E4M3 }),
        entry("index_key_write", Kind::KeyWrite),
    ]
}

/// The cubin of one profile's entries, `owned_writes_{tag}_sm120`.
pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let specs = entries
        .iter()
        .filter(|entry| entry.profile == profile)
        .map(Entry::entry_spec)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CubinSpec {
        name: format!("owned_writes_{}_sm120", profile.tag()),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"owned_writes/owned_writes.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "owned-writes-test",
        tag: "owned_writes_test",
    });

    /// A TP4 rank of six query heads over one KV head of 256 and a
    /// four-head indexer of 128, blocks of 4, 64-token pages.
    const GATED: gated::Config = gated::Config {
        q_heads: 6,
        kv_heads: 1,
        head_dim: 256,
        rotary: 64,
        theta: 1e7,
        sections: [11, 11, 10],
        eps: 1e-6,
        page: 64,
        max_rows: 8192,
    };

    const INDEXER: sparse::Geometry = sparse::Geometry {
        index_heads: 4,
        index_dim: 128,
        rotary_dim: 64,
        theta: 1e7,
        mrope: [11, 11, 10],
        eps: 1e-6,
        block: 4,
        ring: 16,
        top_blocks: 512,
        max_context: 262_144,
        q_heads: 6,
        kv_heads: 1,
        head_dim: 256,
        page_size: 64,
        softmax_scale: 0.0625,
    };

    #[test]
    fn entries_validate_and_plan() {
        let entries = entries(PROFILE, GATED, INDEXER, 2);
        assert_eq!(entries.len(), 3);
        for entry in &entries {
            entry.contract().unwrap();
            assert!(entry.plan(0).is_err());
            assert!(entry.plan(entry.capacity() + 1).is_err());
        }
        assert_eq!(entries[0].plan(300).unwrap().grid, [300, 1, 1]);
        assert_eq!(entries[2].plan(301).unwrap().grid, [76, 1, 1]);
        assert!(
            entries[0]
                .body()
                .contains("owned_qk_norm_rope_kv<6,1,256,256,64,11,10,64,false>")
        );
        assert!(entries[1].body().contains(",64,true>"));
        assert!(
            entries[2]
                .body()
                .contains("Indexer<4,128,64,11,10,4,16,64,512>")
        );
        entries[2].check_rank(1, 2).unwrap();
        assert!(entries[2].check_rank(2, 2).is_err());
        assert!(entries[2].check_rank(0, 4).is_err());
        let spec = cubin_spec(PROFILE, &entries).unwrap();
        assert_eq!(spec.name, "owned_writes_owned_writes_test_sm120");
        assert_eq!(spec.entries.len(), 3);
    }

    /// The owner-routed key write copies the served key write's control
    /// flow (`attention/sparse_indexer.cuh` `indexer_key_write`) and changes
    /// only its block-key destinations. The copied parts must stay the same
    /// source, whitespace aside: the key lambdas, both block sums, the leader
    /// search, and everything from the ring writes on.
    #[test]
    fn the_copied_key_write_control_flow_matches_the_served_one() {
        let root = crate::cuda_source_root();
        let read = |path: &str| std::fs::read_to_string(root.join(path)).unwrap();
        let served = read("attention/sparse_indexer.cuh");
        let owned = read("owned_writes/owned_writes.cuh");
        let body = |source: &str, function: &str| -> String {
            let start = source.find(function).unwrap();
            let rest = &source[start..];
            // The body ends at the first closing brace in column 0.
            rest[..rest.find("\n}\n").unwrap()].to_owned()
        };
        let served = body(&served, "__device__ void indexer_key_write(");
        let owned = body(&owned, "__device__ void owned_index_key_write(");
        let segment = |body: &str, from: &str, to: Option<&str>| -> String {
            let start = body.find(from).unwrap_or_else(|| panic!("no {from}"));
            let rest = &body[start..];
            let end = to.map_or(rest.len(), |to| {
                rest.find(to).unwrap_or_else(|| panic!("no {to}"))
            });
            rest[..end].split_whitespace().collect()
        };
        for (from, to) in [
            ("auto raw_key = ", Some("if (position % I::block")),
            ("for (int r = first; r <= row;", Some("write_block<I>(")),
            (
                "if (row > 0 && row_batch[row - 1] == batch) return;",
                Some("const int completing"),
            ),
            (
                "for (int p = start; p <= completing;",
                Some("write_block<I>("),
            ),
            ("__syncwarp();", None),
        ] {
            assert_eq!(
                segment(&served, from, to),
                segment(&owned, from, to),
                "the copied key write differs from the served one at {from}"
            );
        }
    }

    #[test]
    fn unsupported_shapes_refuse() {
        let bad_gated = gated::Config {
            q_heads: 64,
            ..GATED
        };
        assert!(
            entries(PROFILE, bad_gated, INDEXER, 2)[0]
                .contract()
                .is_err()
        );
        let bad_indexer = sparse::Geometry {
            page_size: 66,
            ..INDEXER
        };
        assert!(
            entries(PROFILE, GATED, bad_indexer, 2)[2]
                .contract()
                .is_err()
        );
        let no_top = sparse::Geometry {
            top_blocks: 0,
            ..INDEXER
        };
        let key_write = &entries(PROFILE, GATED, no_top, 2)[2];
        assert!(key_write.contract().is_err());
        assert!(key_write.plan(4).is_err());
        assert!(key_write.entry_spec().is_err());
        assert!(entries(PROFILE, GATED, INDEXER, 1)[0].contract().is_err());
    }
}
