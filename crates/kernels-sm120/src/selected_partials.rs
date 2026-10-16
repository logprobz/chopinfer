//! Selected attention partials (`cuda/selected_partials/partials.cuh`,
//! `cuda/selected_partials/README.md`): block-sparse grouped-query attention
//! over the keys of each row's selected blocks and incomplete tail that one
//! context-parallel group rank owns, written as the unnormalized partials
//! `context_parallel`'s merge folds. Separate from the block-sparse
//! attention engine (`attention::sparse`), whose entries and bits stay as
//! they are at context-parallel size 1. Planners never allocate; live row
//! counts stay on the device and grids are sized for the prepared rows.

use std::path::PathBuf;

use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    AttentionPhase, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    NumericalContract, OpFamily, Refusal, StaticQuery, Target, Validate,
};

const THREADS: u32 = 256;
/// CUDA's largest grid Y: the grid is `[rows, kv_heads, 1]`.
const MAX_GRID_Y: u32 = 65_535;
/// Keys per tile of the running softmax (`SelectedPartials::tile`).
const TILE_KEYS: u16 = 64;

/// The attention shape of one profile's partial entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    /// Query heads per KV head.
    pub q_heads: u32,
    /// KV heads a rank holds (each page holds them all).
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Tokens of one physical page.
    pub page: u32,
    /// Tokens of one selectable block.
    pub block: u32,
    /// Blocks a row's selected list holds at most.
    pub top_blocks: u32,
}

impl Geometry {
    /// Refuses shapes the CUDA template does not instantiate.
    fn validate(&self) -> Result<(), Refusal> {
        if !(1..=8).contains(&self.q_heads)
            || self.kv_heads == 0
            || self.kv_heads > MAX_GRID_Y
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(32)
            || self.head_dim > 1024
            || !(1..=64).contains(&self.page)
            || self.block == 0
            || !self.page.is_multiple_of(self.block)
            || !(1..=2048).contains(&self.top_blocks)
        {
            return Err(Refusal(
                "selected attention partial geometry is outside the SM120 template",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    /// `{tag}_{kernel}`.
    pub name: String,
    pub geometry: Geometry,
    /// `DType::Bf16` or `DType::Fp8E4M3`.
    pub kv: DType,
    pub phase: AttentionPhase,
    /// Rows a launch may carry.
    pub capacity: u32,
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
        format!("chop_selected_partials_{}_v1", self.name)
    }

    pub fn query(&self) -> ContextParallelQuery {
        let g = self.geometry;
        ContextParallelQuery::SelectedAttentionPartials {
            q_heads: g.q_heads,
            kv_heads: g.kv_heads,
            head_dim: g.head_dim,
            page: g.page,
            block: g.block,
            top_blocks: g.top_blocks,
            kv: self.kv,
            phase: self.phase,
            capacity: self.capacity,
        }
    }

    /// Checks the query and the geometry against the engine.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.geometry.validate()?;
        let query = self.query();
        query.validate()?;
        query.numerical_contract_id()?;
        let spec = InstantiationSpec {
            family: OpFamily::Attention,
            key: CompileKey {
                query: StaticQuery::ContextParallel(query),
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: TILE_KEYS,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (THREADS / 32) as u8,
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

    /// Heads one rank writes: every query head of its KV heads.
    pub fn heads(&self) -> u32 {
        self.geometry.q_heads * self.geometry.kv_heads
    }

    /// Floats of the partials of `rows` rows: `m` and `l` `[rows, heads]`,
    /// then `o` `[rows, heads, head_dim]`.
    pub fn partial_floats(&self, rows: u32) -> u64 {
        u64::from(rows) * u64::from(self.heads()) * (u64::from(self.geometry.head_dim) + 2)
    }

    /// The launch for `prepared_rows` rows (a graph bucket, at most the
    /// capacity): one CTA per row and KV head. CTAs past the live row count
    /// return after the PDL wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        self.plan_rows(prepared_rows)
    }

    /// [`Entry::plan`] of an entry whose contract was checked.
    pub(crate) fn plan_rows(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        if prepared_rows == 0 || prepared_rows > self.capacity {
            return Err(Refusal(
                "selected attention partial rows exceed the prepared capacity",
            ));
        }
        let plan = LaunchPlan {
            grid: [prepared_rows, self.geometry.kv_heads, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Refuses a launch's group rank and size.
    pub fn check_rank(&self, cp_rank: u32, cp_size: u32) -> Result<(), Refusal> {
        if cp_size == 0 || cp_size > 8 || cp_rank >= cp_size {
            return Err(Refusal(
                "selected attention partial group rank is outside its group",
            ));
        }
        Ok(())
    }

    pub fn params(&self) -> Vec<EntryParam> {
        vec![
            pointer("q", "const uint16_t"),
            pointer("k_cache", "const uint8_t"),
            pointer("v_cache", "const uint8_t"),
            pointer("block_tables", "const int32_t"),
            pointer("row_batch", "const int32_t"),
            pointer("positions", "const int32_t"),
            pointer("live_rows", "const uint32_t"),
            pointer("selected", "const int32_t"),
            pointer("out_max", "float"),
            pointer("out_den", "float"),
            pointer("out_num", "float"),
            u32_param("max_pages"),
            u64_param("k_page_stride_bytes"),
            u64_param("v_page_stride_bytes"),
            u32_param("scale_bits"),
            u32_param("k_scale_bits"),
            u32_param("v_scale_bits"),
            u32_param("cp_rank"),
            u32_param("cp_size"),
        ]
    }

    fn body(&self) -> String {
        let g = self.geometry;
        format!(
            "chop::selected_partials::selected_partials<chop::selected_partials::SelectedPartials<\
             {},{},{},{},{},{},{}>>(q,k_cache,v_cache,block_tables,row_batch,positions,live_rows,\
             selected,out_max,out_den,out_num,max_pages,k_page_stride_bytes,v_page_stride_bytes,\
             scale_bits,k_scale_bits,v_scale_bits,cp_rank,cp_size);",
            g.q_heads,
            g.kv_heads,
            g.head_dim,
            g.page,
            g.block,
            g.top_blocks,
            self.kv == DType::Fp8E4M3
        )
    }

    pub fn entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        Ok(EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            THREADS,
            1,
        ))
    }
}

/// The served entries of geometry `g` named `{tag}_{kv}_{phase}` after
/// `profile`: BF16 and FP8 E4M3 KV, decode and verify up to 256 rows and
/// prefill up to 8192.
pub fn entries(profile: Profile, g: Geometry) -> Vec<Entry> {
    let mut entries = Vec::new();
    for (kv_name, kv) in [("bf16", DType::Bf16), ("fp8", DType::Fp8E4M3)] {
        for (phase_name, phase, capacity) in [
            ("decode", AttentionPhase::Decode, 256),
            ("verify", AttentionPhase::Verify, 256),
            ("prefill", AttentionPhase::Prefill, 8192),
        ] {
            entries.push(Entry {
                profile,
                name: format!("{}_{kv_name}_{phase_name}", profile.tag()),
                geometry: g,
                kv,
                phase,
                capacity,
            });
        }
    }
    entries
}

/// The cubin of one profile's entries, `selected_partials_{tag}_sm120`.
pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let specs = entries
        .iter()
        .filter(|entry| entry.profile == profile)
        .map(Entry::entry_spec)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CubinSpec {
        name: format!("selected_partials_{}_sm120", profile.tag()),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"selected_partials/partials.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "selected-partials-test",
        tag: "selected_partials_test",
    });

    /// A TP4 rank of six query heads over one KV head of 256, blocks of 4
    /// and 512 selected blocks.
    const GQA_RANK: Geometry = Geometry {
        q_heads: 6,
        kv_heads: 1,
        head_dim: 256,
        page: 64,
        block: 4,
        top_blocks: 512,
    };

    #[test]
    fn served_entries_validate_and_plan() {
        let entries = entries(PROFILE, GQA_RANK);
        assert_eq!(entries.len(), 6);
        for entry in &entries {
            entry.contract().unwrap();
            assert!(entry.plan(0).is_err());
            assert!(entry.plan(entry.capacity + 1).is_err());
            assert_eq!(
                entry.plan(entry.capacity).unwrap().grid,
                [entry.capacity, 1, 1]
            );
        }
        assert_eq!(
            entries[0].symbol(),
            "chop_selected_partials_selected_partials_test_bf16_decode_v1"
        );
        assert_eq!(entries[5].capacity, 8192);
        assert_eq!(entries[0].partial_floats(2), 2 * 6 * 258);
        assert!(
            entries[0]
                .body()
                .contains("SelectedPartials<6,1,256,64,4,512,false>")
        );
        assert!(
            entries[3]
                .body()
                .contains("SelectedPartials<6,1,256,64,4,512,true>")
        );
        entries[0].check_rank(3, 4).unwrap();
        assert!(entries[0].check_rank(4, 4).is_err());
        assert!(entries[0].check_rank(0, 0).is_err());
        let spec = cubin_spec(PROFILE, &entries).unwrap();
        assert_eq!(spec.name, "selected_partials_selected_partials_test_sm120");
        assert_eq!(spec.entries.len(), 6);
    }

    #[test]
    fn unsupported_shapes_refuse() {
        for g in [
            Geometry {
                q_heads: 9,
                ..GQA_RANK
            },
            Geometry {
                head_dim: 250,
                ..GQA_RANK
            },
            Geometry {
                page: 128,
                ..GQA_RANK
            },
            Geometry {
                kv_heads: 0,
                ..GQA_RANK
            },
            Geometry {
                kv_heads: MAX_GRID_Y + 1,
                ..GQA_RANK
            },
            Geometry {
                block: 3,
                ..GQA_RANK
            },
            Geometry {
                top_blocks: 0,
                ..GQA_RANK
            },
        ] {
            assert!(
                entries(PROFILE, g)
                    .iter()
                    .all(|entry| entry.contract().is_err())
            );
        }
    }
}
