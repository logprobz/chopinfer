//! Native window plus paged compressed latent attention. Every phase uses
//! the same BF16 QK/PV path over exact decoded operands, fixed 64-key tiles,
//! and an explicit split count. The final merge owns the sink and inverse
//! tail rotation. The original latent engine's source closure is unchanged.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    AttentionPhase, CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan, NumericalContract,
    OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat};
use kernel_core::properties::Invariance;

use crate::Profile;

pub const HEADER: &str = "latent/sources.cuh";
pub const KEYS: u32 = 64;
pub const MERGE_THREADS: u32 = 128;

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub name: String,
    pub query: LatentAttentionQuery,
    /// Fixed for every row count; decode and verify use the same count.
    pub splits: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub heads: u32,
    pub latent: u32,
    pub page: u32,
    pub max_keys: u32,
    pub scale_bits: u32,
    pub second: bool,
    pub sink: bool,
    pub tail: u32,
    pub partial: bool,
    pub capacity: u32,
}

pub(crate) fn shape(query: LatentAttentionQuery) -> Result<Shape, Refusal> {
    query.validate()?;
    let LatentAttentionQuery::Attention {
        heads,
        latent,
        page_size,
        max_keys,
        softmax_scale_bits,
        kv,
        sink,
        tail_rope_dims,
        second_fp4_keys,
        output,
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("native latent sources require an attention query"));
    };
    if kv != LatentKvFormat::Fp8E4M3Ue8M0Block32 {
        return Err(Refusal(
            "native latent sources require the FP8 UE8M0 window",
        ));
    }
    if heads > 128 || latent > 512 || !page_size.is_power_of_two() || max_keys > 1_048_576 {
        return Err(Refusal(
            "native latent sources exceed SM120 head, width, page or key limits",
        ));
    }
    Ok(Shape {
        heads,
        latent,
        page: page_size,
        max_keys,
        scale_bits: softmax_scale_bits,
        second: second_fp4_keys,
        sink,
        tail: tail_rope_dims,
        partial: output == LatentAttentionOutput::Partial,
        capacity,
    })
}

fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}
fn u32p(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}
fn u64p(name: &str) -> EntryParam {
    EntryParam::U64 { name: name.into() }
}

impl Entry {
    pub fn symbol(&self) -> String {
        format!("chop_latent_sources_{}_v1", self.name)
    }
    pub fn merge_symbol(&self) -> String {
        format!("chop_latent_sources_{}_merge_v1", self.name)
    }

    /// Lab-only negative control: add the sink once per split instead of
    /// once per query. Serving registrations never include this entry.
    pub fn sink_per_split_entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        let s = shape(self.query)?;
        if s.partial || !s.sink || self.splits < 2 {
            return Err(Refusal(
                "per-split sink control requires a normalized sink and multiple splits",
            ));
        }
        let traits = self.traits()?;
        let rope = if s.tail == 0 { "nullptr" } else { "cos_sin" };
        Ok(EntrySpec::new(
            format!("chop_latent_sources_{}_sink_per_split_merge_v1", self.name),
            self.merge_params()?,
            format!(
                "chop::latent::sources::merge<{traits},true>(workspace,live_rows,sinks,{rope},out,nullptr,nullptr,nullptr);"
            ),
            MERGE_THREADS,
            1,
        ))
    }
    pub fn capacity(&self) -> u32 {
        shape(self.query).map_or(0, |s| s.capacity)
    }

    pub fn dynamic_shared_bytes(&self) -> u32 {
        shape(self.query).map_or(0, |s| {
            KEYS * s.latent * 2 + 8 * s.latent * 2 + (s.latent / 128) * KEYS * 8 * 4 + KEYS * 20
        })
    }

    pub fn workspace_bytes(&self, rows: u32) -> u64 {
        shape(self.query).map_or(0, |s| {
            u64::from(rows)
                * u64::from(self.splits)
                * u64::from(s.heads)
                * (u64::from(s.latent) + 2)
                * 4
        })
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let s = shape(self.query)?;
        if self.splits == 0 || self.splits > 64 || self.dynamic_shared_bytes() > 101_376 {
            return Err(Refusal(
                "native latent split count or shared memory exceeds SM120 limits",
            ));
        }
        self.query.numerical_contract_id()?;
        let spec = InstantiationSpec {
            family: OpFamily::Attention,
            key: CompileKey {
                query: StaticQuery::LatentAttention(self.query),
                knobs: CompileKnobs {
                    tile_m: 8,
                    tile_n: KEYS as u16,
                    tile_k: 128,
                    stages: 1,
                    swizzle: 0,
                    warps: (s.latent / 128) as u8,
                    split_k_capacity: self.splits as u8,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol(),
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn invariance(&self) -> Invariance {
        if self.contract().is_err() {
            return Invariance::NONE;
        }
        // All row counts keep the same key partition. Decode and verify
        // entries with equal splits instantiate identical arithmetic.
        Invariance::any_rows(self.capacity())
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let s = shape(self.query)?;
        if rows == 0 || rows > s.capacity {
            return Err(Refusal("native latent rows exceed entry capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows, s.heads.div_ceil(8), self.splits],
            block: [s.latent / 128 * 32, 1, 1],
            dynamic_shared_bytes: self.dynamic_shared_bytes(),
            workspace_bytes: self.workspace_bytes(rows),
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn merge_plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        let mut plan = self.plan(rows)?;
        plan.grid = [rows, shape(self.query)?.heads, 1];
        plan.block = [MERGE_THREADS, 1, 1];
        plan.dynamic_shared_bytes = 0;
        plan.validate()?;
        Ok(plan)
    }

    pub fn params(&self) -> Result<Vec<EntryParam>, Refusal> {
        let s = shape(self.query)?;
        let mut params = vec![
            ptr("q", "const uint16_t"),
            ptr("row_batch", "const int32_t"),
            ptr("live_rows", "const uint32_t"),
            ptr("window_pool", "const uint8_t"),
            ptr("window_keys", "const int32_t"),
            ptr("workspace", "float"),
            u64p("window_stride"),
            u32p("window_slots"),
            u32p("window_sequences"),
            u32p("window_key_stride"),
            u32p("window_budget"),
            u32p("scale_bits"),
        ];
        if s.second {
            params.extend([
                ptr("compressed_pool", "const uint8_t"),
                ptr("compressed_keys", "const int32_t"),
                ptr("block_tables", "const int32_t"),
                u64p("page_stride"),
                u32p("max_pages"),
                u32p("compressed_pages"),
                u32p("compressed_sequences"),
                u32p("compressed_key_stride"),
                u32p("compressed_budget"),
            ]);
        }
        Ok(params)
    }

    pub fn merge_params(&self) -> Result<Vec<EntryParam>, Refusal> {
        let s = shape(self.query)?;
        let mut params = vec![
            ptr("workspace", "const float"),
            ptr("live_rows", "const uint32_t"),
        ];
        if s.partial {
            params.extend([
                ptr("out_max", "float"),
                ptr("out_den", "float"),
                ptr("out_num", "float"),
            ]);
        } else {
            params.push(ptr("out", "uint16_t"));
            if s.sink {
                params.push(ptr("sinks", "const float"));
            }
            if s.tail != 0 {
                params.push(ptr("cos_sin", "const float"));
            }
        }
        Ok(params)
    }

    fn traits(&self) -> Result<String, Refusal> {
        let s = shape(self.query)?;
        Ok(format!(
            "chop::latent::sources::Attention<{},{},{},{},{},{},{},{},{}>",
            s.heads, s.latent, s.page, s.max_keys, self.splits, s.second, s.sink, s.tail, s.partial
        ))
    }

    pub fn entry_specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.contract()?;
        let s = shape(self.query)?;
        let traits = self.traits()?;
        let sources = if s.second {
            "compressed_pool,compressed_keys,block_tables"
        } else {
            "nullptr,nullptr,nullptr"
        };
        let geometry = if s.second {
            "page_stride,max_pages,compressed_pages,compressed_sequences,compressed_key_stride,compressed_budget"
        } else {
            "0,0,0,0,0,0"
        };
        let output = if s.partial {
            "nullptr,out_max,out_den,out_num"
        } else {
            "out,nullptr,nullptr,nullptr"
        };
        let sinks = if s.sink && !s.partial {
            "sinks"
        } else {
            "nullptr"
        };
        let rope = if s.tail != 0 && !s.partial {
            "cos_sin"
        } else {
            "nullptr"
        };
        Ok(vec![
            EntrySpec::new(
                self.symbol(),
                self.params()?,
                format!(
                    "chop::latent::sources::attention<{traits}>(q,row_batch,live_rows,window_pool,window_keys,{sources},workspace,window_stride,window_slots,window_sequences,window_key_stride,window_budget,{geometry},scale_bits);"
                ),
                s.latent / 128 * 32,
                1,
            ),
            EntrySpec::new(
                self.merge_symbol(),
                self.merge_params()?,
                format!(
                    "chop::latent::sources::merge<{traits}>(workspace,live_rows,{sinks},{rope},{output});"
                ),
                MERGE_THREADS,
                1,
            ),
        ])
    }
}

/// A native source entry for each phase, with the supplied output mode.
/// The model passes
/// its own geometry; a window-only layer sets `second_fp4_keys = false`.
pub fn entries(profile: Profile, query: LatentAttentionQuery) -> Result<Vec<Entry>, Refusal> {
    let s = shape(query)?;
    let mut entries = Vec::new();
    for (phase, name, splits) in [
        (AttentionPhase::Prefill, "prefill", 1),
        (AttentionPhase::Decode, "decode", 8),
        (AttentionPhase::Verify, "verify", 8),
    ] {
        let mut query = query;
        let LatentAttentionQuery::Attention { phase: p, .. } = &mut query else {
            unreachable!()
        };
        *p = phase;
        entries.push(Entry {
            profile,
            name: format!(
                "{}_h{}_d{}_{}_{}_s{}_r{}_{name}",
                profile.tag(),
                s.heads,
                s.latent,
                if s.second { "mixed" } else { "window" },
                if s.partial { "partial" } else { "normalized" },
                u8::from(s.sink),
                s.tail
            ),
            query,
            splits,
        });
    }
    Ok(entries)
}

pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for entry in entries {
        if entry.profile != profile {
            return Err(Refusal("native latent entry belongs to another profile"));
        }
        specs.extend(entry.entry_specs()?);
    }
    Ok(CubinSpec {
        name: format!("latent_sources_{}_sm120", profile.tag()),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "native-latent-builder",
        tag: "native_latent_builder",
    });
    fn query(heads: u32, latent: u32) -> LatentAttentionQuery {
        LatentAttentionQuery::Attention {
            heads,
            latent,
            max_keys: 2176,
            softmax_scale_bits: (512_f32).powf(-0.5).to_bits(),
            page_size: 64,
            kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
            phase: AttentionPhase::Prefill,
            output: LatentAttentionOutput::Normalized,
            sink: true,
            tail_rope_dims: 64,
            second_fp4_keys: true,
            capacity: 16,
        }
    }

    #[test]
    fn native_latent_shapes_plan_within_limits_and_keep_decode_verify_arithmetic() {
        for heads in [1, 3, 4, 8, 16, 32, 64, 128] {
            for latent in [128, 256, 384, 512] {
                let entries = entries(PROFILE, query(heads, latent)).unwrap();
                for e in &entries {
                    e.contract().unwrap();
                    for rows in [1, 3, 16] {
                        crate::backend::check_plan(&e.plan(rows).unwrap()).unwrap();
                        crate::backend::check_plan(&e.merge_plan(rows).unwrap()).unwrap();
                        assert_eq!(
                            e.workspace_bytes(rows),
                            u64::from(rows * e.splits * heads * (latent + 2)) * 4
                        );
                    }
                    assert!(e.dynamic_shared_bytes() <= 101_376);
                    assert_eq!(e.entry_specs().unwrap().len(), 2);
                }
                let decode = entries[1].entry_specs().unwrap();
                let verify = entries[2].entry_specs().unwrap();
                for i in 0..2 {
                    assert_eq!(decode[i].body, verify[i].body);
                    assert_eq!(decode[i].params, verify[i].params);
                }
            }
        }
    }

    #[test]
    fn native_latent_builder_refuses_invalid_geometry_and_legacy_formats() {
        let base = entries(PROFILE, query(16, 512)).unwrap().remove(0);
        for splits in [0, 65] {
            assert!(
                Entry {
                    splits,
                    ..base.clone()
                }
                .entry_specs()
                .is_err()
            );
        }
        for (heads, width) in [(129, 512), (0, 512), (16, 640)] {
            assert!(entries(PROFILE, query(heads, width)).is_err());
        }
        let mut legacy = base.clone();
        if let LatentAttentionQuery::Attention {
            kv,
            second_fp4_keys,
            ..
        } = &mut legacy.query
        {
            *kv = LatentKvFormat::Bf16;
            *second_fp4_keys = false;
        }
        assert!(legacy.contract().is_err());
        assert!(base.plan(0).is_err());
        assert!(base.plan(17).is_err());
        assert_eq!(base.dynamic_shared_bytes(), 83_200);
    }
}
