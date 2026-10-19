//! Additive decode staging candidates. Every entry keeps eight numerical splits.
use super::sources::{self, Entry as Source};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, AttentionPhase, LaunchPlan, Refusal};
use kernel_core::latent_attention::LatentAttentionQuery;

#[derive(Clone, Debug)]
pub struct Entry {
    pub source: Source,
    pub stage_keys: u32,
    /// Zero retains the qualified one-head/small-row schedule.
    pub head_group: u32,
    pub packed: bool,
    pub profile: bool,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.source.contract()?;
        let s = sources::shape(self.source.query)?;
        if self.source.splits != 8
            || !matches!(self.stage_keys, 32 | 64)
            || !matches!(self.head_group, 0 | 1 | 2 | 4 | 8)
            || s.bf16
            || !matches!(
                self.source.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Decode | AttentionPhase::Verify,
                    ..
                }
            )
        {
            return Err(Refusal(
                "decode staging requires eight native decode splits",
            ));
        }
        Ok(())
    }
    pub fn symbol(&self) -> String {
        format!(
            "chop_latent_sources_{}_stage{}_h{}_b{}_profile{}_v1",
            self.source.name,
            self.stage_keys,
            self.head_group,
            u32::from(self.packed),
            u32::from(self.profile)
        )
    }
    pub fn group(&self, rows: u32) -> u32 {
        if self.head_group != 0 {
            self.head_group
        } else if rows <= 4 {
            1
        } else {
            8
        }
    }
    pub fn stamp_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.plan(rows)?;
        Ok(u64::from(rows) * u64::from(sources::shape(self.source.query)?.heads) * 8 * 8 * 8)
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        let s = sources::shape(self.source.query)?;
        let mut plan = self.source.plan(rows)?;
        plan.grid[1] = s.heads.div_ceil(self.group(rows));
        plan.dynamic_shared_bytes = self.stage_keys * s.latent * 2
            + (s.latent / 128) * sources::KEYS * 8 * 4
            + sources::KEYS * 20;
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
    pub fn spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let s = sources::shape(self.source.query)?;
        let mut params = self.source.params()?;
        params.push(EntryParam::DevicePointer {
            name: "stamps".into(),
            cxx_type: "uint64_t".into(),
        });
        let traits = format!(
            "chop::latent::sources::Attention<{},{},{},{},8,{},{},{},{}>",
            s.heads, s.latent, s.page, s.max_keys, s.second, s.sink, s.tail, s.partial
        );
        let second = if s.second {
            "compressed_pool,compressed_keys,block_tables"
        } else {
            "nullptr,nullptr,nullptr"
        };
        let geometry = if s.second {
            "page_stride,max_pages,compressed_pages,compressed_sequences,compressed_key_stride,compressed_budget"
        } else {
            "0,0,0,0,0,0"
        };
        let group = if self.head_group == 0 {
            "(gridDim.x <= 4 ? 1 : 8)".into()
        } else {
            self.head_group.to_string()
        };
        let body = format!(
            "chop::ptx::griddep_launch_dependents();\nextern __shared__ __align__(1024) uint8_t arena[];\nchop::ptx::griddep_wait();\nconst uint32_t group = {group};\nconst chop::latent::sources_decode::Index task{{blockIdx.x,blockIdx.y*group,blockIdx.z,group}};\nchop::latent::sources_decode::task<{traits},{},{},{}>(task,arena,stamps,q,row_batch,live_rows,window_codes,window_scales,window_item_slots,window_keys,{second},workspace,window_code_stride,window_scale_stride,window_slots,window_sequences,window_items,window_key_stride,window_budget,{geometry},scale_bits);",
            self.stage_keys, self.packed, self.profile
        );
        Ok(EntrySpec::new(
            self.symbol(),
            params,
            body,
            s.latent / 128 * 32,
            1,
        ))
    }
    pub fn launch(&self, reference: &Launch, stamps: Option<View>) -> Result<Launch, Refusal> {
        self.validate()?;
        if reference.symbol != self.source.symbol()
            || reference.plan != self.source.plan(reference.rows)?
            || reference.arguments.0.len() != self.source.params()?.len()
        {
            return Err(Refusal("decode staging needs its validated source launch"));
        }
        let mut call = reference.clone();
        let stamp = if self.profile {
            let view = stamps.ok_or(Refusal("profile stamps are absent"))?;
            view.covers(self.stamp_bytes(reference.rows)?)?;
            if call
                .arguments
                .0
                .contains(&Argument::DeviceBuffer(view.handle))
            {
                return Err(Refusal("profile stamps alias an operand"));
            }
            Argument::DeviceBuffer(view.handle)
        } else {
            Argument::NullDevicePointer
        };
        call.arguments.0.push(stamp);
        call.symbol = self.symbol();
        call.plan = self.plan(reference.rows)?;
        Ok(call)
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let first = entries
        .first()
        .ok_or(Refusal("empty decode staging cubin"))?;
    let mut spec = sources::cubin_spec(first.source.profile, &[])?;
    spec.name = name.into();
    spec.headers = vec!["#include \"latent/sources_decode.cuh\"".into()];
    spec.entries = entries.iter().map(Entry::spec).collect::<Result<_, _>>()?;
    Ok(spec)
}

pub fn widen_spec(wrong: bool) -> EntrySpec {
    let params = ["reference", "candidate"]
        .into_iter()
        .map(|name| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: "uint32_t".into(),
        })
        .collect();
    EntrySpec::new(
        format!("chop_latent_decode_widen_control{}_v1", u32::from(wrong)),
        params,
        format!(
            "const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\nif (i >= 65536) return;\nconst uint16_t codes = (i & 255) | (((i ^ 128) & 255) << 8);\nconst uint8_t scale = i >> 8;\nconst auto fp8 = chop::convert::unpack_f32_e4m3(codes);\nconst float s8 = chop::convert::unpack_f32_ue8m0(scale).lo;\nreference[i] = chop::attention::bf16_pair(fp8.lo * s8, fp8.hi * s8);\ncandidate[i] = chop::latent::sources_decode::widen<true,false>(codes,scale);\nconst auto fp4 = chop::convert::unpack_f32_e2m1(codes);\nconst float s4 = chop::convert::unpack_f32_e4m3(scale).lo;\nreference[i+65536] = chop::attention::bf16_pair(fp4.lo * s4, fp4.hi * s4);\ncandidate[i+65536] = chop::latent::sources_decode::widen<true,true>(codes,scale);\nif ({} && i == 0) candidate[i] ^= 1;",
            wrong
        ),
        256,
        1,
    )
}
