//! Decode selected native KV once, keeping all eight numerical source splits.
use super::{sources, sources_decode};
#[path = "sources_cached/prepared.rs"]
mod prepared;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, LaunchPlan, Refusal};
pub use prepared::{Buffers, Layout, prepare};

#[derive(Clone, Debug)]
pub struct Entry {
    pub source: sources::Entry,
    pub profile: bool,
}

impl Entry {
    fn staging(&self) -> sources_decode::Entry {
        sources_decode::Entry {
            source: self.source.clone(),
            stage_keys: 64,
            head_group: 0,
            packed: false,
            profile: self.profile,
        }
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.staging().validate()?;
        let s = sources::shape(self.source.query)?;
        if s.latent != 512
            || s.heads != 16
            || !matches!(s.max_keys, 128 | 640)
            || self.source.schedule != sources::Schedule::IndependentHeads
        {
            return Err(Refusal(
                "selected KV caching needs 16 heads and 512 channels",
            ));
        }
        Ok(())
    }

    pub fn symbol(&self) -> String {
        format!(
            "chop_latent_sources_{}_cached_profile{}_v1",
            self.source.name,
            u32::from(self.profile)
        )
    }

    pub fn decode_symbol(&self, wrong: bool) -> String {
        format!(
            "chop_latent_sources_{}_cache_decode_control{}_v1",
            self.source.name,
            u32::from(wrong)
        )
    }

    pub fn cache_bytes(&self, rows: u32) -> Result<(u64, u64), Refusal> {
        self.validate()?;
        self.source.plan(rows)?;
        let s = sources::shape(self.source.query)?;
        let cells = u64::from(rows) * u64::from(s.max_keys);
        Ok((cells * u64::from(s.latent) * 2, cells * 4))
    }

    pub fn stamp_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.staging().stamp_bytes(rows)
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        let mut plan = self.staging().plan(rows)?;
        // The predecode producer resolves native pointers; the consumer only
        // retains a 64-entry validity table after its tile and QK exchange.
        plan.dynamic_shared_bytes -= sources::KEYS * 16;
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn decode_plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        let s = sources::shape(self.source.query)?;
        let mut plan = self.source.plan(rows)?;
        plan.grid = [s.max_keys.div_ceil(4), rows, 1];
        plan.block = [128, 1, 1];
        plan.dynamic_shared_bytes = 0;
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    fn params(&self) -> Result<Vec<EntryParam>, Refusal> {
        let mut params = self.source.params()?;
        for (name, cxx_type) in [("decoded", "uint16_t"), ("cached_valid", "int32_t")] {
            params.push(EntryParam::DevicePointer {
                name: name.into(),
                cxx_type: cxx_type.into(),
            });
        }
        Ok(params)
    }

    pub fn spec(&self, decode: Option<bool>) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let s = sources::shape(self.source.query)?;
        let mut params = self.params()?;
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
        let window = "window_codes,window_scales,window_item_slots,window_keys";
        let strides = "window_code_stride,window_scale_stride,window_slots,window_sequences,window_items,window_key_stride,window_budget";
        let body = if let Some(wrong) = decode {
            format!(
                "chop::ptx::griddep_launch_dependents();\nchop::ptx::griddep_wait();\nconst chop::latent::sources_cached::CacheIndex task{{blockIdx.y,blockIdx.x}};\nchop::latent::sources_cached::predecode<{traits},{wrong}>(task,nullptr,decoded,cached_valid,row_batch,live_rows,{window},{second},{strides},{geometry});"
            )
        } else {
            params.push(EntryParam::DevicePointer {
                name: "stamps".into(),
                cxx_type: "uint64_t".into(),
            });
            format!(
                "chop::ptx::griddep_launch_dependents();\nextern __shared__ __align__(1024) uint8_t arena[];\nchop::ptx::griddep_wait();\nconst uint32_t group = gridDim.x <= 4 ? 1 : 8;\nconst chop::latent::sources_cached::Index task{{blockIdx.x,blockIdx.y*group,blockIdx.z,group}};\nchop::latent::sources_cached::task<{traits},{}>(task,arena,stamps,decoded,cached_valid,q,row_batch,live_rows,{window},{second},workspace,{strides},{geometry},scale_bits);",
                self.profile
            )
        };
        Ok(EntrySpec::new(
            decode.map_or_else(|| self.symbol(), |wrong| self.decode_symbol(wrong)),
            params,
            body,
            128,
            1,
        ))
    }

    pub fn launches(
        &self,
        reference: &Launch,
        decoded: View,
        valid: View,
        stamps: Option<View>,
    ) -> Result<Vec<Launch>, Refusal> {
        self.validate()?;
        if reference.symbol != self.source.symbol()
            || reference.plan != self.source.plan(reference.rows)?
            || reference.arguments.0.len() != self.source.params()?.len()
        {
            return Err(Refusal("KV cache needs its validated source launch"));
        }
        let (data_bytes, valid_bytes) = self.cache_bytes(reference.rows)?;
        decoded.covers(data_bytes)?;
        valid.covers(valid_bytes)?;
        let mut views = vec![decoded, valid];
        if self.profile {
            let stamp = stamps.ok_or(Refusal("cached source stamps are absent"))?;
            stamp.covers(self.stamp_bytes(reference.rows)?)?;
            views.push(stamp);
        }
        for (i, view) in views.iter().enumerate() {
            if views[..i].iter().any(|other| other.handle == view.handle)
                || reference
                    .arguments
                    .0
                    .contains(&Argument::DeviceBuffer(view.handle))
            {
                return Err(Refusal("selected KV cache aliases a source operand"));
            }
        }
        let mut predecode = reference.clone();
        predecode.symbol = self.decode_symbol(false);
        predecode.plan = self.decode_plan(reference.rows)?;
        predecode.arguments.0.extend([
            Argument::DeviceBuffer(decoded.handle),
            Argument::DeviceBuffer(valid.handle),
        ]);
        let mut source = predecode.clone();
        source.symbol = self.symbol();
        source.plan = self.plan(reference.rows)?;
        source.arguments.0.push(if self.profile {
            Argument::DeviceBuffer(views[2].handle)
        } else {
            Argument::NullDevicePointer
        });
        Ok(vec![predecode, source])
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let first = entries
        .first()
        .ok_or(Refusal("empty selected KV cache catalog"))?;
    let mut spec = sources::cubin_spec(first.source.profile, &[])?;
    spec.name = name.into();
    spec.headers = vec!["#include \"latent/sources_cached.cuh\"".into()];
    for entry in entries {
        spec.entries.push(entry.spec(None)?);
        if !entry.profile {
            spec.entries.push(entry.spec(Some(false))?);
            spec.entries.push(entry.spec(Some(true))?);
        }
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::AttentionPhase;
    use kernel_core::latent_attention::{
        LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat,
    };

    #[test]
    fn preserves_numerical_splits_and_bounds_the_selected_cache() {
        let profile = crate::Profile::synthetic(crate::ProfileNames {
            name: "selected-cache",
            tag: "selected_cache",
        });
        let query = LatentAttentionQuery::Attention {
            heads: 16,
            latent: 512,
            max_keys: 640,
            softmax_scale_bits: (512_f32).powf(-0.5).to_bits(),
            page_size: 64,
            kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
            phase: AttentionPhase::Decode,
            output: LatentAttentionOutput::Normalized,
            sink: true,
            tail_rope_dims: 64,
            second_fp4_keys: true,
            capacity: 32,
        };
        let mut source = sources::entries(profile, query).unwrap().remove(1);
        source.schedule = sources::Schedule::IndependentHeads;
        let entry = Entry {
            source,
            profile: false,
        };
        for rows in [1, 6, 8, 32] {
            assert_eq!(entry.plan(rows).unwrap().grid[2], 8);
            assert_eq!(entry.plan(rows).unwrap().dynamic_shared_bytes, 73984);
            assert_eq!(entry.decode_plan(rows).unwrap().grid, [160, rows, 1]);
            assert_eq!(
                entry.cache_bytes(rows).unwrap(),
                (u64::from(rows) * 655360, u64::from(rows) * 2560)
            );
        }
        assert!(entry.plan(0).is_err());
        assert!(entry.plan(33).is_err());
        let mut changed = entry.clone();
        changed.source.splits = 16;
        assert!(changed.plan(1).is_err());
        changed.source.splits = 1;
        assert!(changed.plan(1).is_err());
    }
}
