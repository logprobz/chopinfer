//! Slice-widened latent attention (`cuda/latent/attention_slices.cuh`,
//! README "Slice-widened attention"): an FP8 one-split (prefill) attention
//! entry's outputs bit for bit, from the same launch ABI and grid, with each
//! warp widening only the FP8 slices its own MMAs read, the queries loaded
//! straight into fragments and the page lookups pipelined. Its 46,336 bytes
//! of shared memory and 255 registers over 128 threads let two CTAs share an
//! SM, where the entry it replaces fits one. In its own cubin,
//! `latent_attention_slices_{tag}_sm120`.

use std::path::PathBuf;

use super::{Entry as Attention, Query};
use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    AttentionPhase, InstantiationSpec, LaunchPlan, Refusal, Target, Validate,
};
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "latent/attention_slices.cuh";
/// CTAs of one entry that share an SM: the launch bounds' minimum.
pub const CTAS_PER_SM: u32 = 2;
/// SM120's shared memory per SM, and what each resident CTA takes beside
/// its dynamic shared memory: the 1 KiB ptxas reports as static and the
/// 1 KiB the driver reserves.
const SM_SHARED: u32 = 102_400;
const CTA_RESERVED: u32 = 2_048;
/// Keys per unit, and BF16 columns of a widened slice.
const UNIT_KEYS: u32 = 32;
const SLICE_COLUMNS: u32 = 16;
/// Lab-only mutant: a unit counts as live only when its first key is, so
/// units whose first entry is a hole are skipped.
pub const SKIP_FIRST_DEFINE: &str = "CHOP_LATENT_SLICES_TEST_SKIP_FIRST";
/// Lab-only mutant: each unit's copies read the page of the unit before.
pub const STALE_PAGE_DEFINE: &str = "CHOP_LATENT_SLICES_TEST_STALE_PAGE";
/// Lab-only mutant: the dependency wait after the query loads.
pub const LATE_WAIT_DEFINE: &str = "CHOP_LATENT_SLICES_TEST_LATE_WAIT";
/// The lab's mutants: (symbol suffix, define).
pub const MUTANTS: [(&str, &str); 3] = [
    ("_skip_first", SKIP_FIRST_DEFINE),
    ("_stale_page", STALE_PAGE_DEFINE),
    ("_late_wait", LATE_WAIT_DEFINE),
];

/// The slice-widened form of one FP8 one-split attention entry.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The entry whose outputs this one writes and whose launch it replaces.
    pub attention: Attention,
    /// Appended to the symbol (lab mutants); empty when served.
    pub suffix: String,
}

/// The attention parts the kernel's traits take.
struct Shape {
    heads: u32,
    latent: u32,
    page_size: u32,
    max_keys: u32,
    partial: bool,
}

impl Entry {
    /// The form's declared exactness: its attention entry's, since it writes
    /// that entry's outputs bit for bit at every row count; nothing for a
    /// lab mutant or a form its kernel refuses.
    pub fn invariance(&self) -> kernel_core::properties::Invariance {
        if !self.suffix.is_empty() || self.contract().is_err() {
            return kernel_core::properties::Invariance::NONE;
        }
        self.attention.invariance()
    }

    /// The slice-widened form of `attention`, which must be an FP8 prefill
    /// attention entry (one split per row).
    pub fn of(attention: &Attention) -> Result<Entry, Refusal> {
        let entry = Entry {
            attention: attention.clone(),
            suffix: String::new(),
        };
        entry.shape()?;
        Ok(entry)
    }

    fn shape(&self) -> Result<Shape, Refusal> {
        match self.attention.query {
            Query::Latent(LatentAttentionQuery::Attention {
                heads,
                latent,
                max_keys,
                page_size,
                kv: LatentKvFormat::Fp8E4M3Block128,
                phase: AttentionPhase::Prefill,
                output,
                ..
            }) => Ok(Shape {
                heads,
                latent,
                page_size,
                max_keys,
                partial: output == LatentAttentionOutput::Partial,
            }),
            _ => Err(Refusal(
                "slice-widened latent attention takes an FP8 prefill attention entry",
            )),
        }
    }

    pub fn profile(&self) -> Profile {
        self.attention.profile
    }

    pub fn symbol(&self) -> String {
        format!(
            "chop_latent_{}_slices{}_v1",
            self.attention.name, self.suffix
        )
    }

    /// Dynamic shared bytes: two stages of codes and scales, a widened
    /// slice per warp, the exchanged chains and the valid flags;
    /// `SliceAttention::smem`.
    pub fn dynamic_shared_bytes(&self) -> u32 {
        let Ok(shape) = self.shape() else { return 0 };
        let blocks = shape.latent / 128;
        let stage = UNIT_KEYS * shape.latent + UNIT_KEYS * blocks * 4;
        2 * stage
            + blocks * UNIT_KEYS * SLICE_COLUMNS * 2
            + blocks * UNIT_KEYS * 16 * 4
            + 2 * UNIT_KEYS * 4
    }

    /// Threads per CTA: a warp per 128-wide latent block.
    pub fn threads(&self) -> u32 {
        self.attention.threads()
    }

    /// The attention entry's contract under this symbol, its knobs marking
    /// the swizzled FP8 stages and the one split.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.shape()?;
        let base = self.attention.contract()?;
        let mut key = base.key;
        key.knobs.swizzle = 1;
        key.knobs.split_k_capacity = 1;
        key.target = Target::Sm120f;
        let spec = InstantiationSpec {
            family: base.family,
            key,
            symbol: self.symbol(),
        };
        spec.validate()?;
        if (self.dynamic_shared_bytes() + CTA_RESERVED) * CTAS_PER_SM > SM_SHARED {
            return Err(Refusal(
                "slice-widened latent attention needs two CTAs' shared memory per SM",
            ));
        }
        Ok(spec)
    }

    /// The attention entry's grid, one CTA per (row, 16 heads); rows past
    /// the live count return after the PDL wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let plan = LaunchPlan {
            dynamic_shared_bytes: self.dynamic_shared_bytes(),
            ..self.attention.plan(prepared_rows)?
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The attention entry's ABI; the kernel reads neither the workspace
    /// nor the split count, which is one.
    pub fn params(&self) -> Vec<EntryParam> {
        self.attention.params()
    }

    fn body(&self) -> Result<String, Refusal> {
        let shape = self.shape()?;
        let outputs = if shape.partial {
            "nullptr,out_max,out_den,out_num"
        } else {
            "out,nullptr,nullptr,nullptr"
        };
        Ok(format!(
            "chop::latent::attention_slices<chop::latent::SliceAttention<{},{},{},{},{}>>(q,pool,\
             block_tables,row_batch,key_lists,live_rows,{outputs},max_pages,page_stride_bytes,\
             budget,scale_bits);",
            shape.heads, shape.latent, shape.page_size, shape.max_keys, shape.partial
        ))
    }

    pub fn entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        Ok(EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body()?,
            self.threads(),
            CTAS_PER_SM,
        ))
    }
}

/// The slice-widened forms of the FP8 prefill attention entries among
/// `entries`, in order.
pub fn entries(entries: &[Attention]) -> Vec<Entry> {
    entries
        .iter()
        .filter_map(|attention| Entry::of(attention).ok())
        .collect()
}

/// The cubin name of `profile`'s slice-widened entries, with a mutant's
/// suffix.
pub fn cubin_name(profile: Profile, suffix: &str) -> String {
    format!("latent_attention_slices{suffix}_{}_sm120", profile.tag())
}

/// The cubin of `profile`'s entries among `entries`, compiled with each of
/// `defines` set to 1.
pub fn cubin_spec(
    profile: Profile,
    entries: &[Entry],
    suffix: &str,
    defines: &[&str],
) -> Result<CubinSpec, Refusal> {
    let specs = entries
        .iter()
        .filter(|entry| entry.profile() == profile)
        .map(Entry::entry_spec)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CubinSpec {
        name: cubin_name(profile, suffix),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: defines
            .iter()
            .map(|define| ((*define).to_owned(), "1".to_owned()))
            .collect(),
        entries: specs,
    })
}

/// The lab's cubins of `profile`'s entries among `entries`: the served
/// form, then each mutant in a cubin of its own with its suffix on every
/// symbol.
pub fn lab_cubin_specs(profile: Profile, entries: &[Entry]) -> Result<Vec<CubinSpec>, Refusal> {
    let mut specs = vec![cubin_spec(profile, entries, "", &[])?];
    for (suffix, define) in MUTANTS {
        let mutants = entries
            .iter()
            .map(|entry| Entry {
                suffix: suffix.into(),
                ..entry.clone()
            })
            .collect::<Vec<_>>();
        specs.push(cubin_spec(profile, &mutants, suffix, &[define])?);
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;
    use crate::latent::{Geometry, entries as latent_entries};

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "latent-slices-test",
        tag: "latent_slices_test",
    });

    fn geometry() -> Geometry {
        Geometry {
            heads: 16,
            latent: 512,
            page_size: 64,
            max_keys: 2051,
            softmax_scale: 1.0 / 16.0,
            nope: 256,
            v_dim: 256,
        }
    }

    #[test]
    fn entries_take_the_fp8_prefill_entries_plans_and_abi() {
        let attention = latent_entries(PROFILE, geometry());
        let slices = entries(&attention);
        // The normalized and the partial FP8 prefill entry.
        assert_eq!(slices.len(), 2);
        assert_eq!(
            slices[0].symbol(),
            "chop_latent_latent_slices_test_attn_fp8_prefill_slices_v1"
        );
        assert_eq!(
            slices[1].symbol(),
            "chop_latent_latent_slices_test_attn_fp8_prefill_partial_slices_v1"
        );
        for entry in &slices {
            entry.contract().unwrap();
            assert_eq!(entry.dynamic_shared_bytes(), 46_336);
            for rows in [1, 30, 2048, 8192] {
                let plan = entry.plan(rows).unwrap();
                let reference = entry.attention.plan(rows).unwrap();
                assert_eq!(plan.grid, reference.grid);
                assert_eq!(plan.block, reference.block);
                assert_eq!(plan.workspace_bytes, 0);
                assert!(plan.pdl);
                assert!(reference.dynamic_shared_bytes > 50_176);
            }
            assert!(entry.plan(8193).is_err());
            assert_eq!(entry.params(), entry.attention.params());
            assert_eq!(entry.entry_spec().unwrap().min_blocks, 2);
        }
        assert!(
            slices[0]
                .body()
                .unwrap()
                .contains("SliceAttention<16,512,64,2051,false>>(q,pool,")
        );
        assert!(slices[1].body().unwrap().contains("2051,true>>(q,pool,"));
        // BF16 pages, decode and verify keep their entries.
        for other in attention
            .iter()
            .filter(|entry| !entry.name.contains("attn_fp8_prefill"))
        {
            assert!(Entry::of(other).is_err(), "{}", other.name);
        }
    }

    #[test]
    fn lab_cubins_hold_the_served_entries_and_each_mutant() {
        let slices = entries(&latent_entries(PROFILE, geometry()));
        let specs = lab_cubin_specs(PROFILE, &slices).unwrap();
        assert_eq!(specs.len(), 1 + MUTANTS.len());
        assert_eq!(
            specs[0].name,
            "latent_attention_slices_latent_slices_test_sm120"
        );
        assert!(specs[0].defines.is_empty());
        for (spec, (suffix, define)) in specs[1..].iter().zip(MUTANTS) {
            assert_eq!(spec.name, cubin_name(PROFILE, suffix));
            assert_eq!(spec.defines, [(define.to_owned(), "1".to_owned())]);
            assert!(
                spec.entries
                    .iter()
                    .all(|entry| entry.symbol.ends_with(&format!("{suffix}_v1")))
            );
        }
    }
}
