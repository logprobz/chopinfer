//! Experimental prefill staging with the native source engine's arithmetic.
//! The 64-key softmax tile is unchanged; only 32 decoded keys reside in
//! shared memory at a time. Call the reference merge on the resulting partials.
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{AttentionPhase, LaunchPlan, Refusal};
use kernel_core::latent_attention::LatentAttentionQuery;

use super::sources;
use crate::Profile;

pub const HEADER: &str = "latent/sources_prefill.cuh";
pub const PROFILE_WORDS: u64 = 9;

#[derive(Clone, Copy, Debug)]
pub enum ProfileMode {
    Full,
    Loads,
    Resident,
}

impl ProfileMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Loads => "loads",
            Self::Resident => "resident",
        }
    }
}

pub fn profile_symbol(reference: &sources::Entry, mode: ProfileMode) -> String {
    format!(
        "chop_latent_sources_{}_profile_{}_v1",
        reference.name,
        mode.name()
    )
}

/// Lab-only ablations. Full keeps reference arithmetic; Loads and Resident
/// are diagnostic workloads with different outputs, never serving entries.
pub fn profile_cubin_spec(
    profile: Profile,
    entries: &[sources::Entry],
) -> Result<CubinSpec, Refusal> {
    let mut spec = sources::cubin_spec(profile, entries)?;
    spec.name = format!("latent_sources_prefill_profile_{}_sm120", profile.tag());
    spec.headers = vec![format!("#include \"{HEADER}\"")];
    spec.entries.clear();
    for entry in entries {
        validate(entry)?;
        for (mode, cxx) in [
            (ProfileMode::Full, "Full"),
            (ProfileMode::Loads, "Loads"),
            (ProfileMode::Resident, "Resident"),
        ] {
            let mut body = entry.entry_specs()?.remove(0);
            if !body.body.starts_with("chop::latent::sources::attention<") {
                return Err(Refusal("profiling requires the grouped reference body"));
            }
            body.symbol = profile_symbol(entry, mode);
            body.params.push(kernel_build::EntryParam::DevicePointer {
                name: "stamps".into(),
                cxx_type: "uint64_t".into(),
            });
            body.body = body.body.replace("chop::latent::sources::attention<", &format!(
                "chop::latent::sources_prefill::profile_attention<chop::latent::sources_prefill::ProfileMode::{cxx},"
            )).replace(");", ",stamps);");
            spec.entries.push(body);
        }
    }
    Ok(spec)
}

fn validate(reference: &sources::Entry) -> Result<sources::Shape, Refusal> {
    reference.contract()?;
    if !matches!(
        reference.query,
        LatentAttentionQuery::Attention {
            phase: AttentionPhase::Prefill,
            ..
        }
    ) || reference.splits != 1
    {
        return Err(Refusal("half-tile staging requires one-split prefill"));
    }
    sources::shape(reference.query)
}

pub fn symbol(reference: &sources::Entry) -> String {
    format!("chop_latent_sources_{}_stage32_v1", reference.name)
}

pub fn plan(reference: &sources::Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    let shape = validate(reference)?;
    let mut plan = reference.plan(rows)?;
    plan.dynamic_shared_bytes =
        32 * shape.latent * 2 + (shape.latent / 128) * sources::KEYS * 8 * 4 + sources::KEYS * 20;
    Ok(plan)
}

pub fn entry_spec(reference: &sources::Entry) -> Result<EntrySpec, Refusal> {
    validate(reference)?;
    let mut spec = reference.entry_specs()?.remove(0);
    if !spec.body.starts_with("chop::latent::sources::attention<") {
        return Err(Refusal(
            "half-tile staging requires the grouped reference body",
        ));
    }
    spec.symbol = symbol(reference);
    spec.body = spec.body.replace(
        "chop::latent::sources::attention<",
        "chop::latent::sources_prefill::attention<",
    );
    // The reduced shared allocation permits two CTAs per SM at width 512.
    spec.min_blocks = 2;
    Ok(spec)
}

pub fn cubin_spec(profile: Profile, entries: &[sources::Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = sources::cubin_spec(profile, entries)?;
    spec.name = format!("latent_sources_prefill_{}_sm120", profile.tag());
    spec.headers = vec![format!("#include \"{HEADER}\"")];
    spec.entries = entries.iter().map(entry_spec).collect::<Result<_, _>>()?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;
    use kernel_core::latent_attention::{LatentAttentionOutput, LatentKvFormat};

    #[test]
    fn prefill_stage_changes_storage_without_changing_the_partial_or_merge_abi() {
        let profile = Profile::synthetic(ProfileNames {
            name: "prefill-stage",
            tag: "prefill_stage",
        });
        for (second, page, kv) in [
            (false, 64, LatentKvFormat::Fp8E4M3Ue8M0Block32),
            (true, 64, LatentKvFormat::Fp8E4M3Ue8M0Block32),
            (true, 32, LatentKvFormat::Fp8E4M3Ue8M0Block32),
            (true, 32, LatentKvFormat::Bf16),
        ] {
            let query = LatentAttentionQuery::Attention {
                heads: 16,
                latent: 512,
                page_size: page,
                max_keys: 640,
                softmax_scale_bits: (1.0_f32 / 512.0_f32.sqrt()).to_bits(),
                kv,
                sink: true,
                tail_rope_dims: 64,
                second_fp4_keys: second,
                output: LatentAttentionOutput::Normalized,
                capacity: 8192,
                phase: AttentionPhase::Prefill,
            };
            for reference in sources::entries(profile, query).unwrap() {
                if reference.splits != 1 {
                    assert!(entry_spec(&reference).is_err());
                    assert!(plan(&reference, 1).is_err());
                    continue;
                }
                let spec = entry_spec(&reference).unwrap();
                assert_eq!(spec.params, reference.params().unwrap());
                assert!(
                    spec.body
                        .starts_with("chop::latent::sources_prefill::attention<")
                );
                assert_eq!(spec.min_blocks, 2);
                let profiling =
                    profile_cubin_spec(profile, std::slice::from_ref(&reference)).unwrap();
                assert_eq!(profiling.entries.len(), 3);
                for (mode, entry) in [ProfileMode::Full, ProfileMode::Loads, ProfileMode::Resident]
                    .into_iter()
                    .zip(&profiling.entries)
                {
                    assert_eq!(entry.symbol, profile_symbol(&reference, mode));
                    assert_eq!(
                        &entry.params[..entry.params.len() - 1],
                        reference.params().unwrap()
                    );
                    assert!(entry.body.ends_with(",stamps);"));
                    assert_eq!(entry.max_threads, 128);
                    assert_eq!(entry.min_blocks, 1);
                }
                for rows in [1, 17, 8192] {
                    let old = reference.plan(rows).unwrap();
                    let new = plan(&reference, rows).unwrap();
                    assert_eq!(old.dynamic_shared_bytes, 83200);
                    assert_eq!(new.dynamic_shared_bytes, 42240);
                    assert_eq!(
                        LaunchPlan {
                            dynamic_shared_bytes: old.dynamic_shared_bytes,
                            ..new
                        },
                        old
                    );
                }
                assert!(plan(&reference, 8193).is_err());
            }
        }
    }
}
