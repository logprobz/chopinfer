//! Sixteen heads share an ordered 64-key tile without changing any chain.
use super::*;
use kernel_core::contract::Validate;

pub const HEADER: &str = "latent/sources_prefill_heads.cuh";
pub const SHARED_BYTES: u32 = 64 * 512 * 2 + 2 * 4 * 64 * 8 * 4 + 64 * 20;
#[path = "sources_prefill_heads_async.rs"]
pub mod asynchronous;
#[path = "sources_prefill_heads_empty.rs"]
pub mod empty;
#[path = "sources_prefill_heads_next.rs"]
pub mod next;

pub fn symbol(reference: &sources::Entry) -> String {
    format!("chop_latent_sources_{}_head_shared_v1", reference.name)
}

pub fn plan(reference: &sources::Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    let shape = validate(reference)?;
    if shape.latent != 512 || shape.heads % 16 != 0 {
        return Err(Refusal(
            "shared-head prefill requires width 512 and complete sixteen-head groups",
        ));
    }
    let mut plan = reference.plan(rows)?;
    plan.grid = [rows, shape.heads / 16, 1];
    plan.block = [256, 1, 1];
    plan.dynamic_shared_bytes = SHARED_BYTES;
    plan.validate()?;
    Ok(plan)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Partial,
    Profile,
    Finish,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Partial => "partial",
            Self::Profile => "profile",
            Self::Finish => "finish",
        }
    }
    fn cxx(self) -> &'static str {
        match self {
            Self::Partial => "Partial",
            Self::Profile => "Profile",
            Self::Finish => "Finish",
        }
    }
}

fn head_entry(
    reference: &sources::Entry,
    asynchronous: bool,
    pipeline: bool,
    mode: Mode,
    skip_empty: bool,
) -> Result<EntrySpec, Refusal> {
    let shape = validate(reference)?;
    let plan = plan(reference, 1)?;
    if reference.schedule != sources::Schedule::Grouped {
        return Err(Refusal(
            "shared heads require the grouped reference schedule",
        ));
    }
    if asynchronous && !shape.bf16 {
        return Err(Refusal(
            "asynchronous shared-head staging requires BF16 keys",
        ));
    }
    if skip_empty && (pipeline || mode != Mode::Finish) {
        return Err(Refusal(
            "empty-unit skipping requires direct finish staging",
        ));
    }
    let traits = format!(
        "chop::latent::sources::Attention<{},{},{},{},{},{},{},{},{}>",
        shape.heads,
        shape.latent,
        shape.page,
        shape.max_keys,
        reference.splits,
        shape.second,
        shape.sink,
        shape.tail,
        shape.partial
    );
    let sources = if shape.second {
        "compressed_pool,compressed_keys,block_tables"
    } else {
        "nullptr,nullptr,nullptr"
    };
    let geometry = if shape.second {
        "page_stride,max_pages,compressed_pages,compressed_sequences,compressed_key_stride,compressed_budget"
    } else {
        "0,0,0,0,0,0"
    };
    let mut params = reference.params()?;
    let outputs = match mode {
        Mode::Partial => "nullptr,nullptr,nullptr,nullptr",
        Mode::Profile => {
            params.push(kernel_build::EntryParam::DevicePointer {
                name: "stamps".into(),
                cxx_type: "uint64_t".into(),
            });
            "nullptr,nullptr,nullptr,stamps"
        }
        Mode::Finish => {
            for (name, cxx_type) in [
                ("sinks", "const float"),
                ("cos_sin", "const float"),
                ("out", "uint16_t"),
            ] {
                params.push(kernel_build::EntryParam::DevicePointer {
                    name: name.into(),
                    cxx_type: cxx_type.into(),
                });
            }
            "sinks,cos_sin,out,nullptr"
        }
    };
    let engine = if asynchronous {
        "head_async_attention"
    } else {
        "head_shared_attention"
    };
    let pipeline_arg = if asynchronous {
        format!("{pipeline},")
    } else {
        String::new()
    };
    let mut symbol = match mode {
        Mode::Partial if asynchronous => asynchronous::symbol(reference, pipeline),
        Mode::Partial => symbol(reference),
        _ => next::symbol(reference, asynchronous, mode),
    };
    if skip_empty {
        symbol = empty::symbol(reference, asynchronous);
    }
    Ok(EntrySpec::new(
        symbol,
        params,
        format!(
            "chop::latent::sources_prefill::{engine}<chop::latent::sources_prefill::HeadMode::{mode},{skip_empty},{pipeline_arg}{traits},chop::latent::sources::NativeAddress,{bf16}>(q,row_batch,live_rows,window_codes,window_scales,window_item_slots,window_keys,{sources},workspace,window_code_stride,window_scale_stride,window_slots,window_sequences,window_items,window_key_stride,window_budget,{geometry},scale_bits,{outputs});",
            mode = mode.cxx(),
            bf16 = shape.bf16
        ),
        plan.block[0],
        1,
    ))
}

pub fn cubin_spec(profile: Profile, entries: &[sources::Entry]) -> Result<CubinSpec, Refusal> {
    Ok(CubinSpec {
        name: format!("latent_sources_prefill_head_shared_{}_sm120", profile.tag()),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: entries
            .iter()
            .map(|e| head_entry(e, false, false, Mode::Partial, false))
            .collect::<Result<_, _>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_head_body_keeps_the_paired_math_statements() {
        let old = include_str!("../../cuda/latent/sources_prefill.cuh")
            .split("void reuse_attention(")
            .nth(1)
            .unwrap()
            .split("// The final merge is the only sink owner.")
            .next()
            .unwrap();
        let new = include_str!("../../cuda/latent/sources_prefill_heads.cuh");
        let nonempty = new.split("    // Both eight-head groups").nth(1).unwrap();
        let math = |s: &str| {
            s.lines()
                .map(str::trim)
                .filter(|l| {
                    l.contains("ptx::mma_bf16")
                        || l.contains("fmaf(")
                        || l.contains("sum +=")
                        || l.contains("denominator[n] +=")
                        || l.contains("p[kt][n][i] = exp_of")
                })
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert_eq!(math(old), math(nonempty));
        assert_eq!(SHARED_BYTES, 83200);
        assert!(new.contains("head0 = blockIdx.y * 16 + group * T::head_tile"));
    }
}
