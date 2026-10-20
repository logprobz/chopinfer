//! Sixteen heads share an ordered 64-key tile without changing any chain.
use super::*;
use kernel_core::contract::Validate;

pub const HEADER: &str = "latent/sources_prefill_heads.cuh";
pub const SHARED_BYTES: u32 = 64 * 512 * 2 + 2 * 4 * 64 * 8 * 4 + 64 * 20;

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

pub fn cubin_spec(profile: Profile, entries: &[sources::Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = reuse_cubin_spec(profile, entries, 2)?;
    spec.name = format!("latent_sources_prefill_head_shared_{}_sm120", profile.tag());
    spec.headers = vec![format!("#include \"{HEADER}\"")];
    for (entry, reference) in spec.entries.iter_mut().zip(entries) {
        plan(reference, 1)?;
        entry.symbol = symbol(reference);
        if !entry.body.contains("::reuse_attention<2,") {
            return Err(Refusal("missing paired entry anchor"));
        }
        entry.body = entry
            .body
            .replace("::reuse_attention<2,", "::head_shared_attention<");
    }
    Ok(spec)
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
        assert_eq!(math(old), math(new));
        assert_eq!(SHARED_BYTES, 83200);
        assert!(new.contains("head0 = blockIdx.y * 16 + group * T::head_tile"));
    }
}
