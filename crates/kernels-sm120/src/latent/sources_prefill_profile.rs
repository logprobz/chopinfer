//! Diagnostic clocks for the current paired body, never a serving schedule.
use super::*;

pub const WORDS: usize = 12;
const DIAGNOSTIC_HEADER: &str = "latent/sources_prefill_profile.cuh";
pub const STAGES: [&str; 6] = ["setup", "gather", "qk", "softmax", "pv", "epilogue"];
const SOURCE: &str = include_str!("../../cuda/latent/sources_prefill.cuh");
const BEGIN: &str = "template<int Rows, class T, class Address = NativeAddress, bool Bf16 = false>";
const END: &str = "// The final merge is the only sink owner.";

fn once(source: &mut String, old: &str, new: &str) -> Result<(), Refusal> {
    if source.matches(old).count() != 1 {
        return Err(Refusal("paired profile source anchor changed"));
    }
    *source = source.replacen(old, new, 1);
    Ok(())
}

fn body(source: &str) -> Result<String, Refusal> {
    let (_, tail) = source
        .split_once(BEGIN)
        .ok_or(Refusal("missing paired body"))?;
    let (body, _) = tail
        .split_once(END)
        .ok_or(Refusal("missing paired body end"))?;
    let mut text = format!("{BEGIN}{body}");
    once(
        &mut text,
        "void reuse_attention(",
        "void reuse_profile_attention(",
    )?;
    once(
        &mut text,
        "uint32_t compressed_budget, uint32_t scale_bits) {",
        "uint32_t compressed_budget, uint32_t scale_bits, uint64_t* stamps) {",
    )?;
    once(
        &mut text,
        "  const bool row_live = row < live;",
        r#"  const bool row_live = row < live;
  const bool sample = stamps != nullptr && threadIdx.x == 0;
  const uint64_t started = sample ? clock64() : 0;
  uint64_t last = started, cycles[6] = {};
  uint32_t same_units = 0, shifted_units = 0, distinct_units = 0;"#,
    )?;
    once(
        &mut text,
        "  for (uint32_t unit = begin; unit < end; ++unit) {",
        "  profile_clock(sample, cycles, 0, last);\n  for (uint32_t unit = begin; unit < end; ++unit) {",
    )?;
    once(
        &mut text,
        "    const int passes = shift >= 0 ? 1 : Rows;",
        r#"    if (sample) {
      same_units += shift == 0;
      shifted_units += shift == 1;
      distinct_units += shift < 0;
    }
    const int passes = shift >= 0 ? 1 : Rows;"#,
    )?;
    once(
        &mut text,
        "      const bool active = row_live && (shift >= 0 || group == pass);",
        "      profile_clock(sample, cycles, 1, last);\n      const bool active = row_live && (shift >= 0 || group == pass);",
    )?;
    let probability = "uint32_t probability[terms][T::key_tiles][T::n_tiles][2];";
    let rescale = "float rescale[T::n_tiles][2];";
    // Extend only these two declarations across the added stage barrier.
    // Arithmetic and every loop's key/channel order remain in the source.
    once(&mut text, &format!("        {probability}"), "")?;
    once(&mut text, &format!("        {rescale}"), "")?;
    once(
        &mut text,
        "      if (active) {\n        // Scores",
        &format!(
            "      profile_clock(sample, cycles, 2, last);\n      {probability}\n      {rescale}\n      if (active) {{\n        // Scores"
        ),
    )?;
    // The current body has no CTA barrier between softmax and PV. Add one
    // in this diagnostic so thread zero also times the other row's passes.
    // Always compare null-stamp probe timing to the unmodified locked body.
    once(
        &mut text,
        "        // PV: one chain from zero",
        r#"      }
      __syncthreads();
      profile_clock(sample, cycles, 3, last);
      if (active) {
        // PV: one chain from zero"#,
    )?;
    once(
        &mut text,
        "      __syncthreads();\n    }\n  }\n  if (!row_live) return;",
        "      __syncthreads();\n      profile_clock(sample, cycles, 4, last);\n    }\n  }\n  if (!row_live) return;",
    )?;
    let end = text
        .trim_end()
        .strip_suffix('}')
        .ok_or(Refusal("invalid paired function end"))?;
    Ok(format!(
        r#"{end}
  profile_clock(sample, cycles, 5, last);
  if (sample) {{
    const uint64_t cta = (uint64_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    uint64_t* record = stamps + cta * 12;
    record[0] = started; record[1] = last;
    uint32_t smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    record[2] = smid;
    for (int stage = 0; stage < 6; ++stage) record[3 + stage] = cycles[stage];
    record[9] = same_units; record[10] = shifted_units; record[11] = distinct_units;
  }}
}}
"#
    ))
}

pub fn symbol(reference: &sources::Entry) -> String {
    format!("chop_latent_sources_{}_paired_profile_v1", reference.name)
}

/// Reproduce the diagnostic header from the current body with checked anchors.
pub fn source() -> Result<String, Refusal> {
    Ok(format!(
        "#pragma once\n#include \"sources_prefill.cuh\"\nnamespace chop::latent::sources_prefill {{\n{}\n}}\n",
        body(SOURCE)?
    ))
}

pub fn cubin_spec(profile: Profile, entries: &[sources::Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = reuse_cubin_spec(profile, entries, 2)?;
    spec.name = format!(
        "latent_sources_prefill_pair_profile_{}_sm120",
        profile.tag()
    );
    spec.headers = vec![format!("#include \"{DIAGNOSTIC_HEADER}\"")];
    for (entry, reference) in spec.entries.iter_mut().zip(entries) {
        entry.symbol = symbol(reference);
        entry.params.push(kernel_build::EntryParam::DevicePointer {
            name: "stamps".into(),
            cxx_type: "uint64_t".into(),
        });
        once(
            &mut entry.body,
            "::reuse_attention<",
            "::reuse_profile_attention<",
        )?;
        once(&mut entry.body, ");", ",stamps);")?;
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_profile_rejects_missing_or_ambiguous_anchors() {
        assert!(body("").is_err());
        let duplicate = SOURCE.replace(
            "  const bool row_live = row < live;",
            "  const bool row_live = row < live;\n  const bool row_live = row < live;",
        );
        assert!(body(&duplicate).is_err());
        assert!(body(&SOURCE.replace("// PV: one chain from zero", "// PV changed")).is_err());
    }
    #[test]
    fn paired_profile_keeps_math_and_declares_its_extra_barrier() {
        assert_eq!(
            source().unwrap(),
            include_str!("../../cuda/latent/sources_prefill_profile.cuh")
        );
        let probe = body(SOURCE).unwrap();
        let old = SOURCE
            .split_once(BEGIN)
            .unwrap()
            .1
            .split_once(END)
            .unwrap()
            .0;
        let math = |source: &str| {
            source
                .lines()
                .map(str::trim)
                .filter(|line| {
                    line.contains("ptx::mma_bf16")
                        || line.contains("fmaf(")
                        || line.contains("sum +=")
                        || line.contains("denominator[n] +=")
                        || line.contains("p[kt][n][i] = exp_of")
                })
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert_eq!(math(&probe), math(old));
        assert_eq!(
            probe.matches("__syncthreads();").count(),
            old.matches("__syncthreads();").count() + 1
        );
        for stage in 0..6 {
            assert_eq!(
                probe
                    .matches(&format!("profile_clock(sample, cycles, {stage}, last)"))
                    .count(),
                1
            );
        }
        assert!(probe.contains("record[11] = distinct_units"));
    }
}
