//! Derived head-stage clocks and an exact single-split output epilogue.
use super::*;
use kernel_build::EntryParam;

const NATIVE: &str = include_str!("../../cuda/latent/sources_prefill_heads.cuh");
const ASYNC: &str = include_str!("../../cuda/latent/sources_prefill_heads_async.cuh");
pub const WORDS: usize = 14;
pub const STAGES: [&str; 6] = ["setup", "gather", "qk", "softmax", "pv", "epilogue"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Profile,
    Finish,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Profile => "profile",
            Self::Finish => "finish",
        }
    }
}
fn once(text: &mut String, old: &str, new: &str) -> Result<(), Refusal> {
    if text.matches(old).count() != 1 {
        return Err(Refusal("head source anchor changed"));
    }
    *text = text.replacen(old, new, 1);
    Ok(())
}
pub fn add_global_timer(text: &mut String) -> Result<(), Refusal> {
    once(
        text,
        "  const uint64_t started = sample ? clock64() : 0;",
        "  uint64_t global_start = 0;\n  if (sample) asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(global_start));\n  const uint64_t started = sample ? clock64() : 0;",
    )?;
    once(text, "stamps + cta * 12", "stamps + cta * 14")?;
    once(
        text,
        "    record[2] = smid;",
        "    record[2] = smid;\n    uint64_t global_end; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(global_end));\n    record[12] = global_start; record[13] = global_end;",
    )
}
fn body(original: &str, asynchronous: bool, mode: Mode) -> Result<String, Refusal> {
    let start = original
        .find("template<")
        .ok_or(Refusal("missing head template"))?;
    let end = original
        .rfind("} // namespace")
        .ok_or(Refusal("missing head namespace"))?;
    let mut text = original[start..end].trim().to_owned();
    let base = if asynchronous {
        "head_async_attention"
    } else {
        "head_shared_attention"
    };
    once(
        &mut text,
        &format!("void {base}("),
        &format!("void {base}_{}(", mode.name()),
    )?;
    match mode {
        Mode::Finish => {
            once(
                &mut text,
                "uint32_t compressed_budget, uint32_t scale_bits) {",
                "uint32_t compressed_budget, uint32_t scale_bits, const float* sinks, const float* cos_sin, uint16_t* out) {",
            )?;
            once(
                &mut text,
                "  if (!row_live) return;",
                "  if (!row_live) return;\n  __syncthreads();",
            )?;
            once(
                &mut text,
                "      float* dest = workspace + partial_index<T>(row, split, head);",
                "      float* dest = head_finish_partial<T>(tile, group * T::head_tile + n * 8 + 2 * c + r);\n      float* audit = workspace ? workspace + partial_index<T>(row, split, head) : nullptr;",
            )?;
            once(
                &mut text,
                "      if (warp == 0 && g == 0) { dest[0] = m_head; dest[1] = l_head; }",
                "      if (warp == 0 && g == 0) {\n        dest[0] = m_head; dest[1] = l_head;\n        if (audit) { audit[0] = m_head; audit[1] = l_head; }\n      }",
            )?;
            once(
                &mut text,
                "          dest[2 + warp * 128 + m * 16 + g + half * 8] = output[m][n].v[2 * half + r];",
                "        {\n          const int dim = warp * 128 + m * 16 + g + half * 8;\n          dest[4 + dim] = output[m][n].v[2 * half + r];\n          if (audit) audit[2 + dim] = output[m][n].v[2 * half + r];\n        }",
            )?;
            let last = text
                .strip_suffix('}')
                .ok_or(Refusal("missing head body end"))?;
            text = format!(
                "{last}\n  head_finish<T>(tile, row, blockIdx.y * 16, sinks, cos_sin, out);\n}}"
            );
        }
        Mode::Profile => {
            once(
                &mut text,
                "uint32_t compressed_budget, uint32_t scale_bits) {",
                "uint32_t compressed_budget, uint32_t scale_bits, uint64_t* stamps) {",
            )?;
            once(
                &mut text,
                "  const bool row_live = row < live;",
                "  const bool row_live = row < live;\n  const bool sample = stamps != nullptr && threadIdx.x == 0;\n  const uint64_t started = sample ? clock64() : 0;\n  uint64_t last = started, cycles[6] = {};",
            )?;
            once(
                &mut text,
                "  for (uint32_t unit = begin; unit < end; ++unit) {",
                "  profile_clock(sample, cycles, 0, last);\n  for (uint32_t unit = begin; unit < end; ++unit) {",
            )?;
            once(
                &mut text,
                "      const bool active = row_live;",
                "      profile_clock(sample, cycles, 1, last);\n      const bool active = row_live;",
            )?;
            let probability = "uint32_t probability[terms][T::key_tiles][T::n_tiles][2];";
            let rescale = "float rescale[T::n_tiles][2];";
            once(&mut text, &format!("        {probability}"), "")?;
            once(&mut text, &format!("        {rescale}"), "")?;
            once(
                &mut text,
                "      if (active) {\n        // Scores",
                &format!(
                    "      profile_clock(sample, cycles, 2, last);\n      {probability}\n      {rescale}\n      if (active) {{\n        // Scores"
                ),
            )?;
            let pv = if asynchronous {
                "        if constexpr (Pipeline) {"
            } else {
                "        // PV: one chain from zero"
            };
            once(
                &mut text,
                pv,
                &format!(
                    "      }}\n      __syncthreads();\n      profile_clock(sample, cycles, 3, last);\n      if (active) {{\n{pv}"
                ),
            )?;
            once(
                &mut text,
                "      __syncthreads();\n    }\n  }\n  if (!row_live) return;",
                "      __syncthreads();\n      profile_clock(sample, cycles, 4, last);\n    }\n  }\n  if (!row_live) return;",
            )?;
            let last = text
                .strip_suffix('}')
                .ok_or(Refusal("missing head body end"))?;
            text = format!(
                r#"{last}
  __syncthreads();
  profile_clock(sample, cycles, 5, last);
  if (sample) {{
    const uint64_t cta = (uint64_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    uint64_t* record = stamps + cta * 12;
    record[0] = started; record[1] = last;
    uint32_t smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    record[2] = smid;
    for (int stage = 0; stage < 6; ++stage) record[3 + stage] = cycles[stage];
    record[9] = end - begin; record[10] = row; record[11] = blockIdx.y;
  }}
}}"#
            );
        }
    }
    if mode == Mode::Profile {
        add_global_timer(&mut text)?;
    }
    Ok(text)
}
pub fn source(mode: Mode) -> Result<String, Refusal> {
    let mut code = "#pragma once\n#include \"sources_prefill_heads_async.cuh\"\n#include \"sources_prefill_finish.cuh\"\nnamespace chop::latent::sources_prefill {\n".to_owned();
    code.push_str(&body(NATIVE, false, mode)?);
    code.push('\n');
    code.push_str(&body(ASYNC, true, mode)?);
    code.push_str("\n} // namespace chop::latent::sources_prefill\n");
    Ok(code)
}
pub fn symbol(entry: &sources::Entry, asynchronous: bool, mode: Mode) -> String {
    format!(
        "chop_latent_sources_{}_head_{}_{}_v1",
        entry.name,
        if asynchronous { "async" } else { "shared" },
        mode.name()
    )
}
pub fn cubin_spec(
    profile: Profile,
    entries: &[sources::Entry],
    mode: Mode,
) -> Result<CubinSpec, Refusal> {
    let mut spec = super::cubin_spec(profile, entries)?;
    spec.entries.clear();
    spec.name = format!(
        "latent_sources_prefill_head_{}_{}_sm120",
        mode.name(),
        profile.tag()
    );
    spec.headers = vec![format!(
        "#include \"latent/sources_prefill_heads_{}.cuh\"",
        mode.name()
    )];
    for entry in entries {
        let asynchronous = entry.name.ends_with("_decoded");
        let mut one = if asynchronous {
            super::asynchronous::cubin_spec(profile, std::slice::from_ref(entry), false)?
        } else {
            super::cubin_spec(profile, std::slice::from_ref(entry))?
        };
        let mut body = one.entries.remove(0);
        body.symbol = symbol(entry, asynchronous, mode);
        let old = if asynchronous {
            "::head_async_attention<"
        } else {
            "::head_shared_attention<"
        };
        once(
            &mut body.body,
            old,
            &old.replace("<", &format!("_{}<", mode.name())),
        )?;
        let parameters = if mode == Mode::Profile {
            vec![("stamps", "uint64_t")]
        } else {
            vec![
                ("sinks", "const float"),
                ("cos_sin", "const float"),
                ("out", "uint16_t"),
            ]
        };
        for (name, cxx_type) in &parameters {
            body.params.push(EntryParam::DevicePointer {
                name: (*name).into(),
                cxx_type: (*cxx_type).into(),
            });
        }
        once(
            &mut body.body,
            ");",
            &format!(
                ",{});",
                parameters.iter().map(|p| p.0).collect::<Vec<_>>().join(",")
            ),
        )?;
        spec.entries.push(body);
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn math(text: &str) -> Vec<&str> {
        text.lines()
            .map(str::trim)
            .filter(|s| {
                s.contains("ptx::mma_bf16")
                    || s.contains("fmaf(")
                    || s.contains("sum +=")
                    || s.contains("denominator[n] +=")
                    || s.contains("p[kt][n][i] = exp_of")
            })
            .collect()
    }
    #[test]
    fn derived_heads_preserve_the_winning_chains_and_reject_stale_anchors() {
        for (original, asynchronous) in [(NATIVE, false), (ASYNC, true)] {
            let original_body = &original[original.find("template<").unwrap()..];
            for mode in [Mode::Profile, Mode::Finish] {
                let new = body(original, asynchronous, mode).unwrap();
                assert_eq!(math(original_body), math(&new));
                assert!(
                    body(
                        &original.replace("const bool row_live", "const bool changed_live"),
                        asynchronous,
                        Mode::Profile
                    )
                    .is_err()
                );
                assert!(body("", asynchronous, mode).is_err());
            }
        }
        assert_eq!(
            source(Mode::Profile).unwrap(),
            include_str!("../../cuda/latent/sources_prefill_heads_profile.cuh")
        );
        assert_eq!(
            source(Mode::Finish).unwrap(),
            include_str!("../../cuda/latent/sources_prefill_heads_finish.cuh")
        );
    }
    #[test]
    fn finish_shared_layout_is_disjoint_and_fixed_r_stores_visit_every_bank() {
        let mut addresses = std::collections::BTreeSet::new();
        for group in 0..2 {
            for warp in 0..4 {
                for r in 0..2 {
                    for m in 0..8 {
                        for half in 0..2 {
                            let mut banks = std::collections::BTreeSet::new();
                            for lane in 0..32 {
                                let (g, c) = (lane / 4, lane % 4);
                                let at = (group * 8 + 2 * c + r) * 516
                                    + 4
                                    + warp * 128
                                    + m * 16
                                    + g
                                    + half * 8;
                                assert!(addresses.insert(at));
                                banks.insert(at % 32);
                            }
                            assert_eq!(banks.len(), 32);
                        }
                    }
                }
            }
        }
        assert_eq!(addresses.len(), 16 * 512);
        assert!((addresses.last().unwrap() + 1) * 4 <= 64 * 512 * 2);
        for head in 0..16 {
            for metadata in 0..4 {
                assert!(!addresses.contains(&(head * 516 + metadata)));
            }
        }
    }
}
