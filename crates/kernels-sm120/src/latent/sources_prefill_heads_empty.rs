//! Skip key staging and MMA for wholly invalid units, retaining state FMAs.
use super::*;

const UPDATE: &str = r"    // Empty PV chains start at positive zero. Keep the original rescale
    // FMAs: even a zero contribution can canonicalize a negative zero.
    if (!__syncthreads_or(threadIdx.x < T::keys && valid[threadIdx.x] != 0)) {
      float correction[T::n_tiles];
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n) {
        const float next = fmaxf(maximum[n], -INFINITY);
        const float base = next == -INFINITY ? 0.0f : next;
        correction[n] = exp_of(maximum[n] - base);
        maximum[n] = next;
        denominator[n] = fmaf(denominator[n], correction[n], 0.0f);
      }
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float rescale = __shfl_sync(0xffffffff, correction[n], (2 * c + r) * 4);
#pragma unroll
          for (int m = 0; m < column_blocks; ++m)
#pragma unroll
            for (int half = 0; half < 2; ++half)
              output[m][n].v[2 * half + r] = fmaf(output[m][n].v[2 * half + r], rescale, 0.0f);
        }
      continue;
    }
";

fn once(text: &mut String, old: &str, new: &str) -> Result<(), Refusal> {
    if text.matches(old).count() != 1 {
        return Err(Refusal("empty-unit source anchor changed"));
    }
    *text = text.replacen(old, new, 1);
    Ok(())
}

pub fn source() -> Result<String, Refusal> {
    let mut text = next::source(next::Mode::Finish)?;
    for body in [
        "head_shared_attention_finish",
        "head_async_attention_finish",
    ] {
        once(
            &mut text,
            &format!("void {body}("),
            &format!("void {body}_empty("),
        )?;
    }
    once(
        &mut text,
        "    __syncthreads();\n    // Both eight-head groups",
        &format!("{UPDATE}    // Both eight-head groups"),
    )?;
    once(
        &mut text,
        "    if constexpr (!Pipeline) resolve(unit);\n",
        &format!(
            "    static_assert(!Pipeline, \"empty-unit skipping requires direct staging\");\n    resolve(unit);\n{UPDATE}"
        ),
    )?;
    Ok(text)
}

pub fn symbol(entry: &sources::Entry, asynchronous: bool) -> String {
    format!(
        "{}_empty_v1",
        next::symbol(entry, asynchronous, next::Mode::Finish)
    )
}

pub fn cubin_spec(profile: Profile, entries: &[sources::Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = next::cubin_spec(profile, entries, next::Mode::Finish)?;
    spec.name = format!("latent_sources_prefill_head_empty_{}_sm120", profile.tag());
    spec.headers = vec!["#include \"latent/sources_prefill_heads_empty.cuh\"".into()];
    for (body, entry) in spec.entries.iter_mut().zip(entries) {
        body.symbol = symbol(entry, entry.name.ends_with("_decoded"));
        for name in [
            "head_shared_attention_finish",
            "head_async_attention_finish",
        ] {
            body.body = body
                .body
                .replace(&format!("::{name}<"), &format!("::{name}_empty<"));
        }
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_units_keep_the_complete_nonempty_body() {
        let mut text = source().unwrap();
        assert_eq!(
            text,
            include_str!("../../cuda/latent/sources_prefill_heads_empty.cuh")
        );
        assert_eq!(text.matches(UPDATE).count(), 2);
        text = text.replace(UPDATE, "");
        text = text.replace(
            "head_shared_attention_finish_empty",
            "head_shared_attention_finish",
        );
        text = text.replace(
            "head_async_attention_finish_empty",
            "head_async_attention_finish",
        );
        text = text.replacen(
            "    // Both eight-head groups",
            "    __syncthreads();\n    // Both eight-head groups",
            1,
        );
        text = text.replace("    static_assert(!Pipeline, \"empty-unit skipping requires direct staging\");\n    resolve(unit);\n", "    if constexpr (!Pipeline) resolve(unit);\n");
        assert_eq!(text, next::source(next::Mode::Finish).unwrap());
        assert!(once(&mut text, "absent anchor", "").is_err());
    }
    #[test]
    fn an_empty_unit_cannot_drop_the_positive_zero_fma() {
        let before = -0.0_f32;
        assert_ne!(before.to_bits(), before.mul_add(1.0, 0.0).to_bits());
        assert_eq!(before.mul_add(1.0, 0.0).to_bits(), 0);
    }
}
