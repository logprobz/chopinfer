//! Skip wholly invalid units while retaining their positive-zero state FMAs.
use super::*;

pub fn symbol(entry: &sources::Entry, asynchronous: bool) -> String {
    format!(
        "{}_empty_v1",
        next::symbol(entry, asynchronous, next::Mode::Finish)
    )
}

pub fn cubin_spec(profile: Profile, entries: &[sources::Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = next::cubin_spec(profile, &[], Mode::Finish)?;
    spec.name = format!("latent_sources_prefill_head_empty_{}_sm120", profile.tag());
    spec.entries = entries
        .iter()
        .map(|e| head_entry(e, e.name.ends_with("_decoded"), false, Mode::Finish, true))
        .collect::<Result<_, _>>()?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_empty_unit_cannot_drop_the_positive_zero_fma() {
        let before = -0.0_f32;
        assert_ne!(before.to_bits(), before.mul_add(1.0, 0.0).to_bits());
        assert_eq!(before.mul_add(1.0, 0.0).to_bits(), 0);
    }
}
