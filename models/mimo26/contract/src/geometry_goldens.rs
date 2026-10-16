//! Golden captured from the expansion before geometry replaced its literals.
use sha2::{Digest, Sha256};

#[test]
fn flash_catalog_and_rank_views_match_captured_expansion() {
    let mut output = String::new();
    output.push_str(&format!(
        "crate::kernels::attention::catalog() = {:?}\n",
        crate::sm120::attention::catalog()
            .into_iter()
            .filter(|entry| entry
                .profile
                .ranks_of(crate::sm120::config::MODEL)
                .is_some())
            .collect::<Vec<_>>()
    ));
    output.push_str(&format!(
        "crate::kernels::gemm::catalog() = {:?}\n",
        crate::sm120::gemm::catalog()
            .into_iter()
            .filter(|entry| entry
                .profile
                .ranks_of(crate::sm120::config::MODEL)
                .is_some())
            .collect::<Vec<_>>()
    ));
    output.push_str(&format!(
        "crate::kernels::moe::catalog() = {:?}\n",
        crate::sm120::moe::catalog()
            .into_iter()
            .filter(|entry| entry
                .profile
                .ranks_of(crate::sm120::config::MODEL)
                .is_some())
            .collect::<Vec<_>>()
    ));
    output.push_str(&format!(
        "crate::kernels::row::catalog() = {:?}\n",
        crate::sm120::row::catalog_for(crate::sm120::geometry::MODELS[0])
    ));
    output.push_str(&format!(
        "crate::encoders::expected() = {:?}\n",
        crate::encoders::expected_flash()
    ));
    for tp in crate::Tp::SUPPORTED {
        output.push_str(&format!(
            "{:?} {:?} {:?} {:?}\n",
            tp,
            crate::text_views(tp),
            crate::drafters::mtp_fields(tp),
            crate::drafters::fields(tp).collect::<Vec<_>>()
        ));
    }
    // 2026-10-05: the wide-count prefill rows (TP4 down, TP2 gate/up and
    // down) change only those three resolved schedules; with their old
    // knobs the expansion hashes to the previous golden, 7874afe4...
    // 2026-10-10: the sharded argmax queries' new fields change only the four
    // DFlash argmax entries' query text.
    // 2026-10-14: rank counts became data, which changes only each profile's
    // debug text (its model's names keyed by rank count, and the rank count
    // as a number); written as before, the expansion hashes to the previous
    // golden, a3fcbadc...
    assert_eq!(
        format!("{:x}", Sha256::digest(output.as_bytes())),
        "1ef36c74d1dfd85b2b73834a32310432496283a461b258a5a6dc083551b25591",
        "catalog expansion differs:\n{output}"
    );
}
