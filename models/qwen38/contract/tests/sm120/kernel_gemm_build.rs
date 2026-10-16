//! Compile-only audit of the Qwen3.8 fused-weight GEMM cubins, over its
//! catalogs (`sm120::catalogs`).

use kernel_build::{BuildOptions, Toolchain};
use kernels_sm120::gemm::fused;

#[test]
fn qwen_gemm_kernels_compile_without_spills() {
    assert!(
        fused::cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            Some(
                kernels_sm120::Model::new(
                    kernels_sm120::profile_names!("other-model", "other", 4, 2),
                    model_qwen38_contract::sm120::config::CONFIG
                )
                .at(4)
            )
        )
        .is_empty()
    );
    let root = std::env::temp_dir().join(format!("qwen38-gemm-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let reports = kernel_build::build(
        &Toolchain::detect().unwrap(),
        &fused::cubin_specs(model_qwen38_contract::sm120::catalogs(), None),
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    let all = fused::catalog(model_qwen38_contract::sm120::catalogs());
    let profiles = model_qwen38_contract::sm120::PROFILES;
    let precomputed_profiles = profiles
        .iter()
        .filter(|&&profile| {
            all.iter()
                .any(|entry| entry.profile == profile && fused::hc_precompute_spec(entry).is_ok())
        })
        .count();
    assert_eq!(reports.len(), profiles.len() + precomputed_profiles);
    for profile in profiles {
        let owner = model_qwen38_contract::sm120::geometry::owner(profile).unwrap();
        let entries = all
            .iter()
            .filter(|entry| entry.profile == profile)
            .collect::<Vec<_>>();
        let report = reports
            .iter()
            .find(|report| report.name == format!("gemm_{}_own_sm120", profile.tag()))
            .unwrap();
        let mix = profile.ranks() == Some(4);
        assert_eq!(report.entries.len(), entries.len() + usize::from(mix));
        for (built, entry) in report.entries.iter().zip(&entries) {
            assert_eq!(built.symbol, entry.symbol);
            assert_eq!(built.local_bytes, 0, "{} spills", built.symbol);
            let plan = entry.plan(entry.query.m_capacity).unwrap();
            assert!(
                built.shared_static_bytes + plan.dynamic_shared_bytes <= 101_376,
                "{} exceeds shared memory",
                built.symbol
            );
        }
        if mix {
            let built = report.entries.last().unwrap();
            assert_eq!(built.symbol, owner.hc.up_mix_symbol);
            assert_eq!(built.local_bytes, 0, "the up mix spills");
            assert!(
                built.shared_static_bytes
                    + fused::hc_up_mix_plan(owner.hc, 32)
                        .unwrap()
                        .dynamic_shared_bytes
                    <= 101_376
            );
        }
        if let Some(hc) = reports
            .iter()
            .find(|report| report.name == format!("gemm_{}_hc_precompute_sm120", profile.tag()))
        {
            let producers = entries
                .iter()
                .filter(|entry| fused::hc_precompute_spec(entry).is_ok())
                .count();
            assert_eq!(hc.entries.len(), producers + 2);
            for built in &hc.entries {
                assert_eq!(built.local_bytes, 0, "{} spills", built.symbol);
                assert!(
                    built.shared_static_bytes
                        + fused::hc_precompute_shared(
                            model_qwen38_contract::sm120::catalogs(),
                            &built.symbol
                        )
                        .unwrap()
                        <= 101_376,
                    "{} exceeds shared memory",
                    built.symbol
                );
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
