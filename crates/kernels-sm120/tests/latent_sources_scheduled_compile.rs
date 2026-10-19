//! Compile the additive schedules without opening CUDA or changing serving locks.

use kernel_build::{BuildOptions, Toolchain};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat};
use kernels_sm120::latent::sources::{self, Schedule};
use kernels_sm120::{Profile, ProfileNames};

#[test]
fn scheduled_sources_compile_without_spills() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let profile = Profile::synthetic(ProfileNames {
        name: "scheduled-sources-test",
        tag: "scheduled_sources_test",
    });
    let mut entries = Vec::new();
    for latent in [128, 512] {
        for page_size in [32, 64] {
            let query = LatentAttentionQuery::Attention {
                heads: 16,
                latent,
                page_size,
                max_keys: 2176,
                softmax_scale_bits: (latent as f32).powf(-0.5).to_bits(),
                kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
                phase: AttentionPhase::Decode,
                output: LatentAttentionOutput::Normalized,
                sink: true,
                tail_rope_dims: 64,
                second_fp4_keys: true,
                capacity: 192,
            };
            let mut entry = sources::paged_entries(profile, query).unwrap().remove(1);
            entry.schedule = Schedule::IndependentHeads;
            entries.push(entry);
        }
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out = std::env::temp_dir().join(format!("chopinfer-sources-scheduled-{stamp}"));
    std::fs::create_dir(&out).unwrap();
    let spec = sources::cubin_spec(profile, &entries).unwrap();
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &out,
        &out.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit scheduled sources");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), 8);
    for report in &built[0].entries {
        assert_eq!(report.local_bytes, 0, "{} spills", report.symbol);
        assert_eq!(report.stack_bytes, 0, "{} uses a stack", report.symbol);
        let owner = entries
            .iter()
            .find(|e| e.symbol() == report.symbol || e.merge_symbol() == report.symbol)
            .unwrap();
        let dynamic = if owner.symbol() == report.symbol {
            owner.dynamic_shared_bytes()
        } else {
            0
        };
        assert!(dynamic + report.shared_static_bytes <= 101_376);
    }
    let sass = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(sass.status.success());
    let sass = String::from_utf8(sass.stdout).unwrap();
    assert!(sass.contains("HMMA"));
    assert!(!sass.contains("TF32"));
    for entry in &entries {
        let marker = format!("Function : {}\n", entry.symbol());
        let body = sass
            .split_once(&marker)
            .unwrap()
            .1
            .split("Function : ")
            .next()
            .unwrap();
        assert!(body.contains("PREEXIT"));
        assert!(body.contains("ACQBULK"));
        assert!(
            body.contains("STS.128"),
            "source widening must store complete 16-byte sectors"
        );
    }
    std::fs::remove_dir_all(out).unwrap();
}
