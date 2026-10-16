//! Compile and audit every latent attention entry at the served shape (16
//! local heads, latent 512, 64-token pages, 2051-entry key lists, nope and
//! value widths 256) without loading a cubin, and check the SASS for the
//! intended instructions.

use kernel_build::{BuildOptions, Toolchain};
use kernel_core::latent_attention::{HeadWeightLayout, LatentAttentionQuery, LatentKvFormat};
use kernels_sm120::latent::{Entry, Geometry, Query, cubin_spec, entries};
use kernels_sm120::{Profile, ProfileNames};
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "latent-build",
    tag: "latent_build",
});

const SERVED: Geometry = Geometry {
    heads: 16,
    latent: 512,
    page_size: 64,
    max_keys: 2051,
    softmax_scale: 0.0625,
    nope: 256,
    v_dim: 256,
};

/// The SASS of one function, from `cuobjdump -sass` output.
fn function_sass<'a>(sass: &'a str, symbol: &str) -> &'a str {
    let header = format!("Function : {symbol}\n");
    let start = sass
        .find(&header)
        .unwrap_or_else(|| panic!("{symbol} has no SASS"));
    let body = &sass[start + header.len()..];
    &body[..body.find("Function : ").unwrap_or(body.len())]
}

#[test]
fn served_entries_build_without_spills_and_emit_mma() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(
        !toolchain.skipped,
        "latent attention compile audit requires Linux and CUDA"
    );
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "chopinfer-latent-build-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&out_dir).expect("create test output directory");
    let catalog = entries(PROFILE, SERVED);
    let spec = cubin_spec(PROFILE, &catalog).expect("served latent entries are valid");
    let symbols = spec
        .entries
        .iter()
        .map(|entry| entry.symbol.clone())
        .collect::<Vec<_>>();
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &out_dir,
        &out_dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit latent attention cubin");
    assert_eq!(built.len(), 1);
    let cubin = &built[0];
    assert_eq!(
        cubin
            .entries
            .iter()
            .map(|e| e.symbol.clone())
            .collect::<Vec<_>>(),
        symbols
    );
    let owner = |symbol: &str| -> &Entry {
        catalog
            .iter()
            .find(|e| e.symbol() == symbol || e.merge_symbol().as_deref() == Some(symbol))
            .expect("entry belongs to the catalog")
    };
    for entry in &cubin.entries {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{} uses a stack", entry.symbol);
        let main = owner(&entry.symbol);
        let dynamic = if main.symbol() == entry.symbol {
            main.dynamic_shared_bytes()
        } else {
            0
        };
        assert!(
            entry.shared_static_bytes + dynamic <= 101_376,
            "{} exceeds SM120 shared memory",
            entry.symbol
        );
    }

    let output = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&cubin.path)
        .output()
        .expect("cuobjdump SASS");
    assert!(output.status.success(), "cuobjdump failed");
    let sass = String::from_utf8(output.stdout).expect("SASS is UTF-8");
    // Every kernel releases its PDL dependents (PREEXIT) and waits on its
    // producer (ACQBULK); none uses TF32.
    for symbol in &symbols {
        let code = function_sass(&sass, symbol);
        for opcode in ["PREEXIT", "ACQBULK"] {
            assert!(code.contains(opcode), "{symbol} lacks {opcode}");
        }
        assert!(!code.contains("TF32"), "{symbol} uses TF32");
    }
    for main in &catalog {
        let symbol = main.symbol();
        let code = function_sass(&sass, &symbol);
        let expected: &[&str] = match main.query {
            Query::Latent(LatentAttentionQuery::KvWrite { kv, .. }) => {
                if kv == LatentKvFormat::Fp8E4M3Block128 {
                    // Satfinite E4M3 packing after the block maximum's
                    // butterfly reduction.
                    &["F2FP.SATFINITE.E4M3.F32.PACK_AB", "SHFL.BFLY"]
                } else {
                    &["STG"]
                }
            }
            // BF16 MMA on ldmatrix fragments (keys as rows for QK, the
            // transposed tile for PV) and cp.async staging.
            Query::Latent(_) => &[
                "HMMA.16816.F32.BF16",
                "LDGSTS",
                "LDSM.16.M88.4",
                "LDSM.16.MT88.4",
            ],
            Query::HeadGemm(gemm) => {
                if gemm.layout == HeadWeightLayout::KByN {
                    &["HMMA.16816.F32.BF16", "LDGSTS", "LDSM.16.MT88.4"]
                } else {
                    &["HMMA.16816.F32.BF16", "LDGSTS", "LDSM.16.M88.4"]
                }
            }
        };
        for opcode in expected {
            assert!(code.contains(opcode), "{symbol} lacks {opcode}");
        }
        if let Query::Latent(LatentAttentionQuery::Attention { kv, .. }) = main.query {
            assert_eq!(
                code.contains("F2FP.BF16.E4M3.UNPACK_B"),
                kv == LatentKvFormat::Fp8E4M3Block128,
                "{symbol} widens FP8 codes exactly when its pages are FP8"
            );
        }
        if matches!(main.query, Query::HeadGemm(_)) {
            // The first weight chunk's copies are issued before the wait.
            assert!(
                code.find("LDGSTS") < code.find("ACQBULK"),
                "{symbol} does not prefetch weights before the PDL wait"
            );
        }
    }
    fs::remove_dir_all(out_dir).expect("remove hermetic test output");
}

/// The KV write at every latent width the contract accepts (multiples of 128
/// up to 1024), widths that are not powers of two among them: each lane
/// owns four elements of every 128-element block, so each width compiles
/// without spills (the lab GPU test
/// `latent_kv_write_matches_the_oracle_at_every_width` checks the bytes).
#[test]
fn kv_write_builds_at_every_accepted_width() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(!toolchain.skipped, "KV write audit requires Linux and CUDA");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "chopinfer-latent-write-build-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&out_dir).expect("create test output directory");
    let writes = (1..=8)
        .flat_map(|blocks| {
            let latent = 128 * blocks;
            entries(PROFILE, Geometry { latent, ..SERVED })
                .into_iter()
                .filter(|e| matches!(e.query, Query::Latent(LatentAttentionQuery::KvWrite { .. })))
                .map(move |e| Entry {
                    name: format!("{}_w{latent}", e.name),
                    ..e
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(writes.len(), 16);
    for entry in &writes {
        let plan = entry.plan(entry.capacity()).expect("every width plans");
        assert_eq!(plan.block, [128, 1, 1]);
    }
    let spec = cubin_spec(PROFILE, &writes).expect("KV write entries are valid");
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &out_dir,
        &out_dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit KV writes at every width");
    assert_eq!(built[0].entries.len(), 16);
    for entry in &built[0].entries {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{} uses a stack", entry.symbol);
    }
    fs::remove_dir_all(out_dir).expect("remove hermetic test output");
}
