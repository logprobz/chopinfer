//! Head preparation checks and the same locked cubin used by the bundle.

use crate::dspark_entries::{CONFIDENCE, CUBIN, FAMILY, MARKOV};
use crate::own_ops::Sm120Own;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_core::backend::{Counts, Need, View};
use kernel_core::properties::Invariance;
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::dspark::{
    ConfidenceHead, ConfidenceHeadBuffers, MarkovStep, MarkovStepBuffers,
};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn cubin() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        // Queued jobs use an immutable prebuilt snapshot and check its hash.
        if let Some(path) = std::env::var_os("CHOPINFER_DEEPSEEK41_HEAD_CUBIN") {
            return std::fs::read(path).expect("prebuilt DSpark head cubin");
        }
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../bundle")
            .canonicalize()
            .unwrap();
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("dspark-heads");
        std::fs::create_dir_all(&out).unwrap();
        let built = kernel_build::build_in_families(
            &Toolchain::detect().unwrap(),
            &[crate::dspark_cubin::cubin_spec(&bundle)],
            &out,
            &LockDir::new(bundle.join("kernels.lock.d")).family(FAMILY, [CUBIN]),
            BuildOptions {
                lock_update: false,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
            None,
        )
        .expect("DSpark cubin matches its lock");
        std::fs::read(&built[0].path).unwrap()
    })
}

fn parallel_cubin() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        if let Some(path) = std::env::var_os("CHOPINFER_DEEPSEEK41_CONFIDENCE_CUBIN") {
            return std::fs::read(path).expect("prebuilt parallel confidence cubin");
        }
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../bundle")
            .canonicalize()
            .unwrap();
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("confidence-parallel");
        std::fs::create_dir_all(&out).unwrap();
        let built = kernel_build::build_in_families(
            &Toolchain::detect().unwrap(),
            &[crate::confidence_parallel::cubin_spec(&bundle)],
            &out,
            &LockDir::new(bundle.join("kernels.lock.d")).family(
                crate::confidence_parallel::FAMILY,
                [crate::confidence_parallel::CUBIN],
            ),
            BuildOptions {
                lock_update: false,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
            None,
        )
        .expect("parallel confidence cubin matches its lock");
        std::fs::read(&built[0].path).unwrap()
    })
}

fn tiled_cubin() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        if let Some(path) = std::env::var_os("CHOPINFER_DEEPSEEK41_MARKOV_CUBIN") {
            return std::fs::read(path).expect("prebuilt tiled Markov cubin");
        }
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../bundle")
            .canonicalize()
            .unwrap();
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("markov-tiled");
        std::fs::create_dir_all(&out).unwrap();
        let built = kernel_build::build_in_families(
            &Toolchain::detect().unwrap(),
            &[crate::markov_tiled::cubin_spec(&bundle)],
            &out,
            &LockDir::new(bundle.join("kernels.lock.d"))
                .family(crate::markov_tiled::FAMILY, [crate::markov_tiled::CUBIN]),
            BuildOptions {
                lock_update: false,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
            None,
        )
        .expect("tiled Markov cubin matches its lock");
        std::fs::read(&built[0].path).unwrap()
    })
}

#[test]
fn head_cpu_tiled_markov_matches_lock_and_waits_before_reads() {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("markov-tiled-audit.cubin");
    std::fs::write(&path, tiled_cubin()).unwrap();
    let output =
        std::process::Command::new(Toolchain::detect().unwrap().home.join("bin/cuobjdump"))
            .arg("--dump-sass")
            .arg(&path)
            .output()
            .unwrap();
    assert!(output.status.success());
    let sass = String::from_utf8(output.stdout).unwrap();
    let body = sass
        .split(&format!("Function : {}", crate::markov_tiled::SYMBOL))
        .nth(1)
        .expect("tiled Markov symbol");
    assert!(body.find("PREEXIT").unwrap() < body.find("ACQBULK").unwrap());
    assert!(body.find("ACQBULK").unwrap() < body.find("LDG.").unwrap());
    assert!(body.contains("FFMA") && body.contains("FADD") && body.contains("BAR.SYNC"));
    for forbidden in ["TF32", "HMMA", "HGMMA", "UTMMA"] {
        assert!(!body.contains(forbidden), "unexpected {forbidden}");
    }
    // The tile index divisions compile to the integer divide sequence, whose
    // reciprocal truncation is F2I.FTZ. No floating-point arithmetic flushes.
    for line in body.lines().filter(|line| line.contains(".FTZ")) {
        assert!(line.contains("F2I.FTZ.U32.TRUNC"), "unexpected {line}");
    }
}

#[test]
fn head_cpu_parallel_confidence_matches_lock_and_waits_before_reads() {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("confidence-parallel-audit.cubin");
    std::fs::write(&path, parallel_cubin()).unwrap();
    let output =
        std::process::Command::new(Toolchain::detect().unwrap().home.join("bin/cuobjdump"))
            .arg("--dump-sass")
            .arg(&path)
            .output()
            .unwrap();
    assert!(output.status.success());
    let sass = String::from_utf8(output.stdout).unwrap();
    let body = sass
        .split(&format!(
            "Function : {}",
            crate::confidence_parallel::SYMBOL
        ))
        .nth(1)
        .expect("parallel confidence symbol");
    assert!(body.find("PREEXIT").unwrap() < body.find("ACQBULK").unwrap());
    assert!(body.find("ACQBULK").unwrap() < body.find("LDG.").unwrap());
    assert!(body.contains("FFMA") && body.contains("FADD") && body.contains("BAR.SYNC"));
    for forbidden in ["TF32", "HMMA", "HGMMA", "UTMMA", ".FTZ"] {
        assert!(!body.contains(forbidden), "unexpected {forbidden}");
    }
}

#[test]
fn head_cpu_cubin_matches_lock_without_tf32() {
    let bytes = cubin();
    for symbol in [MARKOV, CONFIDENCE] {
        assert!(bytes.windows(symbol.len()).any(|b| b == symbol.as_bytes()));
    }
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("dspark-audit.cubin");
    std::fs::write(&path, bytes).unwrap();
    let output =
        std::process::Command::new(Toolchain::detect().unwrap().home.join("bin/cuobjdump"))
            .arg("--dump-sass")
            .arg(&path)
            .output()
            .unwrap();
    assert!(output.status.success());
    let sass = String::from_utf8(output.stdout).unwrap();
    assert!(sass.contains("FFMA"));
    assert!(sass.contains("FADD"));
    for symbol in [MARKOV, CONFIDENCE] {
        let body = sass.split(&format!("Function : {symbol}")).nth(1).unwrap();
        let body = body.split("Function :").next().unwrap();
        // SM120 lowers the PTX release/wait to PREEXIT and ACQBULK.
        let release = body.find("PREEXIT").expect("PDL release");
        let wait = body.find("ACQBULK").expect("PDL acquire");
        let first_read = body.find("LDG.").expect("device input read");
        assert!(release < wait && wait < first_read);
    }
    for forbidden in ["TF32", "HMMA", "HGMMA", "UTMMA"] {
        assert!(
            !sass.contains(forbidden),
            "unexpected {forbidden} instruction"
        );
    }
}

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

#[test]
fn head_cpu_prepares_every_registered_geometry() {
    for g in model_deepseek41_contract::geometry::GEOMETRIES {
        let rank = g.shape.dspark.markov_rank;
        let op = ConfidenceHead {
            hidden: g.shape.hidden,
            markov_rank: rank,
            hidden_stride: g.shape.hidden * 5,
            score_stride: 5,
        };
        assert!(
            Sm120Own::new(g)
                .dspark_confidence(&op, Need::rows(7))
                .is_ok()
        );
        if g.id == "synthetic_small" {
            assert_eq!(g.layers(), 12);
        }
    }
}

#[test]
fn head_cpu_markov_checks_spans_aliases_and_all_vocab_shards() {
    let rank = FLASH.shape.dspark.markov_rank;
    let need = Need::rows(7).with_exact(Invariance::any_rows(7));
    for tp in [1, 2, 4] {
        let vocab = FLASH.shape.vocab / tp;
        let op = MarkovStep {
            rank,
            vocab_rows: vocab,
            logits_stride: 5 * vocab,
        };
        let prepared = Sm120Own::new(&FLASH).dspark_markov_step(&op, need).unwrap();
        let bytes = u64::from(6 * op.logits_stride + vocab) * 4;
        let b = MarkovStepBuffers {
            embedding: view(1, u64::from(7 * rank) * 2),
            weight: view(2, u64::from(vocab) * u64::from(rank) * 2),
            base: view(3, bytes),
            output: view(4, bytes),
            live_rows: view(5, 4),
        };
        let launch = prepared.launches(&b, View::EMPTY, Counts::rows(7)).unwrap();
        assert_eq!(launch.len(), 1);
        assert!(launch[0].plan.pdl);
        assert_eq!(launch[0].symbol, crate::markov_tiled::SYMBOL);
        assert_eq!(launch[0].plan.grid, [vocab.div_ceil(128), 1, 1]);
        // One row takes the row step: 64 logits and 256 threads per CTA.
        let single = prepared.launches(&b, View::EMPTY, Counts::rows(1)).unwrap();
        assert_eq!(single[0].symbol, crate::markov_row::SYMBOL);
        assert_eq!(single[0].plan.grid, [vocab.div_ceil(64), 1, 1]);
        assert_eq!(single[0].plan.block, [256, 1, 1]);
        assert_eq!(
            single[0].plan.dynamic_shared_bytes,
            crate::markov_row::shared_bytes(rank)
        );
        assert!(single[0].plan.pdl);
        assert_eq!(single[0].arguments, launch[0].arguments);
        let two = prepared.launches(&b, View::EMPTY, Counts::rows(2)).unwrap();
        assert_eq!(two[0].symbol, crate::markov_tiled::SYMBOL);
        let symbols: Vec<_> = prepared
            .info()
            .symbols
            .iter()
            .map(|(s, _)| s.as_str())
            .collect();
        assert_eq!(
            symbols,
            [
                crate::markov_tiled::SYMBOL,
                MARKOV,
                crate::markov_row::SYMBOL
            ]
        );
        let alias = MarkovStepBuffers {
            output: b.base,
            ..b
        };
        assert!(
            prepared
                .launches(&alias, View::EMPTY, Counts::rows(7))
                .is_ok()
        );
        for bad in [
            MarkovStepBuffers {
                output: b.embedding,
                ..b
            },
            MarkovStepBuffers {
                base: view(3, bytes - 1),
                ..b
            },
            MarkovStepBuffers {
                live_rows: view(5, 3),
                ..b
            },
        ] {
            assert!(
                prepared
                    .launches(&bad, View::EMPTY, Counts::rows(7))
                    .is_err()
            );
        }
        for rows in [0, 8] {
            assert!(
                prepared
                    .launches(&b, View::EMPTY, Counts::rows(rows))
                    .is_err()
            );
        }
        assert!(
            Sm120Own::new(&FLASH)
                .dspark_markov_step(&op, Need::rows(65536))
                .is_err()
        );
        assert!(
            Sm120Own::new(&FLASH)
                .dspark_markov_step(
                    &MarkovStep {
                        rank: rank + 1,
                        ..op
                    },
                    need
                )
                .is_err()
        );
    }
}

#[test]
fn head_cpu_confidence_checks_strides_aliases_and_capacity() {
    let (hidden, rank) = (FLASH.shape.hidden, FLASH.shape.dspark.markov_rank);
    let op = ConfidenceHead {
        hidden,
        markov_rank: rank,
        hidden_stride: 5 * hidden,
        score_stride: 5,
    };
    let prepared = Sm120Own::new(&FLASH)
        .dspark_confidence(&op, Need::rows(7))
        .unwrap();
    let b = ConfidenceHeadBuffers {
        hidden: view(1, u64::from(31 * hidden) * 4),
        markov: view(2, u64::from(7 * rank) * 2),
        weight: view(3, u64::from(hidden + rank) * 2),
        scores: view(4, 31 * 4),
        live_rows: view(5, 4),
    };
    assert_eq!(
        prepared.launches(&b, View::EMPTY, Counts::rows(7)).unwrap()[0]
            .plan
            .grid,
        [7, 1, 1],
        "confidence needs one cooperative block per item"
    );
    assert!(
        prepared.launches(&b, View::EMPTY, Counts::rows(7)).unwrap()[0]
            .plan
            .pdl
    );
    for bad in [
        ConfidenceHeadBuffers {
            hidden: view(1, b.hidden.bytes / 2),
            ..b
        },
        ConfidenceHeadBuffers {
            hidden: view(1, b.hidden.bytes - 1),
            ..b
        },
        ConfidenceHeadBuffers {
            scores: view(4, b.scores.bytes - 1),
            ..b
        },
        ConfidenceHeadBuffers {
            scores: b.hidden,
            ..b
        },
    ] {
        assert!(
            prepared
                .launches(&bad, View::EMPTY, Counts::rows(7))
                .is_err()
        );
    }
    assert!(
        Sm120Own::new(&FLASH)
            .dspark_confidence(&op, Need::rows(65536))
            .is_err()
    );
    assert!(
        Sm120Own::new(&FLASH)
            .dspark_confidence(
                &ConfidenceHead {
                    hidden: hidden - 1,
                    ..op
                },
                Need::rows(7)
            )
            .is_err()
    );
}

#[path = "dspark_heads/gpu.rs"]
mod gpu;
