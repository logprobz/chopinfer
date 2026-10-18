//! CPU compilation and SASS audits for trained FP4 key writes.

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::compressed_kv::{CompressedKvStorage, CompressedKvWriteQuery};
use kernel_core::pooled_indexer::{IndexerCandidates, PositionIndexerQuery};
use kernels_sm120::gated_pooling::key_write::{Entry, cubin_spec};
use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn all_fp4_key_write_forms_compile_with_pdl_without_spills_or_tf32() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is required");
    assert!(!toolchain.skipped);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("fp4-key-write-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let mut entries = Vec::new();
    for width in [128, 512] {
        for ratio in [1, 2] {
            entries.push(
                Entry::index(
                    &PositionIndexerQuery {
                        heads: 32,
                        head_dim: width,
                        compression_ratio: ratio,
                        tail_rope_dims: 64,
                        key_norm_eps_bits: 1e-20f32.to_bits(),
                        max_top_k: 512,
                        page_size: 64,
                        candidates: IndexerCandidates::All,
                        capacity: 16,
                    },
                    &format!("chop_index_w{width}_r{ratio}_v1"),
                )
                .unwrap(),
            );
            for (name, storage) in [
                ("packed", CompressedKvStorage::Packed),
                ("bf16", CompressedKvStorage::Bf16),
            ] {
                entries.push(
                    Entry::compressed(
                        &CompressedKvWriteQuery {
                            width,
                            tail_rope_dims: 64,
                            compression_ratio: ratio,
                            page_size: 64,
                            storage,
                            capacity: 16,
                        },
                        &format!("chop_compressed_{name}_w{width}_r{ratio}_v1"),
                    )
                    .unwrap(),
                );
            }
        }
    }
    let specs = [cubin_spec("fp4_key_write_compile_v1", &entries).unwrap()];
    let built = build(
        &toolchain,
        &specs,
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("FP4 key-write cubin builds and audits");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), 12);
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(sass.status.success());
    let text = String::from_utf8(sass.stdout).unwrap();
    assert!(!text.contains("TF32"));
    for entry in &built[0].entries {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{} uses a stack", entry.symbol);
        let header = format!("Function : {}\n", entry.symbol);
        let body = text
            .split_once(&header)
            .expect("entry SASS")
            .1
            .split("Function : ")
            .next()
            .unwrap();
        assert!(
            body.contains("ACQBULK"),
            "{} lacks a PDL wait",
            entry.symbol
        );
        assert!(
            body.contains("PREEXIT"),
            "{} lacks a PDL release",
            entry.symbol
        );
    }
    fs::remove_dir_all(root).unwrap();
}
