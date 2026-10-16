use kernel_build::{BuildOptions, Toolchain};
use model_qwen38_contract::sm120::comm::decode as qwen38_decode;
use std::collections::BTreeMap;
use std::process::Command;

// Inspect the last writer of each F32 value rounded to BF16. Both LL
// and one-shot have ten residual FADDs and ten norm FMULs. A separate
// v3 inject loop once changed the residual producer to FFMA: the BF16
// residual hid the difference, but norm retained the changed F32 value.
fn finish_rounding(sass: &str) -> BTreeMap<String, usize> {
    let mut writers = BTreeMap::<String, String>::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let register = |operand: &str| -> Option<String> {
        let name = operand.trim().split('.').next().unwrap();
        name.strip_prefix('R')
            .filter(|suffix| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()))
            .map(|_| name.to_owned())
    };
    for line in sass.lines() {
        let Some((_, rest)) = line.split_once("*/") else {
            continue;
        };
        let Some((instruction, _)) = rest.split_once(';') else {
            continue;
        };
        let mut instruction = instruction.trim();
        if instruction.starts_with('@') {
            instruction = instruction.split_once(' ').unwrap().1.trim();
        }
        let Some((opcode, args)) = instruction.split_once(' ') else {
            continue;
        };
        let operands: Vec<_> = args.split(',').collect();
        if opcode.starts_with("F2FP.BF16.F32") {
            for source in operands.iter().skip(1).filter_map(|s| register(s)) {
                let producer = writers
                    .get(&source)
                    .expect("BF16 conversion's F32 producer");
                *counts.entry(producer.clone()).or_default() += 1;
            }
        }
        if let Some(dest) = operands.first().and_then(|s| register(s)) {
            writers.insert(dest, opcode.to_owned());
        }
    }
    counts
}

#[test]
fn qwen_decode_comm_compiles_without_local_memory() {
    assert!(
        qwen38_decode::cubin_specs(Some(
            kernels_sm120::Model::new(
                kernels_sm120::profile_names!("other-model", "other", 4, 2),
                model_qwen38_contract::sm120::config::CONFIG
            )
            .at(4)
        ))
        .is_empty()
    );
    let root = std::env::temp_dir().join(format!("qwen38-comm-decode-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let reports = kernel_build::build(
        &Toolchain::detect().unwrap(),
        &qwen38_decode::cubin_specs(None),
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(reports[0].entries[0].local_bytes, 0);
    let toolchain = Toolchain::detect().unwrap();
    for ranks in [2, 4, 8] {
        for symbol in [
            qwen38_decode::symbol_v2(ranks),
            qwen38_decode::symbol_v3(ranks),
        ] {
            let symbol = symbol.unwrap();
            let dump = Command::new(toolchain.home.join("bin/cuobjdump"))
                .args(["-sass", "-fun", symbol])
                .arg(&reports[0].path)
                .output()
                .expect("decode finish SASS");
            assert!(dump.status.success(), "{symbol}: cuobjdump failed");
            let sass = String::from_utf8(dump.stdout).unwrap();
            assert_eq!(
                finish_rounding(&sass),
                BTreeMap::from([("FADD".to_owned(), 20), ("FMUL".to_owned(), 20)]),
                "{symbol}: preserve separate inject multiply/add and norm multiplications"
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn qwen_wide_decode_comm_compiles_without_local_memory() {
    assert!(
        qwen38_decode::wide_cubin_specs(Some(
            kernels_sm120::Model::new(
                kernels_sm120::profile_names!("other-model", "other", 4, 2),
                model_qwen38_contract::sm120::config::CONFIG
            )
            .at(4)
        ))
        .is_empty()
    );
    let root = std::env::temp_dir().join(format!("qwen38-comm-wide-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let reports = kernel_build::build(
        &Toolchain::detect().unwrap(),
        &qwen38_decode::wide_cubin_specs(None),
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(reports[0].entries.len(), qwen38_decode::WIDE_SYMBOLS.len());
    for entry in &reports[0].entries {
        assert!(qwen38_decode::WIDE_SYMBOLS.contains(&entry.symbol.as_str()));
        assert_eq!(
            (entry.local_bytes, entry.stack_bytes),
            (0, 0),
            "{}",
            entry.symbol
        );
        // Its plans count on this many resident CTAs per SM.
        let resident = kernels_sm120::comm::sm120_ctas_per_sm(
            entry.max_threads,
            entry.registers,
            entry.shared_static_bytes,
        );
        assert!(
            entry.min_blocks == kernels_sm120::comm::WIDE_CTAS_PER_SM
                && resident >= kernels_sm120::comm::WIDE_CTAS_PER_SM,
            "{}: {resident} CTAs per SM ({} registers, {} shared bytes)",
            entry.symbol,
            entry.registers,
            entry.shared_static_bytes
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}
