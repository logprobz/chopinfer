#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

pub type TaskResult<T = ()> = Result<T, String>;

pub fn run(arguments: impl IntoIterator<Item = String>) -> TaskResult {
    let mut arguments = arguments.into_iter();
    match arguments.next().as_deref() {
        Some("check-portable") if arguments.next().is_none() => check_portable(),
        _ => Err("usage: xtask check-portable".to_owned()),
    }
}

const REVIEWED_UNSAFE_FILES: &[&str] = &[
    "crates/chopinfer-top/src/gpu/nvml.rs",
    "crates/kernel-core/tests/block32_rows_allocation.rs",
    "crates/kernel-harness/src/gpu_state.rs",
    "crates/runtime-bundle-sdk/src/native_note.rs",
    "crates/kernel-lab/tests/common/kv_offload.rs",
    "crates/runtime-host/src/page_cache.rs",
    "crates/runtime-host/src/allocator.rs",
    "crates/runtime-host/src/generation/heap.rs",
    "crates/runtime-host/src/direct/plan_cache/files.rs",
    "crates/runtime-host/src/signals.rs",
    "crates/runtime-host/src/host_rows.rs",
    "crates/runtime-host/src/weight_keeper/fds.rs",
    // SAFETY-REVIEWED: private anonymous mappings owned by mutex-protected shards.
    "crates/runtime-host/src/host_rows/resident/storage.rs",
    "crates/runtime-host/tests/host_mapped_rows_cost.rs",
    "crates/runtime-host/tests/memory_stability.rs",
    "crates/runtime-kv/tests/prefix_registry_bytes.rs",
    "crates/runtime-bundle/src/atomic_publish.rs",
    "crates/runtime-bundle-abi/src/loader.rs",
    "crates/runtime-bundle-abi/src/prefill.rs",
    "crates/runtime-bundle-abi/src/device_results/sink.rs",
    "crates/runtime-bundle-abi/src/trampoline.rs",
    "crates/runtime-bundle-abi/tests/native_loader.rs",
    "crates/runtime-bundle-abi/tests/abi_contract.rs",
    "crates/runtime-bundle-abi/tests/retirement_export.rs",
    "crates/runtime-cuda/src/ffi.rs",
    "crates/runtime-cuda/src/nvdec.rs",
    "crates/runtime-grammar/tests/retained_bytes.rs",
    "crates/runtime-media/tests/decode_bounds.rs",
];

fn check_portable() -> TaskResult {
    let started = Instant::now();
    audit_unsafe(Path::new("crates"))?;
    audit_unsafe(Path::new("models"))?;
    eprintln!("unsafe source roots: crates/, models/");
    eprintln!(
        "portable unsafe audit: {:.3}s",
        started.elapsed().as_secs_f64()
    );
    for (label, program, args) in [
        ("format", "cargo", vec!["fmt", "--all", "--check"]),
        (
            "portable policy",
            "python3",
            vec!["tests/portable-runtime-policy.py"],
        ),
        (
            "clippy",
            "cargo",
            vec![
                "clippy",
                "--workspace",
                "--all-targets",
                "--locked",
                "--offline",
                "--",
                "-D",
                "warnings",
            ],
        ),
        // cargo-nextest and the doctests, or cargo test without nextest.
        (
            "tests",
            "python3",
            vec!["-B", "ci/rust_tests.py", "--workspace"],
        ),
    ] {
        let started = Instant::now();
        let status = Command::new(program)
            .args(args)
            .status()
            .map_err(|error| format!("{label}: {error}"))?;
        eprintln!("portable {label}: {:.3}s", started.elapsed().as_secs_f64());
        if !status.success() {
            return Err(format!("{label} failed: {status}"));
        }
    }
    Ok(())
}

fn audit_unsafe(root: &Path) -> TaskResult {
    for entry in fs::read_dir(root).map_err(|error| format!("{}: {error}", root.display()))? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            audit_unsafe(&path)?;
            continue;
        }
        if path.extension().is_none_or(|extension| extension != "rs") {
            continue;
        }
        audit_unsafe_file(&path)?;
    }
    Ok(())
}

fn audit_unsafe_file(path: &Path) -> TaskResult {
    // The compiler enforces this file's crate-level unsafe prohibition.
    if path == Path::new("crates/xtask/src/lib.rs") {
        return Ok(());
    }
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    audit_unsafe_source(path, &source)
}

fn audit_unsafe_source(path: &Path, source: &str) -> TaskResult {
    let path_text = path.to_string_lossy();
    let reviewed = REVIEWED_UNSAFE_FILES.iter().any(|item| *item == path_text);
    let lines: Vec<_> = source.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let callback_type = path == Path::new("crates/runtime-bundle-abi/src/lib.rs")
            && (trimmed.starts_with("pub type ")
                || (trimmed.starts_with("unsafe extern \"C\" fn(")
                    && index > 0
                    && lines[index - 1].trim_end().ends_with('=')));
        if callback_type {
            continue;
        }
        let unsafe_line = [
            "unsafe {",
            "unsafe fn ",
            "unsafe extern ",
            "unsafe impl ",
            "#[allow(unsafe_code)]",
            "#![allow(unsafe_code)]",
        ]
        .iter()
        .any(|pattern| line.contains(pattern));
        let nearby_review = lines[index.saturating_sub(3)..index]
            .iter()
            .any(|line| line.trim_start().starts_with("// SAFETY-REVIEWED:"));
        if unsafe_line && !(reviewed && nearby_review) {
            return Err(format!(
                "unreviewed unsafe Rust at {}:{}",
                path.display(),
                index + 1
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_document_dispatch_has_reviewed_contracts() {
        audit_unsafe_source(
            Path::new("crates/runtime-bundle-abi/src/trampoline.rs"),
            include_str!("../../runtime-bundle-abi/src/trampoline.rs"),
        )
        .unwrap();
    }

    #[test]
    fn media_counting_allocator_has_reviewed_contracts() {
        audit_unsafe_source(
            Path::new("crates/runtime-media/tests/decode_bounds.rs"),
            include_str!("../../runtime-media/tests/decode_bounds.rs"),
        )
        .unwrap();
    }

    #[test]
    fn lifecycle_counting_allocators_have_reviewed_contracts() {
        for (path, source) in [
            (
                "crates/runtime-host/tests/memory_stability.rs",
                include_str!("../../runtime-host/tests/memory_stability.rs"),
            ),
            (
                "crates/runtime-kv/tests/prefix_registry_bytes.rs",
                include_str!("../../runtime-kv/tests/prefix_registry_bytes.rs"),
            ),
        ] {
            audit_unsafe_source(Path::new(path), source).unwrap();
            assert!(audit_unsafe_source(Path::new(path), "unsafe { call() }").is_err());
        }
    }

    #[test]
    fn serving_allocator_controls_have_reviewed_contracts() {
        for (path, source) in [
            (
                "crates/runtime-host/src/allocator.rs",
                include_str!("../../runtime-host/src/allocator.rs"),
            ),
            (
                "crates/runtime-host/src/generation/heap.rs",
                include_str!("../../runtime-host/src/generation/heap.rs"),
            ),
        ] {
            audit_unsafe_source(Path::new(path), source).unwrap();
            assert!(audit_unsafe_source(Path::new(path), "unsafe { call() }").is_err());
        }
    }

    #[test]
    fn optional_abi_callbacks_have_reviewed_contracts() {
        for (path, source) in [
            (
                "crates/runtime-bundle-abi/src/prefill.rs",
                include_str!("../../runtime-bundle-abi/src/prefill.rs"),
            ),
            (
                "crates/runtime-bundle-abi/src/device_results/sink.rs",
                include_str!("../../runtime-bundle-abi/src/device_results/sink.rs"),
            ),
        ] {
            audit_unsafe_source(Path::new(path), source).unwrap();
        }
    }

    #[test]
    fn reviewed_files_still_require_a_nearby_review() {
        let path = Path::new("crates/runtime-bundle-abi/src/loader.rs");
        assert!(audit_unsafe_source(path, "unsafe { call() }").is_err());
        assert!(
            audit_unsafe_source(path, "// SAFETY-REVIEWED: fixture.\nunsafe { call() }").is_ok()
        );
        assert!(
            audit_unsafe_source(
                path,
                "// SAFETY-REVIEWED: fixture.\n\n\n\nunsafe { call() }"
            )
            .is_err()
        );
    }

    #[test]
    fn review_markers_do_not_admit_unreviewed_files() {
        assert!(
            audit_unsafe_source(
                Path::new("crates/unreviewed/src/lib.rs"),
                "// SAFETY-REVIEWED: fixture.\nunsafe { call() }"
            )
            .is_err()
        );
    }
}
