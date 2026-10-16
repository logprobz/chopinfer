use kernel_build::{
    BuildOptions, CubinSpec, EntryParam, EntrySpec, Toolchain, audit_cubin, build, generate,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use kernel_build::report::{LockStatus, SourceKind};
use kernel_build::{
    BUILD_REPORT_FILE, BuildContext, BuildReport, EntryDeclaration, KernelSelection, LockDir,
    SelectedCubin, build_in_families, build_with_context,
};
use std::collections::BTreeMap;

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Case(PathBuf);
impl Case {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "kernel-build-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn root(&self) -> PathBuf {
        self.0.join("src")
    }
    fn out(&self) -> PathBuf {
        self.0.join("out")
    }
    fn lock(&self) -> PathBuf {
        self.0.join("kernels.lock")
    }
}
impl Drop for Case {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn toolchain() -> Option<Toolchain> {
    if std::env::consts::OS != "linux" || !Path::new("/usr/local/cuda/bin/nvcc").exists() {
        eprintln!("SKIP: kernel-build compile tests require Linux and nvcc");
        return None;
    }
    Some(Toolchain::detect().unwrap())
}
fn tiny(case: &Case) -> CubinSpec {
    fs::create_dir_all(case.root()).unwrap();
    fs::write(
        case.root().join("ops.h"),
        "__device__ inline int value(int x) { return x + 1; }\n",
    )
    .unwrap();
    CubinSpec {
        name: "tiny".into(),
        sources_root: case.root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"ops.h\"".into()],
        defines: vec![],
        entries: vec![EntrySpec {
            symbol: "tiny_v1".into(),
            params: vec![EntryParam::DevicePointer {
                name: "out".into(),
                cxx_type: "int".into(),
            }],
            body: "out[threadIdx.x] = value(threadIdx.x);".into(),
            max_threads: 32,
            min_blocks: 1,
            allowed_local_bytes: 0,
        }],
    }
}
fn update() -> BuildOptions {
    BuildOptions {
        lock_update: true,
        emit_cargo_directives: false,
        profile_line_info: false,
    }
}
fn verify() -> BuildOptions {
    BuildOptions {
        lock_update: false,
        emit_cargo_directives: false,
        profile_line_info: false,
    }
}
fn nvcc(toolchain: &Toolchain, source: &Path, out: &Path, flags: &[&str]) {
    let output = Command::new(toolchain.home.join("bin/nvcc"))
        .args(flags)
        .arg(source)
        .arg("-o")
        .arg(out)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn codegen_golden_all_abi_kinds() {
    let case = Case::new();
    let mut spec = tiny(&case);
    spec.headers.clear();
    spec.entries[0].body = "(void)ptr; (void)n; (void)u; (void)i; (void)map; (void)arr;".into();
    spec.entries[0].params = vec![
        EntryParam::DevicePointer {
            name: "ptr".into(),
            cxx_type: "float".into(),
        },
        EntryParam::U64 { name: "n".into() },
        EntryParam::U32 { name: "u".into() },
        EntryParam::I32 { name: "i".into() },
        EntryParam::TensorMap { name: "map".into() },
        EntryParam::PointerArray {
            name: "arr".into(),
            slots: 2,
            cxx_type: "float".into(),
        },
    ];
    let generated = generate(&spec).unwrap();
    assert_eq!(
        generated.header,
        "#pragma once\nstruct tiny_v1_arr_params {\n    float* slot_0;\n    float* slot_1;\n};\n"
    );
    assert_eq!(
        generated.translation_unit,
        "#include <stdint.h>\n#include <cuda.h>\n#include \"tiny_params.h\"\nextern \"C\" __global__ void __launch_bounds__(32, 1) tiny_v1(float* ptr, uint64_t n, uint32_t u, int32_t i, const __grid_constant__ CUtensorMap map, tiny_v1_arr_params arr) {\n(void)ptr; (void)n; (void)u; (void)i; (void)map; (void)arr;\n}\n"
    );
    if let Some(toolchain) = toolchain() {
        build(&toolchain, &[spec], &case.out(), &case.lock(), update()).unwrap();
    }
}

#[test]
fn tiny_build_audits_and_lock_workflow() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let case = Case::new();
    let spec = tiny(&case);
    let first = build(
        &toolchain,
        std::slice::from_ref(&spec),
        &case.out(),
        &case.lock(),
        update(),
    )
    .unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].entries[0].symbol, "tiny_v1");
    assert!(first[0].entries[0].registers > 0);
    assert_eq!(first[0].entries[0].local_bytes, 0);
    assert_eq!(first[0].entries[0].shared_static_bytes, 0);
    assert!(!first[0].line_info);
    kernel_build::write_embedded(&case.out(), &first, |_| 0).unwrap();
    let default = fs::read_to_string(case.out().join("embedded.rs")).unwrap();
    assert!(default.contains("\"/tiny.cubin\""));
    kernel_build::write_embedded_with_prefix(&case.out(), Path::new("precision"), &first, |_| 1)
        .unwrap();
    let nested = fs::read_to_string(case.out().join("embedded.rs")).unwrap();
    assert!(nested.contains("\"/precision/tiny.cubin\""));
    for invalid in ["../precision", "/precision", "precision/../escape"] {
        assert!(
            kernel_build::write_embedded_with_prefix(
                &case.out(),
                Path::new(invalid),
                &first,
                |_| 0
            )
            .is_err()
        );
    }
    let dump = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-elf")
        .arg(&first[0].path)
        .output()
        .unwrap();
    let dump = String::from_utf8(dump.stdout).unwrap();
    assert!(
        dump.lines().any(|line| line.starts_with("64-bit ELF:")
            && line.contains("sm=120,")
            && line.contains("flags=0x6007802")),
        "{}",
        dump.lines()
            .find(|line| line.contains("ELF:"))
            .unwrap_or("<empty>")
    );
    assert!(!dump.contains(".nv_debug_ptx_txt"));
    build(
        &toolchain,
        std::slice::from_ref(&spec),
        &case.out(),
        &case.lock(),
        verify(),
    )
    .unwrap();
    fs::write(
        case.root().join("ops.h"),
        "__device__ inline int value(int x) { return x + 2; }\n",
    )
    .unwrap();
    let changed = build(
        &toolchain,
        std::slice::from_ref(&spec),
        &case.out(),
        &case.lock(),
        verify(),
    )
    .unwrap_err();
    assert!(changed.to_string().contains("source_closure_sha256"));
    build(
        &toolchain,
        std::slice::from_ref(&spec),
        &case.out(),
        &case.lock(),
        update(),
    )
    .unwrap();
    let mut lock: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(case.lock()).unwrap()).unwrap();
    lock[0]["cubin_sha256"] = serde_json::Value::String("0".repeat(64));
    fs::write(
        case.lock(),
        format!("{}\n", serde_json::to_string_pretty(&lock).unwrap()),
    )
    .unwrap();
    assert!(
        build(
            &toolchain,
            std::slice::from_ref(&spec),
            &case.out(),
            &case.lock(),
            verify()
        )
        .unwrap_err()
        .to_string()
        .contains("kernels.lock mismatch")
    );
    fs::write(case.lock(), "[]\n").unwrap();
    assert!(
        build(
            &toolchain,
            std::slice::from_ref(&spec),
            &case.out(),
            &case.lock(),
            verify()
        )
        .unwrap_err()
        .to_string()
        .contains("record count")
    );
    build(&toolchain, &[spec], &case.out(), &case.lock(), update()).unwrap();
}

#[test]
fn rejects_invalid_entry_specs() {
    let case = Case::new();
    let mut spec = tiny(&case);
    spec.entries[0].symbol = "tiny".into();
    assert!(
        generate(&spec)
            .unwrap_err()
            .to_string()
            .contains("versioned symbol")
    );
    spec.entries[0].symbol = "tiny_v1".into();
    spec.entries[0].body.clear();
    assert!(
        generate(&spec)
            .unwrap_err()
            .to_string()
            .contains("empty body")
    );
    spec.entries[0].body = "return;".into();
    spec.entries[0].max_threads = 33;
    assert!(
        generate(&spec)
            .unwrap_err()
            .to_string()
            .contains("launch bounds")
    );
    spec.entries[0].max_threads = 32;
    spec.entries[0].min_blocks = 0;
    assert!(
        generate(&spec)
            .unwrap_err()
            .to_string()
            .contains("launch bounds")
    );
    spec.entries[0].min_blocks = 1;
    spec.entries[0].params = vec![EntryParam::PointerArray {
        name: "array".into(),
        slots: 9,
        cxx_type: "int".into(),
    }];
    assert!(generate(&spec).unwrap_err().to_string().contains("slots"));
    spec.entries[0].params.clear();
    let mut duplicate = spec.clone();
    duplicate.name = "other".into();
    let toolchain = Toolchain {
        home: PathBuf::new(),
        version: String::new(),
        nvcc_sha256: String::new(),
        skipped: false,
    };
    assert!(
        build(
            &toolchain,
            &[spec, duplicate],
            &case.out(),
            &case.lock(),
            update()
        )
        .unwrap_err()
        .to_string()
        .contains("duplicate versioned symbol")
    );
}

#[test]
fn architecture_encoding_and_audit_failures() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let case = Case::new();
    let mut spec = tiny(&case);
    let source = case.0.join("direct.cu");
    fs::write(
        &source,
        "extern \"C\" __global__ void tiny_v1(int* out) { out[threadIdx.x] = 1; }\n",
    )
    .unwrap();
    let a = case.0.join("a.cubin");
    nvcc(
        &toolchain,
        &source,
        &a,
        &["-arch=sm_120a", "-cubin", "-std=c++20"],
    );
    let dump = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-elf")
        .arg(&a)
        .output()
        .unwrap();
    let dump = String::from_utf8(dump.stdout).unwrap();
    assert!(
        dump.lines()
            .any(|line| line.starts_with("64-bit ELF:") && line.contains("sm=120a"))
    );
    assert!(
        audit_cubin(&toolchain, &spec, &a)
            .unwrap_err()
            .to_string()
            .contains("architecture")
    );
    let f = case.0.join("f.cubin");
    nvcc(
        &toolchain,
        &source,
        &f,
        &["-arch=sm_120f", "-cubin", "-std=c++20"],
    );
    let dump = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-elf")
        .arg(&f)
        .output()
        .unwrap();
    let dump = String::from_utf8(dump.stdout).unwrap();
    assert!(
        dump.lines()
            .any(|line| line.starts_with("64-bit ELF:") && line.contains("sm=120,"))
    );
    let unresolved_source = case.0.join("unresolved.cu");
    fs::write(&unresolved_source, "extern \"C\" __device__ int unresolved();\nextern \"C\" __global__ void tiny_v1(int* out) { out[threadIdx.x] = unresolved(); }\n").unwrap();
    let unresolved = case.0.join("unresolved.cubin");
    nvcc(
        &toolchain,
        &unresolved_source,
        &unresolved,
        &["-arch=sm_120f", "-cubin", "-rdc=true", "-std=c++20"],
    );
    assert!(
        audit_cubin(&toolchain, &spec, &unresolved)
            .unwrap_err()
            .to_string()
            .contains("unresolved device function")
    );
    spec.entries.push(EntrySpec {
        symbol: "missing_v1".into(),
        ..spec.entries[0].clone()
    });
    assert!(
        audit_cubin(&toolchain, &spec, &f)
            .unwrap_err()
            .to_string()
            .contains("entry symbol")
    );
    let fatbin = case.0.join("with_ptx.fatbin");
    nvcc(
        &toolchain,
        &source,
        &fatbin,
        &[
            "-fatbin",
            "-gencode=arch=compute_120f,code=[sm_120f,compute_120f]",
        ],
    );
    assert!(
        audit_cubin(&toolchain, &spec, &fatbin)
            .unwrap_err()
            .to_string()
            .contains("PTX")
    );
}

#[test]
fn extra_entry_and_local_memory_fail() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let case = Case::new();
    let spec = tiny(&case);
    fs::write(case.root().join("ops.h"), "__device__ inline int value(int x) { return x; }\nextern \"C\" __global__ void extra_v1(int* out) { out[threadIdx.x] = 1; }\n").unwrap();
    assert!(
        build(
            &toolchain,
            std::slice::from_ref(&spec),
            &case.out(),
            &case.lock(),
            update()
        )
        .unwrap_err()
        .to_string()
        .contains("entry symbol")
    );
    fs::write(
        case.root().join("ops.h"),
        "__device__ inline int value(int x) { return x; }\n",
    )
    .unwrap();
    let mut spilling = spec;
    spilling.entries[0].body = "volatile int local[256]; local[threadIdx.x] = value(threadIdx.x); out[threadIdx.x] = local[threadIdx.x];".into();
    let error = build(&toolchain, &[spilling], &case.out(), &case.lock(), update()).unwrap_err();
    assert!(error.to_string().contains("STACK"), "{error}");
}

#[test]
fn reproducible_across_directories() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let a = Case::new();
    let b = Case::new();
    let left = build(&toolchain, &[tiny(&a)], &a.out(), &a.lock(), update()).unwrap();
    let right = build(&toolchain, &[tiny(&b)], &b.out(), &b.lock(), update()).unwrap();
    assert_eq!(left[0].sha256, right[0].sha256);
    assert_eq!(
        fs::read_to_string(a.lock()).unwrap(),
        fs::read_to_string(b.lock()).unwrap()
    );
}

#[test]
fn profiling_line_info_skips_release_lock() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let case = Case::new();
    let spec = tiny(&case);
    fs::write(case.lock(), "release lock sentinel\n").unwrap();
    let built = build(
        &toolchain,
        &[spec],
        &case.out(),
        &case.lock(),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: true,
        },
    )
    .unwrap();
    assert_eq!(built.len(), 1);
    assert!(built[0].line_info);
    assert!(built[0].path.ends_with("tiny.profile.cubin"));
    assert_eq!(
        fs::read_to_string(case.lock()).unwrap(),
        "release lock sentinel\n"
    );
    let identity: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&built[0].identity_path).unwrap()).unwrap();
    assert_eq!(identity["line_info"], true);
    assert!(
        identity["flags"]
            .as_array()
            .unwrap()
            .contains(&serde_json::Value::String("-lineinfo".into()))
    );
}

fn context(case: &Case, owner: &str) -> BuildContext {
    BuildContext::new(format!("{owner}@1"), owner, "test.selection.v1")
        .source_root("test.sources", case.root())
}

fn declaration(symbol: &str) -> EntryDeclaration {
    EntryDeclaration {
        symbol: symbol.into(),
        operation: "integer increment".into(),
        contract: "test.add-one.v1".into(),
        implementation: "test.native.v1".into(),
        requirements: BTreeMap::from([("value".into(), "i32 + 1".into())]),
        correctness_case: Some("test.add-one.v1".into()),
    }
}

#[test]
fn selection_declarations_cover_exact_entries() {
    let case = Case::new();
    let spec = tiny(&case);
    assert!(SelectedCubin::new(spec.clone(), vec![declaration("tiny_v1")]).is_err());
    assert!(SelectedCubin::existing(spec.clone(), vec![]).is_err());
    assert!(SelectedCubin::existing(spec.clone(), vec![declaration("other_v1")]).is_err());
    assert!(
        SelectedCubin::existing(
            spec.clone(),
            vec![declaration("tiny_v1"), declaration("tiny_v1")]
        )
        .is_err()
    );
    assert!(KernelSelection::new(context(&case, "test"), vec![]).is_err());
    let mut renamed = spec.clone();
    renamed.name = "other_v1".into();
    let error = match KernelSelection::new(
        context(&case, "test"),
        vec![
            SelectedCubin::existing(spec.clone(), vec![declaration("tiny_v1")]).unwrap(),
            SelectedCubin::new(renamed, vec![declaration("tiny_v1")]).unwrap(),
        ],
    ) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("duplicate symbols accepted"),
    };
    assert!(error.contains("test.selection.v1"));
    assert!(error.contains("tiny_v1"));
    let selection = KernelSelection::new(
        context(&case, "test"),
        vec![SelectedCubin::existing(spec, vec![declaration("tiny_v1")]).unwrap()],
    )
    .unwrap();
    assert_eq!(selection.specs()[0].name, "tiny");
    let identity = selection.contract_identity().unwrap();
    let mut altered = declaration("tiny_v1");
    altered
        .requirements
        .insert("scales".into(), "per-expert".into());
    let other = KernelSelection::new(
        context(&case, "other-owner"),
        vec![SelectedCubin::existing(selection.specs()[0].clone(), vec![altered]).unwrap()],
    )
    .unwrap();
    assert_ne!(identity, other.contract_identity().unwrap());
    assert!(selection.audited_contract_identity(&[]).is_err());
}

#[test]
fn membership_survives_cache_hits_and_relocation() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let a = Case::new();
    let b = Case::new();
    let mut reports = Vec::new();
    let mut identities = Vec::new();
    for case in [&a, &b] {
        let spec = tiny(case);
        fs::write(case.root().join("ops.h"), "#include \"nested.h\"\n").unwrap();
        fs::write(case.root().join("nested.h"), "#include \"leaf.h\"\n").unwrap();
        fs::write(
            case.root().join("leaf.h"),
            "__device__ inline int value(int x) { return x + 1; }\n",
        )
        .unwrap();
        let selection = KernelSelection::new(
            context(case, "test"),
            vec![SelectedCubin::existing(spec, vec![declaration("tiny_v1")]).unwrap()],
        )
        .unwrap();
        let built = selection
            .build(&toolchain, &case.out(), &case.lock(), update())
            .unwrap();
        identities.push((
            selection.audited_contract_identity(&built).unwrap(),
            selection.audited_dependency_identity(&built).unwrap(),
        ));
        let mut incomplete = built.clone();
        incomplete[0].entries.clear();
        assert!(selection.audited_contract_identity(&incomplete).is_err());
        let mut profiling = built.clone();
        profiling[0].line_info = true;
        assert!(selection.audited_contract_identity(&profiling).is_err());
        let report_path = case.out().join(BUILD_REPORT_FILE);
        let initial = BuildReport::read(&report_path).unwrap();
        assert_eq!(initial.lock_status, LockStatus::Updated);
        assert!(initial.verify_current().is_err());
        let modified = fs::metadata(&built[0].path).unwrap().modified().unwrap();
        fs::remove_file(&report_path).unwrap();
        selection
            .build(&toolchain, &case.out(), &case.lock(), verify())
            .unwrap();
        assert_eq!(
            modified,
            fs::metadata(&built[0].path).unwrap().modified().unwrap()
        );
        let report = BuildReport::read(&report_path).unwrap();
        report.verify_current().unwrap();
        assert!(
            report.source_membership_complete
                && report.source_ownership_complete
                && report.entry_declarations_complete
        );
        assert!(
            report
                .files
                .values()
                .any(|f| f.identity.path == "leaf.h" && f.identity.kind == SourceKind::Project)
        );
        assert_eq!(report.cubins[0].declarations, vec![declaration("tiny_v1")]);
        reports.push(report);
    }
    assert_eq!(identities[0], identities[1]);
    assert_eq!(reports[0].cubins[0].identity, reports[1].cubins[0].identity);
    assert_eq!(
        reports[0].cubins[0].recipe_sha256,
        reports[1].cubins[0].recipe_sha256
    );
    assert_eq!(
        reports[0].files.keys().collect::<Vec<_>>(),
        reports[1].files.keys().collect::<Vec<_>>()
    );
    let spec = tiny(&a);
    // Restore the nested includes after the fixture helper writes ops.h.
    fs::write(a.root().join("ops.h"), "#include \"nested.h\"\n").unwrap();
    fs::write(
        a.root().join("leaf.h"),
        "__device__ inline int value(int x) { return x + 2; }\n",
    )
    .unwrap();
    assert!(
        reports[0]
            .verify_current()
            .unwrap_err()
            .to_string()
            .contains("stale source")
    );
    reports[1].verify_current().unwrap();
    assert!(
        build_with_context(
            &toolchain,
            &[spec],
            &a.out(),
            &a.lock(),
            verify(),
            &context(&a, "test")
        )
        .is_err()
    );
    assert!(!a.out().join(BUILD_REPORT_FILE).exists());
}

#[test]
fn shared_sources_identify_all_consumers_without_changing_others() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let shared = Case::new();
    let unrelated = Case::new();
    let mut left = tiny(&shared);
    fs::create_dir_all(shared.root().join("shared")).unwrap();
    let header = shared.root().join("shared/core.h");
    fs::write(
        &header,
        "__device__ inline int value(int x) { return x + 1; }\n",
    )
    .unwrap();
    left.name = "left_v1".into();
    left.headers = vec!["#include \"shared/core.h\"".into()];
    let mut right = left.clone();
    right.name = "right_v1".into();
    right.sources_root = shared.root().join("shared");
    right.headers = vec!["#include \"core.h\"".into()];
    let third = tiny(&unrelated);
    let mut reports = Vec::new();
    for (name, spec, case) in [
        ("a", &left, &shared),
        ("b", &right, &shared),
        ("c", &third, &unrelated),
    ] {
        let out = case.out().join(name);
        let lock = case.0.join(format!("{name}.lock"));
        let selected = KernelSelection::new(
            context(case, name),
            vec![SelectedCubin::existing(spec.clone(), vec![declaration("tiny_v1")]).unwrap()],
        )
        .unwrap();
        selected.build(&toolchain, &out, &lock, update()).unwrap();
        selected.build(&toolchain, &out, &lock, verify()).unwrap();
        reports.push(BuildReport::read(&out.join(BUILD_REPORT_FILE)).unwrap());
    }
    let shared_ids = reports[..2]
        .iter()
        .map(|report| {
            report.verify_current().unwrap();
            report
                .files
                .iter()
                .find(|(_, f)| f.identity.path == "shared/core.h")
                .unwrap()
                .0
        })
        .collect::<Vec<_>>();
    assert_eq!(shared_ids[0], shared_ids[1]);
    assert_ne!(
        reports[0].cubins[0]
            .membership
            .iter()
            .find(|m| &m.file == shared_ids[0])
            .unwrap()
            .closure_key,
        reports[1].cubins[0]
            .membership
            .iter()
            .find(|m| &m.file == shared_ids[1])
            .unwrap()
            .closure_key
    );
    // A second owner cannot reuse another owner's output directory.
    let before = fs::read(&reports[0].cubins[0].path).unwrap();
    assert!(
        build_with_context(
            &toolchain,
            &[right],
            &shared.out().join("a"),
            &shared.0.join("a.lock"),
            verify(),
            &context(&shared, "b")
        )
        .is_err()
    );
    assert_eq!(before, fs::read(&reports[0].cubins[0].path).unwrap());
    reports[0].verify_current().unwrap();
    fs::write(
        &header,
        "__device__ inline int value(int x) { return x + 2; }\n",
    )
    .unwrap();
    assert!(reports[0].verify_current().is_err());
    assert!(reports[1].verify_current().is_err());
    reports[2].verify_current().unwrap();
}

#[test]
fn report_distinguishes_legacy_profiling_and_skipped_builds() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let case = Case::new();
    let spec = tiny(&case);
    build(
        &toolchain,
        std::slice::from_ref(&spec),
        &case.out(),
        &case.lock(),
        update(),
    )
    .unwrap();
    build(
        &toolchain,
        std::slice::from_ref(&spec),
        &case.out(),
        &case.lock(),
        verify(),
    )
    .unwrap();
    let release = BuildReport::read(&case.out().join(BUILD_REPORT_FILE)).unwrap();
    release.verify_current().unwrap();
    assert!(!release.source_ownership_complete && !release.entry_declarations_complete);
    build(
        &toolchain,
        std::slice::from_ref(&spec),
        &case.out(),
        &case.lock(),
        BuildOptions {
            profile_line_info: true,
            ..verify()
        },
    )
    .unwrap();
    let profiled = BuildReport::read(&case.out().join("kernel-build-report.profile.json")).unwrap();
    assert_eq!(profiled.lock_status, LockStatus::Profiling);
    assert!(profiled.verify_current().is_err());
    build(
        &Toolchain {
            skipped: true,
            ..toolchain
        },
        &[spec],
        &case.out(),
        &case.lock(),
        verify(),
    )
    .unwrap();
    let skipped = BuildReport::read(&case.out().join(BUILD_REPORT_FILE)).unwrap();
    assert!(!skipped.source_membership_complete);
    assert_eq!(skipped.lock_status, LockStatus::Skipped);
    assert!(skipped.verify_current().is_err());
}

fn renamed(case: &Case, name: &str) -> CubinSpec {
    let mut spec = tiny(case);
    spec.name = name.into();
    spec.entries[0].symbol = format!("{name}_v1");
    spec
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}

#[test]
fn lock_directory_rejects_incomplete_family_declarations() {
    // Family assignment is checked before any compiler runs.
    let toolchain = Toolchain {
        home: PathBuf::from("/nonexistent-cuda"),
        version: "release 13.2,".into(),
        nvcc_sha256: "0".repeat(64),
        skipped: false,
    };
    let case = Case::new();
    let specs = [renamed(&case, "alpha"), renamed(&case, "beta")];
    let dir = case.0.join("kernels.lock.d");
    let refused = |lock: LockDir| {
        build_in_families(&toolchain, &specs, &case.out(), &lock, verify(), None)
            .unwrap_err()
            .to_string()
    };
    let error = refused(LockDir::new(&dir).family("alpha", ["alpha"]));
    assert!(error.contains("cubin beta has no lock family"), "{error}");
    let error = refused(LockDir::new(&dir).family("alpha", ["alpha", "beta", "gamma"]));
    assert!(error.contains("names cubin gamma"), "{error}");
    let error = refused(
        LockDir::new(&dir)
            .family("x", ["alpha", "beta"])
            .family("y", ["beta"]),
    );
    assert!(error.contains("more than one lock family"), "{error}");
    let error = refused(LockDir::new(&dir).family("../escape", ["alpha", "beta"]));
    assert!(error.contains("invalid lock family name"), "{error}");
    let error = refused(
        LockDir::new(&dir)
            .family("x", ["alpha", "beta"])
            .family("x", ["gamma"]),
    );
    assert!(error.contains("declared twice"), "{error}");
    assert!(!dir.exists());
}

#[test]
fn lock_directory_gates_on_cubin_and_reports_provenance() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let case = Case::new();
    let spec = tiny(&case);
    let lock = LockDir::new(case.0.join("kernels.lock.d")).family("tiny_family", ["tiny"]);
    let lock_file = lock.lock_file("tiny_family");
    let provenance_file = lock.provenance_file("tiny_family");
    let specs = std::slice::from_ref(&spec);
    let report = || BuildReport::read(&case.out().join(BUILD_REPORT_FILE)).unwrap();
    let run = |options| build_in_families(&toolchain, specs, &case.out(), &lock, options, None);

    let built = run(update()).unwrap();
    assert!(lock_file.ends_with("kernels.lock.d/tiny_family.lock"));
    let gated: serde_json::Value = serde_json::from_str(&read(&lock_file)).unwrap();
    let provenance: serde_json::Value = serde_json::from_str(&read(&provenance_file)).unwrap();
    assert!(gated[0].get("source_closure_sha256").is_none());
    assert_eq!(gated[0]["cubin_sha256"], built[0].sha256);
    assert_eq!(
        provenance,
        serde_json::json!([{"name": "tiny", "source_closure_sha256": built[0].source_closure_sha256}])
    );
    assert_eq!(report().lock_status, LockStatus::Updated);
    run(verify()).unwrap();
    let verified = report();
    assert_eq!(verified.lock_status, LockStatus::Verified);
    assert_eq!(verified.lock_sha256, None);
    assert_eq!(verified.lock_families[0].cubins, ["tiny"]);
    assert!(verified.lock_families[0].provenance_changed.is_empty());
    verified.verify_current().unwrap();

    // A header-only edit with an identical cubin passes and reports provenance.
    let (locked, recorded) = (read(&lock_file), read(&provenance_file));
    fs::write(
        case.root().join("ops.h"),
        "// A comment changes the source closure, not the cubin.\n\
         __device__ inline int value(int x) { return x + 1; }\n",
    )
    .unwrap();
    let edited = run(verify()).unwrap();
    assert_ne!(
        edited[0].source_closure_sha256,
        built[0].source_closure_sha256
    );
    assert_eq!(edited[0].sha256, built[0].sha256);
    let drifted = report();
    assert_eq!(drifted.lock_status, LockStatus::Verified);
    assert_eq!(drifted.lock_families[0].provenance_changed, ["tiny"]);
    drifted.verify_current().unwrap();
    assert_eq!(
        (read(&lock_file), read(&provenance_file)),
        (locked.clone(), recorded.clone())
    );

    // An update refreshes provenance only.
    run(update()).unwrap();
    assert_eq!(read(&lock_file), locked);
    assert_ne!(read(&provenance_file), recorded);
    run(verify()).unwrap();
    assert!(report().lock_families[0].provenance_changed.is_empty());

    // A cubin change fails naming the family, cubin and field.
    fs::write(
        case.root().join("ops.h"),
        "__device__ inline int value(int x) { return x * 3; }\n",
    )
    .unwrap();
    let error = run(verify()).unwrap_err().to_string();
    assert!(error.contains("family tiny_family"), "{error}");
    assert!(error.contains("tiny.cubin_sha256"), "{error}");
    assert!(error.contains("KERNELS_LOCK_UPDATE=1"), "{error}");
    assert!(!case.out().join(BUILD_REPORT_FILE).exists());
    assert_eq!(read(&lock_file), locked);
    fs::write(
        case.root().join("ops.h"),
        "__device__ inline int value(int x) { return x + 1; }\n",
    )
    .unwrap();
    run(verify()).unwrap();

    // An entry ABI difference names the entry symbol and field.
    let mut abi: serde_json::Value = serde_json::from_str(&locked).unwrap();
    abi[0]["entries"][0]["max_threads"] = 64.into();
    fs::write(&lock_file, serde_json::to_string_pretty(&abi).unwrap()).unwrap();
    let error = run(verify()).unwrap_err().to_string();
    assert!(
        error.contains("tiny.entries[tiny_v1].max_threads"),
        "{error}"
    );

    // A legacy record with its source closure is not a directory-form record.
    let mut legacy = abi.clone();
    legacy[0]["entries"][0]["max_threads"] = 32.into();
    legacy[0]["source_closure_sha256"] = built[0].source_closure_sha256.clone().into();
    fs::write(&lock_file, serde_json::to_string_pretty(&legacy).unwrap()).unwrap();
    let error = run(verify()).unwrap_err().to_string();
    assert!(error.contains("invalid lock"), "{error}");

    // A missing family file fails with the update hint; a missing provenance
    // file is reported only.
    fs::remove_file(&lock_file).unwrap();
    let error = run(verify()).unwrap_err().to_string();
    assert!(error.contains("tiny_family.lock"), "{error}");
    assert!(
        error.contains("rerun with KERNELS_LOCK_UPDATE=1"),
        "{error}"
    );
    run(update()).unwrap();
    assert_eq!(read(&lock_file), locked);
    fs::remove_file(&provenance_file).unwrap();
    run(verify()).unwrap();
    assert_eq!(report().lock_families[0].provenance_changed, ["tiny"]);
}

#[test]
fn lock_directory_families_never_touch_each_other() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let case = Case::new();
    let dir = case.0.join("kernels.lock.d");
    fs::create_dir_all(&dir).unwrap();
    // Another branch's family that no build here may read or rewrite.
    fs::write(dir.join("other.lock"), "not a lock\n").unwrap();
    fs::write(dir.join("other.provenance.json"), "not provenance\n").unwrap();
    let alpha = LockDir::new(&dir).family("alpha", ["alpha"]);
    let beta = LockDir::new(&dir).family("beta", ["beta"]);
    let alpha_out = case.0.join("alpha-out");
    let beta_out = case.0.join("beta-out");
    let selection = KernelSelection::new(
        context(&case, "test"),
        vec![
            SelectedCubin::existing(renamed(&case, "beta"), vec![declaration("beta_v1")]).unwrap(),
        ],
    )
    .unwrap();
    let build_alpha = |options| {
        build_in_families(
            &toolchain,
            &[renamed(&case, "alpha")],
            &alpha_out,
            &alpha,
            options,
            None,
        )
    };
    let build_beta = |options| selection.build_in_families(&toolchain, &beta_out, &beta, options);

    build_alpha(update()).unwrap();
    let mut names: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "alpha.lock",
            "alpha.provenance.json",
            "other.lock",
            "other.provenance.json"
        ]
    );
    build_beta(update()).unwrap();
    let beta_lock = read(&dir.join("beta.lock"));
    build_alpha(verify()).unwrap();
    build_beta(verify()).unwrap();
    BuildReport::read(&beta_out.join(BUILD_REPORT_FILE))
        .unwrap()
        .verify_current()
        .unwrap();

    // Updating alpha leaves a stale beta lock alone, and beta still fails.
    fs::write(dir.join("beta.lock"), "[]\n").unwrap();
    build_alpha(update()).unwrap();
    assert_eq!(read(&dir.join("beta.lock")), "[]\n");
    let error = build_beta(verify()).unwrap_err().to_string();
    assert!(error.contains("family beta"), "{error}");
    assert!(error.contains("beta (not locked)"), "{error}");
    build_beta(update()).unwrap();
    assert_eq!(read(&dir.join("beta.lock")), beta_lock);
    assert_eq!(read(&dir.join("other.lock")), "not a lock\n");
    assert_eq!(read(&dir.join("other.provenance.json")), "not provenance\n");

    // One build may produce several families; each gets its own files.
    let both = LockDir::new(case.0.join("both.lock.d"))
        .family("alpha", ["alpha"])
        .family("beta", ["beta"]);
    let specs = [renamed(&case, "alpha"), renamed(&case, "beta")];
    build_in_families(&toolchain, &specs, &case.out(), &both, update(), None).unwrap();
    assert_eq!(read(&both.lock_file("beta")), beta_lock);
    assert_eq!(
        read(&both.lock_file("alpha")),
        read(&alpha.lock_file("alpha"))
    );
    build_in_families(&toolchain, &specs, &case.out(), &both, verify(), None).unwrap();
    // Every failing family is named, not only the first.
    fs::write(both.lock_file("alpha"), "[]\n").unwrap();
    fs::remove_file(both.lock_file("beta")).unwrap();
    let error = build_in_families(&toolchain, &specs, &case.out(), &both, verify(), None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("family alpha: alpha (not locked)"),
        "{error}"
    );
    assert!(error.contains("beta.lock: family beta"), "{error}");

    // Profiling bypasses the lock directory entirely.
    let profiled = LockDir::new(case.0.join("profile.lock.d")).family("alpha", ["alpha"]);
    build_in_families(
        &toolchain,
        &[renamed(&case, "alpha")],
        &case.0.join("profile-out"),
        &profiled,
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: true,
        },
        None,
    )
    .unwrap();
    assert!(!profiled.dir().exists());
}

#[test]
fn lock_directory_reports_verify_from_another_directory() {
    let Some(toolchain) = toolchain() else {
        return;
    };
    let case = Case::new();
    let specs = [tiny(&case)];
    let elsewhere = case.0.join("elsewhere");
    fs::create_dir_all(&elsewhere).unwrap();
    let lock = LockDir::new("relative.lock.d").family("tiny_family", ["tiny"]);
    let original = std::env::current_dir().unwrap();
    // Every other test uses absolute paths, so moving this process is safe.
    // Nothing between the moves panics, so the directory is always restored.
    std::env::set_current_dir(&case.0).unwrap();
    let built = build_in_families(&toolchain, &specs, &case.out(), &lock, update(), None)
        .and_then(|_| {
            // The report-only file is absent, yet still has an absolute location.
            fs::remove_file(lock.provenance_file("tiny_family")).unwrap();
            build_in_families(&toolchain, &specs, &case.out(), &lock, verify(), None)
        })
        .map_err(|e| e.to_string());
    std::env::set_current_dir(&elsewhere).unwrap();
    let report = BuildReport::read(&case.out().join(BUILD_REPORT_FILE)).map_err(|e| e.to_string());
    let verified = report
        .as_ref()
        .map(|r| r.verify_current().map_err(|e| e.to_string()));
    std::env::set_current_dir(original).unwrap();
    built.unwrap();
    verified.unwrap().unwrap();
    let report = report.unwrap();
    let dir = case.0.canonicalize().unwrap().join("relative.lock.d");
    assert_eq!(report.lock_path, dir);
    let family = &report.lock_families[0];
    assert_eq!(family.lock_path, dir.join("tiny_family.lock"));
    assert_eq!(
        family.provenance_path,
        dir.join("tiny_family.provenance.json")
    );
    assert!(!family.provenance_path.exists());
    assert_eq!(family.provenance_changed, ["tiny"]);
}
