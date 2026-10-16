use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use std::sync::Mutex;

// Every test must hold this lock, including tests that only spawn children.
// A concurrent fork can inherit a fake tool's open write descriptor until exec,
// causing ETXTBSY even after fs::write returns. Cover tool rewrites as well as
// creation and keep the guard until all children finish. The cache test still
// runs its two competing builders concurrently within the guarded test.
static TOOL_WORKFLOW: Mutex<()> = Mutex::new(());

fn write_tool(dir: &std::path::Path, name: &str, body: &str) {
    let path = dir.join(name);
    fs::write(&path, body).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn builds_inspects_and_refuses_replacement() {
    let _workflow = TOOL_WORKFLOW.lock().unwrap_or_else(|e| e.into_inner());
    let root = std::env::temp_dir().join(format!("kernel-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let tools = root.join("bin");
    fs::create_dir(&tools).unwrap();
    let version = "fake tool version\n";
    write_tool(
        &tools,
        "nvcc",
        &format!(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf '{}'; exit 0; fi\nwhile [ \"$1\" != \"-o\" ]; do shift; done\nshift\nprintf 'fake cubin' > \"$1\"\n",
            version
        ),
    );
    write_tool(
        &tools,
        "cuobjdump",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'fake tool version\n'; exit 0; fi\ncase \"$1\" in --dump-elf) printf '64-bit ELF: sm=120\n\tTool Command Line Arguments: -arch sm_120f -m 64 \n';; --dump-elf-symbols) echo STO_ENTRY kernel_entry;; --dump-sass) echo SASS-INSTRUCTIONS;; --dump-ptx) ;; esac\n",
    );
    write_tool(
        &tools,
        "readelf",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'fake tool version\n'; exit 0; fi\ncase \"$1\" in -Ws) echo '  1: 0000 0000 FUNC GLOBAL DEFAULT 1 kernel_entry';; -SW) ;; -d) ;; esac\n",
    );
    let source = root.join("kernel.cu");
    let header = root.join("kernel.cuh");
    fs::write(&source, "extern \"C\" __global__ void kernel_entry() {}\n").unwrap();
    fs::write(&header, "// header\n").unwrap();
    let nvcc = tools.join("nvcc");
    let manifest = root.join("kernel.json");
    fs::write(&manifest, format!(r#"{{"source_paths":["kernel.cu"],"header_paths":["kernel.cuh"],"nvcc_path":{},"nvcc_version":{},"cuobjdump_path":{},"cuobjdump_version":{},"readelf_path":{},"readelf_version":{},"target_architecture":"sm_120f","flags":["-std=c++20"],"expected_exported_symbols":["kernel_entry"],"output_path":"out/kernel.cubin"}}"#, serde_json::to_string(&nvcc).unwrap(), serde_json::to_string(version).unwrap(), serde_json::to_string(&tools.join("cuobjdump")).unwrap(), serde_json::to_string(version).unwrap(), serde_json::to_string(&tools.join("readelf")).unwrap(), serde_json::to_string(version).unwrap())).unwrap();
    let bin = env!("CARGO_BIN_EXE_chopinfer");
    let output = Command::new(bin)
        .current_dir(&root)
        .env("PATH", tools.to_str().unwrap())
        .args(["kernel", "build", "--manifest", "kernel.json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(bin)
        .current_dir(&root)
        .env("PATH", tools.to_str().unwrap())
        .args(["kernel", "inspect", "--manifest", "kernel.json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source_bytes = fs::read(&source).unwrap();
    fs::write(&source, b"stale").unwrap();
    let stale = Command::new(bin)
        .current_dir(&root)
        .env("PATH", tools.to_str().unwrap())
        .args(["kernel", "inspect", "--manifest", "kernel.json"])
        .output()
        .unwrap();
    assert!(!stale.status.success());
    fs::write(&source, source_bytes).unwrap();
    let manifest_bytes = fs::read(&manifest).unwrap();
    fs::write(&manifest, b"{}").unwrap();
    let stale = Command::new(bin)
        .current_dir(&root)
        .env("PATH", tools.to_str().unwrap())
        .args(["kernel", "inspect", "--manifest", "kernel.json"])
        .output()
        .unwrap();
    assert!(!stale.status.success());
    fs::write(&manifest, manifest_bytes).unwrap();
    let cuobjdump = tools.join("cuobjdump");
    let tool_bytes = fs::read(&cuobjdump).unwrap();
    for (needle, replacement) in [
        ("-arch sm_120f", "-arch sm_121f"),
        ("--dump-ptx) ;;", "--dump-ptx) echo PTX;;"),
        ("STO_ENTRY kernel_entry", "STO_ENTRY other_entry"),
        ("--dump-sass) echo SASS-INSTRUCTIONS", "--dump-sass) ;;"),
    ] {
        let changed = String::from_utf8(tool_bytes.clone())
            .unwrap()
            .replace(needle, replacement);
        fs::write(&cuobjdump, changed).unwrap();
        let rejected = Command::new(bin)
            .current_dir(&root)
            .env("PATH", tools.to_str().unwrap())
            .args(["kernel", "inspect", "--manifest", "kernel.json"])
            .output()
            .unwrap();
        assert!(!rejected.status.success());
        fs::write(&cuobjdump, &tool_bytes).unwrap();
    }
    let receipt = root.join("out/kernel.receipt.json");
    let receipt_bytes = fs::read(&receipt).unwrap();
    fs::write(&receipt, b"{}").unwrap();
    let rejected = Command::new(bin)
        .current_dir(&root)
        .env("PATH", tools.to_str().unwrap())
        .args(["kernel", "inspect", "--manifest", "kernel.json"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    fs::write(&receipt, receipt_bytes).unwrap();
    let cubin = root.join("out/kernel.cubin");
    let cubin_bytes = fs::read(&cubin).unwrap();
    fs::write(&cubin, b"tampered").unwrap();
    let rejected = Command::new(bin)
        .current_dir(&root)
        .env("PATH", tools.to_str().unwrap())
        .args(["kernel", "inspect", "--manifest", "kernel.json"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    fs::write(&cubin, cubin_bytes).unwrap();
    let output = Command::new(bin)
        .current_dir(&root)
        .env("PATH", tools.to_str().unwrap())
        .args(["kernel", "build", "--manifest", "kernel.json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(root.join("out/kernel.receipt.json").is_file());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn rejects_multiple_sources_before_compilation() {
    let _workflow = TOOL_WORKFLOW.lock().unwrap_or_else(|e| e.into_inner());
    let root = std::env::temp_dir().join(format!("kernel-many-sources-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("manifest.json"), r#"{"source_paths":["a.cu","b.cu"],"nvcc_path":"nvcc","nvcc_version":"","cuobjdump_path":"cuobjdump","cuobjdump_version":"","readelf_path":"readelf","readelf_version":"","target_architecture":"sm_120f","flags":[],"expected_exported_symbols":["entry"],"output_path":"out.cubin"}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_chopinfer"))
        .current_dir(&root)
        .args(["kernel", "inspect", "--manifest", "manifest.json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("exactly one source"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn failed_compile_leaves_no_partial_output_or_temporary() {
    let _workflow = TOOL_WORKFLOW.lock().unwrap_or_else(|e| e.into_inner());
    let root = std::env::temp_dir().join(format!("kernel-partial-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let tools = root.join("bin");
    fs::create_dir(&tools).unwrap();
    write_tool(
        &tools,
        "nvcc",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'v\\n'; exit 0; fi\nexit 1\n",
    );
    write_tool(
        &tools,
        "cuobjdump",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'v\\n'; fi\n",
    );
    write_tool(
        &tools,
        "readelf",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'v\\n'; fi\n",
    );
    fs::write(root.join("a.cu"), "x").unwrap();
    fs::write(root.join("manifest.json"), r#"{"source_paths":["a.cu"],"nvcc_path":"bin/nvcc","nvcc_version":"v\n","cuobjdump_path":"bin/cuobjdump","cuobjdump_version":"v\n","readelf_path":"bin/readelf","readelf_version":"v\n","target_architecture":"sm_120f","flags":[],"expected_exported_symbols":["entry"],"output_path":"out.cubin"}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_chopinfer"))
        .current_dir(&root)
        .args(["kernel", "build", "--manifest", "manifest.json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!root.join("out.cubin").exists());
    assert!(!root.join("out.receipt.json").exists());
    assert_eq!(
        fs::read_dir(&root)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains("tmp-"))
            .count(),
        0
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn incomplete_pairs_fail_closed_and_preserve_bytes() {
    let _workflow = TOOL_WORKFLOW.lock().unwrap_or_else(|e| e.into_inner());
    for (label, cubin_exists) in [("cubin-only", true), ("receipt-only", false)] {
        let root =
            std::env::temp_dir().join(format!("kernel-sentinel-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let source = root.join("a.cu");
        fs::write(&source, "x").unwrap();
        let tools = root.join("bin");
        fs::create_dir(&tools).unwrap();
        for name in ["nvcc", "cuobjdump", "readelf"] {
            fs::write(tools.join(name), "tool").unwrap();
        }
        let output = root.join("out.cubin");
        let receipt = root.join("out.receipt.json");
        let sentinel = if cubin_exists {
            b"cubin sentinel".as_slice()
        } else {
            b"receipt sentinel".as_slice()
        };
        if cubin_exists {
            fs::write(&output, sentinel).unwrap();
        } else {
            fs::write(&receipt, sentinel).unwrap();
        }
        let manifest = root.join("manifest.json");
        fs::write(&manifest, r#"{"source_paths":["a.cu"],"nvcc_path":"bin/nvcc","nvcc_version":"","cuobjdump_path":"bin/cuobjdump","cuobjdump_version":"","readelf_path":"bin/readelf","readelf_version":"","target_architecture":"sm_120f","flags":[],"expected_exported_symbols":["entry"],"output_path":"out.cubin"}"#).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_chopinfer"))
            .current_dir(&root)
            .args(["kernel", "build", "--manifest", "manifest.json"])
            .output()
            .unwrap();
        assert!(!result.status.success());
        if cubin_exists {
            assert_eq!(fs::read(&output).unwrap(), sentinel);
        } else {
            assert_eq!(fs::read(&receipt).unwrap(), sentinel);
        }
        let _ = fs::remove_dir_all(root);
    }
}

#[test]
fn cache_reuses_invalidates_serializes_and_refuses_corruption() {
    let _workflow = TOOL_WORKFLOW.lock().unwrap_or_else(|e| e.into_inner());
    let root = std::env::temp_dir().join(format!("kernel-cache-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("bin")).unwrap();
    let tools = root.join("bin");
    write_tool(
        &tools,
        "nvcc",
        &format!(
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then printf 'v\n'; exit 0; fi
for arg do
  if [ "$arg" = "-M" ]; then
    echo scan >> '{}'
    printf 'module: '
    for source do last=$source; done
    printf '%s\n' "$last"
    exit 0
  fi
done
echo compile >> '{}'
while [ "$1" != "-o" ]; do shift; done
shift
printf 'fake cubin' > "$1"
if [ -f '{}' ]; then exit 1; fi
"#,
            root.join("calls.log").display(),
            root.join("calls.log").display(),
            root.join("fail.flag").display()
        ),
    );
    write_tool(
        &tools,
        "cuobjdump",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'v\\n'; exit 0; fi\ncase \"$1\" in --dump-elf) printf '64-bit ELF: sm=120\n\tTool Command Line Arguments: -arch sm_120f -m 64 \n';; --dump-elf-symbols) echo STO_ENTRY entry;; --dump-sass) echo SASS;; esac\n",
    );
    write_tool(
        &tools,
        "readelf",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'v\\n'; fi\n",
    );
    fs::write(root.join("a.cu"), "a").unwrap();
    fs::write(root.join("b.cu"), "b").unwrap();
    fs::create_dir(root.join("include")).unwrap();
    fs::write(root.join("include/outer.h"), "#include \"inner.h\"\n").unwrap();
    fs::write(root.join("include/inner.h"), "header A").unwrap();
    for name in ["a", "b"] {
        let manifest = serde_json::json!({"source_paths": [format!("{name}.cu")],
            "header_paths": ["include/outer.h"], "nvcc_path": "bin/nvcc", "nvcc_version": "v\n",
            "cuobjdump_path": "bin/cuobjdump", "cuobjdump_version": "v\n", "readelf_path": "bin/readelf",
            "readelf_version": "v\n", "target_architecture": "sm_120f", "flags": [],
            "expected_exported_symbols": ["entry"], "output_path": "absent/ignored.cubin"});
        fs::write(
            root.join(format!("{name}.json")),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }
    let command = |name: &str, verb: &str| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_chopinfer"));
        command.current_dir(&root).args([
            "kernel",
            verb,
            "--manifest",
            &format!("{name}.json"),
            "--cache-dir",
            "cache",
        ]);
        command
    };
    let run = |name: &str, verb: &str| {
        let result = command(name, verb).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        serde_json::from_slice::<serde_json::Value>(&result.stdout).unwrap()
    };
    let calls = || fs::read_to_string(root.join("calls.log")).unwrap_or_default();
    let first = run("a", "build");
    assert_eq!(first["cache_hit"], false);
    run("b", "build");
    let cold = calls();
    assert_eq!(run("a", "build")["cache_hit"], true);
    assert_eq!(run("b", "build")["cache_hit"], true);
    run("a", "inspect");
    assert_eq!(
        calls().matches("compile").count(),
        cold.matches("compile").count(),
        "warm requests scan dependencies without compilation"
    );
    assert!(!root.join("absent").exists());
    let manifest_path = root.join("a.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["output_path"] = "different/unused.cubin".into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert_eq!(run("a", "build")["cache_hit"], true);
    assert_eq!(
        calls().matches("compile").count(),
        cold.matches("compile").count(),
        "publication output is excluded from the module key"
    );
    fs::write(root.join("a.cu"), "edited a").unwrap();
    run("a", "build");
    assert_eq!(run("b", "build")["cache_hit"], true);
    assert_eq!(calls().matches("compile").count(), 3);
    fs::write(root.join("include/inner.h"), "header B").unwrap();
    let changed = run("a", "build");
    assert_ne!(first["key"], changed["key"]);
    assert_eq!(calls().matches("compile").count(), 4);
    fs::write(root.join("a.cu"), "concurrent").unwrap();
    let left = command("a", "build")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let right = command("a", "build")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    assert!(left.wait_with_output().unwrap().status.success());
    assert!(right.wait_with_output().unwrap().status.success());
    assert_eq!(calls().matches("compile").count(), 5);
    let latest = run("a", "build");
    let before = calls();
    for field in ["cubin_path", "sass_path", "receipt_path"] {
        let path = std::path::Path::new(latest[field].as_str().unwrap());
        let saved = fs::read(path).unwrap();
        fs::write(path, b"corrupt").unwrap();
        assert!(!command("a", "build").output().unwrap().status.success());
        assert_eq!(fs::read(path).unwrap(), b"corrupt");
        fs::write(path, &saved).unwrap();
        fs::remove_file(path).unwrap();
        assert!(!command("a", "build").output().unwrap().status.success());
        fs::write(path, saved).unwrap();
    }
    assert_eq!(
        calls().matches("compile").count(),
        before.matches("compile").count(),
        "corrupt entries are never rebuilt silently"
    );
    fs::write(root.join("a.cu"), "partial write").unwrap();
    fs::write(root.join("fail.flag"), "fail").unwrap();
    assert!(!command("a", "build").output().unwrap().status.success());
    assert!(
        !fs::read_dir(root.join("cache")).unwrap().any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-"))
    );
    fs::remove_file(root.join("fail.flag")).unwrap();
    assert_eq!(run("a", "build")["cache_hit"], false);
    let before_tool_edit = calls().matches("compile").count();
    let nvcc_path = tools.join("nvcc");
    let original_tool = fs::read_to_string(&nvcc_path).unwrap();
    fs::write(
        &nvcc_path,
        format!("{original_tool}\n# changed binary identity\n"),
    )
    .unwrap();
    assert_eq!(run("a", "build")["cache_hit"], false);
    assert_eq!(calls().matches("compile").count(), before_tool_edit + 1);
    manifest["flags"] = serde_json::json!(["-O2"]);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert_eq!(run("a", "build")["cache_hit"], false);
    manifest["flags"] = serde_json::json!(["--options-file=hidden"]);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(!command("a", "build").output().unwrap().status.success());
    manifest["flags"] = serde_json::json!([]);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    // A dependency outside the snapshotted trees must fail before compilation.
    let outside = root.join("untracked.dat");
    fs::write(&outside, "dependency").unwrap();
    fs::write(
        &nvcc_path,
        original_tool.replace(
            "printf '%s\\n' \"$last\"",
            &format!("printf '%s\\n' \"$last\"; echo {}", outside.display()),
        ),
    )
    .unwrap();
    let before_uncovered = calls().matches("compile").count();
    let rejected = command("a", "build").output().unwrap();
    assert!(
        !rejected.status.success(),
        "uncovered dependency was accepted"
    );
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("uncovered compiler dependency"));
    assert_eq!(calls().matches("compile").count(), before_uncovered);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn cache_tracks_consumed_extensionless_headers_without_tracking_outputs() {
    let _workflow = TOOL_WORKFLOW.lock().unwrap_or_else(|e| e.into_inner());
    let root = std::env::temp_dir().join(format!("kernel-closure-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("bin")).unwrap();
    fs::create_dir_all(root.join("include")).unwrap();
    let tools = root.join("bin");
    // Use the real CPU preprocessor to exercise transitive include resolution.
    write_tool(
        &tools,
        "nvcc",
        &format!(
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then printf 'v\n'; exit 0; fi
for item do last=$item; done
for arg do
 if [ "$arg" = "-M" ]; then
  exec /usr/bin/g++ -M -MT module -x c++ -I'{}/include' "$last"
 fi
done
echo compile >> '{}/compiler-output'
while [ "$1" != "-o" ]; do shift; done
shift
printf 'fake cubin' > "$1"
"#,
            root.display(),
            root.display()
        ),
    );
    write_tool(
        &tools,
        "cuobjdump",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'v\\n'; exit 0; fi\ncase \"$1\" in --dump-elf) printf '64-bit ELF: sm=120\n\tTool Command Line Arguments: -arch sm_120f -m 64 \n';; --dump-elf-symbols) echo STO_ENTRY entry;; --dump-sass) echo SASS;; esac\n",
    );
    write_tool(
        &tools,
        "readelf",
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'v\\n'; fi\n",
    );
    fs::write(root.join("a.cu"), "#include <operand>\nint a = VALUE;\n").unwrap();
    fs::write(root.join("b.cu"), "int b;\n").unwrap();
    fs::write(root.join("include/operand"), "#include \"nested.cuh\"\n").unwrap();
    fs::write(root.join("include/nested.cuh"), "#define VALUE 1\n").unwrap();
    fs::write(root.join("bin/compiler-data"), "tool data A").unwrap();
    // g++ includes this system header implicitly even for this tiny source.
    let dependency_output = Command::new("/usr/bin/g++")
        .args(["-M", "-MT", "module", "-x", "c++"])
        .arg(format!("-I{}", root.join("include").display()))
        .arg(root.join("a.cu"))
        .output()
        .unwrap();
    assert!(dependency_output.status.success());
    let dependency_text = String::from_utf8(dependency_output.stdout).unwrap();
    assert!(dependency_text.contains("/include/operand"));
    assert!(dependency_text.contains("/include/nested.cuh"));
    let implicit: Vec<_> = dependency_text
        .replace("\\\n", " ")
        .split_whitespace()
        .filter(|p| p.starts_with("/usr/"))
        .map(str::to_owned)
        .collect();
    for name in ["a", "b"] {
        fs::write(
            root.join(format!("{name}.json")),
            serde_json::to_vec(&serde_json::json!({
                "source_paths":[format!("{name}.cu")], "header_paths":implicit,
                "cache_dependency_roots":["include"], "cache_tool_paths":["bin/compiler-data"],
                "nvcc_path":"bin/nvcc", "nvcc_version":"v\n",
                "cuobjdump_path":"bin/cuobjdump", "cuobjdump_version":"v\n",
                "readelf_path":"bin/readelf", "readelf_version":"v\n",
                "target_architecture":"sm_120f", "flags":[],
                "expected_exported_symbols":["entry"], "output_path":"unused.cubin"
            }))
            .unwrap(),
        )
        .unwrap();
    }
    let run = |name: &str| {
        let out = Command::new(env!("CARGO_BIN_EXE_chopinfer"))
            .current_dir(&root)
            .args([
                "kernel",
                "build",
                "--manifest",
                &format!("{name}.json"),
                "--cache-dir",
                "cache",
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice::<serde_json::Value>(&out.stdout).unwrap()
    };
    let count = || {
        fs::read_to_string(root.join("compiler-output"))
            .unwrap()
            .lines()
            .count()
    };
    let first = run("a");
    assert_eq!(
        count(),
        1,
        "new unrelated extensionless output must not invalidate compilation"
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(first["receipt_path"].as_str().unwrap()).unwrap())
            .unwrap();
    let inputs = &receipt["identity"]["inputs"];
    assert!(
        inputs
            .get(root.join("include/operand").to_str().unwrap())
            .is_some()
    );
    assert!(
        inputs
            .get(root.join("include/nested.cuh").to_str().unwrap())
            .is_some()
    );
    assert!(
        inputs
            .get(root.join("compiler-output").to_str().unwrap())
            .is_none()
    );
    assert!(
        receipt["identity"]["tools"]
            .get(root.join("bin/compiler-data").to_str().unwrap())
            .is_some()
    );
    assert_eq!(run("a")["cache_hit"], true);
    assert_eq!(count(), 1);
    let b = run("b");
    fs::write(
        root.join("include/operand"),
        "#include \"nested.cuh\"\n// changed consumed header\n",
    )
    .unwrap();
    assert_ne!(run("a")["key"], first["key"]);
    assert_eq!(
        run("b")["key"],
        b["key"],
        "unconsumed extensionless headers do not invalidate siblings"
    );
    assert_eq!(count(), 3);
    fs::write(
        root.join("a.cu"),
        "#include <operand>\nint a = VALUE + 1;\n",
    )
    .unwrap();
    run("a");
    assert_eq!(run("b")["cache_hit"], true);
    assert_eq!(count(), 4);
    fs::write(root.join("include/nested.cuh"), "#define VALUE 2\n").unwrap();
    assert_eq!(run("a")["cache_hit"], false);
    assert_eq!(count(), 5);
    fs::write(root.join("bin/compiler-data"), "tool data B").unwrap();
    assert_eq!(run("a")["cache_hit"], false);
    assert_eq!(count(), 6);
    // A consumed extensionless transitive header remains part of the closure.
    fs::write(root.join("include/operand"), "#include <selected>\n").unwrap();
    fs::write(root.join("include/selected"), "#define VALUE 3\n").unwrap();
    run("a");
    fs::write(root.join("include/nested-unused"), "unrelated").unwrap();
    assert_eq!(run("a")["cache_hit"], true);
    let before = count();
    fs::write(root.join("include/selected"), "#define VALUE 4\n").unwrap();
    assert_eq!(run("a")["cache_hit"], false);
    assert_eq!(count(), before + 1);
    // Quote includes search the including file's directory before -I directories.
    // Creating a nearer extensionless header changes an otherwise warm lookup.
    fs::write(root.join("a.cu"), "#include \"operand\"\nint a = VALUE;\n").unwrap();
    run("a");
    let before_shadow = count();
    fs::write(root.join("operand"), "#define VALUE 5\n").unwrap();
    let shadow = run("a");
    assert_eq!(shadow["cache_hit"], false);
    assert_eq!(count(), before_shadow + 1);
    let shadow_receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(shadow["receipt_path"].as_str().unwrap()).unwrap())
            .unwrap();
    assert!(
        shadow_receipt["identity"]["compiler_dependencies"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!(root.join("operand")))
    );
    assert_eq!(run("a")["cache_hit"], true);
    fs::remove_dir_all(root).unwrap();
}
