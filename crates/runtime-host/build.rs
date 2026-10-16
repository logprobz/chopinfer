use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Workspace crates by package name, from the root manifest's
/// `[workspace.dependencies]` path entries.
fn workspace_crates(workspace: &Path) -> BTreeMap<String, PathBuf> {
    let text = std::fs::read_to_string(workspace.join("Cargo.toml")).expect("workspace manifest");
    let table = text
        .split("[workspace.dependencies]")
        .nth(1)
        .expect("workspace dependencies")
        .split("\n[")
        .next()
        .unwrap();
    table
        .lines()
        .filter_map(|line| {
            let (name, rest) = line.split_once('=')?;
            let path = rest.split("path = \"").nth(1)?.split('"').next()?;
            Some((name.trim().to_owned(), workspace.join(path)))
        })
        .collect()
}

/// Every dependency a manifest names in a table that builds its library:
/// `[dependencies]`, `[build-dependencies]` and their target-specific forms,
/// optional or not. Development dependencies do not reach the byte path.
fn linked_dependencies(manifest: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut linked = false;
    for line in manifest.lines() {
        let line = line.trim();
        if let Some(header) = line.strip_prefix('[') {
            let header = header.trim_end_matches(']');
            linked = !header.contains("dev-dependencies")
                && (header.ends_with("dependencies") || header.contains("dependencies."));
            if let Some(name) = header.rsplit_once("dependencies.").map(|(_, name)| name) {
                if linked {
                    names.insert(name.to_owned());
                }
                linked = false;
            }
            continue;
        }
        if linked
            && !line.starts_with('#')
            && let Some(name) = line.split(['=', '.']).next()
            && !name.trim().is_empty()
        {
            names.insert(name.trim().to_owned());
        }
    }
    names
}

fn rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_files(&path, found);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
}

/// The sources a planned rank's bytes can depend on, which the direct load's
/// shared implementation digest covers: the whole source trees, build scripts
/// and manifests of this crate and of every workspace crate it links,
/// transitively (the executor, read path and CUDA upload among them), and the
/// lock file that pins every third-party crate. Walking the manifests here
/// keeps the list from going stale. The digest is taken here, so the binary
/// carries 32 bytes and the names, not the sources.
fn direct_sources() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest
        .parent()
        .and_then(Path::parent)
        .expect("workspace directory")
        .to_path_buf();
    let crates = workspace_crates(&workspace);
    let mut pending = vec!["runtime-host".to_owned()];
    let mut covered = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !covered.insert(name.clone()) {
            continue;
        }
        let root = if name == "runtime-host" {
            manifest.clone()
        } else {
            crates[&name].clone()
        };
        let text = std::fs::read_to_string(root.join("Cargo.toml")).expect("crate manifest");
        pending.extend(
            linked_dependencies(&text)
                .into_iter()
                .filter(|dependency| crates.contains_key(dependency)),
        );
    }
    let mut sources: Vec<(String, PathBuf)> = Vec::new();
    for name in &covered {
        let root = if name == "runtime-host" {
            manifest.clone()
        } else {
            crates[name].clone()
        };
        println!("cargo:rerun-if-changed={}", root.join("src").display());
        sources.push((format!("{name}/Cargo.toml"), root.join("Cargo.toml")));
        if root.join("build.rs").is_file() {
            sources.push((format!("{name}/build.rs"), root.join("build.rs")));
        }
        let mut found = Vec::new();
        rust_files(&root.join("src"), &mut found);
        for path in found {
            let relative = path.strip_prefix(&root).expect("under the crate");
            sources.push((format!("{name}/{}", relative.display()), path));
        }
    }
    sources.push(("Cargo.lock".into(), workspace.join("Cargo.lock")));
    sources.sort();
    let mut digest = Sha256::new();
    let mut generated = String::from(
        "/// Generated by build.rs: every source on the direct load's byte path.\n\
         #[cfg(test)]\n\
         pub(crate) const DIRECT_SOURCES: &[&str] = &[\n",
    );
    for (name, path) in &sources {
        println!("cargo:rerun-if-changed={}", path.display());
        let bytes =
            std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        // Each source by name, length and bytes.
        for part in [name.as_bytes(), &(bytes.len() as u64).to_le_bytes(), &bytes] {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
        generated.push_str(&format!("    {name:?},\n"));
    }
    generated.push_str("];\n");
    generated.push_str(&format!(
        "/// SHA-256 of [`DIRECT_SOURCES`] by name, length and bytes.\n\
         pub(crate) const DIRECT_SOURCES_DIGEST: [u8; 32] = {:?};\n",
        <[u8; 32]>::from(digest.finalize())
    ));
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(out.join("direct_sources.rs"), generated).expect("write direct_sources.rs");
}

fn main() {
    direct_sources();
    println!("cargo:rerun-if-env-changed=CHOPINFER_BUILD_REVISION");
    println!("cargo:rerun-if-changed=build.rs");
    // Resolve Git paths so branch changes also invalidate builds in worktrees.
    for name in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--git-path", name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"])
        && let Some(path) = git(&["rev-parse", "--git-path", &reference])
    {
        println!("cargo:rerun-if-changed={path}");
    }
    let revision = std::env::var("CHOPINFER_BUILD_REVISION")
        .ok()
        .or_else(|| git(&["rev-parse", "--verify", "HEAD"]))
        .unwrap_or_default();
    assert!(
        revision.is_empty()
            || (revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())),
        "build revision must be a full Git SHA or empty"
    );
    println!("cargo:rustc-env=CHOPINFER_BUILD_REVISION={revision}");
}
