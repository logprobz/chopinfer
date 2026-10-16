//! The kernel-lab lock families of `crates/kernel-lab/kernels.lock.d`, one
//! module per family file of this directory (`<family>.rs`). A family file
//! exposes `FAMILY`, its file name, and `cubins()`, every cubin the family's
//! `<family>.lock` gates. kernel-lab's build script builds `all()` in one
//! lock directory build; its tests reach a family's items as
//! `kernel_lab::families::<family>`.

use kernel_build::{CubinSpec, FAMILY_LOCK_SUFFIX, FAMILY_PROVENANCE_SUFFIX};
use std::collections::BTreeSet;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/families.rs"));

/// Whether two names are equal, in a constant (each family's `FAMILY`
/// against its file name).
const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// `specs`, after checking that their names are exactly `names`: the cubin
/// names a family's shapes declare for its tests' `Session::module`.
pub fn declared(names: &[&str], specs: Vec<CubinSpec>) -> Vec<CubinSpec> {
    let mut built: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
    let mut names = names.to_vec();
    built.sort_unstable();
    names.sort_unstable();
    assert_eq!(
        built, names,
        "a family's cubins differ from its declared names"
    );
    specs
}

/// The problems of a lock directory against `families`: a file that is not
/// one family's `<family>.lock` or `<family>.provenance.json`, and a family
/// missing either file. Empty when every file pairs with a family.
pub fn unpaired(dir: &Path, families: &[&str]) -> std::io::Result<Vec<String>> {
    let mut found = BTreeSet::new();
    let mut problems = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        let family = name
            .strip_suffix(FAMILY_LOCK_SUFFIX)
            .or_else(|| name.strip_suffix(FAMILY_PROVENANCE_SUFFIX));
        match family {
            Some(family) if families.contains(&family) => {
                found.insert(name.into_owned());
            }
            Some(family) => problems.push(format!(
                "{name}: no family file {family}.rs; remove the lock files of a removed family"
            )),
            None => problems.push(format!("{name}: not a family lock or provenance file")),
        }
    }
    for family in families {
        for suffix in [FAMILY_LOCK_SUFFIX, FAMILY_PROVENANCE_SUFFIX] {
            let file = format!("{family}{suffix}");
            if !found.contains(&file) {
                problems.push(format!(
                    "{file}: missing for family {family}; create it with KERNELS_LOCK_UPDATE=1"
                ));
            }
        }
    }
    problems.sort();
    Ok(problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock_dir(tag: &str, files: &[&str]) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("kernel-lab-families-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for file in files {
            std::fs::write(dir.join(file), "[]").unwrap();
        }
        dir
    }

    #[test]
    fn family_names_are_file_names() {
        let built: Vec<&str> = all().iter().map(|(name, _)| *name).collect();
        assert_eq!(built, NAMES);
        assert!(NAMES.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn paired_lock_directory_passes() {
        let dir = lock_dir("paired", &["a.lock", "a.provenance.json"]);
        assert_eq!(unpaired(&dir, &["a"]).unwrap(), Vec::<String>::new());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn orphan_and_missing_files_fail() {
        let dir = lock_dir(
            "orphan",
            &[
                "a.lock",
                "a.provenance.json",
                "b.lock",
                "b.provenance.json",
                "c.lock",
                "notes.txt",
            ],
        );
        assert_eq!(
            unpaired(&dir, &["a", "c"]).unwrap(),
            [
                "b.lock: no family file b.rs; remove the lock files of a removed family",
                "b.provenance.json: no family file b.rs; remove the lock files of a removed family",
                "c.provenance.json: missing for family c; create it with KERNELS_LOCK_UPDATE=1",
                "notes.txt: not a family lock or provenance file",
            ]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[should_panic(expected = "differ from its declared names")]
    fn undeclared_cubin_name_fails() {
        let spec = CubinSpec {
            name: "built".into(),
            ..all().remove(0).1.remove(0)
        };
        declared(&["declared"], vec![spec]);
    }
}
