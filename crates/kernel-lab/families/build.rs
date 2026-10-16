//! Lists the family files beside this build script (every `<family>.rs` but
//! build.rs) as `OUT_DIR/families.rs`: one module per family, `NAMES` and
//! `all()`, sorted by family name.
use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    // A directory rerun scans every file under it, so an added or removed
    // family file regenerates the list.
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut families = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read the family directory") {
        let path = entry.expect("read a family directory entry").path();
        if !path.is_file()
            || path.extension().is_none_or(|extension| extension != "rs")
            || path.file_name().is_some_and(|name| name == "build.rs")
        {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("UTF-8 family file name")
            .to_owned();
        let mut chars = name.chars();
        assert!(
            chars.next().is_some_and(|c| c.is_ascii_lowercase())
                && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "family file {name}.rs: a family name is lowercase ASCII letters, digits and `_`, \
             starting with a letter"
        );
        families.push((name, path));
    }
    families.sort();
    let mut out = String::new();
    for (name, path) in &families {
        let path = path.to_str().expect("UTF-8 family path");
        writeln!(out, "#[path = {path:?}]\npub mod {name};").unwrap();
        writeln!(
            out,
            "const _: () = assert!(same({name}::FAMILY, {name:?}), \
             \"{name}.rs: FAMILY must be its file name\");"
        )
        .unwrap();
    }
    out.push_str("/// Every family's name, sorted.\npub const NAMES: &[&str] = &[\n");
    for (name, _) in &families {
        writeln!(out, "    {name:?},").unwrap();
    }
    out.push_str("];\n");
    out.push_str(
        "/// Every family's name and cubins, sorted by name.\n\
         pub fn all() -> Vec<(&'static str, Vec<kernel_build::CubinSpec>)> {\n    vec![\n",
    );
    for (name, _) in &families {
        writeln!(out, "        ({name}::FAMILY, {name}::cubins()),").unwrap();
    }
    out.push_str("    ]\n}\n");
    let generated =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("families.rs");
    std::fs::write(generated, out).expect("write the family list");
}
