use kernel_build::{CubinSpec, EntrySpec};
use std::path::Path;

pub fn cubin_spec(bundle: &Path, name: &str, header: &str, entry: EntrySpec) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: bundle.into(),
        include_dirs: vec!["cuda".into()],
        headers: vec![format!("#include \"{header}\"")],
        defines: vec![],
        entries: vec![entry],
    }
}
