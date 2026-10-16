//! `eval files`: the files under the root that `eval` reads for a model,
//! which a release install ships at the same relative paths so `eval
//! verify-references` and the shared kinds' captures work from it.

use std::path::{Component, Path};

use crate::declaration::{self, Input, Loaded};

/// The declared inputs the shared kinds read, each with the file read
/// under it when it is a directory: the teacher texts (`teacher_forced`),
/// the frontend's thinking switch, and the api samples and chat renders
/// (`api`).
const SHARED_INPUTS: [(&str, Option<&str>); 4] = [
    ("fixture_root", Some("teacher-texts.json")),
    ("frontend", None),
    ("api_samples", None),
    ("chat_renders", None),
];

/// A declared path as a relative path under the root; `None` for an
/// absolute path or one that leaves the root.
fn relative(path: &str) -> Option<&str> {
    Path::new(path)
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
        .then_some(path)
}

/// The media tensors a workload file names, relative to its directory
/// (`workload_sequences`).
fn workload_media(root: &Path, workload: &str) -> Vec<String> {
    let Ok((_, value)) = crate::json::read(&root.join(workload)) else {
        return Vec::new();
    };
    let directory = Path::new(workload).parent().unwrap_or(Path::new(""));
    let items = value["sequences"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|sequence| sequence["media"].as_array())
        .flatten();
    items
        .flat_map(|item| [&item["tensor"]["path"], &item["grid"]["path"]])
        .filter_map(|path| path.as_str())
        .map(|path| directory.join(path).display().to_string())
        .collect()
}

/// The files `eval` reads for `loaded`, relative to its root, sorted: the
/// declaration (`qual.json`, or a variant's `qual.VARIANT.json`), its
/// reference lock and module metadata, the checkpoint lock and registry,
/// every pinned reference with its workload (declared, or the
/// `<stem>.workload.json` beside it) and the media tensors the workload
/// names, the capture a reference scores and the manifest that pins it,
/// and the shared kinds' inputs. Environment inputs live outside the root
/// and are not listed; optional files are listed when present, and a
/// missing required file is an error.
pub fn files(loaded: &Loaded) -> Result<Vec<String>, String> {
    let declaration = &loaded.declaration;
    let root = &loaded.root;
    let name = loaded
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(declaration::FILE_NAME);
    let mut required = vec![
        format!("models/{}/{name}", declaration.model),
        declaration::MODULES.to_owned(),
        declaration.checkpoint.lock.clone(),
    ];
    if let Some((_, registry)) = &declaration.checkpoint.registry {
        required.push(registry.clone());
    }
    let lock = crate::lock::path(loaded);
    let mut optional: Vec<String> = lock
        .strip_prefix(root)
        .ok()
        .and_then(|lock| lock.to_str())
        .map(String::from)
        .into_iter()
        .collect();
    for pin in &declaration.references {
        let path = match &pin.path {
            Input::Path(path) => {
                required.push(path.clone());
                Some(Path::new(path))
            }
            _ => None,
        };
        let directory = path.and_then(Path::parent).unwrap_or(Path::new(""));
        let workload = match &pin.workload {
            Some(Input::Path(workload)) => Some(workload.clone()),
            Some(_) => None,
            None => path
                .and_then(Path::file_stem)
                .and_then(|stem| stem.to_str())
                .map(|stem| {
                    let workload = directory.join(format!("{stem}.workload.json"));
                    workload.display().to_string()
                }),
        };
        if let Some(workload) = workload.filter(|workload| relative(workload).is_some()) {
            optional.extend(workload_media(root, &workload));
            optional.push(workload);
        }
        if let Some(capture) = &pin.capture_sha256
            && path.is_some()
        {
            let manifest = directory.join("manifest.json");
            if let Ok((_, value)) = crate::json::read(&root.join(&manifest)) {
                optional.push(manifest.display().to_string());
                for (name, digest) in value["files"].as_object().into_iter().flatten() {
                    if digest.as_str() == Some(capture.as_str()) {
                        optional.push(directory.join(name).display().to_string());
                    }
                }
            }
        }
    }
    for (name, file) in SHARED_INPUTS {
        if let Some(Input::Path(path)) = declaration.input(name) {
            optional.push(match file {
                Some(file) => Path::new(path).join(file).display().to_string(),
                None => path.clone(),
            });
        }
    }
    let mut listed = Vec::new();
    for path in required {
        let path = relative(&path)
            .filter(|path| root.join(path).is_file())
            .ok_or_else(|| format!("{}: not a file under {}", path, root.display()))?;
        listed.push(path.to_owned());
    }
    listed.extend(
        optional
            .iter()
            .filter_map(|path| relative(path))
            .filter(|path| root.join(path).is_file())
            .map(String::from),
    );
    listed.sort();
    listed.dedup();
    Ok(listed)
}
