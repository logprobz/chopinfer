//! Where `--model` and `--package` point: a model's declaration directory
//! `ROOT/models/<m>`, in a checkout or an install.
//!
//! A package names its model in the serving profiles it carries, the
//! `model` of its `serve-profiles` component. A package that carries none
//! names only its bundle (`manifest.json` `id`); the model it belongs to is
//! then the one whose `models/<m>/profiles.json` serves that bundle.

use std::path::{Path, PathBuf};

use crate::Host;
use crate::declaration::FILE_NAME;
use crate::json;

/// The model directory of `ROOT/models/<model>` when it holds a
/// declaration.
fn declared(root: &Path, model: &str) -> Option<PathBuf> {
    let directory = root.join("models").join(model);
    directory.join(FILE_NAME).is_file().then_some(directory)
}

/// The names of the models under `ROOT/models` that hold a declaration and
/// pass `keep`, sorted.
fn models(root: &Path, keep: impl Fn(&Path) -> bool) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root.join("models")) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|entry| entry.path().join(FILE_NAME).is_file() && keep(&entry.path()))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    names.sort();
    names
}

/// Whether `models/<m>/profiles.json` names a profile of bundle `id`.
fn serves_bundle(model: &Path, id: &str) -> bool {
    json::read(&model.join("profiles.json")).is_ok_and(|(_, value)| {
        value["profiles"].as_object().is_some_and(|profiles| {
            profiles
                .values()
                .any(|profile| profile["bundle_id"].as_str() == Some(id))
        })
    })
}

/// The model the serving profiles a package carries name: the `model` of
/// the component whose role is `chop.serve-profiles.v1`. The component is
/// read as serving reads it, as a bounded regular file without following a
/// link, and parsed only when its bytes have the manifest's length and
/// SHA-256. `None` when the package carries no profiles.
fn carried_model(directory: &Path, manifest: &serde_json::Value) -> Result<Option<String>, String> {
    use runtime_bundle::deployment::{DEPLOYMENT_COMPONENT_MAX_BYTES, SERVE_PROFILES_ROLE_V1};
    let declared: Vec<_> = manifest["components"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|component| component["role"] == SERVE_PROFILES_ROLE_V1)
        .collect();
    let component = match declared.as_slice() {
        [] => return Ok(None),
        [component] => component,
        _ => return Err(format!("{}: several serving profiles", directory.display())),
    };
    let invalid = || {
        format!(
            "{}: invalid serving profiles component",
            directory.display()
        )
    };
    let relative = component["path"]
        .as_str()
        .and_then(|path| runtime_bundle::BundleRelativePathV1::new(path).ok())
        .ok_or_else(invalid)?;
    let length = component["byte_length"].as_u64().ok_or_else(invalid)?;
    let sha256 = component["sha256"].as_str().ok_or_else(invalid)?;
    if length > DEPLOYMENT_COMPONENT_MAX_BYTES {
        return Err(format!(
            "{}: the serving profiles exceed {DEPLOYMENT_COMPONENT_MAX_BYTES} bytes",
            directory.display()
        ));
    }
    let file = runtime_bundle::VerifiedComponentFileV1::open(
        directory.join(relative.as_path()),
        length,
        sha256,
    )
    .map_err(|error| format!("{}: serving profiles: {error}", directory.display()))?;
    let value = json::parse(file.bytes()).map_err(|error| {
        format!(
            "{}: serving profiles: not JSON: {error}",
            directory.display()
        )
    })?;
    let model = value["model"].as_str().ok_or_else(invalid)?;
    if model.is_empty()
        || !model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        return Err(invalid());
    }
    Ok(Some(model.to_owned()))
}

/// The directories above `host`'s binary: its install root, or the
/// checkout it was built in.
fn host_roots(host: &Host) -> impl Iterator<Item = &Path> {
    host.executable
        .parent()
        .into_iter()
        .flat_map(Path::ancestors)
}

/// The model directory `--model DIR` names: `DIR` itself when it holds
/// `qual.json` or is not a checkpoint; for a checkpoint directory (holding
/// `config.json`), the model the host selects for it as `serve --model`
/// does, declared under the nearest root above this binary.
pub fn model(directory: &Path, host: &Host) -> Result<PathBuf, String> {
    if directory.join(FILE_NAME).is_file() || !directory.join("config.json").is_file() {
        return Ok(directory.to_path_buf());
    }
    let name = (host.checkpoint_model)(directory)
        .map_err(|error| format!("--model {}: {error}", directory.display()))?;
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        return Err(format!(
            "--model {}: the checkpoint's model name {name:?} is not a directory name",
            directory.display()
        ));
    }
    host_roots(host)
        .find_map(|root| declared(root, &name))
        .ok_or_else(|| {
            format!(
                "--model {}: the checkpoint is a {name} checkpoint, but no models/{name}/{FILE_NAME} \
                 is above {}; pass --model ROOT/models/{name}",
                directory.display(),
                host.executable.display()
            )
        })
}

/// The model directory of `--package DIR`: for a package directory
/// (holding `manifest.json`), the model under the nearest root above it,
/// else above this binary, that its carried serving profiles name, or, for
/// a package without them, whose `profiles.json` serves the package's
/// bundle; for an install root or checkout (holding `models/`), its one
/// declared model. Ambiguity is refused.
pub fn package(directory: &Path, host: &Host) -> Result<PathBuf, String> {
    let manifest = directory.join("manifest.json");
    if manifest.is_file() {
        let (_, value) = json::read(&manifest)?;
        let id = value["id"]
            .as_str()
            .ok_or_else(|| format!("{}: names no bundle id", manifest.display()))?;
        let package = std::fs::canonicalize(directory)
            .map_err(|error| format!("{}: {error}", directory.display()))?;
        if let Some(name) = carried_model(&package, &value)? {
            return package
                .ancestors()
                .chain(host_roots(host))
                .find_map(|root| declared(root, &name))
                .ok_or_else(|| {
                    format!(
                        "{} serves model {name}, but no models/{name}/{FILE_NAME} is above it \
                         or {}; pass --model ROOT/models/{name}",
                        package.display(),
                        host.executable.display()
                    )
                });
        }
        for root in package.ancestors().chain(host_roots(host)) {
            match models(root, |model| serves_bundle(model, id)).as_slice() {
                [] => continue,
                [model] => return Ok(root.join("models").join(model)),
                several => {
                    return Err(format!(
                        "several models under {} serve bundle {id} ({}); pass --model \
                         ROOT/models/<model>",
                        root.join("models").display(),
                        several.join(", ")
                    ));
                }
            }
        }
        return Err(format!(
            "no models/<model>/profiles.json above {} or {} serves bundle {id}; pass --model \
             ROOT/models/<model>",
            package.display(),
            host.executable.display()
        ));
    }
    if directory.join("models").is_dir() {
        return match models(directory, |_| true).as_slice() {
            [model] => Ok(directory.join("models").join(model)),
            [] => Err(format!(
                "{} declares no model (no models/<model>/{FILE_NAME})",
                directory.display()
            )),
            several => Err(format!(
                "{} holds several models ({}); pass --model {}",
                directory.display(),
                several.join(", "),
                directory.join("models/<model>").display()
            )),
        };
    }
    Err(format!(
        "{}: --package takes a package directory (manifest.json) or an install root (models/)",
        directory.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temporary(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("runtime-eval-locate-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// `ROOT/models/<model>` with a declaration and, given a bundle, the
    /// profiles that serve it.
    fn model_at(root: &Path, model: &str, bundle: Option<&str>) {
        let directory = root.join("models").join(model);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(FILE_NAME), "{}").unwrap();
        if let Some(bundle) = bundle {
            let profiles = json!({"profiles": {"tp2": {"bundle_id": bundle}}});
            std::fs::write(directory.join("profiles.json"), profiles.to_string()).unwrap();
        }
    }

    fn package_at(directory: &Path, bundle: &str) {
        std::fs::create_dir_all(directory).unwrap();
        let manifest = json!({"id": bundle, "schema": "chop-model-bundle-v1"});
        std::fs::write(directory.join("manifest.json"), manifest.to_string()).unwrap();
    }

    fn host(executable: PathBuf) -> Host {
        Host {
            executable,
            checkpoint_model: Box::new(|checkpoint| {
                let config = std::fs::read_to_string(checkpoint.join("config.json")).unwrap();
                match config.trim() {
                    "{}" => Err("no installed family implementation accepts it".into()),
                    name => Ok(name.trim_matches('"').into()),
                }
            }),
            serve_listen: None,
        }
    }

    #[test]
    fn a_checkpoint_resolves_through_the_host_to_its_installed_model() {
        let root = temporary("checkpoint");
        model_at(&root, "alpha", None);
        let installed = host(root.join("bin/chopinfer"));
        // A model directory is taken as it is.
        let model_directory = root.join("models/alpha");
        assert_eq!(
            model(&model_directory, &installed).unwrap(),
            model_directory
        );
        let checkpoint = root.join("checkpoint");
        std::fs::create_dir_all(&checkpoint).unwrap();
        std::fs::write(checkpoint.join("config.json"), "\"alpha\"").unwrap();
        assert_eq!(model(&checkpoint, &installed).unwrap(), model_directory);
        // Installed elsewhere, or refused by every helper.
        let nowhere = host(PathBuf::from("/nonexistent/bin/chopinfer"));
        let error = model(&checkpoint, &nowhere).unwrap_err();
        assert!(error.contains("no models/alpha/qual.json"), "{error}");
        for (config, refusal) in [
            ("{}", "no installed family"),
            ("\"../alpha\"", "not a directory name"),
        ] {
            std::fs::write(checkpoint.join("config.json"), config).unwrap();
            let error = model(&checkpoint, &installed).unwrap_err();
            assert!(error.contains(refusal), "{error}");
        }
        // A directory that is neither is left to the declaration's refusal.
        let neither = root.join("empty");
        std::fs::create_dir_all(&neither).unwrap();
        assert_eq!(model(&neither, &installed).unwrap(), neither);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_package_resolves_to_the_model_serving_its_bundle() {
        let root = temporary("package");
        model_at(&root, "alpha", Some("alpha.v1"));
        model_at(&root, "beta", Some("beta.v1"));
        let nowhere = host(PathBuf::from("/nonexistent/bin/chopinfer"));
        // A package emitted inside the install belongs to it.
        let inside = root.join("packages/alpha");
        package_at(&inside, "alpha.v1");
        let canonical = std::fs::canonicalize(&root).unwrap();
        assert_eq!(
            package(&inside, &nowhere).unwrap(),
            canonical.join("models/alpha")
        );
        // A package elsewhere belongs to the install of this binary.
        let outside = temporary("package-outside");
        package_at(&outside, "beta.v1");
        let installed = host(root.join("bin/chopinfer"));
        assert_eq!(
            package(&outside, &installed).unwrap(),
            root.join("models/beta")
        );
        let error = package(&outside, &nowhere).unwrap_err();
        assert!(error.contains("serves bundle beta.v1"), "{error}");
        // Two models serving one bundle are ambiguous.
        model_at(&root, "gamma", Some("beta.v1"));
        let error = package(&outside, &installed).unwrap_err();
        assert!(
            error.contains("several models") && error.contains("beta, gamma"),
            "{error}"
        );
        std::fs::remove_dir_all(outside).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A package that carries its serving profiles belongs to the model
    /// they name, under the nearest root above it or this binary, with no
    /// `profiles.json` beside the model; bytes that differ from the manifest
    /// are refused.
    #[test]
    fn a_package_resolves_to_the_model_its_profiles_name() {
        let root = temporary("carried");
        model_at(&root, "alpha", None);
        model_at(&root, "beta", Some("alpha.v1"));
        let outside = temporary("carried-outside");
        std::fs::create_dir_all(outside.join("components")).unwrap();
        let profiles = json!({"model": "alpha", "profiles": {}}).to_string();
        std::fs::write(outside.join("components/profiles.json"), &profiles).unwrap();
        let manifest = |digest: &str| {
            let manifest = json!({"id": "alpha.v1", "components": [{
                "role": "chop.serve-profiles.v1", "path": "components/profiles.json",
                "byte_length": profiles.len(), "sha256": digest,
            }]});
            std::fs::write(outside.join("manifest.json"), manifest.to_string()).unwrap();
        };
        manifest(&json::sha256_hex(profiles.as_bytes()));
        let installed = host(root.join("bin/chopinfer"));
        assert_eq!(
            package(&outside, &installed).unwrap(),
            root.join("models/alpha")
        );
        let nowhere = host(PathBuf::from("/nonexistent/bin/chopinfer"));
        let error = package(&outside, &nowhere).unwrap_err();
        assert!(error.contains("serves model alpha"), "{error}");
        manifest(&"0".repeat(64));
        let error = package(&outside, &installed).unwrap_err();
        assert!(error.contains("differs from the manifest"), "{error}");
        std::fs::remove_dir_all(outside).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    /// The carried profiles are read as serving reads a component: a link
    /// to the same bytes, a FIFO (without blocking) and a declaration past
    /// the deployment component bound are refused before anything parses.
    #[test]
    fn carried_profiles_are_a_bounded_regular_file() {
        let root = temporary("carried-reader");
        model_at(&root, "alpha", None);
        let installed = host(root.join("bin/chopinfer"));
        let profiles = json!({"model": "alpha", "profiles": {}}).to_string();
        let digest = json::sha256_hex(profiles.as_bytes());
        let carrying = |name: &str, length: u64| {
            let directory = temporary(&format!("carried-reader-{name}"));
            std::fs::create_dir_all(directory.join("components")).unwrap();
            let manifest = json!({"id": "alpha.v1", "components": [{
                "role": "chop.serve-profiles.v1", "path": "components/profiles.json",
                "byte_length": length, "sha256": digest,
            }]});
            std::fs::write(directory.join("manifest.json"), manifest.to_string()).unwrap();
            directory
        };
        let outside = root.join("outside.json");
        std::fs::write(&outside, &profiles).unwrap();
        let linked = carrying("linked", profiles.len() as u64);
        std::os::unix::fs::symlink(&outside, linked.join("components/profiles.json")).unwrap();
        let fifo = carrying("fifo", profiles.len() as u64);
        assert!(
            std::process::Command::new("mkfifo")
                .arg(fifo.join("components/profiles.json"))
                .status()
                .unwrap()
                .success()
        );
        let large = carrying("large", 256 * 1024 + 1);
        std::fs::write(large.join("components/profiles.json"), &profiles).unwrap();
        for (directory, expected) in [
            (&linked, "serving profiles"),
            (&fifo, "serving profiles"),
            (&large, "exceed 262144 bytes"),
        ] {
            let error = package(directory, &installed).unwrap_err();
            assert!(error.contains(expected), "{error}");
            std::fs::remove_dir_all(directory).unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_install_root_resolves_only_to_its_one_model() {
        let root = temporary("install");
        let nowhere = host(PathBuf::from("/nonexistent/bin/chopinfer"));
        std::fs::create_dir_all(root.join("models/empty")).unwrap();
        assert!(
            package(&root, &nowhere)
                .unwrap_err()
                .contains("declares no model")
        );
        model_at(&root, "alpha", None);
        assert_eq!(package(&root, &nowhere).unwrap(), root.join("models/alpha"));
        model_at(&root, "beta", None);
        let error = package(&root, &nowhere).unwrap_err();
        assert!(error.contains("several models (alpha, beta)"), "{error}");
        assert!(error.contains("pass --model"), "{error}");
        let error = package(&root.join("models"), &nowhere).unwrap_err();
        assert!(error.contains("--package takes"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }
}
