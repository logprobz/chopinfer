//! The files `eval files` lists for each shipped declaration and checkpoint
//! variant, staged at the same relative paths as tools/release/build.sh
//! ships them, are all that `eval verify-references` reads: the staged copy
//! verifies as the checkout does.

use std::path::{Path, PathBuf};

use runtime_eval::declaration::{self, Loaded};
use runtime_eval::references::{self, ReferenceError};
use runtime_eval::{Exit, Tool, cli, files};

fn verify(model: &Path, variant: Option<&str>) -> Exit {
    let tool = Tool {
        name: "chopinfer eval".into(),
        version: "test".into(),
        revision: None,
    };
    let mut arguments = vec!["verify-references", "--model", model.to_str().unwrap()];
    arguments.extend(
        variant
            .map(|variant| ["--variant", variant])
            .into_iter()
            .flatten(),
    );
    let arguments: Vec<String> = arguments.into_iter().map(String::from).collect();
    cli::run(&tool, &arguments)
}

/// Each pinned reference's verdict: verified, or the kind of refusal.
fn verdicts(loaded: &Loaded) -> Vec<(String, &'static str)> {
    let status = |result: Result<(), ReferenceError>| match result {
        Ok(()) => "verified",
        Err(ReferenceError::Missing(_)) => "missing",
        Err(ReferenceError::Unverified(_)) => "unverified",
        Err(ReferenceError::Stale(..)) => "stale",
    };
    let lock = references::checkpoint_lock(loaded).map(|_| ());
    std::iter::once(("checkpoint lock".to_owned(), status(lock)))
        .chain(loaded.declaration.references.iter().map(|pin| {
            let verified = references::pinned(loaded, pin, None).map(|_| ());
            (pin.name.clone(), status(verified))
        }))
        .collect()
}

/// Stages what `eval files` lists for the declaration of `model` (or its
/// `variant`) under `stage` and verifies the staged copy as the checkout.
fn stage_and_verify(model: &Path, variant: Option<&str>, stage: &Path) {
    let loaded = declaration::load_variant(model, variant).unwrap();
    let listed = files::files(&loaded).unwrap();
    let name = &loaded.declaration.model;
    let label = format!("{name} {variant:?}");
    // The file the declaration was read from: a configuration expands from
    // qual.json, a variant file is read as itself.
    let file = loaded.path.file_name().unwrap().to_str().unwrap();
    for required in [
        format!("models/{name}/{file}"),
        declaration::MODULES.into(),
        loaded.declaration.checkpoint.lock.clone(),
    ] {
        assert!(listed.contains(&required), "{label}: {required}");
    }
    for pin in &loaded.declaration.references {
        if let declaration::Input::Path(path) = &pin.path {
            assert!(listed.contains(path), "{label}: {path}");
        }
    }
    for file in &listed {
        let target = stage.join(file);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::copy(loaded.root.join(file), target).unwrap();
    }
    let staged = declaration::load_variant(&stage.join("models").join(name), variant).unwrap();
    assert_eq!(verdicts(&staged), verdicts(&loaded), "{label}");
    let exit = verify(model, variant);
    assert_eq!(
        verify(&stage.join("models").join(name), variant),
        exit,
        "{label}"
    );
    // Only a workload kept outside the repository, read where its
    // variable is set, leaves the checkout itself unverified.
    let outside = loaded.declaration.references.iter().any(|pin| {
        matches!(&pin.workload, Some(declaration::Input::Environment { env, .. })
                 if std::env::var_os(env).is_none())
    });
    assert_eq!(
        exit,
        if outside {
            Exit::Insufficient
        } else {
            Exit::Pass
        },
        "{label}"
    );
}

#[test]
fn a_staged_install_verifies_as_the_checkout_does() {
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let stage = std::env::temp_dir().join(format!("runtime-eval-install-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&stage);
    let mut models: Vec<PathBuf> = std::fs::read_dir(checkout.join("models"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|model| model.join(declaration::FILE_NAME).is_file())
        .collect();
    models.sort();
    assert!(!models.is_empty());
    let mut variants = 0;
    for model in models {
        let named = declaration::variants(&model).unwrap();
        variants += named.len();
        for variant in std::iter::once(None).chain(named.iter().map(|name| Some(name.as_str()))) {
            stage_and_verify(&model, variant, &stage);
        }
    }
    assert!(variants > 0, "no shipped model declares a variant");
    std::fs::remove_dir_all(stage).unwrap();
}
