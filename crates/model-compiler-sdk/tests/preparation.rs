#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

//! CPU mechanics of the shared direct-package preparation: the helper
//! protocol and CLI, and pruning a model cache that older builds filled with
//! prepared weights.
use model_compiler_sdk::files::Result;
use model_compiler_sdk::preparation as prep;
use runtime_bundle::preparation::{
    DIRECT_VERSION, Envelope, PrepareDirectRequest, Request, Response, read_envelope, read_record,
    write_record,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Scratch {
    root: PathBuf,
}
impl Scratch {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "sdk-preparation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self { root }
    }
    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

/// A family that describes itself and probes, but emits no package.
struct Family;
impl prep::helper::Provider for Family {
    fn name(&self) -> &str {
        "chopinfer-model-sdk-test"
    }
    fn profiles(&self) -> Vec<String> {
        vec!["test-tp1".into()]
    }
    fn probe(&self, _path: &Path, _ranks: u32) -> Result<()> {
        Ok(())
    }
}

/// One small safetensors checkpoint.
fn checkpoint(root: &Path) {
    fs::create_dir_all(root).unwrap();
    let header = br#"{"used":{"dtype":"U8","shape":[4096],"data_offsets":[0,4096]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend(vec![1; 4096]);
    fs::write(root.join("model.safetensors"), bytes).unwrap();
}

#[test]
fn a_provider_without_a_profile_does_not_describe_the_capability() {
    use runtime_bundle::preparation::{DIRECT_PROFILE_CAPABILITY, PREPARE_DIRECT_CAPABILITY};
    let mut input = vec![];
    write_record(
        &Envelope {
            version: DIRECT_VERSION,
            request: Request::Describe,
        },
        &mut input,
    )
    .unwrap();
    let mut output = vec![];
    prep::helper::protocol(&Family, &mut input.as_slice(), &mut output).unwrap();
    let Response::Description { capabilities, .. } = read_record(&mut output.as_slice()).unwrap()
    else {
        panic!("expected a description")
    };
    // A provider rebuilt unchanged keeps the trait's default direct_profile,
    // so the host must not ask it for one.
    assert!(capabilities.iter().any(|c| c == PREPARE_DIRECT_CAPABILITY));
    assert!(!capabilities.iter().any(|c| c == DIRECT_PROFILE_CAPABILITY));
}

#[test]
fn a_family_without_direct_emission_refuses_a_direct_package() {
    let s = Scratch::new();
    checkpoint(&s.path("a"));
    let mut input = vec![];
    write_record(
        &Envelope {
            version: DIRECT_VERSION,
            request: Request::PrepareDirect(PrepareDirectRequest {
                path: s.path("a"),
                cache_root: s.path("direct-cache"),
                tensor_parallel_size: 1,
                activation_policy: None,
                modalities: Default::default(),
            }),
        },
        &mut input,
    )
    .unwrap();
    let mut output = vec![];
    prep::helper::protocol(&Family, &mut input.as_slice(), &mut output).unwrap();
    assert!(matches!(
        read_record::<Response>(&mut output.as_slice()).unwrap(),
        Response::Error { message, .. } if message.contains("does not load checkpoints directly")
    ));
    // Nothing is left in staging, and no package is published.
    let leftovers: Vec<_> = fs::read_dir(s.path("direct-cache"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.starts_with("t-") || name.starts_with("p-"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    let identity = prep::direct::checkpoint_identity(&s.path("a")).unwrap();
    assert_eq!(
        identity,
        prep::direct::checkpoint_identity(&s.path("a")).unwrap()
    );
    assert_eq!(identity.len(), 64);
}

/// The materializing request and its flags are gone: the protocol does not
/// parse a `prepare` request, and the CLI names the removed flags.
/// A family whose deployment policy is `policy`.
struct Policed(&'static str);
impl prep::helper::Provider for Policed {
    fn name(&self) -> &str {
        "chopinfer-model-sdk-test"
    }
    fn profiles(&self) -> Vec<String> {
        Vec::new()
    }
    fn probe(&self, _path: &Path, _ranks: u32) -> Result<()> {
        Ok(())
    }
    fn deployment_policy(&self) -> Option<&str> {
        Some(self.0)
    }
}

/// `deployment-policy` prints a family's checked policy, nothing for a
/// family without one, and refuses a policy that does not pass the checks
/// its packages' policy component does.
#[test]
fn a_helper_reports_only_a_valid_deployment_policy() {
    let command = || vec!["deployment-policy".to_owned()];
    prep::helper::run(&Family, command()).unwrap();
    let valid = r#"{"schema":"chopinfer-deployment-policy-v1","bundle_id":"test.v1","knob_prefixes":["TEST_"],"drafters":[]}"#;
    prep::helper::run(&Policed(valid), command()).unwrap();
    let invalid = r#"{"schema":"chopinfer-deployment-policy-v1","bundle_id":"test.v1","knob_prefixes":["TEST_"],"drafters":[],"booleans":["MAX_SEQUENCES"]}"#;
    let error = prep::helper::run(&Policed(invalid), command())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not start with a knob prefix"),
        "{error}"
    );
    let mut extra = command();
    extra.push("value".into());
    assert!(prep::helper::run(&Policed(valid), extra).is_err());
}

#[test]
fn materializing_preparation_is_refused() {
    let request = br#"{"version":2,"request":{"command":"prepare","source":{"kind":"directory","path":"m"},"cache_root":"c","tensor_parallel_size":4}}
"#;
    assert!(read_envelope(&mut &request[..]).is_err());
    let check = br#"{"version":2,"request":{"command":"check-activation","bundle":"b","manifest_sha256":"0","policy":"auto"}}
"#;
    assert!(read_envelope(&mut &check[..]).is_err());
    for flag in ["--snapshot", "--refresh", "--native-module"] {
        let mut arguments = vec!["prepare".to_owned(), "--model".into(), "m".into()];
        arguments.push(flag.into());
        if flag != "--refresh" {
            arguments.push("value".into());
        }
        let error = prep::helper::run(&Family, arguments)
            .unwrap_err()
            .to_string();
        assert!(error.contains(&format!("{flag} is gone")), "{error}");
    }
}

/// A cache an older build filled with prepared weights (`w-`), snapshots
/// (`s-`) and import records lists them, and pruning removes them.
#[test]
fn a_cache_of_prepared_weights_lists_and_prunes() {
    let s = Scratch::new();
    let root = s.path("cache");
    let cache = prep::cache::Cache::open(&root).unwrap();
    let digest = |byte: char| byte.to_string().repeat(64);
    let weights = format!("w-{}", digest('a'));
    let snapshot = format!("s-{}", digest('b'));
    let image = vec![0_u8; 4096];
    for (key, kind, dependencies) in [
        (&weights, "weights", vec![]),
        (&snapshot, "snapshot", vec![weights.clone()]),
    ] {
        fs::create_dir(root.join(key)).unwrap();
        fs::write(root.join(key).join("rank0.img"), &image).unwrap();
        cache
            .save(
                key,
                &prep::cache::Entry {
                    kind: kind.into(),
                    state: "complete".into(),
                    dependencies,
                    files: vec![prep::cache::Artifact {
                        name: "rank0.img".into(),
                        bytes: image.len() as u64,
                        sha256: runtime_weights::sha256_bytes(&image),
                        image: true,
                    }],
                    last_selected: 0,
                    payload: serde_json::Value::Null,
                },
            )
            .unwrap();
    }
    prep::cache::write(
        &root.join("imports"),
        &format!("i-{}.json", digest('c')),
        &serde_json::json!({"freshness": {}, "snapshot": snapshot}),
    )
    .unwrap();
    let listed = prep::list(&root).unwrap();
    let keys: Vec<_> = listed.entries.iter().map(|e| e.identity.clone()).collect();
    assert!(
        keys.contains(&weights) && keys.contains(&snapshot),
        "{keys:?}"
    );
    let dry = prep::prune(&root, false, &[]).unwrap();
    assert!(dry.removed.is_empty());
    assert!(root.join(&weights).exists());
    let pruned = prep::prune(&root, true, &[]).unwrap();
    assert!(pruned.removed.contains(&weights) && pruned.removed.contains(&snapshot));
    assert!(!root.join(&weights).exists() && !root.join(&snapshot).exists());
    assert_eq!(fs::read_dir(root.join("imports")).unwrap().count(), 0);
}
