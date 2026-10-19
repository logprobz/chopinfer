use super::*;
use model_compiler_sdk::{EmittedBundleV1, StagedContentV1, package_draft};
use runtime_bundle::{BundleKvDtypeV1, BundleStateLifetimeV1};

#[path = "../../../../../../crates/model-compiler-sdk/test-support/native.rs"]
mod native;

struct Package {
    emitted: EmittedBundleV1,
    root: std::path::PathBuf,
}

impl Package {
    fn new() -> Self {
        // The emitted fixture validates the package and its native symbol.
        // Execution below uses this test binary's real bundle ABI table.
        let module = native::compile_native_fixture(
            "deepseek41-serving-package",
            "deepseek41_serving_fixture",
            "libbundle_deepseek41.so",
            r#"
#[unsafe(no_mangle)]
pub extern "C" fn chop_bundle_get_api_v1() -> *const core::ffi::c_void {
    core::ptr::null()
}
"#,
        );
        let root = module.parent().unwrap().to_path_buf();
        let components = vec![
            package_draft::tokenizer_definition(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
            package_draft::tokenizer_configuration(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
        ];
        let draft = model_deepseek41::package::draft(
            &FLASH,
            FLASH.runtime_defaults(),
            &[(4, 1 << 30)],
            components,
            StagedContentV1::file(module),
        )
        .unwrap();
        let emitted =
            model_compiler_sdk::emit_bundle_directory_v1(&draft, &root.join("package")).unwrap();
        let profiles: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                emitted
                    .root()
                    .join(runtime_bundle::deployment::SERVE_PROFILES_PATH_V1),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            profiles,
            serde_json::from_str::<serde_json::Value>(model_deepseek41::package::PROFILES).unwrap()
        );
        Self { emitted, root }
    }

    fn record(&self, shape: Shape, dtype: BundleKvDtypeV1) -> Vec<u8> {
        self.record_with_options(shape, dtype, &Default::default())
    }

    fn record_with_options(
        &self,
        shape: Shape,
        dtype: BundleKvDtypeV1,
        overrides: &std::collections::BTreeMap<String, String>,
    ) -> Vec<u8> {
        let manifest = self
            .emitted
            .manifest()
            .for_topology(&shape.topology())
            .unwrap();
        let kv = manifest
            .kv_formats()
            .iter()
            .find(|format| format.dtype() == dtype)
            .unwrap()
            .layout();
        let state = manifest.state_layout_for_v1(kv);
        assert_eq!(kv.state_lifetime(), BundleStateLifetimeV1::Transaction);
        let mut roles = manifest
            .components()
            .iter()
            .map(|component| component.role().as_str().as_bytes().to_vec())
            .chain(
                manifest
                    .planned_components()
                    .iter()
                    .map(|component| component.role().as_str().as_bytes().to_vec()),
            )
            .collect::<Vec<_>>();
        roles.extend((0..shape.ranks()).map(|rank| index_role(rank).into_bytes()));
        let mut record = bundle_runtime::config::test_support::encode(
            shape.topology().as_bytes(),
            &(0..shape.ranks())
                .map(|rank| rank as i32)
                .collect::<Vec<_>>(),
            &bundle_runtime::config::KvLayout {
                layout_id: kv.id().as_str().as_bytes().to_vec(),
                page_bytes: kv.page_bytes(),
                alignment: kv.alignment_bytes(),
                page_tokens: u64::from(kv.page_tokens()),
                ranks: u64::from(shape.ranks()),
                lifetime: 0,
            },
            &bundle_runtime::config::StateLayout {
                state_id: state.id().as_str().as_bytes().to_vec(),
                bytes: manifest.state_bytes_for_v1(kv, Some(4)).unwrap(),
                alignment: state.alignment_bytes(),
            },
            &roles,
            Some(4),
        );
        record.extend_from_slice(&kv.maximum_sequence_tokens().to_le_bytes());
        let options = manifest.resolve_model_options(overrides).unwrap();
        let options = serde_json::to_vec(&options).unwrap();
        record.extend_from_slice(runtime_bundle_abi::MODEL_OPTIONS_TAG_V1);
        record.extend_from_slice(&(options.len() as u32).to_le_bytes());
        record.extend_from_slice(&options);
        record
    }
}

impl Drop for Package {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn package_host(shape: Shape, layout: KvLayout) -> Rc<Host> {
    let host = FakeHost::new(shape, layout);
    // Installed rank images contain drafter weights even when drafting is off.
    host.drafter.set(true);
    host
}

fn create_from_record(host: &Rc<Host>, record: &[u8]) -> Result<BundleInstanceV1, BundleAbiError> {
    api().create(
        HostServicesTableV1::new(Rc::new(host::MappedHost::new(Rc::clone(host)))),
        record,
    )
}

fn ar_journal(host: &Rc<Host>, record: &[u8]) -> Vec<String> {
    let mut instance = create_from_record(host, record)
        .unwrap_or_else(|error| panic!("package create: {error:?} {:?}", host.logs.borrow()));
    execute(&mut instance, host, &prompt(7, 0, 0, 3, true));
    instance.commit().unwrap();
    assert!(crate::instance::dspark_context(instance.handle(), 7).is_none());
    for (position, token) in [(3, 7), (4, 8)] {
        let sink = execute(&mut instance, host, &decode(&[(7, 0, position, token)]));
        assert!(sink.drafts.is_empty());
        instance.commit().unwrap();
    }
    let pages = (0..instance.describe_prefix_state().unwrap())
        .map(|page| 40 + page)
        .collect::<Vec<_>>();
    instance.capture_prefix_state(7, 5, &pages).unwrap();
    instance.retire_request(7).unwrap();
    instance
        .restore_prefix_state_in_slot(8, &[0, 1, 2, 7, 8], &pages, 2)
        .unwrap();
    execute(&mut instance, host, &decode(&[(8, 2, 5, 9)]));
    instance.commit().unwrap();
    instance.retire_request(8).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    assert!(
        !host
            .logs
            .borrow()
            .iter()
            .any(|line| line.contains(crate::dspark::calibration::SCHEMA))
    );
    let journal = host.journal.borrow().clone();
    assert!(journal.iter().any(|line| line.starts_with("capture ")));
    assert!(
        journal
            .iter()
            .filter(|line| line.starts_with("replay "))
            .count()
            >= 2
    );
    journal
}

#[test]
fn emitted_package_with_speculation_off_keeps_the_full_ar_journal() {
    let package = Package::new();
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let layout = KV_LAYOUTS[0];
    let legacy = package_host(shape, layout);
    let legacy_record = crate::config::tests::record(&shape, layout, 4, false);
    let expected = ar_journal(&legacy, &legacy_record);
    let installed = package_host(shape, layout);
    let record = package.record_with_options(
        shape,
        BundleKvDtypeV1::MixedPlanes,
        &std::collections::BTreeMap::from([
            ("prefill_chunk_tokens".into(), "0".into()),
            ("prefill_upper_skip".into(), "false".into()),
        ]),
    );
    let actual = ar_journal(&installed, &record);
    assert_eq!(installed.state_bytes.get(), legacy.state_bytes.get());
    assert_eq!(actual, expected);
}

#[test]
fn emitted_package_defaults_keep_the_default_profile_ar_journal() {
    let package = Package::new();
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let layout = KV_LAYOUTS[0];
    let profiles: serde_json::Value =
        serde_json::from_str(model_deepseek41::package::PROFILES).unwrap();
    let overrides =
        serde_json::from_value(profiles["profiles"]["tp4-ar"]["model_options"].clone()).unwrap();
    let explicit = package.record_with_options(shape, BundleKvDtypeV1::MixedPlanes, &overrides);
    let defaults = package.record(shape, BundleKvDtypeV1::MixedPlanes);
    assert_eq!(defaults, explicit);
    let profiled = package_host(shape, layout);
    let installed = package_host(shape, layout);
    assert_eq!(
        ar_journal(&installed, &defaults),
        ar_journal(&profiled, &explicit)
    );
    assert_eq!(installed.state_bytes.get(), profiled.state_bytes.get());
}

#[test]
fn emitted_package_refuses_bf16_speculation_at_startup() {
    let package = Package::new();
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let bf16 = package.record(shape, BundleKvDtypeV1::Bf16);
    let ar = package_host(shape, KV_LAYOUTS[1]);
    ar_journal(&ar, &bf16);

    let native = package_host(shape, KV_LAYOUTS[0]);
    native.speculation.set(Some((5, 32)));
    let record = package.record(shape, BundleKvDtypeV1::MixedPlanes);
    let mut instance = create_from_record(&native, &record).unwrap();
    execute(&mut instance, &native, &prompt(7, 0, 0, 3, true));
    instance.commit().unwrap();
    assert!(crate::instance::dspark_context(instance.handle(), 7).is_some());
    let sink = execute(&mut instance, &native, &super::verify(7, 0, 3, 5));
    assert_eq!(sink.draft_tokens(), vec![(7, vec![0; 5])]);
    instance.commit_accepted(&[kept(7, 6)]).unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(native.acquired.get(), native.released.get());

    for drafts in 1..=5 {
        let host = package_host(shape, KV_LAYOUTS[1]);
        host.speculation.set(Some((drafts, 32)));
        let error = match create_from_record(&host, &bf16) {
            Err(error) => error,
            Ok(mut instance) => {
                instance.destroy().unwrap();
                panic!("BF16 KV admitted {drafts} speculative tokens");
            }
        };
        assert_eq!(error, BundleAbiError::Status(BundleStatusV1::Unsupported));
        assert!(host.logs.borrow().iter().any(|line| {
            line.contains(
                "BF16 KV supports autoregressive serving only; set --speculative-tokens 0",
            )
        }));
        assert_eq!(host.acquired.get(), 0, "startup refusal acquired resources");
        assert_eq!(host.state_bytes.get(), 0);
    }
}
