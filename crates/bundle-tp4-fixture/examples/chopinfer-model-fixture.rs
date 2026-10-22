//! CPU conformance provider. Setup commands are used only by the process test.
use bundle_tp4_fixture::*;
use model_compiler_sdk::files::Result;
use model_compiler_sdk::native_contract::NativeModule;
use model_compiler_sdk::preparation as prep;
use model_compiler_sdk::source::SourceFiles;
use model_compiler_sdk::*;
use runtime_bundle::preparation::{NativeContracts, NativeProfile};
use runtime_bundle::{BundleOpaqueId, BundleRelativePathV1};
use runtime_weights::sha256_bytes;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

struct Family;

/// The fixture's serving profiles: autoregressive, and two drafters that
/// the program image serves under different names and scheduling.
const PROFILES: &str = r#"{"schema":"chopinfer-serve-profiles-v2","model":"fixture","served_model_name":"fixture",
"profiles":{
"tp4-ar":{"bundle_id":"fixture.tp4.tp4.v1","topology":"fixture-tp4","tensor_parallel":4,"devices_required":4,
 "devices":[0,1,2,3],"speculation":{"mode":"fixed","tokens":0,"draft_cost":"per-token","drafter":"none"},
 "max_sequences":1,"qualified_max_sequences":null,"qualification":"CPU fixture only","kv_cache_dtype":"auto",
 "prefix_cache":false,"max_batch_tokens":128,"prefill_chunk_tokens":128,"queue_depth":8,"gpu_memory_utilization":1.0,
 "status":"experimental","evidence":["tests/test_model_preparation.py"]},
"tp4-mtp":{"bundle_id":"fixture.tp4.tp4.v1","topology":"fixture-tp4","tensor_parallel":4,"devices_required":4,
 "devices":[0,1,2,3],"speculation":{"mode":"adaptive","tokens":3,"draft_cost":"per-token","drafter":"mtp"},
 "max_sequences":1,"qualified_max_sequences":null,"qualification":"CPU fixture only","kv_cache_dtype":"auto",
 "prefix_cache":false,"max_batch_tokens":128,"prefill_chunk_tokens":128,"queue_depth":8,"gpu_memory_utilization":1.0,
 "status":"experimental","evidence":["tests/test_model_preparation.py"]},
"tp4-dflash":{"bundle_id":"fixture.tp4.tp4.v1","topology":"fixture-tp4","tensor_parallel":4,"devices_required":4,
 "devices":[0,1,2,3],"speculation":{"mode":"adaptive","tokens":3,"draft_cost":"per-block","drafter":"dflash"},
 "max_sequences":1,"qualified_max_sequences":null,"qualification":"CPU fixture only","kv_cache_dtype":"auto",
 "prefix_cache":false,"max_batch_tokens":128,"prefill_chunk_tokens":128,"queue_depth":8,"gpu_memory_utilization":1.0,
 "prefill_budget":"tbt","decode_tbt_target_ms":150,"status":"experimental","evidence":["tests/test_model_preparation.py"]}
}}"#;
/// A test may name the frontend's model owner through this environment
/// variable, as a family whose emission reads its environment would.
const OWNER: &str = "CHOPINFER_FIXTURE_OWNER";
fn id(s: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(s).unwrap()
}
fn path(s: &str) -> BundleRelativePathV1 {
    BundleRelativePathV1::new(s).unwrap()
}
/// The fixture package for the checkpoint `identity`: the program image is
/// a planned component, which a direct load asks the weight plan for.
fn draft(native: StagedContentV1, identity: &str) -> Result<BundleDraftV1> {
    let base = tp4_fixture_bundle_draft_v1(Path::new("unused.so"))?;
    let mut frontend: serde_json::Value =
        serde_json::from_slice(&tp4_fixture_frontend_declaration_v1())?;
    frontend["models"][0]["id"] = format!("local-{identity}").into();
    frontend["models"][0]["owned_by"] = std::env::var(OWNER)
        .unwrap_or_else(|_| "local".into())
        .into();
    frontend["tokenizer"]["repository"] = "local".into();
    let components = base
        .components()
        .iter()
        .map(|c| {
            if c.role().as_str() == TP4_FIXTURE_FRONTEND_ROLE_V1 {
                ComponentDraftV1::new(
                    c.id().clone(),
                    c.role().clone(),
                    c.format().clone(),
                    c.path().clone(),
                    ComponentContentV1::Whole(StagedContentV1::bytes(
                        serde_json::to_vec(&frontend).unwrap(),
                    )),
                )
            } else {
                c.clone()
            }
        })
        .collect::<Vec<_>>();
    Ok(base
        .with_components(components)
        .with_native_module(NativeModuleDraftV1::new(
            path(TP4_FIXTURE_NATIVE_MODULE_PATH_V1),
            native,
        )))
}

impl Family {
    /// The module note's profile the fixture's native module declares.
    fn profile(&self, ranks: u32) -> Result<NativeProfile> {
        if ranks != 4 {
            return Err("CPU fixture needs TP4".into());
        }
        let declaration =
            draft(StagedContentV1::bytes(vec![]), &"0".repeat(64))?.native_declarations();
        Ok(NativeProfile {
            profile: "fixture-tp4".into(),
            tensor_parallel_size: 4,
            weight_contract: sha256_bytes(b"synthetic-tp4-images-v1"),
            package_contract: prep::cache::digest(&declaration)?,
        })
    }
    fn native_module(&self) -> Result<PathBuf> {
        Ok(std::env::current_exe()?
            .parent()
            .ok_or("no install root")?
            .join("libfixture_preparation.so"))
    }
    /// Admits a synthetic conformance checkpoint.
    fn open(&self, root: &Path, ranks: u32) -> Result<()> {
        self.profile(ranks)?;
        let files = SourceFiles::capture(root, &[("index.json", 4096)], &[], "index.json")?;
        let index: serde_json::Value = serde_json::from_slice(&files.inputs["index.json"])?;
        if index["fixture"] != true {
            return Err("not a synthetic conformance checkpoint".into());
        }
        Ok(())
    }
}
impl prep::helper::Provider for Family {
    fn name(&self) -> &str {
        "chopinfer-model-fixture"
    }
    fn profiles(&self) -> Vec<String> {
        vec!["fixture-tp4".into()]
    }
    fn probe(&self, path: &Path, ranks: u32) -> Result<()> {
        self.open(path, ranks)
    }
    /// A test may build this helper's behavior before the profile request:
    /// `CHOPINFER_FIXTURE_NO_DIRECT_PROFILE=1` withholds the capability.
    fn names_direct_profile(&self) -> bool {
        std::env::var_os("CHOPINFER_FIXTURE_NO_DIRECT_PROFILE").is_none()
    }
    /// A synthetic checkpoint names the drafters it serves with, in
    /// preference order, as a family knows them for a real one.
    fn direct_profile(
        &self,
        root: &Path,
        _request: &runtime_bundle::preparation::PrepareDirectRequest,
    ) -> Result<runtime_bundle::preparation::DirectProfile> {
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
        let drafters: Vec<&str> = match config["fixture_drafters"].as_array() {
            Some(names) => names.iter().filter_map(serde_json::Value::as_str).collect(),
            None => vec!["mtp", "none"],
        };
        prep::direct::default_profile(PROFILES, 4, &drafters)
    }
    /// The native module, and [`OWNER`], which the emission reads.
    fn direct_inputs(
        &self,
        _root: &Path,
        _request: &runtime_bundle::preparation::PrepareDirectRequest,
    ) -> Result<Option<Vec<u8>>> {
        Ok(Some(prep::direct::native_module_inputs(
            &self.native_module()?,
            &[OWNER],
        )?))
    }
    fn emit_direct(
        &self,
        root: &Path,
        request: &runtime_bundle::preparation::PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
    ) -> Result<prep::direct::DirectEmission> {
        self.emit_direct_with_inputs(root, request, checkpoint_identity, output, None)
    }
    fn emit_direct_with_inputs(
        &self,
        root: &Path,
        request: &runtime_bundle::preparation::PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
        inputs: Option<&[u8]>,
    ) -> Result<prep::direct::DirectEmission> {
        self.open(root, request.tensor_parallel_size)?;
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
        let mut profile = prep::helper::Provider::direct_profile(self, root, request)?;
        // A test checkpoint may make emission carry another profile under
        // the same name, as a family whose two selection paths drifted would.
        if config["fixture_profile_drift"] == true {
            profile.profile["max_sequences"] = 2.into();
        }
        let drafter = profile.profile["speculation"]["drafter"]
            .as_str()
            .ok_or("fixture profile has no drafter")?
            .to_owned();
        NativeModule::inspect(&self.native_module()?)?;
        // A test checkpoint may name a marker: the first emission without it
        // replaces the installed module after its inputs were taken, as an
        // installer racing the start would.
        if let Some(marker) = config["fixture_replace_module_once"].as_str()
            && !Path::new(marker).exists()
        {
            fs::write(marker, b"")?;
            let module = self.native_module()?;
            let mut replaced = fs::read(&module)?;
            replaced.extend_from_slice(b"fixture-module-replaced");
            fs::write(&module, replaced)?;
        }
        let native = prep::direct::native_module_staged(&self.native_module()?, inputs)?;
        let mut draft = draft(native, checkpoint_identity)?;
        // Opt in through config.json so legacy fixture packages keep their
        // manifest and create bytes. The option sizes the token input buffer.
        if let Some(default) = config.get("scratch_multiplier") {
            let default = default
                .as_i64()
                .ok_or("scratch_multiplier must be an integer")?;
            draft = draft.with_model_options(runtime_bundle::model_options::ModelOptionsV1 {
                schema: runtime_bundle::model_options::CAPABILITY.into(),
                options: vec![runtime_bundle::model_options::ModelOptionV1 {
                    name: "scratch_multiplier".into(),
                    definition: runtime_bundle::model_options::OptionDefinitionV1::Integer {
                        default,
                        minimum: 1,
                        maximum: 4,
                    },
                    default_is_qualified: false,
                }],
            })?;
        }
        if drafter != "none" {
            // The program image drafts; its kind is the profile's drafter.
            draft = draft.with_speculation(
                runtime_bundle::BundleSpeculationV1::new(
                    3,
                    true,
                    4,
                    vec![2, 4],
                    Some(id(TP4_FIXTURE_PROGRAM_COMPONENT_V1)),
                )?
                .with_drafter_kind(Some(id(&drafter)))?,
            );
        }
        let package = emit_bundle_directory_v1(&draft, output)?;
        // A test checkpoint may make emission slow, to see it counted.
        if let Some(millis) = config["fixture_emit_delay_ms"].as_u64() {
            std::thread::sleep(std::time::Duration::from_millis(millis));
        }
        // A test checkpoint may change a metadata file after emission, as a
        // writer racing the start would.
        if let Some(name) = config["fixture_change_after_emission"].as_str() {
            fs::write(root.join(name), b"{\"changed\":true}")?;
        }
        Ok(prep::direct::DirectEmission { package, profile })
    }
}
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|a| a == "fixture-note") {
        let document = serde_json::to_vec(&NativeContracts {
            version: 1,
            profiles: vec![Family.profile(4)?],
            implementation_contract: None,
        })?;
        let note = model_compiler_sdk::native_contract::encode_native_note(&document);
        std::io::stdout().write_all(&note)?;
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "fixture-source") {
        let root = Path::new(args.get(1).ok_or("fixture-source needs a directory")?);
        write_tp4_fixture_checkpoint_v1(root)?;
        let mut config = serde_json::json!({"architectures":["Tp4FixtureForCausalLM"]});
        if let Some(drafters) = args.get(2) {
            config["fixture_drafters"] = drafters.split(',').collect::<Vec<_>>().into();
        }
        fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
        return Ok(());
    }
    prep::helper::run(&Family, args)
}
