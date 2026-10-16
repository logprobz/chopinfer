//! Private CPU lifecycle capture. The model owns actions, labels and state probes.
//! This is not a public SDK format or a numerical simulator.
use super::*;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

const EXTENSION: &str = "qwen38.lifecycle.v1";
const INPUT_LIMIT: usize = 16 << 20;
const DATA_LIMIT: usize = 64 << 20;
const ENTRY: &str = "fake_host_tests::lifecycle::replay_recording";
const PREFIX: &str = "CHOPINFER_QWEN38_LIFECYCLE_";
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Default)]
pub(super) struct Recorder {
    action: usize,
    bytes: usize,
    observations: Vec<Value>,
}

impl Recorder {
    fn push(&mut self, mut event: Value) {
        event["ordinal"] = json!(self.observations.len());
        event["action"] = json!(self.action);
        self.bytes += serde_json::to_vec(&event).unwrap().len() + 1;
        assert!(self.bytes <= DATA_LIMIT, "lifecycle data budget exceeded");
        self.observations.push(event);
    }
}

impl bundle_runtime::fake_host::JournalObserver for Recorder {
    fn journal(&mut self, line: usize, entry: &str) {
        self.bytes += entry.len() + 1;
        self.push(json!({"kind":"journal", "line":line,
            "service":entry.split(' ').next().unwrap()}));
    }
}

fn observe(host: &Package, event: Value) {
    if let Some(recorder) = host.observer.borrow_mut().as_mut() {
        recorder.push(event);
    }
}

pub(super) fn copy_result(
    host: &Package,
    destination: DeviceBufferSpanV1,
    source: DeviceBufferSpanV1,
    status: &str,
) {
    if host.observer.borrow().is_none() {
        return;
    }
    let span = |s: DeviceBufferSpanV1| {
        json!({"buffer":s.buffer.slot(),
        "generation":s.buffer.generation(), "offset":s.offset, "length":s.length})
    };
    observe(
        host,
        json!({"kind":"copy_result", "line":host.journal.borrow().len()-1,
        "rank":destination.buffer.slot().checked_sub(STATE_SLOT), "source":span(source),
        "destination":span(destination), "status":status}),
    );
}

pub(super) fn actions() -> Vec<Value> {
    let long = long_prompt();
    let execute = |id, name, batch: BatchRequestV1| {
        json!({"id":id, "kind":"execute",
        "name":name, "wire":batch.encode().unwrap()})
    };
    vec![
        json!({"id":0,"kind":"create"}),
        execute(
            1,
            "prefill",
            step(&[(20, 0, 0, &PROMPT), (21, 1, 0, &long)]),
        ),
        json!({"id":2,"kind":"commit"}),
        execute(3, "decode", step(&[(20, 0, 3, &[4]), (21, 1, 2050, &[5])])),
        json!({"id":4,"kind":"commit"}),
        execute(
            5,
            "mixed",
            step(&[(22, 2, 0, &PROMPT), (20, 0, 4, &[6]), (21, 1, 2051, &[7])]),
        ),
        json!({"id":6,"kind":"cancel"}),
        json!({"id":7,"kind":"retire","request":22}),
        json!({"id":8,"kind":"destroy"}),
    ]
}

fn bytes(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::from_value(value.clone())?)
}

fn rows(wire: &[u8]) -> Result<Value> {
    let request = BatchRequestV1::decode(wire).map_err(|e| format!("batch: {e:?}"))?;
    if request.encode().map_err(|e| format!("batch: {e:?}"))? != wire
        || !request.media().is_empty()
        || request.items().is_empty()
    {
        return Err("noncanonical or unsupported batch".into());
    }
    let slots = request.state_slots().ok_or("missing state slots")?;
    let mut result = Vec::new();
    for ((item, append), slot) in request.items().iter().zip(request.appends()).zip(slots) {
        if item.is_verify()
            || item.bundle_drafts()
            || item.prompt_rows()
            || !item.position_axes().is_empty()
            || append.copy().is_some()
            || !item.expects_logits()
            || item.input_tokens().len() != item.phase_tokens() as usize
        {
            return Err("unsupported batch item".into());
        }
        result.push(json!({"request":item.request_id(),"slot":slot,
            "phase":if item.is_decode() {"decode"} else {"prefill"},
            "context":append.previous_tokens(),"tokens":item.phase_tokens(),
            "end":append.new_tokens(),"additional":append.additional_tokens(),"pages":append.page_table()}));
    }
    Ok(json!(result))
}

fn configuration() -> Vec<u8> {
    configuration_served(Shape::TP4, FakeHost::default().layout(), &[], SERVED)
}

fn producer() -> Result<Value> {
    Ok(json!({"binary_sha256":hash(&std::env::current_exe()?)?,
        "kernel_lock_sha256":runtime_weights::sha256_bytes(include_bytes!("../../kernels.lock"))}))
}

fn settings() -> Result<Value> {
    use crate::instance::hc;
    // Wide comm cannot affect this fixture. Refuse if its capacity grows into wide rows.
    if crate::scratch::Served::new(served(SERVED)).rows > DECODE_ENTRY_ROWS {
        return Err("unsupported lifecycle wide decode rows".into());
    }
    let wide =
        crate::launch::wide_comm_enabled(std::env::var(crate::launch::WIDE_COMM).ok().as_deref());
    // Some(None) means an override selected the default, serialized as the legacy null.
    let switches = [
        (crate::launch::WIDE_COMM, Some((!wide).then_some("0"))),
        (
            hc::SWITCH,
            hc::FOR_TEST
                .with(Cell::get)
                .map(|enabled| (!enabled).then_some("0")),
        ),
        (
            "CHOPINFER_QWEN38_HC_PADDED_MIX",
            hc::PADDED_FOR_TEST
                .with(Cell::get)
                .map(|enabled| enabled.then_some("1")),
        ),
        (
            "CHOPINFER_QWEN38_ROW_OWNED_LOGITS",
            crate::instance::ROW_OWNERS_FOR_TEST
                .with(Cell::get)
                .map(|enabled| (!enabled).then_some("0")),
        ),
        ("CHOPINFER_QWEN38_DUMP_DIR", None),
    ];
    if std::env::var_os(runtime_bundle_sdk::taps::TAPS_ENV.dir).is_some() {
        return Err("unsupported lifecycle environment switch CHOPINFER_TAPS_DIR".into());
    }
    let mut environment = serde_json::Map::new();
    for (key, forced) in switches {
        if forced.is_none() && std::env::var_os(key).is_some() {
            return Err(format!("unsupported lifecycle environment switch {key}").into());
        }
        environment.insert(key.into(), json!(forced.flatten()));
    }
    Ok(
        json!({"environment":environment,"effective_hc":hc::enabled(),"topology":"tp4",
        "ranks":4,"fold":false,"kv_layout":std::str::from_utf8(KV_LAYOUT_BF16)?,
        "state_layout":std::str::from_utf8(RING_STATE_ID)?,"served_capacity":SERVED,"kv_pages":KV_PAGES,
        "host_capabilities":"qwen38-default-fake-host","checkpoint":"not-applicable-synthetic",
        "prepared_weights":"not-applicable-synthetic","scope":"cpu-structural-state"}),
    )
}

#[test]
fn settings_follow_scoped_selections_and_hc_execution() {
    with_default_journal_settings(|| {
        let defaults = settings().unwrap();
        for key in [
            crate::instance::hc::SWITCH,
            "CHOPINFER_QWEN38_HC_PADDED_MIX",
            "CHOPINFER_QWEN38_ROW_OWNED_LOGITS",
        ] {
            assert_eq!(defaults["environment"][key], Value::Null);
        }
        assert_eq!(defaults["effective_hc"], true);
        crate::instance::with_hc_padded_mix(true, || {
            assert_eq!(
                settings().unwrap()["environment"]["CHOPINFER_QWEN38_HC_PADDED_MIX"],
                "1"
            );
        });
        crate::instance::with_row_owners(false, || {
            assert_eq!(
                settings().unwrap()["environment"]["CHOPINFER_QWEN38_ROW_OWNED_LOGITS"],
                "0"
            );
        });
        crate::instance::with_hc_precompute(false, || {
            let actual = settings().unwrap();
            assert_eq!(actual["environment"][crate::instance::hc::SWITCH], "0");
            assert_eq!(actual["effective_hc"], false);
            let lines = journal_steps(&Rc::new(FakeHost::default()));
            assert_eq!(journal_digest(&lines), EXPECTED_HC_OFF);
        });
        assert_eq!(settings().unwrap(), defaults);
    });
}

fn input() -> Result<Value> {
    Ok(json!({"extension":EXTENSION,"backend":"qwen38-fake-host",
        "scenario":"mixed-cancellation","producer":producer()?,"settings":settings()?,
        "configuration":configuration(),"actions":actions()}))
}

fn validate_input(value: &Value) -> Result<Vec<Value>> {
    if value["extension"] != EXTENSION
        || value["backend"] != "qwen38-fake-host"
        || value["scenario"] != "mixed-cancellation"
        || value["producer"] != producer()?
        || value["settings"] != settings()?
        || bytes(&value["configuration"])? != configuration()
    {
        return Err("incompatible lifecycle input, producer or configuration".into());
    }
    let actions = value["actions"].as_array().ok_or("missing actions")?;
    if actions.len() > 64 {
        return Err("lifecycle action budget exceeded".into());
    }
    let kinds = [
        "create", "execute", "commit", "execute", "commit", "execute", "cancel", "retire",
        "destroy",
    ];
    if actions.len() != kinds.len() {
        return Err("incomplete actions".into());
    }
    for (id, (action, kind)) in actions.iter().zip(kinds).enumerate() {
        if action["id"] != id || action["kind"] != kind {
            return Err("unsupported or reordered action".into());
        }
        let allowed: &[&str] = match kind {
            "execute" => &["id", "kind", "name", "wire"],
            "retire" => &["id", "kind", "request"],
            _ => &["id", "kind"],
        };
        if action
            .as_object()
            .ok_or("invalid action")?
            .keys()
            .any(|key| !allowed.contains(&key.as_str()))
        {
            return Err("unsupported action field".into());
        }
        if kind == "execute" {
            let expected: &[(u64, u64, u64, u32)] = match id {
                1 => &[(20, 0, 0, 3), (21, 1, 0, 2050)],
                3 => &[(20, 0, 3, 1), (21, 1, 2050, 1)],
                5 => &[(22, 2, 0, 3), (20, 0, 4, 1), (21, 1, 2051, 1)],
                _ => unreachable!(),
            };
            if action["name"] != ["prefill", "decode", "mixed"][(id - 1) / 2] {
                return Err("unsupported step name".into());
            }
            let actual = rows(&bytes(&action["wire"])?)?;
            if actual.as_array().unwrap().len() != expected.len() {
                return Err("changed live membership".into());
            }
            for (row, &(request, slot, context, count)) in
                actual.as_array().unwrap().iter().zip(expected)
            {
                let end = context + u64::from(count);
                let pages: Vec<_> = (0..end.div_ceil(PAGE_TOKENS))
                    .map(|p| slot * 64 + p)
                    .collect();
                if *row
                    != json!({"request":request,"slot":slot,"context":context,"tokens":count,
                    "end":end,"additional":count,"pages":pages,"phase":if count==1 {"decode"} else {"prefill"}})
                {
                    return Err("changed context, phase or slot/page mapping".into());
                }
            }
        } else if kind == "retire" && action["request"] != 22 {
            return Err("unsupported retirement".into());
        }
    }
    Ok(actions.clone())
}

fn status(result: &std::result::Result<(), BundleAbiError>) -> String {
    match result {
        Ok(()) => "Success".into(),
        Err(BundleAbiError::Status(BundleStatusV1::HostError)) => "HostError".into(),
        Err(e) => format!("{e:?}"),
    }
}

/// Both the legacy golden and recorded replay execute this same action sequence.
pub(super) fn execute(
    host: &Rc<FakeHost>,
    config: &[u8],
    actions: &[Value],
    fault: bool,
) -> Result<Value> {
    let mut instance: Option<BundleInstanceV1> = None;
    for (id, action) in actions.iter().enumerate() {
        if let Some(recorder) = host.observer.borrow_mut().as_mut() {
            recorder.action = id;
        }
        observe(host, json!({"kind":"begin"}));
        let kind = action["kind"].as_str().ok_or("missing action kind")?;
        let result = match kind {
            "create" => api()
                .create(HostServicesTableV1::new(Rc::clone(host)), config)
                .map(|i| instance = Some(i)),
            "execute" => {
                let name = action["name"].as_str().ok_or("missing step name")?;
                let wire = bytes(&action["wire"])?;
                let mut sink = ViewSink::default();
                host.note(format!("step {name}"));
                let result = instance.as_mut().unwrap().execute(&wire, &mut sink);
                for view in &sink.0 {
                    host.note(format!("{name} view {:?}", view.raw()));
                }
                result
            }
            "commit" => instance.as_mut().unwrap().commit(),
            "cancel" => {
                if fault {
                    host.fail_restore_rank.set(Some(1));
                }
                instance.as_mut().unwrap().cancel()
            }
            "retire" => instance
                .as_mut()
                .unwrap()
                .retire_request(action["request"].as_u64().unwrap()),
            "destroy" => instance.as_mut().unwrap().destroy(),
            _ => return Err("unsupported lifecycle action".into()),
        };
        let handle = instance.as_ref().map(BundleInstanceV1::handle);
        observe(
            host,
            json!({"kind":"result","status":status(&result),
            "probe":"qwen38.slot-transaction","pending":handle.and_then(crate::instance::pending_step),
            "owners":handle.map(|h| (0..3).map(|slot| crate::instance::owner(h,slot)).collect::<Vec<_>>()),
            "poisoned":instance.as_ref().is_some_and(BundleInstanceV1::is_poisoned),
            "captures":host.captures.get(),"replays":host.replays.get(),
            "acquired":host.acquired.get(),"released":host.released.get()}),
        );
        if result.is_err() {
            if !fault || kind != "cancel" || status(&result) != "HostError" {
                result?;
            }
            let teardown = instance.as_mut().unwrap().destroy();
            return Ok(
                json!({"completion":"closed-failure","completed_actions":id+1,
                "failed_action":id,"teardown":{"status":status(&teardown),
                "acquired":host.acquired.get(),"released":host.released.get()}}),
            );
        }
    }
    Ok(
        json!({"completion":"full","completed_actions":actions.len(),"failed_action":null,
        "teardown":{"status":"Success","acquired":host.acquired.get(),"released":host.released.get()}}),
    )
}

fn external(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path.components().any(|c| matches!(c, Component::ParentDir))
        || path.ancestors().any(|p| {
            p.join(".git").exists()
                || p.symlink_metadata()
                    .is_ok_and(|m| m.file_type().is_symlink())
        })
    {
        return Err("lifecycle path must be external without symlink traversal".into());
    }
    Ok(())
}

fn read(path: &Path) -> Result<Vec<u8>> {
    external(path)?;
    let file = runtime_weights::open_nofollow(path)?;
    if file.metadata()?.len() > INPUT_LIMIT as u64 {
        return Err("lifecycle metadata budget exceeded".into());
    }
    let mut data = Vec::new();
    file.take(INPUT_LIMIT as u64 + 1).read_to_end(&mut data)?;
    if data.len() > INPUT_LIMIT {
        return Err("lifecycle metadata budget exceeded".into());
    }
    Ok(data)
}

fn hash(path: &Path) -> Result<String> {
    Ok(runtime_weights::sha256_reader(
        &mut runtime_weights::open_nofollow(path)?,
    )?)
}

fn write(path: &Path, data: &[u8]) -> Result<Value> {
    external(path)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = runtime_weights::open_regular_nofollow(path, &mut options)?;
    file.write_all(data)?;
    file.sync_all()?;
    Ok(json!({"bytes":data.len(),"sha256":runtime_weights::sha256_bytes(data)}))
}

fn env(key: &str) -> Option<String> {
    std::env::var(format!("{PREFIX}{key}")).ok()
}
fn required(key: &str) -> Result<String> {
    env(key).ok_or_else(|| format!("missing {PREFIX}{key}").into())
}

fn binding(key: &str) -> Result<Value> {
    let path = PathBuf::from(required(key)?);
    let raw = read(&path.join("diagnostic.json"))?;
    let digest = runtime_weights::sha256_bytes(&raw);
    if digest != required(&format!("{key}_SHA256"))? {
        return Err("stale lifecycle binding".into());
    }
    Ok(json!({"path":path,"sha256":digest}))
}

fn replay() -> Result<()> {
    let input_path = PathBuf::from(required("INPUT")?);
    let raw = read(&input_path)?;
    let digest = runtime_weights::sha256_bytes(&raw);
    if digest != required("INPUT_SHA256")?
        || producer()?["binary_sha256"] != required("BINARY_SHA256")?
    {
        return Err("stale replay input or executable".into());
    }
    let input: Value = serde_json::from_slice(&raw)?;
    let actions = validate_input(&input)?;
    let arm = required("ARM")?;
    if !["baseline", "control", "candidate"].contains(&arm.as_str()) {
        return Err("unsupported replay arm".into());
    }
    let fault = env("FAULT").unwrap_or_else(|| "none".into());
    if !["none", "restore-copy-rank-1"].contains(&fault.as_str())
        || (fault != "none" && arm != "candidate")
    {
        return Err("unsupported lifecycle fault".into());
    }
    let reference = if arm != "baseline" {
        binding("REFERENCE")?
    } else {
        Value::Null
    };
    let control = if arm == "candidate" {
        binding("CONTROL")?
    } else {
        Value::Null
    };
    let root = PathBuf::from(required("OUTPUT")?);
    external(&root)?;
    std::fs::create_dir(&root)?;
    let input_artifact = write(&root.join("input.json"), &raw)?;
    let host = Rc::new(FakeHost::default());
    *host.observer.borrow_mut() = Some(Recorder::default());
    let outcome = execute(
        &host,
        &bytes(&input["configuration"])?,
        &actions,
        fault != "none",
    )?;
    let mut journal = host.journal.borrow().join("\n").into_bytes();
    journal.push(b'\n');
    let recorder = host.observer.borrow_mut().take().unwrap();
    let mut observations = Vec::new();
    for observation in &recorder.observations {
        serde_json::to_writer(&mut observations, observation)?;
        observations.push(b'\n');
    }
    assert_eq!(journal.len() + observations.len(), recorder.bytes);
    let artifacts = json!({"input.json":input_artifact,
        "journal.txt":write(&root.join("journal.txt"), &journal)?,
        "observations.jsonl":write(&root.join("observations.jsonl"), &observations)?});
    let workload: Vec<_> = actions
        .iter()
        .filter(|a| a["kind"] == "execute")
        .map(|a| Ok(json!({"action":a["id"],"name":a["name"],"rows":rows(&bytes(&a["wire"])?)?})))
        .collect::<Result<_>>()?;
    let mut reproduction = vec!["env".to_owned()];
    let environment = input["settings"]["environment"].as_object().unwrap();
    for (key, value) in environment {
        if value.is_null() {
            reproduction.extend(["-u".into(), key.clone()]);
        }
    }
    for (key, value) in environment {
        if let Some(value) = value.as_str() {
            reproduction.push(format!("{key}={value}"));
        }
    }
    // Reproduction must use the same fixed arm as the golden capture.
    reproduction.push("CHOPINFER_QWEN38_PREFILL_OVERLAP=0".into());
    for key in [
        "INPUT",
        "INPUT_SHA256",
        "BINARY_SHA256",
        "ARM",
        "FAULT",
        "REFERENCE",
        "REFERENCE_SHA256",
        "CONTROL",
        "CONTROL_SHA256",
    ] {
        if let Some(value) = env(key) {
            reproduction.push(format!("{PREFIX}{key}={value}"));
        }
    }
    reproduction.push(format!(
        "{PREFIX}OUTPUT={}",
        root.with_file_name(format!(
            "{}-reproduced",
            root.file_name().unwrap().to_string_lossy()
        ))
        .display()
    ));
    reproduction.extend([
        std::env::current_exe()?.to_string_lossy().into_owned(),
        "--exact".into(),
        ENTRY.into(),
        "--nocapture".into(),
    ]);
    let value = json!({"format":"chopinfer.diagnostic.v1","mode":"journal","extension":EXTENSION,
        "backend":"qwen38-fake-host","arm":arm,"fault":fault,"producer":producer()?,
        "identity":{"input_sha256":digest,"settings":input["settings"]},"declared_differences":[],
        "workload":workload,"reference":reference,"control":control,
        "completion":outcome["completion"],"execution_result":if outcome["completion"] == "full" { "passed" } else { "failed" },
        "completed_actions":outcome["completed_actions"],
        "expected_actions":actions.len(),"failed_action":outcome["failed_action"],"teardown":outcome["teardown"],
        "events":host.journal.borrow().len(),"observations":recorder.observations.len(),
        "device":{"execution":"not_run","reason":"CPU fake host; idle-device lifecycle evidence is missing"},
        "reproduction":reproduction,"artifacts":artifacts});
    let terminal = serde_json::to_vec(&value)?;
    if terminal.len() > INPUT_LIMIT {
        return Err("lifecycle metadata budget exceeded".into());
    }
    write(&root.join("diagnostic.json"), &terminal)?;
    std::fs::File::open(&root)?.sync_all()?;
    Ok(())
}

#[test]
fn replay_recording() {
    with_default_journal_settings(|| {
        if ["INPUT", "OUTPUT", "ARM"]
            .iter()
            .any(|key| env(key).is_some())
        {
            replay().unwrap();
        } else {
            // The normal suite checks the same decoder; external replay always requires an input.
            let mut value = input().unwrap();
            assert_eq!(validate_input(&value).unwrap().len(), 9);
            // A recorded setting other than the effective one is refused.
            let wide = &mut value["settings"]["environment"][crate::launch::WIDE_COMM];
            let effective = wide.clone();
            *wide = if effective.is_null() {
                json!("0")
            } else {
                Value::Null
            };
            assert!(validate_input(&value).is_err());
            value["settings"]["environment"][crate::launch::WIDE_COMM] = effective;
            value["actions"][6]["kind"] = json!("restore_prefix");
            assert!(validate_input(&value).is_err());
        }
    });
}

fn run_arm(root: &Path, name: &str, arm: &str, fault: &str) -> Result<()> {
    let mut command = std::process::Command::new(std::env::current_exe()?);
    command.args(["--exact", ENTRY, "--nocapture"]);
    // The pinned lifecycle golden describes the default whole-chunk program.
    command.env("CHOPINFER_QWEN38_PREFILL_OVERLAP", "0");
    for (key, value) in [
        ("INPUT", root.join("input.json").display().to_string()),
        ("INPUT_SHA256", hash(&root.join("input.json"))?),
        ("BINARY_SHA256", hash(&std::env::current_exe()?)?),
        ("OUTPUT", root.join(name).display().to_string()),
        ("ARM", arm.into()),
        ("FAULT", fault.into()),
    ] {
        command.env(format!("{PREFIX}{key}"), value);
    }
    for (key, name) in [("REFERENCE", "baseline"), ("CONTROL", "control")] {
        command.env_remove(format!("{PREFIX}{key}"));
        command.env_remove(format!("{PREFIX}{key}_SHA256"));
        if arm != "baseline" && (key != "CONTROL" || arm == "candidate") {
            command.env(format!("{PREFIX}{key}"), root.join(name));
            command.env(
                format!("{PREFIX}{key}_SHA256"),
                hash(&root.join(name).join("diagnostic.json"))?,
            );
        }
    }
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "replay failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    Ok(())
}

#[test]
fn recorded_mixed_cancellation() {
    with_default_journal_settings(|| {
        let retained = env("DIR");
        let root = retained.as_ref().map(PathBuf::from).unwrap_or_else(|| {
            std::env::temp_dir().join(format!("qwen-lifecycle-{}", std::process::id()))
        });
        external(&root).unwrap();
        std::fs::create_dir(&root).unwrap();
        write(
            &root.join("input.json"),
            &serde_json::to_vec(&input().unwrap()).unwrap(),
        )
        .unwrap();
        for (name, arm, fault) in [
            ("baseline", "baseline", "none"),
            ("control", "control", "none"),
            ("matching", "candidate", "none"),
            ("failure", "candidate", "restore-copy-rank-1"),
            ("repeat-failure", "candidate", "restore-copy-rank-1"),
        ] {
            run_arm(&root, name, arm, fault).unwrap();
        }
        let journal = std::fs::read_to_string(root.join("baseline/journal.txt")).unwrap();
        assert_eq!(
            (
                journal.lines().count(),
                fnv1a(journal.trim_end_matches('\n').as_bytes())
            ),
            EXPECTED
        );
        assert!(journal.len() > 8_000_000);
        // The Python suite consumes these actual full-size exports and runs the reported command.
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/test_lifecycle_diagnostics.py");
        let output = std::process::Command::new("python3")
            .arg("-B")
            .arg(script)
            .env("CHOPINFER_LIFECYCLE_TEST_CAPTURE", &root)
            .env("CHOPINFER_LIFECYCLE_TEST_EVENTS", EXPECTED.0.to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        let test_count = String::from_utf8_lossy(&output.stderr)
            .lines()
            .find_map(|line| {
                line.strip_prefix("Ran ")
                    .and_then(|rest| rest.split(' ').next())
                    .and_then(|count| count.parse::<usize>().ok())
            })
            .unwrap();
        assert!(
            test_count >= 14,
            "full-size lifecycle tests did not execute"
        );
        if retained.is_none() {
            std::fs::remove_dir_all(root).unwrap();
        }
    });
}
