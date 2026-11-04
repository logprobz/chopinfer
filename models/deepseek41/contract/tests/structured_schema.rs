//! The shared structured.v1 PERSON schema passes the recipe's strict schema
//! check, so the strict cases of the structured check and of soak's
//! structured mix are requests this frontend admits. The schema is read from
//! its one source, `tools/qual/structured.py`.

use std::path::Path;
use std::process::Command;

use frontend_format::FrontendJsonV1;
use model_deepseek41_contract::frontend::strict_schema;

/// `structured.PERSON` as Python writes it, its keys in order. Python is a
/// CPU-gate prerequisite.
fn person() -> FrontendJsonV1 {
    let tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tools/qual");
    let output = Command::new("python3")
        .args([
            "-B",
            "-c",
            "import json, sys\nsys.path.insert(0, sys.argv[1])\nimport structured\n\
             print(json.dumps(structured.PERSON))",
        ])
        .arg(tools)
        .env_remove("CHOPINFER_QUAL_CONFIG")
        .output()
        .expect("python3 starts");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("PERSON is JSON")
}

#[test]
fn the_shared_person_schema_passes_the_strict_schema_check() {
    let person = person();
    assert!(matches!(&person, FrontendJsonV1::Object(entries) if !entries.is_empty()));
    if let Err(error) = strict_schema::response_schema(&person) {
        panic!("structured.PERSON is refused: {error}");
    }
}
