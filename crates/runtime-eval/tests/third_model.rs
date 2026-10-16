//! The same text/image declaration exercised by the Python gate planner.

use std::path::Path;

use runtime_eval::declaration;
use serde_json::json;

#[test]
fn new_model_names_load_without_registration() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = std::fs::read(source.join("tests/qual-third-model/qual.json")).unwrap();
    for name in ["synthetic-vlm", "another-unregistered-vlm"] {
        let root = std::env::temp_dir().join(format!(
            "runtime-eval-third-model-{}-{name}",
            std::process::id()
        ));
        let model = root.join("models").join(name);
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(root.join("tools/qual")).unwrap();
        std::fs::copy(
            source.join("tools/qual/modules.json"),
            root.join("tools/qual/modules.json"),
        )
        .unwrap();
        let mut value = runtime_eval::json::parse(&fixture).unwrap();
        value["model"] = json!(name);
        std::fs::write(model.join("qual.json"), value.to_string()).unwrap();
        let loaded = declaration::load(&model).unwrap();
        assert_eq!(loaded.declaration.model, name);
        for cohort in ["quick", "full"] {
            assert_eq!(
                loaded.declaration.cohort(cohort).unwrap(),
                ["teacher", "image"]
            );
        }
        assert_eq!(
            loaded.declaration.check("teacher").unwrap().module,
            "teacher"
        );
        assert_eq!(
            loaded.declaration.check("image").unwrap().module,
            "multimodal"
        );
        assert!(loaded.declaration.reference("teacher").is_some());
        assert!(loaded.declaration.reference("media").is_some());
        std::fs::remove_dir_all(root).unwrap();
    }
}
