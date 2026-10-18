//! Catalog uniqueness uses the same fields and equality as admission.
use model_compiler_sdk::geometry::match_geometry;
use serde_json::Value;

pub(crate) fn check_unique(
    id: &str,
    config: &Value,
    existing: &[(&str, Value)],
    compile_fields: &[&str],
) -> Result<(), String> {
    if let Ok(index) = match_geometry(config, existing, compile_fields) {
        return Err(format!(
            "duplicate geometry admission key {id}: matches {}",
            existing[index].0
        ));
    }
    Ok(())
}
