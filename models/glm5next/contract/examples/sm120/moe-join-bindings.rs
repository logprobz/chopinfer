//! Export the served projection choices for a complete-MoE join lab.
use kernel_core::contract::{ScaleMode, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::precision::Compute;
use model_glm5next_contract::{contract::Shape, sm120::precision};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn main() -> Result<(), String> {
    let cubins = precision::cubin_specs(Shape::TP4)?;
    let mut bindings = BTreeMap::new();
    let mut module = None;
    for rows in [8, 16, 32] {
        let mut roles = BTreeMap::new();
        for (role, id, weight) in [
            ("router", "moe.router", WeightFormat::Bf16),
            (
                "shared_down",
                "moe.shared.down",
                WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            ),
        ] {
            let offer = precision::offer(
                id,
                weight,
                OperationUse {
                    phase: ExecutionPhase::Decode,
                    capacity: rows,
                    graph: true,
                },
            )?;
            let Compute::Dense(spec) = &offer.offer.handle().compute else {
                return Err("expected dense offer".into());
            };
            let cubin = cubins
                .iter()
                .find(|c| c.entries.iter().any(|e| e.symbol == spec.symbol))
                .ok_or("projection cubin missing")?;
            if let Some(previous) = &module {
                if previous != &cubin.name {
                    return Err("join projections span multiple cubins".into());
                }
            } else {
                module = Some(cubin.name.clone());
            }
            roles.insert(role, spec.symbol.clone());
        }
        bindings.insert(rows.to_string(), roles);
    }
    let lock = include_bytes!("../../../bundle/kernels.lock.d/projections.lock");
    println!("{}", serde_json::to_string_pretty(&json!({"schema":"a16-moe-join-bindings.v1","bindings":bindings,
        "cubin_name":module.ok_or("empty bindings")?,"lock_sha256":format!("{:x}",Sha256::digest(lock))})).map_err(|e|e.to_string())?);
    Ok(())
}
