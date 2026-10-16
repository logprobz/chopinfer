//! The SM120 offers of the projection API's inventory against the stage
//! records of every program (`crate::launch`).

use std::collections::BTreeMap;

use kernel_core::contract::{ActivationFormat, StaticQuery, WeightFormat};
use kernel_core::latent_attention::LatentKvFormat;
use kernels_sm120::precision::Compute;
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::precision;

use crate::launch::{Op, drafter_records, records as records_of};
use crate::program::{DraftPass, Phase, Stage};

/// The projection API's inventory (`model_glm5next_contract::sm120::precision`)
/// serves exactly the programs' projection stages (the target's in
/// every phase and both drafter passes; the pooling gate is a side
/// tensor), at the uses its declaration's profile lists, each with an
/// offer over the record's rank-local matrix from the rank views
/// (BF16 rows padded to the 64-row tile), weight format, BF16
/// activations, output dtype and residual add.
#[test]
fn the_precision_inventory_serves_every_projection_stage() {
    let declaration: serde_json::Value =
        serde_json::from_str(model_glm5next_contract::projections::DECLARATION).unwrap();
    let declared = |id: &str| {
        declaration["projections"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == id)
            .cloned()
    };
    let shape = Shape::TP4;
    let kv = LatentKvFormat::Fp8E4M3Block128;
    let mut records = Vec::new();
    for phase in [
        Phase::Prefill { rows: 64 },
        Phase::Decode(8),
        Phase::Verify { rows: 32, items: 8 },
    ] {
        records.extend(records_of(shape, kv, phase).unwrap());
    }
    for pass in [DraftPass::Extend, DraftPass::Step] {
        records.extend(drafter_records(shape, kv, Phase::Decode(8), pass).unwrap());
    }
    let mut stages = BTreeMap::new();
    for record in records {
        let (id, gemm) = match (record.stage, record.op) {
            (Stage::Project(_, id), Op::Gemm(gemm)) => (id, gemm),
            (Stage::LmHead, Op::Gemm(gemm)) => ("head", gemm),
            _ => continue,
        };
        if declared(id).is_some() {
            stages.insert(id, gemm);
        }
    }
    let mut lowered = precision::LOWERED.to_vec();
    lowered.sort_unstable();
    assert_eq!(stages.keys().copied().collect::<Vec<_>>(), lowered);
    for (id, gemm) in stages {
        let projection = declared(id).unwrap();
        let profile = projection["use_profile"].as_str().unwrap();
        let profile_uses: Vec<String> = declaration["use_profiles"][profile]
            .as_array()
            .unwrap()
            .iter()
            .map(|u| {
                format!(
                    "{} {} {}",
                    u["phase"].as_str().unwrap(),
                    u["capacity"],
                    u["graph"]
                )
            })
            .collect();
        let uses = precision::uses(id).unwrap();
        assert_eq!(
            uses.iter()
                .map(|(u, _)| format!("{:?} {} {}", u.phase, u.capacity, u.graph))
                .collect::<Vec<_>>(),
            profile_uses,
            "{id}"
        );
        let rows = match gemm.weight {
            WeightFormat::Bf16 => gemm.n.next_multiple_of(64),
            _ => gemm.n,
        };
        for (usage, _) in uses {
            let generated =
                model_glm5next_contract::sm120::precision::offer(id, gemm.weight, usage).unwrap();
            let Compute::Dense(spec) = &generated.offer.handle().compute else {
                panic!("{id} is not a dense GEMM");
            };
            let StaticQuery::Gemm(query) = spec.key.query else {
                panic!("{id} has no GEMM query");
            };
            assert_eq!((query.n, query.k), (rows, gemm.k), "{id} {usage:?}");
            assert_eq!(query.weight, gemm.weight, "{id}");
            assert_eq!(query.activation, ActivationFormat::Bf16, "{id}");
            assert_eq!(query.output, gemm.output, "{id}");
            assert_eq!(query.epilogues, gemm.epilogues, "{id}");
            assert!(query.m_capacity >= usage.capacity, "{id} {usage:?}");
        }
    }
}
