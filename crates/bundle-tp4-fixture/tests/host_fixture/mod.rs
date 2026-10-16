#![allow(dead_code)]

//! The host-side declaration this fixture's four ranks are served from.
//!
//! Both the direct execution test and the served acceptance test build the same
//! instance configuration a real host would build from this package's manifest,
//! so neither is proving a shape the other does not run.

use std::collections::BTreeMap;

use bundle_tp4_fixture::{
    TP4_FIXTURE_FRONTEND_ROLE_V1, TP4_FIXTURE_GEOMETRY_ALIGNMENT_V1, TP4_FIXTURE_GEOMETRY_BYTES_V1,
    TP4_FIXTURE_GEOMETRY_OFFSET_V1, TP4_FIXTURE_GEOMETRY_VIEW_V1, TP4_FIXTURE_KV_LAYOUT_ID_V1,
    TP4_FIXTURE_MARKER_ROLE_V1, TP4_FIXTURE_PROGRAM_IMAGE_ALIGNMENT_V1,
    TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1, TP4_FIXTURE_PROGRAM_ROLE_V1, TP4_FIXTURE_RANK_COUNT_V1,
    TP4_FIXTURE_STATE_ALIGNMENT_V1, TP4_FIXTURE_STATE_BYTES_V1, TP4_FIXTURE_STATE_LAYOUT_ID_V1,
    TP4_FIXTURE_STATE_SEED_OFFSET_V1, TP4_FIXTURE_STATE_SEED_VIEW_V1, program_image_bytes_v1,
    tp4_fixture_frontend_declaration_v1, tp4_fixture_kv_layout_v1, tp4_fixture_marker_v1,
    tp4_fixture_topology_v1,
};
use runtime_host::{
    CompiledImageDescriptionV1, CompiledImageViewV1, CompiledRankImageV1, CompiledViewSpanV1,
    DeclaredKvContractV1, DeclaredStateContractV1, HostInstanceConfigV1, HostTopologyV1,
};

/// The four device ordinals this fixture's declared topology is served on.
pub fn tp4_ordinals() -> Vec<i32> {
    (0..TP4_FIXTURE_RANK_COUNT_V1 as i32).collect()
}

/// Where one rank's image starts inside the staged program component.
///
/// The emitter concatenates the rank images at the declared alignment, and the
/// image length is a whole number of alignments, so this is where each lands.
pub fn rank_image_offset(rank: u32) -> u64 {
    u64::from(rank) * TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1
}

/// The whole program component, as the emitter stages it.
pub fn program_component_bytes() -> Vec<u8> {
    (0..TP4_FIXTURE_RANK_COUNT_V1)
        .flat_map(program_image_bytes_v1)
        .collect()
}

/// The same span of every rank image, as one declared view.
fn rank_spans(offset: u64, byte_length: u64) -> BTreeMap<u32, CompiledViewSpanV1> {
    (0..TP4_FIXTURE_RANK_COUNT_V1)
        .map(|rank| {
            (
                rank,
                CompiledViewSpanV1 {
                    offset: rank_image_offset(rank) + offset,
                    byte_length,
                },
            )
        })
        .collect()
}

/// How the program component divides for the four ranks this fixture serves.
pub fn program_image_description() -> CompiledImageDescriptionV1 {
    let mut views = BTreeMap::new();
    views.insert(
        TP4_FIXTURE_GEOMETRY_VIEW_V1.to_vec(),
        CompiledImageViewV1::new(
            TP4_FIXTURE_GEOMETRY_ALIGNMENT_V1,
            rank_spans(
                TP4_FIXTURE_GEOMETRY_OFFSET_V1,
                TP4_FIXTURE_GEOMETRY_BYTES_V1,
            ),
        ),
    );
    views.insert(
        TP4_FIXTURE_STATE_SEED_VIEW_V1.to_vec(),
        CompiledImageViewV1::new(
            TP4_FIXTURE_STATE_ALIGNMENT_V1,
            rank_spans(TP4_FIXTURE_STATE_SEED_OFFSET_V1, TP4_FIXTURE_STATE_BYTES_V1),
        ),
    );
    CompiledImageDescriptionV1::new(
        TP4_FIXTURE_PROGRAM_IMAGE_ALIGNMENT_V1,
        (0..TP4_FIXTURE_RANK_COUNT_V1)
            .map(|rank| CompiledRankImageV1 {
                offset: rank_image_offset(rank),
                byte_length: TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1,
            })
            .collect(),
        views,
    )
}

/// The declaration a host builds from this fixture's own manifest records.
pub fn instance_config() -> HostInstanceConfigV1 {
    instance_config_with_program(program_component_bytes())
}

/// The same declaration, over program bytes the caller chose.
pub fn instance_config_with_program(program: Vec<u8>) -> HostInstanceConfigV1 {
    let topology = HostTopologyV1::new(tp4_ordinals()).expect("the TP4 test topology is valid");
    let page_layout = tp4_fixture_kv_layout_v1()
        .expect("the declared KV layout is valid")
        .declared_page_layout(&tp4_fixture_topology_v1().expect("the declared topology is valid"))
        .expect("the declared KV layout describes this topology");
    HostInstanceConfigV1::new(topology)
        .with_kv_contract(DeclaredKvContractV1::new(
            TP4_FIXTURE_KV_LAYOUT_ID_V1.as_bytes().to_vec(),
            page_layout,
        ))
        .with_state_contract(DeclaredStateContractV1::new(
            TP4_FIXTURE_STATE_LAYOUT_ID_V1.as_bytes().to_vec(),
            TP4_FIXTURE_STATE_BYTES_V1,
            TP4_FIXTURE_STATE_ALIGNMENT_V1,
        ))
        .with_compiled_image(
            TP4_FIXTURE_PROGRAM_ROLE_V1.as_bytes(),
            program,
            program_image_description(),
        )
        .expect("the program component is valid")
        .with_component(
            TP4_FIXTURE_FRONTEND_ROLE_V1.as_bytes(),
            tp4_fixture_frontend_declaration_v1(),
        )
        .expect("the frontend component is valid")
        .with_component(
            TP4_FIXTURE_MARKER_ROLE_V1.as_bytes(),
            tp4_fixture_marker_v1(),
        )
        .expect("the fixture marker component is valid")
}
