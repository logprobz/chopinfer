//! The fixture's weight plan: the one rank's program image is the tensor
//! `rank0` of `rank0.safetensors`, copied whole, from the synthetic
//! checkpoint [`write_recurrent_fixture_checkpoint_v1`] writes.
use crate::contract::{
    RECURRENT_GEOMETRY_ALIGNMENT_V1, RECURRENT_GEOMETRY_BYTES_V1, RECURRENT_GEOMETRY_OFFSET_V1,
    RECURRENT_GEOMETRY_VIEW_V1, RECURRENT_PROGRAM_IMAGE_ALIGNMENT_V1,
    RECURRENT_PROGRAM_IMAGE_BYTES_V1, RECURRENT_PROGRAM_ROLE_V1, RECURRENT_RANK_COUNT_V1,
    RECURRENT_STATE_ALIGNMENT_V1, RECURRENT_STATE_SEED_OFFSET_V1, RECURRENT_STATE_SEED_VIEW_V1,
    program_image_bytes_v1,
};
use crate::recurrent::RECURRENT_STATE_BYTES_V1;
use kernel_core::weight_transform::WeightTransform;
use runtime_bundle_abi::weight_plan::{
    PlannedViewV1, RecordSourceV1, ViewPlacementV1, WeightPlanRequestV1, WeightPlanV1,
    WeightRecordV1,
};

/// Plans the program image, the primary image; declines every other role.
pub fn recurrent_fixture_weight_plan_v1(
    request: &WeightPlanRequestV1,
) -> Result<Option<WeightPlanV1>, String> {
    if !request.image_role.is_empty() && request.image_role != RECURRENT_PROGRAM_ROLE_V1 {
        return Ok(None);
    }
    if request.ranks != RECURRENT_RANK_COUNT_V1 || request.rank >= request.ranks {
        return Err(format!(
            "the recurrent fixture plans {RECURRENT_RANK_COUNT_V1} rank, not rank {} of {}",
            request.rank, request.ranks
        ));
    }
    let name = format!("rank{}.safetensors", request.rank);
    let file = request
        .files
        .iter()
        .position(|file| file.name == name)
        .ok_or_else(|| format!("the checkpoint has no {name}"))?;
    let view = |id: &[u8], alignment, offset, byte_length| PlannedViewV1 {
        id: String::from_utf8_lossy(id).into_owned(),
        alignment,
        placement: ViewPlacementV1::Device,
        offset,
        byte_length,
    };
    Ok(Some(WeightPlanV1 {
        ranks: request.ranks,
        rank: request.rank,
        image_role: RECURRENT_PROGRAM_ROLE_V1.into(),
        alignment: RECURRENT_PROGRAM_IMAGE_ALIGNMENT_V1,
        arena_bytes: RECURRENT_PROGRAM_IMAGE_BYTES_V1,
        views: vec![
            view(
                RECURRENT_GEOMETRY_VIEW_V1,
                RECURRENT_GEOMETRY_ALIGNMENT_V1,
                RECURRENT_GEOMETRY_OFFSET_V1,
                RECURRENT_GEOMETRY_BYTES_V1,
            ),
            view(
                RECURRENT_STATE_SEED_VIEW_V1,
                RECURRENT_STATE_ALIGNMENT_V1,
                RECURRENT_STATE_SEED_OFFSET_V1,
                RECURRENT_STATE_BYTES_V1,
            ),
        ],
        components: vec![],
        records: vec![WeightRecordV1 {
            arena_offset: 0,
            byte_length: RECURRENT_PROGRAM_IMAGE_BYTES_V1,
            transform: WeightTransform::Copy.id(),
            params: vec![],
            sources: vec![RecordSourceV1 {
                file: u32::try_from(file).map_err(|e| e.to_string())?,
                tensor: format!("rank{}", request.rank),
            }],
            axis: None,
            ranges: vec![],
            stored_shape: vec![RECURRENT_PROGRAM_IMAGE_BYTES_V1],
        }],
    }))
}

/// Writes the synthetic checkpoint the weight plan loads into the new
/// directory `root`: `rank0.safetensors` holding the U8 tensor `rank0`, the
/// program image, and a `config.json` with the fixture's architecture.
pub fn write_recurrent_fixture_checkpoint_v1(root: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir(root)?;
    let data = program_image_bytes_v1();
    let header = serde_json::to_vec(&serde_json::json!({
        "rank0": {"dtype": "U8", "shape": [data.len()], "data_offsets": [0, data.len()]}
    }))?;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    std::fs::write(root.join("rank0.safetensors"), bytes)?;
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&serde_json::json!({"architectures": ["RecurrentFixture"]}))?,
    )
}
