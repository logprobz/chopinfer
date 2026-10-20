//! Fused candidate maxima and register-cached, ordered block selection.
use super::{position_blocks::CandidateEntry, position_candidates_runtime, position_valid};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::{
    backend::Launch,
    contract::{Argument, Refusal},
};

pub fn symbol(entry: &CandidateEntry) -> String {
    format!("{}_cached4_v1", entry.gather_symbol())
}

pub fn spec(entry: &CandidateEntry) -> Result<EntrySpec, Refusal> {
    let mut spec = position_candidates_runtime::specs(entry)?.remove(1);
    spec.symbol = symbol(entry);
    spec.params[0] = EntryParam::DevicePointer {
        name: "scores".into(),
        cxx_type: "const float".into(),
    };
    spec.body = format!(
        "chop::position_candidates_cached::publish<{},{}>(scores,live_rows,tags,visible,source_status,ids,out_tags,out_visible,status,width,top_k_blocks);",
        entry.query.block_size, entry.query.top_k_blocks
    );
    Ok(spec)
}

/// Adapt an already validated producer. Its public buffers retain their
/// original spans; only the private maxima and shard workspace disappear.
pub fn remap(entry: &CandidateEntry, calls: &[Launch]) -> Result<Launch, Refusal> {
    entry.validate()?;
    let [scores, local, merge] = calls else {
        return Err(Refusal("cached candidates require the three-stage valid producer"));
    };
    let runtime = match merge.arguments.0.len() {
        12 => false,
        13 => true,
        _ => return Err(Refusal("cached candidate argument count changed")),
    };
    let original = if runtime {
        position_candidates_runtime::entry(entry)
    } else {
        entry.clone()
    };
    if scores.symbol != position_valid::symbol(&original.score_symbol())
        || local.symbol != position_valid::symbol(&original.gather_symbol())
        || merge.symbol != position_valid::merge_symbol(&original.gather_symbol())
        || scores.arguments.0.len() != 5
        || scores.rows != merge.rows
        || local.rows != merge.rows
        || local.arguments.0 != merge.arguments.0
        || merge.rows == 0
        || merge.rows > entry.query.capacity
        || scores.plan.grid != [32, merge.rows, 1]
        || local.plan.grid != [32, merge.rows, 1]
        || merge.plan.grid != [1, merge.rows, 1]
        || calls.iter().any(|c| c.plan.block != [256, 1, 1])
    {
        return Err(Refusal("cached candidate launch geometry changed"));
    }
    let Argument::U32(width) = merge.arguments.0[9] else {
        return Err(Refusal("cached candidate width is not a scalar"));
    };
    let kept = if runtime {
        let Argument::U32(kept) = merge.arguments.0[10] else {
            return Err(Refusal("cached candidate count is not a scalar"));
        };
        kept
    } else {
        entry.query.top_k_blocks
    };
    if width == 0
        || width > 2 * entry.query.block_size * entry.query.top_k_blocks
        || kept == 0
        || kept > entry.query.top_k_blocks
        || scores.arguments.0[4] != Argument::U32(width)
        || scores.arguments.0[1] != merge.arguments.0[3]
        || scores.arguments.0[2] != merge.arguments.0[1]
        || scores.arguments.0[3] != merge.arguments.0[0]
    {
        return Err(Refusal("cached candidate extent or input bindings changed"));
    }
    let mut result = merge.clone();
    result.symbol = symbol(entry);
    result.arguments.0.truncate(10);
    result.arguments.0[0] = scores.arguments.0[0].clone();
    result.arguments.0.push(Argument::U32(kept));
    Ok(result)
}

pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_valid::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_candidates_cached.cuh\"".into()];
    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use kernel_core::{
        backend::{Backend, Counts, Need, View},
        pooled_indexer::CandidateBlocksQuery,
        position_blocks::{CandidateBlockIds, CandidateBlockIdsBuffers},
    };

    fn prepared(width: u32, kept: Option<u32>) -> (CandidateEntry, Vec<Launch>) {
        let entry = CandidateEntry {
            name: "cached_candidate_test".into(),
            query: CandidateBlocksQuery {
                block_size: 8,
                top_k_blocks: 2048,
                capacity: 8192,
            },
        };
        let backend = Sm120::new()
            .with_indexer_candidate_block_ids([entry.clone()])
            .with_indexer_candidate_block_ids_runtime([entry.clone()])
            .with_valid_position_candidates();
        let need = Need::rows(512);
        let op = match kept {
            Some(k) => backend.indexer_candidate_block_ids_runtime(&entry.query, k, width, need),
            None => backend.indexer_candidate_block_ids(&entry.query, width, need),
        }
        .unwrap();
        let view = |handle| View { handle, bytes: 1 << 32 };
        let buffers = CandidateBlockIdsBuffers {
            scores: view(1),
            visible_positions: view(2),
            row_tags: view(3),
            source_status: view(4),
            live_rows: view(5),
            output: CandidateBlockIds {
                ids: view(6),
                row_tags: view(7),
                visible_positions: view(8),
                status: view(9),
                position_capacity: width,
            },
        };
        let calls = op.launches(&buffers, view(10), Counts::rows(512)).unwrap();
        (entry, calls)
    }

    #[test]
    fn fused_producer_retains_validated_inputs_outputs_and_runtime_counts() {
        for width in [1, 8, 8192, 16384, 16385, 32768] {
            for kept in [None, Some(1), Some(31), Some(2048)] {
                let (entry, calls) = prepared(width, kept);
                let fused = remap(&entry, &calls).unwrap();
                assert_eq!(fused.rows, 512);
                assert_eq!(fused.plan, calls[2].plan);
                assert_eq!(fused.arguments.0.len(), spec(&entry).unwrap().params.len());
                assert_eq!(fused.arguments.0[0], calls[0].arguments.0[0]);
                assert_eq!(&fused.arguments.0[1..10], &calls[2].arguments.0[1..10]);
                assert_eq!(fused.arguments.0[10], Argument::U32(kept.unwrap_or(2048)));
                assert!(!fused.arguments.0.contains(&Argument::DeviceBuffer(10)));
            }
        }
    }

    #[test]
    fn fused_producer_refuses_wider_shapes_and_changed_launches() {
        let (entry, calls) = prepared(32769, None);
        assert!(remap(&entry, &calls).is_err());
        let (entry, calls) = prepared(16384, Some(31));
        let mut bad = calls.clone();
        bad[0].arguments.0[1] = Argument::DeviceBuffer(999);
        assert!(remap(&entry, &bad).is_err());
        let mut bad = calls.clone();
        bad[2].plan.grid[0] = 4;
        assert!(remap(&entry, &bad).is_err());
        let mut bad = calls.clone();
        bad[2].arguments.0[10] = Argument::U32(0);
        assert!(remap(&entry, &bad).is_err());
        assert!(remap(&entry, &calls[..2]).is_err());
    }
}
