//! The numerically qualified small-row query must reach serving through its owned cubin.
use super::*;
use kernel_core::{
    backend::{Counts, Need, View},
    pooled_indexer::PositionQueryBuffers,
};
use model_deepseek41_contract::{indexer, serving::PREFILL_TOKENS};

#[test]
fn parallel_index_query_is_bound_in_every_serving_row_bucket() {
    for shape in crate::shape::topologies() {
        let kernels = crate::Kernels::new(&shape).unwrap();
        let g = shape.geometry;
        let q = indexer::selection(g, g.shape.csa.index_sources[0], PREFILL_TOKENS).unwrap();
        let op = kernels
            .backend
            .position_indexer_query(&q, Need::rows(PREFILL_TOKENS))
            .unwrap();
        for (name, _) in &op.info().symbols {
            let cubin = crate::CUBINS
                .iter()
                .find(|c| c.symbols.contains(&name.as_str()))
                .unwrap();
            assert!(crate::serves(cubin.name, &shape));
        }
        let v = |handle, bytes| View { handle, bytes };
        let b = PositionQueryBuffers {
            q: v(1, 8192 * 4096 * 4),
            cos_sin: v(2, 8 * 64 * 4),
            positions: v(3, 8192 * 4),
            live_rows: v(4, 4),
            output: v(5, 8192 * 4096 * 2),
            quantizer_status: v(6, 8192 * 4),
            quantizer_status_row_offset: 0,
        };
        for rows in [1, 2, 4, 6, 8, 16, 32, 33, 64, 8192] {
            let calls = op
                .launches(
                    &b,
                    v(7, op.info().workspace),
                    Counts {
                        positions: 8,
                        ..Counts::rows(rows)
                    },
                )
                .unwrap();
            if rows <= 32 {
                assert_eq!(calls.len(), 1);
                assert!(calls[0].symbol.ends_with("_parallel_v1"));
                assert_eq!(calls[0].plan.grid, [rows, 1, 1]);
                assert_eq!(calls[0].plan.block, [1024, 1, 1]);
            } else {
                assert_eq!(calls.len(), 2);
                assert!(calls[0].symbol.ends_with("_rotate_v3"));
                assert!(calls[1].symbol.ends_with("_mxfp4_v2"));
            }
        }
    }
}

pub(super) fn assert_query_bindings(lines: &[String]) {
    let mut parallel = 0;
    let mut original = 0;
    for line in lines
        .iter()
        .filter(|s| s.starts_with("launch ") && s.contains("chop_position_query_"))
    {
        let rows: u32 = line
            .split("grid_x: ")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        if line.contains("_parallel_v1 ") {
            assert!(rows <= 32);
            assert!(line.contains("grid_y: 1,") && line.contains("block_x: 1024,"));
            parallel += 1;
        } else {
            assert!(rows > 32, "small query used the serial quantizer: {line}");
            original += 1;
        }
    }
    assert!(parallel > 0 && original > 0);
}
