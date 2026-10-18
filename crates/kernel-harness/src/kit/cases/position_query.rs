//! Query preparation with its F32 rotation exposed to the test oracle.
//! The rotation is bounded against f64, then the final query is exact
//! against the shared quantizer on this run's actual F32 intermediate.

use crate::{
    LabResult,
    conformance::{Expected, Rows},
    kit::{OpCase, Required, Spec},
};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{DType, Refusal};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernel_core::pooled_indexer::{PositionIndexerQuery, PositionQueryBuffers};
use kernel_core::position_indexer_oracle::rotate_tail;
use std::{cell::RefCell, rc::Rc};

const CAPACITY: u32 = 4;
const TABLE_POSITIONS: u32 = 4;

kernel_core::scratch_record! {
    /// The serving op's workspace made observable for this test, with the
    /// same bindings and launches; it remains private workspace in serving.
    #[derive(Clone, Copy)]
    struct ObservedBuffers {
        query: record(PositionQueryBuffers),
        rotated: value,
    }
}

struct Observed {
    inner: Prepared<PositionQueryBuffers<View>>,
    info: OpInfo,
}

impl PreparedOp<ObservedBuffers<View>> for Observed {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &ObservedBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.inner.launches(&b.query, b.rotated, counts)
    }
}

pub fn family(queries: &[PositionIndexerQuery]) -> Vec<Required<'static>> {
    queries
        .iter()
        .copied()
        .map(|query| {
            Required::new(
                format!(
                    "position query h{} d{} tail{}",
                    query.heads, query.head_dim, query.tail_rope_dims
                ),
                Need::rows(CAPACITY),
                move |backend, need| backend.position_indexer_query(&query, need),
                move |prepared| case(query, prepared),
            )
            .required()
        })
        .collect()
}

pub fn case(
    query: PositionIndexerQuery,
    prepared: Prepared<PositionQueryBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    let width = (query.heads * query.head_dim) as usize;
    let dim = query.head_dim as usize;
    let tail = query.tail_rope_dims as usize;
    let mut q = crate::wide_values(CAPACITY as usize * width, 1507);
    for (i, value) in q.iter_mut().enumerate() {
        *value = *value * 1.0013 + (i % 7) as f32 * 0.00013;
    }
    for head in q.chunks_exact_mut(dim) {
        head[0] = 6.015625;
        head[1] = 0.5;
        head[dim - tail] = 1.0038;
        head[dim - tail + 1] = 0.8164;
    }
    let table = (0..TABLE_POSITIONS as usize * tail / 2)
        .flat_map(|i| [[0.8_f32, 0.6], [0.0, 1.0], [1.0, 0.0], [-0.6, 0.8]][i % 4])
        .collect::<Vec<_>>();
    let f32_bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let roles = ObservedBuffers {
        query: PositionQueryBuffers {
            quantizer_status_row_offset: 0,
            q: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&q)),
            cos_sin: Spec::input(
                DType::F32,
                (Rows::Fixed(TABLE_POSITIONS as usize), tail),
                f32_bytes(&table),
            ),
            positions: Spec::input(
                DType::I32,
                (Rows::Live, 1),
                [0_i32, 2, 1, 3]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
            ),
            live_rows: Spec::live(),
            quantizer_status: super::block_quantize::status_role(&vec![0; CAPACITY as usize]),
            output: Spec::output(DType::Bf16, (Rows::Live, width)),
        },
        // A finite poison lets the quantizer run when the omission test
        // deliberately removes rotation; the oracle must reject its result.
        rotated: Spec::output(DType::F32, (Rows::Live, width)).filled_with(&[0xa5]),
    };
    let mut info = prepared.info().clone();
    let workspace = info.workspace;
    info.workspace = 0;
    for (name, _) in &mut info.bytes {
        *name = format!("query.{name}");
    }
    info.bytes.push(("rotated".into(), workspace));
    let observed: Prepared<ObservedBuffers<View>> = Box::new(Observed {
        inner: prepared,
        info,
    });
    let case = OpCase::with_counts(observed, CAPACITY, roles, |rows| Counts {
        positions: TABLE_POSITIONS,
        ..Counts::rows(rows)
    })?;
    Ok(case.oracle(move |images, live| {
        let input = images.f32s("query.q")?;
        let table = images.f32s("query.cos_sin")?;
        let positions = images.i32s("query.positions")?;
        let mut reference = OracleOutput {
            values: Vec::new(),
            sum_abs_products: Vec::new(),
            reduction_lengths: Vec::new(),
        };
        for (row, &position) in positions.iter().enumerate().take(live as usize) {
            let start = position as usize * tail;
            let pairs = table[start..start + tail]
                .chunks_exact(2)
                .map(|p| [p[0], p[1]])
                .collect::<Vec<_>>();
            for head in input[row * width..(row + 1) * width].chunks_exact(dim) {
                let values = head.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
                let head = rotate_tail(query, &values, &pairs).map_err(|e| e.0)?;
                reference.values.extend(head.values);
                reference.sum_abs_products.extend(head.sum_abs_products);
                reference.reduction_lengths.extend(head.reduction_lengths);
            }
        }
        let saved = Rc::new(RefCell::new(None));
        let rotated = saved.clone();
        let n = live as usize * width;
        let judge_rotation = move |bytes: &[u8]| -> LabResult<()> {
            let values = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            if values.len() != CAPACITY as usize * width {
                return Err("rotation output size mismatch".into());
            }
            let actual = values[..n].iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
            check_numeric(&actual, &reference, DType::F32)
                .map_err(|e| format!("F32 query tail rotation: {e:?}"))?;
            *rotated.borrow_mut() = Some(values);
            Ok(())
        };
        let quantizer = query.query_quantization().map_err(|e| e.0)?.format;
        let judge_quantization = move |bytes: &[u8]| -> LabResult<()> {
            let saved = saved.borrow();
            let values = saved
                .as_ref()
                .ok_or("quantization needs this run's checked F32 rotation")?;
            if live == 0 {
                return Ok(());
            }
            let expected = quantizer
                .fake_quant(&values[..n], live as usize, width)
                .map_err(|e| e.0)?;
            let expected = expected
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            if bytes.len() != CAPACITY as usize * width * 2 || bytes[..n * 2] != expected {
                return Err("query quantization differs from the shared quantizer on this run's F32 rotation".into());
            }
            Ok(())
        };
        Ok(vec![
            ("query.quantizer_status".into(),Expected::Bits(vec![0;CAPACITY as usize*4])),
            ("rotated".into(), Expected::Judge(Box::new(judge_rotation))),
            (
                "query.output".into(),
                Expected::Judge(Box::new(judge_quantization)),
            ),
        ])
    }))
}
