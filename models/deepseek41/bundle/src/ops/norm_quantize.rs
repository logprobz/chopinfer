use super::{Asks, Bind, Lowered, exact};
use crate::{Kernels, program::Phase, shape::Shape};
use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;
use model_deepseek41_contract::{rows, serving::DECODE_BATCH};
use runtime_bundle_abi::DeviceBufferSpanV1;

pub(crate) struct NormQuantizeOps {
    input: Prepared<RmsNormQuantizeBuffers<View>>,
    query: Prepared<RmsNormQuantizeBuffers<View>>,
    small_decode: Option<[Prepared<RmsNormQuantizeBuffers<View>>; 2]>,
}

impl NormQuantizeOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
        target: bool,
    ) -> Result<Self, Refused> {
        let [input, query] = rows::norm_quantizers(shape.geometry);
        let need = exact::need(DECODE_BATCH, exact::every_phase(DECODE_BATCH));
        Ok(Self {
            input: asks.ask("rows.input_norm_quantize", need, |need| {
                backend.rmsnorm_quantize(&input, need)
            })?,
            query: asks.ask("rows.query_norm_quantize", need, |need| {
                backend.rmsnorm_quantize(&query, need)
            })?,
            small_decode: if target {
                Kernels::small_decode_norm_backend(shape)
                    .map(|backend| {
                        let need = exact::need(4, exact::every_phase(4));
                        Ok::<_, Refused>([
                            asks.ask("rows.input_norm_small_decode", need, |need| {
                                backend.rmsnorm_quantize(&input, need)
                            })?,
                            asks.ask("rows.query_norm_small_decode", need, |need| {
                                backend.rmsnorm_quantize(&query, need)
                            })?,
                        ])
                    })
                    .transpose()?
            } else {
                None
            },
        })
    }
    pub(crate) fn launch(
        &self,
        query: bool,
        phase: Phase,
        views: RmsNormQuantizeBuffers<DeviceBufferSpanV1>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = RmsNormQuantizeBuffers {
            input: bind.view(views.input),
            weight: bind.view(views.weight),
            normed: bind.view(views.normed),
            widened: views.widened.map(|v| bind.view(v)),
            codes: bind.view(views.codes),
            scales: bind.view(views.scales),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: views.status_row_offset,
        };
        bind.launch(
            self.selected(query, phase),
            &buffers,
            Counts::rows(phase.rows()),
        )
    }

    fn selected(&self, query: bool, phase: Phase) -> &Prepared<RmsNormQuantizeBuffers<View>> {
        if matches!(phase, Phase::Decode(2 | 4))
            && let Some(ops) = &self.small_decode
        {
            return &ops[usize::from(query)];
        }
        if query { &self.query } else { &self.input }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn cached_small_norms_preserve_other_phases_topologies_and_drafter() {
        let view = |handle| View {
            handle,
            bytes: 1 << 30,
        };
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = Kernels::new(&shape).unwrap();
            for target in [false, true] {
                let ops = NormQuantizeOps::prepare(
                    kernels.backend.as_ref(),
                    &shape,
                    &Asks::default(),
                    target,
                )
                .unwrap();
                for (index, query) in rows::norm_quantizers(&FLASH).into_iter().enumerate() {
                    let reference = kernels
                        .backend
                        .rmsnorm_quantize(
                            &query,
                            exact::need(DECODE_BATCH, exact::every_phase(DECODE_BATCH)),
                        )
                        .unwrap();
                    for widen in [false, true] {
                        if index == 1 && widen {
                            continue;
                        }
                        let buffers = RmsNormQuantizeBuffers {
                            input: view(1),
                            weight: view(2),
                            normed: view(3),
                            widened: widen.then(|| view(4)),
                            codes: view(5),
                            scales: view(6),
                            live_rows: view(7),
                            status: view(8),
                            status_row_offset: 3,
                        };
                        for rows in 1..=DECODE_BATCH {
                            let counts = Counts::rows(rows);
                            let old = reference.launches(&buffers, view(9), counts).unwrap();
                            for phase in [
                                Phase::Decode(rows),
                                Phase::Verify { rows, items: 1 },
                                Phase::Prefill { rows },
                            ] {
                                let mut expected = old.clone();
                                let changed = target
                                    && ranks == Ranks::TP4
                                    && matches!(phase, Phase::Decode(2 | 4));
                                if changed {
                                    let (entry, warps, symbol_index) = if index == 0 {
                                        (crate::sm120::norm_quantize_cached::entry(&FLASH), 32, 2)
                                    } else {
                                        (crate::sm120::norm_query_cached::entry(&FLASH), 16, 1)
                                    };
                                    expected[0].symbol = entry.symbols()[symbol_index].clone();
                                    expected[0].plan = entry.plan(rows, rows, warps).unwrap();
                                }
                                let selected = ops.selected(index == 1, phase);
                                let actual = selected.launches(&buffers, view(9), counts).unwrap();
                                assert_eq!(
                                    actual, expected,
                                    "{ranks:?} target={target} {phase:?} query={index} widen={widen}"
                                );
                                if changed {
                                    assert_eq!(
                                        selected.route(counts, &actual[0]).unwrap().class,
                                        kernel_core::route::RouteClass::Fast
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
