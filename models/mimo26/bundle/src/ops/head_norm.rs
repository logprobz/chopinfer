//! DFlash's per-head RMSNorm of its Q and K heads.

use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::contract::{Refusal, RmsNormKind};
use kernel_core::rmsnorm::{RmsNormBuffers, RmsNormLayout, RmsNormRequirement};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Bind, ByPhase, no_workspace, one};
use crate::program_lower::Lowered;
use crate::selection::{MimoPhase as _, Phase};
use crate::topology::Tp;

/// Which of DFlash's head norms: the draft block's Q or K heads, read from
/// a layer's packed QKV rows, or the context's K heads, read from the
/// packed K and V rows of every layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadNorm {
    Q,
    K,
    ContextK,
}

/// The norm of `norm`'s heads of `tp`'s ranks into compact rows of heads.
fn requirement(tp: Tp, norm: HeadNorm) -> RmsNormRequirement {
    let d = tp.geometry.dflash();
    let head_dim = d.head_dim;
    let (q, kv) = (
        tp.dflash_q_heads() as u32 * head_dim,
        tp.dflash_kv_heads() as u32 * head_dim,
    );
    let (width, input_stride) = match norm {
        HeadNorm::Q => (q, q + 2 * kv),
        HeadNorm::K => (kv, q + 2 * kv),
        HeadNorm::ContextK => (kv, 2 * kv * d.num_hidden_layers),
    };
    RmsNormRequirement {
        layout: RmsNormLayout::StridedHeads {
            input_stride: u64::from(input_stride),
            output_stride: u64::from(width),
            alignment_bytes: 16,
        },
        ..RmsNormRequirement::bf16_backend_chosen(
            width,
            d.rms_norm_eps as f32,
            RmsNormKind::QkPerHead { head_dim },
            false,
        )
    }
}

type Norm = Prepared<RmsNormBuffers<View>>;

/// A DFlash instance's head norms by phase: the context's at every phase,
/// the block's at verify buckets.
#[derive(Default)]
pub(super) struct HeadNorms {
    q: ByPhase<Norm>,
    k: ByPhase<Norm>,
    context_k: ByPhase<Norm>,
}

impl HeadNorms {
    pub(super) fn prepare(backend: &dyn Backend, tp: Tp, dflash: bool) -> Result<Self, Refused> {
        let prepare = |norm: HeadNorm| {
            ByPhase::prepare(|phase| {
                if !dflash || (norm != HeadNorm::ContextK && !matches!(phase, Phase::Verify(_))) {
                    return Ok(None);
                }
                let need = Need::rows(phase.capacity()?);
                no_workspace(backend.rmsnorm(&requirement(tp, norm), need)?).map(Some)
            })
        };
        Ok(Self {
            q: prepare(HeadNorm::Q)?,
            k: prepare(HeadNorm::K)?,
            context_k: prepare(HeadNorm::ContextK)?,
        })
    }
}

impl super::Ops {
    /// DFlash's `norm` of a `phase` step's heads (bucket `bucket` of
    /// `COMM_BUCKETS`) at `input`, by the head weight every head shares,
    /// into compact rows at `output`. It grids and records the phase's
    /// capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn head_norm(
        &self,
        norm: HeadNorm,
        phase: Phase,
        bucket: usize,
        input: DeviceBufferSpanV1,
        weight: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
        live_tokens: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let norms = &self.head_norms;
        let op = match norm {
            HeadNorm::Q => &norms.q,
            HeadNorm::K => &norms.k,
            HeadNorm::ContextK => &norms.context_k,
        }
        .get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = RmsNormBuffers {
            x: bind.view(input),
            weight: bind.view(weight),
            residual: None,
            y: bind.view(output),
            live_rows: bind.view(live_tokens),
        };
        let counts = Counts::rows(phase.capacity()?);
        one(bind.launches(op.launches(&buffers, View::EMPTY, counts)?))
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::{Argument, RuntimeCounts};
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::super::tests::{any_gathers, phases, resolved};
    use super::*;
    use crate::dispatch::BufferBindings;
    use crate::ops::{Ops, Plan};
    use crate::pipeline::PreparedLaunch;
    use crate::program_lower::launches;
    use crate::switches::Switches;

    /// A span of its own buffer of `bytes`.
    fn span(buffer: u32, bytes: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            256,
            bytes,
        )
    }

    fn prepared(tp: Tp, dflash: bool) -> Ops {
        let switches = Switches::configured();
        Ops::prepare(
            &*crate::sm120::backend(tp, switches, true).unwrap(),
            &Plan {
                tp,
                switches,
                kv: kernel_core::contract::DType::Bf16,
                items: crate::contract::DECODE_ENTRY_ROWS,
                target: &crate::sm120::own_ops::Sm120Own::new(tp),
                state: &|_| [u64::MAX; 3],
                gathers: &any_gathers,
                dflash: dflash.then_some([u64::MAX; 2]),
                own: crate::ops::OwnPlan::new(
                    &crate::sm120::own_ops::Sm120Own::new(tp),
                    crate::contract::DECODE_ENTRY_ROWS,
                ),
            },
        )
        .unwrap()
    }

    /// The head norm as MiMo's DFlash program planned it: its row entry
    /// over the phase's capacity, the input at its row stride into compact
    /// rows.
    fn planned(
        tp: Tp,
        norm: HeadNorm,
        phase: Phase,
        live: u32,
        [input, weight, output, live_tokens]: [DeviceBufferSpanV1; 4],
    ) -> PreparedLaunch {
        let requirement = requirement(tp, norm);
        let RmsNormLayout::StridedHeads { input_stride, .. } = requirement.layout else {
            unreachable!()
        };
        let name = if norm == HeadNorm::Q {
            "dflash_q_norm"
        } else {
            "dflash_k_norm"
        };
        let symbol = crate::sm120::selection::drafter_row_symbol(tp, name);
        let entry = crate::sm120::launch::row::entries_for(tp)
            .find(|entry| entry.symbol == symbol)
            .unwrap();
        let mut bindings = BufferBindings::default();
        let arguments = vec![
            Argument::DeviceBuffer(bindings.bind(input)),
            Argument::DeviceBuffer(bindings.bind(weight)),
            Argument::DeviceBuffer(bindings.bind(output)),
            Argument::DeviceBuffer(bindings.bind(live_tokens)),
            Argument::U32(input_stride as u32),
            Argument::U32(requirement.width),
            Argument::U32((tp.geometry.dflash().rms_norm_eps as f32).to_bits()),
        ];
        let capacity = phase.capacity().unwrap();
        let (plan, arguments) = crate::sm120::launch::row::plan(
            entry,
            capacity,
            RuntimeCounts {
                m: live,
                batch: 1,
                context: 1,
                routed_rows: 8,
                split_k: 1,
            },
            arguments,
        )
        .unwrap();
        PreparedLaunch {
            symbol,
            rows: capacity,
            plan,
            arguments,
            bindings,
        }
    }

    /// The prepared head norms launch what MiMo's DFlash program planned
    /// for the block's Q and K heads at every verify bucket and the
    /// context's K heads at every phase, at TP2 and TP4; an instance
    /// without DFlash prepares none.
    #[test]
    fn head_norms_launch_what_the_dflash_program_planned() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            let ops = prepared(tp, true);
            for (phase, bucket) in phases() {
                let capacity = phase.capacity().unwrap();
                for norm in [HeadNorm::Q, HeadNorm::K, HeadNorm::ContextK] {
                    let requirement = requirement(tp, norm);
                    let (input_stride, _) = requirement.layout.strides(requirement.width);
                    let width = u64::from(requirement.width);
                    // Input and output hold exactly the capacity's rows.
                    let spans = [
                        span(1, ((u64::from(capacity) - 1) * input_stride + width) * 2),
                        span(2, 128 * 2),
                        span(3, u64::from(capacity) * width * 2),
                        span(4, 4),
                    ];
                    let [input, weight, output, live_tokens] = spans;
                    let lowered =
                        ops.head_norm(norm, phase, bucket, input, weight, output, live_tokens);
                    if norm != HeadNorm::ContextK && !matches!(phase, Phase::Verify(_)) {
                        assert!(lowered.is_err());
                        continue;
                    }
                    assert_eq!(
                        resolved(&launches(vec![lowered.unwrap()])),
                        resolved(&[planned(tp, norm, phase, 1, spans)]),
                        "{tp:?} {norm:?} {phase:?}"
                    );
                    compared += 1;
                    let short = span(
                        1,
                        ((u64::from(capacity) - 1) * input_stride + width) * 2 - 2,
                    );
                    assert!(
                        ops.head_norm(norm, phase, bucket, short, weight, output, live_tokens)
                            .is_err()
                    );
                }
            }
            let ops = prepared(tp, false);
            for (phase, bucket) in phases() {
                let any = DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1::from_parts(1, 1),
                    0,
                    u64::MAX / 2,
                );
                assert!(
                    ops.head_norm(HeadNorm::ContextK, phase, bucket, any, any, any, any)
                        .is_err()
                );
            }
        }
        assert!(compared > 20, "{compared}");
    }
}
