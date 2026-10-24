//! The shared expert's activation (`model_deepseek41_contract::moe::
//! shared_activation`): the clamped SwiGLU of its F32 gate and up columns,
//! rounded once to BF16 and quantized to E4M3 per 32 with a ceil UE8M0
//! scale, which its down projection reads. Its projections are dense
//! (`super::dense`).

use kernel_core::backend::{Backend, Counts, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::gated_mlp::quantized::SiluMulQuantizeBuffers;
use model_deepseek41_contract::moe;
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::row_buckets::RowBuckets;
use super::{Asks, Bind, Lowered, exact};
use crate::program::Phase;
use crate::shape::Shape;

pub(crate) struct SharedOps {
    activation: RowBuckets<SiluMulQuantizeBuffers<View>>,
}

/// A step's spans of the activation: the F32 gate and up rows, the codes
/// and scales, and the phase's row status (`Scratch::row_status`), which a
/// non-finite operand or an unrepresentable product flags, zeroing the row.
pub(crate) struct ActivationViews {
    pub gate: DeviceBufferSpanV1,
    pub up: DeviceBufferSpanV1,
    pub codes: DeviceBufferSpanV1,
    pub scales: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub status: DeviceBufferSpanV1,
}

impl SharedOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<SharedOps, Refused> {
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(shape.verify_rows));
        let query = moe::shared_activation(shape.geometry, shape.ranks());
        let activation = RowBuckets::prepare(asks, "shared.activation", every, |need| {
            backend.silu_mul_quantize(&query, need)
        })?;
        Ok(SharedOps { activation })
    }

    /// The activation of `rows` rows.
    pub(crate) fn activate(
        &self,
        phase: Phase,
        views: &ActivationViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let rows = phase.rows();
        let mut bind = Bind::default();
        let buffers = SiluMulQuantizeBuffers {
            gate: bind.view(views.gate),
            up: bind.view(views.up),
            codes: bind.view(views.codes),
            scales: bind.view(views.scales),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: 0,
        };
        bind.launch(
            self.activation.at_phase(phase),
            &buffers,
            Counts::rows(rows),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;
    use model_deepseek41_contract::state::KvStorage;

    /// SM120 prepares the activation at each rank count's slice; the
    /// shared projections wait with the other dense projections.
    #[test]
    fn sm120_activates_the_shared_expert_at_every_rank_count() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops =
                super::super::Ops::prepare(&kernels, &shape, KvStorage::Native, Default::default());
            assert!(ops.shared.is_some(), "{:?}", ops.missing);
            let ids: Vec<&str> = super::super::dense::projections(&shape)
                .iter()
                .map(|d| d.id)
                .collect();
            for id in ["moe.shared-gate", "moe.shared-up", "moe.shared-down"] {
                assert!(ids.contains(&id));
            }
        }
    }
}
