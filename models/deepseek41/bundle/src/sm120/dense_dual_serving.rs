//! Expose the captured SM120 pair through the model-neutral prepared interface.
use crate::ops::dense_dual::{Buffers, Op};
use kernel_core::backend::{Backend, Counts, Launch, Need, OpInfo, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::properties::Invariance;
use kernels_sm120::{backend::Sm120, gemm::mxfp8_fold_dual::Entry};

impl crate::Kernels {
    pub(crate) fn prepare_dense_dual(
        shape: &crate::shape::Shape,
        need: Need,
    ) -> Result<Option<Op>, Refused> {
        if !super::dense_dual::enabled(shape.geometry, shape.ranks()) {
            return Ok(None);
        }
        let entry = super::dense_dual::entry();
        let kv = &entry.projections[1];
        let initializer = Sm120::new()
            .with_mxfp8_fold_stream(kv.clone())
            .mxfp8_fold_gemm_stream(&kv.ticket.query, false, need)?;
        // Persistent storage belongs to this weight, outside the stage workspace.
        let info = OpInfo::new(
            "native-dual-fold",
            Invariance::any_rows(1),
            vec![(entry.symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Some(Op::new(Box::new(Pair { entry, info }), initializer)))
    }
}

struct Pair {
    entry: Entry,
    info: OpInfo,
}
impl PreparedOp<Buffers> for Pair {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(&self, buffers: &Buffers, _: View, counts: Counts) -> Result<Vec<Launch>, Refusal> {
        if counts.rows != 1 {
            return Err(Refusal("combined target projections require one row"));
        }
        self.entry
            .launch(&buffers.projections, buffers.work, counts.rows, false)
            .map(|launch| vec![launch])
            .map_err(|_| Refusal("invalid dual fold binding"))
    }
}
