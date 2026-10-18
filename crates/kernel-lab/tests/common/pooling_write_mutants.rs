//! Redirect only the prepared symbols to device mutants. The queries,
//! buffers, counts, acceptance cases and their bounds remain unchanged.

use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::compressed_kv::{CompressedKvWriteBuffers, CompressedKvWriteQuery};
use kernel_core::contract::Refusal;
use kernel_core::gated_pooling::{GatedPoolingBuffers, GatedPoolingQuery};
use kernel_core::pooled_indexer::{PositionIndexerQuery, PositionKeyWriteBuffers};
use kernel_lab::families::pooling_write_mutants::Fault;
use kernel_lab::kit::{OpResult, Outcome, Report};
use kernels_sm120::backend::Sm120;

pub struct Mutant {
    pub backend: Sm120,
    pub fault: Fault,
}

struct Op<B> {
    inner: Prepared<B>,
    info: OpInfo,
    fault: Fault,
}

impl<B: 'static> Op<B> {
    fn wrap(inner: Prepared<B>, fault: Fault) -> Prepared<B> {
        let mut info = inner.info().clone();
        for (symbol, _) in &mut info.symbols {
            *symbol = fault.symbol(symbol);
        }
        Box::new(Self { inner, info, fault })
    }
}

impl<B> PreparedOp<B> for Op<B> {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(buffers, workspace, counts)?;
        for launch in &mut launches {
            launch.symbol = self.fault.symbol(&launch.symbol);
        }
        Ok(launches)
    }
}

impl Backend for Mutant {
    fn name(&self) -> &str {
        "pooling-writer-device-control"
    }
    fn gated_pooling(
        &self,
        q: &GatedPoolingQuery,
        need: Need,
    ) -> Result<Prepared<GatedPoolingBuffers<View>>, Refused> {
        self.backend
            .gated_pooling(q, need)
            .map(|op| Op::wrap(op, self.fault))
    }
    fn position_indexer_key_write(
        &self,
        q: &PositionIndexerQuery,
        need: Need,
    ) -> Result<Prepared<PositionKeyWriteBuffers<View>>, Refused> {
        self.backend
            .position_indexer_key_write(q, need)
            .map(|op| Op::wrap(op, self.fault))
    }
    fn compressed_kv_write(
        &self,
        q: &CompressedKvWriteQuery,
        need: Need,
    ) -> Result<Prepared<CompressedKvWriteBuffers<View>>, Refused> {
        self.backend
            .compressed_kv_write(q, need)
            .map(|op| Op::wrap(op, self.fault))
    }
}

/// A refusal, a missing case or a launch error is not mutant detection.
pub fn rejected(report: &Report, count: usize, witness: &str) {
    assert_eq!(report.prepared().len(), count);
    assert_eq!(report.ops.len(), count);
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("{} did not prepare", op.name)
        };
        assert!(checks.iter().any(|check| check.name == "conformance"
            && matches!(&check.outcome, Outcome::Failed(error)
                if error.contains(witness) && !error.contains("CUDA") && !error.contains("unbound"))),
            "{}: mutant was not rejected at {witness}: {:?}", op.name, checks);
    }
}
