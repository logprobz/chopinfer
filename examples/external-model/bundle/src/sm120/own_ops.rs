//! The example's own kernel on SM120: [`OwnOps`] over the probe entry the
//! build compiles (`selection.rs`).

use super::ops::{OwnOps, ProbeBuffers, SigmoidProbe};
use super::selection::{PROBE_SYMBOL, PROBE_THREADS};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;

/// The example's own kernels as its build compiles them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sm120Own;

impl OwnOps for Sm120Own {
    fn name(&self) -> &str {
        "sm120"
    }

    fn sigmoid_probe(
        &self,
        _op: &SigmoidProbe,
        need: Need,
    ) -> Result<Prepared<ProbeBuffers<View>>, Refused> {
        if need.capacity != 1 {
            return Err(Refused::Rule("the sigmoid probe launches one row".into()));
        }
        let info = OpInfo::new(
            "sm120.example.sigmoid-probe.v1",
            Invariance::NONE,
            vec![(PROBE_SYMBOL.into(), 0)],
        );
        need.check(&info)?;
        Ok(Box::new(Probe { info }))
    }
}

/// One block of [`PROBE_THREADS`]; thread zero writes the result.
struct Probe {
    info: OpInfo,
}

impl PreparedOp<ProbeBuffers<View>> for Probe {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &ProbeBuffers<View>,
        _workspace: View,
        _counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        buffers.result.covers(4)?;
        Ok(vec![Launch {
            symbol: PROBE_SYMBOL.into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1; 3],
                block: [PROBE_THREADS, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1; 3],
                pdl: false,
            },
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(buffers.result.handle),
                Argument::U32(buffers.value),
            ]),
            collective: false,
            persistent: false,
        }])
    }
}
