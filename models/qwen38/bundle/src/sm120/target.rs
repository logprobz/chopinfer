//! Qwen's SM120 target: [`Target`] over the SM120 backend, Qwen's own
//! kernels on SM120 and the entry selection of this module.

use bundle_runtime::lanes::{Config, Lane, LaunchFacts};
use bundle_runtime::projection::Needs;
use bundle_runtime::residency::SpinningEntry;
use kernel_core::backend::{Backend, Refused};
use kernel_core::contract::{AttentionPhase, Refusal};
use kernel_core::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery};
use model_qwen38_contract::own_ops::OwnOps;

use super::catalog;
use super::own_ops::Sm120Own;
use crate::ops::Key;
use crate::pipeline::PreparedLaunch;
use crate::scratch::{CommBytes, Served};
use crate::target::{EncoderKernels, Target, Uses};

impl Target for Sm120Own {
    fn backend(&self, wide_comm: bool) -> Box<dyn Backend> {
        super::backend(self.shape(), wide_comm)
    }

    fn own(&self) -> &dyn OwnOps {
        self
    }

    fn expert_contract(
        &self,
        projection: GatedMlpProjection,
        key: Key,
    ) -> Result<GatedMlpProjectionQuery, Refused> {
        super::expert_contract(self.shape(), projection, key)
    }

    fn partials_phase(&self, key: Key) -> AttentionPhase {
        super::partials_phase(self.shape(), key)
    }

    fn lanes(&self) -> Option<Config> {
        super::overlap::config(self.shape().ranks())
    }

    fn adjust(&self, launch: &mut PreparedLaunch, facts: LaunchFacts, lane: Lane, config: &Config) {
        super::overlap::adjust(launch, facts, lane, config);
    }

    fn built(&self, launch: &PreparedLaunch) -> LaunchFacts {
        super::overlap::built(self.shape(), launch)
    }

    fn loads(&self, cubin: &str, uses: Uses) -> bool {
        super::loads(cubin, uses)
    }

    fn loads_symbol(&self, symbol: &str, uses: Uses) -> bool {
        super::loads_symbol(symbol, uses)
    }

    fn comm_bytes(&self, served: Served) -> CommBytes {
        super::comm_bytes(self.shape(), served)
    }

    fn workspace_bytes(&self, served: Served) -> Result<u64, Refusal> {
        catalog::workspace_bytes(self.shape(), served)
    }

    fn dynamic_shared_bytes(&self, symbol: &str) -> Result<u32, Refusal> {
        catalog::dynamic_shared_bytes(symbol)
    }

    fn spinning_entries(
        &self,
        rows: u32,
        f32_norms: bool,
        lanes: Option<&Config>,
    ) -> Result<Vec<SpinningEntry>, Refusal> {
        catalog::spinning_entries(self.shape(), rows, f32_norms, lanes)
    }

    fn registered_dense(&self, symbol: &str) -> Result<Needs, String> {
        super::registered_dense(symbol)
    }

    fn registered_experts(&self, symbol: &str) -> Result<(GatedMlpProjectionQuery, u64), String> {
        super::registered_experts(symbol)
    }

    fn encoder(&self) -> Result<EncoderKernels, String> {
        super::encoder(self.shape())
    }
}
