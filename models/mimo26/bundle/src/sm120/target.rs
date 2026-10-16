//! MiMo's SM120 target (`crate::target::Target`): its backend and own
//! kernels, its catalog's attention queries and the SM120 facts generic code
//! reads, on the own-op set an instance builds at create.

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

use bundle_runtime::encoder::Encoder;
use bundle_runtime::geometry::Items;
use bundle_runtime::residency::SpinningEntry;
use kernel_core::backend::Backend;
use kernel_core::contract::{AttentionLayout, AttentionPhase, AttentionQuery, DType, Refusal};
use kernel_core::precision::execution::{Binding, BoundUse};
use kernel_core::precision::{OperationSpec, OperationUse};
use model_mimo26_contract::encoders::Encoders;
use model_mimo26_contract::own_ops::OwnOps;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span, HostServicesV1};

use super::own_ops::Sm120Own;
use crate::overlap::{Config, Facts, Lane};
use crate::pipeline::PreparedLaunch;
use crate::selection::RowBuckets;
use crate::switches::Switches;
use crate::target::{Registered, Target};

impl Target for Sm120Own {
    fn backend(&self, switches: Switches, wide: bool) -> Result<Box<dyn Backend>, Refusal> {
        super::backend(self.tp(), switches, wide)
    }

    fn own(&self) -> &dyn OwnOps {
        self
    }

    fn row_buckets(&self) -> RowBuckets {
        super::ROW_BUCKETS
    }

    fn folds_combine(&self, rows: u32) -> bool {
        super::folds_combine(self.tp(), rows)
    }

    fn fused_router(&self, rows: u32) -> bool {
        model_mimo26_contract::sm120::moe::prefer_fused_router(rows)
    }

    fn attention_query(
        &self,
        layout: AttentionLayout,
        phase: AttentionPhase,
        kv: DType,
    ) -> Result<AttentionQuery, Refusal> {
        super::attention::query(self.tp(), layout, phase, kv)
    }

    fn block_attention_query(&self) -> Result<AttentionQuery, Refusal> {
        super::attention::block_query(self.tp())
    }

    fn loads(&self, cubin: &str, switches: Switches) -> bool {
        !super::dense::medium_cubin(cubin) || switches.dense_mid
    }

    fn dynamic_shared_bytes(&self, symbol: &str, items: Items) -> Result<u32, Refusal> {
        super::modules::dynamic_shared_bytes(symbol, items)
    }

    fn spinning_entries(
        &self,
        rows: u32,
        lanes: Option<&Config>,
    ) -> Result<Vec<SpinningEntry>, Refusal> {
        super::comm::spinning_entries_for(self.tp(), rows, lanes)
    }

    fn lanes(&self) -> Option<Config> {
        super::overlap::config(self.tp())
    }

    fn adjust(&self, launch: &mut PreparedLaunch, facts: Facts, lane: Lane, config: &Config) {
        super::overlap::adjust(launch, facts, lane, config);
    }

    fn projection_backend(&self) -> Box<dyn Backend> {
        super::projection::backend(self.tp())
    }

    fn projection_rows(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Option<u32>>, String> {
        super::projection::rows(self.tp(), operation, usage)
    }

    fn registered_projection(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Option<Registered>, String> {
        super::projection::registered(self.tp(), operation, usage)
    }

    fn row_alternative(
        &self,
        id: &str,
        bound: &BoundUse,
        switches: Switches,
    ) -> Result<Option<(String, RangeInclusive<u32>)>, String> {
        super::projection::medium_alternative(id, bound, switches)
    }

    fn encoders(
        &self,
        services: &HostServicesV1,
        media: Encoders,
        weights: Vec<BTreeMap<String, Span>>,
        allocate: &mut dyn FnMut(usize, u64) -> Result<Span, BundleAbiError>,
    ) -> Result<Encoder, BundleAbiError> {
        super::encoders::create_for(self.tp(), services, media, weights, allocate)
    }
}
