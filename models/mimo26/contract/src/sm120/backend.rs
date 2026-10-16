//! MiMo's SM120 kernels as the contract's backend ([`crate::backend::Kernels`]).

use super::{Sm120, comm, encoders, gemm, moe, prepared, workspace};
use crate::Tp;
use crate::backend::{Alternative, Kernels, RouteKernels, StateBytes};
use crate::capabilities::Projection;
use kernel_core::contract::{InstantiationSpec, Refusal, Target};
use kernel_core::precision::capability::{CapabilityDocument, OperationCapability};
use kernel_core::precision::{ExecutionPhase, OperationUse};

impl Kernels for Sm120 {
    const TARGET: Target = Target::Sm120f;

    const ARCHITECTURE: &'static str = "sm_120f";

    fn bundle_entry(tp: Tp, name: &str) -> String {
        super::bundle_symbol(tp, name)
    }

    fn bundle_cubin(tp: Tp, name: &str) -> String {
        format!("{}_{name}", crate::topology::kernel_tag(tp))
    }

    fn all_reduce_state(ranks: u32, rows: u32) -> Result<StateBytes, Refusal> {
        comm::fused_state_buffers(ranks, rows).map(|state| StateBytes {
            flags: state.flag_bytes,
            epoch: state.epoch_bytes,
        })
    }

    fn all_reduce_inbox_bytes(ranks: u32, rows: u32) -> Result<u64, Refusal> {
        comm::fused_state_buffers(ranks, rows).map(|state| state.inbox_bytes)
    }

    const GATHER_ROWS_STATE: StateBytes = StateBytes {
        flags: comm::GATHER_ROWS_FLAG_BYTES,
        epoch: comm::GATHER_ROWS_EPOCH_BYTES,
    };

    const EXCHANGE_ROWS_STATE: StateBytes = StateBytes {
        flags: comm::EXCHANGE_ROWS_FLAG_BYTES,
        epoch: comm::EXCHANGE_ROWS_EPOCH_BYTES,
    };

    const ROUTE_TICKET_BYTES: u64 = moe::PARALLEL_ROUTE_TICKET_BYTES;

    fn workspace_bytes(tp: Tp, items: bundle_runtime::geometry::Items) -> Result<u64, Refusal> {
        workspace::workspace_bytes(tp, items)
    }

    fn capabilities(
        geometry: &'static crate::geometry::Geometry,
    ) -> Result<CapabilityDocument, String> {
        crate::capabilities::sm120::inventory(geometry).map(|inventory| inventory.document)
    }

    fn drafter_capability(
        projection: Projection,
        tp: Tp,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal> {
        crate::capabilities::sm120::drafter_capability(projection, tp, usage)
    }

    type Entry = prepared::Entry;

    fn bind(
        tp: Tp,
        projection: Projection,
        usage: OperationUse,
        alternative: Alternative,
        capability: &mut OperationCapability,
    ) -> Result<prepared::Entry, String> {
        prepared::bind(tp, projection, usage, alternative, capability)
    }

    fn has_medium(entry: &prepared::Entry) -> Option<bool> {
        prepared::has_medium(entry)
    }

    fn dependency(tp: Tp) -> String {
        prepared::dependency(tp)
    }

    fn gate_up_entry(
        tp: Tp,
        usage: OperationUse,
        dflash: bool,
    ) -> Result<(String, &'static InstantiationSpec), Refusal> {
        let symbol = if dflash {
            let bucket = if usage.capacity == crate::serving::PREFILL_TOKENS {
                "prefill".into()
            } else {
                format!("m{}", usage.capacity)
            };
            format!(
                "chop_gemm_{}_dflash_gate_up_{bucket}_v1",
                crate::topology::kernel_tag(tp)
            )
        } else {
            crate::capabilities::sm120::symbol(Projection::MlpGateUp, tp, usage)
        };
        let entry = gemm::entry(&symbol).ok_or(Refusal("MiMo legacy dense entry is absent"))?;
        Ok((symbol, &entry.instantiation))
    }

    fn fused_router(usage: OperationUse) -> bool {
        usage.phase != ExecutionPhase::Prefill && moe::prefer_fused_router(usage.capacity)
    }

    fn route_kernels(tp: Tp, usage: OperationUse, fused: bool) -> RouteKernels {
        let wide = crate::serving::prefill_route(usage.phase, usage.capacity);
        let tag = tp.geometry.kernel_tag;
        let symbol = if fused {
            format!("{}_t{}_v1", moe::FUSED_ROUTER_PREFIX, usage.capacity)
        } else if wide {
            format!("chop_row_router_{tag}_sigmoid{}_v1", tp.geometry.experts())
        } else {
            moe::DECODE_ROUTE_SYMBOL.into()
        };
        let permutation = if wide {
            ["count", "prefix", "scan", "scatter"]
                .iter()
                .map(|name| format!("chop_row_permute_tile_{name}_{tag}_b64_v1"))
                .collect()
        } else {
            vec![]
        };
        RouteKernels {
            symbol,
            permutation,
        }
    }

    fn combine_symbol(tp: Tp) -> String {
        format!(
            "chop_row_combine_{}{}_v1",
            tp.geometry.kernel_tag,
            tp.geometry.hidden()
        )
    }

    fn encoder_symbols(
        geometry: &'static crate::geometry::Geometry,
    ) -> Result<Vec<String>, encoder_spec::Error> {
        // MiMo's own row kernels, then the shared catalog's entries.
        let mut symbols: Vec<String> = encoders::rows::kernels_for(geometry)?
            .into_iter()
            .map(|kernel| kernel.symbol)
            .collect();
        symbols.extend(
            encoders::kernels_for(geometry)?
                .iter()
                .map(|entry| entry.symbol.clone()),
        );
        Ok(symbols)
    }
}
