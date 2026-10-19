//! The chained engine's MXFP4 / ceil-MXFP8 projections. Stored K and loop
//! K are separate, and each route is weighted in the gate/up epilogue.
//! Gate/up writes its BF16 intermediate. The prepared backend operation
//! follows it with the shared ceil quantizer before down reads its codes.

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, Refusal, RoutedCapacity, Validate,
};
use kernel_core::gated_mlp::{GatedMlpProjection, SwigluClamp};
use kernel_core::moe_chained::ChainedMlpProjectionQuery;
use kernel_core::properties::Invariance;

use super::{ROW_TILES, THREADS, TILE_N};
use crate::moe::ptr;

#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub symbol: String,
    pub query: ChainedMlpProjectionQuery,
    pub row_tile: u32,
    pub tokens_capacity: u32,
    /// Compact route tiles, four ordered K chains and async operand stages.
    pub active: bool,
    /// Bound the old column grid by nonempty route tiles, keeping its cubin.
    pub compact_columns: bool,
    /// Model-owned selection limit, independent of the compiled capacity.
    pub max_rows: u32,
}

impl Projection {
    pub fn new(
        query: ChainedMlpProjectionQuery,
        row_tile: u32,
        tokens_capacity: u32,
    ) -> Result<Self, Refusal> {
        query.validate()?;
        let p = query.projection;
        let g = p.gemm;
        let grouped = g.grouped.expect("contract requires grouping");
        let n = if p.projection == GatedMlpProjection::GateUp {
            g.n / 2
        } else {
            g.n
        };
        if !ROW_TILES.contains(&row_tile)
            || !n.is_multiple_of(TILE_N)
            || tokens_capacity == 0
            || tokens_capacity > i32::MAX as u32
            || grouped.experts > i32::MAX as u32
            || grouped.top_k > 32
            || g.k > i32::MAX as u32
            || g.n > i32::MAX as u32
        {
            return Err(Refusal(
                "quantized chained geometry exceeds its column schedule",
            ));
        }
        let routes = tokens_capacity
            .checked_mul(grouped.top_k)
            .ok_or(Refusal("quantized chained route count overflows"))?;
        if routes > g.m_capacity || !matches!(grouped.capacity, RoutedCapacity::TotalRows(_)) {
            return Err(Refusal(
                "quantized chained capacity needs enough total routed rows",
            ));
        }
        let clamp = p.swiglu_clamp.map_or(String::new(), |c| {
            format!("_clamp{:08x}{:08x}", c.gate_max_bits, c.up_limit_bits)
        });
        let kind = if p.projection == GatedMlpProjection::GateUp {
            "gateup"
        } else {
            "down"
        };
        let symbol = format!(
            "chop_moe_chain_mxfp4_a8ceil_before_{kind}_e{}k{}_n{}_k{}_r{row_tile}_t{tokens_capacity}{clamp}_v1",
            grouped.experts, grouped.top_k, g.n, g.k
        );
        let result = Self {
            symbol,
            query,
            row_tile,
            tokens_capacity,
            active: false,
            compact_columns: false,
            max_rows: tokens_capacity,
        };
        result.padded_rows(tokens_capacity)?;
        Ok(result)
    }

    pub fn gate_up(&self) -> bool {
        self.query.projection.projection == GatedMlpProjection::GateUp
    }

    pub fn with_max_rows(mut self, rows: u32) -> Result<Self, Refusal> {
        if rows == 0 || rows > self.tokens_capacity {
            return Err(Refusal("chained selection limit exceeds compiled capacity"));
        }
        self.max_rows = rows;
        Ok(self)
    }

    /// Reuse the column kernel with a tighter graph-stable grid. Every
    /// padded tile contains at least one route, so tiles <= tokens * top_k.
    /// This retains 64 columns per CTA and lets CUDA schedule all work,
    /// without a persistent cap, warp-chain handovers or extra K padding.
    pub fn with_compact_columns(mut self) -> Result<Self, Refusal> {
        if self.active {
            return Err(Refusal("compact columns require the column entry"));
        }
        self.compact_columns = true;
        Ok(self)
    }

    pub fn with_active_experts(mut self) -> Result<Self, Refusal> {
        if self.compact_columns {
            return Err(Refusal("active block warps cannot use the column grid"));
        }
        let grouped = self.query.projection.gemm.grouped.expect("routed contract");
        if u64::from(self.tokens_capacity)
            * u64::from(grouped.top_k)
            * u64::from(self.columns() / 16)
            > i32::MAX as u64
        {
            return Err(Refusal("active chained item count overflows"));
        }
        if self.row_tile != 16 {
            return Err(Refusal("active chained projections require 16-row tiles"));
        }
        if !self.active {
            self.symbol = format!("{}_active_bw4_v1", self.symbol.trim_end_matches("_v1"));
            self.active = true;
        }
        Ok(self)
    }

    pub fn columns(&self) -> u32 {
        self.query.projection.gemm.n / if self.gate_up() { 2 } else { 1 }
    }

    pub fn padded_rows(&self, tokens: u32) -> Result<u32, Refusal> {
        let g = self.query.projection.gemm.grouped.expect("routed contract");
        tokens
            .checked_mul(g.top_k)
            .and_then(|r| {
                g.experts
                    .checked_mul(self.row_tile - 1)
                    .and_then(|p| r.checked_add(p))
            })
            .and_then(|r| r.checked_next_multiple_of(self.row_tile))
            .filter(|&r| r <= i32::MAX as u32)
            .ok_or(Refusal("quantized chained padded row count overflows"))
    }

    pub fn invariance(&self) -> Invariance {
        Invariance::any_rows(self.tokens_capacity)
    }

    pub fn plan(&self, tokens: u32) -> Result<LaunchPlan, Refusal> {
        self.query.validate()?;
        if tokens == 0 || tokens > self.tokens_capacity || tokens > self.max_rows {
            return Err(Refusal("quantized chained launch exceeds token capacity"));
        }
        let plan = LaunchPlan {
            grid: if self.active {
                // Every nonempty tile owns at least one route. The grid is
                // fixed for a graph bucket and walks only live tile work.
                let routes = u64::from(tokens)
                    * u64::from(self.query.projection.gemm.grouped.unwrap().top_k);
                [
                    (routes * u64::from(self.columns() / 16)).min(376) as u32,
                    1,
                    1,
                ]
            } else {
                [
                    if self.compact_columns {
                        (self.padded_rows(tokens)? / self.row_tile)
                            .min(tokens * self.query.projection.gemm.grouped.unwrap().top_k)
                    } else {
                        self.padded_rows(tokens)? / self.row_tile
                    },
                    self.columns() / TILE_N,
                    1,
                ]
            },
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Input codes, weight codes, weight scales, activation scales,
    /// offsets, sorted routes, padded count, live tokens, route weights,
    /// output. Only down's unused route-weight pointer may be null. The
    /// active schedule also requires its compact list (`active_arguments`).
    pub fn arguments(&self, handles: [u64; 10]) -> Result<ArgumentList, Refusal> {
        if self.active {
            return Err(Refusal(
                "active chained arguments require the compact expert list",
            ));
        }
        self.base_arguments(handles)
    }

    pub fn active_arguments(
        &self,
        handles: [u64; 10],
        active_experts: u64,
    ) -> Result<ArgumentList, Refusal> {
        if !self.active || active_experts == 0 {
            return Err(Refusal(
                "active chained arguments require an active entry and expert list",
            ));
        }
        let mut args = self.base_arguments(handles)?;
        args.0.push(Argument::DeviceBuffer(active_experts));
        args.validate()?;
        Ok(args)
    }

    fn base_arguments(&self, handles: [u64; 10]) -> Result<ArgumentList, Refusal> {
        let required = [
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            self.gate_up(),
            true,
        ];
        let values = handles
            .into_iter()
            .zip(required)
            .map(|(h, required)| match (h, required) {
                (0, true) => Err(Refusal("quantized chained argument missing")),
                (0, false) => Ok(Argument::NullDevicePointer),
                (h, _) => Ok(Argument::DeviceBuffer(h)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }

    pub fn entry(&self) -> EntrySpec {
        let p = self.query.projection;
        let g = p.gemm;
        let grouped = g.grouped.expect("routed contract");
        let clamp = match p.swiglu_clamp {
            Some(SwigluClamp {
                gate_max_bits,
                up_limit_bits,
            }) => format!(
                "chop::moe::clamp::SwigluClamp<true,{gate_max_bits:#x}u,{up_limit_bits:#x}u>"
            ),
            None => "chop::moe::clamp::SwigluClamp<false,0u,0u>".into(),
        };
        let projection = if self.gate_up() { "gate_up" } else { "down" };
        let body = if self.active {
            format!(
                "chop::moe::active_quantized::run<chop::moe::chained::quantized::Config<chop::moe::chained::Projection::{projection},{},{},{},{},{},{},{clamp}>>(input,weight,weight_scales,activation_scales,offsets,sorted_rows,padded_rows,live_tokens,route_weights,output,active_experts);",
                grouped.experts,
                grouped.top_k,
                self.columns(),
                g.k,
                self.row_tile,
                self.tokens_capacity
            )
        } else {
            format!(
                "chop::moe::chained::run<chop::moe::chained::quantized::Config<chop::moe::chained::Projection::{projection},{},{},{},{},{},{},{clamp}>>(reinterpret_cast<const uint16_t*>(input),weight,nullptr,weight_scales,nullptr,offsets,sorted_rows,padded_rows,live_tokens,route_weights,output,activation_scales);",
                grouped.experts,
                grouped.top_k,
                self.columns(),
                g.k,
                self.row_tile,
                self.tokens_capacity
            )
        };
        let mut entry = EntrySpec::new(
            self.symbol.clone(),
            vec![
                ptr("input", "const uint8_t"),
                ptr("weight", "const uint8_t"),
                ptr("weight_scales", "const uint8_t"),
                ptr("activation_scales", "const uint8_t"),
                ptr("offsets", "const int32_t"),
                ptr("sorted_rows", "const int32_t"),
                ptr("padded_rows", "const int32_t"),
                ptr("live_tokens", "const int32_t"),
                ptr("route_weights", "const float"),
                ptr("output", "void"),
            ],
            body,
            THREADS,
            1,
        );
        if self.active {
            entry.params.push(ptr("active_experts", "const int32_t"));
        }
        entry
    }
}

pub fn cubin_spec(name: &str, projections: &[Projection]) -> CubinSpec {
    assert!(projections.iter().all(|p| !p.active));
    super::spec(
        name.into(),
        "chained_quantized.cuh",
        projections.iter().map(Projection::entry).collect(),
    )
}

pub fn active_cubin_spec(name: &str, projections: &[Projection]) -> CubinSpec {
    assert!(projections.iter().all(|p| p.active));
    super::spec(
        name.into(),
        "chained_active_quantized.cuh",
        projections.iter().map(Projection::entry).collect(),
    )
}

/// Every E2M1 pair under its UE8M0/32 scale, with the exact overflow
/// exclusions the device decoder enforces. Signed zeros remain admitted.
pub fn admit_mxfp4_payload(codes: &[u8], scales: &[u8]) -> Result<(), Refusal> {
    if scales.len().checked_mul(16) != Some(codes.len()) {
        return Err(Refusal("MXFP4 needs one UE8M0 scale per 16 packed bytes"));
    }
    for (group, &scale) in codes.chunks_exact(16).zip(scales) {
        if scale == 255 {
            return Err(Refusal("MXFP4 UE8M0 scale byte 255 is not finite"));
        }
        if scale > 252 {
            let limit = if scale == 253 { 6 } else { 4 };
            if group
                .iter()
                .any(|&v| v & 7 >= limit || (v >> 4) & 7 >= limit)
            {
                return Err(Refusal("MXFP4 scaled weights overflow BF16"));
            }
        }
    }
    Ok(())
}

/// MXFP8 activations retain the same finite and BF16-exact admission as
/// MXFP8 weight payloads; a ceil scale rule never relaxes this domain.
pub use super::formats::admit_mxfp8_payload as admit_activations;
