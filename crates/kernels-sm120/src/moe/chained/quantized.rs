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
        };
        result.padded_rows(tokens_capacity)?;
        Ok(result)
    }

    pub fn gate_up(&self) -> bool {
        self.query.projection.projection == GatedMlpProjection::GateUp
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
        if tokens == 0 || tokens > self.tokens_capacity {
            return Err(Refusal("quantized chained launch exceeds token capacity"));
        }
        let plan = LaunchPlan {
            grid: [
                self.padded_rows(tokens)? / self.row_tile,
                self.columns() / TILE_N,
                1,
            ],
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
    /// output. Only down's unused route-weight pointer may be null.
    pub fn arguments(&self, handles: [u64; 10]) -> Result<ArgumentList, Refusal> {
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
        let body = format!(
            "chop::moe::chained::run<chop::moe::chained::quantized::Config<chop::moe::chained::Projection::{projection},{},{},{},{},{},{},{clamp}>>(reinterpret_cast<const uint16_t*>(input),weight,nullptr,weight_scales,nullptr,offsets,sorted_rows,padded_rows,live_tokens,route_weights,output,activation_scales);",
            grouped.experts,
            grouped.top_k,
            self.columns(),
            g.k,
            self.row_tile,
            self.tokens_capacity
        );
        EntrySpec::new(
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
        )
    }
}

pub fn cubin_spec(name: &str, projections: &[Projection]) -> CubinSpec {
    super::spec(
        name.into(),
        "chained_quantized.cuh",
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
