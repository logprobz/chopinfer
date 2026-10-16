//! Write maps for the grouped fixture's compared launch units.
use super::*;
use replay::{Checks, RouteBuffers, RouteGeometry};

impl FullCase {
    pub(super) fn route_only_checks<'a>(
        &'a self,
        session: &'a Session,
        block: usize,
        active: bool,
    ) -> LabResult<Checks<'a>> {
        Checks::new(session).route(
            RouteGeometry {
                routes: self.tokens * TOP_K,
                experts: 0..self.experts,
                block,
                sorted_offset: 0,
            },
            RouteBuffers {
                ids: &self.ids,
                weights: &self.topk,
                counts: &self.counts,
                offsets: &self.offsets,
                sorted: &self.sorted,
                padded: &self.padded,
                active: active.then_some(&self.active),
            },
            &[0x80],
        )
    }

    pub(super) fn replay_checks<'a>(
        &'a self,
        session: &'a Session,
        routing: bool,
        gate: bool,
        down: bool,
        combine: bool,
    ) -> LabResult<Checks<'a>> {
        self.replay_maps(
            session,
            [routing, gate, down, combine],
            self.use_fused_router(),
        )
    }

    pub(super) fn replay_maps<'a>(
        &'a self,
        session: &'a Session,
        parts: [bool; 4],
        fused_router: bool,
    ) -> LabResult<Checks<'a>> {
        let [routing, gate, down, combine] = parts;
        let routes = self.tokens * TOP_K;
        let mut checks = Checks::new(session);
        if routing {
            // row/router.cuh:237-337; moe/route.cuh; gemm/fragment_router.cuh.
            checks = self.route_only_checks(
                session,
                if self.route_module.is_some() {
                    16
                } else {
                    self.gate.knobs.tile_m as usize
                },
                self.route_module.is_some(),
            )?;
            if self.tiled_route() {
                checks = checks
                    .output(
                        "tile counts",
                        &self.tile_counts,
                        WriteMap::of(DType::I32)
                            .prefix((self.prepared_tokens * TOP_K).div_ceil(256) * self.experts)?,
                    )
                    .output(
                        "tile offsets",
                        &self.tile_offsets,
                        WriteMap::of(DType::I32).prefix(routes.div_ceil(256) * self.experts)?,
                    );
            }
            if let Some(router) = &self.router_gemm {
                checks = checks.output(
                    "router logits",
                    &self.logits,
                    WriteMap::of(DType::F32).prefix(self.tokens * self.experts)?,
                );
                if fused_router {
                    let tile = self.tokens.max(8);
                    let live = self.tokens;
                    checks =
                        checks.scratch_map("router partials", &router.partials, 0, move |_| {
                            // fragment_router.cuh:351-360: live columns in 1024 rows.
                            WriteMap::of(DType::F32).byte_ranges(
                                (0..1024).map(|row| row * tile * 4..(row * tile + live) * 4),
                            )
                        });
                } else {
                    checks = checks.scratch("router partials", &router.partials, 0);
                }
            }
        }
        if gate {
            checks = checks.output(
                "gate/up",
                &self.gate_output,
                WriteMap::of(DType::Bf16).byte_prefix(routes * self.gate.query.n as usize)?,
            );
            checks = projection_scratch(checks, self, &self.gate, &self.gate_partials)?;
        }
        if down {
            checks = checks.output(
                "weighted down",
                &self.down_output,
                WriteMap::of(DType::Bf16).prefix(routes * self.down.query.n as usize)?,
            );
            checks = projection_scratch(checks, self, &self.down, &self.down_partials)?;
        }
        if combine {
            checks = checks.output(
                "combined",
                &self.final_output,
                WriteMap::of(DType::Bf16).prefix(self.tokens * HIDDEN)?,
            );
        }
        Ok(checks)
    }
}

fn projection_scratch<'a>(
    checks: Checks<'a>,
    case: &'a FullCase,
    variant: &Variant,
    partials: &'a DeviceBuffer,
) -> LabResult<Checks<'a>> {
    projection_scratch_region(checks, case, variant, partials, partials.byte_len())
}

pub(super) fn projection_scratch_region<'a>(
    checks: Checks<'a>,
    case: &'a FullCase,
    variant: &Variant,
    partials: &'a DeviceBuffer,
    end: usize,
) -> LabResult<Checks<'a>> {
    let counts = case.counts(variant.knobs.split_k_capacity);
    if variant.reduction_plan(counts).map_err(err)?.is_some() {
        // moe.cuh:668-681,763-775: live route stride, gate/up pair interleaved.
        let bytes = counts.split_k as usize * case.tokens * TOP_K * variant.query.n as usize * 4;
        let map = WriteMap::of(DType::F32).byte_prefix(bytes)?;
        return Ok(
            checks.scratch_map_region(&variant.symbol, partials, 0, end, move |_| Ok(map.clone()))
        );
    }
    if variant.plan(counts).map_err(err)?.workspace_bytes != 0 {
        return Err(format!("{}: unknown arithmetic workspace", variant.symbol));
    }
    Ok(checks)
}
