//! Router preparation and ordered r2 finish for separate cached F32 spans.
use super::{floor::Entry, router_finish};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

pub use router_finish::{STAGES, Stage};
pub const DOMAIN_BYTES: u64 = 256;

/// Independent domain, chain partials and arrival storage precede route scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    pub partials: u64,
    pub partial_bytes: u64,
    pub arrival: u64,
    pub routed: u64,
}

impl Layout {
    pub fn new(e: &Entry, rows: u32) -> Result<Self, Refusal> {
        plan(e, Stage::Finish, rows)?;
        let partial_bytes = u64::from(rows)
            .checked_mul(u64::from(e.experts()))
            .and_then(|v| v.checked_mul(u64::from(e.hidden().div_ceil(128))))
            .and_then(|v| v.checked_mul(4))
            .ok_or(Refusal("wide router partial bytes overflow"))?;
        let arrival = DOMAIN_BYTES
            .checked_add(partial_bytes)
            .and_then(|v| v.checked_next_multiple_of(256))
            .ok_or(Refusal("wide router arrival offset overflows"))?;
        Ok(Self {
            partials: DOMAIN_BYTES,
            partial_bytes,
            arrival,
            routed: arrival
                .checked_add(256)
                .ok_or(Refusal("wide router scratch offset overflows"))?,
        })
    }
}

pub fn plan(e: &Entry, stage: Stage, rows: u32) -> Result<LaunchPlan, Refusal> {
    if !(33..=64).contains(&rows) || e.experts() % 32 != 0 || e.top_k() > 8 {
        return Err(Refusal(
            "wide router finish requires 33..64 rows and whole expert warps",
        ));
    }
    let plan = LaunchPlan {
        grid: [rows, 1, 1],
        block: [
            match stage {
                Stage::Prepare => 128,
                Stage::Finish => e.experts().max(64 * e.top_k()),
            },
            1,
            1,
        ],
        cluster: [1; 3],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn entry(e: &Entry, stage: Stage) -> EntrySpec {
    let mut spec = match stage {
        Stage::Prepare => router_finish::entry(e, stage),
        Stage::Finish => router_finish::r2_entry(e, 7, false),
    };
    spec.symbol = spec
        .symbol
        .replace("_router_finish_", "_router_finish_m64_");
    for param in &mut spec.params {
        if let EntryParam::DevicePointer { name, cxx_type } = param
            && name == "workspace"
        {
            match stage {
                Stage::Prepare => {
                    *name = "row_domain".into();
                    *cxx_type = "uint32_t".into();
                }
                Stage::Finish => *name = "partials".into(),
            }
        }
    }
    spec.body = match stage {
        Stage::Prepare => format!(
            "chop::moe::router_finish64::prepare<{}>(input,live,rows,row_domain,arrived);",
            e.hidden()
        ),
        Stage::Finish => format!(
            "chop::moe::router_finish64::finish<{},{},{}>(partials,logits,live,rows,bias,alternate_bias,bias_select,scores,ids,weights,__uint_as_float(scale_bits),counts,offsets,sorted,padded,active,scratch,gate_tickets,arrived);",
            e.experts(),
            e.top_k(),
            e.hidden()
        ),
    };
    spec
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    super::spec(
        name.into(),
        "router_finish64.cuh",
        entries
            .iter()
            .flat_map(|e| STAGES.into_iter().map(move |stage| entry(e, stage)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::chained::quantized::Projection;
    use kernel_core::contract::{
        ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, OutputQuantization,
        RoutedCapacity, SwigluColumns, WeightFormat,
    };
    use kernel_core::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint};
    use kernel_core::moe_chained::{ChainedK, ChainedMlpProjectionQuery};

    fn fixture() -> Entry {
        let projection = |gate: bool| {
            let k = if gate { 5120 } else { 576 };
            Projection::new(
                ChainedMlpProjectionQuery {
                    projection: GatedMlpProjectionQuery {
                        gemm: GemmQuery {
                            m_capacity: 64 * 6,
                            n: if gate { 1152 } else { 5120 },
                            k,
                            weight: WeightFormat::Mxfp4,
                            activation: ActivationFormat::Mxfp8,
                            output: if gate { DType::Bf16 } else { DType::F32 },
                            epilogues: Epilogues {
                                swiglu: gate.then_some(SwigluColumns::ConcatenatedHalves),
                                output_quantization: gate.then_some(OutputQuantization::Mxfp8),
                                ..Default::default()
                            },
                            grouped: Some(GroupedGemm {
                                experts: 384,
                                top_k: 6,
                                capacity: RoutedCapacity::TotalRows(64 * 6),
                            }),
                        },
                        projection: if gate {
                            GatedMlpProjection::GateUp
                        } else {
                            GatedMlpProjection::Down
                        },
                        swiglu_clamp: None,
                        route_weight: Some(RouteWeightPoint::BeforeDown),
                    },
                    k: ChainedK::new(k).unwrap(),
                },
                64,
                64,
            )
            .unwrap()
            .with_streamed_chains()
            .unwrap()
        };
        Entry::new(
            projection(true),
            projection(false),
            1e-4_f32.to_bits(),
            "chop_test_moe_floor_e384_k6".into(),
        )
        .unwrap()
    }

    #[test]
    fn every_wide_row_separates_domain_partials_arrival_and_routes() {
        let e = fixture();
        for rows in 33..=64 {
            let layout = Layout::new(&e, rows).unwrap();
            assert_eq!(layout.partials, 256);
            assert_eq!(layout.partial_bytes, 61440 * u64::from(rows));
            assert!(layout.partials + layout.partial_bytes <= layout.arrival);
            assert!(layout.arrival + 256 <= layout.routed);
            for offset in [layout.partials, layout.arrival, layout.routed] {
                assert!(offset.is_multiple_of(256));
            }
            for stage in STAGES {
                let p = plan(&e, stage, rows).unwrap();
                assert_eq!(p.grid, [rows, 1, 1]);
                assert!(p.pdl);
                assert!(router_finish::plan(&e, stage, rows).is_err());
            }
        }
        for rows in [0, 1, 16, 32, 65, u32::MAX] {
            assert!(Layout::new(&e, rows).is_err());
        }
    }

    #[test]
    fn wide_entries_name_independent_planes_without_changing_small_abi() {
        let e = fixture();
        let prepare = entry(&e, Stage::Prepare);
        let finish = entry(&e, Stage::Finish);
        assert_eq!(prepare.params.len(), 5);
        assert_eq!(finish.params.len(), 19);
        assert_eq!(
            prepare.params[3],
            EntryParam::DevicePointer {
                name: "row_domain".into(),
                cxx_type: "uint32_t".into(),
            }
        );
        assert_eq!(
            finish.params[0],
            EntryParam::DevicePointer {
                name: "partials".into(),
                cxx_type: "const float".into(),
            }
        );
        assert!(prepare.body.ends_with("rows,row_domain,arrived);"));
        assert!(finish.body.contains("finish<384,6,5120>(partials,"));
        assert!(finish.symbol.contains("_router_finish_m64_"));
        assert!(finish.symbol.ends_with("_finish_r2v7_v1"));
        for stage in STAGES {
            let old = router_finish::entry(&e, stage);
            assert!(!old.symbol.contains("_m64_"));
            assert!(old.body.contains("workspace"));
            assert!(router_finish::plan(&e, stage, 32).is_ok());
        }
    }
}
