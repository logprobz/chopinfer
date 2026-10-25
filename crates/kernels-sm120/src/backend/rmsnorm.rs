//! RMSNorm through the BF16 row builder, and per-head RMSNorms of strided
//! heads through the row engine entries a model instantiates of the
//! per-head norm template (`dflash_qk_norm`, `row/dflash.cuh`).

use kernel_build::EntryParam;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{
    Argument, DType, LaunchPlan, Refusal, RmsNormKind, RuntimeCounts, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::rmsnorm::{
    FusedResidual, RmsNormBuffers, RmsNormLayout, RmsNormNumerics, RmsNormRequirement,
};

use super::check_plan;
use super::rows::{EngineOp, row_launch};
use super::select::{Candidate, Chosen, choose};
use crate::row::rmsnorm::{RmsNorm, RmsNormTarget};
use crate::row::{Grid, RowInstantiation, smallest_bucket};

/// The implementation a prepared RMSNorm with the warp-tree numerics
/// reports.
const TREE_256: &str = "sm120.row.rmsnorm.tree256.v1";
/// The implementation a prepared RMSNorm with HF's ordered numerics reports.
const HF_ORDERED: &str = "sm120.row.rmsnorm.hf-ordered.v1";

/// A requirement with the numerics the row RMSNorm computes for it: its own
/// when it pins them, otherwise those of its kind.
fn resolved(requirement: &RmsNormRequirement) -> RmsNormRequirement {
    RmsNormRequirement {
        numerics: Some(
            requirement
                .numerics
                .unwrap_or(RmsNorm::numerics(requirement.kind)),
        ),
        ..*requirement
    }
}

/// The RMSNorm of `op` through the model's entry for it: entries of equal
/// requirements and different symbols are a tie. Numerics an op or an entry
/// leaves to the backend are the row RMSNorm's for its kind, so such an op
/// launches what today's requirement pinning them does.
pub(super) fn prepare(
    entries: &[(RmsNormRequirement, String)],
    op: &RmsNormRequirement,
    need: Need,
) -> Result<Prepared<RmsNormBuffers<View>>, Refused> {
    let op = resolved(op);
    let symbol = choose(
        "RMSNorm",
        entries
            .iter()
            .filter(|(requirement, _)| resolved(requirement) == op)
            // Every entry claims what the row RMSNorm keeps.
            .map(|entry| {
                let guarantees = Invariance::any_rows(need.capacity);
                Candidate::new(entry.1.clone(), &entry.1, launch(entry), guarantees)
            }),
        need,
    )?
    .entry;
    let op = RmsNormOp::new(op, symbol, need.capacity)?;
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// What an RMSNorm entry launches: the symbol of the row RMSNorm its
/// requirement builds.
fn launch(entry: &(RmsNormRequirement, String)) -> String {
    let (_requirement, symbol) = entry; // the requirement selects the op
    symbol.clone()
}

/// The BF16 row RMSNorm (`row/row.cuh`): one block per row, so every row
/// has the bits it has alone at any row count and chunk split.
struct RmsNormOp {
    norm: RmsNorm,
    symbol: String,
    capacity: u32,
    /// The plan at the prepared capacity; a launch grids its own rows.
    plan: LaunchPlan,
    info: OpInfo,
}

impl RmsNormOp {
    fn new(
        requirement: RmsNormRequirement,
        symbol: String,
        capacity: u32,
    ) -> Result<Self, Refused> {
        let norm = RmsNorm::new(requirement, capacity, RmsNormTarget::SM120F)?;
        let plan = norm.plan(capacity)?;
        check_plan(&plan)?;
        let implementation = if norm.requirement().numerics == Some(RmsNormNumerics::HF_ORDERED) {
            HF_ORDERED
        } else {
            TREE_256
        };
        let mut info = OpInfo::new(
            implementation,
            Invariance::any_rows(capacity),
            vec![(symbol.clone(), 0)],
        );
        if requirement.output == DType::F32 {
            info.bytes.push(("status".into(), u64::from(capacity) * 4));
            info.implementation.push_str(".status-or-zero.v2");
        }
        Ok(Self {
            norm,
            symbol,
            capacity,
            plan,
            info,
        })
    }
}

impl PreparedOp<RmsNormBuffers<View>> for RmsNormOp {
    fn route(&self, _counts: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        Some(kernel_core::route::branch(
            &launch.symbol,
            kernel_core::route::RouteClass::Fast,
            "rmsnorm",
            "standalone",
            [1..=self.capacity],
        ))
    }
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &RmsNormBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 {
            return Err(Refusal("RMSNorm launches at least one row"));
        }
        // One block per row of the grid: the step's rows, or the bucket it
        // pads them to.
        let grid_rows = counts.rows.max(counts.padded_rows);
        if grid_rows > self.capacity {
            return Err(Refusal("RMSNorm rows exceed the prepared capacity"));
        }
        let mut plan = self.plan;
        plan.grid[0] = grid_rows;
        let requirement = self.norm.requirement();
        let row_bytes = u64::from(requirement.width) * 2;
        let rows = u64::from(counts.rows) * row_bytes;
        buffers.x.covers(
            u64::from(counts.rows)
                * u64::from(requirement.width)
                * if requirement.input == DType::F32 {
                    4
                } else {
                    2
                },
        )?;
        let output_bytes = if requirement.output == DType::F32 {
            4
        } else {
            2
        };
        buffers
            .y
            .covers(u64::from(counts.rows) * u64::from(requirement.width) * output_bytes)?;
        if requirement.output == DType::F32
            && [
                buffers.x.handle,
                buffers.weight.handle,
                buffers.live_rows.handle,
            ]
            .contains(&buffers.y.handle)
        {
            return Err(Refusal(
                "F32 RMSNorm output must not overlap its inputs or live count",
            ));
        }
        if requirement.output == DType::F32 {
            buffers
                .status
                .ok_or(Refusal("F32 RMSNorm requires row status"))?
                .covers(kernel_core::precision::quantizer::status_extent(
                    buffers.status_row_offset,
                    counts.rows,
                )?)?;
        }
        buffers.weight.covers(row_bytes)?;
        buffers.live_rows.covers(4)?;
        if let Some(residual) = buffers.residual {
            residual.input.covers(rows)?;
            residual.output.covers(rows)?;
        }
        Ok(vec![Launch {
            symbol: self.symbol.clone(),
            rows: counts.rows,
            plan,
            arguments: self.norm.arguments(RmsNormBuffers {
                status: buffers.status.map(|v| v.handle),
                status_row_offset: buffers.status_row_offset,
                x: buffers.x.handle,
                weight: buffers.weight.handle,
                residual: buffers.residual.map(|r| FusedResidual {
                    input: r.input.handle,
                    output: r.output.handle,
                }),
                y: buffers.y.handle,
                live_rows: buffers.live_rows.handle,
            })?,
            collective: false,
            persistent: false,
        }])
    }
}

/// The head width the per-head norm template compiles in: one thread per
/// value of a head.
const HEAD: u32 = 128;
/// The arguments every per-head norm entry passes its template.
const HEAD_ARGUMENTS: &str =
    "(input,weight,output,live_tokens,input_stride,output_stride,__uint_as_float(eps_bits));";

/// The parameters of every per-head norm entry, in order: the names
/// [`HEAD_ARGUMENTS`] passes the template and the types it takes, so each
/// launch argument reaches the template parameter of its role.
fn head_params() -> Vec<EntryParam> {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let scalar = |name: &str| EntryParam::U32 { name: name.into() };
    vec![
        pointer("input", "const uint16_t"),
        pointer("weight", "const uint16_t"),
        pointer("output", "uint16_t"),
        pointer("live_tokens", "const uint32_t"),
        scalar("input_stride"),
        scalar("output_stride"),
        scalar("eps_bits"),
    ]
}

/// Whether the row engine plans `entry` as the template runs at every
/// capacity bucket up to the one holding `capacity` rows: a block of
/// [`HEAD`] threads per bucket row and head, without shared memory,
/// workspace or clusters. The planner adjusts some symbols' grids and
/// blocks by name, so the entry's grid and threads alone do not say it.
fn plans_heads(entry: &RowInstantiation, heads: u32, capacity: u32) -> bool {
    let mut rows = 1;
    while let Some(bucket) = smallest_bucket(rows) {
        let counts = RuntimeCounts {
            m: bucket,
            batch: 1,
            context: 0,
            routed_rows: 0,
            split_k: 1,
        };
        let Ok(plan) = entry.plan(bucket, counts) else {
            return false;
        };
        let template = plan.grid == [bucket, heads, 1]
            && plan.block == [HEAD, 1, 1]
            && plan.dynamic_shared_bytes == 0
            && plan.workspace_bytes == 0
            && plan.cluster == [1, 1, 1];
        if !template {
            return false;
        }
        if bucket >= capacity {
            break;
        }
        rows = bucket + 1;
    }
    true
}

/// The per-head RMSNorm of `op` (`RmsNormLayout::StridedHeads`) through the
/// selection's entry that instantiates the per-head norm template for its
/// heads: an entry whose query is the op's for one head, whose parameters
/// are the template's in order ([`head_params`]), and which the row engine
/// plans as a block per row and head. Entries that are not the template's
/// are not candidates, so they neither serve the op nor tie with one that
/// does. The template's arithmetic is its own (an F32
/// sum of the head's squares over its threads and a reciprocal square
/// root), named by the entry's declaration in `OpInfo::implementation`, so
/// an op that pins numerics is refused. The op claims what the entry's
/// registrations declare.
pub(super) fn prepare_heads(
    entries: &[RowInstantiation],
    op: &RmsNormRequirement,
    need: Need,
) -> Result<Prepared<RmsNormBuffers<View>>, Refused> {
    op.validate()?;
    let RmsNormLayout::StridedHeads {
        input_stride,
        output_stride,
        alignment_bytes,
    } = op.layout
    else {
        return Err(Refused::Rule("expected strided heads".into()));
    };
    let RmsNormKind::QkPerHead { head_dim } = op.kind else {
        unreachable!("validated strided heads normalize each head");
    };
    if head_dim != HEAD {
        return Err(Refused::Rule(format!(
            "the SM120 per-head RMSNorm normalizes heads of {HEAD} values"
        )));
    }
    if [op.input, op.weight, op.output] != [DType::Bf16; 3] {
        return Err(Refused::Rule(
            "the SM120 per-head RMSNorm reads and writes BF16 only".into(),
        ));
    }
    if op.output_quantization.is_some() {
        return Err(Refused::Rule(
            "the SM120 per-head RMSNorm does not quantize its output".into(),
        ));
    }
    if op.numerics.is_some() {
        return Err(Refused::Rule(
            "the SM120 per-head RMSNorm computes its template's numerics; it serves backend-chosen numerics only".into(),
        ));
    }
    // The template reads and writes one value per thread.
    if alignment_bytes < 2 {
        return Err(Refused::Rule(
            "the SM120 per-head RMSNorm needs element-aligned buffers".into(),
        ));
    }
    let stride = |stride: u64| {
        u32::try_from(stride).map_err(|_| {
            Refused::Rule("the SM120 per-head RMSNorm takes row strides within u32".into())
        })
    };
    let (input_stride, output_stride) = (stride(input_stride)?, stride(output_stride)?);
    let heads = op.width / head_dim;
    let query = RmsNormRequirement {
        width: head_dim,
        ..*op
    }
    .query();
    let body = format!("chop::row::dflash_qk_norm<{heads}>{HEAD_ARGUMENTS}");
    let params = head_params();
    let chosen = choose(
        "per-head RMSNorm",
        entries
            .iter()
            .filter(|entry| {
                entry.query == query
                    && entry.body == body
                    && entry.params == params
                    && entry.grid == Grid::RowsHeads(heads)
                    && entry.threads == head_dim
                    && plans_heads(entry, heads, need.capacity)
            })
            .map(|entry| Candidate::new(entry, &entry.symbol, row_launch(entry), entry.invariance)),
        need,
    )?;
    let op = HeadNormOp {
        capacity: need.capacity,
        width: op.width,
        head_dim,
        input_stride,
        output_stride,
        epsilon_bits: op.epsilon_bits,
        engine: EngineOp::new(
            Chosen {
                entry: chosen.entry.clone(),
                guarantees: chosen.guarantees,
            },
            need,
        )?,
    };
    Ok(Box::new(op))
}

/// A per-head norm entry launched as the row engine plans it, at the
/// smallest capacity bucket that holds a step's rows: a block per row and
/// head, rows at or past the device live count not written.
struct HeadNormOp {
    engine: EngineOp,
    capacity: u32,
    width: u32,
    head_dim: u32,
    input_stride: u32,
    output_stride: u32,
    epsilon_bits: u32,
}

impl HeadNormOp {
    /// The template's arguments over input, weight, output and live row
    /// handles.
    fn arguments(&self, [x, weight, y, live_rows]: [u64; 4]) -> Vec<Argument> {
        vec![
            Argument::DeviceBuffer(x),
            Argument::DeviceBuffer(weight),
            Argument::DeviceBuffer(y),
            Argument::DeviceBuffer(live_rows),
            Argument::U32(self.input_stride),
            Argument::U32(self.output_stride),
            Argument::U32(self.epsilon_bits),
        ]
    }
}

/// The bytes from the first of `rows` BF16 rows `stride` apart to the end
/// of the last row's `width` values.
fn strided_bytes(rows: u32, stride: u32, width: u32) -> Result<u64, Refusal> {
    u64::from(rows - 1)
        .checked_mul(u64::from(stride))
        .and_then(|n| n.checked_add(u64::from(width)))
        .and_then(|n| n.checked_mul(2))
        .ok_or(Refusal("per-head RMSNorm buffer bytes overflow"))
}

impl PreparedOp<RmsNormBuffers<View>> for HeadNormOp {
    fn info(&self) -> &OpInfo {
        &self.engine.info
    }

    fn launches(
        &self,
        buffers: &RmsNormBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if buffers.status.is_some() || buffers.status_row_offset != 0 {
            return Err(Refusal("legacy BF16 RMSNorm has no row status"));
        }
        let rows = self.engine.rows(counts)?;
        if counts.padded_rows > self.capacity {
            return Err(Refusal("RMSNorm rows exceed the prepared capacity"));
        }
        if buffers.residual.is_some() {
            return Err(Refusal("strided heads carry no residual rows"));
        }
        buffers
            .x
            .covers(strided_bytes(rows, self.input_stride, self.width)?)?;
        buffers
            .y
            .covers(strided_bytes(rows, self.output_stride, self.width)?)?;
        buffers.weight.covers(u64::from(self.head_dim) * 2)?;
        buffers.live_rows.covers(4)?;
        self.engine.launch(
            rows,
            self.arguments([
                buffers.x.handle,
                buffers.weight.handle,
                buffers.y.handle,
                buffers.live_rows.handle,
            ]),
        )
    }
}

#[cfg(test)]
pub(super) fn hf(width: u32) -> RmsNormRequirement {
    RmsNormRequirement::bf16(
        width,
        1e-6,
        kernel_core::contract::RmsNormKind::HfBf16Intermediate,
        false,
        kernel_core::rmsnorm::RmsNormNumerics::HF_ORDERED,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use kernel_core::backend::Backend;

    #[test]
    fn f32_norm_requires_full_output_bytes_and_separate_storage() {
        let req = RmsNormRequirement {
            input: DType::F32,
            output: DType::F32,
            numerics: Some(RmsNormNumerics::TREE_256),
            ..RmsNormRequirement::bf16_backend_chosen(1280, 1e-20, RmsNormKind::Plain, false)
        };
        let backend = Sm120::new().with_rmsnorm(req, "plain_f32");
        let prepared = backend.rmsnorm(&req, Need::rows(4)).unwrap();
        let b = RmsNormBuffers {
            status: Some(view(50, 16)),
            status_row_offset: 0,
            x: view(1, 4 * 1280 * 4),
            weight: view(2, 1280 * 2),
            residual: None,
            y: view(3, 4 * 1280 * 4),
            live_rows: view(4, 4),
        };
        assert!(prepared.launches(&b, View::EMPTY, Counts::rows(4)).is_ok());
        let mut shifted = b;
        shifted.status_row_offset = (1u64 << 32) + 3;
        shifted.status = Some(view(50, (shifted.status_row_offset + 4) * 4));
        let launches = prepared
            .launches(&shifted, View::EMPTY, Counts::rows(4))
            .unwrap();
        assert_eq!(
            launches[0].arguments.0.last(),
            Some(&Argument::U64(shifted.status_row_offset))
        );
        shifted.status.as_mut().unwrap().bytes -= 1;
        assert!(
            prepared
                .launches(&shifted, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        shifted.status_row_offset = u64::MAX;
        assert!(
            prepared
                .launches(&shifted, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        for status in [
            None,
            Some(b.x),
            Some(b.weight),
            Some(b.y),
            Some(b.live_rows),
        ] {
            assert!(
                prepared
                    .launches(
                        &RmsNormBuffers { status, ..b },
                        View::EMPTY,
                        Counts::rows(4)
                    )
                    .is_err()
            );
        }

        assert!(
            prepared
                .launches(
                    &RmsNormBuffers {
                        x: view(1, 4 * 1280 * 2),
                        ..b
                    },
                    View::EMPTY,
                    Counts::rows(4)
                )
                .is_err()
        );
        for y in [view(3, 4 * 1280 * 2), view(1, 4 * 1280 * 4)] {
            assert!(
                prepared
                    .launches(&RmsNormBuffers { y, ..b }, View::EMPTY, Counts::rows(4))
                    .is_err()
            );
        }
    }

    #[test]
    fn rmsnorm_launches_what_the_builder_plans() {
        let requirement = hf(4096);
        let backend = Sm120::new().with_rmsnorm(requirement, "chop_row_rmsnorm_test_v1");
        let prepared = backend.rmsnorm(&requirement, Need::rows(16)).unwrap();
        assert_eq!(prepared.info().implementation, HF_ORDERED);
        assert_eq!(prepared.info().guarantees, Invariance::any_rows(16));
        let rows = 16 * 4096 * 2;
        let buffers = RmsNormBuffers {
            status: None,
            status_row_offset: 0,
            x: view(1, rows),
            weight: view(2, 4096 * 2),
            residual: None,
            y: view(3, rows),
            live_rows: view(4, 4),
        };
        let launches = prepared
            .launches(&buffers, View::EMPTY, Counts::rows(16))
            .unwrap();
        let built = RmsNorm::new(requirement, 16, RmsNormTarget::SM120F).unwrap();
        let tag = prepared.route(Counts::rows(16), &launches[0]).unwrap();
        assert_eq!(
            (tag.class, tag.family),
            (kernel_core::route::RouteClass::Fast, "rmsnorm")
        );
        assert_eq!(tag.wrappers, ["standalone"]);
        assert_eq!(
            launches,
            [Launch {
                symbol: "chop_row_rmsnorm_test_v1".into(),
                rows: 16,
                plan: built.plan(16).unwrap(),
                arguments: built
                    .arguments(RmsNormBuffers {
                        status: None,
                        status_row_offset: 0,
                        x: 1,
                        weight: 2,
                        residual: None,
                        y: 3,
                        live_rows: 4,
                    })
                    .unwrap(),
                collective: false,
                persistent: false,
            }]
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(17))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(0))
                .is_err()
        );
        let short = RmsNormBuffers {
            y: view(3, rows - 2),
            ..buffers
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(16))
                .is_err()
        );
    }

    /// A requirement that leaves its numerics to SM120 prepares the entry
    /// and launches of the requirement pinning its kind's numerics, and
    /// reports them; an entry registered either way serves both.
    #[test]
    fn backend_chosen_norms_launch_what_pinned_ones_do() {
        use kernel_core::contract::RmsNormKind;
        for (kind, width, fused, implementation) in [
            (RmsNormKind::Plain, 4096, false, TREE_256),
            (RmsNormKind::Plain, 8192, true, TREE_256),
            (RmsNormKind::GemmaOnePlusWeight, 4096, false, TREE_256),
            (
                RmsNormKind::QkPerHead { head_dim: 128 },
                128,
                false,
                TREE_256,
            ),
            (RmsNormKind::HfBf16Intermediate, 4096, false, HF_ORDERED),
            (RmsNormKind::HfBf16Intermediate, 4096, true, HF_ORDERED),
        ] {
            let pinned =
                RmsNormRequirement::bf16(width, 1e-6, kind, fused, RmsNorm::numerics(kind));
            let chosen = RmsNormRequirement::bf16_backend_chosen(width, 1e-6, kind, fused);
            let rows = 16 * u64::from(width) * 2;
            let buffers = RmsNormBuffers {
                status: None,
                status_row_offset: 0,
                x: view(1, rows),
                weight: view(2, u64::from(width) * 2),
                residual: fused.then_some(FusedResidual {
                    input: view(5, rows),
                    output: view(6, rows),
                }),
                y: view(3, rows),
                live_rows: view(4, 4),
            };
            let padded = Counts {
                padded_rows: 16,
                ..Counts::rows(3)
            };
            let prepared = |registered: RmsNormRequirement, op: &RmsNormRequirement| {
                let op = Sm120::new()
                    .with_rmsnorm(registered, "chop_row_rmsnorm_test_v1")
                    .rmsnorm(op, Need::rows(16))
                    .unwrap();
                let launches = [Counts::rows(1), Counts::rows(16), padded]
                    .map(|counts| op.launches(&buffers, View::EMPTY, counts).unwrap());
                (op.info().clone(), launches)
            };
            let today = prepared(pinned, &pinned);
            assert_eq!(today.0.implementation, implementation);
            for (registered, op) in [(pinned, chosen), (chosen, pinned), (chosen, chosen)] {
                assert_eq!(prepared(registered, &op), today);
            }
        }
        // Pinned numerics other than the kind's find no entry, whichever
        // way the entry was registered.
        let plain = RmsNormRequirement::bf16_backend_chosen(4096, 1e-6, RmsNormKind::Plain, false);
        let crossed = RmsNormRequirement {
            numerics: Some(RmsNormNumerics::HF_ORDERED),
            ..plain
        };
        for registered in [plain, resolved(&plain)] {
            let backend = Sm120::new().with_rmsnorm(registered, "chop_row_rmsnorm_test_v1");
            assert!(matches!(
                backend.rmsnorm(&crossed, Need::rows(16)),
                Err(Refused::Rule(rule)) if rule.contains("no SM120 RMSNorm entry")
            ));
        }
        // A pinned and a backend-chosen registration of one requirement are
        // one entry, and tie under two symbols.
        let one = Sm120::new()
            .with_rmsnorm(plain, "chop_row_rmsnorm_a_v1")
            .with_rmsnorm(resolved(&plain), "chop_row_rmsnorm_a_v1");
        assert!(one.rmsnorm(&plain, Need::rows(16)).is_ok());
        let two = Sm120::new()
            .with_rmsnorm(plain, "chop_row_rmsnorm_a_v1")
            .with_rmsnorm(resolved(&plain), "chop_row_rmsnorm_b_v1");
        assert!(matches!(
            two.rmsnorm(&plain, Need::rows(16)),
            Err(Refused::Rule(rule))
                if rule.contains("chop_row_rmsnorm_a_v1 and chop_row_rmsnorm_b_v1")
        ));
    }

    #[test]
    fn a_launch_grids_its_bucket_not_the_prepared_capacity() {
        // A step builds today's norm at its row bucket and plans its live
        // rows: one block per bucket row, whatever capacity the op serves.
        let requirement = hf(4096);
        let backend = Sm120::new().with_rmsnorm(requirement, "chop_row_rmsnorm_test_v1");
        let prepared = backend.rmsnorm(&requirement, Need::rows(8192)).unwrap();
        let rows = 8192 * 4096 * 2;
        let buffers = RmsNormBuffers {
            status: None,
            status_row_offset: 0,
            x: view(1, rows),
            weight: view(2, 4096 * 2),
            residual: None,
            y: view(3, rows),
            live_rows: view(4, 4),
        };
        let bucket = RmsNorm::new(requirement, 128, RmsNormTarget::SM120F).unwrap();
        let padded = Counts {
            padded_rows: 128,
            ..Counts::rows(100)
        };
        for (counts, recorded) in [(padded, 100), (Counts::rows(128), 128)] {
            let launch = prepared
                .launches(&buffers, View::EMPTY, counts)
                .unwrap()
                .remove(0);
            assert_eq!(launch.plan, bucket.plan(100).unwrap());
            assert_eq!(launch.rows, recorded);
        }
        let past = Counts {
            padded_rows: 8193,
            ..Counts::rows(100)
        };
        assert!(prepared.launches(&buffers, View::EMPTY, past).is_err());
    }

    #[test]
    fn rmsnorm_entries_prepare_alike_in_every_order() {
        let requirement = hf(4096);
        let buffers = RmsNormBuffers {
            status: None,
            status_row_offset: 0,
            x: view(1, 4096 * 2),
            weight: view(2, 4096 * 2),
            residual: None,
            y: view(3, 4096 * 2),
            live_rows: view(4, 4),
        };
        let one = "chop_row_rmsnorm_a_v1";
        let launches = |symbols: &[&str]| {
            let backend = symbols
                .iter()
                .fold(Sm120::new(), |backend, symbol| {
                    backend.with_rmsnorm(requirement, *symbol)
                })
                .with_rmsnorm(hf(2048), "chop_row_rmsnorm_other_v1");
            backend
                .rmsnorm(&requirement, Need::rows(1))
                .map(|op| op.launches(&buffers, View::EMPTY, Counts::rows(1)).unwrap())
        };
        // A duplicate registration is one entry.
        let expected = launches(&[one]).unwrap();
        for order in orders(&[one, one]) {
            assert_eq!(launches(&order).unwrap(), expected);
        }
        // Two symbols for one requirement are a tie.
        for order in orders(&[one, "chop_row_rmsnorm_b_v1", one]) {
            assert!(matches!(
                launches(&order),
                Err(Refused::Rule(rule))
                    if rule.contains("chop_row_rmsnorm_a_v1 and chop_row_rmsnorm_b_v1")
            ));
        }
    }

    /// An RMSNorm entry is its requirement, which the op selects by, and
    /// its symbol, the one key field: it is the launch's symbol.
    #[test]
    fn the_launch_key_is_the_launched_symbol() {
        use crate::backend::select::{Class, Field, keys_follow_launches};
        let requirement = hf(4096);
        let rows = 16 * 4096 * 2;
        let buffers = RmsNormBuffers {
            status: None,
            status_row_offset: 0,
            x: view(1, rows),
            weight: view(2, 4096 * 2),
            residual: None,
            y: view(3, rows),
            live_rows: view(4, 4),
        };
        // The entry is a pair: (requirement, symbol).
        let fields: Vec<Field<(RmsNormRequirement, String)>> = vec![
            ("requirement", Class::Selects, Box::new(|e| e.0 = hf(2048))),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.1 = "chop_row_rmsnorm_other_v1".into()),
            ),
        ];
        keys_follow_launches(
            &(requirement, "chop_row_rmsnorm_test_v1".to_owned()),
            vec!["requirement", "symbol"],
            fields,
            launch,
            |(requirement, symbol)| {
                let op = Sm120::new()
                    .with_rmsnorm(*requirement, symbol.clone())
                    .rmsnorm(&hf(4096), Need::rows(16))?;
                let launches = [1, 16].map(|rows| {
                    op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((String::new(), launches))
            },
        );
    }

    /// A per-head norm entry of `heads` heads as a model's row catalog
    /// instantiates the template.
    fn head_entry(heads: u32) -> RowInstantiation {
        use kernel_build::EntryParam;
        use kernel_core::contract::RowQuery;
        let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: cxx_type.into(),
        };
        let u32_arg = |name: &str| EntryParam::U32 { name: name.into() };
        RowInstantiation::new(
            crate::Profile::StdDenseMoe,
            &format!("head_norm_h{heads}"),
            RowQuery::RmsNorm {
                width: 128,
                eps: 1e-6,
                kind: RmsNormKind::QkPerHead { head_dim: 128 },
                fused_residual: false,
                output_quantization: None,
            },
            128,
            (Grid::RowsHeads(heads), 128),
            vec![
                ptr("input", "const uint16_t"),
                ptr("weight", "const uint16_t"),
                ptr("output", "uint16_t"),
                ptr("live_tokens", "const uint32_t"),
                u32_arg("input_stride"),
                u32_arg("output_stride"),
                u32_arg("eps_bits"),
            ],
            format!(
                "chop::row::dflash_qk_norm<{heads}>(input,weight,output,live_tokens,input_stride,output_stride,__uint_as_float(eps_bits));"
            ),
        )
    }

    fn heads_op(heads: u32, input_stride: u64, output_stride: u64) -> RmsNormRequirement {
        RmsNormRequirement {
            layout: RmsNormLayout::StridedHeads {
                input_stride,
                output_stride,
                alignment_bytes: 16,
            },
            ..RmsNormRequirement::bf16_backend_chosen(
                heads * 128,
                1e-6,
                RmsNormKind::QkPerHead { head_dim: 128 },
                false,
            )
        }
    }

    fn heads_buffers(rows: u64, input_stride: u64, output_stride: u64) -> RmsNormBuffers<View> {
        RmsNormBuffers {
            status: None,
            status_row_offset: 0,
            x: view(1, rows * input_stride * 2),
            weight: view(2, 128 * 2),
            residual: None,
            y: view(3, rows * output_stride * 2),
            live_rows: view(4, 4),
        }
    }

    /// What the row engine's legacy planning gives a step launched at
    /// `bucket` rows: the entry's plan at the bucket, whatever its live
    /// count, and its arguments packed in ABI order.
    fn legacy(
        entry: &RowInstantiation,
        bucket: u32,
        live: u32,
        input_stride: u32,
        output_stride: u32,
    ) -> Launch {
        let counts = kernel_core::contract::RuntimeCounts {
            m: live,
            batch: 1,
            context: 1,
            routed_rows: 8,
            split_k: 1,
        };
        Launch {
            symbol: entry.symbol.clone(),
            rows: bucket,
            plan: entry.plan(bucket, counts).unwrap(),
            arguments: entry
                .pack_arguments(vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                    Argument::U32(input_stride),
                    Argument::U32(output_stride),
                    Argument::U32(1e-6_f32.to_bits()),
                ])
                .unwrap(),
            collective: false,
            persistent: false,
        }
    }

    /// Per-head norms of the shapes a drafter's Q and K heads take launch
    /// what the row engine plans for the entry at a bucket of rows, and
    /// grid the smallest bucket that holds fewer rows. The drafter has 64
    /// Q heads, 8 KV heads and five layers, split over 8, 4 and 2 ranks:
    /// the Q norm reads Q heads from the fused QKV row, the verify K norm
    /// reads K heads from it, and the context K norm reads K heads from a
    /// row of every layer's K and V; each writes compact heads.
    #[test]
    fn strided_heads_launch_what_the_row_engine_plans() {
        let drafter = |ranks: u32| {
            let (q, kv) = (64 / ranks, 8 / ranks);
            let qkv = u64::from(q + 2 * kv) * 128;
            let (q_cols, kv_cols) = (u64::from(q) * 128, u64::from(kv) * 128);
            [
                (q, qkv, q_cols),
                (kv, qkv, kv_cols),
                (kv, 2 * kv_cols * 5, kv_cols),
            ]
        };
        // The pairs at two ranks, written out.
        assert_eq!(
            drafter(2),
            [(32, 5120, 4096), (4, 5120, 512), (4, 5120, 512)]
        );
        let shapes = [8, 4, 2]
            .into_iter()
            .flat_map(drafter)
            .chain([(32, 32 * 128, 32 * 128)]);
        for (heads, input_stride, output_stride) in shapes {
            let entry = head_entry(heads);
            let op = heads_op(heads, input_stride, output_stride);
            let backend = Sm120::new().with_row_rmsnorm(entry.clone());
            for capacity in [1, 16, 64, 512] {
                let prepared = backend.rmsnorm(&op, Need::rows(capacity)).unwrap();
                let info = prepared.info();
                assert!(
                    info.implementation
                        .starts_with(&format!("sm120.row.v1[chop_row_head_norm_h{heads}_v1]#"))
                );
                assert_eq!(info.symbols, [(entry.symbol.clone(), 0)]);
                // The entry declares nothing, so the op claims nothing.
                assert_eq!(info.guarantees, Invariance::NONE);
                let buffers = heads_buffers(u64::from(capacity), input_stride, output_stride);
                let strides = (input_stride as u32, output_stride as u32);
                for live in [1, capacity] {
                    assert_eq!(
                        prepared
                            .launches(&buffers, View::EMPTY, Counts::rows(capacity))
                            .unwrap(),
                        [legacy(&entry, capacity, live, strides.0, strides.1)]
                    );
                }
                if capacity > 1 {
                    let rows = capacity - 1;
                    let bucket = crate::row::smallest_bucket(rows).unwrap();
                    let launch = prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                        .remove(0);
                    assert_eq!(
                        launch,
                        Launch {
                            rows,
                            ..legacy(&entry, bucket, rows, strides.0, strides.1)
                        }
                    );
                    assert_eq!(launch.plan.grid, [bucket, heads, 1]);
                }
            }
        }
    }

    /// The entry's declaration is what the op claims, and an exactness
    /// it does not declare is refused.
    #[test]
    fn strided_heads_claim_what_the_entry_declares() {
        let op = heads_op(8, 10 * 128, 8 * 128);
        let declared = RowInstantiation {
            invariance: crate::row::row_local(),
            ..head_entry(8)
        };
        let prepared = Sm120::new()
            .with_row_rmsnorm(declared.clone())
            .rmsnorm(&op, Need::rows(64).with_exact(Invariance::any_rows(64)))
            .unwrap();
        assert_eq!(prepared.info().guarantees, declared.invariance);
        assert!(matches!(
            Sm120::new()
                .with_row_rmsnorm(head_entry(8))
                .rmsnorm(&op, Need::rows(64).with_exact(Invariance::any_rows(64))),
            Err(Refused::Inexact { .. })
        ));
        // Registered both ways, the op may launch either, so it claims what
        // both declare, in either order.
        for order in orders(&[declared, head_entry(8)]) {
            let backend = order.into_iter().fold(Sm120::new(), |backend, entry| {
                backend.with_row_rmsnorm(entry)
            });
            let prepared = backend.rmsnorm(&op, Need::rows(64)).unwrap();
            assert_eq!(prepared.info().guarantees, Invariance::NONE);
        }
    }

    #[test]
    fn strided_heads_refuse_what_the_template_does_not_compute() {
        let op = heads_op(8, 10 * 128, 8 * 128);
        let backend = Sm120::new().with_row_rmsnorm(head_entry(8));
        let refused = |op: RmsNormRequirement, rule: &str| {
            let refusal = backend
                .rmsnorm(&op, Need::rows(16))
                .map(|_| ())
                .unwrap_err();
            assert!(
                matches!(&refusal, Refused::Rule(r) if r.contains(rule)),
                "{op:?}: {refusal}"
            );
        };
        refused(
            RmsNormRequirement {
                numerics: Some(RmsNormNumerics::TREE_256),
                ..op
            },
            "backend-chosen numerics only",
        );
        refused(
            RmsNormRequirement {
                output: DType::F32,
                ..op
            },
            "BF16 only",
        );
        refused(
            RmsNormRequirement {
                output_quantization: Some(kernel_core::contract::OutputQuantization::Nvfp4),
                ..op
            },
            "quantize",
        );
        refused(
            RmsNormRequirement {
                width: 4 * 64,
                kind: RmsNormKind::QkPerHead { head_dim: 64 },
                ..op
            },
            "heads of 128",
        );
        refused(
            RmsNormRequirement {
                layout: RmsNormLayout::StridedHeads {
                    input_stride: 10 * 128,
                    output_stride: 8 * 128,
                    alignment_bytes: 1,
                },
                ..op
            },
            "element-aligned",
        );
        refused(heads_op(8, u64::from(u32::MAX) + 1, 8 * 128), "within u32");
        // Heads no entry grids, and a contiguous requirement, which the
        // generic row RMSNorm serves.
        refused(
            heads_op(4, 10 * 128, 4 * 128),
            "no SM120 per-head RMSNorm entry",
        );
        refused(
            RmsNormRequirement {
                layout: RmsNormLayout::ContiguousRows {
                    alignment_bytes: 16,
                },
                ..op
            },
            "no SM120 RMSNorm entry",
        );
        // Kernel-core's math refusals come first.
        refused(
            RmsNormRequirement {
                fused_residual: true,
                ..op
            },
            "no residual rows",
        );
    }

    /// An entry serves only with the template's parameters in order: one
    /// whose same-kind parameters are swapped (strides, or input and
    /// weight), renamed or retyped would bind arguments to the wrong roles,
    /// so it is no candidate, alone or beside the template's entry.
    #[test]
    fn strided_heads_take_only_the_templates_parameters_in_order() {
        let (input_stride, output_stride) = (10 * 128, 8 * 128);
        let op = heads_op(8, input_stride, output_stride);
        let buffers = heads_buffers(16, input_stride, output_stride);
        let launches = |backend: Sm120| {
            backend.rmsnorm(&op, Need::rows(16)).map(|op| {
                op.launches(&buffers, View::EMPTY, Counts::rows(16))
                    .unwrap()
            })
        };
        let expected = launches(Sm120::new().with_row_rmsnorm(head_entry(8))).unwrap();
        let changes: [fn(&mut Vec<kernel_build::EntryParam>); 6] = [
            |params| params.swap(4, 5),
            |params| params.swap(0, 1),
            |params| params.swap(3, 4),
            |params| params[6] = kernel_build::EntryParam::U32 { name: "eps".into() },
            |params| {
                params[1] = kernel_build::EntryParam::DevicePointer {
                    name: "weight".into(),
                    cxx_type: "uint16_t".into(),
                }
            },
            |params| params.truncate(6),
        ];
        for change in changes {
            let mut wrong = head_entry(8);
            change(&mut wrong.params);
            assert!(
                matches!(
                    launches(Sm120::new().with_row_rmsnorm(wrong.clone())),
                    Err(Refused::Rule(rule)) if rule.contains("no SM120 per-head RMSNorm entry")
                ),
                "{:?}",
                wrong.params
            );
            for order in orders(&[wrong, head_entry(8)]) {
                let backend = order.into_iter().fold(Sm120::new(), |backend, entry| {
                    backend.with_row_rmsnorm(entry)
                });
                assert_eq!(launches(backend).unwrap(), expected);
            }
        }
    }

    /// The row engine's planner adjusts some symbols' launches by name: a
    /// `permute_count` entry gets 256 threads and a `rope` entry four times
    /// its heads below 512 rows. An entry it would launch so is no
    /// candidate, alone or beside the template's entry, whatever its grid
    /// and threads fields say.
    #[test]
    fn strided_heads_refuse_entries_the_planner_reshapes() {
        let (input_stride, output_stride) = (10 * 128, 8 * 128);
        let op = heads_op(8, input_stride, output_stride);
        let buffers = heads_buffers(512, input_stride, output_stride);
        let launches = |backend: Sm120, capacity| {
            backend.rmsnorm(&op, Need::rows(capacity)).map(|op| {
                [1, capacity].map(|rows| {
                    op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                })
            })
        };
        for name in ["permute_count_head_norm", "rope_head_norm"] {
            let renamed = RowInstantiation {
                symbol: format!("chop_row_{name}_v1"),
                ..head_entry(8)
            };
            for capacity in [16, 512] {
                assert!(
                    matches!(
                        launches(Sm120::new().with_row_rmsnorm(renamed.clone()), capacity),
                        Err(Refused::Rule(rule)) if rule.contains("no SM120 per-head RMSNorm entry")
                    ),
                    "{name} at {capacity}"
                );
                let expected =
                    launches(Sm120::new().with_row_rmsnorm(head_entry(8)), capacity).unwrap();
                for order in orders(&[renamed.clone(), head_entry(8)]) {
                    let backend = order.into_iter().fold(Sm120::new(), |backend, entry| {
                        backend.with_row_rmsnorm(entry)
                    });
                    assert_eq!(launches(backend, capacity).unwrap(), expected);
                }
            }
        }
        // Every launch of the template's entry is a block of 128 threads
        // per bucket row and head.
        let prepared = Sm120::new()
            .with_row_rmsnorm(head_entry(8))
            .rmsnorm(&op, Need::rows(512))
            .unwrap();
        for rows in [1, 3, 32, 100, 511, 512] {
            let launch = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap()
                .remove(0);
            let bucket = crate::row::smallest_bucket(rows).unwrap();
            assert_eq!(launch.plan.grid, [bucket, 8, 1]);
            assert_eq!(launch.plan.block, [128, 1, 1]);
        }
    }

    #[test]
    fn strided_heads_check_buffers_and_rows() {
        let (input_stride, output_stride) = (10 * 128, 8 * 128);
        let op = heads_op(8, input_stride, output_stride);
        let prepared = Sm120::new()
            .with_row_rmsnorm(head_entry(8))
            .rmsnorm(&op, Need::rows(16))
            .unwrap();
        let buffers = heads_buffers(16, input_stride, output_stride);
        let launch = |buffers: &RmsNormBuffers<View>, counts| {
            prepared.launches(buffers, View::EMPTY, counts)
        };
        assert!(launch(&buffers, Counts::rows(16)).is_ok());
        // The last row needs its own width only, not a whole stride.
        let tight = RmsNormBuffers {
            x: view(1, (15 * input_stride + 8 * 128) * 2),
            y: view(3, (15 * output_stride + 8 * 128) * 2),
            ..buffers
        };
        assert!(launch(&tight, Counts::rows(16)).is_ok());
        for short in [
            RmsNormBuffers {
                x: view(1, (15 * input_stride + 8 * 128) * 2 - 2),
                ..buffers
            },
            RmsNormBuffers {
                y: view(3, (15 * output_stride + 8 * 128) * 2 - 2),
                ..buffers
            },
            RmsNormBuffers {
                weight: view(2, 127 * 2),
                ..buffers
            },
            RmsNormBuffers {
                live_rows: view(4, 2),
                ..buffers
            },
            RmsNormBuffers {
                residual: Some(FusedResidual {
                    input: view(5, 1 << 20),
                    output: view(6, 1 << 20),
                }),
                ..buffers
            },
        ] {
            assert!(launch(&short, Counts::rows(16)).is_err(), "{short:?}");
        }
        for counts in [
            Counts::rows(0),
            Counts::rows(17),
            Counts {
                padded_rows: 17,
                ..Counts::rows(4)
            },
        ] {
            assert!(launch(&buffers, counts).is_err(), "{counts:?}");
        }
        // Padded rows within the capacity grid the step's rows.
        let padded = Counts {
            padded_rows: 16,
            ..Counts::rows(3)
        };
        assert_eq!(
            launch(&buffers, padded).unwrap(),
            launch(&buffers, Counts::rows(3)).unwrap()
        );
    }

    /// A per-head norm entry's launch key is the row engine's: the op
    /// selects by its query, code, parameters, heads and threads (the
    /// template), and its symbol is launched.
    #[test]
    fn strided_heads_keys_follow_launches() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        use kernel_core::contract::RowQuery;
        let other = crate::Profile::synthetic(crate::ProfileNames {
            name: "head-norm-other",
            tag: "head_norm_other",
        });
        let (input_stride, output_stride) = (10 * 128, 8 * 128);
        let op = heads_op(8, input_stride, output_stride);
        let buffers = heads_buffers(16, input_stride, output_stride);
        let names = field_names!(RowInstantiation {
            profile,
            symbol,
            query,
            width,
            grid,
            threads,
            params,
            body,
            invariance
        });
        let fields: Vec<Field<RowInstantiation>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(move |e| e.profile = other),
            ),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_row_head_norm_other_v1".into()),
            ),
            (
                "query",
                Class::Selects,
                Box::new(|e| {
                    e.query = RowQuery::RmsNorm {
                        width: 128,
                        eps: 1e-5,
                        kind: RmsNormKind::QkPerHead { head_dim: 128 },
                        fused_residual: false,
                        output_quantization: None,
                    }
                }),
            ),
            ("width", Class::Inert, Box::new(|e| e.width *= 2)),
            (
                "grid",
                Class::Selects,
                Box::new(|e| e.grid = Grid::RowsHeads(4)),
            ),
            ("threads", Class::Selects, Box::new(|e| e.threads *= 2)),
            ("params", Class::Selects, Box::new(|e| e.params.swap(4, 5))),
            ("body", Class::Selects, Box::new(|e| e.body.push_str("//"))),
            (
                "invariance",
                Class::Inert,
                Box::new(|e| e.invariance = crate::row::row_local()),
            ),
        ];
        keys_follow_launches(&head_entry(8), names, fields, row_launch, |entry| {
            let prepared = Sm120::new()
                .with_row_rmsnorm(entry.clone())
                .rmsnorm(&op, Need::rows(16))?;
            let code = format!("{:?} {} {}", entry.params, entry.body, entry.threads);
            let launches = [1, 16].map(|rows| {
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((code, launches))
        });
    }
}
