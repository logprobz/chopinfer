//! Position score/select and candidate pre-filter conformance.

use kernel_core::pooled_indexer::IndexerCandidates;
use kernel_harness::kit::{self, OpResult, Outcome, cases::position_indexer};
use kernel_lab::families::position_indexer as family;
use kernels_sm120::backend::Sm120;

#[test]
fn position_family_builds_the_required_shapes_and_controls() {
    let entries = family::entries();
    assert_eq!(entries.len(), 6);
    for entry in &entries {
        entry.validate().unwrap();
        let score = entry.score_plan(8, 16392).unwrap();
        assert_eq!(score.grid, [4098, 8, 1]);
        assert!(score.pdl);
        entry.select_plan(8, 16392, 3).unwrap();
        assert!(entry.score_plan(9, 16392).is_err());
        assert!(entry.select_plan(8, 16392, 2049).is_err());
        assert_eq!(entry.specs().unwrap().len(), 2);
    }
    assert!(entries.iter().any(|e| e.query.head_dim > 128));
    assert!(
        entries
            .iter()
            .any(|e| e.query.candidates == IndexerCandidates::Mask)
    );
    let filters = family::candidates();
    assert_eq!(filters.len(), 2);
    assert_eq!(filters[1].workspace_bytes(8, 16385).unwrap(), 8 * 2049 * 4);
    let specs = family::cubins();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].entries.len(), 28);
    assert!(
        specs[0]
            .entries
            .iter()
            .any(|e| e.symbol.ends_with("_ignore_mask_v3"))
    );
    assert!(
        specs[0]
            .entries
            .iter()
            .any(|e| e.symbol.ends_with("_wrong_pin_v1"))
    );
}

#[test]
fn position_cubin_has_no_tf32_spills_or_stack_and_keeps_pdl_waits() {
    let toolchain = kernel_build::Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(!toolchain.skipped, "the SASS audit must run");
    let root =
        std::env::temp_dir().join(format!("chopinfer-position-audit-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("position.cubin");
    std::fs::write(&path, kernel_lab::cubin(family::CUBIN).unwrap()).unwrap();
    let resources = kernel_build::audit_cubin(&toolchain, &family::cubins()[0], &path).unwrap();
    assert_eq!(resources.len(), 28);
    for entry in resources {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{} uses a stack", entry.symbol);
    }
    let result = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&path)
        .output()
        .expect("SASS dump");
    assert!(result.status.success(), "cuobjdump failed");
    let sass = String::from_utf8(result.stdout).unwrap();
    for spec in &family::cubins()[0].entries {
        let header = format!("Function : {}\n", spec.symbol);
        let start = sass.find(&header).expect("compiled function") + header.len();
        let body = &sass[start..];
        let body = &body[..body.find("Function : ").unwrap_or(body.len())];
        for required in ["PREEXIT", "ACQBULK"] {
            assert!(body.contains(required), "{} lacks {required}", spec.symbol);
        }
        for forbidden in ["TF32", "LDL", "STL"] {
            assert!(
                !body.contains(forbidden),
                "{} uses {forbidden}",
                spec.symbol
            );
        }
    }
    assert!(sass.contains("FFMA"));
    std::fs::remove_dir_all(root).unwrap();
}

fn report(session: Option<&kernel_lab::Session>) -> kernel_lab::LabResult<kit::Report> {
    let entries = family::entries();
    let filters = family::candidates();
    let ops = position_indexer::family(&entries, &filters);
    let backend = Sm120::new()
        .with_position_indexer(entries)
        .with_indexer_candidate_blocks(filters);
    kit::conform(&backend, &kernel_lab::kit::LabModules, session, &ops)
}

fn check(report: &kit::Report, gpu: bool) {
    assert!(report.failures().is_empty(), "{:#?}", report.failures());
    assert_eq!(report.prepared().len(), 14, "{:#?}", report.ops);
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("required op refused");
            };
            let conformance = checks
                .iter()
                .find(|c| c.name == "conformance")
                .expect("conformance check");
            assert!(
                matches!(conformance.outcome, Outcome::Passed(_)),
                "{}: {:?}",
                op.name,
                conformance.outcome
            );
            for name in [
                kit::PDL,
                "property rows",
                "property chunk",
                "property batch",
            ] {
                let check = checks
                    .iter()
                    .find(|c| c.name == name)
                    .expect("required replay or invariance check");
                assert!(
                    matches!(check.outcome, Outcome::Passed(_)),
                    "{} {name}: {:?}",
                    op.name,
                    check.outcome
                );
            }
        }
    }
}

#[test]
fn position_cases_declare_and_prepare() {
    let report = report(None).expect("position family declarations");
    check(&report, false);
}

#[test]
fn position_cases_pass_on_a_gpu() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let report = report(Some(session))?;
        session.receipt(
            "position-indexer",
            std::collections::BTreeMap::from([
                ("report".into(), report.json()),
                (
                    "not_exercised".into(),
                    serde_json::json!(report.not_exercised()),
                ),
            ]),
        )?;
        check(&report, true);
        Ok(())
    })
    .expect("position family GPU conformance");
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    IgnoreMask,
    WrongPin,
    ReverseTies,
    DropCandidate,
    WrongWeight,
    SwapNibbles,
}

impl Fault {
    fn symbol(self, symbol: &str) -> String {
        let suffix = match self {
            Self::IgnoreMask => "ignore_mask",
            Self::WrongPin => "wrong_pin",
            Self::ReverseTies => "reverse_ties",
            Self::DropCandidate => "drop_candidate",
            Self::WrongWeight => "wrong_weight",
            Self::SwapNibbles => "swap_nibbles",
        };
        let (base, version) = symbol.rsplit_once("_v").unwrap();
        format!("{base}_{suffix}_v{version}")
    }
}

struct Mutant {
    backend: Sm120,
    fault: Fault,
}

struct MutantOp<B> {
    inner: kernel_core::backend::Prepared<B>,
    info: kernel_core::backend::OpInfo,
    fault: Fault,
}

impl<B: 'static> MutantOp<B> {
    fn wrap(
        inner: kernel_core::backend::Prepared<B>,
        fault: Fault,
    ) -> kernel_core::backend::Prepared<B> {
        let mut info = inner.info().clone();
        for (symbol, _) in &mut info.symbols {
            *symbol = fault.symbol(symbol);
        }
        Box::new(Self { inner, info, fault })
    }
}

impl<B> kernel_core::backend::PreparedOp<B> for MutantOp<B> {
    fn info(&self) -> &kernel_core::backend::OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &B,
        workspace: kernel_core::backend::View,
        counts: kernel_core::backend::Counts,
    ) -> Result<Vec<kernel_core::backend::Launch>, kernel_core::contract::Refusal> {
        let mut launches = self.inner.launches(buffers, workspace, counts)?;
        for launch in &mut launches {
            launch.symbol = self.fault.symbol(&launch.symbol);
        }
        Ok(launches)
    }
}

impl kernel_core::backend::Backend for Mutant {
    fn name(&self) -> &str {
        "position-control"
    }
    fn position_indexer(
        &self,
        query: &kernel_core::pooled_indexer::PositionIndexerQuery,
        top_k: u32,
        width: u32,
        need: kernel_core::backend::Need,
    ) -> Result<
        kernel_core::backend::Prepared<
            kernel_core::pooled_indexer::PositionSelectBuffers<kernel_core::backend::View>,
        >,
        kernel_core::backend::Refused,
    > {
        self.backend
            .position_indexer(query, top_k, width, need)
            .map(|op| MutantOp::wrap(op, self.fault))
    }
    fn indexer_candidate_blocks(
        &self,
        query: &kernel_core::pooled_indexer::CandidateBlocksQuery,
        width: u32,
        need: kernel_core::backend::Need,
    ) -> Result<
        kernel_core::backend::Prepared<
            kernel_core::pooled_indexer::CandidateBlocksBuffers<kernel_core::backend::View>,
        >,
        kernel_core::backend::Refused,
    > {
        self.backend
            .indexer_candidate_blocks(query, width, need)
            .map(|op| MutantOp::wrap(op, self.fault))
    }
}

#[test]
fn position_negative_controls_fail_on_a_gpu() {
    kernel_lab::gpu_test(4 << 30, |session| {
        for fault in [
            Fault::IgnoreMask,
            Fault::WrongPin,
            Fault::ReverseTies,
            Fault::DropCandidate,
            Fault::WrongWeight,
            Fault::SwapNibbles,
        ] {
            let entries = match fault {
                Fault::IgnoreMask => vec![family::entries()[1].clone()],
                Fault::WrongPin => vec![],
                _ => vec![family::entries()[0].clone()],
            };
            let filters = match fault {
                Fault::WrongPin => vec![family::candidates()[0].clone()],
                _ => vec![],
            };
            let ops = match fault {
                Fault::IgnoreMask | Fault::WrongPin => position_indexer::family(&entries, &filters),
                _ => witness_ops(&[position_indexer::Witness::Ties]),
            };
            let backend = Mutant {
                backend: Sm120::new()
                    .with_position_indexer(entries)
                    .with_indexer_candidate_blocks(filters),
                fault,
            };
            let report = kit::conform(&backend, &kernel_lab::kit::LabModules, Some(session), &ops)?;
            assert_eq!(report.prepared().len(), ops.len());
            assert!(!report.prepared().is_empty());
            let expected_failure = report.ops.iter().any(|op| {
                let OpResult::Prepared { checks, .. } = &op.result else {
                    return false;
                };
                checks.iter().any(|check| {
                    let Outcome::Failed(reason) = &check.outcome else {
                        return false;
                    };
                    check.name == "conformance"
                        && match fault {
                            Fault::IgnoreMask => {
                                reason.contains("scores")
                                    && (reason.contains("byte") || reason.contains("masked"))
                            }
                            Fault::WrongPin | Fault::ReverseTies | Fault::DropCandidate => {
                                reason.contains("output") && reason.contains("byte")
                            }
                            Fault::WrongWeight | Fault::SwapNibbles => {
                                reason.contains("scores") && reason.contains("byte")
                            }
                        }
                })
            });
            session.receipt(
                &format!("position-control-{fault:?}"),
                std::collections::BTreeMap::from([("report".into(), report.json())]),
            )?;
            assert!(
                expected_failure,
                "{fault:?} did not fail at its intended oracle: {}",
                report.json()
            );
        }
        Ok(())
    })
    .expect("position negative controls");
}

fn witness_ops(witnesses: &[position_indexer::Witness]) -> Vec<kit::Required<'static>> {
    let query = family::entries()[0].query;
    witnesses
        .iter()
        .copied()
        .map(|witness| {
            kit::Required::new(
                format!("position witness {witness:?}"),
                kernel_core::backend::Need::rows(4),
                move |backend, need| backend.position_indexer(&query, 2, 8, need),
                move |op| position_indexer::witness_case(query, witness, op),
            )
            .required()
        })
        .collect()
}

fn failure_witness_report(
    session: Option<&kernel_lab::Session>,
) -> kernel_lab::LabResult<kit::Report> {
    use position_indexer::Witness::*;
    let backend = Sm120::new().with_position_indexer([family::entries()[0].clone()]);
    kit::conform(
        &backend,
        &kernel_lab::kit::LabModules,
        session,
        &witness_ops(&[Ties, Overflow, NanWeight, NanDot]),
    )
}

#[test]
fn position_failure_witnesses_declare_and_prepare() {
    let report = failure_witness_report(None).unwrap();
    assert_eq!(report.prepared().len(), 4);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
}

#[test]
fn position_nonfinite_arithmetic_is_reported_on_a_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let report = failure_witness_report(Some(session))?;
        session.receipt(
            "position-nonfinite",
            std::collections::BTreeMap::from([("report".into(), report.json())]),
        )?;
        assert_eq!(report.prepared().len(), 4);
        assert!(report.failures().is_empty(), "{:?}", report.failures());
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("witness refused")
            };
            for name in [
                "conformance",
                kit::PDL,
                "property rows",
                "property chunk",
                "property batch",
            ] {
                assert!(
                    checks
                        .iter()
                        .any(|c| c.name == name && matches!(c.outcome, Outcome::Passed(_))),
                    "{}: {name}",
                    op.name
                );
            }
        }
        Ok(())
    })
    .expect("non-finite arithmetic reporting");
}
