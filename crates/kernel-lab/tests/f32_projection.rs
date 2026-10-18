//! Required preparations, F64 gates, poisoned replay and device controls.
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{Refusal, WeightFormat};
use kernel_core::precision::{
    OperationSpec, OperationUse, buffers::ProjectionBuffers, execution::Binding,
};
use kernel_lab::families::{f32_projection as family, f32_projection_mutants as mutants};
use kernel_lab::{
    LabResult, Session, gpu_test,
    kit::{
        self, LabModules, OpResult, Outcome,
        cases::f32_projection::{self as cases, Witness},
    },
};
use kernels_sm120::backend::{F32ProjectionOffers, Sm120};
use std::collections::BTreeMap;

fn backend() -> Sm120 {
    let entries = family::entries();
    let available = entries.iter().map(|e| e.symbol().into()).collect();
    Sm120::new()
        .with_f32_projections(F32ProjectionOffers::new(entries, available, "0".repeat(64)).unwrap())
}
fn required() -> Vec<kit::Required<'static>> {
    let mut ops = cases::family(&family::entries());
    let entry = family::entry(17, 257, 9, WeightFormat::Bf16);
    for witness in [
        Witness::LowComponent,
        Witness::SeparateChains,
        Witness::FoldBoundary,
        Witness::Fallback,
    ] {
        ops.push(cases::required(entry.clone(), witness));
    }
    let raw = family::entry(17, 257, 9, WeightFormat::F32);
    for witness in [Witness::WeightBits, Witness::Fallback] {
        ops.push(cases::required(raw.clone(), witness));
    }
    for weight in [WeightFormat::Bf16, WeightFormat::F32] {
        for witness in [
            Witness::Underflow,
            Witness::NonfiniteInput,
            Witness::NonfiniteWeight,
            Witness::Overflow,
        ] {
            ops.push(cases::required(family::entry(17, 257, 9, weight), witness));
        }
    }
    ops
}
fn run(session: Option<&Session>) -> LabResult<()> {
    let report = kit::conform(&backend(), &LabModules, session, &required())?;
    assert_eq!(report.prepared().len(), 36);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    if let Some(session) = session {
        assert!(
            report.not_exercised().is_empty(),
            "{:?}",
            report.not_exercised()
        );
        session.receipt(
            "exact-f32-projection",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
    }
    Ok(())
}
#[test]
fn prepares_every_required_case() {
    run(None).unwrap();
}
#[test]
fn projections_pass_oracle_poison_graph_and_pdl_on_a_gpu() {
    gpu_test(1 << 30, |session| run(Some(session))).unwrap();
}

struct Mutant {
    fault: &'static str,
}
struct MutantOp {
    inner: Prepared<ProjectionBuffers<View>>,
    info: OpInfo,
    fault: &'static str,
}
impl Backend for Mutant {
    fn name(&self) -> &str {
        "f32-projection-device-control"
    }
    fn projection(
        &self,
        op: &OperationSpec,
        usage: OperationUse,
        need: Need,
    ) -> Result<Binding<Prepared<ProjectionBuffers<View>>>, Refused> {
        let binding = backend().projection(op, usage, need)?;
        let mut info = binding.handle.info().clone();
        for (symbol, _) in &mut info.symbols {
            *symbol = mutants::symbol(symbol, self.fault);
        }
        Ok(Binding {
            identity: binding.identity,
            handle: Box::new(MutantOp {
                inner: binding.handle,
                info,
                fault: self.fault,
            }),
        })
    }
}
impl PreparedOp<ProjectionBuffers<View>> for MutantOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &ProjectionBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(b, workspace, counts)?;
        for launch in &mut launches {
            launch.symbol = mutants::symbol(&launch.symbol, self.fault);
        }
        Ok(launches)
    }
}
#[test]
fn faulty_device_arithmetic_is_rejected_by_unchanged_cases() {
    gpu_test(1<<30,|session| {
        let entry = family::entry(17,257,9,WeightFormat::Bf16);
        for (fault,witness) in [("drop_lo",Witness::LowComponent),("merged_chains",Witness::SeparateChains),("tf32",Witness::LowComponent)] {
            let ops = vec![cases::required(entry.clone(),witness)];
            let positive = kit::conform(&backend(),&LabModules,Some(session),&ops)?;
            assert_eq!(positive.prepared().len(),1);
            assert!(positive.failures().is_empty(),"{:?}",positive.failures());
            assert!(positive.not_exercised().is_empty());
            let report = kit::conform(&Mutant {fault},&LabModules,Some(session),&ops)?;
            assert_eq!(report.prepared().len(),1);
            assert_eq!(report.ops.len(),1);
            let OpResult::Prepared {checks,..} = &report.ops[0].result else {panic!("mutant did not prepare")};
            assert!(checks.iter().any(|check| check.name == "conformance" && matches!(&check.outcome, Outcome::Failed(error) if error.contains("output") && !error.contains("CUDA") && !error.contains("unbound"))),"{fault}: {checks:?}");
            session.receipt(&format!("f32-mutant-{}", fault.replace('_', "-")),BTreeMap::from([("report".into(),report.json())]))?;
        }
        Ok(())
    }).unwrap();
}

#[test]
fn compiled_entries_have_pdl_bf16_mma_and_no_tf32() {
    let toolchain = kernel_build::Toolchain::detect().expect("CUDA toolkit required");
    assert!(!toolchain.skipped);
    let root = std::env::temp_dir().join(format!("f32-projection-audit-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("engine.cubin");
    std::fs::write(&path, kernel_lab::cubin(family::CUBIN).unwrap()).unwrap();
    let sass = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&path)
        .output()
        .unwrap();
    assert!(sass.status.success());
    let text = String::from_utf8(sass.stdout).unwrap();
    assert!(!text.contains("TF32"));
    assert!(!text.contains("TRAP"), "device data must never trap");
    for entry in family::entries() {
        let header = format!("Function : {}\n", entry.symbol());
        let body = text
            .split_once(&header)
            .expect("entry SASS")
            .1
            .split("Function : ")
            .next()
            .unwrap();
        for opcode in ["ACQBULK", "PREEXIT", "FFMA"] {
            assert!(body.contains(opcode), "{} lacks {opcode}", entry.symbol());
        }
        for opcode in ["LDL", "STL"] {
            assert!(
                !body.contains(opcode),
                "{} uses local memory",
                entry.symbol()
            );
        }
        if entry.query().weight == WeightFormat::Bf16 {
            assert!(
                body.contains("HMMA.16816.F32.BF16"),
                "{} lacks BF16 tensor math",
                entry.symbol()
            );
        } else {
            assert!(!body.contains("HMMA"), "F32 weights must use SIMT");
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
