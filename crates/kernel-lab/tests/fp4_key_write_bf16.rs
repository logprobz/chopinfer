//! Native-code products and BF16 comparison storage have identical bits.

use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::compressed_kv::{CompressedKvWriteBuffers, CompressedKvWriteQuery};
use kernel_core::contract::Refusal;
use kernel_core::pooled_indexer::{
    IndexerKeyFormat, PositionIndexerQuery, PositionKeyWriteBuffers,
};
use kernel_lab::families::{fp4_key_write_bf16 as family, fp4_key_write_bf16_mutants as mutants};
use kernel_lab::kit::{self, LabModules, OpResult, Outcome, cases::fp4_key_write as cases};
use kernel_lab::{LabResult, gpu_test};
use kernels_sm120::backend::Sm120;
use std::collections::BTreeMap;

fn backend() -> Sm120 {
    Sm120::new().with_fp4_key_writes(family::all_entries())
}

#[test]
fn every_bf16_write_case_prepares() {
    let report = kit::conform(&backend(), &LabModules, None, &cases::bf16_family()).unwrap();
    assert_eq!(report.prepared().len(), 30);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
}

#[test]
fn bf16_writes_match_native_products_poison_and_pdl_on_a_gpu() {
    gpu_test(1 << 30, |session| {
        let backend = backend();
        let report = kit::conform(&backend, &LabModules, Some(session), &cases::bf16_family())?;
        assert_eq!(report.prepared().len(), 30);
        assert!(report.failures().is_empty(), "{:?}", report.failures());
        assert!(
            report.not_exercised().is_empty(),
            "{:?}",
            report.not_exercised()
        );
        let equality = cases::compare_storage_on_device(&backend, &LabModules, session)?;
        assert_eq!(equality["launch_pairs"], 90);
        session.receipt(
            "fp4-bf16-write",
            BTreeMap::from([
                ("report".into(), report.json()),
                ("equality".into(), equality),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}

struct Mutant {
    backend: Sm120,
    fault: &'static str,
}
struct Op<B> {
    inner: Prepared<B>,
    info: OpInfo,
    fault: &'static str,
}
impl<B: 'static> Op<B> {
    fn wrap(inner: Prepared<B>, fault: &'static str) -> Prepared<B> {
        let mut info = inner.info().clone();
        for (symbol, _) in &mut info.symbols {
            *symbol = mutants::symbol(symbol, fault);
        }
        Box::new(Self { inner, info, fault })
    }
}
impl<B> PreparedOp<B> for Op<B> {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(buffers, workspace, counts)?;
        for launch in &mut launches {
            launch.symbol = mutants::symbol(&launch.symbol, self.fault);
        }
        Ok(launches)
    }
}
impl Backend for Mutant {
    fn name(&self) -> &str {
        "bf16-write-store-control"
    }
    fn position_indexer_key_write_with_storage(
        &self,
        q: &PositionIndexerQuery,
        storage: IndexerKeyFormat,
        need: Need,
    ) -> Result<Prepared<PositionKeyWriteBuffers<View>>, Refused> {
        self.backend
            .position_indexer_key_write_with_storage(q, storage, need)
            .map(|op| Op::wrap(op, self.fault))
    }
    fn compressed_kv_write(
        &self,
        q: &CompressedKvWriteQuery,
        need: Need,
    ) -> Result<Prepared<CompressedKvWriteBuffers<View>>, Refused> {
        self.backend
            .compressed_kv_write(q, need)
            .map(|op| Op::wrap(op, self.fault))
    }
}

#[test]
fn truncated_input_and_dropped_scale_fail_unchanged_cases_on_a_gpu() {
    gpu_test(1<<30, |session| {
        let required=cases::family_for(cases::comparison_queries().into_iter().map(cases::bf16_storage));
        let positive=kit::conform(&backend(),&LabModules,Some(session),&required)?;
        assert_eq!(positive.prepared().len(),4);
        assert!(positive.failures().is_empty(),"{:?}",positive.failures());
        for fault in mutants::FAULTS {
            let report=kit::conform(&Mutant{backend:backend(),fault},&LabModules,Some(session),&required)?;
            assert_eq!(report.prepared().len(),4);
            for op in &report.ops {
                let OpResult::Prepared{checks,..}=&op.result else {panic!("mutant refused")};
                assert!(checks.iter().any(|c| c.name=="conformance" && matches!(&c.outcome,Outcome::Failed(e) if e.contains("codes") && !e.contains("CUDA") && !e.contains("unbound"))),"{fault}: {checks:?}");
            }
            session.receipt("fp4-bf16-write-mutant",BTreeMap::from([("fault".into(),serde_json::json!(fault)),("report".into(),report.json())]))?;
        }
        Ok(())
    }).unwrap();
}

#[test]
fn bf16_write_cubin_has_pdl_without_traps_tf32_or_spills() -> LabResult<()> {
    let toolchain = kernel_build::Toolchain::detect().map_err(|e| e.to_string())?;
    assert!(!toolchain.skipped);
    let root = kernel_lab::receipt_root().join(format!("bf16-write-sass-{}", std::process::id()));
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let cubin = root.join("writer.cubin");
    std::fs::write(&cubin, kernel_lab::cubin(family::CUBIN).unwrap()).map_err(|e| e.to_string())?;
    let output = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&cubin)
        .output()
        .map_err(|e| e.to_string())?;
    assert!(output.status.success());
    let sass = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    for forbidden in ["TRAP", "TF32", "LDL", "STL"] {
        assert!(!sass.contains(forbidden), "{forbidden}");
    }
    assert!(sass.contains("ACQBULK") && sass.contains("PREEXIT"));
    std::fs::remove_dir_all(root).map_err(|e| e.to_string())?;
    Ok(())
}
