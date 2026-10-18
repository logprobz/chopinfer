//! Bitwise retained-window staging through poisoned graph and PDL replay.
#[path = "latent_window_stage/chain.rs"]
mod chain;
use kernel_core::backend::{Backend, Need};
use kernel_core::backend::{Counts, Launch, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{Argument, Refusal};
use kernel_core::latent_attention::LatentKvFormat;
use kernel_core::latent_glue::{LatentWindowWrite, LatentWindowWriteBuffers, WindowAddressing};
use kernel_core::latent_window_stage::{LatentWindowStage, LatentWindowStageBuffers};
use kernel_harness::conformance::Probes;
use kernel_harness::kit::cases::latent_window_stage::{self as cases, Scenario};
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernel_lab_families::latent_window_bf16 as family;
use kernels_sm120::backend::Sm120;
use runtime_cuda::Module;
use std::{cell::RefCell, collections::BTreeMap, path::PathBuf};
fn backend() -> Sm120 {
    let mut b = Sm120::new()
        .with_latent_window_stage(kernel_lab_families::latent_window_stage::entry())
        .with_latent_window_stage(family::stage())
        .with_latent_glue(family::store())
        .with_latent_staged_window_keys(kernel_lab_families::latent_staged_window_keys::entry());
    for e in kernel_lab_families::latent_glue::entries() {
        b = b.with_latent_glue(e);
    }
    for e in kernel_lab_families::latent_glue::quantizers() {
        b = b.with_block_quantize(e);
    }
    for e in kernel_lab_families::latent_window_chain::entries() {
        b = b.with_latent_sources(e);
    }
    for e in family::attention() {
        b = b.with_latent_sources(e);
    }
    b
}
struct FamilyModules {
    directory: PathBuf,
}
impl Modules for FamilyModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let mut lookup = BTreeMap::new();
        for spec in family::cubins()
            .into_iter()
            .chain(kernel_lab_families::latent_staged_window_keys::cubins())
            .chain(kernel_lab_families::latent_window_chain::cubins())
            .chain(kernel_lab_families::latent_glue::cubins())
            .chain(kernel_lab_families::latent_window_stage::cubins())
        {
            for entry in spec.entries {
                lookup.insert(entry.symbol, spec.name.clone());
            }
        }
        let modules = RefCell::new(BTreeMap::<String, Module>::new());
        Ok(Box::new(move |symbol| {
            let name = lookup
                .get(symbol)
                .ok_or_else(|| format!("missing latent glue symbol {symbol}"))?;
            let mut modules = modules.borrow_mut();
            if !modules.contains_key(name) {
                let bytes = std::fs::read(self.directory.join(format!("{name}.cubin")))
                    .map_err(|e| e.to_string())?;
                modules.insert(name.clone(), session.load_cubin(&bytes)?);
            }
            modules[name].function(symbol).map_err(|e| e.to_string())
        }))
    }
    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let bytes = std::fs::read(
            self.directory
                .join(format!("{}.cubin", kernels_sm120::probes::CUBIN)),
        )
        .map_err(|e| e.to_string())?;
        let module = session.load_cubin(&bytes)?;
        Ok(Some(Probes {
            live: LiveProducer::with_function(
                session,
                module
                    .function(kernels_sm120::probes::LIVE)
                    .map_err(|e| e.to_string())?,
            )?,
            copy: Some(CopyProducer::with_function(
                module
                    .function(kernels_sm120::probes::COPY)
                    .map_err(|e| e.to_string())?,
            )),
        }))
    }
}

fn query(addressing: WindowAddressing) -> LatentWindowWrite {
    LatentWindowWrite {
        kv: LatentKvFormat::Bf16,
        latent: 512,
        rotary: 64,
        epsilon_bits: 1e-20_f32.to_bits(),
        addressing,
    }
}
fn required() -> Vec<kit::Required<'static>> {
    use kit::cases::latent_glue as glue;
    let mut result = Vec::new();
    for addressing in [
        WindowAddressing::Circular { slots: 136 },
        WindowAddressing::Staged { slots: 256 },
    ] {
        let q = query(addressing);
        result.extend([
            glue::window_write(q, 32, 127),
            glue::window_write(q, 320, 140),
            glue::window_status(q),
            glue::window_f32_regression(q),
            glue::window_invalid_data(q),
            glue::window_status_boundary(q, false),
            glue::window_status_boundary(q, true),
        ]);
    }
    for scenario in Scenario::ALL {
        result.push(
            kit::Required::new(
                format!("BF16 stage {scenario:?}"),
                Need::rows(scenario.capacity()),
                move |b, n| {
                    b.latent_window_stage(
                        &LatentWindowStage {
                            kv: LatentKvFormat::Bf16,
                            ..cases::QUERY
                        },
                        n,
                    )
                },
                move |op| cases::case_with_format(LatentKvFormat::Bf16, scenario, op),
            )
            .required(),
        );
    }
    result.push(kit::cases::latent_staged_window_keys::required());
    result
}
struct StoreMutant {
    inner: Prepared<LatentWindowWriteBuffers<View>>,
    info: OpInfo,
    truncate: bool,
}
impl PreparedOp<LatentWindowWriteBuffers<View>> for StoreMutant {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentWindowWriteBuffers<View>,
        w: View,
        c: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut calls = self.inner.launches(b, w, c)?;
        let store = calls.last_mut().unwrap();
        store.symbol = if self.truncate {
            family::TRUNCATE
        } else {
            family::SWAP
        }
        .into();
        if self.truncate {
            store
                .arguments
                .0
                .push(Argument::DeviceBuffer(b.latent.handle));
        }
        Ok(calls)
    }
}
struct StageMutant {
    inner: Prepared<LatentWindowStageBuffers<View>>,
    info: OpInfo,
}
impl PreparedOp<LatentWindowStageBuffers<View>> for StageMutant {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentWindowStageBuffers<View>,
        w: View,
        c: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        // The fixture's unread scale guard has the full source extent.
        b.ring.scales.covers(b.ring.codes.bytes)?;
        let mut calls = self.inner.launches(b, w, c)?;
        calls[0].symbol = family::STAGE_SWAP.into();
        calls[0].arguments.0[0] = Argument::DeviceBuffer(b.ring.scales.handle);
        Ok(calls)
    }
}
fn controls() -> Vec<kit::Required<'static>> {
    let mut result = Vec::new();
    for truncate in [true, false] {
        let q = query(WindowAddressing::Circular { slots: 136 });
        result.push(
            kit::Required::new(
                format!("BF16 store mutant truncate={truncate}"),
                Need::rows(32),
                move |b, n| {
                    let inner = b.latent_window_write(&q, n)?;
                    let mut info = inner.info().clone();
                    info.symbols.last_mut().unwrap().0 = if truncate {
                        family::TRUNCATE
                    } else {
                        family::SWAP
                    }
                    .into();
                    Ok(Box::new(StoreMutant {
                        inner,
                        info,
                        truncate,
                    })
                        as Prepared<LatentWindowWriteBuffers<View>>)
                },
                move |op| kit::cases::latent_glue::write_case(q, 32, 140, op),
            )
            .required(),
        );
    }
    result.push(
        kit::Required::new(
            "BF16 staging swapped plane",
            Need::rows(2),
            move |b, n| {
                let inner = b.latent_window_stage(
                    &LatentWindowStage {
                        kv: LatentKvFormat::Bf16,
                        ..cases::QUERY
                    },
                    n,
                )?;
                let mut info = inner.info().clone();
                info.symbols[0].0 = family::STAGE_SWAP.into();
                Ok(Box::new(StageMutant { inner, info })
                    as Prepared<LatentWindowStageBuffers<View>>)
            },
            move |op| cases::case_with_format(LatentKvFormat::Bf16, Scenario::Wrap, op),
        )
        .required(),
    );
    result
}
fn check(report: &kit::Report, gpu: bool) {
    assert_eq!(report.prepared().len(), 22, "{:?}", report.ops);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("refused")
            };
            for c in checks {
                assert!(
                    matches!(c.outcome, Outcome::Passed(_)),
                    "{} {}: {:?}",
                    op.name,
                    c.name,
                    c.outcome
                );
            }
        }
    }
}
#[test]
fn bf16_window_cases_prepare_and_mutants_change_expected_bits() {
    let modules = FamilyModules {
        directory: PathBuf::new(),
    };
    check(
        &kit::conform(&backend(), &modules, None, &required()).unwrap(),
        false,
    );
    let controls = kit::conform(&backend(), &modules, None, &controls()).unwrap();
    assert_eq!(controls.prepared().len(), 3);
    assert!(controls.failures().is_empty(), "{:?}", controls.failures());
    let q = query(WindowAddressing::Circular { slots: 136 });
    let case = kit::cases::latent_glue::write_case(
        q,
        32,
        140,
        backend().latent_window_write(&q, Need::rows(32)).unwrap(),
    )
    .unwrap();
    let images = case.images();
    let (latent, encoded, flags) = kernel_core::latent_glue_oracle::window_rows_status(
        q,
        &images.f32s("input").unwrap(),
        &images.u16s("weight").unwrap(),
        &images.f32s("cos_sin").unwrap(),
        &[0; 32],
    )
    .unwrap();
    assert!(flags.iter().all(|&f| f == 0));
    let truncated = latent
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let exact = encoded
        .iter()
        .flat_map(|r| kernel_core::latent_glue_oracle::exact_window_bf16(r, 512).unwrap())
        .collect::<Vec<_>>();
    assert_ne!(truncated, exact, "unquantized truncation must differ");
    let mut swapped = encoded[0].clone();
    for i in 0..512 {
        swapped.values[i] = encoded[0].scale_bytes[i / 32];
    }
    for i in 0..16 {
        swapped.scale_bytes[i] = encoded[0].values[i * 32];
    }
    assert!(
        kernel_core::latent_glue_oracle::exact_window_bf16(&swapped, 512)
            .map_or(true, |v| v != exact[..1024])
    );
}
#[test]
fn bf16_window_passes_poison_graph_pdl_and_controls_on_gpu() {
    let result = kernel_harness::gpu_test(512 << 20, |session| {
        let directory = PathBuf::from(
            std::env::var_os("CHOPINFER_WINDOW_BF16_CUBINS")
                .ok_or("prebuilt BF16 window cubins required")?,
        );
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let modules = FamilyModules { directory };
        let report = kit::conform(&backend(), &modules, Some(session), &required())?;
        session.receipt(
            "window-bf16",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        check(&report, true);
        for control in controls() {
            let report = kit::conform(&backend(), &modules, Some(session), &[control])?;
            session.receipt(
                "window-bf16-control",
                BTreeMap::from([("report".into(), report.json())]),
            )?;
            let OpResult::Prepared { checks, .. } = &report.ops[0].result else {
                panic!("mutant refused")
            };
            assert!(checks.iter().any(|c|c.name=="conformance" && matches!(&c.outcome,Outcome::Failed(e) if e.contains("window.codes") || e.contains("staged.codes") || e.contains("status"))),"mutant escaped: {checks:?}");
            for c in checks {
                if let Outcome::Failed(e) = &c.outcome {
                    assert!(
                        !e.contains("CUDA") && !e.contains("trap") && !e.contains("illegal"),
                        "{e}"
                    );
                }
            }
        }
        chain::compare_bf16(session, &modules)?;
        Ok(())
    });
    result.unwrap();
}

#[test]
fn bf16_window_first_use_on_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let directory = PathBuf::from(
            std::env::var_os("CHOPINFER_WINDOW_BF16_CUBINS")
                .ok_or("prebuilt BF16 window cubins required")?,
        );
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let modules = FamilyModules { directory };
        let selected = required()
            .into_iter()
            .enumerate()
            .filter_map(|(i, c)| matches!(i, 0 | 16).then_some(c))
            .collect::<Vec<_>>();
        let report = kit::conform(&backend(), &modules, Some(session), &selected)?;
        assert_eq!(report.prepared().len(), 2);
        assert!(report.failures().is_empty(), "{:?}", report.failures());
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("refused")
            };
            for c in checks {
                assert!(
                    matches!(c.outcome, Outcome::Passed(_)),
                    "{} {} {:?}",
                    op.name,
                    c.name,
                    c.outcome
                );
            }
        }
        session.receipt(
            "bf16-window-first-use",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        Ok(())
    })
    .unwrap();
}
