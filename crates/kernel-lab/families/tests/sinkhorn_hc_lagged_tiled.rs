//! Exact prefill row tiling, intermediate probes, inactive rows and timing.
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::Argument;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::poison;
use kernel_harness::{Cubins, LabResult, Session};
use kernel_lab_families::sinkhorn_hc_lagged_tiled as family;
use kernels_sm120::sinkhorn_hc::{lagged, tiled};
use runtime_cuda::{DeviceBuffer, Module};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

const H: usize = 5120;
const N: usize = 4;
const FLAT: usize = H * N;
const MIXES: usize = 24;
const WIDTH_BYTES: [usize; 7] = [H * 2, N * 4, N * 4, N * N * 4, MIXES * 4, 4, N * N * 4];
const NAMES: [&str; 7] = [
    "collapsed",
    "pre",
    "post",
    "comb",
    "projection",
    "rstd",
    "cells",
];
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn reference(probe: bool) -> lagged::Entry {
    lagged::lab_catalog()
        .into_iter()
        .find(|e| e.op == lagged::Op::MixCollapse { probe })
        .unwrap()
}
const TIMING_CUBIN: &str = "probe_sm120";
const TIMING_SYMBOLS: &[&str] = &[
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
    "chop_probe_gate_v1",
];

fn timing_identity() -> LabResult<Value> {
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kernels.lock");
    let records: Vec<Value> =
        serde_json::from_slice(&std::fs::read(lock).map_err(err)?).map_err(err)?;
    records
        .into_iter()
        .find(|record| record["name"] == TIMING_CUBIN)
        .ok_or("kernel-lab lock lacks the original timing probe".into())
}

fn timing_symbols(identity: &Value) -> LabResult<Vec<String>> {
    identity["entries"]
        .as_array()
        .ok_or("probe identity has no entries")?
        .iter()
        .map(|entry| {
            entry["symbol"]
                .as_str()
                .map(str::to_owned)
                .ok_or("probe identity entry has no symbol".into())
        })
        .collect()
}

// The family modules used by all three GPU paths. The original kernel-lab
// timing module is added separately, checked against its unchanged lock.
fn gpu_specs() -> Vec<kernel_build::CubinSpec> {
    family::cubins()
        .into_iter()
        .chain(lagged::lab_cubin_specs())
        .chain([kernels_sm120::probes::cubin_spec()])
        .collect()
}

fn named_cubins(images: Vec<(String, Vec<u8>, Vec<String>)>) -> Cubins {
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    // Cubins keeps static tables. The staged image set is loaded once per
    // test process, then retained for every session that uses its functions.
    for (name, image, symbols) in images {
        let name: &'static str = Box::leak(name.into_boxed_str());
        let image: &'static [u8] = Box::leak(image.into_boxed_slice());
        let symbols: &'static [&'static str] = Box::leak(
            symbols
                .into_iter()
                .map(|symbol| &*Box::leak(symbol.into_boxed_str()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        bytes.push((name, image));
        entries.push((name, symbols));
    }
    Cubins::new(
        Box::leak(bytes.into_boxed_slice()),
        Box::leak(entries.into_boxed_slice()),
    )
}

fn staged_cubins() -> LabResult<Cubins> {
    static CUBINS: std::sync::OnceLock<LabResult<Cubins>> = std::sync::OnceLock::new();
    CUBINS
        .get_or_init(|| {
            let root = PathBuf::from(
                std::env::var_os("CHOPINFER_LAGGED_TILED_CUBINS")
                    .ok_or("prebuilt lagged tiled cubins required")?,
            );
            let report =
                kernel_build::report::BuildReport::read(&root.join("kernel-build-report.json"))
                    .map_err(err)?;
            report.verify_current().map_err(err)?;
            let mut images = gpu_specs()
                .into_iter()
                .map(|spec| {
                    if !report.cubins.iter().any(|cubin| cubin.name == spec.name) {
                        return Err(format!("staged build report lacks {}", spec.name));
                    }
                    let image =
                        std::fs::read(root.join(format!("{}.cubin", spec.name))).map_err(err)?;
                    Ok((
                        spec.name,
                        image,
                        spec.entries.into_iter().map(|entry| entry.symbol).collect(),
                    ))
                })
                .collect::<LabResult<Vec<_>>>()?;
            let identity = timing_identity()?;
            let image = std::fs::read(root.join(format!("{TIMING_CUBIN}.cubin"))).map_err(err)?;
            if identity["cubin_sha256"].as_str() != Some(kernel_build::digest(&image).as_str()) {
                return Err("staged probe_sm120 differs from the original kernel-lab lock".into());
            }
            images.push((TIMING_CUBIN.into(), image, timing_symbols(&identity)?));
            Ok(named_cubins(images))
        })
        .clone()
}

fn gpu_test(run: impl FnOnce(&Session) -> LabResult<()>) -> LabResult<Option<()>> {
    if kernel_harness::selected_ordinals()?.is_none() {
        return kernel_harness::gpu_test(512 << 20, run);
    }
    kernel_harness::gpu_test_with(staged_cubins()?, 512 << 20, run)
}

#[test]
fn cpu_resolves_every_module_symbol_and_launch() {
    let specs = gpu_specs();
    let mut symbols = std::collections::BTreeSet::new();
    for spec in &specs {
        assert!(!spec.entries.is_empty());
        for e in &spec.entries {
            assert!(symbols.insert(e.symbol.clone()));
        }
    }
    for tile in tiled::ROWS {
        for probe in [false, true] {
            let e = family::entry(tile, probe);
            assert!(symbols.contains(&e.reference.symbol));
            for rows in [1, 7, 1024, 2048, 8192] {
                let p = e.plan(rows, rows).unwrap();
                assert_eq!(p.block, [256, 1, 1]);
                assert_eq!(p.cluster, [1, 1, 1]);
                assert_eq!(p.dynamic_shared_bytes, 0);
                assert_eq!(p.grid, [rows.div_ceil(tile), 1, 1]);
            }
        }
    }
    assert!(symbols.contains(&family::mutant().reference.symbol));
    let identity = timing_identity().unwrap();
    let timing = timing_symbols(&identity).unwrap();
    for symbol in TIMING_SYMBOLS {
        assert!(timing.iter().any(|s| s == symbol));
    }
    if std::env::var_os("CHOPINFER_LAGGED_TILED_CUBINS").is_some() {
        let staged = staged_cubins().unwrap();
        for spec in &specs {
            let image = staged.bytes(&spec.name).unwrap();
            assert!(!image.is_empty());
            for e in &spec.entries {
                assert!(
                    staged
                        .entries(&spec.name)
                        .unwrap()
                        .contains(&e.symbol.as_str())
                );
                // Cubins contain the ELF symbol strings. The job also checks
                // GLOBAL FUNC definitions using readelf before GPU execution.
                let name = format!("{}\0", e.symbol);
                assert!(image.windows(name.len()).any(|s| s == name.as_bytes()));
            }
        }
        let image = staged.bytes(TIMING_CUBIN).unwrap();
        for symbol in TIMING_SYMBOLS {
            let name = format!("{symbol}\0");
            assert!(image.windows(name.len()).any(|s| s == name.as_bytes()));
        }
    }
}

struct Fixture {
    buffers: Vec<DeviceBuffer>,
    rows: usize,
}
impl Fixture {
    fn new(session: &Session, rows: usize, seed: u64) -> LabResult<Self> {
        let mut x = kernel_harness::kit::bf16_values(seed, rows * FLAT);
        // Finite extremes, signed zero and subnormals supplement random rows.
        if rows > 3 {
            x[..FLAT].fill(0);
            for (i, v) in x[FLAT..2 * FLAT].iter_mut().enumerate() {
                *v = if i % 2 == 0 { 1 } else { 0x8001 };
            }
            for (i, v) in x[2 * FLAT..3 * FLAT].iter_mut().enumerate() {
                *v = if i % 2 == 0 { 0x6000 } else { 0xe000 };
            }
            x[3 * FLAT] = 0x8000;
        }
        let weights: Vec<_> = kernel_harness::kit::bf16_values(seed + 41, MIXES * FLAT)
            .into_iter()
            .map(|v| bf16_to_f32(v) / 128.0 + 2_f32.powi(-22))
            .collect();
        let base: Vec<_> = (0..MIXES).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
        let pre: Vec<_> = (0..rows * N)
            .map(|i| 0.0625 + (i % 7) as f32 * 0.125 + 2_f32.powi(-20))
            .collect();
        let mut buffers = vec![
            session.upload_bf16(&x)?,
            session.upload_f32(&weights)?,
            session.upload_f32(&base)?,
            session.upload_f32(&[0.5, 0.25, 1.0])?,
            session.upload_f32(&pre)?,
            session.allocate(4)?,
        ];
        for _ in 0..2 {
            for bytes in WIDTH_BYTES {
                buffers.push(session.allocate(rows * bytes)?);
            }
        }
        Ok(Self { buffers, rows })
    }
    fn reset(&self, session: &Session, live: usize) -> LabResult<()> {
        self.buffers[5]
            .copy_from_bytes(&(live as u32).to_le_bytes(), &session.stream)
            .map_err(err)?;
        for b in &self.buffers[6..] {
            b.copy_from_bytes(&vec![0xa5; b.byte_len()], &session.stream)
                .map_err(err)?;
        }
        Ok(())
    }
    fn fire(
        &self,
        session: &Session,
        module: &Module,
        tile: u32,
        probe: bool,
        mutant: bool,
    ) -> LabResult<()> {
        let output = if tile == 0 { 6 } else { 13 };
        let values = (0..5)
            .chain(output..output + if probe { 7 } else { 4 })
            .chain([5])
            .map(Argument::DeviceBuffer)
            .collect();
        let (entry, mut plan) = if tile == 0 {
            let e = reference(probe);
            let p = e.plan(self.rows as u32, 0).map_err(err)?;
            (e, p)
        } else {
            let e = if mutant {
                family::mutant()
            } else {
                family::entry(tile, probe)
            };
            let p = e.plan(self.rows as u32, 0).map_err(err)?;
            (e.reference, p)
        };
        // Standalone event timing and comparisons do not enable PDL.
        plan.pdl = false;
        let args = entry.pack_arguments(values).map_err(err)?;
        let bindings = Bindings {
            buffers: self
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64, b))
                .collect(),
        };
        launch(&session.context, &module.function(&entry.symbol).map_err(err)?,
            &session.stream, plan, &args, &bindings)
            .map_err(|e| format!("rows={} tile={tile} probe={probe} symbol={} block={:?} cluster={:?} dynamic_shared={} max_dynamic_shared_attribute=0: {e}",
                self.rows, entry.symbol, plan.block, plan.cluster, plan.dynamic_shared_bytes))
    }
    fn output(&self, session: &Session, index: usize) -> LabResult<Vec<u8>> {
        poison::read(session, &self.buffers[index])
    }
    fn compare(&self, session: &Session, live: usize, count: usize) -> LabResult<()> {
        for (i, name) in NAMES.iter().enumerate().take(count) {
            let old = self.output(session, 6 + i)?;
            let new = self.output(session, 13 + i)?;
            if let Some(at) = old.iter().zip(&new).position(|(a, b)| a != b) {
                return Err(format!(
                    "rows={} live={live} {name} byte={at}: {:02x} != {:02x}",
                    self.rows, old[at], new[at]
                ));
            }
            if new[live * WIDTH_BYTES[i]..].iter().any(|&b| b != 0xa5) {
                return Err(format!("{name}: inactive output overwritten"));
            }
        }
        Ok(())
    }
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_LAGGED_TILED_CUBINS"]
fn prefill_tiled_bitwise_all_outputs_on_gpu() {
    let ran = gpu_test(|session| {
        let old = session.module(&lagged::probe_cubin())?;
        let new = session.module(family::CUBIN)?;
        let changed = session.module(family::MUTANT)?;
        let mut records = vec![];
        for rows in [1, 7, 8, 17, 33, 1024, 2048, 8192] {
            for seed in [17, 97] {
                let fixture = Fixture::new(session, rows, seed)?;
                for live in [0, rows.saturating_sub(1), rows] {
                    for tile in tiled::ROWS {
                        fixture.reset(session, live)?;
                        fixture.fire(session, &old, 0, true, false)?;
                        fixture.fire(session, &new, tile, true, false)?;
                        fixture.compare(session, live, 7)?;
                        println!("L10_HC_BITS rows={rows} live={live} tile={tile} seed={seed}");
                        records.push(json!({"rows":rows,"live":live,"tile":tile,"seed":seed}));
                    }
                }
                if rows == 33 {
                    fixture.reset(session, rows)?;
                    fixture.fire(session, &old, 0, true, false)?;
                    fixture.fire(session, &changed, 8, true, true)?;
                    assert_ne!(
                        fixture.output(session, 10)?,
                        fixture.output(session, 17)?,
                        "changed butterfly escaped the projection comparison"
                    );
                    println!("L10_HC_NEGATIVE seed={seed}");
                }
            }
        }
        session.receipt(
            "lagged-tiled-exact",
            BTreeMap::from([("cases".into(), json!(records))]),
        )?;
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(ran.is_some(), "required bitwise comparison skipped");
    }
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_LAGGED_TILED_CUBINS"]
fn prefill_tiled_timing_on_gpu() {
    let ran = gpu_test(|session| {
        let old = session.module(&lagged::served_cubin())?;
        let new = session.module(family::CUBIN)?;
        let mut records = vec![];
        for rows in [1024, 2048, 8192] {
            let fixture = Fixture::new(session, rows, 1337)?;
            fixture.reset(session, rows)?;
            fixture.fire(session, &old, 0, false, false)?;
            let baseline = kernel_harness::time_named(session, "HC one-row", 2, 11,
                |_| fixture.fire(session, &old, 0, false, false))?;
            for tile in tiled::ROWS {
                fixture.fire(session, &new, tile, false, false)?;
                fixture.compare(session, rows, 4)?;
                let measured = kernel_harness::time_named(session, "HC row tile", 2, 11,
                    |_| fixture.fire(session, &new, tile, false, false))?;
                fixture.compare(session, rows, 4)?;
                let record = json!({"rows":rows,"tile":tile,"old_ms":baseline.median_ms,"new_ms":measured.median_ms});
                println!("L10_HC_TIMING {record}");
                records.push(record);
            }
        }
        session.receipt("lagged-tiled-timing", BTreeMap::from([("cases".into(), json!(records))]))?;
        Ok(())
    }).unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(ran.is_some(), "required timing skipped");
    }
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_LAGGED_TILED_CUBINS"]
fn prefill_tiled_pdl_and_row_properties_on_gpu() {
    use kernel_core::backend::{Backend, Need};
    use kernel_core::properties::Invariance;
    use kernel_harness::conformance::Probes;
    use kernel_harness::kit::{Outcome, cases::sinkhorn_lagged as kit};
    use kernel_harness::poison::{CopyProducer, LiveProducer};
    use kernels_sm120::backend::Sm120;
    let ran = gpu_test(|session| {
        let query = reference(false).query;
        let need = Need::rows(4).with_exact(Invariance::any_rows(4));
        let mut entry = family::entry(8, false);
        entry.from_rows = 1;
        let prepared = Sm120::new()
            .with_sinkhorn_lagged(lagged::catalog())
            .with_sinkhorn_lagged_tiled([entry])
            .sinkhorn_lagged_mix_collapse(&query, need)
            .map_err(err)?;
        let case = kit::mix_collapse_case(query, 4, prepared)?;
        let old = session.module(&lagged::served_cubin())?;
        let new = session.module(family::CUBIN)?;
        let module = session.module(kernels_sm120::probes::CUBIN)?;
        let probes = Probes {
            live: LiveProducer::with_function(
                session,
                module.function(kernels_sm120::probes::LIVE).map_err(err)?,
            )?,
            copy: Some(CopyProducer::with_function(
                module.function(kernels_sm120::probes::COPY).map_err(err)?,
            )),
        };
        let lookup = |symbol: &str| {
            new.function(symbol)
                .or_else(|_| old.function(symbol))
                .map_err(err)
        };
        let checks = case.checks(need, Some(session), Some(&lookup), Some(&probes));
        assert!(!checks.is_empty());
        for check in &checks {
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{}: {:?}",
                check.name,
                check.outcome
            );
        }
        println!("L10_HC_PROPERTIES checks={}", checks.len());
        session.receipt(
            "lagged-tiled-properties",
            BTreeMap::from([(
                "checks".into(),
                json!(checks.iter().map(|c| &c.name).collect::<Vec<_>>()),
            )]),
        )?;
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(ran.is_some(), "required PDL comparison skipped");
    }
}
