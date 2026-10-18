//! Model-owned compressor cases and frozen artifacts. The CPU test builds
//! and checks locks in its target directory. External GPU snapshots are
//! read-only; in-tree GPU runs wait for the local build.

#[path = "../../../bundle/src/sm120/compressor.rs"]
#[allow(dead_code, reason = "the bundle consumes selection declarations")]
mod entries;
#[path = "compressor_probes.rs"]
mod probes;
#[path = "../../../bundle/src/sm120/rows.rs"]
#[allow(dead_code, reason = "only the latent norm is exercised here")]
mod rows;

use entries::{CompressorKernels, WriteQuery};
use kernel_build::{BuildOptions, CubinSpec, LockDir, Toolchain};
use kernel_core::backend::Need;
use kernel_harness::conformance::Probes;
use kernel_harness::kit::{self, Modules, Required, cases};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::{GEOMETRIES, Geometry};
use runtime_cuda::Module;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const ARTIFACTS: &str = "CHOPINFER_DEEPSEEK41_COMPRESSOR_ARTIFACTS";
// Name, relative cubin file, SHA-256, exported entry symbols.
type Index = Vec<(String, String, String, Vec<String>)>;

fn artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("deepseek41-compressor")
}

fn specifications() -> LabResult<Vec<(&'static str, CubinSpec)>> {
    let mut specs = entries::cubins(GEOMETRIES).map_err(|e| e.to_string())?;
    for geometry in GEOMETRIES {
        specs.push((rows::FAMILY, rows::cubin(geometry)?));
    }
    Ok(specs)
}

fn build_artifacts() -> LabResult<()> {
    let lab = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = artifact_dir();
    let toolchain = Toolchain::detect().map_err(|e| e.to_string())?;
    let specs = specifications()?;
    let mut index = Index::new();
    let mut build = |directory: &str, lock_root: &Path, specs: Vec<(&str, CubinSpec)>| {
        let mut families: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (family, spec) in &specs {
            families.entry(family).or_default().push(&spec.name);
        }
        let lock = families
            .into_iter()
            .fold(LockDir::new(lock_root), |lock, (family, names)| {
                lock.family(family, names)
            });
        let specs: Vec<_> = specs.into_iter().map(|(_, spec)| spec).collect();
        let mut options = BuildOptions::from_env();
        options.emit_cargo_directives = false;
        let built = kernel_build::build_in_families(
            &toolchain,
            &specs,
            &root.join(directory),
            &lock,
            options,
            None,
        )
        .map_err(|e| e.to_string())?;
        assert_eq!(
            built.len(),
            specs.len(),
            "every requested cubin must be audited"
        );
        for cubin in built {
            let file = format!(
                "{directory}/{}",
                cubin.path.file_name().unwrap().to_string_lossy()
            );
            let bytes = std::fs::read(&cubin.path).map_err(|e| e.to_string())?;
            let symbols = specs
                .iter()
                .find(|spec| spec.name == cubin.name)
                .unwrap()
                .entries
                .iter()
                .map(|entry| entry.symbol.clone())
                .collect();
            index.push((cubin.name, file, kernel_build::digest(&bytes), symbols));
        }
        Ok::<_, String>(())
    };
    build("model", &lab.join("../bundle/kernels.lock.d"), specs)?;
    build(
        "probes",
        &lab.join("kernels.lock.d"),
        vec![(probes::FAMILY, probes::cubin(lab))],
    )?;
    std::fs::write(
        root.join("index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .map_err(|e| e.to_string())
}

struct Artifacts {
    cubins: BTreeMap<String, (Vec<u8>, Vec<String>)>,
}

impl Artifacts {
    fn read(root: &Path) -> LabResult<Self> {
        let index: Index = serde_json::from_slice(
            &std::fs::read(root.join("index.json"))
                .map_err(|e| format!("run the CPU artifact test before GPU tests: {e}"))?,
        )
        .map_err(|e| e.to_string())?;
        let mut cubins = BTreeMap::new();
        for (name, file, digest, symbols) in index {
            if Path::new(&file)
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                return Err("artifact path must be confined to the snapshot".into());
            }
            let bytes = std::fs::read(root.join(file)).map_err(|e| e.to_string())?;
            if kernel_build::digest(&bytes) != digest || cubins.contains_key(&name) {
                return Err(format!("changed or duplicated cubin {name}"));
            }
            cubins.insert(name, (bytes, symbols));
        }
        let mut expected = specifications()?
            .into_iter()
            .map(|(_, spec)| spec)
            .collect::<Vec<_>>();
        expected.push(probes::cubin(Path::new(env!("CARGO_MANIFEST_DIR"))));
        if cubins.len() != expected.len() {
            return Err("artifact geometry set differs from the test binary".into());
        }
        for spec in expected {
            let (_, symbols) = cubins.get(&spec.name).ok_or("missing model cubin")?;
            if symbols
                != &spec
                    .entries
                    .iter()
                    .map(|entry| entry.symbol.clone())
                    .collect::<Vec<_>>()
            {
                return Err(format!("entry inventory differs for {}", spec.name));
            }
        }
        Ok(Self { cubins })
    }
}

impl Modules for Artifacts {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<kit::modules::Functions<'s>> {
        let modules: BTreeMap<_, Module> = self
            .cubins
            .iter()
            .map(|(name, (bytes, _))| Ok((name, session.load_cubin(bytes)?)))
            .collect::<LabResult<_>>()?;
        Ok(Box::new(move |symbol| {
            let (name, _) = self
                .cubins
                .iter()
                .find(|(_, (_, entries))| entries.iter().any(|entry| entry == symbol))
                .ok_or_else(|| format!("no model cubin holds {symbol}"))?;
            modules[name].function(symbol).map_err(|e| e.to_string())
        }))
    }

    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let module = session.load_cubin(&self.cubins[probes::CUBIN].0)?;
        Ok(Some(Probes {
            live: LiveProducer::with_function(
                session,
                module.function(probes::LIVE).map_err(|e| e.to_string())?,
            )?,
            copy: Some(CopyProducer::with_function(
                module.function(probes::COPY).map_err(|e| e.to_string())?,
            )),
        }))
    }
}

fn required(geometry: &Geometry) -> LabResult<(CompressorKernels, Vec<Required<'static>>)> {
    let entries = CompressorKernels::serving(geometry).map_err(|e| e.to_string())?;
    let mut ops = cases::gated_pooling::family(&entries.pooling);
    ops.extend(cases::gated_pooling::steps::family(&entries.pooling));
    ops.extend(cases::fp4_key_write::family_for(entries.writes.iter().map(
        |(_, query)| match query {
            WriteQuery::Index(q) => cases::fp4_key_write::Query::Index(*q),
            WriteQuery::Compressed(q) => cases::fp4_key_write::Query::Compressed(*q),
        },
    )));
    let norm = entries
        .ratio_one
        .ok_or("the model must exercise ratio one")?;
    ops.push(
        Required::new(
            format!("{} ratio-one compressor", geometry.id),
            Need::rows(8),
            move |backend, need| backend.rmsnorm(&norm, need),
            move |prepared| cases::gated_pooling::ratio_one_case(norm, 8, prepared),
        )
        .required(),
    );
    assert_eq!(ops.len(), 10);
    assert!(ops.iter().all(|op| op.required));
    Ok((entries, ops))
}

fn run(session: Option<&Session>, artifacts: &Artifacts) -> LabResult<()> {
    let mut preparations = 0;
    for geometry in GEOMETRIES {
        let (entries, required) = required(geometry)?;
        for &ranks in SUPPORTED_RANKS {
            let backend = entries.register(rows::register(Sm120::new(), geometry, ranks));
            let report = kit::conform(&backend, artifacts, session, &required)?;
            if !report.failures().is_empty() {
                return Err(format!(
                    "{} TP{ranks}: {:?}",
                    geometry.id,
                    report.failures()
                ));
            }
            assert_eq!(report.prepared().len(), required.len());
            preparations += report.prepared().len();
            if let Some(session) = session {
                if !report.not_exercised().is_empty() {
                    return Err(format!(
                        "{} TP{ranks}: {:?}",
                        geometry.id,
                        report.not_exercised()
                    ));
                }
                session.receipt(
                    "model-compressor",
                    BTreeMap::from([
                        ("geometry".into(), serde_json::json!(geometry.id)),
                        ("ranks".into(), serde_json::json!(ranks)),
                        ("report".into(), report.json()),
                        (
                            "prefill".into(),
                            cases::gated_pooling::steps::prefill_splits(
                                &backend,
                                artifacts,
                                session,
                                &entries.pooling,
                            )?,
                        ),
                    ]),
                )?;
            }
        }
    }
    assert_eq!(preparations, GEOMETRIES.len() * SUPPORTED_RANKS.len() * 10);
    assert!(preparations > 0);
    Ok(())
}

/// Builds the artifacts once per test process, so a consumer in the same
/// run reads them only after the producer finished, whichever test the
/// harness starts first.
fn built() -> LabResult<()> {
    static BUILT: std::sync::OnceLock<LabResult<()>> = std::sync::OnceLock::new();
    BUILT.get_or_init(build_artifacts).clone()
}

#[test]
fn compressor_model_prepares_and_builds_required_cases() {
    built().unwrap();
    run(None, &Artifacts::read(&artifact_dir()).unwrap()).unwrap();
}

#[test]
fn cpu_artifact_directory_ignores_gpu_snapshot() {
    let target = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let snapshot = target.join("external-compressor-snapshot");
    if std::env::var_os(ARTIFACTS).as_deref() == Some(snapshot.as_os_str()) {
        assert_eq!(artifact_dir(), target.join("deepseek41-compressor"));
        compressor_model_passes_on_a_gpu();
        return;
    }
    // Isolate the override from the parallel test harness. The GPU test must
    // skip before reading this absent snapshot when no GPU is selected.
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "compressor::cpu_artifact_directory_ignores_gpu_snapshot",
            "--nocapture",
        ])
        .env(ARTIFACTS, snapshot)
        .env_remove("CHOPINFER_KERNEL_GPU")
        .env_remove("CHOPINFER_KERNEL_GPUS")
        .env_remove("CHOPINFER_KERNEL_REQUIRE")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}

/// Skips before reading anything when no GPU is selected. A GPU job reads
/// the frozen artifacts its snapshot names in the artifacts variable; an
/// in-tree run reads the ones this process builds.
#[test]
fn compressor_model_passes_on_a_gpu() {
    kernel_harness::gpu_test(1 << 30, |session| {
        let root = if let Some(root) = std::env::var_os(ARTIFACTS) {
            PathBuf::from(root)
        } else {
            built()?;
            artifact_dir()
        };
        run(Some(session), &Artifacts::read(&root)?)
    })
    .unwrap();
}
