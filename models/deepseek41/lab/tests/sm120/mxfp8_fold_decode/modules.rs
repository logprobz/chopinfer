//! CPU validation of every module used by the fold comparison job.
use crate::dense_decode_reference as mxfp8_fold;
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_harness::{Cubins, LabResult};
use model_deepseek41_contract::{geometry::FLASH, serving::PREFILL_TOKENS};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const TIMING: &str = "probe_sm120";
// time_named uses the gate and both cache-eviction modes. The conformance
// kit uses authenticated shared dependency probes through ModelModules.
const TIMING_ENTRIES: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];

struct Expected {
    digest: String,
    symbols: BTreeSet<String>,
    embedded: Option<&'static [u8]>,
}

fn expected() -> LabResult<BTreeMap<String, Expected>> {
    let mut result = BTreeMap::<String, Expected>::new();
    let mut add = |symbol: String| -> LabResult<()> {
        let (name, digest, embedded) = if let Some((name, bytes)) = fold_cubin_for_symbol(&symbol) {
            (name.to_owned(), kernel_build::digest(bytes), Some(bytes))
        } else {
            let module = mxfp8_fold::modules()
                .iter()
                .find(|m| m.symbols.contains(&symbol))
                .ok_or_else(|| format!("missing fold symbol {symbol}"))?;
            (module.name.clone(), module.digest.clone(), None)
        };
        result
            .entry(name)
            .or_insert_with(|| Expected {
                digest,
                symbols: BTreeSet::new(),
                embedded,
            })
            .symbols
            .insert(symbol);
        Ok(())
    };
    for entry in mxfp8_fold::decode_catalog(&FLASH)? {
        for (index, symbol) in entry.symbols().into_iter().enumerate() {
            add(if index == 1 {
                kernels_sm120::gemm::mxfp8_fold_packed::symbol(&symbol)
            } else {
                symbol
            })?;
        }
    }
    for entry in mxfp8_fold::ticket_catalog(&FLASH)? {
        for (index, symbol) in entry.symbols().into_iter().enumerate() {
            add(if index == 1 {
                kernels_sm120::gemm::mxfp8_fold_packed::symbol(&symbol)
            } else {
                symbol
            })?;
        }
    }
    for entry in mxfp8_fold::prefetch_catalog(&FLASH)? {
        for (index, symbol) in entry.symbols().into_iter().enumerate() {
            add(if index == 1 {
                kernels_sm120::gemm::mxfp8_fold_packed::symbol(&symbol)
            } else {
                symbol
            })?;
        }
    }
    for entry in mxfp8_fold::catalog(&FLASH)?
        .into_iter()
        .filter(|e| e.query.m_capacity == PREFILL_TOKENS)
    {
        add(entry.packed_spec().map_err(|e| e.to_string())?.symbol)?;
    }
    let (bytes, symbols) = crate::dependency_probes::get();
    for required in [kernels_sm120::probes::LIVE, kernels_sm120::probes::COPY] {
        if !symbols.contains(&required) {
            return Err(format!("missing dependency probe {required}"));
        }
    }
    result.insert(
        kernels_sm120::probes::CUBIN.into(),
        Expected {
            digest: kernel_build::digest(bytes),
            symbols: [kernels_sm120::probes::LIVE, kernels_sm120::probes::COPY]
                .into_iter()
                .map(String::from)
                .collect(),
            embedded: Some(bytes),
        },
    );
    // Pin the external probe to the existing kernel-lab lock. No GPU or
    // kernel-lab dependency is needed to check its identity and exports.
    let lock: Vec<Value> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../crates/kernel-lab/kernels.lock"
    )))
    .map_err(|e| e.to_string())?;
    let probe = lock
        .iter()
        .find(|record| record["name"] == TIMING)
        .ok_or("missing timing probe lock")?;
    let exports: BTreeSet<_> = probe["entries"]
        .as_array()
        .ok_or("missing timing probe entries")?
        .iter()
        .filter_map(|entry| entry["symbol"].as_str())
        .collect();
    for symbol in TIMING_ENTRIES {
        if !exports.contains(symbol) {
            return Err(format!("timing probe lock omits {symbol}"));
        }
    }
    result.insert(
        TIMING.into(),
        Expected {
            digest: probe["cubin_sha256"]
                .as_str()
                .ok_or("missing timing probe digest")?
                .into(),
            symbols: TIMING_ENTRIES.iter().map(|s| (*s).into()).collect(),
            embedded: None,
        },
    );
    Ok(result)
}

pub(crate) fn directory() -> LabResult<PathBuf> {
    std::env::var_os("CHOPINFER_DEEPSEEK41_FOLD_CUBINS")
        .map(PathBuf::from)
        .ok_or("set CHOPINFER_DEEPSEEK41_FOLD_CUBINS to the staged module directory".into())
}

fn read_module(dir: &Path, name: &str, module: &Expected) -> LabResult<Vec<u8>> {
    let bytes = std::fs::read(dir.join(format!("{name}.cubin")))
        .map_err(|e| format!("required module {name}: {e}"))?;
    if kernel_build::digest(&bytes) != module.digest {
        return Err(format!(
            "required module {name}: staged bytes differ from the locked module"
        ));
    }
    Ok(bytes)
}

fn validate(dir: &Path, required: &BTreeMap<String, Expected>) -> LabResult<Value> {
    let mut modules = Vec::new();
    for (name, module) in required {
        read_module(dir, name, module)?;
        modules.push(json!({"name":name,"sha256":module.digest,"required_symbols":module.symbols}));
    }
    Ok(json!({"test":"mxfp8_fold_decode::decode_fold_gpu","modules":modules}))
}

pub(crate) fn check() -> LabResult<Value> {
    validate(&directory()?, &expected()?)
}

/// Register the timing module by name; model functions and authenticated
/// dependency probes are loaded separately.
pub(crate) fn timing_cubins() -> LabResult<Cubins> {
    static BYTES: OnceLock<LabResult<Vec<u8>>> = OnceLock::new();
    static TABLE: OnceLock<[(&str, &[u8]); 1]> = OnceLock::new();
    let bytes = BYTES
        .get_or_init(|| {
            let dir = directory()?;
            let required = expected()?;
            validate(&dir, &required)?;
            read_module(&dir, TIMING, &required[TIMING])
        })
        .as_ref()
        .map_err(Clone::clone)?;
    let table = TABLE.get_or_init(|| [(TIMING, bytes)]);
    Ok(Cubins::new(table, &[(TIMING, TIMING_ENTRIES)]))
}

#[test]
fn dry_run_refuses_missing_or_wrong_timing_probe() {
    let mut required = expected().unwrap();
    assert!(required.values().filter(|m| m.embedded.is_none()).count() >= 4);
    required.retain(|name, module| name == TIMING || module.embedded.is_some());
    let dir = std::env::temp_dir().join(format!("fold-module-check-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    for (name, module) in &required {
        if let Some(bytes) = module.embedded {
            std::fs::write(dir.join(format!("{name}.cubin")), bytes).unwrap();
        }
    }
    let missing = validate(&dir, &required).unwrap_err();
    assert!(missing.contains("required module probe_sm120"), "{missing}");
    std::fs::write(dir.join("probe_sm120.cubin"), b"wrong cubin").unwrap();
    let wrong = validate(&dir, &required).unwrap_err();
    assert!(
        wrong.contains("probe_sm120: staged bytes differ"),
        "{wrong}"
    );
    std::fs::remove_dir_all(dir).unwrap();
    assert!(Cubins::NONE.bytes(TIMING).is_err());
}

pub(crate) struct ReferenceModules;
impl kernel_harness::kit::Modules for ReferenceModules {
    fn load<'s>(
        &'s self,
        session: &'s kernel_harness::Session,
    ) -> LabResult<kernel_harness::kit::modules::Functions<'s>> {
        let loaded = std::cell::RefCell::new(BTreeMap::new());
        Ok(Box::new(move |symbol| {
            let (name, bytes) = crate::dense_decode_reference::load_symbol(symbol)?;
            let mut modules = loaded.borrow_mut();
            if !modules.contains_key(name) {
                modules.insert(name, session.load_cubin(bytes)?);
            }
            modules[name].function(symbol).map_err(|e| e.to_string())
        }))
    }
    fn probes(
        &self,
        session: &kernel_harness::Session,
    ) -> LabResult<Option<kernel_harness::conformance::Probes>> {
        kernel_harness::kit::Modules::probes(&super::super::mxfp8_fold::ModelModules, session)
    }
}
