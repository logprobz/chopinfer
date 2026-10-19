//! Frozen old/new attention comparisons and per-stage timings. No GPU test builds code.

#[path = "../../../bundle/src/sm120/latent_scheduled.rs"]
#[allow(dead_code)]
mod candidate;
#[path = "attention_pair/decode.rs"]
mod decode;
#[path = "../../../bundle/src/sm120/latent_decode.rs"]
#[allow(dead_code)]
mod decode_candidate;
#[path = "attention_pair/fixture.rs"]
mod fixture;
#[path = "../../../bundle/src/sm120/latent_glue.rs"]
#[allow(dead_code)]
mod glue;
#[path = "../../../bundle/src/sm120/latent.rs"]
#[allow(dead_code)]
mod latent;
#[path = "attention_pair/launch.rs"]
mod launch;
#[path = "attention_pair/stages.rs"]
mod stages;
#[path = "attention_pair/probe.rs"]
#[allow(dead_code)]
mod state_probe;

use fixture::{Case, DeviceFixture, Fixture};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernel_harness::{Cubins, LabResult, Session};
use kernels_sm120::latent::sources::Entry;
use model_deepseek41_contract::geometry::{GEOMETRIES, Geometry};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn geometry() -> &'static Geometry {
    GEOMETRIES
        .iter()
        .find(|g| g.kernel_tag == "deepseek41")
        .expect("served Flash geometry")
}

fn reference(case: Case) -> Entry {
    latent::entries(geometry(), 4).unwrap().into_iter().find(|e| {
        matches!(e.query, LatentAttentionQuery::Attention { phase, page_size, second_fp4_keys, .. }
            if phase == (if case.verify { AttentionPhase::Verify } else { AttentionPhase::Decode })
            && second_fp4_keys == (case.ratio != 0)
            && page_size == (if case.ratio == 2 { 32 } else { 64 }))
            && !e.name.ends_with("_draft_block")
    }).unwrap()
}

fn matrix() -> Vec<Case> {
    [1024, 8192, 32768, 131072]
        .into_iter()
        .flat_map(|context| {
            [1, 8, 32].into_iter().flat_map(move |sequences| {
                [false, true].into_iter().flat_map(move |verify| {
                    [0, 1, 2].into_iter().map(move |ratio| Case {
                        verify_width: 6,
                        context,
                        sequences,
                        verify,
                        ratio,
                        holes: false,
                        inactive: 0,
                    })
                })
            })
        })
        .collect()
}

fn artifact_root() -> LabResult<PathBuf> {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_ATTENTION_CUBINS")
            .ok_or("set CHOPINFER_ATTENTION_CUBINS to the frozen module directory")?,
    );
    if !root.is_absolute() {
        return Err("frozen module directory must be absolute".into());
    }
    Ok(root)
}

fn manifest() -> LabResult<Vec<Value>> {
    let bytes = std::fs::read(artifact_root()?.join("attention-modules.json"))
        .map_err(|e| e.to_string())?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if value["schema"] != 1 {
        return Err("unknown attention snapshot schema".into());
    }
    value["modules"]
        .as_array()
        .cloned()
        .ok_or("snapshot modules are absent".into())
}

fn image(record: &Value) -> LabResult<Vec<u8>> {
    let name = record["name"].as_str().ok_or("module name is absent")?;
    if name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
        return Err("module name must be a plain identifier".into());
    }
    let path = artifact_root()?.join(format!("{name}.cubin"));
    if !std::fs::symlink_metadata(&path)
        .map_err(|e| e.to_string())?
        .is_file()
    {
        return Err("snapshot module is not a regular file".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if record["sha256"].as_str() != Some(kernel_build::digest(&bytes).as_str()) {
        return Err(format!("snapshot digest mismatch: {name}"));
    }
    Ok(bytes)
}

fn cubins() -> LabResult<Cubins> {
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    for record in manifest()? {
        let name: &'static str = Box::leak(
            record["name"]
                .as_str()
                .ok_or("module name")?
                .to_owned()
                .into_boxed_str(),
        );
        let data: &'static [u8] = Box::leak(image(&record)?.into_boxed_slice());
        let symbols = record["symbols"]
            .as_array()
            .ok_or("module symbols")?
            .iter()
            .map(|s| {
                s.as_str()
                    .map(|s| &*Box::leak(s.to_owned().into_boxed_str()))
                    .ok_or("invalid symbol".to_owned())
            })
            .collect::<LabResult<Vec<_>>>()?;
        bytes.push((name, data));
        entries.push((name, &*Box::leak(symbols.into_boxed_slice())));
    }
    Ok(Cubins::new(
        Box::leak(bytes.into_boxed_slice()),
        Box::leak(entries.into_boxed_slice()),
    ))
}

struct Modules {
    functions: BTreeMap<String, runtime_cuda::Function>,
    configured: BTreeMap<String, u32>,
}
impl Modules {
    fn load(session: &Session) -> LabResult<Self> {
        let mut functions = BTreeMap::new();
        for record in manifest()? {
            let module = session.load_cubin(&image(&record)?)?;
            for symbol in record["symbols"].as_array().ok_or("symbols")? {
                let symbol = symbol.as_str().ok_or("symbol")?.to_owned();
                let function = module
                    .function(&symbol)
                    .map_err(|e| format!("{symbol}: {e}"))?;
                if functions.insert(symbol, function).is_some() {
                    return Err("duplicate frozen symbol".into());
                }
            }
        }
        let configured = launch::configure(&launch::limits()?, |symbol, shared| {
            let function = functions
                .get(symbol)
                .ok_or_else(|| format!("missing frozen symbol {symbol}"))?;
            function
                .set_max_dynamic_shared_memory(shared)
                .map_err(|e| format!("{symbol}: opt in to {shared} shared bytes: {e}"))?;
            eprintln!(
                "ATTENTION_CONFIGURE {}",
                json!({"symbol": symbol, "dynamic_shared_bytes": shared})
            );
            Ok(())
        })?;
        Ok(Self {
            functions,
            configured,
        })
    }
    fn function(&self, symbol: &str) -> LabResult<&runtime_cuda::Function> {
        self.functions
            .get(symbol)
            .ok_or_else(|| format!("missing frozen symbol {symbol}"))
    }
}

#[test]
fn cpu_matrix_covers_decode_and_six_row_verify() {
    use kernel_core::backend::{Backend, Counts, Need, View};
    use kernel_core::latent_glue::LatentQueryRopeBuffers;
    let backend = candidate::register(kernels_sm120::backend::Sm120::new(), geometry(), 4).unwrap();
    let query = model_deepseek41_contract::latent_glue::query_rope(geometry(), 4).unwrap();
    let cases = matrix();
    assert_eq!(cases.len(), 72);
    let mut shapes = BTreeSet::new();
    for case in cases {
        let entry = reference(case);
        let candidate = candidate::entries(geometry(), 4)
            .unwrap()
            .into_iter()
            .find(|e| e.query == entry.query)
            .unwrap();
        assert_eq!(entry.params().unwrap(), candidate.params().unwrap());
        assert_eq!(
            entry.merge_params().unwrap(),
            candidate.merge_params().unwrap()
        );
        assert_eq!(entry.splits, candidate.splits);
        assert_eq!(
            entry.workspace_bytes(case.rows() as u32),
            candidate.workspace_bytes(case.rows() as u32)
        );
        assert!(candidate.plan(case.rows() as u32).is_ok());
        let rows = case.rows() as u32;
        let prepared = backend.latent_query_rope(&query, Need::rows(rows)).unwrap();
        let elements = u64::from(rows) * 16 * 512;
        let buffers = LatentQueryRopeBuffers {
            query: View {
                handle: 1,
                bytes: elements * 4,
            },
            native_q: View {
                handle: 2,
                bytes: elements * 2,
            },
            cos_sin: View {
                handle: 3,
                bytes: u64::from(rows) * 64 * 4,
            },
            live_rows: View {
                handle: 4,
                bytes: 4,
            },
            status: View {
                handle: 5,
                bytes: u64::from(rows) * 4,
            },
            status_row_offset: 0,
        };
        let calls = prepared
            .launches(&buffers, View::default(), Counts::rows(rows))
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].symbol, candidate::query().symbol);
        assert_eq!(calls[0].plan.block, [1024, 1, 1]);
        shapes.insert((case.context, case.sequences, case.rows(), case.ratio));
    }
    assert_eq!(shapes.len(), 72);
    let (old, new) = state_probe::pair().unwrap();
    assert_eq!(old.splits, 1);
    assert_eq!(new.splits, 1);
    let LatentAttentionQuery::Attention { max_keys, .. } = old.query else {
        unreachable!()
    };
    assert!(max_keys.div_ceil(64) >= 3);
    assert_eq!(old.params().unwrap(), new.params().unwrap());
}

#[test]
fn cpu_visibility_and_bandwidth_floor_use_selected_cells() {
    for context in [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049] {
        for ratio in [0, 1, 2] {
            let case = Case {
                verify_width: 6,
                context,
                sequences: 1,
                verify: false,
                ratio,
                holes: false,
                inactive: 0,
            };
            let entry = reference(case);
            let fixture = Fixture::new(case, &entry);
            fixture.launches(&entry).unwrap();
            let expected_pages = if ratio == 0 {
                0
            } else {
                (context / ratio).div_ceil(64 / ratio)
            };
            assert_eq!(fixture.pages, expected_pages + 1);
            let compressed = if ratio == 0 {
                0
            } else {
                (context / ratio).min(fixture.budget)
            };
            let bytes = context.min(128) * 528 + compressed * 288;
            assert_eq!(fixture.row_kv_bytes, bytes);
            assert_eq!(fixture.unique_kv_bytes, bytes);
            for key in fixture.data["compressed_keys"].chunks_exact(4) {
                let key = u32::from_ne_bytes(key.try_into().unwrap());
                assert!(key == u32::MAX || (key as usize) < context / ratio.max(1));
            }
            assert!(fixture.data["pool"].contains(&0x7f));
            if context == 129 && ratio == 2 {
                let boundary = Case {
                    sequences: 2,
                    verify: true,
                    holes: true,
                    ..case
                };
                let entry = reference(boundary);
                let fixture = Fixture::new(boundary, &entry);
                assert_eq!(fixture.code_stride % 8, 2);
                assert_eq!(fixture.page_stride % 4, 1);
                fixture.launches(&entry).unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires the frozen module snapshot"]
fn frozen_modules() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    let mut specs = latent::cubins(GEOMETRIES)
        .unwrap()
        .into_iter()
        .filter(|(family, _)| *family != "latent_sources_bf16")
        .chain(glue::cubins(GEOMETRIES).unwrap())
        .map(|(_, s)| s)
        .collect::<Vec<_>>();
    specs.extend(candidate::cubins(GEOMETRIES).unwrap());
    specs.push(candidate::query_cubin());
    specs.push(state_probe::cubin().unwrap());
    specs.push(decode_candidate::cubin(geometry()).unwrap());
    specs.push(kernels_sm120::probes::cubin_spec());
    let records = manifest().unwrap();
    assert_eq!(records.len(), specs.len() + 1);
    let names = records
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), records.len());
    let mut count = 0;
    for record in &records {
        image(record).unwrap();
        let name = record["name"].as_str().unwrap();
        let expected = record["symbols"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>();
        assert!(!expected.is_empty());
        let path = artifact_root().unwrap().join(format!("{name}.cubin"));
        if let Some(spec) = specs.iter().find(|s| s.name == name) {
            let reports = kernel_build::audit_cubin(&toolchain, spec, &path).unwrap();
            assert_eq!(
                reports
                    .iter()
                    .map(|e| e.symbol.clone())
                    .collect::<BTreeSet<_>>(),
                expected
            );
        } else {
            assert_eq!(name, "probe_sm120");
            let dump = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
                .arg("--dump-elf-symbols")
                .arg(path)
                .output()
                .unwrap();
            assert!(dump.status.success());
            let text = String::from_utf8(dump.stdout).unwrap();
            let actual = text
                .lines()
                .filter(|line| {
                    line.contains("STT_FUNC")
                        && line.contains("STB_GLOBAL")
                        && line.contains("STO_ENTRY")
                })
                .map(|line| line.split_whitespace().last().unwrap().to_owned())
                .collect::<BTreeSet<_>>();
            assert_eq!(actual, expected);
        }
        count += expected.len();
    }
    assert!(names.contains("probe_sm120"));
    println!(
        "ATTENTION_MODULES_OK modules={} symbols={count}",
        records.len()
    );
}

fn shard() -> usize {
    let shard = std::env::var("CHOPINFER_ATTENTION_SHARD")
        .expect("pair job shard")
        .parse::<usize>()
        .unwrap();
    assert!(shard < 2);
    shard
}

fn compare(
    session: &Session,
    modules: &Modules,
    case: Case,
    negative: bool,
    timing: bool,
) -> LabResult<()> {
    let old = reference(case);
    let new = candidate::entries(geometry(), 4)?
        .into_iter()
        .find(|e| e.query == old.query)
        .ok_or("candidate query")?;
    let host = Fixture::new(case, &old);
    let device = DeviceFixture::new(session, &host)?;
    let calls = [host.launches(&old)?, host.launches(&new)?];
    device.run(session, modules, &calls[0], &session.stream)?;
    let expected = [
        device.download(session, "workspace")?,
        device.download(session, "out")?,
    ];
    device.reset(session)?;
    device.run(session, modules, &calls[1], &session.stream)?;
    for (name, expected) in ["workspace", "out"].into_iter().zip(&expected) {
        let actual = device.download(session, name)?;
        if actual != *expected {
            let first = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap();
            return Err(format!("{case:?}: {name} differs at byte {first}"));
        }
    }
    let output = &expected[1];
    let live_bytes = case.live() * fixture::HEADS * fixture::WIDTH * 2;
    for value in output[..live_bytes].chunks_exact(2) {
        if u16::from_ne_bytes(value.try_into().unwrap()) & 0x7f80 == 0x7f80 {
            return Err(format!("{case:?}: nonfinite output from a stale slot"));
        }
    }
    if output[live_bytes..].iter().any(|&b| b != 0xa5) {
        return Err("inactive output changed".into());
    }
    if negative {
        let mut wrong = calls[1][1].clone();
        wrong.symbol = new
            .sink_per_split_entry_spec()
            .map_err(|e| e.to_string())?
            .symbol;
        device.run(session, modules, &[wrong], &session.stream)?;
        if device.download(session, "out")? == expected[1] {
            return Err("negative control was not rejected".into());
        }
        println!("ATTENTION_NEGATIVE_OK shard={}", shard());
    }
    if timing {
        for (stage, range) in [("sources", 0..1), ("output", 1..2), ("attention", 0..2)] {
            let paired = kernel_harness::time_ab(
                session,
                2,
                9,
                [kernel_harness::eviction::EvictionMode::Write; 2],
                |arm, stream| device.run(session, modules, &calls[arm][range.clone()], stream),
            )?;
            println!(
                "ATTENTION_TIMING {}",
                json!({
                    "stage": stage, "context": case.context, "sequences": case.sequences,
                    "rows": case.rows(), "verify": case.verify, "ratio": case.ratio,
                    "old_us": paired.arms[0].median_ms * 1000.0,
                    "new_us": paired.arms[1].median_ms * 1000.0, "paired_ratio": paired.paired_ratio,
                    "kv_unique_bytes": host.unique_kv_bytes,
                    "kv_payload_bytes_old": host.row_kv_bytes * calls[0][0].plan.grid[1] as usize,
                    "kv_payload_bytes_new": host.row_kv_bytes * calls[1][0].plan.grid[1] as usize,
                    "kv_floor_us_at_1_tb_s": host.unique_kv_bytes as f64 / 1_000_000.0,
                })
            );
        }
        stages::measure(session, modules, case)?;
    }
    println!(
        "ATTENTION_PAIR_OK {}",
        json!({"context": case.context, "sequences": case.sequences, "rows": case.rows(), "verify": case.verify, "ratio": case.ratio, "holes": case.holes, "inactive": case.inactive})
    );
    Ok(())
}

fn signed_zero(session: &Session, modules: &Modules) -> LabResult<()> {
    let case = Case {
        verify_width: 6,
        context: 128,
        sequences: 1,
        verify: false,
        ratio: 1,
        holes: false,
        inactive: 0,
    };
    let (old, new) = state_probe::pair()?;
    let mut host = Fixture::new(case, &old);
    host.data.get_mut("q").unwrap().fill(0);
    for head in 0..fixture::HEADS {
        let at = head * 512 * 2;
        host.data.get_mut("q").unwrap()[at..at + 2]
            .copy_from_slice(&kernel_core::codec::f32_to_bf16(1600.0).to_ne_bytes());
    }
    host.data.get_mut("window_codes").unwrap().fill(0);
    host.data.get_mut("window_scales").unwrap().fill(127);
    for slot in 0..64 {
        host.data.get_mut("window_codes").unwrap()[slot * 512 + 128] = 0xfe;
        host.data.get_mut("window_scales").unwrap()[slot * 16 + 4] = 59;
    }
    for slot in 64..128 {
        host.data.get_mut("window_codes").unwrap()[slot * 512] = 0x38;
    }
    host.data.get_mut("compressed_keys").unwrap().fill(0xff);
    let device = DeviceFixture::new(session, &host)?;
    device.run(session, modules, &host.launches(&old)?, &session.stream)?;
    let expected = device.download(session, "workspace")?;
    // The second tile underflows the negative numerator. The following
    // all-hole tiles must turn its -0 into +0 through the reference FMA.
    let at = (2 + 128) * 4;
    assert_eq!(&expected[at..at + 4], &0_f32.to_ne_bytes());
    device.reset(session)?;
    device.run(session, modules, &host.launches(&new)?, &session.stream)?;
    if device.download(session, "workspace")? != expected {
        return Err("signed-zero state changed across an empty tile".into());
    }
    println!("ATTENTION_SIGNED_ZERO_OK shard={}", shard());
    Ok(())
}

#[test]
#[ignore = "requires two allocated idle GPUs through the staged pair job"]
fn gpu_matrix() {
    let shard = shard();
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 8 << 30, |session| {
        let modules = Modules::load(session)?;
        let mut negative = false;
        let mut count = 0;
        for (index, case) in matrix().into_iter().enumerate() {
            if index % 2 != shard {
                continue;
            }
            let control = !negative && case.ratio != 0;
            compare(session, &modules, case, control, true)?;
            negative |= control;
            count += 1;
        }
        if count != 36 || !negative {
            return Err("incomplete attention shard".into());
        }
        println!("ATTENTION_MATRIX_OK shard={shard} cases={count}");
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}

#[test]
#[ignore = "requires an allocated idle GPU through the staged pair job"]
fn gpu_boundaries_and_poison() {
    let shard = shard();
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 2 << 30, |session| {
        let modules = Modules::load(session)?;
        let mut count = 0;
        for (index, context) in [
            1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 4095,
        ]
        .into_iter()
        .enumerate()
        {
            if index % 2 != shard {
                continue;
            }
            for ratio in [0, 1, 2] {
                for inactive in [0, 1, 12] {
                    compare(
                        session,
                        &modules,
                        Case {
                            verify_width: 6,
                            context,
                            sequences: 2,
                            verify: true,
                            ratio,
                            holes: true,
                            inactive,
                        },
                        false,
                        false,
                    )?;
                    count += 1;
                }
            }
        }
        assert_eq!(count, 63);
        signed_zero(session, &modules)?;
        println!("ATTENTION_EDGES_OK shard={shard} cases={count}");
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}
