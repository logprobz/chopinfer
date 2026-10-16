//! The glm5next vision tower whole, on one GPU, from the pinned
//! checkpoint: the encoder image planned by the bundle's weight plan and
//! loaded by the host's direct loader (`runtime_host::direct`), the
//! bundle's serving encoder (`bundle_glm5next::encoders`) over the host's
//! CUDA services, every image of the probe workload encoded in its own
//! step. Each step's output rows (`enc.vision.out`, read from the media
//! arena) and every block's output rows (`enc.vision.B<nn>.out`) are
//! written as a tap capture, which `tools/qual/encoder_parity.py lab`
//! compares with the GPU encoder reference under the encoder parity gate's
//! tapdiff rules.
//!
//! Opt-in: `CHOPINFER_GLM5NEXT_ENCODER_TOWER` (the capture directory,
//! which must not exist) opts in, and then `CHOPINFER_KERNEL_GPU` (one
//! ordinal), `CHOPINFER_GLM5NEXT_CHECKPOINT` and
//! `CHOPINFER_GLM5NEXT_ENCODER_REFERENCE` (a directory holding the
//! reference runs `vision-exact` and `vision-policy`) are required before
//! any device work; the workload is `CHOPINFER_GLM5NEXT_ENCODER_WORKLOAD` or
//! the R1 probe workload under `$CHOPINFER_DATA`. The test runs the
//! comparison under `CHOPINFER_REFRUN_PYTHON`, writing tapdiff's receipt
//! into the capture directory, and fails unless it matches. Relative paths
//! are resolved against the test's working directory. Run in release:
//!
//! ```sh
//! cargo test -p kernel-lab --release --test glm5next_encoder_tower -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use bundle_glm5next::{
    encoders::{self, views},
    sm120::own_ops::Sm120Own,
};
use bundle_runtime::acquire::{self, Log};
use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::pipeline::Collectives;
use model_glm5next_contract::{admission, catalog, contract::Shape, encoder_plan};
use runtime_bundle_abi::HostServicesTableV1;
use runtime_bundle_sdk::{
    BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaSectionV3, BatchRequestV1,
    BatchWireVersionV1,
};
use runtime_host::cuda::CudaHostDeviceV1;
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};
use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
use serde_json::{Value, json};

const BLOCKS: usize = 24;
const WIDTH: usize = 1024;
const HIDDEN: usize = 4096;
const OUT: &str = "enc.vision.out";

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// One workload image: its sequence name, token count, F32 patch rows and
/// `[t, h, w]` grid.
struct Image {
    name: String,
    tokens: usize,
    payload: Vec<u8>,
    grid: [u32; 3],
}

fn images(workload: &Path) -> Vec<Image> {
    let root = workload.parent().unwrap();
    let value: Value = serde_json::from_slice(&std::fs::read(workload).unwrap()).unwrap();
    value["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|sequence| {
            let name = sequence["name"].as_str().unwrap().to_owned();
            let tokens = sequence["token_ids"].as_array().unwrap().len();
            sequence["media"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|media| media["kind"] == "image")
                .map(move |media| {
                    let read = |key: &str| {
                        std::fs::read(root.join(media[key]["path"].as_str().unwrap())).unwrap()
                    };
                    let grid = read("grid");
                    let axis = |at: usize| {
                        i64::from_le_bytes(grid[at * 8..at * 8 + 8].try_into().unwrap()) as u32
                    };
                    Image {
                        name: name.clone(),
                        tokens,
                        payload: read("tensor"),
                        grid: [axis(0), axis(1), axis(2)],
                    }
                })
        })
        .collect()
}

/// The encoder image of one rank, planned from the checkpoint and checked
/// against the admitted checkpoint's catalog, prepared for the host.
fn planned_image(root: &Path) -> PlannedImageV1 {
    let checkpoint = Arc::new(DirectCheckpointV1::open(root).unwrap());
    let config: Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    let shapes: BTreeMap<String, Vec<u64>> =
        catalog::catalog(admission::admit(&config).unwrap().geometry)
            .into_iter()
            .map(|spec| (spec.name, spec.shape))
            .collect();
    // Any module digest: the bundle's transforms run in this process.
    let implementation = implementation_v1([5; 32]);
    let role = std::str::from_utf8(views::IMAGE_ROLE).unwrap();
    let request = checkpoint.request(role, 1, 0, implementation, vec![], vec![]);
    let declaration = &model_glm5next_contract::encoder::declarations().unwrap()[0];
    let plan =
        encoder_plan::plan_for(&request, declaration, &|name| shapes.get(name).cloned()).unwrap();
    PlannedImageV1::new(
        checkpoint,
        vec![plan],
        Some(bundle_glm5next::chop_bundle_weight_transform_v1),
        implementation,
        DirectLoadOptionsV1::for_ranks(1),
    )
    .unwrap()
}

/// A prefill step of one token that encodes `image` into the first arena
/// pages.
fn step(image: &Image) -> (BatchRequestV1, PhaseInputs) {
    let [t, h, w] = image.grid;
    let rows = (t * h * w) as u64;
    let output_rows = (rows / 4) as u32;
    assert_eq!(
        image.payload.len() as u64,
        rows * 1176 * 4,
        "{}",
        image.name
    );
    let media = BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder: 0,
            geometry: [t, h, w, 0],
            output_rows,
            pages: (0..u64::from(output_rows).div_ceil(encoders::PAGE_ROWS)).collect(),
            payload_offset: 0,
            payload_bytes: image.payload.len() as u64,
        }],
        injects: vec![],
        payload: image.payload.clone(),
    };
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(1, false, 1, true, vec![0])],
        vec![BatchAppendV1::new(0, 1, 1, vec![0], None)],
    )
    .with_media(media);
    let prefill = PhaseInputs {
        item_order: vec![0],
        request_ids: vec![1],
        positions: vec![0],
        q_start: vec![0],
        q_len: vec![1],
        token_ids: vec![0],
        ..PhaseInputs::default()
    };
    (request, prefill)
}

/// Writes one step of the capture: the image's output rows (BF16) and
/// every block's rows (F32), each on the encoder row axis of its sequence.
fn write_step(dir: &Path, image: &Image, taps: &[(String, &'static str, usize, Vec<u8>)]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut index = String::new();
    let mut segments = serde_json::Map::new();
    let mut expected = Vec::new();
    for (name, dtype, cols, bytes) in taps {
        let element = if *dtype == "bf16" { 2 } else { 4 };
        let rows = bytes.len() / (cols * element);
        assert_eq!(rows * cols * element, bytes.len(), "{name}");
        std::fs::write(dir.join(format!("{name}.r0.bin")), bytes).unwrap();
        index += &json!({"name": name, "rank": 0, "first_row": 0, "rows": rows, "cols": cols,
                         "dtype": dtype, "axis": "encoder"})
        .to_string();
        index.push('\n');
        segments.insert(
            name.clone(),
            json!([{"index": 0, "item": 0, "kind": "image", "request": image.name,
                     "rows": rows, "start": 0}]),
        );
        expected.push(json!({"name": name, "rank": 0, "sharding": "replicated"}));
    }
    std::fs::write(dir.join("index.jsonl"), index).unwrap();
    let items = json!({
        "dump_rows": [0, 0], "encoder": segments,
        "items": [{"positions": [], "prefix": 0, "request": image.name, "rows": 0,
                   "seq_len": image.tokens, "start": 0, "tokens": []}],
        "positions": [], "program": "encoder-tower", "row_map": [], "rows": 0, "token_ids": [],
    });
    std::fs::write(dir.join("items.json"), items.to_string()).unwrap();
    let capture = json!({
        "schema": "chopinfer.taps.v1", "producer": "kernel-lab", "program": "encoder-tower",
        "execution": "eager-synchronized", "ranks": 1, "request": image.name, "rows": 0,
        "seq_len": image.tokens, "expected": expected,
    });
    std::fs::write(dir.join("capture.json"), capture.to_string()).unwrap();
    let completion = json!({"completion": "full", "expected": taps.len(),
                            "observed": taps.len(), "missing": [], "reason": null});
    std::fs::write(dir.join("completion.json"), completion.to_string()).unwrap();
}

/// What an opted-in run needs, checked before any device work.
#[derive(Debug, PartialEq)]
struct Run {
    ordinal: i32,
    checkpoint: PathBuf,
    out: PathBuf,
    workload: PathBuf,
    references: PathBuf,
}

/// The run `lookup` (the environment) asks for: `None` when it does not
/// opt in (`CHOPINFER_GLM5NEXT_ENCODER_TOWER` unset), and an error when it
/// opts in without everything the comparison needs, the reference runs
/// included, so an opted-in run never passes without comparing. Relative
/// paths are resolved against `cwd`, the test's working directory, since
/// the comparison runs from the repository root.
fn preflight(lookup: impl Fn(&str) -> Option<String>, cwd: &Path) -> Result<Option<Run>, String> {
    let Some(out) = lookup("CHOPINFER_GLM5NEXT_ENCODER_TOWER") else {
        return Ok(None);
    };
    let path = |value: String| cwd.join(value);
    let required = |name: &str| lookup(name).ok_or_else(|| format!("the tower check needs {name}"));
    let ordinal = required("CHOPINFER_KERNEL_GPU")?
        .parse()
        .map_err(|_| "CHOPINFER_KERNEL_GPU is not one GPU ordinal".to_owned())?;
    let workload = match lookup("CHOPINFER_GLM5NEXT_ENCODER_WORKLOAD") {
        Some(workload) => path(workload),
        None => path(required("CHOPINFER_DATA")?).join("glm5next/R1/step30/w/probes-greedy.json"),
    };
    Ok(Some(Run {
        ordinal,
        checkpoint: path(required("CHOPINFER_GLM5NEXT_CHECKPOINT")?),
        out: path(out),
        workload,
        references: path(required("CHOPINFER_GLM5NEXT_ENCODER_REFERENCE")?),
    }))
}

#[test]
fn an_opted_in_run_needs_the_reference_runs() {
    let set = |missing: &'static str| {
        move |name: &str| {
            (name != missing).then(|| match name {
                "CHOPINFER_KERNEL_GPU" => "0".to_owned(),
                other => format!("/abs/{other}"),
            })
        }
    };
    let cwd = Path::new("/work");
    assert_eq!(preflight(|_| None, cwd), Ok(None));
    let run = preflight(set(""), cwd).unwrap().unwrap();
    assert_eq!(
        run.references,
        Path::new("/abs/CHOPINFER_GLM5NEXT_ENCODER_REFERENCE")
    );
    for missing in [
        "CHOPINFER_GLM5NEXT_ENCODER_REFERENCE",
        "CHOPINFER_GLM5NEXT_CHECKPOINT",
        "CHOPINFER_KERNEL_GPU",
    ] {
        let error = preflight(set(missing), cwd).unwrap_err();
        assert!(error.contains(missing), "{error}");
    }
}

/// Relative paths name the same files in the comparison, which runs from
/// the repository root, as in the test's own working directory.
#[test]
fn relative_paths_resolve_against_the_working_directory() {
    let lookup = |name: &str| {
        Some(match name {
            "CHOPINFER_KERNEL_GPU" => "0".to_owned(),
            "CHOPINFER_GLM5NEXT_ENCODER_REFERENCE" => "/abs/references".to_owned(),
            other => format!("relative/{other}"),
        })
    };
    let run = preflight(lookup, Path::new("/work/lab")).unwrap().unwrap();
    let under = |name: &str| Path::new("/work/lab/relative").join(name);
    assert_eq!(run.out, under("CHOPINFER_GLM5NEXT_ENCODER_TOWER"));
    assert_eq!(run.checkpoint, under("CHOPINFER_GLM5NEXT_CHECKPOINT"));
    assert_eq!(run.workload, under("CHOPINFER_GLM5NEXT_ENCODER_WORKLOAD"));
    assert_eq!(run.references, Path::new("/abs/references"));
}

#[test]
fn the_vision_tower_matches_the_encoder_reference() {
    let cwd = std::env::current_dir().unwrap();
    let Some(run) = preflight(env, &cwd).unwrap() else {
        eprintln!("skipped: CHOPINFER_GLM5NEXT_ENCODER_TOWER is not set");
        return;
    };
    let Run {
        ordinal,
        checkpoint: root,
        out,
        workload,
        references,
    } = run;
    assert!(!out.exists(), "{} exists", out.display());
    let images = images(&workload);
    assert_eq!(images.len(), 6);

    let started = Instant::now();
    let image = planned_image(&root);
    let planned = started.elapsed().as_secs_f64();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![ordinal]).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(
        CudaHostDeviceV1::new().unwrap(),
        config,
    ));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let log = Log {
        services,
        bundle: "glm5next encoder tower",
    };
    let mut held = Vec::new();
    let ranks = acquire::open_ranks(log, &[ordinal], &mut held).unwrap();
    let target = Sm120Own::new(Shape::TP4);
    let mut encoder = encoders::acquire(&target, services, &ranks, &mut held).unwrap();
    encoder.block_taps = true;
    let symbols = encoders::load(&target, log, &ranks, &mut held).unwrap();
    let stream = ranks[0].2;
    let arena = encoder.ranks[0].region("arena").unwrap();
    let loaded = started.elapsed().as_secs_f64();

    let mut seconds = Vec::new();
    for (index, image) in images.iter().enumerate() {
        let begun = Instant::now();
        let (request, prefill) = step(image);
        encoder
            .run(
                services,
                &[stream],
                &symbols,
                &request.encode().unwrap(),
                &prefill,
                &Collectives::default(),
            )
            .unwrap_or_else(|status| panic!("{}: {status:?}", image.name));
        let tokens = request.media().encodes[0].output_rows as usize;
        let mut rows = vec![0; tokens * HIDDEN * 2];
        let span = runtime_bundle_abi::DeviceBufferSpanV1::new(
            arena.buffer,
            arena.offset,
            rows.len() as u64,
        );
        services
            .copy_device_to_host(stream, &mut rows, span)
            .unwrap();
        services.synchronize_stream(stream).unwrap();
        let mut taps = vec![(OUT.to_owned(), "bf16", HIDDEN, rows)];
        // Every block's output; the reference taps no block inputs.
        let blocks: Vec<_> = encoder
            .take_block_rows()
            .into_iter()
            .filter(|block| block.name.ends_with(".out"))
            .collect();
        let names: Vec<String> = (0..BLOCKS)
            .map(|b| format!("enc.vision.B{b:02}.out"))
            .collect();
        assert_eq!(
            blocks.iter().map(|block| &block.name).collect::<Vec<_>>(),
            names.iter().collect::<Vec<_>>(),
            "{}",
            image.name
        );
        for block in blocks {
            taps.push((block.name, "f32", WIDTH, block.bytes));
        }
        write_step(&out.join(format!("step{index:04}")), image, &taps);
        seconds.push((image.name.clone(), begun.elapsed().as_secs_f64()));
    }
    let timing = json!({"plan_seconds": planned, "load_seconds": loaded - planned,
                        "images": seconds, "total_seconds": started.elapsed().as_secs_f64()});
    std::fs::write(out.join("timing.json"), timing.to_string()).unwrap();
    eprintln!("glm5next encoder tower: {timing}");

    let python = env("CHOPINFER_REFRUN_PYTHON").unwrap_or_else(|| "python3".into());
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let status = std::process::Command::new(python)
        .current_dir(&repository)
        .args([
            "-B",
            "tools/qual/encoder_parity.py",
            "lab",
            "--model",
            "glm5next",
        ])
        .arg("--engine")
        .arg(&out)
        .arg("--reference")
        .arg(references.join("vision-exact"))
        .arg("--reference")
        .arg(references.join("vision-policy"))
        .arg("--workload")
        .arg(&workload)
        .arg("--json")
        .arg(out.join("tapdiff.json"))
        .status()
        .unwrap();
    assert!(
        status.success(),
        "the tower leaves the encoder reference's spread"
    );
}
