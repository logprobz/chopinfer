//! Emits the Qwen3.8 Flash package a checkpoint serves with at 2, 4 and 8
//! ranks (each its precision serves), from the checkpoint and the native
//! module alone.
//! The package holds no weights: its text image (with the MTP views of
//! `--drafter mtp`) and, with `--encoders on`, the vision encoders are
//! planned components the weight plan loads from the checkpoint.
use model_compiler_sdk::native_contract::NativeModule;
use model_compiler_sdk::package_cli;
use model_qwen38::{
    Result, encoders::ImagePositions, package::Drafter, preparation::emit_direct_package,
};

fn run() -> Result<()> {
    let package_cli::Options {
        values: mut options,
    } = package_cli::parse(
        std::env::args().skip(1),
        &[
            (
                "--fp8-kv",
                "--fp8-kv is gone: every package declares BF16 and FP8 KV; choose one at serve time with --kv-cache-dtype",
            ),
            (
                "--rank-dir",
                "--rank-dir is gone: the package is computed from the checkpoint alone",
            ),
            (
                "--images",
                "--images is gone: every package plans its images from the checkpoint",
            ),
            (
                "--ranks",
                "--ranks is gone: one package serves every topology the checkpoint's precision serves",
            ),
            (
                "--ngram-placement",
                "--ngram-placement is gone: the weight plan places the n-gram table by the device capacity rule (host memory at TP2, the GPU otherwise)",
            ),
        ],
    )?;
    let mut path = |name: &str, env| package_cli::path(&mut options, name, env);
    let checkpoint = path("checkpoint", Some("CHOPINFER_QWEN38_CHECKPOINT"))?;
    let module = path("module", None)?;
    let output = path("output", None)?;
    // `mtp` plans the drafter's views into the text image.
    let drafter: Drafter = options
        .remove("drafter")
        .as_deref()
        .unwrap_or("none")
        .parse()?;
    // `on` plans the vision encoders, for image inputs.
    let encoders = package_cli::switch(&mut options, "encoders")?;
    // HF's grid positions for image rows by default; `sequential` gives
    // every axis the token index, as the pinned vLLM does.
    let image_positions = options.remove("image-positions");
    let positions = match image_positions.as_deref() {
        None | Some("grid-3axis") => ImagePositions::Grid3Axis,
        Some("sequential") => ImagePositions::Sequential,
        Some(other) => return Err(format!("unknown --image-positions {other}").into()),
    };
    if !encoders && image_positions.is_some() {
        return Err("--image-positions needs --encoders on".into());
    }
    let activation_policy = model_compiler_sdk::quantization::requested_policy(
        options.remove("activation-policy").as_deref(),
    )?;
    // Precision components recorded earlier (a package's, one per
    // topology, or `qwen38-load-direct precision`'s), checked against the
    // checkpoint; the package serves the topologies they record.
    let supplied_precision = options
        .remove("precision-component")
        .iter()
        .flat_map(|paths| paths.split(','))
        .map(|path| {
            model_compiler_sdk::quantization::identity::read_component(std::path::Path::new(path))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    package_cli::refuse_unknown(&options)?;
    let module = NativeModule::inspect(&module)?;
    let identity = model_compiler_sdk::preparation::direct::checkpoint_identity(&checkpoint)?;
    let package = emit_direct_package(
        &checkpoint,
        &module,
        activation_policy.unwrap_or_default(),
        &supplied_precision,
        &identity,
        drafter,
        encoders.then_some(positions),
        None,
        &output,
    )?;
    for (shape, _) in &package.omitted {
        println!("left out {}", shape.topology().id);
    }
    println!(
        "emitted {} at {}",
        package.emitted.manifest().id(),
        package.emitted.root().display()
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    package_cli::main(
        "qwen38-package",
        "qwen38-package [--checkpoint DIR] --module FILE --output DIR [--activation-policy auto|checkpoint|bf16] [--precision-component FILE[,FILE...]] [--drafter none|mtp] [--encoders on|off] [--image-positions grid-3axis|sequential]",
        run,
    )
}
