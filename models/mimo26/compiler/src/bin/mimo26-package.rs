//! Emits into `--output` the MiMo V2.6 Flash package a pinned checkpoint
//! serves with at 2 and 4 ranks, from the checkpoint and the native module
//! alone; `chopinfer serve --bundle DIR --checkpoint DIR --profile NAME`
//! serves it. The package holds no weights: its text image, the drafter's of
//! `--drafter mtp|dflash` and, with `--encoders on`, the vision and audio
//! encoders' are planned components the weight plan loads from the
//! checkpoint.

use model_compiler_sdk::package_cli;
use std::env;
use std::process::ExitCode;

#[path = "mimo26_package/package.rs"]
mod package;

#[cfg(test)]
#[path = "../test_native.rs"]
mod test_native;

use model_compiler_sdk::StagedContentV1;
use model_mimo26::catalog::DrafterKind;
use package::{PackageAssets, emit_verified_direct_package, native_contract, planned_rank_bytes};

fn run() -> Result<(), String> {
    let package_cli::Options {
        values: mut options,
    } = package_cli::parse(
        env::args().skip(1),
        &[
            (
                "--fp8-global-kv",
                "--fp8-global-kv is gone: the geometry declares supported KV formats; choose one at serve time with --kv-cache-dtype",
            ),
            (
                "--images",
                "--images is gone: the package is computed from the checkpoint alone",
            ),
            (
                "--rank-dir",
                "--rank-dir is gone: the package is computed from the checkpoint alone",
            ),
            (
                "--drafter-rank-dir",
                "--drafter-rank-dir is gone: the package is computed from the checkpoint alone",
            ),
            (
                "--encoder-rank-dir",
                "--encoder-rank-dir is gone: the package is computed from the checkpoint alone",
            ),
            (
                "--ranks",
                "--ranks is gone: one package serves 2 and 4 ranks",
            ),
        ],
    )?;
    let checkpoint_identity = options.remove("checkpoint-identity");
    let identity = match checkpoint_identity.as_deref() {
        Some(name) => model_mimo26::checkpoint::identity(name)?,
        None => model_mimo26::checkpoint::default_identity(),
    };
    let checkpoint = package_cli::path(&mut options, "checkpoint", None)?;
    let module = package_cli::path(&mut options, "module", None)?;
    let output = package_cli::path(&mut options, "output", None)?;
    let drafter = match options.remove("drafter").as_deref().unwrap_or("none") {
        "none" => None,
        "mtp" => Some(DrafterKind::Mtp),
        "dflash" => Some(DrafterKind::Dflash),
        _ => return Err("--drafter must be none, mtp, or dflash".into()),
    };
    // `on` plans the vision and audio encoders, for media inputs.
    let encoders = package_cli::switch(&mut options, "encoders")?;
    package_cli::refuse_unknown(&options)?;
    let verifier = model_mimo26::checkpoint_provenance::Verifier::new(&checkpoint, identity)
        .map_err(|error| error.to_string())?;
    let [tokenizer_definition, tokenizer_configuration] =
        PackageAssets::pinned_tokenizer(&checkpoint, identity)
            .map_err(|error| error.to_string())?;
    let assets = PackageAssets {
        tokenizer_definition,
        tokenizer_configuration,
        native_module: StagedContentV1::file(module),
        drafter,
        precision: model_mimo26::catalog::precision_components(&checkpoint, drafter)
            .map_err(|error| error.to_string())?,
        encoders: if encoders {
            model_mimo26_contract::encoders::Encoders::ALL
        } else {
            model_mimo26_contract::encoders::Encoders::NONE
        },
    };
    let contract = native_contract(&assets.native_module).map_err(|error| error.to_string())?;
    let bytes = planned_rank_bytes(&checkpoint, &assets).map_err(|error| error.to_string())?;
    let package = emit_verified_direct_package(assets, &bytes, &output, &verifier)
        .map_err(|error| error.to_string())?;
    println!(
        "emitted {} at {} with kernel contract {}",
        package.manifest().id(),
        package.root().display(),
        contract.selection
    );
    Ok(())
}

fn main() -> ExitCode {
    package_cli::main(
        "mimo26-package",
        "mimo26-package --checkpoint DIR [--checkpoint-identity NAME] --module FILE --output DIR [--drafter none|mtp|dflash] [--encoders on|off]",
        run,
    )
}
