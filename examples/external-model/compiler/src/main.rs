//! Compile a tiny checkpoint into one canonical package using a supplied module.
#![forbid(unsafe_code)]
mod frontend;
#[cfg(feature = "precision-exercise")]
mod precision;
use example_contract as model;
use model_compiler_sdk::{
    BundleDraftV1, ComponentContentV1, ComponentDraftV1, NativeModuleDraftV1, StagedContentV1,
    emit_bundle_directory_v1, source_file::SourceFile,
};
use runtime_bundle::{
    BundleBatchLimitsV1, BundleGraphBucketV1, BundleKvLayoutV1, BundleOpaqueId,
    BundleRankPlacementV1, BundleRelativePathV1, BundleResourceRequirementsV1, BundleStateLayoutV1,
    BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
};
use std::{error::Error, io::Read, path::Path};

fn component(
    id: &str,
    role: &str,
    format: &str,
    path: &str,
    bytes: Vec<u8>,
) -> Result<ComponentDraftV1, Box<dyn Error + Send + Sync>> {
    Ok(ComponentDraftV1::new(
        BundleOpaqueId::new(id)?,
        BundleOpaqueId::new(role)?,
        BundleOpaqueId::new(format)?,
        BundleRelativePathV1::new(path)?,
        ComponentContentV1::Whole(StagedContentV1::bytes(bytes)),
    ))
}
fn compile(
    checkpoint: &Path,
    module: &Path,
    output: &Path,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut source = SourceFile::open(checkpoint)?;
    if source.metadata()?.len() > 4096 {
        return Err("checkpoint is too large".into());
    }
    let mut snapshot = Vec::new();
    source.read_to_end(&mut snapshot)?;
    source.verify()?;
    let config: serde_json::Value = serde_json::from_slice(&snapshot)?;
    if config.as_object().is_none_or(|o| o.len() != 1) {
        return Err("checkpoint requires only seed".into());
    }
    let seed = config["seed"]
        .as_u64()
        .ok_or("checkpoint requires unsigned seed")?;
    let draft = draft(seed, StagedContentV1::file(module), vec![], &[])?;
    let emitted = emit_bundle_directory_v1(&draft, output)?;
    println!("{}", emitted.manifest().id());
    Ok(())
}
/// The package: the seed, frontend and tokenizer components, then `extra`,
/// requiring `capabilities` beside the example's own.
fn draft(
    seed: u64,
    module: StagedContentV1,
    extra: Vec<ComponentDraftV1>,
    capabilities: &[&str],
) -> Result<BundleDraftV1, Box<dyn Error + Send + Sync>> {
    let (tokenizer, tokenizer_config, frontend) = frontend::artifacts();
    let mut components = vec![
        component(
            "checkpoint",
            model::SEED_ROLE,
            "example.seed-u64.v1",
            "checkpoint.bin",
            seed.to_le_bytes().to_vec(),
        )?,
        component(
            "frontend",
            "chop.frontend.openai.v1",
            "chop-frontend-openai-v2",
            "frontend.json",
            frontend,
        )?,
        component(
            "tokenizer-config",
            "chop.tokenizer.configuration.v1",
            "tokenizer-config-json",
            "tokenizer_config.json",
            tokenizer_config,
        )?,
        component(
            "tokenizer-definition",
            "chop.tokenizer.definition.v1",
            "tokenizer-json",
            "tokenizer.json",
            tokenizer,
        )?,
    ];
    components.extend(extra);
    let dependencies = components.iter().map(|c| c.id().clone()).collect();
    let draft = BundleDraftV1::new(
        BundleOpaqueId::new(model::BUNDLE)?,
        BundleTargetV1::new(
            vec![BundleOpaqueId::new("sm_120f")?],
            vec![BundleTopologyV1::new(
                BundleOpaqueId::new(model::TOPOLOGY)?,
                1,
            )?],
        )?,
        BundleResourceRequirementsV1::new(
            1 << 20,
            [
                "chop.frontend.v1",
                "chop.prompt-rows.v2",
                "chop.request-retirement.v1",
            ]
            .iter()
            .chain(capabilities)
            .map(|capability| BundleOpaqueId::new(*capability))
            .collect::<Result<_, _>>()?,
        )?,
        BundleKvLayoutV1::new(
            BundleOpaqueId::new(model::KV)?,
            32,
            32,
            8,
            4,
            1,
            model::CONTEXT,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Transaction,
        )?,
        BundleStateLayoutV1::new(
            BundleOpaqueId::new(model::STATE)?,
            64,
            64,
            BundleStateLifetimeV1::Process,
        )?,
    )
    .with_components(components)
    .with_graph_buckets(vec![BundleGraphBucketV1::new(
        BundleOpaqueId::new("step")?,
        dependencies,
        vec![],
    )?])
    .with_batch_limits(BundleBatchLimitsV1::new(model::ITEMS, 512)?)
    .with_native_module(NativeModuleDraftV1::new(
        BundleRelativePathV1::new("native/module.bin")?,
        module,
    ));
    Ok(draft)
}
fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    #[cfg(feature = "precision-exercise")]
    if matches!(
        args.first().map(String::as_str),
        Some("prepare" | "describe" | "cache" | "--protocol")
    ) {
        return model_compiler_sdk::preparation::helper::run(&precision::Family, args);
    }
    if args.len() != 6
        || args[0] != "--checkpoint"
        || args[2] != "--module"
        || args[4] != "--output"
    {
        return Err("usage: example-compile --checkpoint FILE --module FILE --output DIR".into());
    }
    compile(
        Path::new(&args[1]),
        Path::new(&args[3]),
        Path::new(&args[5]),
    )
}
