//! Exercise packages produced by the independent SDK consumer. Artifact and
//! policy checks run without a GPU; device evidence requires an idle assignment.
//! Each package plans its weights: the adapter loads the package's native
//! module and executes its weight plan over the recorded checkpoint with the
//! host's direct-load executor, as `serve --model` does.
#[path = "../../../bundle/precision_program.rs"]
mod precision_program;
#[path = "../../../bundle/src/sm120/precision_selection.rs"]
mod precision_selection;

use kernel_build::{BuildReport, BuiltCubin};
use kernel_core::{
    backend::{Counts, View},
    codec::{SmallFloat, bf16_to_f32, f32_to_bf16},
    contract::{ArgumentList, DType, LaunchPlan, WeightFormat},
    moe::RouteMetadata,
    oracle::{self, ExpertRowRange, GemmEpilogue, GemmShape},
    precision::{
        DecodeOrder, NibbleOrder, OperationUse,
        buffers::{ProjectionBuffers, ProjectionRouting, ProjectionScratch, QuantizedWeights},
        execution::ActivationPolicy,
        packed::{MatrixView, ScaleBytes},
    },
    quantize::{self, QuantFormat, Quantized, ScaleFormat, Scales},
    transform::{Layout, Packed, ScaleTileLayout, Transform},
};
use kernel_harness::{
    LabResult, Session,
    launch::{Bindings, launch},
};
use kernels_sm120::{backend::Sm120, precision::CompiledInventory};
use model_compiler_sdk::{native_contract::NativeModule, quantization::resolve_operation};
use example_contract::precision as model;
use example_contract::precision as precision_contract;
use precision_program::PreparedExercise;
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::json;
use sha2::Digest;
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

const GUARD: u16 = 0x42c8;
const X: u64 = 1;
const W: u64 = 2;
const WS: u64 = 3;
const WG: u64 = 4;
const AG: u64 = 5;
const Y: u64 = 6;
const LIVE: u64 = 7;
const CODES: u64 = 8;
const SCALES: u64 = 9;
const GATHER: u64 = 10;
const SORTED: u64 = 11;
const OFFSETS: u64 = 12;
const PADDED: u64 = 13;
const ACTIVE: u64 = 14;
const TOPK: u64 = 15;
const PARTIAL: u64 = 16;
const SEMAPHORES: u64 = 17;
const UNUSED: u64 = 18;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

struct Artifacts {
    report: BuildReport,
    /// The SM120 backend over the reported entries, as the bundle builds it.
    backend: Sm120,
    packages: BTreeMap<String, (PreparedExercise, Vec<u8>)>,
}

fn artifacts() -> Artifacts {
    let report_path = std::env::var_os("CHOPINFER_PRECISION_SELECTION_REPORT")
        .expect("set CHOPINFER_PRECISION_SELECTION_REPORT to the exported consumer report");
    let package_path = std::env::var_os("CHOPINFER_PRECISION_PACKAGES")
        .expect("set CHOPINFER_PRECISION_PACKAGES to precision_preparation.py's packages.json");
    let report = BuildReport::read(&PathBuf::from(report_path)).unwrap();
    report.verify_current().unwrap();
    assert!(report.source_ownership_complete && report.entry_declarations_complete);
    let context = report.context.as_ref().unwrap();
    assert_eq!(context.selection, model::ARCHITECTURE);
    let root = |name| &context.sources.iter().find(|s| s.id == name).unwrap().path;
    let selection = precision_selection::selection(
        root("external-model"),
        &root("chopinfer").join("crates/kernels-sm120/cuda"),
    )
    .unwrap();
    assert_eq!(report.cubins.len(), 3);
    let built = report
        .cubins
        .iter()
        .map(|c| BuiltCubin {
            name: c.name.clone(),
            path: c.path.clone(),
            identity_path: c.identity_path.clone(),
            line_info: c.identity["line_info"].as_bool().unwrap(),
            sha256: c.identity["cubin_sha256"].as_str().unwrap().into(),
            source_closure_sha256: c.recipe.source_closure_sha256.clone(),
            entries: serde_json::from_value(c.identity["resources"].clone()).unwrap(),
        })
        .collect::<Vec<_>>();
    for cubin in &report.cubins {
        assert_eq!(cubin.declarations, selection.declarations()[&cubin.name]);
    }
    let offers = precision_selection::offers().unwrap();
    let published =
        kernels_sm120::precision::publish_capabilities(&offers, &selection, &built).unwrap();
    let available = offers
        .iter()
        .flat_map(|o| o.handle().entries().unwrap())
        .map(|e| e.symbol)
        .collect();
    let backend =
        precision_selection::backend(offers, available, published.binding_dependency).unwrap();
    // Each name maps to {"package": <p- directory>, "checkpoint": <directory>}.
    let paths: BTreeMap<String, BTreeMap<String, PathBuf>> =
        serde_json::from_slice(&std::fs::read(package_path).unwrap()).unwrap();
    let mut packages = BTreeMap::new();
    for (name, entry) in paths {
        let (path, checkpoint) = (&entry["package"], &entry["checkpoint"]);
        let package = model_compiler_sdk::verify_emitted_package(path).unwrap();
        let native = package.manifest().native_module().unwrap();
        let module = NativeModule::inspect(&path.join(native.path().as_str())).unwrap();
        assert_eq!(module.sha256, native.sha256());
        let profile = &module.contracts.profiles[0];
        assert_eq!(
            profile.weight_contract,
            kernel_build::digest(model::WEIGHT_CONTRACT)
        );
        assert_eq!(
            profile.package_contract,
            kernel_build::digest(model::PACKAGE_CONTRACT)
        );
        let contract = module.contracts.implementation_contract.as_ref().unwrap();
        assert_eq!(contract.selection, published.selection);
        assert_eq!(contract.dependencies, published.dependencies);
        assert_eq!(contract.capabilities.as_ref(), Some(&published.sha256));
        let portable = module.precision_inventory().unwrap();
        assert_eq!(
            serde_json::to_vec(portable.document()).unwrap(),
            published.bytes
        );
        let component = |role| {
            package
                .read_component(
                    package
                        .manifest()
                        .components()
                        .iter()
                        .find(|c| c.role().as_str() == role)
                        .unwrap(),
                )
                .unwrap()
        };
        let metadata = component(model::PRECISION_ROLE);
        let exercise = PreparedExercise::bind(&metadata, ActivationPolicy::Auto, &backend).unwrap();
        assert_eq!(
            kernel_core::precision::component::decode(&component(model::POLICY_ROLE)).unwrap(),
            exercise.metadata.precision,
        );
        for policy in [
            ActivationPolicy::Auto,
            ActivationPolicy::Checkpoint,
            ActivationPolicy::Bf16,
        ] {
            let allowed = exercise
                .metadata
                .precision
                .operations
                .iter()
                .all(|selection| selection.require_policy(policy).is_ok());
            assert_eq!(
                model_compiler_sdk::quantization::package::check(&package, policy).is_ok(),
                allowed
            );
        }
        // The weights are a planned component the bundle's weight plan fills
        // from the checkpoint, given the package's bundle metadata.
        let manifest = package.manifest();
        assert!(
            manifest
                .planned_components()
                .iter()
                .any(|c| c.role().as_str() == model::WEIGHTS_ROLE)
        );
        assert!(
            manifest
                .resource_requirements()
                .required_capabilities()
                .iter()
                .any(|c| c.as_str() == runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1)
        );
        let mut components: Vec<_> = manifest
            .components()
            .iter()
            .filter(|c| c.role().as_str().starts_with("chop.bundle-metadata."))
            .map(|c| runtime_bundle_abi::weight_plan::NamedBytesV1 {
                name: c.role().as_str().to_owned(),
                bytes: package.read_component(c).unwrap(),
            })
            .collect();
        components.sort_by(|a, b| a.name.cmp(&b.name));
        let image = planned_weights(&path.join(native.path().as_str()), components, checkpoint);
        assert_eq!(image.len() as u64, exercise.metadata.image_bytes);
        packages.insert(name, (exercise, image));
    }
    for name in [
        "auto",
        "checkpoint",
        "bf16",
        "per-expert-auto",
        "changed-auto",
    ] {
        assert!(packages.contains_key(name), "missing prepared case {name}");
    }
    Artifacts {
        report,
        backend,
        packages,
    }
}

#[test]
fn authenticated_bindings_refuse_changes_and_resolve_missing_graph_capacity() {
    let a = artifacts();
    let metadata = &a.packages["checkpoint"].0.metadata;
    let encoded = serde_json::to_vec(metadata).unwrap();
    assert!(PreparedExercise::bind(&encoded, ActivationPolicy::Bf16, &a.backend).is_err());
    let mut bad = metadata.clone();
    bad.weights[1].codes.offset = bad.weights[0].codes.offset;
    assert!(
        PreparedExercise::bind(
            &serde_json::to_vec(&bad).unwrap(),
            ActivationPolicy::Auto,
            &a.backend
        )
        .is_err()
    );
    let mut bad = metadata.clone();
    bad.precision.operations[0].bindings[0]
        .implementation
        .dependency = "0".repeat(64);
    assert!(
        PreparedExercise::bind(
            &serde_json::to_vec(&bad).unwrap(),
            ActivationPolicy::Auto,
            &a.backend
        )
        .is_err()
    );
    let offers = precision_selection::offers()
        .unwrap()
        .into_iter()
        .filter(|offer| !(offer.usage().graph && offer.handle().quantizer.is_some()))
        .collect::<Vec<_>>();
    let symbols = offers
        .iter()
        .flat_map(|o| o.handle().entries().unwrap())
        .map(|e| e.symbol)
        .collect();
    let inventory = CompiledInventory::new(offers, symbols, "a".repeat(64)).unwrap();
    for source in metadata
        .precision
        .operations
        .iter()
        .take(2)
        .map(|s| &s.source)
    {
        assert!(resolve_operation(source, ActivationPolicy::Checkpoint, &inventory).is_err());
        let automatic = resolve_operation(source, ActivationPolicy::Auto, &inventory).unwrap();
        assert!(
            automatic
                .selection()
                .effective
                .activations
                .iter()
                .all(|a| a.is_bf16())
        );
        assert_eq!(automatic.selection().bindings.len(), model::uses().len());
    }
    assert_ne!(a.packages["auto"].1, a.packages["changed-auto"].1);
    assert_eq!(a.packages["auto"].1, a.packages["bf16"].1);
}

/// A package's weight image: the weight plan of its native module `module`
/// for `model::WEIGHTS_ROLE`, executed over `checkpoint` by the host's
/// direct-load executor with the package's bundle metadata `components`, as
/// a direct load executes it.
fn planned_weights(
    module: &std::path::Path,
    components: Vec<runtime_bundle_abi::weight_plan::NamedBytesV1>,
    checkpoint: &std::path::Path,
) -> Vec<u8> {
    let bytes = std::fs::read(module).unwrap();
    let digest: [u8; 32] = sha2::Sha256::digest(&bytes).into();
    let library = runtime_bundle_abi::BundleLibraryV1::open_verified_image(
        &bytes,
        &module.display().to_string(),
    )
    .unwrap();
    let checkpoint = Arc::new(runtime_host::direct::DirectCheckpointV1::open(checkpoint).unwrap());
    let image = runtime_host::direct::plan_image_v1(
        &library,
        digest,
        checkpoint,
        model::WEIGHTS_ROLE,
        1,
        components,
        vec![],
    )
    .unwrap()
    .expect("the bundle plans its weight image");
    assert_eq!(image.image_role(), model::WEIGHTS_ROLE);
    let mut weights = vec![0; image.arena_bytes(0) as usize];
    image.read(0, &mut weights).unwrap();
    image.checkpoint().verify_unchanged().unwrap();
    weights
}

fn span<'a>(image: &'a [u8], s: &model::Span) -> &'a [u8] {
    &image[s.range(image.len() as u64).unwrap()]
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

// Decode the declared physical representation independently of the device code.
// Exact A16 local operands retain the global scale in the oracle weight.
fn weight_values(exercise: &PreparedExercise, image: &[u8], index: usize) -> Vec<f64> {
    let selected = &exercise.metadata.precision.operations[index];
    let views = &exercise.metadata.weights[index];
    let codes = span(image, &views.codes);
    let globals = span(image, &views.global_scales);
    let k = if index == 2 {
        model::TABLE_WIDTH
    } else {
        model::HIDDEN
    } as usize;
    let rows = codes.len() * 2 / k;
    let scales = span(image, &views.local_scales).to_vec();
    // The grouped operand stores its scales in 16-row by 8-block MMA tiles.
    let scales = if index == 1 {
        let tiles = ScaleTileLayout {
            rows,
            k_blocks: k / 16,
            tile_rows: 16,
            k_blocks_per_tile: 8,
        };
        let packed = Packed {
            bytes: scales,
            layout: Layout::ScaleTile(tiles),
        };
        tiles.unpack(&packed).unwrap()
    } else {
        scales
    };
    // One global scale per tensor, or per INTERMEDIATE rows.
    let segment = if globals.len() == 4 {
        rows
    } else {
        model::INTERMEDIATE as usize
    };
    let order = selected.effective.arithmetic.decode_order;
    let mut values = Vec::with_capacity(rows * k);
    for (first, global) in (0..rows).step_by(segment).zip(globals.chunks_exact(4)) {
        let codes = &codes[first * k / 2..(first + segment) * k / 2];
        let local = &scales[first * k / 16..(first + segment) * k / 16];
        let view = MatrixView {
            tensor: &views.operation,
            format: WeightFormat::Nvfp4,
            rows: segment,
            k,
            row_origin: 0,
            k_origin: 0,
            nibble_order: NibbleOrder::EvenLow,
            codes,
            local: Some(ScaleBytes {
                dtype: DType::Ue4M3,
                bytes: local,
            }),
            global: Some(ScaleBytes {
                dtype: DType::F32,
                bytes: global,
            }),
        };
        let nvfp4 = QuantFormat {
            element: SmallFloat::E2M1,
            block: (1, 16),
            scale: ScaleFormat::E4M3,
            chunk_rows: None,
        };
        let global = f32::from_le_bytes(global.try_into().unwrap());
        let exact = Quantized::from_parts(
            segment,
            k,
            nvfp4,
            codes.to_vec(),
            Scales::Ue4M3(local.to_vec()),
            global,
        )
        .unwrap();
        for i in 0..segment * k {
            let (row, col) = (i / k, i % k);
            values.push(match order {
                DecodeOrder::ScaleThenCode
                | DecodeOrder::CodeThenScale
                | DecodeOrder::ExactWeightToBf16 => view.oracle_weight(row, col, order).unwrap(),
                _ => exact.decode(row, col),
            });
        }
    }
    values
}

fn input(rows: usize, k: usize) -> Vec<u16> {
    (0..rows * k)
        .map(|i| f32_to_bf16(((i * 17 + i / k * 13) % 101) as f32 / 256.0 - 0.1875))
        .collect()
}
fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = s.allocate(bytes.len().max(4))?;
    if !bytes.is_empty() {
        buffer.copy_from_bytes(bytes, &s.stream).map_err(err)?;
    }
    Ok(buffer)
}
fn read(s: &Session, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; len];
    buffer.copy_to_bytes(&mut bytes, &s.stream).map_err(err)?;
    Ok(bytes)
}

struct Call {
    symbol: String,
    function: Function,
    plan: LaunchPlan,
    arguments: ArgumentList,
}
fn call(
    s: &Session,
    modules: &BTreeMap<String, runtime_cuda::Module>,
    symbol: &str,
    plan: LaunchPlan,
    arguments: ArgumentList,
) -> LabResult<Call> {
    let _ = s;
    let function = modules
        .values()
        .find_map(|m| m.function(symbol).ok())
        .ok_or_else(|| format!("unbound entry {symbol}"))?;
    if plan.dynamic_shared_bytes >= 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(err)?;
    }
    Ok(Call {
        symbol: symbol.into(),
        function,
        plan,
        arguments,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "device case identifies package, operation, capacity and live rows"
)]
fn execute(
    s: &Session,
    modules: &BTreeMap<String, runtime_cuda::Module>,
    exercise: &PreparedExercise,
    image: &[u8],
    package: &str,
    index: usize,
    usage: OperationUse,
    live: u32,
) -> LabResult<()> {
    let op = exercise.handle(index, usage)?;
    // The bytes the op sizes for a role; zero for a role it does not use.
    let bytes = |role: &str| op.info().bytes_of(role).unwrap_or(0) as usize;
    // This op set quantizes activations into `scratch.codes` and gathers
    // BF16 rows into `scratch.gathered`.
    let quantizes = bytes("scratch.codes") > 0;
    let gathered_elements = bytes("scratch.gathered") / 2;
    let views = &exercise.metadata.weights[index];
    let capacity = usage.capacity as usize;
    let grouped = index == 1;
    let lookup = index == 2;
    let width = if lookup {
        model::TABLE_WIDTH as usize
    } else {
        model::HIDDEN as usize
    };
    let out_capacity = capacity * if grouped { model::TOP_K as usize } else { 1 };
    let output_len = out_capacity * width + 128;
    let live_rows = live as usize * if grouped { model::TOP_K as usize } else { 1 };
    let x = input(capacity, width);
    let ids = (0..capacity)
        .map(|i| {
            if i % 5 == 0 {
                -1
            } else {
                ((i * 13) % (model::VOCAB as usize + 3)) as i32
            }
        })
        .collect::<Vec<_>>();
    let routes = (0..live_rows)
        .map(|r| (r + r / model::TOP_K as usize) % model::EXPERTS as usize)
        .collect::<Vec<_>>();
    // This model pads expert routes to 16 rows. A prefill CTA may cover more
    // rows, but the declared routed capacity does not reserve a full CTA each.
    let permute = oracle::expert_permute(&routes, model::EXPERTS as usize, 16).map_err(err)?;
    let padded = permute.forward.len();
    let mut buffers = BTreeMap::new();
    buffers.insert(
        X,
        if lookup {
            upload(s, &bytes_i32(&ids))?
        } else {
            s.upload_bf16(&x)?
        },
    );
    for (id, view) in [
        (W, &views.codes),
        (WS, &views.local_scales),
        (WG, &views.global_scales),
        (AG, &views.input_scale),
    ] {
        buffers.insert(id, upload(s, span(image, view))?);
    }
    buffers.insert(Y, s.upload_bf16(&vec![GUARD; output_len])?);
    buffers.insert(LIVE, upload(s, &live.to_le_bytes())?);
    let sorted = permute
        .forward
        .iter()
        .map(|v| v.map_or(-1, |r| r as i32))
        .collect::<Vec<_>>();
    let offsets = permute
        .expert_offsets
        .iter()
        .map(|r| *r as i32)
        .collect::<Vec<_>>();
    let active = (0..model::EXPERTS as usize)
        .filter(|e| offsets[*e] < offsets[*e + 1])
        .map(|e| e as i32)
        .collect::<Vec<_>>();
    buffers.insert(SORTED, upload(s, &bytes_i32(&sorted))?);
    buffers.insert(OFFSETS, upload(s, &bytes_i32(&offsets))?);
    buffers.insert(
        ACTIVE,
        upload(s, &bytes_i32(&[vec![active.len() as i32], active].concat()))?,
    );
    buffers.insert(PADDED, upload(s, &(padded as u32).to_le_bytes())?);
    buffers.insert(TOPK, s.upload_f32(&vec![0.5; out_capacity])?);
    // Scratch at the bytes the op sizes, with a guard tail.
    buffers.insert(
        GATHER,
        s.upload_bf16(&vec![GUARD; gathered_elements + 128])?,
    );
    buffers.insert(CODES, upload(s, &vec![0xa5; bytes("scratch.codes") + 128])?);
    buffers.insert(
        SCALES,
        upload(s, &vec![0xa5; bytes("scratch.scales") + 128])?,
    );
    buffers.insert(
        PARTIAL,
        upload(s, &vec![0; bytes("scratch.partial").max(4)])?,
    );
    buffers.insert(
        SEMAPHORES,
        upload(s, &vec![0; bytes("scratch.semaphores").max(4096)])?,
    );
    // The A16 grouped entries bind their unread activation scale slots here.
    buffers.insert(UNUSED, upload(s, &[0; 4])?);
    let view = |id: u64| View {
        handle: id,
        bytes: buffers[&id].byte_len() as u64,
    };
    let roles = ProjectionBuffers {
        activation: view(X),
        live_rows: view(LIVE),
        output: view(Y),
        residual: None,
        weights: QuantizedWeights {
            codes: view(W),
            local_scales: Some(view(WS)),
            global_scales: Some(view(WG)),
        },
        activation_global: Some(view(AG)),
        routing: grouped.then(|| ProjectionRouting {
            weights: view(TOPK),
            routes: RouteMetadata {
                counts: None,
                offsets: view(OFFSETS),
                sorted_rows: view(SORTED),
                padded_rows: view(PADDED),
                active_experts: Some(view(ACTIVE)),
            },
        }),
        scratch: ProjectionScratch {
            partial: view(PARTIAL),
            semaphores: view(SEMAPHORES),
            unused: view(UNUSED),
            temporary: None,
            gathered: Some(view(GATHER)),
            codes: Some(view(CODES)),
            scales: Some(view(SCALES)),
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
            quantizer_status: None,
            quantizer_status_row_offset: 0,
        },
        row_offset: 0,
        shard_rows: model::VOCAB,
    };
    // A launch schedules at least one row; zero live rows launch one
    // scheduled row whose device count writes nothing.
    let counts = Counts {
        padded_rows: padded as u32,
        ..Counts::rows(live.max(1))
    };
    let calls = op
        .launches(&roles, View::EMPTY, counts)
        .map_err(err)?
        .into_iter()
        .map(|l| call(s, modules, &l.symbol, l.plan, l.arguments))
        .collect::<LabResult<Vec<_>>>()?;
    let mut gathered_expected = None;
    let mut decoded_x = x[..live as usize * width]
        .iter()
        .map(|b| f64::from(bf16_to_f32(*b)))
        .collect::<Vec<_>>();
    let mut quant_rows = live as usize;
    if gathered_elements > 0 {
        quant_rows = padded;
        decoded_x = permute
            .forward
            .iter()
            .flat_map(|r| {
                let x = &x;
                (0..width).map(move |col| {
                    r.map_or(0.0, |r| {
                        f64::from(bf16_to_f32(x[r / model::TOP_K as usize * width + col]))
                    })
                })
            })
            .collect();
        gathered_expected = Some(
            decoded_x
                .iter()
                .map(|v| f32_to_bf16(*v as f32))
                .collect::<Vec<_>>(),
        );
    }
    let mut quantized = None;
    if quantizes && quant_rows > 0 {
        let expected = quantize::nvfp4_with_global(
            &decoded_x.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            quant_rows,
            width,
            floats(span(image, &views.input_scale))[0],
        )
        .map_err(err)?;
        decoded_x = expected.decoded();
        quantized = Some(expected);
    }
    let bindings = Bindings {
        buffers: buffers.iter().map(|(id, buffer)| (*id, buffer)).collect(),
    };
    let graph = s.capture(|stream| {
        for c in &calls {
            launch(
                &s.context,
                &c.function,
                stream,
                c.plan,
                &c.arguments,
                &bindings,
            )
            .map_err(|e| format!("{} {:?}: {e}", c.symbol, c.plan))?;
        }
        Ok(())
    })?;
    let mut previous = None;
    for _ in 0..3 {
        graph.launch(&s.stream).map_err(err)?;
        let actual = s.download_bf16(&buffers[&Y], output_len)?;
        if previous.as_ref().is_some_and(|last| last != &actual) {
            return Err("graph replay changed output bits".into());
        }
        if actual[live_rows * width..].iter().any(|v| *v != GUARD) {
            return Err("operation overwrote its output guard".into());
        }
        previous = Some(actual);
    }
    if let Some(expected) = gathered_expected {
        let actual = s.download_bf16(&buffers[&GATHER], gathered_elements + 128)?;
        if actual[..expected.len()] != expected
            || actual[expected.len()..].iter().any(|v| *v != GUARD)
        {
            return Err("BF16 gather differs or overwrote its live-row guard".into());
        }
    }
    if quantizes {
        let codes = read(s, &buffers[&CODES], bytes("scratch.codes") + 128)?;
        let scales = read(s, &buffers[&SCALES], bytes("scratch.scales") + 128)?;
        let (expected_codes, expected_scales) = if let Some(expected) = &quantized {
            let quantize::Scales::Ue4M3(scales) = &expected.scales else {
                unreachable!()
            };
            (expected.codes.as_slice(), scales.as_slice())
        } else {
            (&[][..], &[][..])
        };
        if &codes[..expected_codes.len()] != expected_codes
            || &scales[..expected_scales.len()] != expected_scales
        {
            return Err(
                "calibrated activation codes or scales differ from the exact reference".into(),
            );
        }
        if codes[expected_codes.len()..]
            .iter()
            .chain(&scales[expected_scales.len()..])
            .any(|v| *v != 0xa5)
        {
            return Err("quantizer wrote outside live rows".into());
        }
    }
    let actual = previous.unwrap();
    let weights = weight_values(exercise, image, index);
    let mut receipt = BTreeMap::from([
        ("package".into(), json!(package)),
        ("operation".into(), json!(views.operation)),
        ("usage".into(), json!(usage)),
        ("live".into(), json!(live)),
        ("graph_replays".into(), json!(3)),
    ]);
    if lookup {
        let expected = ids[..live as usize]
            .iter()
            .flat_map(|id| {
                (0..width).map(|col| {
                    if *id < 0 || *id as u64 >= model::VOCAB {
                        0
                    } else {
                        f32_to_bf16(weights[*id as usize * width + col] as f32)
                    }
                })
            })
            .collect::<Vec<_>>();
        if actual[..expected.len()] != expected {
            return Err("packed table lookup differs bitwise".into());
        }
        receipt.insert("lookup_bitwise".into(), json!(true));
    } else if live > 0 {
        let (a, ranges) = if grouped {
            let input = (0..live_rows)
                .flat_map(|r| {
                    let row = if quantizes {
                        permute.inverse[r]
                    } else {
                        r / model::TOP_K as usize
                    };
                    decoded_x[row * width..(row + 1) * width].iter().copied()
                })
                .collect::<Vec<_>>();
            (
                input,
                routes
                    .iter()
                    .enumerate()
                    .map(|(r, e)| ExpertRowRange {
                        expert: *e,
                        start: r,
                        end: r + 1,
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            (decoded_x, vec![])
        };
        let reference = oracle::gemm(
            &a,
            &weights,
            GemmShape {
                m: live_rows,
                n: width * if grouped { 2 } else { 1 },
                k: width,
                experts: if grouped { model::EXPERTS as usize } else { 1 },
            },
            &ranges,
            GemmEpilogue {
                residual: None,
                swiglu: if grouped {
                    Some(kernel_core::contract::SwigluColumns::ConcatenatedHalves)
                } else {
                    None
                },
            },
        )
        .map_err(err)?;
        kernel_harness::check_output(
            &actual[..live_rows * width]
                .iter()
                .map(|v| bf16_to_f32(*v))
                .collect::<Vec<_>>(),
            &reference,
            DType::Bf16,
            &mut receipt,
        )?;
        oracle::check_layer_cosine(
            &actual[..live_rows * width]
                .iter()
                .map(|v| f64::from(bf16_to_f32(*v)))
                .collect::<Vec<_>>(),
            &reference.values,
            width,
        )
        .map_err(|e| format!("layer cosine failed for row {}: {}", e.index, e.actual))?;
        receipt.insert("layer_cosine_minimum".into(), json!(0.9999));
    }
    s.receipt("independent-precision", receipt)?;
    Ok(())
}

#[test]
fn prepared_projection_experts_and_lookup_device_oracles() {
    let artifacts = artifacts();
    let result = kernel_harness::gpu_test(128 * 1024 * 1024, |session| {
        let modules = artifacts
            .report
            .cubins
            .iter()
            .map(|c| {
                Ok((
                    c.name.clone(),
                    session.load_cubin(&std::fs::read(&c.path).map_err(err)?)?,
                ))
            })
            .collect::<LabResult<BTreeMap<_, _>>>()?;
        for (name, (exercise, image)) in &artifacts.packages {
            for usage in model::uses() {
                let live = if usage.capacity == 1 {
                    vec![1]
                } else {
                    vec![1, 3, usage.capacity]
                };
                for index in 0..3 {
                    for &rows in &live {
                        execute(session, &modules, exercise, image, name, index, usage, rows)
                            .map_err(|e| {
                                format!("{name} operation {index} {usage:?} live={rows}: {e}")
                            })?;
                    }
                    if index != 0 {
                        execute(session, &modules, exercise, image, name, index, usage, 0)
                            .map_err(|e| {
                                format!("{name} operation {index} {usage:?} live=0: {e}")
                            })?;
                    }
                }
            }
        }
        Ok(())
    })
    .expect("independent prepared operation GPU cases");
    if result.is_none() {
        eprintln!(
            "UNRUN: independent precision device arithmetic and graph replay require an idle assigned GPU"
        );
    }
}
