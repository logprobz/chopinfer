//! The direct load's weights equal the image the rank-image path packed.
//!
//! For the synthetic checkpoints `tests/make_precision.py` writes, the test
//! builds the former `weights.bin` with the SDK's prepared-weight builder
//! (as the removed `materialize` did), then executes the bundle's weight
//! plan record by record as the host does (`WeightRecord::prepare` and
//! `produce` over the checkpoint's bytes), and compares every section.
//!
//! It needs the native module built with `precision-exercise` for the
//! capability inventory precision selection reads: `libexample_bundle.so`
//! beside the test executable or one directory up, where Cargo places the
//! workspace's cdylib when it builds the workspace's tests.
use super::*;
use example_contract::weight_plan;
use kernel_core::precision::packed::MatrixSlice;
use kernel_core::weight_transform::{PackedSection, SourceTensor, WeightRecord, WeightTransform};
use model_compiler_sdk::quantization::{packing::PackedWeightBuilder, payload::read_operand};
use runtime_bundle_abi::weight_plan::{
    CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1, WeightPlanV1,
};
use std::collections::BTreeMap;
use std::process::Command;

/// A scratch directory, removed when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch() -> Scratch {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "example-precision-plan-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir(&path).unwrap();
    Scratch(path)
}

/// Writes a checkpoint with the preparation test's fixture writer.
fn write_checkpoint(root: &Path, schema: &str, seed: u32, expert_scales: bool) {
    let writer = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/make_precision.py");
    let mut command = Command::new("python3");
    command.args(["-B".as_ref(), writer.as_os_str()]);
    command.args(["--output".as_ref(), root.as_os_str()]);
    command.args(["--schema", schema, "--seed", &seed.to_string()]);
    if expert_scales {
        command.arg("--expert-scales");
    }
    let status = command
        .status()
        .expect("run python3 for the fixture writer");
    assert!(
        status.success(),
        "fixture writer failed for {}",
        root.display()
    );
}

/// Replaces every code byte of the checkpoint with a pattern that does not
/// repeat per expert or per row, as the fixture's codes do. The headers, and
/// so the checkpoint identity, stay the same.
fn scramble_codes(root: &Path) {
    let path = root.join("weights.safetensors");
    let mut bytes = std::fs::read(&path).unwrap();
    let header = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    for (i, byte) in bytes[8 + header..].iter_mut().enumerate() {
        *byte = ((i as u32).wrapping_mul(0x9e37_79b1) >> 24) as u8;
    }
    std::fs::write(path, bytes).unwrap();
}

fn module() -> NativeModule {
    let executable = std::env::current_exe().unwrap();
    let directory = executable.parent().unwrap();
    for path in [
        directory.join(NATIVE_MODULE),
        directory.join("..").join(NATIVE_MODULE),
    ] {
        if path.is_file() {
            let module = NativeModule::inspect(&path).unwrap();
            module
                .require(&profile(1).unwrap())
                .expect("the found module is not built with precision-exercise");
            return module;
        }
    }
    panic!(
        "no {NATIVE_MODULE} near {}; build the workspace's tests with --features precision-exercise",
        executable.display()
    );
}

/// The former `materialize`: every operand packed by the SDK's
/// prepared-weight builder at the resolved plan, and each static
/// calibration copied, at the exercise's spans.
fn packed_image(source: &Source, exercise: &model::Exercise) -> Result<Vec<u8>> {
    let mut image = vec![0; exercise.image_bytes as usize];
    for (index, operand) in source.operands.iter().enumerate() {
        let loaded = read_operand(operand.clone(), |span| {
            source.inventory.read(&source.files, span)
        })?;
        let views = &exercise.weights[index];
        let plan = &views.packing.plan;
        let selected = &exercise.precision.operations[index];
        let mut builder = PackedWeightBuilder::new(plan.clone())?;
        for expert in 0..plan.experts {
            let matrix = loaded.matrix(expert as usize)?;
            if selected.effective.activations.iter().all(|a| a.is_bf16()) {
                matrix.validate_bf16_operands(selected.effective.arithmetic.decode_order)?;
            }
            for projection in 0..plan.projections {
                builder.write_matrix(
                    expert,
                    projection,
                    matrix,
                    &MatrixSlice {
                        rows: (projection as usize * plan.rows as usize
                            ..(projection + 1) as usize * plan.rows as usize)
                            .collect(),
                        k_start: 0,
                        k: plan.k as usize,
                        padded_rows: plan.rows as usize,
                        padded_k: plan.k as usize,
                        nibble_order: NibbleOrder::EvenLow,
                    },
                )?;
            }
        }
        let packed = builder.finish()?;
        let input = calibration(loaded.descriptor())
            .map(|s| loaded.payloads()[&s.tensor].as_slice())
            .unwrap_or(&[]);
        for (span, bytes) in [
            (&views.codes, packed.codes.as_slice()),
            (&views.local_scales, packed.local_scales.as_slice()),
            (&views.global_scales, packed.global_scales.as_slice()),
            (&views.input_scale, input),
        ] {
            if span.bytes != bytes.len() as u64 {
                return Err("packed bytes differ from the prepared example view".into());
            }
            image[span.range(exercise.image_bytes)?].copy_from_slice(bytes);
        }
    }
    source.files.verify()?;
    Ok(image)
}

/// The checkpoint's safetensors files as the host describes them, sorted by
/// name, and their bytes.
fn files(root: &Path) -> (Vec<CheckpointFileV1>, Vec<Vec<u8>>) {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".safetensors"))
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let bytes = std::fs::read(root.join(&name)).unwrap();
            let length = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            let file = CheckpointFileV1 {
                name,
                byte_length: bytes.len() as u64,
                data_offset: 8 + length,
                header: bytes[8..8 + length as usize].to_vec(),
            };
            (file, bytes)
        })
        .unzip()
}

/// The request the host sends for the package of `exercise` over `root`.
fn plan_request(
    root: &Path,
    exercise: &model::Exercise,
    role: &str,
) -> (WeightPlanRequestV1, Vec<Vec<u8>>) {
    let identity = preparation::direct::checkpoint_identity(root).unwrap();
    let mut checkpoint_identity = [0_u8; 32];
    for (byte, pair) in checkpoint_identity
        .iter_mut()
        .zip(identity.as_bytes().chunks(2))
    {
        *byte = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
    }
    let (files, bytes) = files(root);
    let mut components = vec![
        NamedBytesV1 {
            name: model::POLICY_ROLE.into(),
            bytes: kernel_core::precision::component::encode(&exercise.precision).unwrap(),
        },
        NamedBytesV1 {
            name: model::PRECISION_ROLE.into(),
            bytes: serde_json::to_vec(exercise).unwrap(),
        },
    ];
    components.sort_by(|a, b| a.name.cmp(&b.name));
    let request = WeightPlanRequestV1 {
        image_role: role.into(),
        ranks: 1,
        rank: 0,
        checkpoint_identity,
        files,
        components,
        ..Default::default()
    };
    (request, bytes)
}

/// The plan's image, each record checked against its sources' headers and
/// produced from positional reads of the checkpoint, as the host does.
fn execute(
    plan: &WeightPlanV1,
    files: &[CheckpointFileV1],
    bytes: &[Vec<u8>],
) -> std::result::Result<Vec<u8>, String> {
    let mut image = vec![0; plan.arena_bytes as usize];
    for record in &plan.records {
        let mut sources = Vec::new();
        let mut views: Vec<&[u8]> = Vec::new();
        for source in &record.sources {
            let file = &files[source.file as usize];
            let header: serde_json::Value = serde_json::from_slice(&file.header).unwrap();
            let entry = &header[&source.tensor];
            let offsets = [0, 1].map(|i| entry["data_offsets"][i].as_u64().unwrap());
            sources.push(SourceTensor {
                dtype: entry["dtype"].as_str().unwrap().into(),
                shape: entry["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap())
                    .collect(),
                byte_length: offsets[1] - offsets[0],
            });
            let base = file.data_offset;
            views.push(
                &bytes[source.file as usize]
                    [(base + offsets[0]) as usize..(base + offsets[1]) as usize],
            );
        }
        let prepared = WeightRecord {
            transform: WeightTransform::decode(record.transform, &record.params)
                .map_err(|e| e.to_string())?,
            sources,
            axis: record.axis,
            ranges: record.ranges.clone(),
            stored_shape: record.stored_shape.clone(),
        }
        .prepare(record.byte_length)
        .map_err(|e| e.to_string())?;
        let mut read = |index: usize, offset: u64, buffer: &mut [u8]| {
            let view = views[index]
                .get(offset as usize..offset as usize + buffer.len())
                .ok_or_else(|| {
                    kernel_core::weight_transform::WeightTransformError("read past a source".into())
                })?;
            buffer.copy_from_slice(view);
            Ok(())
        };
        let out = &mut image[record.arena_offset as usize..][..record.byte_length as usize];
        prepared
            .produce(0..record.byte_length, out, &mut read, &mut Vec::new())
            .map_err(|e| format!("record at {}: {e}", record.arena_offset))?;
    }
    Ok(image)
}

#[test]
fn the_weight_plan_packs_the_image_the_prepared_weight_builder_wrote() {
    let inventory = module().precision_inventory().unwrap();
    let scratch = scratch();
    let cases: [(&str, &str, u32, bool, &[&str]); 4] = [
        (
            "uniform-a",
            "nested",
            7,
            false,
            &["auto", "checkpoint", "bf16"],
        ),
        ("uniform-b", "flat", 19, false, &["auto"]),
        ("per-expert", "nested", 7, true, &["auto", "bf16"]),
        ("scrambled", "nested", 7, true, &["auto", "bf16"]),
    ];
    let mut images = BTreeMap::new();
    for (name, schema, seed, expert_scales, policies) in cases {
        let root = scratch.0.join(name);
        write_checkpoint(&root, schema, seed, expert_scales);
        if name == "scrambled" {
            scramble_codes(&root);
        }
        let source = open(&root, 1).unwrap();
        let identity = preparation::direct::checkpoint_identity(&root).unwrap();
        for policy in policies {
            let exercise =
                resolve(&source, policy.parse().unwrap(), &inventory, &identity).unwrap();
            let expected = packed_image(&source, &exercise).unwrap();
            let (request, bytes) = plan_request(&root, &exercise, model::WEIGHTS_ROLE);
            let plan = weight_plan::plan(&request).unwrap().unwrap();
            let planned = execute(&plan, &request.files, &bytes).unwrap();
            for spans in &exercise.weights {
                for (section, span) in [
                    ("codes", &spans.codes),
                    ("local scales", &spans.local_scales),
                    ("global scales", &spans.global_scales),
                    ("input scale", &spans.input_scale),
                ] {
                    let range = span.range(exercise.image_bytes).unwrap();
                    assert!(
                        planned[range.clone()] == expected[range],
                        "{name} {policy}: {} {section} differ",
                        spans.operation
                    );
                }
            }
            assert!(planned == expected, "{name} {policy}: padding differs");
            // The primary image is the same plan; other images are declined.
            let (primary, _) = plan_request(&root, &exercise, "");
            assert_eq!(weight_plan::plan(&primary).unwrap(), Some(plan));
            let (other, _) = plan_request(&root, &exercise, "example.other.v1");
            assert_eq!(weight_plan::plan(&other).unwrap(), None);
            images.insert((name, *policy), planned);
        }
    }
    assert_eq!(
        images[&("uniform-a", "auto")],
        images[&("uniform-a", "bf16")]
    );
    assert_ne!(
        images[&("uniform-a", "auto")],
        images[&("uniform-b", "auto")]
    );
    assert_ne!(
        images[&("per-expert", "auto")],
        images[&("scrambled", "auto")]
    );
    // Per-expert calibration refuses the checkpoint recipe.
    let source = open(&scratch.0.join("per-expert"), 1).unwrap();
    assert!(
        resolve(
            &source,
            ActivationPolicy::Checkpoint,
            &inventory,
            &"0".repeat(64)
        )
        .is_err()
    );
}

#[test]
fn the_weight_plan_refuses_other_checkpoints_and_invalid_scales() {
    let inventory = module().precision_inventory().unwrap();
    let scratch = scratch();
    let (a, b) = (scratch.0.join("a"), scratch.0.join("b"));
    write_checkpoint(&a, "nested", 7, false);
    write_checkpoint(&b, "nested", 7, true);
    let identity = preparation::direct::checkpoint_identity(&a).unwrap();
    let exercise = resolve(
        &open(&a, 1).unwrap(),
        ActivationPolicy::Bf16,
        &inventory,
        &identity,
    )
    .unwrap();
    // A package plans only the checkpoint whose headers it was prepared from.
    let (request, _) = plan_request(&b, &exercise, model::WEIGHTS_ROLE);
    let refusal = weight_plan::plan(&request).unwrap_err();
    assert!(refusal.contains("headers differ"), "{refusal}");
    // A document that fails validation is refused before planning.
    let mut changed = exercise.clone();
    changed.weights[1].packing.matrices.pop();
    let (request, _) = plan_request(&a, &changed, model::WEIGHTS_ROLE);
    assert!(weight_plan::plan(&request).is_err());
    // Preparation reads headers only, so a negative zero local scale is
    // refused when the host packs the weights.
    let path = a.join("scales.safetensors");
    let mut payload = std::fs::read(&path).unwrap();
    let header = u64::from_le_bytes(payload[..8].try_into().unwrap()) as usize;
    payload[8 + header] = 0x80;
    std::fs::write(&path, payload).unwrap();
    assert_eq!(
        preparation::direct::checkpoint_identity(&a).unwrap(),
        identity
    );
    let (request, bytes) = plan_request(&a, &exercise, model::WEIGHTS_ROLE);
    let plan = weight_plan::plan(&request).unwrap().unwrap();
    let refusal = execute(&plan, &request.files, &bytes).unwrap_err();
    assert!(refusal.to_lowercase().contains("scale"), "{refusal}");
}

/// The shared derivation selects the rows the example's packing path packs:
/// each derived operation's records, executed over the checkpoint, equal
/// the prepared-weight builder's sections when it is given every expert's
/// operand rows (gate then up for the stacked expert bank).
#[test]
fn derived_records_pack_the_bytes_of_the_example_packing() {
    use model_compiler_sdk::quantization::projection::records;
    let scratch = scratch();
    for (name, schema, seed, expert_scales) in [
        ("uniform", "nested", 7, false),
        ("per-expert", "flat", 9, true),
    ] {
        let root = scratch.0.join(name);
        write_checkpoint(&root, schema, seed, expert_scales);
        scramble_codes(&root);
        let source = open(&root, 1).unwrap();
        let read =
            |span: &kernel_core::precision::TensorSpan| source.inventory.read(&source.files, span);
        for (operand, operation) in source.operands.iter().zip(&source.operations) {
            let derived = source
                .derivation
                .operations
                .iter()
                .find(|o| o.instance.projection == operation.id)
                .unwrap();
            let plan = &derived.packing;
            let rows = derived.logical_geometry.rows as usize;
            let loaded = read_operand(operand.clone(), read).unwrap();
            let mut builder = PackedWeightBuilder::new(plan.clone()).unwrap();
            for expert in 0..plan.experts {
                for projection in 0..plan.projections {
                    let first = projection as usize * rows;
                    builder
                        .write_matrix(
                            expert,
                            projection,
                            loaded.matrix(expert as usize).unwrap(),
                            &MatrixSlice {
                                rows: (first..first + rows).collect(),
                                k_start: 0,
                                k: derived.logical_geometry.k as usize,
                                padded_rows: plan.rows as usize,
                                padded_k: plan.k as usize,
                                nibble_order: NibbleOrder::EvenLow,
                            },
                        )
                        .unwrap();
                }
            }
            let packed = builder.finish().unwrap();
            let records = records(derived).unwrap();
            assert_eq!(records.len(), 3, "{name} {}", operation.id);
            for record in records {
                let tensors: Vec<_> = record
                    .tensors
                    .iter()
                    .map(|t| read(&source.inventory.tensors[t]).unwrap())
                    .collect();
                let bytes = record.record.output_bytes().unwrap();
                let produced = record
                    .record
                    .prepare(bytes)
                    .unwrap()
                    .produce_whole(&mut |index, offset, buffer: &mut [u8]| {
                        let start = offset as usize;
                        buffer.copy_from_slice(&tensors[index][start..start + buffer.len()]);
                        Ok(())
                    })
                    .unwrap();
                let expected = match record.section {
                    PackedSection::Codes => &packed.codes,
                    PackedSection::LocalScales => &packed.local_scales,
                    PackedSection::GlobalScales => &packed.global_scales,
                };
                assert_eq!(
                    &produced, expected,
                    "{name} {} {:?}",
                    operation.id, record.section
                );
            }
        }
    }
}
