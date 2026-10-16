use std::cell::RefCell;

use kernel_build::EntryParam;
use kernel_core::backend::{Launch as BackendLaunch, OpInfo, PreparedOp};
use kernel_core::contract::{Argument, ArgumentList};
use kernel_core::properties::Invariance;
use kernels_sm120::backend::Sm120;
use kernels_sm120::encoder::{self as catalogs, Entry};

use encoder_spec::buffers::Serving;
use encoder_spec::components::Context;
use encoder_spec::{Environment, parse};
use kernel_core::contract::LaunchPlan;

use super::super::Shared;
use super::tests::{STAGED_PAIRS, TOWER};
use super::*;

/// The tower of `vision_variants_launch_over_the_plans_regions`: per-head
/// q/k norms, a clamped SwiGLU MLP and a projector after its merge.
fn variants() -> Declaration {
    let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
    let tower = &mut value["stages"][2];
    tower["attention"]["op"] = "attention.varlen.qk-rms.pairs.v1".into();
    tower["attention"]["qk_norm_eps"] = 1e-5.into();
    tower["mlp"] = serde_json::json!(
        {"op": "mlp.swiglu-clamped.v1", "hidden": 1024, "bias": true, "limit": 10});
    let weights = tower["weights"].as_object_mut().unwrap();
    weights.remove("up.weight");
    weights.remove("up.bias");
    for (role, tensor) in [
        ("q_norm.weight", "b.{i}.qn.weight"),
        ("k_norm.weight", "b.{i}.kn.weight"),
        ("gate_up.weight", "b.{i}.gate_up.weight"),
        ("gate_up.bias", "b.{i}.gate_up.bias"),
    ] {
        weights.insert(role.into(), tensor.into());
    }
    let stages = value["stages"].as_array_mut().unwrap();
    stages.truncate(5);
    stages.push(serde_json::json!(
        {"op": "linear.pairs.v1", "out": 1024, "bias": false,
         "weights": {"weight": "m.proj.weight"}}));
    stages.push(serde_json::json!(
        {"op": "projector.norm-gelu-mlp.v1",
         "norm": {"op": "norm.layer-f32.v1", "eps": 1e-5},
         "gelu": "erf",
         "mlp": {"op": "mlp.swiglu-clamped.v1", "hidden": 2048, "bias": false, "limit": 10},
         "output": "bf16",
         "weights": {"norm.weight": "m.n.weight", "norm.bias": "m.n.bias",
                     "gate_up.weight": "m.gate_up.weight", "down.weight": "m.down.weight"}}));
    value["output"]["hidden"] = 1024.into();
    parse(&value.to_string(), &Environment::new("alpha")).unwrap()
}

/// The towers: learned positions and LayerNorms; q/k norms, clamped
/// SwiGLU and a projector; a staged, stacked tower with sinks, reorders
/// and a carry.
fn towers() -> Vec<(Declaration, BufferPlan)> {
    [
        parse(TOWER, &Environment::new("alpha")).unwrap(),
        variants(),
        parse(STAGED_PAIRS, &Environment::new("alpha")).unwrap(),
    ]
    .into_iter()
    .map(|tower| {
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 1 << 20,
        };
        let plan = buffers::plan(&[&tower], 4096, serving).unwrap();
        (tower, plan)
    })
    .collect()
}

/// A catalog entry's kernel: its symbol, its plan at its capacity and its
/// parameters, the ABI a launch of its slot packs.
struct Kernel {
    symbol: String,
    plan: LaunchPlan,
    params: Vec<EntryParam>,
}

fn kernel_of(entry: &Entry) -> Kernel {
    Kernel {
        symbol: entry.symbol.clone(),
        plan: entry.plan(entry.capacity()).unwrap(),
        params: entry.params.clone(),
    }
}

/// Any part, its parameters unchecked: `named` launches bind what the
/// lowering names.
fn any_part(_: usize, _: &str) -> Result<Vec<&'static str>, String> {
    Ok(Vec::new())
}

/// `tower` lowered on its SM120 catalog, with the catalog entry of every
/// slot, and lowered with stand-in ops that keep each launch's operands by
/// the names the lowering gives them.
struct Both {
    ops: Lowered,
    named: Lowered,
    kernels: BTreeMap<SlotKey, Kernel>,
}

fn both(tower: &Declaration, plan: &BufferPlan) -> Both {
    let slots = catalogs::slots(tower, &catalogs::Shared).unwrap();
    let catalog = catalogs::catalog(tower, &catalogs::Shared).unwrap();
    let entry = |stage, part, block| {
        let slot = catalogs::Slot { stage, part, block };
        slots
            .iter()
            .find(|(known, _)| *known == slot)
            .map(|(_, entry)| entry)
            .ok_or_else(|| format!("no slot {slot:?}"))
    };
    let query = |stage, part, block| Ok(SlotOp::Query(entry(stage, part, block)?.query));
    let backend = Sm120::new().with_encoders(catalog.clone());
    let ops = Lowered::with_backend(tower, plan, &backend, &query, &Shared, 8192).unwrap();
    let layout = (tower.stages.len(), "layout", None);
    let kernels = ops
        .prepared
        .iter()
        .map(|(slot, _)| *slot)
        .chain([layout])
        .map(|slot @ (stage, part, block)| {
            let symbol = &entry(stage, part, block).unwrap().symbol;
            let entry = catalog.iter().find(|known| known.symbol == *symbol);
            (slot, kernel_of(entry.unwrap()))
        })
        .collect();
    Both {
        named: super::tests::lower(tower, plan, any_part, &Shared).unwrap(),
        ops,
        kernels,
    }
}

/// One handle per distinct operand.
#[derive(Default)]
struct Handles(RefCell<Vec<Operand>>);

impl Handles {
    fn of(&self, operand: &Operand) -> u64 {
        let mut known = self.0.borrow_mut();
        let at = match known.iter().position(|seen| seen == operand) {
            Some(at) => at,
            None => {
                known.push(operand.clone());
                known.len() - 1
            }
        };
        at as u64 + 1
    }
}

/// The launch of `kernel` with `named` bound to its parameters by name, as
/// its catalog entry's ABI packs it for `rows` rows and `sequences`
/// sequences.
fn packed(
    handles: &Handles,
    kernel: &Kernel,
    named: &Named<Operand>,
    (rows, sequences): (u32, u32),
) -> (String, u32, LaunchPlan, ArgumentList) {
    assert_eq!(named.len(), kernel.params.len(), "{}", kernel.symbol);
    let values = kernel
        .params
        .iter()
        .map(|param| {
            let name = match param {
                EntryParam::DevicePointer { name, .. }
                | EntryParam::U32 { name }
                | EntryParam::U64 { name } => name,
                other => panic!("{other:?}"),
            };
            let operand = named
                .iter()
                .find(|(known, _)| known == name)
                .map(|(_, operand)| operand)
                .unwrap_or_else(|| panic!("{}: nothing binds `{name}`", kernel.symbol));
            match (operand, param) {
                (Operand::Null, EntryParam::DevicePointer { .. }) => Argument::NullDevicePointer,
                (operand, EntryParam::DevicePointer { .. }) => {
                    Argument::DeviceBuffer(handles.of(operand))
                }
                (Operand::Sequences(_), EntryParam::U32 { .. }) => Argument::U32(sequences),
                (Operand::U32(value), EntryParam::U32 { .. }) => Argument::U32(*value),
                other => panic!("{other:?}"),
            }
        })
        .collect();
    (
        kernel.symbol.clone(),
        rows,
        kernel.plan,
        ArgumentList(values),
    )
}

/// Every launch of `both`'s backend lowering launches its slot's catalog
/// entry, at the plan and with the arguments the entry's ABI packs from the
/// operands the lowering names, at each of `counts(full)` (rows,
/// sequences), its operands bound through `view`.
fn assert_launches_equal(
    both: &Both,
    handles: &Handles,
    counts: &dyn Fn(u32) -> Vec<(u32, u32)>,
    view: &dyn Fn(&Operand) -> Result<View, String>,
) {
    let Both {
        ops,
        named,
        kernels,
    } = both;
    assert_eq!(named.launches.len(), ops.launches.len());
    assert_eq!(named.counts, ops.counts);
    for (index, (named_launch, op_launch)) in named.launches.iter().zip(&ops.launches).enumerate() {
        assert_eq!(named_launch.count, op_launch.count);
        let (slot, ready) = &ops.prepared[op_launch.op];
        let kernel = &kernels[slot];
        let full = ops.counts[op_launch.count] as u32;
        for counts in counts(full) {
            let expected = packed(handles, kernel, super::tests::named(named, index), counts);
            let launches = ready
                .launches(&op_launch.roles, counts, &mut |operand| view(operand))
                .unwrap_or_else(|error| panic!("{} at {counts:?}: {error}", kernel.symbol));
            let [launch] = &launches[..] else {
                panic!("{} launches", launches.len());
            };
            assert_eq!(
                (
                    launch.symbol.clone(),
                    launch.rows,
                    launch.plan,
                    launch.arguments.clone()
                ),
                expected,
                "{}",
                kernel.symbol
            );
            assert!(!launch.collective);
        }
    }
}

/// The override of `both` launches the layout's catalog entry over the
/// regions at each of `rows`, its views `bytes(role, rows)` long
/// (activation, arena, source, live count).
fn assert_overrides_equal(
    both: &Both,
    handles: &Handles,
    rows: &[u32],
    bytes: &dyn Fn(usize, u32) -> u64,
) {
    let layout = (both.ops.declaration.stages.len(), "layout", None);
    let kernel = &both.kernels[&layout];
    let regions = [
        Operand::Region("embed"),
        Operand::Region("arena"),
        Operand::Region("source-rows"),
        Operand::Region("live"),
    ];
    for &rows in rows {
        let view = |at: usize| View {
            handle: handles.of(&regions[at]),
            bytes: bytes(at, rows),
        };
        let buffers = EmbeddingOverrideBuffers {
            activation: view(0),
            arena: view(1),
            source: view(2),
            live_rows: view(3),
            arena_rows: 1024,
        };
        let launch = both
            .ops
            .inject
            .launches(&buffers, View::EMPTY, Counts::rows(rows))
            .unwrap();
        let mut values = regions
            .iter()
            .map(|region| Argument::DeviceBuffer(handles.of(region)))
            .collect::<Vec<_>>();
        values.push(Argument::U64(1024));
        assert_eq!(
            (
                launch[0].symbol.as_str(),
                launch[0].rows,
                launch[0].plan,
                &launch[0].arguments
            ),
            (
                kernel.symbol.as_str(),
                rows,
                kernel.plan,
                &ArgumentList(values)
            )
        );
    }
}

/// Every launch of every tower prepares on the entry its slot resolved
/// to, and launches its symbol, plan and arguments at several row counts.
#[test]
fn backend_launches_pack_their_catalog_entries() {
    for (tower, plan) in towers() {
        let both = both(&tower, &plan);
        let handles = Handles::default();
        let view = |operand: &Operand| {
            Ok(View {
                handle: handles.of(operand),
                bytes: u64::MAX,
            })
        };
        let counts = |full| vec![(1, 1), (37, 1), (full, 1)];
        assert_launches_equal(&both, &handles, &counts, &view);
        assert_overrides_equal(&both, &handles, &[1, 64, 8192], &|_, _| u64::MAX);
    }
}

/// An image tower at serving scale: 16-pixel patches of up to 16M-pixel
/// images, 65536 patch rows, two 1152-wide blocks of 16 heads of 72 with
/// a padded GELU MLP, a 48 x 48 learned position grid and a 2 x 2 merge
/// into 2560-wide rows.
const IMAGE_65536: &str = r#"{
    "schema": "chopinfer.encoder.v1", "model": "alpha", "name": "alpha-image",
    "precision": "f32-pairs.v1",
    "checkpoint": {"prefix": "visual.",
        "config": [{"file": "config.json", "pointer": "/vision_config", "values": {"depth": 2}}]},
    "input": {"recipes": [{"tag": "image.patch-grid.v1", "patch": 16, "temporal_patch": 2,
                           "merge": 2, "maximum_pixels": 16777216}],
              "patchify": "merge-blocks.v1", "payload": "f32", "columns": 1536,
              "capacity_rows": 65536},
    "sequences": {"kind": "item"},
    "image_layout": "view-order.v1",
    "stages": [
        {"op": "linear.pairs.v1", "out": 1152, "bias": true,
         "weights": {"weight": {"tensor": "patch.weight", "transforms": ["flatten"]},
                     "bias": "patch.bias"}},
        {"op": "position.learned-grid.bilinear-align-corners.f32.v1", "table": [48, 48],
         "weights": {"table": "pos.weight"}},
        {"op": "transformer.pre-norm.v1", "blocks": 2, "width": 1152,
         "norm": {"op": "norm.layer-f32.v1", "eps": 1e-6},
         "attention": {"op": "attention.varlen.pairs.v1", "q_heads": 16, "kv_heads": 16,
                       "head_dim": 72, "qkv_bias": true, "proj_bias": true, "mask": "full"},
         "rotary": {"op": "rotary.axial-2d.v1", "pairs_per_axis": 18, "theta": 10000,
                    "style": "neox", "table": "table.hf-f32-angle.v1"},
         "mlp": {"op": "mlp.gelu.v1", "hidden": 4304, "pad_to": 4352, "gelu": "tanh",
                 "bias": true},
         "weights": {
            "norm1.weight": "b.{i}.n1.weight", "norm1.bias": "b.{i}.n1.bias",
            "qkv.weight": "b.{i}.qkv.weight", "qkv.bias": "b.{i}.qkv.bias",
            "proj.weight": "b.{i}.proj.weight", "proj.bias": "b.{i}.proj.bias",
            "norm2.weight": "b.{i}.n2.weight", "norm2.bias": "b.{i}.n2.bias",
            "up.weight": {"tensor": "b.{i}.up.weight", "transforms": [{"pad_rows": 4352}]},
            "up.bias": {"tensor": "b.{i}.up.bias", "transforms": [{"pad_rows": 4352}]},
            "down.weight": {"tensor": "b.{i}.down.weight", "transforms": [{"pad_cols": 4352}]},
            "down.bias": "b.{i}.down.bias"}},
        {"op": "norm.layer-f32.v1", "eps": 1e-6,
         "weights": {"weight": "m.n.weight", "bias": "m.n.bias"}},
        {"op": "merge.consecutive.v1", "rows": 4},
        {"op": "linear.pairs.v1", "out": 4608, "bias": true, "gelu": "erf",
         "weights": {"weight": "m.fc1.weight", "bias": "m.fc1.bias"}},
        {"op": "linear.pairs.v1", "out": 2560, "bias": true, "output": "bf16",
         "weights": {"weight": "m.fc2.weight", "bias": "m.fc2.bias"}}
    ],
    "output": {"op": "layout.tokens.v1", "hidden": 2560, "positions": "grid-3axis.v1",
               "media_mask": false},
    "parallel": ["replicated"]
}"#;

/// An audio tower: 128 log-mel columns of up to 60001 frames in segments
/// of 6000, two convolutions (the second at stride 2), two causal blocks
/// (the even one windowed) with 1D rotary and a carry, a stride-2
/// down-sampling convolution, the segments joined in groups of 4 rows
/// (the last repeated) and a 4-row merge into 2048-wide rows.
const AUDIO_CONV: &str = r#"{
    "schema": "chopinfer.encoder.v1", "model": "alpha", "name": "alpha-audio",
    "precision": "f32-pairs.v1",
    "checkpoint": {"prefix": "audio.",
        "config": [{"file": "config.json", "pointer": "/audio_config", "values": {"layers": 2}}]},
    "input": {"recipes": [{"tag": "audio.log-mel.v1", "mel_bins": 128, "maximum_frames": 60001}],
              "payload": "f32", "columns": 128, "capacity_rows": 8192,
              "chunking": "whole-sequences.v1"},
    "sequences": {"kind": "segments", "input_rows": 6000},
    "image_layout": "view-order.v1",
    "stages": [
        {"op": "conv1d.im2col.v1", "kernel": 3, "stride": 1, "pad": [1, 1], "out": 1024,
         "bias": true, "gelu": "erf",
         "weights": {"weight": {"tensor": "c1.weight", "transforms": ["taps_first"]},
                     "bias": "c1.bias"}},
        {"op": "conv1d.im2col.v1", "kernel": 3, "stride": 2, "pad": [1, 1], "out": 1024,
         "bias": true, "gelu": "erf",
         "weights": {"weight": {"tensor": "c2.weight", "transforms": ["taps_first"]},
                     "bias": "c2.bias"}},
        {"op": "transformer.pre-norm.v1", "blocks": 2, "width": 1024,
         "norm": {"op": "norm.layer-f32.v1", "eps": 1e-5},
         "attention": {"op": "attention.varlen.pairs.v1", "q_heads": 16, "kv_heads": 16,
                       "head_dim": 64, "qkv_bias": true, "proj_bias": true, "mask": "causal"},
         "overrides": [{"blocks": "even", "attention": {"mask": {"causal_window": 128}}}],
         "rotary": {"op": "rotary.1d.v1", "pairs": 32, "theta": 10000, "style": "neox",
                    "table": "table.hf-f32-angle.v1", "positions": "within-sequence"},
         "carry": [{"save_after": 0, "add_after": 1}],
         "mlp": {"op": "mlp.gelu.v1", "hidden": 4096, "gelu": "erf", "bias": true},
         "weights": {
            "norm1.weight": "b.{i}.n1.weight", "norm1.bias": "b.{i}.n1.bias",
            "qkv.weight": "b.{i}.qkv.weight", "qkv.bias": "b.{i}.qkv.bias",
            "proj.weight": "b.{i}.proj.weight", "proj.bias": "b.{i}.proj.bias",
            "norm2.weight": "b.{i}.n2.weight", "norm2.bias": "b.{i}.n2.bias",
            "up.weight": "b.{i}.up.weight", "up.bias": "b.{i}.up.bias",
            "down.weight": "b.{i}.down.weight", "down.bias": "b.{i}.down.bias"}},
        {"op": "norm.layer-f32.v1", "eps": 1e-5,
         "weights": {"weight": "n.weight", "bias": "n.bias"}},
        {"op": "conv1d.im2col.v1", "kernel": 2, "stride": 2, "pad": [0, 1], "out": 1024,
         "bias": false, "gelu": "erf",
         "weights": {"weight": {"tensor": "down.weight", "transforms": ["taps_first"]}}},
        {"op": "norm.layer-f32.v1", "eps": 1e-5,
         "weights": {"weight": "down_n.weight", "bias": "down_n.bias"}},
        {"op": "group.repeat-last.v1", "rows": 4},
        {"op": "merge.consecutive.v1", "rows": 4},
        {"op": "linear.pairs.v1", "out": 4096, "bias": false, "gelu": "erf",
         "weights": {"weight": "p.fc1.weight"}},
        {"op": "linear.pairs.v1", "out": 2048, "bias": false, "output": "bf16",
         "weights": {"weight": "p.fc2.weight"}}
    ],
    "output": {"op": "layout.tokens.v1", "hidden": 2048, "positions": "sequential",
               "media_mask": false},
    "parallel": ["replicated"]
}"#;

/// The bytes of each image view of a `view-order.v1` tower: every role
/// its stages read, at the shape and element the stage takes.
fn image_views(tower: &Declaration) -> BTreeMap<String, u64> {
    let context = Context {
        model: &tower.model,
        precision: tower.precision,
        patchify: tower.input.patchify,
        sequences: &tower.sequences,
        capacity_rows: u64::from(tower.input.capacity_rows),
    };
    let mut views = BTreeMap::new();
    for (stage, input) in tower.stages.iter().zip(tower.inputs().unwrap()) {
        let component = stage.op.component().unwrap();
        for role in component.roles(&context, &input).unwrap() {
            let Source::Tensor(tensor) = &stage.weights[&role.name] else {
                panic!("{}: not a single tensor", role.name);
            };
            let blocks = match (&role.blocks, component.blocks()) {
                (Some(blocks), _) => blocks.clone(),
                (None, Some(blocks)) => (0..blocks).collect(),
                (None, None) => vec![0],
            };
            let bytes = role.shape.iter().product::<u64>() * role.element.bytes();
            for block in blocks {
                let name = tensor.tensor.replace("{i}", &block.to_string());
                views.insert(format!("{}{name}", tower.checkpoint.prefix), bytes);
            }
        }
    }
    views
}

/// Representative towers launch their catalog entries' symbols, plans and
/// arguments with every view bounded as a run over the whole tower binds it:
/// regions at their planned bytes, image views at their roles' shapes,
/// the output at the last stage's rows; at one, some and all rows, with
/// one and the most sequences the plan holds (one per image, several
/// segments of audio). The override's views hold
/// exactly its rows.
#[test]
fn bounded_views_of_serving_scale_towers_launch_their_catalog_entries() {
    let serving = Serving {
        arena_rows: 16384,
        prefill_rows: 8192,
        staging_bytes: 1 << 20,
    };
    let mut most = 0;
    for text in [IMAGE_65536, AUDIO_CONV] {
        let tower = parse(text, &Environment::new("alpha")).unwrap();
        let capacity = u64::from(tower.input.capacity_rows);
        let plan = buffers::plan(&[&tower], capacity, serving).unwrap();
        let both = both(&tower, &plan);
        let ops = &both.ops;
        let images = image_views(&tower);
        // Each live image view is one the image holds, at its size.
        let spans = images
            .iter()
            .map(|(name, &bytes)| {
                let handle = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
                (name.clone(), Span::new(handle, 0, bytes))
            })
            .collect::<BTreeMap<_, _>>();
        ops.check(&spans).unwrap();
        let region = |name: &str, owner| {
            plan.owned(name, owner)
                .map(|region| region.bytes)
                .ok_or_else(|| format!("no region {name} of {owner:?}"))
        };
        let handles = Handles::default();
        let view = |operand: &Operand| {
            let bytes = match operand {
                Operand::Region(name) => region(name, None)?,
                Operand::Side(name, stage) => region(name, Some(*stage))?,
                Operand::Output => ops.output.1,
                Operand::Count(_) => 4,
                Operand::Weight(name) => *images
                    .get(name)
                    .ok_or_else(|| format!("no image view {name}"))?,
                Operand::Payload => region(ops.payload, None)?,
                Operand::RunOutput(name, _) => region(name, None)?,
                other => return Err(format!("{other:?} binds no view")),
            };
            Ok(View {
                handle: handles.of(operand),
                bytes,
            })
        };
        // The most sequences a `sequences` region holds (one bound more).
        let sequences = plan
            .regions
            .iter()
            .filter(|region| region.name == "sequences")
            .map(|region| region.bytes / 4 - 1)
            .min()
            .unwrap() as u32;
        most = most.max(sequences);
        let counts = |full: u32| {
            vec![
                (1, 1),
                (37, sequences.min(3)),
                (full, 1),
                (full, sequences.min(full)),
            ]
        };
        assert_launches_equal(&both, &handles, &counts, &view);
        let row_bytes = ops.row_bytes();
        let arena = region("arena", None).unwrap();
        let bytes = |at: usize, rows: u32| match at {
            0 => u64::from(rows) * row_bytes,
            1 => arena,
            2 => u64::from(rows) * 8,
            _ => 4,
        };
        assert_overrides_equal(&both, &handles, &[1, 64, 8192], &bytes);
    }
    // An image is one sequence; the audio's segments are several.
    assert!(most > 3, "{most} sequences");
}

/// A view an image holds too small for its stage fails the create-time
/// check, which takes the launches at their full rows.
#[test]
fn the_create_check_refuses_an_image_view_too_small() {
    let (tower, plan) = towers().swap_remove(0);
    let ops = both(&tower, &plan).ops;
    let span = |bytes| {
        Span::new(
            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1),
            0,
            bytes,
        )
    };
    let mut weights = BTreeMap::new();
    for launch in &ops.launches {
        for operand in launch.operands() {
            if let Operand::Weight(name) = operand {
                weights.insert(name, span(1 << 30));
            }
        }
    }
    assert!(ops.check(&weights).is_ok());
    // The first block's qkv weight: 768 x 256 BF16.
    weights.insert("visual.b.0.qkv.weight".into(), span(768 * 256 * 2 - 2));
    let refused = ops.check(&weights).unwrap_err();
    assert!(refused.contains("qkv"), "{refused}");
    weights.insert("visual.b.0.qkv.weight".into(), span(768 * 256 * 2));
    assert!(ops.check(&weights).is_ok());
}

/// A slot whose query is another stage's, or whose runtime values are not
/// the declaration's, is refused at lowering.
#[test]
fn slots_of_another_stage_or_other_values_are_refused() {
    let (tower, plan) = towers().swap_remove(0);
    let slots = catalogs::slots(&tower, &catalogs::Shared).unwrap();
    let catalog = catalogs::catalog(&tower, &catalogs::Shared).unwrap();
    let backend = Sm120::new().with_encoders(catalog);
    let query = |stage, part, block| {
        let slot = catalogs::Slot { stage, part, block };
        slots
            .iter()
            .find(|(known, _)| *known == slot)
            .map(|(_, entry)| entry.query)
            .ok_or_else(|| format!("no slot {slot:?}"))
    };
    let norm = query(3, "norm", None).unwrap();
    let position = query(1, "position", None).unwrap();
    // The qkv GEMM resolved to the position add, a row pass.
    let swapped = |stage, part, block| match part {
        "qkv" => Ok(SlotOp::Query(position)),
        _ => query(stage, part, block).map(SlotOp::Query),
    };
    let refused = Lowered::with_backend(&tower, &plan, &backend, &swapped, &Shared, 8192)
        .err()
        .unwrap();
    assert!(refused.contains("nothing binds `input`"), "{refused}");
    // The merger's norm at another eps.
    let MultimodalQuery::LayerNormF32 {
        width, capacity, ..
    } = norm
    else {
        panic!("{norm:?}");
    };
    let other = MultimodalQuery::LayerNormF32 {
        width,
        eps_bits: 1e-5_f32.to_bits(),
        capacity,
    };
    let eps = |stage, part, block| match (stage, part) {
        (3, "norm") => Ok(SlotOp::Query(other)),
        _ => query(stage, part, block).map(SlotOp::Query),
    };
    let refused = Lowered::with_backend(&tower, &plan, &backend, &eps, &Shared, 8192)
        .err()
        .unwrap();
    assert!(
        refused.contains("`eps_bits` is not the declaration's"),
        "{refused}"
    );
    // An override past the backend's rows.
    let refused = Lowered::with_backend(
        &tower,
        &plan,
        &backend,
        &|stage, part, block| query(stage, part, block).map(SlotOp::Query),
        &Shared,
        8193,
    )
    .err()
    .unwrap();
    assert!(refused.contains("override"), "{refused}");
}

/// A model's own op: its buffers by the names its registered stage gives.
struct Echo(OpInfo);

impl PreparedOp<Named<View>> for Echo {
    fn info(&self) -> &OpInfo {
        &self.0
    }

    fn launches(
        &self,
        views: &Named<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<BackendLaunch>, kernel_core::contract::Refusal> {
        Ok(vec![BackendLaunch {
            symbol: views
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(","),
            rows: counts.rows,
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            },
            arguments: ArgumentList(
                views
                    .iter()
                    .map(|(_, view)| Argument::DeviceBuffer(view.handle))
                    .collect(),
            ),
            collective: false,
            persistent: false,
        }])
    }
}

#[test]
fn own_ops_take_their_operands_by_name() {
    let info = OpInfo::new("test.echo.v1", Invariance::NONE, vec![("echo".into(), 0)]);
    let op = Ready::Own(Box::new(Echo(info)));
    let named = vec![
        ("input", Operand::Region("a")),
        ("codes", Operand::Weight("@codes".into())),
        ("live_rows", Operand::Count(2)),
    ];
    let roles = roles(&op, 4, named.clone()).unwrap();
    let handles = Handles::default();
    let launches = op
        .launches(&roles, (9, 0), &mut |operand| {
            Ok(View {
                handle: handles.of(operand),
                bytes: 4,
            })
        })
        .unwrap();
    assert_eq!(launches[0].symbol, "input,codes,live_rows");
    assert_eq!(launches[0].rows, 9);
    assert_eq!(
        launches[0].arguments,
        ArgumentList(
            named
                .iter()
                .map(|(_, operand)| Argument::DeviceBuffer(handles.of(operand)))
                .collect()
        )
    );
    // A launch whose roles are another record's is refused.
    assert!(
        Ready::Own(Box::new(Echo(OpInfo::new(
            "x",
            Invariance::NONE,
            Vec::new()
        ))))
        .launches(
            &Roles::Gemm(EncoderGemmBuffers {
                activation: Operand::Output,
                weight: Operand::Output,
                bias: None,
                output: Operand::Output,
                live_rows: Operand::Output,
            }),
            (1, 0),
            &mut |_| Ok(View::EMPTY)
        )
        .is_err()
    );
}
