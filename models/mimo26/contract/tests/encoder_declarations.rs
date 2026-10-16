//! MiMo's encoder declarations against the code that implements them today:
//! the contract's tensor list, the pinned checkpoint config, the frontend
//! records, and the bundle's audio geometry.

use std::collections::BTreeSet;

use encoder_spec::components::{Boundary, Component, Context, Item, Typed};
use encoder_spec::policy::{self, ParallelClass, Precision};
use encoder_spec::registry::{self, Decoder, Provider, Registration};
use encoder_spec::weights::{Role, TensorInfo};
use encoder_spec::{Declaration, Environment, parse};
use model_mimo26_contract::encoders::expected_flash;
use model_mimo26_contract::encoders::spec;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const VISION_DIGEST: &str = "a04f4c6e77098207d1a68406211f60f532dceb338f499af869c9906d5d1f99ee";
const AUDIO_DIGEST: &str = "d1dfdf4c531cccb51d66cb889dfaba310e831a7b25ce6f13ef398c9bbfdbe8c0";

fn environment() -> Environment {
    spec::environment().unwrap()
}

fn vision() -> Declaration {
    parse(spec::VISION, &environment()).unwrap()
}

fn audio() -> Declaration {
    parse(spec::AUDIO, &environment()).unwrap()
}

fn edited(base: &str, edit: impl FnOnce(&mut Value)) -> String {
    let mut value: Value = serde_json::from_str(base).unwrap();
    edit(&mut value);
    value.to_string()
}

#[track_caller]
fn refuses(base: &str, edit: impl FnOnce(&mut Value), needle: &str) {
    let error = parse(&edited(base, edit), &environment()).expect_err("accepted the edit");
    assert!(error.0.contains(needle), "expected `{needle}` in: {error}");
}

fn lookup(name: &str) -> Option<TensorInfo> {
    expected_flash()
        .get(name)
        .map(|(dtype, shape, file)| TensorInfo {
            dtype: (*dtype).to_owned(),
            shape: shape.clone(),
            file: Some((*file).to_owned()),
        })
}

#[test]
fn declarations_keep_their_digests() {
    assert_eq!(vision().digest(), VISION_DIGEST);
    assert_eq!(audio().digest(), AUDIO_DIGEST);
    for declaration in [vision(), audio()] {
        let canonical = parse(&declaration.canonical_json(), &environment()).unwrap();
        assert_eq!(canonical, declaration);
    }
}

#[test]
fn declarations_bind_to_the_contracts_tensors() {
    let (vision, audio) = (vision(), audio());
    vision.bind(&lookup).unwrap();
    audio.bind(&lookup).unwrap();
    let all: BTreeSet<String> = expected_flash().into_keys().collect();
    let (vision, audio) = (vision.tensors(), audio.tensors());
    assert!(vision.is_disjoint(&audio));
    assert_eq!(vision.union(&audio).cloned().collect::<BTreeSet<_>>(), all);
}

#[test]
fn pins_hold_against_the_checkpoint_config() {
    let config: Value = serde_json::from_str(include_str!(
        "../../compiler/src/encoders/checkpoint_config.json"
    ))
    .unwrap();
    vision().check_config("config.json", &config).unwrap();
    audio().check_config("config.json", &config).unwrap();
    audio()
        .check_config(
            "audio_tokenizer/config.json",
            &config["audio_tokenizer_config"],
        )
        .unwrap();
}

#[test]
fn declarations_agree_with_the_frontend() {
    let frontend: Value =
        serde_json::from_str(include_str!("../../compiler/src/frontend.json")).unwrap();
    let multimodal = &frontend["multimodal"];
    let modality = |kind: &str| {
        multimodal["modalities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["kind"] == kind)
            .unwrap()["recipe"]["parameters"]
            .clone()
    };
    for (index, declaration) in [vision(), audio()].iter().enumerate() {
        let record = &multimodal["encoders"][index];
        assert_eq!(record["name"], declaration.name.as_str());
        assert_eq!(record["payload_columns"], declaration.input.columns);
        assert_eq!(record["payload_element"], json!(declaration.input.payload));
        assert_eq!(
            multimodal["arena"]["hidden"],
            declaration.output.op.params().hidden
        );
    }
    let recipes = serde_json::to_value(&vision().input.recipes).unwrap();
    let image = modality("image");
    for key in ["patch", "temporal_patch", "merge"] {
        assert_eq!(recipes[0][key], image[key]);
    }
    assert_eq!(recipes[0]["maximum_pixels"], image["maximum_pixels"]);
    let video = modality("video");
    assert_eq!(recipes[1]["maximum_frames"], video["maximum_frames"]);
    assert_eq!(
        recipes[1]["maximum_pixels_per_frame"],
        video["maximum_pixels_per_frame"]
    );
    for key in ["patch", "temporal_patch", "merge"] {
        assert_eq!(recipes[1][key], video["image"][key]);
    }
    let sound = modality("audio");
    let recipe = serde_json::to_value(&audio().input.recipes[0]).unwrap();
    assert_eq!(recipe["mel_bins"], sound["mel_bins"]);
    // Centered STFT frames: one per hop, plus one.
    let samples =
        sound["maximum_seconds"].as_u64().unwrap() * sound["sample_rate"].as_u64().unwrap();
    assert_eq!(
        recipe["maximum_frames"],
        samples / sound["hop"].as_u64().unwrap() + 1
    );
    assert_eq!(
        audio().sequences,
        encoder_spec::Sequences::Segments {
            input_rows: sound["segment_frames"].as_u64().unwrap() as u32
        }
    );
}

/// Independent reference: 6,000-frame segments, two stride-2
/// halvings per segment, codes joined, then padded to whole groups of 4.
fn reference_audio(frames: u64) -> (u64, u64, u64, u64) {
    let (mut attention, mut codes) = (0, 0);
    let mut first = 0;
    while first < frames {
        let count = (frames - first).min(6000);
        attention += count.div_ceil(2);
        codes += count.div_ceil(2).div_ceil(2);
        first += count;
    }
    (attention, codes, codes.div_ceil(4) * 4, codes.div_ceil(4))
}

#[test]
fn audio_tail_groups_repeat_the_last_code() {
    let audio = audio();
    let rows = |frames: u64| -> Vec<u64> {
        audio
            .trace(Item {
                rows: frames,
                grid: None,
            })
            .unwrap()
            .iter()
            .map(|item| item.rows)
            .collect()
    };
    // stages: conv1, conv2, tokenizer, norm, downsample, norm, rvq, group,
    // embed, local transformer, norm, merge, two projections, output.
    assert_eq!(rows(17), [17, 9, 9, 9, 5, 5, 5, 8, 8, 8, 8, 2, 2, 2, 2]);
    assert_eq!(rows(6001)[1], 3001);
    assert_eq!(rows(6001)[4], 1501);
    assert_eq!(rows(6001)[7], 1504);
    assert_eq!(rows(6001)[14], 376);
    for frames in (1..=13_000).chain([60_000, 60_001]) {
        let rows = rows(frames);
        let (attention, codes, padded, output) = reference_audio(frames);
        assert_eq!(
            (rows[1], rows[4], rows[7], rows[14]),
            (attention, codes, padded, output),
            "{frames} frames"
        );
    }
    let encoder_spec::components::StageOp::Group(group) = &audio.stages[7].op else {
        panic!("stage 7 is the group op");
    };
    assert_eq!(group.sources(5), [0, 1, 2, 3, 4, 4, 4, 4]);
}

#[test]
fn vision_rows_follow_the_grid() {
    let trace = vision()
        .trace(Item {
            rows: 2 * 32 * 32,
            grid: Some([2, 32, 32]),
        })
        .unwrap();
    assert_eq!(trace.last().unwrap().rows, 2 * 16 * 16);
}

#[test]
fn audio_refusals() {
    refuses(
        spec::AUDIO,
        |d| {
            d["stages"].as_array_mut().unwrap().remove(7);
        },
        "groups of 4 rows need joined items",
    );
    refuses(
        spec::AUDIO,
        |d| {
            let stages = d["stages"].as_array_mut().unwrap();
            stages.remove(7);
            stages.remove(8);
        },
        "merges whole items; join the segments first",
    );
    refuses(
        spec::AUDIO,
        |d| d["stages"][8]["vocab"] = 1.into(),
        "code 0 reaches 1024 entries but each table has 1 rows",
    );
    refuses(
        spec::AUDIO,
        |d| d["stages"][8]["weights"]["tables"]["count"] = 2.into(),
        "stacks 2 tensors where the role stacks 20",
    );
    refuses(
        spec::AUDIO,
        |d| d["stages"][6]["codebooks"][3] = 0.into(),
        "20 non-empty codebooks",
    );
    refuses(
        spec::AUDIO,
        |d| {
            d["output"] = json!({"op": "layout.learned-delimiters.v1", "hidden": 4096, "positions": "sequential",
                                 "media_mask": false, "weights": {"start": "s", "end": "e", "newline": "n"}})
        },
        "this layout needs rows that follow a grid",
    );
    refuses(
        spec::AUDIO,
        |d| {
            d["checkpoint"]["files"]
                .as_object_mut()
                .unwrap()
                .remove("speech_embeddings.");
        },
        "no file holds `speech_embeddings.0.weight`",
    );
    refuses(
        spec::AUDIO,
        |d| d["parallel"] = json!(["replicated", "sequence-split"]),
        "sequence-split needs",
    );
    refuses(
        spec::AUDIO,
        |d| d["stages"][6]["op"] = "mimo26:quantize.rvq21.v1".into(),
        "`mimo26:quantize.rvq21.v1` is not registered by a provider of model mimo26",
    );
    refuses(
        spec::AUDIO,
        |d| d["stages"][2]["rotary"]["theta"] = 1e39.into(),
        "theta 1e39 is inf in F32",
    );
    refuses(
        spec::AUDIO,
        |d| d["stages"][6]["level"] = 1.into(),
        "unknown field `level`",
    );

    // The missing k_proj bias is 1024 zeros; one zero leaves the q/k/v
    // bias 2049 long, which only the checkpoint's shapes reveal.
    let short = edited(spec::AUDIO, |d| {
        d["stages"][2]["weights"]["qkv.bias"]["concat_rows"][1] = json!({"zeros": 1})
    });
    let short: Declaration = {
        let mut declaration: Declaration = serde_json::from_str(&short).unwrap();
        declaration.resolve(&environment()).unwrap();
        declaration
    };
    let error = short.bind(&lookup).unwrap_err();
    assert!(
        error
            .0
            .contains("weights.qkv.bias block 0: gives shape [2049]"),
        "{error}"
    );
}

#[test]
fn vision_refusals() {
    refuses(
        spec::VISION,
        |d| d["stages"][1]["weights"]["sinks"]["blocks"] = "1-8,10-17".into(),
        "is not the blocks that have it",
    );
    refuses(
        spec::VISION,
        |d| {
            d["stages"][1]["weights"]["sinks"]
                .as_object_mut()
                .unwrap()
                .remove("blocks");
        },
        "some blocks have none",
    );
    refuses(
        spec::VISION,
        |d| d["stages"][1]["overrides"][0]["blocks"] = "0,9,18,28".into(),
        "past the last of 28",
    );
    refuses(
        spec::VISION,
        |d| d["stages"][1]["reorder"][0]["blocks"] = "5-8,8-9".into(),
        "block set `5-8,8-9`",
    );
    refuses(
        spec::VISION,
        |d| d["parallel"] = json!(["replicated", "sequence-split"]),
        "sequence-split needs",
    );
    refuses(
        spec::VISION,
        |d| d["stages"][0]["op"] = "linear.bf16.v1".into(),
        "not defined under precision policy f32-pairs.v1",
    );
    refuses(
        spec::VISION,
        |d| {
            d["stages"][1]["rotary"] = json!({"op": "mimo26:rotary.axial-2d-f32-powf.v1",
                "pairs_per_axis": 16, "theta": 10000, "style": "neox"})
        },
        "not defined under precision policy f32-pairs.v1",
    );
    refuses(
        spec::VISION,
        |d| d["stages"][1]["mlp"]["round_linear"] = true.into(),
        "round_linear is allowed only under bf16-legacy.v1",
    );
    refuses(
        spec::VISION,
        |d| d["input"]["capacity_rows"] = 4095.into(),
        "capacity_rows 4095 is not a multiple of 1024 holding the 4096-row sequence",
    );
    refuses(
        spec::VISION,
        |d| d["input"]["capacity_rows"] = 3072.into(),
        "holding the 4096-row sequence",
    );
}

/// Confirm review N1: binding checks each stacked part, not only the
/// total. The kernels index each table and codebook separately.
#[test]
fn stacked_parts_bind_one_by_one() {
    let audio = audio();
    let embeddings = audio.bind(&|name| {
        let mut info = lookup(name)?;
        match name {
            "speech_embeddings.0.weight" => info.shape[0] = 1,
            "speech_embeddings.1.weight" => info.shape[0] = 2559,
            _ => {}
        }
        Some(info)
    });
    let error = embeddings.unwrap_err();
    assert!(
        error.0.contains(
            "weights.tables block 0: part 0 gives shape [1, 1024], the role reads [1280, 1024]"
        ),
        "{error}"
    );
    let codebooks = audio.bind(&|name| {
        let mut info = lookup(name)?;
        match name {
            "encoder.quantizer.vq.layers.0._codebook.embed" => info.shape[0] = 1,
            "encoder.quantizer.vq.layers.1._codebook.embed" => info.shape[0] = 2047,
            _ => {}
        }
        Some(info)
    });
    let error = codebooks.unwrap_err();
    assert!(
        error.0.contains(
            "weights.codebooks block 0: part 0 gives shape [1, 1024], the role reads [1024, 1024]"
        ),
        "{error}"
    );
}

/// Confirm review N1/N2: an unsupported stored dtype is refused, also
/// inside a stack.
#[test]
fn weights_bind_with_the_element_the_kernel_reads() {
    let error = audio()
        .bind(&|name| {
            let mut info = lookup(name)?;
            if name == "speech_embeddings.3.weight" {
                info.dtype = "F16".into();
            }
            Some(info)
        })
        .unwrap_err();
    assert!(
        error
            .0
            .contains("weights.tables block 0: part 3 gives F16, the role reads bf16"),
        "{error}"
    );
    // Sinks widen BF16 to the F32 the kernel reads; stored F32 sinks have
    // nothing to widen.
    let error = vision()
        .bind(&|name| {
            let mut info = lookup(name)?;
            if name == "visual.blocks.1.attn.sinks" {
                info.dtype = "F32".into();
            }
            Some(info)
        })
        .unwrap_err();
    assert!(
        error.0.contains("WidenF32 needs a BF16 tensor, not F32"),
        "{error}"
    );
}

/// Confirm review N3: a resolved declaration revalidated in another
/// environment is admitted only if that registry holds the providers that
/// decoded its variants.
#[test]
fn validation_honors_the_supplied_registry() {
    let audio = audio();
    let bare = Environment::new("mimo26");
    let error = audio.validate(&bare).unwrap_err();
    assert!(
        error.0.contains("`mimo26:quantize.rvq20-f32.v1` was decoded by a provider this environment does not register"),
        "{error}"
    );
    // Another provider under the same namespace is not the one that
    // decoded them.
    static IMPOSTOR: Provider = Provider {
        namespace: "mimo26",
        registrations: &[Registration {
            id: "mimo26:quantize.rvq20-f32.v1",
            decoder: Decoder::Stage(registry::stage::<spec::Rvq20>),
        }],
    };
    let mut impostor = Environment::new("mimo26");
    impostor.registry.register(&IMPOSTOR).unwrap();
    let error = audio.validate(&impostor).unwrap_err();
    assert!(
        error
            .0
            .contains("was decoded by a provider this environment does not register"),
        "{error}"
    );
    audio.validate(&environment()).unwrap();
}

#[test]
fn legacy_precision_is_retired() {
    refuses(
        spec::VISION,
        |d| d["precision"] = "bf16-legacy.v1".into(),
        "precision bf16-legacy.v1 is retired; use f32-pairs.v1",
    );
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoParams {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
enum Identity {
    #[serde(rename = "testmodel:cast.identity.v1")]
    V1(NoParams),
}

impl Typed for Identity {
    fn policies(&self) -> &'static [Precision] {
        policy::ANY
    }
}

impl Component for Identity {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, _: &Context, input: &Boundary) -> Result<Boundary, String> {
        Ok(input.clone())
    }

    fn roles(&self, _: &Context, _: &Boundary) -> Result<Vec<Role>, String> {
        Ok(Vec::new())
    }
}

static TEST_PROVIDER: Provider = Provider {
    namespace: "testmodel",
    registrations: &[Registration {
        id: "testmodel:cast.identity.v1",
        decoder: Decoder::Stage(registry::stage::<Identity>),
    }],
};

#[test]
fn registering_providers_changes_no_declaration() {
    let mut extended = environment();
    extended.registry.register(&TEST_PROVIDER).unwrap();
    for text in [spec::VISION, spec::AUDIO] {
        let plain = parse(text, &environment()).unwrap();
        let more = parse(text, &extended).unwrap();
        assert_eq!(more.canonical_json(), plain.canonical_json());
        assert_eq!(more.digest(), plain.digest());
    }
    let mut foreign = Environment::new("other");
    foreign.registry.register(&spec::PROVIDER).unwrap();
    let text = spec::VISION.replace("\"mimo26\"", "\"other\"");
    let plain = parse(&text, &foreign).unwrap();
    foreign.registry.register(&TEST_PROVIDER).unwrap();
    assert_eq!(parse(&text, &foreign).unwrap().digest(), plain.digest());
}

/// The cost model over MiMo's windows, groups and registered stages: full
/// blocks see every pair of a temporal grid, window blocks at most 129 keys
/// per query, the local transformer's groups of 4 sixteen pairs.
#[test]
fn costs_follow_windows_groups_and_registered_stages() {
    use encoder_spec::components::attention::Mask;
    use encoder_spec::cost::visible_pairs;
    use encoder_spec::geometry::Media;

    let vision = vision();
    let geometry = vision
        .geometry("video.timestamped-grids.v1", Media::Grid([3, 32, 32]))
        .unwrap();
    let [run] = geometry.runs.as_slice() else {
        panic!("three 1,024-patch grids fit one run");
    };
    let costs = vision.cost(run).unwrap();
    let (rows, grid) = (3 * 1024, 1024);
    let gemms = 2 * rows * 1280 * (3072 + 2048 + 2 * 4608 + 4608);
    let norms = 2 * 8 * rows * 1280;
    let window = visible_pairs(Mask::Window([64, 64]), grid);
    assert_eq!(window, 1024 * 129 - 2 * (64 * 65 / 2));
    let attention = |pairs: u64| 4 * 32 * 64 * 3 * pairs;
    assert_eq!(
        costs[1].flops,
        28 * (gemms + norms) + 4 * attention(grid * grid) + 24 * attention(window)
    );

    let audio = audio();
    let geometry = audio
        .geometry("audio.log-mel.v1", Media::Frames(1595))
        .unwrap();
    let [segment, group] = geometry.runs.as_slice() else {
        panic!("one segment, one group run");
    };
    let costs = audio.cost(segment).unwrap();
    let codes = 399;
    let entries = 2 * 1024 + 256 + 17 * 128;
    assert_eq!(costs[6].flops, 3 * codes * 1024 * entries);
    let costs = audio.cost(group).unwrap();
    let rows = 400;
    assert_eq!(costs[1].flops, rows * 20 * 1024);
    let gemms = 2 * rows * 1024 * (3 * 1024 + 1024 + 2 * 4096) + 2 * rows * 4096 * 1024;
    let attention = 4 * 16 * 64 * (rows / 4) * 16;
    assert_eq!(
        costs[2].flops,
        6 * (gemms + attention + 2 * 8 * rows * 1024)
    );
}

/// The record admission the bundle applies: a geometry some recipe of the
/// encoder's declaration emits, in the wire's layout.
#[test]
fn records_are_admitted_by_their_declarations_recipes() {
    for geometry in [
        [1, 64, 64, 0],
        [1, 2, 2048, 0],
        [16, 32, 32, 0],
        [9, 2, 512, 0],
    ] {
        assert!(
            spec::admit_record_flash(0, geometry).is_ok(),
            "{geometry:?}"
        );
    }
    for geometry in [[1, 66, 64, 0], [2, 34, 32, 0], [17, 2, 2, 0], [1, 2, 2, 7]] {
        assert!(
            spec::admit_record_flash(0, geometry).is_err(),
            "{geometry:?}"
        );
    }
    assert!(spec::admit_record_flash(1, [60_001, 128, 0, 0]).is_ok());
    for geometry in [
        [60_002, 128, 0, 0],
        [0, 128, 0, 0],
        [10, 64, 0, 0],
        [10, 128, 1, 0],
    ] {
        assert!(
            spec::admit_record_flash(1, geometry).is_err(),
            "{geometry:?}"
        );
    }
    assert!(spec::admit_record_flash(2, [1, 2, 2, 0]).is_err());
}

/// The RVQ estimate against the kernel's distance loop (`rvq_f32`)
/// walked on a tiny case: per row, stage, entry and element, `delta =
/// residual - code` (one FLOP) and `fmaf(delta, delta, distance)` (two).
#[test]
fn rvq_cost_counts_the_distance_loop() {
    use encoder_spec::components::{Boundary, Component, Data, Element};
    use model_mimo26_contract::encoders::spec::{Rvq20, Rvq20Params};

    let (rows, width) = (3_u64, 4_u32);
    let codebooks: Vec<u32> = (1..=20).collect();
    let mut flops = 0;
    for _row in 0..rows {
        for &entries in &codebooks {
            for _code in 0..entries {
                for _col in 0..width {
                    flops += 1; // delta
                    flops += 2; // fmaf
                }
            }
        }
    }
    let rvq = Rvq20::F32(Rvq20Params { codebooks });
    let boundary = |data| Boundary {
        data,
        run_rows: rows,
        multiple: 1,
        grid: false,
        segmented: false,
    };
    let input = boundary(Data::Rows {
        width,
        element: Element::F32,
    });
    let output = boundary(Data::Codes {
        bounds: vec![1; 20],
    });
    let cost = rvq.cost([rows, rows], &input, &output).unwrap().unwrap();
    assert_eq!(cost.flops, flops);
}
