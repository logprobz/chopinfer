//! CPU goldens for the declarations with no bespoke variants: the DeepSeek
//! V4.1 vision fixture binds against its checkpoint's headers, and adding a
//! provider changes no declaration. MiMo's and Qwen3.8's goldens live in
//! their contract crates, beside the code they are checked against.

use std::collections::BTreeMap;
use std::sync::Arc;

use encoder_spec::components::{Boundary, Component, Context, Item, Typed};
use encoder_spec::policy::{self, ParallelClass, Precision};
use encoder_spec::registry::{self, Decoder, Provider, Registration};
use encoder_spec::weights::{Role, TensorInfo};
use encoder_spec::{Declaration, Environment, parse};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const QWEN: &str = include_str!("../../../models/qwen38/contract/encoders/qwen38-vision.json");
const DSV41: &str = include_str!("fixtures/dsv41-vision.json");
const DSV41_DIGEST: &str = "00d72f6be3fd18c17b4b91e4700a9c897909687ac5862a2abecff4863237e489";
/// `name dtype dims` from the safetensors headers of DeepSeek V4.1 Flash,
/// snapshot fb2764a5cf321eaa5070ca8f9e892818f477c16d.
const DSV41_TENSORS: &str = include_str!("fixtures/tensors/dsv41-vision.txt");
const MIMO_AUDIO: &str = include_str!("../../../models/mimo26/contract/encoders/mimo-audio.json");

fn dsv41() -> Declaration {
    parse(DSV41, &Environment::new("dsv41")).unwrap()
}

fn headers() -> BTreeMap<String, TensorInfo> {
    DSV41_TENSORS
        .lines()
        .map(|line| {
            let mut fields = line.split(' ');
            let name = fields.next().unwrap().to_owned();
            let dtype = fields.next().unwrap().to_owned();
            let shape = fields
                .next()
                .unwrap()
                .split(',')
                .map(|d| d.parse().unwrap())
                .collect();
            (
                name,
                TensorInfo {
                    dtype,
                    shape,
                    file: None,
                },
            )
        })
        .collect()
}

#[test]
fn deepseek_fixture_binds_to_its_checkpoint() {
    let declaration = dsv41();
    assert_eq!(declaration.digest(), DSV41_DIGEST);
    let headers = headers();
    assert!(
        declaration
            .tensors()
            .into_iter()
            .eq(headers.keys().cloned())
    );
    declaration
        .bind(&|name| headers.get(name).cloned())
        .unwrap();

    // The 3x3 aligner unfold needs a 9216-column weight; a 9215-column one
    // has no unfold columns.
    let mut short = headers.clone();
    short.get_mut("aligner.w1.weight").unwrap().shape = vec![5120, 9215];
    let error = declaration
        .bind(&|name| short.get(name).cloned())
        .unwrap_err();
    assert!(error.0.contains("no 3 x 3 unfold columns"), "{error}");
}

#[test]
fn deepseek_rows_follow_the_grid() {
    // 5 x 7 patches: 2 x 3 cells after the zero-padded 3x3 merge, then
    // start, three cells and a newline per cell row, and end.
    let trace = dsv41()
        .trace(Item {
            rows: 35,
            grid: Some([1, 5, 7]),
        })
        .unwrap();
    let rows: Vec<u64> = trace.iter().map(|item| item.rows).collect();
    assert_eq!(rows, [35, 35, 35, 6, 6, 6, 2 * (3 + 1) + 2]);
}

/// DeepSeek's learned-delimiter layout (DS/model.py:1219-1263): per cell
/// row, its cells then a newline row, between a start and an end row, with
/// 1D positions and the media mask over the whole span.
#[test]
fn deepseek_places_learned_delimiters() {
    use encoder_spec::geometry::{Media, Positions};
    use encoder_spec::placement::{Delimiter, Piece};

    let declaration = dsv41();
    let geometry = declaration
        .geometry("image.centered-pad.v1", Media::Grid([1, 5, 7]))
        .unwrap();
    let [run] = geometry.runs.as_slice() else {
        panic!("one image is one run");
    };
    let block = &run.steps[1];
    assert_eq!(block.sequences, Some(vec![0, 35]));
    let Some(Positions::Grid(positions)) = &block.positions else {
        panic!("rotary follows the grid");
    };
    assert_eq!(
        positions[..8],
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [1, 0]
        ]
    );
    let placement = declaration.place(&geometry, 9).unwrap();
    let (start, newline, end) = (
        Piece::Learned(Delimiter::Start),
        Piece::Learned(Delimiter::Newline),
        Piece::Learned(Delimiter::End),
    );
    let rows = |media_row| Piece::Rows { media_row, rows: 3 };
    assert_eq!(
        placement.pieces,
        [start, rows(0), newline, rows(3), newline, end]
    );
    assert!(
        placement
            .positions
            .iter()
            .copied()
            .eq((9..19).map(|p| [p; 3]))
    );
    assert_eq!((placement.next, placement.media_mask), (19, true));
    // The 1,024-token budget: one row of 1,021 cells fits, a second does not.
    for (grid, fits) in [
        ([1, 3, 3063], true),
        ([1, 3, 3064], false),
        ([1, 4, 3063], false),
    ] {
        let result = declaration.geometry("image.centered-pad.v1", Media::Grid(grid));
        assert_eq!(result.is_ok(), fits, "{grid:?}");
    }
    let widest = declaration
        .geometry("image.centered-pad.v1", Media::Grid([1, 3, 3063]))
        .unwrap();
    assert_eq!(widest.rows.last().unwrap().rows, 1024);
}

#[test]
fn pins_hold_against_the_checkpoint_config() {
    let config: Value = serde_json::from_str(include_str!("fixtures/dsv41-config.json")).unwrap();
    dsv41().check_config("config.json", &config).unwrap();
}

/// A provider of an unrelated model, with one stage variant.
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
fn registering_a_provider_changes_no_declaration() {
    for (owner, text) in [("qwen38", QWEN), ("dsv41", DSV41)] {
        let plain = parse(text, &Environment::new(owner)).unwrap();
        let mut environment = Environment::new(owner);
        environment.registry.register(&TEST_PROVIDER).unwrap();
        let registered = parse(text, &environment).unwrap();
        assert_eq!(registered.canonical_json(), plain.canonical_json());
        assert_eq!(registered.digest(), plain.digest());
    }
}

#[test]
fn registered_variants_belong_to_their_model() {
    let mut value: Value = serde_json::from_str(QWEN).unwrap();
    value["stages"]
        .as_array_mut()
        .unwrap()
        .insert(1, serde_json::json!({"op": "testmodel:cast.identity.v1"}));
    let text = value.to_string();
    let mut environment = Environment::new("qwen38");
    environment.registry.register(&TEST_PROVIDER).unwrap();
    let error = parse(&text, &environment).unwrap_err();
    assert!(
        error.0.contains(
            "`testmodel:cast.identity.v1` belongs to model testmodel; model qwen38 cannot use it"
        ),
        "{error}"
    );

    value["model"] = "testmodel".into();
    let mut environment = Environment::new("testmodel");
    environment.registry.register(&TEST_PROVIDER).unwrap();
    let declaration = parse(&value.to_string(), &environment).unwrap();
    assert_eq!(declaration.stages[1].op.id(), "testmodel:cast.identity.v1");
    // Without the provider the id is refused, not carried as opaque JSON.
    let error = parse(&value.to_string(), &Environment::new("testmodel")).unwrap_err();
    assert!(
        error.0.contains(
            "`testmodel:cast.identity.v1` is not registered by a provider of model testmodel"
        ),
        "{error}"
    );
}

#[test]
fn unregistered_bespoke_ids_are_refused() {
    let error = parse(MIMO_AUDIO, &Environment::new("mimo26")).unwrap_err();
    assert!(
        error
            .0
            .contains("is not registered by a provider of model mimo26"),
        "{error}"
    );
}

#[test]
fn registries_refuse_malformed_providers() {
    fn decode(value: Value) -> Result<Arc<dyn Component>, String> {
        registry::stage::<Identity>(value)
    }
    static OUTSIDE: Provider = Provider {
        namespace: "other",
        registrations: &[Registration {
            id: "testmodel:cast.identity.v1",
            decoder: Decoder::Stage(decode),
        }],
    };
    static SHARED: Provider = Provider {
        namespace: "other",
        registrations: &[Registration {
            id: "cast.identity.v1",
            decoder: Decoder::Stage(decode),
        }],
    };
    static SLOT_AS_STAGE: Provider = Provider {
        namespace: "other",
        registrations: &[Registration {
            id: "other:attention.x.v1",
            decoder: Decoder::Stage(decode),
        }],
    };
    static TWICE: Provider = Provider {
        namespace: "other",
        registrations: &[
            Registration {
                id: "other:cast.x.v1",
                decoder: Decoder::Stage(decode),
            },
            Registration {
                id: "other:cast.x.v1",
                decoder: Decoder::Stage(decode),
            },
        ],
    };
    for (provider, needle) in [
        (&OUTSIDE, "outside its namespace"),
        (&SHARED, "outside its namespace"),
        (&SLOT_AS_STAGE, "a decoder for another kind"),
        (&TWICE, "registered twice"),
    ] {
        let error = Environment::new("other")
            .registry
            .register(provider)
            .unwrap_err();
        assert!(error.0.contains(needle), "{error}");
    }
    let mut environment = Environment::new("testmodel");
    environment.registry.register(&TEST_PROVIDER).unwrap();
    let error = environment.registry.register(&TEST_PROVIDER).unwrap_err();
    assert!(error.0.contains("already registered"), "{error}");
    assert_eq!(
        environment.registry.ids().collect::<Vec<_>>(),
        ["testmodel:cast.identity.v1"]
    );
}

/// `grid-3axis.v1` resumes one past the span's largest position on every
/// axis, as the engine does: by `max(t, h, w)`.
#[test]
fn grid_positions_advance_past_the_largest_axis() {
    use encoder_spec::components::Layout;
    use encoder_spec::components::layout::{LayoutParams, PositionRule};

    let layout = Layout::Tokens(LayoutParams {
        hidden: 8,
        positions: PositionRule::Grid3Axis,
        media_mask: false,
    });
    for (grid, next) in [([1, 2, 3], 13), ([4, 1, 1], 14), ([2, 3, 1], 13)] {
        let rows = grid.iter().product();
        let placement = layout
            .place(
                Item {
                    rows,
                    grid: Some(grid),
                },
                10,
            )
            .unwrap();
        assert_eq!(placement.next, next, "{grid:?}");
        let largest = placement.positions.iter().flatten().max().unwrap();
        assert_eq!(placement.next, largest + 1, "{grid:?}");
    }
}

#[test]
fn tagged_float_parameters_keep_their_binary64_value() {
    use encoder_spec::components::StageOp;
    for literal in ["0.00001", "1e-5", "1.000000e-5"] {
        let text = format!(r#"{{"op":"norm.layer-f32.v1","eps":{literal}}}"#);
        let StageOp::Norm(norm) = serde_json::from_str(&text).unwrap() else {
            panic!("norm");
        };
        assert_eq!(norm.params().eps.to_bits(), 1e-5_f64.to_bits());
    }
    assert!(serde_json::from_str::<StageOp>(r#"{"op":"norm.layer-f32.v1","eps":1e400}"#).is_err());
}
