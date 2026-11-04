//! choptop reads the documents chopinfer prints for its model loader. These
//! are chopinfer's own output for the CPU fixture model, with temporary paths
//! replaced; `tests/test_model_preparation.py` checks them against the real
//! commands, so a contract change fails there or here, not in the loader.

use choptop::catalog::{
    Availability, Check, Models, PackageKind, Pick, ServeOptions, Shape, parse_command,
    parse_package_options,
};

const SERVE_OPTIONS: &str = include_str!("fixtures/contracts/fixture/serve-options.json");
const MODELS: &str = include_str!("fixtures/contracts/fixture/models.json");
const PICK: &str = include_str!("fixtures/contracts/fixture/models-pick.json");
const COMMAND: &str = include_str!("fixtures/contracts/fixture/models-command.json");
const CHECK: &str = include_str!("fixtures/contracts/fixture/serve-check.json");
const PACKAGE: &str = include_str!("fixtures/contracts/fixture/list-model-options.json");

#[test]
fn reads_the_options_serve_describes() {
    let options = ServeOptions::parse(SERVE_OPTIONS).unwrap();
    let flag = |name: &str| {
        options
            .options
            .iter()
            .find(|option| option.flag == name)
            .unwrap_or_else(|| panic!("serve describes {name}"))
    };
    assert_eq!(flag("--max-connections").shape, Shape::Count);
    assert!(flag("--max-connections").model_launch);
    assert_eq!(flag("--max-sequences").shape, Shape::Count);
    assert!(flag("--max-sequences").model_launch);
    assert_eq!(
        flag("--prefix-cache").choices.as_deref(),
        Some(&["on".to_owned(), "off".to_owned()][..])
    );
    assert!(flag("--model-option").repeatable);
    assert_eq!(
        flag("--prefill-policy").choices.as_deref(),
        Some(&["fixed".to_owned(), "tbt".to_owned(), "adaptive".to_owned()][..])
    );
    assert_eq!(flag("--decode-tbt-target-ms").shape, Shape::Count);
    for retired in [
        "--prefill-budget",
        "--prefill-timing-cuts",
        "--prefill-itl-target-ms",
        "--prefill-decode-steps",
        "--mixed-min-prefill-tokens",
        "--mixed-decode-steps",
    ] {
        assert!(options.options.iter().all(|option| option.flag != retired));
    }
    assert_eq!(flag("--served-model-name").shape, Shape::Words);
    assert_eq!(flag("--speculation-confidence").shape, Shape::JsonOrOff);
    // Flags a loader must not offer for `serve --model`.
    for owned in ["--bundle", "--profile", "--describe-options"] {
        assert!(!flag(owned).model_launch, "{owned}");
    }
    let key = options
        .environment
        .iter()
        .find(|variable| variable.name == "CHOPINFER_API_KEY")
        .unwrap();
    assert!(key.secret);
}

#[test]
fn reads_the_installed_model_and_what_serve_model_picks() {
    let models = Models::parse(MODELS).unwrap();
    assert!(models.errors.is_empty());
    let [model] = &models.models[..] else {
        panic!("one fixture model: {models:?}");
    };
    assert_eq!(
        (model.model.as_str(), model.helper.as_str()),
        ("fixture", "chopinfer-model-fixture")
    );
    let names: Vec<_> = model
        .profiles
        .iter()
        .map(|profile| profile.name.as_str())
        .collect();
    assert_eq!(names, ["tp4-ar", "tp4-dflash", "tp4-mtp"]);
    let mtp = model.profile("tp4-mtp").unwrap();
    assert_eq!(
        (
            mtp.tensor_parallel,
            mtp.drafter.as_str(),
            mtp.speculative_tokens
        ),
        (4, "mtp", 3)
    );
    assert_eq!(mtp.qualified_max_sequences, None);
    assert_eq!(mtp.settings["max-sequences"], "1");

    let pick = Pick::parse(PICK).unwrap();
    assert_eq!(
        (pick.model.as_deref(), pick.refusal.as_deref()),
        (Some("fixture"), None)
    );
    let [size] = &pick.sizes[..] else {
        panic!("one size: {pick:?}");
    };
    assert_eq!(
        (size.tensor_parallel, size.pick.as_deref()),
        (4, Some("tp4-mtp"))
    );
    assert_eq!(
        size.profile("tp4-mtp").unwrap().availability,
        Availability::Available {
            arguments: Vec::new(),
            inherited_model_options: Vec::new()
        }
    );
    match &size.profile("tp4-ar").unwrap().availability {
        Availability::Available { arguments, .. } => {
            let at = arguments
                .iter()
                .position(|argument| argument == "--speculative-tokens");
            assert_eq!(at.map(|at| arguments[at + 1].as_str()), Some("0"));
        }
        other => panic!("tp4-ar runs over the pick: {other:?}"),
    }
    match &size.profile("tp4-dflash").unwrap().availability {
        Availability::Unavailable(reason) => assert!(reason.contains("dflash"), "{reason}"),
        other => panic!("tp4-dflash drafts with another drafter: {other:?}"),
    }
}

#[test]
fn reads_the_command_without_a_listen_address() {
    let arguments = parse_command(COMMAND).unwrap();
    assert_eq!(
        arguments[..5],
        [
            "serve",
            "--model",
            "/models/fixture",
            "--tensor-parallel-size",
            "4"
        ]
    );
    let sequences = arguments
        .iter()
        .filter(|argument| *argument == "--max-sequences");
    assert_eq!(sequences.count(), 1);
    assert!(!arguments.iter().any(|argument| argument == "--listen"));
}

#[test]
fn a_check_resolves_exactly_the_settings_its_profile_lists() {
    let models = Models::parse(MODELS).unwrap();
    let check = Check::parse(CHECK).unwrap();
    let profile = models.models[0].profile(&check.profile).unwrap();
    // What the loader compares before a start: here the pick with no edits.
    assert_eq!(check.settings, profile.settings);
    let [option] = &check.model_options[..] else {
        panic!("one model option: {check:?}");
    };
    assert_eq!(
        (option.name.as_str(), option.value.as_str()),
        ("scratch_multiplier", "1")
    );
    assert_eq!(option.value, option.default);
    assert!(!option.default_is_qualified);
    assert!(
        check
            .warnings
            .iter()
            .any(|warning| warning.contains("scratch_multiplier"))
    );
}

#[test]
fn reads_the_package_options_and_their_defaults() {
    let options = parse_package_options(PACKAGE).unwrap();
    let [option] = &options[..] else {
        panic!("one package option: {options:?}");
    };
    assert_eq!(option.name, "scratch_multiplier");
    assert_eq!(
        option.kind,
        PackageKind::Integer {
            default: 1,
            minimum: 1,
            maximum: 4
        }
    );
    assert_eq!(option.default_text(), "1");
    assert!(!option.default_is_qualified);
}
