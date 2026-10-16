//! Taps on the fake host: a serving bundle, built without the
//! `diagnostic-taps` feature, refuses to create while the environment asks
//! for taps, so a production server never runs eagerly to write them.

use super::*;

/// The create runs in a child test process, once per variable that turns
/// taps on, because the variables are process-wide.
#[cfg(not(feature = "diagnostic-taps"))]
#[test]
fn serving_create_refuses_taps() {
    const CHILD: &str = "CHOPINFER_TEST_TAPS_CHILD";
    const NAME: &str = "fake_host_tests::taps::serving_create_refuses_taps";
    if let Some(variable) = std::env::var_os(CHILD) {
        let variable = variable.into_string().unwrap();
        let host = Rc::new(FakeHost::default());
        let refused = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &configuration())
            .err();
        assert!(refused.is_some(), "{variable} did not refuse the create");
        let logs = host.logs.borrow();
        assert!(
            logs.iter()
                .any(|line| line.contains(&variable) && line.contains("diagnostic-taps")),
            "{logs:?}"
        );
        assert_eq!(host.released.get(), host.acquired.get());
        return;
    }
    let directory = std::env::temp_dir().join(format!("qwen-taps-serving-{}", std::process::id()));
    for variable in [runtime_bundle_sdk::taps::TAPS_ENV.dir, crate::dump::ENV.dir] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([NAME, "--exact", "--test-threads=1"])
            .env(CHILD, variable)
            .env(variable, &directory)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{variable}: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert!(!directory.exists(), "a refused create wrote taps");
}

/// A diagnostic bundle creates with taps on.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_create_accepts_taps() {
    if create_or_refuse() {
        return;
    }
    create_cases(
        "fake_host_tests::taps::diagnostic_create_accepts_taps",
        &[(&[], "created")],
    );
}

/// Runs test `name` in a child process per case, with taps on and the
/// case's variables set; the child ([`create_or_refuse`]) creates, or checks
/// that the create is refused with the expected message.
#[cfg(feature = "diagnostic-taps")]
fn create_cases(name: &str, cases: &[(&[(&str, &str)], &str)]) {
    let directory = std::env::temp_dir().join(format!("qwen-taps-cases-{}", std::process::id()));
    for (variables, expected) in cases {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([name, "--exact", "--test-threads=1"])
            .env(CHILD, expected)
            .env(runtime_bundle_sdk::taps::TAPS_ENV.dir, &directory)
            .envs(variables.iter().copied())
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{variables:?}: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[cfg(feature = "diagnostic-taps")]
const CHILD: &str = "CHOPINFER_TEST_TAPS_CHILD";

/// In a child of [`create_cases`]: creates, or checks the refusal; true
/// when this is such a child.
#[cfg(feature = "diagnostic-taps")]
fn create_or_refuse() -> bool {
    let Some(expected) = std::env::var_os(CHILD) else {
        return false;
    };
    let expected = expected.into_string().unwrap();
    let host = Rc::new(FakeHost::default());
    let created = api().create(HostServicesTableV1::new(Rc::clone(&host)), &configuration());
    if expected == "created" {
        created.unwrap().destroy().unwrap();
        return true;
    }
    assert!(created.is_err(), "the create took {expected}");
    let logs = host.logs.borrow();
    assert!(logs.iter().any(|line| line.contains(&expected)), "{logs:?}");
    assert_eq!(host.released.get(), host.acquired.get());
    true
}

/// A sites filter, generic or the older Qwen one, that names values that
/// select no tap refuses the create; the older one's suffixes of Qwen's
/// earlier tap names still select their taps.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_create_refuses_sites_that_select_nothing() {
    if create_or_refuse() {
        return;
    }
    create_cases(
        "fake_host_tests::taps::diagnostic_create_refuses_sites_that_select_nothing",
        &[
            (
                &[
                    ("CHOPINFER_TAPS_SITES", "out,x.indexer.topk,L03.attn.core"),
                    ("CHOPINFER_TAPS_LAYERS", "3"),
                ],
                "created",
            ),
            (
                &[("CHOPINFER_TAPS_SITES", "out,attn.selected,bogus")],
                "CHOPINFER_TAPS_SITES selects no tap: attn.selected, bogus",
            ),
            (
                &[("CHOPINFER_QWEN38_DUMP_TAPS", "out.x,attn.selected,final.u")],
                "created",
            ),
            (
                &[("CHOPINFER_QWEN38_DUMP_TAPS", "attn.selected,x.indexer.topk")],
                "CHOPINFER_QWEN38_DUMP_TAPS selects no tap: x.indexer.topk",
            ),
        ],
    );
}

/// Detail layers, generic or the older Qwen ones, that are not a list of
/// the model's layers refuse the create.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_create_refuses_layers_it_cannot_read() {
    if create_or_refuse() {
        return;
    }
    create_cases(
        "fake_host_tests::taps::diagnostic_create_refuses_layers_it_cannot_read",
        &[
            (&[("CHOPINFER_TAPS_LAYERS", "0, 1,3")], "created"),
            (&[("CHOPINFER_QWEN38_DUMP_OPS", "all")], "created"),
            (
                &[("CHOPINFER_TAPS_LAYERS", "0;1")],
                "CHOPINFER_TAPS_LAYERS is not all or comma-separated layer numbers: 0;1",
            ),
            (
                &[("CHOPINFER_QWEN38_DUMP_OPS", "3,,x")],
                "CHOPINFER_QWEN38_DUMP_OPS is not all or comma-separated layer numbers: 3,,x",
            ),
            (
                &[("CHOPINFER_TAPS_LAYERS", "47,48,99")],
                "CHOPINFER_TAPS_LAYERS names layers without taps: 48, 99",
            ),
        ],
    );
}

/// Sites and layers that together select no tap refuse the create.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_create_refuses_filters_that_select_nothing() {
    if create_or_refuse() {
        return;
    }
    let only_detail = "selects only detail taps of layers CHOPINFER_TAPS_LAYERS does not keep";
    create_cases(
        "fake_host_tests::taps::diagnostic_create_refuses_filters_that_select_nothing",
        &[
            (
                &[("CHOPINFER_TAPS_SITES", " , ")],
                "CHOPINFER_TAPS_SITES is empty",
            ),
            (
                &[("CHOPINFER_QWEN38_DUMP_TAPS", "")],
                "CHOPINFER_QWEN38_DUMP_TAPS is empty",
            ),
            (
                &[("CHOPINFER_TAPS_SITES", "out,x.gdn.qkv")],
                &format!("CHOPINFER_TAPS_SITES {only_detail}: x.gdn.qkv"),
            ),
            (
                &[
                    ("CHOPINFER_TAPS_SITES", "x.gdn.qkv"),
                    ("CHOPINFER_TAPS_LAYERS", "3"),
                ],
                &format!("CHOPINFER_TAPS_SITES {only_detail}: x.gdn.qkv"),
            ),
            (
                &[("CHOPINFER_QWEN38_DUMP_TAPS", "lin.qkv")],
                &format!("CHOPINFER_QWEN38_DUMP_TAPS {only_detail}: lin.qkv"),
            ),
            (
                &[
                    ("CHOPINFER_TAPS_SITES", "x.gdn.qkv"),
                    ("CHOPINFER_TAPS_LAYERS", "0"),
                ],
                "created",
            ),
            (
                &[
                    ("CHOPINFER_QWEN38_DUMP_TAPS", "lin.qkv"),
                    ("CHOPINFER_QWEN38_DUMP_OPS", "0"),
                ],
                "created",
            ),
        ],
    );
}

/// A first position that is not one refuses the create.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_create_refuses_a_first_position_it_cannot_read() {
    if create_or_refuse() {
        return;
    }
    create_cases(
        "fake_host_tests::taps::diagnostic_create_refuses_a_first_position_it_cannot_read",
        &[
            (
                &[("CHOPINFER_TAPS_FROM", "100x")],
                "CHOPINFER_TAPS_FROM is not a position: 100x",
            ),
            (
                &[("CHOPINFER_QWEN38_DUMP_FROM", "-3")],
                "CHOPINFER_QWEN38_DUMP_FROM is not a position: -3",
            ),
            (&[("CHOPINFER_TAPS_FROM", "5")], "created"),
        ],
    );
}

/// A byte budget that is not a positive count refuses the create.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_create_refuses_a_byte_budget_it_cannot_read() {
    if create_or_refuse() {
        return;
    }
    create_cases(
        "fake_host_tests::taps::diagnostic_create_refuses_a_byte_budget_it_cannot_read",
        &[
            (
                &[("CHOPINFER_TAPS_BYTES", "10x")],
                "CHOPINFER_TAPS_BYTES is not a byte count from 1: 10x",
            ),
            (
                &[("CHOPINFER_QWEN38_DUMP_BYTES", "0")],
                "CHOPINFER_QWEN38_DUMP_BYTES is not a byte count from 1: 0",
            ),
            (&[("CHOPINFER_QWEN38_DUMP_BYTES", "1048576")], "created"),
        ],
    )
}

/// A comparison scope that is not valid JSON or not a valid scope refuses
/// the create.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_create_refuses_a_scope_it_cannot_read() {
    if create_or_refuse() {
        return;
    }
    let scope = r#"{"model":"qwen38","evidence":"chained","contract":"c",
        "inputs":{"rows":[{"position":0}],"state":{"kind":"fresh"}},
        "probes":[{"name":"z","shape":[1,2]}]}"#;
    create_cases(
        "fake_host_tests::taps::diagnostic_create_refuses_a_scope_it_cannot_read",
        &[
            (
                &[("CHOPINFER_TAPS_SCOPE_JSON", "{bad")],
                "CHOPINFER_TAPS_SCOPE_JSON is not a comparison scope: key must be a string",
            ),
            (
                &[("CHOPINFER_QWEN38_DUMP_SCOPE_JSON", "{}")],
                "CHOPINFER_QWEN38_DUMP_SCOPE_JSON is not a comparison scope: missing scope model",
            ),
            (&[("CHOPINFER_TAPS_SCOPE_JSON", scope)], "created"),
        ],
    );
}
