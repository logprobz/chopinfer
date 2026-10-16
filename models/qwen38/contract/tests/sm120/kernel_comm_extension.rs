//! Compile-only coverage of the comm transports' extension points: input
//! policies, input types and epilogue geometry. Nothing here opens a GPU.

use kernel_build::{BuildError, BuildOptions, BuiltCubin, CubinSpec, Toolchain};
use kernels_sm120::comm::{Epilogue, input};
use model_qwen38_contract::sm120::comm::decode as qwen38_decode;
use model_qwen38_contract::sm120::comm::{self as qwen38, HcShape, RANKS};

/// Builds `spec` into a scratch directory of its own; None without CUDA.
fn build(spec: CubinSpec) -> Option<Result<Vec<BuiltCubin>, BuildError>> {
    let toolchain = Toolchain::detect().expect("CUDA toolchain");
    if toolchain.skipped {
        eprintln!("skipping comm extension compile: CUDA toolchain unavailable");
        return None;
    }
    let scratch = std::env::temp_dir().join(format!(
        "chop-comm-extension-{}-{}",
        spec.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&scratch).expect("create compile scratch");
    let options = BuildOptions {
        lock_update: true,
        emit_cargo_directives: false,
        profile_line_info: false,
    };
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &scratch,
        &scratch.join("kernels.lock"),
        options,
    );
    std::fs::remove_dir_all(&scratch).expect("remove compile scratch");
    Some(built)
}

fn assert_compiles_without_local_memory(spec: CubinSpec) {
    let name = spec.name.clone();
    let Some(built) = build(spec) else { return };
    let built = built.unwrap_or_else(|error| panic!("{name}: {error}"));
    for entry in &built[0].entries {
        assert_eq!(
            (entry.local_bytes, entry.stack_bytes),
            (0, 0),
            "{}",
            entry.symbol
        );
    }
}

fn assert_refused(spec: CubinSpec, diagnostic: &str) {
    let name = spec.name.clone();
    let Some(built) = build(spec) else { return };
    match built {
        Ok(_) => panic!("{name} compiled; expected `{diagnostic}`"),
        Err(error) => assert!(
            error.to_string().contains(diagnostic),
            "{name}: expected `{diagnostic}` in:\n{error}"
        ),
    }
}

/// The push transport takes its input policy as a type: the combine
/// entries with four routed rows per row instead of MiMo's eight compile
/// from the same transport.
#[test]
fn push_takes_another_routed_input_policy() {
    let mut spec = kernels_sm120::comm::cubin_spec();
    spec.name = "comm_extension_topk4".into();
    spec.entries
        .retain(|entry| entry.body.contains("RoutedCombine<8>"));
    assert_eq!(
        spec.entries.len(),
        3,
        "combine, decode combine, wide combine"
    );
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replacen("chop_comm_", "chop_comm_topk4_", 1);
        entry.body = entry.body.replace("RoutedCombine<8>", "RoutedCombine<4>");
    }
    assert_compiles_without_local_memory(spec);
}

/// A model whose all-reduce input is F32 and whose rows are 5120 wide with
/// four streams supplies an epilogue and an input type; every
/// hyper-connection entry (one-shot, phased pair, decode and wide) compiles
/// from the same transports. Its LL layout does not fit (320 vectors per
/// stream part), so its decode cubin builds without LL and its one-shot
/// takes every row count.
#[test]
fn hyper_connection_transports_take_f32_input_at_another_width() {
    let shape = HcShape {
        epilogue: Epilogue {
            width: 5120,
            ..qwen38::EPILOGUE
        },
        input: input::F32,
    };
    assert!(!shape.low_latency_fits());
    assert_eq!(qwen38_decode::low_latency_rows(shape), 0);
    // Without LL the one-shot pads its groups: one row of four streams
    // takes a whole sending cluster of group CTAs, and every compiled decode
    // entry checks that padded grid.
    for (rows, grid) in [(1, 16), (2, 16), (3, 24), (31, 136), (32, 136)] {
        let plan = qwen38_decode::plan_with(shape, rows).unwrap();
        assert_eq!(
            (plan.grid, plan.cluster),
            ([grid, 1, 1], [8, 1, 1]),
            "{rows}"
        );
    }
    hc_specs_compile(shape, "f32_5120");
}

/// The F32 input at Qwen3.8's width keeps the LL layout (160 vectors per
/// stream part), and BF16 rows at 5120 do too.
#[test]
fn hyper_connection_ll_layout_takes_f32_input_and_wider_bf16_rows() {
    for (shape, tag) in [
        (
            HcShape {
                input: input::F32,
                ..qwen38::SHAPE
            },
            "f32_2560",
        ),
        (
            HcShape {
                epilogue: Epilogue {
                    width: 5120,
                    ..qwen38::EPILOGUE
                },
                ..qwen38::SHAPE
            },
            "bf16_5120",
        ),
    ] {
        assert!(shape.low_latency_fits(), "{tag}");
        assert_eq!(
            qwen38_decode::plan_with(shape, 2).unwrap().cluster,
            [4, 1, 1]
        );
        let mut decode = qwen38_decode::cubin_specs_with(shape, &RANKS).remove(0);
        decode.name = format!("comm_extension_{tag}_decode");
        rename(&mut decode, tag);
        assert_compiles_without_local_memory(decode);
    }
}

fn rename(spec: &mut CubinSpec, tag: &str) {
    for entry in &mut spec.entries {
        entry.symbol = entry
            .symbol
            .replacen("chop_comm_", &format!("chop_comm_{tag}_"), 1);
    }
}

fn hc_specs_compile(shape: HcShape, tag: &str) {
    let specs = [
        qwen38::cubin_specs_with(shape),
        qwen38_decode::cubin_specs_with(shape, &RANKS),
        qwen38_decode::wide_cubin_specs_with(shape, &RANKS),
    ];
    for mut spec in specs.into_iter().flatten() {
        spec.name = format!("comm_extension_{tag}_{}", spec.name);
        rename(&mut spec, tag);
        assert!(
            spec.entries
                .iter()
                .all(|entry| entry.body.contains(shape.input.policy))
        );
        assert_compiles_without_local_memory(spec);
    }
}

fn stub_shape(threads: u32, streams: u32) -> HcShape {
    HcShape {
        epilogue: Epilogue {
            template: match threads {
                128 => "stub::Threads_<128>::Epilogue",
                _ => "stub::Threads_<256>::Epilogue",
            },
            streams,
            ..qwen38::EPILOGUE
        },
        input: input::BF16,
    }
}

fn with_stub(mut spec: CubinSpec, name: &str) -> CubinSpec {
    spec.name = format!("comm_extension_{name}");
    spec.headers
        .push("#include \"tests/comm_epilogue_stub.cuh\"".into());
    rename(&mut spec, name);
    spec
}

/// An epilogue whose CTAs are not 256 threads would leave the senders'
/// cluster strides and the epilogue's column strides disagreeing: every
/// hyper-connection entry refuses it at compile time.
#[test]
fn hyper_connection_entries_refuse_other_thread_counts() {
    let shape = stub_shape(128, 4);
    let base = qwen38::cubin_specs_with(shape).remove(0);
    for entry in &base.entries {
        let mut one = base.clone();
        one.entries = vec![entry.clone()];
        assert_refused(
            with_stub(one, &format!("threads128_{}", entry.symbol)),
            "one 256-thread CTA per (row, stream) group",
        );
    }
    assert_refused(
        with_stub(
            qwen38_decode::cubin_specs_with(shape, &[4]).remove(0),
            "threads128_decode",
        ),
        "one 256-thread CTA per (row, stream) group",
    );
}

/// More than eight streams do not fit the LL cluster: the host plans them
/// without LL and their decode cubin compiles, while a cubin that still
/// asks for LL rows is refused at compile time.
#[test]
fn ll_layout_refuses_more_than_eight_streams() {
    let shape = stub_shape(256, 10);
    assert!(!shape.low_latency_fits());
    assert_eq!(
        qwen38_decode::plan_with(shape, 4).unwrap().cluster,
        [8, 1, 1]
    );
    let decode = qwen38_decode::cubin_specs_with(shape, &[4]).remove(0);
    assert_compiles_without_local_memory(with_stub(decode.clone(), "streams10_decode"));
    let mut forced = decode;
    for (name, value) in &mut forced.defines {
        if name == "CHOP_QWEN38_LL_MAX_ROWS" {
            *value = "2".into();
        }
    }
    assert_refused(
        with_stub(forced, "streams10_forced_ll"),
        "the LL layout needs at most eight streams",
    );
}

/// The push transport exchanges BF16 rows and refuses an F32 input type.
#[test]
fn push_refuses_f32_input() {
    let mut spec = kernels_sm120::comm::cubin_spec();
    spec.name = "comm_extension_push_f32".into();
    spec.entries
        .retain(|entry| entry.symbol == "chop_comm_allreduce_oneshot_norm_bf16_v2");
    for entry in &mut spec.entries {
        entry.body = entry
            .body
            .replace(">(input,", ", chop::comm::input::F32>(input,");
        assert!(entry.body.contains("input::F32"));
    }
    spec.headers.push(input::F32.include());
    assert_refused(spec, "the push transport exchanges BF16 rows");
}

/// An input of 8-byte elements would leave the phased and wide sum writers
/// with no whole F32 vector to store: the host refuses the shape and every
/// hyper-connection entry refuses the type at compile time.
#[test]
fn hyper_connection_entries_refuse_eight_byte_inputs() {
    let f64 = kernels_sm120::comm::Input {
        header: "tests/comm_input_stub.cuh",
        policy: "stub::F64",
        cxx_type: "double",
        element_bytes: 8,
    };
    let shape = HcShape {
        input: f64,
        ..qwen38::SHAPE
    };
    assert!(shape.validate().is_err());
    // The F32 specs with the stub type in its place.
    let f32 = HcShape {
        input: input::F32,
        ..qwen38::SHAPE
    };
    let mut spec = qwen38::cubin_specs_with(f32).remove(0);
    spec.name = "comm_extension_f64".into();
    spec.headers.push(f64.include());
    rename(&mut spec, "f64");
    for entry in &mut spec.entries {
        entry.body = entry.body.replace(input::F32.policy, f64.policy);
        for param in &mut entry.params {
            if let kernel_build::EntryParam::DevicePointer { cxx_type, .. }
            | kernel_build::EntryParam::PointerArray { cxx_type, .. } = param
            {
                *cxx_type = cxx_type.replace("float", "double");
            }
        }
    }
    assert_refused(spec, "inputs of BF16 or F32 elements");
}

/// The transports describe their epilogues and inputs by role: no model
/// name appears in their prose or diagnostics. Compatibility identifiers
/// (the qwen38 file, namespace, macro and entry names) are allowed; model
/// attribution lives in the epilogue and input policy headers.
#[test]
fn transports_name_no_model() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../crates/kernels-sm120/cuda/comm");
    for header in [
        "comm.cuh",
        "push.cuh",
        "hyper_connection.cuh",
        "hyper_connection_decode.cuh",
        "hyper_connection_decode_wide.cuh",
        "input/bf16.cuh",
        "input/f32.cuh",
    ] {
        let text = std::fs::read_to_string(root.join(header)).expect("transport header");
        for (number, line) in text.lines().enumerate() {
            for token in line.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
                let token = token.to_ascii_lowercase();
                let named = ["mimo", "gemma", "deepseek", "llama"]
                    .iter()
                    .any(|name| token.starts_with(name))
                    || (token.starts_with("qwen")
                        && token != "qwen38"
                        && !token.starts_with("qwen38_"))
                    || (token.contains("qwen") && !token.contains("qwen38"));
                assert!(!named, "{header}:{}: {line}", number + 1);
            }
        }
    }
}

/// Builds `spec` and returns each entry's SASS instructions; None without
/// CUDA.
fn sass_of(spec: CubinSpec) -> Option<Vec<(String, Vec<String>)>> {
    let toolchain = Toolchain::detect().expect("CUDA toolchain");
    if toolchain.skipped {
        eprintln!("skipping comm extension compile: CUDA toolchain unavailable");
        return None;
    }
    let scratch = std::env::temp_dir().join(format!(
        "chop-comm-extension-{}-{}",
        spec.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&scratch).expect("create compile scratch");
    let options = BuildOptions {
        lock_update: true,
        emit_cargo_directives: false,
        profile_line_info: false,
    };
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &scratch,
        &scratch.join("kernels.lock"),
        options,
    )
    .expect("build the padded entries");
    let output = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .expect("run cuobjdump");
    std::fs::remove_dir_all(&scratch).expect("remove compile scratch");
    assert!(output.status.success(), "cuobjdump failed");
    let mut functions: Vec<(String, Vec<String>)> = vec![];
    for line in String::from_utf8(output.stdout)
        .expect("SASS is UTF-8")
        .lines()
    {
        if let Some(name) = line.trim().strip_prefix("Function : ") {
            functions.push((name.into(), vec![]));
        } else if let Some((_, instruction)) = line.split_once("*/")
            && line.trim_start().starts_with("/*")
            && let Some(function) = functions.last_mut()
        {
            let instruction = instruction.split("/*").next().unwrap_or("").trim();
            if !instruction.is_empty() {
                function.1.push(instruction.into());
            }
        }
    }
    Some(functions)
}

/// A shape without LL pads its one-shot's group CTAs, and padding CTAs
/// leave at entry: in every decode entry that reads its parameters before
/// the PDL wait, a predicated EXIT on the block index comes before the
/// parameter callback (the stub marks it with a PMTRIG) and before any
/// load, barrier or timer read. tests/comm_padding_check.cuh checks at
/// compile time that the exit's predicate marks exactly the padding CTAs.
#[test]
fn padding_ctas_leave_before_the_parameter_callback() {
    let shape = HcShape {
        epilogue: Epilogue {
            template: "stub::Threads_<256>::Epilogue",
            width: 5120,
            ..qwen38::EPILOGUE
        },
        input: input::F32,
    };
    assert!(!shape.low_latency_fits());
    let mut spec = with_stub(
        qwen38_decode::cubin_specs_with(shape, &RANKS).remove(0),
        "padding",
    );
    spec.headers
        .push("#include \"tests/comm_padding_check.cuh\"".into());
    spec.entries.retain(|entry| entry.symbol.ends_with("_v2"));
    assert_eq!(spec.entries.len(), RANKS.len());
    let Some(functions) = sass_of(spec) else {
        return;
    };
    assert_eq!(functions.len(), RANKS.len());
    for (name, instructions) in functions {
        let exit = instructions
            .iter()
            .position(|i| i.starts_with('@') && i.contains(" EXIT"))
            .unwrap_or_else(|| panic!("{name}: no predicated exit"));
        assert!(
            instructions[..exit]
                .iter()
                .any(|i| i.contains("SR_CTAID.X")),
            "{name}: the first exit does not follow the block index"
        );
        let parameters = instructions
            .iter()
            .position(|i| i.contains("PMTRIG"))
            .unwrap_or_else(|| panic!("{name}: no parameter callback"));
        assert!(
            exit < parameters,
            "{name}: parameters before the padding exit"
        );
        for instruction in &instructions[..exit] {
            assert!(
                ![
                    "LDG",
                    "LD.",
                    "LDS",
                    "BAR",
                    "ACQ",
                    "SR_GLOBALTIMER",
                    "PMTRIG"
                ]
                .iter()
                .any(|op| instruction.contains(op)),
                "{name}: `{instruction}` before the padding exit"
            );
        }
    }
}
