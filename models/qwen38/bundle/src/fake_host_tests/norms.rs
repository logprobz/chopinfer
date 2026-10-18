//! Fake host coverage of F32 norm weights: each consumer binds its F32 form
//! by the stored dtype of the view it reads, and images without F32 norm
//! weights keep the BF16 entries and load no F32 form.
use super::*;
use model_qwen38_contract::views::MODEL;

const BF16_NORM: &str = crate::sm120::entries::HC_NORM;
const F32_NORM: &str = "chop_row_qwen38_hc_norm_f32_v1";
const F32_COMM: [&str; 3] = [
    "chop_comm_qwen38_allreduce_hc_f32_v1",
    "chop_comm_qwen38_reduce_scatter_f32_v2",
    "chop_comm_qwen38_allgather_hc_f32_v2",
];
const F32_DECODE: [&str; 2] = [
    "chop_comm_qwen38_allreduce_hc_f32_decode_r4_v2",
    "chop_comm_qwen38_allreduce_hc_f32_decode_r4_v3",
];
const F32_WIDE: &str = "chop_comm_qwen38_allreduce_hc_f32_wide_r4_v1";

/// The norm weights the MXFP8-dense layout stores in F32: every main
/// layer's two hyper-connection norms, the final mixer's and the three PLE
/// norms. Its MTP norms are BF16.
pub(super) fn mxfp8_dense_norms(shape: Shape) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for layer in 0..shape.layers() {
        for sublayer in ["attn", "mlp"] {
            names.insert(format!(
                "{MODEL}.layers.{layer}.{sublayer}_hyper_connection.hc_norm.weight"
            ));
        }
    }
    names.insert(format!("{MODEL}.hyper_connection_mixer.hc_norm.weight"));
    for norm in ["norm_key", "norm_conv", "norm_query"] {
        names.insert(format!(
            "{MODEL}.layers.{}.ple.{norm}.weight",
            shape.ple_layer()
        ));
    }
    names
}

/// A TP4 host serving `served` sequences with MXFP8 dense projections and
/// NVFP4 experts and lookup, as the MXFP8-dense layout prepares, whose rank views
/// store `f32_norms` as F32.
fn prepared(served: u32, f32_norms: BTreeSet<String>) -> Rc<FakeHost> {
    let host = Rc::new(FakeHost::default());
    host.served.set(Some(served));
    *host.precision.borrow_mut() = Some(super::precision::document_for(
        Shape::TP4,
        true,
        true,
        false,
    ));
    *host.f32_norms.borrow_mut() = f32_norms;
    host
}

/// A step of `(request, slot, previous tokens, inputs)` items, each on one
/// KV page numbered by its slot, for more than 32 served sequences.
fn slots_step(items: &[(u64, u64, u64, &[u32])]) -> BatchRequestV1 {
    let (batch, appends): (Vec<_>, Vec<_>) = items
        .iter()
        .map(|&(request, slot, previous, inputs)| {
            let count = inputs.len() as u64;
            (
                BatchItemV1::new(
                    request,
                    previous > 0 && count == 1,
                    count as u32,
                    true,
                    inputs.to_vec(),
                ),
                BatchAppendV1::new(previous, count, previous + count, vec![slot], None),
            )
        })
        .unzip();
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(items.iter().map(|item| item.1).collect())
}

fn resolved(host: &FakeHost) -> BTreeSet<String> {
    host.symbol_names.borrow().values().cloned().collect()
}

fn bf16_comm() -> Vec<&'static str> {
    let decode = &crate::sm120::catalog::geometry_owner(Shape::TP4).decode;
    let mut symbols = vec![
        model_qwen38_contract::sm120::comm::SYMBOL,
        model_qwen38_contract::sm120::comm::SCATTER_SYMBOL,
        model_qwen38_contract::sm120::comm::GATHER_SYMBOL,
        decode.wide_symbols[1],
    ];
    symbols.extend([decode.symbols_v2[1], decode.symbols_v3[1]]);
    symbols
}

/// The MXFP8-dense image binds every main-layer and PLE norm consumer to its
/// F32 form: the grouped norm rows, the one-shot and phased prefill
/// all-reduces, and the decode and wide all-reduces of graph buckets.
#[test]
fn mxfp8_dense_f32_norms_launch_their_f32_forms_at_tp4() {
    let shape = Shape::TP4;
    let host = prepared(100, mxfp8_dense_norms(shape));
    let mut instance = create(&host);
    let names = resolved(&host);
    for symbol in [F32_NORM, F32_WIDE]
        .into_iter()
        .chain(F32_COMM)
        .chain(F32_DECODE)
    {
        assert!(names.contains(symbol), "{symbol}");
    }
    // One short prompt (the one-shot all-reduce) and a 40-row one (the
    // phased pair).
    let long = (0..40).map(|token| token % 50 + 7).collect::<Vec<u32>>();
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 2, 0, &PROMPT), (11, 3, 0, &long)]),
    );
    instance.commit().unwrap();
    let prefill = launched(&host, 0);
    // Four ranks, each reducing twice per layer: once across the rows of
    // both prefills.
    let reduces = 4 * 2 * shape.layers() as usize;
    assert_eq!(count(&prefill, F32_COMM[1]), reduces);
    assert_eq!(count(&prefill, F32_COMM[2]), reduces);
    // FirstNorm, PLE key, convolution and the PLE layer's norm per rank.
    assert_eq!(count(&prefill, F32_NORM), 4 * 4);
    assert_eq!(count(&prefill, BF16_NORM), 0);
    for symbol in bf16_comm() {
        assert_eq!(count(&prefill, symbol), 0, "{symbol}");
    }
    // A one-row decode bucket takes the LL decode entry.
    let before = host.launches.borrow().len();
    run(&host, &mut instance, "decode", step(&[(10, 2, 3, &[7])]));
    instance.commit().unwrap();
    let decode = launched(&host, before);
    assert_eq!(count(&decode, F32_DECODE[1]), reduces);
    assert_eq!(count(&decode, BF16_NORM), 0);
    for symbol in bf16_comm() {
        assert_eq!(count(&decode, symbol), 0, "{symbol}");
    }
    // A short prefill reduces through the one-shot entry.
    let before = host.launches.borrow().len();
    run(&host, &mut instance, "short", step(&[(12, 4, 0, &PROMPT)]));
    instance.commit().unwrap();
    let short = launched(&host, before);
    assert_eq!(count(&short, F32_COMM[0]), reduces);
    for symbol in bf16_comm() {
        assert_eq!(count(&short, symbol), 0, "{symbol}");
    }
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
}

/// Graph buckets above 32 rows reduce through the F32 wide entry.
#[test]
fn mxfp8_dense_f32_norms_take_the_f32_wide_entry() {
    let shape = Shape::TP4;
    let host = prepared(100, mxfp8_dense_norms(shape));
    let mut instance = create(&host);
    let prompts = (0..40_u32)
        .map(|slot| [slot % 50 + 7, 9])
        .collect::<Vec<_>>();
    let prefill = (0..40_u64)
        .map(|slot| (1000 + slot, slot, 0, &prompts[slot as usize][..]))
        .collect::<Vec<_>>();
    run(&host, &mut instance, "prefill", slots_step(&prefill));
    instance.commit().unwrap();
    let decode = (0..40_u64)
        .map(|slot| (1000 + slot, slot, 2, &[11][..]))
        .collect::<Vec<_>>();
    let before = host.launches.borrow().len();
    run(&host, &mut instance, "decode", slots_step(&decode));
    instance.commit().unwrap();
    let names = launched(&host, before);
    let wide_env = std::env::var(crate::launch::WIDE_COMM).as_deref() != Ok("0");
    let reduces = 4 * 2 * shape.layers() as usize;
    if wide_env {
        assert_eq!(count(&names, F32_WIDE), reduces);
    } else {
        assert_eq!(count(&names, F32_COMM[1]), reduces);
    }
    for symbol in bf16_comm() {
        assert_eq!(count(&names, symbol), 0, "{symbol}");
    }
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
}

/// Each consumer binds by its own view: F32 PLE norms and one layer's MLP
/// norm beside BF16 hyper-connection norms everywhere else.
#[test]
fn each_consumer_binds_the_dtype_of_the_view_it_reads() {
    let shape = Shape::TP4;
    let ple = shape.ple_layer();
    let norms = ["norm_key", "norm_conv", "norm_query"]
        .map(|norm| format!("{MODEL}.layers.{ple}.ple.{norm}.weight"))
        .into_iter()
        .chain([format!(
            "{MODEL}.layers.5.mlp_hyper_connection.hc_norm.weight"
        )])
        .collect();
    let host = prepared(SERVED, norms);
    let mut instance = create(&host);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 2, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    let names = launched(&host, 0);
    // PLE key and convolution norms run F32; FirstNorm and the PLE layer's
    // hyper-connection norm stay BF16.
    assert_eq!(count(&names, F32_NORM), 4 * 2);
    assert_eq!(count(&names, BF16_NORM), 4 * 2);
    // The all-reduce after layer 0's MLP applies PLE's query norm and the
    // one after layer 5's attention layer 5's MLP norm.
    let reduces = 2 * shape.layers() as usize;
    assert_eq!(count(&names, F32_COMM[0]), 4 * 2);
    assert_eq!(
        count(&names, model_qwen38_contract::sm120::comm::SYMBOL),
        4 * (reduces - 2)
    );
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
}

/// Prepared and legacy images without F32 norm weights resolve no F32 form
/// and launch the BF16 entries as before.
#[test]
fn images_without_f32_norms_load_no_f32_form() {
    for precision in [false, true] {
        let host = Rc::new(FakeHost::default());
        if precision {
            *host.precision.borrow_mut() = Some(super::precision::document_for(
                Shape::TP4,
                true,
                true,
                false,
            ));
        }
        let mut instance = create(&host);
        let names = resolved(&host);
        for symbol in [F32_NORM, F32_WIDE]
            .into_iter()
            .chain(F32_COMM)
            .chain(F32_DECODE)
        {
            assert!(!names.contains(symbol), "{symbol}");
        }
        run(
            &host,
            &mut instance,
            "prefill",
            step(&[(10, 2, 0, &PROMPT)]),
        );
        instance.commit().unwrap();
        let launched = launched(&host, 0);
        assert_eq!(count(&launched, BF16_NORM), 4 * 4);
        assert_eq!(count(&launched, F32_NORM), 0);
        instance.destroy().unwrap();
        assert_eq!(host.released.get(), host.acquired.get());
    }
}

/// F32 views are admitted only for norm weights with F32 consumers.
#[test]
fn f32_views_without_an_f32_consumer_fail_create() {
    let host = prepared(
        SERVED,
        BTreeSet::from([format!("{MODEL}.layers.0.linear_attn.norm.weight")]),
    );
    let config = configuration_with(
        false,
        &[model_qwen38_contract::prepared::COMPONENT_ROLE
            .as_bytes()
            .to_vec()],
    );
    assert!(
        api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &config)
            .is_err()
    );
    assert!(host.logs.borrow().iter().any(|line| line.contains(
        "view model.language_model.layers.0.linear_attn.norm.weight is F32 f32_row_major_v1"
    )));
    assert_eq!(host.released.get(), host.acquired.get());
}

/// Norm weights are replicated: ranks storing different dtypes fail create.
#[test]
fn ranks_storing_different_norm_dtypes_fail_create() {
    let host = prepared(SERVED, mxfp8_dense_norms(Shape::TP4));
    host.f32_norm_rank.set(Some(1));
    let config = configuration_with(
        false,
        &[model_qwen38_contract::prepared::COMPONENT_ROLE
            .as_bytes()
            .to_vec()],
    );
    assert!(
        api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &config)
            .is_err()
    );
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("rank 1 stores other norm weight dtypes than rank 0")),
        "{:?}",
        host.logs.borrow()
    );
    assert_eq!(host.released.get(), host.acquired.get());
}

/// A prepared image with the drafter admits F32 MTP hyper-connection norms
/// in its drafter views and its drafter launches their F32 forms; its
/// pre-FC norms stay BF16, and an unprepared image keeps every drafter norm
/// BF16.
#[test]
fn prepared_drafter_views_admit_f32_hyper_connection_norms() {
    let drafter = |precision: bool, norms: &[&str]| {
        let host = Rc::new(FakeHost::default());
        host.ngram_host.set(ngram_in_host_memory(Shape::TP4));
        host.drafter.set(true);
        host.speculation.set((3, MAX_VERIFY_TOKENS));
        if precision {
            *host.precision.borrow_mut() = Some(super::precision::document_for(
                Shape::TP4,
                false,
                false,
                false,
            ));
        }
        *host.f32_norms.borrow_mut() = norms.iter().map(|n| n.to_string()).collect();
        host
    };
    let hc = [
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight",
    ];
    let host = drafter(true, &hc);
    let mut instance = create(&host);
    assert!(resolved(&host).contains(F32_NORM));
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 0, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    let prefill = launched(&host, 0);
    // The drafter's layer after the target's prefill: its attention HC norm
    // row and the decode all-reduce applying the mixer's norm read F32 on
    // every rank, while the all-reduce applying its MLP norm and every
    // target consumer stay BF16.
    let bf16 = bf16_comm();
    assert_eq!(count(&prefill, F32_NORM), 4);
    assert_eq!(count(&prefill, F32_DECODE[1]), 4);
    assert_eq!(count(&prefill, bf16[5]), 4);
    assert_eq!(
        count(&prefill, bf16[0]),
        4 * 2 * Shape::TP4.layers() as usize
    );
    instance.destroy().unwrap();
    let refused = |host: &Rc<FakeHost>, view: &str| {
        let extra = if host.precision.borrow().is_some() {
            vec![
                model_qwen38_contract::prepared::COMPONENT_ROLE
                    .as_bytes()
                    .to_vec(),
            ]
        } else {
            vec![]
        };
        assert!(
            api()
                .create(
                    HostServicesTableV1::new(Rc::clone(host)),
                    &configuration_served(host.shape(), host.layout(), &extra, host.served()),
                )
                .is_err()
        );
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains(&format!("view {view} is F32 f32_row_major_v1"))),
            "{:?}",
            host.logs.borrow()
        );
        assert_eq!(host.released.get(), host.acquired.get());
    };
    let pre_fc = "mtp.pre_fc_norm_hidden.weight";
    refused(&drafter(true, &[pre_fc]), pre_fc);
    refused(&drafter(false, &hc), hc[0]);
}
