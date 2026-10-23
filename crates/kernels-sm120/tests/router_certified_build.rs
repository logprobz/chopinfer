//! CPU check of the certified router projection and the certifying prefill
//! norm: their entries compile without spills or local memory, keep the
//! zero-component prefill's launch, and the norm keeps its served ABI with
//! one trailing pointer.
use kernel_build::{BuildOptions, CubinSpec, Toolchain};
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, RmsNormKind, WeightFormat,
};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::rmsnorm::RmsNormRequirement;
use kernel_core::rmsnorm_quantize::RmsNormQuantizeQuery;
use kernels_sm120::f32_projection::{Entry, certified, zero_prefill};
use kernels_sm120::row::rmsnorm_prefill::{self, Mutation};
use std::time::{SystemTime, UNIX_EPOCH};

fn router() -> Entry {
    Entry::new(
        GemmQuery {
            m_capacity: 8192,
            n: 384,
            k: 5120,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        },
        "test_router",
    )
    .unwrap()
}

fn norm(certify: bool, mutation: Mutation) -> rmsnorm_prefill::Entry {
    let query = RmsNormQuantizeQuery {
        norm: RmsNormRequirement::bf16_backend_chosen(5120, 1e-6, RmsNormKind::Plain, false),
        amax_floor_bits: 1e-4f32.to_bits(),
    };
    let mut e = rmsnorm_prefill::Entry::new(
        query,
        8192,
        format!(
            "test_norm_{}_{}_v1",
            u8::from(certify),
            format!("{mutation:?}").to_lowercase()
        ),
        true,
    )
    .unwrap();
    e.certify = certify;
    e.mutation = mutation;
    e.validate().unwrap();
    e
}

#[test]
fn certified_entries_keep_the_zero_launch_and_the_norm_abi() {
    let e = router();
    for rows in [1, 65, 513, 2048, 8191, 8192] {
        assert_eq!(
            certified::plan(&e, 32, rows).unwrap(),
            zero_prefill::plan(&e, 32, rows).unwrap()
        );
    }
    let spec = certified::entry_spec(&e, 32).unwrap();
    assert_eq!(spec.symbol, "test_router_v1_certified_m32_v1");
    assert_eq!(spec.max_threads, 128);
    let served = norm(false, Mutation::None).spec().unwrap();
    let certifying = norm(true, Mutation::None).spec().unwrap();
    assert_eq!(certifying.params[..served.params.len()], served.params[..]);
    assert_eq!(certifying.params.len(), served.params.len() + 1);
    let mut wrong = norm(false, Mutation::None);
    wrong.mutation = Mutation::CertifyAll;
    assert!(wrong.validate().is_err());
}

#[test]
fn certified_entries_compile_without_spills() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "CUDA is required for the certified router compile gate"
    );
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out = std::env::temp_dir().join(format!(
        "chopinfer-router-certified-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir(&out).unwrap();
    let projection =
        certified::cubin_spec("chop_router_certified_compile", &[router()], &[32, 64]).unwrap();
    let norms = rmsnorm_prefill::cubin_spec(
        "chop_norm_certified_compile".into(),
        &[norm(true, Mutation::None), norm(true, Mutation::CertifyAll)],
    )
    .unwrap();
    let specs: Vec<CubinSpec> = vec![projection, norms];
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out,
        &out.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile the certified router and norm");
    for cubin in &built {
        for entry in &cubin.entries {
            assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
            println!(
                "{} registers {} static shared {}",
                entry.symbol, entry.registers, entry.shared_static_bytes
            );
        }
    }
    std::fs::remove_dir_all(out).unwrap();
}
