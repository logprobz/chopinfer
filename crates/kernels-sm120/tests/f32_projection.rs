use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{
    buffers::{ProjectionBuffers, ProjectionScratch, QuantizedWeights},
    *,
};
use kernels_sm120::{
    backend::{F32ProjectionOffers, Sm120},
    f32_projection::{self, Entry},
};

fn entry(weight: WeightFormat) -> Entry {
    Entry::new(
        GemmQuery {
            m_capacity: 17,
            n: 19,
            k: 257,
            weight,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 17,
            graph: true,
        },
        "exact_test",
    )
    .unwrap()
}
fn operation(e: &Entry) -> OperationSpec {
    let q = e.query();
    let f32_weight = q.weight == WeightFormat::F32;
    OperationSpec {
        id: "test".into(),
        kind: OperationKind::Dense,
        target: kernel_core::contract::Target::Sm120f,
        weights: vec![WeightSpec {
            codes: TensorSpan {
                tensor: "weight".into(),
                dtype: if f32_weight { DType::F32 } else { DType::Bf16 },
                shape: vec![q.n.into(), q.k.into()],
                offset: 0,
                byte_length: e.weight_bytes(),
            },
            logical_shape: vec![q.n.into(), q.k.into()],
            format: q.weight,
            encoding: if f32_weight {
                CodeEncoding::F32LittleEndian
            } else {
                CodeEncoding::Bf16LittleEndian
            },
            layout: WeightLayout::row_major(),
            local_scale: None,
            global_scale: None,
            transforms: vec![],
        }],
        geometry: None,
        activations: vec![ActivationSpec::f32()],
        arithmetic: e.capability().arithmetic,
        epilogues: q.epilogues,
        uses: vec![e.usage()],
    }
}
fn backend(entries: Vec<Entry>) -> Sm120 {
    let available = entries.iter().map(|e| e.symbol().into()).collect();
    Sm120::new()
        .with_f32_projections(F32ProjectionOffers::new(entries, available, "0".repeat(64)).unwrap())
}
fn buffers(e: &Entry) -> ProjectionBuffers<View> {
    let v = |handle, bytes| View { handle, bytes };
    let q = e.query();
    ProjectionBuffers {
        activation: v(1, u64::from(q.m_capacity) * u64::from(q.k) * 4),
        live_rows: v(2, 4),
        output: v(3, u64::from(q.m_capacity) * u64::from(q.n) * 4),
        residual: None,
        weights: QuantizedWeights {
            codes: v(4, e.weight_bytes()),
            local_scales: None,
            global_scales: None,
        },
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            partial: View::EMPTY,
            semaphores: View::EMPTY,
            unused: View::EMPTY,
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
        },
        row_offset: 0,
        shard_rows: 0,
    }
}
#[test]
fn f32_binding_covers_original_operands_without_scales_or_scratch() {
    for weight in [WeightFormat::Bf16, WeightFormat::F32] {
        let e = entry(weight);
        let op = operation(&e);
        let usage = e.usage();
        assert!(Sm120::new().projection(&op, usage, Need::rows(17)).is_err());
        let backend = backend(vec![e.clone()]);
        let prepared = backend.projection(&op, usage, Need::rows(17)).unwrap();
        assert_eq!(prepared.identity.stages, vec![e.symbol()]);
        assert_eq!(prepared.handle.info().workspace, 0);
        assert_eq!(
            prepared.handle.info().weight_layouts,
            vec![("weights".into(), WeightLayout::row_major())]
        );
        let counts = Counts {
            rows: 17,
            ..Counts::default()
        };
        let b = buffers(&e);
        let launches = prepared.handle.launches(&b, View::EMPTY, counts).unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].plan.grid, [2, 3, 1]);
        assert_eq!(launches[0].plan.block, [32, 1, 1]);
        assert!(launches[0].plan.pdl);
        assert_eq!(launches[0].arguments.0.len(), 5);
        for role in ["input", "output", "weight", "live"] {
            let mut bad = b;
            match role {
                "input" => bad.activation.bytes -= 1,
                "output" => bad.output.bytes -= 1,
                "weight" => bad.weights.codes.bytes -= 1,
                _ => bad.live_rows.bytes -= 1,
            }
            assert!(
                prepared.handle.launches(&bad, View::EMPTY, counts).is_err(),
                "{role}"
            );
        }
        let mut bad = b;
        bad.weights.local_scales = Some(View::EMPTY);
        assert!(prepared.handle.launches(&bad, View::EMPTY, counts).is_err());
        assert!(
            prepared
                .handle
                .launches(&b, View::EMPTY, Counts { rows: 18, ..counts })
                .is_err()
        );
        assert!(backend.projection(&op, usage, Need::rows(18)).is_err());
        let mut bad = op.clone();
        bad.activations[0] = ActivationSpec::bf16();
        assert!(backend.projection(&bad, usage, Need::rows(17)).is_err());
        let absent = Sm120::new().with_f32_projections(
            F32ProjectionOffers::new(vec![e.clone()], Default::default(), "0".repeat(64)).unwrap(),
        );
        assert!(absent.projection(&op, usage, Need::rows(17)).is_err());
        let mut other = e.query();
        other.activation = ActivationFormat::Bf16;
        assert!(Entry::new(other, usage, "invalid").is_err());
        let spec = f32_projection::cubin_spec("exact_test", &[e]).unwrap();
        assert_eq!(spec.headers, vec!["#include \"gemm/exact_f32.cuh\""]);
    }
}
#[test]
fn ambiguous_entries_refuse_and_duplicate_entries_collapse() {
    let e = entry(WeightFormat::Bf16);
    let op = operation(&e);
    let duplicate = backend(vec![e.clone(), e.clone()]);
    assert!(duplicate.projection(&op, e.usage(), Need::rows(17)).is_ok());
    let other = Entry::new(e.query(), e.usage(), "other_entry").unwrap();
    assert!(
        backend(vec![e.clone(), other])
            .projection(&op, e.usage(), Need::rows(17))
            .is_err()
    );
}

#[test]
fn conservative_tensor_guard_is_inside_the_exact_product_domain() {
    use kernel_core::precision::f32_gemm::{self, DotPath};
    // The CUDA interval endpoints, and low mantissa bits which require
    // all three components. Check every exponent pair in that interval.
    for a_exp in -30..=30 {
        for b_exp in -30..=30 {
            for mantissa in [0, 1, 0x8001, 0x7fffff] {
                let a = f32::from_bits(((a_exp + 127) as u32) << 23 | mantissa);
                let b = f32::from_bits(((b_exp + 127) as u32) << 23 | 0x7f0000);
                for sign in [1., -1.] {
                    assert_eq!(
                        f32_gemm::dot_path(&[a * sign], &[b], DType::Bf16).unwrap(),
                        DotPath::Bf16Expansion3
                    );
                }
            }
        }
    }
}

#[test]
fn dimensions_cannot_overflow_cuda_fold_indices() {
    let e = entry(WeightFormat::Bf16);
    let mut query = e.query();
    query.k = i32::MAX as u32;
    assert!(Entry::new(query, e.usage(), "overflow").is_err());
    query = e.query();
    query.n = u32::MAX;
    assert!(Entry::new(query, e.usage(), "overflow").is_err());
}
