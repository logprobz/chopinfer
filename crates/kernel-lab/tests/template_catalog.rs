//! The grouped-MoE template over every model's served entries and schedule
//! rows (`kernel_lab::catalogs`): rows, instances, candidates and the packed
//! ring's rows. CPU only.

use kernel_build::tuning::{Knob, Row};
use kernel_core::contract::*;
use kernels_sm120::Catalogs;
use kernels_sm120::moe::template::*;
use kernels_sm120::moe::*;

fn cat() -> &'static Catalogs {
    kernel_lab::catalogs()
}

fn capacity(v: &Variant) -> RuntimeCounts {
    RuntimeCounts {
        m: v.tokens_capacity,
        batch: 0,
        context: 0,
        routed_rows: v.query.m_capacity,
        split_k: v.knobs.split_k_capacity,
    }
}

#[test]
fn rows_parse_back_to_their_knobs_and_refuse_strays() {
    for row in served_table(cat()).rows() {
        let knobs = ScheduleKnobs::from_row(row).unwrap();
        assert_eq!(knobs.row_knobs(), row.knobs);
        assert_eq!(knobs.schedule().name(), row.schedule);
        let mut stray = row.clone();
        stray.knobs.insert("tile_m".into(), Knob::Count(16));
        assert!(ScheduleKnobs::from_row(&stray).is_err());
        let mut missing = row.clone();
        missing.knobs.pop_first();
        assert!(ScheduleKnobs::from_row(&missing).is_err());
        let mut other = row.clone();
        other.schedule = "tiles".into();
        assert!(ScheduleKnobs::from_row(&other).is_err());
    }
}

/// The template implements one route weight point and no SwiGLU
/// clamps; an entry declaring others has no template row.
#[test]
fn template_refuses_other_declared_expert_numerics() {
    let v = catalog(cat())
        .into_iter()
        .find(|v| schedule(v).is_some())
        .unwrap();
    assert!(row_knobs(&v, cat().schedules()).is_ok());
    let clamped = Variant {
        swiglu_clamp: Some(SwigluClamp {
            gate_max: 10.0,
            up_limit: 10.0,
        }),
        ..v.clone()
    };
    let before = Variant {
        route_weight: RouteWeight::BeforeDown,
        ..v
    };
    for v in [clamped, before] {
        let refusal = row_knobs(&v, cat().schedules()).unwrap_err();
        assert!(refusal.contains("grouped engines"), "{refusal}");
    }
}

#[test]
fn served_knobs_keep_the_served_plan() {
    for v in catalog(cat()) {
        let Some(knobs) = served_knobs(&v) else {
            continue;
        };
        let served = instance(v.clone(), knobs);
        for m in [1, v.tokens_capacity] {
            let counts = RuntimeCounts {
                m,
                routed_rows: m * v.query.grouped.unwrap().top_k,
                ..capacity(&v)
            };
            assert_eq!(
                served.plan(counts).unwrap(),
                v.engine_plan(counts).unwrap(),
                "{}",
                v.symbol
            );
        }
    }
}

#[test]
fn candidates_keep_the_arithmetic_and_fit() {
    let instances = lab_instances(cat());
    let mut total = 0;
    for instance in &instances {
        let candidates = candidates(instance);
        assert!(!candidates.is_empty(), "{}", instance.baseline.symbol);
        let mut previous: Option<ScheduleKnobs> = None;
        for candidate in &candidates {
            assert_ne!(candidate.knobs, instance.knobs);
            assert_eq!(candidate.knobs.arithmetic(), instance.knobs.arithmetic());
            assert!(previous.is_none_or(|p| p < candidate.knobs));
            previous = Some(candidate.knobs);
            assert!(candidate.template_symbol.starts_with(TUNE_PREFIX));
            assert!(candidate.template_symbol.ends_with("_v1"));
            assert!(candidate.template_cubin.starts_with("tune_"));
            let plan = candidate.plan(capacity(&candidate.baseline)).unwrap();
            assert!(plan.dynamic_shared_bytes <= SHARED_LIMIT);
            let spec = candidate.cubin_spec();
            assert_eq!(spec.entries.len(), 1);
            assert_eq!(spec.entries[0].max_threads, plan.block[0]);
        }
        total += candidates.len();
    }
    let symbols = instances
        .iter()
        .flat_map(candidates)
        .map(|c| c.template_symbol)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(symbols.len(), total);
    // A knob set that changes the K splits is not a candidate.
    let ring = instances
        .iter()
        .find(|i| i.schedule == Schedule::Ring && i.baseline.projection == Projection::GateUp)
        .unwrap();
    let ScheduleKnobs::Ring(mut split) = ring.knobs else {
        unreachable!()
    };
    split.split_k /= 2;
    assert!(ring.with_knobs(ScheduleKnobs::Ring(split)).is_err());
}

/// Every served grouped entry of an instance is the template's at its
/// table row: its body, the entry ABI, the thread count of its knobs
/// and the launch plan of its knobs.
#[test]
fn served_entries_are_template_instances() {
    let served = cubin_specs(cat(), None);
    for instance in lab_instances(cat()) {
        let (spec, entry) = served
            .iter()
            .flat_map(|spec| spec.entries.iter().map(move |entry| (spec, entry)))
            .find(|(_, entry)| entry.symbol == instance.baseline.symbol)
            .expect("served entry");
        assert_eq!(spec.name, instance.template_cubin);
        assert_eq!(entry.body, instance.body);
        assert_eq!(entry.params, parameters());
        assert_eq!(
            entry.max_threads,
            threads(&instance.knobs),
            "{}",
            instance.baseline.symbol
        );
        assert!(
            spec.headers
                .contains(&instance.schedule.header().to_string())
        );
        let v = &instance.baseline;
        let counts = capacity(v);
        assert_eq!(v.plan(counts).unwrap(), instance.plan(counts).unwrap());
    }
}

/// Four K64 stages reuse a route slot with four activation stages.
/// Three activation stages are legal, and every served row stays legal.
#[test]
fn prefill_rings_that_reuse_a_route_slot_are_illegal() {
    let mut baseline = lab_instances(cat())
        .into_iter()
        .find(|i| {
            i.baseline.projection == kernels_sm120::moe::Projection::Down
                && i.schedule == Schedule::Prefill
        })
        .unwrap()
        .baseline;
    baseline.query.k = 256;
    let knobs = |activation_stages| {
        ScheduleKnobs::Prefill(kernels_sm120::moe::template::PrefillKnobs {
            weight_stages: 2,
            activation_stages,
            trim_gather: false,
            bulk_weights: false,
        })
    };
    assert!(legality(&baseline, &knobs(4)).is_err());
    assert_eq!(legality(&baseline, &knobs(3)), Ok(()));
    for instance in lab_instances(cat()) {
        assert_eq!(legality(&instance.baseline, &instance.knobs), Ok(()));
    }
}

/// A row whose launch plan does not fit its capacity is refused before
/// any cubin is generated: MiMo TP4 gate/up prefill with eight weight
/// and four activation stages compiles, but its rings exceed the shared
/// memory the planner admits. Every committed row passes.
#[test]
fn rows_whose_plans_do_not_fit_are_refused() {
    let v = catalog(cat())
        .into_iter()
        .find(|v| v.symbol == "chop_moe_mimo_gateup_prefill_mxfp4_a16_t8192_v1")
        .unwrap();
    let mut row = cat().schedules().require(&key(&v)).unwrap().clone();
    row.knobs.insert("weight_stages".into(), Knob::Count(8));
    row.knobs.insert("activation_stages".into(), Knob::Count(4));
    let refusal = validate_row(&v, &row).map(|_| ()).unwrap_err();
    assert!(refusal.contains("does not fit"), "{refusal}");
    for v in catalog(cat()).into_iter().filter(|v| schedule(v).is_some()) {
        validate_row(&v, cat().schedules().require(&key(&v)).unwrap()).unwrap();
    }
}

#[test]
fn packed_instances_share_the_engine_contract() {
    let instances = model_qwen38_contract::sm120::moe::packed_lab_instances();
    assert_eq!(instances.len(), 2 * 2 * DECODE_BUCKETS.len());
    for instance in &instances {
        let (engine, template) = (&instance.engine, &instance.template);
        assert!(template.packed_ring() && !engine.packed_ring());
        assert_eq!(engine.query, template.query);
        assert_eq!(engine.packed_contract, template.packed_contract);
        // The bank, tensor map and argument order are the engine's.
        assert_eq!(
            engine.weight_tensor_map(1).unwrap(),
            template.weight_tensor_map(1).unwrap()
        );
        assert_eq!(engine.weight_scale_layout(), template.weight_scale_layout());
        let map = engine.weight_tensor_map(16).unwrap();
        assert_eq!(
            engine.arguments([1; 15], map.clone()).unwrap(),
            template.arguments([1; 15], map).unwrap()
        );
        assert_eq!(template.knobs.split_k_capacity, 1);
        assert!(template.reduction_symbol().is_none());
        let counts = RuntimeCounts {
            m: template.tokens_capacity,
            batch: 1,
            context: 0,
            routed_rows: template.query.m_capacity,
            split_k: 1,
        };
        let plan = template.plan(counts).unwrap();
        assert_eq!(plan.workspace_bytes, 0);
        assert!(plan.dynamic_shared_bytes <= 101_376);
        let spec = cubin_spec(&instance.template_cubin, std::slice::from_ref(template)).unwrap();
        assert!(spec.entries[0].body.contains("packed_ring_decode<"));
        assert!(
            spec.headers
                .contains(&"#include \"moe/grouped_packed.cuh\"".to_string())
        );
        let engine_spec = cubin_spec(&instance.engine_cubin, std::slice::from_ref(engine)).unwrap();
        assert!(engine_spec.entries[0].body.starts_with("chop::moe::run<"));
        assert_eq!(
            engine_spec.headers,
            vec!["#include \"moe/moe.cuh\"".to_string()]
        );
        let nv = template.query.weight == WeightFormat::Nvfp4;
        assert_eq!(
            spec.entries[0].body.contains(".expert_globals = true"),
            nv,
            "{}",
            template.symbol
        );
    }
}

#[test]
fn packed_ring_knobs_refuse_other_uses() {
    let shape = model_qwen38_contract::sm120::config::PROFILE
        .rank_shape()
        .map(|rank| Shape::of(rank, model_qwen38_contract::sm120::config::MOE))
        .unwrap();
    let ring = |phase, tokens, weight, activation| {
        packed_ring_knobs(
            cat().schedules(),
            shape,
            Projection::GateUp,
            phase,
            tokens,
            weight,
            activation,
        )
    };
    assert!(
        ring(
            Phase::Decode,
            1,
            WeightFormat::Nvfp4,
            ActivationFormat::Bf16
        )
        .is_some()
    );
    assert!(
        ring(
            Phase::Decode,
            32,
            WeightFormat::Mxfp8,
            ActivationFormat::Bf16
        )
        .is_some()
    );
    assert!(
        ring(
            Phase::Prefill,
            8192,
            WeightFormat::Nvfp4,
            ActivationFormat::Bf16
        )
        .is_none()
    );
    assert!(
        ring(
            Phase::Decode,
            64,
            WeightFormat::Nvfp4,
            ActivationFormat::Bf16
        )
        .is_none()
    );
    assert!(
        ring(
            Phase::Decode,
            3,
            WeightFormat::Nvfp4,
            ActivationFormat::Bf16
        )
        .is_none()
    );
    assert!(
        ring(
            Phase::Decode,
            1,
            WeightFormat::Mxfp4,
            ActivationFormat::Bf16
        )
        .is_none()
    );
    assert!(
        ring(
            Phase::Decode,
            1,
            WeightFormat::Nvfp4,
            ActivationFormat::Nvfp4
        )
        .is_none()
    );
    // Split K would change the accumulation order; the ring never splits.
    let mut knobs = ring(
        Phase::Decode,
        1,
        WeightFormat::Nvfp4,
        ActivationFormat::Bf16,
    )
    .unwrap();
    knobs.split_k_capacity = 2;
    let request = packed_request(
        shape,
        Projection::GateUp,
        1,
        WeightFormat::Nvfp4,
        Some(knobs),
        "x_v1".into(),
    );
    assert!(instantiate(request).is_err());
}

/// Every packed_ring row passes the build's row checks, and the checks
/// refuse another schedule, stray or missing knobs, split parts on down,
/// too many warps and rings that do not fit.
#[test]
fn packed_rows_validate_and_refuse_strays() {
    let shape = model_qwen38_contract::sm120::config::PROFILE
        .rank_shape()
        .map(|rank| Shape::of(rank, model_qwen38_contract::sm120::config::MOE))
        .unwrap();
    for instance in model_qwen38_contract::sm120::moe::packed_lab_instances() {
        let t = &instance.template;
        let row = cat().schedules().require(&instance.key).unwrap();
        let check = |row: &Row| {
            validate_packed_row(shape, t.projection, t.tokens_capacity, t.query.weight, row)
        };
        assert_eq!(check(row), Ok(t.knobs), "{}", instance.key);
        let mut other = row.clone();
        other.schedule = "ring".into();
        assert!(check(&other).is_err());
        let mut stray = row.clone();
        stray.knobs.insert("split_k".into(), Knob::Count(2));
        assert!(check(&stray).is_err());
        let mut missing = row.clone();
        missing.knobs.remove("depth");
        assert!(check(&missing).is_err());
        let mut deep = row.clone();
        deep.knobs
            .insert("depth".into(), Knob::Count(t.query.k / 64 + 1));
        assert!(check(&deep).is_err());
        // More than 16 warps per CTA.
        let mut wide = row.clone();
        wide.knobs.insert("strips".into(), Knob::Count(20));
        assert!(check(&wide).is_err());
        // Rings past the shared-memory limit.
        let mut full = row.clone();
        full.knobs.insert("strips".into(), Knob::Count(8));
        full.knobs
            .insert("depth".into(), Knob::Count(t.query.k / 64));
        if t.tokens_capacity >= 16 {
            assert!(check(&full).is_err(), "{}", instance.key);
        }
        if t.projection == Projection::Down {
            let mut split = row.clone();
            split.knobs.insert("split_parts".into(), Knob::Flag(true));
            assert!(check(&split).is_err());
        }
    }
}

#[test]
fn packed_candidates_keep_the_arithmetic_and_fit() {
    for instance in model_qwen38_contract::sm120::moe::packed_lab_instances() {
        let candidates = packed_candidates(&instance);
        let ok = candidates
            .iter()
            .filter_map(|(_, c)| c.as_ref().ok())
            .collect::<Vec<_>>();
        assert!(ok.len() >= 10, "{}: {} candidates", instance.key, ok.len());
        let symbols = ok
            .iter()
            .map(|c| c.symbol.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(symbols.len(), ok.len());
        for candidate in ok {
            assert!(candidate.packed_ring());
            assert_eq!(candidate.knobs.split_k_capacity, 1);
            assert_eq!(candidate.query, instance.template.query);
            assert_ne!(candidate.knobs, instance.template.knobs);
            assert!(candidate.symbol.starts_with(TUNE_PREFIX));
        }
    }
}

/// Staged rows (one token) parse, build the staged body and refuse
/// other buckets, producer counts that do not divide the converted
/// ring and stray knobs.
#[test]
fn packed_staged_rows_build_the_staged_schedule() {
    let shape = model_qwen38_contract::sm120::config::PROFILE
        .rank_shape()
        .map(|rank| Shape::of(rank, model_qwen38_contract::sm120::config::MOE))
        .unwrap();
    for instance in model_qwen38_contract::sm120::moe::packed_lab_instances() {
        let t = &instance.template;
        let staged = PackedStaged {
            producers: 4,
            depth: 2,
            stages: 4,
        };
        let row = Row {
            key: instance.key.clone(),
            schedule: "packed_staged".into(),
            knobs: packed_staged_row_knobs(staged),
        };
        let check = |row: &Row| {
            validate_packed_row(shape, t.projection, t.tokens_capacity, t.query.weight, row)
        };
        if t.tokens_capacity != 1 {
            assert!(check(&row).is_err(), "{}", instance.key);
            continue;
        }
        let knobs = check(&row).unwrap();
        assert_eq!(knobs, staged.knobs());
        assert_eq!(packed_row(knobs), ("packed_staged", row.knobs.clone()));
        let op = instantiate(packed_request(
            shape,
            t.projection,
            1,
            t.query.weight,
            Some(knobs),
            "staged_v1".into(),
        ))
        .unwrap();
        assert!(op.packed_ring() && op.packed_staged());
        assert!(op.packed_ring_body().contains("packed_staged_decode<"));
        assert!(
            op.packed_ring_body()
                .contains(".producers = 4, .depth = 2, .stages = 4")
        );
        let plan = op
            .plan(RuntimeCounts {
                m: 1,
                batch: 1,
                context: 0,
                routed_rows: op.query.m_capacity,
                split_k: 1,
            })
            .unwrap();
        assert_eq!(plan.block[0], 5 * 32);
        let mut odd = row.clone();
        odd.knobs.insert("stages".into(), Knob::Count(6));
        assert!(check(&odd).is_err());
        let mut stray = row.clone();
        stray.knobs.insert("strips".into(), Knob::Count(1));
        assert!(check(&stray).is_err());
    }
}

/// A ring row may add an L2 prefetch size, which only rows that name one
/// carry into their body; other sizes are refused.
#[test]
fn packed_ring_prefetch_rows() {
    let shape = model_qwen38_contract::sm120::config::PROFILE
        .rank_shape()
        .map(|rank| Shape::of(rank, model_qwen38_contract::sm120::config::MOE))
        .unwrap();
    for instance in model_qwen38_contract::sm120::moe::packed_lab_instances() {
        let t = &instance.template;
        let row = cat().schedules().require(&instance.key).unwrap();
        if row.schedule != "packed_ring" {
            continue;
        }
        assert_eq!(
            t.packed_ring_body().contains(".prefetch"),
            row.knobs.contains_key("prefetch"),
            "{}",
            instance.key
        );
        let check = |row: &Row| {
            validate_packed_row(shape, t.projection, t.tokens_capacity, t.query.weight, row)
        };
        let mut fetch = row.clone();
        fetch.knobs.insert("prefetch".into(), Knob::Count(256));
        let knobs = check(&fetch).unwrap();
        assert_eq!(PackedRing::of(knobs).prefetch, 256);
        assert_eq!(packed_row(knobs), ("packed_ring", fetch.knobs.clone()));
        let op = instantiate(packed_request(
            shape,
            t.projection,
            t.tokens_capacity,
            t.query.weight,
            Some(knobs),
            "fetch_v1".into(),
        ))
        .unwrap();
        assert!(op.packed_ring_body().contains(".prefetch = 256}"));
        fetch.knobs.insert("prefetch".into(), Knob::Count(32));
        assert!(check(&fetch).is_err());
    }
}

/// Oversized strip and producer counts are refused before any
/// narrowing, in every build profile, instead of wrapping into a legal
/// schedule.
#[test]
fn packed_rows_refuse_oversized_counts_without_wrapping() {
    let shape = model_qwen38_contract::sm120::config::PROFILE
        .rank_shape()
        .map(|rank| Shape::of(rank, model_qwen38_contract::sm120::config::MOE))
        .unwrap();
    let instance = model_qwen38_contract::sm120::moe::packed_lab_instances()
        .into_iter()
        .find(|i| {
            i.template.query.weight == WeightFormat::Nvfp4
                && i.template.projection == Projection::GateUp
                && i.template.tokens_capacity == 1
        })
        .unwrap();
    let t = &instance.template;
    let row = cat().schedules().require(&instance.key).unwrap();
    let check = |row: &Row| {
        validate_packed_row(shape, t.projection, t.tokens_capacity, t.query.weight, row)
    };
    for strips in [0u32, 9, 17, 4097, 65535, 65536, u32::MAX] {
        let mut bad = row.clone();
        bad.knobs.insert("strips".into(), Knob::Count(strips));
        // Split parts double the warps; 9 strips need 18.
        assert!(check(&bad).is_err(), "{strips} strips");
    }
    let ring = PackedRing {
        strips: 4097,
        split_parts: true,
        depth: 8,
        prefetch: 0,
    };
    assert_eq!(ring.checked_knobs(), None);
    assert!(ring.knobs().warps > 16 && ring.knobs().tile_n != 16);
    let staged_row = |producers: u32| Row {
        key: instance.key.clone(),
        schedule: "packed_staged".into(),
        knobs: [
            ("depth".to_string(), Knob::Count(1)),
            ("producers".to_string(), Knob::Count(producers)),
            ("stages".to_string(), Knob::Count(producers.max(1))),
        ]
        .into(),
    };
    for producers in [0u32, 16, 255, 256] {
        assert!(
            check(&staged_row(producers)).is_err(),
            "{producers} producers"
        );
    }
    let staged = PackedStaged {
        producers: 255,
        depth: 1,
        stages: 255,
    };
    assert_eq!(staged.checked_knobs(), None);
    assert_eq!(staged.knobs().warps, u8::MAX);
}
