//! CPU checks over every model's catalogs (`kernel_lab::catalogs`):
//! properties that hold across the kernel base's and the models' entries,
//! which the shared crate's own tests cannot see.

use kernel_core::contract::{DenseKernel, RuntimeCounts, StaticQuery};
use kernels_sm120::gemm::{self, cluster, fused};
use kernels_sm120::moe::template::{self, Schedule, ScheduleKnobs};

fn capacity(v: &kernels_sm120::moe::Variant) -> RuntimeCounts {
    RuntimeCounts {
        m: v.tokens_capacity,
        batch: 0,
        context: 0,
        routed_rows: v.query.m_capacity,
        split_k: v.knobs.split_k_capacity,
    }
}

#[test]
fn every_served_a16_entry_has_one_template_instance() {
    let instances = template::lab_instances(kernel_lab::catalogs());
    // MiMo at two rank counts: 2 projections x (5 decode + 1 prefill) and
    // 3 exact entries; Qwen3.8 at two rank counts: 2 x 7.
    assert_eq!(instances.len(), 2 * 12 + 2 * 3 + 2 * 14);
    let symbols = instances
        .iter()
        .map(|i| i.template_symbol.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(symbols.len(), instances.len());
    for instance in &instances {
        assert_eq!(instance.template_symbol, instance.baseline.symbol);
        assert!(instance.engine_symbol.starts_with(template::ENGINE_PREFIX));
        assert_eq!(
            instance
                .engine_symbol
                .replacen(template::ENGINE_PREFIX, "chop_moe_", 1),
            instance.baseline.symbol
        );
    }
    let count = |schedule| instances.iter().filter(|i| i.schedule == schedule).count();
    // Ring: Qwen 24; MiMo gate/up and down 1-16 and the three exact
    // entries at two rank counts.
    assert_eq!(count(Schedule::Ring), 24 + 2 * (10 + 3));
    assert_eq!(count(Schedule::Prefill), 8);
}

#[test]
fn schedule_table_is_canonical_and_keyed_by_the_served_entries() {
    let table = kernel_lab::catalogs().schedules();
    assert_eq!(
        kernel_build::tuning::Table::parse(&table.render()).as_ref(),
        Ok(table)
    );
    let served = template::served_table(kernel_lab::catalogs());
    // Packed ring rows (`packed_lab_instances`) have their own schedule;
    // the others are keyed by the served template entries.
    let (packed_rows, template_rows): (Vec<_>, Vec<_>) = table
        .rows()
        .iter()
        .partition(|r| r.schedule == "packed_ring");
    let keys = template_rows
        .iter()
        .map(|r| r.key.clone())
        .collect::<Vec<_>>();
    let served_keys = served
        .rows()
        .iter()
        .map(|r| r.key.clone())
        .collect::<Vec<_>>();
    assert_eq!(keys, served_keys);
    assert_eq!(
        template_rows.len(),
        template::lab_instances(kernel_lab::catalogs()).len()
    );
    let packed_keys = model_qwen38_contract::sm120::moe::packed_lab_instances()
        .iter()
        .map(|i| i.key.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let rows = packed_rows
        .iter()
        .map(|r| r.key.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(rows, packed_keys);
    for (row, served) in template_rows.into_iter().zip(served.rows()) {
        let knobs = ScheduleKnobs::from_row(row).unwrap();
        let served = ScheduleKnobs::from_row(served).unwrap();
        assert_eq!(knobs.arithmetic(), served.arithmetic(), "{}", row.key);
    }
}

#[test]
fn instance_bodies_carry_the_served_shape_and_schedule() {
    // Bodies at the engines' schedules, whatever the table rows hold now.
    let instances = template::engine_schedule_instances(kernel_lab::catalogs());
    let find = |symbol: &str| {
        instances
            .iter()
            .find(|i| i.baseline.symbol == symbol)
            .unwrap_or_else(|| panic!("missing {symbol}"))
    };
    let qwen = find("chop_moe_qwen38_gateup_decode_fp8block_a16_t1_v1");
    assert!(qwen.body.contains("Weights::fp8_block128"), "{}", qwen.body);
    assert!(
        qwen.body
            .contains(".experts = 128, .top_k = 10, .hidden = 2560, .intermediate = 640")
    );
    assert!(
        qwen.body.contains(".split_k = 10, .warps = 10"),
        "{}",
        qwen.body
    );
    assert!(
        qwen.body
            .contains(".early_row = false, .reload_rows = true")
    );
    let mimo = find("chop_moe_mimo_gateup_decode_mxfp4_a16_t1_v1");
    assert!(
        mimo.body
            .contains(".early_row = true, .reload_rows = false"),
        "{}",
        mimo.body
    );
    assert!(
        mimo.body.contains(".split_k = 16, .warps = 16"),
        "{}",
        mimo.body
    );
    let bulk = find("chop_moe_qwen38_gateup_prefill_fp8block_a16_t8192_v1");
    assert!(
        bulk.body
            .contains(".trim_gather = true, .bulk_weights = true")
    );
    let tp2 = find("chop_moe_qwen38_tp2_gateup_prefill_fp8block_a16_t8192_v1");
    assert!(
        tp2.body
            .contains(".trim_gather = false, .bulk_weights = false")
    );
    // MiMo trims the gather at TP4 (gate/up N = 1024), as `run_fragment` does.
    let mimo = find("chop_moe_mimo_gateup_prefill_mxfp4_a16_t8192_v1");
    assert!(
        mimo.body
            .contains(".trim_gather = true, .bulk_weights = false")
    );
    let mimo_tp2 = find("chop_moe_mimo_tp2_gateup_prefill_mxfp4_a16_t8192_v1");
    assert!(
        mimo_tp2
            .body
            .contains(".trim_gather = false, .bulk_weights = false")
    );
}

#[test]
fn ring_candidates_resize_grid_block_and_rings() {
    // The served schedule, whatever the table row holds.
    let instance = template::engine_schedule_instances(kernel_lab::catalogs())
        .into_iter()
        .find(|i| i.baseline.symbol == "chop_moe_qwen38_gateup_decode_fp8block_a16_t8_v1")
        .unwrap();
    let ScheduleKnobs::Ring(r) = instance.knobs else {
        unreachable!()
    };
    let wide = template::candidates(&instance)
        .into_iter()
        .find(
            |c| matches!(c.knobs, ScheduleKnobs::Ring(k) if k.warps == 2 * r.warps && k.depth == 2),
        )
        .unwrap();
    let counts = capacity(&instance.baseline);
    let (narrow, wide_plan) = (instance.plan(counts).unwrap(), wide.plan(counts).unwrap());
    assert_eq!(wide_plan.grid[0] * 2, narrow.grid[0]);
    assert_eq!(wide_plan.block[0], 2 * narrow.block[0]);
    // 2 parts x 1024 bytes and eight 144-byte rows per slot.
    assert_eq!(
        wide_plan.dynamic_shared_bytes,
        2 * r.warps * 2 * (2048 + 8 * 144)
    );
    assert!(
        wide.body
            .contains(&format!(".warps = {}, .depth = 2", 2 * r.warps))
    );
}

/// Writes the rows that reproduce the served schedules to
/// `CHOPINFER_TUNING_TABLE_OUT`, for a new served model's initial rows.
#[test]
#[ignore = "writes a file; set CHOPINFER_TUNING_TABLE_OUT"]
fn write_served_table() {
    let path = std::env::var_os("CHOPINFER_TUNING_TABLE_OUT")
        .expect("CHOPINFER_TUNING_TABLE_OUT names the output file");
    std::fs::write(
        path,
        template::served_table(kernel_lab::catalogs()).render(),
    )
    .unwrap();
}

#[test]
fn every_served_cluster_decode_entry_has_one_instance() {
    let instances = cluster::lab_instances(kernel_lab::catalogs());
    let qwen = fused::catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|e| matches!(e.kind, fused::Kind::ClusterDecode { .. }))
        .count();
    let hc = fused::catalog(kernel_lab::catalogs())
        .iter()
        .filter(|e| fused::hc_precompute_spec(e).is_ok())
        .count();
    let shared = gemm::catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|v| v.profile.model().is_some())
        .filter(|v| matches!(v.key.dense, Some(DenseKernel::FragmentCluster { .. })))
        .count();
    // TP4's ten launches and TP2's five, at six buckets each.
    assert_eq!(qwen, 6 * (10 + 5));
    assert_eq!(hc, 4);
    assert_eq!(instances.len(), qwen + hc + shared);
    let mut symbols: Vec<_> = instances.iter().map(|i| i.symbol().to_owned()).collect();
    symbols.sort();
    symbols.dedup();
    assert_eq!(symbols.len(), instances.len());
    for item in &instances {
        let qwen = item.profile().model() == Some(model_qwen38_contract::sm120::config::MODEL);
        assert_eq!(item.serves_template(), qwen);
        if qwen {
            // The served entry is the template; kernel-lab keeps the old
            // kernel as the reference.
            assert_eq!(item.symbol(), item.served_symbol);
            assert_eq!(item.template_cubin, item.served_cubin);
            assert_eq!(
                item.reference_symbol,
                cluster::reference_symbol(&item.served_symbol)
            );
            assert!(item.reference_cubin.starts_with("gemm_eng_qwen38_"));
        } else {
            assert_eq!(item.symbol(), cluster::template_symbol(&item.served_symbol));
            assert_eq!(item.reference_symbol, item.served_symbol);
            assert!(
                item.template_cubin.starts_with("gemm_tpl_mimo"),
                "{}",
                item.template_cubin
            );
        }
    }
}

/// The lab-only split variants keep the 174 multi-split fragment decode
/// launches kernel-lab ran before MiMo's and Qwen3.8's entries declared one
/// split: each plans for its splits, which its served entry refuses, from
/// the served entry's compiled wrapper.
#[test]
fn lab_split_variants_keep_the_former_reductions_on_served_wrappers() {
    let served = gemm::catalog(kernel_lab::catalogs());
    let lab = gemm::lab_split_catalog(kernel_lab::catalogs());
    assert_eq!(lab.len(), 102);
    for variant in &lab {
        let entry = served.iter().find(|v| v.symbol == variant.symbol).unwrap();
        let splits = gemm::required_split_k(&variant.key).unwrap();
        assert!(splits > 1, "{}", variant.symbol);
        assert_eq!(gemm::required_split_k(&entry.key), Some(1));
        let counts = |split_k| RuntimeCounts {
            m: 1,
            batch: 1,
            context: 0,
            routed_rows: 0,
            split_k,
        };
        assert!(
            gemm::plan(&variant.key, counts(splits)).is_ok(),
            "{}",
            variant.symbol
        );
        assert!(
            gemm::plan(&entry.key, counts(splits)).is_err(),
            "{}",
            variant.symbol
        );
        assert_eq!(
            format!(
                "{:?}",
                gemm::cubin_spec(std::slice::from_ref(&variant.instantiation)).unwrap()
            ),
            format!(
                "{:?}",
                gemm::cubin_spec(std::slice::from_ref(&entry.instantiation)).unwrap()
            ),
        );
    }
}

/// The medium-M kernels serve the models' fragment prefill
/// references: a Qwen3.8 TP2 bundle's cubin holds the kernels its own and
/// TP4's rank-invariant references select.
#[test]
fn mid_kernels_serve_model_references() {
    let e = gemm::mid::select(
        kernel_lab::catalogs(),
        "chop_gemm_qwen38_shared_gate_up_prefill_v1",
        64,
    )
    .unwrap();
    assert_eq!(e.kernel.symbol(), "chop_gemm_mid_bf16_swiglu_t64_v1");
    let qwen = gemm::mid::cubin_specs(
        kernel_lab::catalogs(),
        Some(model_qwen38_contract::sm120::config::PROFILE),
    );
    assert_eq!(qwen[0].entries.len(), 2);
    let tp2 = gemm::mid::cubin_spec_for(
        kernel_lab::catalogs(),
        "gemm_mid_qwen38_tp2_sm120",
        &[
            model_qwen38_contract::sm120::config::PROFILE,
            model_qwen38_contract::sm120::config::MODEL.at(2),
        ],
    )
    .unwrap();
    assert_eq!(tp2.entries.len(), 2);
}

/// The models' launch-split entries (skinny prefill splits)
/// cannot compile as wrappers without declared arithmetic.
#[test]
fn model_launch_split_entries_require_their_arithmetic() {
    let cubin = gemm::fused::cubin_specs(
        kernel_lab::catalogs(),
        Some(model_qwen38_contract::sm120::config::PROFILE),
    )
    .remove(0);
    assert!(kernels_sm120::arithmetic::DeclaredCubin::compiled(cubin).is_err());
}

/// FP8 fragment entries scale the MMA's partial sums: no code-then-scale
/// capability is stated for them.
#[test]
fn fp8_fragment_entries_offer_no_code_then_scale_capability() {
    use kernel_core::precision::{ExecutionPhase, OperationUse};
    let catalog = kernels_sm120::moe::catalog(kernel_lab::catalogs());
    let fp8 = catalog
        .iter()
        .find(|v| {
            v.layout == kernels_sm120::moe::WeightLayout::Fp8Fragment
                && v.path == kernels_sm120::moe::InputPath::A16Gather
        })
        .unwrap();
    let usage = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: fp8.tokens_capacity.min(4),
        graph: true,
    };
    assert!(
        kernels_sm120::precision::catalog_grouped_capability(fp8, usage)
            .unwrap_err()
            .0
            .contains("MXFP4 fragment entries only")
    );
}

/// The raw contributions, before repeats merge: a repeated symbol is one
/// entry listed twice with identical claims, and no symbol is contributed
/// by two models or by a model and the kernel base (the medium-M kernels
/// models reference are the base's). Every served cubin holds distinct
/// symbols under a distinct name.
#[test]
fn every_contributed_symbol_has_one_owner() {
    use kernels_sm120::Catalogs;
    use kernels_sm120::invariance::contributions;
    use std::collections::{BTreeMap, BTreeSet};
    let all = contributions(kernel_lab::catalogs());
    let mut claims = BTreeMap::new();
    for d in &all {
        let claim = (d.op, d.invariance);
        assert_eq!(
            *claims.entry(d.symbol.as_str()).or_insert(claim),
            claim,
            "{}",
            d.symbol
        );
    }
    let symbols = |catalogs: &Catalogs| {
        contributions(catalogs)
            .into_iter()
            .map(|d| d.symbol)
            .collect::<BTreeSet<_>>()
    };
    // The medium-M kernels are the kernel base's: models' references
    // select them by symbol.
    let mut base = symbols(&Catalogs::base());
    base.extend(kernels_sm120::gemm::mid::Kernel::ALL.map(|kernel| kernel.symbol().to_string()));
    let mut owned = base.clone();
    for model in kernel_lab::catalogs().models() {
        let own = symbols(&Catalogs::new(&[*model]).unwrap());
        let added = own.difference(&base).cloned().collect::<BTreeSet<_>>();
        // A model adds catalog entries, or only generated projection offers.
        assert!(
            !added.is_empty() || !(model.projections)().unwrap().0.is_empty(),
            "{}",
            model.name
        );
        assert!(owned.is_disjoint(&added), "{}", model.name);
        owned.extend(added);
    }
    let contributed = claims
        .keys()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>();
    assert!(contributed.is_subset(&owned));
    assert!(contributed.is_superset(&owned.difference(&base).cloned().collect()));
    let cubins = kernel_lab::catalogs().cubins();
    let names = cubins
        .iter()
        .map(|c| c.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), cubins.len());
    let entries = cubins.iter().map(|c| c.entries.len()).sum::<usize>();
    let distinct = cubins
        .iter()
        .flat_map(|c| c.entries.iter().map(|e| e.symbol.as_str()))
        .collect::<BTreeSet<_>>();
    assert_eq!(distinct.len(), entries);
}

/// Two catalogs cannot declare one model name or one entry symbol.
#[test]
fn catalogs_refuse_duplicate_models() {
    use kernels_sm120::{Catalogs, ModelCatalog};

    let &[mimo, qwen, ..] = kernel_lab::MODELS else {
        unreachable!("the lab's production models")
    };
    assert!(Catalogs::new(&[mimo, mimo]).is_err());
    static RENAMED: std::sync::OnceLock<ModelCatalog> = std::sync::OnceLock::new();
    let renamed = RENAMED.get_or_init(|| ModelCatalog {
        name: "renamed",
        ..*qwen
    });
    assert!(Catalogs::new(&[qwen, renamed]).is_err());
    assert!(Catalogs::new(&[mimo, qwen]).is_ok());
}

#[test]
fn every_mid_reference_is_a_whole_k_prefill_entry() {
    let entries = gemm::mid::catalog(kernel_lab::catalogs());
    let references = kernel_lab::catalogs()
        .models()
        .iter()
        .map(|model| (model.gemm_mid)().len())
        .sum::<usize>();
    assert_eq!(entries.len(), references);
    assert!(
        entries
            .iter()
            .any(|e| e.profile.model() == Some(model_qwen38_contract::sm120::config::MODEL))
    );
    assert!(
        entries
            .iter()
            .any(|e| e.profile.model() == Some(model_mimo26_contract::sm120::config::MODEL))
    );
    for e in &entries {
        assert_eq!(e.query.m_capacity, 8192, "{}", e.reference);
        assert!(e.plan(e.max_rows()).is_ok(), "{}", e.reference);
    }
}

/// The template runs at the served launch plan: its shared memory,
/// cluster and workspace follow from the same shape.
#[test]
fn cluster_instances_fit_the_served_plans() {
    use cluster::{Served, WARPS};
    for item in cluster::lab_instances(kernel_lab::catalogs()) {
        let s = item.shape;
        let plan = match &item.served {
            Served::Fused(entry) | Served::FusedHcPrecompute(entry) => {
                entry.plan(entry.query.m_capacity).unwrap()
            }
            Served::Shared(variant) => {
                let StaticQuery::Gemm(q) = variant.key.query else {
                    unreachable!()
                };
                gemm::plan(
                    &variant.key,
                    RuntimeCounts {
                        m: q.m_capacity,
                        batch: 0,
                        context: 0,
                        routed_rows: 0,
                        split_k: 1,
                    },
                )
                .unwrap()
            }
        };
        let symbol = &item.served_symbol;
        assert_eq!(
            plan.dynamic_shared_bytes,
            s.dynamic_shared_bytes(),
            "{symbol}"
        );
        assert_eq!(plan.cluster, [s.splits(), 1, 1], "{symbol}");
        assert_eq!(plan.block, [WARPS * 32, 1, 1], "{symbol}");
        assert_eq!(item.template.max_threads, WARPS * 32, "{symbol}");
        assert_eq!(plan.grid[0] % s.splits(), 0, "{symbol}");
        if s.splits() > 1 {
            assert_eq!(
                plan.workspace_bytes,
                u64::from(plan.grid[0] * WARPS * 16 * s.token_tile * 4),
                "{symbol}"
            );
        }
    }
}

/// MiMo's kernel fixes 16 fragment tiles per warp and needs the cluster
/// to divide the tile rows; the template keeps that arithmetic.
#[test]
fn shared_cluster_instances_keep_sixteen_tiles_per_warp() {
    let shared: Vec<_> = cluster::lab_instances(kernel_lab::catalogs())
        .into_iter()
        .filter(|i| matches!(i.served, cluster::Served::Shared(_)))
        .collect();
    assert!(!shared.is_empty());
    for item in shared {
        let s = item.shape;
        assert_eq!(s.warp_blocks * s.weights.tiles_per_block(), 16);
        assert_eq!((s.passes, s.segments), (1, 1));
        let rows = s.strips * 16 / if s.swiglu { 2 } else { 1 };
        assert_eq!(rows % s.splits(), 0, "{}", item.served_symbol);
    }
}

/// A model's schedule rows must belong to its schedule keys: MiMo's table
/// without its entries, Qwen3.8's without its packed ring keys, and either
/// table with keys that omit its template entries are refused.
#[test]
fn schedule_tables_hold_only_their_models_keys() {
    use kernel_core::contract::Refusal;
    use kernels_sm120::moe::template;
    use kernels_sm120::{Catalogs, ModelCatalog};

    let &[mimo, qwen, ..] = kernel_lab::MODELS else {
        unreachable!("the lab's production models")
    };
    static ORPHAN: std::sync::OnceLock<ModelCatalog> = std::sync::OnceLock::new();
    let orphan = ORPHAN.get_or_init(|| ModelCatalog {
        name: "orphan",
        gemm: Vec::new,
        gemm_fused: Vec::new,
        gemm_mid: Vec::new,
        gemm_lab_splits: &[],
        moe: Vec::new,
        moe_routes: Vec::new,
        moe_route_cubins: |_| Vec::new(),
        declared: Vec::new,
        schedule_keys: Vec::new,
        ..*mimo
    });
    assert_eq!(
        Catalogs::new(&[orphan]).err(),
        Some(Refusal("a schedule row names no entry of its model"))
    );
    static UNPACKED: std::sync::OnceLock<ModelCatalog> = std::sync::OnceLock::new();
    let unpacked = UNPACKED.get_or_init(|| ModelCatalog {
        schedule_keys: || template::schedule_keys(&model_qwen38_contract::sm120::moe::catalog()),
        ..*qwen
    });
    assert_eq!(
        Catalogs::new(&[unpacked]).err(),
        Some(Refusal("a schedule row names no entry of its model"))
    );
    static KEYLESS: std::sync::OnceLock<ModelCatalog> = std::sync::OnceLock::new();
    let keyless = KEYLESS.get_or_init(|| ModelCatalog {
        schedule_keys: Vec::new,
        schedules: include_str!("../../kernels-sm120/tuning/schedules.json"),
        schedule_table: None,
        ..*mimo
    });
    assert_eq!(
        Catalogs::new(&[keyless]).err(),
        Some(Refusal("a model's schedule keys omit a template entry"))
    );
    assert!(Catalogs::new(&[mimo, qwen]).is_ok());
}

#[test]
fn every_symbol_declares_once_and_claims_follow_the_fields() {
    use kernels_sm120::invariance::{Anchors, Invariance, Property, Ties, declarations};
    let all = declarations(kernel_lab::catalogs());
    assert!(all.len() > 500, "{} declarations", all.len());
    let symbols = all
        .iter()
        .map(|d| d.symbol.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(symbols.len(), all.len());
    assert_eq!(Invariance::NONE.claims(), []);
    assert_eq!(
        Invariance::PREFILL.claims(),
        [Property::Chunk, Property::Batch]
    );
    assert_eq!(
        Invariance::one_row(8)
            .with_ties(Ties::LowerIndex)
            .with_anchors(Anchors::Spare)
            .claims(),
        [
            Property::Rows,
            Property::Batch,
            Property::Ties,
            Property::Anchors
        ]
    );
}
