//! Actual upper-skip references and wider feature tiles against qualified M32.
mod catalog;
mod fixture;
use catalog::*;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_harness::{Cubins, Session};
use kernels_sm120::f32_projection::{blocked, wide};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_F32_WIDE_CUBINS")
        .expect("staged F32 cubins required")
        .into()
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_WIDE_CUBINS"]
fn cpu_build_wide_prefill() {
    let specs = specs();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels.lock.d");
    let options = BuildOptions {
        emit_cargo_directives: false,
        profile_line_info: false,
        ..BuildOptions::from_env()
    };
    if options.lock_update {
        assert!(!dir.join(format!("{FAMILY}.lock")).exists());
    }
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let locks = LockDir::new(dir).family(FAMILY, specs.iter().map(|s| &s.name));
    kernel_build::build_in_families(&toolchain, &specs, &root(), &locks, options, None).unwrap();
}

#[test]
fn cpu_actual_rows_and_launch_inventory() {
    let entries = entries();
    let cases = cases();
    assert_eq!(entries.len(), 5);
    assert_eq!(cases.len(), 336);
    assert_eq!(cases.iter().filter(|c| c.timed()).count(), 224);
    let mut symbols = BTreeMap::<String, Vec<String>>::new();
    let mut launches = Vec::new();
    let mut add = |module: String, symbol: String, p: kernel_core::contract::LaunchPlan| {
        symbols
            .entry(module.clone())
            .or_default()
            .push(symbol.clone());
        launches.push(json!({"module":module,"symbol":symbol,"grid":p.grid,"block":p.block,"cluster":p.cluster,"shared":p.dynamic_shared_bytes,"attribute":p.dynamic_shared_bytes}));
    };
    for c in &cases {
        let e = &entries[c.entry];
        let q = e.query();
        let module = if c.wide {
            CUBIN.to_string()
        } else {
            baseline_module()
        };
        add(module, c.symbol(e), c.plan(e));
        if c.wide {
            add(
                baseline_module(),
                blocked::symbol(e, 32),
                blocked::plan(e, 32, c.rows).unwrap(),
            );
        } else {
            // These rows must compare with the serving generic entry, not a substituted tile.
            let served = bundle_deepseek41::sm120::f32_prefill::serving_entry(
                &model_deepseek41_contract::geometry::FLASH,
                e.clone().with_split_decode(),
            )
            .unwrap();
            let selected = served.entry_for_rows(c.rows);
            assert!(!selected.uses_prefill_tiles());
            assert!(c.rows > selected.decode_max_rows());
            add(
                generic(e).0.into(),
                e.baseline_symbol().into(),
                e.plan(c.rows).unwrap(),
            );
        }
        let prepared = e.clone().with_prefill_tiles(32).unwrap();
        add(
            baseline_module(),
            prepared.stage_symbol("_prepare"),
            stage_plan([c.rows, 1, 1]),
        );
        add(
            baseline_module(),
            prepared.stage_symbol("_weights"),
            stage_plan([q.n.div_ceil(16), 1, 1]),
        );
    }
    let e = entries.iter().find(|e| e.query().n == 384).unwrap();
    // The residency probe resolves all four compute variants for every shape.
    for entry in &entries {
        for (kind, tile) in [(false, 32), (false, 64), (true, 64), (true, 128)] {
            let (module, symbol, plan) = if kind {
                (
                    CUBIN.into(),
                    wide::symbol(entry, tile),
                    wide::plan(entry, tile, 8192),
                )
            } else {
                (
                    baseline_module(),
                    blocked::symbol(entry, tile),
                    blocked::plan(entry, tile, 8192),
                )
            };
            add(module, symbol, plan.unwrap());
        }
    }
    for (kind, tile) in [(false, 32), (false, 64), (true, 64), (true, 128)] {
        let symbol = if kind {
            wide::symbol(e, tile)
        } else {
            blocked::symbol(e, tile)
        };
        let plan = if kind {
            wide::plan(e, tile, 129)
        } else {
            blocked::plan(e, tile, 129)
        }
        .unwrap();
        add(
            if kind {
                CUBIN.into()
            } else {
                baseline_module()
            },
            symbol,
            plan,
        );
    }
    let prepared = e.clone().with_prefill_tiles(32).unwrap();
    add(
        baseline_module(),
        prepared.stage_symbol("_prepare"),
        stage_plan([129, 1, 1]),
    );
    for (features, mutation) in [(64, "chain"), (128, "neighbor")] {
        add(
            CUBIN.into(),
            format!("chop_f32_wide_{mutation}_negative_v1"),
            wide::plan(e, features, 129).unwrap(),
        );
    }
    for (symbol, threads, grid) in [
        (PROBES[0], 32, 1),
        (PROBES[1], 256, 1024),
        (PROBES[2], 256, 1024),
    ] {
        let mut p = stage_plan([grid, 1, 1]);
        p.block = [threads, 1, 1];
        p.pdl = false;
        add("probe_sm120".into(), symbol.into(), p);
    }
    for values in symbols.values_mut() {
        values.sort();
        values.dedup();
    }
    if let Some(path) = std::env::var_os("CHOPINFER_F32_WIDE_EXPORT") {
        let value = json!({"symbols":symbols,"launches":launches,"cases":cases.iter().map(|c| c.json()).collect::<Vec<Value>>(),"patterns":PATTERNS,"timed_patterns":&PATTERNS[..4],"controls":["chain","neighbor"],"isolation_variants":["blocked32","blocked64","wide64","wide128"],"phases":["full","partial","empty","restored"]});
        std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        let dir = root();
        std::fs::create_dir_all(&dir).unwrap();
        for c in cases.iter().filter(|c| !c.wide) {
            let (name, bytes) = generic(&entries[c.entry]);
            std::fs::write(dir.join(format!("{name}.cubin")), bytes).unwrap();
        }
    }
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_WIDE_CUBINS"]
fn gpu_skip_and_wide_bits_timings_and_occupancy() {
    let probe = std::fs::read(root().join("probe_sm120.cubin"))
        .unwrap()
        .leak();
    let cubins = Cubins::new(
        Box::leak(Box::new([("probe_sm120", &*probe)])),
        &[("probe_sm120", PROBES)],
    );
    let shard: usize = std::env::var("CHOPINFER_F32_WIDE_SHARD")
        .unwrap()
        .parse()
        .unwrap();
    assert!(shard < 2);
    kernel_harness::gpu_test_with(cubins, 1 << 30, |s: &Session| {
        let modules = fixture::Modules::load(s)?;
        fixture::occupancy(&modules)?;
        fixture::controls(s, &modules)?;
        fixture::isolation(s, &modules)?;
        let mut executed = 0;
        for (i, c) in cases().into_iter().enumerate() {
            if !c.timed() && i % 2 != shard {
                continue;
            }
            fixture::compare(s, &modules, c)?;
            executed += 1;
        }
        assert_eq!(executed, 280);
        println!(
            "F32_PREFILL_DONE {}",
            json!({"shard":shard,"cases":executed})
        );
        Ok(())
    })
    .unwrap()
    .expect("required F32 GPU qualification skipped");
}
