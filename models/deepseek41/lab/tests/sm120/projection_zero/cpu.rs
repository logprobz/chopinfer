use super::*;
fn all_rows() -> Vec<u32> {
    let mut v = rows();
    v.extend([320, 5695]);
    v.sort_unstable();
    v.dedup();
    v
}
pub fn embedded() -> Vec<(String, &'static [u8], BTreeSet<String>)> {
    let mut modules: BTreeMap<String, (&'static [u8], BTreeSet<String>)> = BTreeMap::new();
    for r in all_rows() {
        let e = old(r);
        let symbols = if e.uses_prefill_tiles() {
            vec![e.symbol().to_string(), e.stage_symbol("_prepare")]
        } else if r <= 32 {
            ["_rows", "_decode", "_reduce"]
                .map(|suffix| e.stage_symbol(suffix))
                .to_vec()
        } else {
            vec![e.symbol().to_string()]
        };
        for symbol in symbols {
            let (name, bytes) = fold_cubin_for_symbol(&symbol)
                .unwrap_or_else(|| panic!("missing reference {symbol}"));
            modules
                .entry(name.into())
                .or_insert((bytes, BTreeSet::new()))
                .1
                .insert(symbol);
        }
    }
    modules.into_iter().map(|(n, (b, s))| (n, b, s)).collect()
}
pub fn launches() -> Vec<(String, LaunchPlan)> {
    let e = family::entry();
    let mut out = Vec::new();
    for r in all_rows() {
        let old = old(r);
        if r <= 32 {
            out.push((old.stage_symbol("_rows"), stage([r, 1, 1])));
            out.push((old.stage_symbol("_decode"), old.decode_plan(r).unwrap()));
            out.push((
                old.stage_symbol("_reduce"),
                stage([(r * n() as u32).div_ceil(128), 1, 1]),
            ));
            out.push((zero_decode::symbol(&e), zero_decode::plan(&e, r).unwrap()));
            out.push((
                family::control(&e, "force_decode"),
                zero_decode::plan(&e, r).unwrap(),
            ));
        } else {
            out.push((old.symbol().to_string(), old_plan(&old, r)));
        }
        if old.uses_prefill_tiles() {
            out.push((old.stage_symbol("_prepare"), stage([r, 1, 1])));
        }
        for m in [32, 64] {
            out.push((zero::symbol(&e, m), zero::plan(&e, m, r).unwrap()));
        }
        out.push((zero::prepare_symbol(&e), stage([r, 1, 1])));
    }
    out.push((
        e.clone()
            .with_prefill_tiles(32)
            .unwrap()
            .stage_symbol("_weights"),
        stage([n() as u32 / 16, 1, 1]),
    ));
    out.push((
        family::control(&e, "omit_chain"),
        zero::plan(&e, 32, 17).unwrap(),
    ));
    out.push((family::control(&e, "force_certificate"), stage([17, 1, 1])));
    out.push((
        "chop_projection_zero_replay_norm_v1".into(),
        family::norm().plan(8192).unwrap(),
    ));
    out
}
#[test]
fn launch_matrix() {
    let specs = family::cubins();
    let entries = specs
        .iter()
        .flat_map(|s| s.entries.iter())
        .map(|e| (&e.symbol, e))
        .collect::<BTreeMap<_, _>>();
    let mut plans = Vec::new();
    for (symbol, p) in launches() {
        p.validate_geometry().unwrap();
        assert_eq!(p.cluster, [1; 3]);
        assert!(p.block[0] <= 1024);
        assert!(p.dynamic_shared_bytes <= 101376);
        if let Some(e) = entries.get(&symbol) {
            assert_eq!(p.block[0], e.max_threads);
        }
        plans.push(json!({"symbol":symbol,"grid":p.grid,"block":p.block,"cluster":p.cluster,"shared_bytes":p.dynamic_shared_bytes,"opt_in_shared_bytes":p.dynamic_shared_bytes}));
    }
    let mut references = BTreeMap::new();
    for (name, bytes, symbols) in embedded() {
        references.insert(
            name.clone(),
            json!({"sha256":kernel_build::digest(bytes),"symbols":symbols}),
        );
        if let Some(p) = std::env::var_os("CHOPINFER_PROJECTION_ZERO_EXPORT") {
            std::fs::write(PathBuf::from(p).join(format!("{name}.cubin")), bytes).unwrap();
        }
    }
    println!(
        "PROJECTION_ZERO_DRY {}",
        json!({"family":family::FAMILY,"rows":rows(),"cases":cases(false),"memcheck_cases":cases(true),"symbols":entries.keys().collect::<Vec<_>>(),"embedded":references,"launches":plans,"serving_enabled":false})
    );
}
#[test]
fn certificate_exhausts_bf16_and_keeps_zero_merges() {
    let mut admitted = 0;
    for bits in 0u16..=u16::MAX {
        let wide = u32::from(bits) << 16;
        let magnitude = wide & 0x7fff_ffff;
        if magnitude != 0 && !((97 << 23)..(158 << 23)).contains(&magnitude) {
            continue;
        }
        let sign = bits & 0x8000;
        let mut x = f32::from_bits(wide);
        let mut parts = [0; 3];
        for p in &mut parts {
            *p = if x == 0.0 {
                sign
            } else {
                (x.to_bits() >> 16) as u16
            };
            x -= f32::from_bits(u32::from(*p) << 16);
        }
        assert_eq!(parts, [bits, sign, sign]);
        admitted += 1;
    }
    assert_eq!(admitted, 15618);
    assert_ne!((-0.0f32).to_bits(), ((-0.0f32 + 0.0) + 0.0).to_bits());
}
#[test]
fn row_certificate_is_independent_and_workspace_is_disjoint() {
    let e = family::entry();
    for rows in rows() {
        let old = e.prefill_workspace_bytes(rows);
        let new = zero::workspace_bytes(&e, rows);
        assert_eq!(new - old, u64::from(rows) * 8);
        assert_eq!(old % 4, 0);
    }
    for pattern in ["bf16", "residual", "mixed", "weight", "edges"] {
        let (x, w) = data::synthetic(17, pattern, 91);
        assert_eq!(w.len(), n() * K);
        let classify = |row: &[f32]| {
            let domain = row.iter().all(|v| {
                let b = v.to_bits() & 0x7fff_ffff;
                b == 0 || ((97 << 23)..(158 << 23)).contains(&b)
            });
            (
                domain,
                domain && row.iter().all(|v| v.to_bits() & 65535 == 0),
            )
        };
        let original = x.chunks_exact(K).map(classify).collect::<Vec<_>>();
        for (i, row) in x.chunks_exact(K).enumerate() {
            assert_eq!(classify(row), original[i]);
        }
        if pattern == "mixed" {
            assert_eq!(
                &original[..3],
                &[(true, true), (true, false), (false, false)]
            );
        }
    }
    let mut raw = e.query();
    raw.weight = kernel_core::contract::WeightFormat::F32;
    let raw = Entry::new(raw, e.usage(), "chop_raw").unwrap();
    assert!(zero::plan(&raw, 32, 1).is_err());
    assert!(zero::plan(&e, 16, 1).is_err());
    assert!(zero::plan(&e, 32, 0).is_err());
    assert!(zero::plan(&e, 32, 8193).is_err());
}
#[test]
#[ignore = "requires sealed CHOPINFER_PROJECTION_ZERO_ROOT inputs"]
fn capture_admission() {
    let manifest = data::manifest().unwrap();
    assert_eq!(manifest["layers"], json!([0, 39]));
    assert_eq!(manifest["rows"], 8192);
    assert_eq!(manifest["shapes"], json!([32, 384, 512]));
    assert_eq!(manifest["k"], 5120);
    assert_eq!(
        manifest["epsilon_bits"],
        model_deepseek41_contract::rows::norm_quantizers(&FLASH)[0]
            .norm
            .epsilon_bits
    );
    for layer in [0, 39] {
        for (name, len) in [
            ("norm-input.bf16", 8192 * K * 2),
            ("norm-weight.bf16", K * 2),
            (&format!("weight-n{}.bf16", n()), n() * K * 2),
        ] {
            assert_eq!(data::read(layer, name).unwrap().len(), len);
        }
    }
}
