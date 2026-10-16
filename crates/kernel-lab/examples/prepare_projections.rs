//! CPU-only preparation of generated instances, outside every committed lock.
fn main() -> Result<(), String> {
    use kernel_build::{BuildContext, BuildOptions, KernelSelection, Toolchain};
    let directory = std::env::args_os()
        .nth(1)
        .ok_or("pass a fresh external output directory")?;
    let cases_only = match std::env::args_os().nth(2) {
        None => false,
        Some(mode) if mode == "--cases-only" => true,
        Some(_) => return Err("expected --cases-only or no mode".into()),
    };
    if std::env::args_os().nth(3).is_some() {
        return Err("unexpected preparation arguments".into());
    }
    let out = std::path::PathBuf::from(directory);
    std::fs::create_dir(&out).map_err(|e| e.to_string())?;
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if out
        .canonicalize()
        .map_err(|e| e.to_string())?
        .starts_with(root.canonicalize().map_err(|e| e.to_string())?)
    {
        return Err("prepared artifacts must be outside the repository".into());
    }
    let (offers, _, copies) = kernel_lab::models::projection_inventory()?;
    let row_shapes = kernel_lab::row_shapes::pairs()?;
    if !cases_only {
        let mut selected =
            kernels_sm120::precision::selected_cubins("generated_projections", &offers)?;
        // Uses of one copy at several capacities share its entry.
        let stages = copies
            .iter()
            .map(|c| (c.stage().entry().symbol.clone(), c.stage().clone()))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .collect::<Vec<_>>();
        selected.push(
            kernel_build::SelectedCubin::new(
                kernels_sm120::row::precision::cubin_spec(
                    "generated_projection_copies_v1",
                    &stages,
                )
                .map_err(|e| e.to_string())?,
                stages.iter().map(|s| s.declaration()).collect(),
            )
            .map_err(|e| e.to_string())?,
        );
        selected.extend(kernel_lab::row_shapes::selected_cubins(&row_shapes)?);
        let selection = KernelSelection::new(
            BuildContext::new("kernel-lab", "chopinfer", "generated-projections.v1")
                .source_root("chopinfer", root),
            selected,
        )
        .map_err(|e| e.to_string())?;
        let toolchain = Toolchain::detect().map_err(|e| e.to_string())?;
        let lock = out.join("prepared.lock");
        for lock_update in [true, false] {
            selection
                .build(
                    &toolchain,
                    &out,
                    &lock,
                    BuildOptions {
                        lock_update,
                        emit_cargo_directives: false,
                        profile_line_info: false,
                    },
                )
                .map_err(|e| e.to_string())?;
        }
    }
    let cases = kernel_lab::projection_cases::inventory()?;
    let copy_cases = copies.iter().map(|copy| {
        let (columns, source, destination) = copy.layout();
        serde_json::json!({"symbol":copy.stage().entry().symbol, "capacity":copy.stage().capacity(), "columns":columns,
            "source":[source.row_stride,source.column_stride,source.column_offset],
            "destination":[destination.row_stride,destination.column_stride,destination.column_offset]})
    }).collect::<Vec<_>>();
    let mut rows = cases
        .iter()
        .map(|c| (c.capability.stages.clone(), c.oracle_rows.clone()))
        .collect::<Vec<_>>();
    rows.extend(copies.iter().map(|c| {
        (
            vec![c.stage().entry().symbol.clone()],
            vec![
                0,
                1,
                c.stage().capacity().saturating_sub(1),
                c.stage().capacity(),
            ],
        )
    }));
    let representatives = kernel_lab::projection_cases::representatives(&rows);
    let dense_families = offers.iter().enumerate().filter_map(|(case_index, offer)| {
        let kernels_sm120::precision::Compute::Dense(spec) = &offer.handle().compute else { return None; };
        let kernel = serde_json::to_value(kernels_sm120::gemm::kernel(&spec.key).expect("validated dense kernel")).expect("kernel JSON");
        let family = kernel.as_object().expect("kernel variant").keys().next().expect("kernel family");
        let tile_rows = [u32::from(spec.key.knobs.tile_m), u32::from(spec.key.knobs.tile_n)];
        let edge_rows = kernel_lab::projection_cases::numerical_rows(offer.usage().capacity, &tile_rows).len();
        Some(serde_json::json!({"case_index":case_index,"kernel_family":family,"tile_rows":tile_rows,"edge_rows":edge_rows}))
    }).collect::<Vec<_>>();
    let row_shape_cases = row_shapes.iter().map(|pair| {
        let symbols = |stage: &kernel_lab::row_shapes::Stage| stage.stages().iter().map(|s| s.entry().symbol.clone()).collect::<Vec<_>>();
        serde_json::json!({"label":pair.label,"origin":pair.origin,"capacity":pair.candidate.capacity(),
            "reference":symbols(&pair.reference),"candidate":symbols(&pair.candidate)})
    }).collect::<Vec<_>>();
    let manifest = serde_json::json!({"operations":cases,"copies":copy_cases,"typed_count":offers.len(),"representatives":representatives,"dense_families":dense_families,"row_shapes":row_shape_cases});
    std::fs::write(
        out.join("cases.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{} cases prepared", cases.len());
    Ok(())
}
