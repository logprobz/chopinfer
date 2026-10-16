use super::*;
use kernels_sm120::precision::Compute;

#[test]
fn generated_dense_offers_have_oracles_and_replay_invariance() {
    gpu_test(8 << 30, |session| {
        let (offers, _, _) = kernel_lab::models::projection_inventory()?;
        let selection = kernel_lab::projection_cases::Selection::read()?;
        let mut tested = 0;
        let mut wrong_output_checked = false;
        let mut graph_entries = std::collections::BTreeSet::new();
        let mut chunk_entries = std::collections::BTreeSet::new();
        let mut rows_entries = std::collections::BTreeSet::new();
        for (case_index, offer) in offers.iter().enumerate() {
            if !selection.contains(case_index) { continue; }
            let Compute::Dense(spec) = &offer.handle().compute else {
                continue;
            };
            let generated = kernel_lab::projection_cases::typed(offer)?;
            if gemm::plan(
                &spec.key,
                RuntimeCounts {
                    m: 0,
                    batch: 1,
                    context: 0,
                    routed_rows: 0,
                    split_k: 1,
                },
            )
            .is_ok()
            {
                return Err("dense zero-live refusal changed".into());
            }
            let mut preparation_s = 0.0;
            let mut module_s = 0.0;
            let mut replay_s = 0.0;
            let mut subcases = 0;
            let mut wrong_output_refused = false;
            for m in generated.oracle_rows.iter().copied().filter(|m| *m > 0) {
                let variant = GemmVariant {
                    profile: Profile::StdDenseMoe,
                    instantiation: spec.clone(),
                    invariance: kernels_sm120::invariance::Invariance::NONE,
                };
                let start = std::time::Instant::now();
                let case = Case::prepare_variant_sampled(session, variant, m, true, false, true)?;
                preparation_s += start.elapsed().as_secs_f64();
                let start = std::time::Instant::now();
                let function = kernel_lab::projection_function(
                    session,
                    &spec.symbol,
                    case.plan.dynamic_shared_bytes,
                )?
                    .ok_or("missing prepared dense function")?;
                module_s += start.elapsed().as_secs_f64();
                let start = std::time::Instant::now();
                let mut previous: Option<Vec<u32>> = None;
                for _ in 0..2 {
                    case.poison(session, &[&case.buffers[&OUTPUT], &case.buffers[&PARTIAL]])?;
                    case.launch(session, &function, &session.stream)?;
                    let actual =
                        case.written(session, &case.buffers[&OUTPUT], "generated projection")?;
                    case.receipt(session, &actual, "generated_projection_oracle")?;
                    let bits = actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    if previous.as_ref().is_some_and(|old| old != &bits) {
                        return Err("generated dense replay differs".into());
                    }
                    previous = Some(bits);
                }
                if m == *generated.oracle_rows.iter().max().unwrap()
                    && graph_entries.insert(spec.symbol.clone())
                {
                    let graph = session.capture(|stream| case.launch(session, &function, stream))?;
                    case.poison(session, &[&case.buffers[&OUTPUT], &case.buffers[&PARTIAL]])?;
                    graph.launch(&session.stream).map_err(|e| e.to_string())?;
                    let actual = case.written(session, &case.buffers[&OUTPUT], "generated graph projection")?;
                    case.receipt(session, &actual, "generated_projection_graph_oracle")?;
                    if previous.as_ref() != Some(&actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()) {
                        return Err("generated dense eager and graph output differ".into());
                    }
                }
                if !wrong_output_checked && m >= 64 {
                    let picks = case.picks.as_ref().ok_or("missing dense samples")?;
                    let width = case.output_len / m as usize;
                    let sample = picks.iter().position(|p| p / width == 47)
                        .ok_or("dense samples miss the later strip row")?;
                    let reference = case.reference.as_ref().ok_or("missing dense reference")?;
                    let output = query(&case.variant).output;
                    let magnitude = reference.values[sample].abs()
                        + 4.0 * oracle::numeric_bound(reference, sample, output) + 1.0;
                    let wrong = if reference.values[sample] >= 0.0 { -magnitude } else { magnitude } as f32;
                    if !wrong.is_finite() { return Err("invalid wrong-output fixture".into()); }
                    let bytes = if output == DType::Bf16 {
                        f32_to_bf16(wrong).to_ne_bytes().to_vec()
                    } else {
                        wrong.to_ne_bytes().to_vec()
                    };
                    case.buffers[&OUTPUT].copy_from_bytes_at(picks[sample] * bytes.len(), &bytes, &session.stream)
                        .map_err(|e| e.to_string())?;
                    let actual = case.written(session, &case.buffers[&OUTPUT], "finite wrong strip output")?;
                    let sampled = picks.iter().map(|p| actual[*p]).collect::<Vec<_>>();
                    if check_output(&sampled, reference, output, &mut BTreeMap::new()).is_ok() {
                        return Err("dense oracle accepted a wrong output on row 47".into());
                    }
                    wrong_output_checked = true;
                    wrong_output_refused = true;
                }
                replay_s += start.elapsed().as_secs_f64();
                subcases += 1;
            }
            let chunk_start = std::time::Instant::now();
            if generated.chunk_invariant && chunk_entries.insert(spec.symbol.clone()) {
                let (rows, splits) = kernel_lab::projection_cases::chunk_splits(
                    offer.usage().capacity,
                    &[u32::from(spec.key.knobs.tile_m), u32::from(spec.key.knobs.tile_n)],
                );
                let variant = GemmVariant {
                    profile: Profile::StdDenseMoe,
                    instantiation: spec.clone(),
                    invariance: kernels_sm120::invariance::Invariance::NONE,
                };
                let mut case =
                    Case::prepare_variant_sampled(session, variant, rows, false, false, true)?;
                case.prepared = true;
                let k = query(&case.variant).k as usize;
                let input = invariance::wide_values(rows as usize * k, 0x513ea);
                let whole = invariance::launch_rows(session, &mut case, &input, None)?;
                for split in splits {
                    if let kernels_sm120::invariance::Chunk::Granule(granule) = offer.invariance().chunk
                        && split % granule != 0 {
                        continue;
                    }
                    let mut chunks = invariance::launch_rows(
                        session,
                        &mut case,
                        &input[..split as usize * k],
                        None,
                    )?;
                    chunks.extend(invariance::launch_rows(
                        session,
                        &mut case,
                        &input[split as usize * k..],
                        None,
                    )?);
                    if whole != chunks {
                        return Err("generated dense chunks differ".into());
                    }
                }
                // Batch: even rows keep their bits when the odd rows change.
                if offer.invariance().batch == kernels_sm120::invariance::Batch::Independent {
                    let mut beside = invariance::wide_values(rows as usize * k, 0x0dd5);
                    for row in (0..rows as usize).step_by(2) {
                        beside[row * k..(row + 1) * k].copy_from_slice(&input[row * k..(row + 1) * k]);
                    }
                    let other = invariance::launch_rows(session, &mut case, &beside, None)?;
                    if (0..rows as usize).step_by(2).any(|row| other[row] != whole[row]) {
                        return Err(format!("{}: prefill rows change with the other rows", spec.symbol));
                    }
                }
            }
            // Rows: each row of a launch of 1 to `max` rows has the bits of
            // that row launched alone through the operation's one-row offer
            // (`one_row_offer`), and keeps them beside other rows' values.
            if let kernels_sm120::invariance::Rows::OneRow { max } = offer.invariance().rows
                && max > 1
                && rows_entries.insert(spec.symbol.clone())
            {
                let one = kernel_lab::projection_cases::one_row_offer(&offers, case_index)?;
                let Compute::Dense(one_spec) = &offers[one].handle().compute else {
                    return Err("a dense offer's one-row offer is not dense".into());
                };
                let variant = |spec: &kernel_core::contract::InstantiationSpec| GemmVariant {
                    profile: Profile::StdDenseMoe,
                    instantiation: spec.clone(),
                    invariance: kernels_sm120::invariance::Invariance::NONE,
                };
                let (wide, single) = (variant(spec), variant(one_spec));
                let capacity = max.min(offer.usage().capacity);
                let k = query(&wide).k as usize;
                let mut case =
                    Case::prepare_variant_sampled(session, wide.clone(), capacity, false, false, true)?;
                case.prepared = true;
                for rows in 1..=capacity {
                    let input = invariance::wide_values(rows as usize * k, 0x5e7 + u64::from(rows));
                    case.variant = wide.clone();
                    let together = invariance::launch_rows(session, &mut case, &input, None)?;
                    let mut beside =
                        invariance::wide_values(rows as usize * k, 0xb47c4 + u64::from(rows));
                    beside[..k].copy_from_slice(&input[..k]);
                    let first = invariance::launch_rows(session, &mut case, &beside, None)?;
                    if first[0] != together[0] {
                        return Err(format!("{}: row 0 of {rows} changes with the other rows", wide.symbol));
                    }
                    case.variant = single.clone();
                    for (row, got) in together.iter().enumerate() {
                        let alone = invariance::launch_rows(
                            session,
                            &mut case,
                            &input[row * k..(row + 1) * k],
                            None,
                        )?;
                        if alone[0] != *got {
                            return Err(format!(
                                "{}: row {row} of {rows} differs from {} alone",
                                wide.symbol, single.symbol
                            ));
                        }
                    }
                }
            }
            if subcases == 0 {
                return Err(format!("assigned dense case {case_index} did not execute"));
            }
            tested += 1;
            session.receipt(
                "generated-projection",
                BTreeMap::from([
                    ("phases".into(), json!({"prepare_and_upload_s":preparation_s,"module_s":module_s,"replay_and_receipts_s":replay_s,"chunk_s":chunk_start.elapsed().as_secs_f64(),"executed_subcases":subcases})),
                    ("case_index".into(), json!(case_index)),
                    ("capability".into(), json!(generated.capability)),
                    ("oracle".into(), json!(true)),
                    ("replay".into(), json!(true)),
                    ("eager_vs_graph".into(), json!(graph_entries.contains(&spec.symbol))),
                    ("one_row_rows".into(), json!(rows_entries.contains(&spec.symbol))),
                    ("wrong_output_refused".into(), json!(wrong_output_refused)),
                ]),
            )?;
        }
        if tested == 0 {
            return Err("no generated dense offers ran".into());
        }
        Ok(())
    })
    .expect("generated dense oracle and invariance");
}

/// Every generated dense offer whose epilogue adds a residual gets a case
/// whose arguments the GEMM accepts: the case binds the residual the
/// epilogue reads (the GEMM refuses arguments without it).
#[test]
fn generated_dense_residual_offers_bind_their_residual() {
    let (offers, _, _) = kernel_lab::models::projection_inventory().unwrap();
    let mut residual = 0;
    for offer in &offers {
        let Compute::Dense(spec) = &offer.handle().compute else {
            continue;
        };
        let variant = GemmVariant {
            profile: Profile::StdDenseMoe,
            instantiation: spec.clone(),
            invariance: kernels_sm120::invariance::Invariance::NONE,
        };
        let q = query(&variant);
        if !q.epilogues.residual_add {
            continue;
        }
        let m = 1;
        let w = generated_host::weight_fixture(&variant).unwrap();
        let a = generated_host::activation_fixture(
            q.m_capacity as usize,
            q.k as usize,
            m,
            q.activation,
        )
        .unwrap();
        let counts = RuntimeCounts {
            m: m as u32,
            batch: m as u32,
            context: 0,
            routed_rows: 0,
            split_k: gemm::choose_split_k(&variant.key).unwrap(),
        };
        if let Err(refusal) = case_arguments(&variant, counts, &w, &a) {
            panic!("{}: {refusal}", spec.symbol);
        }
        residual += 1;
    }
    assert!(residual > 0, "no generated dense offer adds a residual");
}

/// A residual joins each reference element at its picked output element,
/// as one more term of the bound; a pick past the residual is refused.
#[test]
fn a_residual_adds_to_its_picked_reference_elements() {
    let mut reference = OracleOutput::new(vec![1.0, 2.0], vec![3.0, 4.0], 8).unwrap();
    add_residual(&mut reference, Some(&[2, 0]), &[0.5, 9.0, -0.25]).unwrap();
    assert_eq!(reference.values, [0.75, 2.5]);
    assert_eq!(reference.sum_abs_products, [3.25, 4.5]);
    assert!(add_residual(&mut reference, Some(&[3, 0]), &[0.0; 3]).is_err());
    let (bytes, values) = residual_fixture(3, DType::Bf16, false);
    assert_eq!((bytes.len(), values.len()), (6, 3));
    assert!(values.iter().all(|v| v.is_finite()));
    assert_eq!(
        residual_fixture(2, DType::F32, true),
        (vec![0; 8], vec![0.0; 2])
    );
    // F32 residuals carry precision BF16 does not hold, and the reference
    // adds exactly the uploaded values.
    let (bytes, values) = residual_fixture(256, DType::F32, false);
    let uploaded = bytes
        .chunks_exact(4)
        .map(|word| f32::from_ne_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(
        uploaded
            .iter()
            .zip(&values)
            .all(|(&f, &v)| f64::from(f) == v)
    );
    let finer = uploaded
        .iter()
        .filter(|v| v.to_bits() & 0xffff != 0)
        .count();
    assert!(
        finer > 200,
        "{finer} of 256 F32 residuals need more than BF16"
    );
    assert!(uploaded.iter().all(|v| v.abs() <= 1.0 / 16.0));
}
