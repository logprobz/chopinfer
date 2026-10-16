//! MiMo partial live lengths on the production prefill entry.
use super::super::prefill_boundaries as boundary;
use super::*;

#[test]
fn mimo_prefill_boundaries_bitwise() {
    if !boundary::enabled() {
        kernel_lab::skip("mimo_prefill_boundaries_bitwise: set CHOPINFER_MOE_BOUNDARY_BASELINE")
            .unwrap();
        return;
    }
    gpu_test(4 << 30, |session| {
        let mut failures = super::replay::Failures::default();
        let mut case = FullCase::new(
            session,
            model_mimo26_contract::sm120::config::PROFILE,
            Phase::Prefill,
            boundary::CAPACITY,
        )?;
        case.gate_weights = upload_bank(session, &case.gate, GATE_SEED)?;
        case.down_weights = upload_bank(session, &case.down, DOWN_SEED)?;
        case.bias
            .copy_from_bytes(&vec![0; EXPERTS * 4], &session.stream)
            .map_err(err)?;
        let (mut control, identity) = boundary::baseline(
            session,
            &case.gate,
            case.counts(case.gate.knobs.split_k_capacity),
            "fbe01a95ddefe2575cf5e50dd01964f07cb7f8896affa0dc66403ff57be6e9b4",
        )?;
        let intermediate = case.gate.query.n as usize / 2;
        let mut executed = 0;
        for fixture in ["spread", "hot-local"] {
            for tokens in boundary::TOKENS {
                let result = (|| -> LabResult<()> {
                    case.tokens = tokens;
                    case.live
                        .copy_from_bytes(&(tokens as i32).to_ne_bytes(), &session.stream)
                        .map_err(err)?;
                    let mut logits = vec![-4.0_f32; tokens * EXPERTS];
                    let mut expected_counts = vec![0; EXPERTS];
                    for token in 0..tokens {
                        for slot in 0..TOP_K {
                            let expert = if fixture == "hot-local" {
                                EXPERTS - TOP_K + slot
                            } else {
                                (token * TOP_K + slot) % EXPERTS
                            };
                            logits[token * EXPERTS + expert] = 4.0 + slot as f32 * 0.125;
                            expected_counts[expert] += 1;
                        }
                    }
                    case.logits
                        .copy_from_bytes(&f32_bytes(&logits), &session.stream)
                        .map_err(err)?;
                    let buffers = [&case.gate_output, &case.down_output, &case.final_output];
                    let route_mask = vec![true; tokens * TOP_K];
                    let token_mask = vec![true; tokens];
                    let snapshot = || -> LabResult<Vec<Vec<u8>>> {
                        Ok(vec![
                            boundary::checked_output(
                                session,
                                buffers[0],
                                intermediate,
                                &route_mask,
                            )?,
                            boundary::checked_output(session, buffers[1], HIDDEN, &route_mask)?,
                            boundary::checked_output(session, buffers[2], HIDDEN, &token_mask)?,
                        ])
                    };
                    std::mem::swap(&mut case.gate_functions.compute, &mut control);
                    let control_result = (|| -> LabResult<Vec<Vec<u8>>> {
                        let mut routing_checks =
                            case.replay_checks(session, true, false, false, false)?;
                        routing_checks.prepare()?;
                        boundary::poison(session, &buffers)?;
                        case.launch(session, true)?;
                        routing_checks.finish(kernel_lab::poison::Phase::Eager, true)?;
                        snapshot()
                    })();
                    std::mem::swap(&mut case.gate_functions.compute, &mut control);
                    let expected = control_result?;
                    let routing = read_routing(session, &case)?;
                    if routing.counts != expected_counts {
                        return Err("MiMo fixture expert histogram differs".into());
                    }
                    let mut routing_checks =
                        case.replay_checks(session, true, false, false, false)?;
                    routing_checks.prepare()?;
                    boundary::poison(session, &buffers)?;
                    case.launch(session, true)?;
                    routing_checks.finish(kernel_lab::poison::Phase::Eager, true)?;
                    boundary::compare(&expected, &snapshot()?, "candidate-pdl")?;
                    let graph = session.capture(|_| case.launch(session, true))?;
                    for _ in 0..2 {
                        routing_checks.prepare()?;
                        boundary::poison(session, &buffers)?;
                        graph.launch(&session.stream).map_err(err)?;
                        routing_checks.finish(kernel_lab::poison::Phase::Graph, true)?;
                        boundary::compare(&expected, &snapshot()?, "candidate-graph")?;
                    }
                    routing_checks.prepare()?;
                    boundary::poison(session, &buffers)?;
                    case.launch(session, false)?;
                    routing_checks.finish(kernel_lab::poison::Phase::Relaunch, false)?;
                    boundary::compare(&expected, &snapshot()?, "candidate-serial")?;
                    let mut routing_evidence = BTreeMap::new();
                    routing_checks.record(&mut routing_evidence);
                    session.receipt("boundary-routing-replay", routing_evidence)?;
                    boundary::record(
                        session,
                        &format!("mimo-{fixture}"),
                        tokens,
                        &routing.counts,
                        &identity,
                        &expected,
                        4,
                    )?;
                    executed += 1;

                    Ok(())
                })();
                failures.check(&format!("fixture={fixture} tokens={tokens}"), result);
            }
        }
        failures.finish()?;
        assert_eq!(executed, 24);
        eprintln!("boundary executed model=mimo cases={executed}");
        Ok(())
    })
    .expect("MiMo prefill boundary exactness");
}
