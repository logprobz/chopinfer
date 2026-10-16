//! Qwen partial chunks, maximum last-expert load and an empty local rank.
use super::super::prefill_boundaries as boundary;
use super::*;

#[test]
fn qwen_prefill_boundaries_bitwise() {
    if !boundary::enabled() {
        kernel_lab::skip("qwen_prefill_boundaries_bitwise: set CHOPINFER_MOE_BOUNDARY_BASELINE")
            .unwrap();
        return;
    }
    gpu_test(4 << 30, |session| {
        let mut failures = super::replay::Failures::default();
        let mut case = Case::new(session, Phase::Prefill, boundary::CAPACITY)?;
        if case.local != 128 || case.rank != 1 {
            return Err("boundary qualification requires TP4 rank 1".into());
        }
        let banks = Banks::new(session, case.local)?;
        let (mut control, identity) = boundary::baseline(
            session,
            &case.gate,
            counts(&case.gate, boundary::CAPACITY),
            "4e59d89fde01e97752ee995ece7f55f973cb108d92251fc19ac328a51b583732",
        )?;
        let mut executed = 0;
        // Empty follows hot fixtures on the same allocations, including routing
        // workspaces, so stale work descriptors cannot pass as an empty rank.
        for fixture in ["last-expert", "hot-local", "empty"] {
            for tokens in boundary::TOKENS {
                let result = (|| -> LabResult<()> {
                    case.tokens = tokens;
                    case.live
                        .copy_from_bytes(&(tokens as i32).to_ne_bytes(), &session.stream)
                        .map_err(err)?;
                    case.logits_host = vec![-4.0; tokens * GLOBAL];
                    let mut expected_counts = vec![0; case.local];
                    for token in 0..tokens {
                        for slot in 0..QK {
                            let expert = match fixture {
                                "last-expert" if slot == 0 => 255,
                                "hot-local" => 256 - QK + slot,
                                _ => slot,
                            };
                            case.logits_host[token * GLOBAL + expert] = 4.0 + slot as f32 * 0.125;
                            if (128..256).contains(&expert) {
                                expected_counts[expert - 128] += 1;
                            }
                        }
                    }
                    case.logits
                        .copy_from_bytes(&f32_bytes(&case.logits_host), &session.stream)
                        .map_err(err)?;
                    // Routing is checked independently before constructing the write mask.
                    let mut route_checks = case.routing_checks(session)?;
                    route_checks.phase(kernel_lab::poison::Phase::Eager, true, || {
                        case.launch_route(session, true)
                    })?;
                    drop(route_checks);
                    let routing = check_routing(session, &case)?;
                    if routing.counts != expected_counts {
                        return Err("Qwen fixture expert histogram differs".into());
                    }
                    let mut route_mask = vec![false; tokens * QK];
                    for &route in &routing.local.routes {
                        route_mask[route] = true;
                    }
                    let token_mask = vec![true; tokens];
                    let buffers = [&case.inter, &case.routed, &case.output];
                    let snapshot = || -> LabResult<Vec<Vec<u8>>> {
                        Ok(vec![
                            boundary::checked_output(session, buffers[0], I, &route_mask)?,
                            boundary::checked_output(session, buffers[1], H, &route_mask)?,
                            boundary::checked_output(session, buffers[2], H, &token_mask)?,
                        ])
                    };
                    std::mem::swap(&mut case.gate_fn, &mut control);
                    let control_result = (|| -> LabResult<Vec<Vec<u8>>> {
                        let mut routing_checks = case.routing_checks(session)?;
                        routing_checks.prepare()?;
                        boundary::poison(session, &buffers)?;
                        case.launch(session, &banks, true)?;
                        routing_checks.finish(kernel_lab::poison::Phase::Eager, true)?;
                        snapshot()
                    })();
                    std::mem::swap(&mut case.gate_fn, &mut control);
                    let expected = control_result?;
                    let mut routing_checks = case.routing_checks(session)?;
                    routing_checks.prepare()?;
                    boundary::poison(session, &buffers)?;
                    case.launch(session, &banks, true)?;
                    routing_checks.finish(kernel_lab::poison::Phase::Eager, true)?;
                    boundary::compare(&expected, &snapshot()?, "candidate-pdl")?;
                    let graph = session.capture(|_| case.launch(session, &banks, true))?;
                    for _ in 0..2 {
                        routing_checks.prepare()?;
                        boundary::poison(session, &buffers)?;
                        graph.launch(&session.stream).map_err(err)?;
                        routing_checks.finish(kernel_lab::poison::Phase::Graph, true)?;
                        boundary::compare(&expected, &snapshot()?, "candidate-graph")?;
                    }
                    routing_checks.prepare()?;
                    boundary::poison(session, &buffers)?;
                    case.launch(session, &banks, false)?;
                    routing_checks.finish(kernel_lab::poison::Phase::Relaunch, false)?;
                    boundary::compare(&expected, &snapshot()?, "candidate-serial")?;
                    let mut routing_evidence = BTreeMap::new();
                    routing_checks.record(&mut routing_evidence);
                    session.receipt("boundary-routing-replay", routing_evidence)?;
                    boundary::record(
                        session,
                        &format!("qwen-{fixture}"),
                        tokens,
                        &routing.counts,
                        &identity,
                        &expected,
                        5,
                    )?;
                    executed += 1;

                    Ok(())
                })();
                failures.check(&format!("fixture={fixture} tokens={tokens}"), result);
            }
        }
        failures.finish()?;
        assert_eq!(executed, 36);
        eprintln!("boundary executed model=qwen cases={executed}");
        Ok(())
    })
    .expect("Qwen prefill boundary exactness");
}
