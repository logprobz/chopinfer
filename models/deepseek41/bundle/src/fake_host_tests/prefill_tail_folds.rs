//! Compare the candidate's fake-host outputs and committed request state.
//! Fake kernels do not perform GEMM arithmetic; the lab probe checks bits.
use super::*;
use bundle_runtime::fake_host::RecordingSink;
use model_deepseek41_contract::execution_options::PREFILL_TAIL_FOLDS;
use runtime_bundle_sdk::prefill::{Boundary, encode};

#[test]
fn tail_folds_preserve_prefill_outputs_and_committed_state() {
    for dspark in [false, true] {
        let run = |enabled: bool| {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            host.drafter.set(dspark);
            if dspark {
                host.speculation.set(Some((5, 32)));
            }
            let mut instance = prefill_skip::create_profile_with_options(
                &host,
                8192,
                true,
                &[(PREFILL_TAIL_FOLDS, enabled.into())],
            );
            let mut observations = Vec::new();
            for rows in [85, 127, 129, 191, 789, 831, 853, 1557, 1599] {
                let request = encode(
                    &prompt(1, 0, 0, rows, true),
                    &[Boundary {
                        request_id: 1,
                        prompt_tokens: rows,
                        chunk_end: rows,
                        chunk_tokens: 8192,
                    }],
                )
                .unwrap();
                let mut sink = RecordingSink::<conformance::Limits>::default();
                instance
                    .execute_prefill(&request, &mut sink)
                    .unwrap_or_else(|e| {
                        panic!(
                            "candidate={enabled}, dspark={dspark}, rows={rows}: {e:?}: {:?}",
                            host.logs.borrow()
                        )
                    });
                let outputs = sink
                    .views
                    .iter()
                    .map(|view| {
                        let v = view.raw();
                        (
                            v.rank,
                            v.element_type,
                            v.row_stride_elements,
                            v.row_count,
                            v.live_width,
                            v.first_item_index,
                        )
                    })
                    .collect::<Vec<_>>();
                assert_eq!(outputs.iter().map(|v| v.3).sum::<u32>(), 1);
                instance.commit().unwrap();
                let state = crate::instance::ledger_state(instance.handle(), 0);
                assert_eq!(state, (Some((1, rows)), false));
                observations.push((outputs, state));
                instance.retire_request(1).unwrap();
                assert_eq!(
                    crate::instance::ledger_state(instance.handle(), 0),
                    (None, false)
                );
                host.device.journal.borrow_mut().clear();
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
            observations
        };
        assert_eq!(run(false), run(true), "dspark={dspark}");
    }
}
