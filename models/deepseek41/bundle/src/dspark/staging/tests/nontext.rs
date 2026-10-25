//! Synthetic bookkeeping coverage. No encoder or device arithmetic runs here.
use super::*;
use crate::dspark::execution::Noise;
use crate::dspark::state::Snapshot;
use serde_json::json;

#[derive(Default)]
struct SyntheticProgram {
    history: Vec<i64>,
    text: Vec<u8>,
    observed: Vec<serde_json::Value>,
}

impl Program for SyntheticProgram {
    fn project_committed(&mut self, _: Projection) -> Result<(), String> {
        Ok(())
    }
    fn project_rows(&mut self, _: &[RowProjection]) -> Result<(), String> {
        Ok(())
    }
    fn draft(&mut self, _: &[DraftInput]) -> Result<(), String> {
        Ok(())
    }
    fn target(&mut self, _: &[TargetRows]) -> Result<(), String> {
        Ok(())
    }
    fn commit_features(&mut self, _: &CommitPlan) -> Result<(), String> {
        Ok(())
    }
    fn defer_engram(&mut self, _: &Hashed) -> Result<bool, String> {
        Ok(true)
    }
    fn observe_before_draft(
        &mut self,
        state: &State,
        staging: &Staging,
        inputs: &[Input],
    ) -> Result<(), String> {
        for input in inputs {
            let context = state.context(input.rows.request).unwrap();
            assert_eq!(context.projected(), [input.rows.start; 3]);
            assert_eq!(staging.rows.history(input.rows.request), self.history);
            assert_eq!(input.noise, Noise::sampled(91, input.rows.start - 2, 1.0));
        }
        self.observed.push(json!({"state":format!("{state:?}"),
            "staging":staging.capture_metadata(&inputs.iter().map(|i| i.rows.request).collect::<Vec<_>>()),
            "inputs":format!("{inputs:?}")}));
        Ok(())
    }
    fn device_target(
        &mut self,
        _: &[Input],
        drafts: &[DraftInput],
        history: &[i64],
        text: &[u8],
    ) -> Result<Option<DeviceTarget>, String> {
        assert_eq!(drafts.len(), 1);
        assert_eq!(history, self.history);
        assert_eq!(text, self.text);
        Ok(Some(DeviceTarget {
            ids: vec![[10, 11, 12, 13, 14]],
            confidence_bits: vec![[0x3f800000; 5]],
        }))
    }
}

fn expected(tokens: &[u32], text: &[bool]) -> Vec<i64> {
    let hash = HashTables::of(&model_deepseek41_contract::geometry::FLASH).unwrap();
    tokens
        .iter()
        .zip(text)
        .skip(tokens.len().saturating_sub(3))
        .map(|(&token, &is_text)| {
            if is_text {
                hash.compress(token, true).unwrap()
            } else {
                -1
            }
        })
        .collect()
}

struct Fixture {
    runtime: Runtime,
    staging: Staging,
    program: SyntheticProgram,
    tokens: Vec<u32>,
    text: Vec<bool>,
    request: u64,
    snapshots: Vec<serde_json::Value>,
    ranks: Vec<Rank>,
}

impl Fixture {
    fn step(&mut self, depth: u32, kept: Option<u32>, is_text: bool) {
        let table = HostServicesTableV1::new(std::rc::Rc::new(
            runtime_bundle_abi::UnsupportedHostServicesBackendV1,
        ));
        let empty = Span::new(
            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
            0,
            0,
        );
        let readback = Readback {
            stream: BundleStreamHandleV1::from_raw(0),
            draft_ids: empty,
            confidence: empty,
        };
        let mask = BTreeMap::from([(self.request, vec![is_text])]);
        self.program.history = expected(&self.tokens, &self.text);
        self.program.text = [vec![u8::from(is_text)], vec![1; depth as usize]].concat();
        let start = self.tokens.len() as u64;
        let input = Input {
            rows: Rows {
                request: self.request,
                slot: 0,
                start,
                kind: Kind::Decode { drafts: depth },
            },
            tokens: vec![17],
            noise: Noise::sampled(91, start - 2, 1.0),
        };
        let mut adapter = Adapter {
            program: &mut self.program,
            services: table.caller(),
            staging: &mut self.staging,
            ranks: &self.ranks,
            readback: &readback,
            text: &mask,
            times: None,
            diagnostic: Context {
                instance_id: 1,
                step_id: start,
            },
        };
        self.runtime.execute(&mut adapter, &[input]).unwrap();
        assert_eq!(
            adapter.staging.rows.history(self.request),
            expected(&self.tokens, &self.text)
        );
        if let Some(kept) = kept {
            let accepted = [(self.request, kept)];
            self.runtime
                .commit(&mut adapter, if depth == 0 { &[] } else { &accepted })
                .unwrap();
            self.tokens.push(17);
            self.text.push(is_text);
            self.tokens.extend((10..).take(kept as usize - 1));
            self.text
                .extend(std::iter::repeat_n(true, kept as usize - 1));
            assert_eq!(
                adapter.staging.rows.history(self.request),
                expected(&self.tokens, &self.text)
            );
            assert_eq!(
                self.runtime.state().context(self.request).unwrap().end,
                self.tokens.len() as u64
            );
        } else {
            self.runtime.cancel(&mut adapter).unwrap();
            assert_eq!(
                adapter.staging.rows.history(self.request),
                expected(&self.tokens, &self.text)
            );
        }
        assert!(adapter.staging.pending.is_empty());
        assert!(adapter.staging.confidence.is_none());
        self.snapshots.push(json!({"state":format!("{:?}",self.runtime.state()),
            "staging":self.staging.capture_metadata(&[self.request]),"committed_tokens":self.tokens,"text":self.text,"canceled":kept.is_none()}));
    }

    fn snapshot(&mut self) -> Snapshot {
        let table = HostServicesTableV1::new(std::rc::Rc::new(
            runtime_bundle_abi::UnsupportedHostServicesBackendV1,
        ));
        let empty = Span::new(
            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
            0,
            0,
        );
        let readback = Readback {
            stream: BundleStreamHandleV1::from_raw(0),
            draft_ids: empty,
            confidence: empty,
        };
        let text = BTreeMap::new();
        let mut adapter = Adapter {
            program: &mut self.program,
            services: table.caller(),
            staging: &mut self.staging,
            ranks: &self.ranks,
            readback: &readback,
            text: &text,
            times: None,
            diagnostic: Context {
                instance_id: 1,
                step_id: 999,
            },
        };
        self.runtime.catch_up(&mut adapter, self.request).unwrap();
        self.runtime.state().snapshot(self.request).unwrap()
    }
}

#[test]
fn nontext_bookkeeping_survives_long_zero_depth_restore_cancel_and_slot_reuse() {
    let g = &model_deepseek41_contract::geometry::FLASH;
    // The real adapter validates all rank spans before handing hashes to
    // defer_engram. No host transfer is available in this synthetic fixture.
    let columns = HashTables::of(g).unwrap().columns() as u64;
    let dim = u64::from(g.shape.engram.head_dim);
    let ranks = (0..4)
        .map(|rank| Rank {
            rank,
            stream: BundleStreamHandleV1::from_parts(1, rank + 1),
            image: BundleCompiledImageHandleV1::from_parts(1, rank + 1),
            tables: (0..g.engram_layers())
                .map(|_| Table {
                    codes: engram_rows::Plane {
                        view: b"codes".to_vec(),
                        row_bytes: dim,
                    },
                    scales: engram_rows::Plane {
                        view: b"scales".to_vec(),
                        row_bytes: dim / 32,
                    },
                })
                .collect(),
            destinations: (0..g.engram_layers())
                .map(|layer| {
                    let span = |id, bytes| {
                        Span::new(
                            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, id),
                            0,
                            bytes,
                        )
                    };
                    Destination {
                        codes: span(2 * layer + 1, (u64::from(DECODE_BASE) + 4) * columns * dim),
                        scales: span(
                            2 * layer + 2,
                            (u64::from(DECODE_BASE) + 4) * columns * dim / 32,
                        ),
                    }
                })
                .collect(),
        })
        .collect();
    let mut state = State::new(Config {
        slots: 2,
        stages: 3,
        window: 128,
        max_prefill_rows: 512,
        context_tokens: 1024,
        position_capacity: 1024,
    })
    .unwrap();
    let initial = Snapshot {
        start: 0,
        end: 3,
        window: 128,
        stages: 3,
    };
    state.restore(7, 0, initial).unwrap();
    let mut fixture = Fixture {
        runtime: Runtime::new(state, g.shape.vocab, 32).unwrap(),
        staging: Staging::new(g).unwrap(),
        program: SyntheticProgram::default(),
        tokens: vec![1, 2, 3],
        text: vec![true, false, true],
        request: 7,
        snapshots: Vec::new(),
        ranks,
    };
    fixture
        .staging
        .restore(7, &fixture.tokens, Some(&fixture.text))
        .unwrap();
    assert_eq!(
        fixture.staging.rows.history(7),
        expected(&fixture.tokens, &fixture.text)
    );
    for step in 0..270 {
        fixture.step(0, Some(1), step % 3 != 2);
    }
    assert!(fixture.program.observed.is_empty());
    assert_eq!(
        fixture.runtime.state().context(7).unwrap().projected(),
        [0; 3]
    );
    assert!(fixture.staging.rows.history(7).contains(&-1));
    for (depth, kept) in [(1, 1), (1, 2), (3, 4), (1, 1)] {
        fixture.step(depth, Some(kept), false);
    }
    assert_eq!(fixture.program.observed.len(), 4);
    // Prefix capture follows the passive transition observations.
    let snapshot = fixture.snapshot();
    let saved_tokens = fixture.tokens.clone();
    let saved_text = fixture.text.clone();
    fixture.step(1, None, false);
    // These are the production calls used after a canceled instance step.
    fixture.runtime.retire(7).unwrap();
    fixture.staging.forget(7);
    assert!(fixture.runtime.state().context(7).is_none());
    assert!(fixture.staging.rows.history(7).is_empty());
    fixture
        .runtime
        .state()
        .validate_restore(9, 0, snapshot)
        .unwrap();
    fixture.runtime.restore_after_copy(9, 0, snapshot).unwrap();
    fixture
        .staging
        .restore(9, &saved_tokens, Some(&saved_text))
        .unwrap();
    fixture.request = 9;
    fixture.tokens = saved_tokens;
    fixture.text = saved_text;
    assert_eq!(
        fixture.staging.rows.history(9),
        expected(&fixture.tokens, &fixture.text)
    );
    fixture.step(0, Some(1), false);
    fixture.step(1, Some(2), false);
    fixture.step(3, Some(4), true);
    assert_eq!(fixture.runtime.state().context(9).unwrap().slot, 0);
    let root = std::path::PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
        .join("deepseek41-tests");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join(format!("nontext-bookkeeping-{}.json",std::process::id())),serde_json::to_vec_pretty(&json!({
        "passed":true,"synthetic":true,"device_arithmetic":false,"zero_depth_steps":271,
        "snapshots":fixture.snapshots,"before_draft":fixture.program.observed,
        "release_qualification":"An image prompt followed by prefix restore, cancellation and physical slot reuse, with resumed drafting and exact output/state comparison."
    })).unwrap()).unwrap();
}
