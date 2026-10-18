use super::*;
use crate::dspark::execution::{Input, Outcome, Runtime};
use crate::dspark::state::{Config, Rows, State};
use crate::shape::Shape;
use runtime_bundle_abi::{HostServicesBackendV1, HostServicesTableV1};

mod host;

struct FakeProgram<'a> {
    services: &'a HostServicesV1,
    host: &'a bundle_runtime::fake_host::Backend<host::Package>,
    readback: &'a Readback,
    target: Vec<TargetRows>,
}

impl Program for FakeProgram<'_> {
    fn project_committed(&mut self, pass: Projection) -> Result<(), String> {
        self.host.events.borrow_mut().push(format!(
            "project {} {} {}",
            pass.stage, pass.start, pass.count
        ));
        Ok(())
    }
    fn draft(&mut self, inputs: &[DraftInput]) -> Result<(), String> {
        self.host.events.borrow_mut().push("draft".into());
        let mut ids = Vec::new();
        let mut confidence = Vec::new();
        for input in inputs {
            assert_eq!(input.block.compute_rows, 5);
            for position in 0..5 {
                ids.extend_from_slice(
                    &((input.bonus + position + 1) % self.host.geometry.shape.vocab).to_le_bytes(),
                );
            }
            for bits in [
                0x3f80_8000_u32,
                0x8000_0000,
                0x7fc0_0123,
                0x7f80_0000,
                0xff80_0000,
            ] {
                confidence.extend_from_slice(&bits.to_le_bytes());
            }
        }
        self.services
            .copy_host_to_device(
                self.readback.stream,
                prefix(self.readback.draft_ids, ids.len() as u64)?,
                &ids,
            )
            .map_err(|e| e.to_string())?;
        self.services
            .copy_host_to_device(
                self.readback.stream,
                prefix(self.readback.confidence, confidence.len() as u64)?,
                &confidence,
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    fn target(&mut self, rows: &[TargetRows]) -> Result<(), String> {
        self.host.events.borrow_mut().push("target".into());
        self.target = rows.to_vec();
        Ok(())
    }
    fn commit_features(&mut self, _plan: &CommitPlan) -> Result<(), String> {
        self.host.events.borrow_mut().push("commit".into());
        Ok(())
    }
}

struct ResultCase {
    ids: Vec<u32>,
    confidence: [u32; 5],
    history: Vec<i64>,
    target: Vec<u32>,
}

fn run(shape: Shape, budget: u32, kept: Option<u32>) -> ResultCase {
    let host = host::host(shape);
    let table = HostServicesTableV1::new(std::rc::Rc::clone(&host));
    let services = table.caller();
    let hash = HashTables::of(shape.geometry).unwrap();
    let columns = hash.columns() as u64;
    let dim = u64::from(shape.geometry.shape.engram.head_dim);
    let mut ranks = Vec::new();
    let mut pools = Vec::new();
    for rank in 0..shape.ranks() {
        let context = host.select_context(rank, rank as i32).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        pools.push(pool);
        let stream = host.create_stream(context, 0).unwrap();
        let alloc = |bytes| {
            Span::new(
                host.allocate_device_buffer(pool, stream, bytes, 64)
                    .unwrap(),
                0,
                bytes,
            )
        };
        let mut tables = Vec::new();
        let mut destinations = Vec::new();
        for layer in 0..shape.geometry.engram_layers() {
            tables.push(Table {
                codes: engram_rows::Plane {
                    view: format!("{layer}c").into_bytes(),
                    row_bytes: dim,
                },
                scales: engram_rows::Plane {
                    view: format!("{layer}s").into_bytes(),
                    row_bytes: dim / 32,
                },
            });
            destinations.push(Destination {
                codes: alloc(6 * columns * dim),
                scales: alloc(6 * columns * dim / 32),
            });
        }
        ranks.push(Rank {
            stream,
            image: BundleCompiledImageHandleV1::from_parts(1, 1),
            rank,
            tables,
            destinations,
        });
    }
    let pool = pools[0];
    let stream = ranks[0].stream;
    let alloc = |bytes| {
        Span::new(
            host.allocate_device_buffer(pool, stream, bytes, 64)
                .unwrap(),
            0,
            bytes,
        )
    };
    let readback = Readback {
        stream,
        draft_ids: alloc(20),
        confidence: alloc(20),
    };
    let mut program = FakeProgram {
        services,
        host: &host,
        readback: &readback,
        target: Vec::new(),
    };
    let mut staging = Staging::new(shape.geometry).unwrap();
    let config = Config {
        slots: 4,
        stages: shape.geometry.dspark_stages(),
        window: shape.geometry.shape.attention.window,
        max_prefill_rows: 64,
        context_tokens: 1024,
        position_capacity: 2048,
    };
    let mut runtime =
        Runtime::new(State::new(config).unwrap(), shape.geometry.shape.vocab, 32).unwrap();
    let text = BTreeMap::new();
    let mut adapter = Adapter {
        program: &mut program,
        services,
        staging: &mut staging,
        ranks: &ranks,
        readback: &readback,
        text: &text,
        diagnostic: Context {
            instance_id: 1,
            step_id: 1,
        },
    };
    let prompt = Input {
        rows: Rows {
            request: 7,
            slot: 0,
            start: 0,
            kind: Kind::Prefill { rows: 3 },
        },
        tokens: vec![1, 2, 3],
    };
    runtime.execute(&mut adapter, &[prompt]).unwrap();
    runtime.commit(&mut adapter, &[]).unwrap();
    host.events.borrow_mut().clear();
    let input = Input {
        rows: Rows {
            request: 7,
            slot: 0,
            start: 3,
            kind: Kind::Decode { drafts: budget },
        },
        tokens: vec![4],
    };
    let proposed = runtime.execute(&mut adapter, &[input]).unwrap();
    let diagnostic = match kept {
        Some(kept) => {
            let accepted = [(7, kept)];
            runtime
                .commit(&mut adapter, if budget == 0 { &[] } else { &accepted })
                .unwrap()
        }
        None => runtime.cancel(&mut adapter).unwrap(),
    };
    assert!(adapter.staging.pending.is_empty());
    let events = host.events.borrow();
    let target = events.iter().position(|line| line == "target").unwrap();
    let gathers = events
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("gather "))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    assert_eq!(
        gathers.len(),
        shape.ranks() as usize * shape.geometry.engram_layers() as usize * 2
    );
    assert!(
        gathers
            .iter()
            .all(|&i| i < target && events[i + 1] == "upload")
    );
    assert_eq!(adapter.program.target[0].tokens.len(), budget as usize + 1);
    if budget > 0 {
        let reads = events
            .iter()
            .enumerate()
            .filter(|(_, line)| line.starts_with("read "))
            .collect::<Vec<_>>();
        assert_eq!(reads.len(), 2);
        assert!(
            reads
                .iter()
                .all(|(i, line)| line.as_str() == "read 20" && *i < gathers[0])
        );
        assert_eq!(
            diagnostic[0].confidence_bits,
            [
                0x3f80_8000,
                0x8000_0000,
                0x7fc0_0123,
                0x7f80_0000,
                0xff80_0000
            ]
        );
        if kept.is_none() {
            assert!(
                diagnostic[0].outcomes[..budget as usize]
                    .iter()
                    .all(|o| *o == Outcome::Cancelled)
            );
        }
    } else {
        assert!(
            !events
                .iter()
                .any(|line| line == "draft" || line.starts_with("read "))
        );
        assert!(diagnostic.is_empty());
    }
    let mut committed = vec![1, 2, 3];
    if let Some(kept) = kept {
        committed.extend_from_slice(&adapter.program.target[0].tokens[..kept as usize]);
    }
    let expected = committed
        .iter()
        .rev()
        .take(3)
        .rev()
        .map(|&token| hash.compress(token, true).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(adapter.staging.rows.history(7), expected);
    ResultCase {
        ids: proposed
            .first()
            .map_or(Vec::new(), |draft| draft.tokens.clone()),
        confidence: diagnostic.first().map_or([0; 5], |d| d.confidence_bits),
        history: expected,
        target: adapter.program.target[0].tokens.clone(),
    }
}

fn synthetic_topologies() -> Vec<Shape> {
    let shapes = crate::shape::topologies()
        .into_iter()
        .filter(|shape| !shape.geometry.base)
        .collect::<Vec<_>>();
    if std::env::var_os("CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS").is_some() {
        assert!(
            !shapes.is_empty(),
            "synthetic staging gates require the compiled recipe geometry"
        );
    }
    shapes
}

#[test]
fn staged_draft_journal_and_acceptance_depths_cover_every_synthetic_topology() {
    for shape in synthetic_topologies() {
        eprintln!("DSpark staging journal: {}", shape.topology());
        for accepted in 0..=5 {
            run(shape, 5, Some(accepted + 1));
        }
        run(shape, 5, None);
        run(shape, 0, Some(1));
        run(shape, 0, None);
    }
}

#[test]
fn partial_budgets_keep_full_block_scores_and_the_same_committed_prefix() {
    for shape in synthetic_topologies() {
        for budget in 1..5 {
            let full = run(shape, 5, Some(budget + 1));
            let partial = run(shape, budget, Some(budget + 1));
            assert_eq!(partial.ids, full.ids[..budget as usize]);
            assert_eq!(partial.confidence, full.confidence);
            assert_eq!(partial.history, full.history);
            assert_eq!(partial.target, full.target[..budget as usize + 1]);
        }
    }
}

#[test]
fn restore_and_reuse_replace_the_engram_history() {
    for shape in crate::shape::topologies() {
        let hash = HashTables::of(shape.geometry).unwrap();
        let mut staging = Staging::new(shape.geometry).unwrap();
        staging.restore(7, &[1, 2, 3, 4], None).unwrap();
        staging.forget(7);
        assert!(staging.rows.history(7).is_empty());
        staging.restore(9, &[5, 6], Some(&[false, true])).unwrap();
        assert_eq!(
            staging.rows.history(9),
            &[
                hash.compress(5, false).unwrap(),
                hash.compress(6, true).unwrap()
            ]
        );
    }
}
