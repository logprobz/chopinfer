use kernel_core::{
    backend::{Backend, Counts, Need, View},
    contract::{Argument, Validate},
    replicated_embedding::{ReplicatedEmbedding as Query, ReplicatedEmbeddingBuffers as Buffers},
};
use kernels_sm120::{
    backend::Sm120,
    row::replicated_embedding::{self, Entry},
};
fn entry(ranks: u32) -> Entry {
    Entry {
        query: Query {
            vocab: 128,
            width: 1024,
            ranks,
            streams: 4,
        },
        capacity: 32,
        symbol: format!("chop_embedding_replicated_tp{ranks}_v1"),
    }
}
fn buffers(ranks: u32) -> Buffers<View> {
    let v = |handle| View {
        handle,
        bytes: 1 << 30,
    };
    Buffers {
        tables: (0..ranks).map(|r| v(u64::from(r) + 10)).collect(),
        ids: v(1),
        shard: v(2),
        wide: v(3),
        output: v(4),
        expanded: v(5),
        live_rows: v(6),
    }
}
#[test]
fn one_launch_preserves_local_intermediates_without_collective_state() {
    for ranks in [1, 2, 4] {
        let e = entry(ranks);
        for rank in 0..ranks {
            let backend = Sm120::new().with_replicated_embedding(e.clone());
            let op = backend
                .replicated_embedding(&e.query, rank, Need::rows(32))
                .unwrap();
            assert!(op.info().state.is_empty());
            assert_eq!(op.info().workspace, 0);
            for rows in [1, 2, 8, 32] {
                let b = buffers(ranks);
                let launch = op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
                assert_eq!(launch.len(), 1);
                assert!(!launch[0].collective);
                assert_eq!(launch[0].arguments.0[7], Argument::U32(rank));
                assert_eq!(launch[0].plan.grid, [rows, 1, 1]);
                for bad in 0..5 {
                    let mut b = b.clone();
                    match bad {
                        0 => {
                            b.tables.pop();
                        }
                        1 => b.tables[0].bytes = 1,
                        2 => b.expanded.bytes = 1,
                        3 => b.wide = b.output,
                        _ => b.ids.bytes = 1,
                    }
                    assert!(op.launches(&b, View::EMPTY, Counts::rows(rows)).is_err());
                }
            }
            assert!(
                op.launches(&buffers(ranks), View::EMPTY, Counts::rows(33))
                    .is_err()
            );
            assert!(
                backend
                    .replicated_embedding(&e.query, ranks, Need::rows(1))
                    .is_err()
            );
        }
    }
    for q in [
        Query {
            ranks: 0,
            ..entry(1).query
        },
        Query {
            width: 3,
            ..entry(1).query
        },
        Query {
            vocab: 127,
            ..entry(2).query
        },
    ] {
        assert!(q.validate().is_err());
    }
}
#[test]
fn task_has_no_grid_ownership_and_all_rank_entries_compile_without_spills() {
    let body = include_str!("../cuda/row/replicated_embedding.cuh");
    for forbidden in ["blockIdx", "gridDim", "blockDim", "griddep_", "__shared__"] {
        assert!(!body.contains(forbidden), "{forbidden}");
    }
    let t = kernel_build::Toolchain::detect().unwrap();
    assert!(!t.skipped);
    let root = std::env::temp_dir().join(format!("replicated-embedding-{}", std::process::id()));
    let spec =
        replicated_embedding::cubin_spec("embedding_replicated", &[entry(1), entry(2), entry(4)])
            .unwrap();
    let built = kernel_build::build(
        &t,
        &[spec],
        &root.join("out"),
        &root.join("lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    for image in built {
        for e in image.entries {
            assert_eq!((e.local_bytes, e.stack_bytes), (0, 0));
        }
    }
}
