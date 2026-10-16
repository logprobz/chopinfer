//! The shared row copies (`kernels_sm120::row::copy`, lock family
//! `row_copy`) against their CPU oracles at every shape the models serve
//! (`common/row_copies.rs`). The Qwen3.8 stream expansion launches the
//! shared broadcast, MiMo's bundle the shared slice, and GLM's bundle both.
//!
//! Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab
//! --release --test row_copy`; without a GPU the GPU tests skip.

#[path = "common/row_copies.rs"]
mod row_copies;

use kernel_core::backend::{Counts, View};
use kernel_core::contract::DType;
use kernel_core::row_copy::{ColumnSlice, RowCopyBuffers};
use kernels_sm120::row::copy::{BROADCAST_SYMBOL, LAB_CUBIN, SLICE_SYMBOL};
use model_mimo26_contract::Tp;
use row_copies::{CAPACITY, Op, all, check_shared, shared};

/// The shared copies prepare for every shape the models serve, through the
/// shared entries the `row_copy` family's cubin holds; each launch's
/// arguments fill its entry's parameters.
#[test]
fn the_shared_copies_serve_every_model_shape_from_the_family_cubin() {
    let [spec] = &kernel_lab::families::row_copy::cubins()[..] else {
        panic!("the row copy family has one cubin");
    };
    assert_eq!(spec.name, LAB_CUBIN);
    let built = kernel_lab::cubin_entries(LAB_CUBIN).unwrap();
    assert_eq!(built, [BROADCAST_SYMBOL, SLICE_SYMBOL]);
    let copies = shared().unwrap();
    // Qwen's and GLM's expansions, GLM's compaction, and MiMo's shards: one
    // at TP2, two at TP4.
    assert_eq!(copies.len(), 6);
    let models = all().unwrap();
    for copy in &copies {
        let prepared = copy.op.prepare().unwrap();
        let (source, destination) = copy.op.bytes().unwrap();
        let view = |handle, bytes: usize| View {
            handle,
            bytes: bytes as u64,
        };
        let buffers = RowCopyBuffers {
            source: view(1, source),
            destination: view(2, destination),
            live_rows: view(3, 4),
        };
        let launches = prepared
            .launches(&buffers, View::EMPTY, Counts::rows(CAPACITY))
            .unwrap();
        let [launch] = &launches[..] else {
            panic!("{}: one launch", copy.name);
        };
        let entry = spec
            .entries
            .iter()
            .find(|entry| entry.symbol == launch.symbol)
            .unwrap_or_else(|| panic!("{}: {} is not built", copy.name, launch.symbol));
        assert_eq!(
            launch.arguments.0.len(),
            entry.params.len(),
            "{}",
            copy.name
        );
        assert_eq!(launch.plan.grid, [CAPACITY, 1, 1], "{}", copy.name);
        assert!(launch.plan.pdl);
        // A model that launches the shared entry already (Qwen's
        // expansion) names what the backend prepares.
        let model = models.iter().find(|model| model.name == copy.name);
        if let Some(model) = model.filter(|model| model.model.cubin == LAB_CUBIN) {
            assert_eq!(launch.symbol, model.model.symbol, "{}", copy.name);
            assert_eq!(launch.plan, model.model.plan, "{}", copy.name);
        }
    }
}

/// A shard past MiMo's source row, which no rank slices, is refused when
/// the op is prepared.
#[test]
fn a_shard_past_the_source_row_is_refused_when_prepared() {
    for tp in Tp::SUPPORTED {
        let hidden = tp.geometry.hidden();
        let past = ColumnSlice {
            dtype: DType::Bf16,
            width: tp.mtp_eh_k() as u32,
            stride: hidden,
            offset: hidden,
            alignment_bytes: 16,
        };
        assert!(Op::Slice(past).prepare().is_err());
    }
}

/// Each shared copy matches its oracle bit for bit at every live count,
/// leaving every byte past the live rows and columns as filled.
#[test]
fn shared_copies_match_their_oracles_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let runs = check_shared(session, &shared()?)?;
        eprintln!("{runs} shared copy runs match their oracles");
        Ok(())
    })
    .expect("shared row copies against their oracles");
}
