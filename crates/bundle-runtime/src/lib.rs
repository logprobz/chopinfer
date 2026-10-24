//! Model- and target-neutral mechanics shared by native bundles.
//!
//! A bundle lowers its own program into prepared launches; this crate binds
//! them to host buffers, issues them eagerly or as cached graphs, and owns the
//! mixed-step input format and its device layout. It holds no model shape,
//! tensor name or model stage. Its logits exchange uses the model-neutral
//! communication contract in `kernel-core`. See `BUNDLE_ARCHITECTURE.md#ownership-boundary`.

#![deny(unsafe_code)]

pub mod acquire;
pub mod config;
pub mod context_parallel;
pub mod dispatch;
pub mod dma;
pub mod drafter;
pub mod encoder;
pub mod entries;
pub mod entry;
#[cfg(any(test, feature = "test-support"))]
pub mod fake_host;
pub mod geometry;
pub mod graphs;
pub mod index;
pub mod inputs;
pub mod kv;
pub mod lanes;
pub mod ledger;
pub mod logits;
pub mod media;
pub mod pages;
pub mod phase;
pub mod pipeline;
pub mod prefix;
pub mod projection;
pub mod recorded;
pub mod residency;
pub mod resources;
pub mod ring;
pub mod scratch;
pub mod snapshot;
pub mod startup;
pub mod state;
pub mod submission;
pub mod table;
pub mod transaction;
pub mod upload;

pub use geometry::{BundleGeometry, LimitsError, ModelLimits};

// A pointer array argument carries one address per rank.
const _: () = assert!(
    runtime_bundle_abi::BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS as usize
        == kernel_core::contract::MAX_COMM_RANKS
);

pub mod routes;
