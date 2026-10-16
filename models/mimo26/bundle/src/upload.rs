//! MiMo's input arenas over the shared layout.

pub use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena, RowWrite};

pub type BlockTableMirror = bundle_runtime::upload::BlockTableMirror<crate::contract::Mimo>;
