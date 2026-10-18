//! Staged key lists with explicit retained-ring bounds.
use kernel_build::CubinSpec;
use kernels_sm120::latent::staged_window_keys::{self,Entry};
pub const FAMILY:&str="latent_staged_window_keys";
pub fn entry()->Entry{Entry{capacity:32,symbol:"chop_staged_window_keys_lab_v1".into()}}
pub fn cubins()->Vec<CubinSpec>{vec![staged_window_keys::cubin_spec("latent_staged_window_keys_sm120",&[entry()])]}
