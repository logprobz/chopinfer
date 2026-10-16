//! The device memory a package plans each rank against.

/// Free memory of the device a package's ranks are planned for: a 96 GB
/// device less what the driver and CUDA context keep (about 94.5 GiB).
pub const DEVICE_BYTES: u64 = 94 << 30;

/// What the host keeps free on the tightest rank before it sizes the KV
/// pool (docs/serving-kv-pool.md). A model's plan adds what else it keeps
/// outside the pool.
pub const HOST_RESERVE_BYTES: u64 = 1 << 30;
