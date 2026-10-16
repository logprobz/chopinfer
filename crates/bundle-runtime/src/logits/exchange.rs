//! Full-vocabulary rows distributed in contiguous blocks across ranks.

use kernel_core::contract::{DType, Refusal};
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1};

use crate::dispatch::BufferBindings;
use crate::pipeline::PreparedLaunch;

#[derive(Clone, Copy, Debug)]
pub struct ExchangeRowsState {
    pub flags: DeviceBufferSpanV1,
    pub epoch: DeviceBufferSpanV1,
}

impl ExchangeRowsState {
    /// Allocate in protocol order. The caller zeros both spans before use.
    pub fn allocate(
        allocate: &mut impl FnMut(u64) -> Result<DeviceBufferSpanV1, BundleAbiError>,
    ) -> Result<Self, BundleAbiError> {
        Ok(Self {
            flags: allocate(kernel_core::row_exchange::EXCHANGE_ROWS_FLAG_BYTES)?,
            epoch: allocate(kernel_core::row_exchange::EXCHANGE_ROWS_EPOCH_BYTES)?,
        })
    }
}

pub struct ExchangeRowsBuffers<'a> {
    pub local: DeviceBufferSpanV1,
    pub outputs: &'a [DeviceBufferSpanV1],
    pub flags: &'a [DeviceBufferSpanV1],
    pub epoch: DeviceBufferSpanV1,
}

/// Exchange all `rows` of each shard. Each owner keeps full rows at their
/// original row offsets, matching `RowOwners` and the rank-zero gather.
pub fn exchange_rows(
    rows: u32,
    capacity: u32,
    shard: u64,
    rank: u32,
    dtype: DType,
    buffers: ExchangeRowsBuffers<'_>,
) -> Result<PreparedLaunch, Refusal> {
    let element_bytes = match dtype {
        DType::F32 => 4,
        DType::Bf16 => 2,
        _ => return Err(Refusal("logits exchange dtype is unsupported")),
    };
    let ranks =
        u32::try_from(buffers.outputs.len()).map_err(|_| Refusal("logits exchange rank count"))?;
    let local_bytes = u64::from(rows)
        .checked_mul(shard)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or(Refusal("logits exchange size overflow"))?;
    let full_bytes = local_bytes
        .checked_mul(u64::from(ranks))
        .ok_or(Refusal("logits exchange size overflow"))?;
    if rows == 0
        || rows > capacity
        || local_bytes > buffers.local.length
        || buffers.outputs.iter().any(|span| full_bytes > span.length)
        || buffers.flags.len() != buffers.outputs.len()
        || buffers
            .flags
            .iter()
            .any(|span| span.length < kernel_core::row_exchange::EXCHANGE_ROWS_FLAG_BYTES)
        || buffers.epoch.length < kernel_core::row_exchange::EXCHANGE_ROWS_EPOCH_BYTES
    {
        return Err(Refusal("logits row exchange exceeds prepared buffers"));
    }
    let mut bindings = BufferBindings::default();
    let input = bindings.bind(buffers.local);
    let outputs = buffers
        .outputs
        .iter()
        .map(|&span| bindings.bind(span))
        .collect();
    let flags = buffers
        .flags
        .iter()
        .map(|&span| bindings.bind(span))
        .collect();
    let epoch = bindings.bind(buffers.epoch);
    let arguments = kernel_core::row_exchange::exchange_rows_arguments(
        kernel_core::row_exchange::ExchangeRowsCall {
            input,
            outputs,
            flags,
            epoch,
            rows,
            shard,
            rank,
            ranks,
        },
    )?;
    Ok(PreparedLaunch {
        symbol: match dtype {
            DType::Bf16 => "chop_comm_exchange_rows_bf16_v1",
            DType::F32 => "chop_comm_exchange_rows_f32_v1",
            _ => unreachable!(),
        }
        .into(),
        rows,
        plan: kernel_core::row_exchange::exchange_rows_plan(),
        arguments,
        bindings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, KernelArgumentV1};

    #[test]
    fn exchange_binds_synthetic_shards_in_rank_order_and_checks_extents() {
        let span = |slot, length| {
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, slot), 256, length)
        };
        for ranks in [2, 4, 8] {
            for (dtype, bytes) in [(DType::Bf16, 2), (DType::F32, 4)] {
                let rows = 17;
                let shard = 128;
                let local = span(1, rows as u64 * shard * bytes);
                let outputs = (0..ranks)
                    .map(|r| span(10 + r, local.length * ranks as u64))
                    .collect::<Vec<_>>();
                let flags = (0..ranks)
                    .map(|r| span(20 + r, kernel_core::row_exchange::EXCHANGE_ROWS_FLAG_BYTES))
                    .collect::<Vec<_>>();
                let epoch = span(30, kernel_core::row_exchange::EXCHANGE_ROWS_EPOCH_BYTES);
                let plan = |rows,
                            local,
                            outputs: &[DeviceBufferSpanV1],
                            flags: &[DeviceBufferSpanV1],
                            epoch| {
                    exchange_rows(
                        rows,
                        32,
                        shard,
                        ranks - 1,
                        dtype,
                        ExchangeRowsBuffers {
                            local,
                            outputs,
                            flags,
                            epoch,
                        },
                    )
                };
                let launch = plan(rows, local, &outputs, &flags, epoch).unwrap();
                let arguments =
                    crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
                assert_eq!(arguments[0], KernelArgumentV1::DeviceBuffer(local));
                assert_eq!(
                    arguments[1],
                    KernelArgumentV1::DevicePointerArray(
                        outputs.iter().copied().map(Some).collect()
                    )
                );
                assert_eq!(
                    arguments[2],
                    KernelArgumentV1::DevicePointerArray(flags.iter().copied().map(Some).collect())
                );
                assert_eq!(arguments[3], KernelArgumentV1::DeviceBuffer(epoch));
                assert_eq!(arguments[4], KernelArgumentV1::ScalarU32(rows));
                assert_eq!(arguments[5], KernelArgumentV1::ScalarU64(shard));
                assert_eq!(arguments[6], KernelArgumentV1::ScalarU32(ranks - 1));
                assert_eq!(arguments[7], KernelArgumentV1::ScalarU32(ranks));
                assert!(plan(0, local, &outputs, &flags, epoch).is_err());
                assert!(plan(33, local, &outputs, &flags, epoch).is_err());
                assert!(plan(rows, span(1, local.length - 1), &outputs, &flags, epoch).is_err());
                assert!(plan(rows, local, &outputs, &flags[..1], epoch).is_err());
                assert!(plan(rows, local, &outputs, &flags, span(30, epoch.length - 1)).is_err());
                let mut short = outputs.clone();
                short[1].length -= 1;
                assert!(plan(rows, local, &short, &flags, epoch).is_err());
                let mut short = flags.clone();
                short[1].length -= 1;
                assert!(plan(rows, local, &outputs, &short, epoch).is_err());
            }
        }
    }
}
