//! Convert audited engine launch plans into the host's safe CUDA ABI.

use std::collections::BTreeMap;

use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, TensorMap, TensorMapElement, TensorMapInterleave,
    TensorMapL2Promotion, TensorMapOobFill, TensorMapSwizzle, Validate,
};
use runtime_bundle_abi::{
    BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION,
    BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_16, BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_32,
    BUNDLE_TENSOR_MAP_INTERLEAVE_V1_NONE, BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_64,
    BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_128, BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_256,
    BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_NONE, BUNDLE_TENSOR_MAP_OOB_FILL_V1_NAN_REQUEST_ZERO_FMA,
    BUNDLE_TENSOR_MAP_OOB_FILL_V1_NONE, BundleAbiError, BundleCudaSymbolHandleV1, BundleStatusV1,
    BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1, KernelArgumentV1, LaunchClusterV1,
    LaunchConfigV1, TensorMapElementV1, TensorMapSwizzleV1, TensorMapV1,
};

#[derive(Debug)]
pub enum DispatchError {
    InvalidPlan,
    MissingBuffer(u64),
    MissingSymbol(String),
    InvalidTensorMap,
    Host(BundleAbiError),
}

impl From<BundleAbiError> for DispatchError {
    fn from(error: BundleAbiError) -> Self {
        Self::Host(error)
    }
}

/// The ABI status a failed dispatch reports: the host's for a host error,
/// else a bundle error.
pub fn status(error: DispatchError) -> BundleStatusV1 {
    match error {
        DispatchError::Host(error) => crate::resources::status(error),
        _ => BundleStatusV1::BundleError,
    }
}

/// Planner IDs are local to one dispatch and never interpreted as CUDA handles.
/// Distinct views of an arena get distinct IDs even when the host buffer is shared.
#[derive(Clone, Default)]
pub struct BufferBindings {
    next: u64,
    views: BTreeMap<u64, DeviceBufferSpanV1>,
}

impl BufferBindings {
    pub fn bind(&mut self, view: DeviceBufferSpanV1) -> u64 {
        self.next += 1;
        self.views.insert(self.next, view);
        self.next
    }

    pub fn get(&self, id: u64) -> Result<DeviceBufferSpanV1, DispatchError> {
        self.views
            .get(&id)
            .copied()
            .ok_or(DispatchError::MissingBuffer(id))
    }
}

pub fn launch_config(plan: LaunchPlan) -> Result<LaunchConfigV1, DispatchError> {
    plan.validate().map_err(|_| DispatchError::InvalidPlan)?;
    Ok(LaunchConfigV1 {
        flags: if plan.pdl {
            BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION
        } else {
            0
        },
        grid_x: plan.grid[0],
        grid_y: plan.grid[1],
        grid_z: plan.grid[2],
        block_x: plan.block[0],
        block_y: plan.block[1],
        block_z: plan.block[2],
        shared_memory_bytes: plan.dynamic_shared_bytes,
        cluster: LaunchClusterV1 {
            x: plan.cluster[0],
            y: plan.cluster[1],
            z: plan.cluster[2],
        },
    })
}

fn tensor_map(map: &TensorMap, bindings: &BufferBindings) -> Result<TensorMapV1, DispatchError> {
    map.validate()
        .map_err(|_| DispatchError::InvalidTensorMap)?;
    let rank = map.dims.len();
    let element_bytes = match map.element {
        TensorMapElement::F16 | TensorMapElement::Bf16 => 2,
        TensorMapElement::F32 => 4,
        _ => 1,
    };
    if map.strides_bytes[0] != element_bytes || rank > 5 {
        return Err(DispatchError::InvalidTensorMap);
    }
    let mut dims = [0; 5];
    let mut strides = [0; 4];
    let mut boxes = [0; 5];
    let mut element_strides = [0; 5];
    dims[..rank].copy_from_slice(&map.dims);
    strides[..rank - 1].copy_from_slice(&map.strides_bytes[1..]);
    boxes[..rank].copy_from_slice(&map.box_dims);
    element_strides[..rank].copy_from_slice(&map.element_strides);
    let element_type = match map.element {
        TensorMapElement::U8 => TensorMapElementV1::U8,
        TensorMapElement::F16 => TensorMapElementV1::F16,
        TensorMapElement::Bf16 => TensorMapElementV1::Bf16,
        TensorMapElement::F32 => TensorMapElementV1::F32,
        TensorMapElement::Fp8E4M3 => TensorMapElementV1::Fp8E4M3,
        TensorMapElement::Fp8E5M2 => TensorMapElementV1::Fp8E5M2,
        TensorMapElement::U4Align8B => TensorMapElementV1::U4Align8B,
        TensorMapElement::U4Align16B => TensorMapElementV1::U4Align16B,
        TensorMapElement::U6Align16B => TensorMapElementV1::U6Align16B,
    };
    let swizzle = match map.swizzle {
        TensorMapSwizzle::None => TensorMapSwizzleV1::None,
        TensorMapSwizzle::B32 => TensorMapSwizzleV1::Bytes32,
        TensorMapSwizzle::B64 => TensorMapSwizzleV1::Bytes64,
        TensorMapSwizzle::B128 => TensorMapSwizzleV1::Bytes128,
    };
    let interleave = match map.interleave {
        TensorMapInterleave::None => BUNDLE_TENSOR_MAP_INTERLEAVE_V1_NONE,
        TensorMapInterleave::B16 => BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_16,
        TensorMapInterleave::B32 => BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_32,
    };
    let l2_promotion = match map.l2_promotion {
        TensorMapL2Promotion::None => BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_NONE,
        TensorMapL2Promotion::B64 => BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_64,
        TensorMapL2Promotion::B128 => BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_128,
        TensorMapL2Promotion::B256 => BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_256,
    };
    let oob_fill = match map.oob_fill {
        TensorMapOobFill::Zero => BUNDLE_TENSOR_MAP_OOB_FILL_V1_NONE,
        TensorMapOobFill::Nan => BUNDLE_TENSOR_MAP_OOB_FILL_V1_NAN_REQUEST_ZERO_FMA,
    };
    Ok(TensorMapV1 {
        buffer: bindings.get(map.base_buffer)?,
        rank: rank as u32,
        element_type,
        swizzle,
        interleave,
        l2_promotion,
        oob_fill,
        global_dims: dims,
        global_strides_bytes: strides,
        box_dims: boxes,
        element_strides,
    })
}

pub fn arguments(
    planned: &ArgumentList,
    bindings: &BufferBindings,
) -> Result<Vec<KernelArgumentV1>, DispatchError> {
    planned.validate().map_err(|_| DispatchError::InvalidPlan)?;
    planned
        .0
        .iter()
        .map(|argument| {
            Ok(match argument {
                Argument::DeviceBuffer(id) => KernelArgumentV1::DeviceBuffer(bindings.get(*id)?),
                Argument::U64(value) => KernelArgumentV1::ScalarU64(*value),
                Argument::U32(value) => KernelArgumentV1::ScalarU32(*value),
                Argument::I32(value) => KernelArgumentV1::ScalarI32(*value),
                Argument::NullDevicePointer => KernelArgumentV1::NullDevicePointer,
                Argument::TensorMap(map) => KernelArgumentV1::TensorMap(tensor_map(map, bindings)?),
                Argument::DevicePointerArray(ids) => KernelArgumentV1::DevicePointerArray(
                    ids.iter()
                        .map(|id| bindings.get(*id).map(Some))
                        .collect::<Result<_, _>>()?,
                ),
            })
        })
        .collect()
}

/// A launch whose symbol, configuration and arguments the host accepted.
#[derive(Clone)]
pub struct Resolved {
    function: BundleCudaSymbolHandleV1,
    config: LaunchConfigV1,
    arguments: Vec<KernelArgumentV1>,
}

impl Resolved {
    /// Bind a prepared launch to another independent rank without replanning.
    pub fn relocate<E>(
        &self,
        span: impl Fn(DeviceBufferSpanV1) -> Result<DeviceBufferSpanV1, E>,
        symbol: impl Fn(BundleCudaSymbolHandleV1) -> Result<BundleCudaSymbolHandleV1, E>,
    ) -> Result<Self, E> {
        let mut arguments = self.arguments.clone();
        for argument in &mut arguments {
            match argument {
                KernelArgumentV1::DeviceBuffer(view) => *view = span(*view)?,
                KernelArgumentV1::TensorMap(map) => map.buffer = span(map.buffer)?,
                KernelArgumentV1::DevicePointerArray(views) => {
                    for view in views.iter_mut().flatten() {
                        *view = span(*view)?;
                    }
                }
                _ => {}
            }
        }
        Ok(Self {
            function: symbol(self.function)?,
            config: self.config,
            arguments,
        })
    }

    pub fn submit(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
    ) -> Result<(), DispatchError> {
        services.launch_cuda_symbol(stream, self.function, self.config, &self.arguments)?;
        Ok(())
    }
}

/// Resolves a launch on the host without submitting it.
pub fn resolve(
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    symbol: &str,
    plan: LaunchPlan,
    planned: &ArgumentList,
    bindings: &BufferBindings,
) -> Result<Resolved, DispatchError> {
    let function = *symbols
        .get(symbol)
        .ok_or_else(|| DispatchError::MissingSymbol(symbol.into()))?;
    Ok(Resolved {
        function,
        config: launch_config(plan)?,
        arguments: arguments(planned, bindings)?,
    })
}

pub fn launch(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    symbol: &str,
    plan: LaunchPlan,
    planned: &ArgumentList,
    bindings: &BufferBindings,
) -> Result<(), DispatchError> {
    resolve(symbols, symbol, plan, planned, bindings)?.submit(services, stream)
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, TensorMap};
    use runtime_bundle_abi::{
        BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION, BundleDeviceBufferHandleV1,
        DeviceBufferSpanV1, KernelArgumentV1,
    };

    use super::{BufferBindings, arguments, launch_config};

    #[test]
    fn bound_views_keep_distinct_offsets_and_pdl() {
        let handle = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let mut bindings = BufferBindings::default();
        let first = bindings.bind(DeviceBufferSpanV1::new(handle, 256, 4096));
        let second = bindings.bind(DeviceBufferSpanV1::new(handle, 8192, 4096));
        let args = arguments(
            &ArgumentList(vec![
                Argument::DeviceBuffer(first),
                Argument::DevicePointerArray(vec![second, first]),
                Argument::U32(7),
                Argument::NullDevicePointer,
            ]),
            &bindings,
        )
        .expect("bound argument list");
        assert_eq!(
            args[0],
            KernelArgumentV1::DeviceBuffer(bindings.get(first).unwrap())
        );
        assert_eq!(
            args[1],
            KernelArgumentV1::DevicePointerArray(vec![
                Some(bindings.get(second).unwrap()),
                Some(bindings.get(first).unwrap())
            ])
        );
        let config = launch_config(LaunchPlan {
            grid: [2, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 65536,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        })
        .expect("PDL plan");
        assert_eq!(
            config.flags,
            BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION
        );
        assert_eq!(config.shared_memory_bytes, 65536);
    }

    #[test]
    fn tensor_map_uses_the_bound_view_and_host_stride_order() {
        let handle = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let mut bindings = BufferBindings::default();
        let id = bindings.bind(DeviceBufferSpanV1::new(handle, 256, 8192));
        let map = TensorMap {
            base_buffer: id,
            dims: vec![128, 64],
            strides_bytes: vec![1, 128],
            box_dims: vec![128, 1],
            element_strides: vec![1, 1],
            swizzle: kernel_core::contract::TensorMapSwizzle::B128,
            interleave: kernel_core::contract::TensorMapInterleave::None,
            l2_promotion: kernel_core::contract::TensorMapL2Promotion::B256,
            oob_fill: kernel_core::contract::TensorMapOobFill::Zero,
            element: kernel_core::contract::TensorMapElement::Fp8E4M3,
        };
        let args = arguments(&ArgumentList(vec![Argument::TensorMap(map)]), &bindings)
            .expect("host tensor map");
        let KernelArgumentV1::TensorMap(map) = args[0] else {
            panic!("tensor map")
        };
        assert_eq!(map.buffer.offset, 256);
        assert_eq!(map.global_dims[..2], [128, 64]);
        assert_eq!(map.global_strides_bytes[0], 128);
    }
}
