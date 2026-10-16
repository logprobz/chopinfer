//! The sole conversion point from kernel-core plans to runtime-cuda launches.

use crate::Session;
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, TensorMap as MapSpec, TensorMapElement,
    TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill, TensorMapSwizzle,
};
use runtime_cuda::{
    Context, DeviceBuffer, Function, KernelArgument, LaunchConfig, Module, Stream, TensorMap,
    TensorMapDataType, TensorMapDescriptor, TensorMapSwizzle as CudaSwizzle,
};
use std::collections::BTreeMap;

pub struct Bindings<'a> {
    pub buffers: BTreeMap<u64, &'a DeviceBuffer>,
}

impl Bindings<'_> {
    fn buffer(&self, handle: u64) -> Result<&DeviceBuffer, String> {
        self.buffers
            .get(&handle)
            .copied()
            .ok_or_else(|| format!("unbound device buffer {handle}"))
    }
}

pub fn tensor_map(
    context: &Context,
    spec: &MapSpec,
    buffer: &DeviceBuffer,
) -> Result<TensorMap, String> {
    let descriptor = descriptor(spec)?;
    context
        .tensor_map(buffer, 0, buffer.byte_len(), descriptor)
        .map_err(|e| e.to_string())
}

pub fn descriptor(spec: &MapSpec) -> Result<TensorMapDescriptor, String> {
    let rank = spec.dims.len();
    if !(1..=5).contains(&rank)
        || spec.strides_bytes.len() != rank
        || spec.box_dims.len() != rank
        || spec.element_strides.len() != rank
    {
        return Err("tensor-map geometry must have matching rank 1 to 5".into());
    }
    let data_type = match spec.element {
        TensorMapElement::U8 => TensorMapDataType::Uint8,
        TensorMapElement::F16 => TensorMapDataType::Float16,
        TensorMapElement::Bf16 => TensorMapDataType::Bfloat16,
        TensorMapElement::F32 => TensorMapDataType::Float32,
        TensorMapElement::Fp8E4M3 => TensorMapDataType::Fp8E4M3,
        TensorMapElement::Fp8E5M2 => TensorMapDataType::Fp8E5M2,
        TensorMapElement::U4Align8B => TensorMapDataType::U4Align8B,
        TensorMapElement::U4Align16B => TensorMapDataType::U4Align16B,
        TensorMapElement::U6Align16B => TensorMapDataType::U6Align16B,
    };
    let swizzle = match spec.swizzle {
        TensorMapSwizzle::None => CudaSwizzle::None,
        TensorMapSwizzle::B128 => CudaSwizzle::Bytes128,
        TensorMapSwizzle::B32 => CudaSwizzle::Bytes32,
        TensorMapSwizzle::B64 => CudaSwizzle::Bytes64,
    };
    let mut descriptor = TensorMapDescriptor {
        rank: rank as u32,
        data_type,
        swizzle,
        interleave: match spec.interleave {
            TensorMapInterleave::None => 0,
            TensorMapInterleave::B16 => 1,
            TensorMapInterleave::B32 => 2,
        },
        l2_promotion: match spec.l2_promotion {
            TensorMapL2Promotion::None => 0,
            TensorMapL2Promotion::B64 => 1,
            TensorMapL2Promotion::B128 => 2,
            TensorMapL2Promotion::B256 => 3,
        },
        oob_fill: match spec.oob_fill {
            TensorMapOobFill::Zero => 0,
            TensorMapOobFill::Nan => 1,
        },
        global_dims: [0; 5],
        global_strides_bytes: [0; 4],
        box_dims: [0; 5],
        element_strides: [0; 5],
    };
    descriptor.global_dims[..rank].copy_from_slice(&spec.dims);
    descriptor.global_strides_bytes[..rank - 1].copy_from_slice(&spec.strides_bytes[1..]);
    descriptor.box_dims[..rank].copy_from_slice(&spec.box_dims);
    descriptor.element_strides[..rank].copy_from_slice(&spec.element_strides);
    Ok(descriptor)
}

pub fn simple_config(
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_memory_bytes: u32,
    cluster: (u32, u32, u32),
    pdl: bool,
) -> Result<LaunchConfig, String> {
    LaunchConfig {
        grid,
        block,
        shared_memory_bytes,
        cluster,
        programmatic_stream_serialization: pdl,
    }
    .validate()
    .map_err(|e| e.to_string())
}

pub fn config(plan: LaunchPlan) -> Result<LaunchConfig, String> {
    simple_config(
        plan.grid.into(),
        plan.block.into(),
        plan.dynamic_shared_bytes,
        plan.cluster.into(),
        plan.pdl,
    )
}

/// Device-side argument storage of one call: its tensor maps and pointer
/// arrays, which the packed arguments borrow.
struct Prepared {
    maps: Vec<(usize, TensorMap)>,
    arrays: Vec<(usize, Vec<u64>)>,
}

impl Prepared {
    fn new(
        context: &Context,
        args: &ArgumentList,
        bindings: &Bindings<'_>,
    ) -> Result<Self, String> {
        let mut maps = Vec::new();
        let mut arrays = Vec::new();
        for (index, argument) in args.0.iter().enumerate() {
            if let Argument::TensorMap(spec) = argument {
                let buffer = bindings.buffer(spec.base_buffer)?;
                maps.push((index, tensor_map(context, spec, buffer)?));
            }
            if let Argument::DevicePointerArray(handles) = argument {
                let addresses = handles
                    .iter()
                    .map(|&handle| {
                        let buffer = bindings.buffer(handle)?;
                        buffer
                            .device_address_at(0, buffer.byte_len())
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                arrays.push((index, addresses));
            }
        }
        Ok(Self { maps, arrays })
    }

    fn pack<'a>(
        &'a self,
        args: &ArgumentList,
        bindings: &'a Bindings<'a>,
    ) -> Result<Vec<KernelArgument<'a>>, String> {
        let mut packed = Vec::with_capacity(args.0.len());
        for (index, argument) in args.0.iter().enumerate() {
            packed.push(match argument {
                Argument::DeviceBuffer(handle) => KernelArgument::Buffer(bindings.buffer(*handle)?),
                Argument::U64(value) => KernelArgument::U64(*value),
                Argument::U32(value) => KernelArgument::U32(*value),
                Argument::I32(value) => KernelArgument::I32(*value),
                Argument::NullDevicePointer => KernelArgument::NullBuffer,
                Argument::TensorMap(_) => KernelArgument::TensorMap(
                    &self
                        .maps
                        .iter()
                        .find(|(i, _)| *i == index)
                        .ok_or("missing map")?
                        .1,
                ),
                Argument::DevicePointerArray(_) => KernelArgument::DevicePointerArray(
                    &self
                        .arrays
                        .iter()
                        .find(|(i, _)| *i == index)
                        .ok_or("missing pointer array")?
                        .1,
                ),
            });
        }
        Ok(packed)
    }
}

/// One rank's launch of a rank-local entry (it waits on no other rank),
/// prepared under [`crate::topology::preflight`] and not submitted.
#[track_caller]
pub fn launch(
    context: &Context,
    function: &Function,
    stream: &Stream,
    plan: LaunchPlan,
    args: &ArgumentList,
    bindings: &Bindings<'_>,
) -> Result<(), String> {
    let shape = config(plan)?;
    let prepared = Prepared::new(context, args, bindings)?;
    let packed = prepared.pack(args, bindings)?;
    if crate::topology::checking() {
        return Ok(());
    }
    let site = std::panic::Location::caller();
    crate::topology::record_launch(stream, || {
        format!(
            "at {site}, grid {:?}, args {args:?}, peer pointer arrays {:?}",
            plan.grid, prepared.arrays
        )
    })?;
    function
        .launch(stream, shape, &packed)
        .map_err(|e| e.to_string())
}

/// [`launch`] with a launch config and packed arguments.
#[track_caller]
pub fn launch_with(
    function: &Function,
    stream: &Stream,
    config: LaunchConfig,
    args: &[KernelArgument<'_>],
) -> Result<(), String> {
    if crate::topology::checking() {
        return Ok(());
    }
    let site = std::panic::Location::caller();
    crate::topology::record_launch(stream, || format!("at {site}, grid {:?}", config.grid))?;
    function
        .launch(stream, config, args)
        .map_err(|e| e.to_string())
}

/// One rank's arguments of a collective call.
pub struct RankCall<'a> {
    pub args: ArgumentList,
    pub bindings: Bindings<'a>,
}

/// A comm cubin loaded for collective calls on each rank's device.
pub struct CollectiveModules {
    ranks: Vec<Module>,
}

impl CollectiveModules {
    pub fn load(sessions: &[Session], name: &str) -> Result<Self, String> {
        for session in sessions {
            if !session.cubins().comm(name)? {
                return Err(format!("{name} is not a comm cubin"));
            }
        }
        let ranks = sessions
            .iter()
            .map(|session| session.module(name))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { ranks })
    }

    /// A comm cubin a bundle embeds, loaded on every rank, so a lab case
    /// runs the exact bytes the bundle loads.
    pub fn from_cubin(sessions: &[Session], bytes: &[u8]) -> Result<Self, String> {
        let ranks = sessions
            .iter()
            .map(|session| session.load_cubin(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { ranks })
    }

    /// The collective entry `symbol` resolved on every rank.
    pub fn function(&self, symbol: &str) -> Result<CollectiveFunction, String> {
        let ranks = self
            .ranks
            .iter()
            .map(|module| {
                module
                    .function(symbol)
                    .map_err(|e| format!("{symbol}: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CollectiveFunction {
            symbol: symbol.to_owned(),
            ranks,
        })
    }
}

/// A collective entry of [`CollectiveModules`], resolved on every rank.
pub struct CollectiveFunction {
    symbol: String,
    ranks: Vec<Function>,
}

/// Every rank, or just `only_rank` when specified.
fn participants(sessions: &[Session], only_rank: Option<usize>) -> Vec<usize> {
    (0..sessions.len())
        .filter(|&rank| only_rank.is_none_or(|only| only == rank))
        .collect()
}

/// Prepares every item, then submits them, none under
/// [`crate::topology::preflight`]: a refusal while preparing any device's
/// launch leaves every device's unsubmitted, since a submitted collective
/// whose peers never launch waits forever.
fn prepare_then_submit<I, T>(
    items: impl IntoIterator<Item = I>,
    prepare: impl FnMut(I) -> Result<T, String>,
    submit: impl FnMut(T) -> Result<(), String>,
) -> Result<(), String> {
    let ready = items
        .into_iter()
        .map(prepare)
        .collect::<Result<Vec<_>, _>>()?;
    if crate::topology::checking() {
        return Ok(());
    }
    ready.into_iter().try_for_each(submit)
}

/// One call of the collective `function` on every rank (or just
/// `only_rank`), rank `r` taking `args(r)`. Prepare every rank's arguments
/// before submitting any launch so a refusal leaves every rank unsubmitted.
#[track_caller]
pub fn submit_collective<'a>(
    sessions: &[Session],
    function: &CollectiveFunction,
    config: LaunchConfig,
    only_rank: Option<usize>,
    args: impl Fn(usize) -> Result<Vec<KernelArgument<'a>>, String>,
) -> Result<(), String> {
    let site = std::panic::Location::caller();
    let symbol = &function.symbol;
    prepare_then_submit(
        participants(sessions, only_rank),
        |rank| Ok((rank, args(rank)?)),
        |(rank, packed)| {
            let stream = &sessions[rank].stream;
            crate::topology::record_launch(stream, || {
                format!("at {site}, {symbol} grid {:?}", config.grid)
            })?;
            function.ranks[rank]
                .launch(stream, config, &packed)
                .map_err(|e| format!("{symbol}: {e}"))
        },
    )
}

/// [`submit_collective`] of `modules`' entry `symbol` at `plan`, rank `r`
/// taking `call(r)`'s arguments.
#[track_caller]
pub fn launch_collective<'a>(
    sessions: &[Session],
    modules: &CollectiveModules,
    symbol: &str,
    plan: LaunchPlan,
    only_rank: Option<usize>,
    call: impl Fn(usize) -> Result<RankCall<'a>, String>,
) -> Result<(), String> {
    launch_resolved(sessions, &modules.function(symbol)?, plan, only_rank, call)
}

/// [`launch_collective`] of an entry resolved before. Under CUDA's lazy
/// module loading resolving an entry loads it, which can wait for the
/// device's running work, so a call that must overlap a PDL producer
/// resolves its entry before submitting that producer.
#[track_caller]
pub fn launch_resolved<'a>(
    sessions: &[Session],
    function: &CollectiveFunction,
    plan: LaunchPlan,
    only_rank: Option<usize>,
    call: impl Fn(usize) -> Result<RankCall<'a>, String>,
) -> Result<(), String> {
    let calls = participants(sessions, only_rank)
        .into_iter()
        .map(|rank| {
            let call = call(rank)?;
            let prepared = Prepared::new(&sessions[rank].context, &call.args, &call.bindings)?;
            Ok((rank, call, prepared))
        })
        .collect::<Result<Vec<_>, String>>()?;
    submit_collective(sessions, function, config(plan)?, only_rank, |rank| {
        let (_, call, prepared) = calls
            .iter()
            .find(|(r, _, _)| *r == rank)
            .ok_or("no call for a participating rank")?;
        prepared.pack(&call.args, &call.bindings)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal while preparing a later device's launch, or a preflight,
    /// submits no device's launch.
    #[test]
    fn a_later_refusal_or_a_preflight_submits_nothing() {
        let submitted = std::cell::Cell::new(0);
        let refused = prepare_then_submit(
            [0, 1, 2, 3],
            |device| {
                if device == 2 {
                    Err(format!("device {device}: launch preparation refused"))
                } else {
                    Ok(device)
                }
            },
            |_| {
                submitted.set(submitted.get() + 1);
                Ok(())
            },
        );
        assert_eq!(refused, Err("device 2: launch preparation refused".into()));
        assert_eq!(submitted.get(), 0);
        crate::topology::preflight(|| {
            prepare_then_submit([0, 1], Ok, |_| {
                submitted.set(submitted.get() + 1);
                Ok(())
            })
        })
        .unwrap();
        assert_eq!(submitted.get(), 0);
        prepare_then_submit([0, 1], Ok, |_| {
            submitted.set(submitted.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(submitted.get(), 2);
    }

    #[test]
    fn maps_rank_five_and_pdl_without_a_device() {
        let spec = MapSpec {
            base_buffer: 0,
            dims: vec![128, 2, 3, 4, 5],
            strides_bytes: vec![0, 256, 512, 1536, 6144],
            box_dims: vec![64, 1, 1, 1, 1],
            element_strides: vec![1; 5],
            element: TensorMapElement::Bf16,
            swizzle: TensorMapSwizzle::B128,
            interleave: TensorMapInterleave::None,
            l2_promotion: TensorMapL2Promotion::B256,
            oob_fill: TensorMapOobFill::Zero,
        };
        let encoded = descriptor(&spec).expect("rank-five descriptor");
        assert_eq!(encoded.rank, 5);
        assert_eq!(encoded.global_strides_bytes, [256, 512, 1536, 6144]);
        assert_eq!(encoded.global_dims, [128, 2, 3, 4, 5]);
        assert_eq!(encoded.l2_promotion, 3);
        let plan = LaunchPlan {
            grid: [2, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        assert!(
            config(plan)
                .expect("PDL launch")
                .programmatic_stream_serialization
        );
    }
}
