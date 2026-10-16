//! Lab instances of grouped rows and all three comm epilogue call forms.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::DType;
use kernels_sm120::comm::epilogue::{GroupedRmsNorm, NormReduction};

pub const CUBIN: &str = "zz_norm_weights_sm120";
pub const SHAPES: [(u32, u32); 3] = [(2560, 4), (10240, 1), (2560, 1)];

pub fn row_symbol(width: u32, groups: u32, dtype: DType) -> String {
    format!(
        "chop_lab_norm_w{width}_g{groups}_{}_v1",
        if dtype == DType::F32 { "f32" } else { "bf16" }
    )
}

pub fn cubin_spec() -> CubinSpec {
    let mut spec = GroupedRmsNorm {
        reduction: NormReduction::ShuffleDown,
        weight_dtype: DType::F32,
        group_width: 2560,
        groups: 4,
    }
    .cubin_spec(CUBIN, "unused")
    .unwrap();
    spec.entries.clear();
    spec.defines = kernels_sm120::comm::cubin_spec().defines;
    spec.headers
        .push("#include \"comm/epilogue/hc_inject_norm_f32.cuh\"".into());
    for (group_width, groups) in SHAPES {
        for weight_dtype in [DType::Bf16, DType::F32] {
            spec.entries.push(
                GroupedRmsNorm {
                    reduction: if groups == 1 {
                        NormReduction::XorThenSerial
                    } else {
                        NormReduction::ShuffleDown
                    },
                    weight_dtype,
                    group_width,
                    groups,
                }
                .entry(&row_symbol(group_width, groups, weight_dtype))
                .unwrap(),
            );
        }
    }
    for mode in 0..3 {
        let params = [
            ("input", "const float"),
            ("residual", "const uint16_t"),
            ("logits", "const uint16_t"),
            ("weight", "const float"),
            ("residual_out", "uint16_t"),
            ("normed", "uint16_t"),
            ("live_rows", "const uint32_t"),
        ]
        .into_iter()
        .map(|(name, cxx_type)| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: cxx_type.into(),
        })
        .chain([EntryParam::U32 {
            name: "ranks".into(),
        }])
        .collect();
        let finish = match mode {
            0 => "e.finish(y, x, e.scale(group), group, sums);",
            1 => {
                "uint16_t w[E::kPer] = {}; e.parameters(blockIdx.y, w); e.finish(y, x, e.scale(group), group, w, sums);"
            }
            _ => "e.finish_token(y, blockIdx.x, sums);",
        };
        spec.entries.push(EntrySpec::new(
            format!("chop_lab_norm_comm_f32_m{mode}_v1"),
            params,
            format!(
                r#"chop::row::begin();
  if (blockIdx.x >= *live_rows) return;
  using E = chop::comm::epilogue::HcInjectNormF32<2560, 4>;
  E e{{residual, logits, weight, residual_out, normed}};
  __shared__ float sums[8];
  float y[E::kPer] = {{}};
  const uint64_t group = uint64_t(blockIdx.x) * 4 + blockIdx.y;
  uint16_t x[E::kPer]; e.operand(group, x);
  for (uint32_t rank = 0; rank < ranks; ++rank)
    for (uint32_t j = 0; j < E::kPer; ++j)
      y[j] += input[(uint64_t(blockIdx.x) * ranks + rank) * 2560 + threadIdx.x + j * 256];
  {finish}"#
            ),
            256,
            1,
        ));
    }
    spec
}

/// Opposite-tree controls have their own appended identity. Existing probes
/// and serving instances keep their original wrapper and source identities.
pub const WRONG_TREE_CUBIN: &str = "zz_norm_weights_v2_wrong_tree_sm120";
pub const LEGACY_CUBIN: &str = "zz_norm_weights_v3_legacy_sm120";

pub fn wrong_tree_spec() -> CubinSpec {
    let mut spec = cubin_spec();
    spec.name = WRONG_TREE_CUBIN.into();
    spec.entries = SHAPES
        .into_iter()
        .map(|(group_width, groups)| {
            GroupedRmsNorm {
                reduction: if groups == 1 {
                    NormReduction::ShuffleDown
                } else {
                    NormReduction::XorThenSerial
                },
                weight_dtype: DType::Bf16,
                group_width,
                groups,
            }
            .entry(&row_symbol(group_width, groups, DType::Bf16))
            .unwrap()
        })
        .collect();
    // Entries live in a separate module, but keep globally unique symbols.
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replace("_v1", "_wrong_tree_v1");
    }
    spec
}

/// Use the supplied legacy transport's exact headers, defines and source roots.
/// Only its local epilogue wrapper is new; the legacy CUDA is never copied.
pub fn legacy_epilogue_spec(mut transport: CubinSpec) -> CubinSpec {
    transport.name = LEGACY_CUBIN.into();
    transport.entries = cubin_spec()
        .entries
        .into_iter()
        .filter(|entry| entry.symbol.contains("_comm_"))
        .collect();
    for entry in &mut transport.entries {
        entry.symbol = entry.symbol.replace("_f32_", "_legacy_bf16_");
        entry.body = entry.body.replace("HcInjectNormF32", "HcInjectNorm");
        for param in &mut entry.params {
            if let EntryParam::DevicePointer { name, cxx_type } = param
                && name == "weight"
            {
                *cxx_type = "const uint16_t".into();
            }
        }
    }
    transport
}
