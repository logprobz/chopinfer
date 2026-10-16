//! Fused all-reduce epilogues: the model-supplied functors the comm
//! transport runs on each reduced row (`cuda/comm/epilogue/`).

/// A model's fused all-reduce epilogue. `header` declares the functor
/// template `template <uint32_t Width, uint32_t Streams>` named `template`;
/// the transport entries take its instance as their first template argument
/// and build it from their epilogue arguments, and `width` and `streams`
/// size the planners' rows. A new model adds a header and a value of this
/// type without editing the transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Epilogue {
    /// The functor's header, relative to the CUDA source root.
    pub header: &'static str,
    /// The functor's C++ class template.
    pub template: &'static str,
    /// Columns per stream of a row (one token).
    pub width: u32,
    /// Residual streams per row: 1 without hyper-connections.
    pub streams: u32,
}

/// The residual add and HF RMSNorm of 4096-wide rows
/// (`comm/epilogue/residual_rmsnorm.cuh`), which the fused entries of
/// `comm_sm120` instantiate (`super::FUSED_EPILOGUE`).
pub const RESIDUAL_RMSNORM_4096: Epilogue = Epilogue {
    header: "comm/epilogue/residual_rmsnorm.cuh",
    template: "chop::comm::epilogue::ResidualRmsNorm",
    width: 4096,
    streams: 1,
};

/// Epsilon used by both finish overloads in `hc_inject_norm.cuh`.
pub const HC_INJECT_NORM_EPS_BITS: u32 = 1e-6_f32.to_bits();
/// Stream count required by the shipped inject scale.
pub const HC_INJECT_NORM_STREAMS: u32 = 4;

impl Epilogue {
    pub(super) fn is_hc_inject_norm(&self) -> bool {
        matches!(
            self.template,
            "chop::comm::epilogue::HcInjectNorm" | "chop::comm::epilogue::HcInjectNormF32"
        )
    }

    /// Select the side tensor's stored dtype without changing the reduction.
    /// BF16 keeps the caller's original header and template verbatim.
    pub fn with_norm_weight_dtype(
        self,
        dtype: kernel_core::contract::DType,
    ) -> Result<Self, kernel_core::contract::Refusal> {
        use kernel_core::contract::{DType, Refusal};
        if !self.is_hc_inject_norm() || self.streams != HC_INJECT_NORM_STREAMS {
            return Err(Refusal("unsupported norm epilogue"));
        }
        match dtype {
            DType::Bf16 if self.template == "chop::comm::epilogue::HcInjectNorm" => Ok(self),
            DType::F32 => Ok(Self {
                header: "comm/epilogue/hc_inject_norm_f32.cuh",
                template: "chop::comm::epilogue::HcInjectNormF32",
                ..self
            }),
            _ => Err(Refusal("unsupported norm weight dtype")),
        }
    }

    /// Weight pointer spelling for the transport's generated entry ABI.
    pub fn weight_pointer_type(&self) -> &'static str {
        if self.template == "chop::comm::epilogue::HcInjectNormF32" {
            "const float"
        } else {
            "const uint16_t"
        }
    }

    /// Check the shipped functor's fixed arithmetic. A custom functor owns
    /// its epsilon contract; the transport does not pass epsilon to CUDA.
    pub(super) fn validate_hc_epsilon(
        &self,
        eps_bits: u32,
    ) -> Result<(), kernel_core::contract::Refusal> {
        if self.is_hc_inject_norm() && eps_bits != HC_INJECT_NORM_EPS_BITS {
            return Err(kernel_core::contract::Refusal(
                "unsupported grouped norm epsilon",
            ));
        }
        Ok(())
    }

    /// The C++ type the entries instantiate the transport with.
    pub fn functor(&self) -> String {
        format!("{}<{}, {}>", self.template, self.width, self.streams)
    }

    /// The cubin header line that declares the functor.
    pub fn include(&self) -> String {
        format!("#include \"{}\"", self.header)
    }
}

pub use kernel_core::contract::NormReduction;

/// A row split into independent RMSNorm groups, with BF16 activations and
/// output, F32 accumulation and `(1 + weight)`, and epsilon 1e-6.
/// One 256-thread CTA reduces each group in strided column order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedRmsNorm {
    pub reduction: NormReduction,
    pub weight_dtype: kernel_core::contract::DType,
    pub group_width: u32,
    pub groups: u32,
}

impl GroupedRmsNorm {
    pub fn validate(&self) -> Result<(), kernel_core::contract::Refusal> {
        use kernel_core::contract::{DType, Refusal};
        if !matches!(self.weight_dtype, DType::Bf16 | DType::F32)
            || self.group_width == 0
            || self.group_width > i32::MAX as u32 - 256
            || self.groups == 0
            || self.groups > 65535
            || self.group_width.checked_mul(self.groups).is_none()
        {
            return Err(Refusal("unsupported grouped RMSNorm shape or weight dtype"));
        }
        Ok(())
    }

    pub fn plan(
        &self,
        capacity: u32,
        live: u32,
    ) -> Result<kernel_core::contract::LaunchPlan, kernel_core::contract::Refusal> {
        use kernel_core::contract::{LaunchPlan, Refusal, Validate};
        self.validate()?;
        if capacity == 0 || capacity > i32::MAX as u32 || live > capacity {
            return Err(Refusal("invalid grouped RMSNorm row count"));
        }
        let plan = LaunchPlan {
            grid: [capacity, self.groups, 1],
            block: [256, 1, 1],
            cluster: [1, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Arguments are BF16 input, stored weights, BF16 output and live rows.
    pub fn entry(
        &self,
        symbol: &str,
    ) -> Result<kernel_build::EntrySpec, kernel_core::contract::Refusal> {
        use kernel_build::{EntryParam, EntrySpec};
        self.validate()?;
        let weight = if self.weight_dtype == kernel_core::contract::DType::F32 {
            "float"
        } else {
            "uint16_t"
        };
        let params = [
            ("input", "const uint16_t".into()),
            ("weight", format!("const {weight}")),
            ("output", "uint16_t".into()),
            ("live_rows", "const uint32_t".into()),
        ]
        .into_iter()
        .map(|(name, cxx_type)| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type,
        })
        .collect();
        Ok(EntrySpec::new(
            symbol.into(),
            params,
            format!(
                "chop::row::grouped_rmsnorm<{}, {}, {weight}, {}>(input, weight, output, live_rows);",
                self.group_width,
                self.groups,
                self.reduction == NormReduction::XorThenSerial
            ),
            256,
            1,
        ))
    }

    pub fn cubin_spec(
        &self,
        cubin: &str,
        symbol: &str,
    ) -> Result<kernel_build::CubinSpec, kernel_core::contract::Refusal> {
        Ok(kernel_build::CubinSpec {
            name: cubin.into(),
            sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
            include_dirs: vec![],
            defines: vec![],
            headers: vec!["#include \"row/grouped_rmsnorm.cuh\"".into()],
            entries: vec![self.entry(symbol)?],
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    /// Each functor names its header and instance as the entries and cubin
    /// specs spell them, and the header exists.
    #[test]
    fn epilogues_name_their_headers_and_instances() {
        let norm = super::RESIDUAL_RMSNORM_4096;
        assert_eq!(
            norm.functor(),
            "chop::comm::epilogue::ResidualRmsNorm<4096, 1>"
        );
        assert_eq!(
            norm.include(),
            "#include \"comm/epilogue/residual_rmsnorm.cuh\""
        );
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda");
        assert!(root.join(norm.header).is_file(), "{norm:?}");
    }
}
