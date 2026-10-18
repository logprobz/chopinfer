//! Typed construction of BF16 row RMSNorm and its plain F32-output mode.

use super::{capacity_bucket, ptr, u32_arg};
use kernel_build::{CubinSpec, EntryDeclaration, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, LaunchPlan, NumericalContract, Refusal, RmsNormKind, Target,
    Validate,
};
use kernel_core::rmsnorm::{RmsNormLayout, RmsNormNumerics, RmsNormRequirement};

/// Available hardware bounds, separate from the requested arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct RmsNormTarget {
    pub target: Target,
    pub max_threads: u32,
    pub shared_bytes: u32,
    pub pdl: bool,
}

impl RmsNormTarget {
    pub const SM120F: Self = Self {
        target: Target::Sm120f,
        max_threads: 1024,
        shared_bytes: 101_376,
        pdl: true,
    };
}

// Transitional: the contract's buffer roles live in `kernel_core::rmsnorm`;
// import them from there, never from this path. This re-export goes (plan
// step 31) once no kernel-lab test or example imports them from
// `kernels_sm120::row::rmsnorm`.
pub use kernel_core::rmsnorm::{FusedResidual, RmsNormBuffers};

/// The HF RMSNorm with [`RmsNormNumerics::HF_TREE_256`]'s sum of squares,
/// relative to the CUDA source root.
pub const TREE_HEADER: &str = "row/rmsnorm_tree.cuh";

/// The static shared memory a CTA may declare in `__shared__` arrays.
const STATIC_SHARED_BYTES: u32 = 48 * 1024;

/// The least eps the HF kind serves, 2^-101 (biased exponent 26).
///
/// The HF numerics round the mean of squares twice, `fl(fl(s / width) +
/// eps)`. Both HF forms (`row.cuh`'s ordered kind and `rmsnorm_tree.cuh`)
/// write `1 / sqrtf(total / width + eps)`, which nvcc compiles at a
/// power-of-two width (512, 4096) to one `fma.rn.f32(s, 2^-k, eps)`, rounded
/// once, and otherwise (1536) to `div.rn.f32` then `add.f32` (PTX of the
/// served entries, `-O3`). The forms differ only where `s / width` is not
/// exact, below 2^-126: from eps = 2^-101 the sum's ulp is at least 2^-124,
/// so that quotient, rounded or not, is under half an ulp and both give
/// eps, while a normal quotient is exact. Smaller epsilons are refused.
pub const HF_MIN_EPS_BITS: u32 = 26 << 23;

/// Validated selection. Its symbol and owner never determine arithmetic.
#[derive(Clone, Debug)]
pub struct RmsNorm {
    requirement: RmsNormRequirement,
    capacity: u32,
    head: u32,
    kind: u32,
    shared_bytes: u32,
    /// The HF kind with the 256-lane tree sum (`row/rmsnorm_tree.cuh`).
    tree: bool,
}

impl RmsNorm {
    pub const IMPLEMENTATION: &'static str = "sm120.row.rmsnorm.v1";

    /// The numerics the row RMSNorm computes for `kind`: HF's order for the
    /// HF kind and the 256-thread warp tree for the others. A requirement
    /// that leaves its numerics to the backend gets these.
    pub fn numerics(kind: RmsNormKind) -> RmsNormNumerics {
        if kind == RmsNormKind::HfBf16Intermediate {
            RmsNormNumerics::HF_ORDERED
        } else {
            RmsNormNumerics::TREE_256
        }
    }

    /// The row RMSNorm of `requirement`; numerics it leaves to the backend
    /// become [`Self::numerics`] of its kind.
    pub fn new(
        requirement: RmsNormRequirement,
        capacity: u32,
        target: RmsNormTarget,
    ) -> Result<Self, Refusal> {
        requirement.validate()?;
        if [requirement.input, requirement.weight] != [DType::Bf16; 2]
            || !matches!(requirement.output, DType::Bf16 | DType::F32)
        {
            return Err(Refusal(
                "RMSNorm requires BF16 input and weight formats, with BF16 or F32 output",
            ));
        }
        if requirement.output == DType::F32
            && (requirement.kind != RmsNormKind::Plain || requirement.fused_residual)
        {
            return Err(Refusal(
                "F32-output RMSNorm requires plain kind without a fused residual",
            ));
        }
        match requirement.layout {
            RmsNormLayout::ContiguousRows { alignment_bytes }
                if alignment_bytes >= 16 && alignment_bytes.is_power_of_two() => {}
            _ => {
                return Err(Refusal(
                    "RMSNorm requires contiguous rows with 16-byte alignment",
                ));
            }
        }
        if requirement.output_quantization.is_some() {
            return Err(Refusal(
                "RMSNorm output quantization is unsupported by this builder",
            ));
        }
        let (head, kind) = match requirement.kind {
            RmsNormKind::Plain => (0, 0),
            RmsNormKind::GemmaOnePlusWeight => (0, 1),
            RmsNormKind::QkPerHead { head_dim } => (head_dim, 2),
            RmsNormKind::HfBf16Intermediate => (0, 3),
        };
        // Positive finite epsilons order as their bits.
        if kind == 3 && requirement.epsilon_bits < HF_MIN_EPS_BITS {
            return Err(Refusal(
                "the HF RMSNorm rounds its mean of squares with eps once at power-of-two widths; it serves epsilons from 2^-101, where that equals the contract's two roundings",
            ));
        }
        let group = if head == 0 { requirement.width } else { head };
        if !group.is_multiple_of(8) || requirement.width > i32::MAX as u32 {
            return Err(Refusal(
                "RMSNorm groups must be multiples of eight within the CUDA width limit",
            ));
        }
        // row.cuh advances an int vector index by 256 * 8 after the last
        // element at group - 8. That terminating increment must also fit.
        if group > i32::MAX as u32 - (256 * 8 - 8) {
            return Err(Refusal(
                "RMSNorm group overflows the CUDA vector loop index",
            ));
        }
        let numerics = Self::numerics(requirement.kind);
        let tree = kind == 3 && requirement.numerics == Some(RmsNormNumerics::HF_TREE_256);
        if requirement
            .numerics
            .is_some_and(|pinned| pinned != numerics)
            && !tree
        {
            return Err(Refusal(
                "RMSNorm accumulation, reduction or rounding rules are unsupported",
            ));
        }
        if tree && requirement.fused_residual {
            return Err(Refusal("the HF RMSNorm's tree sum takes no fused residual"));
        }
        let requirement = RmsNormRequirement {
            numerics: Some(if tree {
                RmsNormNumerics::HF_TREE_256
            } else {
                numerics
            }),
            ..requirement
        };
        if !capacity_bucket(capacity) {
            return Err(Refusal(
                "RMSNorm requires a supported prepared capacity bucket",
            ));
        }
        // Eight reduction slots and the aligned ordered-square buffer; the
        // tree's eight warp sums and its reciprocal.
        let shared_bytes = if tree {
            48
        } else if kind == 3 {
            group
                .checked_mul(4)
                .and_then(|n| n.checked_add(32))
                .ok_or(Refusal("RMSNorm shared memory size overflows"))?
        } else {
            48
        };
        if target.target != Target::Sm120f || target.max_threads < 256 || !target.pdl {
            return Err(Refusal("RMSNorm requires SM120f, 256 threads and PDL"));
        }
        // Both forms declare static `__shared__` arrays, which a CTA holds
        // up to 48 KiB of (the HF kind's ordered squares fill it at 12,280
        // columns).
        if shared_bytes > target.shared_bytes.min(STATIC_SHARED_BYTES) {
            return Err(Refusal(
                "RMSNorm static shared memory exceeds the hardware capacity",
            ));
        }
        Ok(Self {
            requirement,
            capacity,
            head,
            kind,
            shared_bytes,
            tree,
        })
    }

    /// The requirement with the numerics this RMSNorm computes.
    pub fn requirement(&self) -> &RmsNormRequirement {
        &self.requirement
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn shared_bytes(&self) -> u32 {
        self.shared_bytes
    }

    pub fn declaration(&self, symbol: impl Into<String>) -> EntryDeclaration {
        let req = self.requirement;
        let numerics = if self.tree {
            RmsNormNumerics::HF_TREE_256
        } else {
            Self::numerics(req.kind)
        };
        let fields = [
            ("width", req.width.to_string()),
            ("epsilon_bits", format!("{:08x}", req.epsilon_bits)),
            ("kind", format!("{:?}", req.kind)),
            ("input", format!("{:?}", req.input)),
            ("weight", format!("{:?}", req.weight)),
            ("output", format!("{:?}", req.output)),
            ("layout", format!("{:?}", req.layout)),
            ("fused_residual", req.fused_residual.to_string()),
            (
                "output_quantization",
                format!("{:?}", req.output_quantization),
            ),
            ("accumulator", format!("{:?}", numerics.accumulator)),
            ("reduction", format!("{:?}", numerics.reduction)),
            (
                "square_accumulation",
                format!("{:?}", numerics.square_accumulation),
            ),
            ("reciprocal", format!("{:?}", numerics.reciprocal)),
            ("normalized_bf16", numerics.normalized_bf16.to_string()),
            (
                "residual_rounding",
                format!("{:?}", numerics.residual_rounding),
            ),
            ("output_rounding", format!("{:?}", numerics.output_rounding)),
            ("state", "stateless; only live rows are written".into()),
            ("scales", "none".into()),
            ("capacity", self.capacity.to_string()),
            ("target", "sm120f".into()),
            ("threads", "256".into()),
            ("shared_bytes_bound", self.shared_bytes.to_string()),
            ("pdl", "true".into()),
        ];
        EntryDeclaration {
            symbol: symbol.into(),
            operation: "RMSNorm".into(),
            contract: req
                .query()
                .numerical_contract_id()
                .expect("validated RMSNorm")
                .to_string(),
            implementation: Self::IMPLEMENTATION.into(),
            requirements: fields.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            correctness_case: None,
        }
    }

    /// Whether this RMSNorm is the requested one; a request that leaves its
    /// numerics to the backend takes this RMSNorm's.
    pub fn require(&self, requirement: &RmsNormRequirement) -> Result<(), Refusal> {
        let requirement = RmsNormRequirement {
            numerics: requirement.numerics.or(self.requirement.numerics),
            ..*requirement
        };
        if self.requirement != requirement {
            return Err(Refusal(
                "RMSNorm selection does not match the complete requested contract",
            ));
        }
        Ok(())
    }

    pub fn entry(&self, symbol: impl Into<String>) -> EntrySpec {
        let width = self.requirement.width;
        let head = self.head;
        let kind = self.kind;
        let residual = if self.requirement.fused_residual {
            "true"
        } else {
            "false"
        };
        let body = if self.requirement.output == DType::F32 {
            format!(
                "chop::row::rmsnorm_f32<{width}>(x,weight,y,live_rows,__uint_as_float(eps_bits));"
            )
        } else if self.tree {
            format!(
                "chop::row::rmsnorm_hf_tree<{width}>(x,weight,y,live_rows,__uint_as_float(eps_bits));"
            )
        } else {
            format!(
                "chop::row::rmsnorm<{width},{head},{kind},{residual}>(x,weight,residual,residual_out,y,live_rows,__uint_as_float(eps_bits));"
            )
        };
        EntrySpec::new(
            symbol.into(),
            vec![
                ptr("x", "const uint16_t"),
                ptr("weight", "const uint16_t"),
                ptr("residual", "const uint16_t"),
                ptr("residual_out", "uint16_t"),
                ptr(
                    "y",
                    if self.requirement.output == DType::F32 {
                        "float"
                    } else {
                        "uint16_t"
                    },
                ),
                ptr("live_rows", "const uint32_t"),
                u32_arg("eps_bits"),
            ],
            body,
            256,
            1,
        )
    }

    /// The `#include` lines a cubin holding this entry needs.
    pub fn headers(&self) -> Vec<String> {
        let mut headers = vec!["#include \"row/row.cuh\"".to_owned()];
        if self.requirement.output == DType::F32 {
            headers.push("#include \"row/rmsnorm_f32.cuh\"".to_owned());
        }
        if self.tree {
            headers.push(format!("#include \"{TREE_HEADER}\""));
        }
        headers
    }

    pub fn cubin_spec(&self, name: impl Into<String>, symbol: impl Into<String>) -> CubinSpec {
        CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: self.headers(),
            defines: vec![],
            entries: vec![self.entry(symbol)],
        }
    }

    pub fn plan(&self, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        if live_rows > self.capacity {
            return Err(Refusal("RMSNorm live rows exceed prepared capacity"));
        }
        let plan = LaunchPlan {
            grid: [self.capacity, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn arguments(&self, buffers: RmsNormBuffers<u64>) -> Result<ArgumentList, Refusal> {
        let (residual, residual_out) = match (self.requirement.fused_residual, buffers.residual) {
            (false, None) => (Argument::NullDevicePointer, Argument::NullDevicePointer),
            (true, Some(FusedResidual { input, output })) if output != buffers.y => (
                Argument::DeviceBuffer(input),
                Argument::DeviceBuffer(output),
            ),
            _ => {
                return Err(Refusal(
                    "RMSNorm residual buffers do not match the selected epilogue",
                ));
            }
        };
        let args = ArgumentList(vec![
            Argument::DeviceBuffer(buffers.x),
            Argument::DeviceBuffer(buffers.weight),
            residual,
            residual_out,
            Argument::DeviceBuffer(buffers.y),
            Argument::DeviceBuffer(buffers.live_rows),
            Argument::U32(self.requirement.epsilon_bits),
        ]);
        args.validate()?;
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::OutputQuantization;
    use kernel_core::rmsnorm::RmsNormRounding;

    fn plain() -> RmsNormRequirement {
        RmsNormRequirement::bf16(
            1280,
            1e-6,
            RmsNormKind::Plain,
            false,
            RmsNormNumerics::TREE_256,
        )
    }

    fn buffers() -> RmsNormBuffers<u64> {
        RmsNormBuffers {
            x: 1,
            weight: 2,
            residual: None,
            y: 3,
            live_rows: 4,
        }
    }

    #[test]
    fn f32_output_is_an_additive_plain_entry() {
        let req = RmsNormRequirement {
            output: DType::F32,
            ..plain()
        };
        let norm = RmsNorm::new(req, 4, RmsNormTarget::SM120F).unwrap();
        assert!(format!("{:?}", norm.entry("norm")).contains("rmsnorm_f32<1280>"));
        assert!(norm.headers().iter().any(|h| h.contains("rmsnorm_f32.cuh")));
        let old = RmsNorm::new(plain(), 4, RmsNormTarget::SM120F).unwrap();
        assert_eq!(old.headers(), ["#include \"row/row.cuh\""]);
        for kind in [
            RmsNormKind::GemmaOnePlusWeight,
            RmsNormKind::HfBf16Intermediate,
        ] {
            assert!(
                RmsNorm::new(RmsNormRequirement { kind, ..req }, 4, RmsNormTarget::SM120F).is_err()
            );
        }
        assert!(
            RmsNorm::new(
                RmsNormRequirement {
                    fused_residual: true,
                    ..req
                },
                4,
                RmsNormTarget::SM120F
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_incompatible_formats_layouts_and_numerics() {
        let bad = |req, field: &str| {
            let error = RmsNorm::new(req, 4, RmsNormTarget::SM120F).unwrap_err();
            assert!(error.0.contains(field), "{error}");
        };
        bad(
            RmsNormRequirement {
                input: DType::F16,
                ..plain()
            },
            "formats",
        );
        bad(
            RmsNormRequirement {
                weight: DType::F32,
                ..plain()
            },
            "formats",
        );
        bad(
            RmsNormRequirement {
                output: DType::Fp8E4M3,
                ..plain()
            },
            "formats",
        );
        bad(
            RmsNormRequirement {
                layout: RmsNormLayout::StridedRows {
                    stride_elements: 1296,
                    alignment_bytes: 16,
                },
                ..plain()
            },
            "contiguous",
        );
        // Per-head norms of strided heads are a row engine entry's, not
        // this builder's.
        bad(
            RmsNormRequirement {
                kind: RmsNormKind::QkPerHead { head_dim: 128 },
                layout: RmsNormLayout::StridedHeads {
                    input_stride: 1536,
                    output_stride: 1280,
                    alignment_bytes: 16,
                },
                ..plain()
            },
            "contiguous",
        );
        bad(
            RmsNormRequirement {
                layout: RmsNormLayout::ContiguousRows { alignment_bytes: 8 },
                ..plain()
            },
            "alignment",
        );
        bad(
            RmsNormRequirement {
                output_quantization: Some(OutputQuantization::Mxfp8),
                ..plain()
            },
            "quantization",
        );
        bad(
            RmsNormRequirement {
                width: 1279,
                ..plain()
            },
            "groups",
        );
        bad(
            RmsNormRequirement {
                kind: RmsNormKind::QkPerHead { head_dim: 96 },
                ..plain()
            },
            "divide",
        );
        for epsilon in [0.0_f32, -1.0, f32::INFINITY, f32::NAN] {
            bad(
                RmsNormRequirement {
                    epsilon_bits: epsilon.to_bits(),
                    ..plain()
                },
                "epsilon",
            );
        }
        bad(
            RmsNormRequirement {
                numerics: Some(RmsNormNumerics::HF_ORDERED),
                ..plain()
            },
            "rounding",
        );
        let pinned = |numerics| RmsNormRequirement {
            numerics: Some(numerics),
            ..plain()
        };
        bad(
            pinned(RmsNormNumerics {
                accumulator: DType::F16,
                ..RmsNormNumerics::TREE_256
            }),
            "accumulation",
        );
        bad(
            pinned(RmsNormNumerics {
                output_rounding: RmsNormRounding::TowardZero,
                ..RmsNormNumerics::TREE_256
            }),
            "rounding",
        );
        let mut request = plain();
        request.kind = RmsNormKind::HfBf16Intermediate;
        bad(request, "reduction");
    }

    /// A requirement that leaves its numerics to the backend builds the
    /// entry of its kind's numerics: the same declaration, code and plan.
    #[test]
    fn backend_chosen_numerics_build_the_kinds_entry() {
        for (kind, numerics) in [
            (RmsNormKind::Plain, RmsNormNumerics::TREE_256),
            (RmsNormKind::GemmaOnePlusWeight, RmsNormNumerics::TREE_256),
            (
                RmsNormKind::QkPerHead { head_dim: 128 },
                RmsNormNumerics::TREE_256,
            ),
            (RmsNormKind::HfBf16Intermediate, RmsNormNumerics::HF_ORDERED),
        ] {
            assert_eq!(RmsNorm::numerics(kind), numerics);
            for fused in [false, true] {
                let pinned = RmsNormRequirement::bf16(4096, 1e-6, kind, fused, numerics);
                let chosen = RmsNormRequirement::bf16_backend_chosen(4096, 1e-6, kind, fused);
                let from_pinned = RmsNorm::new(pinned, 64, RmsNormTarget::SM120F).unwrap();
                let from_chosen = RmsNorm::new(chosen, 64, RmsNormTarget::SM120F).unwrap();
                assert_eq!(from_chosen.requirement(), &pinned);
                assert_eq!(from_chosen.declaration("s"), from_pinned.declaration("s"));
                assert_eq!(
                    format!("{:?}", from_chosen.entry("s")),
                    format!("{:?}", from_pinned.entry("s"))
                );
                assert_eq!(from_chosen.shared_bytes(), from_pinned.shared_bytes());
                assert_eq!(from_chosen.plan(7), from_pinned.plan(7));
                // Either selection serves either request.
                for selected in [&from_pinned, &from_chosen] {
                    selected.require(&pinned).unwrap();
                    selected.require(&chosen).unwrap();
                }
            }
        }
    }

    #[test]
    fn hf_kind_takes_the_tree_sum() {
        let hf = RmsNormRequirement::bf16(
            4096,
            1e-5,
            RmsNormKind::HfBf16Intermediate,
            false,
            RmsNormNumerics::HF_TREE_256,
        );
        let tree = RmsNorm::new(hf, 32, RmsNormTarget::SM120F).unwrap();
        assert_eq!(tree.shared_bytes(), 48);
        assert_eq!(
            tree.entry("chop_test_v1").body,
            "chop::row::rmsnorm_hf_tree<4096>(x,weight,y,live_rows,__uint_as_float(eps_bits));"
        );
        assert_eq!(
            tree.headers(),
            [
                "#include \"row/row.cuh\"",
                "#include \"row/rmsnorm_tree.cuh\""
            ]
        );
        assert_eq!(tree.cubin_spec("c", "s").headers, tree.headers());
        assert_eq!(tree.entry("s").params, {
            let ordered = RmsNormRequirement {
                numerics: Some(RmsNormNumerics::HF_ORDERED),
                ..hf
            };
            RmsNorm::new(ordered, 32, RmsNormTarget::SM120F)
                .unwrap()
                .entry("s")
                .params
        });
        assert_eq!(tree.plan(5).unwrap().block, [256, 1, 1]);
        let declared = tree.declaration("s").requirements;
        assert_eq!(declared["reduction"], "WarpTree { threads: 256 }");
        assert_eq!(declared["square_accumulation"], "SeparateNearestEven");
        // The ordered kind keeps row.cuh alone.
        let ordered = RmsNormRequirement {
            numerics: Some(RmsNormNumerics::HF_ORDERED),
            ..hf
        };
        let ordered = RmsNorm::new(ordered, 32, RmsNormTarget::SM120F).unwrap();
        assert_eq!(ordered.headers(), ["#include \"row/row.cuh\""]);
        assert!(
            ordered
                .entry("s")
                .body
                .starts_with("chop::row::rmsnorm<4096,0,3,false>")
        );
        // The tree is the HF kind's alone, without a fused residual.
        let error = RmsNorm::new(
            RmsNormRequirement {
                fused_residual: true,
                ..hf
            },
            32,
            RmsNormTarget::SM120F,
        )
        .unwrap_err();
        assert!(error.0.contains("residual"), "{error}");
        assert!(
            RmsNorm::new(
                RmsNormRequirement {
                    kind: RmsNormKind::Plain,
                    ..hf
                },
                32,
                RmsNormTarget::SM120F
            )
            .is_err()
        );
    }

    /// The HF kind's mean of squares as nvcc compiles it at `width`: one
    /// FMA by the exact reciprocal at a power of two, else the rounded
    /// quotient, then the rounded sum (see [`HF_MIN_EPS_BITS`]).
    fn kernel_mean(s: f32, width: u32, eps: f32) -> f32 {
        if width.is_power_of_two() {
            s.mul_add(1.0 / width as f32, eps)
        } else {
            s / width as f32 + eps
        }
    }

    #[test]
    fn hf_kind_refuses_epsilons_its_mean_rounds_otherwise() {
        // A row of one 2^-69 then zeros: s = 2^-138, and at eps bits 1 the
        // contract's mean is eps where the FMA's is 2 eps.
        let s = 2.0_f32.powi(-69) * 2.0_f32.powi(-69);
        let tiny = f32::from_bits(1);
        assert_eq!((s / 4096.0 + tiny).to_bits(), 1);
        assert_eq!(kernel_mean(s, 4096, tiny).to_bits(), 2);
        assert_eq!(f32::from_bits(HF_MIN_EPS_BITS), 2.0_f32.powi(-101));
        for numerics in [RmsNormNumerics::HF_TREE_256, RmsNormNumerics::HF_ORDERED] {
            for width in [512, 1536, 4096] {
                for (bits, served) in [
                    (1, false),
                    (HF_MIN_EPS_BITS - 1, false),
                    (HF_MIN_EPS_BITS, true),
                    (1e-6_f32.to_bits(), true),
                    (1e-5_f32.to_bits(), true),
                ] {
                    let requirement = RmsNormRequirement::bf16(
                        width,
                        f32::from_bits(bits),
                        RmsNormKind::HfBf16Intermediate,
                        false,
                        numerics,
                    );
                    let built = RmsNorm::new(requirement, 32, RmsNormTarget::SM120F);
                    assert_eq!(built.is_ok(), served, "{numerics:?} {width} {bits:#x}");
                    if let Err(refusal) = built {
                        assert!(refusal.0.contains("2^-101"), "{refusal}");
                    }
                }
            }
        }
        // The other kinds' approximate numerics claim no rounding of the
        // mean, so they keep any positive eps.
        let tiny_plain = RmsNormRequirement {
            epsilon_bits: 1,
            ..plain()
        };
        assert!(RmsNorm::new(tiny_plain, 4, RmsNormTarget::SM120F).is_ok());
    }

    /// From the least served eps the kernel's mean equals the contract's at
    /// the served HF widths: s sampled over every quotient that is not
    /// exact (s below 2^-126 width) and at that edge, at the least eps, its
    /// neighbours and the served epsilons.
    #[test]
    fn from_the_least_hf_eps_the_kernels_mean_is_the_contracts() {
        let least = HF_MIN_EPS_BITS;
        let mut epsilons = vec![least, least + 1, least + 0x3f_ffff, least + 0x40_0000];
        epsilons.extend([1e-6_f32.to_bits(), 1e-5_f32.to_bits()]);
        for width in [512_u32, 1536, 4096] {
            let top = (2.0_f32.powi(-126) * width as f32).to_bits();
            let samples = (0..=top).step_by(997).chain([top, top - 1]);
            for s in samples.map(f32::from_bits) {
                for &bits in &epsilons {
                    let eps = f32::from_bits(bits);
                    assert_eq!(
                        kernel_mean(s, width, eps).to_bits(),
                        (s / width as f32 + eps).to_bits(),
                        "width {width} s {:#x} eps {bits:#x}",
                        s.to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn refuses_vector_loop_index_overflow() {
        let requirement = RmsNormRequirement {
            width: (i32::MAX as u32) & !7,
            ..plain()
        };
        let error = RmsNorm::new(requirement, 1, RmsNormTarget::SM120F).unwrap_err();
        assert!(error.0.contains("loop index"), "{error}");
        // The largest aligned group whose final increment still fits in int.
        let bounded = RmsNormRequirement {
            width: ((i32::MAX as u32) - (256 * 8 - 8)) & !7,
            ..requirement
        };
        assert!(RmsNorm::new(bounded, 1, RmsNormTarget::SM120F).is_ok());
    }

    #[test]
    fn bounds_hardware_and_prepared_resources() {
        assert!(RmsNorm::new(plain(), 3, RmsNormTarget::SM120F).is_err());
        for target in [
            RmsNormTarget {
                max_threads: 128,
                ..RmsNormTarget::SM120F
            },
            RmsNormTarget {
                shared_bytes: 0,
                ..RmsNormTarget::SM120F
            },
            RmsNormTarget {
                pdl: false,
                ..RmsNormTarget::SM120F
            },
        ] {
            assert!(RmsNorm::new(plain(), 4, target).is_err());
        }
        let hf = RmsNormRequirement::bf16(
            4096,
            1e-6,
            RmsNormKind::HfBf16Intermediate,
            false,
            RmsNormNumerics::HF_ORDERED,
        );
        assert!(
            RmsNorm::new(
                hf,
                4,
                RmsNormTarget {
                    shared_bytes: 4096,
                    ..RmsNormTarget::SM120F
                }
            )
            .is_err()
        );
        assert!(
            RmsNorm::new(
                RmsNormRequirement { width: 32768, ..hf },
                4,
                RmsNormTarget::SM120F
            )
            .is_err()
        );
        // The ordered squares and eight sums are static: 12,280 columns
        // fill the 48 KiB, and nvcc refuses 12,288 (0xc010 bytes) although
        // the target's dynamic capacity is larger.
        let at = |width| RmsNorm::new(RmsNormRequirement { width, ..hf }, 4, RmsNormTarget::SM120F);
        assert_eq!(at(12280).unwrap().shared_bytes(), 48 * 1024);
        assert!(at(12288).unwrap_err().0.contains("static shared"));
    }

    #[test]
    fn full_contract_binding_and_scalar_arguments() {
        let req = RmsNormRequirement {
            epsilon_bits: 2e-6_f32.to_bits(),
            ..plain()
        };
        let selected = RmsNorm::new(req, 4, RmsNormTarget::SM120F).unwrap();
        selected.require(&req).unwrap();
        assert!(selected.require(&plain()).is_err());
        let hf = RmsNormRequirement {
            kind: RmsNormKind::HfBf16Intermediate,
            numerics: Some(RmsNormNumerics::HF_ORDERED),
            ..req
        };
        assert!(RmsNorm::new(hf, 4, RmsNormTarget::SM120F).is_ok());
        assert!(selected.require(&hf).is_err());
        let args = selected.arguments(buffers()).unwrap();
        assert_eq!(args.0.len(), 7);
        assert_eq!(args.0[6], Argument::U32(req.epsilon_bits));
        assert_eq!(args.0[2], Argument::NullDevicePointer);
        let mut supplied = buffers();
        supplied.residual = Some(FusedResidual {
            input: 5,
            output: 6,
        });
        assert!(selected.arguments(supplied).is_err());
        let residual = RmsNorm::new(
            RmsNormRequirement {
                fused_residual: true,
                ..req
            },
            4,
            RmsNormTarget::SM120F,
        )
        .unwrap();
        assert!(residual.arguments(buffers()).is_err());
        let mut supplied = buffers();
        supplied.residual = Some(FusedResidual {
            input: 5,
            output: 5,
        });
        assert!(residual.arguments(supplied).is_ok());
        let mut supplied = buffers();
        supplied.residual = Some(FusedResidual {
            input: 5,
            output: supplied.y,
        });
        assert!(residual.arguments(supplied).is_err());
    }
}
