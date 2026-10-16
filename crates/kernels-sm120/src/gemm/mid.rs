//! Medium-M dense GEMMs for the fragment prefill shapes at a few dozen
//! live rows: verify buckets above the decode batch and decode above 32
//! sequences.
//!
//! Each catalog entry names the fragment prefill entry it stands in for
//! (`reference`) and reproduces that entry's outputs bit for bit at the same
//! live M. Both run `run_fragment_prefill` (`cuda/gemm/fragment_prefill.cuh`),
//! in which every output element sums the same fixed K segments in the same
//! order whatever the tile or schedule; the medium entry stages 64-token
//! tiles instead of 128, which a live M of 64 or less leaves half unused,
//! and so holds a deeper ring. It reads the same weights, scales and
//! activation buffer and writes the same output; the argument list is the
//! reference's. Only the activation tensor map's box (`tokens` rows) and the
//! launch plan differ.
//!
//! The kernels are generic over shapes: N, K and the output type are
//! runtime arguments, so six kernels (weight format, SwiGLU) serve every
//! model; a bundle's cubin holds the ones its profiles' selected entries
//! use. An entry's shape is its reference's catalog query, and the live rows
//! it is selected for are an explicit field (`rows`), never inferred from a
//! model's widths.
//!
//! Bundles call [`select`] with the prefill entry they would launch and the
//! live rows; when it returns an entry, they build the activation map with
//! [`Entry::activation_box_rows`] rows and launch [`Entry::plan`] and
//! [`Entry::arguments`] instead.

use super::{GemmBuffers, SM_COUNT};
use crate::arithmetic::{self, DeclaredCubin};
use crate::invariance::Invariance;
use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, GemmQuery, LaunchPlan, Refusal, StaticQuery, SwigluColumns,
    TensorMap, TensorMapElement, TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill,
    TensorMapSwizzle, Validate, WeightFormat,
};
use std::ops::RangeInclusive;
use std::path::PathBuf;

/// Math warps plus the four-warp producer group, as the prefill kernel.
const THREADS: u32 = 12 * 32;
/// Math warps per CTA.
const MATH_WARPS: u32 = 8;
/// Partial tiles per CTA: the prefill's `PREFILL_SEGMENTS`.
const PARTIALS: u32 = 8;
/// Tokens per CTA tile.
const TOKENS: u32 = 64;
/// Live rows every entry is selected for. The 64-token kernel was faster
/// than the prefill entry for every shape at these rows (kernel-lab
/// 48-layer graph bench, `cuda/gemm/README.md`).
const ROWS: RangeInclusive<u32> = 33..=64;

/// One compiled kernel: weight format and SwiGLU epilogue. N, K, live M and
/// the output type are runtime arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Kernel {
    /// FP8 E4M3 weights with F32 scales; BF16 otherwise.
    pub fp8: bool,
    /// FP8 scales per 64 weight rows (`Fp8Rows64x128`) instead of 128.
    pub rows64: bool,
    pub swiglu: bool,
}

impl Kernel {
    /// BF16, 128-row and 64-row FP8 scales, each with and without SwiGLU.
    pub const ALL: [Kernel; 6] = {
        let mut all = [Kernel {
            fp8: false,
            rows64: false,
            swiglu: false,
        }; 6];
        let mut i = 0;
        while i < 6 {
            all[i] = Kernel {
                fp8: i % 3 != 0,
                rows64: i % 3 == 2,
                swiglu: i / 3 == 1,
            };
            i += 1;
        }
        all
    };

    fn of(weight: WeightFormat, swiglu: bool) -> Self {
        Kernel {
            fp8: weight != WeightFormat::Bf16,
            rows64: matches!(weight, WeightFormat::Fp8Rows64x128(_)),
            swiglu,
        }
    }

    fn weight_tag(self) -> &'static str {
        match (self.fp8, self.rows64) {
            (false, _) => "bf16",
            (true, false) => "fp8b",
            (true, true) => "fp8r64",
        }
    }

    pub fn symbol(self) -> String {
        format!(
            "chop_gemm_mid_{}{}_t{TOKENS}_v1",
            self.weight_tag(),
            if self.swiglu { "_swiglu" } else { "" },
        )
    }

    /// Stage bytes: the activation rows' 128-byte swizzled slice of one 64-K
    /// stage and eight weight strips.
    fn stage_bytes(self) -> u32 {
        TOKENS * 128 + 8 * if self.fp8 { 1024 } else { 2048 }
    }

    /// As many stages as fit 96 KiB, plus 1 KiB to align the swizzled ring
    /// (`FragmentPrefillTraits::shared_bytes`).
    pub fn dynamic_shared_bytes(self) -> u32 {
        let stage = self.stage_bytes();
        96 * 1024 / stage * stage + 1024
    }

    /// F32 values of one partial tile: eight warps of a 64-token tile's
    /// share of eight strips (two each) by four n8 tiles.
    fn partial_floats(self) -> u32 {
        MATH_WARPS * 2 * 4 * 32 * 4
    }

    fn entry_spec(self) -> EntrySpec {
        let mut entry = EntrySpec::new(
            self.symbol(),
            super::params(true, false),
            format!(
                "chop::gemm::run_fragment_prefill<chop::gemm::WeightKind::{}, {}, {TOKENS}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual);",
                match (self.fp8, self.rows64) {
                    (false, _) => "bf16",
                    (true, false) => "fp8_block128",
                    (true, true) => "fp8_row64",
                },
                self.swiglu,
            ),
            THREADS,
            1,
        );
        // As the prefill entries: nvcc may reserve a small stack frame.
        entry.allowed_local_bytes = 1024;
        entry
    }
}

/// A medium-M stand-in for one fragment prefill entry.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The reference's profile.
    pub profile: Profile,
    /// The fragment prefill entry whose outputs this entry reproduces.
    pub reference: String,
    /// The reference's query: N, K, formats, epilogue and the activation
    /// capacity its tensor map addresses.
    pub query: GemmQuery,
    pub kernel: Kernel,
    /// The reference's fixed K segments.
    pub segments: u8,
    /// Live rows `select` picks this entry for.
    pub rows: RangeInclusive<u32>,
    /// No claim of its own: the entry reproduces its reference's bits at
    /// the same live M (`gemm_mid`).
    pub invariance: Invariance,
}

/// The fragment prefill entries with a medium-M stand-in: the models'
/// in `catalogs`, every model's base (rank-invariant) references first, then
/// the others, each model's in its own order.
fn references(catalogs: &Catalogs) -> Vec<(Profile, String)> {
    let symbols = catalogs
        .models()
        .iter()
        .flat_map(|model| (model.gemm_mid)())
        .collect::<Vec<_>>();
    let (base, others): (Vec<_>, Vec<_>) = symbols
        .into_iter()
        .partition(|(profile, _)| profile.is_base());
    base.into_iter().chain(others).collect()
}

/// Every medium-M entry of `catalogs`: one per reference.
pub fn catalog(catalogs: &Catalogs) -> Vec<Entry> {
    catalogs.mid_entries().to_vec()
}

/// Builds [`catalog`]'s entries; `Catalogs` retains them.
pub(crate) fn entries(catalogs: &Catalogs) -> Vec<Entry> {
    let dense = super::catalog(catalogs);
    let mut entries = Vec::new();
    for (profile, reference) in references(catalogs) {
        let item = dense
            .iter()
            .find(|item| item.symbol == reference)
            .unwrap_or_else(|| panic!("{reference} is not a GEMM catalog entry"));
        assert_eq!(
            item.profile, profile,
            "{reference} is another profile's entry"
        );
        let StaticQuery::Gemm(query) = item.key.query else {
            unreachable!("GEMM catalog entries hold GEMM queries")
        };
        let segments = super::prefill_segments(&item.key)
            .unwrap_or_else(|_| panic!("{reference} is not a fragment prefill entry"));
        entries.push(Entry {
            profile,
            reference: reference.clone(),
            query,
            kernel: Kernel::of(query.weight, query.epilogues.swiglu.is_some()),
            segments,
            rows: ROWS,
            invariance: Invariance::NONE,
        });
    }
    entries
}

/// The medium-M entry to launch instead of fragment prefill entry
/// `reference` at `live_m` rows, when one is selected there. Its outputs are
/// bitwise equal to the reference's at the same live M.
pub fn select(catalogs: &Catalogs, reference: &str, live_m: u32) -> Option<Entry> {
    catalogs
        .mid_entries()
        .iter()
        .find(|entry| entry.reference == reference && entry.rows.contains(&live_m))
        .cloned()
}

/// Opaque buffers of one launch: the reference prefill entry's.
#[derive(Clone, Copy, Debug)]
pub struct Buffers {
    pub activation: u64,
    pub weight: u64,
    pub weight_scales: Option<u64>,
    pub output: u64,
    pub partial: u64,
    pub semaphores: u64,
}

impl Entry {
    /// Rows of the activation tensor map's box.
    pub fn activation_box_rows(&self) -> u32 {
        TOKENS
    }

    /// Largest live M the kernel runs: one 64-token tile per feature tile.
    pub fn max_rows(&self) -> u32 {
        TOKENS
    }

    fn output_rows(&self) -> u32 {
        if self.query.epilogues.swiglu.is_some() {
            self.query.n / 2
        } else {
            self.query.n
        }
    }

    fn check(&self, live_m: u32) -> Result<(), Refusal> {
        let q = self.query;
        if live_m == 0 || live_m > self.max_rows() || live_m > q.m_capacity {
            return Err(Refusal("live M is outside the medium GEMM's rows"));
        }
        if q.k == 0
            || !q.k.is_multiple_of(128)
            || !self.output_rows().is_multiple_of(64)
            || q.epilogues
                .swiglu
                .is_some_and(|s| s != SwigluColumns::ConcatenatedHalves)
            || self.kernel != Kernel::of(q.weight, q.epilogues.swiglu.is_some())
        {
            return Err(Refusal("medium GEMM shape does not match its kernel"));
        }
        Ok(())
    }

    /// One persistent CTA per SM (so the stream-K waits cannot deadlock),
    /// each with the prefill's partial tiles at the 64-token tile.
    pub fn plan(&self, live_m: u32) -> Result<LaunchPlan, Refusal> {
        self.check(live_m)?;
        let plan = LaunchPlan {
            grid: [SM_COUNT, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: self.kernel.dynamic_shared_bytes(),
            workspace_bytes: u64::from(SM_COUNT * PARTIALS * self.kernel.partial_floats()) * 4,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The activation map the kernel expects: BF16 rows of K over the
    /// reference's capacity, a 64-K by `tokens` box with the 128-byte swizzle.
    pub fn activation_map(&self, buffer: u64, l2_promotion: TensorMapL2Promotion) -> TensorMap {
        let k = self.query.k;
        TensorMap {
            base_buffer: buffer,
            dims: vec![u64::from(k), u64::from(self.query.m_capacity)],
            strides_bytes: vec![2, u64::from(k) * 2],
            box_dims: vec![64, TOKENS],
            element_strides: vec![1, 1],
            swizzle: TensorMapSwizzle::B128,
            interleave: TensorMapInterleave::None,
            l2_promotion,
            oob_fill: TensorMapOobFill::Zero,
            element: TensorMapElement::Bf16,
        }
    }

    /// The reference's fifteen arguments with this entry's activation map.
    pub fn arguments(
        &self,
        live_m: u32,
        activation_map: TensorMap,
        buffers: Buffers,
    ) -> Result<ArgumentList, Refusal> {
        self.check(live_m)?;
        let expected = self.activation_map(activation_map.base_buffer, activation_map.l2_promotion);
        if activation_map != expected || activation_map.base_buffer != buffers.activation {
            return Err(Refusal(
                "activation map does not match the medium GEMM tile",
            ));
        }
        if self.kernel.fp8 != buffers.weight_scales.is_some() {
            return Err(Refusal("scale pointer does not match GEMM format"));
        }
        let args = ArgumentList(vec![
            Argument::TensorMap(activation_map),
            Argument::DeviceBuffer(buffers.weight),
            Argument::DeviceBuffer(buffers.output),
            Argument::DeviceBuffer(buffers.partial),
            Argument::DeviceBuffer(buffers.semaphores),
            Argument::U32(live_m),
            Argument::U32(self.query.n),
            Argument::U32(self.query.k),
            Argument::U32(u32::from(self.segments)),
            Argument::U32(u32::from(self.query.output == DType::F32)),
            buffers
                .weight_scales
                .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
            Argument::NullDevicePointer,
            Argument::NullDevicePointer,
            Argument::NullDevicePointer,
            Argument::NullDevicePointer,
        ]);
        args.validate()?;
        Ok(args)
    }

    /// The same arguments from a reference launch's buffers, for callers
    /// that hold [`GemmBuffers`] and a weight map.
    pub fn arguments_from(
        &self,
        live_m: u32,
        activation_map: TensorMap,
        weight: u64,
        buffers: GemmBuffers,
    ) -> Result<ArgumentList, Refusal> {
        if buffers.activation_scales.is_some()
            || buffers.residual.is_some()
            || buffers.weight_global_scale.is_some()
            || buffers.activation_global_scale.is_some()
        {
            return Err(Refusal(
                "medium GEMM takes BF16 activations and no residual",
            ));
        }
        self.arguments(
            live_m,
            activation_map.clone(),
            Buffers {
                activation: activation_map.base_buffer,
                weight,
                weight_scales: buffers.weight_scales,
                output: buffers.output,
                partial: buffers.partial,
                semaphores: buffers.semaphores,
            },
        )
    }
}

/// The shared medium-M cubin with the kernels `profile`'s selected entries
/// use (every profile's when `None`), or nothing when none is selected.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    let profiles = match profile {
        Some(owner) => vec![owner],
        None => catalog(catalogs).into_iter().map(|e| e.profile).collect(),
    };
    cubin_spec_for(catalogs, &cubin_name(profile), &profiles)
        .into_iter()
        .collect()
}

/// One cubin named `name` with the kernels the selected entries of all
/// `profiles` use, for a bundle that serves entries of several profiles
/// (a TP2 bundle also runs TP4's rank-invariant shapes); the
/// kernels are shape-generic, so one copy of each serves them all. `None`
/// when no entry is selected.
pub fn cubin_spec_for(catalogs: &Catalogs, name: &str, profiles: &[Profile]) -> Option<CubinSpec> {
    declared_cubin_spec_for(catalogs, name, profiles).map(DeclaredCubin::into_spec)
}

/// [`cubin_spec_for`] with each kernel's declared arithmetic: the
/// references and K segments of the selected entries it serves.
pub fn declared_cubin_spec_for(
    catalogs: &Catalogs,
    name: &str,
    profiles: &[Profile],
) -> Option<DeclaredCubin> {
    let mut kernels: Vec<(Kernel, Vec<(String, u8)>)> = Vec::new();
    for entry in catalog(catalogs)
        .into_iter()
        .filter(|entry| profiles.contains(&entry.profile))
    {
        let served = (entry.reference, entry.segments);
        match kernels
            .iter_mut()
            .find(|(kernel, _)| *kernel == entry.kernel)
        {
            Some((_, references)) => references.push(served),
            None => kernels.push((entry.kernel, vec![served])),
        }
    }
    kernels.sort_by_key(|(k, _)| (k.fp8, k.rows64, k.swiglu));
    let arithmetic = kernels
        .iter()
        .map(|(kernel, references)| (kernel.symbol(), Some(arithmetic::mid(references))))
        .collect();
    let spec = CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/fragment_prefill.cuh\"".into()],
        defines: vec![],
        entries: kernels.iter().map(|(k, _)| k.entry_spec()).collect(),
    };
    (!kernels.is_empty())
        .then(|| DeclaredCubin::new(spec, arithmetic).expect("medium-M kernels declare all"))
}

/// The name of `profile`'s medium-M cubin (every profile's when `None`).
pub fn cubin_name(profile: Option<Profile>) -> String {
    match profile {
        None => "gemm_mid_sm120".to_owned(),
        Some(owner) => format!("gemm_mid_{}_sm120", owner.tag()),
    }
}

/// The dynamic shared memory of a medium-M kernel symbol.
pub fn dynamic_shared_bytes(symbol: &str) -> Option<u32> {
    Kernel::ALL
        .into_iter()
        .find(|kernel| kernel.symbol() == symbol)
        .map(Kernel::dynamic_shared_bytes)
}
