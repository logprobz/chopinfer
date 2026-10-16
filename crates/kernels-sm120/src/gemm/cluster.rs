//! The shared dense cluster decode template (`cuda/gemm/cluster_decode.cuh`)
//! and its kernel-lab instances.
//!
//! One template covers the local cluster decode
//! (`run_fragment_decode_cluster` in `fragment_decode_cluster.cuh`, BF16 and
//! FP8 weights) and the fused-weight cluster decode (`run_cluster_decode` in
//! `qwen38.cuh`: one or two 128-K blocks per warp, output rows that the
//! cluster does not divide evenly, up to four weights per launch, tokens
//! staged in two passes, and the hyper-connection precompute epilogue).
//! [`Shape`] holds its compile-time parameters.
//!
//! [`lab_instances`] lists every served cluster decode entry of a consumer's
//! catalogs (the fused `_c{M}` buckets, their HC precompute producers, and the
//! dense catalog's local decode entries), with its template entry and the
//! reference that entry must reproduce. While a served entry declares its own
//! kernel ([`Served::runs_template`]), the reference is the served entry and
//! the template entry a lab instance at the same shape, ABI, threads and
//! launch plan under the served symbol with [`TEMPLATE_PREFIX`]. Fused-weight
//! entries are template instances; kernel-lab keeps their old kernel as the
//! reference ([`REFERENCE_PREFIX`]). Kernel-lab compares the two bit for bit.

use super::{GemmVariant, fused};
use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{Refusal, WeightFormat};
use std::path::PathBuf;

/// The template header.
pub const HEADER: &str = "#include \"gemm/cluster_decode.cuh\"";
/// The hyper-connection precompute epilogue for the template.
pub const HC_EPILOGUE_HEADER: &str = "#include \"gemm/qwen38_hc_bottleneck.cuh\"";
/// Prefix that replaces `chop_gemm_` in lab template instance symbols and
/// `gemm_` in their cubin names.
pub const TEMPLATE_PREFIX: &str = "chop_gemm_tpl_";
/// Prefix that replaces `chop_gemm_` in kernel-lab's frozen references of a
/// model's own kernel, once the model serves the template, and `gemm_` in
/// their cubin names.
pub const REFERENCE_PREFIX: &str = "chop_gemm_eng_";
/// Warps per CTA.
pub const WARPS: u32 = 8;
/// Dynamic shared memory an instance may use: the SM120 limit less the
/// static shared memory and slack the fused-weight planner reserves.
const SHARED_LIMIT: u32 = 101_376 - 1152;

/// Weight format: BF16, or FP8 E4M3 with F32 scales per 128-K block and
/// 128 or 64 rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Weights {
    Bf16,
    Fp8Block128,
    Fp8Rows64,
}

impl Weights {
    pub fn of(format: WeightFormat) -> Option<Self> {
        match format {
            WeightFormat::Bf16 => Some(Weights::Bf16),
            WeightFormat::Fp8Block128x128(_) => Some(Weights::Fp8Block128),
            WeightFormat::Fp8Rows64x128(_) => Some(Weights::Fp8Rows64),
            _ => None,
        }
    }

    /// `chop::gemm::WeightKind` enumerator.
    fn kind(self) -> &'static str {
        match self {
            Weights::Bf16 => "bf16",
            Weights::Fp8Block128 => "fp8_block128",
            Weights::Fp8Rows64 => "fp8_row64",
        }
    }

    /// Fragment tiles per 128-K block: k16 BF16 tiles, or k32 FP8 tiles.
    pub fn tiles_per_block(self) -> u32 {
        match self {
            Weights::Bf16 => 8,
            _ => 4,
        }
    }
}

/// The template's compile-time parameters (`Traits` and `run`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub weights: Weights,
    /// Token tile: 8, 16 or 32 rows.
    pub token_tile: u32,
    /// 128-K blocks of K.
    pub k_blocks: u32,
    /// 16-row strips per CTA.
    pub strips: u32,
    /// 128-K blocks each warp holds in registers.
    pub warp_blocks: u32,
    /// Token passes: 1, or 2 when one slab of the whole tile would not fit.
    pub passes: u32,
    /// Weights (segments) per launch.
    pub segments: u32,
    pub swiglu: bool,
}

impl Shape {
    pub fn k_parts(&self) -> u32 {
        WARPS / self.strips
    }

    /// 128-K blocks per CTA.
    pub fn chunk_blocks(&self) -> u32 {
        self.warp_blocks * self.k_parts()
    }

    /// CTAs per row tile, which form one cluster.
    pub fn splits(&self) -> u32 {
        self.k_blocks / self.chunk_blocks()
    }

    /// The parameters that decide every output bit: the format, K, the
    /// 128-K blocks each partial sums (the partials are added in K order
    /// from zero) and the epilogue. Strips, splits, token tile and passes
    /// only distribute the work.
    pub fn arithmetic(&self) -> (Weights, u32, u32, bool) {
        (self.weights, self.k_blocks, self.warp_blocks, self.swiglu)
    }

    /// The activation slab of one pass plus the F32 partial tile.
    pub fn dynamic_shared_bytes(&self) -> u32 {
        self.token_tile / self.passes * (self.chunk_blocks() * 256 + 16)
            + WARPS * 16 * self.token_tile * 4
    }

    /// The template's static assertions and the shared memory limit.
    pub fn validate(&self) -> Result<(), Refusal> {
        let tiles = self.warp_blocks * self.weights.tiles_per_block();
        let fits = matches!(self.token_tile, 8 | 16 | 32)
            && (self.passes == 1 || (self.passes == 2 && self.token_tile >= 16))
            && matches!(self.strips, 1 | 2 | 4 | 8)
            && self.warp_blocks >= 1
            && tiles.is_multiple_of(4)
            && tiles <= 32
            && self.k_blocks.is_multiple_of(self.chunk_blocks())
            && (1..=8).contains(&self.splits())
            && (1..=4).contains(&self.segments)
            && (!self.swiglu || (self.segments == 1 && self.strips >= 2))
            && (self.weights == Weights::Bf16 || self.segments == 1);
        if !fits {
            return Err(Refusal("cluster decode template shape is not compiled"));
        }
        if self.dynamic_shared_bytes() > SHARED_LIMIT {
            return Err(Refusal("cluster decode slab exceeds shared memory"));
        }
        Ok(())
    }

    /// A call of the template with C++ argument expressions: per segment
    /// its weight, output, rows and output stride, then the activation,
    /// workspace, live M, the F32 output flag, the FP8 scales and an
    /// epilogue (its type and value).
    pub fn call(&self, args: &Arguments) -> String {
        let count = self.segments as usize;
        assert!(
            [&args.weights, &args.outputs, &args.rows, &args.strides]
                .iter()
                .all(|list| list.len() == count),
            "one expression per segment"
        );
        let epilogue = args
            .epilogue
            .as_ref()
            .map_or(String::new(), |(ty, _)| format!(", {ty}"));
        let tail = match (&args.epilogue, args.scales) {
            (Some((_, value)), scales) => format!(", {}, {value}", scales.unwrap_or("nullptr")),
            (None, Some(scales)) => format!(", {scales}"),
            (None, None) => String::new(),
        };
        format!(
            "chop::gemm::cluster::run<chop::gemm::WeightKind::{}, {}, {}, {}, {}, {}, {count}, {}{epilogue}>(chop::gemm::cluster::Segments<{count}>{{{{{}}}, {{{}}}, {{{}}}, {{{}}}}}, {}, {}, {}, {}{tail});",
            self.weights.kind(),
            self.token_tile,
            self.k_blocks,
            self.strips,
            self.warp_blocks,
            self.passes,
            self.swiglu,
            args.weights.join(", "),
            args.outputs.join(", "),
            args.rows.join(", "),
            args.strides.join(", "),
            args.activation,
            args.workspace,
            args.live_m,
            args.output_f32,
        )
    }
}

/// C++ argument expressions of one template call ([`Shape::call`]).
#[derive(Clone, Debug, Default)]
pub struct Arguments {
    pub weights: Vec<String>,
    pub outputs: Vec<String>,
    pub rows: Vec<String>,
    pub strides: Vec<String>,
    pub activation: String,
    pub workspace: String,
    pub live_m: String,
    pub output_f32: String,
    pub scales: Option<&'static str>,
    /// Epilogue type and value; the template's `Store` when absent.
    pub epilogue: Option<(String, String)>,
}

/// The served entry a template instance reproduces.
#[derive(Clone, Debug)]
pub enum Served {
    /// A fused-weight cluster decode entry
    /// (`gemm::fused::Kind::ClusterDecode`).
    Fused(fused::Entry),
    /// An HC precompute producer, which extends this cluster decode entry
    /// by the bottleneck output (`fused::hc_precompute_spec`).
    FusedHcPrecompute(fused::Entry),
    /// A shared-catalog local decode entry (`run_fragment_decode_cluster`).
    Shared(GemmVariant),
}

impl Served {
    /// Whether the served entry's declared kernel is the template: a
    /// `fused::Kind::ClusterDecode` entry (and its HC precompute producer)
    /// is a template instance, while a `DenseKernel::FragmentCluster` entry
    /// runs `run_fragment_decode_cluster`. An entry on its own kernel gets a
    /// `chop_gemm_tpl_` lab instance beside it; a template entry keeps its
    /// former kernel as a frozen `chop_gemm_eng_` reference
    /// ([`REFERENCE_PREFIX`]).
    pub fn runs_template(&self) -> bool {
        match self {
            Served::Fused(_) | Served::FusedHcPrecompute(_) => true,
            Served::Shared(_) => false,
        }
    }
}

/// A served cluster decode entry with its template entry and the reference
/// the template must reproduce bit for bit.
#[derive(Clone, Debug)]
pub struct Instance {
    pub served: Served,
    pub served_symbol: String,
    pub served_cubin: String,
    pub shape: Shape,
    /// The template entry: the served entry once its model serves the
    /// template, otherwise a lab instance with the served ABI, threads and
    /// minimum blocks, a template body and the `chop_gemm_tpl_` symbol.
    pub template: EntrySpec,
    pub template_cubin: String,
    /// The model's own kernel: the served entry, or kernel-lab's frozen
    /// `chop_gemm_eng_` copy once the model serves the template.
    pub reference_symbol: String,
    pub reference_cubin: String,
}

impl Instance {
    pub fn profile(&self) -> Profile {
        match &self.served {
            Served::Fused(entry) | Served::FusedHcPrecompute(entry) => entry.profile,
            Served::Shared(variant) => variant.profile,
        }
    }

    /// The template entry's symbol.
    pub fn symbol(&self) -> &str {
        &self.template.symbol
    }

    /// Whether the served entry is the template.
    pub fn serves_template(&self) -> bool {
        self.served.runs_template()
    }
}

/// `chop_gemm_x` becomes `chop_gemm_tpl_x`, and cubin `gemm_x` becomes
/// `gemm_tpl_x`.
pub fn template_symbol(served: &str) -> String {
    served.replacen("chop_gemm_", TEMPLATE_PREFIX, 1)
}

/// `chop_gemm_x` becomes `chop_gemm_eng_x`, and cubin `gemm_x` becomes
/// `gemm_eng_x`.
pub fn reference_symbol(served: &str) -> String {
    served.replacen("chop_gemm_", REFERENCE_PREFIX, 1)
}

fn instance(
    served: Served,
    served_spec: &EntrySpec,
    served_cubin: String,
    shape: Shape,
    body: String,
) -> Instance {
    shape
        .validate()
        .expect("served cluster decode shapes compile");
    let mut item = Instance {
        served,
        served_symbol: served_spec.symbol.clone(),
        template_cubin: served_cubin.replacen("gemm_", "gemm_tpl_", 1),
        reference_symbol: served_spec.symbol.clone(),
        reference_cubin: served_cubin.clone(),
        served_cubin,
        shape,
        template: served_spec.clone(),
    };
    if item.serves_template() {
        assert_eq!(
            served_spec.body, body,
            "{} does not run the template",
            served_spec.symbol
        );
        item.template_cubin = item.served_cubin.clone();
        item.reference_symbol = reference_symbol(&served_spec.symbol);
        item.reference_cubin = item.served_cubin.replacen("gemm_", "gemm_eng_", 1);
    } else {
        item.template.symbol = template_symbol(&served_spec.symbol);
        item.template.body = body;
        // The template needs no local memory, whatever its served entry
        // allows.
        item.template.allowed_local_bytes = 0;
    }
    item
}

/// Every served cluster decode entry of `catalogs` with its template
/// instance: the fused-weight `_c{M}` buckets per profile, their HC
/// precompute producers, then the dense catalog's local decode entries.
pub fn lab_instances(catalogs: &Catalogs) -> Vec<Instance> {
    let mut instances = Vec::new();
    let fused = fused::catalog(catalogs);
    for entry in fused.iter().cloned() {
        let Some((shape, body)) = entry.template() else {
            continue;
        };
        let spec = entry.entry_spec();
        let cubin = format!("gemm_{}_own_sm120", entry.profile.tag());
        instances.push(instance(Served::Fused(entry), &spec, cubin, shape, body));
    }
    for entry in fused {
        let (Ok(spec), Ok((shape, body))) = (
            fused::hc_precompute_spec(&entry),
            fused::hc_precompute_template(&entry),
        ) else {
            continue;
        };
        let cubin = format!("gemm_{}_hc_precompute_sm120", entry.profile.tag());
        instances.push(instance(
            Served::FusedHcPrecompute(entry),
            &spec,
            cubin,
            shape,
            body,
        ));
    }
    // Served models only: the standard profile's entries are catalog probes.
    for variant in super::catalog(catalogs)
        .into_iter()
        .filter(|variant| variant.profile.model().is_some())
    {
        let Some((shape, body)) = super::local_decode_template(&variant) else {
            continue;
        };
        let spec = super::cubin_spec(std::slice::from_ref(&variant.instantiation))
            .expect("catalog entries are valid")
            .entries
            .remove(0);
        let cubin = format!("gemm_{}_sm120", variant.profile.tag());
        instances.push(instance(Served::Shared(variant), &spec, cubin, shape, body));
    }
    instances
}

/// One lab cubin per served cubin, holding the template instances of
/// entries whose model does not serve the template yet.
pub fn lab_cubin_specs(catalogs: &Catalogs) -> Vec<CubinSpec> {
    let mut specs: Vec<CubinSpec> = Vec::new();
    for item in lab_instances(catalogs) {
        if item.serves_template() {
            continue;
        }
        let index = match specs
            .iter()
            .position(|spec| spec.name == item.template_cubin)
        {
            Some(index) => index,
            None => {
                let mut headers = vec![HEADER.to_owned()];
                if matches!(item.served, Served::FusedHcPrecompute(_)) {
                    headers.push(HC_EPILOGUE_HEADER.to_owned());
                }
                specs.push(CubinSpec {
                    name: item.template_cubin.clone(),
                    sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
                    include_dirs: vec![],
                    headers,
                    defines: vec![],
                    entries: Vec::new(),
                });
                if matches!(item.served, Served::FusedHcPrecompute(_)) {
                    catalogs.fused_sources(item.profile(), specs.last_mut().unwrap());
                }
                specs.len() - 1
            }
        };
        specs[index].entries.push(item.template);
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_name_every_argument() {
        let shape = Shape {
            weights: Weights::Bf16,
            token_tile: 8,
            k_blocks: 20,
            strips: 2,
            warp_blocks: 1,
            passes: 1,
            segments: 2,
            swiglu: false,
        };
        let args = Arguments {
            weights: vec!["weight0".into(), "weight1".into()],
            outputs: vec!["output0".into(), "output1".into()],
            rows: vec!["320u".into(), "4u".into()],
            strides: vec!["stride0".into(), "stride1".into()],
            activation: "activation".into(),
            workspace: "partial".into(),
            live_m: "live_m".into(),
            output_f32: "output_f32".into(),
            scales: None,
            epilogue: None,
        };
        assert_eq!(
            shape.call(&args),
            "chop::gemm::cluster::run<chop::gemm::WeightKind::bf16, 8, 20, 2, 1, 1, 2, false>(chop::gemm::cluster::Segments<2>{{weight0, weight1}, {output0, output1}, {320u, 4u}, {stride0, stride1}}, activation, partial, live_m, output_f32);"
        );
        let hc = Arguments {
            epilogue: Some(("E".into(), "E{bottleneck}".into())),
            ..args.clone()
        };
        assert!(shape.call(&hc).ends_with(", 2, false, E>(chop::gemm::cluster::Segments<2>{{weight0, weight1}, {output0, output1}, {320u, 4u}, {stride0, stride1}}, activation, partial, live_m, output_f32, nullptr, E{bottleneck});"));
        let fp8 = Shape {
            weights: Weights::Fp8Block128,
            warp_blocks: 4,
            segments: 1,
            ..shape
        };
        assert!(
            fp8.validate().is_err(),
            "a 20-block K does not split 16 blocks per CTA"
        );
        let bad = Shape { segments: 2, ..fp8 };
        assert!(bad.validate().is_err());
    }
}
