//! Stacked strip chains (`cuda/gemm/strip_chains_stacked_ordered.cuh`, the
//! kernel of `strip_chains_stacked.cuh` with each ring stage released only
//! after a `fence.proxy.async` orders its reads, as FIX-SC1's ordered strip
//! chains do): several
//! BF16 strip-chain decode GEMMs (`DenseKernel::StripChains`) that read the
//! same activation rows, in one launch. Each part keeps its own weights,
//! tensor map, tiles and per-element arithmetic, so its output elements
//! are bitwise those of its own strip-chain entry (and of the one-split
//! tiled GEMM), whatever the strip and split counts; it writes its first
//! `width` columns at `row_stride` elements per token row, which places
//! them where a layout copy of the part's contiguous output would.
//!
//! With `prefetch`, each CTA also asks L2, before its grid dependency wait,
//! for the weight chunks its ring does not stage ahead, so a launch that
//! starts early behind PDL streams its weights while earlier kernels run.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, DType, DenseKernel, LaunchPlan, Refusal, Target, TensorMap,
    Validate,
};

use super::{DenseRequest, instantiate, plan as gemm_plan, row_counts, tensor_maps};

/// The entries' header, relative to the CUDA source root.
pub const HEADER: &str = "gemm/strip_chains_stacked_ordered.cuh";
/// The most parts one entry stacks (`StackedParts` in the header).
pub const MAX_PARTS: usize = 6;

/// One stacked GEMM's weight rows and output placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    /// Weight rows (the part's own GEMM's n, padding included).
    pub rows: u32,
    /// Leading output columns written, at most `rows`.
    pub width: u32,
    /// Output elements per token row, at least `width`.
    pub row_stride: u32,
}

impl Part {
    /// A part whose contiguous `[tokens, rows]` output is written as is.
    pub fn contiguous(rows: u32) -> Self {
        Self {
            rows,
            width: rows,
            row_stride: rows,
        }
    }
}

/// One stacked entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stacked {
    pub symbol: String,
    /// Token rows of the bucket (at most 32).
    pub capacity: u32,
    pub k: u32,
    /// The output type every part writes (BF16 or F32).
    pub output: DType,
    pub parts: Vec<Part>,
    pub strips: u8,
    pub splits: u8,
    pub prefetch: bool,
}

impl Stacked {
    /// Part `index`'s own strip-chain entry at this entry's strips and
    /// splits: the tiles, ring, tensor maps and arithmetic the part keeps.
    pub fn part_key(&self, index: usize) -> Result<CompileKey, Refusal> {
        let part = self
            .parts
            .get(index)
            .ok_or(Refusal("stacked GEMM part out of range"))?;
        Ok(instantiate(DenseRequest {
            query: super::bf16_query(self.capacity, part.rows, self.k, self.output),
            kernel: DenseKernel::StripChains {
                strips: self.strips,
                splits: self.splits,
            },
            target: Target::Sm120f,
            symbol: "chop_gemm_stacked_part_v1".into(),
            knobs: None,
        })?
        .key)
    }

    fn check(&self) -> Result<Vec<CompileKey>, Refusal> {
        if self.parts.is_empty() || self.parts.len() > MAX_PARTS {
            return Err(Refusal("a stacked GEMM holds 1 to 6 parts"));
        }
        if !matches!(self.output, DType::Bf16 | DType::F32) {
            return Err(Refusal("a stacked GEMM writes BF16 or F32 outputs"));
        }
        if self
            .parts
            .iter()
            .any(|p| p.width == 0 || p.width > p.rows || p.row_stride < p.width)
        {
            return Err(Refusal(
                "a stacked part writes 1 to its rows' columns within its row stride",
            ));
        }
        let keys = (0..self.parts.len())
            .map(|i| self.part_key(i))
            .collect::<Result<Vec<_>, _>>()?;
        if keys.iter().any(|key| key.knobs != keys[0].knobs) {
            return Err(Refusal("stacked parts must share tiles and ring stages"));
        }
        Ok(keys)
    }

    /// Rows per CTA tile: 16 per strip.
    pub fn tile_m(&self) -> u32 {
        16 * u32::from(self.strips)
    }

    /// The stacked tiles, every part's in part order.
    pub fn tiles(&self) -> u32 {
        self.parts
            .iter()
            .map(|p| p.rows.div_ceil(self.tile_m()))
            .sum()
    }

    /// The launch over `live` rows: one CTA per stacked tile and K split,
    /// part 0's strip-chain block and ring, and one F32 temporary tile per
    /// (stacked tile, 64-K chunk) above one split, as `gemm::plan` gives a
    /// strip-chain entry.
    pub fn plan(&self, live: u32) -> Result<LaunchPlan, Refusal> {
        let keys = self.check()?;
        let part = gemm_plan(&keys[0], row_counts(&keys[0], live))?;
        let tiles = u64::from(self.tiles());
        let knobs = keys[0].knobs;
        let chunks = u64::from(self.k.div_ceil(u32::from(knobs.tile_k)));
        let temporaries = if self.splits == 1 {
            0
        } else {
            tiles * chunks * u64::from(knobs.tile_m) * u64::from(knobs.tile_n)
        };
        let result = LaunchPlan {
            grid: [
                u32::try_from(tiles * u64::from(self.splits))
                    .map_err(|_| Refusal("stacked GEMM grid exceeds u32"))?,
                1,
                1,
            ],
            workspace_bytes: (temporaries + tiles) * 4,
            ..part
        };
        result.validate()?;
        Ok(result)
    }

    /// External ticket bytes, zeroed by the owner before the first launch:
    /// one per stacked tile above one split.
    pub fn semaphore_bytes(&self) -> u64 {
        if self.splits == 1 {
            0
        } else {
            u64::from(self.tiles()) * 4
        }
    }

    /// The parameters: each part's weight map, the activation map, each
    /// part's output, workspace, tickets, the live rows, K, splits, the
    /// output type, then each part's rows, width and row stride.
    pub fn params(&self) -> Vec<EntryParam> {
        let parts = self.parts.len();
        let mut params: Vec<EntryParam> = (0..parts)
            .map(|p| EntryParam::TensorMap {
                name: format!("map_a{p}"),
            })
            .collect();
        params.push(EntryParam::TensorMap {
            name: "map_b".into(),
        });
        params.extend((0..parts).map(|p| EntryParam::DevicePointer {
            name: format!("output{p}"),
            cxx_type: "void".into(),
        }));
        params.push(EntryParam::DevicePointer {
            name: "partial".into(),
            cxx_type: "float".into(),
        });
        params.push(EntryParam::DevicePointer {
            name: "semaphores".into(),
            cxx_type: "uint32_t".into(),
        });
        for name in ["live_m", "k", "splits", "output_f32"] {
            params.push(EntryParam::U32 { name: name.into() });
        }
        for field in ["n", "width", "stride"] {
            params.extend((0..parts).map(|p| EntryParam::U32 {
                name: format!("{field}{p}"),
            }));
        }
        params
    }

    /// The entry: the parts' geometry gathered, then the stacked kernel.
    pub fn entry(&self) -> Result<EntrySpec, Refusal> {
        let keys = self.check()?;
        let knobs = keys[0].knobs;
        let list = |field: &str| {
            (0..self.parts.len())
                .map(|p| format!("{field}{p}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let maps = (0..self.parts.len())
            .map(|p| format!("&map_a{p}"))
            .collect::<Vec<_>>()
            .join(",");
        let body = format!(
            "const chop::gemm::StackedParts<{parts}> parts{{{{{maps}}},{{{outputs}}},{{{n}}},{{{width}}},{{{stride}}}}};\n\
             chop::gemm::run_a16_strips_stacked_ordered<{parts}, {strips}, {tile_n}, {stages}, {prefetch}>(parts, map_b, partial, semaphores, live_m, k, splits, output_f32);",
            parts = self.parts.len(),
            outputs = list("output"),
            n = list("n"),
            width = list("width"),
            stride = list("stride"),
            strips = self.strips,
            tile_n = knobs.tile_n,
            stages = knobs.stages,
            prefetch = self.prefetch,
        );
        // Two strip-chain CTAs share each SM, as `gemm::cubin_spec` declares.
        Ok(EntrySpec::new(
            self.symbol.clone(),
            self.params(),
            body,
            (u32::from(self.strips) + 1) * 32,
            2,
        ))
    }

    /// Each part's weight map over `weights[p]` and the activation map
    /// over `activation`: the maps of the part's own strip-chain entry.
    pub fn tensor_maps(
        &self,
        weights: &[u64],
        activation: u64,
    ) -> Result<(Vec<TensorMap>, TensorMap), Refusal> {
        let keys = self.check()?;
        if weights.len() != keys.len() {
            return Err(Refusal("a stacked GEMM takes one weight per part"));
        }
        let mut activation_map = None;
        let mut maps = Vec::with_capacity(keys.len());
        for (key, &weight) in keys.iter().zip(weights) {
            let [map_a, map_b] = tensor_maps(key, weight, activation)?;
            maps.push(map_a);
            activation_map.get_or_insert(map_b);
        }
        Ok((maps, activation_map.expect("at least one part")))
    }

    /// The launch's arguments over `live` rows, in parameter order.
    #[allow(clippy::too_many_arguments)]
    pub fn arguments(
        &self,
        live: u32,
        weights: &[u64],
        activation: u64,
        outputs: &[u64],
        partial: u64,
        semaphores: u64,
    ) -> Result<ArgumentList, Refusal> {
        self.plan(live)?;
        if outputs.len() != self.parts.len() {
            return Err(Refusal("a stacked GEMM takes one output per part"));
        }
        let (maps, map_b) = self.tensor_maps(weights, activation)?;
        let mut args: Vec<Argument> = maps.into_iter().map(Argument::TensorMap).collect();
        args.push(Argument::TensorMap(map_b));
        args.extend(outputs.iter().map(|&o| Argument::DeviceBuffer(o)));
        args.push(Argument::DeviceBuffer(partial));
        args.push(Argument::DeviceBuffer(semaphores));
        args.extend(
            [
                live,
                self.k,
                u32::from(self.splits),
                u32::from(self.output == DType::F32),
            ]
            .map(Argument::U32),
        );
        let fields: [fn(&Part) -> u32; 3] = [|p| p.rows, |p| p.width, |p| p.row_stride];
        for field in fields {
            args.extend(self.parts.iter().map(|p| Argument::U32(field(p))));
        }
        let args = ArgumentList(args);
        args.validate()?;
        Ok(args)
    }
}

/// A cubin named `name` holding `entries`.
pub fn cubin_spec(name: &str, entries: &[Stacked]) -> Result<CubinSpec, Refusal> {
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: entries
            .iter()
            .map(Stacked::entry)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qkv(capacity: u32, strips: u8, splits: u8) -> Stacked {
        Stacked {
            symbol: "chop_gemm_stacked_test_v1".into(),
            capacity,
            k: 4096,
            output: DType::Bf16,
            parts: (0..3)
                .map(|_| Part {
                    rows: 2048,
                    width: 2048,
                    row_stride: 6144,
                })
                .collect(),
            strips,
            splits,
            prefetch: false,
        }
    }

    #[test]
    fn plans_follow_each_part_strip_entry() {
        for (capacity, strips, splits) in [(1, 2, 1), (4, 1, 2), (32, 4, 1), (16, 2, 2)] {
            let stacked = qkv(capacity, strips, splits);
            let plan = stacked.plan(capacity).unwrap();
            let key = stacked.part_key(0).unwrap();
            let part = gemm_plan(&key, row_counts(&key, capacity)).unwrap();
            let tiles = 3 * 2048 / (16 * u32::from(strips));
            assert_eq!(plan.grid[0], tiles * u32::from(splits));
            assert_eq!(plan.block, part.block);
            assert_eq!(plan.dynamic_shared_bytes, part.dynamic_shared_bytes);
            // Three parts' temporaries and tickets, one part's times three.
            assert_eq!(plan.workspace_bytes, 3 * part.workspace_bytes);
            assert_eq!(stacked.semaphore_bytes() > 0, splits > 1);
        }
    }

    #[test]
    fn refuses_misplaced_parts() {
        let mut stacked = qkv(4, 2, 1);
        stacked.parts[1].width = 4096;
        assert!(stacked.plan(4).is_err());
        let mut stacked = qkv(4, 2, 1);
        stacked.parts[2].row_stride = 1024;
        assert!(stacked.plan(4).is_err());
        let mut stacked = qkv(4, 2, 1);
        stacked.parts = vec![Part::contiguous(64); 7];
        assert!(stacked.entry().is_err());
    }

    #[test]
    fn arguments_follow_the_parameters() {
        let stacked = qkv(8, 2, 1);
        let args = stacked
            .arguments(5, &[1, 2, 3], 4, &[5, 5, 5], 6, 7)
            .unwrap();
        assert_eq!(args.0.len(), stacked.params().len());
        let spec = cubin_spec("stacked_test", std::slice::from_ref(&stacked)).unwrap();
        assert!(
            spec.entries[0]
                .body
                .contains("run_a16_strips_stacked_ordered<3, 2, 8,")
        );
    }
}
