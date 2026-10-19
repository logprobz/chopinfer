//! The clamped SwiGLU with BF16 rounding and ceil E4M3/32 quantization
//! (`cuda/row/silu_mul_quantize.cuh`): F32 gate and up rows in, E4M3 codes
//! and UE8M0 scales out, the code and scale rule of the block quantizer
//! (`row/block_quantize.cuh`). One CTA of 32 threads per row validates all
//! groups before writing; each row reads only its own columns, so an entry
//! claims one-row bits up to its capacity, any chunk split and row
//! independence. Each column is a scalar load, so the gate and up views
//! need only F32 alignment. An entry refuses a clamp whose product can
//! round to a BF16 infinity ([`clamp_fits_bf16`]). Device failures report
//! sticky status and zero the whole row; no data path traps.

use super::{pack_arguments, ptr};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, NumericalContract, Refusal, Validate,
};
use kernel_core::gated_mlp::SwigluClamp;
use kernel_core::gated_mlp::quantized::SiluMulQuantizeQuery;
use kernel_core::properties::Invariance;

pub const HEADER: &str = "row/silu_mul_quantize.cuh";
pub const LAB_CUBIN: &str = "silu_mul_quantize_sm120";
pub const LAB_CAPACITY: u32 = 16;
/// Columns of one quantization group.
pub const GROUP: u32 = 32;
/// The largest finite BF16 value, `0x7f7f`.
const BF16_MAX: f64 = 3.389_531_389_251_535_5e38;

/// Whether every finite gate and up within `clamp` gives a finite BF16
/// product: `max(gate_max, 1) * up_limit` at most the largest BF16. The
/// SiLU of a gate at most `gate_max` is at most `max(gate_max, 1)` in
/// magnitude after F32 rounding (`g / fl(1 + t)` with `t >= 0` is at most
/// `g` for `g >= 0`, and under 0.28 in magnitude for `g < 0`), the up value
/// is at most `up_limit` in magnitude, and the F32 product and its BF16
/// rounding are monotone, so the bound holds through every rounding. For a
/// symmetric clamp of at least 1 this is its square.
pub fn clamp_fits_bf16(clamp: SwigluClamp) -> bool {
    f64::from(clamp.gate_max().max(1.0)) * f64::from(clamp.up_limit()) <= BF16_MAX
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub query: SiluMulQuantizeQuery,
    pub capacity: u32,
    pub symbol: String,
    /// Opt-in warp-parallel groups with a barrier before row output.
    pub parallel_groups: bool,
}

impl Entry {
    pub fn new(
        query: SiluMulQuantizeQuery,
        capacity: u32,
        symbol: impl Into<String>,
    ) -> Result<Self, Refusal> {
        let entry = Self {
            query,
            capacity,
            symbol: symbol.into(),
            parallel_groups: false,
        };
        entry.validate()?;
        Ok(entry)
    }

    /// Select the parallel implementation for a separately named entry.
    pub fn parallel(mut self) -> Result<Self, Refusal> {
        self.parallel_groups = true;
        self.validate()?;
        Ok(self)
    }

    pub fn threads(&self) -> u32 {
        if self.parallel_groups {
            (self.query.width / 32).min(32) * 32
        } else {
            32
        }
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        self.query.numerical_contract_id()?;
        if !clamp_fits_bf16(self.query.clamp) {
            return Err(Refusal(
                "quantized SwiGLU clamp lets the product overflow BF16",
            ));
        }
        if self.capacity == 0 || self.capacity > crate::backend::MAX_CAPACITY {
            return Err(Refusal("quantized SwiGLU capacity exceeds SM120 bounds"));
        }
        if self.symbol.is_empty()
            || !self
                .symbol
                .bytes()
                .enumerate()
                .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            return Err(Refusal("quantized SwiGLU symbol must be a C identifier"));
        }
        self.plan(self.capacity)?;
        Ok(())
    }

    /// The plan of a launch of `rows` token rows (at most the capacity):
    /// one CTA per row. The kernel reads the live count on the
    /// device, so rows past it write nothing.
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows > self.capacity {
            return Err(Refusal("quantized SwiGLU rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows.max(1), 1, 1],
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn invariance(&self) -> Invariance {
        Invariance::any_rows(self.capacity)
    }

    /// Bytes of one F32 gate or up matrix of `rows` rows.
    pub fn input_bytes(&self, rows: u32) -> u64 {
        u64::from(rows) * u64::from(self.query.width) * 4
    }

    /// Bytes of the E4M3 codes of `rows` rows.
    pub fn code_bytes(&self, rows: u32) -> u64 {
        u64::from(rows) * u64::from(self.query.width)
    }

    /// Bytes of the UE8M0 scales of `rows` rows.
    pub fn scale_bytes(&self, rows: u32) -> u64 {
        u64::from(rows) * u64::from(self.query.width / GROUP)
    }

    pub fn spec(&self) -> EntrySpec {
        let engine = if self.parallel_groups {
            "silu_mul_quantize_parallel"
        } else {
            "silu_mul_quantize"
        };
        let SiluMulQuantizeQuery {
            width,
            clamp:
                SwigluClamp {
                    gate_max_bits,
                    up_limit_bits,
                },
            amax_floor_bits,
        } = self.query;
        EntrySpec::new(
            self.symbol.clone(),
            vec![
                ptr("gate", "const float"),
                ptr("up", "const float"),
                ptr("codes", "uint8_t"),
                ptr("scales", "uint8_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("status", "uint32_t"),
                EntryParam::U64 {
                    name: "status_row_offset".into(),
                },
            ],
            format!(
                "chop::row::{engine}::run<{width},{gate_max_bits:#x}u,{up_limit_bits:#x}u,{amax_floor_bits:#x}u>(gate,up,codes,scales,live_rows,status,status_row_offset);"
            ),
            self.threads(),
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn arguments(
        &self,
        gate: u64,
        up: u64,
        codes: u64,
        scales: u64,
        live: u64,
        status: u64,
        status_row_offset: u64,
    ) -> Result<ArgumentList, Refusal> {
        pack_arguments(
            &self.spec().params,
            [gate, up, codes, scales, live, status]
                .map(Argument::DeviceBuffer)
                .into_iter()
                .chain([Argument::U64(status_row_offset)])
                .collect(),
        )
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("quantized SwiGLU cubin requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(&entry.symbol) {
            return Err(Refusal("duplicate quantized SwiGLU symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!(
            "#include \"{}\"",
            if entries.iter().any(|e| e.parallel_groups) {
                "row/ceil_parallel.cuh"
            } else {
                HEADER
            }
        )],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    })
}

/// The lab's entries: the clamp at 10 and floor 1e-4 at widths of one
/// group, a tensor-parallel shard of 576 columns and an unsplit 2304.
pub fn lab_entries() -> Vec<Entry> {
    [32, 576, 2304]
        .into_iter()
        .map(|width| {
            Entry::new(
                SiluMulQuantizeQuery {
                    width,
                    clamp: SwigluClamp::symmetric(10.0),
                    amax_floor_bits: 1e-4_f32.to_bits(),
                },
                LAB_CAPACITY,
                format!("chop_silu_mul_quantize_c10_w{width}_v2"),
            )
            .unwrap()
        })
        .collect()
}

/// The lab's cubin of [`lab_entries`].
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    vec![cubin_spec(LAB_CUBIN, &lab_entries()).unwrap()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_entries_validate_plan_and_pack() {
        for entry in lab_entries() {
            entry.validate().unwrap();
            let groups = entry.query.width / GROUP;
            for rows in [0, 1, LAB_CAPACITY] {
                let plan = entry.plan(rows).unwrap();
                assert_eq!(plan.grid, [rows.max(1), 1, 1]);
                assert_eq!(plan.block, [32, 1, 1]);
                assert!(plan.pdl);
            }
            assert!(entry.plan(LAB_CAPACITY + 1).is_err());
            entry.arguments(1, 2, 3, 4, 5, 6, 3).unwrap();
            assert!(
                entry.spec().body.contains("0x41200000u"),
                "{}",
                entry.spec().body
            );
            assert_eq!(entry.code_bytes(2), 2 * u64::from(entry.query.width));
            assert_eq!(entry.scale_bytes(2), 2 * u64::from(groups));
            assert_eq!(entry.invariance(), Invariance::any_rows(LAB_CAPACITY));
        }
        assert!(
            Entry::new(lab_entries()[0].query, 0, "chop_x_v1").is_err()
                && Entry::new(lab_entries()[0].query, 1, "1x").is_err()
        );
        let mut bad = lab_entries()[0].query;
        bad.width = 48;
        assert!(Entry::new(bad, 1, "chop_x_v1").is_err());
        assert_eq!(lab_cubin_specs()[0].entries.len(), 3);
    }

    #[test]
    fn entries_refuse_clamps_whose_product_can_overflow_bf16() {
        let entry = |clamp| {
            Entry::new(
                SiluMulQuantizeQuery {
                    clamp,
                    ..lab_entries()[0].query
                },
                1,
                "chop_x_v1",
            )
        };
        let bf16_max = f32::from_bits(0x7f7f_0000);
        // The square of the largest symmetric clamp is the largest BF16.
        let root = (f64::from(bf16_max)).sqrt() as f32;
        let mut fits = root;
        while !clamp_fits_bf16(SwigluClamp::symmetric(fits)) {
            fits = fits.next_down();
        }
        while clamp_fits_bf16(SwigluClamp::symmetric(fits.next_up())) {
            fits = fits.next_up();
        }
        assert!(clamp_fits_bf16(SwigluClamp::symmetric(fits)));
        assert!(!clamp_fits_bf16(SwigluClamp::symmetric(fits.next_up())));
        entry(SwigluClamp::symmetric(fits)).unwrap();
        assert!(entry(SwigluClamp::symmetric(fits.next_up())).is_err());
        // The review's overflow: clamp 2e19, whose square passes 3.4e38.
        assert!(entry(SwigluClamp::symmetric(2e19)).is_err());
        // A gate clamp under 1 still bounds the SiLU by 1, not by itself.
        let small_gate = SwigluClamp {
            gate_max_bits: 0.5_f32.to_bits(),
            up_limit_bits: bf16_max.to_bits(),
        };
        entry(small_gate).unwrap();
        let wide_up = SwigluClamp {
            gate_max_bits: 0.5_f32.to_bits(),
            up_limit_bits: f32::MAX.to_bits(),
        };
        assert!(entry(wide_up).is_err());
    }
}
