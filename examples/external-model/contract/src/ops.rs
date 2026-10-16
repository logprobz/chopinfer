//! The example's kernel ops as contracts: a plain RMSNorm, which kernel-core
//! defines, and the example's own sigmoid probe, which no other model runs.
//! Neither is called by the synthetic serving program; an instance prepares
//! both through a backend when it is created, so a build whose kernels cannot
//! serve them creates no instance.
//!
//! The probe is the pattern for a model-owned op: a contract type, a role
//! record of its buffers, a CPU oracle, and a method of [`OwnOps`], which a
//! backend implements or refuses. SM120's implementation lives in the
//! bundle's `sm120` module.

use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::contract::RmsNormKind;
use kernel_core::rmsnorm::{RmsNormBuffers, RmsNormRequirement};

/// The RMSNorm's row width.
pub const WIDTH: usize = 1280;
/// The most rows one RMSNorm launch normalizes.
pub const CAPACITY: u32 = 4;
/// The RMSNorm's epsilon.
pub const EPSILON: f32 = 1e-6;

/// The RMSNorm the example needs: BF16 rows, a plain weight, no residual. It
/// pins no reduction order, so the backend chooses one and names it.
pub fn rmsnorm() -> RmsNormRequirement {
    RmsNormRequirement::bf16_backend_chosen(WIDTH as u32, EPSILON, RmsNormKind::Plain, false)
}

/// The sigmoid probe: writes `1 / (1 + exp(-x))` in F32 to its result, where
/// `x` is the U32 input converted to F32 with nearest-even rounding. Its
/// numerical contract is `example.sigmoid-u32-f32.v1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SigmoidProbe;

kernel_core::scratch_record! {
    /// A sigmoid probe's buffers: one 4-byte aligned F32 `result`, and the
    /// U32 `value` the launch passes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ProbeBuffers {
        pub result: value,
        pub value: scalar(u32),
    }
}

/// The probe's oracle in f64.
pub fn sigmoid_probe(value: u32) -> f64 {
    1.0 / (1.0 + (-f64::from(value as f32)).exp())
}

/// The largest absolute difference from [`sigmoid_probe`] a kernel may give:
/// it evaluates in F32.
pub const PROBE_ABS_BOUND: f64 = 2e-7;

/// A backend of the example's own kernels. Each method prepares one op or
/// refuses it; every method refuses by default, so a backend implements the
/// ops it has kernels for.
pub trait OwnOps: Send + Sync {
    /// The backend's name in refusals and reports, such as `sm120`.
    fn name(&self) -> &str;

    /// The sigmoid probe. Launches read no counts and record one row.
    fn sigmoid_probe(
        &self,
        op: &SigmoidProbe,
        need: Need,
    ) -> Result<Prepared<ProbeBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }
}

/// The example's ops, prepared once.
pub struct Ops {
    pub rmsnorm: Prepared<RmsNormBuffers<View>>,
    pub probe: Prepared<ProbeBuffers<View>>,
}

impl Ops {
    /// Prepares every op on `backend` and `own`, or names the first refused.
    pub fn prepare(backend: &dyn Backend, own: &dyn OwnOps) -> Result<Ops, String> {
        let refused = |op: &str, backend: &str, refusal: Refused| {
            format!("{backend} refuses the example's {op}: {refusal}")
        };
        Ok(Ops {
            rmsnorm: backend
                .rmsnorm(&rmsnorm(), Need::rows(CAPACITY))
                .map_err(|r| refused("RMSNorm", backend.name(), r))?,
            probe: own
                .sigmoid_probe(&SigmoidProbe, Need::rows(1))
                .map_err(|r| refused("sigmoid probe", own.name(), r))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Nothing;

    impl Backend for Nothing {
        fn name(&self) -> &str {
            "nothing"
        }
    }

    impl OwnOps for Nothing {
        fn name(&self) -> &str {
            "nothing"
        }
    }

    #[test]
    fn a_backend_without_the_ops_prepares_nothing() {
        let error = Ops::prepare(&Nothing, &Nothing).err().unwrap();
        assert!(
            error.starts_with("nothing refuses the example's RMSNorm"),
            "{error}"
        );
    }

    #[test]
    fn the_probe_oracle_is_a_sigmoid_of_the_rounded_value() {
        assert_eq!(sigmoid_probe(0), 0.5);
        assert!((0.5..1.0).contains(&sigmoid_probe(8)));
        // 2^24 + 1 rounds to 2^24 in F32.
        assert_eq!(sigmoid_probe((1 << 24) + 1), sigmoid_probe(1 << 24));
    }
}
