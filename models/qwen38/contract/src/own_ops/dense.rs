//! Qwen's dense projections: one launch's BF16 GEMMs of an activation
//! against the weights of a projection (a layer's input projections side by
//! side, an output, router, shared expert, hyper-connection, PLE, head or
//! MTP projection), with the hyper-connection bottleneck as an optional
//! epilogue.
//!
//! The math is the member GEMMs; which projection a group computes is part
//! of the contract because a backend keeps one arithmetic (split of K,
//! accumulation order) per projection and step use, so every step of a
//! projection gives a row the same bits.

use kernel_core::contract::{DType, Refusal, SigmoidMeanHyperConnectionQuery, SwigluColumns};
use kernel_core::hyper_connection_oracle::bottleneck;
use kernel_core::oracle::{GemmEpilogue, GemmShape, OracleOutput, gemm};

use crate::contract::Shape;

/// The dense projections of Qwen's layers, as one launch computes them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Projection {
    /// Linear attention's `in_proj_qkv`, `in_proj_z`, `in_proj_a` and
    /// `in_proj_b` of a decode bucket, each to its own output.
    LinearIn,
    LinearQkv,
    LinearZ,
    /// `in_proj_a` and `in_proj_b` side by side in one output row.
    LinearAb,
    LinearOutput,
    /// Attention's `q_proj`, `k_proj`, `v_proj` and the indexer's
    /// `index_qk_proj` of a decode bucket, each to its own output.
    AttentionIn,
    /// `q_proj`, `k_proj` and `v_proj` stored back to back as one weight.
    AttentionQkv,
    IndexQk,
    AttentionOutput,
    Router,
    /// The shared expert's gate and up halves, with their SwiGLU.
    SharedGateUp,
    SharedDown,
    /// A hyper-connection's `input_mix_weight_down` and
    /// `block_inject_weight` of a decode bucket, each to its own output.
    HcIn,
    HcDown,
    HcInject,
    HcUp,
    PleKey,
    PleValue,
    /// The vocabulary head of the rank's shard, in BF16 (the drafter's
    /// argmax) or F32 (logits).
    Head(DType),
    /// The MTP drafter's `fc_hidden` and `fc_embedding`.
    MtpFc,
}

/// One launch's GEMMs: member `i` multiplies the activation rows
/// (`[rows][k]` BF16) by its weight (`[members[i]][k]` BF16) transposed,
/// accumulating in F32, and rounds once to `output`. A SwiGLU member's rows
/// are its gate then up halves, and it writes `silu(gate) * up`, half as
/// many columns. Members write their own output at that output's row
/// stride (`strides`), or, `packed`, one output with member `i` at the
/// columns after members `0..i`. With `bottleneck` (the hyper-connection's
/// streams), the launch also writes `silu(d / streams)` rounded to BF16 of
/// member 0's stored BF16 values `d`. All math.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseGroup {
    pub projection: Projection,
    pub k: u32,
    pub members: Vec<u32>,
    pub swiglu: Option<SwigluColumns>,
    pub output: DType,
    pub packed: bool,
    /// Each output's row stride in elements: one per member, or one when
    /// `packed`.
    pub strides: Vec<u32>,
    pub bottleneck: Option<u32>,
}

impl DenseGroup {
    /// `projection` of `shape`'s rank, without the bottleneck.
    pub fn of(shape: Shape, projection: Projection) -> DenseGroup {
        use Projection::*;
        let (h, wide) = (shape.hidden(), shape.wide());
        let heads = shape.value_heads;
        let (q, kv) = (
            shape.q_heads * 2 * shape.attn_head(),
            shape.kv_heads() * shape.attn_head(),
        );
        let ab = shape.linear_ab_stride;
        let (k, members, strides): (u32, Vec<u32>, Vec<u32>) = match projection {
            LinearIn => (
                h,
                vec![shape.linear_qkv, shape.linear_value, heads, heads],
                vec![shape.linear_qkv, shape.linear_value, ab, ab],
            ),
            LinearQkv => (h, vec![shape.linear_qkv], vec![shape.linear_qkv]),
            LinearZ => (h, vec![shape.linear_value], vec![shape.linear_value]),
            LinearAb => (h, vec![heads, heads], vec![ab]),
            LinearOutput => (shape.linear_value, vec![h], vec![h]),
            AttentionIn => (
                h,
                vec![q, kv, kv, shape.index_row()],
                vec![
                    shape.attn_row,
                    shape.attn_row,
                    shape.attn_row,
                    shape.index_row(),
                ],
            ),
            AttentionQkv => (h, vec![shape.attn_row], vec![shape.attn_row]),
            IndexQk => (h, vec![shape.index_row()], vec![shape.index_row()]),
            AttentionOutput => (shape.attn_value, vec![h], vec![h]),
            Router => (h, vec![shape.experts()], vec![shape.experts()]),
            SharedGateUp => (
                h,
                vec![2 * shape.shared_intermediate],
                vec![shape.shared_intermediate],
            ),
            SharedDown => (shape.shared_intermediate, vec![h], vec![h]),
            HcIn => (
                wide,
                vec![shape.hc_rank(), shape.streams()],
                vec![shape.hc_rank(), shape.streams()],
            ),
            HcDown => (wide, vec![shape.hc_rank()], vec![shape.hc_rank()]),
            HcInject => (wide, vec![shape.streams()], vec![shape.streams()]),
            HcUp => (shape.hc_rank(), vec![wide], vec![wide]),
            PleKey => (shape.ple_embed_dim(), vec![wide], vec![wide]),
            PleValue => (shape.ple_embed_dim(), vec![h], vec![h]),
            Head(_) => (h, vec![shape.vocab_per_rank], vec![shape.vocab_per_rank]),
            MtpFc => (h, vec![h], vec![h]),
        };
        DenseGroup {
            projection,
            k,
            members,
            swiglu: (projection == SharedGateUp).then_some(SwigluColumns::ConcatenatedHalves),
            output: match projection {
                Router => DType::F32,
                Head(output) => output,
                _ => DType::Bf16,
            },
            packed: projection == LinearAb,
            strides,
            bottleneck: None,
        }
    }

    /// This group, also writing the bottleneck of `streams` streams.
    pub fn with_bottleneck(self, streams: u32) -> DenseGroup {
        DenseGroup {
            bottleneck: Some(streams),
            ..self
        }
    }

    /// The columns member `index` writes.
    pub fn columns(&self, index: usize) -> u32 {
        self.members[index] / (1 + u32::from(self.swiglu.is_some()))
    }

    /// The column of its output where member `index` starts.
    pub fn first_column(&self, index: usize) -> u32 {
        if self.packed {
            (0..index).map(|member| self.columns(member)).sum()
        } else {
            0
        }
    }
}

impl kernel_core::contract::Validate for DenseGroup {
    fn validate(&self) -> Result<(), Refusal> {
        let outputs = if self.packed { 1 } else { self.members.len() };
        let halves = 1 + u32::from(self.swiglu.is_some());
        if self.k == 0
            || self.members.is_empty()
            || self.members.iter().any(|&n| n == 0 || n % halves != 0)
            || self.strides.len() != outputs
            || !matches!(self.output, DType::Bf16 | DType::F32)
        {
            return Err(Refusal(
                "a dense group needs K, members and a BF16 or F32 output per stride",
            ));
        }
        let last = self.members.len() - 1;
        let widths = if self.packed {
            vec![self.first_column(last) + self.columns(last)]
        } else {
            (0..self.members.len()).map(|m| self.columns(m)).collect()
        };
        if widths.iter().zip(&self.strides).any(|(w, s)| s < w) {
            return Err(Refusal(
                "a dense group's row stride is shorter than its row",
            ));
        }
        if self.bottleneck == Some(0)
            || self.bottleneck.is_some() && (self.output != DType::Bf16 || self.swiglu.is_some())
        {
            return Err(Refusal(
                "a dense group's bottleneck reads member 0's BF16 values over streams",
            ));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A dense group's buffers: each member's weight in the layout the
    /// backend names for `weights` (`OpInfo::weight_layouts`), the
    /// activation rows, the outputs (rows at the contract's strides; a
    /// member's view may start inside another's row), the bottleneck rows
    /// (`[rows][members[0]]` BF16) exactly when the contract writes them,
    /// and the split-K `partial` sums and zeroed `semaphores` of the
    /// backend's schedule, at least the bytes `OpInfo::bytes` names. Views
    /// start at 16-byte alignment, except outputs, which start at their
    /// element's.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct DenseGroupBuffers {
        pub weights: values,
        pub activation: value,
        pub outputs: values,
        pub bottleneck: option,
        pub partial: value,
        pub semaphores: value,
    }
}

/// The F64 reference of each member over `rows` rows of `activation`
/// (`[rows][k]` BF16 bits) and `weights` (`[members[i]][k]`, unpacked BF16
/// bits): values `[rows][columns(i)]` with their accumulation magnitudes,
/// for `kernel_core::oracle::check_numeric` at the contract's output.
pub fn dense_group(
    op: &DenseGroup,
    weights: &[&[u16]],
    activation: &[u16],
    rows: u32,
) -> Result<Vec<OracleOutput>, Refusal> {
    kernel_core::contract::Validate::validate(op)?;
    let (rows, k) = (rows as usize, op.k as usize);
    if weights.len() != op.members.len() {
        return Err(Refusal(
            "a dense group reference needs one weight per member",
        ));
    }
    let widen = |bits: &[u16], len: usize| -> Result<Vec<f64>, Refusal> {
        Ok(bits
            .get(..len)
            .ok_or(Refusal("a dense group buffer is too small"))?
            .iter()
            .map(|&b| f64::from(kernel_core::codec::bf16_to_f32(b)))
            .collect())
    };
    let a = widen(activation, rows * k)?;
    op.members
        .iter()
        .zip(weights)
        .map(|(&n, weight)| {
            let n = n as usize;
            gemm(
                &a,
                &widen(weight, n * k)?,
                GemmShape {
                    m: rows,
                    n,
                    k,
                    experts: 1,
                },
                &[],
                GemmEpilogue {
                    residual: None,
                    swiglu: op.swiglu,
                },
            )
        })
        .collect()
}

/// The bottleneck reference over `rows` rows of member 0's stored BF16
/// values `down`, as the hyper-connection's bottleneck row op computes it.
pub fn dense_group_bottleneck(
    op: &DenseGroup,
    down: &[u16],
    rows: u32,
) -> Result<OracleOutput, Refusal> {
    kernel_core::contract::Validate::validate(op)?;
    let streams = op
        .bottleneck
        .ok_or(Refusal("the dense group writes no bottleneck"))?;
    let rank = op.members[0];
    let down = down
        .get(..rows as usize * rank as usize)
        .ok_or(Refusal("a dense group buffer is too small"))?
        .iter()
        .map(|&b| f64::from(kernel_core::codec::bf16_to_f32(b)))
        .collect::<Vec<_>>();
    bottleneck(
        SigmoidMeanHyperConnectionQuery::Bottleneck {
            streams,
            rank,
            capacity: rows,
        },
        &down,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::Validate;
    use kernel_core::oracle::check_numeric;

    #[test]
    fn every_projection_of_each_topology_is_valid() {
        use Projection::*;
        for shape in [Shape::TP4, Shape::TP2] {
            for projection in [
                LinearIn,
                LinearQkv,
                LinearZ,
                LinearAb,
                LinearOutput,
                AttentionIn,
                AttentionQkv,
                IndexQk,
                AttentionOutput,
                Router,
                SharedGateUp,
                SharedDown,
                HcIn,
                HcDown,
                HcInject,
                HcUp,
                PleKey,
                PleValue,
                Head(DType::Bf16),
                Head(DType::F32),
                MtpFc,
            ] {
                DenseGroup::of(shape, projection).validate().unwrap();
            }
            let hc = DenseGroup::of(shape, HcIn).with_bottleneck(shape.streams());
            hc.validate().unwrap();
            assert!(
                DenseGroup::of(shape, Router)
                    .with_bottleneck(4)
                    .validate()
                    .is_err()
            );
        }
        let ab = DenseGroup::of(Shape::TP4, LinearAb);
        assert_eq!(ab.first_column(1), ab.members[0]);
        let short = DenseGroup {
            strides: vec![1],
            ..ab
        };
        assert!(short.validate().is_err());
    }

    /// The kernels' F32 accumulation rounded once to BF16 is within the
    /// numeric bound of each member's reference; a wrong member is not.
    #[test]
    fn rounded_f32_members_are_within_the_reference_bounds() {
        let bf = kernel_core::codec::f32_to_bf16;
        let f = kernel_core::codec::bf16_to_f32;
        let op = DenseGroup {
            projection: Projection::LinearAb,
            k: 3,
            members: vec![2, 4],
            swiglu: Some(SwigluColumns::ConcatenatedHalves),
            output: DType::Bf16,
            packed: true,
            strides: vec![4],
            bottleneck: None,
        };
        let activation = (0..2 * 3)
            .map(|v| bf(v as f32 / 3.0 - 0.7))
            .collect::<Vec<_>>();
        let weights = [
            (0..2 * 3)
                .map(|v| bf(((v * 5) % 7) as f32 / 4.0 - 0.5))
                .collect::<Vec<_>>(),
            (0..4 * 3)
                .map(|v| bf(((v * 3) % 11) as f32 / 8.0 - 0.6))
                .collect::<Vec<_>>(),
        ];
        let reference = dense_group(
            &op,
            &[weights[0].as_slice(), weights[1].as_slice()],
            &activation,
            2,
        )
        .unwrap();
        let silu = |g: f32| g / (1.0 + (-g).exp());
        for (member, weight) in weights.iter().enumerate() {
            let n = op.members[member] as usize;
            let dot = |row: usize, col: usize| {
                (0..3).fold(0_f32, |sum, k| {
                    f(activation[row * 3 + k]).mul_add(f(weight[col * 3 + k]), sum)
                })
            };
            let emulated = (0..2)
                .flat_map(|row| (0..n / 2).map(move |col| (row, col)).collect::<Vec<_>>())
                .map(|(row, col)| f64::from(f(bf(silu(dot(row, col)) * dot(row, col + n / 2)))))
                .collect::<Vec<_>>();
            check_numeric(&emulated, &reference[member], DType::Bf16).unwrap();
            let mut wrong = emulated.clone();
            wrong[0] += 0.25;
            assert!(check_numeric(&wrong, &reference[member], DType::Bf16).is_err());
        }
        assert!(dense_group(&op, &[weights[0].as_slice()], &activation, 2).is_err());
        assert!(dense_group_bottleneck(&op, &activation, 2).is_err());
        // The bottleneck reads member 0's stored values over the streams.
        let hc = DenseGroup {
            projection: Projection::HcDown,
            k: 3,
            members: vec![3],
            swiglu: None,
            output: DType::Bf16,
            packed: false,
            strides: vec![3],
            bottleneck: Some(2),
        };
        // A group without members is refused, not indexed.
        let empty = DenseGroup {
            members: vec![],
            strides: vec![],
            bottleneck: Some(4),
            ..hc.clone()
        };
        assert!(dense_group_bottleneck(&empty, &activation, 2).is_err());
        let formed = dense_group_bottleneck(&hc, &activation, 2).unwrap();
        let emulated = activation
            .iter()
            .map(|&d| {
                let v = f(d) / 2.0;
                f64::from(f(bf(v / (1.0 + (-v).exp()))))
            })
            .collect::<Vec<_>>();
        check_numeric(&emulated, &formed, DType::Bf16).unwrap();
    }
}
