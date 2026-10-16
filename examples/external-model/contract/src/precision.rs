//! Small independent architecture: one projection, one fused expert pair and
//! one integer-index table. Encoding and execution precision are SDK contracts.
//! The projections themselves are data: `contract/projections.json` names
//! their checkpoint tensors, shapes, kinds, fused operands and uses.
use kernel_core::contract::WeightFormat;
use kernel_core::precision::execution::PrecisionDocument;
use kernel_core::precision::layout::PackingPlan;
use kernel_core::precision::{Calibration, DecodeOrder, OperationUse};
use kernel_core::weight_transform::PackedMatrixSource;
use projection_spec::registry::Registry;
use projection_spec::{Declaration, Geometry, Projection, ProjectionKind};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub const HIDDEN: u32 = 256;
pub const INTERMEDIATE: u32 = 256;
pub const EXPERTS: u32 = 4;
pub const TOP_K: u32 = 2;
pub const VOCAB: u64 = 37;
pub const TABLE_WIDTH: u32 = 160;
pub const ARCHITECTURE: &str = "example.precision.v1";
pub const PRECISION_ROLE: &str = "chop.bundle-metadata.example.precision.v1";
pub const WEIGHTS_ROLE: &str = "example.precision.weights.v1";
pub const MAX_IMAGE_BYTES: u64 = 2 * 1024 * 1024;
pub const WEIGHT_CONTRACT: &[u8] =
    b"example.precision.weights.v1:h256:i256:e4:top2:table37x160:aligned256:packed-nvfp4-v1";
pub const PACKAGE_CONTRACT: &[u8] = b"example.stateful.precision.v2:seed:frontend:tokenizer:precision:shared-policy:planned-weights:weight-plan-v1:abi1:tp1";
pub const POLICY_ROLE: &str = "chop.bundle-metadata.example.precision-policy.v1";
/// The [`Exercise`] document version. Version 2 carries the packing each
/// operation's weight plan needs, so a direct load reads no importer.
pub const EXERCISE_VERSION: u32 = 2;
/// Alignment of the weight image and of every span in it.
pub const IMAGE_ALIGNMENT: u64 = 256;

/// The projection declaration. Its digest is pinned by a test.
pub const DECLARATION: &str = include_str!("../projections.json");

/// The values of the geometry keys the declaration names.
pub fn geometry() -> Geometry {
    Geometry::from([
        ("hidden".into(), u64::from(HIDDEN)),
        ("intermediate".into(), u64::from(INTERMEDIATE)),
        ("experts".into(), u64::from(EXPERTS)),
        ("top_k".into(), u64::from(TOP_K)),
        ("vocab".into(), VOCAB),
        ("table_width".into(), u64::from(TABLE_WIDTH)),
    ])
}

/// The example registers no variants: every projection uses shared formats
/// and shared kernels.
pub fn registry() -> Registry {
    Registry::default()
}

/// The validated declaration with its use profiles expanded, so every
/// projection lists its own uses. Projections keep declaration order, which
/// is also operation and image order.
pub fn declaration() -> Result<&'static Declaration, String> {
    static DECLARED: OnceLock<Result<Declaration, String>> = OnceLock::new();
    DECLARED
        .get_or_init(|| {
            let geometry = geometry();
            projection_spec::parse(DECLARATION, &geometry, &registry())
                .and_then(|d| d.expanded_profiles(&geometry))
                .map_err(|e| format!("projection declaration: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// A declared dimension at [`geometry`].
pub fn dim(d: &projection_spec::Dim) -> Result<u64, String> {
    d.resolve(&geometry()).map_err(|e| e.to_string())
}

/// A projection's logical source shape: operands stacked along rows, and
/// for an expert bank the experts as the leading axis.
pub fn source_shape(p: &Projection) -> Result<Vec<u64>, String> {
    let rows = dim(&p.shape[0])? * u64::from(p.operand_count());
    let k = dim(&p.shape[1])?;
    Ok(match &p.kind {
        ProjectionKind::ExpertBank { experts, .. } => vec![dim(experts)?, rows, k],
        _ => vec![rows, k],
    })
}

/// Declared uses as operation uses, in declaration order.
pub fn operation_uses(uses: &[projection_spec::Use]) -> Result<Vec<OperationUse>, String> {
    uses.iter()
        .map(|u| {
            Ok(OperationUse {
                phase: u.phase,
                capacity: u32::try_from(dim(&u.capacity)?).map_err(|e| e.to_string())?,
                graph: u.graph,
            })
        })
        .collect()
}

/// The declaration's `served` use profile, which every projection uses. A
/// graph may use a prefill entry; its recipe is still operation-wide.
pub fn uses() -> Vec<OperationUse> {
    declaration()
        .and_then(|d| {
            operation_uses(
                d.use_profiles
                    .get("served")
                    .ok_or("no served use profile")?,
            )
        })
        .expect("the compiled-in projection declaration is valid")
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub offset: u64,
    pub bytes: u64,
}

impl Span {
    pub fn range(&self, image_bytes: u64) -> Result<std::ops::Range<usize>, String> {
        let end = self
            .offset
            .checked_add(self.bytes)
            .ok_or("image span overflow")?;
        if end > image_bytes || image_bytes > MAX_IMAGE_BYTES || self.offset % IMAGE_ALIGNMENT != 0
        {
            return Err("image span exceeds its declared bound or alignment".into());
        }
        Ok(self.offset as usize..end as usize)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightSpans {
    pub operation: String,
    pub codes: Span,
    pub local_scales: Span,
    pub global_scales: Span,
    /// Static checkpoint calibration stays present even when BF16 execution
    /// does not consume it. KV and recurrent state are separate declarations.
    pub input_scale: Span,
    /// How the weight plan produces these spans from the checkpoint.
    pub packing: Packing,
}

/// What one operation's weight plan records need: the codes, local-scale
/// and global-scale spans are `PACKED_OPERAND` records of `plan`, and the
/// input-scale span copies `input_scale_tensor` unchanged.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Packing {
    /// The prepared physical contract the selected implementation reads.
    pub plan: PackingPlan,
    /// The effective decode order. The packer checks BF16 operand ranges
    /// when scales apply before a BF16 MMA.
    pub decode_order: DecodeOrder,
    /// Checkpoint tensors the packed records read, in record source order:
    /// codes, then the local and global scales when the weight has them.
    pub sources: Vec<String>,
    /// One matrix per `(expert, projection)` of `plan`, naming `sources` by
    /// index, with its source geometry, scale-grid origins, bank index and
    /// the rows and K range it keeps.
    pub matrices: Vec<PackedMatrixSource>,
    /// The static input calibration tensor, when the checkpoint has one.
    pub input_scale_tensor: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exercise {
    pub version: u32,
    /// The checkpoint the package was prepared from: the direct load's hex
    /// SHA-256 of its safetensors file names, lengths and headers. The weight
    /// plan refuses any other checkpoint, since its tensor names and
    /// geometry come from these headers.
    pub checkpoint_identity: String,
    pub precision: PrecisionDocument,
    pub weights: Vec<WeightSpans>,
    pub image_bytes: u64,
}

impl Exercise {
    pub fn validate(&self) -> Result<(), String> {
        self.precision.validate().map_err(|e| e.to_string())?;
        let projections = &declaration()?.projections;
        if self.version != EXERCISE_VERSION
            || self.image_bytes > MAX_IMAGE_BYTES
            || self.image_bytes % IMAGE_ALIGNMENT != 0
            || self.checkpoint_identity.len() != 64
            || !self
                .checkpoint_identity
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || self.weights.len() != projections.len()
            || self.precision.operations.len() != projections.len()
        {
            return Err("unsupported precision exercise geometry".into());
        }
        let mut occupied = Vec::new();
        for ((weight, precision), declared) in self
            .weights
            .iter()
            .zip(&self.precision.operations)
            .zip(projections)
        {
            let id = declared.id.as_str();
            if weight.operation != id
                || precision.source.id != id
                || precision.source.uses != operation_uses(&declared.uses)?
            {
                return Err("precision exercise operation order or phase coverage differs".into());
            }
            let shape = source_shape(declared)?;
            if precision
                .source
                .weights
                .iter()
                .any(|w| w.logical_shape != shape || w.format != WeightFormat::Nvfp4)
            {
                return Err(format!(
                    "operation {id}: source geometry or encoding differs from the example"
                ));
            }
            let codes = shape.iter().product::<u64>() / 2;
            let globals =
                if precision.effective.weights[0].layout.id == "moe-packed-ExpertProjection" {
                    shape[0] * u64::from(declared.operand_count()) * 4
                } else {
                    4
                };
            let calibration = precision
                .source
                .activations
                .first()
                .and_then(|a| a.quantizer.as_ref())
                .and_then(|q| {
                    if let Calibration::Static(s) = &q.calibration {
                        Some(s.tensor.byte_length)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            if weight.codes.bytes != codes
                || weight.local_scales.bytes != codes / 8
                || weight.global_scales.bytes != globals
                || weight.input_scale.bytes != calibration
            {
                return Err(format!(
                    "operation {id}: packed view lengths disagree with its geometry and scales"
                ));
            }
            weight.packing.validate(declared, weight, &shape)?;
            for span in [
                &weight.codes,
                &weight.local_scales,
                &weight.global_scales,
                &weight.input_scale,
            ] {
                let range = span.range(self.image_bytes)?;
                if !range.is_empty() {
                    if occupied.iter().any(|r: &std::ops::Range<usize>| {
                        r.start < range.end && range.start < r.end
                    }) {
                        return Err("precision exercise weight spans overlap".into());
                    }
                    occupied.push(range);
                }
            }
        }
        Ok(())
    }
}

impl Packing {
    /// Checks that the packing produces exactly the operation's spans, from
    /// a complete bank of NVFP4 matrices that name only listed sources.
    fn validate(
        &self,
        declared: &Projection,
        spans: &WeightSpans,
        shape: &[u64],
    ) -> Result<(), String> {
        let id = &declared.id;
        let refuse = |reason: &str| Err(format!("operation {id}: packing {reason}"));
        let plan = &self.plan;
        plan.validate()
            .map_err(|e| format!("operation {id}: {e}"))?;
        let (experts, projections) = match &declared.kind {
            ProjectionKind::ExpertBank { experts, .. } => (dim(experts)?, declared.operand_count()),
            _ => (1, 1),
        };
        if plan.format != WeightFormat::Nvfp4
            || u64::from(plan.experts) != experts
            || plan.projections != projections
        {
            return refuse("differs from the example's NVFP4 bank geometry");
        }
        let sections = plan
            .section_bytes()
            .map_err(|e| format!("operation {id}: {e}"))?;
        if sections.codes != spans.codes.bytes
            || sections.local_scales != spans.local_scales.bytes
            || sections.global_scales != spans.global_scales.bytes
        {
            return refuse("sections differ from the prepared spans");
        }
        let mut names = std::collections::BTreeSet::new();
        if self.sources.is_empty()
            || self.sources.len() > 3
            || self
                .sources
                .iter()
                .any(|name| name.is_empty() || !names.insert(name))
        {
            return refuse("sources are missing, repeated or unnamed");
        }
        // Each matrix is one `[rows, k]` of the source, whose bank (when it
        // has one) is the leading dimension.
        let (bank, rows, k) = match shape {
            [bank, rows, k] => (*bank, *rows, *k),
            [rows, k] => (1, *rows, *k),
            _ => return refuse("source is neither a matrix nor a bank"),
        };
        let mut seen = std::collections::BTreeSet::new();
        for matrix in &self.matrices {
            let named = [Some(matrix.codes), matrix.local, matrix.global]
                .into_iter()
                .flatten()
                .all(|index| (index as usize) < self.sources.len());
            let kept: u64 = matrix.rows_kept.iter().map(|r| r.end - r.start).sum();
            if !named
                || matrix.format != plan.format
                || matrix.expert >= plan.experts
                || matrix.projection >= plan.projections
                || !seen.insert((matrix.expert, matrix.projection))
                || (matrix.rows, matrix.k, matrix.source_experts) != (rows, k, bank)
                || matrix.source_expert >= matrix.source_experts
                || matrix
                    .rows_kept
                    .iter()
                    .any(|r| r.start >= r.end || r.end > matrix.rows)
                || kept > u64::from(plan.rows)
                || matrix.k_kept.start >= matrix.k_kept.end
                || matrix.k_kept.end > matrix.k
                || matrix.k_kept.end - matrix.k_kept.start > u64::from(plan.k)
            {
                return refuse("matrix names a missing source or an invalid slice");
            }
        }
        if seen.len() != (plan.experts * plan.projections) as usize {
            return refuse("does not supply every bank matrix");
        }
        if self
            .input_scale_tensor
            .as_ref()
            .is_some_and(String::is_empty)
            || self.input_scale_tensor.is_some() != (spans.input_scale.bytes != 0)
        {
            return refuse("input calibration disagrees with its span");
        }
        Ok(())
    }
}
