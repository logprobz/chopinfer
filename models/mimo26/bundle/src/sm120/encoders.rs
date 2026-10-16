//! MiMo's encoders on SM120. The shared encoder runtime
//! (`bundle_runtime::encoder`) prepares every slot of both
//! declarations once at create: a shared stage's slot on the `kernels_sm120`
//! backend over MiMo's generated encoder catalog, from the query its catalog
//! entry was generated from (`kernels_sm120::encoder::kernels`), and a
//! registered stage's `registered` slot on MiMo's own encoder ops
//! (`Sm120Own`), which bind their buffers by the names `MimoVariants` gives
//! the stage's launch. The layout's embedding override runs in the language
//! model's prefill through the runtime's `Rank::inject_part`.

use std::collections::BTreeMap;

use bundle_runtime::encoder::registered::{Bind, Launch as StageLaunch, Stage};
use bundle_runtime::encoder::{Config, Encoder, Named, Registered, SlotOp};
use encoder_spec::Declaration;
use encoder_spec::components::rotary::RotaryComponent;
use encoder_spec::components::{Boundary, Component, StageOp, Variant};
use encoder_spec::geometry::Positions;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{MultimodalQuery, Refusal};
use kernels_sm120::backend::Sm120;
use kernels_sm120::encoder::Slot;
use model_mimo26_contract::encoders::spec;
use model_mimo26_contract::encoders::{Encoders, STAGING_BYTES};
use model_mimo26_contract::own_ops::OwnOps;
use model_mimo26_contract::own_ops::encoder::{EmbeddingSum, EmbeddingSumBuffers, Rvq, RvqBuffers};
use model_mimo26_contract::sm120::encoders as encoder_kernels;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span, HostServicesV1};

use super::own_ops::Sm120Own;
use crate::encoders::arena::{self, CHUNK_ROWS};
use crate::topology::Tp;

/// The host side of MiMo's registered variants (`spec::PROVIDER`): the
/// residual quantizer and the embedding sum. Rotary uses shared variants.
pub struct MimoVariants;

/// Decodes a registered variant from the parameters it was written with.
fn typed<T: serde::de::DeserializeOwned>(variant: &dyn Variant) -> Result<T, String> {
    serde_json::from_value(variant.to_value())
        .map_err(|error| format!("`{}`: {error}", variant.id()))
}

impl Registered for MimoVariants {
    fn rotary(
        &self,
        rotary: &dyn RotaryComponent,
        _: &Positions,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        Err(format!(
            "`{}` has no MiMo rotary table; legacy variants are retired",
            rotary.id()
        ))
    }

    /// The residual quantizer searches the stacked F32 codebooks
    /// (`@codebooks`) from each stage's first row, staged once per record;
    /// the embedding sums add the 20 stacked tables' (`@embeddings`) rows of
    /// each row's codes. The names are the roles of the own ops
    /// (`RvqBuffers`, `EmbeddingSumBuffers`).
    fn stage(&self, component: &dyn Component, _: &Boundary) -> Result<Stage, String> {
        let launch = |part, arguments: Vec<(&'static str, Bind)>| StageLaunch { part, arguments };
        match component.id().as_str() {
            "mimo26:quantize.rvq20-f32.v1" => {
                // The residual search over the F32 rows and the F32
                // codebooks (MiMo's own op, `OwnOps::encoder_rvq`).
                let rvq: spec::Rvq20 = typed(component)?;
                Ok(Stage {
                    launches: vec![launch(
                        "registered",
                        vec![
                            ("input", Bind::Input),
                            ("codebooks", Bind::View("@codebooks".into())),
                            ("codebook_offsets", Bind::Region("codebook-offsets")),
                            ("codes", Bind::Output),
                            ("live_rows", Bind::Live),
                        ],
                    )],
                    record: vec![("codebook-offsets", offsets(&rvq.params().codebooks))],
                })
            }
            "mimo26:embed.sum20-f32.v1" => Ok(Stage {
                launches: vec![launch(
                    "registered",
                    vec![
                        ("tables", Bind::View("@embeddings".into())),
                        ("codes", Bind::Input),
                        ("output", Bind::Output),
                        ("live_rows", Bind::Live),
                    ],
                )],
                record: Vec::new(),
            }),
            id => Err(format!("`{id}` has no MiMo launches")),
        }
    }
}

/// The first codebook row of each RVQ stage plus the end, little-endian
/// i32.
fn offsets(codebooks: &[u32]) -> Vec<u8> {
    let mut offsets = vec![0_i32];
    for &entries in codebooks {
        offsets.push(offsets[offsets.len() - 1] + entries as i32);
    }
    offsets
        .iter()
        .flat_map(|offset| offset.to_le_bytes())
        .collect()
}

/// An own op over the buffers a registered stage's launch names: `roles`
/// builds the op's role record from them.
struct ByName<B> {
    op: Prepared<B>,
    roles: fn(&Named<View>) -> Result<B, Refusal>,
}

impl<B: 'static> PreparedOp<Named<View>> for ByName<B> {
    fn info(&self) -> &OpInfo {
        self.op.info()
    }

    fn launches(
        &self,
        named: &Named<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.op.launches(&(self.roles)(named)?, workspace, counts)
    }
}

/// The views of `names`, each named exactly once in `named`, which names
/// nothing else.
fn take<const N: usize>(named: &Named<View>, names: [&str; N]) -> Result<[View; N], Refusal> {
    if named.len() != N {
        return Err(Refusal("an own encoder op binds exactly its roles"));
    }
    let mut views = [View::EMPTY; N];
    for (view, name) in views.iter_mut().zip(names) {
        let mut found = named.iter().filter(|(known, _)| *known == name);
        *view = match (found.next(), found.next()) {
            (Some(&(_, bound)), None) => bound,
            _ => return Err(Refusal("an own encoder op binds each role once")),
        };
    }
    Ok(views)
}

fn rvq_roles(named: &Named<View>) -> Result<RvqBuffers<View>, Refusal> {
    let [input, codebooks, codebook_offsets, codes, live_rows] = take(
        named,
        [
            "input",
            "codebooks",
            "codebook_offsets",
            "codes",
            "live_rows",
        ],
    )?;
    Ok(RvqBuffers {
        input,
        codebooks,
        codebook_offsets,
        codes,
        live_rows,
    })
}

fn embedding_sum_roles(named: &Named<View>) -> Result<EmbeddingSumBuffers<View>, Refusal> {
    let [tables, codes, output, live_rows] =
        take(named, ["tables", "codes", "output", "live_rows"])?;
    Ok(EmbeddingSumBuffers {
        tables,
        codes,
        output,
        live_rows,
    })
}

/// The slots of a geometry's declarations: each shared slot's query, and
/// MiMo's own ops for the registered stages.
struct Slots<'a> {
    declarations: &'a [Declaration; 2],
    /// Per declaration, its slots' queries, or why the walk refused it.
    queries: Vec<Result<Vec<(Slot, MultimodalQuery)>, String>>,
    own: &'a dyn OwnOps,
}

impl<'a> Slots<'a> {
    fn new(declarations: &'a [Declaration; 2], own: &'a dyn OwnOps) -> Self {
        let queries = declarations
            .iter()
            .map(|declaration| {
                kernels_sm120::encoder::kernels(declaration, &encoder_kernels::Kernels)
                    .map(|kernels| {
                        kernels
                            .into_iter()
                            .map(|(slot, kernel)| (slot, kernel.query))
                            .collect()
                    })
                    .map_err(|error| error.0)
            })
            .collect();
        Self {
            declarations,
            queries,
            own,
        }
    }

    /// What slot (stage, part, block) of declaration `encoder` runs.
    fn op(
        &self,
        encoder: usize,
        stage: usize,
        part: &'static str,
        block: Option<u32>,
    ) -> Result<SlotOp, String> {
        let declaration = self
            .declarations
            .get(encoder)
            .ok_or_else(|| format!("MiMo has no encoder {encoder}"))?;
        let queries = self.queries[encoder].as_ref().map_err(Clone::clone)?;
        let slot = Slot { stage, part, block };
        if let Some((_, query)) = queries.iter().find(|(known, _)| *known == slot) {
            return Ok(SlotOp::Query(*query));
        }
        // MiMo's own ops fill their stages' `registered` slots.
        let registered = match declaration.stages.get(stage).map(|stage| &stage.op) {
            Some(StageOp::Registered(registered)) if part == "registered" => registered,
            _ => return Err(format!("the MiMo catalog has no slot {slot:?}")),
        };
        let component = registered.component.as_ref();
        let need = Need::rows(declaration.input.capacity_rows);
        let refused = |refused: Refused| format!("`{}`: {refused}", component.id());
        match component.id().as_str() {
            "mimo26:quantize.rvq20-f32.v1" => {
                let rvq: spec::Rvq20 = typed(component)?;
                let inputs = declaration.inputs().map_err(|error| error.0)?;
                let op = Rvq::of(rvq.params(), inputs[stage].width()?);
                Ok(SlotOp::Own(Box::new(ByName {
                    op: self.own.encoder_rvq(&op, need).map_err(refused)?,
                    roles: rvq_roles,
                })))
            }
            "mimo26:embed.sum20-f32.v1" => {
                let sum: spec::Sum20 = typed(component)?;
                let op = EmbeddingSum::of(sum.params());
                Ok(SlotOp::Own(Box::new(ByName {
                    op: self.own.encoder_embedding_sum(&op, need).map_err(refused)?,
                    roles: embedding_sum_roles,
                })))
            }
            id => Err(format!("`{id}` has no MiMo own op")),
        }
    }
}

/// The SM120 backend serving `geometry`'s encoder catalog.
fn backend(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
) -> Result<Sm120, BundleAbiError> {
    let catalog = encoder_kernels::kernels_for(geometry)
        .map_err(|_| BundleAbiError::InvalidRange("MiMo encoder declarations are invalid"))?;
    Ok(Sm120::new().with_encoders(catalog.iter().cloned()))
}

/// What MiMo states about its encoders at `geometry` besides their
/// declarations.
fn config(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
) -> Result<Config, BundleAbiError> {
    Ok(Config {
        arena: arena::for_geometry(geometry).map_err(BundleAbiError::InvalidRange)?,
        label: "MiMo",
        capture_schema: "chopinfer.mimo26.encoder-capture.v1",
        prefill_rows: CHUNK_ROWS,
        registered: &MimoVariants,
        staging_bytes: STAGING_BYTES,
    })
}

/// Lowers the declarations of the encoders serving `media` onto one buffer
/// plan over each rank's encoder image `weights`, allocating every rank's
/// regions with `allocate`, every slot prepared for `tp`'s geometry.
pub fn create_for(
    tp: Tp,
    services: &HostServicesV1,
    media: Encoders,
    weights: Vec<BTreeMap<String, Span>>,
    allocate: impl FnMut(usize, u64) -> Result<Span, BundleAbiError>,
) -> Result<Encoder, BundleAbiError> {
    let geometry = tp.geometry;
    let invalid = |_| BundleAbiError::InvalidRange("MiMo encoder declarations are invalid");
    let mut selected = media
        .declarations_for(geometry)
        .map_err(invalid)?
        .into_iter();
    let declarations = [
        if media.vision() {
            selected.next()
        } else {
            None
        },
        if media.holds("audio") {
            selected.next()
        } else {
            None
        },
    ];
    let declarations: Vec<_> = declarations.iter().map(Option::as_ref).collect();
    // Slots come from the whole declarations, which the catalog was
    // generated from; a media set only narrows the vision recipes.
    let own = Sm120Own::new(tp);
    let slots = Slots::new(spec::declarations_for(geometry).map_err(invalid)?, &own);
    Encoder::with_backend(
        services,
        &declarations,
        config(geometry)?,
        &backend(geometry)?,
        &|encoder, stage, part, block| slots.op(encoder, stage, part, block),
        weights,
        allocate,
    )
}

/// `create_for` at the flash geometry.
#[cfg(test)]
pub fn create(
    services: &HostServicesV1,
    media: Encoders,
    weights: Vec<BTreeMap<String, Span>>,
    allocate: impl FnMut(usize, u64) -> Result<Span, BundleAbiError>,
) -> Result<Encoder, BundleAbiError> {
    create_for(Tp::TP2, services, media, weights, allocate)
}

#[cfg(test)]
#[path = "encoders_tests.rs"]
mod tests;
