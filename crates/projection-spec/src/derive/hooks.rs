//! Provider behavior hooks. Registration admits typed parameters; these hooks
//! perform only the explicitly named provider's pure preparation behavior.

use super::{
    DerivedFusion, DerivedOperation, DerivedStage, FusionContext, ImportedProjection,
    ProjectionContext, SourceTransformResult,
};
use crate::contracts::Stage;
use crate::{Error, Part, Variant};

/// Registered source transforms, ordered stages and legacy implementations.
/// Missing behavior refuses rather than silently falling back to another recipe.
pub trait VariantDeriver {
    fn transform(
        &self,
        variant: &Variant,
        _part: Option<&Part>,
        _context: &ProjectionContext<'_>,
        _imported: ImportedProjection,
    ) -> Result<SourceTransformResult, Error> {
        Err(Error(format!(
            "variant `{}` has no source transform behavior",
            variant.id
        )))
    }

    fn stage(
        &self,
        stage: &Stage,
        _context: &ProjectionContext<'_>,
    ) -> Result<DerivedStage, Error> {
        Err(Error(format!(
            "variant `{}` has no stage behavior",
            stage.contract.id
        )))
    }

    fn implementation(
        &self,
        variant: &Variant,
        _context: &ProjectionContext<'_>,
        _imported: &ImportedProjection,
    ) -> Result<DerivedOperation, Error> {
        Err(Error(format!(
            "variant `{}` has no implementation behavior",
            variant.id
        )))
    }

    fn fusion(
        &self,
        variant: &Variant,
        _context: &FusionContext<'_>,
        _members: &[DerivedOperation],
    ) -> Result<DerivedFusion, Error> {
        Err(Error(format!(
            "variant `{}` has no fusion behavior",
            variant.id
        )))
    }
}

/// Default provider for declarations with no registered behavior.
#[derive(Debug, Default)]
pub struct NoVariants;
impl VariantDeriver for NoVariants {}
