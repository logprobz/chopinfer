//! The host side of a bundle's registered variants (`encoder-spec`'s
//! model-namespaced components): their rotary tables, and the launches and
//! record inputs of their stages, in the shared runtime's terms. The bundle
//! implements `Registered` beside the variants; shared code never names
//! them.

use encoder_spec::components::rotary::RotaryComponent;
use encoder_spec::components::{Boundary, Component};
use encoder_spec::geometry::Positions;

/// What one argument of a registered stage's launch binds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Bind {
    /// The rows the stage reads.
    Input,
    /// The rows the stage writes: the run's rows of them when a later stage
    /// joins the runs.
    Output,
    /// A region of the stage (a scratch region it declares).
    Region(&'static str),
    /// A view of the encoder image.
    View(String),
    /// The stage's live row count.
    Live,
}

/// One launch of a registered stage: the slot part its kernel fills (its
/// role in the catalog, else `registered`) and its arguments by the
/// kernel's parameter names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Launch {
    pub part: &'static str,
    pub arguments: Vec<(&'static str, Bind)>,
}

/// What a registered stage runs: its launches in order, and the bytes it
/// stages into its own regions once per record, before the first run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Stage {
    pub launches: Vec<Launch>,
    pub record: Vec<(&'static str, Vec<u8>)>,
}

/// The host side of a bundle's registered variants.
pub trait Registered: Sync {
    /// The cos and sin rows, F32 `[positions, rotary_dims / 2]`, of
    /// registered rotary `rotary` at `positions`.
    fn rotary(
        &self,
        rotary: &dyn RotaryComponent,
        positions: &Positions,
    ) -> Result<(Vec<f32>, Vec<f32>), String>;

    /// The launches of registered stage `component` reading `input`.
    fn stage(&self, component: &dyn Component, input: &Boundary) -> Result<Stage, String>;
}

/// No registered variants: declarations of shared variants only.
pub struct Shared;

impl Registered for Shared {
    fn rotary(
        &self,
        rotary: &dyn RotaryComponent,
        _: &Positions,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        Err(format!("`{}` is not a shared variant", rotary.id()))
    }

    fn stage(&self, component: &dyn Component, _: &Boundary) -> Result<Stage, String> {
        Err(format!("`{}` is not a shared variant", component.id()))
    }
}
