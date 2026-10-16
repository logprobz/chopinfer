//! Generated conformance using this model's serving cubins.
#[macro_use]
#[path = "encoder_conformance/suite.rs"]
mod suite;
const MODEL: suite::Model = suite::Model::Glm;
use bundle_glm5next::embedded_cubin;
glm_cases!(conformance);
