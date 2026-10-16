//! Generated conformance using this model's serving cubins.
#[macro_use]
#[path = "encoder_conformance/suite.rs"]
mod suite;
#[path = "encoder_conformance/registered.rs"]
mod registered;
const MODEL: suite::Model = suite::Model::Mimo;
use bundle_mimo26::embedded_cubin;
mimo_cases!(conformance);
macro_rules! registered_conformance {
    ($($name:ident: $id:literal),+ $(,)?) => {
        $(#[test] fn $name() {registered::run($id);})+
    };
}
registered_cases!(registered_conformance);
