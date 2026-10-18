//! Numerical device controls, isolated from the production glue cubin.
use kernel_build::CubinSpec;
use kernels_sm120::latent::glue::{self, Mutation};
pub const FAMILY: &str = "latent_glue_mutants";
pub fn cubins() -> Vec<CubinSpec> {
    let mut spec = glue::cubin_spec("latent_glue_mutants_sm120", &[]);
    spec.headers = vec!["#include \"latent/glue_mutants.cuh\"".into()];
    spec.entries = Mutation::ALL.into_iter().map(Mutation::spec).collect();
    vec![spec]
}
