//! Prefix gather and its four isolated device controls.
use kernel_build::CubinSpec;
use kernels_sm120::latent::window_stage::{self,Entry,Mutation};
pub const FAMILY:&str="latent_window_stage";
pub const CUBIN:&str="latent_window_stage_sm120";
pub fn entry()->Entry{Entry{capacity:16,symbol:"chop_window_stage_v1".into(),mutation:None}}
pub fn cubins()->Vec<CubinSpec>{
    let entries=std::iter::once(entry()).chain(Mutation::ALL.map(|mutation|Entry{capacity:16,symbol:mutation.symbol(),mutation:Some(mutation)})).collect::<Vec<_>>();
    vec![window_stage::cubin_spec(CUBIN,&entries)]
}
