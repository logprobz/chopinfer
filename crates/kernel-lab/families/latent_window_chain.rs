//! Attention consumer of the real staged prefill chain.
use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput,LatentAttentionQuery,LatentKvFormat};
use kernels_sm120::{Profile,ProfileNames};
use kernels_sm120::latent::sources::{self,Entry};
pub const FAMILY:&str="latent_window_chain";
pub const PROFILE:Profile=Profile::synthetic(ProfileNames{name:"latent-window-chain",tag:"latent_window_chain"});
pub fn query()->LatentAttentionQuery{LatentAttentionQuery::Attention{heads:16,latent:512,max_keys:128,softmax_scale_bits:(512_f32).powf(-0.5).to_bits(),page_size:16,kv:LatentKvFormat::Fp8E4M3Ue8M0Block32,phase:AttentionPhase::Prefill,output:LatentAttentionOutput::Normalized,sink:true,tail_rope_dims:64,second_fp4_keys:false,capacity:32}}
pub fn entries()->Vec<Entry>{sources::entries(PROFILE,query()).unwrap().into_iter().filter(|e|matches!(e.query,LatentAttentionQuery::Attention{phase:AttentionPhase::Prefill,..})).collect()}
pub fn cubins()->Vec<CubinSpec>{vec![sources::cubin_spec(PROFILE,&entries()).unwrap()]}
