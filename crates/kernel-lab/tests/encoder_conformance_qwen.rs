//! Generated conformance using this model's serving cubins.
#[macro_use]
#[path = "encoder_conformance/suite.rs"]
mod suite;
const MODEL: suite::Model = suite::Model::Qwen;
use bundle_qwen38::embedded_cubin;
qwen_cases!(conformance);

#[path = "encoder_conformance/chain.rs"]
mod block_chain;
