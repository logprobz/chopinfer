//! A model's F32 norm-weight forms beside the BF16 forms they stand in for,
//! as its bundle embeds them. A catalog declares them
//! (`ModelCatalog::norm_weights`) so kernel-lab enumerates their oracle,
//! parity and invariance cases without naming a model; each form names the
//! embedded cubin a bundle loads it from.

use super::epilogue::GroupedRmsNorm;
use super::hyper_connection::HcTransport;
use super::hyper_connection_decode::HcDecodeTransport;

/// One entry of a bundle's embedded cubin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Embedded {
    pub cubin: String,
    pub symbol: String,
}

/// A grouped norm row over F32 weights, and the BF16 row of the same groups,
/// reduction and arguments it stands in for.
#[derive(Clone, Debug)]
pub struct NormRowForms {
    pub row: GroupedRmsNorm,
    pub f32: Embedded,
    pub bf16: Embedded,
}

/// The all-reduce epilogue's forms at one rank count, BF16 then F32: the
/// prefill transports (loaded from their `cubin`), and the decode transports
/// whose decode entries lie in `decode_cubins` and wide entries in
/// `wide_cubins`. Both forms of a transport share one comm state.
#[derive(Clone, Debug)]
pub struct NormTransportForms {
    pub ranks: u32,
    pub prefill: [HcTransport; 2],
    pub decode: [HcDecodeTransport; 2],
    pub decode_cubins: [String; 2],
    pub wide_cubins: [String; 2],
}

/// Every F32 norm-weight form a model's bundle embeds.
#[derive(Clone, Debug, Default)]
pub struct NormWeightForms {
    pub rows: Vec<NormRowForms>,
    pub transports: Vec<NormTransportForms>,
}
