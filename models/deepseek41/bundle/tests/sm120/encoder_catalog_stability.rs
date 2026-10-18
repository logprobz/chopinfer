//! deepseek41's encoder cubins, generated from its committed vision
//! declaration, build to exactly the records its `kernels.lock.d` encoder
//! family holds, and to the records it adopted for that declaration
//! (`encoder-adoptions.json`). A change to a shared variant, engine or
//! header that would alter them fails here unless this declaration changes
//! and a new adoption is appended; refreshing the lock alone, or with a
//! rewritten adoption, does not pass. CPU only.

use std::path::Path;

use kernels_sm120::encoder::stability;
use model_deepseek41_contract::encoder::{FAMILY, declarations};

#[path = "../../src/sm120/encoder.rs"]
#[allow(dead_code, reason = "the check reads the catalog alone")]
mod encoder;

#[test]
fn encoder_catalog_stability() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let specs = kernels_sm120::encoder::cubin_specs(encoder::kernels().unwrap()).unwrap();
    let digests = declarations()
        .unwrap()
        .iter()
        .map(|declaration| declaration.digest())
        .collect::<Vec<_>>();
    let adoptions = root.join("encoder-adoptions.json");
    // The approved ledger on main; without git only the chain is checked here,
    // and tests/encoder-adoptions.py checks the base in CI.
    let base = stability::approved_base(&adoptions).unwrap_or_else(|error| {
        eprintln!("encoder adoptions checked without an approved base: {error}");
        None
    });
    stability::check_family(
        &specs,
        &digests,
        &root.join("kernels.lock.d"),
        FAMILY,
        &adoptions,
        base.as_deref(),
    )
    .unwrap();
}
