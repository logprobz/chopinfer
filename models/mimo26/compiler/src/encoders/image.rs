//! The encoder image's geometry. Every rank holds the same image, laid out
//! as the encoder kernels read it (`model_mimo26_contract::encoders::prepared_flash`).

use crate::MimoResult;
use model_mimo26_contract::Tp;

use super::catalog::EncoderCatalog;

/// Every rank's encoder image bytes: the contract's layout of the catalog's
/// tensors, which must be the contract's.
pub fn estimated_rank_bytes(
    catalog: &EncoderCatalog,
    tp: Tp,
    encoders: model_mimo26_contract::encoders::Encoders,
) -> MimoResult<Vec<u64>> {
    let expected = model_mimo26_contract::encoders::expected_for(tp.geometry);
    if catalog.tensors.len() != expected.len()
        || catalog
            .tensors
            .keys()
            .any(|name| !expected.contains_key(name))
    {
        return Err("the encoder catalog differs from the contract's tensors".into());
    }
    Ok(vec![encoders.image_bytes_for(tp.geometry); tp.count()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn real_estimate_is_the_same_on_every_rank() {
        let Ok(root) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
            return;
        };
        let catalog = EncoderCatalog::load(Path::new(&root)).unwrap();
        let estimate = estimated_rank_bytes(
            &catalog,
            Tp::TP4,
            model_mimo26_contract::encoders::Encoders::ALL,
        )
        .unwrap();
        assert_eq!(
            estimate,
            [model_mimo26_contract::encoders::image_bytes_flash(); 4]
        );
        assert_eq!(
            estimated_rank_bytes(
                &catalog,
                Tp::TP2,
                model_mimo26_contract::encoders::Encoders::ALL
            )
            .unwrap(),
            [estimate[0]; 2]
        );
        assert!(estimate[0] > 2_000_000_000 && estimate[0] < 3_000_000_000);
    }
}
