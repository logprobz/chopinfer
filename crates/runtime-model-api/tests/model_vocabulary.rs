use runtime_model_api::{ModelVocabulary, ModelVocabularyError};

#[test]
fn model_vocabulary_is_exact_nonzero_and_token_addressable() {
    assert_eq!(
        ModelVocabulary::new(0),
        Err(ModelVocabularyError::ZeroVocabulary)
    );
    assert_eq!(ModelVocabulary::new(154_880).unwrap().get(), 154_880);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn model_vocabulary_accepts_every_u32_token_id_and_rejects_one_more() {
    let exact = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    assert_eq!(ModelVocabulary::new(exact).unwrap().get(), exact);
    let overflow = exact + 1;
    assert_eq!(
        ModelVocabulary::new(overflow),
        Err(ModelVocabularyError::TokenIdOverflow { provided: overflow })
    );
}

#[cfg(target_pointer_width = "32")]
#[test]
fn model_vocabulary_accepts_the_largest_representable_count() {
    assert_eq!(ModelVocabulary::new(usize::MAX).unwrap().get(), usize::MAX);
}
