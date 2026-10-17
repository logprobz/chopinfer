//! The HF differential corpus (`models/deepseek41/tokenizer_goldens.py
//! corpus DIR`): about 30,000 texts encoded with every added token
//! recognized must give HF's ids and decode back to their bytes.
//! `CHOPINFER_DEEPSEEK_TOKENIZER_CORPUS` names DIR.
mod common;

#[test]
fn the_hf_corpus_encodes_and_decodes_identically() {
    let Some(corpus) = std::env::var_os("CHOPINFER_DEEPSEEK_TOKENIZER_CORPUS") else {
        eprintln!("CHOPINFER_DEEPSEEK_TOKENIZER_CORPUS is unset: skipped");
        return;
    };
    let Some(dir) = common::checkpoint() else {
        return;
    };
    let corpus = std::path::PathBuf::from(corpus);
    let read = |name: &str| std::fs::read(corpus.join(name)).unwrap();
    let texts: Vec<String> = serde_json::from_slice(&read("corpus.json")).unwrap();
    let expected: Vec<Vec<u32>> = serde_json::from_slice(&read("hf-ids.json")).unwrap();
    assert_eq!(texts.len(), expected.len());
    let tokenizer = model_deepseek41::tokenizer::load(&dir).unwrap();
    let added = common::added_tokens(&dir);
    let allowed = added.iter().map(String::as_str).collect::<Vec<_>>();
    let mut mismatches = Vec::new();
    for (index, (text, ids)) in texts.iter().zip(&expected).enumerate() {
        match tokenizer.encode(text, &allowed) {
            Ok(actual) if &actual == ids => {
                assert_eq!(
                    tokenizer.decode_bytes(&actual).unwrap(),
                    text.as_bytes(),
                    "text {index}"
                );
            }
            Ok(_) => mismatches.push(format!("text {index}: ids differ")),
            Err(error) => mismatches.push(format!("text {index}: {error}")),
        }
    }
    println!(
        "{} corpus texts, {} mismatches",
        texts.len(),
        mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "{}",
        mismatches[..mismatches.len().min(10)].join("\n")
    );
}
