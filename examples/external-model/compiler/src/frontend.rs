//! A byte tokenizer and model-owned declaration for the real HTTP frontend.
use example_contract::{CONTEXT, MARKERS, MODEL, VOCAB};
use serde_json::{Value, json};

fn bytes(value: Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}
fn identity(name: &str, data: &[u8]) -> Value {
    json!({"name":name,"byte_length":data.len(),"sha256":runtime_weights::sha256_bytes(data)})
}

pub fn artifacts() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut vocab = serde_json::Map::new();
    let mut extra = 256;
    for byte in 0..=255_u32 {
        let direct = (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174;
        let scalar = if direct {
            byte
        } else {
            let n = extra;
            extra += 1;
            n
        };
        vocab.insert(char::from_u32(scalar).unwrap().to_string(), json!(byte));
    }
    // The reasoning and tool-call markers the parser reads from a completion
    // are not special: a chat request skips special tokens by default, and the
    // decoder clears a special token's bytes before the parser sees them.
    let output = [6, 7, 8, 9, 12, 13, 14, 15];
    let added: Vec<_> = MARKERS
        .iter()
        .enumerate()
        .map(|(i, text)| {
            json!({
        "id":256+i,"content":text,"single_word":false,"lstrip":false,"rstrip":false,
        "normalized":true,"special":!output.contains(&i)})
        })
        .collect();
    let definition = bytes(json!({
        "version":"1.0","truncation":null,"padding":null,"added_tokens":added,
        "normalizer":{"type":"Sequence","normalizers":[]},
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[
            {"type":"Split","pattern":{"Regex":"(?s)."},"behavior":"Isolated","invert":false},
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
        "post_processor":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
        "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,
            "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":vocab,"merges":[]}
    }));
    let token = |s: &str| {
        json!({"__type":"AddedToken","content":s,"single_word":false,
        "lstrip":false,"rstrip":false,"normalized":true})
    };
    let configuration = bytes(json!({"add_bos_token":false,"add_eos_token":false,
        "clean_up_tokenization_spaces":false,"legacy":true,"tokenizer_class":"PreTrainedTokenizerFast",
        "unk_token":null,"model_max_length":CONTEXT,"bos_token":token(MARKERS[0]),
        "eos_token":token(MARKERS[1]),"pad_token":token(MARKERS[1]),"sp_model_kwargs":{}}));
    let boundaries: Vec<_> = MARKERS
        .iter()
        .enumerate()
        .map(|(i, s)| json!({"text":s,"id":256+i}))
        .collect();
    let prompt: Vec<_> = MARKERS
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 1)
        .map(|(_, s)| *s)
        .collect();
    // What the shared parsers read of a completion: the key-value tool
    // grammar over the example's markers. The renderer in
    // `example_contract::frontend` writes the prompt.
    let parser = json!({"tool_grammar":"xml-key-value-arguments-v1","json_encoding":"spaced-json-v1",
        "reasoning_open":MARKERS[6],"reasoning_close":MARKERS[7],
        "tool_call_open":MARKERS[8],"tool_call_close":MARKERS[9],
        "parameter_open":MARKERS[12],"parameter_name_close":MARKERS[13],
        "parameter_value_open":MARKERS[14],"parameter_close":MARKERS[15],
        "reserved_markers":MARKERS,
        "turn_open":[MARKERS[4],MARKERS[2],MARKERS[5],MARKERS[3]],"assistant_header":MARKERS[4]});
    let declaration = bytes(json!({
        "schema":"chop-frontend-openai-v2","version":2,
        "models":[{"id":MODEL,"owned_by":"external-example"}],"context":{"maximum_total_tokens":CONTEXT},
        "limits":{"maximum_prompt_bytes":65536,"maximum_completion_bytes":65536,
            "maximum_generated_tokens":64,"maximum_client_stops":4,"maximum_messages":64,
            "maximum_tools":8,"maximum_tool_calls":8,"maximum_tool_parameters":16,
            "maximum_name_bytes":64,"maximum_json_bytes":65536,"maximum_json_depth":16,"maximum_json_values":4096},
        "reasoning":{"thinking_keywords":["enable_thinking"],"disable_value":"off","default_effort":"standard","efforts":[{"name":"standard","preamble":""}],
            "end_markers":[MARKERS[7],MARKERS[8]]},
        "tokenizer":{"repository":"example/byte-tokenizer",
            "revision":&runtime_weights::sha256_bytes(&definition)[..40],
            "definition":identity("tokenizer.json", &definition),
            "configuration":identity("tokenizer_config.json", &configuration),
            "policy":{"normalizer":"identity-sequence",
                "pretokenizer_byte_level":{"add_prefix_space":false,"trim_offsets":true,"use_regex":false},
                "post_processor_byte_level":{"add_prefix_space":true,"trim_offsets":false,"use_regex":true},
                "decoder_byte_level":{"add_prefix_space":true,"trim_offsets":true,"use_regex":true},
                "merge_results":{"kind":"sequential","first_id":256},"merge_records":"space-separated-string",
                "base_tokens":{"kind":"literal-prefix-then-byte-level","byte_start_id":0},
                "config_tokens":{"kind":"added-token","normalized":true},
                "add_bos_token":{"kind":"required","value":false},
                "bos_token":{"kind":"required","value":{"kind":"contract-value"}},
                "unk_token":{"kind":"required-null"},"tokenizer_class":"PreTrainedTokenizerFast",
                "continuing_subword_prefix":null,"end_of_word_suffix":null,"ignore_merges":"absent",
                "add_eos_token":false,"legacy":true,"empty_sentencepiece_kwargs":true},
            "base_vocabulary_size":256,"merge_count":0,"added_token_count":16,
            "model_max_length":CONTEXT,"bos_token":MARKERS[0],"eos_token":MARKERS[1],"pad_token":MARKERS[1],
            "pretokenizer_patterns":["(?s)."]},
        "tools":{"name_bytes":{"ascii_alphanumeric":true,"extra":"_-"},"arguments":"json-object-integral-numbers"},
        "vocabulary":{"size":VOCAB,"bos_token_id":256,"eos_token_id":257,"stop_token_ids":[257],
            "boundary_tokens":boundaries,"prompt_special_tokens":prompt},
        "response":{"reasoning_field":"reasoning_content"},
        "format":{"parser":parser}
    }));
    (definition, configuration, declaration)
}

#[cfg(test)]
mod tests {
    #[test]
    fn declaration_is_accepted_by_the_shared_parser() {
        let (_, _, declaration) = super::artifacts();
        runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(&declaration)
            .unwrap();
    }
}
