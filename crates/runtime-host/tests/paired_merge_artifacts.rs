#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use runtime_bundle::{
    BundleHostCompatibilityV1, BundleManifestV1, BundleOpaqueId, BundlePackageV1,
};
use runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1;
use runtime_frontend::test_support::fixture_export_v1;
use runtime_frontend::{
    FRONTEND_COMPONENT_ROLE_V1, TOKENIZER_CONFIGURATION_ROLE_V1, TOKENIZER_DEFINITION_ROLE_V1,
};
use runtime_host::deployment_from_package_v1;
use runtime_server::ServerConfig;
use runtime_tokenizer::Tokenizer;
use runtime_weights::sha256_bytes;
use serde_json::{Value, json};

const DECLARATION: &str = r##"{
  "schema": "chop-frontend-openai-v2",
  "version": 2,
  "models": [
    {
      "id": "fixture/paired",
      "owned_by": "fixture"
    }
  ],
  "context": {
    "maximum_total_tokens": 256
  },
  "limits": {
    "maximum_prompt_bytes": 8192,
    "maximum_completion_bytes": 8192,
    "maximum_generated_tokens": 256,
    "maximum_client_stops": 2,
    "maximum_messages": 16,
    "maximum_tools": 4,
    "maximum_tool_calls": 4,
    "maximum_tool_parameters": 8,
    "maximum_name_bytes": 32,
    "maximum_json_bytes": 2048,
    "maximum_json_depth": 8,
    "maximum_json_values": 128
  },
  "reasoning": {
    "disable_value": "off",
    "default_effort": "on",
    "efforts": [
      {
        "name": "on",
        "preamble": ""
      }
    ],
    "end_markers": ["<|/think|>", "<|calls|>"]
  },
  "tools": {
    "name_bytes": {"ascii_alphanumeric": true, "extra": "_-"},
    "arguments": "json-object-integral-numbers"
  },
  "response": {"reasoning_field": "reasoning_content"},
  "format": {
    "parser": {
      "tool_grammar": "xml-key-value-arguments-v1",
      "json_encoding": "spaced-json-v1",
      "reasoning_open": "<|think|>",
      "reasoning_close": "<|/think|>",
      "tool_call_open": "<|calls|>",
      "tool_call_close": "<|/calls|>",
      "parameter_open": "<|k|>",
      "parameter_name_close": "<|/k|>",
      "parameter_value_open": "<|v|>",
      "parameter_close": "<|/v|>",
      "reserved_markers": ["<|"],
      "turn_open": ["<|a|>", "<|u|>"],
      "assistant_header": "<|a|>"
    }
  }
}"##;

struct Artifacts {
    root: PathBuf,
    definition: Vec<u8>,
    configuration: Vec<u8>,
}

impl Artifacts {
    fn new(change: impl FnOnce(&mut Value, &mut Value)) -> Self {
        Self::with_bytes(change, |_, _| {})
    }

    fn with_bytes(
        change: impl FnOnce(&mut Value, &mut Value),
        change_bytes: impl FnOnce(&mut Vec<u8>, &mut Vec<u8>),
    ) -> Self {
        Self::with_pins(change, change_bytes, false)
    }

    fn with_pins(
        change: impl FnOnce(&mut Value, &mut Value),
        change_bytes: impl FnOnce(&mut Vec<u8>, &mut Vec<u8>),
        pin_before_change: bool,
    ) -> Self {
        // Byte token IDs equal their raw byte values. The standard byte-level
        // alphabet keeps printable Latin-1 and maps the remaining bytes above 255.
        let visible = |b: u32| {
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b)
        };
        let mut vocab = serde_json::Map::new();
        let mut extra = 256;
        for b in 0..256 {
            let code = if visible(b) {
                b
            } else {
                let code = extra;
                extra += 1;
                code
            };
            vocab.insert(char::from_u32(code).unwrap().to_string(), json!(b));
        }
        vocab.insert("bc".into(), json!(256));
        vocab.insert("ab".into(), json!(257));
        vocab.insert("abc".into(), json!(258));
        let mut declaration: Value = serde_json::from_str(DECLARATION).unwrap();
        let specials = [
            "<|s|>",
            "<|end|>",
            "<|/think|>",
            "<|calls|>",
            "<|u|>",
            "<|a|>",
        ];
        let added: Vec<Value> = specials
            .iter()
            .enumerate()
            .map(|(i, text)| {
                json!({
                    "id":259+i,"content":text,"single_word":false,"lstrip":false,
                    "rstrip":false,"normalized":false,"special":true
                })
            })
            .collect();
        let mut definition = json!({
            "version":"1.0","truncation":null,"padding":null,"added_tokens":added,
            "normalizer":null,
            "pre_tokenizer":{"type":"Sequence","pretokenizers":[
                {"type":"Split","pattern":{"Regex":"(?s).+"},"behavior":"Isolated","invert":false},
                {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
            "post_processor":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
            "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            "model":{"type":"BPE","dropout":null,"unk_token":null,
                "continuing_subword_prefix":null,"end_of_word_suffix":null,
                "fuse_unk":false,"byte_fallback":false,"ignore_merges":true,
                "vocab":vocab,"merges":[["b","c"],["a","b"]]}
        });
        let mut configuration = json!({"clean_up_tokenization_spaces":false,
            "tokenizer_class":"TokenizersBackend","model_max_length":512,
            "eos_token":specials[1],"pad_token":specials[1]});
        change(&mut definition, &mut configuration);
        let mut definition = serde_json::to_vec(&definition).unwrap();
        let mut configuration = serde_json::to_vec(&configuration).unwrap();
        if pin_before_change {
            definition.push(b' ');
            configuration.push(b' ');
        }
        let pinned_definition = definition.clone();
        let pinned_configuration = configuration.clone();
        change_bytes(&mut definition, &mut configuration);
        declaration["tokenizer"] = json!({
            "repository":"fixture/paired",
            "revision":"abcdefabcdefabcdefabcdefabcdefabcdefabcd",
            "eos_token":specials[1],"pad_token":specials[1]
        });
        declaration["vocabulary"] = json!({
            "boundary_tokens":specials.iter().enumerate().map(|(i, text)|
                json!({"id":259+i,"text":text})).collect::<Vec<_>>(),
            "prompt_special_tokens":specials
        });
        declaration["tokenizer"]["policy"] = json!({
            "normalizer":"null",
            "pretokenizer_byte_level":{"add_prefix_space":false,"trim_offsets":true,"use_regex":false},
            "post_processor_byte_level":{"add_prefix_space":true,"trim_offsets":false,"use_regex":true},
            "decoder_byte_level":{"add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            "merge_results":{"kind":"vocabulary"},"merge_records":"string-pair-array",
            "base_tokens":{"kind":"byte-level"},"config_tokens":{"kind":"string"},
            "add_bos_token":{"kind":"absent"},"bos_token":{"kind":"absent"},
            "unk_token":{"kind":"absent"},"tokenizer_class":"TokenizersBackend",
            "continuing_subword_prefix":null,"end_of_word_suffix":null,
            "ignore_merges":"whole-piece","add_eos_token":null,"legacy":null,
            "empty_sentencepiece_kwargs":false
        });
        declaration["tokenizer"]["base_vocabulary_size"] = json!(259);
        declaration["tokenizer"]["merge_count"] = json!(2);
        declaration["tokenizer"]["added_token_count"] = json!(6);
        declaration["tokenizer"]["bos_token"] = Value::Null;
        declaration["tokenizer"]["model_max_length"] = json!(512);
        declaration["tokenizer"]["pretokenizer_patterns"] = json!(["(?s).+"]);
        declaration["vocabulary"]["size"] = json!(265);
        declaration["vocabulary"]["bos_token_id"] = json!(259);
        declaration["vocabulary"]["eos_token_id"] = json!(260);
        declaration["vocabulary"]["stop_token_ids"] = json!([260]);
        for (field, name, bytes) in [
            (
                "definition",
                "tokenizer.json",
                if pin_before_change {
                    &pinned_definition
                } else {
                    &definition
                },
            ),
            (
                "configuration",
                "tokenizer_config.json",
                if pin_before_change {
                    &pinned_configuration
                } else {
                    &configuration
                },
            ),
        ] {
            declaration["tokenizer"][field] =
                json!({"name":name,"byte_length":bytes.len(),"sha256":sha256_bytes(bytes)});
        }
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "paired-artifacts-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let declaration = serde_json::to_vec(&declaration).unwrap();
        let mut components: Vec<Value> = [
            ("frontend",FRONTEND_COMPONENT_ROLE_V1,"frontend.json",&declaration),
            ("definition",TOKENIZER_DEFINITION_ROLE_V1,"tokenizer.json",&definition),
            ("configuration",TOKENIZER_CONFIGURATION_ROLE_V1,"tokenizer_config.json",&configuration)
        ].into_iter().map(|(id,role,name,bytes)| {
            fs::write(root.join(name),bytes).unwrap();
            json!({"id":id,"role":role,"format":"fixture.json.v1","path":name,"byte_length":bytes.len(),"sha256":sha256_bytes(bytes)})
        }).collect();
        components.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        let manifest = json!({"schema":"chop-model-bundle-v1","abi_version":1,"id":"fixture.paired.v1",
            "target":{"architectures":["sm_120f"],"topologies":[{"id":"tp1","rank_count":1}]},
            "components":components,"resource_requirements":{"minimum_device_memory_bytes":1,"required_capabilities":[FRONTEND_CAPABILITY_V1]},
            "kv_layout":{"id":"fixture.kv.v1","page_bytes":64,"alignment_bytes":64,"page_tokens":1,
                "bytes_per_token":64,"segments_per_page":1,"maximum_sequence_tokens":512,"rank_placement":"replicated","state_lifetime":"transaction"},
            "state_layout":{"id":"fixture.state.v1","bytes":64,"alignment_bytes":64,"lifetime":"request"},
            "graph_buckets":[{"id":"decode-1","components":["configuration","definition","frontend"],"operations":[]}],
            "operation_requirements":[],"implementations":[]});
        let manifest: BundleManifestV1 = serde_json::from_value(manifest).unwrap();
        fs::write(
            root.join("manifest.json"),
            manifest.to_canonical_json().unwrap(),
        )
        .unwrap();
        Self {
            root,
            definition,
            configuration,
        }
    }

    fn package(&self) -> Result<BundlePackageV1, runtime_bundle::BundleError> {
        BundlePackageV1::verify_directory(
            &self.root,
            &BundleHostCompatibilityV1::new("sm_120f", 1)
                .unwrap()
                .with_capabilities([BundleOpaqueId::new(FRONTEND_CAPABILITY_V1).unwrap()]),
        )
    }

    fn load(&self) -> Result<Tokenizer, String> {
        let package = self.package().unwrap();
        deployment_from_package_v1(&package, &fixture_export_v1(), ServerConfig::default())
            .map(|deployment| deployment.into_tokenizer())
            .map_err(|error| error.to_string())
    }
}

impl Drop for Artifacts {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn paired_artifacts_round_trip_unicode_and_raw_bytes() {
    let artifacts = Artifacts::new(|_, _| {});
    let tokenizer = artifacts.load().unwrap();
    for (text, ids) in [
        ("ab", vec![257]),
        ("é", vec![195, 169]),
        ("e\u{301}", vec![101, 204, 129]),
        ("\u{1f600}", vec![240, 159, 152, 128]),
        ("a\n", vec![97, 10]),
    ] {
        assert_eq!(tokenizer.encode(text, &[]).unwrap(), ids);
        assert_eq!(tokenizer.decode_bytes(&ids).unwrap(), text.as_bytes());
        assert_eq!(tokenizer.decode_utf8(&ids).unwrap(), text);
    }
    assert_eq!(tokenizer.decode_bytes(&[255]).unwrap(), [255]);
    assert_eq!(
        tokenizer.decode_utf8(&[255]).unwrap_err().to_string(),
        "decoded tokens are not valid UTF-8: invalid utf-8 sequence of 1 bytes from index 0"
    );
    assert_eq!(
        tokenizer.decode_utf8(&[195]).unwrap_err().to_string(),
        "decoded tokens are not valid UTF-8: incomplete utf-8 byte sequence from index 0"
    );
}

#[test]
fn paired_artifacts_rank_merges_and_bypass_whole_pieces() {
    let first = Artifacts::new(|_, _| {});
    let second = Artifacts::new(|d, _| d["model"]["merges"] = json!([["a", "b"], ["b", "c"]]));
    // "abc" exists and bypasses merging; "abca" does not, so the
    // overlapping pairs compete by rank, independently of their token IDs.
    assert_eq!(first.load().unwrap().encode("abc", &[]).unwrap(), [258]);
    assert_eq!(
        first.load().unwrap().encode("abca", &[]).unwrap(),
        [97, 256, 97]
    );
    assert_eq!(
        second.load().unwrap().encode("abca", &[]).unwrap(),
        [257, 99, 97]
    );
}

#[test]
fn paired_artifacts_keep_special_boundaries() {
    let artifacts = Artifacts::new(|_, _| {});
    let tokenizer = artifacts.load().unwrap();
    let text = "ab<|end|><|calls|>bc";
    assert_eq!(
        tokenizer.encode(text, &["<|end|>", "<|calls|>"]).unwrap(),
        [257, 260, 262, 256]
    );
    assert_eq!(tokenizer.decode_utf8(&[257, 260, 262, 256]).unwrap(), text);
    assert_eq!(
        tokenizer.encode(text, &[]).unwrap_err().to_string(),
        "disallowed special token '<|end|>' in input"
    );
    assert_eq!(
        tokenizer
            .encode(text, &["<|end|>"])
            .unwrap_err()
            .to_string(),
        "disallowed special token '<|calls|>' in input"
    );
    assert_eq!(
        tokenizer.encode("<|end", &[]).unwrap(),
        [60, 124, 101, 110, 100]
    );
}

#[test]
fn paired_artifacts_refuse_package_digest_drift_and_retain_verified_bytes() {
    let artifacts = Artifacts::new(|_, _| {});
    let package = artifacts.package().unwrap();
    for (name, bytes) in [
        ("tokenizer.json", &artifacts.definition),
        ("tokenizer_config.json", &artifacts.configuration),
    ] {
        let mut changed = bytes.clone();
        changed[0] = b' ';
        fs::write(artifacts.root.join(name), &changed).unwrap();
        let error = artifacts.package().unwrap_err().to_string();
        assert_eq!(
            error,
            format!(
                "bundle component {} SHA-256 {} differs from the manifest's {}",
                artifacts.root.join(name).display(),
                sha256_bytes(&changed),
                sha256_bytes(bytes)
            )
        );
        assert_eq!(
            deployment_from_package_v1(&package, &fixture_export_v1(), ServerConfig::default())
                .unwrap_or_else(|error| panic!("{error}"))
                .tokenizer()
                .encode("abca", &[])
                .unwrap(),
            [97, 256, 97]
        );
        let component = package
            .manifest()
            .components()
            .iter()
            .find(|c| c.path().as_str() == name)
            .unwrap();
        assert_eq!(package.read_component(component).unwrap(), *bytes);
        fs::write(artifacts.root.join(name), bytes).unwrap();
    }
}

#[test]
fn paired_artifacts_refuse_authenticated_malformed_contents() {
    for (change, expected) in [
        (0, "BPE merge count does not match policy"),
        (
            1,
            "malformed BPE merge record 0: expected a two-string array",
        ),
        (2, "duplicate or out-of-range BPE token ID"),
        (3, "duplicate added token content"),
        (4, "BPE merge 0 has unknown left token"),
        (5, "tokenizer model_max_length does not match policy"),
        (6, "duplicate BPE merge pair"),
        (7, "JSON field 'bos_token' must be absent"),
    ] {
        let artifacts = Artifacts::new(|d, c| match change {
            0 => {
                d["model"]["merges"].as_array_mut().unwrap().pop();
            }
            1 => d["model"]["merges"][0] = json!(["b"]),
            2 => d["model"]["vocab"]["ab"] = json!(256),
            3 => d["added_tokens"][1]["content"] = json!("<|s|>"),
            4 => d["model"]["merges"][0] = json!(["missing", "c"]),
            5 => c["model_max_length"] = json!(513),
            6 => d["model"]["merges"][1] = json!(["b", "c"]),
            7 => c["bos_token"] = json!("<|s|>"),
            _ => unreachable!(),
        });
        let error = artifacts.load().unwrap_err();
        assert_eq!(
            error,
            format!("declared tokenizer does not build: {expected}"),
            "case {change}"
        );
    }
}

#[test]
fn paired_artifacts_refuse_authenticated_truncated_json_and_duplicate_keys() {
    for configuration in [false, true] {
        for duplicate in [false, true] {
            let artifacts = Artifacts::with_bytes(
                |_, _| {},
                |d, c| {
                    let bytes = if configuration { c } else { d };
                    if duplicate {
                        let prefix: &[u8] = if configuration {
                            br#"{"model_max_length":512,"#
                        } else {
                            br#"{"version":"1.0","#
                        };
                        bytes.splice(..1, prefix.iter().copied());
                    } else {
                        bytes.pop();
                    }
                },
            );
            let error = artifacts.load().unwrap_err();
            let bytes = if configuration {
                &artifacts.configuration
            } else {
                &artifacts.definition
            };
            let expected = if duplicate {
                let key = if configuration {
                    "model_max_length"
                } else {
                    "version"
                };
                let marker = if configuration {
                    b"\"model_max_length\":512".as_slice()
                } else {
                    b"\"version\":\"1.0\"".as_slice()
                };
                let offset = bytes
                    .windows(marker.len())
                    .rposition(|part| part == marker)
                    .unwrap()
                    + marker.len();
                format!("JSON error at byte {offset}: duplicate JSON key '{key}'")
            } else {
                format!("JSON error at byte {}: expected ','", bytes.len())
            };
            assert_eq!(
                error,
                format!("declared tokenizer does not build: {expected}")
            );
        }
    }
}

#[test]
fn paired_artifacts_allow_unknown_metadata_keys() {
    // These JSON formats validate configured fields and allow extra metadata.
    let artifacts = Artifacts::new(|d, c| {
        d["fixture_metadata"] = json!("extra");
        c["fixture_metadata"] = json!("extra");
    });
    assert_eq!(
        artifacts.load().unwrap().encode("abca", &[]).unwrap(),
        [97, 256, 97]
    );
}

#[test]
fn paired_artifacts_refuse_stale_declared_digests_after_package_verification() {
    for configuration in [false, true] {
        let artifacts = Artifacts::with_pins(
            |_, _| {},
            |d, c| {
                // JSON meaning and length stay valid; only the pinned digest changes.
                let bytes = if configuration { c } else { d };
                *bytes.last_mut().unwrap() = b'\n';
            },
            true,
        );
        let package = artifacts.package().unwrap();
        let error =
            deployment_from_package_v1(&package, &fixture_export_v1(), ServerConfig::default())
                .err()
                .expect("stale declaration must be refused")
                .to_string();
        let name = if configuration {
            "tokenizer_config.json"
        } else {
            "tokenizer.json"
        };
        assert_eq!(
            error,
            format!(
                "declared tokenizer does not build: tokenizer artifact {name} SHA-256 mismatch"
            )
        );
    }
}
