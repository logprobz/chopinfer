//! The fixture declarations format and host tests are driven by: one
//! `format.parser` declaration on each shared tool grammar.
//!
//! They are declarations, not code: whatever differs between the behaviour of
//! two tests that share one renderer (`test_support::FixtureFormatV1`) and
//! the shared parsers came out of these bytes. They declare only what the
//! parsers read; a test that renders serves them with that fixture format.

/// A frontend shaped like a chat model with key-value tool calls
/// (`xml-key-value-arguments-v1`), an effort preamble for its `high` effort,
/// and a reasoning close that is a boundary token.
pub const KEY_VALUE: &str = r##"{
  "schema": "chop-frontend-openai-v2",
  "version": 2,
  "models": [
    {"id": "fixture/alpha-1", "owned_by": "fixture-alpha"},
    {"id": "alpha", "owned_by": "fixture-alias"}
  ],
  "context": {"maximum_total_tokens": 512},
  "limits": {
    "maximum_prompt_bytes": 65536,
    "maximum_completion_bytes": 65536,
    "maximum_generated_tokens": 4096,
    "maximum_client_stops": 4,
    "maximum_messages": 64,
    "maximum_tools": 8,
    "maximum_tool_calls": 4,
    "maximum_tool_parameters": 16,
    "maximum_name_bytes": 64,
    "maximum_json_bytes": 4096,
    "maximum_json_depth": 8,
    "maximum_json_values": 256
  },
  "reasoning": {
    "disable_value": "none",
    "default_effort": "low",
    "efforts": [
      {"name": "low", "preamble": ""},
      {"name": "high", "preamble": "THINK HARDER.\n"}
    ],
    "end_markers": ["</t>"]
  },
  "tokenizer": {
    "repository": "fixture/alpha",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "definition": {
      "name": "tokenizer.json",
      "byte_length": 4096,
      "sha256": "abcdef1111111111111111111111111111111111111111111111111111111111"
    },
    "configuration": {
      "name": "tokenizer_config.json",
      "byte_length": 512,
      "sha256": "fedcba2222222222222222222222222222222222222222222222222222222222"
    },
    "policy": {
      "normalizer": "identity-sequence",
      "pretokenizer_byte_level": {"add_prefix_space": false, "trim_offsets": true, "use_regex": false},
      "post_processor_byte_level": {"add_prefix_space": true, "trim_offsets": false, "use_regex": true},
      "decoder_byte_level": {"add_prefix_space": true, "trim_offsets": true, "use_regex": true},
      "merge_results": {"kind": "sequential", "first_id": 300},
      "merge_records": "space-separated-string",
      "base_tokens": {"kind": "literal-prefix-then-byte-level", "byte_start_id": 44},
      "config_tokens": {"kind": "added-token", "normalized": true},
      "add_bos_token": {"kind": "required", "value": false},
      "bos_token": {"kind": "required", "value": {"kind": "contract-value"}},
      "unk_token": {"kind": "required-null"},
      "tokenizer_class": "PreTrainedTokenizerFast",
      "continuing_subword_prefix": null,
      "end_of_word_suffix": null,
      "ignore_merges": "absent",
      "add_eos_token": false,
      "legacy": true,
      "empty_sentencepiece_kwargs": true
    },
    "base_vocabulary_size": 512,
    "merge_count": 212,
    "added_token_count": 6,
    "model_max_length": 512,
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "pad_token": "[EOS]",
    "pretokenizer_patterns": ["\\s+"]
  },
  "tools": {
    "name_bytes": {"ascii_alphanumeric": true, "extra": "_-"},
    "arguments": "json-object"
  },
  "vocabulary": {
    "size": 4096,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "stop_token_ids": [1],
    "boundary_tokens": [
      {"text": "[BOS]", "id": 0},
      {"text": "[EOS]", "id": 1},
      {"text": "[U]", "id": 2},
      {"text": "[A]", "id": 3},
      {"text": "<t>", "id": 4},
      {"text": "</t>", "id": 5}
    ],
    "prompt_special_tokens": ["[BOS]", "[EOS]", "[U]", "[A]", "<t>", "</t>"]
  },
  "response": {"reasoning_field": "reasoning_content"},
  "format": {
    "parser": {
      "tool_grammar": "xml-key-value-arguments-v1",
      "json_encoding": "python-json-v1",
      "reasoning_open": "<t>",
      "reasoning_close": "</t>",
      "tool_call_open": "<tool_call>",
      "tool_call_close": "</tool_call>",
      "parameter_open": "<arg_key>",
      "parameter_name_close": "</arg_key>",
      "parameter_value_open": "<arg_value>",
      "parameter_close": "</arg_value>",
      "reserved_markers": ["[BOS]", "[EOS]", "[U]", "[A]", "<t>", "</t>"],
      "turn_open": ["[A]", "[U]"],
      "assistant_header": "[A]"
    }
  }
}"##;

/// A frontend that shares little with the first: other identities, half the
/// context, a distinct vocabulary, one effort, and function-parameter tool
/// calls (`xml-function-parameters-v1`) with the markers and the stop text of
/// the shared function-parameter cases (`function_parameter_cases.json`).
pub const FUNCTION_PARAMETERS: &str = r##"{
  "schema": "chop-frontend-openai-v2",
  "version": 2,
  "models": [{"id": "fixture/gamma-3", "owned_by": "fixture-gamma"}],
  "context": {"maximum_total_tokens": 256},
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
    "efforts": [{"name": "on", "preamble": ""}],
    "end_markers": ["</think>"]
  },
  "tokenizer": {
    "repository": "fixture/gamma",
    "revision": "abcdefabcdefabcdefabcdefabcdefabcdefabcd",
    "definition": {
      "name": "tokenizer.json",
      "byte_length": 1024,
      "sha256": "5555555555555555555555555555555555555555555555555555555555555555"
    },
    "configuration": {
      "name": "tokenizer_config.json",
      "byte_length": 256,
      "sha256": "6666666666666666666666666666666666666666666666666666666666666666"
    },
    "policy": {
      "normalizer": "identity-sequence",
      "pretokenizer_byte_level": {"add_prefix_space": false, "trim_offsets": true, "use_regex": false},
      "post_processor_byte_level": {"add_prefix_space": true, "trim_offsets": false, "use_regex": true},
      "decoder_byte_level": {"add_prefix_space": true, "trim_offsets": true, "use_regex": true},
      "merge_results": {"kind": "sequential", "first_id": 256},
      "merge_records": "space-separated-string",
      "base_tokens": {"kind": "literal-prefix-then-byte-level", "byte_start_id": 0},
      "config_tokens": {"kind": "added-token", "normalized": true},
      "add_bos_token": {"kind": "required", "value": false},
      "bos_token": {"kind": "required", "value": {"kind": "contract-value"}},
      "unk_token": {"kind": "required-null"},
      "tokenizer_class": "PreTrainedTokenizerFast",
      "continuing_subword_prefix": null,
      "end_of_word_suffix": null,
      "ignore_merges": "absent",
      "add_eos_token": false,
      "legacy": true,
      "empty_sentencepiece_kwargs": true
    },
    "base_vocabulary_size": 256,
    "merge_count": 0,
    "added_token_count": 4,
    "model_max_length": 256,
    "bos_token": "<|s|>",
    "eos_token": "[EOS]",
    "pad_token": "[EOS]",
    "pretokenizer_patterns": [".+"]
  },
  "tools": {
    "name_bytes": {"ascii_alphanumeric": true, "extra": "_-"},
    "arguments": "json-object"
  },
  "vocabulary": {
    "size": 300,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "stop_token_ids": [1],
    "boundary_tokens": [
      {"text": "<|s|>", "id": 0},
      {"text": "[EOS]", "id": 1},
      {"text": "</think>", "id": 2},
      {"text": "<|u|>", "id": 3},
      {"text": "<|a|>", "id": 4}
    ],
    "prompt_special_tokens": ["<|s|>", "[EOS]", "</think>", "<|u|>", "<|a|>"]
  },
  "response": {"reasoning_field": "reasoning_content"},
  "format": {
    "parser": {
      "tool_grammar": "xml-function-parameters-v1",
      "json_encoding": "python-json-v1",
      "reasoning_open": "<think>",
      "reasoning_close": "</think>",
      "tool_call_open": "<tool_call>\n",
      "tool_call_close": "</tool_call>",
      "function_open": "<function=",
      "function_name_close": ">\n",
      "function_close": "</function>\n",
      "parameter_open": "<parameter=",
      "parameter_name_close": ">\n",
      "parameter_close": "\n</parameter>\n",
      "reserved_markers": ["<|s|>", "[EOS]", "<|u|>", "<|a|>", "<think>", "</think>"],
      "turn_open": ["<|a|>", "<|u|>"],
      "assistant_header": "<|a|>"
    }
  }
}"##;
