//! The two tokenizer artifacts this fixture generates, and what they are not.
//!
//! These are a working tokenizer. The shared reader parses them, assigns every
//! one of the 154,880 model token identifiers, and binds each boundary the
//! served declaration pins to the text that identifier decodes to. That is what
//! lets `deployment_from_package_v1` succeed on this package, and therefore what
//! lets a real engine and a real transport be driven against it.
//!
//! They are generated here and nothing is copied. No upstream `tokenizer.json`
//! is read, no upstream digest is reused, and the identity the declaration pins
//! is the identity of what this crate just produced. The repository and revision
//! name this fixture.
//!
//! The encoding is this fixture's own, not TP4's. Its declared policy has the
//! base-token representation this construction needs: identifiers below the
//! byte alphabet are literal, the 256 byte tokens sit directly beneath the
//! added block, and merge results are sequential from a fixed identifier. TP4's published
//! tokenizer is a different artifact under a different encoding with 321,649
//! merges, and this package neither ships it nor claims to.
//!
//! The layout is exact:
//!
//! ```text
//!            0 .. 154_563   literal base tokens, unique and unreachable by
//!                           ordinary encoding, which is byte-level here
//!      154_564 .. 154_819   the 256 byte-alphabet tokens, in byte order
//!      154_820              the published TP4 stop token, at its published id
//!      154_820 .. 154_879   the added block: the published TP4 markers at their
//!                           published identifiers, and a unique fixture filler
//!                           for every remaining identifier
//! ```
//!
//! There are no merges, so ordinary text encodes byte by byte and decodes back
//! to exactly the bytes it came from.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::metadata::{
    TP4_PRETOKENIZER_PATTERN, TP4_TEXT_PROMPT_SPECIAL_TOKENS, tokenizer_metadata,
};
use runtime_weights::sha256_bytes;

use crate::contract::{TP4_PROMPT_TOKEN_IDS_V1, tp4_fixture_vocabulary_v1};

/// The repository identity the served declaration pins these bytes under.
pub const TP4_FIXTURE_TOKENIZER_REPOSITORY_V1: &str = "chopinfer/tp4-fixture-tokenizer";

/// The salt the synthetic revision is derived from.
///
/// The revision a declaration carries is 40 hexadecimal digits. Taking them from
/// this salt keeps the value deterministic and unmistakably this fixture's,
/// rather than an upstream commit this package never read.
const TP4_FIXTURE_TOKENIZER_REVISION_SALT_V1: &[u8] = b"chop-tp4-fixture-tokenizer-revision-v1";

/// The identifier the added block starts at, which is also the base vocabulary.
///
/// It is TP4's published stop-token identifier, so the block of identifiers this
/// fixture assigns begins exactly where the published stop token sits.
pub const TP4_FIXTURE_BASE_VOCABULARY_V1: u32 = 154_820;

/// The identifier the 256 byte-alphabet tokens start at.
pub const TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1: u32 = TP4_FIXTURE_BASE_VOCABULARY_V1 - 256;

/// How many merges the generated definition declares.
///
/// None. Ordinary text is therefore one token per byte, which is what makes an
/// encode and a decode through these artifacts exactly reversible.
pub const TP4_FIXTURE_MERGE_COUNT_V1: usize = 0;

/// The prefix every literal base token carries.
///
/// Two characters at minimum, so no literal base token can collide with one of
/// the 256 single-character byte-alphabet tokens.
const TP4_FIXTURE_BASE_TOKEN_PREFIX_V1: &str = "f";

/// The longest sequence the generated tokenizer declares it can read.
///
/// It is this fixture's own bound. TP4 declares a far larger one, and
/// nothing here is that declaration.
pub const TP4_FIXTURE_TOKENIZER_MODEL_MAX_LENGTH_V1: usize = 1_024;

/// The tokenizer class the declared policy requires the configuration to name.
pub(crate) const TP4_FIXTURE_TOKENIZER_CLASS_V1: &str = "PreTrainedTokenizerFast";

/// How many identifiers the added block covers.
#[must_use]
pub fn tp4_fixture_added_token_count_v1() -> usize {
    (tp4_fixture_vocabulary_v1() - TP4_FIXTURE_BASE_VOCABULARY_V1) as usize
}

/// The 40 hexadecimal digits the served declaration pins as the revision.
#[must_use]
pub fn tp4_fixture_tokenizer_revision_v1() -> String {
    sha256_bytes(TP4_FIXTURE_TOKENIZER_REVISION_SALT_V1)[..40].to_owned()
}

/// One filler token's text, which no TP4 vocabulary entry carries.
///
/// It is deliberately not bracketed: a filler is ordinary visible text, not a
/// marker, and a completion that decodes to one has to read as text.
#[must_use]
pub fn tp4_fixture_filler_token_v1(token_id: u32) -> String {
    format!("chop-fixture-filler-{token_id}")
}

/// One literal base token's text.
#[must_use]
pub fn tp4_fixture_base_token_v1(token_id: u32) -> String {
    format!("{TP4_FIXTURE_BASE_TOKEN_PREFIX_V1}{token_id}")
}

/// Every published TP4 marker this fixture reuses, with its published id.
///
/// The 16 prompt markers come from the model crate; the stop token comes from
/// the same crate's tokenizer metadata. Nothing here is invented.
#[must_use]
pub fn tp4_published_markers_v1() -> Vec<(u32, &'static str)> {
    let metadata = tokenizer_metadata();
    let mut markers = vec![(metadata.eos_token_id(), metadata.eos_token())];
    markers.extend(
        TP4_PROMPT_TOKEN_IDS_V1
            .iter()
            .copied()
            .zip(TP4_TEXT_PROMPT_SPECIAL_TOKENS),
    );
    markers.sort_unstable();
    markers
}

/// The published markers a completion carries to the frontend's parser: the
/// reasoning and tool-call markers the served declaration's grammar reads.
///
/// They are not special. A chat request skips special tokens by default, and
/// the decoder clears a special token's bytes before the parser sees them, so
/// a special reasoning close or call marker would never be parsed.
pub const TP4_FIXTURE_OUTPUT_MARKERS_V1: [&str; 8] = [
    TP4_TEXT_PROMPT_SPECIAL_TOKENS[6],
    TP4_TEXT_PROMPT_SPECIAL_TOKENS[7],
    TP4_TEXT_PROMPT_SPECIAL_TOKENS[8],
    TP4_TEXT_PROMPT_SPECIAL_TOKENS[9],
    TP4_TEXT_PROMPT_SPECIAL_TOKENS[12],
    TP4_TEXT_PROMPT_SPECIAL_TOKENS[13],
    TP4_TEXT_PROMPT_SPECIAL_TOKENS[14],
    TP4_TEXT_PROMPT_SPECIAL_TOKENS[15],
];

/// One entry of the added block: its text, and whether it is a control marker.
///
/// The published markers and the stop token are special, apart from the output
/// markers ([`TP4_FIXTURE_OUTPUT_MARKERS_V1`]). A filler is not, so a sampler
/// that lands on one produces visible text rather than a boundary a decoder
/// would have to police.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddedTokenV1 {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

/// The whole added block, in identifier order, with every identifier assigned.
#[must_use]
pub fn tp4_fixture_added_tokens_v1() -> Vec<AddedTokenV1> {
    let published = tp4_published_markers_v1()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    (TP4_FIXTURE_BASE_VOCABULARY_V1..tp4_fixture_vocabulary_v1())
        .map(|id| match published.get(&id) {
            Some(marker) => AddedTokenV1 {
                id,
                content: (*marker).to_owned(),
                special: !TP4_FIXTURE_OUTPUT_MARKERS_V1.contains(marker),
            },
            None => AddedTokenV1 {
                id,
                content: tp4_fixture_filler_token_v1(id),
                special: false,
            },
        })
        .collect()
}

/// The byte-alphabet character every byte is written as.
///
/// This is the byte-level alphabet the shared reader decodes base tokens with,
/// regenerated here because the generator has to emit the exact characters that
/// reader will look up. The test below builds a real tokenizer from these bytes,
/// which is what proves the two agree.
#[must_use]
pub fn tp4_fixture_byte_alphabet_v1() -> [char; 256] {
    let mut direct = Vec::new();
    direct.extend(b'!'..=b'~');
    direct.extend(0xa1..=0xac_u8);
    direct.extend(0xae..=0xff_u8);
    let mut alphabet = ['\0'; 256];
    let mut extra = 0_u32;
    for byte in 0_u8..=u8::MAX {
        alphabet[usize::from(byte)] = if direct.contains(&byte) {
            char::from(byte)
        } else {
            let value = char::from_u32(256 + extra).expect("a byte-alphabet scalar is valid");
            extra += 1;
            value
        };
    }
    alphabet
}

/// The generated tokenizer definition, as the bytes this package ships.
#[must_use]
pub fn tp4_fixture_tokenizer_definition_v1() -> Vec<u8> {
    let alphabet = tp4_fixture_byte_alphabet_v1();
    let mut json = String::with_capacity(3 << 20);
    json.push_str("{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,");
    json.push_str("\"added_tokens\":[");
    for (index, added) in tp4_fixture_added_tokens_v1().into_iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        write!(
            json,
            "{{\"id\":{},\"content\":{},\"single_word\":false,\"lstrip\":false,\
             \"rstrip\":false,\"normalized\":true,\"special\":{}}}",
            added.id,
            quoted_v1(&added.content),
            added.special
        )
        .expect("writing to a string cannot fail");
    }
    json.push_str("],\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":[]},");
    write!(
        json,
        "\"pre_tokenizer\":{{\"type\":\"Sequence\",\"pretokenizers\":[\
         {{\"type\":\"Split\",\"pattern\":{{\"Regex\":{}}},\"behavior\":\"Isolated\",\
         \"invert\":false}},\
         {{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\
         \"use_regex\":false}}]}},",
        quoted_v1(TP4_PRETOKENIZER_PATTERN)
    )
    .expect("writing to a string cannot fail");
    json.push_str(
        "\"post_processor\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\
         \"trim_offsets\":false,\"use_regex\":true},",
    );
    json.push_str(
        "\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\
         \"trim_offsets\":true,\"use_regex\":true},",
    );
    // `ignore_merges` is absent on purpose: the declared policy configures no
    // value for it, and the reader refuses a definition that carries one.
    json.push_str(
        "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\
         \"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,\
         \"fuse_unk\":false,\"byte_fallback\":false,\"vocab\":{",
    );
    for id in 0..TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1 {
        if id > 0 {
            json.push(',');
        }
        write!(json, "\"{}{id}\":{id}", TP4_FIXTURE_BASE_TOKEN_PREFIX_V1)
            .expect("writing to a string cannot fail");
    }
    for byte in 0..=u8::MAX {
        let id = TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1 + u32::from(byte);
        write!(
            json,
            ",{}:{id}",
            quoted_v1(&alphabet[usize::from(byte)].to_string())
        )
        .expect("writing to a string cannot fail");
    }
    json.push_str("},\"merges\":[]}}");
    json.into_bytes()
}

/// The generated tokenizer configuration, as the bytes this package ships.
#[must_use]
pub fn tp4_fixture_tokenizer_configuration_v1() -> Vec<u8> {
    let metadata = tokenizer_metadata();
    let added = |content: &str| {
        format!(
            "{{\"__type\":\"AddedToken\",\"content\":{},\"single_word\":false,\
             \"lstrip\":false,\"rstrip\":false,\"normalized\":true}}",
            quoted_v1(content)
        )
    };
    format!(
        "{{\"add_bos_token\":false,\"add_eos_token\":false,\
         \"clean_up_tokenization_spaces\":false,\"legacy\":true,\
         \"tokenizer_class\":\"{TP4_FIXTURE_TOKENIZER_CLASS_V1}\",\
         \"unk_token\":null,\
         \"model_max_length\":{TP4_FIXTURE_TOKENIZER_MODEL_MAX_LENGTH_V1},\
         \"bos_token\":{},\"eos_token\":{},\"pad_token\":{},\
         \"sp_model_kwargs\":{{}}}}",
        added(metadata.bos_prefix()[0]),
        added(metadata.eos_token()),
        added(metadata.pad_token()),
    )
    .into_bytes()
}

/// One JSON string literal, escaped for the characters this generator emits.
fn quoted_v1(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('"');
    for character in value.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            control if control < ' ' => {
                write!(quoted, "\\u{:04x}", control as u32)
                    .expect("writing to a string cannot fail");
            }
            other => quoted.push(other),
        }
    }
    quoted.push('"');
    quoted
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::metadata::{TP4_TEXT_PROMPT_SPECIAL_TOKENS, tokenizer_metadata};

    use crate::contract::tp4_fixture_vocabulary_v1;

    use super::{
        TP4_FIXTURE_BASE_VOCABULARY_V1, TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1,
        TP4_FIXTURE_OUTPUT_MARKERS_V1, tp4_fixture_added_token_count_v1,
        tp4_fixture_added_tokens_v1, tp4_fixture_byte_alphabet_v1, tp4_fixture_filler_token_v1,
        tp4_fixture_tokenizer_configuration_v1, tp4_fixture_tokenizer_definition_v1,
        tp4_fixture_tokenizer_revision_v1, tp4_published_markers_v1,
    };

    /// The added block starts at TP4's published stop token and covers every
    /// identifier up to the declared width, leaving none unassigned.
    #[test]
    fn the_added_block_assigns_every_identifier_above_the_base_vocabulary() {
        let metadata = tokenizer_metadata();
        assert_eq!(TP4_FIXTURE_BASE_VOCABULARY_V1, metadata.eos_token_id());
        assert_eq!(TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1, 154_564);
        assert_eq!(tp4_fixture_added_token_count_v1(), 60);

        let added = tp4_fixture_added_tokens_v1();
        assert_eq!(added.len(), tp4_fixture_added_token_count_v1());
        let mut contents = BTreeSet::new();
        for (offset, entry) in added.iter().enumerate() {
            assert_eq!(entry.id, TP4_FIXTURE_BASE_VOCABULARY_V1 + offset as u32);
            assert!(
                contents.insert(entry.content.as_str()),
                "added content {} is declared twice",
                entry.content
            );
        }
        assert_eq!(
            added.last().map(|entry| entry.id),
            Some(tp4_fixture_vocabulary_v1() - 1)
        );

        // Exactly the published markers but the output markers are special,
        // and each sits at its own published identifier.
        let special = added
            .iter()
            .filter(|entry| entry.special)
            .map(|entry| (entry.id, entry.content.as_str()))
            .collect::<Vec<_>>();
        let published = tp4_published_markers_v1();
        assert_eq!(
            special,
            published
                .iter()
                .copied()
                .filter(|(_, marker)| !TP4_FIXTURE_OUTPUT_MARKERS_V1.contains(marker))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            special.len(),
            TP4_TEXT_PROMPT_SPECIAL_TOKENS.len() + 1 - TP4_FIXTURE_OUTPUT_MARKERS_V1.len()
        );
        assert!(
            added
                .iter()
                .filter(|entry| !entry.special)
                .all(|entry| entry.content.starts_with("chop-fixture-filler-")
                    || (TP4_FIXTURE_OUTPUT_MARKERS_V1.contains(&entry.content.as_str())
                        && published.contains(&(entry.id, entry.content.as_str())))),
            "every non-special added token has to be an output marker or a recognisable fixture filler"
        );
    }

    /// The byte alphabet this generator emits is 256 distinct scalars.
    #[test]
    fn the_generated_byte_alphabet_is_a_bijection() {
        let alphabet = tp4_fixture_byte_alphabet_v1();

        let distinct = alphabet.iter().collect::<BTreeSet<_>>();

        assert_eq!(distinct.len(), 256);
        assert_eq!(alphabet[usize::from(b'!')], '!');
        assert_eq!(alphabet[0], '\u{100}');
    }

    #[test]
    fn the_pinned_revision_is_stable_and_hexadecimal() {
        let revision = tp4_fixture_tokenizer_revision_v1();
        assert_eq!(revision.len(), 40);
        assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(revision, tp4_fixture_tokenizer_revision_v1());
    }

    /// The artifacts this crate generates build a tokenizer through the shared
    /// reader, and every identifier the declaration pins decodes to its text.
    ///
    /// This is the whole point of generating them rather than shipping a
    /// stand-in: a deployment reads these bytes, and a deployment that cannot
    /// build a tokenizer from them is one no engine and no transport could be
    /// driven against.
    #[test]
    fn the_generated_artifacts_build_a_tokenizer_that_binds_every_declared_boundary() {
        let descriptor = runtime_frontend::FrontendDescriptorV1::from_component_bytes(
            &crate::frontend::tp4_fixture_frontend_declaration_v1(),
        )
        .expect("the served declaration parses and validates");

        let tokenizer = runtime_frontend::tokenizer_from_components_v1(
            &descriptor,
            &tp4_fixture_tokenizer_definition_v1(),
            &tp4_fixture_tokenizer_configuration_v1(),
        )
        .expect("the generated artifacts build a tokenizer and bind their boundaries");

        // Ordinary text is one token per byte here, and it round-trips exactly.
        let encoded = tokenizer
            .encode("hello fixture", &[])
            .expect("ordinary text encodes");
        assert_eq!(encoded.len(), "hello fixture".len());
        assert!(
            encoded.iter().all(|id| (TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1
                ..TP4_FIXTURE_BASE_VOCABULARY_V1)
                .contains(id)),
            "with no merges every ordinary token is a byte token"
        );
        assert_eq!(
            tokenizer.decode_utf8(&encoded).expect("the tokens decode"),
            "hello fixture"
        );

        // Every published marker decodes to its published text and is special
        // unless the parser reads it from a completion.
        for (id, marker) in tp4_published_markers_v1() {
            assert_eq!(
                tokenizer.decode_bytes(&[id]).expect("a marker decodes"),
                marker.as_bytes(),
                "identifier {id} must decode to {marker}"
            );
            assert_eq!(
                tokenizer.token_is_special(id).expect("the flag reads"),
                !TP4_FIXTURE_OUTPUT_MARKERS_V1.contains(&marker),
                "{marker} is special exactly when it is not an output marker"
            );
        }

        // The peak the synthetic row lands on is not one of them: it decodes to
        // visible text a completion can carry.
        let peak = crate::synthetic::TP4_FIXTURE_PEAK_TOKEN_ID_V1;
        assert!(!tokenizer.token_is_special(peak).expect("the flag reads"));
        assert_eq!(
            tokenizer.decode_utf8(&[peak]).expect("the peak decodes"),
            tp4_fixture_filler_token_v1(peak)
        );
    }

    #[test]
    fn both_generated_artifacts_are_deterministic() {
        assert_eq!(
            tp4_fixture_tokenizer_definition_v1(),
            tp4_fixture_tokenizer_definition_v1()
        );
        assert_eq!(
            tp4_fixture_tokenizer_configuration_v1(),
            tp4_fixture_tokenizer_configuration_v1()
        );
    }
}
