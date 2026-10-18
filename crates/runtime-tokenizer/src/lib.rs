#![forbid(unsafe_code)]

mod json;

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::error::Error;
use std::fmt;
use std::fs::{self, Metadata, OpenOptions};
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Component, Path};

use fancy_regex::{Expr, Regex};
use json::Value;
use runtime_weights::sha256_bytes;
use unicode_normalization::UnicodeNormalization;

const MAX_TOKENIZER_FILE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_CONFIG_FILE_BYTES: u64 = 64 * 1024;
const MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONFIGURABLE_INPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_VOCABULARY_SIZE: u32 = 1_000_000;
const MAX_MERGE_COUNT: usize = 1_000_000;
const MAX_TOKEN_BYTES: usize = 16 * 1024;
const MAX_SPECIAL_TOKEN_BYTES: usize = 512;

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerError {
    message: String,
    disallowed_special_token: Option<String>,
}

impl TokenizerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            disallowed_special_token: None,
        }
    }

    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self::new(message)
    }

    /// The special token refused by the caller's prompt policy, if any.
    #[must_use]
    pub fn disallowed_special_token(&self) -> Option<&str> {
        self.disallowed_special_token.as_deref()
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for TokenizerError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileIdentity {
    name: String,
    size: u64,
    sha256: String,
}

impl FileIdentity {
    pub fn new(name: &str, size: u64, sha256: &str) -> Result<Self> {
        let path = Path::new(name);
        if name.is_empty()
            || name.contains(['/', '\\'])
            || path.components().count() != 1
            || !matches!(path.components().next(), Some(Component::Normal(_)))
            || path.file_name().and_then(|value| value.to_str()) != Some(name)
        {
            return Err(TokenizerError::new(
                "tokenizer artifact name must be one file name",
            ));
        }
        if size == 0 {
            return Err(TokenizerError::new(
                "tokenizer artifact size must be nonzero",
            ));
        }
        if sha256.len() != 64
            || !sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(TokenizerError::new(
                "tokenizer artifact SHA-256 must be 64 lowercase hexadecimal characters",
            ));
        }
        Ok(Self {
            name: name.to_owned(),
            size,
            sha256: sha256.to_owned(),
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerContract {
    repository: String,
    revision: String,
    definition: FileIdentity,
    configuration: FileIdentity,
    base_vocabulary_size: u32,
    vocabulary_size: u32,
    merge_count: usize,
    added_token_count: usize,
    model_max_length: usize,
    dialect: TokenizerDialect,
    bos_token: Option<String>,
    eos_token: String,
    pad_token: String,
    pretokenizer_patterns: Vec<String>,
    prompt_special_tokens: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizerPolicy {
    IdentitySequence,
    Nfc,
    Null,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteLevelPolicy {
    pub add_prefix_space: bool,
    pub trim_offsets: bool,
    pub use_regex: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeResultIdPolicy {
    Sequential { first_id: u32 },
    Vocabulary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeRecordPolicy {
    SpaceSeparatedString,
    StringPairArray,
    StringOrPair,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IgnoreMergesPolicy {
    Absent,
    Disabled,
    WholePiece,
}

impl IgnoreMergesPolicy {
    const fn artifact_value(self) -> Option<bool> {
        match self {
            Self::Absent => None,
            Self::Disabled => Some(false),
            Self::WholePiece => Some(true),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BaseTokenRepresentation {
    LiteralPrefixThenByteLevel { byte_start_id: u32 },
    ByteLevel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigTokenRepresentation {
    AddedToken { normalized: bool },
    String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigFieldPolicy<T> {
    Required(T),
    RequiredNull,
    Absent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigTokenValue {
    ContractValue,
    Literal(String),
}

/// The component policies a tokenizer's artifacts are checked and read with.
///
/// Every member is declared by the bundle that ships the artifacts, so this
/// crate names no tokenizer family: it checks each component against what
/// the declaration says and reads the artifacts that way.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerDialect {
    pub normalizer: NormalizerPolicy,
    pub pretokenizer_byte_level: ByteLevelPolicy,
    pub post_processor_byte_level: ByteLevelPolicy,
    pub decoder_byte_level: ByteLevelPolicy,
    pub merge_results: MergeResultIdPolicy,
    pub merge_records: MergeRecordPolicy,
    pub base_tokens: BaseTokenRepresentation,
    pub config_tokens: ConfigTokenRepresentation,
    pub add_bos_token: ConfigFieldPolicy<bool>,
    pub bos_token: ConfigFieldPolicy<ConfigTokenValue>,
    pub unk_token: ConfigFieldPolicy<ConfigTokenValue>,
    pub tokenizer_class: String,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    pub ignore_merges: IgnoreMergesPolicy,
    pub add_eos_token: Option<bool>,
    pub legacy: Option<bool>,
    pub empty_sentencepiece_kwargs: bool,
}

impl TokenizerContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_dialect(
        repository: &str,
        revision: &str,
        definition: FileIdentity,
        configuration: FileIdentity,
        base_vocabulary_size: u32,
        vocabulary_size: u32,
        merge_count: usize,
        added_token_count: usize,
        model_max_length: usize,
        bos_token: Option<&str>,
        eos_token: &str,
        pad_token: &str,
        pretokenizer_patterns: &[&str],
        prompt_special_tokens: &[&str],
        dialect: TokenizerDialect,
    ) -> Result<Self> {
        if repository.is_empty()
            || !matches!(revision.len(), 40 | 64)
            || !revision.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(TokenizerError::new(
                "tokenizer repository or revision is invalid",
            ));
        }
        if definition.size > MAX_TOKENIZER_FILE_BYTES {
            return Err(TokenizerError::new(
                "tokenizer definition exceeds the file-size limit",
            ));
        }
        if configuration.size > MAX_CONFIG_FILE_BYTES {
            return Err(TokenizerError::new(
                "tokenizer configuration exceeds the file-size limit",
            ));
        }
        if base_vocabulary_size == 0
            || vocabulary_size < base_vocabulary_size
            || vocabulary_size > MAX_VOCABULARY_SIZE
            || model_max_length == 0
            || model_max_length > MAX_INPUT_BYTES
        {
            return Err(TokenizerError::new(
                "tokenizer vocabulary or input limit is invalid",
            ));
        }
        if pretokenizer_patterns.is_empty() || prompt_special_tokens.is_empty() {
            return Err(TokenizerError::new(
                "tokenizer policy lists cannot be empty",
            ));
        }
        if merge_count > MAX_MERGE_COUNT {
            return Err(TokenizerError::new(
                "tokenizer merge count exceeds the merge-count resource limit",
            ));
        }
        let merge_count_u32 = u32::try_from(merge_count)
            .map_err(|_| TokenizerError::new("tokenizer merge count exceeds u32"))?;
        let merge_policy_valid = match dialect.merge_results {
            MergeResultIdPolicy::Sequential { first_id } => first_id
                .checked_add(merge_count_u32)
                .is_some_and(|end| end == base_vocabulary_size),
            MergeResultIdPolicy::Vocabulary => true,
        };
        if !merge_policy_valid || added_token_count > MAX_VOCABULARY_SIZE as usize {
            return Err(TokenizerError::new(
                "tokenizer merge or added-token count is invalid",
            ));
        }
        if let BaseTokenRepresentation::LiteralPrefixThenByteLevel { byte_start_id } =
            dialect.base_tokens
            && byte_start_id
                .checked_add(256)
                .is_none_or(|end| end > base_vocabulary_size)
        {
            return Err(TokenizerError::new(
                "tokenizer byte tokens do not fit in the base vocabulary",
            ));
        }
        supported_dialect(&dialect)?;
        Ok(Self {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
            definition,
            configuration,
            base_vocabulary_size,
            vocabulary_size,
            merge_count,
            added_token_count,
            model_max_length,
            dialect,
            bos_token: bos_token.map(str::to_owned),
            eos_token: eos_token.to_owned(),
            pad_token: pad_token.to_owned(),
            pretokenizer_patterns: pretokenizer_patterns
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            prompt_special_tokens: prompt_special_tokens
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
        })
    }

    #[must_use]
    pub fn repository(&self) -> &str {
        &self.repository
    }
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }
    #[must_use]
    pub const fn definition(&self) -> &FileIdentity {
        &self.definition
    }
    #[must_use]
    pub const fn configuration(&self) -> &FileIdentity {
        &self.configuration
    }
    #[must_use]
    pub const fn base_vocabulary_size(&self) -> u32 {
        self.base_vocabulary_size
    }
    #[must_use]
    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }
    #[must_use]
    pub const fn merge_count(&self) -> usize {
        self.merge_count
    }
    #[must_use]
    pub const fn added_token_count(&self) -> usize {
        self.added_token_count
    }
    #[must_use]
    pub const fn model_max_length(&self) -> usize {
        self.model_max_length
    }
    #[must_use]
    pub fn bos_token(&self) -> Option<&str> {
        self.bos_token.as_deref()
    }
    #[must_use]
    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }
    #[must_use]
    pub fn pad_token(&self) -> &str {
        &self.pad_token
    }
    #[must_use]
    pub const fn dialect(&self) -> &TokenizerDialect {
        &self.dialect
    }
    #[must_use]
    pub fn prompt_special_tokens(&self) -> Vec<&str> {
        self.prompt_special_tokens
            .iter()
            .map(String::as_str)
            .collect()
    }
}

#[derive(Clone, Debug)]
struct Merge {
    rank: u32,
    result: u32,
}

#[derive(Clone, Debug, Default)]
struct TrieNode {
    children: BTreeMap<u8, usize>,
    terminal: Option<(u32, String)>,
}

/// The alternatives that end the byte-level split patterns. Together they
/// split a whitespace run through its last line break; without one, before
/// its last character when text follows; otherwise whole. The lookahead
/// backtracks once per character, so a long run exceeds the regex stack.
const WHITESPACE_TAIL: [&str; 3] = [r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+"];

/// A pattern whose top-level alternation ends with `WHITESPACE_TAIL` runs
/// with that tail replaced by one capturing `\s+`, and a run that group
/// matched is cut where the tail would have ended it.
#[derive(Debug)]
struct PretokenizerPattern {
    regex: Regex,
    whitespace_group: Option<usize>,
}

impl PretokenizerPattern {
    fn new(pattern: &str) -> Result<Self> {
        let compile = |pattern: &str| {
            Regex::new(pattern).map_err(|error| {
                TokenizerError::new(format!("invalid pretokenizer regex: {error}"))
            })
        };
        if let Some(rewritten) = whitespace_tail_rewrite(pattern) {
            let regex = compile(&rewritten)?;
            let whitespace_group = regex.captures_len() - 1;
            return Ok(Self {
                regex,
                whitespace_group: Some(whitespace_group),
            });
        }
        Ok(Self {
            regex: compile(pattern)?,
            whitespace_group: None,
        })
    }

    /// Whether the whitespace group matched `matched`, which the search from
    /// `search` found. The probe repeats that search: from another position
    /// an anchor such as `\G` or a reset such as `\K` could select another
    /// alternative.
    fn whitespace_group_matched(
        &self,
        piece: &str,
        search: usize,
        matched: &fancy_regex::Match<'_>,
        group: usize,
    ) -> Result<bool> {
        let captures = self
            .regex
            .captures_from_pos(piece, search)
            .map_err(|error| TokenizerError::new(format!("pretokenizer regex failed: {error}")))?;
        match captures {
            Some(captures)
                if captures.get(0).is_some_and(|whole| {
                    (whole.start(), whole.end()) == (matched.start(), matched.end())
                }) =>
            {
                Ok(captures.get(group).is_some())
            }
            _ => Err(TokenizerError::new(
                "pretokenizer regex found another match when probed",
            )),
        }
    }

    /// Appends the pieces of `piece` an `Isolated` split gives: the matches
    /// and the text between them.
    fn split<'a>(&self, piece: &'a str, split: &mut Vec<&'a str>) -> Result<()> {
        let failed = |error| TokenizerError::new(format!("pretokenizer regex failed: {error}"));
        let mut end = 0;
        let Some(group) = self.whitespace_group else {
            for matched in self.regex.find_iter(piece) {
                let matched = matched.map_err(failed)?;
                if matched.start() > end {
                    split.push(&piece[end..matched.start()]);
                }
                if matched.end() > matched.start() {
                    split.push(matched.as_str());
                }
                end = matched.end();
            }
            if end < piece.len() {
                split.push(&piece[end..]);
            }
            return Ok(());
        };
        // As `find_iter`, but each search resumes where the cut match ends.
        let mut search = 0;
        let mut last_match = None;
        while search <= piece.len() {
            let Some(matched) = self.regex.find_from_pos(piece, search).map_err(failed)? else {
                break;
            };
            let start = matched.start();
            let mut stop = matched.end();
            if start == stop {
                search = piece[stop..]
                    .chars()
                    .next()
                    .map_or(piece.len() + 1, |next| stop + next.len_utf8());
                if last_match == Some(stop) {
                    continue;
                }
            } else {
                let text = matched.as_str();
                if text.chars().all(char::is_whitespace)
                    && self.whitespace_group_matched(piece, search, &matched, group)?
                {
                    if let Some(line_break) = text.rfind(['\r', '\n']) {
                        stop = start + line_break + 1;
                    } else if stop < piece.len() {
                        let last = text.chars().next_back().expect("nonempty match");
                        if text.len() > last.len_utf8() {
                            stop -= last.len_utf8();
                        }
                    }
                }
                search = stop;
            }
            last_match = Some(stop);
            if start > end {
                split.push(&piece[end..start]);
            }
            if stop > start {
                split.push(&piece[start..stop]);
            }
            end = stop;
        }
        if end < piece.len() {
            split.push(&piece[end..]);
        }
        Ok(())
    }
}

/// `pattern` with its trailing `WHITESPACE_TAIL` alternatives replaced by
/// `(\s+)`, when its parsed top-level alternation ends with them, however
/// they are spelled (a class may hold literal line breaks), and the rewrite
/// parses to the same earlier alternatives.
fn whitespace_tail_rewrite(pattern: &str) -> Option<String> {
    let alternatives = |pattern: &str| match Expr::parse_tree(pattern).ok()?.expr {
        Expr::Alt(alternatives) => Some(alternatives),
        _ => None,
    };
    let original = alternatives(pattern)?;
    let kept = original.len().checked_sub(WHITESPACE_TAIL.len())?;
    if kept == 0 {
        return None;
    }
    for (alternative, expected) in original[kept..].iter().zip(WHITESPACE_TAIL) {
        if *alternative != Expr::parse_tree(expected).ok()?.expr {
            return None;
        }
    }
    let group = Expr::parse_tree(r"(\s+)").ok()?.expr;
    pattern
        .rmatch_indices('|')
        .map(|(index, _)| format!(r"{}|(\s+)", &pattern[..index]))
        .find(|rewritten| {
            alternatives(rewritten).is_some_and(|parsed| {
                parsed.len() == kept + 1
                    && parsed[..kept] == original[..kept]
                    && parsed[kept] == group
            })
        })
}

#[derive(Debug)]
pub struct Tokenizer {
    contract: TokenizerContract,
    patterns: Vec<PretokenizerPattern>,
    byte_alphabet: [char; 256],
    byte_ids: [u32; 256],
    merges: HashMap<(u32, u32), Merge>,
    whole_piece_ids: Option<HashMap<String, u32>>,
    decoded: Vec<Vec<u8>>,
    token_strings: Vec<String>,
    added_by_text: BTreeMap<String, u32>,
    special_ids: BTreeSet<u32>,
    trie: Vec<TrieNode>,
}

impl Tokenizer {
    pub fn load(root: &Path, contract: &TokenizerContract) -> Result<Self> {
        let root_metadata = fs::symlink_metadata(root).map_err(|error| {
            TokenizerError::new(format!(
                "cannot inspect tokenizer root {}: {error}",
                root.display()
            ))
        })?;
        if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
            return Err(TokenizerError::new(format!(
                "tokenizer root {} must be a real directory, not a symlink",
                root.display()
            )));
        }
        let definition = read_verified(root, &contract.definition, MAX_TOKENIZER_FILE_BYTES)?;
        let configuration = read_verified(root, &contract.configuration, MAX_CONFIG_FILE_BYTES)?;
        let final_root_metadata = fs::symlink_metadata(root).map_err(|error| {
            TokenizerError::new(format!(
                "cannot re-inspect tokenizer root {}: {error}",
                root.display()
            ))
        })?;
        if !final_root_metadata.is_dir() || !same_file(&root_metadata, &final_root_metadata) {
            return Err(TokenizerError::new(format!(
                "tokenizer root {} changed while artifacts were read",
                root.display()
            )));
        }
        Self::from_verified_bytes(&definition, &configuration, contract)
    }

    pub fn from_verified_bytes(
        definition: &[u8],
        configuration: &[u8],
        contract: &TokenizerContract,
    ) -> Result<Self> {
        verify_bytes(definition, &contract.definition, MAX_TOKENIZER_FILE_BYTES)?;
        verify_bytes(
            configuration,
            &contract.configuration,
            MAX_CONFIG_FILE_BYTES,
        )?;
        validate_configuration(configuration, contract)?;
        parse_definition(definition, contract)
    }

    pub fn encode(&self, text: &str, allowed_special_tokens: &[&str]) -> Result<Vec<u32>> {
        self.encode_with_input_limit(text, allowed_special_tokens, MAX_INPUT_BYTES)
    }

    /// Encodes text under a caller-supplied byte limit, up to 64 MiB.
    pub fn encode_with_input_limit(
        &self,
        text: &str,
        allowed_special_tokens: &[&str],
        max_input_bytes: usize,
    ) -> Result<Vec<u32>> {
        if max_input_bytes == 0 || max_input_bytes > MAX_CONFIGURABLE_INPUT_BYTES {
            return Err(TokenizerError::new("tokenizer input byte limit is invalid"));
        }
        if text.len() > max_input_bytes {
            return Err(TokenizerError::new(
                "tokenizer input exceeds the byte limit",
            ));
        }
        let mut allowed = BTreeSet::new();
        for token in allowed_special_tokens {
            if !self.added_by_text.contains_key(*token) {
                return Err(TokenizerError::new(format!(
                    "unknown allowed special token '{token}'"
                )));
            }
            allowed.insert(*token);
        }
        let mut output = Vec::new();
        output
            .try_reserve(text.len().min(self.contract.model_max_length))
            .map_err(|_| TokenizerError::new("token output allocation failed"))?;
        let bytes = text.as_bytes();
        let mut ordinary_start = 0;
        let mut offset = 0;
        while offset < bytes.len() {
            if let Some((length, id, token)) = self.special_at(bytes, offset) {
                self.encode_ordinary(&text[ordinary_start..offset], &mut output, max_input_bytes)?;
                if !allowed.contains(token.as_str()) {
                    return Err(TokenizerError {
                        message: format!("disallowed special token '{token}' in input"),
                        disallowed_special_token: Some(token),
                    });
                }
                output.push(id);
                offset = offset
                    .checked_add(length)
                    .ok_or_else(|| TokenizerError::new("special-token offset overflow"))?;
                ordinary_start = offset;
            } else {
                offset += 1;
            }
        }
        self.encode_ordinary(&text[ordinary_start..], &mut output, max_input_bytes)?;
        if output.len() > self.contract.model_max_length {
            return Err(TokenizerError::new("encoded input exceeds the token limit"));
        }
        Ok(output)
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        if ids.len() > self.contract.model_max_length {
            return Err(TokenizerError::new("decode input exceeds the token limit"));
        }
        let mut size = 0_usize;
        for id in ids {
            let bytes = self
                .decoded
                .get(
                    usize::try_from(*id)
                        .map_err(|_| TokenizerError::new("token ID conversion overflow"))?,
                )
                .ok_or_else(|| {
                    TokenizerError::new(format!("token ID {id} is outside the fixed vocabulary"))
                })?;
            size = size
                .checked_add(bytes.len())
                .ok_or_else(|| TokenizerError::new("decoded byte length overflow"))?;
            if size > MAX_INPUT_BYTES {
                return Err(TokenizerError::new("decoded output exceeds the byte limit"));
            }
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(size)
            .map_err(|_| TokenizerError::new("decode output allocation failed"))?;
        for id in ids {
            output.extend_from_slice(&self.decoded[*id as usize]);
        }
        Ok(output)
    }

    /// Returns the exact decoded bytes for one token without allocating.
    ///
    /// This method validates the fixed-vocabulary bound but does not hide or
    /// reinterpret special tokens. Call [`Self::token_is_special`] when the
    /// distinction is relevant to the caller's grammar.
    pub fn token_bytes(&self, id: u32) -> Result<&[u8]> {
        let index =
            usize::try_from(id).map_err(|_| TokenizerError::new("token ID conversion overflow"))?;
        self.decoded.get(index).map(Vec::as_slice).ok_or_else(|| {
            TokenizerError::new(format!("token ID {id} is outside the fixed vocabulary"))
        })
    }

    pub fn decode_utf8(&self, ids: &[u32]) -> Result<String> {
        String::from_utf8(self.decode_bytes(ids)?).map_err(|error| {
            TokenizerError::new(format!("decoded tokens are not valid UTF-8: {error}"))
        })
    }

    pub fn token_is_special(&self, id: u32) -> Result<bool> {
        if id >= self.contract.vocabulary_size {
            return Err(TokenizerError::new(format!(
                "token ID {id} is outside the fixed vocabulary"
            )));
        }
        Ok(self.special_ids.contains(&id))
    }

    /// Returns the authenticated vocabulary spelling, before byte decoding.
    pub fn token_string(&self, id: u32) -> Result<&str> {
        self.token_strings
            .get(id as usize)
            .map(String::as_str)
            .ok_or_else(|| TokenizerError::new("token ID is outside the vocabulary"))
    }

    fn encode_ordinary(
        &self,
        text: &str,
        output: &mut Vec<u32>,
        max_input_bytes: usize,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let normalized;
        let text = match self.contract.dialect.normalizer {
            NormalizerPolicy::IdentitySequence | NormalizerPolicy::Null => text,
            NormalizerPolicy::Nfc => {
                normalized = text.nfc().collect::<String>();
                if normalized.len() > max_input_bytes {
                    return Err(TokenizerError::new(
                        "normalized tokenizer input exceeds the byte limit",
                    ));
                }
                &normalized
            }
        };
        let mut pieces = vec![text];
        for pattern in &self.patterns {
            let mut split = Vec::new();
            for piece in pieces {
                split
                    .try_reserve(piece.len())
                    .map_err(|_| TokenizerError::new("pretokenizer allocation failed"))?;
                pattern.split(piece, &mut split)?;
            }
            pieces = split;
        }
        for piece in pieces {
            self.encode_piece(piece.as_bytes(), output)?;
        }
        Ok(())
    }

    fn encode_piece(&self, bytes: &[u8], output: &mut Vec<u32>) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if let Some(whole_piece_ids) = &self.whole_piece_ids {
            let capacity = bytes
                .len()
                .checked_mul(2)
                .ok_or_else(|| TokenizerError::new("whole-piece token size overflow"))?;
            let mut encoded = String::new();
            encoded
                .try_reserve_exact(capacity)
                .map_err(|_| TokenizerError::new("whole-piece token allocation failed"))?;
            encoded.extend(
                bytes
                    .iter()
                    .map(|byte| self.byte_alphabet[usize::from(*byte)]),
            );
            if let Some(id) = whole_piece_ids.get(&encoded) {
                output.push(*id);
                return Ok(());
            }
        }
        let count = bytes.len();
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| TokenizerError::new("BPE token allocation failed"))?;
        ids.extend(bytes.iter().map(|byte| self.byte_ids[usize::from(*byte)]));
        let mut next = Vec::new();
        next.try_reserve_exact(count)
            .map_err(|_| TokenizerError::new("BPE link allocation failed"))?;
        next.extend((0..count).map(|index| (index + 1 < count).then_some(index + 1)));
        let mut previous = Vec::new();
        previous
            .try_reserve_exact(count)
            .map_err(|_| TokenizerError::new("BPE reverse-link allocation failed"))?;
        previous.extend((0..count).map(|index| index.checked_sub(1)));
        let mut live = Vec::new();
        live.try_reserve_exact(count)
            .map_err(|_| TokenizerError::new("BPE liveness allocation failed"))?;
        live.resize(count, true);
        let mut heap: BinaryHeap<Reverse<(u32, usize, u32, u32)>> = BinaryHeap::new();
        heap.try_reserve(
            count
                .checked_mul(3)
                .ok_or_else(|| TokenizerError::new("BPE work queue size overflow"))?,
        )
        .map_err(|_| TokenizerError::new("BPE work queue allocation failed"))?;
        for index in 0..count.saturating_sub(1) {
            self.push_pair(&ids, &next, index, &mut heap);
        }
        while let Some(Reverse((_, left, expected_left, expected_right))) = heap.pop() {
            if !live[left] || ids[left] != expected_left {
                continue;
            }
            let Some(right) = next[left] else { continue };
            if !live[right] || ids[right] != expected_right {
                continue;
            }
            let Some(merge) = self.merges.get(&(expected_left, expected_right)) else {
                continue;
            };
            ids[left] = merge.result;
            live[right] = false;
            next[left] = next[right];
            if let Some(after) = next[right] {
                previous[after] = Some(left);
            }
            if let Some(before) = previous[left] {
                self.push_pair(&ids, &next, before, &mut heap);
            }
            self.push_pair(&ids, &next, left, &mut heap);
        }
        let mut current = Some(0);
        while let Some(index) = current {
            if live[index] {
                output.push(ids[index]);
            }
            current = next[index];
        }
        Ok(())
    }

    fn push_pair(
        &self,
        ids: &[u32],
        next: &[Option<usize>],
        left: usize,
        heap: &mut BinaryHeap<Reverse<(u32, usize, u32, u32)>>,
    ) {
        if let Some(right) = next[left]
            && let Some(merge) = self.merges.get(&(ids[left], ids[right]))
        {
            heap.push(Reverse((merge.rank, left, ids[left], ids[right])));
        }
    }

    fn special_at(&self, bytes: &[u8], offset: usize) -> Option<(usize, u32, String)> {
        let mut node = 0;
        let mut best = None;
        for (relative, byte) in bytes[offset..].iter().enumerate() {
            let Some(next) = self.trie[node].children.get(byte) else {
                break;
            };
            node = *next;
            if let Some((id, token)) = &self.trie[node].terminal {
                best = Some((relative + 1, *id, token.clone()));
            }
        }
        best
    }
}

fn read_verified(root: &Path, identity: &FileIdentity, limit: u64) -> Result<Vec<u8>> {
    read_verified_with_open_hook(root, identity, limit, || {})
}

fn read_verified_with_open_hook(
    root: &Path,
    identity: &FileIdentity,
    limit: u64,
    after_open: impl FnOnce(),
) -> Result<Vec<u8>> {
    let path = root.join(&identity.name);
    let path_metadata = fs::symlink_metadata(&path).map_err(|error| {
        TokenizerError::new(format!("cannot inspect {}: {error}", path.display()))
    })?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(TokenizerError::new(format!(
            "tokenizer artifact {} must be a regular non-symlink file",
            path.display()
        )));
    }
    if path_metadata.len() != identity.size || path_metadata.len() > limit {
        return Err(TokenizerError::new(format!(
            "tokenizer artifact {} has mismatched or excessive size",
            identity.name
        )));
    }

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(no_follow_flag());
    #[cfg(windows)]
    options.custom_flags(0x0020_0000);
    let mut file = options.open(&path).map_err(|error| {
        TokenizerError::new(format!(
            "cannot open {} without following links: {error}",
            path.display()
        ))
    })?;
    let opened_metadata = file.metadata().map_err(|error| {
        TokenizerError::new(format!("cannot inspect opened {}: {error}", path.display()))
    })?;
    if !opened_metadata.is_file()
        || !same_file(&path_metadata, &opened_metadata)
        || opened_metadata.len() != identity.size
        || opened_metadata.len() > limit
    {
        return Err(TokenizerError::new(format!(
            "tokenizer artifact {} changed while it was opened",
            identity.name
        )));
    }
    after_open();

    let capacity = usize::try_from(opened_metadata.len())
        .map_err(|_| TokenizerError::new("tokenizer artifact size does not fit memory"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| TokenizerError::new("tokenizer artifact allocation failed"))?;
    bytes.resize(capacity, 0);
    file.read_exact(&mut bytes).map_err(|error| {
        TokenizerError::new(format!("cannot read exact {}: {error}", path.display()))
    })?;
    let mut trailing = [0_u8; 1];
    if file.read(&mut trailing).map_err(|error| {
        TokenizerError::new(format!("cannot verify EOF for {}: {error}", path.display()))
    })? != 0
    {
        return Err(TokenizerError::new(format!(
            "tokenizer artifact {} grew while it was read",
            identity.name
        )));
    }

    let final_opened_metadata = file.metadata().map_err(|error| {
        TokenizerError::new(format!(
            "cannot re-inspect opened {}: {error}",
            path.display()
        ))
    })?;
    let final_path_metadata = fs::symlink_metadata(&path).map_err(|error| {
        TokenizerError::new(format!("cannot re-inspect {}: {error}", path.display()))
    })?;
    if !final_opened_metadata.is_file()
        || !same_file(&opened_metadata, &final_opened_metadata)
        || final_opened_metadata.len() != identity.size
        || !final_path_metadata.is_file()
        || !same_file(&opened_metadata, &final_path_metadata)
        || final_path_metadata.len() != identity.size
    {
        return Err(TokenizerError::new(format!(
            "tokenizer artifact {} changed while it was read",
            identity.name
        )));
    }
    if sha256_bytes(&bytes) != identity.sha256 {
        return Err(TokenizerError::new(format!(
            "tokenizer artifact {} SHA-256 mismatch",
            identity.name
        )));
    }
    Ok(bytes)
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
const fn no_follow_flag() -> i32 {
    0x100
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "solaris"))]
const fn no_follow_flag() -> i32 {
    0x2_0000
}

#[cfg(target_os = "netbsd")]
const fn no_follow_flag() -> i32 {
    0x40_0000
}

#[cfg(unix)]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
        && left.created().ok() == right.created().ok()
}

fn verify_bytes(bytes: &[u8], identity: &FileIdentity, limit: u64) -> Result<()> {
    let length =
        u64::try_from(bytes.len()).map_err(|_| TokenizerError::new("artifact length overflow"))?;
    if length != identity.size || length > limit {
        return Err(TokenizerError::new(format!(
            "tokenizer artifact {} has mismatched or excessive size",
            identity.name
        )));
    }
    if sha256_bytes(bytes) != identity.sha256 {
        return Err(TokenizerError::new(format!(
            "tokenizer artifact {} SHA-256 mismatch",
            identity.name
        )));
    }
    Ok(())
}

/// Refuses a policy whose encoding or decoding this reader does not implement.
///
/// Members that only check an artifact field may take any value: the reader
/// compares the field and tokenizes the same way. Members that change token
/// ids or decoded bytes are accepted only with the values the encoder and
/// decoder below implement, so a declaration cannot select semantics that
/// would be checked and then silently not applied.
fn supported_dialect(dialect: &TokenizerDialect) -> Result<()> {
    let unsupported = |what: &str| {
        Err(TokenizerError::new(format!(
            "unsupported tokenizer policy: {what}"
        )))
    };
    // Pieces come from the declared split patterns alone and map byte for byte
    // onto the byte alphabet. Nothing prepends a space or applies the
    // byte-level regex split.
    if dialect.pretokenizer_byte_level.add_prefix_space {
        return unsupported("the pretokenizer does not add a prefix space");
    }
    if dialect.pretokenizer_byte_level.use_regex {
        return unsupported("the pretokenizer does not apply the byte-level regex");
    }
    // Merges join spellings as they are; no piece is marked as continuing a
    // word or ending one.
    if dialect
        .continuing_subword_prefix
        .as_deref()
        .is_some_and(|prefix| !prefix.is_empty())
    {
        return unsupported("a continuing-subword prefix is not applied");
    }
    if dialect
        .end_of_word_suffix
        .as_deref()
        .is_some_and(|suffix| !suffix.is_empty())
    {
        return unsupported("an end-of-word suffix is not applied");
    }
    // A whole-piece hit is looked up by its byte-alphabet spelling. A literal
    // entry with that spelling decodes as its own text, not as the bytes that
    // were encoded.
    if matches!(
        dialect.base_tokens,
        BaseTokenRepresentation::LiteralPrefixThenByteLevel { .. }
    ) && matches!(dialect.ignore_merges, IgnoreMergesPolicy::WholePiece)
    {
        return unsupported(
            "whole-piece lookup cannot be combined with literal-prefix base tokens",
        );
    }
    Ok(())
}

fn parse_definition(bytes: &[u8], contract: &TokenizerContract) -> Result<Tokenizer> {
    let root_value = json::parse(bytes)?;
    let root = root_value.object("tokenizer definition")?;
    expect_string(root, "version", "1.0")?;
    expect_null(root, "truncation")?;
    expect_null(root, "padding")?;
    validate_normalizer(required(root, "normalizer")?, contract.dialect.normalizer)?;
    let patterns = validate_pretokenizer(required(root, "pre_tokenizer")?, contract)?;
    validate_byte_level(
        required(root, "post_processor")?,
        contract.dialect.post_processor_byte_level,
        "post processor",
    )?;
    validate_byte_level(
        required(root, "decoder")?,
        contract.dialect.decoder_byte_level,
        "decoder",
    )?;
    let mut parsed = parse_model(required(root, "model")?, contract)?;
    parse_added_tokens(required(root, "added_tokens")?, contract, &mut parsed)?;
    let mut token_strings = vec![String::new(); contract.vocabulary_size as usize];
    for (text, id) in parsed
        .vocab_by_encoded
        .iter()
        .chain(parsed.added_by_text.iter())
    {
        token_strings[*id as usize] = text.clone();
    }
    let whole_piece_ids = matches!(
        contract.dialect.ignore_merges,
        IgnoreMergesPolicy::WholePiece
    )
    .then(|| std::mem::take(&mut parsed.vocab_by_encoded));
    Ok(Tokenizer {
        contract: contract.clone(),
        patterns,
        byte_alphabet: parsed.byte_alphabet,
        byte_ids: parsed.byte_ids,
        merges: parsed.merges,
        whole_piece_ids,
        decoded: parsed.decoded,
        token_strings,
        added_by_text: parsed.added_by_text,
        special_ids: parsed.special_ids,
        trie: parsed.trie,
    })
}

struct ParsedModel {
    byte_alphabet: [char; 256],
    byte_ids: [u32; 256],
    merges: HashMap<(u32, u32), Merge>,
    decoded: Vec<Vec<u8>>,
    vocab_by_encoded: HashMap<String, u32>,
    added_by_text: BTreeMap<String, u32>,
    special_ids: BTreeSet<u32>,
    trie: Vec<TrieNode>,
}

fn parse_model(value: &Value, contract: &TokenizerContract) -> Result<ParsedModel> {
    let model = value.object("tokenizer model")?;
    expect_string(model, "type", "BPE")?;
    for key in ["dropout", "unk_token"] {
        expect_null(model, key)?;
    }
    expect_optional_string(
        model,
        "continuing_subword_prefix",
        contract.dialect.continuing_subword_prefix.as_deref(),
    )?;
    expect_optional_string(
        model,
        "end_of_word_suffix",
        contract.dialect.end_of_word_suffix.as_deref(),
    )?;
    expect_bool(model, "fuse_unk", false)?;
    expect_bool(model, "byte_fallback", false)?;
    expect_optional_bool(
        model,
        "ignore_merges",
        contract.dialect.ignore_merges.artifact_value(),
    )?;
    let vocab = required(model, "vocab")?.object("BPE vocabulary")?;
    if vocab.len() != contract.base_vocabulary_size as usize {
        return Err(TokenizerError::new(
            "BPE vocabulary count does not match policy",
        ));
    }
    let mut ids = BTreeSet::new();
    let mut vocab_by_encoded = HashMap::new();
    vocab_by_encoded
        .try_reserve(vocab.len())
        .map_err(|_| TokenizerError::new("BPE vocabulary allocation failed"))?;
    let mut decoded = vec![Vec::new(); contract.vocabulary_size as usize];
    let byte_alphabet = byte_alphabet_decoder();
    for (token, id_value) in vocab {
        let id = to_u32(id_value.number("BPE token ID")?, "BPE token ID")?;
        if id >= contract.base_vocabulary_size || !ids.insert(id) {
            return Err(TokenizerError::new(
                "duplicate or out-of-range BPE token ID",
            ));
        }
        let token_bytes = match contract.dialect.base_tokens {
            BaseTokenRepresentation::LiteralPrefixThenByteLevel { byte_start_id }
                if id < byte_start_id =>
            {
                token.as_bytes().to_vec()
            }
            BaseTokenRepresentation::LiteralPrefixThenByteLevel { .. }
            | BaseTokenRepresentation::ByteLevel => decode_byte_alphabet(token, &byte_alphabet)?,
        };
        if token_bytes.len() > MAX_TOKEN_BYTES {
            return Err(TokenizerError::new(
                "BPE token exceeds the token-byte limit",
            ));
        }
        decoded[id as usize] = token_bytes;
        vocab_by_encoded.insert(token.clone(), id);
    }
    if ids.len() != contract.base_vocabulary_size as usize
        || ids.first() != Some(&0)
        || ids.last() != Some(&(contract.base_vocabulary_size - 1))
    {
        return Err(TokenizerError::new(
            "BPE token IDs are not the fixed contiguous range",
        ));
    }
    let byte_alphabet = byte_alphabet_reverse();
    let mut byte_ids = [u32::MAX; 256];
    for (byte, encoded) in byte_alphabet.iter().enumerate() {
        byte_ids[byte] = *vocab_by_encoded
            .get(&encoded.to_string())
            .ok_or_else(|| TokenizerError::new(format!("BPE vocabulary lacks byte {byte}")))?;
        // Encoding emits this identifier for the byte, so decoding it has to
        // give the byte back. Only a literal base entry can break this.
        if decoded[byte_ids[byte] as usize] != [byte as u8] {
            return Err(TokenizerError::new(format!(
                "BPE token for byte {byte} does not decode to that byte"
            )));
        }
    }
    let merge_values = required(model, "merges")?.array("BPE merges")?;
    if merge_values.len() != contract.merge_count {
        return Err(TokenizerError::new("BPE merge count does not match policy"));
    }
    let mut merges = HashMap::new();
    merges
        .try_reserve(merge_values.len())
        .map_err(|_| TokenizerError::new("BPE merge allocation failed"))?;
    for (rank, value) in merge_values.iter().enumerate() {
        let (left_text, right_text) =
            parse_merge_record(value, rank, contract.dialect.merge_records)?;
        let left = *vocab_by_encoded.get(left_text).ok_or_else(|| {
            TokenizerError::new(format!("BPE merge {rank} has unknown left token"))
        })?;
        let right = *vocab_by_encoded.get(right_text).ok_or_else(|| {
            TokenizerError::new(format!("BPE merge {rank} has unknown right token"))
        })?;
        let joined = format!("{left_text}{right_text}");
        let result = *vocab_by_encoded
            .get(&joined)
            .ok_or_else(|| TokenizerError::new(format!("BPE merge {rank} has no result token")))?;
        if let MergeResultIdPolicy::Sequential { first_id } = contract.dialect.merge_results {
            let expected = first_id
                .checked_add(
                    u32::try_from(rank).map_err(|_| TokenizerError::new("merge rank overflow"))?,
                )
                .ok_or_else(|| TokenizerError::new("merge result ID overflow"))?;
            if result != expected {
                return Err(TokenizerError::new(format!(
                    "BPE merge {rank} has a mismatched result rank"
                )));
            }
        }
        // Encoding replaces the pair with the result, so the result has to
        // decode to the pair's bytes. Only a literal base entry can break this.
        let (left_bytes, right_bytes) = (&decoded[left as usize], &decoded[right as usize]);
        let result_bytes = &decoded[result as usize];
        if result_bytes.len() != left_bytes.len() + right_bytes.len()
            || !result_bytes.starts_with(left_bytes)
            || !result_bytes.ends_with(right_bytes)
        {
            return Err(TokenizerError::new(format!(
                "BPE merge {rank} result does not decode to its two parts"
            )));
        }
        let rank = u32::try_from(rank).map_err(|_| TokenizerError::new("merge rank overflow"))?;
        if merges
            .insert((left, right), Merge { rank, result })
            .is_some()
        {
            return Err(TokenizerError::new("duplicate BPE merge pair"));
        }
    }
    Ok(ParsedModel {
        byte_alphabet,
        byte_ids,
        merges,
        decoded,
        vocab_by_encoded,
        added_by_text: BTreeMap::new(),
        special_ids: BTreeSet::new(),
        trie: vec![TrieNode::default()],
    })
}

fn parse_merge_record(
    value: &Value,
    rank: usize,
    policy: MergeRecordPolicy,
) -> Result<(&str, &str)> {
    let (left, right) = match policy {
        MergeRecordPolicy::SpaceSeparatedString => {
            let record = value.string("BPE merge string record")?;
            let mut fields = record.split(' ');
            let left = fields.next().unwrap_or_default();
            let right = fields.next().unwrap_or_default();
            if fields.next().is_some() {
                return Err(TokenizerError::new(format!(
                    "malformed BPE merge record {rank}"
                )));
            }
            (left, right)
        }
        MergeRecordPolicy::StringOrPair if matches!(value, Value::String(_)) => {
            let record = value.string("BPE merge string record")?;
            let mut fields = record.split(' ');
            let left = fields.next().unwrap_or_default();
            let right = fields.next().unwrap_or_default();
            if fields.next().is_some() {
                return Err(TokenizerError::new(format!(
                    "malformed BPE merge record {rank}"
                )));
            }
            (left, right)
        }
        MergeRecordPolicy::StringPairArray | MergeRecordPolicy::StringOrPair => {
            let fields = value.array("BPE merge two-string array")?;
            if fields.len() != 2 {
                return Err(TokenizerError::new(format!(
                    "malformed BPE merge record {rank}: expected a two-string array"
                )));
            }
            let left = fields[0].string(&format!("BPE merge record {rank} left token"))?;
            let right = fields[1].string(&format!("BPE merge record {rank} right token"))?;
            (left, right)
        }
    };
    if left.is_empty() || right.is_empty() {
        return Err(TokenizerError::new(format!(
            "malformed BPE merge record {rank}: tokens must be nonempty"
        )));
    }
    if left.len() > MAX_TOKEN_BYTES || right.len() > MAX_TOKEN_BYTES {
        return Err(TokenizerError::new(format!(
            "BPE merge record {rank} exceeds the token-byte limit"
        )));
    }
    Ok((left, right))
}

fn parse_added_tokens(
    value: &Value,
    contract: &TokenizerContract,
    parsed: &mut ParsedModel,
) -> Result<()> {
    let tokens = value.array("added tokens")?;
    if tokens.len() != contract.added_token_count {
        return Err(TokenizerError::new(
            "added-token count does not match policy",
        ));
    }
    let mut ids = BTreeSet::new();
    for token_value in tokens {
        let token = token_value.object("added token")?;
        let id = to_u32(
            required(token, "id")?.number("added token ID")?,
            "added token ID",
        )?;
        let content = required(token, "content")?.string("added token content")?;
        if content.is_empty() || content.len() > MAX_SPECIAL_TOKEN_BYTES {
            return Err(TokenizerError::new("added token has invalid byte length"));
        }
        for key in ["single_word", "lstrip", "rstrip"] {
            expect_bool(token, key, false)?;
        }
        let _normalized = required(token, "normalized")?.boolean("added token normalized flag")?;
        let special = required(token, "special")?.boolean("added token special flag")?;
        if id >= contract.vocabulary_size || !ids.insert(id) {
            return Err(TokenizerError::new(
                "duplicate or out-of-range added token ID",
            ));
        }
        if let Some(base_id) = parsed.vocab_by_encoded.get(content) {
            if *base_id != id || parsed.decoded[id as usize] != content.as_bytes() {
                return Err(TokenizerError::new(
                    "added token collides with a base token",
                ));
            }
        } else {
            if !parsed.decoded[id as usize].is_empty() {
                return Err(TokenizerError::new("added token ID collision"));
            }
            parsed.decoded[id as usize] = content.as_bytes().to_vec();
        }
        if parsed
            .added_by_text
            .insert(content.to_owned(), id)
            .is_some()
        {
            return Err(TokenizerError::new("duplicate added token content"));
        }
        if special {
            parsed.special_ids.insert(id);
        }
        trie_insert(&mut parsed.trie, content, id)?;
    }
    if parsed.decoded.iter().any(Vec::is_empty) {
        return Err(TokenizerError::new(
            "tokenizer vocabulary contains an unassigned token ID",
        ));
    }
    for expected in &contract.prompt_special_tokens {
        if !parsed.added_by_text.contains_key(expected) {
            return Err(TokenizerError::new(format!(
                "prompt special token '{expected}' is absent"
            )));
        }
    }
    Ok(())
}

fn trie_insert(trie: &mut Vec<TrieNode>, token: &str, id: u32) -> Result<()> {
    let mut node = 0;
    for byte in token.bytes() {
        let next = if let Some(next) = trie[node].children.get(&byte) {
            *next
        } else {
            let next = trie.len();
            trie.try_reserve(1)
                .map_err(|_| TokenizerError::new("special-token trie allocation failed"))?;
            trie.push(TrieNode::default());
            trie[node].children.insert(byte, next);
            next
        };
        node = next;
    }
    if trie[node]
        .terminal
        .replace((id, token.to_owned()))
        .is_some()
    {
        return Err(TokenizerError::new("duplicate special-token trie entry"));
    }
    Ok(())
}

fn validate_configuration(bytes: &[u8], contract: &TokenizerContract) -> Result<()> {
    let value = json::parse(bytes)?;
    let config = value.object("tokenizer configuration")?;
    validate_bool_config_field(config, "add_bos_token", contract.dialect.add_bos_token)?;
    expect_optional_bool(config, "add_eos_token", contract.dialect.add_eos_token)?;
    expect_bool(config, "clean_up_tokenization_spaces", false)?;
    expect_optional_bool(config, "legacy", contract.dialect.legacy)?;
    expect_string(config, "tokenizer_class", &contract.dialect.tokenizer_class)?;
    validate_token_config_field(
        config,
        "unk_token",
        &contract.dialect.unk_token,
        None,
        contract.dialect.config_tokens,
        "unknown",
    )?;
    if required(config, "model_max_length")?.number("model_max_length")?
        != contract.model_max_length as u64
    {
        return Err(TokenizerError::new(
            "tokenizer model_max_length does not match policy",
        ));
    }
    validate_token_config_field(
        config,
        "bos_token",
        &contract.dialect.bos_token,
        contract.bos_token.as_deref(),
        contract.dialect.config_tokens,
        "BOS",
    )?;
    validate_optional_config_token(
        required(config, "eos_token")?,
        Some(&contract.eos_token),
        contract.dialect.config_tokens,
        "EOS",
    )?;
    validate_optional_config_token(
        required(config, "pad_token")?,
        Some(&contract.pad_token),
        contract.dialect.config_tokens,
        "pad",
    )?;
    if contract.dialect.empty_sentencepiece_kwargs {
        let kwargs = required(config, "sp_model_kwargs")?.object("sp_model_kwargs")?;
        if !kwargs.is_empty() {
            return Err(TokenizerError::new(
                "tokenizer sentencepiece policy mismatch",
            ));
        }
    } else if config.contains_key("sp_model_kwargs") {
        return Err(TokenizerError::new(
            "tokenizer has unconfigured sentencepiece kwargs",
        ));
    }
    Ok(())
}

fn validate_bool_config_field(
    config: &HashMap<String, Value>,
    key: &str,
    policy: ConfigFieldPolicy<bool>,
) -> Result<()> {
    match policy {
        ConfigFieldPolicy::Required(expected) => expect_bool(config, key, expected),
        ConfigFieldPolicy::RequiredNull => expect_null(config, key),
        ConfigFieldPolicy::Absent => expect_absent(config, key),
    }
}

fn validate_token_config_field(
    config: &HashMap<String, Value>,
    key: &str,
    policy: &ConfigFieldPolicy<ConfigTokenValue>,
    contract_value: Option<&str>,
    representation: ConfigTokenRepresentation,
    name: &str,
) -> Result<()> {
    match policy {
        ConfigFieldPolicy::Required(value) => {
            let expected = match value {
                ConfigTokenValue::ContractValue => contract_value.ok_or_else(|| {
                    TokenizerError::new(format!(
                        "tokenizer config field '{key}' requires a missing contract value"
                    ))
                })?,
                ConfigTokenValue::Literal(value) => value,
            };
            validate_optional_config_token(
                required(config, key)?,
                Some(expected),
                representation,
                name,
            )
        }
        ConfigFieldPolicy::RequiredNull => expect_null(config, key),
        ConfigFieldPolicy::Absent => expect_absent(config, key),
    }
}

fn validate_optional_config_token(
    value: &Value,
    expected: Option<&str>,
    representation: ConfigTokenRepresentation,
    name: &str,
) -> Result<()> {
    let Some(expected) = expected else {
        return if matches!(value, Value::Null) {
            Ok(())
        } else {
            Err(TokenizerError::new(format!(
                "tokenizer {name} token must be null"
            )))
        };
    };
    match representation {
        ConfigTokenRepresentation::AddedToken { normalized } => {
            let token = value.object(&format!("{name} token"))?;
            expect_string(token, "__type", "AddedToken")?;
            expect_string(token, "content", expected)?;
            for key in ["lstrip", "rstrip", "single_word"] {
                expect_bool(token, key, false)?;
            }
            expect_bool(token, "normalized", normalized)
        }
        ConfigTokenRepresentation::String => {
            if value.string(&format!("{name} token"))? == expected {
                Ok(())
            } else {
                Err(TokenizerError::new(format!(
                    "tokenizer {name} token does not match policy"
                )))
            }
        }
    }
}

fn validate_normalizer(value: &Value, policy: NormalizerPolicy) -> Result<()> {
    match policy {
        NormalizerPolicy::IdentitySequence => {
            let normalizer = value.object("normalizer")?;
            expect_string(normalizer, "type", "Sequence")?;
            if !required(normalizer, "normalizers")?
                .array("normalizers")?
                .is_empty()
            {
                return Err(TokenizerError::new("tokenizer normalization must be empty"));
            }
            Ok(())
        }
        NormalizerPolicy::Nfc => expect_string(value.object("normalizer")?, "type", "NFC"),
        NormalizerPolicy::Null if matches!(value, Value::Null) => Ok(()),
        NormalizerPolicy::Null => Err(TokenizerError::new("tokenizer normalizer must be null")),
    }
}

fn validate_pretokenizer(
    value: &Value,
    contract: &TokenizerContract,
) -> Result<Vec<PretokenizerPattern>> {
    let sequence = value.object("pretokenizer")?;
    expect_string(sequence, "type", "Sequence")?;
    let values = required(sequence, "pretokenizers")?.array("pretokenizer sequence")?;
    let expected_count = contract
        .pretokenizer_patterns
        .len()
        .checked_add(1)
        .ok_or_else(|| TokenizerError::new("pretokenizer count overflow"))?;
    if values.len() != expected_count {
        return Err(TokenizerError::new(
            "pretokenizer sequence length does not match policy",
        ));
    }
    let mut patterns = Vec::new();
    for (index, expected) in contract.pretokenizer_patterns.iter().enumerate() {
        let split = values[index].object("split pretokenizer")?;
        expect_string(split, "type", "Split")?;
        expect_string(split, "behavior", "Isolated")?;
        expect_bool(split, "invert", false)?;
        let pattern = required(split, "pattern")?.object("split pattern")?;
        expect_string(pattern, "Regex", expected)?;
        patterns.push(PretokenizerPattern::new(expected)?);
    }
    validate_byte_level(
        &values[values.len() - 1],
        contract.dialect.pretokenizer_byte_level,
        "byte-level pretokenizer",
    )?;
    Ok(patterns)
}

fn validate_byte_level(value: &Value, policy: ByteLevelPolicy, context: &str) -> Result<()> {
    let object = value.object(context)?;
    expect_string(object, "type", "ByteLevel")?;
    expect_bool(object, "add_prefix_space", policy.add_prefix_space)?;
    expect_bool(object, "trim_offsets", policy.trim_offsets)?;
    expect_bool(object, "use_regex", policy.use_regex)
}

fn expect_optional_string(
    object: &HashMap<String, Value>,
    key: &str,
    expected: Option<&str>,
) -> Result<()> {
    match expected {
        Some(expected) => expect_string(object, key, expected),
        None => expect_null(object, key),
    }
}

fn expect_optional_bool(
    object: &HashMap<String, Value>,
    key: &str,
    expected: Option<bool>,
) -> Result<()> {
    match expected {
        Some(expected) => expect_bool(object, key, expected),
        None if object.contains_key(key) => Err(TokenizerError::new(format!(
            "JSON field '{key}' is not configured by the tokenizer contract"
        ))),
        None => Ok(()),
    }
}

fn required<'a>(object: &'a HashMap<String, Value>, key: &str) -> Result<&'a Value> {
    object
        .get(key)
        .ok_or_else(|| TokenizerError::new(format!("missing JSON field '{key}'")))
}

fn expect_string(object: &HashMap<String, Value>, key: &str, expected: &str) -> Result<()> {
    if required(object, key)?.string(key)? == expected {
        Ok(())
    } else {
        Err(TokenizerError::new(format!(
            "JSON field '{key}' does not match policy"
        )))
    }
}

fn expect_bool(object: &HashMap<String, Value>, key: &str, expected: bool) -> Result<()> {
    if required(object, key)?.boolean(key)? == expected {
        Ok(())
    } else {
        Err(TokenizerError::new(format!(
            "JSON field '{key}' does not match policy"
        )))
    }
}

fn expect_null(object: &HashMap<String, Value>, key: &str) -> Result<()> {
    if matches!(required(object, key)?, Value::Null) {
        Ok(())
    } else {
        Err(TokenizerError::new(format!(
            "JSON field '{key}' must be null"
        )))
    }
}

fn expect_absent(object: &HashMap<String, Value>, key: &str) -> Result<()> {
    if object.contains_key(key) {
        Err(TokenizerError::new(format!(
            "JSON field '{key}' must be absent"
        )))
    } else {
        Ok(())
    }
}

fn to_u32(value: u64, context: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| TokenizerError::new(format!("{context} exceeds u32")))
}

fn byte_alphabet_reverse() -> [char; 256] {
    let mut direct = Vec::new();
    direct.extend(b'!'..=b'~');
    direct.extend(0xa1..=0xac);
    direct.extend(0xae..=0xff);
    let direct_set: BTreeSet<u8> = direct.iter().copied().collect();
    let mut output = ['\0'; 256];
    let mut extra = 0_u32;
    for byte in 0_u8..=u8::MAX {
        output[usize::from(byte)] = if direct_set.contains(&byte) {
            char::from(byte)
        } else {
            let value = char::from_u32(256 + extra).expect("byte alphabet scalar is valid");
            extra += 1;
            value
        };
    }
    output
}

fn byte_alphabet_decoder() -> [u16; 324] {
    let alphabet = byte_alphabet_reverse();
    let mut reverse = [u16::MAX; 324];
    for (byte, character) in alphabet.iter().enumerate() {
        reverse[*character as usize] = byte as u16;
    }
    reverse
}

fn decode_byte_alphabet(token: &str, reverse: &[u16; 324]) -> Result<Vec<u8>> {
    token
        .chars()
        .map(|character| {
            let value = reverse.get(character as usize).copied().unwrap_or(u16::MAX);
            u8::try_from(value).map_err(|_| {
                TokenizerError::new(format!(
                    "BPE token contains invalid byte-alphabet scalar U+{:04X}",
                    character as u32
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_BOS: &str = "<bos>";
    const FIXTURE_EOS: &str = "<eos>";
    const FIXTURE_PAD: &str = "<pad>";

    #[test]
    fn strict_json_rejects_duplicate_keys_and_invalid_utf8() {
        assert!(
            json::parse(br#"{"a":1,"a":2}"#)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        assert!(
            json::parse(b"{\"a\":\"\xff\"}")
                .unwrap_err()
                .to_string()
                .contains("UTF-8")
        );
    }

    #[test]
    fn byte_alphabet_is_bijective() {
        let alphabet = byte_alphabet_reverse();
        let reverse = byte_alphabet_decoder();
        assert_eq!(alphabet.iter().copied().collect::<BTreeSet<_>>().len(), 256);
        for byte in 0_u8..=u8::MAX {
            assert_eq!(
                decode_byte_alphabet(&alphabet[usize::from(byte)].to_string(), &reverse).unwrap(),
                [byte]
            );
        }
    }

    #[test]
    fn sequential_merge_ids_reject_gaps_while_vocabulary_policy_allows_them() {
        let definition = fixture_definition();
        let configuration = fixture_configuration();
        let definition_identity = FileIdentity::new(
            "tokenizer.json",
            definition.len() as u64,
            &sha256_bytes(&definition),
        )
        .unwrap();
        let configuration_identity = FileIdentity::new(
            "tokenizer_config.json",
            configuration.len() as u64,
            &sha256_bytes(&configuration),
        )
        .unwrap();

        let error = TokenizerContract::new_with_dialect(
            "test/model",
            "0123456789abcdef0123456789abcdef01234567",
            definition_identity.clone(),
            configuration_identity.clone(),
            260,
            260,
            0,
            3,
            1024,
            Some(FIXTURE_BOS),
            FIXTURE_EOS,
            FIXTURE_EOS,
            &[r".+"],
            &[FIXTURE_BOS],
            literal_prefix_policy(259, 3),
        )
        .unwrap_err();
        assert!(error.to_string().contains("merge or added-token count"));

        TokenizerContract::new_with_dialect(
            "test/model",
            "0123456789abcdef0123456789abcdef01234567",
            definition_identity,
            configuration_identity,
            260,
            260,
            0,
            3,
            1024,
            None,
            FIXTURE_EOS,
            FIXTURE_EOS,
            &[r".+"],
            &[FIXTURE_BOS],
            nfc_policy(),
        )
        .unwrap();
    }

    #[test]
    fn literal_prefix_byte_tokens_must_fit_the_base_vocabulary() {
        let definition = fixture_definition();
        let configuration = fixture_configuration();
        for byte_start_id in [4, u32::MAX - 255] {
            let error = TokenizerContract::new_with_dialect(
                "test/model",
                "0123456789abcdef0123456789abcdef01234567",
                FileIdentity::new(
                    "tokenizer.json",
                    definition.len() as u64,
                    &sha256_bytes(&definition),
                )
                .unwrap(),
                FileIdentity::new(
                    "tokenizer_config.json",
                    configuration.len() as u64,
                    &sha256_bytes(&configuration),
                )
                .unwrap(),
                259,
                259,
                0,
                3,
                1024,
                Some(FIXTURE_BOS),
                FIXTURE_EOS,
                FIXTURE_EOS,
                &[r".+"],
                &[FIXTURE_BOS],
                literal_prefix_policy(259, byte_start_id),
            )
            .unwrap_err();
            assert!(error.to_string().contains("byte tokens do not fit"));
        }
    }

    #[test]
    fn vocabulary_merge_count_uses_an_independent_resource_limit() {
        let definition = fixture_definition();
        let configuration = fixture_configuration();
        let definition_identity = FileIdentity::new(
            "tokenizer.json",
            definition.len() as u64,
            &sha256_bytes(&definition),
        )
        .unwrap();
        let configuration_identity = FileIdentity::new(
            "tokenizer_config.json",
            configuration.len() as u64,
            &sha256_bytes(&configuration),
        )
        .unwrap();

        for merge_count in [260, MAX_MERGE_COUNT] {
            TokenizerContract::new_with_dialect(
                "test/model",
                "0123456789abcdef0123456789abcdef01234567",
                definition_identity.clone(),
                configuration_identity.clone(),
                259,
                259,
                merge_count,
                3,
                1024,
                None,
                FIXTURE_EOS,
                FIXTURE_EOS,
                &[r".+"],
                &[FIXTURE_BOS],
                nfc_policy(),
            )
            .unwrap();
        }

        for merge_count in [MAX_MERGE_COUNT + 1, usize::MAX] {
            let error = TokenizerContract::new_with_dialect(
                "test/model",
                "0123456789abcdef0123456789abcdef01234567",
                definition_identity.clone(),
                configuration_identity.clone(),
                259,
                259,
                merge_count,
                3,
                1024,
                None,
                FIXTURE_EOS,
                FIXTURE_EOS,
                &[r".+"],
                &[FIXTURE_BOS],
                nfc_policy(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("merge-count resource limit"));
        }
    }

    #[test]
    fn compact_fixture_covers_byte_round_trips_and_special_policy() {
        let (definition, configuration, contract) = fixture();
        let tokenizer =
            Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();
        assert_eq!(tokenizer.encode("Aé", &[]).unwrap(), [68, 198, 172]);
        assert_eq!(tokenizer.decode_utf8(&[68, 198, 172]).unwrap(), "Aé");
        assert_eq!(tokenizer.token_bytes(68).unwrap(), b"A");
        assert!(!tokenizer.token_is_special(68).unwrap());
        assert_eq!(tokenizer.token_bytes(0).unwrap(), FIXTURE_BOS.as_bytes());
        assert!(tokenizer.token_is_special(0).unwrap());
        assert_eq!(
            tokenizer.decode_bytes(&[3, 131, 257, 258]).unwrap(),
            [0, 128, 254, 255]
        );
        assert!(tokenizer.decode_utf8(&[3, 131]).is_err());
        assert_eq!(tokenizer.encode(FIXTURE_BOS, &[FIXTURE_BOS]).unwrap(), [0]);
        assert!(
            tokenizer
                .encode(FIXTURE_BOS, &[])
                .unwrap_err()
                .to_string()
                .contains("disallowed")
        );
        assert!(
            tokenizer
                .encode("x", &["<unknown>"])
                .unwrap_err()
                .to_string()
                .contains("unknown")
        );
        assert!(
            tokenizer
                .decode_bytes(&[259])
                .unwrap_err()
                .to_string()
                .contains("outside")
        );
        assert!(tokenizer.token_bytes(259).is_err());
    }

    #[test]
    fn caller_limit_admits_text_above_the_standalone_default() {
        let (definition, configuration, mut contract) = fixture();
        let configuration = replace_once(
            &configuration,
            "\"model_max_length\":1024",
            "\"model_max_length\":1048576",
        );
        contract.configuration = FileIdentity::new(
            "tokenizer_config.json",
            configuration.len() as u64,
            &sha256_bytes(&configuration),
        )
        .unwrap();
        contract.model_max_length = 1_048_576;
        let tokenizer =
            Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();
        let count = MAX_INPUT_BYTES / FIXTURE_BOS.len() + 1;
        let text = FIXTURE_BOS.repeat(count);
        assert!(text.len() > MAX_INPUT_BYTES);
        assert!(tokenizer.encode(&text, &[FIXTURE_BOS]).is_err());
        assert!(
            tokenizer
                .encode_with_input_limit(&text, &[FIXTURE_BOS], text.len() - 1)
                .is_err()
        );
        let ids = tokenizer
            .encode_with_input_limit(&text, &[FIXTURE_BOS], text.len())
            .unwrap();
        assert_eq!(ids.len(), count);
        assert!(ids.iter().all(|id| *id == 0));
    }

    #[test]
    fn null_normalizer_requires_an_explicit_dialect_policy() {
        let definition = replace_once(
            &fixture_definition(),
            r#""normalizer":{"type":"Sequence","normalizers":[]}"#,
            r#""normalizer":null"#,
        );
        let configuration = fixture_configuration();

        let existing = fixture_contract(&definition, &configuration, 0);
        assert!(
            Tokenizer::from_verified_bytes(&definition, &configuration, &existing)
                .unwrap_err()
                .to_string()
                .contains("normalizer")
        );
        assert!(
            validate_normalizer(&Value::Null, nfc_policy().normalizer)
                .unwrap_err()
                .to_string()
                .contains("normalizer")
        );

        let mut dialect = literal_prefix_policy(259, 3);
        dialect.normalizer = NormalizerPolicy::Null;
        let contract = fixture_contract_with_dialect(&definition, &configuration, &dialect);
        Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();

        let definition = fixture_definition();
        let contract = fixture_contract_with_dialect(&definition, &configuration, &dialect);
        assert!(
            Tokenizer::from_verified_bytes(&definition, &configuration, &contract)
                .unwrap_err()
                .to_string()
                .contains("normalizer")
        );
    }

    #[test]
    fn absent_config_fields_require_an_explicit_policy() {
        let definition = replace_once(
            &fixture_definition(),
            r#""normalizer":{"type":"Sequence","normalizers":[]}"#,
            r#""normalizer":null"#,
        );
        let configuration = fixture_configuration_without_optional_dialect_fields();
        let mut dialect = literal_prefix_policy(259, 3);
        dialect.normalizer = NormalizerPolicy::Null;
        dialect.add_bos_token = ConfigFieldPolicy::Absent;
        dialect.bos_token = ConfigFieldPolicy::Absent;
        dialect.unk_token = ConfigFieldPolicy::Absent;
        let contract = fixture_contract_with_dialect(&definition, &configuration, &dialect);

        Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();

        for (field, configuration) in [
            (
                "add_bos_token",
                replace_once(
                    &configuration,
                    r#"{"add_eos_token"#,
                    r#"{"add_bos_token":false,"add_eos_token"#,
                ),
            ),
            (
                "bos_token",
                replace_once(
                    &configuration,
                    r#"{"add_eos_token"#,
                    r#"{"bos_token":null,"add_eos_token"#,
                ),
            ),
            (
                "unk_token",
                replace_once(
                    &configuration,
                    r#""tokenizer_class":"PreTrainedTokenizerFast""#,
                    r#""unk_token":"wrong","tokenizer_class":"PreTrainedTokenizerFast""#,
                ),
            ),
        ] {
            let contract = fixture_contract_with_dialect(&definition, &configuration, &dialect);
            let error =
                Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap_err();
            assert!(error.to_string().contains(field));
            assert!(error.to_string().contains("absent"));
        }
    }

    #[test]
    fn existing_dialects_still_require_current_config_fields() {
        let definition = fixture_definition();
        let configuration = fixture_configuration();
        for (field, configuration) in [
            (
                "add_bos_token",
                replace_once(&configuration, r#""add_bos_token":false,"#, ""),
            ),
            (
                "bos_token",
                replace_once(
                    &configuration,
                    r#""bos_token":{"__type":"AddedToken","content":"<bos>","lstrip":false,"normalized":true,"rstrip":false,"single_word":false},"#,
                    "",
                ),
            ),
            (
                "unk_token",
                replace_once(&configuration, r#","unk_token":null"#, ""),
            ),
        ] {
            let contract = fixture_contract(&definition, &configuration, 0);
            let error =
                Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap_err();
            assert!(error.to_string().contains(field));
            assert!(error.to_string().contains("missing"));
        }

        let configuration = fixture_nfc_configuration();
        for (field, configuration) in [
            (
                "add_bos_token",
                replace_once(&configuration, r#""add_bos_token":false,"#, ""),
            ),
            (
                "bos_token",
                replace_once(&configuration, r#""bos_token":null,"#, ""),
            ),
            (
                "unk_token",
                replace_once(&configuration, r#","unk_token":null"#, ""),
            ),
        ] {
            let contract =
                fixture_contract_with_dialect(&definition, &configuration, &nfc_policy());
            let error = validate_configuration(&configuration, &contract).unwrap_err();
            assert!(error.to_string().contains(field));
            assert!(error.to_string().contains("missing"));
        }
    }

    #[test]
    fn required_and_null_config_fields_reject_wrong_values() {
        let definition = fixture_definition();
        for (expected_error, configuration) in [
            (
                "add_bos_token",
                replace_once(
                    &fixture_configuration(),
                    r#""add_bos_token":false"#,
                    r#""add_bos_token":true"#,
                ),
            ),
            (
                "content",
                replace_once(&fixture_configuration(), "<bos>", "<wrong>"),
            ),
            (
                "unk_token",
                replace_once(
                    &fixture_configuration(),
                    r#""unk_token":null"#,
                    r#""unk_token":"wrong""#,
                ),
            ),
        ] {
            let contract = fixture_contract(&definition, &configuration, 0);
            let error = validate_configuration(&configuration, &contract).unwrap_err();
            assert!(error.to_string().contains(expected_error));
        }
    }

    #[test]
    fn exact_config_field_policy_accepts_only_its_value() {
        let definition = fixture_definition();
        let mut dialect = literal_prefix_policy(259, 3);
        dialect.unk_token =
            ConfigFieldPolicy::Required(ConfigTokenValue::Literal("<unknown>".to_owned()));
        let configuration = replace_once(
            &fixture_configuration(),
            r#""unk_token":null"#,
            r#""unk_token":{"__type":"AddedToken","content":"<unknown>","lstrip":false,"normalized":true,"rstrip":false,"single_word":false}"#,
        );
        let contract = fixture_contract_with_dialect(&definition, &configuration, &dialect);
        Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();

        for configuration in [
            replace_once(&configuration, "<unknown>", "<wrong>"),
            replace_once(
                &configuration,
                r#"{"__type":"AddedToken","content":"<unknown>","lstrip":false,"normalized":true,"rstrip":false,"single_word":false}"#,
                "null",
            ),
            replace_once(
                &configuration,
                r#","unk_token":{"__type":"AddedToken","content":"<unknown>","lstrip":false,"normalized":true,"rstrip":false,"single_word":false}"#,
                "",
            ),
        ] {
            let contract = fixture_contract_with_dialect(&definition, &configuration, &dialect);
            assert!(
                Tokenizer::from_verified_bytes(&definition, &configuration, &contract).is_err()
            );
        }
    }

    #[test]
    fn compact_merges_choose_the_lowest_rank() {
        let tokenizer = merged_tokenizer();
        assert_eq!(tokenizer.encode("abc", &[]).unwrap(), [100, 259]);
    }

    #[test]
    fn compact_merges_break_equal_rank_ties_leftmost() {
        let tokenizer = merged_tokenizer();
        assert_eq!(tokenizer.encode("aaaa", &[]).unwrap(), [261, 261]);
    }

    #[test]
    fn compact_merges_requeue_a_changed_neighbor() {
        let tokenizer = merged_tokenizer();
        assert_eq!(tokenizer.encode("def", &[]).unwrap(), [263]);
        assert_eq!(tokenizer.decode_utf8(&[263]).unwrap(), "def");
    }

    #[test]
    fn compact_pretokenizer_prevents_cross_piece_merges() {
        let tokenizer = merged_tokenizer();
        assert_eq!(tokenizer.encode("x,y", &[]).unwrap(), [123, 47, 124]);
    }

    #[test]
    fn pair_array_merges_preserve_ranked_bpe_behavior() {
        let (definition, configuration, mut dialect) = pair_array_fixture();
        dialect.merge_records = MergeRecordPolicy::StringPairArray;
        let contract = fixture_contract_counts_with_dialect(
            &definition,
            &configuration,
            265,
            265,
            6,
            &dialect,
        );
        let tokenizer =
            Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();

        assert_eq!(tokenizer.encode("abc", &[]).unwrap(), [100, 259]);
        assert_eq!(tokenizer.encode("def", &[]).unwrap(), [263]);
    }

    #[test]
    fn flexible_merge_records_accept_both_artifact_shapes() {
        let (string_definition, string_configuration, string_contract) = merged_fixture();
        let string_definition =
            replace_once(&string_definition, "\"Regex\":\"[,]\"", "\"Regex\":\".+\"");
        for (definition, configuration, mut dialect) in [
            pair_array_fixture(),
            (
                string_definition,
                string_configuration,
                string_contract.dialect,
            ),
        ] {
            dialect.merge_records = MergeRecordPolicy::StringOrPair;
            let contract = fixture_contract_counts_with_dialect(
                &definition,
                &configuration,
                265,
                265,
                6,
                &dialect,
            );
            let tokenizer =
                Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();
            assert_eq!(tokenizer.encode("def", &[]).unwrap(), [263]);
        }
    }

    #[test]
    fn merge_record_representation_must_match_the_contract() {
        let (array_definition, configuration, mut dialect) = pair_array_fixture();
        dialect.merge_records = MergeRecordPolicy::SpaceSeparatedString;
        let string_contract = fixture_contract_counts_with_dialect(
            &array_definition,
            &configuration,
            265,
            265,
            6,
            &dialect,
        );
        assert!(
            Tokenizer::from_verified_bytes(&array_definition, &configuration, &string_contract,)
                .unwrap_err()
                .to_string()
                .contains("string record")
        );

        let (string_definition, configuration, _) = merged_fixture();
        let string_definition =
            replace_once(&string_definition, "\"Regex\":\"[,]\"", "\"Regex\":\".+\"");
        dialect.merge_records = MergeRecordPolicy::StringPairArray;
        let array_contract = fixture_contract_counts_with_dialect(
            &string_definition,
            &configuration,
            265,
            265,
            6,
            &dialect,
        );
        assert!(
            Tokenizer::from_verified_bytes(&string_definition, &configuration, &array_contract,)
                .unwrap_err()
                .to_string()
                .contains("two-string array")
        );
    }

    #[test]
    fn pair_array_merges_reject_malformed_records() {
        let (definition, configuration, mut dialect) = pair_array_fixture();
        dialect.merge_records = MergeRecordPolicy::StringPairArray;
        for replacement in [
            "[]",
            "[\"b\"]",
            "[\"b\",\"c\",\"d\"]",
            "[1,\"c\"]",
            "[\"\",\"c\"]",
        ] {
            let malformed = replace_once(&definition, "[\"b\",\"c\"]", replacement);
            let contract = fixture_contract_counts_with_dialect(
                &malformed,
                &configuration,
                265,
                265,
                6,
                &dialect,
            );
            assert!(
                Tokenizer::from_verified_bytes(&malformed, &configuration, &contract)
                    .unwrap_err()
                    .to_string()
                    .contains("merge record 0")
            );
        }
    }

    #[test]
    fn pair_array_merges_reject_duplicates_order_drift_and_large_fields() {
        let (definition, configuration, mut dialect) = pair_array_fixture();
        dialect.merge_records = MergeRecordPolicy::StringPairArray;

        let duplicate = replace_once(&definition, "[\"a\",\"b\"]", "[\"b\",\"c\"]");
        dialect.merge_results = MergeResultIdPolicy::Vocabulary;
        let contract =
            fixture_contract_counts_with_dialect(&duplicate, &configuration, 265, 265, 6, &dialect);
        assert!(
            Tokenizer::from_verified_bytes(&duplicate, &configuration, &contract)
                .unwrap_err()
                .to_string()
                .contains("duplicate BPE merge pair")
        );

        let reordered = replace_once(
            &definition,
            "[\"b\",\"c\"],[\"a\",\"b\"]",
            "[\"a\",\"b\"],[\"b\",\"c\"]",
        );
        dialect.merge_results = MergeResultIdPolicy::Sequential { first_id: 259 };
        let contract =
            fixture_contract_counts_with_dialect(&reordered, &configuration, 265, 265, 6, &dialect);
        assert!(
            Tokenizer::from_verified_bytes(&reordered, &configuration, &contract)
                .unwrap_err()
                .to_string()
                .contains("mismatched result rank")
        );

        let oversized_field = "a".repeat(MAX_TOKEN_BYTES + 1);
        let oversized = replace_once(
            &definition,
            "[\"b\",\"c\"]",
            &format!("[{} ,\"c\"]", json_string(&oversized_field)),
        );
        let contract =
            fixture_contract_counts_with_dialect(&oversized, &configuration, 265, 265, 6, &dialect);
        assert!(
            Tokenizer::from_verified_bytes(&oversized, &configuration, &contract)
                .unwrap_err()
                .to_string()
                .contains("token-byte limit")
        );
    }

    #[test]
    fn ignore_merges_prefers_whole_pieces_and_falls_back_to_bpe() {
        let byte_alphabet = byte_alphabet_reverse();
        let snow = "雪"
            .bytes()
            .map(|byte| byte_alphabet[usize::from(byte)])
            .collect::<String>();
        let definition = replace_once(
            &fixture_definition(),
            "},\"merges\":[]",
            &format!(
                ",{}:259,\"ab\":260}},\"ignore_merges\":true,\"merges\":[[\"a\",\"b\"]]",
                json_string(&snow),
            ),
        );
        let configuration = fixture_configuration();
        let mut dialect = literal_prefix_policy(259, 3);
        dialect.base_tokens = BaseTokenRepresentation::ByteLevel;
        dialect.merge_results = MergeResultIdPolicy::Vocabulary;
        dialect.merge_records = MergeRecordPolicy::StringPairArray;
        dialect.ignore_merges = IgnoreMergesPolicy::WholePiece;
        let contract = fixture_contract_counts_with_dialect(
            &definition,
            &configuration,
            261,
            261,
            1,
            &dialect,
        );
        let tokenizer =
            Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();

        assert_eq!(tokenizer.encode("雪", &[]).unwrap(), [259]);
        assert_eq!(tokenizer.decode_utf8(&[259]).unwrap(), "雪");
        assert_eq!(tokenizer.encode("abx", &[]).unwrap(), [260, 123]);
        assert_eq!(tokenizer.decode_utf8(&[260, 123]).unwrap(), "abx");
    }

    #[test]
    fn a_prefix_space_pretokenizer_is_refused_rather_than_misread() {
        // The reference splits "abc" into single characters and, with the
        // pretokenizer's add_prefix_space, encodes each as a space and the
        // character. This fixture spells byte b as identifier b + 3.
        const REFERENCE: [u32; 6] = [35, 100, 35, 101, 35, 102];
        let definition = replace_once(&fixture_definition(), r#"".+""#, r#""[\\s\\S]""#);
        let definition = replace_once(
            &definition,
            r#""add_prefix_space":false"#,
            r#""add_prefix_space":true"#,
        );
        let configuration = fixture_configuration();
        let mut dialect = literal_prefix_policy(259, 3);
        dialect.pretokenizer_byte_level.add_prefix_space = true;
        let contract = TokenizerContract::new_with_dialect(
            "test/model",
            "0123456789abcdef0123456789abcdef01234567",
            FileIdentity::new(
                "tokenizer.json",
                definition.len() as u64,
                &sha256_bytes(&definition),
            )
            .unwrap(),
            FileIdentity::new(
                "tokenizer_config.json",
                configuration.len() as u64,
                &sha256_bytes(&configuration),
            )
            .unwrap(),
            259,
            259,
            0,
            3,
            1024,
            Some(FIXTURE_BOS),
            FIXTURE_EOS,
            FIXTURE_EOS,
            &[r"[\s\S]"],
            &[FIXTURE_BOS],
            dialect,
        );
        // Either the policy is refused, or the reader reproduces the reference.
        match contract {
            Err(error) => assert!(
                error.to_string().contains("unsupported tokenizer policy"),
                "{error}"
            ),
            Ok(contract) => {
                let tokenizer =
                    Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();
                assert_eq!(tokenizer.encode("abc", &[]).unwrap(), REFERENCE);
            }
        }
    }

    #[test]
    fn policies_the_reader_does_not_implement_are_refused() {
        let definition = fixture_definition();
        let configuration = fixture_configuration();
        type PolicyChange = fn(&mut TokenizerDialect);
        let cases: [(&str, PolicyChange); 5] = [
            ("prefix space", |dialect| {
                dialect.pretokenizer_byte_level.add_prefix_space = true;
            }),
            ("byte-level regex", |dialect| {
                dialect.pretokenizer_byte_level.use_regex = true;
            }),
            ("continuing-subword prefix", |dialect| {
                dialect.continuing_subword_prefix = Some("##".to_owned());
            }),
            ("end-of-word suffix", |dialect| {
                dialect.end_of_word_suffix = Some("</w>".to_owned());
            }),
            ("whole-piece lookup", |dialect| {
                dialect.ignore_merges = IgnoreMergesPolicy::WholePiece;
            }),
        ];
        for (what, change) in cases {
            let mut dialect = literal_prefix_policy(259, 3);
            change(&mut dialect);
            let error = TokenizerContract::new_with_dialect(
                "test/model",
                "0123456789abcdef0123456789abcdef01234567",
                FileIdentity::new(
                    "tokenizer.json",
                    definition.len() as u64,
                    &sha256_bytes(&definition),
                )
                .unwrap(),
                FileIdentity::new(
                    "tokenizer_config.json",
                    configuration.len() as u64,
                    &sha256_bytes(&configuration),
                )
                .unwrap(),
                259,
                259,
                0,
                3,
                1024,
                Some(FIXTURE_BOS),
                FIXTURE_EOS,
                FIXTURE_EOS,
                &[r".+"],
                &[FIXTURE_BOS],
                dialect,
            )
            .expect_err(what)
            .to_string();
            assert!(
                error.contains("unsupported tokenizer policy"),
                "{what}: {error}"
            );
            assert!(error.contains(what), "{what}: {error}");
        }

        // Empty affixes are no affix, and whole-piece lookup is implemented
        // over byte-level base tokens.
        let mut dialect = literal_prefix_policy(259, 3);
        dialect.continuing_subword_prefix = Some(String::new());
        dialect.end_of_word_suffix = Some(String::new());
        fixture_contract_with_dialect(&definition, &configuration, &dialect);
        dialect.base_tokens = BaseTokenRepresentation::ByteLevel;
        dialect.ignore_merges = IgnoreMergesPolicy::WholePiece;
        fixture_contract_with_dialect(&definition, &configuration, &dialect);
    }

    #[test]
    fn members_that_only_check_the_artifact_accept_any_value() {
        let definition = fixture_definition();
        let configuration = fixture_configuration();
        let unchecked = ByteLevelPolicy {
            add_prefix_space: false,
            trim_offsets: false,
            use_regex: false,
        };
        let mut dialect = literal_prefix_policy(259, 3);
        dialect.pretokenizer_byte_level.trim_offsets = false;
        dialect.post_processor_byte_level = unchecked;
        dialect.decoder_byte_level = unchecked;
        dialect.merge_results = MergeResultIdPolicy::Vocabulary;
        dialect.merge_records = MergeRecordPolicy::StringOrPair;
        dialect.normalizer = NormalizerPolicy::Nfc;
        dialect.ignore_merges = IgnoreMergesPolicy::Disabled;
        dialect.config_tokens = ConfigTokenRepresentation::String;
        dialect.add_bos_token = ConfigFieldPolicy::Required(true);
        dialect.bos_token = ConfigFieldPolicy::Absent;
        dialect.unk_token =
            ConfigFieldPolicy::Required(ConfigTokenValue::Literal("<u>".to_owned()));
        dialect.tokenizer_class = "AnyTokenizer".to_owned();
        dialect.add_eos_token = None;
        dialect.legacy = Some(false);
        dialect.empty_sentencepiece_kwargs = false;
        fixture_contract_with_dialect(&definition, &configuration, &dialect);
    }

    #[test]
    fn literal_base_entries_must_decode_as_the_bytes_they_encode() {
        let configuration = fixture_configuration();

        // Byte 0's token sits below the declared byte start, so it would
        // decode as the text of its spelling rather than as byte 0.
        let definition = layout_definition(&[], &[("ab", 259), ("cd", 260)], "[]");
        let contract = fixture_contract_counts_with_dialect(
            &definition,
            &configuration,
            261,
            261,
            0,
            &literal_prefix_policy(261, 4),
        );
        let error =
            Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap_err();
        assert!(
            error.to_string().contains("byte 0 does not decode"),
            "{error}"
        );

        // A literal entry merged with a byte token produces a byte-level
        // result whose bytes are not the literal's bytes and the byte.
        let definition = layout_definition(
            &["\u{100}\u{100}"],
            &[("\u{100}\u{100}\u{100}", 260)],
            "[\"\u{100}\u{100} \u{100}\"]",
        );
        let contract = fixture_contract_counts_with_dialect(
            &definition,
            &configuration,
            261,
            261,
            1,
            &literal_prefix_policy(260, 4),
        );
        let error =
            Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not decode to its two parts"),
            "{error}"
        );

        // A literal entry whose text is also its byte-level reading loads.
        let definition = layout_definition(&["<x>"], &[("<x>a", 260)], "[\"<x> a\"]");
        let contract = fixture_contract_counts_with_dialect(
            &definition,
            &configuration,
            261,
            261,
            1,
            &literal_prefix_policy(260, 4),
        );
        let tokenizer =
            Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap();
        assert_eq!(tokenizer.decode_utf8(&[260]).unwrap(), "<x>a");
    }

    #[test]
    fn artifact_names_reject_both_separator_styles() {
        for name in ["nested/tokenizer.json", "nested\\tokenizer.json"] {
            assert!(
                FileIdentity::new(
                    name,
                    1,
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                )
                .unwrap_err()
                .to_string()
                .contains("one file name")
            );
        }
    }

    #[test]
    fn authenticated_definition_above_legacy_limit_is_admissible() {
        let definition = vec![b' '; 20 * 1024 * 1024];
        let configuration = fixture_configuration();
        let contract = fixture_contract(&definition, &configuration, 0);

        let error = Tokenizer::from_verified_bytes(&definition, &configuration, &contract)
            .expect_err("generated data is authenticated but is not tokenizer JSON");
        assert!(error.to_string().contains("JSON"));
        assert!(!error.to_string().contains("size"));
    }

    #[test]
    fn definition_above_global_safety_limit_is_rejected() {
        let configuration = fixture_configuration();
        let definition = FileIdentity::new(
            "tokenizer.json",
            MAX_TOKENIZER_FILE_BYTES + 1,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        let configuration = FileIdentity::new(
            "tokenizer_config.json",
            configuration.len() as u64,
            &sha256_bytes(&configuration),
        )
        .unwrap();

        let error = TokenizerContract::new_with_dialect(
            "test/model",
            "0123456789abcdef0123456789abcdef01234567",
            definition,
            configuration,
            259,
            259,
            0,
            3,
            1024,
            Some(FIXTURE_BOS),
            FIXTURE_EOS,
            FIXTURE_EOS,
            &[r".+"],
            &[FIXTURE_BOS],
            literal_prefix_policy(259, 3),
        )
        .unwrap_err();
        assert!(error.to_string().contains("file-size limit"));
    }

    #[cfg(unix)]
    #[test]
    fn tokenizer_load_rejects_an_artifact_symlink() {
        use std::os::unix::fs::symlink;

        let (definition, configuration, contract) = fixture();
        let parent = std::env::temp_dir().join(format!(
            "runtime-tokenizer-symlink-artifact-{}",
            std::process::id()
        ));
        let outside = parent.join("outside.json");
        let root = parent.join("root");
        fs::create_dir_all(&root).unwrap();
        fs::write(&outside, &definition).unwrap();
        symlink(&outside, root.join("tokenizer.json")).unwrap();
        fs::write(root.join("tokenizer_config.json"), configuration).unwrap();

        let error = Tokenizer::load(&root, &contract).unwrap_err();

        fs::remove_dir_all(&parent).unwrap();
        assert!(error.to_string().contains("regular non-symlink file"));
    }

    #[test]
    fn tokenizer_load_rejects_digest_drift() {
        let (mut definition, configuration, contract) = fixture();
        definition[0] = b'[';
        let root = std::env::temp_dir().join(format!(
            "runtime-tokenizer-digest-drift-{}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("tokenizer.json"), definition).unwrap();
        fs::write(root.join("tokenizer_config.json"), configuration).unwrap();

        let error = Tokenizer::load(&root, &contract).unwrap_err();

        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("SHA-256 mismatch"));
    }

    #[cfg(unix)]
    #[test]
    fn authenticated_read_rejects_path_substitution() {
        let (definition, _, contract) = fixture();
        let root = std::env::temp_dir().join(format!(
            "runtime-tokenizer-path-substitution-{}",
            std::process::id()
        ));
        let path = root.join("tokenizer.json");
        let replacement = root.join("replacement.json");
        fs::create_dir(&root).unwrap();
        fs::write(&path, &definition).unwrap();
        fs::write(&replacement, &definition).unwrap();

        let error = read_verified_with_open_hook(
            &root,
            contract.definition(),
            MAX_TOKENIZER_FILE_BYTES,
            || fs::rename(&replacement, &path).unwrap(),
        )
        .unwrap_err();

        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("changed while it was read"));
    }

    #[cfg(unix)]
    #[test]
    fn tokenizer_root_rejects_a_directory_symlink() {
        use std::os::unix::fs::symlink;

        let (_, _, contract) = fixture();
        let parent = std::env::temp_dir().join(format!(
            "runtime-tokenizer-symlink-root-{}",
            std::process::id()
        ));
        let real = parent.join("real");
        let link = parent.join("link");
        fs::create_dir_all(&real).unwrap();
        symlink(&real, &link).unwrap();
        let error = Tokenizer::load(&link, &contract).unwrap_err();
        fs::remove_file(&link).unwrap();
        fs::remove_dir_all(&parent).unwrap();
        assert!(error.to_string().contains("real directory"));
        assert!(error.to_string().contains("symlink"));
    }

    #[test]
    fn authenticated_boundary_rejects_size_hash_missing_and_invalid_utf8() {
        let (definition, configuration, contract) = fixture();
        let mut extra = definition.clone();
        extra.push(b' ');
        assert!(
            Tokenizer::from_verified_bytes(&extra, &configuration, &contract)
                .unwrap_err()
                .to_string()
                .contains("size")
        );

        let invalid = b"{\"bad\":\"\xff\"}".to_vec();
        let invalid_contract = fixture_contract(&invalid, &configuration, 0);
        assert!(
            Tokenizer::from_verified_bytes(&invalid, &configuration, &invalid_contract)
                .unwrap_err()
                .to_string()
                .contains("UTF-8")
        );

        let root =
            std::env::temp_dir().join(format!("runtime-tokenizer-missing-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let error = Tokenizer::load(&root, &contract).unwrap_err();
        fs::remove_dir(&root).unwrap();
        assert!(error.to_string().contains("cannot inspect"));
    }

    #[test]
    fn parser_rejects_duplicate_ids_collisions_and_malformed_merges() {
        let (definition, configuration, _) = fixture();

        let duplicate_id = replace_once(&definition, "\"Ā\":3", "\"Ā\":4");
        let contract = fixture_contract(&duplicate_id, &configuration, 0);
        assert!(
            Tokenizer::from_verified_bytes(&duplicate_id, &configuration, &contract)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );

        let collision = replace_once(
            &definition,
            "\"id\":1,\"content\":\"<eos>\"",
            "\"id\":1,\"content\":\"<bos>\"",
        );
        let contract = fixture_contract(&collision, &configuration, 0);
        assert!(
            Tokenizer::from_verified_bytes(&collision, &configuration, &contract)
                .unwrap_err()
                .to_string()
                .contains("collides")
        );

        let malformed = replace_once(
            &definition,
            "},\"merges\":[]",
            ",\"AA\":259},\"merges\":[\"bad\"]",
        );
        let contract = fixture_contract_counts(&malformed, &configuration, 260, 260, 1);
        assert!(
            Tokenizer::from_verified_bytes(&malformed, &configuration, &contract)
                .unwrap_err()
                .to_string()
                .contains("malformed")
        );
    }

    fn fixture() -> (Vec<u8>, Vec<u8>, TokenizerContract) {
        let definition = fixture_definition();
        let configuration = fixture_configuration();
        let contract = fixture_contract(&definition, &configuration, 0);
        (definition, configuration, contract)
    }

    fn merged_fixture() -> (Vec<u8>, Vec<u8>, TokenizerContract) {
        let definition = replace_once(
            &fixture_definition(),
            "\"Regex\":\".+\"",
            "\"Regex\":\"[,]\"",
        );
        let definition = replace_once(
            &definition,
            "},\"merges\":[]",
            concat!(
                ",\"bc\":259,\"ab\":260,\"aa\":261,\"de\":262,",
                "\"def\":263,\"x,\":264},",
                "\"merges\":[\"b c\",\"a b\",\"a a\",\"d e\",\"de f\",\"x ,\"]"
            ),
        );
        let configuration = fixture_configuration();
        let contract =
            fixture_contract_counts_with_pattern(&definition, &configuration, 265, 265, 6, "[,]");
        (definition, configuration, contract)
    }

    /// Byte-level split patterns that end with `WHITESPACE_TAIL`: three
    /// published spellings with escaped line breaks, the last split of the
    /// whitespace-tail fixture with literal ones, and alternatives that test
    /// the rewrite (an earlier whitespace match, capture groups, an empty
    /// match, a lookbehind, and a search anchor and a match-start reset,
    /// which the group probe must read from the search's own position).
    const TAIL_PATTERNS: [&str; 10] = [
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
        r"[ \t]{2}|x+|\s*[\r\n]+|\s+(?!\S)|\s+",
        r"(a)(b)?|\s*[\r\n]+|\s+(?!\S)|\s+",
        r"\d*|\s*[\r\n]+|\s+(?!\S)|\s+",
        r"(?<=x)\s|\p{L}+|\s*[\r\n]+|\s+(?!\S)|\s+",
        r"\G\s+|\s*[\r\n]+|\s+(?!\S)|\s+",
        r"a\K\s+|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];

    /// The pattern as written, without the rewrite.
    fn lookahead_pattern(pattern: &str) -> PretokenizerPattern {
        PretokenizerPattern {
            regex: Regex::new(pattern).unwrap(),
            whitespace_group: None,
        }
    }

    fn tail_pattern(pattern: &str) -> PretokenizerPattern {
        let pattern = PretokenizerPattern::new(pattern).unwrap();
        assert!(pattern.whitespace_group.is_some());
        pattern
    }

    fn split_text(pattern: &PretokenizerPattern, text: &str) -> Result<Vec<String>> {
        let mut split = Vec::new();
        pattern.split(text, &mut split)?;
        Ok(split.into_iter().map(str::to_owned).collect())
    }

    #[test]
    fn whitespace_tail_rewrite_requires_the_top_level_alternatives() {
        for pattern in TAIL_PATTERNS {
            let rewritten = whitespace_tail_rewrite(pattern).unwrap();
            let prefix = rewritten.strip_suffix(r"|(\s+)").unwrap();
            assert!(pattern.starts_with(prefix), "{pattern}");
            assert_eq!(
                pattern[prefix.len()..]
                    .chars()
                    .filter(|c| *c == '|')
                    .count(),
                3
            );
        }
        assert_eq!(
            PretokenizerPattern::new(TAIL_PATTERNS[5])
                .unwrap()
                .whitespace_group,
            Some(3)
        );
        for pattern in [
            r"\s*[\r\n]+|\s+(?!\S)|\s+",
            r"(?:x|\s*[\r\n]+|\s+(?!\S)|\s+)",
            r"x\|\s*[\r\n]+|\s+(?!\S)|\s+",
            r"x|\s*[\r\n]+|\s+(?!\S)|\s+x",
            r"x|\s*[\r\n]+|\s+(?!\s)|\s+",
            r"x|\s*[\n]+|\s+(?!\S)|\s+",
            r"x|\s+|\s*[\r\n]+|\s+(?!\S)",
            r"(?x)x#|\s*[\r\n]+|\s+(?!\S)|\s+",
            r"[x|\s*[\r\n]+|\s+(?!\S)|\s+]",
            r".+",
        ] {
            assert_eq!(whitespace_tail_rewrite(pattern), None, "{pattern}");
            let parsed = PretokenizerPattern::new(pattern);
            assert!(parsed.is_err() || parsed.unwrap().whitespace_group.is_none());
        }
    }

    #[test]
    fn whitespace_tail_split_matches_the_lookahead_pattern() {
        let mut alphabet: Vec<&str> = vec![
            " ",
            " ",
            " ",
            "\t",
            "\n",
            "\n",
            "\r",
            "\r\n",
            "\x0b",
            "\x0c",
            "\u{85}",
            "\u{a0}",
            "\u{2028}",
            "\u{3000}",
            "x",
            "x",
            "a",
            "b",
            "Z",
            "é",
            "\u{301}",
            "1",
            "2",
            ".",
            "!",
            "-",
            "'s",
            "'S",
            "_",
            "一",
            "あ",
            "\u{1f600}",
            "\u{200d}",
        ];
        alphabet.sort_unstable();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        let mut texts = Vec::new();
        for _ in 0..20_000 {
            let length = 1 + next(24);
            texts.push(
                (0..length)
                    .map(|_| alphabet[next(alphabet.len())])
                    .collect::<String>(),
            );
        }
        for source in TAIL_PATTERNS {
            let (pattern, lookahead) = (tail_pattern(source), lookahead_pattern(source));
            for text in &texts {
                assert_eq!(
                    split_text(&pattern, text).unwrap(),
                    split_text(&lookahead, text).unwrap(),
                    "{source} {text:?}"
                );
            }
        }
    }

    #[test]
    fn whitespace_tail_split_reads_anchors_from_the_search_position() {
        // HF tokenizers 0.23.2 `Split(Regex(pattern), "isolated")`.
        for (pattern, text, expected) in [
            (TAIL_PATTERNS[8], "a  b", &["a", " ", " ", "b"][..]),
            (TAIL_PATTERNS[8], "  b", &["  ", "b"]),
            (TAIL_PATTERNS[8], "a \n b", &["a", " \n", " ", "b"]),
            (
                TAIL_PATTERNS[8],
                "a\t\t\tb ",
                &["a", "\t\t", "\t", "b", " "],
            ),
            (TAIL_PATTERNS[9], "a  b", &["a", "  ", "b"]),
            (TAIL_PATTERNS[9], "  b", &[" ", " ", "b"]),
            (TAIL_PATTERNS[9], "a \n b", &["a", " \n ", "b"]),
            (TAIL_PATTERNS[9], "a\t\t\tb ", &["a", "\t\t\t", "b", " "]),
        ] {
            assert_eq!(
                split_text(&tail_pattern(pattern), text).unwrap(),
                expected,
                "{pattern} {text:?}"
            );
        }
    }

    #[test]
    fn whitespace_tail_split_handles_runs_beyond_the_regex_stack() {
        let count = 1 << 20;
        for source in &TAIL_PATTERNS[..4] {
            let pattern = tail_pattern(source);
            for (text, expected) in [
                (" ".repeat(count), vec![" ".repeat(count)]),
                (
                    format!("{}x", " ".repeat(count - 1)),
                    vec![" ".repeat(count - 2), " x".to_owned()],
                ),
                (
                    format!("{} \nx", "\t".repeat(count - 2)),
                    vec![format!("{} \n", "\t".repeat(count - 2)), "x".to_owned()],
                ),
                (
                    format!("{}\r\n  x", "\u{3000}".repeat(count / 3)),
                    vec![
                        format!("{}\r\n", "\u{3000}".repeat(count / 3)),
                        " ".to_owned(),
                        " x".to_owned(),
                    ],
                ),
            ] {
                assert_eq!(split_text(&pattern, &text).unwrap(), expected, "{source}");
            }
        }
        // The pattern as written runs out of backtracking stack on such a run.
        let text = format!("{}x", " ".repeat(count));
        assert!(split_text(&lookahead_pattern(TAIL_PATTERNS[3]), &text).is_err());
    }

    #[test]
    fn whitespace_tail_tokenizer_matches_hf() {
        let definition =
            include_bytes!("../tests/fixtures/whitespace-tail/tokenizer.json").as_slice();
        let configuration =
            include_bytes!("../tests/fixtures/whitespace-tail/tokenizer_config.json").as_slice();
        let cases = json::parse(include_bytes!(
            "../tests/fixtures/whitespace-tail/cases.json"
        ))
        .unwrap();
        let parsed = json::parse(definition).unwrap();
        let splits = required(parsed.object("").unwrap(), "pre_tokenizer")
            .unwrap()
            .object("")
            .unwrap();
        let patterns: Vec<&str> = required(splits, "pretokenizers")
            .unwrap()
            .array("")
            .unwrap()
            .iter()
            .filter_map(|split| split.object("").unwrap().get("pattern"))
            .map(|pattern| {
                required(pattern.object("").unwrap(), "Regex")
                    .unwrap()
                    .string("")
                    .unwrap()
            })
            .collect();
        assert_eq!(patterns.len(), 3);
        let contract = TokenizerContract::new_with_dialect(
            "test/model",
            "0123456789abcdef0123456789abcdef01234567",
            FileIdentity::new(
                "tokenizer.json",
                definition.len() as u64,
                &sha256_bytes(definition),
            )
            .unwrap(),
            FileIdentity::new(
                "tokenizer_config.json",
                configuration.len() as u64,
                &sha256_bytes(configuration),
            )
            .unwrap(),
            279,
            279,
            20,
            3,
            1 << 20,
            Some(FIXTURE_BOS),
            FIXTURE_EOS,
            FIXTURE_EOS,
            &patterns,
            &[FIXTURE_BOS],
            literal_prefix_policy(259, 3),
        )
        .unwrap();
        let tokenizer =
            Tokenizer::from_verified_bytes(definition, configuration, &contract).unwrap();
        assert!(tokenizer.patterns[2].whitespace_group.is_some());
        let cases = cases.object("cases").unwrap();
        let short = required(cases, "cases").unwrap().array("cases").unwrap();
        assert!(short.len() > 1000);
        for case in short {
            let [text, ids] = case.array("case").unwrap() else {
                panic!("case is a pair")
            };
            let text = text.string("text").unwrap();
            let expected: Vec<u32> = ids
                .string("ids")
                .unwrap()
                .split_whitespace()
                .map(|id| id.parse().unwrap())
                .collect();
            assert_eq!(tokenizer.encode(text, &[]).unwrap(), expected, "{text:?}");
        }
        let long = required(cases, "long").unwrap().array("long").unwrap();
        assert!(!long.is_empty());
        for case in long {
            let case = case.object("long case").unwrap();
            let field = |key| required(case, key).unwrap();
            let mut text = field("repeat")
                .string("repeat")
                .unwrap()
                .repeat(field("count").number("count").unwrap() as usize);
            text.push_str(field("suffix").string("suffix").unwrap());
            let mut expected = Vec::new();
            for run in field("runs").array("runs").unwrap() {
                let [id, count] = run.array("run").unwrap() else {
                    panic!("run is a pair")
                };
                expected.extend(std::iter::repeat_n(
                    id.number("id").unwrap() as u32,
                    count.number("count").unwrap() as usize,
                ));
            }
            assert!(text.chars().count() >= 349_525);
            assert_eq!(
                tokenizer.encode(&text, &[]).unwrap(),
                expected,
                "{:?} x {}",
                field("repeat"),
                text.len()
            );
        }
    }

    fn pair_array_fixture() -> (Vec<u8>, Vec<u8>, TokenizerDialect) {
        let (definition, configuration, contract) = merged_fixture();
        let definition = replace_once(&definition, "\"Regex\":\"[,]\"", "\"Regex\":\".+\"");
        let definition = replace_once(
            &definition,
            "[\"b c\",\"a b\",\"a a\",\"d e\",\"de f\",\"x ,\"]",
            concat!(
                "[[\"b\",\"c\"],[\"a\",\"b\"],[\"a\",\"a\"],",
                "[\"d\",\"e\"],[\"de\",\"f\"],[\"x\",\",\"]]"
            ),
        );
        (definition, configuration, contract.dialect().clone())
    }

    /// Literal special tokens below the byte tokens, and merges whose results
    /// follow each other from one first identifier.
    fn literal_prefix_policy(first_merge_id: u32, byte_start_id: u32) -> TokenizerDialect {
        TokenizerDialect {
            normalizer: NormalizerPolicy::IdentitySequence,
            pretokenizer_byte_level: ByteLevelPolicy {
                add_prefix_space: false,
                trim_offsets: true,
                use_regex: false,
            },
            post_processor_byte_level: ByteLevelPolicy {
                add_prefix_space: true,
                trim_offsets: false,
                use_regex: true,
            },
            decoder_byte_level: ByteLevelPolicy {
                add_prefix_space: true,
                trim_offsets: true,
                use_regex: true,
            },
            merge_results: MergeResultIdPolicy::Sequential {
                first_id: first_merge_id,
            },
            merge_records: MergeRecordPolicy::SpaceSeparatedString,
            base_tokens: BaseTokenRepresentation::LiteralPrefixThenByteLevel { byte_start_id },
            config_tokens: ConfigTokenRepresentation::AddedToken { normalized: true },
            add_bos_token: ConfigFieldPolicy::Required(false),
            bos_token: ConfigFieldPolicy::Required(ConfigTokenValue::ContractValue),
            unk_token: ConfigFieldPolicy::RequiredNull,
            tokenizer_class: "PreTrainedTokenizerFast".to_owned(),
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            ignore_merges: IgnoreMergesPolicy::Absent,
            add_eos_token: Some(false),
            legacy: Some(true),
            empty_sentencepiece_kwargs: true,
        }
    }

    /// NFC normalization, vocabulary merge results and plain string tokens.
    fn nfc_policy() -> TokenizerDialect {
        let byte_level = ByteLevelPolicy {
            add_prefix_space: false,
            trim_offsets: false,
            use_regex: false,
        };
        TokenizerDialect {
            normalizer: NormalizerPolicy::Nfc,
            pretokenizer_byte_level: byte_level,
            post_processor_byte_level: byte_level,
            decoder_byte_level: byte_level,
            merge_results: MergeResultIdPolicy::Vocabulary,
            merge_records: MergeRecordPolicy::StringOrPair,
            base_tokens: BaseTokenRepresentation::ByteLevel,
            config_tokens: ConfigTokenRepresentation::String,
            add_bos_token: ConfigFieldPolicy::Required(false),
            bos_token: ConfigFieldPolicy::RequiredNull,
            unk_token: ConfigFieldPolicy::RequiredNull,
            tokenizer_class: "NfcTestTokenizer".to_owned(),
            continuing_subword_prefix: Some(String::new()),
            end_of_word_suffix: Some(String::new()),
            ignore_merges: IgnoreMergesPolicy::Disabled,
            add_eos_token: None,
            legacy: None,
            empty_sentencepiece_kwargs: false,
        }
    }

    fn merged_tokenizer() -> Tokenizer {
        let (definition, configuration, contract) = merged_fixture();
        Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap()
    }

    fn fixture_contract(
        definition: &[u8],
        configuration: &[u8],
        merge_count: usize,
    ) -> TokenizerContract {
        fixture_contract_counts(definition, configuration, 259, 259, merge_count)
    }

    fn fixture_contract_counts(
        definition: &[u8],
        configuration: &[u8],
        base_vocabulary_size: u32,
        vocabulary_size: u32,
        merge_count: usize,
    ) -> TokenizerContract {
        fixture_contract_counts_with_pattern(
            definition,
            configuration,
            base_vocabulary_size,
            vocabulary_size,
            merge_count,
            r".+",
        )
    }

    fn fixture_contract_counts_with_pattern(
        definition: &[u8],
        configuration: &[u8],
        base_vocabulary_size: u32,
        vocabulary_size: u32,
        merge_count: usize,
        pretokenizer_pattern: &str,
    ) -> TokenizerContract {
        TokenizerContract::new_with_dialect(
            "test/model",
            "0123456789abcdef0123456789abcdef01234567",
            FileIdentity::new(
                "tokenizer.json",
                definition.len() as u64,
                &sha256_bytes(definition),
            )
            .unwrap(),
            FileIdentity::new(
                "tokenizer_config.json",
                configuration.len() as u64,
                &sha256_bytes(configuration),
            )
            .unwrap(),
            base_vocabulary_size,
            vocabulary_size,
            merge_count,
            3,
            1024,
            Some(FIXTURE_BOS),
            FIXTURE_EOS,
            FIXTURE_EOS,
            &[pretokenizer_pattern],
            &[FIXTURE_BOS],
            literal_prefix_policy(259, 3),
        )
        .unwrap()
    }

    fn fixture_contract_counts_with_dialect(
        definition: &[u8],
        configuration: &[u8],
        base_vocabulary_size: u32,
        vocabulary_size: u32,
        merge_count: usize,
        dialect: &TokenizerDialect,
    ) -> TokenizerContract {
        TokenizerContract::new_with_dialect(
            "test/model",
            "0123456789abcdef0123456789abcdef01234567",
            FileIdentity::new(
                "tokenizer.json",
                definition.len() as u64,
                &sha256_bytes(definition),
            )
            .unwrap(),
            FileIdentity::new(
                "tokenizer_config.json",
                configuration.len() as u64,
                &sha256_bytes(configuration),
            )
            .unwrap(),
            base_vocabulary_size,
            vocabulary_size,
            merge_count,
            3,
            1024,
            Some(FIXTURE_BOS),
            FIXTURE_EOS,
            FIXTURE_EOS,
            &[r".+"],
            &[FIXTURE_BOS],
            dialect.clone(),
        )
        .unwrap()
    }

    fn fixture_contract_with_dialect(
        definition: &[u8],
        configuration: &[u8],
        dialect: &TokenizerDialect,
    ) -> TokenizerContract {
        TokenizerContract::new_with_dialect(
            "test/model",
            "0123456789abcdef0123456789abcdef01234567",
            FileIdentity::new(
                "tokenizer.json",
                definition.len() as u64,
                &sha256_bytes(definition),
            )
            .unwrap(),
            FileIdentity::new(
                "tokenizer_config.json",
                configuration.len() as u64,
                &sha256_bytes(configuration),
            )
            .unwrap(),
            259,
            259,
            0,
            3,
            1024,
            Some(FIXTURE_BOS),
            FIXTURE_EOS,
            FIXTURE_EOS,
            &[r".+"],
            &[FIXTURE_BOS],
            dialect.clone(),
        )
        .unwrap()
    }

    fn fixture_definition() -> Vec<u8> {
        let mut vocab = vec![
            format!("{}:0", json_string(FIXTURE_BOS)),
            format!("{}:1", json_string(FIXTURE_EOS)),
            format!("{}:2", json_string(FIXTURE_PAD)),
        ];
        for (byte, encoded) in byte_alphabet_reverse().iter().enumerate() {
            vocab.push(format!(
                "{}:{}",
                json_string(&encoded.to_string()),
                byte + 3
            ));
        }
        format!(
            concat!(
                "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,",
                "\"added_tokens\":[",
                "{{\"id\":0,\"content\":\"<bos>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}},",
                "{{\"id\":1,\"content\":\"<eos>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}},",
                "{{\"id\":2,\"content\":\"<pad>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}}],",
                "\"normalizer\":{{\"type\":\"Sequence\",\"normalizers\":[]}},",
                "\"pre_tokenizer\":{{\"type\":\"Sequence\",\"pretokenizers\":[",
                "{{\"type\":\"Split\",\"pattern\":{{\"Regex\":\".+\"}},\"behavior\":\"Isolated\",\"invert\":false}},",
                "{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}}]}},",
                "\"post_processor\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":false,\"use_regex\":true}},",
                "\"decoder\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}},",
                "\"model\":{{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,",
                "\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,",
                "\"fuse_unk\":false,\"byte_fallback\":false,\"vocab\":{{{}}},\"merges\":[]}}}}"
            ),
            vocab.join(",")
        )
        .into_bytes()
    }

    /// The fixture definition with `literals` between the special tokens and
    /// the byte tokens, `tail` entries after the byte tokens, and `merges`.
    fn layout_definition(literals: &[&str], tail: &[(&str, u32)], merges: &str) -> Vec<u8> {
        let mut vocab = vec![
            format!("{}:0", json_string(FIXTURE_BOS)),
            format!("{}:1", json_string(FIXTURE_EOS)),
            format!("{}:2", json_string(FIXTURE_PAD)),
        ];
        for (offset, literal) in literals.iter().enumerate() {
            vocab.push(format!("{}:{}", json_string(literal), offset + 3));
        }
        let byte_start = literals.len() + 3;
        for (byte, encoded) in byte_alphabet_reverse().iter().enumerate() {
            vocab.push(format!(
                "{}:{}",
                json_string(&encoded.to_string()),
                byte + byte_start
            ));
        }
        for (text, id) in tail {
            vocab.push(format!("{}:{id}", json_string(text)));
        }
        let definition = String::from_utf8(fixture_definition()).unwrap();
        let start = definition.find("\"vocab\":{").unwrap() + "\"vocab\":{".len();
        format!(
            "{}{}}},\"merges\":{merges}}}}}",
            &definition[..start],
            vocab.join(",")
        )
        .into_bytes()
    }

    fn fixture_configuration() -> Vec<u8> {
        concat!(
            "{\"add_bos_token\":false,\"add_eos_token\":false,",
            "\"bos_token\":{\"__type\":\"AddedToken\",\"content\":\"<bos>\",\"lstrip\":false,\"normalized\":true,\"rstrip\":false,\"single_word\":false},",
            "\"clean_up_tokenization_spaces\":false,",
            "\"eos_token\":{\"__type\":\"AddedToken\",\"content\":\"<eos>\",\"lstrip\":false,\"normalized\":true,\"rstrip\":false,\"single_word\":false},",
            "\"legacy\":true,\"model_max_length\":1024,",
            "\"pad_token\":{\"__type\":\"AddedToken\",\"content\":\"<eos>\",\"lstrip\":false,\"normalized\":true,\"rstrip\":false,\"single_word\":false},",
            "\"sp_model_kwargs\":{},\"tokenizer_class\":\"PreTrainedTokenizerFast\",\"unk_token\":null}"
        )
        .as_bytes()
        .to_vec()
    }

    fn fixture_configuration_without_optional_dialect_fields() -> Vec<u8> {
        let configuration = replace_once(&fixture_configuration(), r#""add_bos_token":false,"#, "");
        let configuration = replace_once(
            &configuration,
            r#""bos_token":{"__type":"AddedToken","content":"<bos>","lstrip":false,"normalized":true,"rstrip":false,"single_word":false},"#,
            "",
        );
        replace_once(&configuration, r#","unk_token":null"#, "")
    }

    fn fixture_nfc_configuration() -> Vec<u8> {
        concat!(
            "{\"add_bos_token\":false,\"bos_token\":null,",
            "\"clean_up_tokenization_spaces\":false,\"eos_token\":\"<eos>\",",
            "\"model_max_length\":1024,\"pad_token\":\"<eos>\",",
            "\"tokenizer_class\":\"NfcTestTokenizer\",\"unk_token\":null}"
        )
        .as_bytes()
        .to_vec()
    }

    fn json_string(input: &str) -> String {
        let mut output = String::from("\"");
        for character in input.chars() {
            match character {
                '"' => output.push_str("\\\""),
                '\\' => output.push_str("\\\\"),
                _ => output.push(character),
            }
        }
        output.push('"');
        output
    }

    fn replace_once(input: &[u8], from: &str, to: &str) -> Vec<u8> {
        let text = std::str::from_utf8(input).unwrap();
        assert_eq!(text.matches(from).count(), 1);
        text.replacen(from, to, 1).into_bytes()
    }
}
