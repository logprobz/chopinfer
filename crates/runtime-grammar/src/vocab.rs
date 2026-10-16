//! The vocabulary's token bytes, arranged once per tokenizer.
//!
//! Ordinary tokens are stored as a byte trie in depth-first order. Each node
//! records where its subtree ends, so a scan that rejects a prefix skips every
//! token under it in one step.

use crate::expr::GrammarError;

/// One vocabulary entry as the tokenizer decodes it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VocabularyToken {
    pub bytes: Vec<u8>,
    /// Tokens the tokenizer marks special never match grammar bytes, unless
    /// a grammar admits them by text.
    pub special: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TrieNode {
    pub(crate) byte: u8,
    /// Distance from the root; the first byte of a token is at depth 1.
    pub(crate) depth: u32,
    /// Index one past this node's subtree.
    pub(crate) end: u32,
    /// Range in the token list of the tokens ending at this node.
    pub(crate) tokens: (u32, u32),
}

/// Token bytes for a model vocabulary.
#[derive(Debug)]
pub struct TokenVocabulary {
    size: usize,
    pub(crate) nodes: Vec<TrieNode>,
    pub(crate) node_tokens: Vec<u32>,
    bytes: Vec<Option<Box<[u8]>>>,
    special: Vec<(u32, Box<[u8]>)>,
    pub(crate) max_depth: usize,
    by_length: Vec<u32>,
    whitespace_horizon: usize,
    whitespace_prefixes: Vec<(usize, u32)>,
}

impl TokenVocabulary {
    /// A key's lower frame can be entered in the middle of a token, so count
    /// whitespace runs anywhere, including runs in mixed tokens.
    pub(crate) fn whitespace_horizon(&self) -> usize {
        self.whitespace_horizon
    }

    pub(crate) fn longer_whitespace_prefixes(
        &self,
        budget: usize,
    ) -> impl Iterator<Item = u32> + '_ {
        let start = self
            .whitespace_prefixes
            .partition_point(|&(length, _)| length <= budget);
        self.whitespace_prefixes[start..]
            .iter()
            .map(|&(_, token)| token)
    }

    /// Ordinary tokens too long for a proven byte budget, longest last.
    pub(crate) fn longer_tokens(&self, budget: usize) -> &[u32] {
        let start = self.by_length.partition_point(|&token| {
            self.ordinary_bytes(token).expect("ordinary token").len() <= budget
        });
        &self.by_length[start..]
    }

    /// Builds the trie for a vocabulary `size` wide; `tokens[id]` describes
    /// token `id`, and identifiers past `tokens` (logits padding) have no
    /// bytes.
    pub fn new(size: usize, tokens: Vec<VocabularyToken>) -> Result<Self, GrammarError> {
        if tokens.len() > size || size == 0 || u32::try_from(size).is_err() {
            return Err(GrammarError::invalid(
                "vocabulary tokens exceed the declared vocabulary width",
            ));
        }
        let mut ordinary = Vec::new();
        let mut bytes = Vec::with_capacity(tokens.len());
        let mut special = Vec::new();
        let mut whitespace_horizon = 0;
        let mut whitespace_prefixes = Vec::new();
        for (id, token) in tokens.into_iter().enumerate() {
            let id = id as u32;
            if token.bytes.is_empty() {
                bytes.push(None);
            } else if token.special {
                special.push((id, token.bytes.into_boxed_slice()));
                bytes.push(None);
            } else {
                let prefix = token
                    .bytes
                    .iter()
                    .take_while(|byte| b" \n\t\r".contains(byte))
                    .count();
                if prefix > 0 {
                    whitespace_prefixes.push((prefix, id));
                }
                let mut run = 0;
                for byte in &token.bytes {
                    run = if b" \n\t\r".contains(byte) {
                        run + 1
                    } else {
                        0
                    };
                    whitespace_horizon = whitespace_horizon.max(run);
                }
                ordinary.push(id);
                bytes.push(Some(token.bytes.into_boxed_slice()));
            }
        }
        let token_bytes = |id: u32| bytes[id as usize].as_deref().expect("ordinary token");
        ordinary.sort_by(|&left, &right| token_bytes(left).cmp(token_bytes(right)));

        let mut nodes: Vec<TrieNode> = Vec::new();
        let mut node_tokens = Vec::with_capacity(ordinary.len());
        // Open path from the root: node index for each depth.
        let mut path: Vec<usize> = Vec::new();
        let mut previous: &[u8] = &[];
        let mut max_depth = 0;
        for &id in &ordinary {
            let current = token_bytes(id);
            let common = previous
                .iter()
                .zip(current)
                .take_while(|(left, right)| left == right)
                .count();
            while path.len() > common {
                let node = path.pop().expect("open node");
                nodes[node].end = nodes.len() as u32;
            }
            for &byte in &current[common..] {
                path.push(nodes.len());
                nodes.push(TrieNode {
                    byte,
                    depth: path.len() as u32,
                    end: 0,
                    tokens: (0, 0),
                });
            }
            let node = *path.last().expect("token has bytes");
            if nodes[node].tokens.1 == 0 {
                nodes[node].tokens = (node_tokens.len() as u32, node_tokens.len() as u32);
            }
            node_tokens.push(id);
            nodes[node].tokens.1 = node_tokens.len() as u32;
            max_depth = max_depth.max(current.len());
            previous = current;
        }
        while let Some(node) = path.pop() {
            nodes[node].end = nodes.len() as u32;
        }
        ordinary.sort_unstable_by_key(|&id| token_bytes(id).len());
        whitespace_prefixes.sort_unstable();
        Ok(Self {
            size,
            nodes,
            node_tokens,
            bytes,
            special,
            max_depth,
            by_length: ordinary,
            whitespace_horizon,
            whitespace_prefixes,
        })
    }

    /// Width of the model's logits row, and of every mask.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Number of `u32` words in one mask.
    #[must_use]
    pub fn mask_words(&self) -> usize {
        self.size.div_ceil(32)
    }

    /// The bytes an ordinary token matches, or `None` for special tokens,
    /// empty tokens and padding.
    #[must_use]
    pub fn ordinary_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.bytes.get(token_id as usize)?.as_deref()
    }

    /// Special tokens whose text is exactly `text`.
    pub(crate) fn special_with_text<'a>(
        &'a self,
        text: &'a [u8],
    ) -> impl Iterator<Item = u32> + 'a {
        self.special
            .iter()
            .filter(move |(_, bytes)| bytes.as_ref() == text)
            .map(|(id, _)| *id)
    }

    pub(crate) fn special_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.special
            .iter()
            .find(|(id, _)| *id == token_id)
            .map(|(_, bytes)| bytes.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(text: &str) -> VocabularyToken {
        VocabularyToken {
            bytes: text.as_bytes().to_vec(),
            special: false,
        }
    }

    #[test]
    fn trie_lists_every_ordinary_token_once_in_order() {
        let mut tokens = vec![
            token("ab"),
            token("a"),
            token("abc"),
            token("b"),
            token(""),
            token("ab"),
        ];
        tokens.push(VocabularyToken {
            bytes: b"<s>".to_vec(),
            special: true,
        });
        let vocabulary = TokenVocabulary::new(9, tokens).unwrap();
        let mut spelled = Vec::new();
        let mut prefix: Vec<u8> = Vec::new();
        for node in &vocabulary.nodes {
            prefix.truncate(node.depth as usize - 1);
            prefix.push(node.byte);
            for &id in &vocabulary.node_tokens[node.tokens.0 as usize..node.tokens.1 as usize] {
                spelled.push((prefix.clone(), id));
            }
        }
        assert_eq!(
            spelled,
            vec![
                (b"a".to_vec(), 1),
                (b"ab".to_vec(), 0),
                (b"ab".to_vec(), 5),
                (b"abc".to_vec(), 2),
                (b"b".to_vec(), 3),
            ]
        );
        assert_eq!(vocabulary.nodes[0].end, 3);
        assert_eq!(vocabulary.ordinary_bytes(4), None);
        assert_eq!(vocabulary.ordinary_bytes(6), None);
        assert_eq!(vocabulary.special_bytes(6), Some(&b"<s>"[..]));
        assert_eq!(vocabulary.mask_words(), 1);
    }
}
