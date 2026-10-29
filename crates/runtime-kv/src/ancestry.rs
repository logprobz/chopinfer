//! Content radix index. Ownership remains entirely in the pool's page refs.
//!
//! Nodes exist only at roots, anchor boundaries and branch points, and as
//! junctions where merging two insertions' edges would have copied more than
//! [`MERGE_COPY_TOKENS`]. An edge is a range of an immutable segment shared by
//! every node split from one insertion, so a split copies nothing. A segment
//! keeps the chain digest of each full block that ends inside it, so a
//! descent compares one digest and less than a block of token and media
//! atoms per node, and a divergence is found by binary search over digests.
//! Every comparison stops at the requested boundary.
//!
//! Fractional nominations flow toward the root until a cached ancestor at or
//! below their target absorbs them; one walk carries every target of a
//! change. Fork nominations stop at the first anchor. Updates propagate
//! nomination deltas, never scan all resident anchors.
//!
//! Each anchor also carries three flags for its lifetime. Extended: a
//! successful public restore found no resident descendant below it, so the
//! request continued from it. Consumed: an anchor below it was extended, so a
//! conversation has moved past it. Shared: it was a fork, or a successful
//! restore used it while it had a resident descendant. Only consumed,
//! unshared ancestors rank below recency. Each node counts the children whose
//! subtree holds an extended or consumed anchor, so a new anchor learns
//! whether it is consumed without a search.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::{PrefixCacheKeyV1, PrefixChainV1, PrefixMediaSpanV1, prefix::PrefixAtom};

/// The most tokens a merge of two insertions' edges copies. Longer
/// neighbours keep a junction node instead.
const MERGE_COPY_TOKENS: usize = 8192;

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum PrefixAnchorClassV1 {
    Superseded,
    Retained,
    #[default]
    Maximal,
}

impl PrefixAnchorClassV1 {
    pub(crate) const fn index(self) -> usize {
        self as usize
    }
}

/// Eviction priority, lowest first; least recently used within each tier.
/// The metric class labels Retained anchors of both the spaced and the live
/// tier, so ordering never uses the class.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PrefixEvictionTierV1 {
    /// A strict ancestor a conversation has moved past: one nearest
    /// descendant, consumed, never shared and not a spaced ancestor.
    Superseded,
    /// The same, but the deepest cached ancestor at or below half, a quarter
    /// or an eighth of a maximal frontier.
    Spaced,
    /// Everything else, by recency alone: maximal frontiers, shared
    /// anchors, ancestors nothing has moved past, private and digest-only
    /// anchors.
    #[default]
    Live,
}

/// Work the ancestry index and its pool did, to bound the cost of each
/// operation. Observability only: it is never part of pool equality.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefixAncestryWorkV1 {
    /// Radix nodes visited by descents, splits, merges, propagation and walks.
    pub nodes: u64,
    /// Block digests compared.
    pub digests_compared: u64,
    /// Token and media atoms compared one at a time.
    pub atoms_compared: u64,
    /// Tokens copied into new or merged edges.
    pub tokens_copied: u64,
    /// Nomination target entries updated.
    pub target_updates: u64,
    /// Page references the pool took or released for anchors.
    pub page_references: u64,
}

impl PrefixAncestryWorkV1 {
    /// The work done since `earlier`.
    #[must_use]
    pub const fn since(self, earlier: Self) -> Self {
        Self {
            nodes: self.nodes - earlier.nodes,
            digests_compared: self.digests_compared - earlier.digests_compared,
            atoms_compared: self.atoms_compared - earlier.atoms_compared,
            tokens_copied: self.tokens_copied - earlier.tokens_copied,
            target_updates: self.target_updates - earlier.target_updates,
            page_references: self.page_references - earlier.page_references,
        }
    }
}

type NodeId = u64;

/// Immutable content of one insertion, shared by every node split from it.
/// Ordinary tokens are stored densely; media is a separate sparse list.
#[derive(Debug, Default, Eq, PartialEq)]
struct Segment {
    /// Absolute position of the first token.
    start: usize,
    tokens: Box<[u32]>,
    /// Media spans starting inside the segment, by absolute start.
    media: Box<[PrefixMediaSpanV1]>,
    block_tokens: usize,
    /// Chain digests of the full blocks ending inside the segment:
    /// `digests[i]` names the first `start / block_tokens + 1 + i` blocks.
    digests: Box<[PrefixCacheKeyV1]>,
}

impl Segment {
    /// `start..end` of `chain`: its tokens, the media spans starting there
    /// and the chain digest of every full block ending there.
    fn from_chain(chain: &PrefixChainV1, start: usize, end: usize) -> Self {
        let block_tokens = chain.block_tokens();
        Self {
            start,
            tokens: chain.tokens()[start..end].into(),
            media: chain.media_starting(start, end).into(),
            block_tokens,
            digests: (start / block_tokens + 1..=end / block_tokens)
                .map(|blocks| chain.chain(blocks).expect("a full block of the chain"))
                .collect(),
        }
    }

    /// `start..middle` of `upper` followed by `middle..end` of `lower`.
    fn joined(upper: &Self, start: usize, middle: usize, lower: &Self, end: usize) -> Self {
        let block_tokens = upper.block_tokens;
        Self {
            start,
            tokens: upper.tokens[start - upper.start..middle - upper.start]
                .iter()
                .chain(&lower.tokens[middle - lower.start..end - lower.start])
                .copied()
                .collect(),
            media: upper
                .media
                .iter()
                .filter(|span| (start..middle).contains(&span.start))
                .chain(
                    lower
                        .media
                        .iter()
                        .filter(|span| (middle..end).contains(&span.start)),
                )
                .copied()
                .collect(),
            block_tokens,
            digests: (start / block_tokens + 1..=end / block_tokens)
                .map(|blocks| {
                    if blocks * block_tokens <= middle {
                        upper.digest(blocks)
                    } else {
                        lower.digest(blocks)
                    }
                })
                .collect(),
        }
    }

    fn end(&self) -> usize {
        self.start + self.tokens.len()
    }

    fn atom(&self, position: usize) -> PrefixAtom {
        PrefixAtom {
            token: self.tokens[position - self.start],
            media: self
                .media
                .binary_search_by_key(&position, |span| span.start)
                .ok()
                .map(|index| self.media[index]),
        }
    }

    /// The chain digest of the first `blocks` full blocks, which must end
    /// inside the segment.
    fn digest(&self, blocks: usize) -> PrefixCacheKeyV1 {
        self.digests[blocks - (self.start / self.block_tokens + 1)]
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Node {
    parent: Option<NodeId>,
    root: Option<PrefixCacheKeyV1>,
    /// The edge from the parent covers `start..depth` of this segment.
    segment: Arc<Segment>,
    start: usize,
    depth: usize,
    children: BTreeMap<PrefixAtom, NodeId>,
    anchor: Option<PrefixCacheKeyV1>,
    // Unabsorbed nominations arriving from children, with multiplicity.
    targets: BTreeMap<usize, usize>,
    forks: usize,
    /// Children whose subtree holds an extended or consumed anchor.
    marked: usize,
    // Lifetime flags of the anchor at this node; false without an anchor.
    extended: bool,
    consumed: bool,
    shared: bool,
}

impl Node {
    fn maximal(&self) -> bool {
        self.anchor.is_some() && self.children.is_empty()
    }

    fn fork(&self) -> bool {
        self.forks > 0 || self.children.len() >= 2
    }

    /// Whether this subtree holds an extended or consumed anchor.
    fn marks(&self) -> bool {
        self.marked > 0 || self.anchor.is_some() && (self.extended || self.consumed)
    }

    fn first_atom(&self) -> PrefixAtom {
        self.segment.atom(self.start)
    }

    fn rank(&self) -> (PrefixAnchorClassV1, PrefixEvictionTierV1) {
        use PrefixAnchorClassV1::{Maximal, Retained, Superseded};
        if self.maximal() {
            (Maximal, PrefixEvictionTierV1::Live)
        } else if self.fork() || self.shared || !self.consumed {
            (Retained, PrefixEvictionTierV1::Live)
        } else if self.targets.range(self.depth..).next().is_some() {
            (Retained, PrefixEvictionTierV1::Spaced)
        } else {
            (Superseded, PrefixEvictionTierV1::Superseded)
        }
    }

    fn outgoing(&self) -> (BTreeMap<usize, usize>, usize) {
        let mut targets: BTreeMap<_, _> = self
            .targets
            .iter()
            .filter(|(target, _)| self.anchor.is_none() || **target < self.depth)
            .map(|(target, count)| (*target, *count))
            .collect();
        if self.maximal() {
            for divisor in [2, 4, 8] {
                *targets.entry(self.depth / divisor).or_default() += 1;
            }
        }
        let forks = if self.anchor.is_none() {
            self.forks + usize::from(self.children.len() >= 2)
        } else {
            0
        };
        (targets, forks)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PrefixAncestry {
    roots: BTreeMap<PrefixCacheKeyV1, NodeId>,
    anchors: BTreeMap<PrefixCacheKeyV1, NodeId>,
    nodes: BTreeMap<NodeId, Node>,
    next: NodeId,
    dirty: BTreeSet<NodeId>,
    work: PrefixAncestryWorkV1,
}

// Work counters are observability, not index state.
impl PartialEq for PrefixAncestry {
    fn eq(&self, other: &Self) -> bool {
        self.roots == other.roots
            && self.anchors == other.anchors
            && self.nodes == other.nodes
            && self.next == other.next
            && self.dirty == other.dirty
    }
}

impl Eq for PrefixAncestry {}

impl PrefixAncestry {
    pub(crate) const fn work(&self) -> PrefixAncestryWorkV1 {
        self.work
    }

    fn allocate(&mut self, node: Node) -> NodeId {
        self.next += 1;
        let id = self.next;
        self.nodes.insert(id, node);
        id
    }

    /// Adds or removes counted nominations from `id` toward the root. Each
    /// passes anchors deeper than its target and stops at the first anchor
    /// at or below it; one walk carries them all.
    fn targets_delta(&mut self, mut id: NodeId, mut targets: Vec<(usize, usize)>, add: bool) {
        while !targets.is_empty() {
            self.work.nodes += 1;
            self.work.target_updates += targets.len() as u64;
            let node = self.nodes.get_mut(&id).expect("live radix node");
            for (target, count) in &targets {
                let value = node.targets.entry(*target).or_default();
                if add {
                    *value += count;
                } else {
                    *value -= count;
                }
                if *value == 0 {
                    node.targets.remove(target);
                }
            }
            self.dirty.insert(id);
            if node.anchor.is_some() {
                let depth = node.depth;
                targets.retain(|(target, _)| *target < depth);
            }
            let Some(parent) = node.parent else {
                break;
            };
            id = parent;
        }
    }

    fn fork_delta(&mut self, mut id: NodeId, count: usize, add: bool) {
        loop {
            self.work.nodes += 1;
            let node = self.nodes.get_mut(&id).expect("live radix node");
            if add {
                node.forks += count;
            } else {
                node.forks -= count;
            }
            self.dirty.insert(id);
            if node.anchor.is_some() {
                break;
            }
            let Some(parent) = node.parent else {
                break;
            };
            id = parent;
        }
    }

    /// Counts a change of `id`'s marked subtree into its ancestors until one
    /// already agrees.
    fn mark_delta(&mut self, mut parent: Option<NodeId>, add: bool) {
        while let Some(id) = parent {
            self.work.nodes += 1;
            let node = self.nodes.get_mut(&id).expect("live radix node");
            let before = node.marks();
            if add {
                node.marked += 1;
            } else {
                node.marked -= 1;
            }
            if node.marks() == before {
                break;
            }
            parent = node.parent;
        }
    }

    // Structural changes affect only this node's outgoing nominations. Ordinary
    // propagation above changes counted targets along its path.
    fn modify(&mut self, id: NodeId, change: impl FnOnce(&mut Node)) {
        self.work.nodes += 1;
        let node = &self.nodes[&id];
        let old_maximal = node.maximal();
        let old_anchor = node.anchor.is_some();
        let old_forks = if old_anchor {
            0
        } else {
            node.forks + usize::from(node.children.len() >= 2)
        };
        change(self.nodes.get_mut(&id).expect("live radix node"));
        self.dirty.insert(id);
        let node = &self.nodes[&id];
        let Some(parent) = node.parent else {
            return;
        };
        let new_anchor = node.anchor.is_some();
        let new_maximal = node.maximal();
        let depth = node.depth;
        let new_forks = if new_anchor {
            0
        } else {
            node.forks + usize::from(node.children.len() >= 2)
        };
        // Targets at or below a new anchor are absorbed; a removed anchor
        // passes them on.
        let mut passed: Vec<_> = if old_anchor != new_anchor {
            node.targets
                .range(depth..)
                .map(|(target, count)| (*target, *count))
                .collect()
        } else {
            Vec::new()
        };
        let mut frontier = Vec::new();
        if old_maximal != new_maximal {
            frontier = [2, 4, 8].map(|divisor| (depth / divisor, 1)).to_vec();
        }
        // Passed targets are added when the anchor goes; the frontier's own
        // targets are added when it becomes maximal. One walk serves both
        // when they move the same way.
        let (pass, nominate) = (!new_anchor, new_maximal);
        if pass == nominate {
            passed.append(&mut frontier);
            self.targets_delta(parent, passed, pass);
        } else {
            self.targets_delta(parent, passed, pass);
            self.targets_delta(parent, frontier, nominate);
        }
        if old_forks != new_forks {
            self.fork_delta(parent, old_forks.abs_diff(new_forks), new_forks > old_forks);
        }
    }

    /// Splits `child`'s edge at absolute position `at`, returning the new
    /// node above it. Both halves keep the child's segment.
    fn split(&mut self, child: NodeId, at: usize) -> NodeId {
        self.work.nodes += 1;
        let old = &self.nodes[&child];
        let parent = old.parent.expect("a root has no edge");
        let (targets, forks) = old.outgoing();
        let middle = self.allocate(Node {
            parent: Some(parent),
            segment: Arc::clone(&old.segment),
            start: old.start,
            depth: at,
            targets,
            forks,
            marked: usize::from(old.marks()),
            children: BTreeMap::from([(old.segment.atom(at), child)]),
            ..Node::default()
        });
        let first = {
            let node = self.nodes.get_mut(&child).expect("live radix node");
            let first = node.first_atom();
            node.parent = Some(middle);
            node.start = at;
            first
        };
        // The intermediate node forwards exactly the old child's summary.
        self.nodes
            .get_mut(&parent)
            .expect("live radix node")
            .children
            .insert(first, middle);
        middle
    }

    /// The first position in `start..limit` where `chain` differs from
    /// node `id`'s edge, or `limit`. Digests of the full blocks ending in
    /// the range locate the divergence; atoms are compared within one block.
    fn common(&mut self, id: NodeId, chain: &PrefixChainV1, start: usize, limit: usize) -> usize {
        let segment = Arc::clone(&self.nodes[&id].segment);
        let block_tokens = segment.block_tokens;
        let first = start / block_tokens + 1;
        let last = limit / block_tokens;
        let mut matched = start;
        let mut end = limit;
        if first <= last {
            let same = |work: &mut PrefixAncestryWorkV1, blocks: usize| {
                work.digests_compared += 1;
                Some(segment.digest(blocks)) == chain.chain(blocks)
            };
            if same(&mut self.work, last) {
                matched = last * block_tokens;
            } else {
                // The chain digests are prefix-closed, so their agreement is
                // monotone: find the last agreeing block.
                let (mut good, mut bad) = (first - 1, last);
                while bad - good > 1 {
                    let middle = good.midpoint(bad);
                    if same(&mut self.work, middle) {
                        good = middle;
                    } else {
                        bad = middle;
                    }
                }
                matched = matched.max(good * block_tokens);
                end = bad * block_tokens;
            }
        }
        for position in matched..end {
            self.work.atoms_compared += 1;
            if segment.atom(position) != chain.atom(position) {
                return position;
            }
        }
        end
    }

    pub(crate) fn insert(&mut self, chain: &PrefixChainV1, tokens: usize, key: PrefixCacheKeyV1) {
        let root = chain.chain(0).expect("chain root");
        let mut node = if let Some(id) = self.roots.get(&root) {
            *id
        } else {
            let id = self.allocate(Node {
                root: Some(root),
                ..Node::default()
            });
            self.roots.insert(root, id);
            id
        };
        loop {
            self.work.nodes += 1;
            let depth = self.nodes[&node].depth;
            if depth == tokens {
                break;
            }
            let atom = chain.atom(depth);
            let Some(child) = self.nodes[&node].children.get(&atom).copied() else {
                self.work.tokens_copied += (tokens - depth) as u64;
                let id = self.allocate(Node {
                    parent: Some(node),
                    segment: Arc::new(Segment::from_chain(chain, depth, tokens)),
                    start: depth,
                    depth: tokens,
                    ..Node::default()
                });
                self.modify(node, |parent| {
                    parent.children.insert(atom, id);
                });
                node = id;
                break;
            };
            let end = self.nodes[&child].depth;
            let common = self.common(child, chain, depth, end.min(tokens));
            node = if common == end {
                child
            } else {
                self.split(child, common)
            };
        }
        let consumed = self.nodes[&node].marked > 0;
        self.modify(node, |node| {
            node.anchor = Some(key);
            node.extended = false;
            node.consumed = consumed;
            node.shared = false;
        });
        self.anchors.insert(key, node);
    }

    /// Records a successful public restore of `key`, which the pool
    /// classified when it resumed: a branch restore had a resident
    /// descendant, an extension restore had none.
    pub(crate) fn restored(&mut self, key: PrefixCacheKeyV1, branch: bool) {
        let Some(&id) = self.anchors.get(&key) else {
            return;
        };
        if branch {
            self.nodes.get_mut(&id).expect("live radix node").shared = true;
            self.dirty.insert(id);
            return;
        }
        let node = self.nodes.get_mut(&id).expect("live radix node");
        let before = node.marks();
        node.extended = true;
        let parent = node.parent;
        if !before {
            self.mark_delta(parent, true);
        }
        // Every subtree above now holds the extended anchor, so marking
        // ancestors consumed changes no count.
        let mut parent = parent;
        while let Some(id) = parent {
            self.work.nodes += 1;
            let node = self.nodes.get_mut(&id).expect("live radix node");
            if node.anchor.is_some() {
                if node.consumed {
                    break;
                }
                node.consumed = true;
                self.dirty.insert(id);
            }
            parent = node.parent;
        }
    }

    /// Whether `key` has a resident descendant.
    pub(crate) fn has_descendant(&self, key: PrefixCacheKeyV1) -> bool {
        self.anchors
            .get(&key)
            .is_some_and(|id| !self.nodes[id].children.is_empty())
    }

    /// Whether `key` is shared now or was shared during its lifetime.
    pub(crate) fn shared(&self, key: PrefixCacheKeyV1) -> bool {
        self.anchors.get(&key).is_some_and(|id| {
            let node = &self.nodes[id];
            node.shared || node.fork()
        })
    }

    pub(crate) fn remove(&mut self, key: PrefixCacheKeyV1) {
        let Some(id) = self.anchors.remove(&key) else {
            return;
        };
        let before = self.nodes[&id].marks();
        self.modify(id, |node| {
            node.anchor = None;
            node.extended = false;
            node.consumed = false;
            node.shared = false;
        });
        if before && !self.nodes[&id].marks() {
            self.mark_delta(self.nodes[&id].parent, false);
        }
        let mut stop = Some(id);
        while let Some(id) = stop {
            let node = &self.nodes[&id];
            if node.anchor.is_some() || !node.children.is_empty() {
                break;
            }
            self.work.nodes += 1;
            let node = self.nodes.remove(&id).expect("live radix node");
            self.dirty.remove(&id);
            let Some(parent) = node.parent else {
                self.roots.remove(&node.root.expect("root identity"));
                stop = None;
                break;
            };
            self.modify(parent, |parent| {
                parent.children.remove(&node.first_atom());
            });
            stop = Some(parent);
        }
        // Only the node pruning stopped at changed its children, and insert
        // never leaves a mergeable node, so no other node needs a merge.
        if let Some(id) = stop {
            self.merge(id);
        }
    }

    /// Removes `id` when it is neither a root, an anchor nor a branch point,
    /// giving its edge to its only child. A child whose segment already
    /// covers the edge takes it for free; otherwise the edges are copied
    /// into one segment up to [`MERGE_COPY_TOKENS`], and longer ones keep `id`
    /// as a junction.
    fn merge(&mut self, id: NodeId) {
        let node = &self.nodes[&id];
        let Some(parent) = node.parent else {
            return;
        };
        if node.anchor.is_some() || node.children.len() != 1 {
            return;
        }
        self.work.nodes += 1;
        let child = *node.children.values().next().expect("one child");
        let lower = &self.nodes[&child];
        let segment = if lower.segment.start <= node.start {
            None
        } else if lower.depth - node.start <= MERGE_COPY_TOKENS {
            self.work.tokens_copied += (lower.depth - node.start) as u64;
            Some(Arc::new(Segment::joined(
                &node.segment,
                node.start,
                node.depth,
                &lower.segment,
                lower.depth,
            )))
        } else {
            return;
        };
        let node = self.nodes.remove(&id).expect("live radix node");
        self.dirty.remove(&id);
        let lower = self.nodes.get_mut(&child).expect("live radix node");
        lower.parent = Some(parent);
        lower.start = node.start;
        if let Some(segment) = segment {
            lower.segment = segment;
        }
        // The removed node forwarded exactly its child's summary and marks.
        self.nodes
            .get_mut(&parent)
            .expect("live radix node")
            .children
            .insert(node.first_atom(), child);
    }

    /// Classes and tiers of the anchors changed since the last call. A fork
    /// seen here marks its anchor shared for the rest of its lifetime.
    pub(crate) fn changes(
        &mut self,
    ) -> Vec<(PrefixCacheKeyV1, PrefixAnchorClassV1, PrefixEvictionTierV1)> {
        let mut changes = Vec::new();
        for id in std::mem::take(&mut self.dirty) {
            let Some(node) = self.nodes.get_mut(&id) else {
                continue;
            };
            let Some(key) = node.anchor else {
                continue;
            };
            if node.fork() {
                node.shared = true;
            }
            let (class, tier) = node.rank();
            changes.push((key, class, tier));
        }
        changes
    }

    pub(crate) fn rank(
        &self,
        key: PrefixCacheKeyV1,
    ) -> Option<(PrefixAnchorClassV1, PrefixEvictionTierV1)> {
        self.anchors.get(&key).map(|id| self.nodes[id].rank())
    }

    pub(crate) fn audit(&self, expected: impl Iterator<Item = (PrefixCacheKeyV1, usize)>) -> bool {
        let mut count = 0;
        for (key, depth) in expected {
            let Some(node) = self.anchors.get(&key).and_then(|id| self.nodes.get(id)) else {
                return false;
            };
            if node.depth != depth {
                return false;
            }
            count += 1;
        }
        if count != self.anchors.len() || !self.dirty.is_empty() {
            return false;
        }
        for (id, node) in &self.nodes {
            if node.children.is_empty() && node.anchor.is_none() {
                return false;
            }
            if let Some(parent) = node.parent {
                if node.start >= node.depth
                    || node.segment.start > node.start
                    || node.segment.end() < node.depth
                    || node.segment.digests.len()
                        != node.segment.end() / node.segment.block_tokens
                            - node.segment.start / node.segment.block_tokens
                    || self
                        .nodes
                        .get(&parent)
                        .and_then(|parent| parent.children.get(&node.first_atom()))
                        != Some(id)
                {
                    return false;
                }
                // A remaining single-child node is a junction between two
                // insertions' segments; one segment would have merged free.
                if node.anchor.is_none()
                    && node.children.len() == 1
                    && self
                        .nodes
                        .get(node.children.values().next().expect("one child"))
                        .is_none_or(|child| child.segment.start <= node.start)
                {
                    return false;
                }
            } else if node.depth != 0
                || node.root.and_then(|root| self.roots.get(&root)) != Some(id)
            {
                return false;
            }
            let mut targets = BTreeMap::new();
            let mut forks = 0;
            let mut marked = 0;
            for (atom, child) in &node.children {
                let Some(child) = self.nodes.get(child) else {
                    return false;
                };
                if child.parent != Some(*id)
                    || child.start != node.depth
                    || child.first_atom() != *atom
                {
                    return false;
                }
                let outgoing = child.outgoing();
                for (target, count) in outgoing.0 {
                    *targets.entry(target).or_default() += count;
                }
                forks += outgoing.1;
                marked += usize::from(child.marks());
            }
            if targets != node.targets || forks != node.forks || marked != node.marked {
                return false;
            }
            if node.anchor.is_none() && (node.extended || node.consumed || node.shared)
                || node.anchor.is_some() && node.fork() && !node.shared
            {
                return false;
            }
            if let Some(key) = node.anchor
                && self.anchors.get(&key) != Some(id)
            {
                return false;
            }
        }
        // Consumed is closed upward over anchors: every anchor above an
        // extended or consumed anchor is consumed.
        for node in self.nodes.values() {
            if node.anchor.is_none() || !(node.extended || node.consumed) {
                continue;
            }
            let mut parent = node.parent;
            while let Some(id) = parent {
                let Some(above) = self.nodes.get(&id) else {
                    return false;
                };
                if above.anchor.is_some() && !above.consumed {
                    return false;
                }
                parent = above.parent;
            }
        }
        self.roots
            .values()
            .all(|id| self.nodes.get(id).is_some_and(|node| node.parent.is_none()))
            && self.anchors.iter().all(|(key, id)| {
                self.nodes
                    .get(id)
                    .is_some_and(|node| node.anchor == Some(*key))
            })
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use crate::{KvError, KvGeometry, KvPool};

    fn fixture() -> (KvPool, PrefixChainV1, PrefixCacheKeyV1) {
        let chain = PrefixChainV1::new(4, vec![1, 2, 3, 4].into(), vec![]).unwrap();
        let mut pool = KvPool::new(KvGeometry::new(4, 16).unwrap(), 2)
            .unwrap()
            .with_prefix_ancestry(true);
        let source = pool.create_sequence(4).unwrap();
        pool.append(source, 4).unwrap();
        pool.insert_content_anchor(&chain, source, 4, 1)
            .unwrap()
            .unwrap();
        pool.release(source).unwrap();
        let key = chain.key_at(4).unwrap();
        (pool, chain, key)
    }

    #[test]
    fn pool_equality_ignores_capture_age() {
        let (pool, _, key) = fixture();
        let mut clone = pool.clone();
        clone.anchors.get_mut(&key).unwrap().created_at += std::time::Duration::from_secs(1);
        assert_eq!(pool, clone);
    }

    #[test]
    fn unsupported_content_boundaries_skip_without_pool_changes() {
        let (mut pool, _, _) = fixture();
        let source = pool.create_sequence(8).unwrap();
        pool.append(source, 8).unwrap();
        let before = pool.clone();
        let wrong_blocks = PrefixChainV1::new(8, vec![1; 8].into(), vec![]).unwrap();
        assert_eq!(
            pool.insert_content_anchor(&wrong_blocks, source, 4, 1),
            Ok(None)
        );
        let media = PrefixChainV1::new(
            4,
            vec![1; 8].into(),
            vec![crate::PrefixMediaSpanV1 {
                identity: [2; 32],
                start: 3,
                rows: 2,
                media_row: 0,
            }],
        )
        .unwrap();
        assert_eq!(pool.insert_content_anchor(&media, source, 4, 1), Ok(None));
        assert_eq!(pool, before);
        assert_eq!(pool.metrics().snapshot(), before.metrics().snapshot());
        pool.audit().unwrap();
    }

    #[test]
    fn audit_accepts_content_legacy_and_private_anchors_together() {
        let (mut pool, _, _) = fixture();
        let source = pool.create_sequence(4).unwrap();
        pool.append(source, 4).unwrap();
        pool.insert_anchor(
            PrefixCacheKeyV1::new([7; 32]),
            PrefixCacheKeyV1::new([8; 32]),
            source,
            1,
        )
        .unwrap()
        .unwrap();
        pool.insert_private_anchor(PrefixCacheKeyV1::new([9; 32]), source, 1)
            .unwrap()
            .unwrap();
        pool.release(source).unwrap();
        pool.audit().unwrap();
    }

    #[test]
    fn audit_rejects_missing_ancestry_membership() {
        let (mut pool, _, _) = fixture();
        pool.ancestry = Some(PrefixAncestry::default());
        assert_eq!(pool.audit(), Err(KvError::InvariantViolation));
    }

    #[test]
    fn audit_rejects_ancestry_without_a_pool_anchor() {
        let (mut pool, _, _) = fixture();
        let orphan = PrefixChainV1::new(4, vec![9, 8, 7, 6].into(), vec![]).unwrap();
        let index = pool.ancestry.as_mut().unwrap();
        index.insert(&orphan, 4, orphan.key_at(4).unwrap());
        index.changes();
        assert_eq!(pool.audit(), Err(KvError::InvariantViolation));
    }

    #[test]
    fn audit_rejects_indexed_private_anchors() {
        let (mut pool, _, key) = fixture();
        pool.anchors.get_mut(&key).unwrap().private = true;
        assert_eq!(pool.audit(), Err(KvError::InvariantViolation));
    }

    #[test]
    fn audit_rejects_index_depth_different_from_anchor_tokens() {
        let (mut pool, _, key) = fixture();
        // Both lengths fit the same physical page, so the page audit alone
        // cannot detect this mismatch with the content index.
        pool.anchors.get_mut(&key).unwrap().tokens = 3;
        assert_eq!(pool.audit(), Err(KvError::InvariantViolation));
    }
}
