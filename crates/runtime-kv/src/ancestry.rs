//! Content radix index. Ownership remains entirely in the pool's page refs.
//!
//! Full-block digests shortcut radix traversal. Edges retain shared block
//! material so a later partial anchor can split them exactly, including media.
//! Fractional nominations flow toward the root until a cached ancestor at or
//! below their target absorbs them. Fork nominations stop at the first anchor.
//! Updates propagate nomination deltas, never scan all resident anchors.
//!
//! Each anchor also carries three flags for its lifetime. Extended: a
//! successful public restore found no resident descendant below it, so the
//! request continued from it. Consumed: an anchor below it was extended, so a
//! conversation has moved past it. Shared: it was a fork, or a successful
//! restore used it while it had a resident descendant. Only consumed,
//! unshared ancestors rank below recency.
use std::collections::{BTreeMap, BTreeSet};

use crate::{PrefixCacheKeyV1, PrefixChainV1, prefix::PrefixAtom};

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

type NodeId = u64;

// Store ordinary tokens densely. An optional media record per token would
// otherwise multiply the index's memory even for text-only content.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Edge {
    tokens: Vec<u32>,
    media: BTreeMap<usize, crate::PrefixMediaSpanV1>,
}

impl Edge {
    fn new(atoms: impl IntoIterator<Item = PrefixAtom>) -> Self {
        let mut edge = Self::default();
        for atom in atoms {
            if let Some(media) = atom.media {
                edge.media.insert(edge.tokens.len(), media);
            }
            edge.tokens.push(atom.token);
        }
        edge
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn atom(&self, position: usize) -> PrefixAtom {
        PrefixAtom {
            token: self.tokens[position],
            media: self.media.get(&position).copied(),
        }
    }

    fn iter(&self) -> impl Iterator<Item = PrefixAtom> + '_ {
        (0..self.len()).map(|position| self.atom(position))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Node {
    parent: Option<NodeId>,
    root: Option<PrefixCacheKeyV1>,
    edge: Edge,
    depth: usize,
    digest: Option<PrefixCacheKeyV1>,
    children: BTreeMap<PrefixAtom, NodeId>,
    anchor: Option<PrefixCacheKeyV1>,
    // Unabsorbed nominations arriving from children, with multiplicity.
    targets: BTreeMap<usize, usize>,
    forks: usize,
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefixAncestry {
    roots: BTreeMap<PrefixCacheKeyV1, NodeId>,
    full: BTreeMap<PrefixCacheKeyV1, NodeId>,
    anchors: BTreeMap<PrefixCacheKeyV1, NodeId>,
    nodes: BTreeMap<NodeId, Node>,
    next: NodeId,
    dirty: BTreeSet<NodeId>,
}

impl PrefixAncestry {
    fn allocate(&mut self, node: Node) -> NodeId {
        self.next += 1;
        let id = self.next;
        self.nodes.insert(id, node);
        id
    }

    fn target_delta(&mut self, mut id: NodeId, target: usize, count: usize, add: bool) {
        loop {
            let node = self.nodes.get_mut(&id).expect("live radix node");
            let value = node.targets.entry(target).or_default();
            if add {
                *value += count;
            } else {
                *value -= count;
            }
            if *value == 0 {
                node.targets.remove(&target);
            }
            self.dirty.insert(id);
            if node.anchor.is_some() && target >= node.depth {
                break;
            }
            let Some(parent) = node.parent else {
                break;
            };
            id = parent;
        }
    }

    fn fork_delta(&mut self, mut id: NodeId, count: usize, add: bool) {
        loop {
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

    // Structural changes affect only this node's outgoing nominations. Ordinary
    // propagation above changes one counted target at a time along its path.
    fn modify(&mut self, id: NodeId, change: impl FnOnce(&mut Node)) {
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
        let changed: Vec<_> = if old_anchor != new_anchor {
            node.targets
                .range(depth..)
                .map(|(target, count)| (*target, *count))
                .collect()
        } else {
            Vec::new()
        };
        for (target, count) in changed {
            self.target_delta(parent, target, count, !new_anchor);
        }
        if old_maximal != new_maximal {
            for divisor in [2, 4, 8] {
                self.target_delta(parent, depth / divisor, 1, new_maximal);
            }
        }
        if old_forks != new_forks {
            self.fork_delta(parent, old_forks.abs_diff(new_forks), new_forks > old_forks);
        }
    }

    fn split(&mut self, child: NodeId, common: usize) -> NodeId {
        let old = &self.nodes[&child];
        let parent = old.parent.expect("a root has no edge");
        let edge = Edge::new(old.edge.iter().take(common));
        let depth = old.depth - old.edge.len() + common;
        let (targets, forks) = old.outgoing();
        let first = old.edge.atom(common);
        let middle = self.allocate(Node {
            parent: Some(parent),
            edge: edge.clone(),
            depth,
            targets,
            forks,
            children: BTreeMap::from([(first, child)]),
            ..Node::default()
        });
        let node = self.nodes.get_mut(&child).unwrap();
        node.parent = Some(middle);
        node.edge = Edge::new(node.edge.iter().skip(common));
        // The intermediate node forwards exactly the old child's summary.
        self.nodes
            .get_mut(&parent)
            .unwrap()
            .children
            .insert(edge.atom(0), middle);
        middle
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
        let block_tokens = chain.block_tokens();
        let mut start = 0;
        while start < tokens {
            let end = (start + block_tokens).min(tokens);
            let digest = end
                .is_multiple_of(block_tokens)
                .then(|| chain.chain(end / block_tokens).unwrap());
            if let Some(id) = digest.and_then(|digest| self.full.get(&digest)) {
                node = *id;
                start = end;
                continue;
            }
            let atoms = chain.ancestry_atoms(start, end);
            let mut offset = 0;
            while offset < atoms.len() {
                let Some(child) = self.nodes[&node].children.get(&atoms[offset]).copied() else {
                    let id = self.allocate(Node {
                        parent: Some(node),
                        edge: Edge::new(atoms[offset..].iter().copied()),
                        depth: end,
                        ..Node::default()
                    });
                    self.modify(node, |parent| {
                        parent.children.insert(atoms[offset], id);
                    });
                    node = id;
                    break;
                };
                let common = self.nodes[&child]
                    .edge
                    .iter()
                    .zip(atoms[offset..].iter().copied())
                    .take_while(|(a, b)| a == b)
                    .count();
                node = if common < self.nodes[&child].edge.len() {
                    self.split(child, common)
                } else {
                    child
                };
                offset += common;
            }
            if let Some(digest) = digest {
                self.nodes.get_mut(&node).unwrap().digest = Some(digest);
                self.full.insert(digest, node);
            }
            start = end;
        }
        let consumed = self.consumed_below(node);
        self.modify(node, |node| {
            node.anchor = Some(key);
            node.extended = false;
            node.consumed = consumed;
            node.shared = false;
        });
        self.anchors.insert(key, node);
    }

    /// Whether an anchor below `id` was extended or moved past. Consumed is
    /// closed upward over anchors, so the search stops at the nearest
    /// anchors and returns at the first one that answers it.
    fn consumed_below(&self, id: NodeId) -> bool {
        let mut stack: Vec<NodeId> = self.nodes[&id].children.values().copied().collect();
        while let Some(id) = stack.pop() {
            let node = &self.nodes[&id];
            if node.anchor.is_some() {
                if node.extended || node.consumed {
                    return true;
                }
            } else {
                stack.extend(node.children.values().copied());
            }
        }
        false
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
        self.nodes.get_mut(&id).expect("live radix node").extended = true;
        let mut parent = self.nodes[&id].parent;
        while let Some(id) = parent {
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
        let Some(mut id) = self.anchors.remove(&key) else {
            return;
        };
        self.modify(id, |node| {
            node.anchor = None;
            node.extended = false;
            node.consumed = false;
            node.shared = false;
        });
        while self.nodes[&id].anchor.is_none() && self.nodes[&id].children.is_empty() {
            let node = self.nodes.remove(&id).unwrap();
            self.dirty.remove(&id);
            if let Some(digest) = node.digest {
                self.full.remove(&digest);
            }
            let Some(parent) = node.parent else {
                self.roots.remove(&node.root.expect("root identity"));
                break;
            };
            self.modify(parent, |parent| {
                parent.children.remove(&node.edge.atom(0));
            });
            id = parent;
        }
        // Remove obsolete tail splits. Block boundaries remain digest shortcuts.
        while let Some(parent) = self.nodes.get(&id).and_then(|node| node.parent) {
            let node = &self.nodes[&id];
            if node.anchor.is_none() && node.digest.is_none() && node.children.len() == 1 {
                let child = *node.children.first_key_value().unwrap().1;
                let edge = node.edge.clone();
                let first = edge.atom(0);
                let child_node = self.nodes.get_mut(&child).unwrap();
                child_node.edge = Edge::new(edge.iter().chain(child_node.edge.iter()));
                child_node.parent = Some(parent);
                self.nodes
                    .get_mut(&parent)
                    .unwrap()
                    .children
                    .insert(first, child);
                self.nodes.remove(&id);
                self.dirty.remove(&id);
            }
            id = parent;
        }
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
                if node.edge.len() == 0
                    || self
                        .nodes
                        .get(&parent)
                        .and_then(|parent| parent.children.get(&node.edge.atom(0)))
                        != Some(id)
                {
                    return false;
                }
            } else if node.root.and_then(|root| self.roots.get(&root)) != Some(id) {
                return false;
            }
            let mut targets = BTreeMap::new();
            let mut forks = 0;
            for (atom, child) in &node.children {
                let Some(child) = self.nodes.get(child) else {
                    return false;
                };
                if child.parent != Some(*id)
                    || child.edge.len() == 0
                    || child.edge.atom(0) != *atom
                    || child.depth != node.depth + child.edge.len()
                {
                    return false;
                }
                let outgoing = child.outgoing();
                for (target, count) in outgoing.0 {
                    *targets.entry(target).or_default() += count;
                }
                forks += outgoing.1;
            }
            if targets != node.targets || forks != node.forks {
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
            if let Some(digest) = node.digest
                && self.full.get(&digest) != Some(id)
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
