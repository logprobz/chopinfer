//! Keep the target prefix and every Engram layer graph in one context bank.
use super::Instance;
use crate::dspark::cycle::LayerGather;
use crate::indexer_workloads::Route;
use bundle_runtime::graphs::{GraphCache, GraphKey};
use model_deepseek41_contract::serving::DeepSeek;

pub(super) struct Pair {
    target: GraphCache<DeepSeek>,
    layers: Vec<(u32, GraphCache<DeepSeek>)>,
}

/// The bank a run's graphs live in: its indexer route, and whether it is an
/// allocated verify step, whose features and head gather into the reserved
/// rows, so its programs differ from an unallocated step's of the same key.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Bank {
    pub(crate) route: Route,
    pub(crate) allocated: bool,
}

impl Bank {
    /// The bank of an unallocated run on `route`.
    pub(crate) const fn of(route: Route) -> Self {
        Self {
            route,
            allocated: false,
        }
    }
}

impl Instance {
    /// Every prepared layer gather whose graphs the banks hold: DSpark's
    /// target layers, or AR's decode layers (`super::ar_engram`).
    fn layer_gathers(&self) -> &[LayerGather] {
        match (&self.dspark, &self.ar_engram) {
            (Some(device), _) => &device.cycle.layers,
            (None, Some(state)) => &state.layers,
            (None, None) => &[],
        }
    }

    fn layer_gathers_mut(&mut self) -> &mut [LayerGather] {
        match (&mut self.dspark, &mut self.ar_engram) {
            (Some(device), _) => &mut device.cycle.layers,
            (None, Some(state)) => &mut state.layers,
            (None, None) => &mut [],
        }
    }

    /// The layer gathers a target graph of `key` joins: DSpark's at every
    /// key, AR's at the decode buckets it serves, none otherwise.
    fn joined_layers(&self, key: GraphKey) -> Option<&[LayerGather]> {
        if let Some(device) = &self.dspark {
            return Some(&device.cycle.layers);
        }
        self.ar_engram
            .as_ref()
            .filter(|state| state.serves((key.bucket, key.items)))
            .map(|state| state.layers.as_slice())
    }

    /// Count uncaptured pieces in the bank that will execute this run. All
    /// banks share the same ledger, but their cached keys are independent.
    pub(super) fn indexer_graph_missing(&self, bank: Bank, key: GraphKey) -> usize {
        let joined = self.joined_layers(key);
        if bank == self.graph_bank {
            usize::from(!self.graphs.contains(key))
                + joined.map_or(0, |layers| {
                    layers
                        .iter()
                        .filter(|layer| !layer.graphs.contains(key))
                        .count()
                })
        } else if let Some(pair) = self.indexer_banks.get(&bank) {
            usize::from(!pair.target.contains(key))
                + joined.map_or(0, |_| {
                    pair.layers
                        .iter()
                        .filter(|(_, graphs)| !graphs.contains(key))
                        .count()
                })
        } else {
            1 + joined.map_or(0, <[_]>::len)
        }
    }

    pub(super) fn indexer_graph_contains(&self, bank: Bank, key: GraphKey) -> bool {
        let joined = self.joined_layers(key);
        if bank == self.graph_bank {
            self.graphs.contains(key)
                && joined.is_none_or(|layers| {
                    !layers.is_empty() && layers.iter().all(|layer| layer.graphs.contains(key))
                })
        } else if let Some(pair) = self.indexer_banks.get(&bank) {
            pair.target.contains(key)
                && joined.is_none_or(|layers| {
                    !pair.layers.is_empty()
                        && pair.layers.len() == layers.len()
                        && pair
                            .layers
                            .iter()
                            .zip(layers)
                            .all(|((id, graphs), layer)| *id == layer.layer && graphs.contains(key))
                })
        } else {
            false
        }
    }

    pub(super) fn activate_indexer_graphs(&mut self, bank: Bank) {
        if bank == self.graph_bank {
            return;
        }
        let mut next = self.indexer_banks.remove(&bank).unwrap_or_else(|| Pair {
            target: GraphCache::sharing(&self.graphs),
            layers: self
                .layer_gathers()
                .iter()
                .map(|layer| (layer.layer, GraphCache::sharing(&layer.graphs)))
                .collect(),
        });
        std::mem::swap(&mut self.graphs, &mut next.target);
        let layers = self.layer_gathers_mut();
        assert_eq!(next.layers.len(), layers.len());
        for ((id, graphs), layer) in next.layers.iter_mut().zip(layers) {
            assert_eq!(*id, layer.layer);
            std::mem::swap(&mut layer.graphs, graphs);
        }
        let previous = self.indexer_banks.insert(self.graph_bank, next);
        debug_assert!(previous.is_none());
        self.graph_bank = bank;
        // Every bank shares the existing acquisition ledger. Switching neither
        // frees in-flight graphs nor changes streams or gather ownership.
    }
}
