//! Keep the target prefix and every Engram layer graph in one context bank.
use super::Instance;
use crate::indexer_workloads::Route;
use bundle_runtime::graphs::{GraphCache, GraphKey};
use model_deepseek41_contract::serving::DeepSeek;

pub(super) struct Pair {
    target: GraphCache<DeepSeek>,
    layers: Vec<(u32, GraphCache<DeepSeek>)>,
}

impl Instance {
    pub(super) fn indexer_graph_contains(&self, route: Route, key: GraphKey) -> bool {
        if route == self.indexer_route {
            self.graphs.contains(key)
                && self.dspark.as_ref().is_none_or(|device| {
                    !device.cycle.layers.is_empty()
                        && device
                            .cycle
                            .layers
                            .iter()
                            .all(|layer| layer.graphs.contains(key))
                })
        } else if let Some(pair) = self.indexer_banks.get(&route) {
            pair.target.contains(key)
                && self.dspark.as_ref().is_none_or(|device| {
                    !pair.layers.is_empty()
                        && pair.layers.len() == device.cycle.layers.len()
                        && pair
                            .layers
                            .iter()
                            .zip(&device.cycle.layers)
                            .all(|((id, graphs), layer)| *id == layer.layer && graphs.contains(key))
                })
        } else {
            false
        }
    }

    pub(super) fn activate_indexer_graphs(&mut self, route: Route) {
        if route == self.indexer_route {
            return;
        }
        let mut next = self.indexer_banks.remove(&route).unwrap_or_else(|| Pair {
            target: GraphCache::sharing(&self.graphs),
            layers: self
                .dspark
                .iter()
                .flat_map(|device| &device.cycle.layers)
                .map(|layer| (layer.layer, GraphCache::sharing(&layer.graphs)))
                .collect(),
        });
        std::mem::swap(&mut self.graphs, &mut next.target);
        if let Some(device) = self.dspark.as_mut() {
            assert_eq!(next.layers.len(), device.cycle.layers.len());
            for ((id, graphs), layer) in next.layers.iter_mut().zip(&mut device.cycle.layers) {
                assert_eq!(*id, layer.layer);
                std::mem::swap(&mut layer.graphs, graphs);
            }
        }
        let previous = self.indexer_banks.insert(self.indexer_route, next);
        debug_assert!(previous.is_none());
        self.indexer_route = route;
        // Every bank shares the existing acquisition ledger. Switching neither
        // frees in-flight graphs nor changes streams or gather ownership.
    }
}
