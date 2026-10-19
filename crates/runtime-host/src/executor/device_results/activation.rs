//! Trace actual result collection, separately from capability admission.
use super::DeviceResults;

#[derive(Debug)]
pub(super) struct ActivationReceipt {
    epoch: u64,
    items: usize,
    rows: usize,
    verify_items: usize,
    prefill_items: usize,
    packets: usize,
    acceptance: bool,
}

impl ActivationReceipt {
    pub(super) fn new(device: &DeviceResults, prefill_items: usize) -> Option<Self> {
        let acceptance = device.completion.is_some();
        let packets = device.ready.len() + usize::from(acceptance);
        (packets > 0).then(|| Self {
            epoch: device.metadata.cycle_epoch,
            items: device.metadata.items.len(),
            rows: device.expanded.len(),
            verify_items: device
                .metadata
                .items
                .iter()
                .filter(|item| item.acceptance.draft_tokens > 0)
                .count(),
            prefill_items,
            packets,
            acceptance,
        })
    }

    pub(super) fn record(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": "chopinfer.device-results.v1",
            "cycle_epoch": self.epoch,
            "kind": if self.acceptance { "acceptance_packet" } else { "ready_logits" },
            "items": self.items,
            "sample_rows": self.rows,
            "verify_items": self.verify_items,
            "prefill_items": self.prefill_items,
            "readback_packets": self.packets,
        })
    }
}
