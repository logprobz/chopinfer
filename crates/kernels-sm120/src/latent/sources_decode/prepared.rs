//! Exact-row routing over the validated source binding and its ordered merge.
use super::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::latent_attention::LatentSourcesBuffers;
use std::collections::BTreeMap;

/// A model supplies only GPU-qualified row/entry pairs. Unlisted row counts
/// keep the reference. Preparation declares every possible shared-memory
/// opt-in; the merge, workspace and numerical guarantees stay unchanged.
pub fn prepare(
    reference: Prepared<LatentSourcesBuffers<View>>,
    need: Need,
    routes: &[(u32, Entry)],
) -> Result<Prepared<LatentSourcesBuffers<View>>, Refused> {
    need.check(reference.info())?;
    if routes.is_empty() {
        return Ok(reference);
    }
    let mut selected = BTreeMap::new();
    let mut info = reference.info().clone();
    for (rows, entry) in routes {
        entry.validate()?;
        if *rows == 0 || *rows > need.capacity || entry.profile {
            return Err(
                Refusal("decode routes need admitted rows and uninstrumented entries").into(),
            );
        }
        if reference.info().symbols.len() != 2
            || reference.info().symbols[0].0 != entry.source.symbol()
            || reference.info().symbols[1].0 != entry.source.merge_symbol()
        {
            return Err(Refusal("decode route differs from its prepared source or merge").into());
        }
        if selected.insert(*rows, entry.clone()).is_some() {
            return Err(Refusal("decode route repeats a row count").into());
        }
        let symbol = entry.symbol();
        let shared = entry.plan(*rows)?.dynamic_shared_bytes;
        if let Some((_, bytes)) = info.symbols.iter_mut().find(|(name, _)| *name == symbol) {
            *bytes = (*bytes).max(shared);
        } else {
            info.symbols
                .insert(info.symbols.len() - 1, (symbol, shared));
        }
    }
    let digest = kernel_build::digest(format!("{selected:?}").as_bytes());
    info.implementation
        .push_str(&format!(":decode-staging#{}", &digest[..16]));
    need.check(&info)?;
    Ok(Box::new(RoutedSources {
        reference,
        selected,
        info,
    }))
}

struct RoutedSources {
    reference: Prepared<LatentSourcesBuffers<View>>,
    selected: BTreeMap<u32, Entry>,
    info: OpInfo,
}

impl PreparedOp<LatentSourcesBuffers<View>> for RoutedSources {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &LatentSourcesBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        // The reference still checks every view, stride, budget and live count.
        let mut calls = self.reference.launches(buffers, workspace, counts)?;
        if let Some(entry) = self.selected.get(&counts.rows) {
            if calls.len() != 2
                || calls[1].symbol != entry.source.merge_symbol()
                || calls[1].plan != entry.source.merge_plan(counts.rows)?
            {
                return Err(Refusal(
                    "decode routing requires the original ordered merge",
                ));
            }
            calls[0] = entry.launch(&calls[0], None)?;
        }
        Ok(calls)
    }
}
