//! Prepared indexed retention over the existing checked copy-list bindings.
use super::{
    device::{DECODE_BASE, TARGET_ROWS, slice},
    features::{Copy, Layout},
};
use crate::ops::{Bind, Lowered};
use kernel_core::backend::{Counts, Need, Prepared as Operation, View};
use model_deepseek41_contract::own_ops::{
    OwnOps,
    retain_rows::{MAX_SEGMENTS, RetainRows, RetainRowsBuffers, Segment},
};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

pub(crate) const METADATA_BYTES: u64 = MAX_SEGMENTS as u64 * 12;
pub(crate) const MIN_COPIES: usize = 8;

pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    return ENABLED.get();
    #[cfg(not(test))]
    std::env::var_os("CHOPINFER_DSPARK_INDEXED_RETENTION").is_some_and(|value| value == "1")
}

#[cfg(test)]
thread_local! { static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
pub(crate) fn with_enabled<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.set(self.0);
        }
    }
    let _restore = Restore(ENABLED.replace(enabled));
    run()
}

pub(crate) struct Prepared {
    pub query: RetainRows,
    pub op: Operation<RetainRowsBuffers<View>>,
}

impl Prepared {
    pub fn new(own: &dyn OwnOps, layout: Layout) -> Result<Self, String> {
        let query = RetainRows {
            row_bytes: u32::try_from(layout.row_bytes())
                .map_err(|_| "retention row exceeds U32")?,
            source_rows: TARGET_ROWS - DECODE_BASE,
            destination_rows: u32::try_from(layout.retained_bytes()? / layout.row_bytes())
                .map_err(|_| "retention destination rows exceed U32")?,
        };
        let op = own
            .dspark_retain_rows(&query, Need::rows(MAX_SEGMENTS))
            .map_err(|error| error.to_string())?;
        Ok(Self { query, op })
    }

    pub fn bind(
        &self,
        retained: Span,
        packed: Span,
        metadata: Span,
        copies: &[Copy],
    ) -> Result<(Vec<u8>, Lowered), String> {
        let source = slice(
            packed,
            u64::from(DECODE_BASE) * u64::from(self.query.row_bytes),
            self.query.source_bytes(),
        )?;
        if retained.length != self.query.destination_bytes() || metadata.length != METADATA_BYTES {
            return Err("indexed retention extents differ from prepared geometry".into());
        }
        for (a, b) in [(source, retained), (source, metadata), (retained, metadata)] {
            let a_end = a
                .offset
                .checked_add(a.length)
                .ok_or("retention span overflows")?;
            let b_end = b
                .offset
                .checked_add(b.length)
                .ok_or("retention span overflows")?;
            if a.buffer == b.buffer && a.offset < b_end && b.offset < a_end {
                return Err("indexed retention buffers overlap".into());
            }
        }
        let row_bytes = u64::from(self.query.row_bytes);
        let relative = |span: Span, base: Span| -> Result<u32, String> {
            let offset = span
                .offset
                .checked_sub(base.offset)
                .ok_or("retention copy precedes its base")?;
            if span.buffer != base.buffer
                || span.length == 0
                || !offset.is_multiple_of(row_bytes)
                || !span.length.is_multiple_of(row_bytes)
                || offset
                    .checked_add(span.length)
                    .is_none_or(|end| end > base.length)
            {
                return Err("retention copy differs from its prepared byte extent".into());
            }
            u32::try_from(offset / row_bytes).map_err(|_| "retention row exceeds U32".into())
        };
        let segments = copies
            .iter()
            .map(|copy| {
                if copy.source.length != copy.destination.length {
                    return Err("retention copy lengths differ".to_owned());
                }
                Ok(Segment {
                    source: relative(copy.source, source)?,
                    destination: relative(copy.destination, retained)?,
                    rows: u32::try_from(copy.source.length / row_bytes)
                        .map_err(|_| "retention count exceeds U32")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let encoded = self
            .query
            .encode(&segments)
            .map_err(|error| error.to_string())?;
        let mut bind = Bind::default();
        let buffers = RetainRowsBuffers {
            source: bind.view(source),
            segments: bind.view(metadata),
            destination: bind.view(retained),
        };
        let mut calls = bind
            .launch(&self.op, &buffers, Counts::rows(segments.len() as u32))
            .map_err(|error| error.to_string())?;
        if calls.len() != 1 || calls[0].collective {
            return Err("indexed retention must bind one local launch".into());
        }
        Ok((encoded, calls.remove(0)))
    }
}
