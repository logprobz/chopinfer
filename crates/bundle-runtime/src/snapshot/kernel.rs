//! A snapshot's pieces as one launch of a copy kernel
//! (`kernel_core::snapshot_copy`): the table's descriptors become the
//! launch's segments over the spans that hold them, and one rank's actions
//! upload the page IDs and segments, launch the kernel and then write or
//! read the header, as [`Table::actions`] does after its device copies.
//! Everything is checked and bound before any rank submits: the pieces'
//! buffers and bounds, the pages, the segment count and view count the
//! kernel was prepared for, and that no two of the pieces' bytes, the
//! snapshot's pages and the table buffer overlap. The kernel is the
//! bundle's prepared op of the contract, so nothing here names a backend.

use kernel_core::backend::{Counts, Launch, PreparedOp, View};
use kernel_core::contract::Refusal;
use kernel_core::snapshot_copy::{
    Direction, MAX_VIEWS, Segment, SnapshotCopy, SnapshotCopyBuffers,
};
use runtime_bundle_abi::{
    BundleCudaSymbolHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1,
    KernelArgumentV1,
};

use super::{Action, HeaderAction, HeldBuffers, Pages, Table};
use crate::dispatch::{BufferBindings, DispatchError};

/// The copy kernel of one direction, prepared for a snapshot's pages and
/// views, and where one rank uploads the table a launch reads.
pub struct KernelCopy<'a> {
    pub op: SnapshotCopy,
    pub prepared: &'a dyn PreparedOp<SnapshotCopyBuffers<View>>,
    /// The rank's table buffer: at least the op's page IDs and the most
    /// segments it was prepared for.
    pub table: DeviceBufferSpanV1,
}

/// A copy kernel launch bound to one rank's spans.
#[derive(Clone, Debug)]
pub struct BoundLaunch {
    pub launch: Launch,
    pub bindings: BufferBindings,
}

impl BoundLaunch {
    /// The launch's ABI arguments.
    pub fn arguments(&self) -> Result<Vec<KernelArgumentV1>, DispatchError> {
        crate::dispatch::arguments(&self.launch.arguments, &self.bindings)
    }

    /// Launches it on `stream` through `symbols`, the rank's resolved
    /// entries.
    pub fn submit(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        symbols: &std::collections::BTreeMap<String, BundleCudaSymbolHandleV1>,
    ) -> Result<(), DispatchError> {
        crate::dispatch::launch(
            services,
            stream,
            symbols,
            &self.launch.symbol,
            self.launch.plan,
            &self.launch.arguments,
            &self.bindings,
        )
    }
}

impl Table {
    /// The pieces as copy kernel segments, in snapshot order, and the spans
    /// their views bind, each declared buffer (or spare) once in the order
    /// the pieces first name it.
    pub fn segments(
        &self,
        held: &HeldBuffers,
    ) -> Result<(Vec<DeviceBufferSpanV1>, Vec<Segment>), Refusal> {
        let mut named = Vec::new();
        let mut views = Vec::new();
        let mut segments = Vec::with_capacity(self.descriptors.len());
        for descriptor in &self.descriptors {
            let span = held
                .span(descriptor.buffer, descriptor.spare)
                .ok_or(Refusal("a snapshot piece's buffer is not held"))?;
            if descriptor
                .offset
                .checked_add(descriptor.length)
                .is_none_or(|end| end > span.length)
            {
                return Err(Refusal("a snapshot piece lies outside its buffer"));
            }
            let key = (descriptor.buffer, descriptor.spare);
            let view = match named.iter().position(|&named| named == key) {
                Some(view) => view,
                None => {
                    named.push(key);
                    views.push(span);
                    views.len() - 1
                }
            };
            if views.len() > MAX_VIEWS as usize {
                return Err(Refusal("a snapshot copy kernel binds at most eight views"));
            }
            segments.push(Segment {
                view: view as u32,
                offset: descriptor.offset,
                at: descriptor.at,
                length: descriptor.length,
            });
        }
        Ok((views, segments))
    }

    /// One rank's actions with `kernel` moving every piece: the table
    /// upload, the launch, then `header`'s actions. The pages must be the
    /// snapshot's (`Pages::check`). Refuses, with nothing to submit, a
    /// kernel prepared for another direction, page size, page count or view
    /// count; a piece outside its buffer; a table past the kernel's
    /// capacity or its buffer; and any overlap between the pieces' bytes,
    /// the snapshot's pages and the table.
    pub fn kernel_actions(
        &self,
        held: &HeldBuffers,
        pages: &Pages,
        to_pages: bool,
        header: HeaderAction,
        kernel: &KernelCopy,
    ) -> Result<Vec<Action<BoundLaunch>>, Refusal> {
        let op = kernel.op;
        let direction = if to_pages {
            Direction::ToPages
        } else {
            Direction::FromPages
        };
        if op.direction != direction
            || op.page_bytes != self.page_bytes
            || pages.page_bytes != self.page_bytes
            || pages.ids.len() as u64 != self.pages()
            || u64::from(op.pages) != self.pages()
        {
            return Err(Refusal(
                "the snapshot copy kernel is prepared for another snapshot",
            ));
        }
        let (views, segments) = self.segments(held)?;
        if segments.is_empty() {
            return Err(Refusal("a snapshot copy kernel needs pieces to copy"));
        }
        if views.len() as u64 != u64::from(op.views) {
            return Err(Refusal(
                "the snapshot copy kernel is prepared for another view count",
            ));
        }
        let bytes = op.encode(pages.ids, &segments)?;
        if bytes.len() as u64 > kernel.table.length {
            return Err(Refusal("a snapshot copy table exceeds its buffer"));
        }
        let table =
            DeviceBufferSpanV1::new(kernel.table.buffer, kernel.table.offset, bytes.len() as u64);
        check_disjoint(&views, &segments, pages, table)?;
        let view_bytes = views.iter().map(|view| view.length).collect::<Vec<_>>();
        op.check(pages.ids, &segments, &view_bytes, pages.pool.length)?;
        let mut bindings = BufferBindings::default();
        let mut bind = |span: DeviceBufferSpanV1| View {
            handle: bindings.bind(span),
            bytes: span.length,
        };
        let buffers = SnapshotCopyBuffers {
            views: views.iter().map(|&span| bind(span)).collect(),
            pool: bind(pages.pool),
            table: bind(table),
        };
        let mut launches =
            kernel
                .prepared
                .launches(&buffers, View::EMPTY, Counts::rows(segments.len() as u32))?;
        if launches.len() != 1
            || launches[0].collective
            || launches[0].persistent
            || launches[0].plan.workspace_bytes != 0
        {
            return Err(Refusal("a snapshot copy kernel is one local launch"));
        }
        let mut actions = vec![
            Action::Upload { to: table, bytes },
            Action::Launch(BoundLaunch {
                launch: launches.remove(0),
                bindings,
            }),
        ];
        actions.extend(pages.header_actions(self, header)?);
        Ok(actions)
    }
}

/// Refuses a launch whose bytes alias: any two of the segments' bytes in
/// their views, the snapshot's pages in the pool and the table overlap.
fn check_disjoint(
    views: &[DeviceBufferSpanV1],
    segments: &[Segment],
    pages: &Pages,
    table: DeviceBufferSpanV1,
) -> Result<(), Refusal> {
    let range = |span: DeviceBufferSpanV1, offset: u64, length: u64| {
        let start = span
            .offset
            .checked_add(offset)
            .ok_or(Refusal("a snapshot copy range overflows"))?;
        let end = start
            .checked_add(length)
            .ok_or(Refusal("a snapshot copy range overflows"))?;
        Ok::<_, Refusal>((span.buffer.raw, start, end))
    };
    let mut ranges = Vec::with_capacity(segments.len() + pages.ids.len() + 1);
    for segment in segments {
        ranges.push(range(
            views[segment.view as usize],
            segment.offset,
            segment.length,
        )?);
    }
    for &page in pages.ids {
        let at = page
            .checked_mul(pages.page_bytes)
            .ok_or(Refusal("a snapshot copy range overflows"))?;
        ranges.push(range(pages.pool, at, pages.page_bytes)?);
    }
    ranges.push(range(table, 0, table.length)?);
    ranges.sort_unstable();
    if ranges
        .windows(2)
        .any(|pair| pair[0].0 == pair[1].0 && pair[1].1 < pair[0].2)
    {
        return Err(Refusal("a snapshot copy's pieces, pages and table overlap"));
    }
    Ok(())
}

/// The device copies a launch of a snapshot copy kernel with ABI
/// `arguments` makes, as `(source, destination)` in snapshot order, its
/// table read by `read`: the copies [`Table::copies`] issues for the same
/// pieces. A fake host runs the kernel through these.
#[cfg(any(test, feature = "test-support"))]
pub fn launched_copies(
    arguments: &[KernelArgumentV1],
    read: impl FnOnce(DeviceBufferSpanV1) -> Vec<u8>,
) -> Result<Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)>, Refusal> {
    let [
        KernelArgumentV1::DevicePointerArray(views),
        KernelArgumentV1::DeviceBuffer(pool),
        KernelArgumentV1::DeviceBuffer(table),
        KernelArgumentV1::ScalarU32(segments),
        KernelArgumentV1::ScalarU32(pages),
        KernelArgumentV1::ScalarU64(page_bytes),
        KernelArgumentV1::ScalarU32(to_pages),
    ] = arguments
    else {
        return Err(Refusal("not a snapshot copy launch"));
    };
    let views = views
        .iter()
        .map(|view| view.ok_or(Refusal("a snapshot copy view is null")))
        .collect::<Result<Vec<_>, _>>()?;
    let op = SnapshotCopy {
        direction: if *to_pages != 0 {
            Direction::ToPages
        } else {
            Direction::FromPages
        },
        views: views.len() as u32,
        page_bytes: *page_bytes,
        pages: *pages,
    };
    let (ids, listed) = op.decode(&read(*table), *segments)?;
    let view_bytes = views.iter().map(|view| view.length).collect::<Vec<_>>();
    op.check(&ids, &listed, &view_bytes, pool.length)?;
    Ok(op
        .fragments(&ids, &listed)?
        .into_iter()
        .map(|fragment| {
            let view = views[fragment.view as usize];
            let held = DeviceBufferSpanV1::new(
                view.buffer,
                view.offset + fragment.offset,
                fragment.length,
            );
            let paged =
                DeviceBufferSpanV1::new(pool.buffer, pool.offset + fragment.pool, fragment.length);
            if *to_pages != 0 {
                (held, paged)
            } else {
                (paged, held)
            }
        })
        .collect())
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
