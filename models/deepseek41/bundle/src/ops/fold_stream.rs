//! Private packed weights, tickets and workspace for a qualified fold operation.
use super::{Bind, Lowered, exact};
use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery, Mxfp8FoldStreamBuffers};
use model_deepseek41_contract::geometry::Geometry;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;
use std::collections::BTreeMap;

pub(crate) type Op = Prepared<Mxfp8FoldStreamBuffers<View>>;
#[derive(Clone, Copy, Debug)]
pub(crate) struct Storage {
    pub packed: [Span; 2],
    pub state: Span,
    pub workspace: Span,
}

pub(crate) fn prepare(
    backend: &dyn Backend,
    g: &Geometry,
    query: Mxfp8FoldGemmQuery,
    rows: u32,
    paired: bool,
) -> Result<Option<Op>, Refused> {
    if !crate::Kernels::stream_fold_qualified(g, query, rows, paired) {
        return Ok(None);
    }
    let query = crate::Kernels::dense_decode_query(query, rows);
    backend
        .mxfp8_fold_gemm_stream(&query, paired, exact::need(rows, exact::verify(rows)))
        .map(Some)
}

pub(crate) fn name(weight: &str, paired: bool, role: &str) -> String {
    format!(
        "{weight}.fold_{}.{}",
        if paired { "pair" } else { "single" },
        role
    )
}

pub(crate) fn storage(
    views: &BTreeMap<String, Span>,
    weight: &str,
    paired: bool,
) -> Option<Storage> {
    Some(Storage {
        packed: [
            *views.get(&name(weight, paired, "packed0"))?,
            if paired {
                *views.get(&name(weight, paired, "packed1"))?
            } else {
                Span::new(
                    runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
                    0,
                    0,
                )
            },
        ],
        state: *views.get(&name(weight, paired, "state"))?,
        workspace: views
            .get(&name(weight, paired, "workspace"))
            .copied()
            .unwrap_or_else(|| {
                Span::new(
                    runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
                    0,
                    0,
                )
            }),
    })
}

impl Storage {
    fn bind(
        self,
        bind: &mut Bind,
        first: Mxfp8FoldGemmBuffers<View>,
        second: Option<Mxfp8FoldGemmBuffers<View>>,
    ) -> Mxfp8FoldStreamBuffers<View> {
        Mxfp8FoldStreamBuffers {
            first,
            second,
            packed_weights: [
                bind.view(self.packed[0]),
                if second.is_some() {
                    bind.view(self.packed[1])
                } else {
                    View::EMPTY
                },
            ],
            state: bind.view(self.state),
        }
    }
}

pub(crate) fn launch(
    op: &Op,
    mut bind: Bind,
    first: Mxfp8FoldGemmBuffers<View>,
    second: Option<Mxfp8FoldGemmBuffers<View>>,
    storage: Option<Storage>,
    rows: u32,
) -> Result<Vec<Lowered>, Refusal> {
    let storage = storage.ok_or(Refusal("qualified fold has no initialized private storage"))?;
    let buffers = storage.bind(&mut bind, first, second);
    bind.launch_within(op, &buffers, storage.workspace, Counts::rows(rows))
}

pub(crate) fn initialize(
    op: &Op,
    weights: [Span; 2],
    storage: Storage,
    paired: bool,
) -> Result<Vec<Lowered>, Refusal> {
    let mut bind = Bind::default();
    let first = Mxfp8FoldGemmBuffers {
        activations: View::EMPTY,
        activation_scales: View::EMPTY,
        weights: bind.view(weights[0]),
        weight_scales: View::EMPTY,
        output: View::EMPTY,
        live_rows: View::EMPTY,
        status: View::EMPTY,
        status_row_offset: 0,
    };
    let second = paired.then(|| Mxfp8FoldGemmBuffers {
        weights: bind.view(weights[1]),
        ..first
    });
    let buffers = storage.bind(&mut bind, first, second);
    bind.initialize(op, &buffers)
}

pub(crate) fn allocation_bytes(op: &Op, paired: bool) -> [u64; 4] {
    [
        op.info().bytes_of("packed_weights.0").unwrap(),
        if paired {
            op.info().bytes_of("packed_weights.1").unwrap()
        } else {
            0
        },
        op.info().bytes_of("state").unwrap(),
        op.info().workspace,
    ]
}
