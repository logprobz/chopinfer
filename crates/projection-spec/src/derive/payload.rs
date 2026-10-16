//! Execute a rank selection through the format plug-in and shared repacker.

use super::OperandSlice;
use crate::Error;
use kernel_core::precision::format::{PaddingTile, SliceAxis, SliceRequest, registry};
use kernel_core::precision::packed::{MatrixSlice, MatrixView, PackedMatrix};

/// Materialize the selected code/scale bytes without decoding or requantizing.
/// The view is one source expert matrix, with its original scale-grid origins.
pub fn slice(selection: &OperandSlice, source: MatrixView<'_>) -> Result<PackedMatrix, Error> {
    if selection.rows.is_empty()
        || selection.rows.iter().any(|r| r.is_empty())
        || selection.k.is_empty()
    {
        return Err(Error("slice must select nonempty row and K ranges".into()));
    }
    let weight = &selection.source.weight;
    weight.validate().map_err(|e| Error(e.to_string()))?;
    let rank = weight.logical_shape.len();
    if source.tensor != weight.codes.tensor
        || source.format != weight.format
        || source.rows as u64 != weight.logical_shape[rank - 2]
        || source.k as u64 != weight.logical_shape[rank - 1]
    {
        return Err(Error(
            "payload view disagrees with declared source matrix".into(),
        ));
    }
    for (axis, origin) in [(rank - 2, source.row_origin), (rank - 1, source.k_origin)] {
        let expected = weight
            .local_scale
            .as_ref()
            .and_then(|s| s.groups.iter().find(|g| usize::from(g.axis) == axis))
            .map_or(0, |g| g.origin);
        if expected != origin as u64 {
            return Err(Error(
                "payload view changes the imported scale grid origin".into(),
            ));
        }
    }
    let plugin = registry::by_id(selection.format).map_err(|e| Error(e.to_string()))?;
    if !plugin.accepts(source.format) {
        return Err(Error("slice plug-in disagrees with source format".into()));
    }
    let rows = selection
        .rows
        .iter()
        .flat_map(|r| r.clone())
        .map(usize::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error("slice rows exceed address space".into()))?;
    let k_start = usize::try_from(selection.k.start)
        .map_err(|_| Error("slice K offset exceeds address space".into()))?;
    let k = usize::try_from(selection.k.end - selection.k.start)
        .map_err(|_| Error("slice K extent exceeds address space".into()))?;
    let padded_rows = usize::try_from(selection.geometry.rows)
        .map_err(|_| Error("padded rows exceed address space".into()))?;
    let padded_k = usize::try_from(selection.geometry.k)
        .map_err(|_| Error("padded K exceeds address space".into()))?;
    let row_only = selection.rows.len() == 1 && k_start == 0 && k == source.k;
    let k_only = selection.rows.len() == 1 && selection.rows[0] == (0..source.rows as u64);
    if row_only || k_only {
        let (axis, range) = if row_only {
            (SliceAxis::Rows, rows[0]..rows[0] + rows.len())
        } else {
            (SliceAxis::K, k_start..k_start + k)
        };
        return plugin
            .slice(
                source,
                &SliceRequest {
                    axis,
                    range,
                    padding: PaddingTile {
                        rows: padded_rows,
                        k: padded_k,
                    },
                    nibble_order: source.nibble_order,
                },
            )
            .map_err(|e| Error(e.to_string()));
    }
    // Gathered head runs use the same repacker after the plug-in admits the format.
    source
        .repack(&MatrixSlice {
            rows,
            k_start,
            k,
            padded_rows,
            padded_k,
            nibble_order: source.nibble_order,
        })
        .map_err(|e| Error(e.to_string()))
}

/// One ordinary row-major source buffer, with its original storage dtype.
pub struct ActivationBuffer<'a> {
    pub dtype: kernel_core::contract::DType,
    pub rows: u64,
    pub columns: u64,
    pub bytes: &'a [u8],
}

/// Gather activation columns bit-for-bit and zero the target padding.
pub fn input(
    input: &super::DerivedInput,
    sources: &std::collections::BTreeMap<String, ActivationBuffer<'_>>,
) -> Result<Vec<u8>, Error> {
    let view = &input.view;
    let bits = kernel_core::precision::dtype_bits(view.dtype);
    if view.shape.len() != 2 || bits % 8 != 0 || bits == 0 {
        return Err(Error(
            "activation gather needs byte-addressable matrix storage".into(),
        ));
    }
    let width = bits / 8;
    let expected = view.shape[0]
        .checked_mul(view.shape[1])
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| Error("activation byte extent overflows".into()))?;
    if expected != view.bytes || expected > kernel_core::precision::MAX_TENSOR_BYTES {
        return Err(Error("activation view byte extent disagrees".into()));
    }
    let mut output = vec![
        0;
        usize::try_from(expected).map_err(|_| Error(
            "activation extent exceeds address space".into()
        ))?
    ];
    let mut end = 0;
    for segment in &input.segments {
        let source = sources
            .get(&segment.source)
            .ok_or_else(|| Error(format!("missing activation source `{}`", segment.source)))?;
        if segment.source_columns.is_some_and(|n| n != source.columns)
            || source.dtype != view.dtype
            || source.rows != view.shape[0]
            || segment.columns.is_empty()
            || segment.columns.end > source.columns
            || segment.target_columns.start != end
            || segment.target_columns.end > view.shape[1]
            || segment.columns.end - segment.columns.start
                != segment.target_columns.end - segment.target_columns.start
        {
            return Err(Error(
                "activation source or target segment disagrees with its view".into(),
            ));
        }
        let bytes = source
            .rows
            .checked_mul(source.columns)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| Error("activation source extent overflows".into()))?;
        if bytes != source.bytes.len() as u64 {
            return Err(Error("activation source byte length disagrees".into()));
        }
        let length = ((segment.columns.end - segment.columns.start) * width) as usize;
        for row in 0..view.shape[0] {
            let from = ((row * source.columns + segment.columns.start) * width) as usize;
            let to = ((row * view.shape[1] + segment.target_columns.start) * width) as usize;
            output[to..to + length].copy_from_slice(&source.bytes[from..from + length]);
        }
        end = segment.target_columns.end;
    }
    if end == 0 {
        return Err(Error("activation gather has no live segments".into()));
    }
    Ok(output)
}
