//! Tail rotation and physical window addressing through additive entries.
use super::select::{Candidate, choose};
use crate::latent::glue::{Entry, Kernel};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, NumericalContract, Refusal, Validate};
use kernel_core::latent_glue::{
    LatentQueryRope, LatentQueryRopeBuffers, LatentWindowKeyBuffers, LatentWindowKeys,
    WindowAddressing,
};

fn select(entries: &[Entry], kernel: Kernel, need: Need) -> Result<(Entry, OpInfo), Refused> {
    if need.capacity == 0 || need.capacity > super::MAX_CAPACITY {
        return Err(Refusal("latent glue capacity exceeds SM120 bounds").into());
    }
    let chosen = choose(
        "latent glue",
        entries.iter().filter(|e| e.kernel == kernel).map(|e| {
            Candidate::new(
                e.clone(),
                &e.symbol,
                format!("{:?}", e.spec()),
                e.invariance(),
            )
            .capacity(e.capacity)
        }),
        need,
    )?;
    let entry = chosen.entry;
    super::check_plan(&entry.plan(need.capacity, 1)?)?;
    let info = OpInfo::new(
        format!("sm120.latent-glue.{kernel:?}.v1"),
        chosen.guarantees,
        vec![(entry.symbol.clone(), 0)],
    );
    Ok((entry, info))
}

fn pointer(view: View, bytes: u64) -> Result<Argument, Refusal> {
    view.covers(bytes)?;
    Ok(Argument::DeviceBuffer(view.handle))
}

fn launch(
    entry: &Entry,
    capacity: u32,
    counts: Counts,
    heads: u32,
    args: Vec<Argument>,
) -> Result<Vec<Launch>, Refusal> {
    if counts.rows > capacity {
        return Err(Refusal("latent glue launch exceeds prepared rows"));
    }
    let arguments = ArgumentList(args);
    arguments.validate()?;
    Ok(vec![Launch {
        symbol: entry.symbol.clone(),
        rows: counts.rows,
        plan: entry.plan(counts.rows, heads)?,
        arguments,
        collective: false,
        persistent: false,
    }])
}

pub(super) fn query_rope(
    entries: &[Entry],
    op: &LatentQueryRope,
    need: Need,
) -> Result<Prepared<LatentQueryRopeBuffers<View>>, Refused> {
    op.validate()?;
    if op.latent != 512 || op.rotary != 64 {
        return Err(
            Refusal("SM120 latent query glue requires a 64-channel tail in 512 channels").into(),
        );
    }
    let (entry, mut info) = select(entries, Kernel::QueryRope, need)?;
    super::check_plan(&entry.plan(need.capacity, op.heads)?)?;
    let elements = u64::from(need.capacity) * u64::from(op.heads) * 512;
    info.bytes = vec![
        ("query".into(), elements * 4),
        ("native_q".into(), elements * 2),
        ("cos_sin".into(), u64::from(need.capacity) * 64 * 4),
        ("live_rows".into(), 4),
    ];
    info.implementation
        .push_str(&format!(":{}", op.numerical_contract_id()?));
    need.check(&info)?;
    Ok(Box::new(QueryOp {
        entry,
        heads: op.heads,
        capacity: need.capacity,
        info,
    }))
}

struct QueryOp {
    entry: Entry,
    heads: u32,
    capacity: u32,
    info: OpInfo,
}
impl PreparedOp<LatentQueryRopeBuffers<View>> for QueryOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentQueryRopeBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if [b.query.handle, b.cos_sin.handle, b.live_rows.handle].contains(&b.native_q.handle)
            || b.query.handle == b.cos_sin.handle
            || b.query.handle == b.live_rows.handle
        {
            return Err(Refusal(
                "latent query F32 input, BF16 output and tables must not overlap",
            ));
        }
        let elements = u64::from(counts.rows) * u64::from(self.heads) * 512;
        launch(
            &self.entry,
            self.capacity,
            counts,
            self.heads,
            vec![
                pointer(b.query, elements * 4)?,
                pointer(b.native_q, elements * 2)?,
                pointer(b.cos_sin, u64::from(counts.rows) * 256)?,
                pointer(b.live_rows, 4)?,
                Argument::U32(self.heads),
            ],
        )
    }
}

pub(super) fn window_keys(
    entries: &[Entry],
    op: &LatentWindowKeys,
    need: Need,
) -> Result<Prepared<LatentWindowKeyBuffers<View>>, Refused> {
    op.validate()?;
    let (entry, mut info) = select(entries, Kernel::WindowKeys, need)?;
    info.bytes = vec![
        ("positions".into(), u64::from(need.capacity) * 4),
        (
            "key_lists".into(),
            u64::from(need.capacity) * u64::from(op.window) * 4,
        ),
        ("live_rows".into(), 4),
    ];
    if matches!(op.addressing, WindowAddressing::Staged { .. }) {
        info.bytes.extend([
            ("row_batch".into(), u64::from(need.capacity) * 4),
            ("first_positions".into(), 4),
        ]);
    }
    info.implementation
        .push_str(&format!(":{}", op.numerical_contract_id()?));
    need.check(&info)?;
    Ok(Box::new(KeysOp {
        entry,
        query: *op,
        capacity: need.capacity,
        info,
    }))
}

struct KeysOp {
    entry: Entry,
    query: LatentWindowKeys,
    capacity: u32,
    info: OpInfo,
}
impl PreparedOp<LatentWindowKeyBuffers<View>> for KeysOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentWindowKeyBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (batch, first, circular) = match (self.query.addressing, b.row_batch, b.first_positions)
        {
            (WindowAddressing::Circular { .. }, None, None) => {
                (Argument::NullDevicePointer, Argument::NullDevicePointer, 1)
            }
            (WindowAddressing::Staged { .. }, Some(batch), Some(first)) if b.sequences > 0 => (
                pointer(batch, u64::from(counts.rows) * 4)?,
                pointer(first, u64::from(b.sequences) * 4)?,
                0,
            ),
            _ => {
                return Err(Refusal(
                    "window key-list sequence views differ from its addressing mode",
                ));
            }
        };
        if [
            Some(b.positions.handle),
            Some(b.live_rows.handle),
            b.row_batch.map(|v| v.handle),
            b.first_positions.map(|v| v.handle),
        ]
        .contains(&Some(b.key_lists.handle))
        {
            return Err(Refusal("window key-list output must not overlap inputs"));
        }
        launch(
            &self.entry,
            self.capacity,
            counts,
            1,
            vec![
                pointer(b.positions, u64::from(counts.rows) * 4)?,
                batch,
                first,
                pointer(
                    b.key_lists,
                    u64::from(counts.rows) * u64::from(self.query.window) * 4,
                )?,
                pointer(b.live_rows, 4)?,
                Argument::U32(b.sequences),
                Argument::U32(self.query.window),
                Argument::U32(self.query.addressing.slots()),
                Argument::U32(circular),
            ],
        )
    }
}

use kernel_core::backend::Backend;
use kernel_core::contract::DType;
use kernel_core::latent_attention::{LATENT_WINDOW_SLOTS, LatentWindowStorage};
use kernel_core::latent_glue::{LatentWindowWrite, LatentWindowWriteBuffers};
use kernel_core::precision::quantizer::{
    BlockQuantization, BlockQuantizeBuffers, BlockQuantizeQuery, QuantizeOutput,
};

pub(super) fn window_write(
    backend: &dyn Backend,
    entries: &[Entry],
    op: &LatentWindowWrite,
    need: Need,
) -> Result<Prepared<LatentWindowWriteBuffers<View>>, Refused> {
    op.validate()?;
    if op.latent != 512 || op.rotary != 64 {
        return Err(
            Refusal("SM120 native window glue requires a 64-channel tail in 512 channels").into(),
        );
    }
    let (rotate, mut info) = select(entries, Kernel::WindowRotate, need)?;
    let (store, store_info) = select(entries, Kernel::WindowStore, need)?;
    let quantizer = backend
        .block_quantize(
            &BlockQuantizeQuery {
                width: 512,
                input: DType::F32,
                format: BlockQuantization::E4M3_CEIL32,
                output: QuantizeOutput::Codes,
            },
            need,
        )
        .map_err(|error| {
            Refused::Rule(format!(
                "native window write requires the shared F32-input ceil quantizer: {error}"
            ))
        })?;
    if !quantizer.info().state.is_empty() || !quantizer.info().weight_layouts.is_empty() {
        return Err(Refusal("native window quantizer must be stateless and weightless").into());
    }
    let r = u64::from(need.capacity);
    let slots = u64::from(op.addressing.slots());
    info.bytes = vec![
        ("input".into(), r * 512 * 4),
        ("weight".into(), 512 * 2),
        ("latent".into(), r * 512 * 4),
        ("cos_sin".into(), r * 64 * 4),
        ("positions".into(), r * 4),
        ("row_batch".into(), r * 4),
        ("last_positions".into(), 4),
        ("window.codes".into(), slots * 512),
        ("window.scales".into(), slots * 16),
        ("quantized_codes".into(), r * 512),
        ("quantized_scales".into(), r * 16),
        ("live_rows".into(), 4),
    ];
    if matches!(op.addressing, WindowAddressing::Staged { .. }) {
        info.bytes.push(("first_positions".into(), 4));
    }
    for (role, bytes) in &quantizer.info().bytes {
        let target = match role.as_str() {
            "input" => "latent",
            "output" => "quantized_codes",
            "scales" => "quantized_scales",
            "live_rows" => "live_rows",
            _ => return Err(Refusal("native window quantizer declares an unknown role").into()),
        };
        let (_, count) = info
            .bytes
            .iter_mut()
            .find(|(path, _)| path == target)
            .expect("quantizer role");
        *count = (*count).max(*bytes);
    }
    info.guarantees = info
        .guarantees
        .meet(quantizer.info().guarantees)
        .meet(store_info.guarantees);
    info.symbols
        .extend(quantizer.info().symbols.iter().cloned());
    info.symbols.extend(store_info.symbols);
    info.workspace = quantizer.info().workspace;
    info.implementation.push_str(&format!(
        ":{}+{}",
        op.numerical_contract_id()?,
        quantizer.info().implementation
    ));
    need.check(&info)?;
    Ok(Box::new(WriteOp {
        rotate,
        store,
        quantizer,
        query: *op,
        capacity: need.capacity,
        info,
    }))
}

struct WriteOp {
    rotate: Entry,
    store: Entry,
    quantizer: Prepared<BlockQuantizeBuffers<View>>,
    query: LatentWindowWrite,
    capacity: u32,
    info: OpInfo,
}
impl PreparedOp<LatentWindowWriteBuffers<View>> for WriteOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentWindowWriteBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let slots = self.query.addressing.slots();
        let view_slots = match b.window.storage {
            LatentWindowStorage::Ring128 => LATENT_WINDOW_SLOTS,
            LatentWindowStorage::Staged { slots } => slots,
        };
        let w = b.window;
        if view_slots != slots || w.sequences == 0 {
            return Err(Refusal("native window writer and reader extents differ"));
        }
        for (view, stride, plane) in [
            (w.codes, w.code_stride, u64::from(slots) * 512),
            (w.scales, w.scale_stride, u64::from(slots) * 16),
        ] {
            if stride < plane {
                return Err(Refusal("native window plane stride is too short"));
            }
            let end = u64::from(w.sequences - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(plane))
                .ok_or(Refusal("native window plane extent overflows"))?;
            view.covers(end)?;
        }
        if !w.code_stride.is_multiple_of(2) {
            return Err(Refusal(
                "native window code stride must preserve reader pair alignment",
            ));
        }
        let inputs = [
            b.input.handle,
            b.weight.handle,
            b.cos_sin.handle,
            b.positions.handle,
            b.row_batch.handle,
            b.last_positions.handle,
            b.live_rows.handle,
            b.first_positions.map_or(0, |v| v.handle),
        ];
        let outputs = [
            b.latent.handle,
            b.quantized_codes.handle,
            b.quantized_scales.handle,
            w.codes.handle,
            w.scales.handle,
        ];
        for (index, handle) in outputs.iter().enumerate() {
            if inputs.contains(handle) || outputs[..index].contains(handle) {
                return Err(Refusal(
                    "native window mutable planes and scratch must not overlap",
                ));
            }
        }
        let (first, circular) = match (self.query.addressing, b.first_positions) {
            (WindowAddressing::Circular { .. }, None) => (Argument::NullDevicePointer, 1),
            (WindowAddressing::Staged { .. }, Some(first)) => {
                (pointer(first, u64::from(w.sequences) * 4)?, 0)
            }
            _ => {
                return Err(Refusal(
                    "native window first positions differ from its addressing mode",
                ));
            }
        };
        let r = u64::from(counts.rows);
        b.latent.covers(r * 512 * 4)?;
        workspace.covers(self.info.workspace)?;
        let mut launches = launch(
            &self.rotate,
            self.capacity,
            counts,
            1,
            vec![
                pointer(b.input, r * 512 * 4)?,
                pointer(b.weight, 512 * 2)?,
                pointer(b.latent, r * 512 * 4)?,
                pointer(b.cos_sin, r * 64 * 4)?,
                pointer(b.live_rows, 4)?,
                Argument::U32(self.query.epsilon_bits),
            ],
        )?;
        launches.extend(self.quantizer.launches(
            &BlockQuantizeBuffers {
                input: b.latent,
                output: b.quantized_codes,
                scales: Some(b.quantized_scales),
                live_rows: b.live_rows,
            },
            workspace,
            counts,
        )?);
        launches.extend(launch(
            &self.store,
            self.capacity,
            counts,
            1,
            vec![
                pointer(b.quantized_codes, r * 512)?,
                pointer(b.quantized_scales, r * 16)?,
                pointer(b.positions, r * 4)?,
                pointer(b.row_batch, r * 4)?,
                first,
                pointer(b.last_positions, u64::from(w.sequences) * 4)?,
                Argument::DeviceBuffer(w.codes.handle),
                Argument::DeviceBuffer(w.scales.handle),
                pointer(b.live_rows, 4)?,
                Argument::U32(w.sequences),
                Argument::U32(slots),
                Argument::U32(circular),
                Argument::U64(w.code_stride),
                Argument::U64(w.scale_stride),
            ],
        )?);
        Ok(launches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};

    #[test]
    fn window_writer_uses_shared_quantization_and_independent_reader_planes() {
        let entries = [Kernel::WindowRotate, Kernel::WindowStore].map(|kernel| Entry {
            kernel,
            capacity: 2,
            symbol: format!("test_{kernel:?}"),
        });
        let backend = entries
            .into_iter()
            .fold(Sm120::new(), Sm120::with_latent_glue);
        let query = LatentWindowWrite {
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        assert!(
            backend
                .latent_window_write(&query, Need::rows(2))
                .err()
                .unwrap()
                .to_string()
                .contains("shared F32-input ceil quantizer")
        );
        let quantizer = crate::row::block_quantize::Entry::new(
            BlockQuantizeQuery {
                width: 512,
                input: DType::F32,
                format: BlockQuantization::E4M3_CEIL32,
                output: QuantizeOutput::Codes,
            },
            2,
            "shared_quantize",
        )
        .unwrap();
        let backend = backend.with_block_quantize(quantizer);
        let op = backend.latent_window_write(&query, Need::rows(2)).unwrap();
        let codes = 136 * 512;
        let scales = 136 * 16;
        let b = LatentWindowWriteBuffers {
            input: view(1, 2 * 512 * 4),
            weight: view(2, 512 * 2),
            latent: view(3, 2 * 512 * 4),
            cos_sin: view(4, 2 * 64 * 4),
            positions: view(5, 8),
            row_batch: view(6, 8),
            first_positions: None,
            last_positions: view(7, 8),
            window: kernel_core::latent_attention::LatentWindowBuffers {
                codes: view(8, codes * 2 + 66),
                scales: view(9, scales * 2 + 7),
                sequences: 2,
                code_stride: codes + 66,
                scale_stride: scales + 7,
                storage: LatentWindowStorage::Staged { slots: 136 },
                key_lists: View::EMPTY,
                key_stride: 0,
                budget: 0,
            },
            quantized_codes: view(10, 2 * 512),
            quantized_scales: view(11, 2 * 16),
            live_rows: view(12, 4),
        };
        assert!(
            op.launches(
                &LatentWindowWriteBuffers {
                    input: view(1, 2 * 512 * 2),
                    ..b
                },
                View::EMPTY,
                Counts::rows(2)
            )
            .is_err()
        );
        let launches = op.launches(&b, View::EMPTY, Counts::rows(2)).unwrap();
        assert_eq!(launches.len(), 3);
        assert_eq!(launches[1].symbol, "shared_quantize");
        for (variant, mut bad) in [b; 4].into_iter().enumerate() {
            match variant {
                0 => bad.window.codes.bytes -= 1,
                1 => bad.window.scale_stride = scales - 1,
                2 => bad.window.code_stride += 1,
                _ => bad.latent.handle = bad.input.handle,
            }
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(2)).is_err());
        }
    }
}
