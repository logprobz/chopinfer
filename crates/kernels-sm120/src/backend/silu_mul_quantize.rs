//! The clamped SwiGLU with BF16 rounding and ceil E4M3/32 quantization
//! through the model's registered entries (`row::silu_mul_quantize`). The
//! entry whose query equals the op's is chosen by SM120's rule
//! (`select::choose`), whatever the order the model registered entries in;
//! it claims what it declares (one-row bits up to its capacity, any chunk
//! split, row independence).

use super::select::{Candidate, choose};
use crate::row::silu_mul_quantize::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Refusal, Validate};
use kernel_core::gated_mlp::quantized::{SiluMulQuantizeBuffers, SiluMulQuantizeQuery};

const IMPLEMENTATION: &str = "sm120.silu-mul-quantize.v1";

fn key(entry: &Entry) -> String {
    format!("{:?}", entry.spec())
}

pub(super) fn prepare(
    entries: &[Entry],
    query: &SiluMulQuantizeQuery,
    need: Need,
) -> Result<Prepared<SiluMulQuantizeBuffers<View>>, Refused> {
    query.validate()?;
    if need.capacity == 0 || need.capacity > super::MAX_CAPACITY {
        return Err(Refusal("quantized SwiGLU preparation capacity exceeds SM120 bounds").into());
    }
    let chosen = choose(
        "silu mul quantize",
        entries.iter().filter(|e| e.query == *query).map(|entry| {
            Candidate::new(entry.clone(), &entry.symbol, key(entry), entry.invariance())
                .capacity(entry.capacity)
        }),
        need,
    )?;
    let entry = chosen.entry;
    entry.validate()?;
    let info = OpInfo::new(
        IMPLEMENTATION,
        chosen
            .guarantees
            .meet(kernel_core::properties::Invariance::any_rows(need.capacity)),
        vec![(entry.symbol.clone(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Op {
        entry,
        capacity: need.capacity,
        info,
    }))
}

struct Op {
    entry: Entry,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<SiluMulQuantizeBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &SiluMulQuantizeBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows > self.capacity {
            return Err(Refusal("quantized SwiGLU launch exceeds prepared capacity"));
        }
        let entry = &self.entry;
        let rows = counts.rows;
        buffers.gate.covers(entry.input_bytes(rows))?;
        buffers.up.covers(entry.input_bytes(rows))?;
        buffers.codes.covers(entry.code_bytes(rows))?;
        buffers.scales.covers(entry.scale_bytes(rows))?;
        buffers.live_rows.covers(4)?;
        let written = [buffers.codes.handle, buffers.scales.handle];
        let read = [
            buffers.gate.handle,
            buffers.up.handle,
            buffers.live_rows.handle,
        ];
        if written[0] == written[1] || written.iter().any(|w| read.contains(w)) {
            return Err(Refusal(
                "quantized SwiGLU outputs overlap each other or its inputs",
            ));
        }
        Ok(vec![Launch {
            symbol: entry.symbol.clone(),
            rows,
            plan: entry.plan(rows)?,
            arguments: entry.arguments(
                buffers.gate.handle,
                buffers.up.handle,
                buffers.codes.handle,
                buffers.scales.handle,
                buffers.live_rows.handle,
            )?,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use crate::row::silu_mul_quantize::lab_entries;
    use kernel_core::backend::Backend;
    use kernel_core::contract::Argument;
    use kernel_core::gated_mlp::SwigluClamp;
    use kernel_core::properties::Invariance;

    fn buffers() -> SiluMulQuantizeBuffers<View> {
        let v = |handle| View {
            handle,
            bytes: 16 * 2304 * 4,
        };
        SiluMulQuantizeBuffers {
            gate: v(1),
            up: v(2),
            codes: v(3),
            scales: v(4),
            live_rows: View {
                handle: 5,
                bytes: 4,
            },
        }
    }

    #[test]
    fn prepares_each_lab_entry_and_checks_roles() {
        for entry in lab_entries() {
            let query = entry.query;
            let backend = Sm120::new().with_silu_mul_quantize(entry.clone());
            let op = backend
                .silu_mul_quantize(&query, Need::rows(8).with_exact(Invariance::any_rows(8)))
                .unwrap();
            assert_eq!(op.info().implementation, IMPLEMENTATION);
            let mut buffers = buffers();
            for rows in [0, 1, 7, 8] {
                let launches = op
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                assert_eq!(launches.len(), 1);
                assert_eq!(launches[0].symbol, entry.symbol);
                assert_eq!(launches[0].plan.grid, [rows.max(1), query.width / 32, 1]);
                assert_eq!(
                    launches[0].arguments.0,
                    (1..=5).map(Argument::DeviceBuffer).collect::<Vec<_>>()
                );
            }
            assert!(op.launches(&buffers, View::EMPTY, Counts::rows(9)).is_err());
            buffers.scales.bytes = entry.scale_bytes(8) - 1;
            assert!(op.launches(&buffers, View::EMPTY, Counts::rows(8)).is_err());
            let mut aliased = self::buffers();
            aliased.codes.handle = aliased.gate.handle;
            assert!(op.launches(&aliased, View::EMPTY, Counts::rows(1)).is_err());
        }
    }

    #[test]
    fn selection_refuses_other_clamps_floors_and_widths() {
        let entry = lab_entries().remove(1);
        let backend = Sm120::new().with_silu_mul_quantize(entry.clone());
        let need = Need::rows(4);
        backend.silu_mul_quantize(&entry.query, need).unwrap();
        for query in [
            SiluMulQuantizeQuery {
                clamp: SwigluClamp::symmetric(7.0),
                ..entry.query
            },
            SiluMulQuantizeQuery {
                amax_floor_bits: 1e-3_f32.to_bits(),
                ..entry.query
            },
            SiluMulQuantizeQuery {
                width: 608,
                ..entry.query
            },
        ] {
            assert!(backend.silu_mul_quantize(&query, need).is_err());
        }
        assert!(
            backend
                .silu_mul_quantize(&entry.query, Need::rows(17))
                .is_err()
        );
        assert!(Sm120::new().silu_mul_quantize(&entry.query, need).is_err());
    }
}
