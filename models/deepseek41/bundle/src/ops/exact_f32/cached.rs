//! Named cached projection stages that fused consumers can replace explicitly.
use crate::ops::Lowered;
use kernel_core::backend::OpInfo;
use kernel_core::contract::{Argument, Refusal};
use runtime_bundle_abi::DeviceBufferSpanV1;

#[derive(Clone, Copy)]
pub(crate) struct Spans {
    pub activation: DeviceBufferSpanV1,
    pub weight: DeviceBufferSpanV1,
    pub row_domain: DeviceBufferSpanV1,
    pub partials: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub weight_domain: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
}
impl Spans {
    pub(crate) fn validate(self) -> Result<(), Refusal> {
        let spans = [
            (self.activation, 16),
            (self.weight, 16),
            (self.row_domain, 256),
            (self.partials, 256),
            (self.live_rows, 4),
            (self.weight_domain, 4),
            (self.output, 4),
        ];
        for (i, (a, alignment)) in spans.iter().enumerate() {
            if a.length == 0
                || !a.offset.is_multiple_of(*alignment)
                || a.offset.checked_add(a.length).is_none()
            {
                return Err(Refusal(
                    "cached projection span is empty, unaligned or overflowing",
                ));
            }
            for (b, _) in &spans[..i] {
                if a.buffer == b.buffer
                    && a.offset < b.offset + b.length
                    && b.offset < a.offset + a.length
                {
                    return Err(Refusal("cached projection spans overlap"));
                }
            }
        }
        Ok(())
    }
}

pub(crate) trait CachedProjection: Send + Sync {
    fn info(&self) -> &OpInfo;
    fn partial_bytes(&self, rows: u32) -> Result<u64, Refusal>;
    fn lower(&self, rows: u32, spans: Spans) -> Result<CachedProjectionStages, Refusal>;
}

pub(crate) struct CachedProjectionStages {
    classifier: Lowered,
    producer: Lowered,
    reducer: Lowered,
    row_domain: DeviceBufferSpanV1,
    partials: DeviceBufferSpanV1,
}
impl CachedProjectionStages {
    pub(crate) fn new(
        calls: Vec<Lowered>,
        symbols: [String; 3],
        rows: u32,
        spans: Spans,
    ) -> Result<Self, Refusal> {
        spans.validate()?;
        let [classifier, producer, reducer]: [Lowered; 3] = calls
            .try_into()
            .map_err(|_| Refusal("cached projection needs three named stages"))?;
        for (call, symbol) in [&classifier, &producer, &reducer].into_iter().zip(symbols) {
            if call.launch.symbol != symbol || call.launch.rows != rows {
                return Err(Refusal("cached projection stage identity differs"));
            }
        }
        let call = &producer.launch;
        let arguments = &call.arguments.0;
        if arguments.len() != 7 || arguments[5] != Argument::U32(rows) {
            return Err(Refusal("cached producer argument layout differs"));
        }
        for (i, expected) in [
            (0, spans.activation),
            (1, spans.weight),
            (2, spans.row_domain),
            (3, spans.partials),
            (4, spans.live_rows),
            (6, spans.weight_domain),
        ] {
            let Argument::DeviceBuffer(handle) = arguments[i] else {
                return Err(Refusal("cached producer needs bound spans"));
            };
            if call.bindings.get(handle).ok() != Some(expected) {
                return Err(Refusal("cached producer span differs"));
            }
        }
        Ok(Self {
            classifier,
            producer,
            reducer,
            row_domain: spans.row_domain,
            partials: spans.partials,
        })
    }
    pub(crate) fn into_launches(self) -> Vec<Lowered> {
        vec![self.classifier, self.producer, self.reducer]
    }
    pub(crate) fn into_producer(
        self,
        rows: u32,
        row_domain: DeviceBufferSpanV1,
        partials: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        if self.producer.launch.rows != rows
            || self.row_domain != row_domain
            || self.partials != partials
        {
            return Err(Refusal(
                "fused consumer differs from cached producer layout",
            ));
        }
        Ok(self.producer)
    }
}
