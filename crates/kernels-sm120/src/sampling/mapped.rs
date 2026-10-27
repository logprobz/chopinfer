//! Indirect logits input with unchanged logical metadata, reductions and outputs.
use kernel_build::{CubinSpec, EntryParam};

/// Separate entries keep the ordinary sampling cubin and its ABI unchanged.
/// Each block runs one logical row. Only its input base changes; metadata,
/// partials, random counters and output still use that logical row.
pub fn cubin_spec() -> CubinSpec {
    let mut spec = super::cubin_spec();
    spec.name = "sampling_mapped_sm120".into();
    spec.entries.retain(|entry| {
        entry.symbol != "chop_sampling_f32_v1" && entry.symbol != "chop_sampling_bf16_v1"
    });
    for entry in &mut spec.entries {
        let finish = entry.symbol.contains("_finish_");
        let axis = if finish || entry.symbol.contains("_resolve_") {
            "x"
        } else {
            "y"
        };
        let body = entry
            .body
            .replace("(logits, stride,", "(mapped_logits, uint64_t(0),");
        assert_ne!(
            body, entry.body,
            "mapped sampler must replace its only logits input"
        );
        entry.body = format!(
            "const uint32_t logical = blockIdx.{axis}; if (logical >= rows) return; const uint32_t physical = row_indices[logical]; const bool valid = physical < source_rows; const auto* mapped_logits = logits + uint64_t(valid ? physical : 0) * stride; {body}"
        );
        if finish {
            // Every invalid map still uses an in-bounds source row while the
            // finish marks its record unusable. No consumer may accept it.
            entry
                .body
                .push_str(" if (!valid && threadIdx.x == 0) output[logical].status = 1;");
        }
        entry.symbol = entry.symbol.replace("_v1", "_mapped_v1");
        entry.params.extend([
            EntryParam::DevicePointer {
                name: "row_indices".into(),
                cxx_type: "const uint32_t".into(),
            },
            EntryParam::U32 {
                name: "source_rows".into(),
            },
        ]);
    }
    spec
}

#[cfg(test)]
mod tests {
    #[test]
    fn mapped_entries_keep_logical_rows_and_bound_every_source_read() {
        let original = super::super::cubin_spec();
        let mapped = super::cubin_spec();
        assert_eq!(mapped.entries.len(), 20);
        for entry in mapped.entries {
            let source = original
                .entries
                .iter()
                .find(|old| entry.symbol == old.symbol.replace("_v1", "_mapped_v1"))
                .unwrap();
            assert_eq!(&entry.params[..source.params.len()], source.params);
            assert!(entry.body.contains("physical < source_rows"));
            assert!(entry.body.contains("valid ? physical : 0"));
            assert!(!entry.body.contains("(logits, stride,"));
            assert!(
                entry
                    .body
                    .contains("(mapped_logits, uint64_t(0), live, rows, meta,")
            );
            if entry.symbol.contains("_finish_") {
                assert!(entry.body.contains("output[logical].status = 1"));
            }
        }
    }
}
