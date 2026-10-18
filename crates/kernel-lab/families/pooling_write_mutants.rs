//! Device negative controls generated from the current serving bodies.
//! Only these lab cubins change; the serving sources and symbols stay intact.

use kernel_build::CubinSpec;
use kernel_core::contract::RecurrentPhase;

pub const FAMILY: &str = "pooling_write_mutants";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fault {
    WrongGate,
    DropSplitTail,
    SwapNibbles,
    WrongPage,
    NarrowCodes,
    NarrowScales,
    DropInputStatus,
    DropDecodeStatus,
    MissingStatusOr,
    FlagSaturation,
    HoleStatus,
}

impl Fault {
    pub const ALL: [Self; 11] = [
        Self::WrongGate,
        Self::DropSplitTail,
        Self::SwapNibbles,
        Self::WrongPage,
        Self::NarrowCodes,
        Self::NarrowScales,
        Self::DropInputStatus, Self::DropDecodeStatus, Self::MissingStatusOr, Self::FlagSaturation, Self::HoleStatus,
    ];

    pub fn tag(self) -> &'static str {
        match self {
            Self::WrongGate => "wrong_gate",
            Self::DropSplitTail => "drop_split_tail",
            Self::SwapNibbles => "swap_nibbles",
            Self::WrongPage => "wrong_page",
            Self::NarrowCodes => "narrow_codes",
            Self::NarrowScales => "narrow_scales",
            Self::DropInputStatus => "drop_input_status",
            Self::DropDecodeStatus => "drop_decode_status",
            Self::MissingStatusOr => "missing_status_or",
            Self::FlagSaturation => "flag_saturation",
            Self::HoleStatus => "hole_status",
        }
    }

    pub fn symbol(self, served: &str) -> String {
        let (stem, version) = served.rsplit_once("_v").expect("versioned serving symbol");
        format!("{stem}_{}_v{version}", self.tag())
    }

    pub fn cubin_name(self) -> String {
        format!("pooling_write_{}_v1", self.tag())
    }

    pub fn cubin(self) -> CubinSpec {
        let mut spec = if matches!(self, Self::WrongGate | Self::DropSplitTail) {
            let entries: Vec<_> = crate::gated_pooling_f32::entries()
                .into_iter()
                .filter(|e| {
                    self != Self::DropSplitTail || e.query().phase == RecurrentPhase::Prefill
                })
                .collect();
            kernels_sm120::gated_pooling::cubin_spec(self.cubin_name(), &entries)
                .expect("pooling control entries")
        } else {
            kernels_sm120::gated_pooling::key_write::cubin_spec(
                self.cubin_name(),
                &crate::fp4_key_write::entries(),
            )
            .expect("key writer control entries")
        };
        for entry in &mut spec.entries {
            match self {
                Self::WrongGate if entry.symbol.ends_with("_pool_v1") => {
                    assert_eq!(entry.body.matches("(key,gate,positions").count(), 1);
                    entry.body = entry
                        .body
                        .replace("(key,gate,positions", "(key,key,positions");
                }
                Self::DropSplitTail if entry.symbol.ends_with("_pool_v1") => {
                    assert_eq!(entry.body.matches("chop::gated_pooling::pool2<").count(), 1);
                    entry.body = entry
                        .body
                        .replace(
                            "chop::gated_pooling::pool2<",
                            "chop::pooling_write_tests::drop_split_tail<",
                        )
                        .replace(",0>", ">");
                }
                Self::SwapNibbles if !entry.symbol.contains("_bf16_") => {
                    let served = crate::fp4_key_write::entries()
                        .into_iter()
                        .find(|e| e.symbol() == entry.symbol)
                        .expect("served writer");
                    entry.body.push_str(&format!(
                        "chop::pooling_write_tests::swap_nibbles<{},{},{}>(codes,positions,row_batch,live_rows,page_tables,items,pages,max_pages,code_stride,table_rows);",
                        served.width(), served.ratio(), served.page_size()));
                }
                Self::WrongPage => {
                    assert_eq!(entry.body.matches("code_stride,scale_stride").count(), 1);
                    entry.body = entry.body.replace("code_stride,scale_stride", "0,0");
                }
                Self::NarrowCodes => {
                    assert_eq!(entry.body.matches(",code_stride,").count(), 1);
                    entry.body = entry
                        .body
                        .replace(",code_stride,", ",uint32_t(code_stride),");
                }
                Self::NarrowScales => {
                    assert_eq!(entry.body.matches(",scale_stride,").count(), 1);
                    entry.body = entry
                        .body
                        .replace(",scale_stride,", ",uint32_t(scale_stride),");
                }
                Self::DropInputStatus | Self::DropDecodeStatus | Self::MissingStatusOr => {
                    let policy = match self {
                        Self::DropInputStatus => "StatusDrop<1>", Self::DropDecodeStatus => "StatusDrop<4>", _ => "StatusReplace",
                    };
                    assert_eq!(entry.body.matches(">(key,").count(),1);
                    entry.body=entry.body.replacen(">(key,", &format!(",chop::pooling_write_tests::{policy}>(key,"),1);
                }
                Self::FlagSaturation => {
                    let served=crate::fp4_key_write::entries().into_iter().find(|e|e.symbol()==entry.symbol).unwrap();
                    entry.body=format!("chop::pooling_write_tests::flag_saturation<{}>(key,live_rows,status,status_row_offset);{}",served.width(),entry.body);
                }
                Self::HoleStatus => {
                    let served = crate::fp4_key_write::entries().into_iter().find(|e|e.symbol()==entry.symbol).unwrap();
                    entry.body.push_str(&format!("chop::pooling_write_tests::mark_holes<{},{}>(positions,row_batch,live_rows,page_tables,items,pages,max_pages,table_rows,status,status_row_offset);",served.ratio(),served.page_size()));
                }
                _ => {}
            }
            entry.symbol = self.symbol(&entry.symbol);
        }
        spec.sources_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        spec.include_dirs = vec!["kernels-sm120/cuda".into()];
        spec.headers = vec!["#include \"kernel-lab/cuda/probes/pooling_write_mutants.cuh\"".into()];
        spec
    }
}

pub fn cubins() -> Vec<CubinSpec> {
    Fault::ALL.into_iter().map(Fault::cubin).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_controls_have_one_fault_and_distinct_entry_names() {
        let mut symbols = std::collections::BTreeSet::new();
        for fault in Fault::ALL {
            let spec = fault.cubin();
            assert_eq!(spec.name, fault.cubin_name());
            assert!(!spec.entries.is_empty());
            for entry in spec.entries {
                assert!(symbols.insert(entry.symbol));
                assert!(!entry.body.is_empty());
            }
        }
    }
}
