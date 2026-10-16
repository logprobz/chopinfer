//! The Case runner's GPU negative controls for reads before the PDL wait:
//! the lab latent shape's decode split and its merge (`latent_shapes.rs`),
//! with one stage's dependency wait made a no-op
//! (`cuda/probes/latent_split_merge_no_wait.cuh`). In the split control the
//! split reads the live count and its inputs before its wait; in the merge
//! control the merge reads the split's partials and the live count before
//! its wait. Each cubin holds the two stages under their own symbols.

use kernel_build::CubinSpec;
use std::path::PathBuf;

use crate::latent_attention::shapes;

pub const FAMILY: &str = "pdl_misorder";

/// The lab latent entry whose chain the controls run.
pub const KERNEL: &str = "decode";

/// The stage whose wait a control drops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Split,
    Merge,
}

impl Stage {
    pub const ALL: [Stage; 2] = [Stage::Split, Stage::Merge];

    fn tag(self) -> &'static str {
        match self {
            Stage::Split => "split",
            Stage::Merge => "merge",
        }
    }

    /// The control's cubin.
    pub fn cubin_name(self) -> String {
        format!("pdl_misorder_latent_{}_no_wait_sm120", self.tag())
    }

    /// The control's symbol for the served lab symbol `served`
    /// (`chop_latent_<name>_v1` or `chop_latent_<name>_merge_v1`).
    pub fn symbol(self, served: &str) -> String {
        let name = served
            .strip_prefix("chop_latent_")
            .expect("a latent symbol");
        format!("chop_pdl_{}_no_wait_{name}", self.tag())
    }

    /// The control's cubin: the lab cubin's [`KERNEL`] split and merge
    /// entries, renamed, over the probe header.
    pub fn cubin(self) -> CubinSpec {
        let entry = shapes::catalog()
            .into_iter()
            .find(|entry| entry.name == format!("{}_attn_bf16_{KERNEL}", shapes::PROFILE.tag()))
            .expect("the lab decode entry");
        let served = [
            entry.symbol(),
            entry.merge_symbol().expect("a split merge"),
        ];
        let lab = shapes::cubin();
        let entries = served
            .iter()
            .map(|symbol| {
                let mut spec = lab
                    .entries
                    .iter()
                    .find(|spec| &spec.symbol == symbol)
                    .expect("the served entry in the lab cubin")
                    .clone();
                spec.symbol = self.symbol(symbol);
                spec
            })
            .collect();
        CubinSpec {
            name: self.cubin_name(),
            sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
            include_dirs: Vec::new(),
            headers: vec![
                "#include \"kernel-lab/cuda/probes/latent_split_merge_no_wait.cuh\"".into(),
            ],
            defines: vec![(
                "CHOP_PDL_MISORDER_MERGE".into(),
                u8::from(self == Stage::Merge).to_string(),
            )],
            entries,
        }
    }
}

pub fn cubins() -> Vec<CubinSpec> {
    Stage::ALL.map(Stage::cubin).into()
}
