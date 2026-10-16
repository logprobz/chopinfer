//! Component conformance generated from the committed encoder declarations.
//! Every case runs its bundle's serving cubin against the component oracle.
use std::collections::BTreeSet;

use encoder_spec::Declaration;
use kernel_lab::encoder::Case;
use kernels_sm120::encoder::{self, Registered};
use model_glm5next_contract::{encoder as glm, sm120};
use model_mimo26_contract::{
    encoders::spec, sm120::encoders as encoder_kernels, sm120::encoders::rows,
};
use model_qwen38_contract::encoder as vision;

pub(crate) use kernel_lab::encoder as shared;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Model {
    Qwen,
    Mimo,
    Glm,
}
impl Model {
    pub(crate) fn cubin(self, name: &str) -> &'static [u8] {
        assert_eq!(self, crate::MODEL);
        crate::embedded_cubin(name)
            .unwrap_or_else(|| panic!("{self:?}: missing serving cubin {name}"))
    }
}
/// The serving cubins of `model`: the test's own bundle's, none of the
/// others'.
fn cubins(model: Model) -> fn(&str) -> Option<&'static [u8]> {
    if model == crate::MODEL {
        crate::embedded_cubin
    } else {
        |_| None
    }
}
fn declarations() -> Vec<(Model, &'static Declaration, &'static dyn Registered)> {
    vision::declarations()
        .unwrap()
        .iter()
        .map(|d| (Model::Qwen, d, &encoder::Shared as &dyn Registered))
        .chain(
            spec::declarations_flash()
                .unwrap()
                .iter()
                .map(|d| (Model::Mimo, d, &encoder_kernels::Kernels as &dyn Registered)),
        )
        .chain(
            glm::declarations()
                .unwrap()
                .iter()
                .map(|d| (Model::Glm, d, &encoder::Shared as &dyn Registered)),
        )
        .collect()
}
/// Every model's cases, each model's from its declarations in order.
fn cases() -> Vec<(Model, Case)> {
    let declarations = declarations();
    let mut models = declarations
        .iter()
        .map(|(model, _, _)| *model)
        .collect::<Vec<_>>();
    models.dedup();
    let mut cases = Vec::new();
    for model in models {
        let catalog = match model {
            Model::Qwen => vision::kernels(),
            Model::Mimo => encoder_kernels::kernels_flash(),
            Model::Glm => sm120::encoder::kernels(),
        }
        .unwrap();
        let own = declarations
            .iter()
            .filter(|(owner, _, _)| *owner == model)
            .map(|&(_, declaration, registered)| (declaration, registered))
            .collect::<Vec<_>>();
        let found = kernel_lab::encoder::cases(&own, catalog, cubins(model)).unwrap();
        cases.extend(found.into_iter().map(|case| (model, case)));
    }
    cases
}
const KERNEL_FREE: &[&str] = &[
    "merge.consecutive.v1",
    "group.repeat-last.v1",
    "mlp.gelu.v1",
];
// The same lists generate the test functions and the coverage registry.
macro_rules! qwen_cases {
    ($consumer:ident) => { $consumer! {
    linear_pairs: "linear.pairs.v1" => shared::gemm,
    position_learned_grid: "position.learned-grid.bilinear-align-corners.f32.v1" => shared::position,
    norm_layer: "norm.layer-f32.v1" => shared::norm,
    attention_varlen: "attention.varlen.pairs.v1" => shared::attention,
    rotary_axial: "rotary.axial-2d.v1" => shared::rotary,
    layout_tokens: "layout.tokens.v1" => shared::layout,
    } };
}
macro_rules! mimo_cases {
    ($consumer:ident) => {
        $consumer! {
        linear_pairs: "linear.pairs.v1" => shared::gemm,
        norm_layer: "norm.layer-f32.v1" => shared::norm,
        norm_rms: "norm.rms-f32.v1" => shared::norm,
        attention_varlen: "attention.varlen.pairs.v1" => shared::attention,
        rotary_axial: "rotary.axial-2d.v1" => shared::rotary,
        rotary_1d: "rotary.1d.v1" => shared::rotary,
        reorder: "reorder.merge-group-columns.v1" => shared::rows,
        mlp_swiglu: "mlp.swiglu.v1" => shared::rows,
        transformer_carry: "transformer.pre-norm.v1" => shared::rows,
        conv1d: "conv1d.im2col.v1" => shared::rows,
        layout_tokens: "layout.tokens.v1" => shared::layout,
        }
    };
}
macro_rules! glm_cases {
    ($consumer:ident) => {
        $consumer! {
        pad_columns: "pad.zero-columns.v1" => shared::rows,
        linear_pairs: "linear.pairs.v1" => shared::gemm,
        norm_rms: "norm.rms-f32.v1" => shared::norm,
        norm_layer: "norm.layer-f32.v1" => shared::norm,
        attention_qk_rms: "attention.varlen.qk-rms.pairs.v1" => shared::qk_rms_attention,
        rotary_axial: "rotary.axial-2d.v1" => shared::rotary,
        mlp_swiglu_clamped: "mlp.swiglu-clamped.v1" => shared::rows,
        projector_gelu: "projector.norm-gelu-mlp.v1" => shared::rows,
        layout_tokens: "layout.tokens.v1" => shared::layout,
        }
    };
}
use kernel_lab::encoder::Runner;
macro_rules! variant_ids {
    ($($name:ident: $id:literal => $runner:path),+ $(,)?) => { &[$(($id, $runner as Runner)),+] };
}
const QWEN: &[(&str, Runner)] = qwen_cases!(variant_ids);
const MIMO: &[(&str, Runner)] = mimo_cases!(variant_ids);
const GLM: &[(&str, Runner)] = glm_cases!(variant_ids);
macro_rules! conformance {
    ($($name:ident: $id:literal => $runner:path),+ $(,)?) => {
        $(#[test]
        fn $name() { suite::run($id); })+
    };
}
pub(crate) fn run(id: &str) {
    let registry = match crate::MODEL {
        Model::Qwen => QWEN,
        Model::Mimo => MIMO,
        Model::Glm => GLM,
    };
    let cases = cases()
        .into_iter()
        .filter(|(model, _)| *model == crate::MODEL)
        .map(|(_, case)| case)
        .collect::<Vec<_>>();
    kernel_lab::encoder::run(id, &cases, registry).expect(id);
}
macro_rules! registered_cases {
    ($consumer:ident) => {
        $consumer! {
            quantize_rvq20: "mimo26:quantize.rvq20-f32.v1",
            embed_sum20: "mimo26:embed.sum20-f32.v1",
        }
    };
}
macro_rules! registered_ids {
    ($($name:ident: $id:literal),+ $(,)?) => { &[$($id),+] };
}
pub(crate) const REGISTERED: &[&str] = registered_cases!(registered_ids);

#[test]
fn every_declared_variant_has_a_generated_test() {
    let cases = cases();
    let mut used: BTreeSet<String> = cases.iter().map(|(_, c)| c.variant.clone()).collect();
    for (_, declaration, _) in declarations() {
        for stage in 0..declaration.stages.len() {
            if let Some(kernel) = rows::stage_kernel(declaration, stage).unwrap() {
                let id = declaration.stages[stage].op.id();
                assert!(
                    REGISTERED.contains(&id.as_str()),
                    "missing registered test: {id}"
                );
                used.insert(id);
                if crate::MODEL == Model::Mimo {
                    assert!(
                        Model::Mimo
                            .cubin(rows::CUBIN)
                            .windows(kernel.symbol.len())
                            .any(|s| s == kernel.symbol.as_bytes())
                    );
                }
            }
        }
    }
    let generated: BTreeSet<String> = QWEN
        .iter()
        .chain(MIMO)
        .chain(GLM)
        .map(|(id, _)| *id)
        .chain(REGISTERED.iter().copied())
        .map(String::from)
        .collect();
    assert_eq!(used, generated);
    for (model, registry) in [(Model::Qwen, QWEN), (Model::Mimo, MIMO), (Model::Glm, GLM)] {
        let actual: BTreeSet<_> = cases
            .iter()
            .filter(|(owner, _)| *owner == model)
            .map(|(_, c)| c.variant.as_str())
            .collect();
        let expected: BTreeSet<_> = registry.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            actual, expected,
            "{model:?}: tests must cover its own variants"
        );
    }
    for (model, case) in cases.iter().filter(|(model, _)| *model == crate::MODEL) {
        let cubin = model.cubin(if case.entry.is_attention() {
            encoder::ATTENTION_CUBIN
        } else {
            encoder::ROWS_GEMM_CUBIN
        });
        assert!(
            cubin
                .windows(case.entry.symbol.len())
                .any(|s| s == case.entry.symbol.as_bytes()),
            "missing {}",
            case.entry.symbol
        );
    }
    fn ops(value: &serde_json::Value, found: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(id) = map.get("op").and_then(|v| v.as_str()) {
                    found.insert(id.into());
                }
                for v in map.values() {
                    ops(v, found);
                }
            }
            serde_json::Value::Array(values) => {
                for v in values {
                    ops(v, found);
                }
            }
            _ => {}
        }
    }
    let mut declared = BTreeSet::new();
    for (_, d, _) in declarations() {
        ops(
            &serde_json::from_str(&d.canonical_json()).unwrap(),
            &mut declared,
        );
    }
    for id in &declared {
        assert!(
            generated.contains(id) || KERNEL_FREE.contains(&id.as_str()),
            "no conformance test or kernel-free classification for {id}"
        );
    }
    for id in KERNEL_FREE {
        assert!(declared.contains(*id));
        assert!(
            !used.contains(*id),
            "kernel-free variant {id} launches a kernel"
        );
    }
}
