//! Stateless F32 RMSNorm over promoted BF16 checkpoint weights.

use super::{HEADER, THREADS, pointer, scalar};
use kernel_build::{CubinSpec, EntryDeclaration, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, NumericalContract, Refusal, RmsNormKind, Validate};
use kernel_core::rmsnorm::RmsNormRequirement;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F32Norm {
    requirement: RmsNormRequirement,
    pub capacity: u32,
}

pub fn ratio_one_norm(width: u32, epsilon: f32, capacity: u32) -> Result<F32Norm, Refusal> {
    let requirement = RmsNormRequirement {
        input: DType::F32,
        output: DType::F32,
        ..RmsNormRequirement::bf16_backend_chosen(width, epsilon, RmsNormKind::Plain, false)
    };
    requirement.validate()?;
    if !matches!(width, 128 | 512) || capacity == 0 || capacity > crate::backend::MAX_CAPACITY {
        return Err(Refusal(
            "F32 RMSNorm needs width 128/512 and a bounded positive capacity",
        ));
    }
    Ok(F32Norm {
        requirement,
        capacity,
    })
}

impl F32Norm {
    pub fn requirement(&self) -> &RmsNormRequirement {
        &self.requirement
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("F32 RMSNorm rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn entry(&self, symbol: impl Into<String>) -> EntrySpec {
        EntrySpec::new(
            symbol.into(),
            vec![
                pointer("x", "const float"),
                pointer("weight", "const uint16_t"),
                pointer("y", "float"),
                pointer("live_rows", "const uint32_t"),
                scalar("eps_bits"),
            ],
            format!(
                "chop::gated_pooling::norm<{},false>(x,weight,nullptr,y,live_rows,__uint_as_float(eps_bits));",
                self.requirement.width
            ),
            THREADS,
            1,
        )
    }

    pub fn cubin_spec(&self, name: impl Into<String>, symbol: impl Into<String>) -> CubinSpec {
        CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: vec![format!("#include \"{HEADER}\"")],
            defines: vec![],
            entries: vec![self.entry(symbol)],
        }
    }

    pub fn declaration(&self, symbol: impl Into<String>) -> EntryDeclaration {
        EntryDeclaration {
            symbol: symbol.into(),
            operation: "RMSNorm".into(),
            contract: self
                .requirement
                .query()
                .numerical_contract_id()
                .expect("validated F32 norm")
                .to_string(),
            implementation: "sm120.gated-pooling.f32-rmsnorm.tree128.v1".into(),
            requirements: [
                ("requirement".into(), format!("{:?}", self.requirement)),
                (
                    "rounding".into(),
                    "F32 input, normalization and output; promoted BF16 weight".into(),
                ),
            ]
            .into(),
            correctness_case: Some("f32_rmsnorm".into()),
        }
    }
}
