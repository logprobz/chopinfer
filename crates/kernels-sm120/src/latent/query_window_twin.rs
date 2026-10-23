//! The decode query RoPE and the fused window write of the same rows in one
//! clustered launch (`latent/query_window_twin.cuh`): the RoPE's CTA runs
//! the served four-group rotation, the window's CTA the served fused window
//! write, and the window's first status commit waits for the RoPE's.
use super::{query_parallel, window_fused};
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::latent_attention::LatentKvFormat;

pub const THREADS: u32 = query_parallel::THREADS;

#[derive(Clone, Debug)]
pub struct Entry {
    pub kv: LatentKvFormat,
    pub capacity: u32,
    pub symbol: String,
}

impl Entry {
    /// The fused window write this entry runs beside the RoPE.
    pub fn window(&self) -> window_fused::Entry {
        window_fused::Entry {
            kv: self.kv,
            capacity: self.capacity,
            symbol: self.symbol.clone(),
        }
    }

    /// The RoPE this entry runs beside the window write.
    pub fn rope(&self) -> query_parallel::Entry {
        query_parallel::Entry {
            capacity: self.capacity,
            symbol: self.symbol.clone(),
        }
    }

    /// One row a cluster of two CTAs: y = 0 the RoPE, y = 1 the window.
    pub fn plan(&self, rows: u32, heads: u32) -> Result<LaunchPlan, Refusal> {
        self.rope().plan(rows, heads)?;
        self.window().plan(rows)?;
        let plan = LaunchPlan {
            grid: [rows.max(1), 2, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 2, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The RoPE's parameters with its F32 queries as `query` and its table
    /// as `query_cos_sin`, less the live count and status, then the fused
    /// window's, which both halves share.
    pub fn spec(&self) -> EntrySpec {
        let rename = |param: P, name: &str| match param {
            P::DevicePointer { cxx_type, .. } => P::DevicePointer {
                name: name.into(),
                cxx_type,
            },
            other => other,
        };
        let rope = self.rope().spec().params;
        let window = self.window().spec().params;
        let name = |p: &P| match p {
            P::DevicePointer { name, .. } | P::U32 { name } | P::U64 { name } => name.clone(),
            _ => String::new(),
        };
        let mut params = Vec::new();
        for param in rope {
            match name(&param).as_str() {
                "values" => params.push(rename(param, "query")),
                "cos_sin" => params.push(rename(param, "query_cos_sin")),
                "live_rows" | "status" | "status_row_offset" => {}
                _ => params.push(param),
            }
        }
        params.extend(window);
        EntrySpec::new(
            self.symbol.clone(),
            params,
            format!(
                "chop::latent::query_window_twin::run<{}>(query,native_q,query_cos_sin,heads,input,weight,latent,cos_sin,live_rows,eps_bits,status,status_row_offset,quantized_codes,quantized_scales,positions,row_batch,first_positions,last_positions,codes,scales,sequences,slots,circular,code_stride,scale_stride);",
                self.kv == LatentKvFormat::Bf16
            ),
            THREADS,
            1,
        )
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("query window twin cubin requires entries"));
    }
    for entry in entries {
        entry.plan(entry.capacity, 1)?;
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"latent/query_window_twin.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(kv: LatentKvFormat) -> Entry {
        Entry {
            kv,
            capacity: 32,
            symbol: "query_window_twin_test_v1".into(),
        }
    }

    #[test]
    fn the_twin_keeps_both_served_abis_in_one_clustered_launch() {
        for kv in [LatentKvFormat::Fp8E4M3Ue8M0Block32, LatentKvFormat::Bf16] {
            let twin = entry(kv);
            let spec = twin.spec();
            let names: Vec<String> = spec
                .params
                .iter()
                .map(|p| match p {
                    P::DevicePointer { name, .. } | P::U32 { name } | P::U64 { name } => {
                        name.clone()
                    }
                    _ => panic!("unexpected parameter {p:?}"),
                })
                .collect();
            assert_eq!(
                names[..4],
                ["query", "native_q", "query_cos_sin", "heads"].map(String::from)
            );
            assert_eq!(spec.params[4..], twin.window().spec().params[..]);
            // Every argument of the body is a parameter, in order.
            let arguments = spec
                .body
                .split_once('(')
                .unwrap()
                .1
                .trim_end_matches(");")
                .split(',')
                .map(str::to_owned)
                .collect::<Vec<_>>();
            assert_eq!(arguments, names);
            assert!(
                spec.body
                    .contains(&format!("run<{}>", kv == LatentKvFormat::Bf16))
            );
            for rows in [1, 8, 32] {
                let plan = twin.plan(rows, 16).unwrap();
                assert_eq!(plan.grid, [rows, 2, 1]);
                assert_eq!(plan.cluster, [1, 2, 1]);
                assert_eq!(plan.block, [1024, 1, 1]);
                assert!(plan.pdl);
            }
            assert!(twin.plan(33, 16).is_err());
            assert!(twin.plan(1, 0).is_err());
        }
        let spec = cubin_spec("m", &[entry(LatentKvFormat::Bf16)]).unwrap();
        assert_eq!(spec.headers, ["#include \"latent/query_window_twin.cuh\""]);
        assert!(cubin_spec("m", &[]).is_err());
    }
}
