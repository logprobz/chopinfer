//! Query rotation with four head groups inside one row-wide failure barrier.

use super::glue;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

pub const THREADS: u32 = 1024;

#[derive(Clone, Debug)]
pub struct Entry {
    pub capacity: u32,
    pub symbol: String,
}

impl Entry {
    pub fn binding(&self) -> glue::Entry {
        glue::Entry {
            kernel: glue::Kernel::QueryRopeParallel,
            capacity: self.capacity,
            symbol: self.symbol.clone(),
        }
    }

    fn reference(&self) -> glue::Entry {
        glue::Entry {
            kernel: glue::Kernel::QueryRope,
            capacity: self.capacity,
            symbol: self.symbol.clone(),
        }
    }

    pub fn plan(&self, rows: u32, heads: u32) -> Result<LaunchPlan, Refusal> {
        let mut plan = self.reference().plan(rows, heads)?;
        plan.block[0] = THREADS;
        plan.validate()?;
        Ok(plan)
    }

    pub fn spec(&self) -> EntrySpec {
        let mut spec = self.reference().spec();
        spec.max_threads = THREADS;
        spec.body = spec.body.replace(
            "chop::latent::glue::rotate<true>",
            "chop::latent::query_parallel::rotate",
        );
        spec
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    let mut spec = glue::cubin_spec(name, &[]);
    spec.headers = vec!["#include \"latent/query_parallel.cuh\"".into()];
    spec.entries = entries.iter().map(Entry::spec).collect();
    spec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_head_groups_preserve_the_row_and_argument_contract() {
        let entry = Entry {
            capacity: 192,
            symbol: "query_groups_test_v1".into(),
        };
        assert_eq!(entry.spec().params, entry.reference().spec().params);
        assert_eq!(entry.spec().body, entry.binding().spec().body);
        for rows in [1, 6, 8, 32, 48, 192] {
            for heads in [1, 3, 4, 16, 32, 64] {
                let plan = entry.plan(rows, heads).unwrap();
                assert_eq!(plan, entry.binding().plan(rows, heads).unwrap());
                assert_eq!(plan.grid, [rows, 1, 1]);
                assert_eq!(plan.block, [1024, 1, 1]);
                let mut owners = vec![0; heads as usize];
                for group in 0..4 {
                    for head in (group..heads).step_by(4) {
                        owners[head as usize] += 1;
                    }
                }
                assert!(owners.iter().all(|&count| count == 1));
            }
        }
        assert!(entry.plan(193, 16).is_err());
    }
}
