//! The pooled indexer's context-parallel page addressing against the host's
//! page ownership (`runtime_kv::ContextParallelPagesV1`): every pool a group
//! rank owns is the one the host places in that rank's local page, at the
//! same row, and a rank's local pools enumerate exactly its positions.
use kernel_core::contract::DType;
use kernels_sm120::pooled_indexer::{Geometry, pool_address, rank_pool, rank_pools};
use runtime_kv::ContextParallelPagesV1;

fn geometry() -> Geometry {
    Geometry {
        heads: 32,
        head_dim: 128,
        group: 4,
        page_size: 64,
        max_top_k: 2048,
        always_tail: true,
        position_dtype: DType::Bf16,
        key_norm_eps: 1e-6,
        verify_rows: 6,
    }
}

#[test]
fn pool_addresses_follow_the_host_stripe_tables() {
    let g = geometry();
    let group = g.group as usize;
    for size in [1_usize, 2, 4, 8] {
        let pages = ContextParallelPagesV1::new(size, g.page_size as usize, 8).unwrap();
        for tokens in [1_usize, 63, 64, 65, 255, 256, 1000, 4096, 4099, 70_001] {
            let stripes = tokens.div_ceil(pages.stripe_tokens());
            // Local page ids in an order unlike the stripe index.
            let table = (0..stripes).map(|s| 3 * stripes - s).collect::<Vec<_>>();
            let pools = tokens / group;
            for rank in 0..size {
                let held = pages
                    .rank_pages(rank, table.iter().copied(), tokens)
                    .unwrap();
                for pool in 0..pools {
                    let token = pool * group;
                    let address = pool_address(&g, pool as u64, rank as u32, size as u32);
                    assert_eq!(address.is_some(), pages.owner(token) == rank);
                    let Some(address) = address else { continue };
                    let page = held
                        .iter()
                        .find(|p| (p.first_position..p.first_position + p.tokens).contains(&token))
                        .unwrap();
                    assert_eq!(table[address.stripe as usize], page.local_page);
                    assert_eq!(address.row as usize, (token - page.first_position) / group);
                }
                let owned = rank_pools(&g, pools as u64, rank as u32, size as u32) as usize;
                // Pages hold whole pools: a rank's pools are its positions'
                // complete groups.
                assert_eq!(owned * group, pages.rank_tokens(rank, pools * group));
                for local in 0..owned {
                    let pool = rank_pool(&g, local as u64, rank as u32, size as u32) as usize;
                    assert_eq!(pool * group, pages.rank_position(rank, local * group));
                    let address = pool_address(&g, pool as u64, rank as u32, size as u32).unwrap();
                    assert_eq!(address.stripe as usize, local / 16);
                    assert_eq!(address.row as usize, local % 16);
                }
            }
            // At size 1 the table is the plain page table.
            if size == 1 {
                for pool in 0..pools {
                    let address = pool_address(&g, pool as u64, 0, 1).unwrap();
                    assert_eq!(address.stripe as usize, pool / 16);
                    assert_eq!(rank_pool(&g, pool as u64, 0, 1) as usize, pool);
                }
            }
        }
    }
}
