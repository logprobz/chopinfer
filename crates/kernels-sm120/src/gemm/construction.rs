//! Dense construction from a typed operation rather than a model profile.

use super::*;

#[derive(Clone, Debug)]
pub struct DenseRequest {
    pub query: GemmQuery,
    /// The kernel and the K reduction it fixes, which decide output bits.
    pub kernel: DenseKernel,
    pub target: Target,
    pub symbol: String,
    /// A consumer can retain measured tuning. Every override passes the same
    /// geometry, pipeline and resource checks as automatically derived knobs.
    pub knobs: Option<CompileKnobs>,
}

pub fn instantiate(request: DenseRequest) -> Result<InstantiationSpec, Refusal> {
    dense_supported(request.kernel, &request.query)?;
    let query = request.query;
    if is_fragment(request.kernel) && !query.k.is_multiple_of(128) {
        return Err(Refusal(
            "fragment GEMM requires complete 128-column K blocks",
        ));
    }
    let weight_row_bytes = match query.weight {
        WeightFormat::Bf16 => u64::from(query.k) * 2,
        WeightFormat::Mxfp4 | WeightFormat::Nvfp4 => u64::from(query.k) / 2,
        _ => u64::from(query.k),
    };
    if !is_fragment(request.kernel) && !weight_row_bytes.is_multiple_of(16) {
        return Err(Refusal(
            "dense packed row stride requires 16-byte alignment; provide explicit padded geometry",
        ));
    }
    let tile_n = if query.m_capacity <= 32 {
        query.m_capacity.next_power_of_two().max(8) as u16
    } else if output_n(query) >= 128 && output_n(query).is_multiple_of(128) {
        128
    } else {
        64
    };
    let mut spec = construct(request.symbol, query, 64, tile_n, request.kernel);
    spec.key.target = request.target;
    if let Some(knobs) = request.knobs {
        spec.key.knobs = knobs;
    }
    spec.validate()?;
    let split_k = required_split_k(&spec.key).unwrap_or(choose_split_k(&spec.key)?);
    plan(
        &spec.key,
        RuntimeCounts {
            m: query.m_capacity,
            batch: 1,
            context: 0,
            routed_rows: 0,
            split_k,
        },
    )?;
    // Code generation is pure. Check it here so a legal-looking format/query
    // cannot escape preparation without an executable engine body.
    cubin_spec(std::slice::from_ref(&spec))?;
    Ok(spec)
}
