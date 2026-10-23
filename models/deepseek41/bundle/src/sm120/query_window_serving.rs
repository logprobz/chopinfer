//! The query window twin's launch (`query_window`) from the served launches
//! it replaces in a one-row decode step: the served RoPE's and the served
//! fused window write's, built by their prepared ops over the step's
//! bindings, so the twin binds exactly what the pair binds.
use super::{latent_scheduled, query_window, window_fused};
use kernel_core::backend::Launch;
use kernel_core::contract::{Argument, ArgumentList, Refusal};

/// The RoPE's arguments: query, native query, table, live count, heads,
/// status, status row offset.
const ROPE_ARGUMENTS: usize = 7;
/// The fused window write's arguments; its live count, status and status
/// row offset are at 4, 6 and 7.
const WINDOW_ARGUMENTS: usize = 21;

impl crate::Kernels {
    /// Whether `shape`'s one-row decode runs its query RoPE and window
    /// write as one launch (`query_window::serves`).
    pub(crate) fn query_window_serves(shape: &crate::shape::Shape) -> bool {
        query_window::serves(shape.geometry, shape.ranks())
    }

    /// The twin's launch from one step's served RoPE launch and served
    /// window write launch, of one row each, sharing their live count and
    /// status row.
    pub(crate) fn query_window_launch(
        rope: Vec<Launch>,
        window: Vec<Launch>,
    ) -> Result<Launch, Refusal> {
        twin_launch(rope, window)
    }
}

fn twin_launch(rope: Vec<Launch>, window: Vec<Launch>) -> Result<Launch, Refusal> {
    let [rope] = <[Launch; 1]>::try_from(rope)
        .map_err(|_| Refusal("the query window twin needs the one served RoPE launch"))?;
    let [window] = <[Launch; 1]>::try_from(window)
        .map_err(|_| Refusal("the query window twin needs the one served window launch"))?;
    let served_rope = latent_scheduled::query();
    let served_window = window_fused::entries()
        .into_iter()
        .find(|w| w.symbol == window.symbol)
        .ok_or(Refusal(
            "the query window twin needs the served fused window write",
        ))?;
    let twin = query_window::entries()
        .into_iter()
        .find(|t| t.kv == served_window.kv)
        .ok_or(Refusal("no query window twin holds this window format"))?;
    let (r, w) = (&rope.arguments.0, &window.arguments.0);
    let Some(&Argument::U32(heads)) = r.get(4) else {
        return Err(Refusal("the served RoPE launch has no head count"));
    };
    if rope.symbol != served_rope.symbol
        || rope.rows != 1
        || window.rows != 1
        || rope.plan != served_rope.plan(1, heads)?
        || window.plan != served_window.plan(1)?
        || r.len() != ROPE_ARGUMENTS
        || w.len() != WINDOW_ARGUMENTS
        || rope.collective
        || window.collective
        || rope.persistent
        || window.persistent
    {
        return Err(Refusal(
            "the query window twin replaces one-row launches of the served RoPE and window write",
        ));
    }
    if r[3] != w[4] || r[5..7] != w[6..8] {
        return Err(Refusal(
            "the query window twin needs one live count and status row for both halves",
        ));
    }
    let mut arguments = vec![r[0].clone(), r[1].clone(), r[2].clone(), r[4].clone()];
    arguments.extend(w.iter().cloned());
    Ok(Launch {
        symbol: twin.symbol.clone(),
        rows: 1,
        plan: twin.plan(1, heads)?,
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served(kv_symbol: &str) -> (Launch, Launch) {
        let rope = latent_scheduled::query();
        let window = window_fused::entries()
            .into_iter()
            .find(|w| w.symbol == kv_symbol)
            .unwrap();
        let buffer = Argument::DeviceBuffer;
        let rope = Launch {
            symbol: rope.symbol.clone(),
            rows: 1,
            plan: rope.plan(1, 16).unwrap(),
            arguments: ArgumentList(vec![
                buffer(1),
                buffer(2),
                buffer(3),
                buffer(4),
                Argument::U32(16),
                buffer(5),
                Argument::U64(0),
            ]),
            collective: false,
            persistent: false,
        };
        let mut arguments = (0..WINDOW_ARGUMENTS as u64)
            .map(|i| buffer(100 + i))
            .collect::<Vec<_>>();
        arguments[4] = buffer(4);
        arguments[5] = Argument::U32(7);
        arguments[6] = buffer(5);
        arguments[7] = Argument::U64(0);
        let window = Launch {
            symbol: window.symbol.clone(),
            rows: 1,
            plan: window.plan(1).unwrap(),
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        };
        (rope, window)
    }

    /// The twin takes the RoPE's query, native query, table and heads, then
    /// every window argument, and refuses launches it would not bind alike.
    #[test]
    fn the_twin_binds_the_served_pairs_arguments() {
        for (window_symbol, twin_symbol) in [
            (
                "chop_deepseek41_window_fused_native_v1",
                "chop_deepseek41_query_window_twin_native_v1",
            ),
            (
                "chop_deepseek41_window_fused_bf16_v1",
                "chop_deepseek41_query_window_twin_bf16_v1",
            ),
        ] {
            let (rope, window) = served(window_symbol);
            let twin = twin_launch(vec![rope.clone()], vec![window.clone()]).unwrap();
            assert_eq!(twin.symbol, twin_symbol);
            assert_eq!(twin.plan.grid, [1, 2, 1]);
            assert_eq!(twin.plan.cluster, [1, 2, 1]);
            assert_eq!(
                twin.arguments.0[..4],
                [0, 1, 2, 4].map(|i| rope.arguments.0[i].clone())
            );
            assert_eq!(twin.arguments.0[4..], window.arguments.0[..]);
            let entry = query_window::entries()
                .into_iter()
                .find(|t| t.symbol == twin_symbol)
                .unwrap();
            assert_eq!(twin.arguments.0.len(), entry.spec().params.len());

            let refused =
                |rope: Launch, window: Launch| twin_launch(vec![rope], vec![window]).is_err();
            let mut other = window.clone();
            other.arguments.0[4] = Argument::DeviceBuffer(9);
            assert!(refused(rope.clone(), other));
            let mut other = window.clone();
            other.arguments.0[7] = Argument::U64(1);
            assert!(refused(rope.clone(), other));
            let mut other = rope.clone();
            other.symbol = "chop_deepseek41_latent_query_rope_v1".into();
            assert!(refused(other, window.clone()));
            let mut other = rope.clone();
            other.rows = 2;
            assert!(refused(other, window.clone()));
            let mut other = window.clone();
            other.symbol = "chop_deepseek41_latent_window_store_v1".into();
            assert!(refused(rope.clone(), other));
            assert!(twin_launch(vec![rope.clone(), rope.clone()], vec![window.clone()]).is_err());
            assert!(twin_launch(vec![rope.clone()], Vec::new()).is_err());
        }
    }
}
