//! Hosts for indexer graph planning tests; device arithmetic is not executed.
use super::*;

pub(crate) fn host(ranks: Ranks, dspark: bool) -> (Rc<Host>, BundleInstanceV1) {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(32 * 4096);
    if dspark {
        host.speculation.set(Some((3, 32)));
    }
    let instance = create(&host, 32);
    (host, instance)
}

/// A complete (not partial) instance serving `items` on `layout`, with
/// the DSpark drafter at five drafts when `dspark`.
pub(crate) fn complete_host(
    ranks: Ranks,
    layout: KvLayout,
    dspark: bool,
    items: u32,
) -> (Rc<Host>, BundleInstanceV1) {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let host = FakeHost::new(shape, layout);
    host.kv_pages.set(32 * 4200);
    let services = if dspark {
        host.drafter.set(true);
        host.speculation.set(Some((5, (items * 6).min(32))));
        HostServicesTableV1::new(Rc::new(super::dspark::host::MappedHost::new(Rc::clone(
            &host,
        ))))
    } else {
        HostServicesTableV1::new(Rc::clone(&host))
    };
    let record = crate::config::tests::record(&shape, host.kv, items, false);
    let instance = api()
        .create(services, &record)
        .unwrap_or_else(|error| panic!("create: {error:?} {:?}", host.logs.borrow()));
    (host, instance)
}

/// A complete instance serving `items` with the served prefill schedule:
/// 8192-row chunks with the upper-layer skip, and prepared Engram gathers
/// where the host offers them (always with the DSpark drafter's mapped
/// host, else when `mapped`).
pub(crate) fn skip_host(
    ranks: Ranks,
    dspark: bool,
    mapped: bool,
    items: u32,
    options: &[(&str, serde_json::Value)],
) -> (Rc<Host>, BundleInstanceV1) {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(32 * 4200);
    // The drafter's package maps the Engram tables a prepared gather reads.
    host.drafter.set(dspark || mapped);
    if dspark {
        host.speculation.set(Some((5, (items * 6).min(32))));
    }
    let mut record = crate::config::tests::record(&shape, host.kv, items, false);
    for (name, value) in [
        ("prefill_chunk_tokens", serde_json::json!(8192)),
        ("prefill_upper_skip", serde_json::json!(true)),
    ]
    .into_iter()
    .chain(options.iter().map(|(name, value)| (*name, value.clone())))
    {
        record = crate::config::tests::with_option(record, name, value);
    }
    let services = if dspark || mapped {
        HostServicesTableV1::new(Rc::new(super::dspark::host::MappedHost::new(Rc::clone(
            &host,
        ))))
    } else {
        HostServicesTableV1::new(Rc::clone(&host))
    };
    let instance = api()
        .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
        .unwrap()
        .create(services, &record)
        .unwrap_or_else(|error| panic!("create: {error:?} {:?}", host.logs.borrow()));
    (host, instance)
}
