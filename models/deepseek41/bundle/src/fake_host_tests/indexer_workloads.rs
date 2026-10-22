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
