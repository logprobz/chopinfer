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
