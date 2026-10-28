//! Optional, thread-local host annotations. Observers must not submit device work.
use std::cell::Cell;
use std::ffi::CStr;
use std::marker::PhantomData;
use std::rc::Rc;

pub type Observer = fn(&'static CStr, bool);

thread_local! {
    static OBSERVER: Cell<Option<Observer>> = const { Cell::new(None) };
}

/// Restores this thread's observer when dropped. Existing ranges keep their observer.
pub struct Subscription {
    previous: Option<Observer>,
    _thread: PhantomData<Rc<()>>,
}

#[must_use]
pub fn subscribe(observer: Observer) -> Subscription {
    Subscription {
        previous: OBSERVER.replace(Some(observer)),
        _thread: PhantomData,
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        OBSERVER.set(self.previous);
    }
}

/// A balanced annotation on its creating thread, including error unwinding.
pub struct Range {
    observer: Observer,
    name: &'static CStr,
    _thread: PhantomData<Rc<()>>,
}

impl Range {
    #[must_use]
    pub fn new(name: &'static CStr) -> Option<Self> {
        let observer = OBSERVER.get()?;
        observer(name, true);
        Some(Self {
            observer,
            name,
            _thread: PhantomData,
        })
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        (self.observer)(self.name, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static EVENTS: RefCell<Vec<(&'static CStr, bool)>> = const { RefCell::new(Vec::new()) };
    }

    fn observe(name: &'static CStr, begin: bool) {
        EVENTS.with_borrow_mut(|events| events.push((name, begin)));
    }

    #[test]
    fn disabled_ranges_do_nothing() {
        assert!(Range::new(c"unused").is_none());
    }

    #[test]
    fn ranges_balance_on_unwind_and_stay_on_their_thread() {
        EVENTS.with_borrow_mut(Vec::clear);
        let subscription = subscribe(observe);
        let result = std::panic::catch_unwind(|| {
            let _outer = Range::new(c"outer");
            std::thread::spawn(|| assert!(Range::new(c"child").is_none()))
                .join()
                .unwrap();
            let _inner = Range::new(c"inner");
            panic!("fixture unwind");
        });
        assert!(result.is_err());
        drop(subscription);
        assert!(Range::new(c"after").is_none());
        EVENTS.with_borrow(|events| {
            assert_eq!(
                events.as_slice(),
                &[
                    (c"outer", true),
                    (c"inner", true),
                    (c"inner", false),
                    (c"outer", false),
                ]
            )
        });
    }
}
