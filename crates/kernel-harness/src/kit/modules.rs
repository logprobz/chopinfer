//! How the kit loads a backend's symbols, which `Backend` does not cover:
//! a backend names symbols in its launches and its `OpInfo`, but not the
//! modules that hold them. A backend's modules also supply the PDL
//! producer probes of its architecture, if it has them. A consumer
//! implements [`Modules`] over the cubins it builds or embeds.

use crate::conformance::Probes;
use crate::{LabResult, Session};
use runtime_cuda::Function;

/// A symbol's function on a session, configured by the caller.
pub type Lookup<'s> = dyn Fn(&str) -> LabResult<Function> + 's;

/// An owned [`Lookup`].
pub type Functions<'s> = Box<Lookup<'s>>;

/// Loads the modules that hold a backend's symbols.
pub trait Modules {
    /// The lookup of every symbol the backend may launch on `session`.
    /// Modules load at most once per lookup.
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>>;

    /// The PDL producer probes of the backend's architecture on `session`,
    /// which the conformance run launches ahead of an op's first launch to
    /// probe its dependency order ([`Probes`]). A backend that supplies
    /// none (the default) gets that check reported as not exercised.
    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let _ = session;
        Ok(None)
    }
}
