//! Lowered programs kept as templates and reused across steps.
//!
//! A bundle lowers a program in full, records as sites the launches that
//! read typed per-step inputs, and keeps the program under a key it derives
//! from every other input its lowering read. A later step with an equal key
//! lowers only the sites again, from its own inputs, and writes them over
//! the kept launches (`patch`). A site that lowers to another launch count
//! or other collective flags is structure the key does not hold: the patch
//! writes nothing and refuses, and the step tries the key's other
//! templates, then lowers in full and keeps another. So one key may keep a
//! template for each launch count its sites take, such as the batches of
//! two context classes. `same_programs` compares a patched program with a
//! fresh lowering, for a bundle's verify mode.
//!
//! Keys and site inputs are the bundle's; this module never reads them.
//! Kept launches hold buffer views, so a template stays valid only while
//! the resources the bundle lowered against stay fixed.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::dispatch::{DispatchError, arguments, launch_config};
use crate::pipeline::PreparedLaunch;

/// Launches `at..at + len` of rank `rank`'s program, which the bundle
/// lowers again from `input` on every reuse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Site<S> {
    pub rank: usize,
    pub at: usize,
    pub len: usize,
    pub input: S,
}

/// Why a patch wrote nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Refusal {
    /// A site lies outside its rank's program, or before or across the
    /// previous site of its rank.
    Extent,
    /// A site lowered to another launch count.
    Launches,
    /// A site lowered to other collective flags.
    Collectives,
}

/// What a patch did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub enum Patch {
    Applied,
    Refused(Refusal),
}

/// Whether `sites` are in rank and launch order, disjoint, and within
/// `programs` and their collective flags. A site may be empty: it records
/// where a lowering issued nothing, which a patch must lower to nothing
/// again.
pub fn check_sites<S>(
    programs: &[Vec<PreparedLaunch>],
    collectives: &[Vec<bool>],
    sites: &[Site<S>],
) -> Result<(), Refusal> {
    if programs.len() != collectives.len() {
        return Err(Refusal::Extent);
    }
    let mut end = (0, 0);
    for site in sites {
        let within = programs
            .get(site.rank)
            .zip(collectives.get(site.rank))
            .is_some_and(|(program, flags)| {
                program.len() == flags.len()
                    && site
                        .at
                        .checked_add(site.len)
                        .is_some_and(|last| last <= program.len())
            });
        if !within || (site.rank, site.at) < end {
            return Err(Refusal::Extent);
        }
        end = (site.rank, site.at + site.len);
    }
    Ok(())
}

/// Lowers every site again with `lower`, as launches and their collective
/// flags, and writes them over `programs`. Every site lowers before any is
/// written, so an error or a refusal leaves `programs` as they were.
pub fn patch<S, E>(
    programs: &mut [Vec<PreparedLaunch>],
    collectives: &[Vec<bool>],
    sites: &[Site<S>],
    mut lower: impl FnMut(&Site<S>) -> Result<Vec<(PreparedLaunch, bool)>, E>,
) -> Result<Patch, E> {
    if let Err(refusal) = check_sites(programs, collectives, sites) {
        return Ok(Patch::Refused(refusal));
    }
    let mut lowered = Vec::with_capacity(sites.len());
    for site in sites {
        let launches = lower(site)?;
        if launches.len() != site.len {
            return Ok(Patch::Refused(Refusal::Launches));
        }
        let kept = &collectives[site.rank][site.at..site.at + site.len];
        if launches
            .iter()
            .map(|(_, flag)| *flag)
            .ne(kept.iter().copied())
        {
            return Ok(Patch::Refused(Refusal::Collectives));
        }
        lowered.push(launches);
    }
    for (site, launches) in sites.iter().zip(lowered) {
        let slots = &mut programs[site.rank][site.at..site.at + site.len];
        for (slot, (launch, _)) in slots.iter_mut().zip(launches) {
            *slot = launch;
        }
    }
    Ok(Patch::Applied)
}

/// Whether two launches are the same call as the host resolves them: the
/// symbol, rows, plan and configuration, every argument with its bound
/// view, and the selected route entry their counters report.
pub fn same_launch(a: &PreparedLaunch, b: &PreparedLaunch) -> Result<bool, DispatchError> {
    Ok(a.symbol == b.symbol
        && a.rows == b.rows
        && a.plan == b.plan
        && a.bindings.route() == b.bindings.route()
        && launch_config(a.plan)? == launch_config(b.plan)?
        && arguments(&a.arguments, &a.bindings)? == arguments(&b.arguments, &b.bindings)?)
}

/// Whether two multi-rank programs issue the same launches (`same_launch`)
/// with the same collective flags on every rank.
pub fn same_programs(
    a: &[impl AsRef<[PreparedLaunch]>],
    a_collectives: &[impl AsRef<[bool]>],
    b: &[impl AsRef<[PreparedLaunch]>],
    b_collectives: &[impl AsRef<[bool]>],
) -> Result<bool, DispatchError> {
    if a.len() != b.len() || a_collectives.len() != b_collectives.len() {
        return Ok(false);
    }
    for (a, b) in a_collectives.iter().zip(b_collectives) {
        if a.as_ref() != b.as_ref() {
            return Ok(false);
        }
    }
    for (a, b) in a.iter().zip(b) {
        let (a, b) = (a.as_ref(), b.as_ref());
        if a.len() != b.len() {
            return Ok(false);
        }
        for (a, b) in a.iter().zip(b) {
            if !same_launch(a, b)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// What a cache did so far, one count per reuse and per keep.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Counts {
    /// Reuses that patched a template.
    pub hits: u64,
    /// Reuses of a key without templates.
    pub misses: u64,
    /// Reuses whose key's templates a step still held.
    pub busy: u64,
    /// Reuses whose key's free templates all refused their patch.
    pub refused: u64,
    /// Templates kept.
    pub kept: u64,
    /// Templates dropped to stay within the limit, or after a failed patch.
    pub evicted: u64,
}

struct Kept<K, T> {
    key: K,
    template: Rc<T>,
    /// The cache's use count when this template was last kept or reused.
    used: u64,
}

/// At most `limit` templates by key; keeping one more drops the least
/// recently used. A template is patched in place, so it is reused only when
/// no earlier step still holds it.
pub struct Templates<K, T> {
    limit: usize,
    kept: RefCell<Vec<Kept<K, T>>>,
    uses: Cell<u64>,
    counts: Cell<Counts>,
}

impl<K: PartialEq, T> Templates<K, T> {
    /// A cache of at most `limit` templates, at least one.
    pub fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            kept: RefCell::new(Vec::new()),
            uses: Cell::new(0),
            counts: Cell::new(Counts::default()),
        }
    }

    pub fn counts(&self) -> Counts {
        self.counts.get()
    }

    /// The most templates kept at once.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// The templates kept now.
    pub fn len(&self) -> usize {
        self.kept.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn count(&self, change: impl FnOnce(&mut Counts)) {
        let mut counts = self.counts.get();
        change(&mut counts);
        self.counts.set(counts);
    }

    fn next_use(&self) -> u64 {
        self.uses.set(self.uses.get() + 1);
        self.uses.get()
    }

    /// A template of `key` once `patch` applied this step's sites to it,
    /// trying the key's free templates from the most recently used; `None`
    /// when the key has none, a step still holds them, or every patch
    /// refused. A template whose patch fails is dropped.
    pub fn reuse<E>(
        &self,
        key: &K,
        mut patch: impl FnMut(&mut T) -> Result<Patch, E>,
    ) -> Result<Option<Rc<T>>, E> {
        let mut kept = self.kept.borrow_mut();
        let mut order: Vec<_> = (0..kept.len()).filter(|&at| kept[at].key == *key).collect();
        if order.is_empty() {
            self.count(|c| c.misses += 1);
            return Ok(None);
        }
        order.sort_unstable_by_key(|&at| std::cmp::Reverse(kept[at].used));
        let mut refused = false;
        for at in order {
            let Some(template) = Rc::get_mut(&mut kept[at].template) else {
                continue;
            };
            match patch(template) {
                Ok(Patch::Applied) => {
                    self.count(|c| c.hits += 1);
                    kept[at].used = self.next_use();
                    return Ok(Some(Rc::clone(&kept[at].template)));
                }
                Ok(Patch::Refused(_)) => refused = true,
                Err(error) => {
                    kept.remove(at);
                    self.count(|c| c.evicted += 1);
                    return Err(error);
                }
            }
        }
        if refused {
            self.count(|c| c.refused += 1);
        } else {
            self.count(|c| c.busy += 1);
        }
        Ok(None)
    }

    /// Keeps `template` under `key`, beside the key's other templates;
    /// returns the least recently used template when it was dropped to stay
    /// within the limit, for the caller to free when it chooses.
    pub fn keep(&self, key: K, template: Rc<T>) -> Option<Rc<T>> {
        let used = self.next_use();
        let mut kept = self.kept.borrow_mut();
        self.count(|c| c.kept += 1);
        let mut dropped = None;
        if kept.len() >= self.limit
            && let Some(oldest) = (0..kept.len()).min_by_key(|&at| kept[at].used)
        {
            dropped = Some(kept.remove(oldest).template);
            self.count(|c| c.evicted += 1);
        }
        kept.push(Kept {
            key,
            template,
            used,
        });
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::BufferBindings;
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    /// One launch of `symbol` reading `offset` of a buffer and `scalar`,
    /// bound after `unused` other views, so planner IDs differ.
    fn launch(symbol: &str, offset: u64, scalar: u32, unused: u64) -> PreparedLaunch {
        let mut bindings = BufferBindings::default();
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        for at in 0..unused {
            bindings.bind(DeviceBufferSpanV1::new(buffer, at * 512, 256));
        }
        let view = bindings.bind(DeviceBufferSpanV1::new(buffer, offset, 256));
        PreparedLaunch {
            symbol: symbol.into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: false,
            },
            arguments: ArgumentList(vec![Argument::DeviceBuffer(view), Argument::U32(scalar)]),
            bindings,
        }
    }

    fn tagged(mut launch: PreparedLaunch, family: &'static str) -> PreparedLaunch {
        launch
            .bindings
            .set_route(Some(kernel_core::route::RouteTag::new(
                "k",
                kernel_core::route::RouteClass::Fast,
                family,
                [1..=8],
            )));
        launch
    }

    #[test]
    fn launches_compare_by_what_the_host_resolves_and_their_route() {
        let same = |a, b| same_launch(&a, &b).unwrap();
        assert!(same(launch("k", 256, 7, 0), launch("k", 256, 7, 0)));
        // Other planner IDs of the same views resolve to the same launch.
        assert!(same(launch("k", 256, 7, 0), launch("k", 256, 7, 3)));
        assert!(!same(launch("k", 256, 7, 0), launch("j", 256, 7, 0)));
        assert!(!same(launch("k", 256, 7, 0), launch("k", 768, 7, 0)));
        assert!(!same(launch("k", 256, 7, 0), launch("k", 256, 8, 0)));
        let mut wider = launch("k", 256, 7, 0);
        wider.plan.grid[0] = 2;
        assert!(!same(launch("k", 256, 7, 0), wider));
        assert!(!same(
            launch("k", 256, 7, 0),
            tagged(launch("k", 256, 7, 0), "f")
        ));
        assert!(!same(
            tagged(launch("k", 256, 7, 0), "f"),
            tagged(launch("k", 256, 7, 0), "g")
        ));
        // An unbound view never compares equal: the host would refuse it.
        let mut unbound = launch("k", 256, 7, 0);
        unbound.arguments.0[0] = Argument::DeviceBuffer(9);
        assert!(same_launch(&unbound, &launch("k", 256, 7, 0)).is_err());
    }

    #[test]
    fn programs_compare_launches_and_collective_flags_on_every_rank() {
        let program = |scalar, flag| (vec![vec![launch("k", 256, scalar, 0)]], vec![vec![flag]]);
        let same = |a: (Vec<Vec<PreparedLaunch>>, Vec<Vec<bool>>),
                    b: (Vec<Vec<PreparedLaunch>>, Vec<Vec<bool>>)| {
            same_programs(&a.0, &a.1, &b.0, &b.1).unwrap()
        };
        assert!(same(program(7, false), program(7, false)));
        assert!(!same(program(7, false), program(8, false)));
        assert!(!same(program(7, false), program(7, true)));
        let (mut two, flags) = program(7, false);
        two[0].push(launch("k", 256, 7, 0));
        assert!(!same(program(7, false), (two, flags)));
        assert!(!same(program(7, false), (Vec::new(), Vec::new())));
    }

    /// Programs, their collective flags and their sites.
    type Program = (Vec<Vec<PreparedLaunch>>, Vec<Vec<bool>>, Vec<Site<u32>>);

    /// Two ranks of three launches with scalar `n`, the middle one of each
    /// a site, and the second rank's last launch a collective.
    fn kept(n: u32) -> Program {
        let programs = (0..2)
            .map(|_| (0..3).map(|_| launch("k", 256, n, 0)).collect())
            .collect();
        let flags = vec![vec![false; 3], vec![false, false, true]];
        let sites = (0..2)
            .map(|rank| Site {
                rank,
                at: 1,
                len: 1,
                input: rank as u32,
            })
            .collect();
        (programs, flags, sites)
    }

    fn scalars(programs: &[Vec<PreparedLaunch>]) -> Vec<Vec<Argument>> {
        programs
            .iter()
            .map(|program| program.iter().map(|l| l.arguments.0[1].clone()).collect())
            .collect()
    }

    #[test]
    fn patches_rewrite_only_their_sites_from_the_new_inputs() {
        let (mut programs, flags, sites) = kept(7);
        let mut seen = Vec::new();
        let patched = patch(&mut programs, &flags, &sites, |site| {
            seen.push(site.input);
            Ok::<_, ()>(vec![(launch("j", 512, 40 + site.input, 0), false)])
        });
        assert_eq!(patched, Ok(Patch::Applied));
        assert_eq!(seen, [0, 1]);
        assert_eq!(
            scalars(&programs),
            [
                [Argument::U32(7), Argument::U32(40), Argument::U32(7)],
                [Argument::U32(7), Argument::U32(41), Argument::U32(7)],
            ]
        );
        assert_eq!(programs[1][1].symbol, "j");
        assert_eq!(programs[1][2].symbol, "k");
    }

    #[test]
    fn refused_and_failed_patches_write_nothing() {
        let untouched = scalars(&kept(7).0);
        // The second rank's site lowers to two launches, then to a
        // collective, then fails: the first rank's lowered site is never
        // written.
        let refusals: [(&str, Result<Patch, &str>); 3] = [
            ("two", Ok(Patch::Refused(Refusal::Launches))),
            ("collective", Ok(Patch::Refused(Refusal::Collectives))),
            ("error", Err("lowering failed")),
        ];
        for (how, expected) in refusals {
            let (mut programs, flags, sites) = kept(7);
            let patched = patch(&mut programs, &flags, &sites, |site| {
                let one = || (launch("j", 512, 9, 0), false);
                Ok(match (site.input, how) {
                    (0, _) => vec![one()],
                    (_, "two") => vec![one(), one()],
                    (_, "collective") => vec![(launch("j", 512, 9, 0), true)],
                    _ => return Err("lowering failed"),
                })
            });
            assert_eq!(patched, expected, "{how}");
            assert_eq!(scalars(&programs), untouched, "{how}");
        }
    }

    #[test]
    fn sites_must_be_ordered_disjoint_and_within_their_programs() {
        let (programs, flags, sites) = kept(7);
        assert_eq!(check_sites(&programs, &flags, &sites), Ok(()));
        let site = |rank, at, len| Site {
            rank,
            at,
            len,
            input: 0,
        };
        for bad in [
            vec![site(2, 0, 1)],
            vec![site(0, 2, 2)],
            vec![site(0, 4, 0)],
            vec![site(0, 0, 2), site(0, 1, 1)],
            vec![site(0, 1, 1), site(0, 1, 0)],
            vec![site(1, 0, 1), site(0, 0, 1)],
            vec![site(0, usize::MAX, 2)],
        ] {
            assert_eq!(check_sites(&programs, &flags, &bad), Err(Refusal::Extent));
            let mut programs = kept(7).0;
            assert_eq!(
                patch(&mut programs, &flags, &bad, |_| Ok::<_, ()>(Vec::new())),
                Ok(Patch::Refused(Refusal::Extent))
            );
        }
        let short = vec![vec![false; 2], vec![false; 3]];
        assert_eq!(check_sites(&programs, &short, &sites), Err(Refusal::Extent));
        // An empty site must lower to nothing again.
        let empty = [site(0, 1, 0), site(0, 1, 1)];
        assert_eq!(check_sites(&programs, &flags, &empty), Ok(()));
        let mut patched = kept(7).0;
        assert_eq!(
            patch(&mut patched, &flags, &empty, |site| Ok::<_, ()>(
                (0..site.len + 1)
                    .map(|_| (launch("j", 512, 9, 0), false))
                    .collect()
            )),
            Ok(Patch::Refused(Refusal::Launches))
        );
        assert_eq!(scalars(&patched), scalars(&kept(7).0));
    }

    #[test]
    fn caches_reuse_only_free_templates_and_drop_failed_ones() {
        let cache = Templates::<u32, Vec<u32>>::new(4);
        let applied = |t: &mut Vec<u32>| {
            t.push(1);
            Ok::<_, ()>(Patch::Applied)
        };
        assert_eq!(cache.reuse(&1, applied), Ok(None));
        assert!(cache.keep(1, Rc::new(vec![0])).is_none());
        let held = cache.reuse(&1, applied).unwrap().unwrap();
        assert_eq!(*held, [0, 1]);
        // A step still holds the template: it is neither patched nor dropped.
        assert_eq!(cache.reuse(&1, applied), Ok(None));
        assert_eq!(*held, [0, 1]);
        drop(held);
        assert_eq!(*cache.reuse(&1, applied).unwrap().unwrap(), [0, 1, 1]);
        // A refusal writes nothing, so the template stays.
        assert_eq!(
            cache.reuse(&1, |_| Ok::<_, ()>(Patch::Refused(Refusal::Launches))),
            Ok(None)
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(
            cache.reuse(&1, |_| Err::<Patch, _>("failed")),
            Err("failed")
        );
        assert!(cache.is_empty());
        assert_eq!(
            cache.counts(),
            Counts {
                hits: 2,
                misses: 1,
                busy: 1,
                refused: 1,
                kept: 1,
                evicted: 1,
            }
        );
    }

    #[test]
    fn a_key_keeps_a_template_per_site_shape_and_patches_the_one_that_fits() {
        // Each template is the launch count its sites take; a step fits
        // the template of its own count.
        let cache = Templates::<u32, u32>::new(4);
        let fits = |count: u32| {
            move |t: &mut u32| {
                Ok::<_, ()>(if *t == count {
                    Patch::Applied
                } else {
                    Patch::Refused(Refusal::Launches)
                })
            }
        };
        cache.keep(1, Rc::new(8));
        assert_eq!(cache.reuse(&1, fits(4)), Ok(None));
        cache.keep(1, Rc::new(4));
        let mut tried = Vec::new();
        let found = cache
            .reuse(&1, |t: &mut u32| {
                tried.push(*t);
                fits(8)(t)
            })
            .unwrap();
        // The most recently used first.
        assert_eq!(tried, [4, 8]);
        assert_eq!(found.as_deref(), Some(&8));
        assert_eq!(cache.reuse(&1, fits(4)).unwrap().as_deref(), Some(&4));
        assert_eq!(cache.reuse(&2, fits(4)), Ok(None));
        assert_eq!(cache.counts().refused, 1);
        assert_eq!(cache.counts().hits, 2);
    }

    #[test]
    fn caches_drop_the_least_recently_used_template() {
        let cache = Templates::<u32, u32>::new(2);
        let applied = |_: &mut u32| Ok::<_, ()>(Patch::Applied);
        assert!(cache.keep(1, Rc::new(10)).is_none());
        assert!(cache.keep(2, Rc::new(20)).is_none());
        // Reusing 1 makes 2 the oldest, which the caller receives.
        assert_eq!(*cache.reuse(&1, applied).unwrap().unwrap(), 10);
        assert_eq!(cache.keep(3, Rc::new(30)).as_deref(), Some(&20));
        assert_eq!(cache.reuse(&2, applied), Ok(None));
        assert_eq!(*cache.reuse(&1, applied).unwrap().unwrap(), 10);
        assert_eq!(*cache.reuse(&3, applied).unwrap().unwrap(), 30);
        assert_eq!(cache.counts().evicted, 1);
        assert_eq!(Templates::<u32, u32>::new(0).limit, 1);
    }
}
