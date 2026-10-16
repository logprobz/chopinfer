use runtime_bundle_abi::BundleStreamHandleV1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamTransactionPhaseV1 {
    Ready,
    Enqueuing,
    Synchronizing,
    Releasing,
    Released,
    Quarantined,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamTransactionEvidenceV1 {
    pub phase: StreamTransactionPhaseV1,
    pub retained_streams: usize,
    pub potentially_enqueued_ranks: usize,
    pub accepted_ranks: usize,
    pub synchronized_ranks: usize,
    pub cleanup_failures: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamTransactionErrorV1<E> {
    RankSet,
    UnknownRank(u32),
    DuplicateLaunch(u32),
    NotReady,
    Released,
    Host {
        service: &'static str,
        rank: u32,
        error: E,
    },
}

#[derive(Debug)]
struct RankStreamV1<P> {
    rank: u32,
    stream: Option<BundleStreamHandleV1>,
    prepared: P,
    potentially_enqueued: bool,
    accepted: bool,
    synchronized: bool,
}

#[derive(Debug)]
pub struct StreamTransactionV1<P> {
    stream_kind: u32,
    phase: StreamTransactionPhaseV1,
    ranks: Vec<RankStreamV1<P>>,
    cleanup_failures: Vec<u32>,
}

#[derive(Debug)]
pub struct StreamAcquisitionFailureV1<P, E> {
    transaction: StreamTransactionV1<P>,
    error: StreamTransactionErrorV1<E>,
}

impl<P, E> StreamAcquisitionFailureV1<P, E> {
    #[must_use]
    pub const fn error(&self) -> &StreamTransactionErrorV1<E> {
        &self.error
    }

    #[must_use]
    pub fn evidence(&self) -> StreamTransactionEvidenceV1 {
        self.transaction.evidence()
    }

    pub fn retry_close<S, R>(
        &mut self,
        synchronize: S,
        release: R,
    ) -> Result<StreamTransactionEvidenceV1, StreamCloseFailureV1<E>>
    where
        S: FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
        R: FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
    {
        self.transaction.retry_close(synchronize, release)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamCloseFailureV1<E> {
    evidence: StreamTransactionEvidenceV1,
    failures: Vec<StreamTransactionErrorV1<E>>,
}

impl<E> StreamCloseFailureV1<E> {
    #[must_use]
    pub const fn evidence(&self) -> &StreamTransactionEvidenceV1 {
        &self.evidence
    }

    #[must_use]
    pub fn failures(&self) -> &[StreamTransactionErrorV1<E>] {
        &self.failures
    }
}

impl<P> StreamTransactionV1<P> {
    pub fn acquire<E>(
        stream_kind: u32,
        expected_ranks: &[u32],
        ranks: Vec<(u32, P)>,
        mut acquire: impl FnMut(u32, u32) -> Result<BundleStreamHandleV1, E>,
        mut rollback: impl FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
    ) -> Result<Self, StreamAcquisitionFailureV1<P, E>> {
        if ranks
            .iter()
            .map(|(rank, _)| *rank)
            .ne(expected_ranks.iter().copied())
        {
            return Err(StreamAcquisitionFailureV1 {
                transaction: Self {
                    stream_kind,
                    phase: StreamTransactionPhaseV1::Released,
                    ranks: Vec::new(),
                    cleanup_failures: Vec::new(),
                },
                error: StreamTransactionErrorV1::RankSet,
            });
        }

        let mut held = Vec::with_capacity(ranks.len());
        for (rank, prepared) in ranks {
            match acquire(rank, stream_kind) {
                Ok(stream) => held.push(RankStreamV1 {
                    rank,
                    stream: Some(stream),
                    prepared,
                    potentially_enqueued: false,
                    accepted: false,
                    synchronized: false,
                }),
                Err(error) => {
                    let mut transaction = Self {
                        stream_kind,
                        phase: StreamTransactionPhaseV1::Releasing,
                        ranks: held,
                        cleanup_failures: Vec::new(),
                    };
                    for held in transaction.ranks.iter_mut().rev() {
                        let stream = held.stream.expect("acquired streams are retained");
                        if rollback(held.rank, stream).is_ok() {
                            held.stream = None;
                        } else {
                            transaction.cleanup_failures.push(held.rank);
                        }
                    }
                    transaction.update_terminal_phase();
                    return Err(StreamAcquisitionFailureV1 {
                        transaction,
                        error: StreamTransactionErrorV1::Host {
                            service: "create_stream",
                            rank,
                            error,
                        },
                    });
                }
            }
        }

        Ok(Self {
            stream_kind,
            phase: StreamTransactionPhaseV1::Ready,
            ranks: held,
            cleanup_failures: Vec::new(),
        })
    }

    #[must_use]
    pub const fn stream_kind(&self) -> u32 {
        self.stream_kind
    }

    pub fn launch_rank<E>(
        &mut self,
        rank: u32,
        mut launch: impl FnMut(&P, BundleStreamHandleV1) -> Result<(), E>,
    ) -> Result<(), StreamTransactionErrorV1<E>> {
        if !matches!(
            self.phase,
            StreamTransactionPhaseV1::Ready | StreamTransactionPhaseV1::Enqueuing
        ) {
            return Err(if self.phase == StreamTransactionPhaseV1::Released {
                StreamTransactionErrorV1::Released
            } else {
                StreamTransactionErrorV1::NotReady
            });
        }
        let held = self
            .ranks
            .iter_mut()
            .find(|held| held.rank == rank)
            .ok_or(StreamTransactionErrorV1::UnknownRank(rank))?;
        if held.potentially_enqueued {
            return Err(StreamTransactionErrorV1::DuplicateLaunch(rank));
        }
        let stream = held.stream.ok_or(StreamTransactionErrorV1::Released)?;
        held.potentially_enqueued = true;
        self.phase = StreamTransactionPhaseV1::Enqueuing;
        launch(&held.prepared, stream).map_err(|error| StreamTransactionErrorV1::Host {
            service: "launch",
            rank,
            error,
        })?;
        held.accepted = true;
        Ok(())
    }

    pub fn launch_all<E>(
        &mut self,
        mut launch: impl FnMut(u32, &P, BundleStreamHandleV1) -> Result<(), E>,
    ) -> Result<(), StreamTransactionErrorV1<E>> {
        let ranks: Vec<u32> = self.ranks.iter().map(|held| held.rank).collect();
        for rank in ranks {
            self.launch_rank(rank, |prepared, stream| launch(rank, prepared, stream))?;
        }
        Ok(())
    }

    pub fn close<E>(
        &mut self,
        synchronize: impl FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
        release: impl FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
    ) -> Result<StreamTransactionEvidenceV1, StreamCloseFailureV1<E>> {
        if self.phase == StreamTransactionPhaseV1::Released {
            return Err(StreamCloseFailureV1 {
                evidence: self.evidence(),
                failures: vec![StreamTransactionErrorV1::Released],
            });
        }
        if self.phase == StreamTransactionPhaseV1::Quarantined {
            return Err(StreamCloseFailureV1 {
                evidence: self.evidence(),
                failures: vec![StreamTransactionErrorV1::NotReady],
            });
        }
        self.cleanup(synchronize, release)
    }

    pub fn retry_close<E>(
        &mut self,
        synchronize: impl FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
        release: impl FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
    ) -> Result<StreamTransactionEvidenceV1, StreamCloseFailureV1<E>> {
        if self.phase != StreamTransactionPhaseV1::Quarantined {
            return Err(StreamCloseFailureV1 {
                evidence: self.evidence(),
                failures: vec![StreamTransactionErrorV1::NotReady],
            });
        }
        self.cleanup(synchronize, release)
    }

    fn cleanup<E>(
        &mut self,
        mut synchronize: impl FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
        mut release: impl FnMut(u32, BundleStreamHandleV1) -> Result<(), E>,
    ) -> Result<StreamTransactionEvidenceV1, StreamCloseFailureV1<E>> {
        self.cleanup_failures.clear();
        let mut failures = Vec::new();
        self.phase = StreamTransactionPhaseV1::Synchronizing;
        for held in &mut self.ranks {
            if !held.potentially_enqueued || held.synchronized {
                continue;
            }
            let Some(stream) = held.stream else { continue };
            match synchronize(held.rank, stream) {
                Ok(()) => held.synchronized = true,
                Err(error) => {
                    self.cleanup_failures.push(held.rank);
                    failures.push(StreamTransactionErrorV1::Host {
                        service: "synchronize_stream",
                        rank: held.rank,
                        error,
                    });
                }
            }
        }

        self.phase = StreamTransactionPhaseV1::Releasing;
        for held in self.ranks.iter_mut().rev() {
            let Some(stream) = held.stream else { continue };
            if held.potentially_enqueued && !held.synchronized {
                continue;
            }
            match release(held.rank, stream) {
                Ok(()) => held.stream = None,
                Err(error) => {
                    if !self.cleanup_failures.contains(&held.rank) {
                        self.cleanup_failures.push(held.rank);
                    }
                    failures.push(StreamTransactionErrorV1::Host {
                        service: "release_stream",
                        rank: held.rank,
                        error,
                    });
                }
            }
        }
        self.update_terminal_phase();
        let evidence = self.evidence();
        if failures.is_empty() && self.phase == StreamTransactionPhaseV1::Released {
            Ok(evidence)
        } else {
            Err(StreamCloseFailureV1 { evidence, failures })
        }
    }

    fn update_terminal_phase(&mut self) {
        self.phase = if self.ranks.iter().all(|held| held.stream.is_none()) {
            StreamTransactionPhaseV1::Released
        } else {
            StreamTransactionPhaseV1::Quarantined
        };
    }

    #[must_use]
    pub fn evidence(&self) -> StreamTransactionEvidenceV1 {
        StreamTransactionEvidenceV1 {
            phase: self.phase,
            retained_streams: self
                .ranks
                .iter()
                .filter(|held| held.stream.is_some())
                .count(),
            potentially_enqueued_ranks: self
                .ranks
                .iter()
                .filter(|held| held.potentially_enqueued)
                .count(),
            accepted_ranks: self.ranks.iter().filter(|held| held.accepted).count(),
            synchronized_ranks: self.ranks.iter().filter(|held| held.synchronized).count(),
            cleanup_failures: self.cleanup_failures.clone(),
        }
    }
}
