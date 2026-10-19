use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Routing {
    Mixed,
    All,
    Seeded,
}

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub rows: u32,
    pub phase: ExecutionPhase,
    pub set: moe::Experts,
    pub routing: Routing,
}
impl Case {
    pub fn label(self) -> String {
        format!(
            "e{}k{}-{:?}-{}-{:?}",
            self.set.count, self.set.top_k, self.phase, self.rows, self.routing
        )
    }
    pub fn capacity(self) -> u32 {
        match self.phase {
            ExecutionPhase::Decode => 32,
            ExecutionPhase::Verify => 192,
            ExecutionPhase::Prefill => 8192,
        }
    }
    pub fn logits(self) -> Vec<u8> {
        let mut seed = 0x8315_a7d2_046f_39c1_u64;
        let mut bytes = Vec::new();
        for row in 0..self.rows {
            let first = match self.routing {
                Routing::All => row * self.set.top_k % self.set.count,
                Routing::Mixed => {
                    if row % 3 == 0 {
                        0
                    } else {
                        row * 37 % self.set.count
                    }
                }
                Routing::Seeded => 0,
            };
            for e in 0..self.set.count {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let value = if self.routing == Routing::Seeded {
                    ((seed >> 32) as u32) as f32 / u32::MAX as f32 * 8.0 - 4.0
                } else {
                    -(((e + self.set.count - first) % self.set.count) as f32) / 32.0
                };
                bytes.extend(value.to_le_bytes());
            }
        }
        bytes
    }
}

pub fn cases(set: moe::Experts) -> Vec<Case> {
    let mut out = Vec::new();
    for (phase, rows) in [
        (ExecutionPhase::Decode, &[1, 2, 4, 8, 16, 32][..]),
        // Graph powers of two plus the eager 5/6-row and maximum boundaries.
        (
            ExecutionPhase::Verify,
            &[1, 2, 4, 5, 6, 8, 16, 32, 64, 128, 192][..],
        ),
        (ExecutionPhase::Prefill, &[128][..]),
    ] {
        out.extend(rows.iter().map(|&rows| Case {
            rows,
            phase,
            set,
            routing: Routing::Mixed,
        }));
    }
    for routing in [Routing::All, Routing::Seeded] {
        out.push(Case {
            rows: 192,
            phase: ExecutionPhase::Verify,
            set,
            routing,
        });
    }
    out
}

pub fn speed_gate(old: f32, selected: f32) -> LabResult<()> {
    if !old.is_finite() || !selected.is_finite() || old <= 0.0 || selected <= 0.0 {
        return Err("non-finite or non-positive timing".into());
    }
    if selected > old * 1.03 {
        return Err(format!(
            "selected/old = {:.4}, exceeds the 1.03 limit",
            selected / old
        ));
    }
    Ok(())
}

#[test]
fn speed_gate_rejects_the_reported_regressions() {
    for pair in [
        [0.504368, 0.847328],
        [1.2117441, 2.614176],
        [4.0082397, 6.685808],
    ] {
        assert!(speed_gate(pair[0], pair[1]).is_err());
    }
    assert!(speed_gate(0.3488, 0.285616).is_ok());
    assert!(speed_gate(1.0, 1.03).is_ok());
    assert!(speed_gate(1.0, 1.031).is_err());
    for pair in [[0.0, 0.0], [1.0, f32::NAN], [f32::INFINITY, 1.0]] {
        assert!(speed_gate(pair[0], pair[1]).is_err());
    }
}

#[test]
fn all_active_routes_cover_each_bank_and_seeded_routes_reproduce() {
    for set in moe::expert_sets(&FLASH) {
        let case = Case {
            rows: 192,
            phase: ExecutionPhase::Verify,
            set,
            routing: Routing::All,
        };
        let used = (0..case.rows)
            .flat_map(|r| (0..set.top_k).map(move |k| (r * set.top_k + k) % set.count))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(used.len(), set.count as usize);
        let seeded = Case {
            routing: Routing::Seeded,
            ..case
        };
        assert_eq!(seeded.logits(), seeded.logits());
        assert_ne!(seeded.logits(), case.logits());
    }
}
