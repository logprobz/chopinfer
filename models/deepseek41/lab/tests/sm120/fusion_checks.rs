//! CPU admission and complete failure collection for producer-fusion replays.
use kernel_core::contract::LaunchPlan;
use kernel_harness::LabResult;

pub(super) fn live_counts(rows: u32) -> [u32; 5] {
    [rows, 0, rows.saturating_sub(1), rows / 2, rows]
}

/// Every buffer holds `rows` rows and every entry uses one CTA per row.
/// A device live count may deactivate rows, never extend the allocation.
pub(super) fn validate_replay(
    rows: u32,
    live_rows: u32,
    capacity: u32,
    plans: &[LaunchPlan],
) -> LabResult<()> {
    if rows == 0 || rows > capacity || live_rows > rows {
        return Err(format!(
            "invalid fusion replay: rows={rows} live={live_rows} capacity={capacity}"
        ));
    }
    if plans.is_empty() {
        return Err("fusion replay requires launch plans".into());
    }
    for plan in plans {
        plan.validate_geometry().map_err(|e| e.to_string())?;
        if plan.grid != [rows, 1, 1] {
            return Err(format!(
                "fusion replay grid {:?} differs from {rows} allocated rows",
                plan.grid
            ));
        }
    }
    Ok(())
}

pub(super) fn compare_roles(
    names: &[&str],
    expected: &[Vec<u8>],
    actual: &[Vec<u8>],
) -> LabResult<()> {
    if expected.len() != names.len() || actual.len() != names.len() {
        return Err("comparison output role count differs".into());
    }
    let mut errors = Vec::new();
    for (name, (a, b)) in names.iter().zip(expected.iter().zip(actual)) {
        if a == b {
            continue;
        }
        if a.len() != b.len() {
            errors.push(format!(
                "{name} length: expected={} actual={}",
                a.len(),
                b.len()
            ));
        }
        if let Some(at) = a.iter().zip(b).position(|(a, b)| a != b) {
            let count = a.iter().zip(b).filter(|(a, b)| a != b).count();
            let word = at / 4 * 4;
            errors.push(format!(
                "{name}: {count} differing bytes, first={at}, expected={:02x?} actual={:02x?}",
                &a[word..(word + 4).min(a.len())],
                &b[word..(word + 4).min(b.len())]
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

/// Keep unused outputs as explicit poisoned guards, including an absent
/// widening output and the BF16 window's absent scale plane.
pub(super) fn guards(
    names: &[&str],
    outputs: &[Vec<u8>],
    rows: u32,
    live_rows: u32,
    untouched: &[usize],
    status_offset: usize,
    sticky: bool,
) -> LabResult<()> {
    if rows == 0 || live_rows > rows || outputs.len() != names.len() || outputs.is_empty() {
        return Err("invalid guard-check shape".into());
    }
    let mut errors = Vec::new();
    for (role, bytes) in outputs.iter().enumerate().take(outputs.len() - 1) {
        if bytes.len() % rows as usize != 0 {
            return Err(format!("{} does not hold complete rows", names[role]));
        }
        let start = if untouched.contains(&role) {
            0
        } else {
            bytes.len() / rows as usize * live_rows as usize
        };
        if let Some(at) = bytes[start..].iter().position(|b| *b != 0xa5) {
            errors.push(format!(
                "{} unwritten guard changed at byte {}",
                names[role],
                start + at
            ));
        }
    }
    let status = outputs.last().unwrap();
    if status.len() != (status_offset + rows as usize + 2) * 4 {
        return Err("status guard extent differs".into());
    }
    for (index, word) in status.chunks_exact(4).enumerate() {
        let expected = if index < status_offset || index >= status_offset + rows as usize {
            Some(0x1357_2468u32)
        } else if index >= status_offset + live_rows as usize {
            Some(if sticky { 0x80 } else { 0 })
        } else {
            None
        };
        let actual = u32::from_ne_bytes(word.try_into().unwrap());
        if let Some(expected) = expected
            && actual != expected
        {
            errors.push(format!(
                "status guard word {index}: expected={expected:#x} actual={actual:#x}"
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

pub(super) struct Failures {
    prefix: &'static str,
    messages: Vec<String>,
}
impl Failures {
    pub(super) fn new(prefix: &'static str) -> Self {
        Self {
            prefix,
            messages: Vec::new(),
        }
    }
    pub(super) fn record(&mut self, case: &str, result: LabResult<()>) {
        if let Err(detail) = result {
            let message = format!("{case}: {detail}");
            eprintln!("{}_MISMATCH {message}", self.prefix);
            self.messages.push(message);
        }
    }
    pub(super) fn require(&mut self, case: &str, passed: bool, detail: &str) {
        self.record(case, if passed { Ok(()) } else { Err(detail.into()) });
    }
    pub(super) fn len(&self) -> usize {
        self.messages.len()
    }
    pub(super) fn messages(&self) -> &[String] {
        &self.messages
    }
    pub(super) fn finish(&self) -> LabResult<()> {
        if self.messages.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} mismatching checks; see every {}_MISMATCH record",
                self.len(),
                self.prefix
            ))
        }
    }
}

#[test]
fn failure_collection_reports_all_cases_and_roles_before_failing() {
    let expected = vec![vec![0; 8]; 2];
    let mut bad = expected.clone();
    bad[0][0] = 1;
    bad[1][4] = 2;
    let mut failures = Failures::new("FIXTURE");
    for label in ["first", "middle", "last"] {
        failures.record(label, compare_roles(&["left", "right"], &expected, &bad));
    }
    assert_eq!(failures.len(), 3);
    for (label, message) in ["first", "middle", "last"].iter().zip(failures.messages()) {
        assert!(message.contains(label) && message.contains("left:") && message.contains("right:"));
    }
    assert!(failures.finish().is_err());
    let mut passing = Failures::new("FIXTURE");
    passing.record(
        "same",
        compare_roles(&["left", "right"], &expected, &expected),
    );
    assert!(passing.finish().is_ok());
    assert!(compare_roles(&["left"], &expected, &bad).is_err());
}

#[test]
fn unused_outputs_and_inactive_status_remain_required_guards() {
    let mut output = vec![vec![0xa5; 8], vec![0; 16]];
    output[1][..4].copy_from_slice(&0x1357_2468u32.to_ne_bytes());
    output[1][8..12].copy_from_slice(&0x1357_2468u32.to_ne_bytes());
    output[1][12..16].copy_from_slice(&0x1357_2468u32.to_ne_bytes());
    guards(&["unused", "status"], &output, 1, 0, &[0], 1, false).unwrap();
    output[0][0] = 0;
    assert!(guards(&["unused", "status"], &output, 1, 1, &[0], 1, false).is_err());
    output[0][0] = 0xa5;
    output[1][4] = 1;
    assert!(guards(&["unused", "status"], &output, 1, 0, &[0], 1, false).is_err());
}
