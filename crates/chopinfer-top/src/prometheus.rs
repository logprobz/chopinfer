//! The Prometheus text exposition format, as `/metrics` publishes it.

use std::collections::HashMap;

/// One sample: its labels in the order written, and its value.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub labels: Vec<(String, String)>,
    pub value: f64,
}

impl Sample {
    pub fn label(&self, name: &str) -> Option<&str> {
        self.labels
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Every sample of one scrape, grouped by series name.
#[derive(Clone, Debug, Default)]
pub struct Exposition {
    series: HashMap<String, Vec<Sample>>,
}

/// A cumulative histogram: `(le, count)` in increasing `le`, ending with
/// `+Inf`, and the sum and count of observations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Histogram {
    pub buckets: Vec<(f64, f64)>,
    pub sum: f64,
    pub count: f64,
}

impl Exposition {
    /// Parses a scrape. Lines that do not parse are skipped, so one series
    /// this reader does not understand never hides the others.
    pub fn parse(text: &str) -> Self {
        let mut series: HashMap<String, Vec<Sample>> = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((name, sample)) = parse_line(line) {
                series.entry(name).or_default().push(sample);
            }
        }
        Self { series }
    }

    pub fn samples(&self, name: &str) -> &[Sample] {
        self.series.get(name).map_or(&[], Vec::as_slice)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.series.contains_key(name)
    }

    /// The sum of every sample of `name` whose labels include `filter`, or
    /// `None` when there is none. Series of one server differ only in
    /// constant labels such as `model_name`, so the sum is the value.
    pub fn value(&self, name: &str, filter: &[(&str, &str)]) -> Option<f64> {
        let mut found = None;
        for sample in self.samples(name) {
            if matches(sample, filter) {
                *found.get_or_insert(0.0) += sample.value;
            }
        }
        found
    }

    /// Values of `name` keyed by `label`, summed over the other labels, in
    /// first-seen order.
    pub fn by_label(&self, name: &str, label: &str) -> Vec<(String, f64)> {
        let mut values: Vec<(String, f64)> = Vec::new();
        for sample in self.samples(name) {
            let Some(key) = sample.label(label) else {
                continue;
            };
            match values.iter_mut().find(|(seen, _)| seen == key) {
                Some((_, value)) => *value += sample.value,
                None => values.push((key.to_owned(), sample.value)),
            }
        }
        values
    }

    /// The histogram `name`, summing every label set that includes
    /// `filter`; `None` when it has no buckets.
    pub fn histogram(&self, name: &str, filter: &[(&str, &str)]) -> Option<Histogram> {
        let mut buckets: Vec<(f64, f64)> = Vec::new();
        for sample in self.samples(&format!("{name}_bucket")) {
            if !matches(sample, filter) {
                continue;
            }
            let Some(le) = sample.label("le").and_then(parse_value) else {
                continue;
            };
            match buckets.iter_mut().find(|(bound, _)| *bound == le) {
                Some((_, count)) => *count += sample.value,
                None => buckets.push((le, sample.value)),
            }
        }
        if buckets.is_empty() {
            return None;
        }
        buckets.sort_by(|left, right| left.0.total_cmp(&right.0));
        Some(Histogram {
            buckets,
            sum: self.value(&format!("{name}_sum"), filter).unwrap_or(0.0),
            count: self.value(&format!("{name}_count"), filter).unwrap_or(0.0),
        })
    }
}

fn matches(sample: &Sample, filter: &[(&str, &str)]) -> bool {
    filter
        .iter()
        .all(|(name, value)| sample.label(name) == Some(*value))
}

impl Histogram {
    /// Observations in this histogram that are not in `earlier`, bucket by
    /// bucket; `None` when the bounds differ.
    pub fn since(&self, earlier: &Self) -> Option<Self> {
        if self.buckets.len() != earlier.buckets.len()
            || self
                .buckets
                .iter()
                .zip(&earlier.buckets)
                .any(|(now, then)| now.0 != then.0)
        {
            return None;
        }
        Some(Self {
            buckets: self
                .buckets
                .iter()
                .zip(&earlier.buckets)
                .map(|(now, then)| (now.0, (now.1 - then.1).max(0.0)))
                .collect(),
            sum: (self.sum - earlier.sum).max(0.0),
            count: (self.count - earlier.count).max(0.0),
        })
    }

    /// The `q` quantile with Prometheus `histogram_quantile` interpolation:
    /// linear within the bucket the rank falls in, the highest finite bound
    /// when it falls in `+Inf`, and `None` without observations or without
    /// the `+Inf` bucket, as Prometheus refuses such a histogram.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        let (last, total) = *self.buckets.last()?;
        if last != f64::INFINITY {
            return None;
        }
        if total.is_nan() || total <= 0.0 || !(0.0..=1.0).contains(&q) {
            return None;
        }
        let rank = q * total;
        let index = self
            .buckets
            .iter()
            .position(|(_, count)| *count >= rank)
            .unwrap_or(self.buckets.len() - 1);
        let (upper, count) = self.buckets[index];
        if upper.is_infinite() {
            return self
                .buckets
                .iter()
                .rev()
                .find(|(bound, _)| bound.is_finite())
                .map(|(bound, _)| *bound);
        }
        if index == 0 && upper <= 0.0 {
            return Some(upper);
        }
        let (lower, below) = if index == 0 {
            (0.0, 0.0)
        } else {
            self.buckets[index - 1]
        };
        let within = count - below;
        if within <= 0.0 {
            return Some(upper);
        }
        Some(lower + (upper - lower) * ((rank - below) / within))
    }
}

/// Parses `name{labels} value [timestamp]`.
fn parse_line(line: &str) -> Option<(String, Sample)> {
    let name_end = line
        .find(|character: char| character == '{' || character.is_whitespace())
        .unwrap_or(line.len());
    let name = &line[..name_end];
    if name.is_empty()
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | ':'))
    {
        return None;
    }
    let mut rest = &line[name_end..];
    let mut labels = Vec::new();
    if let Some(inner) = rest.strip_prefix('{') {
        let (parsed, after) = parse_labels(inner)?;
        labels = parsed;
        rest = after;
    }
    let value = parse_value(rest.split_whitespace().next()?)?;
    Some((name.to_owned(), Sample { labels, value }))
}

/// Parses `a="x",b="y"}` and returns the labels and the text after `}`.
fn parse_labels(mut text: &str) -> Option<(Vec<(String, String)>, &str)> {
    let mut labels = Vec::new();
    loop {
        text = text.trim_start();
        if let Some(after) = text.strip_prefix('}') {
            return Some((labels, after));
        }
        let (name, after) = text.split_once('=')?;
        let mut value = String::new();
        let mut characters = after.trim_start().strip_prefix('"')?.char_indices();
        let end = loop {
            let (index, character) = characters.next()?;
            match character {
                '"' => break index,
                '\\' => match characters.next()?.1 {
                    'n' => value.push('\n'),
                    other => value.push(other),
                },
                other => value.push(other),
            }
        };
        labels.push((name.trim().to_owned(), value));
        text = after.trim_start()[1 + end + 1..].trim_start();
        text = text.strip_prefix(',').unwrap_or(text);
    }
}

fn parse_value(text: &str) -> Option<f64> {
    match text {
        "+Inf" | "Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        "NaN" => Some(f64::NAN),
        _ => text.parse().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCRAPE: &str = r#"# TYPE chop_process_uptime_seconds gauge
chop_process_uptime_seconds 12.500000
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="m",engine="0"} 1500
vllm:request_success_total{model_name="m",engine="0",finished_reason="stop"} 7
vllm:request_success_total{model_name="m",engine="0",finished_reason="length"} 2 1700000000000
odd_label{path="C:\\dir",quote="say \"hi\"",line="a\nb"} -Inf
chop_step_execution_seconds_bucket{model_name="m",engine="0",phase="decode",le="0.01"} 4
chop_step_execution_seconds_bucket{model_name="m",engine="0",phase="decode",le="0.1"} 9
chop_step_execution_seconds_bucket{model_name="m",engine="0",phase="decode",le="+Inf"} 10
chop_step_execution_seconds_sum{model_name="m",engine="0",phase="decode"} 0.4
chop_step_execution_seconds_count{model_name="m",engine="0",phase="decode"} 10
chop_step_execution_seconds_bucket{model_name="m",engine="0",phase="mixed",le="0.01"} 0
chop_step_execution_seconds_bucket{model_name="m",engine="0",phase="mixed",le="0.1"} 1
chop_step_execution_seconds_bucket{model_name="m",engine="0",phase="mixed",le="+Inf"} 1
not a sample line
"#;

    #[test]
    fn parses_names_labels_escapes_values_and_timestamps() {
        let scrape = Exposition::parse(SCRAPE);
        assert_eq!(scrape.value("chop_process_uptime_seconds", &[]), Some(12.5));
        assert_eq!(
            scrape.value("vllm:generation_tokens_total", &[]),
            Some(1500.0)
        );
        assert_eq!(scrape.value("vllm:request_success_total", &[]), Some(9.0));
        assert_eq!(
            scrape.by_label("vllm:request_success_total", "finished_reason"),
            [("stop".to_owned(), 7.0), ("length".to_owned(), 2.0)]
        );
        let odd = &scrape.samples("odd_label")[0];
        assert_eq!(odd.label("path"), Some("C:\\dir"));
        assert_eq!(odd.label("quote"), Some("say \"hi\""));
        assert_eq!(odd.label("line"), Some("a\nb"));
        assert_eq!(odd.value, f64::NEG_INFINITY);
        assert!(!scrape.contains("not"));
        assert_eq!(scrape.value("missing", &[]), None);
    }

    #[test]
    fn histograms_filter_and_sum_label_sets() {
        let scrape = Exposition::parse(SCRAPE);
        let decode = scrape
            .histogram("chop_step_execution_seconds", &[("phase", "decode")])
            .unwrap();
        assert_eq!(
            decode.buckets,
            [(0.01, 4.0), (0.1, 9.0), (f64::INFINITY, 10.0)]
        );
        assert_eq!((decode.sum, decode.count), (0.4, 10.0));
        let all = scrape
            .histogram("chop_step_execution_seconds", &[])
            .unwrap();
        assert_eq!(
            all.buckets,
            [(0.01, 4.0), (0.1, 10.0), (f64::INFINITY, 11.0)]
        );
        assert!(
            scrape
                .histogram("vllm:time_to_first_token_seconds", &[])
                .is_none()
        );
    }

    #[test]
    fn quantiles_interpolate_like_histogram_quantile() {
        let histogram = Histogram {
            buckets: vec![(0.1, 10.0), (0.5, 30.0), (1.0, 40.0), (f64::INFINITY, 40.0)],
            sum: 0.0,
            count: 40.0,
        };
        // Rank 20 lies halfway through (0.1, 0.5].
        assert!((histogram.quantile(0.5).unwrap() - 0.3).abs() < 1e-12);
        // Rank 4 lies in the first bucket, interpolated from zero.
        assert!((histogram.quantile(0.1).unwrap() - 0.04).abs() < 1e-12);
        assert_eq!(histogram.quantile(1.0), Some(1.0));
        let overflow = Histogram {
            buckets: vec![(0.1, 1.0), (1.0, 2.0), (f64::INFINITY, 10.0)],
            sum: 0.0,
            count: 10.0,
        };
        assert_eq!(overflow.quantile(0.99), Some(1.0));
        let empty = Histogram {
            buckets: vec![(0.1, 0.0), (f64::INFINITY, 0.0)],
            sum: 0.0,
            count: 0.0,
        };
        assert_eq!(empty.quantile(0.5), None);
        let incomplete = Histogram {
            buckets: vec![(1.0, 1.0), (2.0, 2.0)],
            sum: 0.0,
            count: 2.0,
        };
        assert_eq!(incomplete.quantile(0.5), None);
    }

    #[test]
    fn deltas_need_matching_bounds() {
        let now = Histogram {
            buckets: vec![(0.1, 10.0), (f64::INFINITY, 12.0)],
            sum: 3.0,
            count: 12.0,
        };
        let then = Histogram {
            buckets: vec![(0.1, 4.0), (f64::INFINITY, 4.0)],
            sum: 1.0,
            count: 4.0,
        };
        let delta = now.since(&then).unwrap();
        assert_eq!(delta.buckets, [(0.1, 6.0), (f64::INFINITY, 8.0)]);
        assert_eq!((delta.sum, delta.count), (2.0, 8.0));
        let other = Histogram {
            buckets: vec![(0.2, 1.0), (f64::INFINITY, 1.0)],
            sum: 0.0,
            count: 1.0,
        };
        assert!(now.since(&other).is_none());
    }
}
