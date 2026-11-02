//! Boxes, sparklines, gauges and number formats.

use super::buffer::{Buffer, Color, Style};

/// Keeps printable ASCII and replaces everything else with `?`. Every
/// string that comes from a server, a process or a container passes
/// through this, so none can move the cursor or inject escape sequences.
pub fn sanitize(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character == ' ' || character.is_ascii_graphic() {
                character
            } else {
                '?'
            }
        })
        .collect()
}

/// `text` cut or padded on the right to `width` characters.
pub fn pad(text: &str, width: usize) -> String {
    let mut out = text.chars().take(width).collect::<String>();
    let length = out.chars().count();
    out.extend(std::iter::repeat_n(' ', width - length));
    out
}

/// `text` in lines of at most `width` characters, broken at spaces; a
/// word longer than a line is split. Text that fits is kept as it is.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 || text.chars().count() <= width {
        return vec![text.chars().take(width).collect()];
    }
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut length = 0;
    for word in text.split_whitespace() {
        let mut word = word.chars().collect::<Vec<_>>();
        if length > 0 && length + 1 + word.len() <= width {
            line.push(' ');
            line.extend(&word);
            length += 1 + word.len();
            continue;
        }
        if length > 0 {
            lines.push(std::mem::take(&mut line));
        }
        while word.len() > width {
            lines.push(word.drain(..width).collect());
        }
        length = word.len();
        line = word.into_iter().collect();
    }
    if length > 0 {
        lines.push(line);
    }
    lines
}

/// `text` cut on the right, padded on the left, to `width` characters.
pub fn right(text: &str, width: usize) -> String {
    let cut = text.chars().take(width).collect::<String>();
    format!("{cut:>width$}")
}

/// `1,842`.
pub fn integer(value: f64) -> String {
    let rounded = value.round().max(0.0) as u64;
    let digits = rounded.to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// A rate with precision fitting its size.
pub fn rate(value: f64) -> String {
    if value >= 100.0 {
        integer(value)
    } else if value >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

/// `38ms`, `1.20s`, `12.5s`; `-` for none.
pub fn latency(seconds: Option<f64>) -> String {
    match seconds {
        None => "-".to_owned(),
        Some(seconds) if seconds < 0.001 => format!("{:.0}us", seconds * 1e6),
        Some(seconds) if seconds < 1.0 => format!("{:.0}ms", seconds * 1000.0),
        Some(seconds) if seconds < 10.0 => format!("{seconds:.2}s"),
        Some(seconds) => format!("{seconds:.1}s"),
    }
}

/// [`integer`] when it fits in `width` characters, else `540k`, `12.3M` or
/// `4.1G`, so a narrow column never cuts digits off.
pub fn count(value: f64, width: usize) -> String {
    let full = integer(value);
    if full.chars().count() <= width {
        return full;
    }
    let (scaled, suffix) = if value >= 1e9 {
        (value / 1e9, "G")
    } else if value >= 1e6 {
        (value / 1e6, "M")
    } else {
        (value / 1e3, "k")
    };
    if scaled < 100.0 {
        format!("{scaled:.1}{suffix}")
    } else {
        format!("{scaled:.0}{suffix}")
    }
}

/// `63%`; `-` for none.
pub fn percent(ratio: Option<f64>) -> String {
    ratio.map_or_else(|| "-".to_owned(), |ratio| format!("{:.0}%", ratio * 100.0))
}

/// `3h12m`, `2m14s`, `41s`.
pub fn age(seconds: f64) -> String {
    let seconds = seconds.max(0.0) as u64;
    let (hours, minutes, rest) = (seconds / 3600, seconds % 3600 / 60, seconds % 60);
    if hours >= 48 {
        format!("{}d{:02}h", hours / 24, hours % 24)
    } else if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m{rest:02}s")
    } else {
        format!("{rest}s")
    }
}

/// `512 MiB`.
pub fn bytes(value: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = value.max(0.0);
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 || value >= 10.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// A bordered box with a title of styled pieces.
pub fn boxed(
    buffer: &mut Buffer,
    area: (u16, u16, u16, u16),
    title: &[(&str, Style)],
    focus: bool,
) {
    let (x, y, width, height) = area;
    if width < 2 || height < 2 {
        return;
    }
    let border = Style::fg(if focus { Color::Cyan } else { Color::Dim });
    let inner = usize::from(width - 2);
    buffer.put(x, y, &format!("┌{}┐", "─".repeat(inner)), border);
    for row in 1..height - 1 {
        buffer.put(x, y + row, "│", border);
        buffer.put(x + width - 1, y + row, "│", border);
    }
    buffer.put(
        x,
        y + height - 1,
        &format!("└{}┘", "─".repeat(inner)),
        border,
    );
    if title.is_empty() || width < 6 {
        return;
    }
    let limit = x + width - 2;
    let mut column = buffer.put(x + 2, y, " ", border);
    for (text, style) in title {
        let room = usize::from(limit.saturating_sub(column + 1));
        column = buffer.put(
            column,
            y,
            &text.chars().take(room).collect::<String>(),
            *style,
        );
    }
    buffer.put(column.min(limit), y, " ", border);
}

/// Averages a series into one value a column, for a plot `width` columns
/// wide that spans `span` polls. `data[i]` is poll `first + i` of its
/// history ([`crate::stats::Stats::oldest_poll`]). Poll `n` goes to bucket
/// `n * width / span`, so bucket edges fall on fixed poll numbers: a bucket
/// holds the same polls (3 or 4 for 300 polls in 80 columns) for as long as
/// it is shown, a finished one keeps its value as the series slides, and a
/// full span fills every column. A bucket whose oldest polls have left the
/// series is dropped rather than shown changing.
pub fn buckets(data: &[f64], first: u64, span: usize, width: usize) -> Vec<f64> {
    // One poll a bucket when the span fits the width.
    let (share, of) = if span > width {
        (width as u64, span as u64)
    } else {
        (1, 1)
    };
    let bucket = |poll: u64| poll * share / of;
    // (bucket, sum, polls) of each bucket in turn.
    let mut sums: Vec<(u64, f64, u64)> = Vec::new();
    for (poll, value) in (first..).zip(data) {
        match sums.last_mut() {
            Some((seen, sum, polls)) if *seen == bucket(poll) => {
                *sum += value;
                *polls += 1;
            }
            _ => sums.push((bucket(poll), *value, 1)),
        }
    }
    // The oldest bucket began before `first` if the poll before is in it.
    if sums.len() > 1 && first > 0 && bucket(first - 1) == bucket(first) {
        sums.remove(0);
    }
    sums[sums.len().saturating_sub(width)..]
        .iter()
        .map(|(_, sum, polls)| sum / *polls as f64)
        .collect()
}

/// A sparkline `height` rows high of one value per column, the newest at the
/// right edge, scaled to `top` or else to the largest value shown. Values
/// beyond `width` are the oldest and are not drawn; [`buckets`] fits a long
/// series.
pub fn sparkline(
    buffer: &mut Buffer,
    area: (u16, u16, u16, u16),
    data: &[f64],
    top: Option<f64>,
    color: Color,
) {
    const LEVELS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let (x, y, width, height) = area;
    if width == 0 || height == 0 {
        return;
    }
    let values = &data[data.len().saturating_sub(usize::from(width))..];
    let maximum = top.unwrap_or_else(|| values.iter().copied().fold(0.0_f64, f64::max));
    let offset = usize::from(width).saturating_sub(values.len());
    let style = Style::fg(color);
    for (index, value) in values.iter().enumerate() {
        let level = if maximum > 0.0 {
            value / maximum * f64::from(height) * 8.0
        } else {
            0.0
        };
        for row in 0..height {
            let mut eighths = (level - f64::from(row) * 8.0).round().clamp(0.0, 8.0) as usize;
            if row == 0 && eighths == 0 && *value > 0.0 {
                eighths = 1;
            }
            let column = x + (offset + index) as u16;
            buffer.put(
                column,
                y + height - 1 - row,
                &LEVELS[eighths].to_string(),
                style,
            );
        }
    }
}

/// The smallest of 1, 1.5, 2, 3, 5 and 7.5 times a power of ten at or above
/// `value`, so a plot scaled to it keeps its tallest bar between two thirds
/// and all of its height; 0 for none.
pub fn nice_ceiling(value: f64) -> f64 {
    if value <= 0.0 || !value.is_finite() {
        return 0.0;
    }
    let power = 10_f64.powf(value.log10().floor());
    [1.0, 1.5, 2.0, 3.0, 5.0, 7.5]
        .into_iter()
        .map(|step| step * power)
        .find(|ceiling| *ceiling >= value * (1.0 - 1e-9))
        .unwrap_or(10.0 * power)
}

/// A [`nice_ceiling`] exactly: `2,000`, `75`, `7.5`, `0.2`.
pub fn ceiling_label(value: f64) -> String {
    if value >= 100.0 {
        return integer(value);
    }
    if value <= 0.0 || !value.is_finite() {
        return "0".to_owned();
    }
    let decimals = (1 - value.log10().floor() as i32).clamp(0, 9) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text
    }
}

/// A line gauge `width` wide ending in `label`.
pub fn gauge(
    buffer: &mut Buffer,
    x: u16,
    y: u16,
    width: u16,
    ratio: f64,
    color: Color,
    label: &str,
) {
    let bar = usize::from(width).saturating_sub(label.chars().count() + 1);
    let filled = ((bar as f64) * ratio.clamp(0.0, 1.0)).round() as usize;
    let column = buffer.put(x, y, &"━".repeat(filled), Style::fg(color));
    let column = buffer.put(column, y, &"─".repeat(bar - filled), Style::fg(Color::Dim));
    buffer.put(column + 1, y, label, Style::PLAIN);
}

/// Draws whole groups of styled pieces from `x` on line `y` while each
/// ends by column `end`, so none is cut. Returns the column after the last.
pub fn groups(
    buffer: &mut Buffer,
    x: u16,
    y: u16,
    groups: &[Vec<(String, Style)>],
    end: u16,
) -> u16 {
    let mut column = x;
    for group in groups {
        let length: usize = group.iter().map(|(text, _)| text.chars().count()).sum();
        if usize::from(column) + length > usize::from(end) {
            break;
        }
        for (text, style) in group {
            column = buffer.put(column, y, text, *style);
        }
    }
    column
}

/// A horizontal bar `width` cells long at full scale, in eighths.
pub fn bar(ratio: f64, width: usize) -> String {
    const PARTS: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    let eighths = (ratio.clamp(0.0, 1.0) * width as f64 * 8.0).round() as usize;
    let mut out = "█".repeat(eighths / 8);
    if eighths % 8 > 0 {
        out.push(PARTS[eighths % 8]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_breaks_at_spaces_and_splits_long_words() {
        assert_eq!(wrap("fits as  it is", 14), ["fits as  it is"]);
        assert_eq!(
            wrap("SM clock 1,650 of 3,090 MHz, 2C under", 16),
            ["SM clock 1,650", "of 3,090 MHz, 2C", "under"]
        );
        assert_eq!(wrap("a 0123456789ab c", 5), ["a", "01234", "56789", "ab c"]);
        assert_eq!(wrap("", 8), [""]);
        assert_eq!(wrap("text", 0), [""]);
    }

    #[test]
    fn sanitizing_keeps_printable_ascii_only() {
        assert_eq!(sanitize("Qwen3.8-Flash"), "Qwen3.8-Flash");
        assert_eq!(sanitize("evil\u{1b}[2Jname\n\t"), "evil?[2Jname??");
        assert_eq!(sanitize("caf\u{e9} \u{202e}x"), "caf? ?x");
    }

    #[test]
    fn formats_numbers() {
        assert_eq!(integer(1842.4), "1,842");
        assert_eq!(integer(1_234_567.0), "1,234,567");
        assert_eq!(integer(12.0), "12");
        assert_eq!(count(540_123.0, 7), "540,123");
        assert_eq!(count(540_123.0, 6), "540k");
        assert_eq!(count(12_345_678.0, 8), "12.3M");
        assert_eq!(count(4_100_000_000.0, 8), "4.1G");
        assert_eq!(rate(3.0), "3.00");
        assert_eq!(rate(47.25), "47.2");
        assert_eq!(latency(Some(0.2104)), "210ms");
        assert_eq!(latency(Some(9.4)), "9.40s");
        assert_eq!(latency(None), "-");
        assert_eq!(percent(Some(0.634)), "63%");
        assert_eq!(age(11523.0), "3h12m");
        assert_eq!(age(134.0), "2m14s");
        assert_eq!(age(200_000.0), "2d07h");
        assert_eq!(bytes(3.0 * 1024.0 * 1024.0), "3.0 MiB");
        assert_eq!(bytes(512.0 * 1024.0 * 1024.0), "512 MiB");
        assert_eq!(pad("abcdef", 4), "abcd");
        assert_eq!(right("7", 3), "  7");
        assert_eq!(bar(0.5, 2), "█");
        assert_eq!(bar(0.55, 2), "█▏");
    }

    #[test]
    fn plot_scales_round_up_to_labelled_ceilings() {
        for (value, ceiling, label) in [
            (76.0, 100.0, "100"),
            (1805.0, 2000.0, "2,000"),
            (4755.0, 5000.0, "5,000"),
            (35_200.0, 50_000.0, "50,000"),
            (150.0, 150.0, "150"),
            (61.0, 75.0, "75"),
            (12.0, 15.0, "15"),
            (7.1, 7.5, "7.5"),
            (1.2, 1.5, "1.5"),
            (0.2, 0.2, "0.2"),
        ] {
            let top = nice_ceiling(value);
            assert!((top - ceiling).abs() < ceiling * 1e-12, "{value}: {top}");
            assert_eq!(ceiling_label(top), label, "{value}");
        }
        assert_eq!(nice_ceiling(0.0), 0.0);
        assert_eq!(nice_ceiling(1000.0), 1000.0);
    }

    #[test]
    fn draws_boxes_sparklines_and_gauges() {
        let mut buffer = Buffer::new(12, 4);
        boxed(
            &mut buffer,
            (0, 0, 12, 4),
            &[("Title", Style::PLAIN)],
            false,
        );
        assert_eq!(
            buffer.text(),
            "┌─ Title ──┐\n│          │\n│          │\n└──────────┘\n"
        );
        let mut buffer = Buffer::new(4, 2);
        sparkline(
            &mut buffer,
            (0, 0, 4, 2),
            &[0.0, 1.0, 2.0, 4.0],
            None,
            Color::Cyan,
        );
        assert_eq!(buffer.text(), "   █\n ▄██\n");
        // Scaled to a top above the data, the tallest bar leaves headroom.
        let mut buffer = Buffer::new(4, 2);
        sparkline(
            &mut buffer,
            (0, 0, 4, 2),
            &[0.0, 1.0, 2.0, 4.0],
            Some(8.0),
            Color::Cyan,
        );
        assert_eq!(buffer.text(), "\n ▂▄█\n");
        let mut buffer = Buffer::new(10, 1);
        gauge(&mut buffer, 0, 0, 10, 0.5, Color::Green, "50%");
        assert_eq!(buffer.text(), "━━━─── 50%\n");
        assert_eq!(buckets(&[1.0, 3.0, 5.0, 7.0], 0, 4, 2), [2.0, 6.0]);
    }

    #[test]
    fn full_history_fills_the_plot_and_slides_without_changing() {
        let value = |poll: u64| ((poll * 37) % 101) as f64;
        let columns = |first: u64, span: usize, width: usize| {
            let data = (first..first + span as u64).map(value).collect::<Vec<_>>();
            buckets(&data, first, span, width)
        };
        // Five minutes of polls at 1 s: 5, 3.75 and about 2.8 polls a column.
        for width in [60, 80, 107] {
            let (mut shifts, mut grows) = (0, 0);
            for first in 400..440 {
                let before = columns(first, 300, width);
                let after = columns(first + 1, 300, width);
                assert_eq!(before.len(), width, "{width} columns at {first}");
                // A new poll joins the newest column, or starts a new one
                // and every finished column steps left unchanged.
                let last = width - 1;
                if before[..last] == after[..last] {
                    grows += 1;
                } else {
                    assert_eq!(before[1..last], after[..last - 1], "{width} at {first}");
                    shifts += 1;
                }
            }
            assert!(shifts > 0 && grows > 0, "{width}: {shifts} {grows}");
        }
    }

    #[test]
    fn buckets_drop_expiring_polls_and_keep_the_newest() {
        // Poll 1 is the only poll of bucket 0 left: dropped.
        assert_eq!(buckets(&[1.0, 2.0, 3.0, 4.0, 5.0], 1, 4, 2), [2.5, 4.5]);
        // A lone partial bucket is also the newest: kept.
        assert_eq!(buckets(&[7.0], 3, 8, 2), [7.0]);
        // A span that fits gives one value per poll, the newest `width`.
        assert_eq!(buckets(&[1.0, 2.0, 3.0], 5, 10, 20), [1.0, 2.0, 3.0]);
        assert_eq!(
            buckets(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0, 4, 4),
            [3.0, 4.0, 5.0, 6.0]
        );
        assert!(buckets(&[], 0, 300, 80).is_empty());
    }
}
