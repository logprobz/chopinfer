//! A grid of styled cells, and writing only what changed between frames.

use std::io::Write;

use crossterm::cursor::MoveTo;
use crossterm::style::{
    Attribute, Color as TerminalColor, Print, SetAttribute, SetBackgroundColor, SetForegroundColor,
};
use crossterm::terminal::{Clear, ClearType};
use crossterm::{QueueableCommand, queue};

/// The 16-color palette roles, so the terminal's own theme applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    Default,
    Dim,
    White,
    Cyan,
    Green,
    Yellow,
    Red,
    Magenta,
    Blue,
    /// The header and footer bars.
    Bar,
    /// The selected row.
    Selection,
}

impl Color {
    fn terminal(self) -> TerminalColor {
        match self {
            Self::Default => TerminalColor::Reset,
            Self::Dim => TerminalColor::DarkGrey,
            Self::White => TerminalColor::White,
            Self::Cyan => TerminalColor::Cyan,
            Self::Green => TerminalColor::Green,
            Self::Yellow => TerminalColor::Yellow,
            Self::Red => TerminalColor::Red,
            Self::Magenta => TerminalColor::Magenta,
            Self::Blue => TerminalColor::Blue,
            Self::Bar => TerminalColor::DarkGrey,
            Self::Selection => TerminalColor::DarkBlue,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    pub fg: Color,
    pub bg: Option<Color>,
    pub bold: bool,
}

impl Style {
    pub const PLAIN: Self = Self {
        fg: Color::Default,
        bg: None,
        bold: false,
    };

    pub const fn fg(fg: Color) -> Self {
        Self {
            fg,
            bg: None,
            bold: false,
        }
    }

    pub const fn bold(self) -> Self {
        Self { bold: true, ..self }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub character: char,
    pub style: Style,
}

const BLANK: Cell = Cell {
    character: ' ',
    style: Style::PLAIN,
};

/// One frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    pub width: u16,
    pub height: u16,
    cells: Vec<Cell>,
}

impl Buffer {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            cells: vec![BLANK; usize::from(width) * usize::from(height)],
        }
    }

    fn index(&self, x: u16, y: u16) -> Option<usize> {
        (x < self.width && y < self.height)
            .then(|| usize::from(y) * usize::from(self.width) + usize::from(x))
    }

    pub fn cell(&self, x: u16, y: u16) -> Option<&Cell> {
        self.index(x, y).map(|index| &self.cells[index])
    }

    /// Writes `text` from `(x, y)`, clipped at the right edge, keeping the
    /// background already there unless `style` has one. Returns the column
    /// after the text.
    pub fn put(&mut self, x: u16, y: u16, text: &str, style: Style) -> u16 {
        let mut column = x;
        for character in text.chars() {
            if column >= self.width {
                break;
            }
            if let Some(index) = self.index(column, y) {
                let cell = &mut self.cells[index];
                cell.character = character;
                cell.style = Style {
                    bg: style.bg.or(cell.style.bg),
                    ..style
                };
            }
            column += 1;
        }
        // Text cut at the edge ends there, so callers can add to the result.
        column.min(self.width)
    }

    /// Writes styled pieces one after another.
    pub fn pieces(&mut self, x: u16, y: u16, pieces: &[(&str, Style)]) -> u16 {
        pieces
            .iter()
            .fold(x, |column, (text, style)| self.put(column, y, text, *style))
    }

    /// Sets the background of `width` cells from `(x, y)`.
    pub fn background(&mut self, x: u16, y: u16, width: u16, color: Color) {
        for column in x..x.saturating_add(width) {
            if let Some(index) = self.index(column, y) {
                self.cells[index].style.bg = Some(color);
            }
        }
    }

    /// Blanks a rectangle, filling it with `bg`.
    pub fn clear(&mut self, x: u16, y: u16, width: u16, height: u16, bg: Option<Color>) {
        for row in y..y.saturating_add(height) {
            for column in x..x.saturating_add(width) {
                if let Some(index) = self.index(column, row) {
                    self.cells[index] = Cell {
                        character: ' ',
                        style: Style { bg, ..Style::PLAIN },
                    };
                }
            }
        }
    }

    /// The characters of each row, trailing spaces removed, for tests.
    pub fn text(&self) -> String {
        let mut text = String::new();
        for row in self.cells.chunks(usize::from(self.width.max(1))) {
            let line = row.iter().map(|cell| cell.character).collect::<String>();
            text.push_str(line.trim_end());
            text.push('\n');
        }
        text
    }
}

/// Writes `next` to `out`: every cell when there is no `previous` frame of
/// the same size, else only the cells that changed. Without `color`, colors
/// are dropped and the selection and bars are drawn in reverse video.
pub fn flush(
    previous: Option<&Buffer>,
    next: &Buffer,
    out: &mut impl Write,
    color: bool,
) -> std::io::Result<()> {
    let full = previous
        .is_none_or(|previous| previous.width != next.width || previous.height != next.height);
    if full {
        out.queue(SetAttribute(Attribute::Reset))?
            .queue(Clear(ClearType::All))?;
    }
    let mut current: Option<Style> = None;
    let mut cursor: Option<(u16, u16)> = None;
    for y in 0..next.height {
        for x in 0..next.width {
            let cell = next.cell(x, y).expect("in bounds");
            if !full && previous.and_then(|previous| previous.cell(x, y)) == Some(cell) {
                continue;
            }
            if cursor != Some((x, y)) {
                out.queue(MoveTo(x, y))?;
            }
            if current != Some(cell.style) {
                apply_style(out, cell.style, color)?;
                current = Some(cell.style);
            }
            out.queue(Print(cell.character))?;
            cursor = Some((x + 1, y));
        }
    }
    queue!(out, SetAttribute(Attribute::Reset))?;
    out.flush()
}

fn apply_style(out: &mut impl Write, style: Style, color: bool) -> std::io::Result<()> {
    out.queue(SetAttribute(Attribute::Reset))?;
    if style.bold {
        out.queue(SetAttribute(Attribute::Bold))?;
    }
    if color {
        out.queue(SetForegroundColor(style.fg.terminal()))?;
        if let Some(bg) = style.bg {
            out.queue(SetBackgroundColor(bg.terminal()))?;
        }
    } else if style.bg.is_some() {
        out.queue(SetAttribute(Attribute::Reverse))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_clip_and_keep_backgrounds() {
        let mut buffer = Buffer::new(6, 2);
        buffer.background(0, 0, 6, Color::Selection);
        // The returned column stops at the edge, so adding to it is safe
        // even after a very long text.
        let end = buffer.put(3, 0, "abcdef", Style::fg(Color::Cyan));
        assert_eq!(end, 6);
        assert_eq!(buffer.put(1, 1, &"x".repeat(70_000), Style::PLAIN), 6);
        assert_eq!(buffer.text(), "   abc\n xxxxx\n");
        assert_eq!(buffer.cell(3, 0).unwrap().style.bg, Some(Color::Selection));
        assert_eq!(buffer.cell(3, 0).unwrap().style.fg, Color::Cyan);
        buffer.put(0, 5, "outside", Style::PLAIN);
    }

    #[test]
    fn flushes_only_changed_cells() {
        let mut first = Buffer::new(4, 1);
        first.put(0, 0, "abcd", Style::PLAIN);
        let mut all = Vec::new();
        flush(None, &first, &mut all, true).unwrap();
        let mut second = first.clone();
        second.put(2, 0, "X", Style::PLAIN);
        let mut changed = Vec::new();
        flush(Some(&first), &second, &mut changed, true).unwrap();
        let changed = String::from_utf8(changed).unwrap();
        assert!(changed.contains('X'));
        assert!(
            !changed.contains('a') && !changed.contains('d'),
            "{changed:?}"
        );
        assert!(changed.len() < all.len());
        let mut plain = Vec::new();
        let mut selected = first.clone();
        selected.background(0, 0, 4, Color::Selection);
        flush(None, &selected, &mut plain, false).unwrap();
        let plain = String::from_utf8(plain).unwrap();
        // Reverse video, and no color sequences.
        assert!(plain.contains("\u{1b}[7m"), "{plain:?}");
        assert!(!plain.contains("\u{1b}[38;"), "{plain:?}");
    }
}
