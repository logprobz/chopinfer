//! The interface state: which view, which instance, which window, and
//! what each key does.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

use crate::instances::{Instance, Registry};
use crate::stats::Window;
use crate::ui::list::{TABLE_TOP, first_row, table_rows};

/// Seconds between two clicks on one row that open it.
const DOUBLE_CLICK_SECONDS: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    List,
    Detail,
}

/// What the main loop should do after an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Nothing,
    Redraw,
    Rescan,
    Quit,
}

#[derive(Debug)]
pub struct App {
    pub registry: Registry,
    pub view: View,
    /// The selected row's id, so the selection survives rows coming and
    /// going.
    pub selected: Option<u64>,
    pub window: Window,
    pub paused: bool,
    pub help: bool,
    pub interval: f64,
    pub rescan: f64,
    /// The sources in use, for the header: `proc`, `docker`, `--target`.
    pub sources: Vec<&'static str>,
    /// The first list row shown, kept so the list scrolls only when the
    /// selection leaves it.
    pub scroll: usize,
    /// The terminal height at the last [`App::settle`], which clicks are
    /// mapped with.
    height: u16,
    last_click: Option<(u64, f64)>,
}

impl App {
    pub fn new(registry: Registry, interval: f64, rescan: f64, sources: Vec<&'static str>) -> Self {
        Self {
            registry,
            view: View::List,
            selected: None,
            window: Window::Minute,
            paused: false,
            help: false,
            interval,
            rescan,
            sources,
            scroll: 0,
            height: 0,
            last_click: None,
        }
    }

    /// Scrolls the list so the selected row is shown on a terminal `height`
    /// rows tall. Called before each frame.
    pub fn settle(&mut self, height: u16) {
        self.height = height;
        let room = table_rows(self.rows().len(), height);
        self.scroll = first_row(self.scroll, self.selected_index(), room);
    }

    /// The rows shown, in order.
    pub fn rows(&self) -> Vec<&Instance> {
        self.registry.visible()
    }

    /// The selected row's position, the first row when it is gone.
    pub fn selected_index(&self) -> usize {
        let rows = self.rows();
        self.selected
            .and_then(|id| rows.iter().position(|row| row.id == id))
            .unwrap_or(0)
    }

    pub fn current(&self) -> Option<&Instance> {
        self.rows().get(self.selected_index()).copied()
    }

    fn select(&mut self, index: usize) {
        self.selected = self.rows().get(index).map(|row| row.id);
    }

    fn step(&mut self, forward: bool, wrap: bool) {
        let count = self.rows().len();
        if count == 0 {
            return;
        }
        let index = self.selected_index();
        let next = match (forward, wrap) {
            (true, true) => (index + 1) % count,
            (false, true) => (index + count - 1) % count,
            (true, false) => (index + 1).min(count - 1),
            (false, false) => index.saturating_sub(1),
        };
        self.select(next);
    }

    pub fn key(&mut self, event: KeyEvent) -> Action {
        let control = event.modifiers.contains(KeyModifiers::CONTROL);
        if control && matches!(event.code, KeyCode::Char('c')) {
            return Action::Quit;
        }
        if self.help {
            self.help = false;
            return Action::Redraw;
        }
        match (self.view, event.code) {
            (_, KeyCode::Char('q')) => return Action::Quit,
            (_, KeyCode::Char('?')) => self.help = true,
            (_, KeyCode::Char('p')) => self.paused = !self.paused,
            (_, KeyCode::Char('w')) => self.window = self.window.next(),
            (_, KeyCode::Char('r')) => return Action::Rescan,
            (View::List, KeyCode::Down | KeyCode::Char('j')) => self.step(true, false),
            (View::List, KeyCode::Up | KeyCode::Char('k')) => self.step(false, false),
            (View::List, KeyCode::Enter) => {
                if self.current().is_some() {
                    self.select(self.selected_index());
                    self.view = View::Detail;
                }
            }
            (View::Detail, KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('h')) => {
                self.view = View::List;
            }
            (View::Detail, KeyCode::Right | KeyCode::Tab | KeyCode::Char('l')) => {
                self.step(true, true);
            }
            (View::Detail, KeyCode::Left | KeyCode::BackTab) => self.step(false, true),
            _ => return Action::Nothing,
        }
        Action::Redraw
    }

    /// A click selects a row of the list; a second click on it within half
    /// a second opens it.
    pub fn mouse(&mut self, event: MouseEvent, now: f64) -> Action {
        if self.view != View::List
            || self.help
            || event.kind != MouseEventKind::Down(MouseButton::Left)
        {
            return Action::Nothing;
        }
        let Some(line) = event.row.checked_sub(TABLE_TOP + 2).map(usize::from) else {
            return Action::Nothing;
        };
        let rows = self.rows();
        let room = table_rows(rows.len(), self.height);
        if line >= room {
            return Action::Nothing;
        }
        let first = first_row(self.scroll, self.selected_index(), room);
        let Some(id) = rows.get(first + line).map(|row| row.id) else {
            return Action::Nothing;
        };
        let double = self
            .last_click
            .is_some_and(|(clicked, at)| clicked == id && now - at <= DOUBLE_CLICK_SECONDS);
        self.selected = Some(id);
        if double {
            self.view = View::Detail;
            self.last_click = None;
        } else {
            self.last_click = Some((id, now));
        }
        Action::Redraw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::targets::Target;

    fn app(rows: usize) -> App {
        let mut registry = Registry::new(1.0);
        for port in 0..rows {
            registry.add_target(Target::parse(&format!(":{}", 8310 + port)).unwrap(), 0.0);
        }
        App::new(registry, 1.0, 15.0, vec!["--target"])
    }

    fn press(app: &mut App, code: KeyCode) -> Action {
        app.key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn keys_select_open_switch_and_quit() {
        let mut app = app(3);
        assert_eq!(app.selected_index(), 0);
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Char('j'));
        press(&mut app, KeyCode::Down);
        assert_eq!(app.selected_index(), 2);
        press(&mut app, KeyCode::Char('k'));
        assert_eq!(app.selected_index(), 1);
        assert_eq!(press(&mut app, KeyCode::Enter), Action::Redraw);
        assert_eq!(app.view, View::Detail);
        press(&mut app, KeyCode::Tab);
        press(&mut app, KeyCode::Right);
        assert_eq!(app.selected_index(), 0);
        press(&mut app, KeyCode::BackTab);
        assert_eq!(app.selected_index(), 2);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::List);
        press(&mut app, KeyCode::Char('w'));
        assert_eq!(app.window, Window::FiveMinutes);
        press(&mut app, KeyCode::Char('p'));
        assert!(app.paused);
        assert_eq!(press(&mut app, KeyCode::Char('r')), Action::Rescan);
        press(&mut app, KeyCode::Char('?'));
        assert!(app.help);
        // Any key closes help, even q.
        assert_eq!(press(&mut app, KeyCode::Char('q')), Action::Redraw);
        assert!(!app.help);
        assert_eq!(press(&mut app, KeyCode::Char('q')), Action::Quit);
        assert_eq!(
            app.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Quit
        );
        assert_eq!(press(&mut app, KeyCode::Char('x')), Action::Nothing);
    }

    #[test]
    fn clicks_select_and_double_clicks_open() {
        let mut app = app(3);
        app.settle(24);
        let click = |row| MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 10,
            row,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(app.mouse(click(TABLE_TOP + 3), 1.0), Action::Redraw);
        assert_eq!((app.selected_index(), app.view), (1, View::List));
        assert_eq!(app.mouse(click(TABLE_TOP + 3), 1.3), Action::Redraw);
        assert_eq!(app.view, View::Detail);
        app.view = View::List;
        assert_eq!(app.mouse(click(TABLE_TOP + 9), 2.0), Action::Nothing);
        assert_eq!(app.mouse(click(0), 2.0), Action::Nothing);
    }

    #[test]
    fn a_long_list_scrolls_with_the_selection_and_clicks_follow() {
        let mut app = app(30);
        app.settle(24);
        for _ in 0..20 {
            press(&mut app, KeyCode::Down);
            app.settle(24);
        }
        // 13 rows fit; row 20 is the last one shown.
        assert_eq!((app.selected_index(), app.scroll), (20, 8));
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 10,
            row: TABLE_TOP + 2,
            modifiers: KeyModifiers::NONE,
        };
        app.mouse(click, 1.0);
        app.settle(24);
        assert_eq!((app.selected_index(), app.scroll), (8, 8));
        // A click below the table, on the summary, selects nothing.
        let below = MouseEvent {
            row: TABLE_TOP + 2 + 13,
            ..click
        };
        assert_eq!(app.mouse(below, 5.0), Action::Nothing);
    }

    #[test]
    fn the_selection_follows_its_row() {
        let mut app = app(3);
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Down);
        let id = app.current().unwrap().id;
        app.registry.instances.remove(0);
        assert_eq!(app.current().unwrap().id, id);
        app.registry.instances.clear();
        assert!(app.current().is_none());
        assert_eq!(press(&mut app, KeyCode::Enter), Action::Redraw);
        assert_eq!(app.view, View::List);
    }
}
