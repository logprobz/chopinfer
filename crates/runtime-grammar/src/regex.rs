//! The supported regular expression subset, matched against whole outputs.
//!
//! Literals and escapes, `.`, bracket classes, `\d \w \s` and their
//! negations, groups, alternation, and the counted and open quantifiers.
//! `^` and `$` are accepted only at the ends of the pattern, where they mean
//! nothing more than a whole match already does. Everything that is not a
//! regular language (backreferences, lookaround) is refused.

use crate::expr::{CharClass, Expr, GrammarError};

const MAX_COUNT: u32 = 1 << 12;

/// Parses `pattern`, turning every matched character class into an
/// expression through `class`, so a caller can encode characters for the
/// context the pattern is matched in.
pub(crate) fn regex_expr(
    pattern: &str,
    class: &dyn Fn(CharClass) -> Expr,
) -> Result<Expr, GrammarError> {
    if pattern.len() > 65_536 {
        return Err(GrammarError::too_large(
            "regex source exceeds compilation budget",
        ));
    }
    let mut chars: Vec<char> = pattern.chars().collect();
    if chars.first() == Some(&'^') {
        chars.remove(0);
    }
    if chars.last() == Some(&'$') && !ends_escaped(&chars) {
        chars.pop();
    }
    let mut parser = Parser {
        chars,
        position: 0,
        class,
        depth: 0,
    };
    let expr = parser.alternation()?;
    if parser.position != parser.chars.len() {
        return Err(parser.error("unbalanced ')'"));
    }
    Ok(expr)
}

fn ends_escaped(chars: &[char]) -> bool {
    let backslashes = chars[..chars.len() - 1]
        .iter()
        .rev()
        .take_while(|&&c| c == '\\')
        .count();
    backslashes % 2 == 1
}

struct Parser<'a> {
    chars: Vec<char>,
    position: usize,
    class: &'a dyn Fn(CharClass) -> Expr,
    depth: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> GrammarError {
        GrammarError::unsupported(format!("regex {message} at offset {}", self.position))
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.position).copied()
    }

    fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn alternation(&mut self) -> Result<Expr, GrammarError> {
        let mut branches = vec![self.concatenation()?];
        while self.eat('|') {
            branches.push(self.concatenation()?);
        }
        Ok(Expr::alt(branches))
    }

    fn concatenation(&mut self) -> Result<Expr, GrammarError> {
        let mut items = Vec::new();
        while let Some(next) = self.peek() {
            if next == '|' || next == ')' {
                break;
            }
            let atom = self.atom()?;
            items.push(self.quantified(atom)?);
        }
        Ok(Expr::seq(items))
    }

    /// At most one quantifier per atom, as in Python's `re`: a stacked
    /// quantifier such as `a{0,1}{0,1}` is refused, which also keeps the
    /// nesting of the expression bounded by the group depth.
    fn quantified(&mut self, atom: Expr) -> Result<Expr, GrammarError> {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.position += 1;
                (0, None)
            }
            Some('+') => {
                self.position += 1;
                (1, None)
            }
            Some('?') => {
                self.position += 1;
                (0, Some(1))
            }
            Some('{') if counted_at(&self.chars, self.position) => self.counted()?,
            _ => return Ok(atom),
        };
        // A lazy quantifier matches the same language.
        self.eat('?');
        match self.peek() {
            Some('+') => Err(self.error("possessive quantifiers are not supported")),
            Some('*' | '?') => Err(self.error("multiple repeat")),
            Some('{') if counted_at(&self.chars, self.position) => {
                Err(self.error("multiple repeat"))
            }
            _ => Ok(atom.repeat(min, max)),
        }
    }

    fn number(&mut self) -> Result<u32, GrammarError> {
        let start = self.position;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.position += 1;
        }
        let text: String = self.chars[start..self.position].iter().collect();
        text.parse::<u32>()
            .ok()
            .filter(|&value| value <= MAX_COUNT)
            .ok_or_else(|| GrammarError::too_large("a regex repetition count is too large"))
    }

    fn counted(&mut self) -> Result<(u32, Option<u32>), GrammarError> {
        self.position += 1;
        let min = self.number()?;
        let max = if self.eat(',') {
            if self.peek() == Some('}') {
                None
            } else {
                Some(self.number()?)
            }
        } else {
            Some(min)
        };
        if !self.eat('}') {
            return Err(self.error("malformed repetition"));
        }
        if max.is_some_and(|max| max < min) {
            return Err(GrammarError::invalid(
                "a regex repetition has max below min",
            ));
        }
        Ok((min, max))
    }

    fn atom(&mut self) -> Result<Expr, GrammarError> {
        let Some(next) = self.peek() else {
            return Err(self.error("ends early"));
        };
        self.position += 1;
        match next {
            '(' => {
                if self.eat('?') && !self.eat(':') {
                    return Err(self.error("group extensions are not supported"));
                }
                self.depth += 1;
                if self.depth > 64 {
                    return Err(GrammarError::too_large("a regex nests too deeply"));
                }
                let inner = self.alternation()?;
                self.depth -= 1;
                if !self.eat(')') {
                    return Err(self.error("unclosed group"));
                }
                Ok(inner)
            }
            '[' => {
                let class = self.bracket()?;
                Ok((self.class)(class))
            }
            '.' => Ok((self.class)(CharClass::of("\n").complement())),
            '\\' => {
                let class = self.escape(false)?;
                Ok((self.class)(class))
            }
            '^' | '$' => Err(self.error("anchors are supported only at the pattern ends")),
            '*' | '+' | '?' => Err(self.error("a quantifier has nothing to repeat")),
            '{' if counted_at(&self.chars, self.position - 1) => {
                Err(self.error("a quantifier has nothing to repeat"))
            }
            literal => Ok((self.class)(CharClass::from_ranges([(
                literal as u32,
                literal as u32,
            )]))),
        }
    }

    /// One escape after a backslash, as the class it matches.
    fn escape(&mut self, in_class: bool) -> Result<CharClass, GrammarError> {
        let Some(next) = self.peek() else {
            return Err(self.error("ends in an escape"));
        };
        self.position += 1;
        let single = |c: char| CharClass::from_ranges([(c as u32, c as u32)]);
        Ok(match next {
            'd' => digits(),
            'D' => digits().complement(),
            'w' => word(),
            'W' => word().complement(),
            's' => space(),
            'S' => space().complement(),
            'n' => single('\n'),
            't' => single('\t'),
            'r' => single('\r'),
            'f' => single('\x0c'),
            'v' => single('\x0b'),
            '0' => single('\0'),
            'x' => {
                let value = self.hex(2)?;
                CharClass::from_ranges([(value, value)])
            }
            'u' => {
                let value = self.hex(4)?;
                CharClass::from_ranges([(value, value)])
            }
            'b' if in_class => single('\x08'),
            c if c.is_ascii_alphanumeric() => {
                return Err(self.error("this escape is not supported"));
            }
            c => single(c),
        })
    }

    fn hex(&mut self, digits: usize) -> Result<u32, GrammarError> {
        let end = self.position + digits;
        let text: String = self
            .chars
            .get(self.position..end)
            .ok_or_else(|| self.error("short hex escape"))?
            .iter()
            .collect();
        let value = u32::from_str_radix(&text, 16).map_err(|_| self.error("bad hex escape"))?;
        self.position = end;
        if char::from_u32(value).is_none() {
            return Err(self.error("hex escape is not a scalar value"));
        }
        Ok(value)
    }

    fn bracket(&mut self) -> Result<CharClass, GrammarError> {
        let negated = self.eat('^');
        let mut class = CharClass::default();
        let mut first = true;
        loop {
            let Some(next) = self.peek() else {
                return Err(self.error("unclosed class"));
            };
            if next == ']' && !first {
                self.position += 1;
                break;
            }
            first = false;
            if next == '[' && self.chars.get(self.position + 1) == Some(&':') {
                return Err(self.error("POSIX classes are not supported"));
            }
            let low = self.class_item()?;
            let is_range = self.peek() == Some('-')
                && self.chars.get(self.position + 1).is_some_and(|&c| c != ']');
            if is_range {
                self.position += 1;
                let high = self.class_item()?;
                let (Some(low), Some(high)) = (single_scalar(&low), single_scalar(&high)) else {
                    return Err(self.error("a class range needs single characters"));
                };
                if low > high {
                    return Err(GrammarError::invalid("a regex class range is reversed"));
                }
                class = class.union(&CharClass::from_ranges([(low, high)]));
            } else {
                class = class.union(&low);
            }
        }
        Ok(if negated { class.complement() } else { class })
    }

    fn class_item(&mut self) -> Result<CharClass, GrammarError> {
        let next = self.peek().expect("checked by caller");
        self.position += 1;
        if next == '\\' {
            self.escape(true)
        } else {
            Ok(CharClass::from_ranges([(next as u32, next as u32)]))
        }
    }
}

/// Whether a counted quantifier such as `{2}` or `{1,3}` starts at `position`;
/// any other brace is a literal.
fn counted_at(chars: &[char], position: usize) -> bool {
    let rest = &chars[position + 1..];
    rest.iter().position(|&c| c == '}').is_some_and(|end| {
        let body = &rest[..end];
        !body.is_empty()
            && body[0].is_ascii_digit()
            && body.iter().all(|c| c.is_ascii_digit() || *c == ',')
            && body.iter().filter(|&&c| c == ',').count() <= 1
    })
}

fn single_scalar(class: &CharClass) -> Option<u32> {
    match class.ranges() {
        [(lo, hi)] if lo == hi => Some(*lo),
        _ => None,
    }
}

pub(crate) fn digits() -> CharClass {
    CharClass::from_ranges([('0' as u32, '9' as u32)])
}

fn word() -> CharClass {
    CharClass::from_ranges([
        ('0' as u32, '9' as u32),
        ('A' as u32, 'Z' as u32),
        ('a' as u32, 'z' as u32),
        ('_' as u32, '_' as u32),
    ])
}

fn space() -> CharClass {
    CharClass::of(" \t\n\r\x0b\x0c")
}
