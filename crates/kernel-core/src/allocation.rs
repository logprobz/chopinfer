//! Portable allocation sizes and declaration-ordered scratch traversal.

/// Physical backing and the visible prefix of that backing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Allocation {
    allocation_bytes: u64,
    span_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationError {
    Overflow,
    SpanExceedsAllocation,
}

impl Allocation {
    pub const fn new(allocation_bytes: u64, span_bytes: u64) -> Result<Self, AllocationError> {
        if span_bytes > allocation_bytes {
            return Err(AllocationError::SpanExceedsAllocation);
        }
        Ok(Self {
            allocation_bytes,
            span_bytes,
        })
    }

    pub const fn bytes(bytes: u64) -> Self {
        Self {
            allocation_bytes: bytes,
            span_bytes: bytes,
        }
    }

    /// Size a visible row count with additional backing rows.
    pub fn rows(rows: u64, slack_rows: u64, row_bytes: u64) -> Result<Self, AllocationError> {
        let physical = rows
            .checked_add(slack_rows)
            .and_then(|n| n.checked_mul(row_bytes))
            .ok_or(AllocationError::Overflow)?;
        let visible = rows
            .checked_mul(row_bytes)
            .ok_or(AllocationError::Overflow)?;
        Self::new(physical, visible)
    }

    pub const fn allocation_bytes(self) -> u64 {
        self.allocation_bytes
    }
    pub const fn span_bytes(self) -> u64 {
        self.span_bytes
    }
}

/// A record maps leaves in declaration order, stopping at the first error.
pub trait Traversal<T>: Sized {
    type Mapped<U>;
    fn try_map<U, E>(self, f: &mut impl FnMut(T) -> Result<U, E>) -> Result<Self::Mapped<U>, E>;
    fn visit(&self, f: &mut impl FnMut(&T));
    /// Visit leaves with declaration paths, such as `rows[1].flags`.
    fn visit_named(&self, f: &mut impl FnMut(&str, &T));
    fn map<U>(self, mut f: impl FnMut(T) -> U) -> Self::Mapped<U> {
        match self.try_map(&mut |value| Ok::<U, std::convert::Infallible>(f(value))) {
            Ok(value) => value,
            Err(error) => match error {},
        }
    }
}

/// Fallible fixed-array mapping without requiring default or copy leaves.
pub fn try_array<T, U, E, const N: usize>(
    values: [T; N],
    f: &mut impl FnMut(T) -> Result<U, E>,
) -> Result<[U; N], E> {
    let mut mapped = Vec::with_capacity(N);
    for value in values {
        mapped.push(f(value)?);
    }
    Ok(mapped
        .try_into()
        .unwrap_or_else(|_| unreachable!("fixed array length")))
}

/// Declare a model-owned generic record once. Field kinds are `value`,
/// `array(N)`, `record(Name)`, `records(Name, N)`, `optional(Name)`,
/// `list(Name)`, plain optional leaves (`option`), plain lists (`values`)
/// and fields of a fixed type that are not leaves (`scalar(Type)`), which
/// mapping copies and visits skip. Nested names are generic records from
/// this macro.
#[macro_export]
macro_rules! scratch_record {
    ($(#[$meta:meta])* $vis:vis struct $name:ident { $( $(#[$fmeta:meta])* $fv:vis $field:ident : $kind:ident $(($($arg:tt)*))? ),* $(,)? }) => {
        $crate::scratch_record!(@fields
            [$(#[$meta])* $vis struct $name]
            [$( $(#[$fmeta])* $fv $field : $kind $(($($arg)*))?, )*]
            []
            $( [$(#[$fmeta])* $fv $field] $kind $(($($arg)*))?, )*
        );
    };
    // Resolve types before emitting the struct so built-in derives see types,
    // rather than unexpanded type-position macro calls.
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]) => {
        $crate::scratch_record!(@emit [$($head)*] [$($decl)*] [$($fields)*]);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] value, $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: T,] $($rest)*);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] array($n:expr), $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: [T; $n],] $($rest)*);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] record($r:ident), $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: $r<T>,] $($rest)*);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] records($r:ident, $n:expr), $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: [$r<T>; $n],] $($rest)*);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] optional($r:ident), $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: Option<$r<T>>,] $($rest)*);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] list($r:ident), $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: Vec<$r<T>>,] $($rest)*);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] option, $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: Option<T>,] $($rest)*);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] values, $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: Vec<T>,] $($rest)*);
    };
    (@fields [$($head:tt)*] [$($decl:tt)*] [$($fields:tt)*]
        [$($field:tt)*] scalar($t:ty), $($rest:tt)*) => {
        $crate::scratch_record!(@fields [$($head)*] [$($decl)*]
            [$($fields)* $($field)*: $t,] $($rest)*);
    };
    (@emit [$(#[$meta:meta])* $vis:vis struct $name:ident]
        [$( $(#[$fmeta:meta])* $fv:vis $field:ident : $kind:ident $(($($arg:tt)*))?, )*]
        [$($fields:tt)*]) => {
        $(#[$meta])* $vis struct $name<T> { $($fields)* }
        impl<T> $crate::allocation::Traversal<T> for $name<T> {
            type Mapped<U> = $name<U>;
            fn try_map<U, E>(self, f: &mut impl FnMut(T) -> Result<U, E>) -> Result<$name<U>, E> {
                Ok($name { $( $field: $crate::scratch_record!(@map self.$field, f, $kind $(($($arg)*))?), )* })
            }
            fn visit(&self, f: &mut impl FnMut(&T)) {
                $( $crate::scratch_record!(@visit &self.$field, f, $kind $(($($arg)*))?); )*
            }
            fn visit_named(&self, f: &mut impl FnMut(&str, &T)) {
                $( $crate::scratch_record!(@named &self.$field, stringify!($field), f, $kind $(($($arg)*))?); )*
            }
        }
    };
    (@map $v:expr, $f:ident, option) => { $v.map(&mut *$f).transpose()? };
    (@visit $v:expr, $f:ident, option) => { if let Some(v) = $v { $f(v); } };
    (@named $v:expr, $name:expr, $f:ident, value) => { $f($name, $v) };
    (@named $v:expr, $name:expr, $f:ident, option) => { if let Some(v) = $v { $f($name, v); } };
    (@map $v:expr, $f:ident, values) => {
        $v.into_iter().map(&mut *$f).collect::<Result<Vec<_>, E>>()?
    };
    (@visit $v:expr, $f:ident, values) => { for v in $v { $f(v); } };
    (@named $v:expr, $name:expr, $f:ident, values) => {
        for (index, v) in ($v).iter().enumerate() { $f(&format!("{}[{}]", $name, index), v); }
    };
    (@map $v:expr, $f:ident, scalar($t:ty)) => { $v };
    (@visit $v:expr, $f:ident, scalar($t:ty)) => {};
    (@named $v:expr, $name:expr, $f:ident, scalar($t:ty)) => {};
    (@named $v:expr, $name:expr, $f:ident, array($n:expr)) => {
        for (index, v) in ($v).iter().enumerate() { $f(&format!("{}[{}]", $name, index), v); }
    };
    (@named $v:expr, $name:expr, $f:ident, record($r:ident)) => {
        $crate::allocation::Traversal::visit_named($v, &mut |name, v| $f(&format!("{}.{}", $name, name), v));
    };
    (@named $v:expr, $name:expr, $f:ident, optional($r:ident)) => {
        if let Some(v) = $v {
            $crate::allocation::Traversal::visit_named(v, &mut |name, v| $f(&format!("{}.{}", $name, name), v));
        }
    };
    (@named $v:expr, $name:expr, $f:ident, $kind:ident($($args:tt)*)) => {
        for (index, v) in ($v).iter().enumerate() {
            $crate::allocation::Traversal::visit_named(v, &mut |name, v| $f(&format!("{}[{}].{}", $name, index, name), v));
        }
    };
    (@map $v:expr, $f:ident, value) => { $f($v)? };
    (@map $v:expr, $f:ident, array($n:expr)) => { $crate::allocation::try_array($v, $f)? };
    (@map $v:expr, $f:ident, record($r:ident)) => { $crate::allocation::Traversal::try_map($v, $f)? };
    (@map $v:expr, $f:ident, records($r:ident, $n:expr)) => {
        $crate::allocation::try_array($v, &mut |v| $crate::allocation::Traversal::try_map(v, $f))?
    };
    (@map $v:expr, $f:ident, optional($r:ident)) => {
        $v.map(|v| $crate::allocation::Traversal::try_map(v, $f)).transpose()?
    };
    (@map $v:expr, $f:ident, list($r:ident)) => {
        $v.into_iter().map(|v| $crate::allocation::Traversal::try_map(v, $f)).collect::<Result<Vec<_>, E>>()?
    };
    (@visit $v:expr, $f:ident, value) => { $f($v) };
    (@visit $v:expr, $f:ident, array($n:expr)) => { for v in $v { $f(v); } };
    (@visit $v:expr, $f:ident, record($r:ident)) => { $crate::allocation::Traversal::visit($v, $f) };
    (@visit $v:expr, $f:ident, optional($r:ident)) => {
        if let Some(v) = $v { $crate::allocation::Traversal::visit(v, $f); }
    };
    (@visit $v:expr, $f:ident, $kind:ident($($args:tt)*)) => {
        for v in $v { $crate::allocation::Traversal::visit(v, $f); }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    crate::scratch_record! { #[derive(Clone, Copy, Debug, PartialEq)] struct Pair { a: value, b: value } }
    crate::scratch_record! { struct Rows {
        first: value, fixed: array(2), nested: record(Pair),
        groups: records(Pair, 1), optional: optional(Pair), tail: list(Pair),
    } }
    crate::scratch_record! {
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Documented {
            /// A documented allocation field.
            #[allow(dead_code)]
            pub plain: value,
            /// Fixed rows retain their type.
            rows: array(2),
            nested: record(Pair),
            groups: records(Pair, 2),
            optional: optional(Pair),
            temporary: option,
        }
    }
    #[test]
    fn allocation_record_supports_derives_and_field_attributes() {
        let record = Documented {
            plain: 1,
            rows: [2, 3],
            nested: Pair { a: 4, b: 5 },
            groups: [Pair { a: 6, b: 7 }, Pair { a: 8, b: 9 }],
            optional: Some(Pair { a: 10, b: 11 }),
            temporary: Some(12),
        };
        let copied = record;
        assert_eq!(record, copied);
        assert!(format!("{record:?}").starts_with("Documented { plain: 1, rows: [2, 3]"));
        fn clone_record<T: Clone>(value: &T) -> T {
            value.clone()
        }
        assert_eq!(clone_record(&record), record);
    }
    fn rows(present: bool) -> Rows<u32> {
        Rows {
            first: 1,
            fixed: [2, 3],
            nested: Pair { a: 4, b: 5 },
            groups: [Pair { a: 6, b: 7 }],
            optional: present.then_some(Pair { a: 8, b: 9 }),
            tail: vec![Pair { a: 10, b: 11 }],
        }
    }
    #[test]
    fn allocation_order_map_visit_and_every_failure() {
        for present in [false, true] {
            let expected = if present {
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            } else {
                vec![1, 2, 3, 4, 5, 6, 7, 10, 11]
            };
            let mut visited = Vec::new();
            rows(present)
                .map(|v| v + 20)
                .visit(&mut |v| visited.push(*v));
            assert_eq!(visited, expected.iter().map(|v| v + 20).collect::<Vec<_>>());
            for cutoff in 0..expected.len() {
                let mut seen = Vec::new();
                let result = rows(present).try_map(&mut |v| {
                    if seen.len() == cutoff {
                        return Err(cutoff);
                    }
                    seen.push(v);
                    Ok(v)
                });
                assert_eq!(result.err(), Some(cutoff));
                assert_eq!(seen, expected[..cutoff]);
            }
        }
    }
    #[test]
    fn allocation_named_paths_follow_declaration_order() {
        for present in [false, true] {
            let mut named = Vec::new();
            rows(present).visit_named(&mut |name, value| named.push((name.to_owned(), *value)));
            let mut expected = vec![
                ("first", 1),
                ("fixed[0]", 2),
                ("fixed[1]", 3),
                ("nested.a", 4),
                ("nested.b", 5),
                ("groups[0].a", 6),
                ("groups[0].b", 7),
            ];
            if present {
                expected.extend([("optional.a", 8), ("optional.b", 9)]);
            }
            expected.extend([("tail[0].a", 10), ("tail[0].b", 11)]);
            assert_eq!(
                named,
                expected
                    .into_iter()
                    .map(|(name, v)| (name.to_owned(), v))
                    .collect::<Vec<_>>()
            );
        }
        for value in [None, Some(19)] {
            crate::scratch_record! { struct Temporary { value: option } }
            let mut named = Vec::new();
            Temporary { value }
                .map(|v: u32| v.to_string())
                .visit_named(&mut |name, v| named.push((name.to_owned(), v.clone())));
            assert_eq!(
                named,
                value
                    .map(|v| ("value".to_owned(), v.to_string()))
                    .into_iter()
                    .collect::<Vec<_>>()
            );
            let mut visited = Vec::new();
            Temporary { value }.visit(&mut |v| visited.push(*v));
            assert_eq!(visited, value.into_iter().collect::<Vec<_>>());
        }
        crate::scratch_record! { struct Plain { head: value, items: values, count: scalar(u64) } }
        let mut named = Vec::new();
        let plain = Plain {
            head: 1,
            items: vec![2, 3],
            count: 7,
        };
        plain
            .map(|v: u32| v * 10)
            .visit_named(&mut |name, v| named.push((name.to_owned(), *v)));
        assert_eq!(
            named,
            [("head", 10), ("items[0]", 20), ("items[1]", 30)].map(|(n, v)| (n.to_owned(), v))
        );
        let failed = Plain {
            head: 1,
            items: vec![2, 3],
            count: 7,
        }
        .try_map(&mut |v: u32| if v == 3 { Err(v) } else { Ok(v) });
        assert_eq!(failed.err(), Some(3));
        let mapped = Plain {
            head: 1,
            items: vec![2],
            count: 7,
        }
        .map(|v: u32| u64::from(v) + 1);
        assert_eq!((mapped.head, mapped.items, mapped.count), (2, vec![3], 7));
    }
    #[test]
    fn allocation_checks_sizes_and_overflow() {
        assert_eq!(Allocation::rows(3, 2, 4), Allocation::new(20, 12));
        assert_eq!(
            Allocation::new(3, 4),
            Err(AllocationError::SpanExceedsAllocation)
        );
        for (rows, slack, bytes) in [(u64::MAX, 1, 1), (u64::MAX, 0, 2), (1, u64::MAX - 1, 2)] {
            assert_eq!(
                Allocation::rows(rows, slack, bytes),
                Err(AllocationError::Overflow)
            );
        }
    }
}
