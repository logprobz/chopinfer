//! The RoPE and KV write family's ops and buffer roles: rotary embedding of
//! query and key heads on its own, RoPE of packed QKV rows fused with the
//! write of K and V into a paged cache or a window ring, and the write of
//! K and V rows into window rings. The rotation is [`RowQuery::Rope`] with
//! its oracle (`oracle::rope`); the ring write is [`RowQuery::RingWrite`]
//! (`oracle::ring_kv_write`); FP8 storage is `kv_fp8::encode`.
//!
//! The ops add what the queries leave out and a model needs: the multiplier
//! applied to V as it is written (a model's attention value scale), the
//! cache layout and the positions' integer type. FP8 K and V scales are
//! values of the cache ([`KvCache::k_scale`], [`KvCache::v_scale`]), passed
//! with each launch. Every field is math or a layout the model shares with
//! its attention; block shapes, head groupings per block and capacity
//! buckets are the backend's choices.
//!
//! The KV cache layout is the attention family's: K and V planes of `slots`
//! pages (paged) or rings (ring), each `[token][kv_head][dim]` in the
//! cache's dtype. A BF16 cache stores values rounded to BF16; an FP8 E4M3
//! cache stores one code per value, `E4M3(value / scale)` with the plane's
//! tensor scale (`kv_fp8::encode`), and attention multiplies the scale back
//! in. Scales are not stored in the planes.
//!
//! The value a fused RoPE write stores is the rotated K value, or the V
//! value times `value_scale`, in F32 before any BF16 rounding (a backend
//! computing wider rounds it to F32 once). A BF16 cache stores its BF16
//! rounding, equal to `k_new` and `v_new`. An FP8 cache stores
//! `kv_fp8::encode` of the F32 value: `value / scale` divided in F32, then
//! rounded to the nearest E4M3. It does not encode the BF16 rounding, which
//! can give another code where it lands on an E4M3 rounding boundary. A
//! ring write's inputs are BF16 rows, so it encodes those. A check of an FP8
//! store admits each code that correct rounding of a value within the
//! rotation's allowance gives.

pub use crate::attention::KvCache;
use crate::contract::{DType, Refusal, RowQuery, Validate};

/// A rotation of each row's heads on its own, written to other rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeOp {
    /// `RowQuery::Rope` without a fused KV write: the head width, the
    /// rotated width, theta, the pairing and the angle source.
    pub rope: RowQuery,
    /// Heads of each row.
    pub heads: u32,
    /// The positions' type: `I32` or `I64`.
    pub positions: DType,
}

impl Validate for RopeOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.rope.validate()?;
        if !matches!(
            self.rope,
            RowQuery::Rope {
                fused_paged_kv_write: false,
                ..
            }
        ) {
            return Err(Refusal("a RoPE op rotates without a fused KV write"));
        }
        if self.heads == 0 {
            return Err(Refusal("a RoPE op rotates at least one head"));
        }
        check_positions(self.positions)
    }
}

/// RoPE of packed QKV rows fused with the write of each row's K and V into
/// a KV cache. Each row is `[q_heads * head_dim | kv_heads * head_dim |
/// kv_heads * v_dim]`. Q and K heads are rotated at the row's position as
/// `rope` describes; V is multiplied by `value_scale`. The op writes the
/// rotated Q rows, and, when bound, the rotated K and scaled V rows
/// (`k_new`, `v_new`, rounded to BF16), and stores K and V in the cache at
/// the row's item and position (see [`KvWrite`]; the module docs state the
/// stored value).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeKvOp {
    /// `RowQuery::Rope`; `fused_paged_kv_write` is set for a paged cache.
    pub rope: RowQuery,
    pub q_heads: u32,
    pub kv_heads: u32,
    /// The width of a V head; K heads are `rope`'s head width.
    pub v_dim: u32,
    /// The multiplier of V before it is written, one for none.
    pub value_scale: f32,
    /// The positions' type: `I32` or `I64`.
    pub positions: DType,
    pub cache: KvWrite,
}

/// Where a fused write stores K and V.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KvWrite {
    /// Pages of `page_size` tokens: a row at `position` of item `b` goes to
    /// page `block_tables[b][position / page_size]`, token `position %
    /// page_size`. An item's block table row holds `max_context / page_size`
    /// pages, rounded up, as attention reads it.
    Paged {
        page_size: u32,
        max_context: u32,
        kv_dtype: DType,
    },
    /// Rings of `ring_size` tokens, one per item: a row at `position` of
    /// item `b` goes to ring `slot_ids[b]`, token `position % ring_size`.
    /// Without `store` the op writes only `k_new` and `v_new`, and a later
    /// ring write (`RowQuery::RingWrite`) stores them.
    Ring {
        ring_size: u32,
        kv_dtype: DType,
        store: bool,
    },
}

impl KvWrite {
    pub fn kv_dtype(self) -> DType {
        match self {
            Self::Paged { kv_dtype, .. } | Self::Ring { kv_dtype, .. } => kv_dtype,
        }
    }
}

impl Validate for RopeKvOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.rope.validate()?;
        let RowQuery::Rope {
            fused_paged_kv_write,
            ..
        } = self.rope
        else {
            return Err(Refusal("a RoPE KV write rotates by a RoPE query"));
        };
        if fused_paged_kv_write != matches!(self.cache, KvWrite::Paged { .. }) {
            return Err(Refusal(
                "a RoPE query fuses a paged KV write exactly when the cache is paged",
            ));
        }
        if self.q_heads == 0 || self.kv_heads == 0 || self.v_dim == 0 {
            return Err(Refusal(
                "a RoPE KV write needs Q and KV heads and a V width",
            ));
        }
        if !self.value_scale.is_finite() {
            return Err(Refusal("a RoPE KV write needs a finite value scale"));
        }
        check_kv_dtype(self.cache.kv_dtype())?;
        if let KvWrite::Paged { page_size: 0, .. }
        | KvWrite::Paged { max_context: 0, .. }
        | KvWrite::Ring { ring_size: 0, .. } = self.cache
        {
            return Err(Refusal(
                "a KV write needs a page size and context, or a ring size",
            ));
        }
        check_positions(self.positions)
    }
}

/// The write of K and V rows into window rings (`RowQuery::RingWrite`): a
/// row at `position` of item `b` goes to ring `slot_ids[b]`, token
/// `position % ring_size`, unless a later row of its item will take that
/// token (`position + ring_size <= last_positions[b]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingWriteOp {
    pub query: RowQuery,
    /// The positions' type: `I32` or `I64`.
    pub positions: DType,
    /// The model launches it for one item whose rows hold consecutive
    /// positions, so only the last `ring_size` rows are written.
    pub single_sequence: bool,
}

impl Validate for RingWriteOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !matches!(self.query, RowQuery::RingWrite { .. }) {
            return Err(Refusal("a ring write writes by a ring write query"));
        }
        check_positions(self.positions)
    }
}

fn check_positions(positions: DType) -> Result<(), Refusal> {
    if matches!(positions, DType::I32 | DType::I64) {
        Ok(())
    } else {
        Err(Refusal("positions are I32 or I64"))
    }
}

/// The cache dtypes whose storage the family defines.
fn check_kv_dtype(dtype: DType) -> Result<(), Refusal> {
    if matches!(dtype, DType::Bf16 | DType::Fp8E4M3) {
        Ok(())
    } else {
        Err(Refusal("a KV cache stores BF16 or FP8 E4M3"))
    }
}

crate::scratch_record! {
    /// A RoPE op's buffers by role: BF16 `input` and `output` rows
    /// `[row][heads * head_dim]`, each row's position (`positions`), the F32
    /// `[position][rotary / 2][cos, sin]` table of a table-backed rotation
    /// (`cos_sin`) and the device count of live rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RopeBuffers {
        pub input: value,
        pub output: value,
        pub positions: value,
        pub cos_sin: option,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A RoPE KV write's buffers by role: the BF16 packed rows `qkv`, the
    /// rotated BF16 Q rows `q` (`[row][q_heads * head_dim]`), the optional
    /// BF16 K and V rows `k_new` (`[row][kv_heads * head_dim]`) and `v_new`
    /// (`[row][kv_heads * v_dim]`), the cache, each item's I32 block table
    /// row (`block_tables`, paged) or ring (`slot_ids`, ring), each row's
    /// I32 item (`token_batch`) and position, the table of a table-backed
    /// rotation (`cos_sin`, as [`RopeBuffers`]) and the device count of
    /// live rows. A ring op without `store` needs `k_new` and `v_new`.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct RopeKvBuffers {
        pub qkv: value,
        pub q: value,
        pub k_new: option,
        pub v_new: option,
        pub cache: record(KvCache),
        pub block_tables: option,
        pub slot_ids: option,
        pub token_batch: value,
        pub positions: value,
        pub cos_sin: option,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A ring write's buffers by role: the BF16 rows `k_new`
    /// (`[row][kv_heads * qk_dim]`) and `v_new` (`[row][kv_heads * v_dim]`),
    /// the rings, each item's I32 ring (`slot_ids`) and I64 last position
    /// (`last_positions`), each row's I32 item (`token_batch`) and position,
    /// and the device count of live rows.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct RingWriteBuffers {
        pub k_new: value,
        pub v_new: value,
        pub cache: record(KvCache),
        pub slot_ids: value,
        pub token_batch: value,
        pub positions: value,
        pub last_positions: value,
        pub live_rows: value,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::allocation::Traversal;
    use crate::contract::{RotarySource, RotaryStyle};

    pub(crate) fn rope(fused_paged_kv_write: bool) -> RowQuery {
        RowQuery::Rope {
            head_dim: 128,
            rotary_dim: 64,
            theta: 1e4,
            style: RotaryStyle::Neox,
            source: RotarySource::CosSinTable,
            fused_paged_kv_write,
        }
    }

    pub(crate) fn paged() -> RopeKvOp {
        RopeKvOp {
            rope: rope(true),
            q_heads: 16,
            kv_heads: 2,
            v_dim: 128,
            value_scale: 1.0,
            positions: DType::I32,
            cache: KvWrite::Paged {
                page_size: 64,
                max_context: 4096,
                kv_dtype: DType::Bf16,
            },
        }
    }

    #[test]
    fn ops_check_their_math_only() {
        assert_eq!(paged().validate(), Ok(()));
        // Any head counts and widths are math; a backend refuses what it
        // lacks.
        let odd = RopeKvOp {
            q_heads: 3,
            kv_heads: 5,
            v_dim: 96,
            value_scale: 0.707,
            cache: KvWrite::Paged {
                page_size: 48,
                max_context: 100,
                kv_dtype: DType::Fp8E4M3,
            },
            ..paged()
        };
        assert_eq!(odd.validate(), Ok(()));
        let ring = RopeKvOp {
            rope: rope(false),
            cache: KvWrite::Ring {
                ring_size: 128,
                kv_dtype: DType::Bf16,
                store: false,
            },
            ..paged()
        };
        assert_eq!(ring.validate(), Ok(()));
        // The query's fused flag names the cache.
        assert!(
            RopeKvOp {
                rope: rope(false),
                ..paged()
            }
            .validate()
            .is_err()
        );
        for bad in [
            RopeKvOp {
                value_scale: f32::NAN,
                ..paged()
            },
            RopeKvOp {
                positions: DType::U8,
                ..paged()
            },
            RopeKvOp {
                v_dim: 0,
                ..paged()
            },
            RopeKvOp {
                cache: KvWrite::Paged {
                    page_size: 64,
                    max_context: 4096,
                    kv_dtype: DType::Fp8E5M2,
                },
                ..paged()
            },
            RopeKvOp {
                cache: KvWrite::Paged {
                    page_size: 0,
                    max_context: 4096,
                    kv_dtype: DType::Bf16,
                },
                ..paged()
            },
            RopeKvOp {
                cache: KvWrite::Paged {
                    page_size: 64,
                    max_context: 0,
                    kv_dtype: DType::Bf16,
                },
                ..paged()
            },
            RopeKvOp {
                cache: KvWrite::Ring {
                    ring_size: 0,
                    kv_dtype: DType::Bf16,
                    store: true,
                },
                ..ring
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        let alone = RopeOp {
            rope: rope(false),
            heads: 4,
            positions: DType::I64,
        };
        assert_eq!(alone.validate(), Ok(()));
        assert!(
            RopeOp {
                rope: rope(true),
                ..alone
            }
            .validate()
            .is_err()
        );
        assert!(RopeOp { heads: 0, ..alone }.validate().is_err());
        let write = RingWriteOp {
            query: RowQuery::RingWrite {
                qk_dim: 192,
                v_dim: 128,
                kv_heads: 3,
                ring_size: 100,
                kv_dtype: DType::Fp8E4M3,
            },
            positions: DType::I32,
            single_sequence: false,
        };
        assert_eq!(write.validate(), Ok(()));
        assert!(
            RingWriteOp {
                query: rope(false),
                ..write
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn roles_are_named_by_path() {
        let buffers = RopeKvBuffers {
            qkv: 1,
            q: 2,
            k_new: Some(3),
            v_new: None,
            cache: KvCache {
                k: 4,
                v: 5,
                slots: 8,
                k_slot_stride: 64,
                v_slot_stride: 64,
                k_scale: 1.0,
                v_scale: 1.0,
            },
            block_tables: Some(6),
            slot_ids: None,
            token_batch: 7,
            positions: 8,
            cos_sin: Some(9),
            live_rows: 10,
        };
        let mut named = Vec::new();
        buffers.visit_named(&mut |role, &view| named.push((role.to_owned(), view)));
        let roles: Vec<_> = named.iter().map(|(role, _)| role.as_str()).collect();
        assert_eq!(
            roles,
            [
                "qkv",
                "q",
                "k_new",
                "cache.k",
                "cache.v",
                "block_tables",
                "token_batch",
                "positions",
                "cos_sin",
                "live_rows"
            ]
        );
    }
}
