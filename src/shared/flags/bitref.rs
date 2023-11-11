//! Quick access to a bit within [`AtomicFlags`]

use super::{iter, AtomicFlags, AtomicWord, NonZeroWord, Word, WORD_BITS};
use std::{
    debug_assert,
    fmt::{self, Debug},
    sync::atomic::Ordering,
};

/// Quick access to a bit within [`AtomicFlags`]
#[derive(Clone)]
pub struct BitRef<'flags, const CACHE_ITER_MASKS: bool> {
    /// Word which the bit of interest belongs to
    word: &'flags AtomicWord,

    /// Index of the bit of interest within this word (smaller than Word::Bits)
    bit_shift: u32,

    /// Cache of significant word bits if CACHE_ITER_MASKS is set
    iter_masks_cache: Option<IteratorMasksCache>,
}
//
impl<'flags> BitRef<'flags, false> {
    /// Construct from flags and a bit index
    pub fn new(flags: &'flags AtomicFlags, bit_idx: usize) -> Option<Self> {
        (bit_idx < flags.len).then(|| {
            let word_idx = bit_idx / WORD_BITS;
            let bit_shift = (bit_idx % WORD_BITS) as u32;
            BitRef {
                word: &flags.words[word_idx],
                bit_shift,
                iter_masks_cache: None,
            }
        })
    }

    /// Cache the masks for neighbours iteration
    ///
    /// If you are going to do keep this BitRef around and do lots of neighbour
    /// iteration around this specific bit, having this cache around can greatly
    /// speed things up in the common case where there is a single word of flags
    /// and all flags are set or unset.
    pub fn with_cache(self, flags: &'flags AtomicFlags) -> BitRef<'flags, true> {
        BitRef {
            word: self.word,
            bit_shift: self.bit_shift,
            iter_masks_cache: Some(IteratorMasksCache {
                set_wo_self: iter::init_mask::<true, false, false>(flags, &self),
                unset_wo_self: NonZeroWord::new(iter::init_mask::<false, false, false>(
                    flags, &self,
                ))
                .unwrap(),
            }),
        }
    }
}
//
impl<'flags, const CACHE_ITER_MASKS: bool> BitRef<'flags, CACHE_ITER_MASKS> {
    /// Check if this bit is set with specified memory ordering
    pub fn is_set(&self, order: Ordering) -> bool {
        self.word.load(order) & self.bit_mask() != 0
    }

    /// Set this bit with the specified memory ordering, return former value
    pub fn fetch_set(&self, order: Ordering) -> bool {
        let bit = self.bit_mask();
        self.word.fetch_or(bit, order) & bit != 0
    }

    /// Clear this bit with the specified memory ordering, return former value
    pub fn fetch_clear(&self, order: Ordering) -> bool {
        let bit = self.bit_mask();
        self.word.fetch_and(!bit, order) & bit != 0
    }

    /// Distance from this bit to another bit
    pub fn distance(&self, other: &Self, flags: &'flags AtomicFlags) -> usize {
        self.linear_idx(flags).abs_diff(other.linear_idx(flags))
    }

    /// Check that this bit does belong to the specified flags
    pub(crate) fn belongs_to(&self, flags: &'flags AtomicFlags) -> bool {
        let word: *const AtomicWord = self.word;
        let start = flags.words.as_ptr();
        let end = start.wrapping_add(flags.words.len());
        word >= start && word < end
    }

    /// Index of the word that this bit belongs to within `flags`
    pub(crate) fn word_idx(&self, flags: &'flags AtomicFlags) -> usize {
        assert!(self.belongs_to(flags));
        unsafe { self.word_idx_unchecked(flags) }
    }

    /// Unchecked version of `word_idx`
    ///
    /// # Safety
    ///
    /// This `BitRef` must come from the specified flags, which you can check
    /// with `self.belongs_to(flags)`.
    pub(crate) unsafe fn word_idx_unchecked(&self, flags: &'flags AtomicFlags) -> usize {
        debug_assert!(self.belongs_to(flags));
        let word: *const AtomicWord = self.word;
        let start = flags.words.as_ptr();
        usize::try_from(word.offset_from(start)).unwrap_unchecked()
    }

    /// Index of this bit within the word of interest
    pub(crate) fn bit_shift(&self) -> u32 {
        if self.bit_shift < Word::BITS {
            self.bit_shift
        } else {
            unsafe { std::hint::unreachable_unchecked() }
        }
    }

    /// Word with only the bit at our bit_shift set
    pub(crate) fn bit_mask(&self) -> Word {
        1 << self.bit_shift
    }

    /// Linear index of the bit within the flags
    pub(crate) fn linear_idx(&self, flags: &'flags AtomicFlags) -> usize {
        self.word_idx(flags) * WORD_BITS + self.bit_shift() as usize
    }

    /// Initialization mask for a certain flavor of bit iterator
    pub(crate) fn iter_mask<const FIND_SET: bool, const INCLUDE_SELF: bool>(
        &self,
        flags: &'flags AtomicFlags,
    ) -> Word {
        if !INCLUDE_SELF && CACHE_ITER_MASKS {
            let cache = unsafe { self.iter_masks_cache.as_ref().unwrap_unchecked() };
            if FIND_SET {
                cache.set_wo_self
            } else {
                Word::from(cache.unset_wo_self)
            }
        } else {
            iter::init_mask::<FIND_SET, INCLUDE_SELF, CACHE_ITER_MASKS>(flags, self)
        }
    }

    /// Version of this bit reference without the iterator cache
    pub(crate) fn without_cache(&self) -> BitRef<'flags, false> {
        BitRef {
            word: self.word,
            bit_shift: self.bit_shift,
            iter_masks_cache: None,
        }
    }
}
//
impl<'flags, const CACHE_ITER_MASKS: bool> Debug for BitRef<'flags, CACHE_ITER_MASKS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitRef")
            .field(
                "word",
                &format!("{:064b}", self.word.load(Ordering::Relaxed)),
            )
            .field("bit_shift", &self.bit_shift)
            .field("iter_masks_cache", &self.iter_masks_cache)
            .finish()
    }
}
//
impl<'flags, const CACHE_ITER_MASKS: bool> Eq for BitRef<'flags, CACHE_ITER_MASKS> {}
//
impl<'flags, const LEFT_CACHED: bool, const RIGHT_CACHED: bool>
    PartialEq<BitRef<'flags, RIGHT_CACHED>> for BitRef<'flags, LEFT_CACHED>
{
    fn eq(&self, other: &BitRef<'flags, RIGHT_CACHED>) -> bool {
        std::ptr::eq(self.word, other.word) && self.bit_shift == other.bit_shift
    }
}

/// Cached masks for neighbour iterators
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct IteratorMasksCache {
    /// Iterator over unset indices, not including self
    unset_wo_self: NonZeroWord,

    /// Iterator over set indices, not including self
    set_wo_self: Word,
}

// TODO: Add tests
