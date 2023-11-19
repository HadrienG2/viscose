//! Quick access to a bit within [`AtomicFlags`]

use super::{AtomicFlags, AtomicWord, NonZeroWord, Word, WORD_BITS};
use std::{
    debug_assert,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    sync::atomic::Ordering,
};

/// Quick access to a bit within [`AtomicFlags`]
#[derive(Clone)]
pub struct BitRef<'flags, const CACHE_SEARCH_MASKS: bool> {
    /// Word which the bit of interest belongs to
    word: &'flags AtomicWord,

    /// Index of the bit of interest within this word (smaller than Word::Bits)
    bit_shift: u32,

    /// Cache of masks for bit value searches, if CACHE_SEARCH_MASKS is set
    search_masks_cache: Option<SearchMasksCache>,
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
                search_masks_cache: None,
            }
        })
    }

    /// Cache the masks for neighbours iteration
    ///
    /// If you are going to do keep this BitRef around and do lots of neighbour
    /// iteration around this specific bit, having this cache around can greatly
    /// speed things up in the common case where there is a single word of flags
    /// and all flags are set or unset.
    pub fn with_cache(&self, flags: &'flags AtomicFlags) -> BitRef<'flags, true> {
        BitRef {
            word: self.word,
            bit_shift: self.bit_shift,
            search_masks_cache: Some(SearchMasksCache {
                set_with_self: NonZeroWord::new(self.compute_search_mask::<true, true>(flags))
                    .unwrap(),
                set_wo_self: self.compute_search_mask::<true, false>(flags),
                unset_with_self: self.compute_search_mask::<false, true>(flags),
                unset_wo_self: NonZeroWord::new(self.compute_search_mask::<false, false>(flags))
                    .unwrap(),
            }),
        }
    }
}
//
impl<'flags> BitRef<'flags, true> {
    /// Set this bit and check if the surrounding word was all-zeros before
    ///
    /// This can be used to tell whether the surrounding AtomicFlags could have
    /// been all-zeroes before setting this flag.
    pub fn check_empty_and_set(&self, order: Ordering) -> FormerWordState {
        let bit = self.bit_mask();
        match self.word.fetch_or(bit, order) & self.cached_search_mask::<true, true>() {
            Word::MIN => FormerWordState::EmptyOrFull,
            other => FormerWordState::OtherWithBit(other & bit != 0),
        }
    }

    /// Clear this bit and check if the surrounding word was all-ones before
    ///
    /// This can be used to tell whether the surrounding AtomicFlags could have
    /// been all-ones before clearing this flag.
    pub fn check_full_and_clear(&self, order: Ordering) -> FormerWordState {
        let bit = self.bit_mask();
        match self.word.fetch_and(!bit, order) | self.cached_search_mask::<false, true>() {
            Word::MAX => FormerWordState::EmptyOrFull,
            other => FormerWordState::OtherWithBit(other & bit != 0),
        }
    }

    /// Cached version of `Self::search_mask()`
    fn cached_search_mask<const FIND_SET: bool, const INCLUDE_SELF: bool>(&self) -> Word {
        // SAFETY: Cannot construct a BitRef where CACHE_SEARCH_MASKS is true
        //         but a search mask cache isn't present.
        unsafe { self.cached_search_mask_unchecked::<FIND_SET, INCLUDE_SELF>() }
    }
}
//
impl<'flags, const CACHE_SEARCH_MASKS: bool> BitRef<'flags, CACHE_SEARCH_MASKS> {
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

    /// Number of left bit shifts from `origin` to `self`
    pub fn offset_from(&self, origin: &Self, flags: &'flags AtomicFlags) -> usize {
        self.linear_idx(flags) - origin.linear_idx(flags)
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

    /// Bit mask used to search for a given bit value in a word of AtomicFlags
    ///
    /// Bit masks used to search for set bits have all significant bits set to 1
    /// and other insignificant bits set to 0. They are AND-ed with the word
    /// value to clear all set bits from the insignificant portion of the word,
    /// so that they are ignored by the search for set bits.
    ///
    /// Bit masks used to search for unset bits have all significant bits set to
    /// 0 and other insignificant bits set to 1. They are OR-ed with the word
    /// value to set all the unset bits in the insignificant portion of the
    /// word, so that they are ignored by the search for unset bits.
    ///
    /// Workers will often want to exclude their own bit from the search by
    /// clearing INCLUDE_SELF, since they are looking to exchange work with
    /// others, not themselves. However, threads external to the thread pool
    /// must search over all bits.
    pub(crate) fn search_mask<const FIND_SET: bool, const INCLUDE_SELF: bool>(
        &self,
        flags: &'flags AtomicFlags,
    ) -> Word {
        if CACHE_SEARCH_MASKS {
            // SAFETY: Checked that CACHE_SEARCH_MASKS is true
            unsafe { self.cached_search_mask_unchecked::<FIND_SET, INCLUDE_SELF>() }
        } else {
            self.compute_search_mask::<FIND_SET, INCLUDE_SELF>(flags)
        }
    }

    /// Compute the bit mask to search for a given bit value in a word
    ///
    /// See [`Self::search_mask()`] for details.
    fn compute_search_mask<const FIND_SET: bool, const INCLUDE_START: bool>(
        &self,
        flags: &'flags AtomicFlags,
    ) -> Word {
        let word_idx = self.word_idx(flags);
        let significant = Word::from(flags.significant_bits(word_idx));
        let bit = self.bit_mask();
        match [FIND_SET, INCLUDE_START] {
            [false, false] => !significant | bit,
            [false, true] => !significant,
            [true, false] => significant & !bit,
            [true, true] => significant,
        }
    }

    /// Unchecked access to the cached version of `Self::search_mask()`
    ///
    /// # Safety
    ///
    /// This should only be called if `CACHE_SEARCH_MASKS` is true, and thus a
    /// search masks cache is guaranteed to be present.
    unsafe fn cached_search_mask_unchecked<const FIND_SET: bool, const INCLUDE_SELF: bool>(
        &self,
    ) -> Word {
        // SAFETY: Per precondition
        let cache = unsafe { self.search_masks_cache.as_ref().unwrap_unchecked() };
        match (FIND_SET, INCLUDE_SELF) {
            (false, false) => cache.unset_wo_self.into(),
            (false, true) => cache.unset_with_self,
            (true, false) => cache.set_wo_self,
            (true, true) => cache.set_with_self.into(),
        }
    }

    /// Version of this bit reference without the iterator cache
    pub fn without_cache(&self) -> BitRef<'flags, false> {
        BitRef {
            word: self.word,
            bit_shift: self.bit_shift,
            search_masks_cache: None,
        }
    }
}
//
impl<'flags, const CACHE_SEARCH_MASKS: bool> Debug for BitRef<'flags, CACHE_SEARCH_MASKS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitRef")
            .field(
                "word",
                &format!("{:064b}", self.word.load(Ordering::Relaxed)),
            )
            .field("bit_shift", &self.bit_shift)
            .field("search_masks_cache", &self.search_masks_cache)
            .finish()
    }
}
//
impl<'flags, const CACHE_SEARCH_MASKS: bool> Eq for BitRef<'flags, CACHE_SEARCH_MASKS> {}
//
impl<'flags, const CACHE_SEARCH_MASKS: bool> Hash for BitRef<'flags, CACHE_SEARCH_MASKS> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let word: *const AtomicWord = self.word;
        word.hash(state);
        self.bit_shift.hash(state);
    }
}
//
impl<'flags, const LEFT_CACHED: bool, const RIGHT_CACHED: bool>
    PartialEq<BitRef<'flags, RIGHT_CACHED>> for BitRef<'flags, LEFT_CACHED>
{
    fn eq(&self, other: &BitRef<'flags, RIGHT_CACHED>) -> bool {
        std::ptr::eq(self.word, other.word) && self.bit_shift == other.bit_shift
    }
}

/// State of a word before a check_xyz_and_abc operation
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum FormerWordState {
    /// Word was in the state of interest, empty or full
    EmptyOrFull,

    /// Word was in any other state with the bit of interest in this state
    OtherWithBit(bool),
}

/// Cached masks for bit value searches
#[derive(Copy, Clone, Eq, Hash, PartialEq)]
struct SearchMasksCache {
    /// Mask to be used when searching for set bits, including self
    ///
    /// All significant bits including self are set to 1, all other bits are set
    /// to 0. Use by AND-ing with the current word value, neutral is Word::MIN.
    ///
    /// Cannot be all-zeroes because each word has at least one significant bit
    /// (otherwise, why would we allocate it?)
    set_with_self: NonZeroWord,

    /// Mask to be used when searching for set bits, not including self
    ///
    /// All significant bits except self are set to 1, all other bits are set to
    /// 0. Use by AND-ing with the current word value, neutral is Word::MIN.
    ///
    /// Can be all-zeroes if self is the only significant bit.
    set_wo_self: Word,

    /// Mask to be used when searching for unset bits, including self
    ///
    /// All significant bits including self are set to 0, all other bits are set
    /// to 1. Use by OR-ing with the current word value, neutral is Word::MAX.
    ///
    /// Can be all-zeroes if all bits are significant.
    unset_with_self: Word,

    /// Mask to be used when searching for unset bits, not including self
    ///
    /// All significant bits except self are set to 0, all other bits are set to
    /// 1. Use by OR-ing with the current word value, neutral is Word::MAX.
    ///
    /// Cannot be all-zeroes because at least one bit associated with self witll
    /// be set to 1.
    unset_wo_self: NonZeroWord,
}
//
impl Debug for SearchMasksCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchMasksCache")
            .field("set_with_self", &format!("{:064b}", self.set_with_self))
            .field("set_wo_self", &format!("{:064b}", self.set_wo_self))
            .field("unset_with_self", &format!("{:064b}", self.unset_with_self))
            .field("unset_wo_self", &format!("{:064b}", self.unset_wo_self))
            .finish()
    }
}

// TODO: Add tests
