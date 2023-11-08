//! Set of flags that can be atomically set, checked or unset

mod iter;

#[cfg(test)]
use proptest::prelude::*;
use std::{
    fmt::{self, Debug, Write},
    hash::{Hash, Hasher},
    iter::FusedIterator,
    sync::atomic::{self, AtomicUsize, Ordering},
    write,
};

/// Word::BITS but it's a Word
const WORD_BITS: usize = Word::BITS as usize;

/// Atomic flags
#[derive(Default)]
pub struct AtomicFlags {
    /// Concrete flags data
    words: Box<[AtomicWord]>,

    /// Mask of the significant bits in the last word of the flags
    end_bits_mask: Word,

    /// Number of actually meaningful bits
    len: usize,
}
//
impl AtomicFlags {
    /// Create a set of N flags, initially all unset
    pub fn new(len: usize) -> Self {
        Self {
            words: std::iter::repeat_with(|| AtomicWord::new(0))
                .take(len.div_ceil(WORD_BITS))
                .collect(),
            end_bits_mask: Self::end_bits_mask(len % WORD_BITS),
            len,
        }
    }

    /// end_bits_mask for a given number of trailing bits
    fn end_bits_mask(trailing_bits: usize) -> Word {
        if trailing_bits == 0 {
            Word::MAX
        } else {
            (1 << trailing_bits) - 1
        }
    }

    /// Check how many flags there are in there
    pub fn len(&self) -> usize {
        self.len
    }

    /// Truth that there is at least one flag in there
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if the Nth flag is set with certain atomic ordering
    pub fn is_set(&self, bit_idx: usize, order: Ordering) -> bool {
        let (word, bit) = self.word_and_bit(bit_idx);
        word.load(order) & bit != 0
    }

    /// Set the Nth flag with certain atomic ordering, tell whether that flag
    /// was set beforehand
    pub fn fetch_set(&self, bit_idx: usize, order: Ordering) -> bool {
        let (word, bit) = self.word_and_bit(bit_idx);
        word.fetch_or(bit, order) & bit != 0
    }

    /// Clear the Nth flag with certain atomic ordering, tell whether that flag
    /// was set beforehand
    pub fn fetch_clear(&self, bit_idx: usize, order: Ordering) -> bool {
        let (word, bit) = self.word_and_bit(bit_idx);
        word.fetch_and(!bit, order) & bit != 0
    }

    /// Set all the flags
    pub fn set_all(&self, order: Ordering) {
        if order != Ordering::Relaxed {
            atomic::fence(order);
        }
        self.words
            .iter()
            .for_each(|word| word.store(Word::MAX, Ordering::Relaxed));
    }

    /// Clear all the flags
    pub fn clear_all(&self, order: Ordering) {
        if order != Ordering::Relaxed {
            atomic::fence(order);
        }
        self.words
            .iter()
            .for_each(|word| word.store(0, Ordering::Relaxed));
    }

    /// Iterate over the global bit positions of set flags at increasing
    /// distance from a certain position
    ///
    /// The underlying words are read with the specified ordering, but beware
    /// that not every bit readout requires a word readout.
    ///
    /// Returns None if it can be proven early that iteration will yield no bit
    pub fn iter_set_around<const INCLUDE_CENTER: bool>(
        &self,
        center_bit_idx: usize,
        order: Ordering,
    ) -> Option<impl Iterator<Item = usize> + '_> {
        iter::NearestFlagIterator::<true, INCLUDE_CENTER>::new(self, center_bit_idx, order)
    }

    /// Like iter_set_around, but looks for unset flags instead of set flags
    pub fn iter_unset_around<const INCLUDE_CENTER: bool>(
        &self,
        center_bit_idx: usize,
        order: Ordering,
    ) -> Option<impl Iterator<Item = usize> + '_> {
        iter::NearestFlagIterator::<false, INCLUDE_CENTER>::new(self, center_bit_idx, order)
    }

    /// Convert a global flag index into a (word, subword bit) pair
    fn word_and_bit(&self, bit_idx: usize) -> (&AtomicWord, Word) {
        let (word_idx, bit) = self.word_idx_and_bit(bit_idx);
        (&self.words[word_idx], bit)
    }

    /// Convert a global flag index into a (word idx, subword bit) pair
    fn word_idx_and_bit(&self, bit_idx: usize) -> (usize, Word) {
        let (word_idx, bit_shift) = self.word_idx_and_bit_shift(bit_idx);
        (word_idx, 1 << bit_shift)
    }

    /// Convert a global flag index into a (word idx, bit shift) pair
    fn word_idx_and_bit_shift(&self, bit_idx: usize) -> (usize, u32) {
        assert!(bit_idx < self.len, "requested flag is out of bounds");
        let word_idx = bit_idx / WORD_BITS;
        let bit_shift = (bit_idx % WORD_BITS) as u32;
        (word_idx, bit_shift)
    }

    /// Iterate over the value of inner words
    fn words(
        &self,
        order: Ordering,
    ) -> impl DoubleEndedIterator<Item = Word> + Clone + ExactSizeIterator + FusedIterator + '_
    {
        self.words.iter().map(move |word| word.load(order))
    }

    /// Mask of significant bits in a given word of the flags
    fn bits_mask(&self, word_idx: usize) -> Word {
        if word_idx == self.words.len() - 1 {
            self.end_bits_mask
        } else {
            Word::MAX
        }
    }
}
//
#[cfg(test)]
impl Arbitrary for AtomicFlags {
    type Parameters = <Vec<bool> as Arbitrary>::Parameters;
    type Strategy = prop::strategy::Map<<Vec<bool> as Arbitrary>::Strategy, fn(Vec<bool>) -> Self>;

    fn arbitrary_with(parameters: Self::Parameters) -> Self::Strategy {
        Vec::<bool>::arbitrary_with(parameters).prop_map(|bits| {
            // Set up word storage
            let len = bits.len();
            let mut words = Vec::with_capacity(len.div_ceil(WORD_BITS));

            // Construct full words from input bits
            let mut current_word = 0 as Word;
            let mut current_bit = 0;
            for bit in bits {
                current_word = (current_word << 1) | (bit as Word);
                current_bit += 1;
                if current_bit == WORD_BITS {
                    words.push(AtomicWord::new(current_word));
                    current_word = 0;
                    current_bit = 0;
                }
            }

            // Push trailing word, if any
            if current_bit != 0 {
                words.push(AtomicWord::new(current_word));
            }

            // Emit final flags
            Self {
                words: words.into_boxed_slice(),
                end_bits_mask: AtomicFlags::end_bits_mask(current_bit),
                len,
            }
        })
    }
}
//
impl Clone for AtomicFlags {
    fn clone(&self) -> Self {
        let words = self
            .words(Ordering::Relaxed)
            .map(AtomicUsize::new)
            .collect();
        Self {
            words,
            end_bits_mask: self.end_bits_mask,
            len: self.len,
        }
    }
}
//
impl Debug for AtomicFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut display = "AtomicFlags(".to_owned();
        display.reserve(self.len + self.len.div_ceil(WORD_BITS));
        let mut words = self.words(Ordering::Relaxed).rev();
        let mut has_word_before = false;

        let num_heading_bits = self.len % WORD_BITS;
        if num_heading_bits != 0 {
            let partial_word = words.next().unwrap();
            for bit_idx in (0..num_heading_bits).rev() {
                let bit = 1 << bit_idx;
                if partial_word & bit == 0 {
                    display.push('0');
                } else {
                    display.push('1');
                }
            }
            has_word_before = true;
        }

        for full_word in words.peekable() {
            if has_word_before {
                display.push('_');
            }
            write!(display, "{full_word:064b}")?;
            has_word_before = true;
        }

        display.push(')');
        f.pad(&display)
    }
}
//
impl Eq for AtomicFlags {}
//
impl Hash for AtomicFlags {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for word in self.words(Ordering::Relaxed) {
            word.hash(state)
        }
        self.len.hash(state);
    }
}
//
impl PartialEq for AtomicFlags {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self
                .words(Ordering::Relaxed)
                .eq(other.words(Ordering::Relaxed))
    }
}

/// Word of bits
type Word = usize;

/// Atomic version of Word
type AtomicWord = AtomicUsize;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::sample::SizeRange;
    use std::{collections::hash_map::RandomState, hash::BuildHasher};

    /// Generate a set of >= 1 atomic flags and a vald index within it
    pub(crate) fn flags_and_bit_idx() -> impl Strategy<Value = (AtomicFlags, usize)> {
        any::<AtomicFlags>()
            .prop_filter(
                "need at least one element to have at least one index",
                |flags| flags.len > 0,
            )
            .prop_flat_map(|flags| {
                let bit_idx = 0..flags.len;
                (Just(flags), bit_idx)
            })
    }

    proptest! {
        #[test]
        fn new(len in 0..SizeRange::default().end_excl()) {
            let flags = AtomicFlags::new(len);
            prop_assert_eq!(flags.len, len);
            prop_assert_eq!(flags.words.len(), len.div_ceil(WORD_BITS));
            prop_assert!(flags.words(Ordering::Relaxed).all(|word| word == 0));
        }

        #[test]
        fn unary(flags: AtomicFlags) {
            let hasher = RandomState::new();
            let clone = flags.clone();
            prop_assert_eq!(&clone, &flags);
            prop_assert_eq!(hasher.hash_one(&clone), hasher.hash_one(&flags));
            prop_assert!(flags.words(Ordering::Relaxed).eq(flags.words.iter().map(|word| word.load(Ordering::Relaxed))));
        }
    }

    proptest! {
        #[test]
        fn op_idx((flags, bit_idx) in flags_and_bit_idx()) {
            let initial_flags = flags.clone();

            let (word_idx, bit) = flags.word_idx_and_bit(bit_idx);
            prop_assert_eq!(word_idx, bit_idx / WORD_BITS);
            prop_assert_eq!(bit, 1 << (bit_idx % WORD_BITS));
            prop_assert_eq!(&flags, &initial_flags);

            let (word, bit2) = flags.word_and_bit(bit_idx);
            prop_assert!(std::ptr::eq(word, &flags.words[word_idx]));
            prop_assert_eq!(bit, bit2);
            prop_assert_eq!(&flags, &initial_flags);

            let is_set = flags.is_set(bit_idx, Ordering::Relaxed);
            prop_assert_eq!(
                is_set,
                word.load(Ordering::Relaxed) & bit != 0
            );
            prop_assert_eq!(&flags, &initial_flags);

            if is_set {
                let was_set = flags.fetch_set(bit_idx, Ordering::Relaxed);
                prop_assert!(was_set);
                prop_assert_eq!(&flags, &initial_flags);

                let was_set = flags.fetch_clear(bit_idx, Ordering::Relaxed);
                prop_assert!(was_set);
                word.fetch_or(bit, Ordering::Relaxed);
                prop_assert_eq!(&flags, &initial_flags);
            } else {
                let was_set = flags.fetch_clear(bit_idx, Ordering::Relaxed);
                prop_assert!(!was_set);
                prop_assert_eq!(&flags, &initial_flags);

                let was_set = flags.fetch_set(bit_idx, Ordering::Relaxed);
                prop_assert!(!was_set);
                word.fetch_and(!bit, Ordering::Relaxed);
                prop_assert_eq!(&flags, &initial_flags);
            }
        }
    }
}
