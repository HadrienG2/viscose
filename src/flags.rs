//! Set of flags that can be atomically set, checked or unset

#[cfg(test)]
use proptest::prelude::*;
use std::{
    fmt::{self, Debug, Write},
    hash::{Hash, Hasher},
    iter::FusedIterator,
    sync::atomic::{AtomicUsize, Ordering},
    write,
};

/// Word::BITS but it's a Word
const WORD_BITS: usize = Word::BITS as usize;

/// Atomic flags
#[derive(Default)]
pub(crate) struct AtomicFlags {
    /// Concrete flags data
    words: Box<[AtomicWord]>,

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
            len,
        }
    }

    /// Check if the Nth flag is set with certain atomic ordering
    pub fn is_set(&self, bit_idx: usize, order: Ordering) -> bool {
        let (word, bit) = self.bit(bit_idx);
        word.load(order) & bit != 0
    }

    /// Set the Nth flag with certain atomic ordering, tell whether that flag
    /// was set beforehand
    pub fn fetch_set(&self, bit_idx: usize, order: Ordering) -> bool {
        let (word, bit) = self.bit(bit_idx);
        word.fetch_or(bit, order) & bit != 0
    }

    /// Clear the Nth flag with certain atomic ordering, tell whether that flag
    /// was set beforehand
    pub fn fetch_clear(&self, bit_idx: usize, order: Ordering) -> bool {
        let (word, bit) = self.bit(bit_idx);
        word.fetch_and(!bit, order) & bit != 0
    }

    /// Iterate over the global bit positions of set flags at increasing
    /// distance from a certain position
    ///
    /// The underlying words are read with the specified ordering, but beware
    /// that not every bit readout requires a word readout.
    pub fn iter_set_around(
        &self,
        center_bit_idx: usize,
        order: Ordering,
    ) -> impl Iterator<Item = usize> + '_ {
        self.enumerate_bits_around(center_bit_idx, order)
            .filter_map(|(idx, bit)| bit.then_some(idx))
    }

    /// Like iter_set_around, but for unset flags
    pub fn iter_unset_around(
        &self,
        center_bit_idx: usize,
        order: Ordering,
    ) -> impl Iterator<Item = usize> + '_ {
        self.enumerate_bits_around(center_bit_idx, order)
            .filter_map(|(idx, bit)| (!bit).then_some(idx))
    }

    /// Convert a global flag index into a (word, subword bit) pair
    fn bit(&self, bit_idx: usize) -> (&AtomicWord, Word) {
        let (word_idx, bit) = self.bit_pos(bit_idx);
        (&self.words[word_idx], bit)
    }

    /// Convert a global flag index into a (word pos, subword bit) pair
    fn bit_pos(&self, bit_idx: usize) -> (usize, Word) {
        assert!(bit_idx < self.len, "requested flag is out of bounds");
        let word_idx = bit_idx / WORD_BITS;
        let bit = 1 << (bit_idx % WORD_BITS);
        (word_idx, bit)
    }

    /// Enumerate bits around a certain central position
    ///
    /// Word readout is performed with the specified ordering. Beware that not
    /// every bit readout requires a word readout.
    ///
    /// Returns an iterator over increasingly remote bits on the left side of
    /// the specified bit, and on the right side of the specified bit,
    /// respectively.
    fn enumerate_bits_around(
        &self,
        center_bit_idx: usize,
        order: Ordering,
    ) -> impl Iterator<Item = (usize, bool)> + '_ {
        // Determine the position of the central bit
        let (center_word_idx, center_bit) = self.bit_pos(center_bit_idx);

        // Read out central word
        let center_word = self.words[center_word_idx].load(order);

        // Set up bit iteration towards higher-order bits via left-shifting
        let mut left_bit_idx = center_bit_idx;
        let mut left_bit = center_bit;
        let mut left_word = center_word;
        let mut left_word_idx = center_word_idx;

        // Set up bit iteration towards lower-order bits via right-shifting
        let mut right_bit_idx = center_bit_idx;
        let mut right_bit = center_bit;
        let mut right_word = center_word;
        let mut right_word_idx = center_word_idx;

        // Emit iterator that alternates between left bits and right bits,
        // starting left unless the left iterator is exhausted from the start
        let last_bit_idx = self.len - 1;
        let mut go_left = left_bit_idx != last_bit_idx;
        std::iter::once((center_bit_idx, center_word & center_bit != 0)).chain(std::iter::from_fn(
            move || {
                // Handle case where both left and right iterators are over
                if left_bit_idx == last_bit_idx && right_bit_idx == 0 {
                    return None;
                }

                // Handle case where we're going left
                if go_left {
                    // Move to the next bit on the left
                    left_bit_idx += 1;
                    left_bit <<= 1;
                    if left_bit == 0 {
                        left_bit = 1;
                        left_word_idx += 1;
                        left_word = self.words[left_word_idx].load(order);
                    }

                    // Iterate right next time unless right-hand side is done
                    go_left = right_bit_idx == 0;

                    // Emit bit value and index
                    Some((left_bit_idx, left_word & left_bit != 0))
                } else {
                    // Move to the next bit on the right
                    right_bit_idx -= 1;
                    right_bit >>= 1;
                    if right_bit == 0 {
                        right_bit = 1 << (WORD_BITS - 1);
                        right_word_idx -= 1;
                        right_word = self.words[right_word_idx].load(order);
                    }

                    // Iterate left next time unless left-hand side is done
                    go_left = left_bit_idx != last_bit_idx;

                    // Emit bit value and index
                    Some((right_bit_idx, right_word & right_bit != 0))
                }
            },
        ))
    }

    /// Iterate over the value of inner words
    fn words(
        &self,
        order: Ordering,
    ) -> impl DoubleEndedIterator<Item = Word> + Clone + ExactSizeIterator + FusedIterator + '_
    {
        self.words.iter().map(move |word| word.load(order))
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
        let len = self.len;
        Self { words, len }
    }
}
//
impl Debug for AtomicFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut display = "AtomicFlags(".to_owned();
        display.reserve(self.len + 1);
        let mut words = self.words(Ordering::Relaxed).rev();

        let num_heading_bits = self.len % WORD_BITS;
        if num_heading_bits != 0 {
            if let Some(partial_word) = words.next() {
                for bit_idx in (0..num_heading_bits).rev() {
                    let bit = 1 << bit_idx;
                    if partial_word & bit == 0 {
                        display.push('0');
                    } else {
                        display.push('1');
                    }
                }
            }
        }

        for full_word in words {
            write!(display, "{full_word:b}")?;
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
mod tests {
    use super::*;
    use proptest::sample::SizeRange;
    use std::{collections::hash_map::RandomState, hash::BuildHasher};

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

    /// Generate a set of >= 1 atomic flags and a vald index within it
    fn flags_and_bit_idx() -> impl Strategy<Value = (AtomicFlags, usize)> {
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
        fn op_idx((flags, bit_idx) in flags_and_bit_idx()) {
            let initial_flags = flags.clone();

            let (word_idx, bit) = flags.bit_pos(bit_idx);
            prop_assert_eq!(word_idx, bit_idx / WORD_BITS);
            prop_assert_eq!(bit, 1 << (bit_idx % WORD_BITS));
            prop_assert_eq!(&flags, &initial_flags);

            let (word, bit2) = flags.bit(bit_idx);
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

            let mut left_idx = bit_idx;
            let mut right_idx = bit_idx;
            let mut go_left = left_idx != flags.len - 1;
            let mut iter = flags.enumerate_bits_around(bit_idx, Ordering::Relaxed);
            assert_eq!(iter.next(), Some((bit_idx, is_set)));
            for (bit_idx, is_set) in iter {
                if go_left {
                    left_idx += 1;
                    prop_assert_eq!(bit_idx, left_idx);
                    prop_assert_eq!(is_set, flags.is_set(left_idx, Ordering::Relaxed));
                    go_left = right_idx == 0;
                } else {
                    right_idx -= 1;
                    prop_assert_eq!(bit_idx, right_idx);
                    prop_assert_eq!(is_set, flags.is_set(right_idx, Ordering::Relaxed));
                    go_left = left_idx != flags.len - 1;
                }
                prop_assert_eq!(&flags, &initial_flags);
            }

            prop_assert!(
                flags
                    .iter_set_around(bit_idx, Ordering::Relaxed)
                    .inspect(|_| assert_eq!(flags, initial_flags))
                    .eq(
                        flags.enumerate_bits_around(bit_idx, Ordering::Relaxed)
                            .filter_map(|(bit_idx, is_set)| is_set.then_some(bit_idx))
                    )
            );
            prop_assert!(
                flags
                    .iter_unset_around(bit_idx, Ordering::Relaxed)
                    .inspect(|_| assert_eq!(flags, initial_flags))
                    .eq(
                        flags.enumerate_bits_around(bit_idx, Ordering::Relaxed)
                            .filter_map(|(bit_idx, is_set)| (!is_set).then_some(bit_idx))
                    )
            );
        }
    }
}
