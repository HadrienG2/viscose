//! Set of flags that can be atomically set, checked or unset

use std::sync::atomic::{AtomicUsize, Ordering};

/// Word::BITS but it's a Word
const WORD_BITS: usize = Word::BITS as usize;

/// Atomic flags
pub struct AtomicFlags {
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
        center: usize,
        order: Ordering,
    ) -> impl Iterator<Item = usize> + '_ {
        self.enumerate_bits_around(center, order)
            .filter_map(|(idx, bit)| bit.then_some(idx))
    }

    /// Like iter_set_around, but for unset flags
    pub fn iter_unset_around(
        &self,
        center: usize,
        order: Ordering,
    ) -> impl Iterator<Item = usize> + '_ {
        self.enumerate_bits_around(center, order)
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
        std::iter::from_fn(move || {
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
                go_left = right_bit_idx != 0;

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
        })
    }
}

/// Word of bits
type Word = usize;

/// Atomic version of Word
type AtomicWord = AtomicUsize;

// TODO: Add proptests
