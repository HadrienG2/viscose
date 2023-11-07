//! [`AtomicFlags`] iterator

use super::{AtomicFlags, Word, WORD_BITS};
use std::{
    debug_assert,
    fmt::{self, Debug},
    iter::FusedIterator,
    sync::atomic::Ordering,
};

/// Iterator over the position of set/unset bits at increasing distances from a
/// certain central point
#[derive(Debug, Clone)]
pub(crate) struct NearestFlagIterator<'flags, const FIND_SET: bool, const INCLUDE_CENTER: bool> {
    /// Iterator going towards higher-order bits via left shifts
    left_indices: FlagIdxIterator<'flags, FIND_SET, true>,

    /// Center bit position
    center_bit_idx: usize,

    /// Truth that center_bit_idx must be yielded and hasn't been yielded yet
    yield_center: bool,

    /// Iterator going towards lower-order bits via right shifts
    right_indices: FlagIdxIterator<'flags, FIND_SET, false>,
}
//
impl<'flags, const FIND_SET: bool, const INCLUDE_CENTER: bool>
    NearestFlagIterator<'flags, FIND_SET, INCLUDE_CENTER>
{
    /// Start iterating over set/uset bits around a central position
    #[inline(always)]
    pub(crate) fn new(flags: &'flags AtomicFlags, center_bit_idx: usize, order: Ordering) -> Self {
        let left_indices =
            FlagIdxIterator::<'flags, FIND_SET, true>::new(flags, center_bit_idx, order);
        let right_indices =
            FlagIdxIterator::<'flags, FIND_SET, false>::new(flags, center_bit_idx, order);
        let yield_center = INCLUDE_CENTER && (flags.is_set(center_bit_idx, order) == FIND_SET);
        Self {
            left_indices,
            center_bit_idx: (INCLUDE_CENTER as usize) * center_bit_idx,
            yield_center,
            right_indices,
        }
    }
}
//
impl<'flags, const FIND_SET: bool, const INCLUDE_CENTER: bool> Iterator
    for NearestFlagIterator<'flags, FIND_SET, INCLUDE_CENTER>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // Yield the central index first
        if INCLUDE_CENTER && self.yield_center {
            self.yield_center = false;
            return Some(self.center_bit_idx);
        }

        // Otherwise, yield the closest of the next left and right bits
        match (self.left_indices.peek(), self.right_indices.peek()) {
            (Some(left_idx), Some(right_idx)) => {
                if left_idx - self.center_bit_idx <= self.center_bit_idx - right_idx {
                    self.left_indices.next()
                } else {
                    self.right_indices.next()
                }
            }
            (Some(_), None) => self.left_indices.next(),
            (None, Some(_)) => self.right_indices.next(),
            (None, None) => None,
        }
    }
}
//
impl<const FIND_SET: bool, const INCLUDE_CENTER: bool> FusedIterator
    for NearestFlagIterator<'_, FIND_SET, INCLUDE_CENTER>
{
}

/// Iterator over the position of set/unset bits in a specific direction
///
/// - `FIND_SET` tells whether we're looking at bits that are set or unset
/// - `GOING_LEFT` tells whether we're going towards high-order bits (left side)
///   or towards low-order bits (right side)
#[derive(Clone)]
struct FlagIdxIterator<'flags, const FIND_SET: bool, const GOING_LEFT: bool> {
    /// Flags that we're iterating over
    flags: &'flags AtomicFlags,

    /// Atomic memory ordering that we're using for readouts
    order: Ordering,

    /// Index of the word we're currently looking at
    word_idx: usize,

    /// Position of the bit we're looking at within that word
    bit_shift: u32,

    /// Normalized remainder of the word that we are processing
    ///
    /// The original word is negated and shifted in such a way that the problem
    /// we're solving is enumerating the position of set bits inside of the
    /// word, in the direction of interest (from LSB to MSB if GOING_LEFT,
    /// otherwise from MSB to LSB).
    ///
    /// Bits from the original word which are not associated with this flags are
    /// set to zero.
    normalized_remainder: Word,
}
//
impl<'flags, const FIND_SET: bool, const GOING_LEFT: bool>
    FlagIdxIterator<'flags, FIND_SET, GOING_LEFT>
{
    /// Start iteration
    ///
    /// The bit at initial position `after_bit_idx` will not be emitted.
    #[inline(always)]
    pub fn new(flags: &'flags AtomicFlags, after_bit_idx: usize, order: Ordering) -> Self {
        let (word_idx, bit_shift) = flags.word_idx_and_bit_shift(after_bit_idx);
        let word = flags.words[word_idx].load(order);
        let normalized_remainder = Self::normalize_word(word, bit_shift);
        let mut result = Self {
            flags,
            order,
            word_idx,
            bit_shift,
            normalized_remainder,
        };
        result.find_next_bit();
        result
    }

    /// Peek next iterator element without advancing the iterator
    pub fn peek(&self) -> Option<usize> {
        let bit_idx = self.word_idx * WORD_BITS + self.bit_shift as usize;
        (bit_idx < self.flags.len).then_some(bit_idx)
    }

    /// Go to the next occurence of the bit value of interest in the flags, or
    /// to the end of iteration.
    #[inline(always)]
    fn find_next_bit(&mut self) -> Option<()> {
        self.seek_in_word(1)
            .and_then(|()| self.find_bit_in_word())
            .or_else(|| {
                self.find_next_word()?;
                self.find_bit_in_word()
            })
    }

    /// Seek active word to the bit value of interest or return None
    fn find_bit_in_word(&mut self) -> Option<()> {
        let shift = if GOING_LEFT {
            self.normalized_remainder.trailing_zeros()
        } else {
            self.normalized_remainder.leading_zeros()
        };
        debug_assert!(shift < self.remaining_bits_in_word() || shift == Word::BITS);
        (shift != Word::BITS).then(|| self.seek_in_word_unchecked(shift))
    }

    /// Move forward by N bits within the current word if there are that enough
    /// bits remaining, otherwise return None
    #[inline]
    fn seek_in_word(&mut self, shift: u32) -> Option<()> {
        (shift < self.remaining_bits_in_word()).then(|| self.seek_in_word_unchecked(shift))
    }

    /// Move forward by N bits with the current word, assuming there are enough
    /// bits remaining for this operation to make sense
    #[inline]
    fn seek_in_word_unchecked(&mut self, shift: u32) {
        debug_assert!(shift < self.remaining_bits_in_word());
        if GOING_LEFT {
            self.normalized_remainder >>= shift;
            self.bit_shift += shift;
        } else {
            self.normalized_remainder <<= shift;
            self.bit_shift -= shift;
        }
    }

    /// Go to the next word that features the bit value of interest, if any,
    /// else record that we're at the end of iteration and return None.
    #[inline]
    fn find_next_word(&mut self) -> Option<()> {
        // Find the first word featuring a set/unset bit, if any
        let words = self.flags.words(self.order).enumerate();
        let not_empty = |(_idx, word): &(usize, Word)| {
            if FIND_SET {
                *word != 0
            } else {
                *word != Word::MAX
            }
        };
        let find_result = if GOING_LEFT {
            words.skip(self.word_idx + 1).find(not_empty)
        } else {
            words.take(self.word_idx).rfind(not_empty)
        };

        // If successful, update the state accordingly
        if let Some((word_idx, word)) = find_result {
            // Determine the first bit to be probed within this word
            self.word_idx = word_idx;
            self.bit_shift = if GOING_LEFT { 0 } else { Word::BITS - 1 };
            self.normalized_remainder = Self::normalize_word(word, self.bit_shift);
            Some(())
        } else {
            // We didn't find a word, record that we reached end of iteration by
            // putting the iterator in a past-the-end state
            self.word_idx = self.flags.words.len();
            self.bit_shift = Word::BITS;
            self.normalized_remainder = 0;
            None
        }
    }

    /// Convert a word from AtomicFlags to normalized_remainder format
    #[inline]
    fn normalize_word(mut word: Word, bit_shift: u32) -> Word {
        // Normalize word for iteration over set bits
        if !FIND_SET {
            word = !word;
        }

        // Shift word in such a way that the bit of interest is the first bit in
        // the direction of interest (MSB or LSB)
        debug_assert!(bit_shift < Word::BITS);
        if GOING_LEFT {
            word >>= bit_shift;
        } else {
            word <<= Word::BITS - 1 - bit_shift;
        };
        word
    }

    /// Number of bits from the original word that haven't been processed yet
    #[inline]
    fn remaining_bits_in_word(&self) -> u32 {
        debug_assert!(self.bit_shift < Word::BITS);
        if GOING_LEFT {
            Word::BITS - self.bit_shift
        } else {
            self.bit_shift + 1
        }
    }
}
//
impl<const FIND_SET: bool, const GOING_LEFT: bool> Debug
    for FlagIdxIterator<'_, FIND_SET, GOING_LEFT>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlagIdxIterator")
            .field("flags", &self.flags)
            .field("order", &self.order)
            .field("word_idx", &self.word_idx)
            .field("bit_shift", &self.bit_shift)
            .field(
                "normalized_remainder",
                &format!("{:064b}", self.normalized_remainder),
            )
            .finish()
    }
}
//
impl<const FIND_SET: bool, const GOING_LEFT: bool> Iterator
    for FlagIdxIterator<'_, FIND_SET, GOING_LEFT>
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let bit_idx = self.peek()?;
        self.find_next_bit();
        Some(bit_idx)
    }
}
//
impl<const FIND_SET: bool, const GOING_LEFT: bool> FusedIterator
    for FlagIdxIterator<'_, FIND_SET, GOING_LEFT>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flags::tests::flags_and_bit_idx;
    use proptest::prelude::*;

    /// Check outcome of iterating over FlagIdxIterator
    fn check_iterate<const FIND_SET: bool, const GOING_LEFT: bool>(
        flags: &AtomicFlags,
        start_idx: usize,
    ) -> Result<(), TestCaseError> {
        let mut iterator =
            FlagIdxIterator::<FIND_SET, GOING_LEFT>::new(flags, start_idx, Ordering::Relaxed);
        let process_bit = |bit_idx: usize, iterator: &mut FlagIdxIterator<FIND_SET, GOING_LEFT>| {
            if flags.is_set(bit_idx, Ordering::Relaxed) == FIND_SET {
                prop_assert_eq!(iterator.next(), Some(bit_idx));
            }
            Ok(())
        };
        if GOING_LEFT {
            for bit_idx in (start_idx..flags.len()).skip(1) {
                process_bit(bit_idx, &mut iterator)?;
            }
        } else {
            for bit_idx in (0..start_idx).rev() {
                process_bit(bit_idx, &mut iterator)?;
            }
        }
        prop_assert_eq!(iterator.next(), None);
        Ok(())
    }

    /// Check outcome of iterating over NearestFlagIterator
    fn check_nearest<const FIND_SET: bool>(
        flags: &AtomicFlags,
        center_idx: usize,
    ) -> Result<(), TestCaseError> {
        let mut iterator =
            NearestFlagIterator::<FIND_SET>::new(flags, center_idx, Ordering::Relaxed);
        let left_indices = (center_idx..flags.len()).skip(1);
        let right_indices = (0..center_idx).rev();
        let indices =
            std::iter::once(center_idx).chain(itertools::interleave(left_indices, right_indices));
        for idx in indices {
            if flags.is_set(idx, Ordering::Relaxed) == FIND_SET {
                prop_assert_eq!(iterator.next(), Some(idx));
            }
        }
        prop_assert_eq!(iterator.next(), None);
        Ok(())
    }

    proptest! {
        #[test]
        fn iterate((flags, start_idx) in flags_and_bit_idx()) {
            check_iterate::<false, false>(&flags, start_idx)?;
            check_iterate::<false, true>(&flags, start_idx)?;
            check_iterate::<true, false>(&flags, start_idx)?;
            check_iterate::<true, true>(&flags, start_idx)?;

            check_nearest::<false>(&flags, start_idx)?;
            check_nearest::<true>(&flags, start_idx)?;
        }
    }

    // TODO: Moar tests
}
