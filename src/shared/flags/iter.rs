//! [`AtomicFlags`] iterator

use crate::shared::distances::Distance;

use super::{bitref::BitRef, AtomicFlags, Word, WORD_BITS};
use std::{
    debug_assert,
    fmt::{self, Debug},
    iter::FusedIterator,
    sync::atomic::Ordering,
};

/// Iterator over set/unset bits at increasing distances from a central point
#[derive(Debug, Clone)]
pub(crate) struct NearestBitIterator<'flags, const FIND_SET: bool, const INCLUDE_CENTER: bool> {
    /// Flags that we're iterating over
    flags: &'flags AtomicFlags,

    /// Iterator going towards higher-order bits via left shifts
    left_bits: BitIterator<'flags, FIND_SET, true>,

    /// Reference to the center bit
    center: BitRef<'flags, false>,

    /// Distances from the center bit to other bits
    distances_from_center: &'flags [Distance],

    /// Truth that center must be yielded and hasn't been yielded yet
    yield_center: bool,

    /// Iterator going towards lower-order bits via right shifts
    right_bits: BitIterator<'flags, FIND_SET, false>,
}
//
impl<'flags, const FIND_SET: bool, const INCLUDE_CENTER: bool>
    NearestBitIterator<'flags, FIND_SET, INCLUDE_CENTER>
{
    /// Start iterating over set/uset bits around a central position
    #[inline(always)]
    pub(crate) fn new<const CACHE_SEARCH_MASKS: bool>(
        flags: &'flags AtomicFlags,
        center: &BitRef<'flags, CACHE_SEARCH_MASKS>,
        distances_from_center: &'flags [Distance],
        order: Ordering,
    ) -> Option<Self> {
        let initial_state = InitialState::new::<FIND_SET, INCLUDE_CENTER, CACHE_SEARCH_MASKS>(
            flags, center, order,
        )?;
        let left_bits = BitIterator::<'flags, FIND_SET, true>::from_initial_state(initial_state);
        let right_bits = BitIterator::<'flags, FIND_SET, false>::from_initial_state(initial_state);
        let yield_center = INCLUDE_CENTER && (center.is_set(order) == FIND_SET);
        Some(Self {
            flags,
            left_bits,
            center: center.without_cache(),
            distances_from_center,
            yield_center,
            right_bits,
        })
    }
}
//
impl<'flags, const FIND_SET: bool, const INCLUDE_CENTER: bool> Iterator
    for NearestBitIterator<'flags, FIND_SET, INCLUDE_CENTER>
{
    type Item = BitRef<'flags, false>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // Yield the central index first
        if INCLUDE_CENTER && self.yield_center {
            self.yield_center = false;
            return Some(self.center.clone());
        }

        // Otherwise, yield the closest of the next left and right bits
        match (self.left_bits.peek(), self.right_bits.peek()) {
            (Some(left), Some(right)) => {
                let flags = self.flags;
                if self.distances_from_center[right.linear_idx(flags)]
                    <= self.distances_from_center[left.linear_idx(flags)]
                {
                    self.right_bits.next()
                } else {
                    self.left_bits.next()
                }
            }
            (Some(_), None) => self.left_bits.next(),
            (None, Some(_)) => self.right_bits.next(),
            (None, None) => None,
        }
    }
}
//
impl<const FIND_SET: bool, const INCLUDE_CENTER: bool> FusedIterator
    for NearestBitIterator<'_, FIND_SET, INCLUDE_CENTER>
{
}

/// Iterator over set/unset bits in a specific direction
///
/// - `FIND_SET` tells whether we're looking at bits that are set or unset
/// - `GOING_LEFT` tells whether we're going towards high-order bits (left side)
///   or towards low-order bits (right side)
#[derive(Clone)]
struct BitIterator<'flags, const FIND_SET: bool, const GOING_LEFT: bool> {
    /// Shared iteration state
    shared: SharedState<'flags>,

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
    BitIterator<'flags, FIND_SET, GOING_LEFT>
{
    /// Start iteration
    ///
    /// The bit at initial position `after` will not be emitted.
    #[cfg(test)]
    pub(crate) fn new<const CACHE_SEARCH_MASKS: bool>(
        flags: &'flags AtomicFlags,
        after: &BitRef<'flags, CACHE_SEARCH_MASKS>,
        order: Ordering,
    ) -> Option<Self> {
        let initial =
            InitialState::new::<FIND_SET, false, CACHE_SEARCH_MASKS>(flags, after, order)?;
        Some(Self::from_initial_state(initial))
    }

    /// Start iteration from a prepared initial state
    #[inline]
    fn from_initial_state(initial: InitialState<'flags>) -> Self {
        let InitialState {
            shared,
            word_idx,
            bit_shift,
            normalized_word,
        } = initial;
        let mut result = Self {
            shared,
            word_idx,
            bit_shift,
            normalized_remainder: Self::shift_word(normalized_word, bit_shift),
        };
        result.find_next_bit();
        result
    }

    /// Peek next iterator element without advancing the iterator
    pub fn peek(&self) -> Option<BitRef<'flags, false>> {
        let linear_idx = self.word_idx * WORD_BITS + self.bit_shift as usize;
        (linear_idx < self.shared.flags.len).then(|| self.shared.flags.bit(linear_idx))
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
    fn seek_in_word(&mut self, shift: u32) -> Option<()> {
        (shift < self.remaining_bits_in_word()).then(|| self.seek_in_word_unchecked(shift))
    }

    /// Move forward by N bits with the current word, assuming there are enough
    /// bits remaining for this operation to make sense
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
    fn find_next_word(&mut self) -> Option<()> {
        // Find the first word featuring a set/unset bit, if any
        let words = self.shared.flags.words(self.shared.order).enumerate();
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
            self.bit_shift = Self::first_bit_shift();
            self.normalized_remainder = Self::normalize_word(word, self.bit_shift);
            Some(())
        } else {
            // We didn't find a word, record that we reached end of iteration by
            // putting the iterator in a past-the-end state
            self.word_idx = self.shared.flags.words.len();
            self.bit_shift = Word::BITS;
            self.normalized_remainder = 0;
            None
        }
    }

    /// Convert a word from AtomicFlags to normalized_remainder format
    fn normalize_word(mut word: Word, bit_shift: u32) -> Word {
        // Normalize word for iteration over set bits
        if !FIND_SET {
            word = !word;
        }

        // Shift word in such a way that the bit of interest is the first bit in
        // the direction of interest (MSB or LSB)
        Self::shift_word(word, bit_shift)
    }

    /// Shift word in such a way that the bit of interest is the first bit in
    /// the direction of interest (MSB or LSB)
    fn shift_word(word: Word, bit_shift: u32) -> Word {
        debug_assert!(bit_shift < Word::BITS);
        if GOING_LEFT {
            word >> Self::first_to_bit(bit_shift)
        } else {
            word << Self::first_to_bit(bit_shift)
        }
    }

    /// First bit in a word, in our direction of iteration
    fn first_bit_shift() -> u32 {
        if GOING_LEFT {
            0
        } else {
            Word::BITS - 1
        }
    }

    /// Number of bits to skip from first bit to a certain bit shift, in our
    /// direction of iteration
    fn first_to_bit(bit_shift: u32) -> u32 {
        if GOING_LEFT {
            bit_shift
        } else {
            Word::BITS - 1 - bit_shift
        }
    }

    /// Number of bits from the original word that haven't been processed yet
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
impl<const FIND_SET: bool, const GOING_LEFT: bool> Debug for BitIterator<'_, FIND_SET, GOING_LEFT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitIterator")
            .field("shared", &self.shared)
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
impl<'flags, const FIND_SET: bool, const GOING_LEFT: bool> Iterator
    for BitIterator<'flags, FIND_SET, GOING_LEFT>
{
    type Item = BitRef<'flags, false>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let bit_idx = self.peek()?;
        self.find_next_bit();
        Some(bit_idx)
    }
}
//
impl<const FIND_SET: bool, const GOING_LEFT: bool> FusedIterator
    for BitIterator<'_, FIND_SET, GOING_LEFT>
{
}

/// State shared by all iterators
///
/// Currently, each iterator stores a copy of this because truly sharing the
/// state would require creating a self-referential iterator struct. But it is
/// hoped that given enough inlining, the compiler will manage to optimize this
/// duplication  out.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct SharedState<'flags> {
    /// Flags that we're iterating over
    flags: &'flags AtomicFlags,

    /// Atomic memory ordering that we're using for readouts
    order: Ordering,
}

/// Direction-independent part of the iterator initialization work
///
/// Used to deduplicate redundant work in the initialization of the two inner
/// iterators of [`NearestFlagIterator`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct InitialState<'flags> {
    /// Stuff that will remain shared even after we start iterating
    shared: SharedState<'flags>,

    /// Index of the center word we're initially looking at
    word_idx: usize,

    /// Position of the bit we're initially looking at within that word
    bit_shift: u32,

    /// Word that's normalized by FIND_SET, but not GOING_LEFT
    ///
    /// This subset of the word initialization work is shared for both
    /// directions of iteration.
    normalized_word: Word,
}
//
impl<'flags> InitialState<'flags> {
    /// Do the shared subset of iterator initialization work
    ///
    /// This will return None if it can be cheaply proven at this stage that no
    /// iteration is possible.
    pub fn new<const FIND_SET: bool, const INCLUDE_START: bool, const CACHE_SEARCH_MASKS: bool>(
        flags: &'flags AtomicFlags,
        start: &BitRef<'flags, CACHE_SEARCH_MASKS>,
        order: Ordering,
    ) -> Option<Self> {
        // Part that's independent of FIND_SET
        let (shared, word_idx, bit_shift, word) = Self::set_independent_init(flags, start, order);

        // If that's the only word in the flags and the bit value we're looking
        // for doesn't appear in it, we can skip iteration entirely
        if flags.words.len() == 1 {
            let mask = start.search_mask::<FIND_SET, INCLUDE_START>(flags);
            let word_empty = if FIND_SET {
                word & mask == Word::MIN
            } else {
                word | mask == Word::MAX
            };
            if word_empty {
                return None;
            }
        }

        // Otherwise, normalize by FIND_SET and return
        Some(Self {
            shared,
            word_idx,
            bit_shift,
            normalized_word: if FIND_SET { word } else { !word },
        })
    }

    /// Subset of the iteration work that's independent of FIND_SET
    fn set_independent_init<const CACHE_SEARCH_MASKS: bool>(
        flags: &'flags AtomicFlags,
        start: &BitRef<'flags, CACHE_SEARCH_MASKS>,
        order: Ordering,
    ) -> (SharedState<'flags>, usize, u32, Word) {
        let word_idx = start.word_idx(flags);
        let bit_shift = start.bit_shift();
        // SAFETY: word_idx does the bounds checking
        let word = unsafe { flags.words.get_unchecked(word_idx).load(order) };
        (SharedState { flags, order }, word_idx, bit_shift, word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::flags::tests::flags_and_bit_idx;
    use proptest::prelude::*;

    /// Check output of BitIterator
    fn check_linear<const FIND_SET: bool, const GOING_LEFT: bool>(
        flags: &AtomicFlags,
        start_idx: usize,
    ) -> Result<(), TestCaseError> {
        let start = flags.bit(start_idx);
        let iterator = BitIterator::<FIND_SET, GOING_LEFT>::new(flags, &start, Ordering::Relaxed);
        let bit_indices: Box<dyn Iterator<Item = usize>> = if GOING_LEFT {
            Box::new((start_idx..flags.len()).skip(1))
        } else {
            Box::new((0..start_idx).rev())
        };
        let check_bit = |bit_idx: usize, iterator: &mut BitIterator<FIND_SET, GOING_LEFT>| {
            let bit = flags.bit(bit_idx);
            if bit.is_set(Ordering::Relaxed) == FIND_SET {
                prop_assert_eq!(iterator.next(), Some(bit));
            }
            Ok(())
        };
        if let Some(mut iterator) = iterator {
            for bit_idx in bit_indices {
                check_bit(bit_idx, &mut iterator)?;
            }
            prop_assert_eq!(iterator.next(), None);
        } else {
            for bit_idx in bit_indices {
                let bit = flags.bit(bit_idx);
                prop_assert_ne!(bit.is_set(Ordering::Relaxed), FIND_SET);
            }
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn iterate_linear((flags, start_idx) in flags_and_bit_idx()) {
            check_linear::<false, false>(&flags, start_idx)?;
            check_linear::<false, true>(&flags, start_idx)?;
            check_linear::<true, false>(&flags, start_idx)?;
            check_linear::<true, true>(&flags, start_idx)?;
        }
    }

    /// Check output of NearestBitIterator
    fn check_nearest<const FIND_SET: bool, const INCLUDE_CENTER: bool>(
        flags: &AtomicFlags,
        center_idx: usize,
        distances_from_center: &[Distance],
    ) -> Result<(), TestCaseError> {
        let center = flags.bit(center_idx);
        let iterator = NearestBitIterator::<FIND_SET, INCLUDE_CENTER>::new(
            flags,
            &center,
            distances_from_center,
            Ordering::Relaxed,
        );
        let mut left_indices = (center_idx..flags.len()).skip(1).peekable();
        let mut right_indices = (0..center_idx).rev().peekable();
        let bit_indices = std::iter::once(center_idx)
            .skip(usize::from(!INCLUDE_CENTER))
            .chain(std::iter::from_fn(|| {
                match (left_indices.peek(), right_indices.peek()) {
                    (Some(left_idx), Some(right_idx)) => {
                        if distances_from_center[*right_idx] <= distances_from_center[*left_idx] {
                            right_indices.next()
                        } else {
                            left_indices.next()
                        }
                    }
                    (Some(_), None) => left_indices.next(),
                    (None, Some(_)) => right_indices.next(),
                    (None, None) => None,
                }
            }));
        if let Some(mut iterator) = iterator {
            for bit_idx in bit_indices {
                let bit = flags.bit(bit_idx);
                if bit.is_set(Ordering::Relaxed) == FIND_SET {
                    prop_assert_eq!(
                        iterator.next(),
                        Some(bit),
                        "upon reaching bit {} of check_nearest<FIND_SET={}, INCLUDE_CENTER={}>",
                        bit_idx,
                        FIND_SET,
                        INCLUDE_CENTER,
                    );
                }
            }
            prop_assert_eq!(iterator.next(), None);
        } else {
            for bit_idx in bit_indices {
                let bit = flags.bit(bit_idx);
                prop_assert_ne!(
                    bit.is_set(Ordering::Relaxed),
                    FIND_SET,
                    "failed for FIND_SET={} and INCLUDE_CENTER={}",
                    FIND_SET,
                    INCLUDE_CENTER
                );
            }
        }
        Ok(())
    }

    fn flags_bit_distances() -> impl Strategy<Value = (AtomicFlags, usize, Vec<Distance>)> {
        flags_and_bit_idx().prop_flat_map(|(flags, bit_idx)| {
            prop::collection::vec(any::<Distance>(), flags.len()).prop_map(
                move |mut raw_distances| {
                    raw_distances[bit_idx] = 0;
                    raw_distances[..bit_idx].sort_unstable_by_key(|&dist| -(dist as isize));
                    raw_distances[bit_idx..].sort_unstable();
                    (flags.clone(), bit_idx, raw_distances)
                },
            )
        })
    }

    proptest! {
        #[test]
        fn iterate_nearest((flags, start_idx, distances) in flags_bit_distances()) {
            check_nearest::<false, false>(&flags, start_idx, &distances)?;
            check_nearest::<false, true>(&flags, start_idx, &distances)?;
            check_nearest::<true, false>(&flags, start_idx, &distances)?;
            check_nearest::<true, true>(&flags, start_idx, &distances)?;
        }
    }
}
