# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# July 2021

# Adapted from: https://github.com/eric-wieser/numpy_ringbuffer

import numpy as np
from collections import Sequence

class RingBuffer2:
    def __init__(self, array, indices):
        """
        Create a new ring buffer with the given (empty) np.ndarray
        Parameters
        ----------
        array: and empty np.ndarray
        indices: 2-element array corresponding to left index [0] and right index [1]
        """
        self._arr = array
        self._indices = indices
        self._capacity = self._arr.shape[0]

    @staticmethod
    def unwrap(array, indices):
        """ Return two slices of array that, when combined, are array's unwrapped form.
        Parameters
        ----------
        array: array to be sliced according to indices
        indices: 2-element array corresponding to left index [0] and right index [1] to be used to slice array
        """
        return array[indices[0]:min(indices[1], array.shape[0])], \
               array[:max(indices[1] - array.shape[0], 0)]

    def _unwrap(self):
        """ Copy the data from this buffer into unwrapped form """
        return np.concatenate((
            self._arr[self._indices[0]:min(self._indices[1], self._capacity)],
            self._arr[:max(self._indices[1] - self._capacity, 0)]
        ))

    @staticmethod
    def fix_indices(indices, capacity):
        """
        Enforce our invariant that 0 <= self._left_index < self._capacity
        """
        with indices.get_lock():
            if indices[0] >= capacity:
                indices[0] -= capacity
                indices[1] -= capacity
            elif indices[0] < 0:
                indices[0] += capacity
                indices[1] += capacity

        return indices

    def _fix_indices(self):
        """
        Enforce our invariant that 0 <= self._left_index < self._capacity
        """
        if self._indices[0] >= self._capacity:
            self._indices[0] -= self._capacity
            self._indices[1] -= self._capacity
        elif self._indices[0] < 0:
            self._indices[0] += self._capacity
            self._indices[1] += self._capacity

    @staticmethod
    def is_full(array, indices):
        return RingBuffer.len_stat(indices) == array.shape[0]

    def _is_full(self):
        """ True if there is no more space in the buffer """
        return self.len() == self._capacity

    @staticmethod
    def len(indices):
        with indices.get_lock():
            return indices[1] - indices[0]

    def _len(self):
        return self._indices[1] - self._indices[0]

    @staticmethod
    def append(array, indices, value):
        with indices.get_lock():
            if RingBuffer.is_full(array, indices):
                indices[0] += 1

            array[indices[1] % array.shape[0]] = value
            indices[1] += 1
            RingBuffer.fix_indices(indices, array.shape[0])

    def _append(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._left_index += 1

        self._arr[self._right_index % self._capacity] = value
        self._right_index += 1
        self._fix_indices()

    # LMD implemented:
    @staticmethod
    def peek(array, indices):
        if RingBuffer.len(array) == 0:
            raise IndexError("Peek from an empty RingBuffer.")
        return array[(indices[1] % array.shape[0])]

    def peek(self):
        if self._len() == 0:
            raise IndexError("Peek from an empty RingBuffer.")
        return self._arr[(self._indices[1] - 1) % self._capacity]
