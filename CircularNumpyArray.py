# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# July 2021

# Description: Thread-safe implementation of a circular numpy array.
# Based on:
# https://github.com/eric-wieser/numpy_ringbuffer/blob/master/numpy_ringbuffer/__init__.py
# https://github.com/Dennis-van-Gils/python-dvg-ringbuffer/blob/master/src/dvg_ringbuffer.py

from collections.abc import Sequence
import numpy as np
import threading


class CircularNumpyArray(Sequence):
    def __init__(self, capacity, max_num_grid_cells):
        print("__init__")
        self._lock = threading.Lock()

        #self._array = np.full(shape=capacity, dtype=np.ndarray, fill_value=np.nan)
        self._array = np.full(shape=(capacity, max_num_grid_cells), dtype=(float, max_num_grid_cells), fill_value=np.nan)
        self._unwrap_buffer = np.full(shape=capacity, dtype=np.ndarray, fill_value=np.nan)  # At fixed memory address

        self._capacity = capacity
        self._head = 0
        self._tail = 0
        self._unwrap_buffer_is_dirty = False

    def _fix_indices(self):
        """ Enforce invariant that 0 <= self._tail < self._capacity. """
        print("_fix_indices")
        if self._tail >= self._capacity:
            self._tail -= self._capacity
            self._head -= self._capacity
        elif self._tail < 0:
            self._tail += self._capacity
            self._head += self._capacity

    @property
    def is_full(self):
        """ True if there is no more space in the buffer."""
        print("is_full")
        return len(self) == self._capacity

    # Numpy compatibility
    def __array__(self):
        print("__array__")
        return self._unwrap()

    @property
    def dtype(self):
        print("dtype")
        return self._array.dtype

    @property
    def shape(self):
        print("shape")
        return (len(self),) + self._array.shape[1:]

    # These mirror methods from deque
    def maxlen(self):
        print("maxlen")
        return self._capacity

    def append(self, value):
        print("append")
        self._lock.acquire()

        if self.is_full:  # if (self._head - self._tail) == self._capacity
            if not len(self):  # if (self._head - self._tail) == 0
                return  # Mimic behavior of deque(maxlen=0)
            else:
                self._tail += 1

        # Insert value at head's position in buffer:
        self._array[self._head % self._capacity] = value
        # Advance head's position:
        self._head += 1
        self._fix_indices()

        self._lock.release()

    def peek(self):
        """ Return result at head position in buffer without removing it from buffer. """
        print("peek")
        self._lock.acquire()

        if not len(self):  # if (self._head - self._tail) == 0
            raise IndexError("Pop from empty buffer.")
        # else:
        result = self._array[(self._head - 1) % self._capacity]

        self._lock.release()

        return result

    def pop(self):
        """ Return result at head position in buffer and remove it from buffer. """
        print("pop")
        self._lock.acquire()

        if not len(self):  # if (self._head - self._tail) == 0
            raise IndexError("Pop from empty buffer.")
        # else:
        self._head -= 1
        self._fix_indices()  # I don't think this is necessary as _fix_indices only checks value of self._tail
        result = self._array[self._head % self._capacity]

        self._lock.release()

        return result

    def _unwrap(self):
        """ Copy the data from this buffer into unwrapped form. """
        print("_unwrap")
        return np.concatenate(
            (
                self._array[self._tail : min(self._head, self._capacity)],
                self._array[: max((self._head - self._capacity), 0)]
            )
        )

    def _unwrap_into_buffer(self):
        """ Copy the data from this buffer into unwrapped form to the unwrap
        buffer at a fixed memory address. Only call when the buffer is full. """
        print("_unwrap_into_buffer")
        if self._unwrap_buffer_is_dirty:
            np.concatenate(
                (
                    self._array[self._tail : min(self._head, self._capacity)],
                    self._array[: max((self._head - self._capacity), 0)]
                ),
                out=self._uwrap_buffer,
            )
            self._unwrap_buffer_is_dirty = False
        else:
            pass

    # Implement Sequence methods:
    def __len__(self):
        print("__len__")
        return self._head - self._tail

    def __getitem__(self, item):
        """ DVG RingBuffer version. """
        print("__getitem__")
        self._lock.acquire()
        # --------------------------
        #   ringbuffer[slice]
        #   ringbuffer[tuple]
        #   ringbuffer[None]
        # --------------------------
        try:
            if isinstance(item, (slice, tuple)) or item is None:
                if self.is_full:
                    self._unwrap_into_buffer()
                    result = self._unwrap_buffer[item]
                else:
                    result = self._unwrap()[item]

            # ----------------------------------
            #   ringbuffer[int]
            #   ringbuffer[list of ints]
            #   ringbuffer[np.ndarray of ints]
            # ----------------------------------
            else:
                item_arr = np.asarray(item)
                if not issubclass(item_arr.dtype.type, np.integer):
                    raise TypeError("RingBuffer indices must be integers.")
                if not len(self):  # if (self._head - self._tail) == 0
                    raise IndexError("RingBuffer list index out of range. RingBuffer has length 0.")

                if not hasattr(item, "__len__"):
                    # Single element; We can speed up the code!
                    # Check for list index out of range:
                    if item_arr < -len(self) or item_arr >= len(self):
                        raise IndexError("RingBuffer list index {} out of range. RingBuffer has length {}."
                                         .format(item_arr, len(self)))
                    elif item_arr < 0:
                        item_arr = (self._head + item_arr) % self._capacity
                    else:
                        item_arr = (self._tail + item_arr) % self._capacity

                else:
                    # Multiple elements
                    # Check for list index out of range:
                    if np.any(item_arr < -len(self)) or np.any(item_arr >= len(self)):
                        idx_under = item_arr[np.where(item_arr < -len(self))]
                        idx_over = item_arr[np.where(item_arr >= len(self))]
                        raise IndexError("RingBuffer list indices {} out of range. RingBuffer has length {}."
                                         .format(np.concatenate((idx_under, idx_over)), len(self)))
                    # else:
                    idx_neg = np.where(item_arr < 0)
                    idx_pos = np.where(item_arr >= 0)

                    if len(idx_neg) > 0:
                        item_arr[idx_neg] = (self._head + item_arr[idx_neg]) % self._capacity
                    if len(idx_pos) > 0:
                        item_arr[idx_pos] = (self._tail + item_arr[idx_pos]) % self._capacity

                result = self._array[item_arr]
        finally:
            self._lock.release()

        return result

    def __iter__(self):
        """ DVG RingBuffer version. """
        print("__iter__")
        if self.is_full:
            self._unwrap_into_buffer()
            return iter(self._unwrap_buffer)
        # else:
        return iter(self._unwrap())

    def __repr__(self):
        print("__repr__")
        return "<RingBuffer of {!r}".format(np.asarray(self))

if __name__ == "__main__":
    cna = CircularNumpyArray(3, 2)
    cna.append([[1, 3], [3, 4]])
    cna.append([[2, 9], [0, 1]])
    cna.append([[4, 6], [7, 8]])
    #cna.append(1)
    #cna.append(1)
    print("cna:", cna)
    print("peek:", cna.peek())
    print("cna._array: ", cna._array)
    print("cna[1]: ", cna[1])



