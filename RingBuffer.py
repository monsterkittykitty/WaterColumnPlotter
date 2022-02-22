# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

# Adapted from: https://github.com/eric-wieser/numpy_ringbuffer and
# https://stackoverflow.com/questions/8908998/ring-buffer-with-numpy-ctypes

from multiprocessing import shared_memory
import numpy as np

class RingBuffer:
    #def __init__(self, settings, indices):
    def __init__(self, settings, counter, full_flag, size, dtype, shmem_buf=None):

        self.settings = settings

        self.counter = counter  # multiprocessing.Value
        #self.counter_copy = counter.value
        self.full_flag = full_flag

        self.size = size
        self.dtype = dtype  # numpy.dtype
        self.shmem_buf = shmem_buf  # multiprocessing.shared_memory

        self.ALONG_TRACK_PINGS = self.settings['processing_settings']['alongTrackAvg_ping']
        self.MAX_NUM_GRID_CELLS = settings['buffer_settings']['maxGridCells']
        #self.MAX_LENGTH_BUFFER = settings['buffer_settings']['maxBufferSize']
        self.TOTAL_BUFFER_SIZE = self.size * 2

        self.buffer_array = np.ndarray(shape=self.TOTAL_BUFFER_SIZE, dtype=self.dtype, buffer=self.shmem_buf)

        self.full = False

    # def append(self, data):
    #     """this is an O(n) operation"""
    #
    #     # Ensure data block to add does not exceed total buffer length; if so, trim
    #     data = data[-self.shape:]
    #
    #     n = len(data)
    #
    #     # Compact buffers if length of data to be added exceeds remaining space in buffer
    #     with self.counter.get_lock():
    #         if self.remaining() < n:
    #             self.compact_all()
    #
    #         self.buffer_array[self.counter.value + self.shape:][:n] = data
    #
    #         # TODO: We need to return n value? Instead of incrementing count...
    #         # self.counter.value += n
    #
    # def remaining(self):
    #     with self.counter.get_lock():
    #         return self.shape - self.counter.value
    #
    # def _view(self):
    #     """this is always an O(1) operation"""
    #     print("in view:")
    #     with self.counter.get_lock():
    #         return self.buffer_array[self.counter.value:][:self.RING_BUFFER_SIZE]
    #
    # @staticmethod
    # def view_static(self, buffer, counter, size):
    #     with counter.get_lock():
    #         return buffer[counter.value:][:size]
    #
    # def view_buffer_elements(self):
    #     with self.counter.get_lock():
    #         if self.full:
    #             return self.buffer_array[self.counter.value:][:self.RING_BUFFER_SIZE]
    #         else:
    #             return self.buffer_array[self.counter.value:][:self.RING_BUFFER_SIZE][-self.counter.value:]
    #
    # def view_recent_pings(self, pings):
    #     with self.counter.get_lock():
    #         return self.buffer_array[self.counter.value:][:self.RING_BUFFER_SIZE][-self.counter.value:][:-pings]
    #
    # def compact_all(self):
    #     """
    #     note: only when this function is called, is an O(size) performance hit incurred,
    #     and this cost is amortized over the whole padding space
    #     """
    #     print('compacting all')
    #     self.full = True
    #     with self.counter.get_lock():
    #         self.buffer_array[:self.RING_BUFFER_SIZE] = self._view()
    #
    #         self.counter.value = 0
    #
    # def close_shmem(self):
    #     self.shmem_buf.close()