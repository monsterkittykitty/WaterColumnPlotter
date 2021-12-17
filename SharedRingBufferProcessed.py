# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

# Adapted from: https://github.com/eric-wieser/numpy_ringbuffer and
# https://stackoverflow.com/questions/8908998/ring-buffer-with-numpy-ctypes

from multiprocessing import shared_memory
import numpy as np


class SharedRingBufferProcessed:
    def __init__(self, settings, counter, full_flag, create_shmem=False):

        self.settings = settings

        self.ALONG_TRACK_PINGS = self.settings['processing_settings']['alongTrackAvg_ping']
        self.MAX_NUM_GRID_CELLS = settings['buffer_settings']['maxGridCells']
        self.SIZE_BUFFER = settings['buffer_settings']['maxBufferSize'] // self.ALONG_TRACK_PINGS
        self.FULL_SIZE_BUFFER = self.SIZE_BUFFER * 2

        self.counter = counter
        self.full_flag = full_flag
        self.create_shmem = create_shmem

        self.slice_dtype = np.dtype((np.float16, self.MAX_NUM_GRID_CELLS))
        self.timestamp_dtype = np.dtype(np.float64)
        self.lat_lon_dtype = np.dtype((np.float32, 2))

        self.shmem_vertical_slice_buffer = None
        self.shmem_horizontal_slice_buffer = None
        self.shmem_timestamp_buffer_avg = None
        self.shmem_lat_lon_buffer_avg = None

        self._initialize_shmem()

        self.vertical_slice_buffer = None
        self.horizontal_slice_buffer = None
        self.timestamp_buffer_avg = None
        self.lat_lon_buffer_avg = None

        self._initialize_buffers()

    def _initialize_shmem(self):
        # Create shared memory in the backend: note create=False
        self.shmem_vertical_slice_buffer = shared_memory.SharedMemory(name="shmem_vertical_slice_buffer",
                                                                      create=self.create_shmem,
                                                                      size=(self.SIZE_BUFFER * 2 *
                                                                           self.slice_dtype.itemsize))
        self.shmem_horizontal_slice_buffer = shared_memory.SharedMemory(name="shmem_horizontal_slice_buffer",
                                                                        create=self.create_shmem,
                                                                        size=(self.SIZE_BUFFER * 2 *
                                                                             self.slice_dtype.itemsize))
        self.shmem_timestamp_buffer_avg = shared_memory.SharedMemory(name="shmem_timestamp_buffer_avg",
                                                                     create=self.create_shmem,
                                                                     size=(self.SIZE_BUFFER * 2 *
                                                                          self.timestamp_dtype.itemsize))
        self.shmem_lat_lon_buffer_avg = shared_memory.SharedMemory(name="shmem_lat_lon_buffer_avg",
                                                                   create=self.create_shmem,
                                                                   size=(self.SIZE_BUFFER * 2 *
                                                                        self.lat_lon_dtype.itemsize))

    def _initialize_buffers(self):
        # Create numpy arrays from the shared memory
        self.vertical_slice_buffer = np.ndarray(shape=(self.SIZE_BUFFER * 2), dtype=self.slice_dtype,
                                                buffer=self.shmem_vertical_slice_buffer.buf)
        self.horizontal_slice_buffer = np.ndarray(shape=(self.SIZE_BUFFER * 2), dtype=self.slice_dtype,
                                                  buffer=self.shmem_horizontal_slice_buffer.buf)
        self.timestamp_buffer_avg = np.ndarray(shape=(self.SIZE_BUFFER * 2), dtype=self.timestamp_dtype,
                                               buffer=self.shmem_timestamp_buffer_avg.buf)
        self.lat_lon_buffer_avg = np.ndarray(shape=(self.SIZE_BUFFER * 2), dtype=self.lat_lon_dtype,
                                             buffer=self.shmem_lat_lon_buffer_avg.buf)

    def append_all(self, vertical_data, horizontal_data, timestamp_data, lat_lon_data):
        """this is an O(n) operation"""

        # Ensure data block to add does not exceed total buffer length; if so, trim
        vertical_data = vertical_data[-self.SIZE_BUFFER:]
        horizontal_data = horizontal_data[-self.SIZE_BUFFER:]
        timestamp_data = timestamp_data[-self.SIZE_BUFFER:]
        lat_lon_data = lat_lon_data[-self.SIZE_BUFFER:]

        assert len(vertical_data) == len(horizontal_data) == len(timestamp_data) == len(lat_lon_data)

        n = len(vertical_data)

        # Compact buffers if length of data to be added exceeds remaining space in buffer
        with self.counter.get_lock():
            if self.remaining() < n:
                self.compact_all()

            self.vertical_slice_buffer[self.counter.value + self.SIZE_BUFFER:][:n] = vertical_data
            self.horizontal_slice_buffer[self.counter.value + self.SIZE_BUFFER:][:n] = horizontal_data
            self.timestamp_buffer_avg[self.counter.value + self.SIZE_BUFFER:][:n] = timestamp_data
            self.lat_lon_buffer_avg[self.counter.value + self.SIZE_BUFFER:][:n] = lat_lon_data

            self.counter.value += n

    def remaining(self):
        with self.counter.get_lock():
            return self.SIZE_BUFFER-self.counter.value

    def view(self, buffer):
        """this is always an O(1) operation"""
        print("in view:")
        with self.counter.get_lock():
            return buffer[self.counter.value:][:self.SIZE_BUFFER]

    def view_buffer_elements(self, buffer):
        with self.counter.get_lock():
            if self.full_flag.value:
                return buffer[self.counter.value:][:self.SIZE_BUFFER]
            elif self.counter.value > 0:
                return buffer[self.counter.value:][:self.SIZE_BUFFER][-self.counter.value:]

    def view_recent_pings(self, buffer, pings):
        with self.counter.get_lock():
            temp = self.view_buffer_elements(buffer)
            return temp[-pings:]
            # return buffer[self.counter.value:][:self.FULL_SIZE_BUFFER][-self.counter.value:][-pings:]

    def compact_all(self):
        """
        note: only when this function is called, is an O(size) performance hit incurred,
        and this cost is amortized over the whole padding space
        """
        print('compacting all')
        self.full_flag.value = True
        with self.counter.get_lock():
            self.vertical_slice_buffer[:self.SIZE_BUFFER] = self.view(self.vertical_slice_buffer)
            self.horizontal_slice_buffer[:self.SIZE_BUFFER] = self.view(self.horizontal_slice_buffer)
            self.timestamp_buffer_avg[:self.SIZE_BUFFER] = self.view(self.timestamp_buffer_avg)
            self.lat_lon_buffer_avg[:self.SIZE_BUFFER] = self.view(self.lat_lon_buffer_avg)

            self.counter.value = 0

    def get_num_elements_in_buffer(self):
        if self.full_flag.value:
            return self.SIZE_BUFFER
        else:
            with self.counter.get_lock():
                return self.counter.value

    def close_shmem(self):
        self.shmem_vertical_slice_buffer.close()
        self.shmem_horizontal_slice_buffer.close()
        self.shmem_timestamp_buffer_avg.close()
        self.shmem_lat_lon_buffer_avg.close()

    def unlink_shmem(self):
        self.shmem_vertical_slice_buffer.unlink()
        self.shmem_horizontal_slice_buffer.unlink()
        self.shmem_timestamp_buffer_avg.unlink()
        self.shmem_lat_lon_buffer_avg.unlink()
