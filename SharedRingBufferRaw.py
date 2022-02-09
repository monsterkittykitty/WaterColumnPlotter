# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

# Adapted from: https://github.com/eric-wieser/numpy_ringbuffer and
# https://stackoverflow.com/questions/8908998/ring-buffer-with-numpy-ctypes

from multiprocessing import shared_memory
import numpy as np
from RingBuffer import RingBuffer


class SharedRingBufferRaw:
    def __init__(self, settings, counter, full_flag, create_shmem=False):

        self.settings = settings

        self.ALONG_TRACK_PINGS = self.settings['processing_settings']['alongTrackAvg_ping']
        self.MAX_NUM_GRID_CELLS = settings['buffer_settings']['maxGridCells']
        self.SIZE_BUFFER = settings['buffer_settings']['maxBufferSize_ping']
        self.FULL_SIZE_BUFFER = self.SIZE_BUFFER * 2

        self.counter = counter  # multiprocessing.Value; all raw buffers protected with this lock
        self.full_flag = full_flag  # multiprocessing.Value
        self.create_shmem = create_shmem

        self.amplitude_dtype = np.dtype((np.float16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        self.count_dtype = np.dtype((np.uint16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        self.timestamp_dtype = np.dtype(np.float64)
        self.lat_lon_dtype = np.dtype((np.float32, 2))

        self.shmem_amplitude_buffer = None
        self.shmem_count_buffer = None
        self.shmem_timestamp_buffer = None
        self.shmem_lat_lon_buffer = None

        self._initialize_shmem()

        self.amplitude_buffer = None
        self.count_buffer = None
        self.timestamp_buffer = None
        self.lat_lon_buffer = None

        self._initialize_buffers()

    def _initialize_shmem(self):
        # Create shared memory in the backend: note create=False
        self.shmem_amplitude_buffer = shared_memory.SharedMemory(name="shmem_amplitude_buffer",
                                                                 create=self.create_shmem,
                                                                 size=(self.SIZE_BUFFER * 2 *
                                                                       self.amplitude_dtype.itemsize))
        self.shmem_count_buffer = shared_memory.SharedMemory(name="shmem_count_buffer",
                                                             create=self.create_shmem,
                                                             size=self.SIZE_BUFFER * 2 *
                                                                  self.count_dtype.itemsize)
        self.shmem_timestamp_buffer = shared_memory.SharedMemory(name="shmem_timestamp_buffer",
                                                                 create=self.create_shmem,
                                                                 size=self.SIZE_BUFFER * 2 *
                                                                      self.timestamp_dtype.itemsize)
        self.shmem_lat_lon_buffer = shared_memory.SharedMemory(name="shmem_lat_lon_buffer",
                                                               create=self.create_shmem,
                                                               size=self.SIZE_BUFFER * 2 *
                                                                    self.lat_lon_dtype.itemsize)

    def _initialize_buffers(self):
        # Create numpy arrays from the shared memory
        self.amplitude_buffer = np.ndarray(shape=self.SIZE_BUFFER * 2, dtype=self.amplitude_dtype,
                                           buffer=self.shmem_amplitude_buffer.buf)
        self.count_buffer = np.ndarray(shape=self.SIZE_BUFFER * 2, dtype=self.count_dtype,
                                       buffer=self.shmem_count_buffer.buf)
        self.timestamp_buffer = np.ndarray(shape=self.SIZE_BUFFER * 2, dtype=self.timestamp_dtype,
                                           buffer=self.shmem_timestamp_buffer.buf)
        self.lat_lon_buffer = np.ndarray(shape=self.SIZE_BUFFER * 2, dtype=self.lat_lon_dtype,
                                         buffer=self.shmem_lat_lon_buffer.buf)

    def get_lock(self):
        return self.counter.get_lock()

    def clear(self):
        """
        Resets counter to zero to effectively empty buffer.
        """
        print("clearing raw ring buffer")
        with self.counter.get_lock():
            self.counter.value = 0

    def append_all(self, amplitude_data, count_data, timestamp_data, lat_lon_data):
        """this is an O(n) operation"""

        # Ensure data block to add does not exceed total buffer length; if so, trim
        amplitude_data = amplitude_data[-self.SIZE_BUFFER:]
        count_data = count_data[-self.SIZE_BUFFER:]
        timestamp_data = timestamp_data[-self.SIZE_BUFFER:]
        lat_lon_data = lat_lon_data[-self.SIZE_BUFFER:]

        assert len(amplitude_data) == len(count_data) == len(timestamp_data) == len(lat_lon_data)

        n = len(amplitude_data)

        # Compact buffers if length of data to be added exceeds remaining space in buffer
        with self.counter.get_lock():
            if self.remaining() < n:
                self.compact_all()

            self.amplitude_buffer[self.counter.value + self.SIZE_BUFFER:][:n] = amplitude_data
            self.count_buffer[self.counter.value + self.SIZE_BUFFER:][:n] = count_data
            self.timestamp_buffer[self.counter.value + self.SIZE_BUFFER:][:n] = timestamp_data
            self.lat_lon_buffer[self.counter.value + self.SIZE_BUFFER:][:n] = lat_lon_data

            self.counter.value += n

    def remaining(self):
        with self.counter.get_lock():
            return self.SIZE_BUFFER - self.counter.value

    def view(self, buffer):
        """
        Returns all elements of a given buffer, including the empty elements. This is always an O(1) operation.
        :param buffer: The buffer from which to return a view.
        """
        # print("In view:")
        with self.counter.get_lock():
            return buffer[self.counter.value:][:self.SIZE_BUFFER]

    def view_buffer_elements(self, buffer):
        """
        Returns all elements of a given buffer, minus the empty elements.
        :param buffer: The buffer from which to return a view.
        """
        with self.counter.get_lock():
            if self.full_flag.value:
                return buffer[self.counter.value:][:self.SIZE_BUFFER]
            else:
                return buffer[self.counter.value:][:self.SIZE_BUFFER][-self.counter.value:]

    def view_recent_pings(self, buffer, pings):
        with self.counter.get_lock():
            temp = self.view_buffer_elements(buffer)
            return temp[-pings:]
            # return buffer[self.counter.value:][:self.SIZE_BUFFER][-self.counter.value:][-pings:]

    def view_recent_pings_as_pie(self, pings):
        with self.counter.get_lock():
            temp_amp = self.view_recent_pings(self.amplitude_buffer, pings)
            temp_cnt = self.view_recent_pings(self.count_buffer, pings)
            # print("temp.shape before collapse: ", temp_amp.shape)

            # "Collapse" arrays by adding every self.num_pings_to_average so that
            temp_amp = np.sum(temp_amp, axis=0)
            temp_cnt = np.sum(temp_cnt, axis=0)
            # print("temp.shape after collapse: ", temp_amp.shape)

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                temp_avg = temp_amp / temp_cnt

            return temp_avg


    def compact_all(self):
        """
        note: only when this function is called, is an O(size) performance hit incurred,
        and this cost is amortized over the whole padding space
        """
        # print('Compacting all.')
        self.full_flag.value = True
        with self.counter.get_lock():
            self.amplitude_buffer[:self.SIZE_BUFFER] = self.view(self.amplitude_buffer)
            self.count_buffer[:self.SIZE_BUFFER] = self.view(self.count_buffer)
            self.timestamp_buffer[:self.SIZE_BUFFER] = self.view(self.timestamp_buffer)
            self.lat_lon_buffer[:self.SIZE_BUFFER] = self.view(self.lat_lon_buffer)

            self.counter.value = 0

    def get_num_elements_in_buffer(self):
        if self.full_flag.value:
            return self.SIZE_BUFFER
        else:
            with self.counter.get_lock():
                return self.counter.value

    def close_shmem(self):
        self.shmem_amplitude_buffer.close()
        self.shmem_count_buffer.close()
        self.shmem_timestamp_buffer.close()
        self.shmem_lat_lon_buffer.close()

    def unlink_shmem(self):
        self.shmem_amplitude_buffer.unlink()
        self.shmem_count_buffer.unlink()
        self.shmem_timestamp_buffer.unlink()
        self.shmem_lat_lon_buffer.unlink()
