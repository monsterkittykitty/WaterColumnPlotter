from PyQt5.QtCore import QObject
import multiprocessing
import numpy as np
from NumpyRingBuffer import NumpyRingBuffer

# class Main:
#     def __init__(self):
#         test_process = TestProcess()
#
#         process = multiprocessing.Process(target=test_process.run)
#         print("Starting process.")
#         process.start()
#         print("Joining process.")
#         print("Other things...")
#         print("Other things...")
#         print("Other things...")
#         process.join()  # This blocks until process returns.
#         print("Done joining.")
#
# class TestProcess:
#     # def __init__(self, process_boolean):
#     def __init__(self, process_boolean):
#         self.process_boolean = process_boolean
#         pass
#
#     def run(self):
#         print("TestProcess.run()")
#         # for i in range(1000):
#         #     print("Running test process: ", i)
#         while self.process_boolean.value:
#             #print("hi")
#             pass
#
#
# if __name__ == "__main__":
#     main = Main()

class InitializeBigArrays:
    def __init__(self):
        self.MAX_NUM_GRID_CELLS = 500
        self.MAX_LENGTH_BUFFER = 10000  # Based on ~1000 MWC datagrams per minute for 10 minutes (~16 per second).

        # TODO: Shared memory experiment:
        # https://medium.com/@sampsa.riikonen/doing-python-multiprocessing-the-right-way-a54c1880e300
        # https://github.com/elsampsa/medium/blob/main/multiprocess/example2.py
        self.name = "PlotterProcess"

        amplitude_dtype = np.dtype((np.float16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        count_dtype = np.dtype((np.uint16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        timestamp_dtype = np.dtype(np.float32)
        lat_lon_dtype = np.dtype((np.float32, 2))

        total_size = (amplitude_dtype.itemsize + count_dtype.itemsize + timestamp_dtype.itemsize +
                      lat_lon_dtype.itemsize) * self.MAX_LENGTH_BUFFER

        print("TOTAL SIZE (bytes): ", total_size)
        print("TOTAL SIZE (MB): ", total_size / 1024 / 1024)

        # Create shared memory in the frontend: note create=True
        # self.shmem_amplitude_buffer = shared_memory.SharedMemory(name=self.name + "_amplitude_buffer", create=True,
        #                                                          size=self.MAX_LENGTH_BUFFER * amplitude_dtype.itemsize)
        # self.shmem_count_buffer = shared_memory.SharedMemory(name=self.name + "_count_buffer", create=True,
        #                                                      size=self.MAX_LENGTH_BUFFER * count_dtype.itemsize)
        # self.shmem_timestamp_buffer = shared_memory.SharedMemory(name=self.name + "_timestamp_buffer", create=True,
        #                                                          size=self.MAX_LENGTH_BUFFER * timestamp_dtype.itemsize)
        # self.shmem_lat_lon_buffer = shared_memory.SharedMemory(name=self.name + "_lat_lon_buffer", create=True,
        #                                                        size=self.MAX_LENGTH_BUFFER * lat_lon_dtype.itemsize)

        # Create numpy arrays from the shared memory
        self.amplitude_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=amplitude_dtype)
        self.count_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=count_dtype)
        self.timestamp_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=timestamp_dtype)
        self.lat_lon_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=lat_lon_dtype)
        print("Big arrays initialized")