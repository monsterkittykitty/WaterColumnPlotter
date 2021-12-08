# Using examples from:
# https://medium.com/@sampsa.riikonen/doing-python-multiprocessing-the-right-way-a54c1880e300
# "Back-end" methods -- code that runs "on the other side of the fork" shall have a double underscore in their name.

# 1. Import libraries that do not use multithreading:
from KongsbergDGPlot import KongsbergDGPlot
from multiprocessing import Process, shared_memory, Queue
import numpy as np
from NumpyRingBuffer import NumpyRingBuffer
# from Plotter import Plotter
from Plotter2 import Plotter2

import TestProcess
from TestProcess import InitializeBigArrays


#class PlotterMain2(Process):
class PlotterMain2:

    def __init__(self, settings, queue_pie_object, queue_plot, queue_lat_lon,
                 queue_timestamp, raw_buffer_indices, processed_buffer_indices, process_flag):
        #super().__init__()

        self.settings = settings

        # multiprocessing.Queues
        self.queue_pie_object = queue_pie_object
        self.queue_plot = queue_plot
        self.queue_lat_lon = queue_lat_lon
        self.queue_timestamp = queue_timestamp

        self.raw_buffer_indices = raw_buffer_indices
        self.processed_buffer_indices = processed_buffer_indices

        self.process_flag = process_flag

        # self.test_buffer_a = np.empty((500 * 10, 500 * 10), dtype=np.float32)
        # self.test_buffer_b = np.empty((500 * 10, 500 * 10), dtype=np.float32)
        # self.test_buffer_c = np.empty((500 * 10, 500 * 10), dtype=np.float32)
        # self.test_buffer_d = np.empty((500 * 10, 500 * 10), dtype=np.float32)

        # 2. Create interprocess communication primitives and shared resources
        # that are shared with the main process and your current multiprocess
        # self.queue_plot = Queue()  # multiprocessing.Queue
        # self.queue_lat_lon = Queue()  # multiprocessing.Queue
        # self.queue_timestamp = Queue()  # multiprocessing.Queue
        #
        self.plotter = None

        self.daemon = True

        # TODO: Delete this
        # self.test_process = TestProcess.TestProcess(process_boolean)

        # self.MAX_LENGTH_BUFFER = 10000
        # self.MAX_NUM_GRID_CELLS = 500
        # self.name = "Plotter"
        #
        # amplitude_dtype = np.dtype((np.float16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        # count_dtype = np.dtype((np.uint16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        # timestamp_dtype = np.dtype(np.float32)
        # lat_lon_dtype = np.dtype((np.float32, 2))
        #
        # # Create shared memory in the frontend: note create=True
        # self.shmem_amplitude_buffer = shared_memory.SharedMemory(name=self.name + "_amplitude_buffer", create=True,
        #                                                          size=self.MAX_LENGTH_BUFFER * amplitude_dtype.itemsize)
        # self.shmem_count_buffer = shared_memory.SharedMemory(name=self.name + "_count_buffer", create=True,
        #                                                      size=self.MAX_LENGTH_BUFFER * count_dtype.itemsize)
        # self.shmem_timestamp_buffer = shared_memory.SharedMemory(name=self.name + "_timestamp_buffer", create=True,
        #                                                          size=self.MAX_LENGTH_BUFFER * timestamp_dtype.itemsize)
        # self.shmem_lat_lon_buffer = shared_memory.SharedMemory(name=self.name + "_lat_lon_buffer", create=True,
        #                                                        size=self.MAX_LENGTH_BUFFER * lat_lon_dtype.itemsize)
        #
        # # Create numpy arrays from the shared memory
        # self.amplitude_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=amplitude_dtype,
        #                                         buffer=self.shmem_amplitude_buffer.buf)
        # self.count_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=count_dtype,
        #                                     buffer=self.shmem_count_buffer.buf)
        # self.timestamp_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=timestamp_dtype,
        #                                         buffer=self.shmem_timestamp_buffer.buf)
        # self.lat_lon_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=lat_lon_dtype,
        #                                       buffer=self.shmem_lat_lon_buffer.buf)




    def run(self):
        print("In PlotterMain.run()")

        # self.process_plotter = multiprocessing.Process(target=self.plotter.run())
        # self.process_plotter = multiprocessing.Process(target=self.plotter.get_and_plot_pie())

        # This does not print the true conditions, instead it results in a pickle error:
        # _pickle.UnpicklingError: pickle data was truncated
        # And GUI_Main does not exit if play is pressed, (then stop pressed or unpressed), then main window closed.
        #self.process_plotter = multiprocessing.Process(target=self.plotter.get_and_buffer_pie)

        # This does print the true conditions, and GUI_Main does exit when main window closed if stop has been pressed.
        # But this keeps printing if main window is closed and stop is not pressed.
        # self.process_plotter = multiprocessing.Process(target=self.plotter.get_and_buffer_pie())

        # TODO: Delete this
        # No pickle error with this. This works great. No problems with or without stop button, GUI_Main closes.
        # self.process_plotter =multiprocessing.Process(target=self.test_process.run)

        # self.process_plotter.daemon = True
        # self.process_plotter.start()
        # self.process_plotter.join()
        # print("Initializing big arrays")
        # bigArrays = InitializeBigArrays()
        # print("Done initializing big arrays")
        #exit()

        self.plotter = Plotter2(self.settings, self.queue_pie_object, self.queue_plot,
                                self.queue_lat_lon, self.queue_timestamp, self.raw_buffer_indices,
                                self.processed_buffer_indices, self.process_flag)
        #
        # # self.plotter = Plotter2(self.settings, self.queue_pie_object, self.queue_plot,
        # #                         self.queue_lat_lon, self.queue_timestamp, self.process_flag,
        # #                         self.shmem_amplitude_buffer, self.shmem_count_buffer, self.shmem_timestamp_buffer,
        # #                         self.shmem_lat_lon_buffer, self.amplitude_buffer, self.count_buffer,
        # #                         self.timestamp_buffer, self.lat_lon_buffer)
        #
        self.plotter.daemon = True
        self.plotter.start()
