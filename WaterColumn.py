# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

import ctypes
from KongsbergDGMain import KongsbergDGMain
from multiprocessing import Array, Process, Queue, shared_memory, Value
import numpy as np
from NumpyRingBuffer import NumpyRingBuffer
from PlotterMain2 import PlotterMain2
from PyQt5.QtWidgets import QMessageBox

class WaterColumn:
    def __init__(self, settings):

        self.settings = settings

        # multiprocessing.Queues
        self.queue_datagram = Queue()  # .put() by KongsbergDGCaptureFromSonar; .get() by KongsbergDGProcess
        self.queue_pie_object = Queue()  # .put() by KongsbergDGProcess; .get() by Plotter
        self.queue_plot = Queue()  # .put() by Plotter; .get() by GUI_Main
        self.queue_lat_lon = Queue()  # .put() by Plotter; .get() by GUI_Main
        self.queue_timestamp = Queue()  # .put() by Plotter; .get() by GUI_Main

        self.process_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value
        self.raw_buffer_indices = Array(ctypes.c_uint16, [0, 0], lock=True)  # multiprocessing.Array
        self.processed_buffer_indices = Array(ctypes.c_uint16, [0,0], lock=True)  # multiprocessing.Array

        # capacity = 10000 // self.settings["processing_settings"]["alongTrackAvg_ping"]
        # self.vertical_slice_buffer = NumpyRingBuffer(capacity=capacity, dtype=(np.float16, 500))
        # self.horizontal_slice_buffer = NumpyRingBuffer(capacity=capacity, dtype=(np.float16, 500))
        # self.timestamp_slice_buffer = NumpyRingBuffer(capacity=capacity, dtype=np.float32)
        # self.lat_lon_slice_buffer = NumpyRingBuffer(capacity=capacity, dtype=(np.float32, 2))
        # # ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
        #
        # self._lock_raw_buffers = threading.Lock()

        self.MAX_NUM_GRID_CELLS = self.settings['buffer_settings']['maxGridCells']
        self.MAX_LENGTH_BUFFER = self.settings['buffer_settings']['maxBufferSize']

        self.amplitude_dtype = np.dtype((np.float16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        self.count_dtype = np.dtype((np.uint16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        self.timestamp_dtype = np.dtype(np.float32)
        self.lat_lon_dtype = np.dtype((np.float32, 2))
        self.slice_dtype = np.dtype((np.float16, self.MAX_NUM_GRID_CELLS))


        # Create shared memory in the frontend: note create=True
        print("***WATERCOLUMN: Initializing shared memory.")
        self.shmem_amplitude_buffer = shared_memory.SharedMemory(name="shmem_amplitude_buffer", create=True,
                                                                 size=self.MAX_LENGTH_BUFFER * self.amplitude_dtype.itemsize)
        self.shmem_count_buffer = shared_memory.SharedMemory(name="shmem_count_buffer", create=True,
                                                             size=self.MAX_LENGTH_BUFFER * self.count_dtype.itemsize)
        self.shmem_timestamp_buffer = shared_memory.SharedMemory(name="shmem_timestamp_buffer", create=True,
                                                                 size=self.MAX_LENGTH_BUFFER * self.timestamp_dtype.itemsize)
        self.shmem_lat_lon_buffer = shared_memory.SharedMemory(name="shmem_lat_lon_buffer", create=True,
                                                               size=self.MAX_LENGTH_BUFFER * self.lat_lon_dtype.itemsize)

        self.shmem_vertical_slice_buffer = shared_memory.SharedMemory(name="shmem_vertical_slice_buffer",
                                                                      create=True,
                                                                      size=(self.MAX_LENGTH_BUFFER //
                                                                            self.settings['processing_settings']
                                                                            ['alongTrackAvg_ping']) *
                                                                           self.slice_dtype.itemsize)
        self.shmem_horizontal_slice_buffer = shared_memory.SharedMemory(name="shmem_horizontal_slice_buffer",
                                                                        create=True,
                                                                        size=(self.MAX_LENGTH_BUFFER //
                                                                              self.settings['processing_settings']
                                                                              ['alongTrackAvg_ping']) *
                                                                             self.slice_dtype.itemsize)
        self.shmem_timestamp_buffer_avg = shared_memory.SharedMemory(name="shmem_timestamp_buffer_avg", create=True,
                                                                     size=self.MAX_LENGTH_BUFFER //
                                                                           self.settings['processing_settings']
                                                                           ['alongTrackAvg_ping'] *
                                                                          self.timestamp_dtype.itemsize)

        self.shmem_lat_lon_buffer_avg = shared_memory.SharedMemory(name="shmem_lat_lon_buffer_avg", create=True,
                                                                   size=self.MAX_LENGTH_BUFFER //
                                                                         self.settings['processing_settings']
                                                                         ['alongTrackAvg_ping'] *
                                                                        self.lat_lon_dtype.itemsize)

        print("***WATERCOLUMN: Done initializing shared memory.")

        print("***WATERCOLUMN: Initializing big arrays.")
        # # ! ! ! ! ! ALWAYS USE #self._lock_raw_buffers WHEN ACCESSING THESE BUFFERS ! ! ! ! ! :
        # Create numpy arrays from the shared
        #self.amplitude_dtype
        self.amplitude_buffer = np.ndarray(shape=self.MAX_LENGTH_BUFFER, dtype=self.amplitude_dtype,
                                           buffer=self.shmem_amplitude_buffer.buf)
        # print("SHAPE: ", self.amplitude_buffer.shape)
        self.count_buffer = np.ndarray(shape=self.MAX_LENGTH_BUFFER, dtype=self.count_dtype,
                                       buffer=self.shmem_count_buffer.buf)
        self.timestamp_buffer = np.ndarray(shape=self.MAX_LENGTH_BUFFER, dtype=self.timestamp_dtype,
                                           buffer=self.shmem_timestamp_buffer.buf)
        self.lat_lon_buffer = np.ndarray(shape=self.MAX_LENGTH_BUFFER, dtype=self.lat_lon_dtype,
                                         buffer=self.shmem_lat_lon_buffer.buf)

        self.vertical_slice_buffer = np.ndarray(shape=self.MAX_LENGTH_BUFFER // self.settings['processing_settings']
            ['alongTrackAvg_ping'], dtype=self.slice_dtype, buffer=self.shmem_vertical_slice_buffer.buf)
        self.horizontal_slice_buffer = np.ndarray(shape=self.MAX_LENGTH_BUFFER // self.settings['processing_settings']
            ['alongTrackAvg_ping'], dtype=self.slice_dtype, buffer=self.shmem_horizontal_slice_buffer.buf)
        self.timestamp_buffer_avg = np.ndarray(shape=int(self.MAX_LENGTH_BUFFER / self.settings['processing_settings']
            ['alongTrackAvg_ping']), dtype=self.timestamp_dtype, buffer=self.shmem_timestamp_buffer_avg.buf)
        self.lat_lon_buffer = np.ndarray(shape=int(self.MAX_LENGTH_BUFFER / self.settings['processing_settings']
            ['alongTrackAvg_ping']), dtype=self.lat_lon_dtype, buffer=self.shmem_lat_lon_buffer_avg.buf)

        # self.amplitude_buffer = np.ndarray(shape=(self.MAX_LENGTH_BUFFER, self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS),
        #                                    dtype=np.float16, buffer=self.shmem_amplitude_buffer.buf)
        # self.amplitude_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER,
        #                                          dtype=self.amplitude_dtype,
        #                                          buffer=self.shmem_amplitude_buffer.buf)
        # self.count_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER,
        #                                         dtype=(np.uint16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)),
        #                                         buffer=self.shmem_count_buffer.buf)
        # self.timestamp_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=np.float32,
        #                                         buffer=self.shmem_timestamp_buffer.buf)
        # self.lat_lon_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=(np.float32, 2),
        #                                       buffer=self.shmem_lat_lon_buffer.buf)
        print("***WATERCOLUMN: Done initializing big arrays.")

        #print(self.pie_values_buffer.shape)
        #print(self.pie_values_buffer[1])

        self.sonarMain = None
        self.plotterMain = None

    def startProcesses(self):
        """
        Initiates both self.sonarMain and self.plotterMain processes.
        """
        # TODO: Unsure whether lock is needed here?
        with self.process_flag.get_lock():
            self.process_flag.value = True

        print("watercolumn.startProcesses")
        self.__startSonarMain()
        self.__startPlotterMain()

    def __startSonarMain(self):
        """
        Initiates self.sonarMain process.
        """
        if self.settings["system_settings"]["system"] == "Kongsberg":  # Kongsberg system
            self.sonarMain = KongsbergDGMain(self.settings, self.queue_datagram,
                                             self.queue_pie_object, self.process_flag)
            self.sonarMain.run()

        else:  # Other system
            # TODO: Alter code when additional systems supported.
            QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")
            #self.sonarMain = <SystemMain>

    def __startPlotterMain(self):
        """
        Initiates self.plotterMain process.
        """
        self.plotterMain = PlotterMain2(self.settings, self.queue_pie_object, self.queue_plot,
                                        self.queue_lat_lon, self.queue_timestamp, self.raw_buffer_indices,
                                        self.processed_buffer_indices, self.process_flag)
        self.plotterMain.run()

    def stopProcesses(self):
        # TODO: Unsure whether lock is needed here?
        with self.process_flag.get_lock():
            self.process_flag.value = False

        self.__stopSonarMain()
        self.__stopPlotterMain()

    def __stopSonarMain(self):
        # I don't think we actually want this. Method join() will block...
        # if self.sonarMain.is_alive():
        #     self.sonarMain.join()
        pass

    def __stopPlotterMain(self):
        # I don't think we actually want this. Method join() will block...
        # if self.plotterMain.is_alive():
        #     self.plotterMain.join()
        pass

    def get_raw_buffer_length(self):
        with self.raw_buffer_indices.get_lock():
            len = self.raw_buffer_indices[1] - self.raw_buffer_indices[0]
        return len

    def get_processed_buffer_length(self):
        with self.processed_buffer_indices.get_lock():
            len = self.processed_buffer_indices[1] - self.processed_buffer_indices[0]
        return len

    def get_vertical_slice(self):
        # TODO: Implement some kind of unwrap function!
        return self.vertical_slice_buffer[self.processed_buffer_indices[0]:self.processed_buffer_indices[1]]

    def get_horizontal_slice(self):
        # TODO: Implement some kind of unwrap function!
        return self.horizontal_slice_buffer[self.processed_buffer_indices[0]:self.processed_buffer_indices[1]]

    # def get_vertical_slice(self):
    #     print("in get_vertical_slice")
    #     # VERTICAL SLICE:
    #     # Trim arrays to omit values outside of self.vertical_slice_width_m
    #     # start_index       end_index
    #     #          |X|_|_|_|X|
    #     #          |X|_|_|_|X|
    #     #          |X|_|_|_|X|
    #     #          |X|_|_|_|X|
    #
    #     pie_values_vertical = self.amplitude_buffer[:, :,
    #                           self.plotterMain.plotter.vertical_slice_start_index:
    #                           self.plotterMain.plotter.vertical_slice_end_index]
    #     pie_count_vertical = self.amplitude_buffer[:, :,
    #                          self.plotterMain.plotter.vertical_slice_start_index:
    #                          self.plotterMain.plotter.vertical_slice_end_index]
    #
    #     # "Collapse" arrays by adding every self.num_pings_to_average so that
    #     # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average = 1
    #     pie_values_vertical = np.sum(pie_values_vertical, axis=0)
    #     pie_count_vertical = np.sum(pie_count_vertical, axis=0)
    #
    #     with self.raw_buffer_indices.get_lock():
    #         print("get_buffer_length: ", self.get_buffer_length())
    #         if (self.raw_buffer_indices[1] - self.raw_buffer_indices[0]) >= \
    #             self.settings['processing_settings']['alongTrackAvg_ping']:
    #             print("in get_vertical_slice, length is greater")
    #             print("indices: ", self.raw_buffer_indices[0], ", ", self.raw_buffer_indices[1])
    #
    #             # VERTICAL SLICE:
    #             # Trim arrays to omit values outside of self.vertical_slice_width_m
    #             # start_index       end_index
    #             #          |X|_|_|_|X|
    #             #          |X|_|_|_|X|
    #             #          |X|_|_|_|X|
    #             #          |X|_|_|_|X|
    #
    #             # Slice buffer, this should be a view, not a copy:
    #             # Slicing to select only populated fields and relevant across-track data:
    #             temp_slice_amplitude = self.amplitude_buffer[self.raw_buffer_indices[0]:
    #                                                          (self.raw_buffer_indices[1] - 1), :,
    #                                    self.plotterMain.plotter.vertical_slice_start_index:
    #                                    self.plotterMain.plotter.vertical_slice_end_index]
    #             temp_slice_count = self.count_buffer[self.raw_buffer_indices[0]:(self.raw_buffer_indices[1] - 1), :,
    #                                self.plotterMain.plotter.vertical_slice_start_index:
    #                                self.plotterMain.plotter.vertical_slice_end_index]
    #
    #             print("temp_slice.shape before reduce at: ", temp_slice_amplitude.shape)
    #
    #
    #             temp_slice_amplitude = np.add.reduceat(temp_slice_amplitude, np.arange(self.raw_buffer_indices[0],
    #                                                                                    self.raw_buffer_indices[1],
    #                                                                                    self.settings['processing_settings']['alongTrackAvg_ping']))
    #
    #             temp_slice_count = np.add.reduceat(temp_slice_count, np.arange(self.raw_buffer_indices[0],
    #                                                                            self.raw_buffer_indices[1],
    #                                                                            self.settings['processing_settings']['alongTrackAvg_ping']))
    #
    #             print("temp_slice.shape after reduce at: ", temp_slice_amplitude.shape)
    #
    #             temp_slice_amplitude = np.sum(temp_slice_amplitude, axis=2)
    #             temp_slice_count = np.sum(temp_slice_count, axis=2)
    #
    #             print("temp_slice.shape after sum: ", temp_slice_amplitude.shape)
    #
    #             with np.errstate(divide='ignore', invalid='ignore'):
    #                 temp = temp_slice_amplitude / temp_slice_count
    #             return temp
    #
    #
    #
    #             # TODO: This makes a copy. :(
    #             # temp_amplitude = np.add.reduceat(self.amplitude_buffer,
    #             #                                  np.arange(self.raw_buffer_indices[0],
    #             #                                            (self.raw_buffer_indices[1] - 1),
    #             #                                            self.settings['processing_settings']['alongTrackAvg_ping']))
    #
    #             # temp_count = np.add.reduceat(self.count_buffer,
    #             #                              np.arange(self.raw_buffer_indices[0],
    #             #                                        self.get_buffer_length(),
    #             #                                        self.settings['processing_settings']['alongTrackAvg_ping']))
    #
    #             # To silence "RuntimeWarning: invalid value encountered in true_divide" when dividing by zero or nan:
    #             #with np.errstate(divide='ignore', invalid='ignore'):
    #                 # TODO: Slice before averaging? To reduce amount that needs to be averaged?
    #                 #  Or will the slicing create a copy instead of a view?
    #             # temp = temp_amplitude / temp_count
    #             #
    #             # temp_vertical = temp[:, :,
    #             #                 self.plotterMain.plotter.vertical_slice_start_index:
    #             #                 self.plotterMain.plotter.vertical_slice_end_index]
    #             #print("type(temp_amplitude): ", type(temp_amplitude))
    #             #print("end get vertical slice")
    #             # return temp_vertical
    #
    #         return None

    def get_pie(self):
        with self.raw_buffer_indices.get_lock():
            if (self.raw_buffer_indices[1] - self.raw_buffer_indices[0]) > 0:
                print("get pie index: (", ((self.raw_buffer_indices[1] % self.MAX_LENGTH_BUFFER) - 1))
                temp_pie_amplitude = self.amplitude_buffer[(self.raw_buffer_indices[1] % self.MAX_LENGTH_BUFFER) - 1]
                temp_pie_count = self.count_buffer[(self.raw_buffer_indices[1] % self.MAX_LENGTH_BUFFER) - 1]
                # To silence "RuntimeWarning: invalid value encountered in true_divide" when dividing by zero or nan:
                with np.errstate(divide='ignore', invalid='ignore'):
                    pie = temp_pie_amplitude / temp_pie_count
                return pie

    def getSharedMatrices(self):
        return self.amplitude_buffer, self.count_buffer, self.timestamp_buffer, self.lat_lon_buffer

    def closeSharedMemory(self):
        # Quit using shared memory in the frontend
        self.shmem_amplitude_buffer.close()
        self.shmem_count_buffer.close()
        self.shmem_timestamp_buffer.close()
        self.shmem_lat_lon_buffer.close()

        # if self.plotterMain.plotter:
        #     self.plotterMain.plotter.closeSharedMemory()

    def unlinkSharedMemory(self):
        # Release shared memory definitely
        self.shmem_amplitude_buffer.unlink()
        self.shmem_count_buffer.unlink()
        self.shmem_timestamp_buffer.unlink()
        self.shmem_lat_lon_buffer.unlink()

        # if self.plotterMain.plotter:
        #     self.plotterMain.plotter.unlinkSharedMemory()

