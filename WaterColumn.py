# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

import ctypes
from KongsbergDGMain import KongsbergDGMain
from multiprocessing import Process, Queue, shared_memory, Value
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

        self.process_flag = Value(ctypes.c_bool, False)  # multiprocessing.Value

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
        print("***WATERCOLUMN: Done initializing shared memory.")

        print("***WATERCOLUMN: Initializing big arrays.")
        # # ! ! ! ! ! ALWAYS USE #self._lock_raw_buffers WHEN ACCESSING THESE BUFFERS ! ! ! ! ! :
        # Create numpy arrays from the shared memory
        self.amplitude_buffer = np.ndarray(shape=(self.MAX_LENGTH_BUFFER, self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS),
                                           dtype=np.float16, buffer=self.shmem_amplitude_buffer.buf)
        # self.amplitude_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER,
        #                                          dtype=self.amplitude_dtype,
        #                                          buffer=self.shmem_amplitude_buffer.buf)
        self.count_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER,
                                                dtype=(np.uint16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)),
                                                buffer=self.shmem_count_buffer.buf)
        self.timestamp_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=np.float32,
                                                buffer=self.shmem_timestamp_buffer.buf)
        self.lat_lon_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=(np.float32, 2),
                                              buffer=self.shmem_lat_lon_buffer.buf)
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
                                        self.queue_lat_lon, self.queue_timestamp, self.process_flag)
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


    def getSharedMatrices(self):
        # amplitude_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=self.amplitude_dtype,
        #                                    buffer=self.shmem_amplitude_buffer.buf)
        # count_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=self.count_dtype,
        #                                     buffer=self.shmem_count_buffer.buf)
        # timestamp_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=self.timestamp_dtype,
        #                                         buffer=self.shmem_timestamp_buffer.buf)
        # lat_lon_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=self.lat_lon_dtype,
        #                                       buffer=self.shmem_lat_lon_buffer.buf)

        print("id(amplitude_buffer) in get: ", id(self.amplitude_buffer))
        return self.amplitude_buffer


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

