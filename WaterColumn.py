# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

import ctypes
from KongsbergDGMain import KongsbergDGMain
from multiprocessing import Array, Process, Queue, shared_memory, Value
import numpy as np
from PlotterMain import PlotterMain
from PyQt5.QtWidgets import QMessageBox
from SharedRingBufferProcessed import SharedRingBufferProcessed
from SharedRingBufferRaw import SharedRingBufferRaw


class WaterColumn:
    def __init__(self, settings):

        self.settings = settings

        # multiprocessing.Queues
        self.queue_datagram = Queue()  # .put() by KongsbergDGCaptureFromSonar; .get() by KongsbergDGProcess
        self.queue_pie_object = Queue()  # .put() by KongsbergDGProcess; .get() by Plotter

        self.process_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value
        self.raw_buffer_full_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value
        self.processed_buffer_full_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value

        self.raw_buffer_count = Value(ctypes.c_uint16, 0, lock=True)  # multiprocessing.Value
        self.processed_buffer_count = Value(ctypes.c_uint16, 0, lock=True)  # multiprocessing.Value

        self.MAX_NUM_GRID_CELLS = self.settings['buffer_settings']['maxGridCells']
        self.MAX_LENGTH_BUFFER = self.settings['buffer_settings']['maxBufferSize']
        self.ALONG_TRACK_PINGS = self.settings['processing_settings']['alongTrackAvg_ping']

        self.shared_ring_buffer_raw = SharedRingBufferRaw(self.settings, self.raw_buffer_count,
                                                          self.raw_buffer_full_flag, create_shmem=True)
        self.shared_ring_buffer_processed = SharedRingBufferProcessed(self.settings, self.processed_buffer_count,
                                                                      self.processed_buffer_full_flag, create_shmem=True)

        self.sonarMain = None
        self.plotterMain = None

    def startProcesses(self):
        """
        Initiates both self.sonarMain and self.plotterMain processes.
        """
        # TODO: Unsure whether lock is needed here?
        with self.process_flag.get_lock():
            self.process_flag.value = True

        self.__startSonarMain()
        self.__startPlotterMain()

    def __startSonarMain(self):
        """
        Initiates and runs self.sonarMain process.
        """
        if self.settings["system_settings"]["system"] == "Kongsberg":  # Kongsberg system
            self.sonarMain = KongsbergDGMain(self.settings, self.queue_datagram,
                                             self.queue_pie_object, self.process_flag)
            self.sonarMain.run()

        else:  # Other system
            # TODO: Alter code when additional systems supported.
            QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")
            # self.sonarMain = <SystemMain>
            # self.sonarMain.run()

    def __startPlotterMain(self):
        """
        Initiates and runs self.plotterMain process.
        """
        self.plotterMain = PlotterMain(self.settings, self.queue_pie_object,
                                       self.raw_buffer_count, self.processed_buffer_count,
                                       self.raw_buffer_full_flag, self.processed_buffer_full_flag,
                                       self.process_flag)

        self.plotterMain.run()

    def stopProcesses(self):
        # TODO: Some sort of error handling and graceful closing of threads
        #  if system is changed while another system thread is running!
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
        return self.shared_ring_buffer_raw.get_num_elements_in_buffer()

    def get_processed_buffer_length(self):
        return self.shared_ring_buffer_processed.get_num_elements_in_buffer()

    def get_pie(self):
        with self.raw_buffer_count.get_lock():
            temp_amp = self.shared_ring_buffer_raw.view_recent_pings(self.shared_ring_buffer_raw.amplitude_buffer, 1)
            temp_cnt = self.shared_ring_buffer_raw.view_recent_pings(self.shared_ring_buffer_raw.count_buffer, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            pie = temp_amp / temp_cnt
        return self._trim_nans_pie(pie[0])

    def get_vertical_slice(self):
        temp_slice = self.shared_ring_buffer_processed.view_buffer_elements(
            self.shared_ring_buffer_processed.vertical_slice_buffer)
        return self._trim_nans_vertical(temp_slice)

    def get_horizontal_slice(self):
        temp_slice = self.shared_ring_buffer_processed.view_buffer_elements(
            self.shared_ring_buffer_processed.horizontal_slice_buffer)
        return self._trim_nans_horizontal(temp_slice)

    def _trim_nans_pie(self, slice):
        # Trim NaNs from matrices to be plotted:
        # This method will look for the index of the last row that is not completely filled with NaNs.
        # Add one to that index for the first full row of NaNs after all data.
        index = np.argwhere(~np.isnan(slice).all(axis=1))[-1][0] + 1
        index_port = np.argwhere(~np.isnan(slice).all(axis=0))[0][0] - 1
        index_stbd = np.argwhere(~np.isnan(slice).all(axis=0))[-1][0] + 1

        # Ensure that 'index' plus some small buffer does not exceed grid size.
        # (Because we want to allow some small buffer around bottom of data if possible.)
        index = min((index + 10), self.MAX_NUM_GRID_CELLS)
        index_port = max((index_port - 10), 0)
        index_stbd = min((index_stbd + 10), self.MAX_NUM_GRID_CELLS)

        return slice[:index, index_port:index_stbd]

    def _trim_nans_vertical(self, slice):
        # Trim NaNs from matrices to be plotted:
        # This method will look for the index of the last row that is not completely filled with NaNs.
        # Add one to that index for the first full row of NaNs after all data.
        index = np.argwhere(~np.isnan(slice).all(axis=0))[-1][0] + 1

        # Ensure that 'index' plus some small buffer does not exceed grid size.
        # (Because we want to allow some small buffer around bottom of data if possible.)
        index = min((index + 10), self.MAX_NUM_GRID_CELLS)

        return slice[:, :index]

    def _trim_nans_horizontal(self, slice):
        # Trim NaNs from matrices to be plotted:
        # This method will look for the index of the last row that is not completely filled with NaNs.
        # Add one to that index for the first full row of NaNs after all data.
        index_port = np.argwhere(~np.isnan(slice).all(axis=0))[0][0] - 1
        index_stbd = np.argwhere(~np.isnan(slice).all(axis=0))[-1][0] + 1

        # Ensure that 'index' plus some small buffer does not exceed grid size.
        # (Because we want to allow some small buffer around bottom of data if possible.)
        index_port = max((index_port - 10), 0)
        index_stbd = min((index_stbd + 10), self.MAX_NUM_GRID_CELLS)

        return slice[:, index_port:index_stbd]

    def closeSharedMemory(self):
        self.shared_ring_buffer_raw.close_shmem()
        self.shared_ring_buffer_processed.close_shmem()

    def unlinkSharedMemory(self):
        self.shared_ring_buffer_raw.unlink_shmem()
        self.shared_ring_buffer_processed.unlink_shmem()
