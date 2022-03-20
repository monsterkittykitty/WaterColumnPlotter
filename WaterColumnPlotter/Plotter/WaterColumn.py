# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

import ctypes
from multiprocessing import Array, Queue, Value
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from WaterColumnPlotter.Kongsberg.KongsbergDGMain import KongsbergDGMain
from WaterColumnPlotter.Plotter.PlotterMain import PlotterMain
from WaterColumnPlotter.Plotter.SharedRingBufferProcessed import SharedRingBufferProcessed
from WaterColumnPlotter.Plotter.SharedRingBufferRaw import SharedRingBufferRaw


class WaterColumn:
    def __init__(self, settings):

        self.settings = settings

        self.processing_settings_edited = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value

        # Break out individual settings so they can be shared / updated across processes

        # Note on why using datatype ctypes.c_wchar_p causes a crash when the variable is located in a second process
        # (from https://docs.python.org/2.7/library/multiprocessing.html#module-multiprocessing.sharedctypes):
        # "Note: Although it is possible to store a pointer in shared memory remember that this will refer to a
        # location in the address space of a specific process. However, the pointer is quite likely to be invalid in the
        # context of a second process and trying to dereference the pointer from the second process may cause a crash."

        # multiprocessing.Array and multiprocessing.Values
        # Array must be initialized to maximum number of characters in a valid IP address (15)
        self.ip = Array('u', self.editIP(self.settings['ip_settings']['ip'], append=True), lock=True)
        self.port = Value(ctypes.c_uint16, self.settings['ip_settings']['port'], lock=True)
        # Use only first letter of protocol for shared value: T = TCP; U = UDP; M = Multicast
        self.protocol = Value(ctypes.c_wchar, self.settings['ip_settings']['protocol'][0], lock=True)
        self.socket_buffer_multiplier = Value(ctypes.c_uint8,
                                              self.settings['ip_settings']['socketBufferMultiplier'], lock=True)
        self.bin_size = Value(ctypes.c_float, self.settings['processing_settings']['binSize_m'], lock=True)
        self.across_track_avg = Value(ctypes.c_float,
                                      self.settings['processing_settings']['acrossTrackAvg_m'], lock=True)
        self.depth = Value(ctypes.c_float, self.settings['processing_settings']['depth_m'], lock=True)
        self.depth_avg = Value(ctypes.c_float, self.settings['processing_settings']['depthAvg_m'], lock=True)
        self.along_track_avg = Value(ctypes.c_uint8,
                                     self.settings['processing_settings']['alongTrackAvg_ping'], lock=True)
        self.max_heave = Value(ctypes.c_float, self.settings['processing_settings']['maxHeave_m'], lock=True)
        self.max_grid_cells = Value(ctypes.c_uint16, self.settings['buffer_settings']['maxGridCells'], lock=True)
        self.max_ping_buffer = Value(ctypes.c_uint16, self.settings['buffer_settings']['maxBufferSize_ping'], lock=True)

        # Set to true when IP settings are edited; pass argument to sonarMain when signaling setting changes
        self.ip_settings_edited = False

        # multiprocessing.Queues
        self.queue_datagram = Queue()  # .put() by KongsbergDGCaptureFromSonar; .get() by KongsbergDGProcess
        self.queue_pie_object = Queue()  # .put() by KongsbergDGProcess; .get() by Plotter

        # A count to track the number of full #MWC records (pings) received and reconstructed
        self.full_ping_count = Value(ctypes.c_uint32, 0, lock=True)  # multiprocessing.Value
        # A count to track the number of #MWC records (pings) that could not be reconstructed
        self.discard_ping_count = Value(ctypes.c_uint32, 0, lock=True)  # multiprocessing.Value

        # self.process_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value
        self.sonar_process_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value
        self.plotter_process_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value

        # Shared values for implementation of ring buffers in shared memory
        self.raw_buffer_count = Value(ctypes.c_uint16, 0, lock=True)  # multiprocessing.Value
        self.processed_buffer_count = Value(ctypes.c_uint16, 0, lock=True)  # multiprocessing.Value
        self.raw_buffer_full_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value
        self.processed_buffer_full_flag = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value

        # TODO: Make these multiprocessing.Values?
        self.MAX_NUM_GRID_CELLS = self.settings['buffer_settings']['maxGridCells']
        self.MAX_LENGTH_BUFFER = self.settings['buffer_settings']['maxBufferSize_ping']
        # self.ALONG_TRACK_PINGS = self.settings['processing_settings']['alongTrackAvg_ping']

        self.shared_ring_buffer_raw = None
        self.shared_ring_buffer_processed = None

        self.sonarMain = None
        self.plotterMain = None

    def initRingBuffers(self, create_shmem=False):
        """
        Initialize raw and processed ring buffers.
        :param create_shmem: Boolean indicating whether shared memory must be created
        (it only needs to be created once).
        """
        self.shared_ring_buffer_raw = SharedRingBufferRaw(self.settings, self.raw_buffer_count,
                                                          self.raw_buffer_full_flag, create_shmem=create_shmem)
        self.shared_ring_buffer_processed = SharedRingBufferProcessed(self.settings, self.processed_buffer_count,
                                                                      self.processed_buffer_full_flag,
                                                                      create_shmem=create_shmem)

    def editIP(self, ip, append=True):
        """
        IP addresses shared between processes must be 15 characters in length when stored as a multiprocessing.Array.
        When append is true, this method appends characters ("_") to the left side of the IP address string to meet the
        15-character requirement; when append is false, those characters ("_") are stripped from the left side of the
        IP address string so that the string can be interpreted as a valid ip address.
        :param ip: IP address, with or without special characters ("_") appended
        :param append: When true, special characters are appended to ip;
        when false, special characters are stripped from ip
        """
        if append:
            while len(ip) < 15:
                ip = "_" + ip
        else:  # Strip
            ip = ip.lstrip("_")

        return ip

    def playProcesses(self):
        """
        Initiates processes managed by both self.sonarMain and self.plotterMain.
        """
        # with self.process_flag.get_lock():
        #     self.process_flag.value = True
        print("watercolumn, play_processes before, ip: {}".format(self.ip_settings_edited))
        self._playSonarMain()
        self._playPlotterMain()
        print("watercolumn, play_processes after, ip: {}".format(self.ip_settings_edited))

    def _playSonarMain(self):
        """
        Initiates and runs processes managed by self.sonarMain.
        """
        if self.settings["system_settings"]["system"] == "Kongsberg":  # Kongsberg system

            self.sonarMain = KongsbergDGMain(self.settings, self.ip, self.port, self.protocol,
                                             self.socket_buffer_multiplier, self.bin_size, self.max_heave,
                                             self.max_grid_cells, self.queue_datagram, self.queue_pie_object,
                                             self.full_ping_count, self.discard_ping_count)

            self.sonarMain.play_processes()

            self.sonarMain.run()

        else:  # Other system
            # Disable warning when other systems are supported.
            QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")
            # self.sonarMain = <SystemMain>
            # self.sonarMain.run()

    def _playPlotterMain(self):
        """
        Initiates and runs process managed by self.plotterMain.
        """
        self.plotterMain = PlotterMain(self.settings, self.bin_size, self.across_track_avg, self.depth, self.depth_avg,
                                       self.along_track_avg, self.max_heave, self.queue_pie_object,
                                       self.raw_buffer_count, self.processed_buffer_count,
                                       self.raw_buffer_full_flag, self.processed_buffer_full_flag)

        self.plotterMain.play_processes()

        self.plotterMain.run()

    def pauseProcesses(self):
        """
        Signals processes managed by both sonarMain and plotterMain than pause has been pressed.
        """
        self._pauseSonarMain()
        self._pausePlotterMain()

    def _pauseSonarMain(self):
        """
        Signals processes managed by sonarMain that pause has been pressed.
        """
        if self.sonarMain:
            self.sonarMain.pause_processes()

    def _pausePlotterMain(self):
        """
        Signals process managed by plotterMain that pause has been pressed.
        """
        if self.plotterMain:
            self.plotterMain.pause_processes()

    def stopProcesses(self):
        """
        Signals processes managed by both sonarMain and plotterMain than stop has been pressed.
        """
        self._stopSonarMain()
        self._stopPlotterMain()

    def _stopSonarMain(self):
        """
        Signals processes managed by sonarMain that stop has been pressed.
        """
        if self.sonarMain:
            self.sonarMain.stop_processes()

    def _stopPlotterMain(self):
        """
        Signals process managed by plotterMain that stop has been pressed.
        """
        if self.plotterMain:
            self.plotterMain.stop_processes()

    def get_raw_buffer_length(self):
        """
        Returns number of elements in raw ring buffer in shared memory.
        :return: Number of elements in raw ring buffer in shared memory.
        """
        return self.shared_ring_buffer_raw.get_num_elements_in_buffer()

    def get_processed_buffer_length(self):
        """
        Returns number of elements in processed ring buffer in shared memory.
        :return: Number of elements in processed ring buffer in shared memory.
        """
        return self.shared_ring_buffer_processed.get_num_elements_in_buffer()

    def get_pie(self):
        """
        Calculates average amplitude values for most recent along_track_avg number of pings in raw ring buffer.
        :return: A numpy matrix of average amplitude values for the most recent along_track_avg number of pings
        in raw ring buffer if at least one valid entry exists; otherwise, returns None.
        """
        with self.raw_buffer_count.get_lock():
            # This pulls most recent alongTrackAvg_ping from 'raw' buffer:
            pie = self.shared_ring_buffer_raw.view_recent_pings_as_pie(
                self.settings['processing_settings']['alongTrackAvg_ping'])
        # Check that temp arrays are not all NaNs (from 'discarded' pings)
        if not np.all(np.isnan(pie)):
            return self._trim_nans_pie(pie)
        return None  # If temp arrays are all zero

    def get_vertical_slice(self):
        """
        Retrieves all valid entries from processed ring buffer's vertical slice buffer.
        :return: A numpy matrix of all valid entries from processed ring buffer's vertical slice buffer
        if at least one valid entry exists; otherwise, returns None.
        """
        temp_slice = self.shared_ring_buffer_processed.view_buffer_elements(
            self.shared_ring_buffer_processed.vertical_slice_buffer)
        # return None  # If temp arrays are all zero
        if not np.all(np.isnan(temp_slice)):
            return self._trim_nans_vertical(temp_slice)
        return None

    def get_horizontal_slice(self):
        """
        Retrieves all valid entries from processed ring buffer's horizontal slice buffer.
        :return: A numpy matrix of all valid entries from processed ring buffer's horizontal slice buffer
        if at least one valid entry exists; otherwise, returns None.
        """
        temp_slice = self.shared_ring_buffer_processed.view_buffer_elements(
            self.shared_ring_buffer_processed.horizontal_slice_buffer)
        # return None  # If temp arrays are all zero
        if not np.all(np.isnan(temp_slice)):
            return self._trim_nans_horizontal(temp_slice)
        return None

    def _trim_nans_pie(self, slice):
        """
        Trims excess rows and columns of nans from data before plotting.
        :param slice: A numpy matrix representing average amplitude values for pie slice display.
        :return: A numpy matrix representing average amplitude values for pie slice display,
        with excess rows and columns of nans removed.
        """
        # Trim NaNs from matrices to be plotted:
        # This method will look for the index of the last row that is not completely filled with NaNs.
        # Add one to that index for the first full row of NaNs after all data.
        index_depth = np.argwhere(~np.isnan(slice).all(axis=1))[-1][0] + 1
        index_port = np.argwhere(~np.isnan(slice).all(axis=0))[0][0] - 1
        index_stbd = np.argwhere(~np.isnan(slice).all(axis=0))[-1][0] + 1

        # Ensure that 'index' plus some small buffer does not exceed grid size.
        # (Because we want to allow some small buffer around bottom of data if possible.)
        index_depth = min((index_depth + 10), self.MAX_NUM_GRID_CELLS)
        index_port = max((index_port - 10), 0)
        index_stbd = min((index_stbd + 10), self.MAX_NUM_GRID_CELLS)

        # Simplify trimming to trim the same amount from both port and stbd sides
        index_across_track = min(index_port, (self.MAX_NUM_GRID_CELLS - index_stbd))

        # This value keeps track of the 'zero' position along the x-axis. This, in theory,
        # should be the position of the sonar and should always be placed at x = 0.
        # x_zero = int(slice.shape[0] / 2) - index_port
        # return slice[:index_depth, index_port:index_stbd], x_zero

        return slice[:index_depth, index_across_track:-index_across_track]

    def _trim_nans_vertical(self, slice):
        """
        Trims excess rows of nans from data before plotting.
        :param slice: A numpy matrix representing processed data for vertical slice display.
        :return: A numpy matrix representing processed data for vertical slice display,
        with excess rows of nans removed.
        """
        # Trim NaNs from matrices to be plotted:
        # This method will look for the index of the last row that is not completely filled with NaNs.
        # Add one to that index for the first full row of NaNs after all data.
        # index_heave = np.argwhere(~np.isnan(slice).all(axis=0))[0][0] - 1
        index_depth = np.argwhere(~np.isnan(slice).all(axis=0))[-1][0] + 1

        # Ensure that 'index' plus some small buffer does not exceed grid size.
        # (Because we want to allow some small buffer around bottom of data if possible.)
        # index_heave = max((index_heave - 10), 0)
        index_depth = min((index_depth + 10), self.MAX_NUM_GRID_CELLS)

        # return slice[:, index_heave:index_depth], index_heave
        return slice[:, :index_depth]

    def _trim_nans_horizontal(self, slice):
        """
        Trims excess rows of nans from data before plotting.
        :param slice: A numpy matrix representing processed data for horizontal slice display.
        :return: A numpy matrix representing processed data for horizontal slice display,
        with excess rows of nans removed.
        """
        # Trim NaNs from matrices to be plotted:
        # This method will look for the index of the last row that is not completely filled with NaNs.
        # Add one to that index for the first full row of NaNs after all data.
        index_port = np.argwhere(~np.isnan(slice).all(axis=0))[0][0] - 1
        index_stbd = np.argwhere(~np.isnan(slice).all(axis=0))[-1][0] + 1

        # Ensure that 'index' plus some small buffer does not exceed grid size.
        # (Because we want to allow some small buffer around bottom of data if possible.)
        index_port = max((index_port - 10), 0)
        index_stbd = min((index_stbd + 10), self.MAX_NUM_GRID_CELLS)

        # Simplify trimming to trim the same amount from both port and stbd sides
        index_across_track = min(index_port, (self.MAX_NUM_GRID_CELLS - index_stbd))

        # return slice[:, index_port:index_stbd]
        return slice[:, index_across_track:-index_across_track]

    def settingsChanged(self):
        """
        Called when settings have been changed.
        """
        self.update_buffers()

    def signalSubprocessSettingsChanged(self):
        """
        Signals subprocesses managed by sonarMain and plotterMain when settings have been changed.
        """
        print("WaterColumn, signalSubprocessSettingsChanged")
        if self.sonarMain:
            self.sonarMain.settings_changed(self.ip_settings_edited)
        if self.plotterMain:
            self.plotterMain.settings_changed()

    def update_buffers(self):
        """
        Called when settings are changed. Updates raw and processed ring buffers.
        Signals to subprocesses that settings have been changed.
        """
        print("WaterColumn, update_buffers.")
        if self.plotterMain:
            # Get lock on shared_ring_buffer_raw; this will ensure that no other
            # changes can be made to shared_ring_buffer_raw while we make updates
            # with self.shared_ring_buffer_raw.get_lock():
            with self.shared_ring_buffer_raw.counter.get_lock():
                self.plotterMain.plotter.update_local_settings()
                if self.plotterMain.plotter.bin_size_edited:
                    # If bin size is edited, clear both raw and processed ring buffers.
                    self.shared_ring_buffer_raw.clear()  # This methods gets lock
                    self.shared_ring_buffer_processed.clear()  # This method gets lock
                    self.plotterMain.plotter.bin_size_edited = False
                else:
                    if self.plotterMain.plotter.max_heave_edited:
                        print("**************************************************MAX HEAVE EDITED")
                        # Note that we already hold lock on shared_ring_buffer_raw
                        temp_amplitude_buffer_raw = self.shared_ring_buffer_raw.view_buffer_elements(
                            self.shared_ring_buffer_raw.amplitude_buffer)
                        temp_count_buffer_raw = self.shared_ring_buffer_raw.view_buffer_elements(
                            self.shared_ring_buffer_raw.count_buffer)
                        self.plotterMain.plotter.shift_heave(temp_amplitude_buffer_raw, temp_count_buffer_raw,
                                                             self.plotterMain.plotter.outdated_heave,
                                                             self.plotterMain.plotter.max_heave_local)
                        self.plotterMain.plotter.max_heave_edited = False

                    # Recalculate processed ring buffers based on update settings / updated raw ring buffers
                    # Note that this method holds lock on raw buffers for entire calculation and only get lock on
                    # processed buffer for final phase of adding processed data to processed buffer.
                    print("prior to recalculate, raw ring buffer len: ", self.shared_ring_buffer_raw.get_num_elements_in_buffer())
                    print("prior to recalculate, processed ring buffer len: ",
                          self.shared_ring_buffer_processed.get_num_elements_in_buffer())
                    self.plotterMain.plotter.recalculate_processed_buffer(self.shared_ring_buffer_raw,
                                                                          self.shared_ring_buffer_processed)
                    print("after recalculate, raw ring buffer len: ",
                          self.shared_ring_buffer_raw.get_num_elements_in_buffer())
                    print("after recalculate, processed ring buffer len: ",
                          self.shared_ring_buffer_processed.get_num_elements_in_buffer())
                # Signal to subprocesses that settings have changed.
                self.signalSubprocessSettingsChanged()
                # Reset IP settings edited flag
                if self.ip_settings_edited:
                    self.ip_settings_edited = False

    def closeSharedMemory(self):
        """
        Closes shared memory used by raw and processed ring buffers.
        """
        self.shared_ring_buffer_raw.close_shmem()
        self.shared_ring_buffer_processed.close_shmem()

    def unlinkSharedMemory(self):
        """
        Unlinks shared memory used by raw and processed ring buffers.
        """
        self.shared_ring_buffer_raw.unlink_shmem()
        self.shared_ring_buffer_processed.unlink_shmem()
