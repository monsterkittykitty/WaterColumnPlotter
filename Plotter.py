# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

import datetime
import logging
import math
from multiprocessing import Process, shared_memory, Value
import numpy as np
# import pyvista as pv  # "Easier Pythonic interface to VTK"
# from pyvistaqt import BackgroundPlotter, QtInteractor
from SharedRingBufferProcessed import SharedRingBufferProcessed
from SharedRingBufferRaw import SharedRingBufferRaw
import threading
import time
import queue

logger = logging.getLogger(__name__)

class Plotter(Process):
    def __init__(self, settings, bin_size, across_track_avg, depth, depth_avg, along_track_avg, max_heave,
                 settings_edited, queue_pie_object, raw_buffer_count, processed_buffer_count,
                 raw_buffer_full_flag, processed_buffer_full_flag, process_flag):
        super().__init__()
        print("New instance of Plotter.")

        self.settings = settings

        # multiprocessing.Values (shared between processes)
        self.bin_size = bin_size
        self.across_track_avg = across_track_avg
        self.depth = depth
        self.depth_avg = depth_avg
        # self.along_track_avg = along_track_avg
        self.max_heave = max_heave

        self.settings_edited = settings_edited

        # Local copies of above multiprocessing.Values (to avoid frequent accessing of locks)
        self.bin_size_local = None
        self.across_track_avg_local = None
        self.depth_local = None
        self.depth_avg_local = None
        # self.along_track_avg_local = None
        self.max_heave_local = None
        # Initialize above local copies
        self.update_local_settings()

        self.queue_pie_object = queue_pie_object  # multiprocessing.Queue

        self.raw_buffer_count = raw_buffer_count  # multiprocessing.Value
        self.processed_buffer_count = processed_buffer_count  # multiprocessing.Value
        self.raw_buffer_full_flag = raw_buffer_full_flag  # multiprocessing.Value
        self.processed_buffer_full_flag = processed_buffer_full_flag  # multiprocessing.Value
        self.process_flag = process_flag  # multiprocessing.Value

        # multiprocessing.shared_memory implementation based on:
        # https://medium.com/@sampsa.riikonen/doing-python-multiprocessing-the-right-way-a54c1880e300
        # https://github.com/elsampsa/medium/blob/main/multiprocess/example2.py
        #self.name = "PlotterProcess"
        self.shared_ring_buffer_raw = None
        self.shared_ring_buffer_processed = None

        self.MAX_NUM_GRID_CELLS = self.settings['buffer_settings']['maxGridCells']
        self.MAX_LENGTH_BUFFER = self.settings['buffer_settings']['maxBufferSize_ping']
        self.ALONG_TRACK_PINGS = self.settings['processing_settings']['alongTrackAvg_ping']

        # TODO: Add constants to setting dictionary?
        self.QUEUE_RX_TIMEOUT = 60  # Seconds

        self.PIE_VMIN = -95
        self.PIE_VMAX = 10
        self.PLOT_UPDATE_INTERVAL = 1000  # Milliseconds

        # VERTICAL SLICE:
        # Trim arrays to omit values outside of self.vertical_slice_width_m
        # start_index       end_index
        #          |X|_|_|_|X|
        #          |X|_|_|_|X|
        #          |X|_|_|_|X|
        #          |X|_|_|_|X|

        self.vertical_slice_start_index = None
        self.vertical_slice_end_index = None
        self.set_vertical_indices()

        # HORIZONTAL SLICE:
        # Trim arrays to omit values outside of self.horizontal_slice_width_m
        # |X|X|X|X|X| start_index
        # |_|_|_|_|_|
        # |_|_|_|_|_|
        # |X|X|X|X|X| end_index
        self.horizontal_slice_start_index = None
        self.horizontal_slice_end_index = None
        self.set_horizontal_indices()

        self.collapse_times = []  # For testing

    def update_local_settings(self):
        # print("^^^^^^^ Plotter UPDATE LOCAL SETTINGS")
        with self.settings_edited.get_lock():  # Outer lock to ensure atomicity of updates:
            with self.bin_size.get_lock():
                self.bin_size_local = self.bin_size.value
            with self.across_track_avg.get_lock():
                self.across_track_avg_local = self.across_track_avg.value
            with self.depth.get_lock():
                self.depth_local = self.depth.value
            with self.depth_avg.get_lock():
                self.depth_avg_local = self.depth_avg.value
            # with self.along_track_avg.get_lock():
            #     self.along_track_avg_local = self.along_track_avg.value
            with self.max_heave.get_lock():
                self.max_heave_local = self.max_heave.value

    def set_vertical_indices(self):
        # TODO: Double check that this is calculated correctly
        # self.vertical_slice_start_index = math.floor((self.MAX_NUM_GRID_CELLS / 2) -
        #                                              ((self.settings['processing_settings']['acrossTrackAvg_m'] / 2) /
        #                                               self.settings['processing_settings']['binSize_m']))
        # self.vertical_slice_end_index = math.ceil((self.MAX_NUM_GRID_CELLS / 2) +
        #                                           ((self.settings['processing_settings']['acrossTrackAvg_m'] / 2) /
        #                                            self.settings["processing_settings"]["binSize_m"]))

        # self.vertical_slice_start_index = math.floor((self.MAX_NUM_GRID_CELLS / 2) -
        #                                              ((self.across_track_avg.value / 2) /
        #                                               self.bin_size.value))
        # self.vertical_slice_end_index = math.ceil((self.MAX_NUM_GRID_CELLS / 2) +
        #                                           ((self.across_track_avg.value / 2) /
        #                                            self.bin_size.value))

        self.vertical_slice_start_index = math.floor((self.MAX_NUM_GRID_CELLS / 2) -
                                                     ((self.across_track_avg_local / 2) /
                                                      self.bin_size_local))
        self.vertical_slice_end_index = math.ceil((self.MAX_NUM_GRID_CELLS / 2) +
                                                  ((self.across_track_avg_local / 2) /
                                                   self.bin_size_local))

    def set_horizontal_indices(self):
        # TODO: Double check that this is calculated correctly
        # First, find index of zero depth: this is found by dividing max_heave by bin_size
        # Then, find index of desired depth of slice: this is found by dividing horizontal_slice_depth_m by bin_size
        # Add the above two values to get true index of desired depth.
        # Then, find number of bins that must be included to achieve horizontal_slice_width_m above / below index of
        # desired depth: this is found by dividing horizontal_slice_width_m by 2 and dividing again by bin_size.
        # self.horizontal_slice_start_index = math.ceil(self.settings["processing_settings"]["maxHeave_m"] /
        #                                               self.settings["processing_settings"]["binSize_m"]) + \
        #                                     math.floor(self.settings["processing_settings"]["depth_m"] /
        #                                                self.settings["processing_settings"]["binSize_m"]) - \
        #                                     math.ceil((self.settings["processing_settings"]["depthAvg_m"] / 2) /
        #                                               self.settings["processing_settings"]["binSize_m"])
        # self.horizontal_slice_end_index = math.ceil(self.settings["processing_settings"]["maxHeave_m"] /
        #                                             self.settings["processing_settings"]["binSize_m"]) + \
        #                                   math.floor(self.settings["processing_settings"]["depth_m"] /
        #                                              self.settings["processing_settings"]["binSize_m"]) + \
        #                                   math.ceil((self.settings["processing_settings"]["depthAvg_m"] / 2) /
        #                                             self.settings["processing_settings"]["binSize_m"])

        # self.horizontal_slice_start_index = math.ceil(self.max_heave.value / self.bin_size.value) + \
        #                                     math.floor(self.depth.value / self.bin_size.value) - \
        #                                     math.ceil((self.depth_avg.value / 2) / self.bin_size.value)
        # self.horizontal_slice_end_index = math.ceil(self.max_heave.value / self.bin_size.value) + \
        #                                   math.floor(self.depth.value / self.bin_size.value) + \
        #                                   math.ceil((self.depth_avg.value / 2) / self.bin_size.value)

        self.horizontal_slice_start_index = math.ceil(self.max_heave_local / self.bin_size_local) + \
                                            math.floor(self.depth_local / self.bin_size_local) - \
                                            math.ceil((self.depth_avg_local / 2) / self.bin_size_local)
        self.horizontal_slice_end_index = math.ceil(self.max_heave_local / self.bin_size_local) + \
                                          math.floor(self.depth_local / self.bin_size_local) + \
                                          math.ceil((self.depth_avg_local / 2) / self.bin_size_local)

    # def get_and_plot_pie(self):
    #     print("DGPlot: get_and_plot_pie")  # For debugging
    #     self.start_time = datetime.datetime.now()
    #
    #     threading.Thread(target=self.get_and_buffer_pie, daemon=True).start()
    #
    #     # print("qtimer start")
    #     # timer = QTimer()
    #     # timer.timeout.connect(self.update)
    #     # timer.start(self.PLOT_UPDATE_INTERVAL)
    #     # print("qtimer end")
    #
    #     # self.animation = anim.FuncAnimation(self.fig, self.__animate, fargs=(),
    #     #                                     interval=self.PLOT_UPDATE_INTERVAL)
    #
    #     #plt.show(block=True)
    #     pass

    # def retrieve_plot_matrices(self):
    #     # For testing:
    #     # self.plot_count += 1
    #     # print("Plot count: ", self.plot_count)
    #     # print("Animate timestamp: ", datetime.datetime.now())
    #
    #     pie_display = []
    #     #self._lock_raw_buffers.acquire()
    #     try:
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             # Quick method of averaging!
    #             # TODO: This will give the most recent 'pie' in the buffer with no averaging--is that what we want?
    #             pie_display = np.array(self.pie_values_buffer.peek() / self.pie_count_buffer.peek())
    #     except IndexError:
    #         logger.warning("Excepted IndexError in retrieving values from buffer.")
    #     finally:
    #         #self._lock_raw_buffers.release()
    #         pass
    #
    #     vertical_slice = []
    #     horizontal_slice = []
    #     timestamp_slice = []
    #     lat_lon_slice = []
    #
    #     # For testing:
    #     start = datetime.datetime.now()
    #
    #     #self._lock_slice_buffers.acquire()  # NOTE: Time to acquire this lock is pretty much always zero (0019 test file).
    #
    #     try:  # NOTE: Time to copy these data structures is pretty much always zero (0019 test file).
    #         # TODO: This will make a new temporary object--is that what we want? Is it necessary?
    #         # vertical_slice = np.array(self.vertical_slice_buffer).astype(np.float32)
    #         vertical_slice = np.array(self.vertical_slice_buffer).__array__(np.float32)
    #         horizontal_slice = np.array(self.horizontal_slice_buffer).__array__(np.float32)
    #         timestamp_slice = np.array(self.timestamp_slice_buffer)
    #         lat_lon_slice = np.array(self.lat_lon_slice_buffer)
    #     finally:
    #         #self._lock_slice_buffers.release()
    #         pass
    #
    #     # For testing:
    #     end = datetime.datetime.now()
    #     diff = (end - start).total_seconds()
    #     print("Time to copy data structures: ", diff)  # This always appears to be zero.
    #
    #     if np.any(vertical_slice) and np.any(horizontal_slice):
    #         # For testing:
    #         print("len(vertical_slice) at animate update: ", len(vertical_slice))
    #
    #         # Trim NaNs from matrices to be plotted:
    #         # This method will look for the index of the last row that is not completely filled with NaNs.
    #         # Add one to that index for the first full row of NaNs after all data.
    #         index_pie_display = np.argwhere(~np.isnan(pie_display).all(axis=1))[-1][0] + 1
    #         index_vertical_slice = np.argwhere(~np.isnan(vertical_slice).all(axis=0))[-1][0] + 1
    #         # TODO:
    #         # Minus one to find first full row of NaNs before data.
    #
    #         # index_horizontal_slice_a = np.argwhere(~np.isnan(horizontal_slice).all(axis=0))[0][0] - 1
    #         # index_horizontal_slice_b = np.argwhere(~np.isnan(horizontal_slice).all(axis=0))[-1][0] + 1
    #
    #         # Ensure that 'index' plus some small buffer does not exceed grid size.
    #         # (Because we want to allow some small buffer around bottom of data if possible.)
    #         index_pie_display = min((index_pie_display + 10), self.MAX_NUM_GRID_CELLS)
    #         index_vertical_slice = min((index_vertical_slice + 10), self.MAX_NUM_GRID_CELLS)
    #         # TODO:
    #         # index_horizontal_slice_a = max((index_horizontal_slice_a - 10), 0)
    #         # index_horizontal_slice_b = min((index_horizontal_slice_b + 10), self.MAX_NUM_GRID_CELLS)
    #
    #         print("****************verticalslice.shape: ", vertical_slice[:index_vertical_slice, :].shape)
    #         print("len(verticalslice): ", len(vertical_slice[:index_vertical_slice, :]))
    #
    #         return pie_display[:][:index_pie_display].T, \
    #                vertical_slice[:index_vertical_slice, :], horizontal_slice
    #                # horizontal_slice[index_horizontal_slice_a:index_horizontal_slice_b, :]
    #
    #         # # Update plots:
    #         # self.ax_pie.clear()
    #         # self.ax_vert.clear()
    #         # self.ax_horiz.clear()
    #         # print("*****PLOTTING*****")
    #         # self.ax_pie.imshow(pie_display[:][:index_pie_display], cmap='gray',
    #         #                    vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
    #         # # TODO: NOTE: matplotlib gives "unsupported dtype" error with np.float16; use np.float32.
    #         # # self.ax_vert.imshow(vertical_slice, cmap='gray',
    #         # #                     vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
    #         # self.ax_vert.imshow(vertical_slice.T[:index_vertical_slice, :], cmap='gray',
    #         #                     vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
    #         # self.ax_horiz.imshow(horizontal_slice.T[index_horizontal_slice_a:index_horizontal_slice_b, :], cmap='gray',
    #         #                      vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
    #         # plt.draw()
    #         # plt.pause(0.001)
    #
    #     else:
    #         logger.warning("Nothing to plot; water column data matrix buffer is empty.")

    def get_sliced_matrices(self):
        pass

    def get_pie_matrix(self):
        pass

    def get_vertical_slice(self):
        pass

    def get_horizontal_slice(self):
        pass

    def get_and_buffer_pie(self):
        # TODO: Should inserting values into shared_memory buffers be done in KongsbergDGProcess?
        #  Or is queueing PieObjects best for compatibility with other systems?

        count_temp = 0

        temp_pie_values = []
        temp_pie_count = []
        temp_timestamp = []
        temp_lat_lon = []

        # while self.process_flag.value:
        while True:
            # Check for signal to end loop / exit:
            with self.process_flag.get_lock():
                if not self.process_flag.value:
                    break

            # Check for signal to update settings:
            with self.settings_edited.get_lock():
                if self.settings_edited.value:
                    # print("In plotter. settings_edited is true")
                    self.update_local_settings()
                    self.set_vertical_indices()
                    self.set_horizontal_indices()
                    self.settings_edited.value = False

            try:
                # print("&&&&&& PLOTTER DEPTH: ", self.depth_local)
                # print("trying get item from queue_pie_object, size: ", self.queue_pie_object.qsize())
                pie_object = self.queue_pie_object.get(block=True, timeout=self.QUEUE_RX_TIMEOUT)
                # print("got item from queue_pie_object")

                with self.raw_buffer_count.get_lock():
                    self.shared_ring_buffer_raw.append_all([pie_object.pie_chart_values],
                                                           [pie_object.pie_chart_count],
                                                           [pie_object.timestamp],
                                                           [(pie_object.latitude, pie_object.longitude)])

                    #print("self.raw_buffer_count: ", self.raw_buffer_count.value)

                    count_temp += 1

                    if count_temp == self.ALONG_TRACK_PINGS:
                        temp_pie_values = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                            self.shared_ring_buffer_raw.amplitude_buffer, self.ALONG_TRACK_PINGS))
                        temp_pie_count = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                            self.shared_ring_buffer_raw.count_buffer, self.ALONG_TRACK_PINGS))
                        temp_timestamp = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                            self.shared_ring_buffer_raw.timestamp_buffer, self.ALONG_TRACK_PINGS))
                        temp_lat_lon = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                            self.shared_ring_buffer_raw.lat_lon_buffer, self.ALONG_TRACK_PINGS))

                # Release lock

                if count_temp == self.ALONG_TRACK_PINGS:
                    # threading.Thread(target=self.collapse_and_buffer_pings,
                    #                  args=(temp_pie_values, temp_pie_count, temp_timestamp, temp_lat_lon)).start()
                    self.collapse_and_buffer_pings(temp_pie_values, temp_pie_count, temp_timestamp, temp_lat_lon)
                    count_temp = 0

            except queue.Empty as e:
                # TODO: Shutdown processes when queue is empty?
                logger.exception("Datagram queue empty exception.")
                break

            # if self.queue_pie_object.qsize() == 0:  # For testing
            #     end_rx_time = datetime.datetime.now()
            #     diff = (end_rx_time - start_rx_time).total_seconds()
            # print("DGPLOT, time to deque {} MWC plots: {}".format(test_count, diff))

        # print("TIME TO DEQUE ALL ITEMS IN QUEUE: {}".format(self.start_time - datetime.datetime.now()))
        self.closeSharedMemory()

    def collapse_and_buffer_pings(self, temp_pie_values, temp_pie_count, temp_timestamp, temp_lat_lon):
        # For testing:
        start = datetime.datetime.now()
        # print("collapse_and_buffer timestamp: ", start)

        if np.any(temp_pie_values) and np.any(temp_pie_count):
            #print("Collapse buffer")

            # VERTICAL SLICE:
            # Trim arrays to omit values outside of self.vertical_slice_width_m
            # start_index       end_index
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            pie_values_vertical = temp_pie_values[:, :, self.vertical_slice_start_index:self.vertical_slice_end_index]
            pie_count_vertical = temp_pie_count[:, :, self.vertical_slice_start_index:self.vertical_slice_end_index]

            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average = 1
            pie_values_vertical = np.sum(pie_values_vertical, axis=0)
            pie_count_vertical = np.sum(pie_count_vertical, axis=0)

            # Sum rows of matrices:
            pie_values_vertical = np.sum(pie_values_vertical, axis=1)
            pie_count_vertical = np.sum(pie_count_vertical, axis=1)

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_values_vertical_average = pie_values_vertical / pie_count_vertical

            # HORIZONTAL SLICE:
            # Trim arrays to omit values outside of self.horizontal_slice_width_m
            # |X|X|X|X|X| start_index
            # |_|_|_|_|_|
            # |_|_|_|_|_|
            # |X|X|X|X|X| end_index
            # TODO:
            #self.set_horizontal_indices()

            pie_values_horizontal = temp_pie_values[:, self.horizontal_slice_start_index:
                                                       self.horizontal_slice_end_index, :]
            pie_count_horizontal = temp_pie_count[:, self.horizontal_slice_start_index:
                                                     self.horizontal_slice_end_index, :]

            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average = 1
            pie_values_horizontal = np.sum(pie_values_horizontal, axis=0)
            pie_count_horizontal = np.sum(pie_count_horizontal, axis=0)

            # Sum columns of matrices:
            pie_values_horizontal = np.sum(pie_values_horizontal, axis=0)
            pie_count_horizontal = np.sum(pie_count_horizontal, axis=0)
            # print("pie_values.shape after sum rows:", pie_values_vertical.shape)
            # print("pie_count.shape after sum rows:", pie_count_vertical.shape)

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_values_horizontal_average = pie_values_horizontal / pie_count_horizontal
        else:
            logger.warning("Water column data matrix buffers are empty.")
            pie_values_vertical_average = pie_values_horizontal_average = np.empty((self.MAX_NUM_GRID_CELLS))
            pie_values_vertical_average[:] = np.nan

        # TIMESTAMP:
        if np.any(temp_timestamp):
            # print("$$$$$$$$$$$$$$$$$$$$TEMP TIMESTAMP: ", temp_timestamp)
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            pie_timestamp = np.sum(temp_timestamp)
            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_timestamp_average = pie_timestamp / self.settings["processing_settings"]["alongTrackAvg_ping"]
                # print("$$$$$$$$$$$$$$$$$$$$TEMP TIMESTAMP AVG: ", pie_timestamp_average)
        else:
            logger.warning("Water column timestamp matrix buffer is empty.")
            pie_timestamp_average = np.empty(1)
            pie_timestamp_average[:] = np.nan

        # LAT LON:
        if np.any(temp_lat_lon):
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            pie_lat_lon = np.sum(temp_lat_lon, axis=0)
            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_lat_lon_average = pie_lat_lon / self.settings["processing_settings"]["alongTrackAvg_ping"]
        else:
            logger.warning("Nothing to plot; water column latitude / longitude matrix buffer is empty.")
            pie_lat_lon_average = np.empty(2)
            pie_lat_lon_average[:] = np.nan

        # print("appending data to processed buffer")
        with self.processed_buffer_count.get_lock():
            self.shared_ring_buffer_processed.append_all([pie_values_vertical_average], [pie_values_horizontal_average],
                                              [pie_timestamp_average], [pie_lat_lon_average])

        # For testing:
        # end = datetime.datetime.now()
        # diff = (end - start).total_seconds()
        # self.collapse_times.append(diff)
        # print("Time for single collapse: ", diff)
        # print("Min collapse time: ", min(self.collapse_times))
        # print("Max collapse time: ", max(self.collapse_times))
        # print("Average collapse time: ", sum(self.collapse_times) / len(self.collapse_times))

    def closeSharedMemory(self):
        self.shared_ring_buffer_raw.close_shmem()
        self.shared_ring_buffer_processed.close_shmem()

    def unlinkSharedMemory(self):
        self.shared_ring_buffer_raw.unlink_shmem()
        self.shared_ring_buffer_processed.unlink_shmem()

    def run(self):
        self.shared_ring_buffer_raw = SharedRingBufferRaw(self.settings, self.raw_buffer_count,
                                             self.raw_buffer_full_flag, create_shmem=False)

        self.shared_ring_buffer_processed = SharedRingBufferProcessed(self.settings, self.processed_buffer_count,
                                                         self.processed_buffer_full_flag, create_shmem=False)

        self.get_and_buffer_pie()