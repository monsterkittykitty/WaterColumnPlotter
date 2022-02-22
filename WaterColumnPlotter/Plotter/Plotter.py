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
from WaterColumnPlotter.Plotter.SharedRingBufferProcessed import SharedRingBufferProcessed
from WaterColumnPlotter.Plotter.SharedRingBufferRaw import SharedRingBufferRaw
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
        self.along_track_avg = along_track_avg
        self.max_heave = max_heave

        self.settings_edited = settings_edited

        # To be set to True when bin_size or max_heave is edited
        self.bin_size_edited = False
        self.max_heave_edited = False
        # Need to maintain record of 'old' heave when max_heave is updated
        self.outdated_heave = None

        # Local copies of above multiprocessing.Values (to avoid frequent accessing of locks)
        self.bin_size_local = None
        self.across_track_avg_local = None
        self.depth_local = None
        self.depth_avg_local = None
        self.along_track_avg_local = None
        self.max_heave_local = None
        # Initialize above local copies
        # self.update_local_settings()

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
        # Protected with self.raw_buffer_count lock:
        # self.shared_ring_buffer_raw = SharedRingBufferRaw(self.settings, self.raw_buffer_count,
        #                                                   self.raw_buffer_full_flag, create_shmem=False)
        #
        # # Protected with self.processed_buffer_count lock:
        # self.shared_ring_buffer_processed = SharedRingBufferProcessed(self.settings, self.processed_buffer_count,
        #                                                               self.processed_buffer_full_flag,
        #                                                               create_shmem=False)

        # TODO: Make this a multiprocessing Value?
        self.MAX_NUM_GRID_CELLS = self.settings['buffer_settings']['maxGridCells']
        # self.MAX_LENGTH_BUFFER = self.settings['buffer_settings']['maxBufferSize_ping']
        # self.ALONG_TRACK_PINGS = self.settings['processing_settings']['alongTrackAvg_ping']

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

        # HORIZONTAL SLICE:
        # Trim arrays to omit values outside of self.horizontal_slice_width_m
        # |X|X|X|X|X| start_index
        # |_|_|_|_|_|
        # |_|_|_|_|_|
        # |X|X|X|X|X| end_index
        self.horizontal_slice_start_index = None
        self.horizontal_slice_end_index = None

        self.initialization_flag = False
        # Initialize local copies and vertical and horizontal indices
        self.update_local_settings()
        # self.set_vertical_indices()
        # self.set_horizontal_indices()

        self.counter = 0

        self.collapse_times = []  # For testing

    def update_local_settings(self):
        with self.settings_edited.get_lock():  # Outer lock to ensure atomicity of update--I don't think I actually need this as long as I hold appropriate locks when calling this method:
            with self.bin_size.get_lock():
                print("update_local_settings: ", self.bin_size_local, ",", self.bin_size.value)
                if self.bin_size_local and round(self.bin_size_local, 2) != round(self.bin_size.value, 2):
                    # bin_size edits cannot be applied retroactively;
                    # when this value changes, ring_buffers must be cleared
                    print("setting bin_size_edited to true")
                    self.bin_size_edited = True
                self.bin_size_local = self.bin_size.value
            with self.across_track_avg.get_lock():
                self.across_track_avg_local = self.across_track_avg.value
            with self.depth.get_lock():
                self.depth_local = self.depth.value
            with self.depth_avg.get_lock():
                self.depth_avg_local = self.depth_avg.value
            with self.along_track_avg.get_lock():
                self.along_track_avg_local = self.along_track_avg.value
            with self.max_heave.get_lock():
                if self.max_heave_local and self.max_heave_local != self.max_heave.value:
                    # max_heave edits can be applied retroactively
                    self.max_heave_edited = True
                    self.outdated_heave = self.max_heave_local
                self.max_heave_local = self.max_heave.value

            self.set_vertical_indices()
            self.set_horizontal_indices()

    def shift_heave(self, amplitude_buffer, count_buffer, old_heave, new_heave):
        print("shift heave")
        # NOTE: This method will only be called if self.bin_size_local has not changed. When self.bin_size_local
        # changes, both raw and processed buffers will be cleared; no need to recalculate heave.
        num_bins_old_heave = int(round(old_heave, 2) / round(self.bin_size_local, 2))
        num_bins_new_heave = int(round(new_heave, 2) / round(self.bin_size_local, 2))
        # Negative indicates reducing heave allotment;
        # positive indicates increasing heave allotment
        num_bins_adjustment = num_bins_new_heave - num_bins_old_heave

        # Adjustment method based on (shift5):
        # https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array

        if num_bins_adjustment > 0:  # Increase heave allotment
            if len(amplitude_buffer.shape) == 2:
                # Shift data downward to accommodate extra heave cells at top of matrix
                amplitude_buffer[num_bins_adjustment:] = amplitude_buffer[:-num_bins_adjustment]
                count_buffer[num_bins_adjustment:] = count_buffer[:-num_bins_adjustment]
                # Fill extra heave cells at top of matrix with zero
                amplitude_buffer[:num_bins_adjustment] = 0
                count_buffer[:num_bins_adjustment] = 0

            elif len(amplitude_buffer.shape) == 3:
                # Shift data downward to accommodate extra heave cells at top of matrix
                amplitude_buffer[:, num_bins_adjustment:, :] = amplitude_buffer[:, :-num_bins_adjustment, :]
                count_buffer[:, num_bins_adjustment:, :] = count_buffer[:, :-num_bins_adjustment, :]
                # Fill extra heave cells at top of matrix with zero
                amplitude_buffer[:, :num_bins_adjustment, :] = 0
                count_buffer[:, :num_bins_adjustment, :] = 0
            else:
                logger.warning("Error in Plotter.py, shift_heave() method. Invalid buffer shape: {}."
                               .format(amplitude_buffer.shape))

        elif num_bins_adjustment < 0:  # Decrease heave allotment
            if len(amplitude_buffer.shape) == 2:
                # Shift data upward to accommodate fewer heave cells at top of matrix
                amplitude_buffer[:num_bins_adjustment] = amplitude_buffer[-num_bins_adjustment:]
                count_buffer[:num_bins_adjustment] = count_buffer[-num_bins_adjustment:]
                # Fill extra cells at bottom of matrix with zero
                amplitude_buffer[num_bins_adjustment:] = 0
                count_buffer[num_bins_adjustment:] = 0

            elif len(amplitude_buffer.shape) == 3:
                # Shift data upward to accommodate fewer heave cells at top of matrix
                amplitude_buffer[:, :num_bins_adjustment, :] = amplitude_buffer[:, -num_bins_adjustment:, :]
                count_buffer[:, :num_bins_adjustment, :] = count_buffer[:, -num_bins_adjustment:, :]
                # Fill extra cells at bottom of matrix with zero
                amplitude_buffer[:, num_bins_adjustment:, :] = 0
                count_buffer[:, num_bins_adjustment:, :] = 0
            else:
                logger.warning("Error in Plotter.py, shift_heave() method. Invalid buffer shape: {}."
                               .format(amplitude_buffer.shape))

    def set_vertical_indices(self):
        # TODO: Double check that this is calculated correctly
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

    def get_and_buffer_pie(self):

        count_temp = 0

        temp_pie_values = []
        temp_pie_count = []
        temp_timestamp = []
        temp_lat_lon = []

        while True:
            # print("at top of plotter while loop")

            # Check for signal to play / pause / stop:
            with self.process_flag.get_lock():
                local_process_flag_value = self.process_flag.value

            try:
                print("in plotter, getting pie object: ", self.queue_pie_object.qsize())
                pie_object = self.queue_pie_object.get(block=True, timeout=self.QUEUE_RX_TIMEOUT)

                if pie_object:  # pie_object will be of type DGPie if valid record, or type None if poison pill
                    print("in plotter; got pie object")
                    if local_process_flag_value == 1 or local_process_flag_value == 2:  # Play pressed or pause pressed

                        # TODO: 2/3 New attempt
                        print("in plotter, getting raw buffer lock")
                        with self.shared_ring_buffer_raw.get_lock():
                            print("in plotter, got raw buffer lock")
                            print("in plotter; checking settings_edited.value: ", self.settings_edited.value)
                            with self.settings_edited.get_lock():
                                print("in plotter; checking settings_edited.value: ", self.settings_edited.value)
                                if self.settings_edited.value:  # If settings are edited...
                                    print("in plotter; updating local settings")
                                    self.update_local_settings()
                                    # if not self.bin_size_edited:
                                    #     print("in plotter; recalculating processed buffer")
                                    #     # Called with self.settings_edited lock held to ensure atomicity
                                    #     self.recalculate_processed_buffer(self.shared_ring_buffer_raw,
                                    #                                       self.shared_ring_buffer_processed)
                                    # else:
                                    #     self.shared_ring_buffer_raw.clear()
                                    #     self.shared_ring_buffer_processed.clear()
                                    self.settings_edited.value = False

                            # if not self.bin_size_edited:
                            #     self.recalculate_processed_buffer()

                            # If self.bin_size_edited is True, raw and processed ring buffers have already been cleared in
                            # update_local_settings() call. We only need to empty queue_pie_object of outdated pie_objects.
                            # We DO NOT need to call self.recalculate_processed_buffer, as buffers are empty.
                            print("in plotter; checking bin_size_edited: ", self.bin_size_edited)
                            if self.bin_size_edited:
                                if round(pie_object.bin_size, 2) != round(self.bin_size_local, 2):
                                    # If the current pie_object contains a record processed with the 'old' bin_size,
                                    # do not process it--discard it and get another from the queue
                                    print(round(pie_object.bin_size, 2), round(self.bin_size_local, 2))
                                    print("continue called")
                                    continue  # Return to start of while loop
                                else:
                                    # If the current pie_object contains a record processed with the 'new' bin_size...
                                    count_temp = 0  # ...Reset count...
                                    print("setting bin_size_edited to false")
                                    self.bin_size_edited = False  # ...Reset self.bin_size_edited...
                                    # ...And continue to process pie_object as usual.
                            # else:
                            #     self.recalculate_processed_buffer()


                            # If self.max_heave_edited is True, raw and processed ring buffers have already been adjusted
                            # in update_local_settings() call. We only need to monitor queue_pie_object for outdated
                            # pie_objects and adjust them accordingly.
                            print("in plotter; checking max_heave_edited: ", self.max_heave_edited)
                            if self.max_heave_edited:
                                print("MAX HEAVE IS EDITED")
                                if round(pie_object.max_heave, 2) != round(self.max_heave_local, 2):
                                    print("in plotter; shifting heave of individual pie object")
                                    self.shift_heave(pie_object.pie_chart_values, pie_object.pie_chart_count,
                                                     pie_object.max_heave, self.max_heave_local)
                                else:
                                    self.max_heave_edited = False

                            with self.raw_buffer_count.get_lock():
                                print("in plotter; adding pie objects to raw buffer")
                                self.shared_ring_buffer_raw.append_all([pie_object.pie_chart_values],
                                                                       [pie_object.pie_chart_count],
                                                                       [pie_object.timestamp],
                                                                       [(pie_object.latitude, pie_object.longitude)])

                                count_temp += 1

                                if count_temp == self.along_track_avg_local:
                                    temp_pie_values = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                        self.shared_ring_buffer_raw.amplitude_buffer, self.along_track_avg_local))
                                    temp_pie_count = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                        self.shared_ring_buffer_raw.count_buffer, self.along_track_avg_local))
                                    temp_timestamp = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                        self.shared_ring_buffer_raw.timestamp_buffer, self.along_track_avg_local))
                                    temp_lat_lon = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                        self.shared_ring_buffer_raw.lat_lon_buffer, self.along_track_avg_local))

                            # Release lock

                            if count_temp == self.along_track_avg_local:
                                print("collapsing and buffering pings. count: ", count_temp)
                                # threading.Thread(target=self.collapse_and_buffer_pings,
                                #                  args=(temp_pie_values, temp_pie_count, temp_timestamp, temp_lat_lon)).start()
                                self.collapse_and_buffer_pings(temp_pie_values, temp_pie_count, temp_timestamp, temp_lat_lon)
                                count_temp = 0

                    elif local_process_flag_value == 3:  # Stop pressed
                        # Do not process pie. Instead, only empty queue.
                        pass

                    else:  # local_process_flag_value is invalid
                        logger.error("Error in Plotter. Invalid process_flag value: {}."
                                     .format(local_process_flag_value))
                        break  # Exit loop

                else:
                    print("breaking out of plotter loop because pie_object is None")
                    break

            except queue.Empty as e:
                # TODO: Shutdown processes when queue is empty?
                logger.exception("Datagram queue empty exception.")
                break

        self.closeSharedMemory()

    def collapse_and_buffer_pings(self, temp_pie_values, temp_pie_count, temp_timestamp, temp_lat_lon):
        # For testing:
        start = datetime.datetime.now()
        # print("collapse_and_buffer timestamp: ", start)

        if np.any(temp_pie_values) and np.any(temp_pie_count):
            #print("Collapse buffer")

            # print("len(temp_pie_values):", len(temp_pie_values))
            # print("number of non-nan elements in index 0: ", np.count_nonzero(temp_pie_values[0]))
            # print("number of non-nan elements in index 1: ", np.count_nonzero(temp_pie_values[1]))
            # print("number of non-nan elements in index 2: ", np.count_nonzero(temp_pie_values[2]))
            # print("number of non-nan elements in index 3: ", np.count_nonzero(temp_pie_values[2]))
            # print("number of non-nan elements in index 4: ", np.count_nonzero(temp_pie_values[4]))

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



    def recalculate_processed_buffer(self, ring_buffer_raw, ring_buffer_processed):
        # pass
        print("recalculating processed buffer")
        # TODO: Is it better to collapse buffers first, then trim for vertical and horizontal slices?

        # along_track_avg_ping = self.settings['processing_settings']['alongTrackAvg_ping']
        # TODO: NEW
        with ring_buffer_raw.counter.get_lock():
            temp_amplitude_buffer = ring_buffer_raw.view_buffer_elements(ring_buffer_raw.amplitude_buffer)
            temp_count_buffer = ring_buffer_raw.view_buffer_elements(ring_buffer_raw.count_buffer)
            temp_timestamp_buffer = ring_buffer_raw.view_buffer_elements(ring_buffer_raw.timestamp_buffer)
            temp_lat_lon_buffer = ring_buffer_raw.view_buffer_elements(ring_buffer_raw.lat_lon_buffer)

        # TODO: OLD
        # with self.raw_buffer_count.get_lock():
        #     # Produces views of self.shared_ring_buffer_raw arrays
        #     temp_amplitude_buffer = self.shared_ring_buffer_raw.view_buffer_elements(
        #         self.shared_ring_buffer_raw.amplitude_buffer)
        #     temp_count_buffer = self.shared_ring_buffer_raw.view_buffer_elements(
        #         self.shared_ring_buffer_raw.count_buffer)
        #     temp_timestamp_buffer = self.shared_ring_buffer_raw.view_buffer_elements(
        #         self.shared_ring_buffer_raw.timestamp_buffer)
        #     temp_lat_lon_buffer = self.shared_ring_buffer_raw.view_buffer_elements(
        #         self.shared_ring_buffer_raw.lat_lon_buffer)

            assert len(temp_amplitude_buffer) == len(temp_count_buffer) \
                   == len(temp_timestamp_buffer) == len(temp_lat_lon_buffer)  # All buffers should be of equal length

            # Ensure that buffer can be divided evenly by along_track_avg_ping;
            # if not, discard remainder from end of buffer for following calculations
            indices_to_trim = len(temp_amplitude_buffer) % self.along_track_avg_local

            # TODO: For debugging:
            print("Shape before trim, temp_amplitude_buffer: {}; temp_count_buffer: {}; "
                  "temp_timestamp_buffer: {}; temp_lat_lon_buffer: {}"
                  .format(temp_amplitude_buffer.shape, temp_count_buffer.shape,
                          temp_timestamp_buffer.shape, temp_lat_lon_buffer.shape))

            # TODO: For debugging:
            print("indices_to_trim: {}".format(indices_to_trim))

            if indices_to_trim > 0:
                temp_amplitude_buffer = temp_amplitude_buffer[:-indices_to_trim]
                temp_count_buffer = temp_count_buffer[:-indices_to_trim]
                temp_timestamp_buffer = temp_timestamp_buffer[:-indices_to_trim]
                temp_lat_lon_buffer = temp_lat_lon_buffer[:-indices_to_trim]


            # TODO: For debugging:
            print("Shape after trim, temp_amplitude_buffer: {}; temp_count_buffer: {}; "
                  "temp_timestamp_buffer: {}; temp_lat_lon_buffer: {}"
                  .format(temp_amplitude_buffer.shape, temp_count_buffer.shape,
                          temp_timestamp_buffer.shape, temp_lat_lon_buffer.shape))







            # Take vertical slice, then collapse?
            # VERTICAL SLICE:
            # Trim arrays to omit values outside of self.vertical_slice_width_m
            # start_index       end_index
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            # This creates view of array
            temp_amplitude_vertical = temp_amplitude_buffer[:, :,
                                      self.vertical_slice_start_index:self.vertical_slice_end_index]
            temp_count_vertical = temp_count_buffer[:, :,
                                  self.vertical_slice_start_index:self.vertical_slice_end_index]

            # TODO: For debugging:
            print("Shape temp_amplitude_vertical: {}; temp_count_vertical: {}"
                  .format(temp_count_vertical.shape, temp_count_vertical.shape))

            # if indices_to_trim > 0:
            #     temp_amplitude_vertical = temp_amplitude_vertical[:-indices_to_trim]
            #     temp_count_vertical = temp_count_vertical[:-indices_to_trim]
            #
            # # TODO: For debugging:
            # print("Shape temp_amplitude_vertical_trimmed: {}; temp_count_vertical_trimmed: {}"
            #       .format(temp_count_vertical.shape, temp_count_vertical.shape))

            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average = 1
            # This creates copy of array
            temp_amplitude_vertical_collapsed = np.add.reduceat(temp_amplitude_vertical,
                                                                np.arange(0, len(temp_amplitude_vertical),
                                                                          self.along_track_avg_local))

            temp_count_vertical_collapsed = np.add.reduceat(temp_count_vertical,
                                                           np.arange(0, len(temp_count_vertical),
                                                                     self.along_track_avg_local))

            # TODO: For debugging:
            print("Shape temp_amplitude_vertical_collapsed: {}; temp_count_vertical_collapsed: {}"
                  .format(temp_count_vertical_collapsed.shape, temp_count_vertical_collapsed.shape))

            # Sum rows of matrices:
            # This creates copy of array
            amplitude_vertical = np.sum(temp_amplitude_vertical_collapsed, axis=2)
            count_vertical = np.sum(temp_count_vertical_collapsed, axis=2)

            # TODO: For debugging:
            print("Shape amplitude_vertical: {}; count_vertical: {}"
                  .format(amplitude_vertical.shape, count_vertical.shape))

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                vertical_average = amplitude_vertical / count_vertical

            # TODO: For debugging: OK!
            print("Shape vertical average: {}".format(vertical_average.shape))

            # Take horizontal slice, then collapse?
            # HORIZONTAL SLICE:
            # Trim arrays to omit values outside of self.horizontal_slice_width_m
            # |X|X|X|X|X| start_index
            # |_|_|_|_|_|
            # |_|_|_|_|_|
            # |X|X|X|X|X| end_index
            # This creates view of array
            temp_amplitude_horizontal = temp_amplitude_buffer[:, self.horizontal_slice_start_index:
                                                       self.horizontal_slice_end_index, :]
            temp_count_horizontal = temp_count_buffer[:, self.horizontal_slice_start_index:
                                                     self.horizontal_slice_end_index, :]

            # TODO: For debugging:
            print("Shape temp_amplitude_horizontal: {}; temp_count_horizontal: {}"
                  .format(temp_count_horizontal.shape, temp_count_horizontal.shape))

            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average = 1
            # This creates copy of array
            temp_amplitude_horizontal_collapsed = np.add.reduceat(temp_amplitude_horizontal,
                                                                  np.arange(0, len(temp_amplitude_horizontal),
                                                                            self.along_track_avg_local))

            temp_count_horizontal_collapsed = np.add.reduceat(temp_count_horizontal,
                                                              np.arange(0, len(temp_count_horizontal),
                                                                        self.along_track_avg_local))

            # TODO: For debugging:
            print("Shape temp_amplitude_horizontal_collapsed: {}; temp_count_horizontal_collapsed: {}"
                  .format(temp_count_horizontal_collapsed.shape, temp_count_horizontal_collapsed.shape))

            # Sum columns of matrices:
            # This creates copy of array
            amplitude_horizontal = np.sum(temp_amplitude_horizontal_collapsed, axis=1)
            count_horizontal = np.sum(temp_count_horizontal_collapsed, axis=1)

            # TODO: For debugging:
            print("Shape amplitude_horizontal: {}; count_horizontal: {}"
                  .format(amplitude_horizontal.shape, count_horizontal.shape))

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                horizontal_average = amplitude_horizontal / count_horizontal

            # TODO: For debugging: OK!
            print("Shape horizontal average: {}".format(horizontal_average.shape))









            # TIMESTAMP:
            # if np.any(temp_timestamp):
            # print("$$$$$$$$$$$$$$$$$$$$TEMP TIMESTAMP: ", temp_timestamp)
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            # timestamp = np.sum(temp_timestamp_buffer)

            # This creates copy of array
            timestamp_collapsed = np.add.reduceat(temp_timestamp_buffer,
                                                  np.arange(0, len(temp_timestamp_buffer), self.along_track_avg_local))

            # TODO: For debugging:
            print("Shape timestamp_collapsed: {}"
                  .format(timestamp_collapsed.shape))

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                timestamp_average = timestamp_collapsed / self.along_track_avg_local

            # TODO: For debugging:
            print("Shape timestamp average: {}".format(timestamp_average.shape))
                # print("$$$$$$$$$$$$$$$$$$$$TEMP TIMESTAMP AVG: ", pie_timestamp_average)
            # else:
            #     logger.warning("Water column timestamp matrix buffer is empty.")
            #     pie_timestamp_average = np.empty(1)
            #     pie_timestamp_average[:] = np.nan

            # LAT LON:
            # if np.any(temp_lat_lon):
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            # lat_lon = np.sum(temp_lat_lon_buffer, axis=0)

            # This creates copy of array
            lat_lon_collapsed = np.add.reduceat(temp_lat_lon_buffer,
                                                np.arange(0, len(temp_lat_lon_buffer), self.along_track_avg_local))

            # TODO: For debugging:
            print("Shape lat_lon_collapsed: {}"
                  .format(lat_lon_collapsed.shape))

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                lat_lon_average = lat_lon_collapsed / self.along_track_avg_local

            # TODO: For debugging:
            print("Shape lat_lon average: {}".format(lat_lon_average.shape))
            # else:
            #     logger.warning("Nothing to plot; water column latitude / longitude matrix buffer is empty.")
            #     pie_lat_lon_average = np.empty(2)
            #     pie_lat_lon_average[:] = np.nan

        # Release raw buffer lock

        # print("appending data to processed buffer")

        # TODO: NEW
        with ring_buffer_processed.counter.get_lock():
            ring_buffer_processed.clear_and_append_all(vertical_average, horizontal_average,
                                                       timestamp_average, lat_lon_average)

        # TODO: OLD
        # with self.processed_buffer_count.get_lock():
        #     self.shared_ring_buffer_processed.clear_and_append_all(vertical_average, horizontal_average,
        #                                                  timestamp_average, lat_lon_average)











            # amplitude_reduced = np.add.reduceat(temp_amplitude_buffer[:-indices_to_trim],
            #                                     np.arange(0, len(temp_amplitude_buffer), along_track_avg_ping))
            # count_reduced = np.add.reduceat(temp_count_buffer[:-indices_to_trim],
            #                                 np.arange(0, len(temp_count_buffer), along_track_avg_ping))
            # timestamp_reduced = np.add.reduceat(temp_timestamp_buffer[:-indices_to_trim],
            #                                     np.arange(0, len(temp_timestamp_buffer), along_track_avg_ping))
            # temp_lat_lon_buffer = np.add.reduceat(temp_lat_lon_buffer[:-indices_to_trim],
            #                                       np.arange(0, len(temp_lat_lon_buffer), along_track_avg_ping))




    def closeSharedMemory(self):
        self.shared_ring_buffer_raw.close_shmem()
        self.shared_ring_buffer_processed.close_shmem()

    def unlinkSharedMemory(self):
        self.shared_ring_buffer_raw.unlink_shmem()
        self.shared_ring_buffer_processed.unlink_shmem()

    def run(self):
        # Protected with self.raw_buffer_count lock:
        self.shared_ring_buffer_raw = SharedRingBufferRaw(self.settings, self.raw_buffer_count,
                                             self.raw_buffer_full_flag, create_shmem=False)

        # Protected with self.processed_buffer_count lock:
        self.shared_ring_buffer_processed = SharedRingBufferProcessed(self.settings, self.processed_buffer_count,
                                                         self.processed_buffer_full_flag, create_shmem=False)

        self.get_and_buffer_pie()