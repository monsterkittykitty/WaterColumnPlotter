# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description: Receives standard format pie records from shared multiprocessing.Queue;
# performs along-track, across-track, and depth slicing and averaging as specified by user settings.

import logging
import math
from multiprocessing import Process
import numpy as np
import queue
from WaterColumnPlotter.Plotter.SharedRingBufferProcessed import SharedRingBufferProcessed
from WaterColumnPlotter.Plotter.SharedRingBufferRaw import SharedRingBufferRaw

logger = logging.getLogger(__name__)


class Plotter(Process):
    def __init__(self, settings, bin_size, across_track_avg, depth, depth_avg, along_track_avg, max_heave,
                 settings_edited, queue_pie_object, raw_buffer_count, processed_buffer_count,
                 raw_buffer_full_flag, processed_buffer_full_flag, process_flag):
        super().__init__()

        self.settings = settings

        # multiprocessing.Values (shared between processes)
        self.bin_size = bin_size
        self.across_track_avg = across_track_avg
        self.depth = depth
        self.depth_avg = depth_avg
        self.along_track_avg = along_track_avg
        self.max_heave = max_heave

        self.settings_edited = settings_edited

        # To be set to True when bin_size or max_heave or along_track_avg is edited
        self.bin_size_edited = False
        self.max_heave_edited = False
        self.along_track_avg_edited = False
        # Need to maintain record of 'old' max_heave when updated
        self.outdated_heave = None
        # Need to maintain record of 'old' along_track_avg when updated
        self.outdated_along_track_avg = None

        # Local copies of above multiprocessing.Values (to avoid frequent accessing of locks)
        self.bin_size_local = None
        self.across_track_avg_local = None
        self.depth_local = None
        self.depth_avg_local = None
        self.along_track_avg_local = None
        self.max_heave_local = None

        self.queue_pie_object = queue_pie_object  # multiprocessing.Queue

        self.raw_buffer_count = raw_buffer_count  # multiprocessing.Value
        self.processed_buffer_count = processed_buffer_count  # multiprocessing.Value
        self.raw_buffer_full_flag = raw_buffer_full_flag  # multiprocessing.Value
        self.processed_buffer_full_flag = processed_buffer_full_flag  # multiprocessing.Value
        self.process_flag = process_flag  # multiprocessing.Value

        # multiprocessing.shared_memory implementation based on:
        # https://medium.com/@sampsa.riikonen/doing-python-multiprocessing-the-right-way-a54c1880e300
        # https://github.com/elsampsa/medium/blob/main/multiprocess/example2.py
        self.shared_ring_buffer_raw = None  # Protected with self.raw_buffer_count lock
        self.shared_ring_buffer_processed = None  # Protected with self.processed_buffer_count lock

        # TODO: Make this a multiprocessing Value?
        self.MAX_NUM_GRID_CELLS = self.settings['buffer_settings']['maxGridCells']
        self.QUEUE_RX_TIMEOUT = 60  # Seconds

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

        # Initialize local copies and vertical and horizontal indices
        self.update_local_settings()


    def update_local_settings(self):
        """
        At object initialization, this method initializes local copies of shared variables as well as vertical and
        horizontal slice indices; after initialization, this method updates local copies of shared variables and
        vertical and horizontal slice indices when settings are changed.
        """
        # Outer lock to ensure atomicity of updates; this lock must be held when updating settings.
        with self.settings_edited.get_lock():
            with self.bin_size.get_lock():
                if self.bin_size_local and round(self.bin_size_local, 2) != round(self.bin_size.value, 2):
                    # Bin size edits cannot be applied retroactively; when this value changes,
                    # set bin_size_edited flag to true to indicate that ring buffers must be cleared
                    self.bin_size_edited = True
                self.bin_size_local = self.bin_size.value
            with self.across_track_avg.get_lock():
                self.across_track_avg_local = self.across_track_avg.value
            with self.depth.get_lock():
                self.depth_local = self.depth.value
            with self.depth_avg.get_lock():
                self.depth_avg_local = self.depth_avg.value
            with self.along_track_avg.get_lock():
                if self.along_track_avg_local and self.along_track_avg_local != self.along_track_avg.value:
                    self.along_track_avg_edited = True
                    self.outdated_along_track_avg = self.along_track_avg_local
                self.along_track_avg_local = self.along_track_avg.value
            with self.max_heave.get_lock():
                if self.max_heave_local and self.max_heave_local != self.max_heave.value:
                    # Max heave edits can be applied retroactively; when this value changes, set max_heave_edited
                    # flag to true to indicate that further adjustments must be made and save outdated heave value
                    self.max_heave_edited = True
                    self.outdated_heave = self.max_heave_local
                self.max_heave_local = self.max_heave.value

            # Set vertical and horizontal indices for matrix slicing
            self.set_vertical_indices()
            self.set_horizontal_indices()

    def shift_heave(self, amplitude_buffer, count_buffer, old_heave, new_heave):
        """
        Shifts values in numpy amplitude_buffer and numpy count_buffer to accommodate changes in maximum heave settings.
        :param amplitude_buffer: Numpy matrix containing binned water column amplitude values for a single ping
        :param count_buffer: Numpy matrix containing binned water column counts for a single ping
        :param old_heave: Old heave value to be replaced
        :param new_heave: New heave value to be applied
        """
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
        """
        Sets starting and ending indices for vertical slices based on user settings.
        """
        self.vertical_slice_start_index = math.floor((self.MAX_NUM_GRID_CELLS / 2) -
                                                     ((self.across_track_avg_local / 2) /
                                                      self.bin_size_local))
        self.vertical_slice_end_index = math.ceil((self.MAX_NUM_GRID_CELLS / 2) +
                                                  ((self.across_track_avg_local / 2) /
                                                   self.bin_size_local))

    def set_horizontal_indices(self):
        """
        Sets starting and ending indices for horizontal slices based on user settings.
        """
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

    def get_and_buffer_pie(self):
        """
        Receives processed standard format pie objects from shared multiprocessing queue. (Note that queue's get()
        method is blocking, but does have a timeout.) Appends raw data to raw ring buffers in shared memory; slices and
        averages data according to user settings, and appends this data to processed ring buffers in shared memory.
        """
        # Counter to track number of records processed; triggers additional processing
        # when number of records processed reaches the number of pings to average along track
        # (as defined by user settings); counter is reset to zero after these additional actions are triggered.
        count_temp = 0

        # Temporarily stores copies of data for additional processing.
        temp_pie_amplitudes = []
        temp_pie_counts = []
        temp_timestamp = []
        temp_lat_lon = []

        while True:
            # Check for signal to play / pause / stop:
            with self.process_flag.get_lock():
                local_process_flag_value = self.process_flag.value

            try:
                print("plotter, getting pie object: count_temp: {}, self.along-track-avg-local: {}".format(count_temp, self.along_track_avg_local))
                pie_object = self.queue_pie_object.get(block=True, timeout=self.QUEUE_RX_TIMEOUT)
                print("plotter, got pie object")

                if pie_object:  # pie_object will be of type DGPie if valid record, or type None if poison pill
                    if local_process_flag_value == 1 or local_process_flag_value == 2:  # Play pressed or pause pressed

                        with self.shared_ring_buffer_raw.get_lock():
                            with self.settings_edited.get_lock():
                                if self.settings_edited.value:  # If settings are edited...
                                    self.update_local_settings()
                                    self.settings_edited.value = False

                            # If self.bin_size_edited is True, raw and processed ring buffers will have already
                            # been cleared. We only need to empty queue_pie_object of outdated pie_objects.
                            # We DO NOT need to call self.recalculate_processed_buffer, as buffers are empty.
                            if self.bin_size_edited:
                                if round(pie_object.bin_size, 2) != round(self.bin_size_local, 2):
                                    # If the current pie_object contains a record processed with the 'old' bin_size,
                                    # do not process it--discard it and get another from the queue
                                    continue  # Return to start of while loop
                                else:
                                    # If the current pie_object contains a record processed with the 'new' bin_size...
                                    count_temp = 0  # ...Reset count...
                                    self.bin_size_edited = False  # ...Reset self.bin_size_edited...
                                    # ...And continue to process pie_object as usual.

                            # If self.max_heave_edited is True, raw and processed ring buffers will have already
                            # been adjusted. We only need to monitor queue_pie_object for outdated pie_objects
                            # and adjust them accordingly.
                            if self.max_heave_edited:
                                print("####################In plotter, max_heave_edited is True.")
                                if round(pie_object.max_heave, 2) != round(self.max_heave_local, 2):
                                    self.shift_heave(pie_object.pie_chart_amplitudes, pie_object.pie_chart_counts,
                                                     pie_object.max_heave, self.max_heave_local)
                                else:
                                    print("####################In plotter, max_heave_edited is False.")
                                    self.max_heave_edited = False

                            # If self.along_track_avg_edited is True, processed ring buffer will have already been
                            # adjusted. We need to know the 'remainder' of items in the raw ring buffer than were not
                            # included in calculations for the revised processed ring buffer.
                            print("self.along_track_avg_edited: ", self.along_track_avg_edited)
                            if self.along_track_avg_edited:
                                print("self.shared_ring_buffer_raw.get_num_elements_in_buffer(): ", self.shared_ring_buffer_raw.get_num_elements_in_buffer())
                                print("self.shared_ring_buffer_processed.get_num_elements_in_buffer(): ",
                                      self.shared_ring_buffer_processed.get_num_elements_in_buffer())
                                print("count temp before change: ", count_temp)
                                # Note: Lock for this buffer is already held
                                count_temp = self.shared_ring_buffer_raw.get_num_elements_in_buffer() % \
                                             self.along_track_avg_local
                                print("count temp after change: ", count_temp)

                            # with self.raw_buffer_count.get_lock():
                            # Add raw data to raw ring buffer in shared memory
                            self.shared_ring_buffer_raw.append_all([pie_object.pie_chart_amplitudes],
                                                                   [pie_object.pie_chart_counts],
                                                                   [pie_object.timestamp],
                                                                   [(pie_object.latitude, pie_object.longitude)])
                            # Increment count_temp
                            count_temp += 1

                            if count_temp == self.along_track_avg_local:
                                # Make temporary copies of last along_track_avg_local
                                # number of pings for further processing:
                                temp_pie_amplitudes = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                    self.shared_ring_buffer_raw.amplitude_buffer, self.along_track_avg_local))
                                temp_pie_counts = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                    self.shared_ring_buffer_raw.count_buffer, self.along_track_avg_local))
                                temp_timestamp = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                    self.shared_ring_buffer_raw.timestamp_buffer, self.along_track_avg_local))
                                temp_lat_lon = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                    self.shared_ring_buffer_raw.lat_lon_buffer, self.along_track_avg_local))

                            # if count_temp == self.along_track_avg_local:
                                # Note: Attempt at multithreading here does not seem to affect performance.
                                # threading.Thread(target=self.collapse_and_buffer_pings,
                                #                  args=(temp_pie_amplitudes, temp_pie_counts,
                                #                        temp_timestamp, temp_lat_lon)).start()
                                self.collapse_and_buffer_pings(temp_pie_amplitudes, temp_pie_counts,
                                                               temp_timestamp, temp_lat_lon)
                                # Reset counter
                                count_temp = 0

                            elif self.along_track_avg_edited:
                                if count_temp < self.along_track_avg_local:
                                    # Ready to use new, local value for processing
                                    self.along_track_avg_edited = False
                                elif count_temp > self.along_track_avg_local:
                                    # Not ready to use new, local value for processing
                                    # Make temporary copies of most recent count_temp number of pings
                                    temp_pie_amplitudes = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                        self.shared_ring_buffer_raw.amplitude_buffer, count_temp))
                                    temp_pie_counts = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                        self.shared_ring_buffer_raw.count_buffer, count_temp))
                                    temp_timestamp = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                        self.shared_ring_buffer_raw.timestamp_buffer, count_temp))
                                    temp_lat_lon = np.copy(self.shared_ring_buffer_raw.view_recent_pings(
                                        self.shared_ring_buffer_raw.lat_lon_buffer, count_temp))

                                    print("count_temp before recursive call: {}".format(count_temp))
                                    count_temp = self.adjust_along_track_avg(temp_pie_amplitudes, temp_pie_counts,
                                                                             temp_timestamp, temp_lat_lon, count_temp)
                                    print("count_temp after recursive call: {}".format(count_temp))
                                    if count_temp < self.along_track_avg_local:
                                        # This should always be true at this point!
                                        print("Resetting along_track_avg_edited to False")
                                        self.along_track_avg_edited = False
                                    else:
                                        logger.error("Recursive method for recalculating along-track average failed.")

                    elif local_process_flag_value == 3:  # Stop pressed
                        # Do not process pie. Instead, only empty queue.
                        pass

                    else:  # local_process_flag_value is invalid
                        logger.error("Error in Plotter. Invalid process_flag value: {}."
                                     .format(local_process_flag_value))
                        break  # Exit loop

                else:  # Break out of loop because pie_object is None (poison pill)
                    break

            except queue.Empty as e:
                logger.exception("Datagram queue empty exception.")
                break

        # When process is stopped or queue's get method times out, close shared memory and allow process to terminate
        self.closeSharedMemory()

    def adjust_along_track_avg(self, temp_pie_amplitudes, temp_pie_counts, temp_timestamp, temp_lat_lon, count_temp):
        """
        Recursive method called when along_track_avg setting has been changed and count_temp exceeds
        self.along_track_avg_local. Slices entries from the start of input arrays to pass to collapse_and_buffer_pings
        method until count_temp is less than self.along_track_avg_local.
        :param temp_pie_amplitudes: Temporary copy of raw amplitude matrices from standard format pie objects.
        Number of entries equal to count_temp.
        :param temp_pie_counts: Temporary copy of raw count matrices from standard format pie objects.
        Number of entries equal to count_temp.
        :param temp_timestamp: Temporary copy of raw timestamp values from standard format pie objects.
        Number of entries equal to count_temp.
        :param temp_lat_lon: Temporary copy of raw latitude, longitude values from standard format pie objects.
        Number of entries equal to count_temp.
        :param count_temp: Temporary count of number of entries added to raw ring buffer since
        last call to collapse_and_buffer_pings. (Note, this should be equal to length of input matrices.)
        """
        print("start of recursive call")
        print("count_temp: {}, self.along_trav_avg_local: {}".format(count_temp, self.along_track_avg_local))
        print("len(arrays): ", len(temp_pie_amplitudes), ", ", len(temp_pie_counts), ", ", len(temp_timestamp), ", ", len(temp_lat_lon))
        print("len(arrays[:self.along_track_avg_local]): ", len(temp_pie_amplitudes[:self.along_track_avg_local]), ", ",
              len(temp_pie_counts[:self.along_track_avg_local]), ", ",  len(temp_timestamp[:self.along_track_avg_local]), ", ", len(temp_lat_lon[:self.along_track_avg_local]))

        if count_temp < self.along_track_avg_local:  # Terminating case
            self.outdated_along_track_avg = None
            return count_temp
        else:
            # Collapse and buffer self.along_track_avg_local number of pings from start of array:
            self.collapse_and_buffer_pings(temp_pie_amplitudes[:self.along_track_avg_local],
                                           temp_pie_counts[:self.along_track_avg_local],
                                           temp_timestamp[:self.along_track_avg_local],
                                           temp_lat_lon[:self.along_track_avg_local])
            # Decrement count_temp
            count_temp -= self.along_track_avg_local
            # Recursively call self.adjust_along_track_avg with shortened matrices
            count_temp = self.adjust_along_track_avg(temp_pie_amplitudes[self.along_track_avg_local:],
                                                     temp_pie_counts[self.along_track_avg_local:],
                                                     temp_timestamp[self.along_track_avg_local:],
                                                     temp_lat_lon[self.along_track_avg_local:], count_temp)
        return count_temp

    def collapse_and_buffer_pings(self, temp_pie_amplitudes, temp_pie_counts, temp_timestamp, temp_lat_lon):
        """
        Slices and averages amplitude, count, timestamp, and latitude / longitude entries according to user settings.
        Appends these "collapsed" matrices / values to processed ring buffer in shared memory.
        :param temp_pie_amplitudes: Temporary copy of raw amplitude matrices from standard format pie objects.
        Number of entries equal to along_track_avg.
        :param temp_pie_counts: Temporary copy of raw count matrices from standard format pie objects.
        Number of entries equal to along_track_avg.
        :param temp_timestamp: Temporary copy of raw timestamp values from standard format pie objects.
        Number of entries equal to along_track_avg.
        :param temp_lat_lon: Temporary copy of raw latitude, longitude values from standard format pie objects.
        Number of entries equal to along_track_avg.
        """
        print("plotter, collapsing and buffering pings")

        if np.any(temp_pie_amplitudes) and np.any(temp_pie_counts):
            print("plotter: len(amplitudes): {}".format(len(temp_pie_amplitudes)))

            # VERTICAL SLICE:
            # Trim arrays to omit values outside of self.vertical_slice_width_m
            # start_index       end_index
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            pie_values_vertical = temp_pie_amplitudes[:, :, self.vertical_slice_start_index:
                                                            self.vertical_slice_end_index]
            pie_count_vertical = temp_pie_counts[:, :, self.vertical_slice_start_index:
                                                       self.vertical_slice_end_index]

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

            pie_values_horizontal = temp_pie_amplitudes[:, self.horizontal_slice_start_index:
                                                           self.horizontal_slice_end_index, :]
            pie_count_horizontal = temp_pie_counts[:, self.horizontal_slice_start_index:
                                                      self.horizontal_slice_end_index, :]

            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average = 1
            pie_values_horizontal = np.sum(pie_values_horizontal, axis=0)
            pie_count_horizontal = np.sum(pie_count_horizontal, axis=0)

            # Sum columns of matrices:
            pie_values_horizontal = np.sum(pie_values_horizontal, axis=0)
            pie_count_horizontal = np.sum(pie_count_horizontal, axis=0)

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_values_horizontal_average = pie_values_horizontal / pie_count_horizontal
        else:
            logger.warning("Water column data matrix buffers are empty.")
            pie_values_vertical_average = pie_values_horizontal_average = np.empty((self.MAX_NUM_GRID_CELLS))
            pie_values_vertical_average[:] = np.nan

        # TIMESTAMP:
        if np.any(temp_timestamp):
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            pie_timestamp = np.sum(temp_timestamp)
            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                # pie_timestamp_average = pie_timestamp / self.settings["processing_settings"]["alongTrackAvg_ping"]
                pie_timestamp_average = pie_timestamp / self.along_track_avg_local
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
                # pie_lat_lon_average = pie_lat_lon / self.settings["processing_settings"]["alongTrackAvg_ping"]
                pie_lat_lon_average = pie_lat_lon / self.along_track_avg_local
        else:
            logger.warning("Nothing to plot; water column latitude / longitude matrix buffer is empty.")
            pie_lat_lon_average = np.empty(2)
            pie_lat_lon_average[:] = np.nan

        with self.processed_buffer_count.get_lock():
            self.shared_ring_buffer_processed.append_all([pie_values_vertical_average], [pie_values_horizontal_average],
                                                         [pie_timestamp_average], [pie_lat_lon_average])

    def recalculate_processed_buffer(self, ring_buffer_raw, ring_buffer_processed):
        """
        Called when settings are changed.
        Recalculates matrices in ring_buffer_processed from matrices in ring_buffer_raw according to new settings.
        :param ring_buffer_raw: Reference to raw ring buffer in shared memory.
        :param ring_buffer_processed: Reference to processed ring buffer in shared memory.
        """
        print("in recalculate_processed_buffer. self.along_track_avg_local: ", self.along_track_avg_local)
        with ring_buffer_raw.counter.get_lock():
            temp_amplitude_buffer = ring_buffer_raw.view_buffer_elements(ring_buffer_raw.amplitude_buffer)
            temp_count_buffer = ring_buffer_raw.view_buffer_elements(ring_buffer_raw.count_buffer)
            temp_timestamp_buffer = ring_buffer_raw.view_buffer_elements(ring_buffer_raw.timestamp_buffer)
            temp_lat_lon_buffer = ring_buffer_raw.view_buffer_elements(ring_buffer_raw.lat_lon_buffer)

            assert len(temp_amplitude_buffer) == len(temp_count_buffer) \
                   == len(temp_timestamp_buffer) == len(temp_lat_lon_buffer)  # All buffers should be of equal length

            # Ensure that buffer can be divided evenly by along_track_avg_ping;
            # if not, discard remainder from end of buffer for following calculations
            indices_to_trim = len(temp_amplitude_buffer) % self.along_track_avg_local

            # For debugging:
            # print("Shape before trim, temp_amplitude_buffer: {}; temp_count_buffer: {}; "
            #       "temp_timestamp_buffer: {}; temp_lat_lon_buffer: {}"
            #       .format(temp_amplitude_buffer.shape, temp_count_buffer.shape,
            #               temp_timestamp_buffer.shape, temp_lat_lon_buffer.shape))

            if indices_to_trim > 0:
                temp_amplitude_buffer = temp_amplitude_buffer[:-indices_to_trim]
                temp_count_buffer = temp_count_buffer[:-indices_to_trim]
                temp_timestamp_buffer = temp_timestamp_buffer[:-indices_to_trim]
                temp_lat_lon_buffer = temp_lat_lon_buffer[:-indices_to_trim]

            # For debugging:
            # print("Shape after trim, temp_amplitude_buffer: {}; temp_count_buffer: {}; "
            #       "temp_timestamp_buffer: {}; temp_lat_lon_buffer: {}"
            #       .format(temp_amplitude_buffer.shape, temp_count_buffer.shape,
            #               temp_timestamp_buffer.shape, temp_lat_lon_buffer.shape))

            # VERTICAL SLICE:
            # Trim arrays to omit values outside of self.vertical_slice_width_m
            # start_index       end_index
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            #          |X|_|_|_|X|
            # This creates view of array
            temp_amplitude_vertical = temp_amplitude_buffer[:, :, self.vertical_slice_start_index:
                                                                  self.vertical_slice_end_index]
            temp_count_vertical = temp_count_buffer[:, :, self.vertical_slice_start_index:
                                                          self.vertical_slice_end_index]

            # For debugging:
            # print("Shape temp_amplitude_vertical: {}; temp_count_vertical: {}"
            #       .format(temp_count_vertical.shape, temp_count_vertical.shape))

            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average = 1
            # Note that this creates copy of array
            temp_amplitude_vertical_collapsed = np.add.reduceat(temp_amplitude_vertical,
                                                                np.arange(0, len(temp_amplitude_vertical),
                                                                          self.along_track_avg_local))

            temp_count_vertical_collapsed = np.add.reduceat(temp_count_vertical,
                                                           np.arange(0, len(temp_count_vertical),
                                                                     self.along_track_avg_local))

            # For debugging:
            print("Shape temp_amplitude_vertical_collapsed: {}; temp_count_vertical_collapsed: {}"
                  .format(temp_count_vertical_collapsed.shape, temp_count_vertical_collapsed.shape))

            # Sum rows of matrices:
            # Note that this creates copy of array
            amplitude_vertical = np.sum(temp_amplitude_vertical_collapsed, axis=2)
            count_vertical = np.sum(temp_count_vertical_collapsed, axis=2)

            # For debugging:
            # print("Shape amplitude_vertical: {}; count_vertical: {}"
            #       .format(amplitude_vertical.shape, count_vertical.shape))

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                vertical_average = amplitude_vertical / count_vertical

            # For debugging:
            # print("Shape vertical average: {}".format(vertical_average.shape))

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

            # For debugging:
            # print("Shape temp_amplitude_horizontal: {}; temp_count_horizontal: {}"
            #       .format(temp_count_horizontal.shape, temp_count_horizontal.shape))

            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average = 1
            # Note that this creates copy of array
            temp_amplitude_horizontal_collapsed = np.add.reduceat(temp_amplitude_horizontal,
                                                                  np.arange(0, len(temp_amplitude_horizontal),
                                                                            self.along_track_avg_local))

            temp_count_horizontal_collapsed = np.add.reduceat(temp_count_horizontal,
                                                              np.arange(0, len(temp_count_horizontal),
                                                                        self.along_track_avg_local))

            # For debugging:
            print("Shape temp_amplitude_horizontal_collapsed: {}; temp_count_horizontal_collapsed: {}"
                  .format(temp_count_horizontal_collapsed.shape, temp_count_horizontal_collapsed.shape))

            # Sum columns of matrices:
            # Note that this creates copy of array
            amplitude_horizontal = np.sum(temp_amplitude_horizontal_collapsed, axis=1)
            count_horizontal = np.sum(temp_count_horizontal_collapsed, axis=1)

            # For debugging:
            # print("Shape amplitude_horizontal: {}; count_horizontal: {}"
            #       .format(amplitude_horizontal.shape, count_horizontal.shape))

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                horizontal_average = amplitude_horizontal / count_horizontal

            # For debugging
            # print("Shape horizontal average: {}".format(horizontal_average.shape))

            # TIMESTAMP:
            # Note that this creates copy of array
            timestamp_collapsed = np.add.reduceat(temp_timestamp_buffer,
                                                  np.arange(0, len(temp_timestamp_buffer), self.along_track_avg_local))

            # For debugging:
            # print("Shape timestamp_collapsed: {}".format(timestamp_collapsed.shape))

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                timestamp_average = timestamp_collapsed / self.along_track_avg_local

            # For debugging:
            # print("Shape timestamp average: {}".format(timestamp_average.shape))

            # LAT LON:
            # Note that this creates copy of array
            lat_lon_collapsed = np.add.reduceat(temp_lat_lon_buffer,
                                                np.arange(0, len(temp_lat_lon_buffer), self.along_track_avg_local))

            # For debugging:
            # print("Shape lat_lon_collapsed: {}".format(lat_lon_collapsed.shape))

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                lat_lon_average = lat_lon_collapsed / self.along_track_avg_local

            # For debugging:
            # print("Shape lat_lon average: {}".format(lat_lon_average.shape))

        # Release raw buffer lock

        # Append newly processed water column date to processed ring buffer in shared memory, overwriting old data.
        with ring_buffer_processed.counter.get_lock():
            ring_buffer_processed.clear_and_append_all(vertical_average, horizontal_average,
                                                       timestamp_average, lat_lon_average)

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

    def run(self):
        """
        Initializes raw and processed ring buffers in shared memory and runs process. Process pulls standard format pie
        objects from a shared queue, adds raw data to raw ring buffers, add processed data to processed ring buffers.
        """
        # Protected with self.raw_buffer_count lock:
        self.shared_ring_buffer_raw = SharedRingBufferRaw(self.settings, self.raw_buffer_count,
                                             self.raw_buffer_full_flag, create_shmem=False)

        # Protected with self.processed_buffer_count lock:
        self.shared_ring_buffer_processed = SharedRingBufferProcessed(self.settings, self.processed_buffer_count,
                                                         self.processed_buffer_full_flag, create_shmem=False)

        self.get_and_buffer_pie()
