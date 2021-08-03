# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:
# See https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib for
# adding updating plot to GUI.

from collections import deque
import datetime
import io
import itertools
#import KMALL
from KmallReaderForWaterColumn import KmallReaderForWaterColumn as k
from KongsbergDGPie import KongsbergDGPie
import logging
import math
import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("Qt5Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
#from numpy_ringbuffer import RingBuffer
from NumpyRingBuffer import NumpyRingBuffer
import sched
import statistics
import threading
import time
import queue
import warnings

logger = logging.getLogger(__name__)


class KongsbergDGPlot:
    def __init__(self, bin_size=None, max_heave=None, vertical_slice_width_m=None, horizontal_slice_width_m=None,
                 horizontal_slice_depth_m=None, num_pings_to_average=None, queue_pie=None):
        print("init_dgplot")
        self.bin_size = bin_size
        self.MAX_HEAVE = max_heave

        self.vertical_slice_width_m = vertical_slice_width_m
        self.horizontal_slice_width_m = horizontal_slice_width_m

        self.horizontal_slice_depth_m = horizontal_slice_depth_m
        # TODO: Ensure that this is an integer
        self.num_pings_to_average = num_pings_to_average

        self.queue_rx_pie = queue_pie

        self.QUEUE_RX_PIE_TIMEOUT = 60  # Seconds
        # TODO: Ensure that this is an integer
        self.MAX_LENGTH_BUFFER = 10000  # Based on ~1000 MWC datagrams per minute for 10 minutes (~16 per second).
        # Above doesn't work because NumpyRingBuffer allocates full memory for 10000 * 500 * 500 matrix--that's ~20 GB
        # Update: It works now with dtype=float changed to dtype=np.float16! I think this should still give us good enough precision.
        #self.MAX_LENGTH_BUFFER = 1000  # Based on ~1000 MWC datagrams per minute for 1 minutes (~16 per second).
        # TODO: Should this be passed as an argument to both DGProcess and DGPlot to ensure consistency?
        # TODO: Ensure that this is an integer
        self.MAX_NUM_GRID_CELLS = 500





        self._lock_slice_buffers = threading.Lock()
        # ! ! ! ! ! ALWAYS USE self._lock_slice_buffers WHEN ACCESSING THESE BUFFERS ! ! ! ! ! :
        # self.vertical_slice_buffer = NumpyRingBuffer(capacity=(self.MAX_LENGTH_BUFFER // self.num_pings_to_average),
        #                                              dtype=(np.float16, (self.MAX_NUM_GRID_CELLS,
        #                                                                  self.MAX_NUM_GRID_CELLS)))
        self.vertical_slice_buffer = NumpyRingBuffer(capacity=(self.MAX_LENGTH_BUFFER // self.num_pings_to_average),
                                                     dtype=(np.float16, self.MAX_NUM_GRID_CELLS))
        self.horizontal_slice_buffer = NumpyRingBuffer(capacity=(self.MAX_LENGTH_BUFFER // self.num_pings_to_average),
                                                       dtype=(np.float16, self.MAX_NUM_GRID_CELLS))
        self.timestamp_slice_buffer = NumpyRingBuffer(capacity=(self.MAX_LENGTH_BUFFER // self.num_pings_to_average),
                                                      dtype=np.float32)
        self.lat_lon_slice_buffer = NumpyRingBuffer(capacity=(self.MAX_LENGTH_BUFFER // self.num_pings_to_average),
                                    dtype=(np.float32, 2))
        # ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !







        self._lock_raw_buffers = threading.Lock()
        # ! ! ! ! ! ALWAYS USE self._lock_raw_buffers WHEN ACCESSING THESE BUFFERS ! ! ! ! ! :
        self.pie_values_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER,
                                                 dtype=(np.float16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        self.pie_count_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER,
                                                dtype=(np.uint16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        self.timestamp_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=np.float32)
        self.lat_lon_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=(np.float32, 2))
        # ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

        # TODO: Create a setter method for this.
        # TODO: Double check that this is calculated correctly
        self.vertical_slice_start_index = math.floor((self.MAX_NUM_GRID_CELLS / 2) -
                                                ((self.vertical_slice_width_m / 2) / self.bin_size))
        self.vertical_slice_end_index = math.ceil((self.MAX_NUM_GRID_CELLS / 2) +
                                             ((self.vertical_slice_width_m / 2) / self.bin_size))
        # TODO: Create a setter method for this.
        # TODO: Double check that this is calculated correctly
        # First, find index of zero depth: this is found by dividing max_heave by bin_size
        # Then, find index of desired depth of slice: this is found by dividing horizontal_slice_depth_m by bin_size
        # Add the above two values to get true index of desired depth.
        # Then, find number of bins that must be included to achieve horizontal_slice_width_m above / below index of
        # desired depth: this is found by dividing horizontal_slice_width_m by 2 and dividing again by bin_size.
        self.horizontal_slice_start_index = math.ceil(self.MAX_HEAVE / self.bin_size) + \
                                          math.floor(self.horizontal_slice_depth_m / self.bin_size) - \
                                          math.ceil((self.horizontal_slice_width_m / 2) / self.bin_size)
        self.horizontal_slice_end_index = math.ceil(self.MAX_HEAVE / self.bin_size) + \
                                        math.floor(self.horizontal_slice_depth_m / self.bin_size) + \
                                        math.ceil((self.horizontal_slice_width_m / 2) / self.bin_size)


        # TODO: Should these be set in DGProcess by actually calculating them? Possibly they could change...
        self.PIE_VMIN = -95
        self.PIE_VMAX = 10
        self.PLOT_UPDATE_INTERVAL = 1000  # Milliseconds

        # self.fig_pie, self.ax_pie, self.im_pie = self.__init_pie_plot()
        # self.fig_vert, self.ax_vert, self.im_vert = self.__init_vertical_plot()
        self.fig, self.ax_pie, self.ax_vert, self.ax_horiz, self.im_pie, self.im_vert, self.im_horiz = \
            self.__init_plots()

        self.animation = None

        self.start_time = None
        self.plot_count = 0
        self.old_pie = None

    def get_and_plot_pie(self):
        print("DGPlot: get_and_plot_pie")  # For debugging
        self.start_time = datetime.datetime.now()

        threading.Thread(target=self.get_and_buffer_pie, daemon=True).start()

        print("before animation")
        self.animation = anim.FuncAnimation(self.fig, self.__animate, fargs=(),
                                                 interval=self.PLOT_UPDATE_INTERVAL)
        print("after animation")

        #self.save_animation(self.animation)
        #plt.ioff()
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        plt.show(block=True)
        # These do not print unless block=False, but then plot does not update--only empty plot appears.
        # print("*******************************************************************************************************")
        # print("*******************************************************************************************************")
        # print("*******************************************************************************************************")
        # print("*******************************************************************************************************")
        # print("*******************************************************************************************************")


    def get_and_buffer_pie(self):
        print("DGPlot: get_and_buffer_pie")  # For debugging
        start_rx_time = 0
        test_count = 0

        count = 0

        temp_pie_values = []
        temp_pie_count = []
        temp_timestamp = []
        temp_lat_lon = []

        while True:
            try:
                pie_object = self.queue_rx_pie.get(block=True, timeout=self.QUEUE_RX_PIE_TIMEOUT)
                #print("DGPlot: get_and_deque_pie: APPENDING")
                count += 1

                if test_count == 0:  # For testing
                    start_rx_time = datetime.datetime.now()
                test_count += 1

                self._lock_raw_buffers.acquire()
                try:
                    self.pie_values_buffer.append(pie_object.pie_chart_values)
                    self.pie_count_buffer.append(pie_object.pie_chart_count)
                    self.timestamp_buffer.append(pie_object.timestamp)
                    self.lat_lon_buffer.append([pie_object.latitude, pie_object.longitude])

                    if count == self.num_pings_to_average:
                        temp_pie_values = np.array(self.pie_values_buffer[-self.num_pings_to_average:])
                        temp_pie_count = np.array(self.pie_count_buffer[-self.num_pings_to_average:])
                        temp_timestamp = np.array(self.timestamp_buffer[-self.num_pings_to_average:])
                        temp_lat_lon = np.array(self.lat_lon_buffer[-self.num_pings_to_average:])
                # TODO: Catch exceptions here?
                finally:
                    self._lock_raw_buffers.release()

                if count == self.num_pings_to_average:
                    vertical_slice_average, horizontal_slice_average, timestamp_average, lat_lon_average = \
                        self.collapse_and_buffer_pings(temp_pie_values, temp_pie_count, temp_timestamp, temp_lat_lon)
                    count = 0

                    self._lock_slice_buffers.acquire()
                    try:
                        print("Appending to slice buffers")
                        self.vertical_slice_buffer.append(vertical_slice_average)
                        self.horizontal_slice_buffer.append(horizontal_slice_average)
                        self.timestamp_slice_buffer.append(timestamp_average)
                        self.lat_lon_slice_buffer.append(timestamp_average)
                    finally:
                        self._lock_slice_buffers.release()

                    print("self.vertical_slice_buffer.shape:", self.vertical_slice_buffer.shape)

            except queue.Empty:
                # TODO: Shutdown processes when queue is empty?
                logger.exception("Datagram queue empty exception.")
                break

            if self.queue_rx_pie.qsize() == 0:  # For testing
                end_rx_time = datetime.datetime.now()
                diff = (end_rx_time - start_rx_time).total_seconds()
                # print("DGPLOT, time to deque {} MWC plots: {}".format(test_count, diff))

        #print("TIME TO DEQUE ALL ITEMS IN QUEUE: {}".format(self.start_time - datetime.datetime.now()))

    def collapse_and_buffer_pings(self, temp_pie_values, temp_pie_count, temp_timestamp, temp_lat_lon):
        pie_values_vertical_average = []
        pie_values_horizontal_average = []
        if np.any(temp_pie_values) and np.any(temp_pie_count):
            print("Collapse buffer")

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
            # print("pie_values.shape after sum rows:", pie_values_vertical.shape)
            # print("pie_count.shape after sum rows:", pie_count_vertical.shape)

            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_values_vertical_average = pie_values_vertical / pie_count_vertical
            print("pie_values_vertical_average.shape: ", pie_values_vertical_average.shape)

            # HORIZONTAL SLICE:
            # Trim arrays to omit values outside of self.horizontal_slice_width_m
            # |X|X|X|X|X| start_index
            # |_|_|_|_|_|
            # |_|_|_|_|_|
            # |X|X|X|X|X| end_index
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

        pie_timestamp_average = []
        if np.any(temp_timestamp):
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            pie_timestamp = np.sum(temp_timestamp)
            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_timestamp_average = pie_timestamp / self.num_pings_to_average
        else:
            logger.warning("Water column timestamp matrix buffer is empty.")

        pie_lat_lon_average = []
        if np.any(temp_lat_lon):
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            pie_lat_lon = np.sum(temp_lat_lon, axis=0)
            # Ignore divide by zero warnings. Division by zero results in NaN, which is what we want.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_lat_lon_average = pie_lat_lon / self.num_pings_to_average
        else:
            logger.warning("Nothing to plot; water column latitude / longitude matrix buffer is empty.")

        return pie_values_vertical_average, pie_values_horizontal_average, pie_timestamp_average, pie_lat_lon_average

    def __init_plots(self):
        print("__init_plots")
        array1 = np.zeros([self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS])
        array1[:] = np.nan

        array2_3 = np.zeros([self.MAX_NUM_GRID_CELLS, int(self.MAX_LENGTH_BUFFER / self.num_pings_to_average)])
        array2_3[:] = np.nan

        plt.ion()

        fig = plt.figure(figsize=(6, 6), dpi=150)
        # ax1 = fig.add_subplot(2, 1, 1)
        # ax2 = fig.add_subplot(2, 1, 2)
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        im1 = ax1.imshow(array1, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
        im2 = ax2.imshow(array2_3, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
        im3 = ax3.imshow(array2_3, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale

        #plt.colorbar(im1)
        fig.colorbar(im1, orientation="horizontal", pad=0.2)
        plt.draw()
        plt.pause(0.001)

        return fig, ax1, ax2, ax3, im1, im2, im3

    def __init_pie_plot(self):
        # Plotting finally works following this model:
        # https://stackoverflow.com/questions/43966427/matplotlib-does-not-update-plot-when-used-in-an-ide-pycharm/43967137#43967137

        array = np.zeros([self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS])
        array[:] = np.nan

        plt.ion()

        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        #im = ax.imshow(array, cmap='gray_r', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Reverse greyscale
        im = ax.imshow(array, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale

        plt.colorbar(im)
        plt.draw()
        plt.pause(0.001)

        return fig, ax, im

    def __init_vertical_plot(self):
        array = np.zeros([self.MAX_NUM_GRID_CELLS, int(self.MAX_LENGTH_BUFFER / self.num_pings_to_average)])
        array[:] = np.nan

        plt.ion()

        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        # im = ax.imshow(array, cmap='gray_r', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Reverse greyscale
        im = ax.imshow(array, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
        plt.colorbar(im)
        plt.draw()
        plt.pause(0.001)

        return fig, ax, im

    def __animate(self, i):
        self.plot_count += 1
        print("Plot count: ", self.plot_count)

        pie_display = []
        self._lock_raw_buffers.acquire()
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                # Quick method of averaging!
                # TODO: This will give the most recent 'pie' in the buffer with no averaging--is that what we want?
                pie_display = np.array(self.pie_values_buffer.peek() / self.pie_count_buffer.peek())
        except IndexError:
            logger.warning("Excepted IndexError in retrieving values from buffer.")
        finally:
            self._lock_raw_buffers.release()

        vertical_slice = []
        horizontal_slice = []
        timestamp_slice = []
        lat_lon_slice = []
        self._lock_slice_buffers.acquire()
        try:
            # TODO: This will make a new temporary object--is that what we want? Is it necessary?
            #vertical_slice = np.array(self.vertical_slice_buffer).astype(np.float32)
            vertical_slice = np.array(self.vertical_slice_buffer).__array__(np.float32)
            horizontal_slice = np.array(self.horizontal_slice_buffer).__array__(np.float32)
            timestamp_slice = np.array(self.timestamp_slice_buffer)
            lat_lon_slice = np.array(self.lat_lon_slice_buffer)
        finally:
            self._lock_slice_buffers.release()

        if np.any(vertical_slice) and np.any(horizontal_slice):

            # Trim NaNs from matrices to be plotted:
            # This method will look for the index of the last row that is not completely filled with NaNs.
            # Add one to that index for the first full row of NaNs after all data.
            index_pie_display = np.argwhere(~np.isnan(pie_display).all(axis=1))[-1][0] + 1
            index_vertical_slice = np.argwhere(~np.isnan(vertical_slice).all(axis=0))[-1][0] + 1
            # TODO:
            # Minus one to find first full row of NaNs before data.
            index_horizontal_slice_a = np.argwhere(~np.isnan(horizontal_slice).all(axis=0))[0][0] - 1
            index_horizontal_slice_b = np.argwhere(~np.isnan(horizontal_slice).all(axis=0))[-1][0] + 1

            # Ensure that 'index' plus some small buffer does not exceed grid size.
            # (Because we want to allow some small buffer around bottom of data if possible.)
            index_pie_display = min((index_pie_display + 10), self.MAX_NUM_GRID_CELLS)
            index_vertical_slice = min((index_vertical_slice + 10), self.MAX_NUM_GRID_CELLS)
            # TODO:
            index_horizontal_slice_a = max((index_horizontal_slice_a - 10), 0)
            index_horizontal_slice_b = min((index_horizontal_slice_b + 10), self.MAX_NUM_GRID_CELLS)

            # Update plots:
            self.ax_pie.clear()
            self.ax_vert.clear()
            self.ax_horiz.clear()
            self.ax_pie.imshow(pie_display[:][:index_pie_display], cmap='gray',
                               vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            # TODO: NOTE: matplotlib gives "unsupported dtype" error with np.float16; use np.float32.
            # self.ax_vert.imshow(vertical_slice, cmap='gray',
            #                     vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            self.ax_vert.imshow(vertical_slice.T[:index_vertical_slice, :], cmap='gray',
                                vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            self.ax_horiz.imshow(horizontal_slice.T[index_horizontal_slice_a:index_horizontal_slice_b, :], cmap='gray',
                                vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            plt.draw()
            plt.pause(0.001)

        else:
            logger.warning("Nothing to plot; water column data matrix buffer is empty.")

    def animate_pie(self, i):
        #print("animate")
        self.plot_count += 1
        print(self.plot_count)

        # Animate pie plot:
        # Get most recent entry from pie_buffer
        #if self.pie_buffer:
        pie = []
        self._lock_raw_buffers.acquire()
        try:
            if self.pie_count_buffer:
                #temp_pie = np.array(self.pie_buffer)[-1]
                #temp_pie = self.pie_buffer[-1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Quick method of averaging!

                    pie = self.pie_values_buffer.peek() / self.pie_count_buffer.peek()
        finally:
            self._lock_raw_buffers.release()
        if np.any(pie):

            # This method will look for the index of the last row that is not completely filled with
            # NaNs. Add one to that index for the first full row of NaNs after all data.
            index = np.argwhere(~np.isnan(pie).all(axis=1))[-1][0] + 1

            # Ensure that 'index' plus some small buffer does not exceed grid size.
            # (Because we want to allow some small buffer around bottom of data if possible.)
            index = min((index + 10), self.MAX_NUM_GRID_CELLS)

            self.ax_pie.clear()
            #self.im_pie.set_data(pie)  # This doesn't update plot :(
            #self.ax_pie.imshow(pie, cmap='gray_r', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Reverse greyscale
            self.ax_pie.imshow(pie[:][:index], cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            plt.draw()
            plt.pause(0.001)
        else:
            logger.warning("Nothing to plot; pie matrix buffer is empty.")

    def animate_vert(self, i):
        # Animate vertical curtain plot:

        self._lock_raw_buffers.acquire()
        # We only want a number of pings that is evenly divisible by self.num_pings_to_average;
        # trim the 'remainder' from the end of the array:
        try:
            pie_values = self.pie_values_buffer[:-(self.pie_values_buffer.shape[0] % self.num_pings_to_average)]
            pie_count = self.pie_count_buffer[:-(self.pie_count_buffer.shape[0] % self.num_pings_to_average)]
            pie_timestamp = self.timestamp_buffer[:-(self.timestamp_buffer.shape[0] % self.num_pings_to_average)]
            pie_lat_lon = self.lat_lon_buffer[:-(self.lat_lon_buffer.shape[0] % self.num_pings_to_average)]
        finally:
            self._lock_raw_buffers.release()

        pie_values_average = []

        if np.any(pie_values) and np.any(pie_count):
            # Trim arrays to omit values outside of self.vertical_slice_width
            # TODO: verify that self.vertical_slice_start_index, self.vertical_slice_end_index are calculated correctly
            pie_values = pie_values[:, :, self.vertical_slice_start_index:self.vertical_slice_end_index]
            pie_count = pie_count[:, :, self.vertical_slice_start_index:self.vertical_slice_end_index]
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            pie_values = np.add.reduceat(pie_values, range(0, len(pie_values), self.num_pings_to_average))
            pie_count = np.add.reduceat(pie_count, range(0, len(pie_count), self.num_pings_to_average))
            # Sum rows of matrices:
            pie_values = np.sum(pie_values, axis=2)
            pie_count = np.sum(pie_count, axis=2)

            # Average trimmed, collapsed, summed arrays:
            # Note that division by zero results in a warning and a value of NaN.
            # In cases of divide by zero, NaN is the desired result. Suppress warning.
            # TODO: Need to deal with adding/averaging nans! Maybe not. Arrays initialized with zeros, not NaNs.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_values_average = pie_values / pie_count

            pie_timestamp_average = []

            if np.any(pie_timestamp):
                # "Collapse" arrays by adding every self.num_pings_to_average so that
                # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
                pie_timestamp = np.add.reduceat(pie_timestamp, range(0, len(pie_timestamp), self.num_pings_to_average))
                # Average collapsed arrays:
                # Note that division by zero results in a warning and a value of NaN.
                # In cases of divide by zero, NaN is the desired result. Suppress warning.
                # TODO: Need to deal with adding/averaging nans!
                with np.errstate(divide='ignore', invalid='ignore'):
                    pie_timestamp_average = pie_timestamp / self.num_pings_to_average

            pie_lat_lon_average = []

            if np.any(pie_lat_lon):
                # "Collapse" arrays by adding every self.num_pings_to_average so that
                # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
                pie_lat_lon = np.add.reduceat(pie_lat_lon, range(0, len(pie_lat_lon), self.num_pings_to_average))
                # Average collapsed arrays:
                # Note that division by zero results in a warning and a value of NaN.
                # In cases of divide by zero, NaN is the desired result. Suppress warning.
                # TODO: Need to deal with adding/averaging nans!
                with np.errstate(divide='ignore', invalid='ignore'):
                    pie_lat_lon_average = pie_lat_lon / self.num_pings_to_average

            # Trim NaNs from matrix to be plotted:
            # This method will look for the index of the last row that is not completely filled with
            # NaNs. Add one to that index for the first full row of NaNs after all data.
            index = np.argwhere(~np.isnan(pie_values_average).all(axis=0))[-1][0] + 1

            # Ensure that 'index' plus some small buffer does not exceed grid size.
            # (Because we want to allow some small buffer around bottom of data if possible.)
            index = min((index + 10), self.MAX_NUM_GRID_CELLS)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!index: ", index)

            #print("pie_values_average: ", pie_values_average[50])
            self.ax_vert.clear()
            #self.ax_vert.imshow(pie_values_average[:, :index].T, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            self.ax_vert.imshow(pie_values_average.T[:index], cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            plt.draw()
            plt.pause(0.001)

        else:
            logger.warning("Nothing to plot; value and count buffers are empty.")


        # if self.mean_buffer:
        # if self.vert_mean_buffer:
        #     # # TODO: Probably a better way to do this and not calculate every time:
        #     # start_index = (self.MAX_NUM_GRID_CELLS / 2) - round((self.vertical_slice_width_m / 2) / self.bin_size)
        #     # end_index = (self.MAX_NUM_GRID_CELLS / 2) + round((self.vertical_slice_width_m / 2) / self.bin_size) + 1
        #     # # TODO: I think converting to numpy array can be constly. Time this to confirm. Maybe find better solution.
        #     # #  (mean_buffer is a deque)
        #     # if len(self.mean_buffer) > 1:
        #     #     vert = np.array(self.mean_buffer)[:, :, start_index:end_index]
        #     # else:
        #     #     vert = np.array(self.mean_buffer)[:, start_index:end_index]
        #     self.ax_vert.clear()
        #     #self.ax_vert.imshow((self.mean_buffer[-1])[:, 240:260], cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
        #     self.ax_vert.imshow(self.vert_mean_buffer[-1], cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
        #     plt.draw()
        #     plt.pause(0.001)
        # else:
        #     logger.warning("Nothing to plot; mean buffer is empty.")

    def save_animation(self, animation):
        writer_video = anim.ImageMagickWriter(fps=10)
        # writer_video = anim.FFMpegWriter(fps=10)
        writer_video = anim.PillowWriter(fps=10)
        animation.save("testanimation.gif", writer=writer_video)

    def compare_nan_arrays(self, a, b):
        return np.all((a == b) | np.isnan(a) | np.isnan(b))

