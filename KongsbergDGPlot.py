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
    def __init__(self, bin_size=None, vert_curt_width_m=None, num_pings_to_average=None, queue_pie=None):
        print("init_dgplot")
        self.bin_size = bin_size

        self.vert_curt_width_m = vert_curt_width_m

        self.num_pings_to_average = num_pings_to_average

        self.queue_rx_pie = queue_pie

        self.QUEUE_RX_PIE_TIMEOUT = 60  # Seconds
        self.MAX_LENGTH_BUFFER = 10000  # Based on ~1000 MWC datagrams per minute for 10 minutes (~16 per second).
        # Above doesn't work because NumpyRingBuffer allocates full memory for 10000 * 500 * 500 matrix--that's ~20 GB
        # Update: It works now with dtype=float changed to dtype=np.float16! I think this should still give us good enough precision.
        #self.MAX_LENGTH_BUFFER = 1000  # Based on ~1000 MWC datagrams per minute for 1 minutes (~16 per second).
        # TODO: Should this be passed as an argument to both DGProcess and DGPlot to ensure consistency?
        self.MAX_NUM_GRID_CELLS = 500

        #self.pie_buffer = deque([], maxlen=self.MAX_LENGTH_BUFFER)
        #self.pie_buffer = RingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=np.ndarray)
        #self.pie_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=np.ndarray)
        self._lock = threading.Lock()
        self.pie_values_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER,
                                                 dtype=(np.float16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        self.pie_count_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER,
                                                dtype=(np.uint16, (self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS)))
        self.timestamp_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=np.float32)
        self.lat_lon_buffer = NumpyRingBuffer(capacity=self.MAX_LENGTH_BUFFER, dtype=(np.float32, 2))

        # TODO: I'm not sure whether another data structure is the best way to do this. Probably not??? Deal with dual swath here.
        #self.vert_mean_buffer = deque([], maxlen=int(self.MAX_LENGTH_BUFFER / self.num_pings_to_average))



        # TODO: Create a setter method for this.
        self.vert_curt_start_index = math.floor((self.MAX_NUM_GRID_CELLS / 2) -
                                                ((self.vert_curt_width_m / 2) / self.bin_size))
        self.vert_curt_end_index = math.ceil((self.MAX_NUM_GRID_CELLS / 2) +
                                             ((self.vert_curt_width_m / 2) / self.bin_size))

        # TODO: Should these be set in DGProcess by actually calculating them? Possibly they could change...
        self.PIE_VMIN = -95
        self.PIE_VMAX = 10
        self.PLOT_UPDATE_INTERVAL = 500  # Milliseconds

        # self.fig_pie, self.ax_pie, self.im_pie = self.__init_pie_plot()
        # self.fig_vert, self.ax_vert, self.im_vert = self.__init_vertical_plot()
        self.fig_pie, self.ax_pie, self.ax_vert, self.im_pie, self.im_vert = self.__init_plots()

        self.animation = None

        self.start_time = None
        self.plot_count = 0
        self.old_pie = None

    def get_and_plot_pie(self):
        print("DGPlot: get_and_plot_pie")  # For debugging
        self.start_time = datetime.datetime.now()

        # start_time = datetime.datetime.now()
        #
        # while True:
        #     try:
        #         pie = self.queue_rx_pie.get(block=True, timeout=self.QUEUE_RX_PIE_TIMEOUT)
        #         end_time = datetime.datetime.now()
        #         self.pie_buffer.append(pie)
        #
        #         if (end_time - start_time).total_seconds() >= 0.25:
        #             self.plot_pie(pie)
        #             start_time = datetime.datetime.now()
        #
        #     except queue.Empty:
        #         # TODO: Shutdown processes when queue is empty?
        #         logger.exception("Datagram queue empty exception.")
        #         break

        # In a new thread, begin transferring data from self.queue_rx_pie to self.pie_buffer
        # Note: multiprocessing.Queue() (self.queue_rx_pie) is thread-safe;
        # collections.deque (self.pie_buffer) append and pop methods are thread-safe; RingBu
        threading.Thread(target=self.get_and_buffer_pie, daemon=True).start()
        # while True:
        #     print(len(self.pie_buffer))
        # Animate plot:
        # self.animation_pie = anim.FuncAnimation(self.fig_pie, self.animate_pie, fargs=(),
        #                                         interval=self.PLOT_UPDATE_INTERVAL)
        # self.animation_vert = anim.FuncAnimation(self.fig_vert, self.animate_vert, fargs=(),
        #                                         interval=self.PLOT_UPDATE_INTERVAL)
        self.animation_vert = anim.FuncAnimation(self.fig_pie, self.animate_vert, fargs=(),
                                                 interval=self.PLOT_UPDATE_INTERVAL)
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

        mwc_mean_count = 0
        nonnan_count = 0

        while True:
            try:
                pie_object = self.queue_rx_pie.get(block=True, timeout=self.QUEUE_RX_PIE_TIMEOUT)
                #print("DGPlot: get_and_deque_pie: APPENDING")

                if test_count == 0:  # For testing
                    start_rx_time = datetime.datetime.now()
                test_count += 1

                self._lock.acquire()
                #self.pie_buffer.append(pie_object)
                self.pie_values_buffer.append(pie_object.pie_chart_values)
                self.pie_count_buffer.append(pie_object.pie_chart_count)
                self.timestamp_buffer.append(pie_object.timestamp)
                self.lat_lon_buffer.append([pie_object.latitude, pie_object.longitude])
                self._lock.release()
                mwc_mean_count += 1


                # if mwc_mean_count == 22:
                #     start = datetime.datetime.now()
                #     self.animate_vert(2)
                #     end = datetime.datetime.now()
                #     print("time: ", (end - start).total_seconds())
                #     #exit()


                # # Average 'blocks' of pings as able:
                # if mwc_mean_count == self.num_pings_to_average:
                #     # Slice deque (self.pie_buffer) to retrieve desired number (self.num_pings_to_average)
                #     # of newest pings and convert to a numpy array:
                #     slice_num_pings = np.array(list(itertools.islice(self.pie_buffer,
                #                                            (len(self.pie_buffer) - self.num_pings_to_average),
                #                                            len(self.pie_buffer))))
                #
                #     slice_vert = slice_num_pings[:, :, self.vert_curt_start_index:self.vert_curt_end_index]
                #
                #
                #     # Suppress "RuntimeWarning: Mean of empty slice," which results from taking a nanmean of all nans.
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore", category=RuntimeWarning)
                #         # Average slice
                #         slice_vert_mean = np.nanmean(slice_vert, axis=1)
                #
                #     # For testing:  ALL TRUE! GOOD!
                #     # print("1: ", self.compare_nan_arrays(slice[-1], self.pie_buffer[-1]))
                #     # print("2: ", self.compare_nan_arrays(slice[-2], self.pie_buffer[-2]))
                #     # print("3: ", self.compare_nan_arrays(slice[-3], self.pie_buffer[-3]))
                #     # print("4: ", self.compare_nan_arrays(slice[-4], self.pie_buffer[-4]))
                #     # print("5: ", self.compare_nan_arrays(slice[-5], self.pie_buffer[-5]))
                #     # print("6: ", self.compare_nan_arrays(slice[-6], self.pie_buffer[-6]))
                #     # print("7: ", self.compare_nan_arrays(slice[-7], self.pie_buffer[-7]))
                #     # print("8: ", self.compare_nan_arrays(slice[-8], self.pie_buffer[-8]))
                #     # print("9: ", self.compare_nan_arrays(slice[-9], self.pie_buffer[-9]))
                #     # print("10: ", self.compare_nan_arrays(slice[-10], self.pie_buffer[-10]))
                #
                #
                #
                #     print('slice_vert.shape:', slice_vert.shape)
                #     print('slice_vert_mean.shape:', slice_vert_mean.shape)
                #     print('slice_vert non nan: ', np.count_nonzero(~np.isnan(slice_vert)))
                #     print('slice_vert_mean non nan: ', np.count_nonzero(~np.isnan(slice_vert_mean)))
                #
                #     self.vert_mean_buffer.append(slice_vert_mean)
                #     # TODO: This boolean is not thread-safe.
                #     mwc_mean_count = 0




            except queue.Empty:
                # TODO: Shutdown processes when queue is empty?
                logger.exception("Datagram queue empty exception.")
                break

            if self.queue_rx_pie.qsize() == 0:  # For testing
                end_rx_time = datetime.datetime.now()
                diff = (end_rx_time - start_rx_time).total_seconds()
                print("DGPLOT, time to deque {} MWC plots: {}".format(test_count, diff))

        #print("TIME TO DEQUE ALL ITEMS IN QUEUE: {}".format(self.start_time - datetime.datetime.now()))

    def __init_plots(self):
        array1 = np.zeros([self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS])
        array1[:] = np.nan

        array2 = np.zeros([self.MAX_NUM_GRID_CELLS, int(self.MAX_LENGTH_BUFFER / self.num_pings_to_average)])
        array2[:] = np.nan

        plt.ion()

        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        im1 = ax1.imshow(array1, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
        im2 = ax2.imshow(array2, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale

        plt.colorbar(im1)
        plt.draw()
        plt.pause(0.001)

        return fig, ax1, ax2, im1, im2

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


        # TODO: EXP:
        # array_s = np.zeros([self.MAX_NUM_GRID_CELLS, int(self.MAX_LENGTH_BUFFER / self.num_pings_to_average)])
        # array_s[:] = np.nan
        # ax2 = fig.add_subplot(2, 1, 1)
        # im2 = ax2.imshow(array_s, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)


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

    def animate_pie(self, i):
        #print("animate")
        self.plot_count += 1
        print(self.plot_count)

        # Animate pie plot:
        # Get most recent entry from pie_buffer
        #if self.pie_buffer:
        pie = []
        self._lock.acquire()
        try:
            if self.pie_count_buffer:
                #temp_pie = np.array(self.pie_buffer)[-1]
                #temp_pie = self.pie_buffer[-1]

                with np.errstate(divide='ignore', invalid='ignore'):
                    # Quick method of averaging!
                    pie = self.pie_values_buffer.peek() / self.pie_count_buffer.peek()
        finally:
            self._lock.release()
        if np.any(pie):

            print(pie[50])
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

        self._lock.acquire()
        # We only want a number of pings that is evenly divisible by self.num_pings_to_average;
        # trim the 'remainder' from the end of the array:
        try:
            pie_values = self.pie_values_buffer[:-(self.pie_values_buffer.shape[0] % self.num_pings_to_average)]
            pie_count = self.pie_count_buffer[:-(self.pie_count_buffer.shape[0] % self.num_pings_to_average)]
            pie_timestamp = self.timestamp_buffer[:-(self.timestamp_buffer.shape[0] % self.num_pings_to_average)]
            pie_lat_lon = self.lat_lon_buffer[:-(self.lat_lon_buffer.shape[0] % self.num_pings_to_average)]
        finally:
            self._lock.release()

        pie_values_average = []

        if np.any(pie_values) and np.any(pie_count):
            # Trim arrays to omit values outside of self.vert_curt_width
            # TODO: verify that self.vert_curt_start_index, self.vert_curt_end_index are calculated correctly
            pie_values = pie_values[:, :, self.vert_curt_start_index:self.vert_curt_end_index]
            pie_count = pie_count[:, :, self.vert_curt_start_index:self.vert_curt_end_index]
            # "Collapse" arrays by adding every self.num_pings_to_average so that
            # len(_collapsed_array_) = len(_array_) / self.num_pings_to_average
            pie_values = np.add.reduceat(pie_values, range(0, len(pie_values), self.num_pings_to_average))
            pie_count = np.add.reduceat(pie_count, range(0, len(pie_count), self.num_pings_to_average))
            # Sum rows of matrices:
            pie_values = np.sum(pie_values, axis=1)
            pie_count = np.sum(pie_count, axis=1)

            # Average trimmed, collapsed, summed arrays:
            # Note that division by zero results in a warning and a value of NaN.
            # In cases of divide by zero, NaN is the desired result. Suppress warning.
            # TODO: Need to deal with adding/averaging nans! Maybe not. Arrays initialized with zeros, not NaNs.
            with np.errstate(divide='ignore', invalid='ignore'):
                pie_values_average = pie_values / pie_count

            print("pie_values.shape after collapse:", pie_values.shape)
            print("pie_count.shape after collapse:", pie_count.shape)
            print("pie_values_average.shape after collapse:", pie_values_average.shape)

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


            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #print("pie_values_average: ", pie_values_average[50])
            self.ax_vert.clear()
            self.ax_vert.imshow(pie_values_average.T, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            plt.draw()
            plt.pause(0.001)

        else:
            logger.warning("Nothing to plot; value and count buffers are empty.")


        # if self.mean_buffer:
        # if self.vert_mean_buffer:
        #     # # TODO: Probably a better way to do this and not calculate every time:
        #     # start_index = (self.MAX_NUM_GRID_CELLS / 2) - round((self.vert_curt_width_m / 2) / self.bin_size)
        #     # end_index = (self.MAX_NUM_GRID_CELLS / 2) + round((self.vert_curt_width_m / 2) / self.bin_size) + 1
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

