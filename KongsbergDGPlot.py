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
import KMALL
from KmallReaderForWaterColumn import KmallReaderForWaterColumn as k
import logging
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import sched
import threading
import time
import queue

logger = logging.getLogger(__name__)


class KongsbergDGPlot:
    def __init__(self, queue_pie=None):
        print("init_dgplot")
        self.queue_rx_pie = queue_pie

        self.QUEUE_RX_PIE_TIMEOUT = 60  # Seconds

        self.MAX_LENGTH_BUFFER = 10000  # Based on ~1000 MWC datagrams per minute for 10 minutes (~16 per second).
        self.pie_buffer = deque([], maxlen=self.MAX_LENGTH_BUFFER)

        # TODO: Should this be passed as an argument to both DGProcess and DGPlot to ensure consistency?
        self.MAX_NUM_GRID_CELLS = 500

        # TODO: Should these be set in DGProcess by actually calculating them? Possibly they could change...
        self.PIE_VMIN = -95
        self.PIE_VMAX = 10

        self.fig_pie, self.ax_pie, self.im_pie = self.__init_plots()

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
        # collections.deque (self.pie_buffer) append and pop methods are thread-safe.
        threading.Thread(target=self.get_and_deque_pie, daemon=True).start()
        # while True:
        #     print(len(self.pie_buffer))
        # Animate plot:
        self.animation = anim.FuncAnimation(self.fig_pie, self.animate, fargs=(), interval=500)  # Milliseconds
        #self.save_animation(self.animation)
        #plt.ioff()
        plt.show(block=True)

    def get_and_deque_pie(self):
        print("DGPlot: get_and_deque_pie")  # For debugging
        start_rx_time = 0
        mwc_count = 0

        while True:
            try:
                pie = self.queue_rx_pie.get(block=True, timeout=self.QUEUE_RX_PIE_TIMEOUT)
                #print("DGPlot: get_and_deque_pie: APPENDING")

                if mwc_count == 0:  # For testing
                    start_rx_time = datetime.datetime.now()
                mwc_count += 1

                self.pie_buffer.append(pie)
                #print(len(self.pie_buffer))

            except queue.Empty:
                # TODO: Shutdown processes when queue is empty?
                logger.exception("Datagram queue empty exception.")
                break

            if self.queue_rx_pie.qsize() == 0:
                end_rx_time = datetime.datetime.now()
                diff = (end_rx_time - start_rx_time).total_seconds()
                print("DGPLOT, time to deque {} MWC plots: {}".format(mwc_count, diff))

        #print("TIME TO DEQUE ALL ITEMS IN QUEUE: {}".format(self.start_time - datetime.datetime.now()))

    def __init_plots(self):
        # Plotting finally works following this model:
        # https://stackoverflow.com/questions/43966427/matplotlib-does-not-update-plot-when-used-in-an-ide-pycharm/43967137#43967137

        array = np.zeros([self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS])
        array[:] = np.nan

        plt.ion()

        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        #im = ax.imshow(array, cmap='gray_r', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Reverse greyscale
        im = ax.imshow(array, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
        plt.colorbar(im)  # This isn't displaying anything
        plt.draw()
        plt.pause(0.001)

        return fig, ax, im

    def plot_pie(self, pie):
        print("PLOT PIE!", datetime.datetime.now())
        print("max(pie): ", np.nanmax(pie), "; min(pie): ", np.nanmin(pie))
        self.im_pie.set_data(pie)
        plt.draw()
        plt.pause(0.001)

    def animate(self, i):
        #print("animate")
        self.plot_count += 1
        print(self.plot_count)
        # Get most recent entry from pie_buffer
        if self.pie_buffer:
            pie = self.pie_buffer[-1]
            #self.im_pie.set_data(pie)
            self.ax_pie.clear()
            #self.ax_pie.imshow(pie, cmap='gray_r', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Reverse greyscale
            self.ax_pie.imshow(pie, cmap='gray', vmin=self.PIE_VMIN, vmax=self.PIE_VMAX)  # Greyscale
            plt.draw()
            plt.pause(0.001)
        else:
            logger.warning("Nothing to plot; pie matrix buffer is empty.")

    def save_animation(self, animation):
        writer_video = anim.ImageMagickWriter(fps=10)
        # writer_video = anim.FFMpegWriter(fps=10)
        writer_video = anim.PillowWriter(fps=10)
        animation.save("testanimation.gif", writer=writer_video)
