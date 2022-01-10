# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

import ctypes
from Plotter import Plotter
import logging
from multiprocessing import Value

logger = logging.getLogger(__name__)

__appname__ = "PlotterMain"


class PlotterMain:

    def __init__(self, settings, bin_size, across_track_avg, depth, depth_avg, along_track_avg, max_heave,
                 queue_pie_object, raw_buffer_count, processed_buffer_count, raw_buffer_full_flag,
                 processed_buffer_full_flag, process_flag):

        self.settings = settings

        # multiprocessing.Values
        self.bin_size = bin_size
        self.across_track_avg = across_track_avg
        self.depth = depth
        self.depth_avg = depth_avg
        self.along_track_avg = along_track_avg
        self.max_heave = max_heave
        # Flags to indicate to processes when settings have changed in main
        self.plotter_settings_edited = Value(ctypes.c_bool, False, lock=True)

        # multiprocessing.Queues
        self.queue_pie_object = queue_pie_object

        self.raw_buffer_count = raw_buffer_count
        self.processed_buffer_count = processed_buffer_count

        self.raw_buffer_full_flag = raw_buffer_full_flag
        self.processed_buffer_full_flag = processed_buffer_full_flag

        self.process_flag = process_flag
        self.plotter_process_flag = Value(ctypes.c_bool, False, lock=True)

        self.plotter = None

    def settings_changed(self):
        print("in plottermain settings_changed")
        with self.plotter_settings_edited.get_lock():
            self.plotter_settings_edited.value = True

    def start_processes(self):
        self._start_plotter()

    def _start_plotter(self):
        with self.plotter_process_flag.get_lock():
            self.plotter_process_flag.value = True

    def stop_processes(self):
        self._stop_plotter()

    def _stop_plotter(self):
        with self.plotter_process_flag.get_lock():
            self.plotter_process_flag.value = False

    def run(self):
        # With daemon flag set to True, these should be terminated when main process completes:
        # https://stackoverflow.com/questions/25391025/what-exactly-is-python-multiprocessing-modules-join-method-doing
        # https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/

        # self.plotter = Plotter(self.settings, self.bin_size, self.across_track_avg, self.depth, self.depth_avg,
        #                        self.along_track_avg, self.max_heave, self.queue_pie_object,
        #                        self.raw_buffer_count, self.processed_buffer_count,
        #                        self.raw_buffer_full_flag, self.processed_buffer_full_flag,
        #                        self.process_flag)

        self.plotter = Plotter(self.settings, self.bin_size, self.across_track_avg, self.depth, self.depth_avg,
                               self.along_track_avg, self.max_heave, self.plotter_settings_edited,
                               self.queue_pie_object, self.raw_buffer_count, self.processed_buffer_count,
                               self.raw_buffer_full_flag, self.processed_buffer_full_flag,
                               self.plotter_process_flag)

        self.plotter.daemon = True
        self.plotter.start()
