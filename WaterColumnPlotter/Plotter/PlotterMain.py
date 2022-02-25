# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description: Launches and manages subprocess Plotter.

import ctypes
import logging
from multiprocessing import Value
from WaterColumnPlotter.Plotter.Plotter import Plotter

logger = logging.getLogger(__name__)


class PlotterMain:
    def __init__(self, settings, bin_size, across_track_avg, depth, depth_avg, along_track_avg, max_heave,
                 queue_pie_object, raw_buffer_count, processed_buffer_count, raw_buffer_full_flag,
                 processed_buffer_full_flag):

        self.settings = settings

        self.bin_size = bin_size  # multiprocessing.Value
        self.across_track_avg = across_track_avg  # multiprocessing.Value
        self.depth = depth  # multiprocessing.Value
        self.depth_avg = depth_avg  # multiprocessing.Value
        self.along_track_avg = along_track_avg  # multiprocessing.Value
        self.max_heave = max_heave  # multiprocessing.Value

        # Boolean flag to indicate to processes when settings have changed in main
        self.plotter_settings_edited = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value

        # Queue to share data between processes
        self.queue_pie_object = queue_pie_object  # multiprocessing.Queue

        # Shared values for implementation of ring buffers in shared memory
        self.raw_buffer_count = raw_buffer_count  # multiprocessing.Value
        self.processed_buffer_count = processed_buffer_count  # multiprocessing.Value
        self.raw_buffer_full_flag = raw_buffer_full_flag
        self.processed_buffer_full_flag = processed_buffer_full_flag

        # A flag to indicate status of process. # 0 = initialization; 1 = play; 2 = pause; 3 = stop
        self.plotter_process_flag = Value(ctypes.c_uint8, 0, lock=True)

        self.plotter = None

    def settings_changed(self):
        """
        Signals subprocess (Plotter) when settings have changed through the use of shared multiprocessing.Values.
        """
        print("in plottermain settings_changed")
        with self.plotter_settings_edited.get_lock():
            self.plotter_settings_edited.value = True

    def play_processes(self):
        """
        Signals to subprocesses when play has been pressed through the use of shared multiprocessing.Values.
        """
        self._play_plotter()

    def _play_plotter(self):
        """
        Signals to Plotter subprocess when play has been pressed through the use of shared multiprocessing.Values.
        """
        with self.plotter_process_flag.get_lock():
            self.plotter_process_flag.value = 1

    def pause_processes(self):
        """
        Signals to subprocesses when pause has been pressed through the use of shared multiprocessing.Values.
        """
        self._pause_plotter()

    def _pause_plotter(self):
        """
        Signals to Plotter subprocess when pause has been pressed through the use of shared multiprocessing.Values.
        """
        with self.plotter_process_flag.get_lock():
            self.plotter_process_flag.value = 2

    def stop_processes(self):
        """
        Signals to subprocesses when stop has been pressed through the use of shared multiprocessing.Values.
        """
        self._stop_plotter()

    def _stop_plotter(self):
        """
        Signals to Plotter subprocess when stop has been pressed through the use of shared multiprocessing.Values.
        """
        with self.plotter_process_flag.get_lock():
            self.plotter_process_flag.value = 3

    def run(self):
        """
        Initializes and runs subprocess (Plotter).
        """
        # With daemon flag set to True, these should be terminated when main process completes:
        # https://stackoverflow.com/questions/25391025/what-exactly-is-python-multiprocessing-modules-join-method-doing
        # https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/

        self.plotter = Plotter(self.settings, self.bin_size, self.across_track_avg, self.depth, self.depth_avg,
                               self.along_track_avg, self.max_heave, self.plotter_settings_edited,
                               self.queue_pie_object, self.raw_buffer_count, self.processed_buffer_count,
                               self.raw_buffer_full_flag, self.processed_buffer_full_flag,
                               self.plotter_process_flag)

        self.plotter.daemon = True
        self.plotter.start()
