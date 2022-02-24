# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description: Launches and manages Kongsberg-specific subprocesses KongsbergDGCaptureFromSonar and KongsbergDGProcess.

import ctypes
from WaterColumnPlotter.Kongsberg.KongsbergDGCaptureFromSonar import KongsbergDGCaptureFromSonar
from WaterColumnPlotter.Kongsberg.KongsbergDGProcess import KongsbergDGProcess
import logging
from multiprocessing import Value

logger = logging.getLogger(__name__)

__appname__ = "KongsbergDGMain"


class KongsbergDGMain:
    def __init__(self, settings, ip, port, protocol, socket_buffer_multiplier, bin_size, max_heave,
                 max_grid_cells, queue_datagram, queue_pie_object, full_ping_count, discard_ping_count):

        self.settings = settings

        self.ip = ip  # multiprocessing.Array
        self.port = port  # multiprocessing.Value
        self.protocol = protocol  # multiprocessing.Value
        self.socket_buffer_multiplier = socket_buffer_multiplier  # multiprocessing.Value
        self.bin_size = bin_size  # multiprocessing.Value
        self.max_heave = max_heave  # multiprocessing.Value
        self.max_grid_cells = max_grid_cells  # multiprocessing.Value

        # Boolean flags to indicate to processes when settings have changed in main
        self.capture_settings_edited = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value
        self.process_settings_edited = Value(ctypes.c_bool, False, lock=True)  # multiprocessing.Value

        # Queues to share data between processes
        self.queue_datagram = queue_datagram  # multiprocessing.Queue
        self.queue_pie_object = queue_pie_object  # multiprocessing.Queue

        # A count to track the number of full #MWC records (pings) received and reconstructed
        self.full_ping_count = full_ping_count  # multiprocessing.Value
        # A count to track the number of #MWC records (pings) that could not be reconstructed
        self.discard_ping_count = discard_ping_count  # multiprocessing.Value

        # 0 = initialization; 1 = play; 2 = pause; 3 = stop
        self.capture_process_flag = Value(ctypes.c_uint8, 0, lock=True)
        self.process_process_flag = Value(ctypes.c_uint8, 0, lock=True)

        self.dg_capture = None
        self.dg_process = None

    def settings_changed(self, ip_settings_edited):
        """
        Signals subprocesses (KongsbergDGCapture and KongsbergDGProcess) when
        settings have changed through the use of shared multiprocessing.Values.
        :param ip_settings_edited: Boolean indicating whether IP settings have been edited.
        """
        if ip_settings_edited:
            with self.capture_settings_edited.get_lock():
                self.capture_settings_edited.value = True
        with self.process_settings_edited.get_lock():
            self.process_settings_edited.value = True

    def play_processes(self):
        """
        Signals to both subprocesses (KongsbergDGCapture and KongsbergDGProcess)
        when play has been pressed through the use of shared multiprocessing.Values.
        """
        self._play_capture()
        self._play_process()

    def _play_capture(self):
        """
        Signals to KongsbergDGCapture subprocess when play has been
        pressed through the use of shared multiprocessing.Values.
        """
        with self.capture_process_flag.get_lock():
            self.capture_process_flag.value = 1

    def _play_process(self):
        """
        Signals to KongsbergDGProcess subprocess when play has been
        pressed through the use of shared multiprocessing.Values.
        """
        with self.process_process_flag.get_lock():
            self.process_process_flag.value = 1

    def pause_processes(self):
        """
        Signals to both subprocesses (KongsbergDGCapture and KongsbergDGProcess)
        when pause has been pressed through the use of shared multiprocessing.Values.
        """
        self._pause_capture()
        self._pause_process()

    def _pause_capture(self):
        """
        Signals to KongsbergDGCapture subprocess when pause has been
        pressed through the use of shared multiprocessing.Values.
        """
        with self.capture_process_flag.get_lock():
            self.capture_process_flag.value = 2

    def _pause_process(self):
        """
        Signals to KongsbergDGProcess subprocess when play has been
        pressed through the use of shared multiprocessing.Values.
        """
        with self.process_process_flag.get_lock():
            self.process_process_flag.value = 2

    def stop_processes(self):
        """
        Signals to both subprocesses (KongsbergDGCapture and KongsbergDGProcess)
        when stop has been pressed through the use of shared multiprocessing.Values.
        """
        self._stop_capture()
        self._stop_process()

    def _stop_capture(self):
        """
        Signals to KongsbergDGCapture subprocess when stop has been
        pressed through the use of shared multiprocessing.Values.
        """
        with self.capture_process_flag.get_lock():
            self.capture_process_flag.value = 3

    def _stop_process(self):
        """
        Signals to KongsbergDGProcess subprocess when stop has been
        pressed through the use of shared multiprocessing.Values.
        """
        with self.process_process_flag.get_lock():
            self.process_process_flag.value = 3

    def run(self):
        """
        Initializes and runs subprocesses (KongsbergDGCapture and KongsbergDGProcess).
        """
        # With daemon flag set to True, these should be terminated when main process completes:
        # https://stackoverflow.com/questions/25391025/what-exactly-is-python-multiprocessing-modules-join-method-doing
        # https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/

        self.dg_capture = KongsbergDGCaptureFromSonar(ip=self.ip, port=self.port, protocol=self.protocol,
                                                      socket_buffer_multiplier=self.socket_buffer_multiplier,
                                                      settings_edited=self.capture_settings_edited,
                                                      queue_datagram=self.queue_datagram,
                                                      full_ping_count=self.full_ping_count,
                                                      discard_ping_count=self.discard_ping_count,
                                                      process_flag=self.capture_process_flag)

        self.dg_process = KongsbergDGProcess(bin_size=self.bin_size,
                                             max_heave=self.max_heave,
                                             max_grid_cells=self.max_grid_cells,
                                             settings_edited=self.process_settings_edited,
                                             queue_datagram=self.queue_datagram,
                                             queue_pie_object=self.queue_pie_object,
                                             process_flag=self.process_process_flag)

        self.dg_capture.daemon = True
        self.dg_process.daemon = True

        self.dg_capture.start()
        self.dg_process.start()
