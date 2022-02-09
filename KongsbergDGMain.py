# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

import argparse
import ctypes
from KongsbergDGCaptureFromSonar import KongsbergDGCaptureFromSonar
from KongsbergDGProcess import KongsbergDGProcess
import logging
from multiprocessing import Value

logger = logging.getLogger(__name__)

__appname__ = "KongsbergDGMain"


class KongsbergDGMain:
    def __init__(self, settings, ip, port, protocol, socket_buffer_multiplier, bin_size, max_heave, max_grid_cells,
                 queue_datagram, queue_pie_object, full_ping_count, discard_ping_count, process_flag):

        self.settings = settings

        # multiprocessing.Values
        self.ip = ip
        self.port = port
        self.protocol = protocol
        self.socket_buffer_multiplier = socket_buffer_multiplier
        self.bin_size = bin_size
        self.max_heave = max_heave
        self.max_grid_cells = max_grid_cells
        # Flags to indicate to processes when settings have changed in main
        self.capture_settings_edited = Value(ctypes.c_bool, False, lock=True)
        self.process_settings_edited = Value(ctypes.c_bool, False, lock=True)

        # multiprocessing.Queues
        self.queue_datagram = queue_datagram
        self.queue_pie_object = queue_pie_object

        # multiprocessing.Values
        self.full_ping_count = full_ping_count
        self.discard_ping_count = discard_ping_count

        # 0 = initialization; 1 = play; 2 = pause; 3 = stop
        self.capture_process_flag = Value(ctypes.c_uint8, 0, lock=True)
        self.process_process_flag = Value(ctypes.c_uint8, 0, lock=True)

        self.dg_capture = None
        self.dg_process = None


    def settings_changed(self, ip_settings_edited):
        print("in sonarmain settings_changed")
        if ip_settings_edited:
            print("ip settings edited!")
            with self.capture_settings_edited.get_lock():
                self.capture_settings_edited.value = True
        with self.process_settings_edited.get_lock():
            self.process_settings_edited.value = True

    # def settings_changed(self, capture_settings_edited, process_settings_edited):
    #     print("in sonarmain settings_changed")
    #     if capture_settings_edited:
    #         with self.capture_settings_edited.get_lock():
    #             self.capture_settings_edited.value = True
    #     if process_settings_edited:
    #         with self.process_settings_edited.get_lock():
    #             self.process_settings_edited.value = True

    def play_processes(self):
        self._play_capture()
        self._play_process()

    def _play_capture(self):
        with self.capture_process_flag.get_lock():
            self.capture_process_flag.value = 1

    def _play_process(self):
        print("setting process flag to 1")
        with self.process_process_flag.get_lock():
            self.process_process_flag.value = 1

    def pause_processes(self):
        self._pause_capture()
        # self._pause_process()

    def _pause_capture(self):
        with self.capture_process_flag.get_lock():
            self.capture_process_flag.value = 2

    def _pause_process(self):
        print("setting process flag to 2")
        with self.process_process_flag.get_lock():
            self.process_process_flag.value = 2

    def stop_processes(self):
        self._stop_capture()
        self._stop_process()

    def _stop_capture(self):
        with self.capture_process_flag.get_lock():
            self.capture_process_flag.value = 3

    def _stop_process(self):
        print("setting process flag to 3")
        with self.process_process_flag.get_lock():
            self.process_process_flag.value = 3

    def buffer_flushed(self):
        print("BUFFER FLUSHED")

    def run(self):
        # With daemon flag set to True, these should be terminated when main process completes:
        # https://stackoverflow.com/questions/25391025/what-exactly-is-python-multiprocessing-modules-join-method-doing
        # https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/

        # self.dg_capture = KongsbergDGCaptureFromSonar(rx_ip=self.settings['ip_settings']['ip'],
        #                                               rx_port=self.settings['ip_settings']['port'],
        #                                               ip_protocol=self.settings['ip_settings']['protocol'],
        #                                               socket_buffer_multiplier=
        #                                               self.settings['ip_settings']['socketBufferMultiplier'],
        #                                               settings_edited=self.capture_settings_edited,
        #                                               queue_datagram=self.queue_datagram,
        #                                               full_ping_count=self.full_ping_count,
        #                                               discard_ping_count=self.discard_ping_count,
        #                                               process_flag=self.capture_process_flag)

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


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("rx_ip", help="IP address to receive Kongsberg datagrams.")
#     parser.add_argument("rx_port", help="Port to receive Kongsberg datagrams.", type=int)
#     parser.add_argument("--connection", default="UDP", help="Connection type: TCP, UDP, or Multicast.",
#                         choices={"TCP", "UDP", "Multicast"})
#     parser.add_argument("bin_size", help="Bin size.", type=float)
#
#     args = parser.parse_args()
#
#     dg_main = KongsbergDGMain(rx_ip=args.rx_ip, rx_port=args.rx_port, connection=args.connection, bin_size=args.bin_size)
#     dg_main.run()
