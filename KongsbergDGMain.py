# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

import argparse
from KongsbergDGCaptureFromSonar import KongsbergDGCaptureFromSonar
from KongsbergDGPlot import KongsbergDGPlot
from KongsbergDGProcess import KongsbergDGProcess
import logging
from multiprocessing import Process
from PyQt5.QtCore import QRunnable, QObject
import queue

logger = logging.getLogger(__name__)

__appname__ = "KongsbergDGMain"

class KongsbergDGMain:
    def __init__(self, settings, queue_datagram, queue_pie_object, process_flag):

        self.settings = settings

        # multiprocessing.Queues
        self.queue_datagram = queue_datagram
        self.queue_pie_object = queue_pie_object

        self.process_flag = process_flag

        self.dg_capture = None
        self.dg_process = None

        self.dg_capture = KongsbergDGCaptureFromSonar(rx_ip=self.settings["ip_settings"]["ip"],
                                                      rx_port=self.settings["ip_settings"]["port"],
                                                      connection="UDP", queue_datagram=self.queue_datagram,
                                                      process_flag=self.process_flag)

        self.dg_process = KongsbergDGProcess(bin_size=self.settings["processing_settings"]["binSize_m"],
                                             max_heave=self.settings["buffer_settings"]["maxHeave_m"],
                                             queue_datagram=self.queue_datagram, queue_pie_object=self.queue_pie_object,
                                             process_flag=self.process_flag)

    def run(self):
        # TODO: Do I need to set process_consumer daemon value to True?
        #  https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/

        # With daemon flag set to True, these should be terminated when main process completes:
        # https://stackoverflow.com/questions/25391025/what-exactly-is-python-multiprocessing-modules-join-method-doing

        # print("kongsbergdgmain run")
        # self.process_producer = Process(target=self.dg_capture.receive_dg_and_queue)
        # self.process_producer.daemon = True
        # self.process_producer.start()
        # print("producer started")
        #
        # self.process_consumer = Process(target=self.dg_process.get_and_process_dg)
        # self.process_consumer.daemon = True
        # self.process_consumer.start()
        # print("consumer started")

        # process_plotter = multiprocessing.Process(target=self.dg_plot.get_and_plot_pie())
        # process_plotter.start()
        # print("plotter started")

        # self.process_producer.join()
        # self.process_consumer.join()

        # process_plotter.join()
        # print("after join")

        self.dg_capture = KongsbergDGCaptureFromSonar(rx_ip=self.settings["ip_settings"]["ip"],
                                                      rx_port=self.settings["ip_settings"]["port"],
                                                      connection="UDP", queue_datagram=self.queue_datagram,
                                                      process_flag=self.process_flag)
        self.dg_capture.daemon = True

        self.dg_process = KongsbergDGProcess(bin_size=self.settings["processing_settings"]["binSize_m"],
                                             max_heave=self.settings["buffer_settings"]["maxHeave_m"],
                                             queue_datagram=self.queue_datagram, queue_pie_object=self.queue_pie_object,
                                             process_flag=self.process_flag)
        self.dg_process.daemon = True

        print("starting dg_capture: ")
        self.dg_capture.start()
        print("starting dg_process: ")
        self.dg_process.start()
        print("started both dg_capture and dg_process: ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("rx_ip", help="IP address to receive Kongsberg datagrams.")
    parser.add_argument("rx_port", help="Port to receive Kongsberg datagrams.", type=int)
    parser.add_argument("--connection", default="UDP", help="Connection type: TCP, UDP, or Multicast.",
                        choices={"TCP", "UDP", "Multicast"})
    parser.add_argument("bin_size", help="Bin size.", type=float)

    args = parser.parse_args()

    dg_main = KongsbergDGMain(rx_ip=args.rx_ip, rx_port=args.rx_port, connection=args.connection, bin_size=args.bin_size)
    dg_main.run()
