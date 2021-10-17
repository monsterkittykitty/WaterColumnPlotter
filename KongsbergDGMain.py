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
import multiprocessing
import queue

logger = logging.getLogger(__name__)

class KongsbergDGMain:
    def __init__(self, rx_ip, rx_port, bin_size, connection="UDP"):
        self.connection = connection
        self.rx_ip = rx_ip
        self.rx_port = rx_port

        self.bin_size = bin_size

        self.queue_data = multiprocessing.Queue()
        self.queue_pie = multiprocessing.Queue()
        self.dg_capture = KongsbergDGCaptureFromSonar(rx_ip, rx_port, connection, queue_data=self.queue_data)
        self.dg_process = KongsbergDGProcess(bin_size=self.bin_size, water_depth=10, max_heave=1, queue_data=self.queue_data,
                                             queue_pie=self.queue_pie)

        # TODO: Experiment to launch KongsbergDGPlot from WaterColumnGUI.
        #  Is it better to create KongsbergDGPlot object here and pass it as argument to WaterColumnGUI?
        #  I think it's OK to do it this way because the format of everything in self.queue_pie
        #  should be standard regardless of sonar system...
        self.dg_plot = KongsbergDGPlot(bin_size=self.bin_size, max_heave=1, vertical_slice_width_m=1,
                                       horizontal_slice_width_m=1, horizontal_slice_depth_m=1, num_pings_to_average=10,
                                       queue_pie=self.queue_pie)
        # self.gui = WaterColumnGUI(queue_pie=self.queue_pie)

    # def __init__(self, settings):
    #     """
    #     :param settings: Python dictionary of format:
    #                     {"ip_settings: {"ip": __, "port": __},
    #                     "processing_settings": {"binSize_m": __, "acrossTrackAvg_m": __, "depthAvg_m": __,
    #                                             "alongTrackAvg_ping": __, "dualSwathPolicy": __}}
    #     """
    #     self.queue_data = multiprocessing.Queue()
    #     self.queue_pie = multiprocessing.Queue()
    #     self.dg_capture = KongsbergDGCaptureFromSonar(rx_ip=settings["ip_settings"]["ip"],
    #                                                   rx_port=settings["ip_settings"]["port"],
    #                                                   "UDP", queue_data=self.queue_data)
    #     self.dg_process = KongsbergDGProcess(bin_size=settings["processing_settings"]["binSize_m"],
    #                                          water_depth=10, max_heave=1, queue_data=self.queue_data,
    #                                          queue_pie=self.queue_pie)
    #
    #     self.dg_plot = KongsbergDGPlot(bin_size=settings["processing_settings"]["binSize_m"], max_heave=1,
    #                                     vertical_slice_width_m=settings["processing_settings"]["acrossTrackAvg_m"],
    #                                     horizontal_slice_width_m=settings["processing_settings"]["depthAvg_m"],
    #                                     horizontal_slice_depth_m=1,
    #                                     num_pings_to_average=settings["processing_settings"]["alongTrackAvg_ping"],
    #                                     queue_pie=self.queue_pie)


    def run(self):
        # TODO: Do I need to set process_consumer daemon value to True?
        #  https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/

        process_producer = multiprocessing.Process(target=self.dg_capture.receive_dg_and_queue)
        process_producer.daemon = True
        process_producer.start()
        print("producer started")

        process_consumer = multiprocessing.Process(target=self.dg_process.get_and_process_dg)
        process_consumer.daemon = True
        process_consumer.start()
        print("consumer started")

        process_plotter = multiprocessing.Process(target=self.dg_plot.get_and_plot_pie())
        process_plotter.start()
        print("plotter started")

        # process_gui = multiprocessing.Process(target=self.gui.run())
        # #process_gui.daemon = True
        # process_gui.start()
        # # TODO: This doesn't print...
        # print("********************************************************************************************gui started")

        process_producer.join()
        process_consumer.join()
        process_plotter.join()
        # process_gui.join()
        print("after join")


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
