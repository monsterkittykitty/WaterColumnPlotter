# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

import argparse
import logging
import multiprocessing
from Old.Plotter import Plotter

logger = logging.getLogger(__name__)

class LaunchProcesses:

    def __init__(self, settings, queue_pie, vertical_plot):
        """
        :param settings: Python dictionary of format:
                        {"ip_settings: {"ip": __, "port": __},
                        "processing_settings": {"binSize_m": __, "acrossTrackAvg_m": __, "depthAvg_m": __,
                                                "alongTrackAvg_ping": __, "dualSwathPolicy": __}}
        """
        self.queue_data = multiprocessing.Queue()
        # self.queue_pie = multiprocessing.Queue()
        self.queue_pie = queue_pie
        # self.dg_capture = KongsbergDGCaptureFromSonar(rx_ip=settings["ip_settings"]["ip"],
        #                                               rx_port=settings["ip_settings"]["port"],
        #                                               connection="UDP", queue_data=self.queue_data)
        # self.dg_process = KongsbergDGProcess(bin_size=settings["processing_settings"]["binSize_m"],
        #                                      water_depth=10, max_heave=1, queue_data=self.queue_data,
        #                                      queue_pie=self.queue_pie)

        self.plotter = Plotter(vertical_plot)


    def run(self):
        # TODO: Do I need to set process_consumer daemon value to True?
        #  https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/

        # process_producer = multiprocessing.Process(target=self.dg_capture.receive_dg_and_queue)
        # process_producer.daemon = True
        # process_producer.start()
        # print("producer started")
        #
        # process_consumer = multiprocessing.Process(target=self.dg_process.get_and_process_dg)
        # process_consumer.daemon = True
        # process_consumer.start()
        # print("consumer started")

        self.process_plotter = multiprocessing.Process(target=self.plotter.run())
        self.process_plotter.start()
        print("plotter started")

        # process_producer.join()
        # process_consumer.join()
        # process_plotter.join()
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
