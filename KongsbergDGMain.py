# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

import argparse
from KongsbergDGCapture import KongsbergDGCapture
from KongsbergDGCaptureFromSonar import KongsbergDGCaptureFromSonar
from KongsbergDGPlot import KongsbergDGPlot
from KongsbergDGProcess import KongsbergDGProcess
import logging
import multiprocessing
import queue

logger = logging.getLogger(__name__)

class KongsbergDGMain:
    def __init__(self, rx_ip, rx_port, connection="UDP"):
        self.connection = connection
        self.rx_ip = rx_ip
        self.rx_port = rx_port

        self.queue_data = multiprocessing.Queue()
        self.queue_pie = multiprocessing.Queue()
        self.dg_capture = KongsbergDGCaptureFromSonar(rx_ip, rx_port, connection, queue_data=self.queue_data)
        self.dg_process = KongsbergDGProcess(bin_size=0.05, water_depth=10, queue_data=self.queue_data,
                                             queue_pie=self.queue_pie)
        self.dg_plot = KongsbergDGPlot(queue_pie=self.queue_pie)

    def receive_dg(self):
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
        #process_plotter.daemon = True
        process_plotter.start()
        print("plotter started")

        process_producer.join()
        process_consumer.join()
        process_plotter.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("rx_ip", help="IP address to receive Kongsberg datagrams.")
    parser.add_argument("rx_port", help="Port to receive Kongsberg datagrams.", type=int)
    parser.add_argument("--connection", default="UDP", help="Connection type: TCP or UDP.", choices={"TCP", "UDP"})

    args = parser.parse_args()

    dg_process = KongsbergDGMain(args.rx_ip, args.rx_port, connection=args.connection)
    dg_process.receive_dg()
