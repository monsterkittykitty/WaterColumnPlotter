# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

from Plotter import Plotter
import logging

logger = logging.getLogger(__name__)

__appname__ = "PlotterMain"


class PlotterMain:

    def __init__(self, settings, queue_pie_object,
                 raw_buffer_count, processed_buffer_count, raw_buffer_full_flag,
                 processed_buffer_full_flag, process_flag):

        self.settings = settings

        # multiprocessing.Queues
        self.queue_pie_object = queue_pie_object

        self.raw_buffer_count = raw_buffer_count
        self.processed_buffer_count = processed_buffer_count

        self.raw_buffer_full_flag = raw_buffer_full_flag
        self.processed_buffer_full_flag = processed_buffer_full_flag

        self.process_flag = process_flag

        self.plotter = None

    def run(self):
        # With daemon flag set to True, these should be terminated when main process completes:
        # https://stackoverflow.com/questions/25391025/what-exactly-is-python-multiprocessing-modules-join-method-doing
        # https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/

        self.plotter = Plotter(self.settings, self.queue_pie_object,
                               self.raw_buffer_count, self.processed_buffer_count,
                               self.raw_buffer_full_flag, self.processed_buffer_full_flag,
                               self.process_flag)

        self.plotter.daemon = True
        self.plotter.start()
