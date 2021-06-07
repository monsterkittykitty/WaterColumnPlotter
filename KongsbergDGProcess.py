# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

from collections import deque
import datetime
import io
import KMALL
from KmallReaderForWaterColumn import KmallReaderForWaterColumn as k
import logging
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import queue

logger = logging.getLogger(__name__)

class KongsbergDGProcess:
    def __init__(self, bin_size=None, water_depth=None, queue_data=None, queue_pie=None):
        print("init_dgprocess")
        # TODO: Create a function that ensure bin size is not larger than range resolution and will not exceed max 1000 x 1000 matrix
        self.bin_size = bin_size  # Meters
        self.water_depth = water_depth  # Meters

        # Queue shared between DGCapture and DGProcess ('get' data from this queue)
        self.queue_rx_data = queue_data

        # Queue shared between DGProcess and DGPlot ('put' pie in this queue)
        self.queue_tx_pie = queue_pie

        self.k = KMALL.kmall(filename=None)

        self.mrz = None
        self.mwc = None
        self.skm = None

        self.QUEUE_RX_DATA_TIMEOUT = 60  # Seconds
        self.MAX_NUM_GRID_CELLS = 500

    def get_and_process_dg(self):
        print("DGProcess: get_and_process")  # For debugging

        count = 0  # For testing
        while True:
            try:
                bytes = self.queue_rx_data.get(block=True, timeout=self.QUEUE_RX_DATA_TIMEOUT)

                self.process_dgm(bytes)

                # count += 1  # For testing
                # print("DGProcess Count: ", count)  # For testing
                #print("DGProcess Queue Size: ", self.queue_rx_data.qsize())

            except queue.Empty:
                # TODO: Shutdown processes when queue is empty?
                logger.exception("Datagram queue empty exception.")
                break

        #self.__join_subprocesses()

    def process_dgm(self, bytes):
        bytes_io = io.BytesIO(bytes)
        header = k.read_EMdgmHeader(bytes_io, return_fields=True)

        if header[1] == b'#MRZ':
            self.mrz = bytes
            self.process_MRZ(header, bytes_io)

        elif header[1] == b'#MWC':
            self.mwc = bytes

            pie_matrix = self.process_MWC(header, bytes_io)
            self.queue_tx_pie.put(pie_matrix)

        elif header[1] == b'#SKM':
            self.skm = bytes
            self.process_SKM(header, bytes_io)

    def process_MRZ(self, header, bytes_io):
        pass

    def process_MWC(self, header, bytes_io):
        # print("DGProcess: process_MWC()")
        dg = k.read_EMdgmMWC(bytes_io)

        # Header fields:
        timestamp = dg['header']['dgtime']
        dg_datetime = dg['header']['dgdatetime']

        # CmnPart fields:
        swaths_per_ping = dg['cmnPart']['swathsPerPing']

        # TxInfo fields:
        num_tx_sectors = dg['txInfo']['numTxSectors']
        heave = dg['txInfo']['heave_m']

        # SectorData fields:
        tilt_angle_re_tx_deg_3_sectors = dg['sectorData']['tiltAngleReTx_deg']

        # RxInfo fields:
        num_beams = dg['rxInfo']['numBeams']
        tvg_offset_db = dg['rxInfo']['TVGoffset_dB']
        sample_freq = dg['rxInfo']['sampleFreq_Hz']
        sound_speed = dg['rxInfo']['soundVelocity_mPerSec']

        # TODO: Adjust this based on bin size and water depth?
        #pie_chart_list = [[[] for value in range(self.MAX_NUM_GRID_CELLS)] for value in range(self.MAX_NUM_GRID_CELLS)]
        #pie_chart_list = [[[] for value in range(self.num_bins_width)] for value in range(self.num_bins_depth)]

        pie_chart_values = np.zeros(shape=(self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS))
        pie_chart_count = np.zeros(shape=(self.MAX_NUM_GRID_CELLS, self.MAX_NUM_GRID_CELLS))

        for beam in range(num_beams):
            # Across-track beam angle:
            beam_point_angle_re_vertical = dg['beamData']['beamPointAngReVertical_deg'][beam]
            # Along-track beam angle:
            sector_tilt_angle_re_tx_deg = tilt_angle_re_tx_deg_3_sectors[dg['beamData']['beamTxSectorNum'][beam]]
            # TODO: Interpolate pitch to find tilt_angle_re_vertical_deg:
            #  tilt_angle_re_vertical_deg = tilt_angle_re_tx_deg + interpolated_pitch
            temp_tilt_angle_re_vertical_deg = sector_tilt_angle_re_tx_deg

            # Index in sampleAmplitude05dB array where bottom detected
            detected_range = dg['beamData']['detectedRangeInSamples'][beam]

            # TODO: Use harmonic sound speed to determine bottom strike point; assume all other points for this
            #  beam on straight line from bottom strike point to transducer.

            # For each water column data point in a single beam:
            for i in range(detected_range + 1):  # 0 to detected_range
                range_to_wc_data_point = (sound_speed * i) / (sample_freq * 2)

                # TODO: Change temp_tilt_angle_re_vertical_deg to tilt_angle_re_vertical_deg
                kongs_x = range_to_wc_data_point * math.sin(math.radians(temp_tilt_angle_re_vertical_deg))
                kongs_y = range_to_wc_data_point * math.sin(math.radians(beam_point_angle_re_vertical))
                kongs_z = range_to_wc_data_point * math.cos(math.radians(temp_tilt_angle_re_vertical_deg)) * \
                          math.cos(math.radians(beam_point_angle_re_vertical)) - heave

                # TODO: I dont' think I need -1 here or the max(, ...) because we want to allow negative indices...
                # Note: We need "(self.MAX_NUM_GRID_CELLS / 2)" to 'normalize position'--otherwise, negative indices
                # insert values at the end of the array (think negative indexing into array).
                # Determine corresponding bin based on across-track position (x):
                # Note: We will approximate a swath as a 2-dimensional y, z plane rotated about the z axis.
                # We do not need x position except at bottom strike point
                if i == detected_range:
                    bin_index_x = math.floor(kongs_x / self.bin_size) + int(self.MAX_NUM_GRID_CELLS / 2)
                # Determine corresponding bin based on across-track position (y):
                # bin_index_y = max(0, (math.floor(kongs_y / self.bin_size) - 1))
                bin_index_y = math.floor(kongs_y / self.bin_size) + int(self.MAX_NUM_GRID_CELLS / 2)
                # Determine corresponding bin based on depth (z):
                # bin_index_z = max(0, (math.floor(kongs_z / self.bin_size) - 1))
                bin_index_z = math.floor(kongs_z / self.bin_size)

                #pie_chart_list[bin_index_z][bin_index_y].append(dg['beamData']['sampleAmplitude05dB_p'][beam][i])
                pie_chart_values[bin_index_z][bin_index_y] += (dg['beamData']['sampleAmplitude05dB_p'][beam][i] *
                                                               0.5) - tvg_offset_db
                pie_chart_count[bin_index_z][bin_index_y] += 1

        # TODO: We need to find an efficient way of averaging this matrix.
        # From: https://stackoverflow.com/questions/20572316/numpy-average-over-one-dimension-in-jagged-3d-array
        # After capturing value and position of every data point in every beam in a given ping:
        # pie_chart_np_array_3d = np.array(pie_chart_list, dtype=object)
        # do_average = np.vectorize(np.average)
        # pie_chart_np_array_average_2d = do_average(pie_chart_np_array_3d)

        start = datetime.datetime.now()
        # This appears to be a very quick way of doing the averaging! 0.002 sec for 500 x 500 matrix!
        pie_chart_average = pie_chart_values / pie_chart_count
        end = datetime.datetime.now()

        # print("ARRAY AVERAGE TAKES {}###########################################################".format((end - start)))
        # print("*****************************************************************************************averaging done")

        return pie_chart_average

    def process_SKM(self, header, bytes_io):
        pass

    def print_MWC(self, bytes_io):
        print("In print_MWC.")
        bytes_io.seek(0, 0)
        header = k.read_EMdgmHeader(bytes_io, return_fields=True)
        print("Header: ", header)
        if header[1] == b'#MWC':
            print("Header: ", header)
            #print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            partition = k.read_EMdgmMpartition(bytes_io, header[1], header[2], return_fields=True)
            print("Partition: ", partition)
            #print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            cmn_part = k.read_EMdgmMbody(bytes_io, header[1], header[2], return_fields=True)
            print("CmnPart: ", cmn_part)
            #print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            tx_info = k.read_EMdgmMWC_txInfo(bytes_io, header[2], return_fields=True)
            print("TxInfo: ", tx_info)
            #print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            sectorData = []
            for i in range(tx_info[1]):
                sectorData.append(k.read_EMdgmMWC_txSectorData(bytes_io, header[2], return_fields=True))
            print("SectorData: ", sectorData)
            #print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            rx_info = k.read_EMdgmMWC_rxInfo(bytes_io, header[2], return_fields=True)
            print("Rx Info: ", rx_info)
            #print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            beamData = []
            for i in range(rx_info[1]):
                print("In DGProcess. i: ", i)
                beamData.append(k.read_EMdgmMWC_rxBeamData(bytes_io, header[2], rx_info[3], return_fields=True))
            print("Beam Data: ", beamData)

    def __init_plots(self, x, y):

        array = np.zeros([y, x])

        plt.ion()

        fig1 = plt.figure(figsize=(11, 8.5), dpi=150)
        ax1 = plt.axes()
        im = ax1.imshow(array, cmap="gray")  # origin=?
        im.set_data(array.shape)
        return im

        # fig, ax = plt.subplots()
        # plt.gca().invert_yaxis()
        # ax.set_aspect('equal')
        # im = ax.imshow(array, cmap='gray')
        # plt.show(block=False)
        #
        # return fig, ax, im



    def plot_pie_chart(self, water_column_2d):
        print("******************************************************************************************Plotting!")
        #to_plot = np.transpose(water_column_2d)
        #fig1 = plt.figure(figsize=(11, 8.5), dpi=150)
        #x = [i for i in range(-int(self.MAX_NUM_GRID_CELLS / 2), int(self.MAX_NUM_GRID_CELLS / 2))]
        #y = [i for i in range(int(self.MAX_NUM_GRID_CELLS / 2))]
        #x = np.arange(-(self.MAX_NUM_GRID_CELLS / 2), (self.MAX_NUM_GRID_CELLS / 2), self.bin_size)
        #y = np.arange(0, self.MAX_NUM_GRID_CELLS, self.bin_size)
        #plt.gca().invert_yaxis()
        #plt.pcolormesh(water_column_2d, cmap='Greys')
        #plt.show()


        #plt.gca().invert_yaxis()
        #plt.pcolormesh(water_column_2d, cmap='Greys')
        #plt.show(block=True)
        #plt.show(block=False)

        #self.pie_plot.set_data(water_column_2d)

        # plt.pcolormesh(water_column_2d, cmap='gray')
        # self.pie_plot.canvas.draw()
        # plt.show()

        # self.pie_plot.set_data(water_column_2d)
        # plt.draw()

        self.im_pie.set_data(water_column_2d)
        plt.show(block=False)

