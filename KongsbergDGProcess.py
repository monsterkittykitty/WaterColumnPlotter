# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

from collections import deque
import datetime
import io
#import KMALL
from KmallReaderForWaterColumn import KmallReaderForWaterColumn as k
from KongsbergDGPie import KongsbergDGPie
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
        #self.water_depth = water_depth  # Meters

        # Queue shared between DGCapture and DGProcess ('get' data from this queue)
        self.queue_rx_data = queue_data

        # Queue shared between DGProcess and DGPlot ('put' pie in this queue)
        self.queue_tx_pie = queue_pie

        #self.k = KMALL.kmall(filename=None)

        self.mrz = None
        self.mwc = None
        self.skm = None

        self.QUEUE_RX_DATA_TIMEOUT = 60  # Seconds
        self.MAX_NUM_GRID_CELLS = 500

        self.MAX_HEAVE = 1  # Meter(s)

        self.dg_counter = 0  # For testing
        self.mwc_counter = 0  # For testing

    def get_and_process_dg(self):
        print("DGProcess: get_and_process")  # For debugging
        first_tx_time = None  # For testing

        count = 0  # For testing
        while True:
            try:
                dg_bytes = self.queue_rx_data.get(block=True, timeout=self.QUEUE_RX_DATA_TIMEOUT)

                if self.dg_counter == 0:  # For testing
                    first_tx_time = datetime.datetime.now()
                self.dg_counter += 1

                self.process_dgm(dg_bytes)

                # count += 1  # For testing
                # print("DGProcess Count: ", count)  # For testing
                #print("DGProcess Queue Size: ", self.queue_rx_data.qsize())

            except queue.Empty:
                # TODO: Shutdown processes when queue is empty?
                logger.exception("Datagram queue empty exception.")
                break

            if self.queue_rx_data.qsize() == 0:
                last_tx_time = datetime.datetime.now()
                print("DGPROCESS, queue_rx_data is empty.")
                print("DGPROCESS, Received: ", self.dg_counter)
                print("DGPROCESS, Received MWCs: ", self.mwc_counter)
                print("DGPROCESS, First transmit: {}; Final transmit: {}; Total time: {}".format(first_tx_time,
                                                                                                 last_tx_time,
                                                                                                 (last_tx_time - first_tx_time).total_seconds()))

    def process_dgm(self, dg_bytes):
        bytes_io = io.BytesIO(dg_bytes)
        header = k.read_EMdgmHeader(bytes_io, return_fields=True)

        if header[1] == b'#MRZ':
            self.mrz = dg_bytes
            self.process_MRZ(header, bytes_io)

        elif header[1] == b'#MWC':
            self.mwc_counter += 1  # For testing
            print("mwc_counter:", self.mwc_counter)
            self.mwc = dg_bytes

            # TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # pie_matrix = self.process_MWC(header, bytes_io)
            # self.queue_tx_pie.put(pie_matrix)
            pie_object = self.process_MWC(header, bytes_io)
            self.queue_tx_pie.put(pie_object)

        elif header[1] == b'#SKM':
            self.skm = dg_bytes
            self.process_SKM(header, bytes_io)

    def process_MRZ(self, header, bytes_io):
        pass

    def process_MWC(self, header, bytes_io):
        # print("DGProcess: process_MWC()")  # For debugging
        process_MWC_start_time = datetime.datetime.now()  # For testing

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

        # ###################### START NEW - OUTSIDE ###################### #
        # for beam in range(num_beams):
        # Across-track beam angle array:
        beam_point_angle_re_vertical_np = np.array(dg['beamData']['beamPointAngReVertical_deg'])
        # Along-track beam angle array:
        sector_tilt_angle_re_tx_deg_np = np.array([tilt_angle_re_tx_deg_3_sectors[i] for i
                                                   in dg['beamData']['beamTxSectorNum']])

        # TODO: Interpolate pitch to find tilt_angle_re_vertical_deg:
        #  tilt_angle_re_vertical_deg = tilt_angle_re_tx_deg + interpolated_pitch
        temp_tilt_angle_re_vertical_deg = sector_tilt_angle_re_tx_deg_np

        # Index in sampleAmplitude05dB array where bottom detected
        # detected_range = dg['beamData']['detectedRangeInSamples'][beam]

        detected_range_np = np.array(dg['beamData']['detectedRangeInSamples'])
        # Compute average for non-zero values:
        average_detected_range_for_swath = np.average(detected_range_np[detected_range_np > 0])
        # Replace zero values with average value:
        detected_range_np[detected_range_np == 0] = average_detected_range_for_swath

        # TODO: Use harmonic sound speed to determine bottom strike point; assume all other points for this
        #  beam on straight line from bottom strike point to transducer.

        start_wc_i = datetime.datetime.now()  # For testing

        # #*#*#*#*#*#*#*#*#*# START NEW, FAST VERSION - INSIDE #*#*#*#*#*#*#*#*#*# #
        # Create an array from 0 to max(detected_range_np), with a step size of 1
        # Tile above array num_beams number of times
        range_indices_np = np.tile(np.arange(0, (np.max(detected_range_np) + 1), 1), (num_beams, 1))
        # Mask values beyond actual reported detected range for any given beam
        # Based on: https://stackoverflow.com/questions/67978532/how-to-mask-rows-of-a-2d-numpy-matrix-by-values-in-1d-list
        # And: https: // stackoverflow.com / questions / 29046162 / numpy - array - loss - of - dimension - when - masking
        range_indices_np = np.where(range_indices_np <= detected_range_np[:, None], range_indices_np, np.nan)

        # Calculate range (distance) to every point from 0 to detected range:
        range_to_wc_data_point_np = (sound_speed * range_indices_np) / (sample_freq * 2)

        # TODO: Change temp_tilt_angle_re_vertical_deg to tilt_angle_re_vertical_deg
        kongs_x_np = range_to_wc_data_point_np * (np.sin(np.radians(temp_tilt_angle_re_vertical_deg)))[:, np.newaxis]
        kongs_y_np = range_to_wc_data_point_np * (np.sin(np.radians(beam_point_angle_re_vertical_np)))[:, np.newaxis]
        kongs_z_np = range_to_wc_data_point_np * (np.cos(np.radians(temp_tilt_angle_re_vertical_deg)))[:, np.newaxis] \
                     * (np.cos(np.radians(beam_point_angle_re_vertical_np)))[:, np.newaxis] + heave

        # Note: For x and y, we need "(self.MAX_NUM_GRID_CELLS / 2)" to 'normalize position'--otherwise, negative
        # indices insert values at the end of the array (think negative indexing into array).
        # Note: For z, (self.MAX_HEAVE / self.bin_size) results in number of bins allowable above '0' (neutral sea
        # surface). For example, for a negative (upward) heave that results in a bin index of -20, if self.MAX_HEAVE
        # is 1 and self.bin_size is 0.05, we will add 20 to the bin index. -20 (bin_index) + 20 (adjustment) = 0
        # (*new* bin_index).
        # Note: We will approximate a swath as a 2-dimensional y, z plane rotated about the z axis.

        # We only need x bin index for bottom strike points (the last value in the np array).
        # (Though, I'm not sure we need the x bin index at all, given that we have actual positions (kongs_x_np).)
        bin_index_x_np = np.floor(kongs_x_np[-1] / self.bin_size) + int(self.MAX_NUM_GRID_CELLS / 2)
        bin_index_y_np = np.floor(kongs_y_np / self.bin_size) + int(self.MAX_NUM_GRID_CELLS / 2)
        bin_index_z_np = np.floor(kongs_z_np / self.bin_size) + int(self.MAX_HEAVE / self.bin_size)

        # Mask indices that fall outside of accepted values: 0 to (MAX_NUM_GRID_CELLS - 1)
        # Mask will read False for values outside of range, True for values inside range
        # TODO: Do we need to do this for x indices too?
        mask_index_y = np.ma.masked_inside(bin_index_y_np, 0, (self.MAX_NUM_GRID_CELLS - 1))
        mask_index_z = np.ma.masked_inside(bin_index_z_np, 0, (self.MAX_NUM_GRID_CELLS - 1))

        # Error checking and warning if data will be lost:
        # if len(bin_index_y_np[~mask_index_y.mask]) > 0:  # This doesn't work because NaNs are masked.
        # np.count_nonzero(np.isnan(bin_index_y_np[~mask_index_y.mask])) will count the number of nans that have been
        # masked; only if length of masked array is greater than this are real values being masked.
        if len(bin_index_y_np[~mask_index_y.mask]) > np.count_nonzero(np.isnan(bin_index_y_np[~mask_index_y.mask])):
            print("Masked y values: ", bin_index_y_np[~mask_index_y.mask])
            logger.warning("Across-track width exceed maximum grid bounds. "
                           "{} data points beyond bounds will be lost. Consider increasing bin size."
                           .format(len(bin_index_y_np[~mask_index_y.mask]) -
                                   np.count_nonzero(np.isnan(bin_index_y_np[~mask_index_y.mask]))))
                           #.format(len(bin_index_y_np[~mask_index_y.mask])))

        # if len(bin_index_z_np[~mask_index_z.mask]) > 0:  # This doesn't work because NaNs are masked.
        # np.count_nonzero(np.isnan(bin_index_z_np[~mask_index_z.mask])) will count the number of nans that have been
        # masked; only if length of masked array is greater than this are real values being masked.
        if len(bin_index_z_np[~mask_index_z.mask]) > np.count_nonzero(np.isnan(bin_index_z_np[~mask_index_z.mask])):
            print("Masked z values: ", bin_index_z_np[~mask_index_z.mask])
            logger.warning("Heave ({:.5f}) exceeds maximum heave ({}) by {:.5f} meters. "
                           "{} data points beyond maximum heave will be lost. Consider increasing maximum heave."
                           .format(heave, self.MAX_HEAVE, (heave + self.MAX_HEAVE),
                                   len(bin_index_z_np[~mask_index_z.mask])))

        # Combine y, z masks:
        mask_index_y_z = np.logical_and(mask_index_y.mask, mask_index_z.mask)

        # Pie chart will be approximated as a 2-dimensional y, z grid.
        # Combine y, z indices, convert from float to int:
        y_z_indices = np.vstack((bin_index_z_np[mask_index_y_z], bin_index_y_np[mask_index_y_z])).astype(int)

        amplitude_np = (np.array(dg['beamData']['sampleAmplitude05dB_p']) * 0.5) - tvg_offset_db

        # Trim amplitude_np to only include values of interest (up to np.max(detected_range_np) + 1)
        amplitude_np = amplitude_np[:, :(np.max(detected_range_np) + 1)]

        # Mask amplitude_np with same combination of y, z masks
        amplitude_np = amplitude_np[mask_index_y_z]

        # This method of indexing based on:
        # https://stackoverflow.com/questions/47015578/numpy-assigning-values-to-2d-array-with-list-of-indices
        pie_chart_values[tuple(y_z_indices)] += amplitude_np
        pie_chart_count[tuple(y_z_indices)] += 1
        # ###################### END NEW - OUTSIDE ###################### #




        # # ###################### START OLD - OUTSIDE ###################### #
        # # TODO: I think I can speed this up by getting rid of this loop and just using matrix math?
        # for beam in range(num_beams):
        #     # Across-track beam angle:
        #     beam_point_angle_re_vertical = dg['beamData']['beamPointAngReVertical_deg'][beam]
        #     # Along-track beam angle:
        #     sector_tilt_angle_re_tx_deg = tilt_angle_re_tx_deg_3_sectors[dg['beamData']['beamTxSectorNum'][beam]]
        #     # TODO: Interpolate pitch to find tilt_angle_re_vertical_deg:
        #     #  tilt_angle_re_vertical_deg = tilt_angle_re_tx_deg + interpolated_pitch
        #     temp_tilt_angle_re_vertical_deg = sector_tilt_angle_re_tx_deg
        #
        #     # Index in sampleAmplitude05dB array where bottom detected
        #     # detected_range = dg['beamData']['detectedRangeInSamples'][beam]
        #
        #     detected_range_np = np.array(dg['beamData']['detectedRangeInSamples'])
        #     # Compute average for non-zero values:
        #     average_detected_range_for_swath = np.average(detected_range_np[detected_range_np > 0])
        #     # Replace zero values with average value:
        #     detected_range_np[detected_range_np == 0] = average_detected_range_for_swath
        #
        #     # Index in sampleAmplitude05dB array where bottom detected
        #     detected_range = detected_range_np[beam]
        #
        #     # For testing:
        #     # histo = np.histogram(detected_range)
        #     # detected_range_np_flatten = np.array(detected_range).flatten()
        #     # mask = np.ma.masked_less(detected_range_np_flatten, 400)
        #     # detected_range_np_flatten_masked = detected_range_np_flatten[~mask.mask]
        #     # if len(detected_range_np_flatten_masked) > 1:
        #     #     plt.hist(detected_range_np_flatten_masked, bins=np.arange(0, 650, 10))
        #     #     plt.show()
        #
        #     # TODO: Use harmonic sound speed to determine bottom strike point; assume all other points for this
        #     #  beam on straight line from bottom strike point to transducer.
        #
        #     start_wc_i = datetime.datetime.now()  # For testing
        #
        #     # #*#*#*#*#*#*#*#*#*# START NEW, FAST VERSION #*#*#*#*#*#*#*#*#*# #
        #     # Create an array from 0 to detected_range, with a step size of 1
        #     range_indices_np = np.arange(0, (detected_range + 1), 1)
        #     # Calculate range (distance) to every point from 0 to detected range:
        #     range_to_wc_data_point_np = (sound_speed * range_indices_np) / (sample_freq * 2)
        #
        #     # TODO: Change temp_tilt_angle_re_vertical_deg to tilt_angle_re_vertical_deg
        #     kongs_x_np = range_to_wc_data_point_np * math.sin(math.radians(temp_tilt_angle_re_vertical_deg))
        #     kongs_y_np = range_to_wc_data_point_np * math.sin(math.radians(beam_point_angle_re_vertical))
        #     kongs_z_np = range_to_wc_data_point_np * math.cos(math.radians(temp_tilt_angle_re_vertical_deg)) * \
        #               math.cos(math.radians(beam_point_angle_re_vertical)) + heave
        #
        #     # For testing:
        #     # mask = np.ma.masked_less(kongs_z_np, 0)
        #     # kongs_z_np_masked = kongs_z_np[mask.mask]
        #     # if len(kongs_z_np_masked) > 0:
        #     #     print("-----------------------------------------------------------------------------------------------")
        #     #     print("kongs_z_np_masked: ", kongs_z_np_masked)
        #     #     print("range_indices_np: ", range_indices_np[mask.mask])
        #     #     print("range_to_wc_data_point: ", range_to_wc_data_point_np[mask.mask])
        #     #     print("temp_tilt_angle_re_vertical_deg", temp_tilt_angle_re_vertical_deg)
        #     #     print("beam_point_angle_re_vertical", beam_point_angle_re_vertical)
        #     #     print("heave: ", heave)
        #     #     print("-----------------------------------------------------------------------------------------------")
        #
        #
        #     # Note: For x and y, we need "(self.MAX_NUM_GRID_CELLS / 2)" to 'normalize position'--otherwise, negative
        #     # indices insert values at the end of the array (think negative indexing into array).
        #     # Note: For z, (self.MAX_HEAVE / self.bin_size) results in number of bins allowable above '0' (neutral sea
        #     # surface). For example, for a negative (upward) heave that results in a bin index of -20, if self.MAX_HEAVE
        #     # is 1 and self.bin_size is 0.05, we will add 20 to the bin index. -20 (bin_index) + 20 (adjustment) = 0
        #     # (*new* bin_index).
        #     # Note: We will approximate a swath as a 2-dimensional y, z plane rotated about the z axis.
        #
        #     # We only need x bin index for bottom strike points (the last value in the np array).
        #     # (Though, I'm not sure we need the x bin index at all, given that we have actual positions (kongs_x_np).)
        #     bin_index_x_np = np.floor(kongs_x_np[-1] / self.bin_size) + int(self.MAX_NUM_GRID_CELLS / 2)
        #     bin_index_y_np = np.floor(kongs_y_np / self.bin_size) + int(self.MAX_NUM_GRID_CELLS / 2)
        #     bin_index_z_np = np.floor(kongs_z_np / self.bin_size) + int(self.MAX_HEAVE / self.bin_size)
        #
        #     # Mask indices that fall outside of accepted values: 0 to (MAX_NUM_GRID_CELLS - 1)
        #     # TODO: Do we need to do this for x indices too?
        #     mask_index_y = np.ma.masked_inside(bin_index_y_np, 0, (self.MAX_NUM_GRID_CELLS - 1))
        #     mask_index_z = np.ma.masked_inside(bin_index_z_np, 0, (self.MAX_NUM_GRID_CELLS - 1))
        #
        #     # Error checking and warning if data will be lost:
        #     if len(bin_index_y_np[~mask_index_y.mask]) > 0:
        #         logger.warning("Across-track width exceed maximum grid bounds. "
        #                        "{} data points beyond bounds will be lost. Consider increasing bin size."
        #                        .format(len(bin_index_y_np[~mask_index_y.mask])))
        #
        #     if len(bin_index_z_np[~mask_index_z.mask]) > 0:
        #         logger.warning("Heave ({:.5f}) exceeds maximum heave ({}) by {:.5f} meters. "
        #                        "{} data points beyond maximum heave will be lost. Consider increasing maximum heave."
        #                        .format(heave, self.MAX_HEAVE, (heave + self.MAX_HEAVE),
        #                                len(bin_index_z_np[~mask_index_z.mask])))
        #
        #     # Combine y, z masks:
        #     mask_index_y_z = np.logical_and(mask_index_y.mask, mask_index_z.mask)
        #
        #     # For testing:
        #     # mask = np.ma.masked_less(bin_index_z_np, 0)
        #     # bin_index_z_np_masked = bin_index_z_np[mask.mask]
        #     # if len(bin_index_z_np_masked) > 0:
        #     #     print("###############################################################################################")
        #     #     print("bin_index_z_np_masked, should be ints: ", bin_index_z_np_masked)
        #     #     print("###############################################################################################")
        #
        #     # Pie chart will be approximated as a 2-dimensional y, z grid.
        #     # Combine y, z indices, convert from float to int:
        #     y_z_indices = np.vstack((bin_index_z_np[mask_index_y_z], bin_index_y_np[mask_index_y_z])).astype(int)
        #
        #     amplitude_np = (np.array(dg['beamData']['sampleAmplitude05dB_p'][beam][:detected_range + 1]) * 0.5) - \
        #                    tvg_offset_db
        #
        #     # Mask amplitude_np with same combination of y, z masks
        #     amplitude_np = amplitude_np[mask_index_y_z]
        #
        #     # This method of indexing based on:
        #     # https://stackoverflow.com/questions/47015578/numpy-assigning-values-to-2d-array-with-list-of-indices
        #     pie_chart_values[tuple(y_z_indices)] += amplitude_np
        #     pie_chart_count[tuple(y_z_indices)] += 1
        #     # pie_chart_values[tuple(y_z_indices)] += \
        #     #     (np.array(dg['beamData']['sampleAmplitude05dB_p'][beam][:detected_range + 1]) * 0.5) - tvg_offset_db
        #     # pie_chart_count[tuple(y_z_indices)] += 1
        # # ###################### END OLD - OUTSIDE ###################### #

        # TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Ignore divide by zero runtime warning. In this case, divide by 0 results in NaN, which is what we want.
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     # Quick method of averaging!
        #     pie_chart_average = pie_chart_values / pie_chart_count
        #
        # return pie_chart_average

        pie_object = KongsbergDGPie(pie_chart_values, pie_chart_count, dg['header']['dgtime'])
        return pie_object

    def process_SKM(self, header, bytes_io):
        pass

    def print_MWC(self, bytes_io):
        print("In print_MWC.")
        bytes_io.seek(0, 0)
        header = k.read_EMdgmHeader(bytes_io, return_fields=True)
        print("Header: ", header)
        if header[1] == b'#MWC':
            print("Header: ", header)
            # print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            partition = k.read_EMdgmMpartition(bytes_io, header[1], header[2], return_fields=True)
            print("Partition: ", partition)
            # print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            cmn_part = k.read_EMdgmMbody(bytes_io, header[1], header[2], return_fields=True)
            print("CmnPart: ", cmn_part)
            # print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            tx_info = k.read_EMdgmMWC_txInfo(bytes_io, header[2], return_fields=True)
            print("TxInfo: ", tx_info)
            # print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            sectorData = []
            for i in range(tx_info[1]):
                sectorData.append(k.read_EMdgmMWC_txSectorData(bytes_io, header[2], return_fields=True))
            print("SectorData: ", sectorData)
            # print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            rx_info = k.read_EMdgmMWC_rxInfo(bytes_io, header[2], return_fields=True)
            print("Rx Info: ", rx_info)
            # print("At position {} of length {}".format(bytes_io.tell(), len(bytes)))
            beamData = []
            for i in range(rx_info[1]):
                print("In DGProcess. i: ", i)
                beamData.append(k.read_EMdgmMWC_rxBeamData(bytes_io, header[2], rx_info[3], return_fields=True))
            print("Beam Data: ", beamData)


