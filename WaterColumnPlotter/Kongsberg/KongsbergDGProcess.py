# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description: Receives reconstructed #MWC records from KongsbergDGCaptureFromSonar via a shared multiprocessing Queue.
# Reads data from #MWC records, bins water column data, creates standard format pie records,
# and adds this record to a shared multiprocessing.Queue for use by the next process.

import io
import logging
from multiprocessing import Process, Value
import numpy as np
import struct
import queue
from WaterColumnPlotter.Kongsberg.KmallReaderForMDatagrams import KmallReaderForMDatagrams as k
from WaterColumnPlotter.Plotter.PieStandardFormat import PieStandardFormat

logger = logging.getLogger(__name__)


class KongsbergDGProcess(Process):
    def __init__(self, bin_size, max_heave, max_grid_cells, settings_edited,
                 queue_datagram, queue_pie_object, process_flag):
        super(KongsbergDGProcess, self).__init__()

        # multiprocessing.Values (shared between processes)
        self.bin_size = bin_size  # multiprocessing.Value
        self.max_heave = max_heave  # multiprocessing.Value
        self.max_grid_cells = max_grid_cells  # multiprocessing.Value

        # A boolean flag to indicate when settings have been edited
        self.settings_edited = settings_edited  # multiprocessing.Value

        # Local copies of above multiprocessing.Values (to avoid frequent accessing of locks)
        self.bin_size_local = None
        self.max_heave_local = None
        self.max_grid_cells_local = None
        # Initialize above local copies
        self.update_local_settings()

        # Queue shared between DGCapture and DGProcess ('get' data from this queue)
        self.queue_datagram = queue_datagram

        # Queue shared between DGProcess and DGPlot ('put' pie in this queue)
        self.queue_pie_object = queue_pie_object

        # A flag to indicate status of process. # 0 = initialization; 1 = play; 2 = pause; 3 = stop
        self.process_flag = process_flag  # multiprocessing.Value

        # self.mrz = None
        # self.mwc = None
        # self.skm = None

        self.QUEUE_DATAGRAM_TIMEOUT = 60  # Seconds

        self.dg_counter = 0  # For debugging
        self.mwc_counter = 0  # For debugging

    def update_local_settings(self):
        """
        At object initialization, this method initializes local copies of shared variables;
        after initialization, this method updates local copies of shared variables when settings are changed.
        """
        # Outer lock to ensure atomicity of updates; this lock must be held when updating settings.
        with self.settings_edited.get_lock():
            with self.bin_size.get_lock():
                self.bin_size_local = self.bin_size.value
            with self.max_heave.get_lock():
                self.max_heave_local = self.max_heave.value
            with self.max_grid_cells.get_lock():
                self.max_grid_cells_local = self.max_grid_cells.value

    def get_and_process_dg(self):
        """
        Receives datagrams from shared multiprocessing queue and processes data according to datagram type.
        Note that queue's get() method is blocking, but does have a timeout.
        """
        while True:
            # Check for signal to play / pause / stop:
            with self.process_flag.get_lock():
                local_process_flag_value = self.process_flag.value

            try:
                dg_bytes = self.queue_datagram.get(block=True, timeout=self.QUEUE_DATAGRAM_TIMEOUT)

                if dg_bytes:
                    if local_process_flag_value == 1 or local_process_flag_value == 2:  # Play pressed or pause pressed
                        # Check for signal to update settings:
                        with self.settings_edited.get_lock():
                            if self.settings_edited.value:
                                self.update_local_settings()
                                self.settings_edited.value = False
                        # Process data pulled from queue
                        self.process_dgm(dg_bytes)
                    elif local_process_flag_value == 3:  # Stop pressed
                        # Do not process datagram. Instead, only empty queue.
                        pass
                    else:
                        logger.error("Error in KongsbergDGProcess. Invalid process_flag value: {}."
                                     .format(local_process_flag_value))
                        break  # Exit loop
                else:
                    # Poison pill received; pass poison pill to next process
                    self.queue_pie_object.put(None)
                    break

            except queue.Empty:
                logger.exception("Datagram queue empty exception.")
                break

    def process_dgm(self, dg_bytes):
        """
        Reads header of datagram and initiates processing of datagram based on datagram type.
        :param dg_bytes: Datagram as pulled from shared multiprocessing.Queue (bytes).
        """
        bytes_io = io.BytesIO(dg_bytes)
        header = k.read_EMdgmHeader(bytes_io)

        if header['dgmType'] == b'#MRZ':
            # self.mrz = dg_bytes
            self.process_MRZ(header, bytes_io)

        elif header['dgmType'] == b'#MWC':
            # self.mwc = dg_bytes
            pie_object = self.process_MWC(header, bytes_io)
            self.queue_pie_object.put(pie_object)

        elif header['dgmType'] == b'#SKM':
            # self.skm = dg_bytes
            self.process_SKM(header, bytes_io)

    def process_MRZ(self, header, bytes_io):
        """
        Process #MRZ datagram; not currently implemented.
        :param header: Header field of #MRZ datagram.
        :param bytes_io: #MRZ datagram as BytesIO object.
        :return: None
        """
        pass

    def process_MWC(self, header, bytes_io):
        """
        Process #MWC datagram. Bins water column data, creates standard format pie records.
        :param header: Header field of #MWC datagram.
        :param bytes_io: #MWC datagram as BytesIO object.
        :return: #MWC data as a PieStandardFormat object.
        """
        header_struct_format = k.read_EMdgmHeader(None, return_format=True)
        partition_struct_format = k.read_EMdgmMpartition(None, header['dgmType'],
                                                         header['dgmVersion'], return_format=True)

        length_to_strip = struct.calcsize(header_struct_format) + \
                          struct.calcsize(partition_struct_format)

        # If #MWC record is 'empty' (did not receive all partitions):
        if header['numBytesDgm'] == length_to_strip:
            # print("Processing empty datagram.")

            pie_chart_amplitudes = np.zeros(shape=(self.max_grid_cells_local, self.max_grid_cells_local))
            pie_chart_counts = np.zeros(shape=(self.max_grid_cells_local, self.max_grid_cells_local))

            # Create an 'empty' PieStandardFormat record
            pie_object = PieStandardFormat(self.bin_size_local, self.max_heave_local,
                                        pie_chart_amplitudes, pie_chart_counts, header['dgTime'])

        # Full datagram (all partitions received):
        else:
            dg = k.read_EMdgmMWC(bytes_io)

            # TxInfo fields:
            heave = dg['txInfo']['heave_m']

            # SectorData fields:
            tilt_angle_re_tx_deg_3_sectors = dg['sectorData']['tiltAngleReTx_deg']

            # RxInfo fields:
            num_beams = dg['rxInfo']['numBeams']
            tvg_offset_db = dg['rxInfo']['TVGoffset_dB']
            sample_freq = dg['rxInfo']['sampleFreq_Hz']
            sound_speed = dg['rxInfo']['soundVelocity_mPerSec']

            pie_chart_amplitudes = np.zeros(shape=(self.max_grid_cells_local, self.max_grid_cells_local))
            pie_chart_counts = np.zeros(shape=(self.max_grid_cells_local, self.max_grid_cells_local))

            # ###################### START NEW - OUTSIDE ###################### #
            # Across-track beam angle array:
            beam_point_angle_re_vertical_np = np.array(dg['beamData']['beamPointAngReVertical_deg'])
            # Along-track beam angle array:
            sector_tilt_angle_re_tx_deg_np = np.array([tilt_angle_re_tx_deg_3_sectors[i] for i
                                                       in dg['beamData']['beamTxSectorNum']])

            # TODO: With access to #SKM datagrams, interpolate pitch to find tilt_angle_re_vertical_deg:
            # tilt_angle_re_vertical_deg = sector_tilt_angle_re_tx_deg + interpolated_pitch
            # TODO: For now, use sector_tilt_angle_re_tx_deg_np as an approximation for tilt_angle_re_vertical_deg.
            tilt_angle_re_vertical_deg = sector_tilt_angle_re_tx_deg_np

            # Detected range indicates bottom-detect point (zero bottom not detected)
            detected_range_np = np.array(dg['beamData']['detectedRangeInSamples'])
            # Compute average for non-zero values:
            average_detected_range_for_swath = np.average(detected_range_np[detected_range_np > 0])
            # Replace zero values (no bottom detect) with average value:
            detected_range_np[detected_range_np == 0] = average_detected_range_for_swath

            # Create an array from 0 to max(detected_range_np), with a step size of 1
            # Tile above array num_beams number of times
            range_indices_np = np.tile(np.arange(0, (np.max(detected_range_np) + 1), 1), (num_beams, 1))
            # Mask values beyond actual reported detected range for any given beam
            # Based on: https://stackoverflow.com/questions/67978532/
            # how-to-mask-rows-of-a-2d-numpy-matrix-by-values-in-1d-list
            # And: https://stackoverflow.com/questions/29046162/numpy-array-loss-of-dimension-when-masking
            range_indices_np = np.where(range_indices_np <= detected_range_np[:, None], range_indices_np, np.nan)

            # Calculate range (distance) to every point from 0 to detected range:
            range_to_wc_data_point_np = (sound_speed * range_indices_np) / (sample_freq * 2)

            kongs_x_np = range_to_wc_data_point_np * \
                         (np.sin(np.radians(tilt_angle_re_vertical_deg)))[:, np.newaxis]
            kongs_y_np = range_to_wc_data_point_np * \
                         (np.sin(np.radians(beam_point_angle_re_vertical_np)))[:, np.newaxis]
            kongs_z_np = range_to_wc_data_point_np * \
                         (np.cos(np.radians(tilt_angle_re_vertical_deg)))[:, np.newaxis] * \
                         (np.cos(np.radians(beam_point_angle_re_vertical_np)))[:, np.newaxis] + heave

            # Note: For x and y, we need "(self.max_grid_cells_local / 2)" to 'normalize position'--otherwise, negative
            # indices insert values at the end of the array (think negative indexing into array).
            # Note: For z, (self.max_heave / self.bin_size) results in number of bins allowable above '0' (neutral sea
            # surface). For example, for a negative (upward) heave that results in a bin index of -20, if self.max_heave
            # is 1 and self.bin_size is 0.05, we will add 20 to the bin index. -20 (bin_index) + 20 (adjustment) = 0
            # (*new* bin_index).
            # Note: We will approximate a swath as a 2-dimensional y, z plane rotated about the z axis.

            # We only need x bin index for bottom strike points (the last value in the np array).
            # (Though, I'm not sure we need the x bin index at all, given that we have actual positions (kongs_x_np).)

            bin_index_x_np = np.floor(kongs_x_np[-1] / round(self.bin_size_local, 2)) + \
                             int(self.max_grid_cells_local / 2)
            bin_index_y_np = np.floor(kongs_y_np / round(self.bin_size_local, 2)) + \
                             int(self.max_grid_cells_local / 2)
            bin_index_z_np = np.floor(kongs_z_np / self.bin_size_local) + \
                             int(round(self.max_heave_local, 2) / round(self.bin_size_local, 2))

            # Mask indices that fall outside of accepted values: 0 to (MAX_NUM_GRID_CELLS - 1)
            # Mask will read False for values outside of range, True for values inside range
            mask_index_y = np.ma.masked_inside(bin_index_y_np, 0, (self.max_grid_cells_local - 1))
            mask_index_z = np.ma.masked_inside(bin_index_z_np, 0, (self.max_grid_cells_local - 1))

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

            # if len(bin_index_z_np[~mask_index_z.mask]) > 0:  # This doesn't work because NaNs are masked.
            # np.count_nonzero(np.isnan(bin_index_z_np[~mask_index_z.mask])) will count the number of nans that have
            # been masked; only if length of masked array is greater than this are real values being masked.
            # TODO: Something might be wrong here.
            #  Occasional warning like: "Heave (-0.2) exceeds maximum heave (1) by 1.2 meters.
            if len(bin_index_z_np[~mask_index_z.mask]) > np.count_nonzero(np.isnan(bin_index_z_np[~mask_index_z.mask])):
                logger.warning("Heave ({:.5f}) exceeds maximum heave ({}) by {:.5f} meters. {} data points "
                               "beyond maximum heave will be lost. Consider increasing maximum heave."
                               .format(heave, round(self.max_heave_local, 2),
                                       (heave + round(self.max_heave_local, 2)),
                                       len(bin_index_z_np[~mask_index_z.mask])))

            # Combine y, z masks:
            mask_index_y_z = np.logical_and(mask_index_y.mask, mask_index_z.mask)

            # Pie chart will be approximated as a 2-dimensional y, z grid.
            # Combine y, z indices, convert from float to int:
            y_z_indices = np.vstack((bin_index_z_np[mask_index_y_z], bin_index_y_np[mask_index_y_z])).astype(int)

            # amplitude_np = (np.array(dg['beamData']['sampleAmplitude05dB_p']) * 0.5) - tvg_offset_db
            # NOTE: This method results in two errors:
            # Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays
            # with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object'
            # when creating the ndarray.
            # TypeError: can't multiply sequence by non-int of type 'float'
            # NOTE: Fixed based on: https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
            # First, make all sub-lists (nested lists) the same length:
            max_len = max(map(len, (dg['beamData']['sampleAmplitude05dB_p'])))
            amplitude_np = (np.array([list(sub_list) + [np.nan] * (max_len - len(sub_list))
                                      for sub_list in dg['beamData']['sampleAmplitude05dB_p']]) * 0.5) - tvg_offset_db

            # Trim amplitude_np to only include values of interest (up to np.max(detected_range_np) + 1)
            amplitude_np = amplitude_np[:, :(np.max(detected_range_np) + 1)]

            # Mask amplitude_np with same combination of y, z masks
            amplitude_np = amplitude_np[mask_index_y_z]

            # This method of indexing based on:
            # https://stackoverflow.com/questions/47015578/numpy-assigning-values-to-2d-array-with-list-of-indices
            pie_chart_amplitudes[tuple(y_z_indices)] += amplitude_np
            pie_chart_counts[tuple(y_z_indices)] += 1

            # This results in mirror-image pie display. Use flip!
            # pie_object = PieStandardFormat(pie_chart_amplitudes, pie_chart_counts, dg['header']['dgTime'])
            pie_object = PieStandardFormat(self.bin_size_local, self.max_heave_local,
                                           np.flip(pie_chart_amplitudes, axis=1),
                                           np.flip(pie_chart_counts, axis=1), dg['header']['dgTime'])

        return pie_object

    def process_SKM(self, header, bytes_io):
        """
        Process #SKM datagram; not currently implemented.
        :param header: Header field of #SKM datagram.
        :param bytes_io: #SKM datagram as BytesIO object.
        :return: None
        """
        pass

    def print_MWC(self, bytes_io):
        """
        Prints full #MWC record as dictionary. For debugging.
        :param bytes_io: #MWC datagram as BytesIO object.
        """
        print("Printing #MWC record:")
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

    def run(self):
        """
        Runs process. Process pulls data from multiprocessing.Queue; reads data from #MWC records,
        bins water column data, creates standard format pie records, adds this record to shared
        multiprocessing.Queue for use by next process.
        """
        self.get_and_process_dg()
