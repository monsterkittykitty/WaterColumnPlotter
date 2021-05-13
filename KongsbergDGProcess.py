# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

from collections import deque
import logging
import math
import numpy as np
import queue

logger = logging.getLogger(__name__)

class KongsbergDGProcess:
    def __init__(self, bin_size=None, queue=None):
        # TODO: Create a function that ensure bin size is not larger than range resolution and will not exceed max 1000 x 1000 matrix
        self.bin_size = bin_size
        # Queue shared between processes
        self.queue = queue
        #self.dg_deque = deque()

    def get_and_process_dg(self):
        count = 0  # For testing
        while True:
            try:
                #print("process q size: ", self.queue.qsize())
                # TODO: Choose more appropriate timeout number. Current timeout (sec) only for testing.
                #print("q.get: ", self.queue.get(block=True, timeout=60))
                #self.dg_deque.append(self.queue.get(block=True, timeout=60))

                dg = self.queue.get(block=True, timeout=60)

                self.process_dg(dg)

                count += 1  # For testing
                #print("process q size: ", self.queue.qsize())
                print("process count: ", count)  # For testing
            except queue.Empty:
                # TODO: Shutdown processes when queue is empty?
                logger.exception("Datagram queue empty exception.")
                break

    def process_dg(self, dg):
        if dg['header']['dgmType'] == b'#MRZ':
            pass

        if dg['header']['dgmType'] == b'#MWC':
            self.process_MWC(dg)

        if dg['header']['dgmType'] == b'#SKM':
            pass

    def process_MRZ(self, dg):
        pass

    def process_MWC(self, dg):
        print("In process_MWC")
        #print(dg['beamData']['beamTxSectorNum'])
        # Header fields:
        datetime_mwc = dg['header']['dgdatetime']

        # CmnPart fields:
        swaths_per_ping = dg['cmnPart']['swathsPerPing']

        # TxInfo fields:
        num_tx_sectors = dg['txInfo']['numTxSectors']
        heave = dg['txInfo']['heave_m']

        # SectorData fields:
        tilt_angle_re_tx_deg_3_sectors = dg['sectorData']['tiltAngleReTx_deg']

        # RxInfo fields:
        num_beams = dg['rxInfo']['numBeams']
        sample_freq = dg['rxInfo']['sampleFreq_Hz']
        sound_speed = dg['rxInfo']['soundVelocity_mPerSec']

        # pie_chart_matrix = np.empty((1000, 1000))
        # pie_chart_matrix[:] = np.array()
        # TODO: Adjust this based on bin size and water depth?
        pie_chart_list = [[[] for value in range(1000)] for value in range(1000)]

        for beam in range(num_beams):
            # Across-track beam angle:
            beam_point_angle_re_vertical = dg['beamData']['beamPointAngReVertical_deg'][beam]
            # Along-track beam angle:
            #print("beam: ", beam)
            #print("dg['beamData']['beamTxSectorNum'][beam]: ", dg['beamData']['beamTxSectorNum'][beam])
            tilt_angle_re_tx_deg = tilt_angle_re_tx_deg_3_sectors[dg['beamData']['beamTxSectorNum'][beam]]
            # TODO: Interpolate pitch to find tilt_angle_re_vertical_deg:
            #  tilt_angle_re_vertical_deg = tilt_angle_re_tx_deg + interpolated_pitch
            temp_tilt_angle_re_vertical_deg = tilt_angle_re_tx_deg

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

                # Determine corresponding bin based on across-track position (y):
                bin_index_y = max(0, (math.floor(kongs_y / self.bin_size) - 1))
                # Determine corresponding bin based on depth (z):
                bin_index_z = max(0, (math.floor(kongs_z / self.bin_size) - 1))

                pie_chart_list[bin_index_z][bin_index_y].append(dg['beamData']['sampleAmplitude05dB_p'][beam][i])

        # After capturing value and position of every data point in every beam in a given ping:
        pie_chart_np_array_3d = np.array(pie_chart_list)

        # From: https://stackoverflow.com/questions/20572316/numpy-average-over-one-dimension-in-jagged-3d-array
        do_average = np.vectorize(np.average)
        pie_chart_np_array_2d = do_average(pie_chart_np_array_3d)
        #print(pie_chart_np_array_2d)

    def process_SKM(self, dg):
        pass
