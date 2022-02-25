# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# May 2021

# Description: Read file "testlogX.txt", a file that was logged on RVGS on 13 May 2021. This file should contain
# #SKM, #MRZ, #MWC (and #SPO?) datagrams. The goal of this program is to figure out how Kongsberg splits 'oversized'
# M datagrams so that we can: 1) Mimic this behavior in KmallPlayer, and 2) Reconstruct these packets appropriately.

import argparse
import io
from WaterColumnPlotter.Other.KmallPlayer import KMALL
from WaterColumnPlotter.Kongsberg.KmallReaderForMDatagrams import KmallReaderForMDatagrams as k
import logging
import struct
import sys

logger = logging.getLogger(__name__)


class ReadTestLog:
    def __init__(self, in_file):
        self.in_file = in_file

        self.start_file_position = None
        self.current_file_position = None

        self.corrupt_file_count = 0

        self.HEADER_STRUCT_FORMAT = '1I4s2B1H2I'
        self.M_PARTITION_STRUCT_FORMAT = '2H'
        self.M_BODY_STRUCT_FORMAT = '2H8B'

        self.buffer = None

    def close_file(self):
        self.file_io.close()

    def read_file(self):
        with open(self.in_file, "rb") as file_io:
            mrz_buffer = None
            mwc_buffer = None

            while True:
                self.start_file_position = file_io.tell()
                header = k.read_EMdgmHeader(file_io)
                print(header)

                if header[1] == b'#MRZ':
                    pass

                elif header[1] == b'#MWC':
                    temp_mwc_buffer = self.read_mwc(header, file_io)

                    if mwc_buffer is None:
                        if len(temp_mwc_buffer["beamDataBuffer"]) == temp_mwc_buffer["partition"][0]:
                            print(temp_mwc_buffer)
                    else:  # mwc_buffer is not None
                        if temp_mwc_buffer["cmnPart"][1] == mwc_buffer["cmnPart"][1]:  # pngCnt matches
                            mwc_buffer["beamDataBuffer"].append(temp_mwc_buffer["beamDataBuffer"])
                            if len(mwc_buffer["beamDataBuffer"]) == mwc_buffer["partition"][0]:
                                # Sort list based on dgmNum
                                beam_data_buffer_sorted = sorted(mwc_buffer["beamDataBuffer"], key=lambda tup: tup[0])
                                # Remove dgmNum
                                beam_data_cleaned = [el[1:] for el in beam_data_buffer_sorted]
                                print("beam_data_cleaned: ", beam_data_cleaned)
                                exit()


                elif header[1] == b'#SKM':
                    pass

                elif header[1] == b'#SPO':
                    pass

                else:
                    logger.warning("Datagram type does not match any datagram subscribed to. Datagram: {}".format(header[1]))
                    sys.exit(1)


    def read_mwc(self, header, file_io):
        dg = {}

        dg["header"] = header

        partition = k.read_EMdgmMpartition(file_io, header[2])
        dg["partition"] = partition

        # For dgmType 0 and 1, 'MBody' only exists in partition 1:
        if header[2] == 0 or header[2] == 1:  # header[2] = dgmVersion
            if partition[1] == 1:
                m_body = k.read_EMdgmMbody(file_io, header[2])
                dg["cmnPart"] = m_body
                print("mbody: ", m_body)
        # For dgmType 2, 'MBody' exists in all partitions:
        elif header[2] == 2:
            m_body = k.read_EMdgmMbody()
            dg["cmnPart"] = m_body
            print("mbody: ", m_body)

        tx_info = k.read_EMdgmMWCtxInfo(file_io, header[2])
        dg["txInfo"] = tx_info
        print("tx_info: ", tx_info)
        print("tx_info[1]: ", tx_info[1])
        sector_data = []
        #for i in range(tx_info[1]):  # Loop for all i = numTxSectors
        for i in range(3):
            sector_data.append(k.read_EMdgmMWCtxSectorData(file_io, header[2]))
        dg["sectorData"] = sector_data

        print("file_io.tell: ", file_io.tell())
        rx_info = k.read_EMdgmMWCrxInfo(file_io, header[2])
        dg["rxInfo"] = rx_info

        print("start_file_pos: ", self.start_file_position)
        print("file_io.tell: ", file_io.tell())
        bytes_read = file_io.tell() - self.start_file_position
        print("bytes_read: ", bytes_read)
        bytes_to_read = header[0] - bytes_read
        print("bytes_to_read: ", bytes_to_read)
        beam_data_buffer = file_io.read(bytes_to_read)
        dg["beamDataBuffer"] = (partition[1], beam_data_buffer)

        return dg

    def read_headers_only(self):
        with open(self.in_file, "rb") as file_io:

            mwc_count = 0

            while True:
                # Find current position in file:
                self.start_file_position = file_io.tell()

                # Read header:
                header = k.read_EMdgmHeader(file_io)

                if header[1] == b'#MWC':
                    mwc_count += 1

                    print("header: ", header)

                    # Read partition:
                    partition = k.read_EMdgmMpartition(file_io, header[2])
                    # print("partition: ", partition)

                    # Read cmnPart:
                    # For dgmType 0 and 1, 'MBody' only exists in partition 1:
                    if header[2] == 0 or header[2] == 1:  # header[2] = dgmVersion
                        if partition[1] == 1:
                            m_body = k.read_EMdgmMbody(file_io, header[2])
                            # print("mbody: ", m_body)
                            if m_body[0] != 12:
                                self.corrupt_file_count += 1

                                # Seek back to the beginning of this datagram:
                                file_io.seek(self.start_file_position, 0)
                                # Read number of bytes specified in header:
                                file_io.read(header[0])

                                continue

                    # Read txInfo:
                    tx_info = k.read_EMdgmMWCtxInfo(file_io, header[2])
                    # print("tx_info: ", tx_info)
                    if tx_info[0] != 12:
                        self.corrupt_file_count += 1

                    # Seek back to the beginning of this datagram:
                    file_io.seek(self.start_file_position, 0)
                    # Read number of bytes specified in header:
                    file_io.read(header[0])

                print("mwc_count: ", mwc_count)
                print("corrupt_file_count: ", self.corrupt_file_count)

    def kmall_index_mwc(self):
        kmall = KMALL.kmall(self.in_file)
        kmall.index_file()

        print(kmall.Index)
        exit()

        # Keep track of total number of MWC datagrams
        mwc_count = 0
        # Keep track of how many datagrams are dropped by comparing timestamps
        dgm_count_expected = None
        timestamp = None
        timestamp_count = 0
        unique_timestamp_count = 0
        total_dropped_dgms = 0

        # Get MWC offsets:
        mwc_offsets = [x for x, y in zip(kmall.msgoffset, kmall.msgtype) if y == "b'#MWC'"]
        for offset in mwc_offsets:
            mwc_count += 1
            kmall.FID.seek(offset, 0)

            header = kmall.read_EMdgmHeader()
            partition = kmall.read_EMdgmMpartition()

            if timestamp is None:
                # TODO: Double-check for out-of-order packets?
                kmall.FID.seek(offset, 0)
                self.buffer = kmall.FID.read((header["numBytesDgm"] - 4))
                timestamp = header['dgdatetime']  # For determining how many datagrams are dropped
                dgm_count_expected = partition["numOfDgms"]
                timestamp_count = 1
                unique_timestamp_count = 1
            elif timestamp == header['dgdatetime']:
                # TODO: Double-check for out-of-order packets?
                bytes_read = struct.Struct(self.HEADER_STRUCT_FORMAT).size + struct.Struct(self.M_PARTITION_STRUCT_FORMAT).size
                size_to_read = header["numBytesDgm"] - bytes_read - 4
                self.buffer += kmall.FID.read(size_to_read)
                timestamp_count += 1
            else:  # timestamp != header['dgdatetime']
                if partition["dgmNum"] == 1:
                    kmall.FID.seek(offset, 0)
                    self.buffer = kmall.FID.read((header["numBytesDgm"] - 4))
                total_dropped_dgms += (dgm_count_expected - timestamp_count)
                timestamp = header['dgdatetime']
                dgm_count_expected = partition["numOfDgms"]
                timestamp_count = 1
                unique_timestamp_count += 1

            print("dgm {} of {}, {}".format(partition["dgmNum"], partition["numOfDgms"], header['dgdatetime']))

            if timestamp_count == dgm_count_expected:
                print("ALT")
                print(self.buffer)
                bytes_io = io.BytesIO(self.buffer)

                header_a = k.read_EMdgmHeader(bytes_io)
                print("header_a: ", header_a)

                partition_a = k.read_EMdgmMpartition(bytes_io, header_a[2])
                print("partition_a: ", partition_a)

                cmn_part_a = k.read_EMdgmMbody(bytes_io, header_a[2])
                print("cmn_part_a: ", cmn_part_a)

                tx_info_a = k.read_EMdgmMWCtxInfo(bytes_io, header_a[2])
                print("tx_info_a: ", tx_info_a)

                sector_data_a = []
                for i in range(tx_info_a[1]):
                    sector_data_a.append(k.read_EMdgmMWCtxSectorData(bytes_io, header_a[2]))
                print("sector_data_a: ", sector_data_a)

                rx_info_a = k.read_EMdgmMWCrxInfo(bytes_io, header_a[2])
                print("rx_info_a: ", rx_info_a)

                beam_data_a = []
                for i in range(rx_info_a[1]):
                    beam_data_p = k.read_EMdgmMWCrxBeamData(bytes_io, header_a[2], rx_info_a[3])
                    beam_data_a.append(beam_data_p)
                    if i == (rx_info_a[1] - 1):
                        print("beam_data_p: ", beam_data_p)
                        print("len(beam_data_p[6]): ", len(beam_data_p[6]))

                # Final 4 bytes:
                fields = struct.unpack("I", kmall.FID.read(struct.Struct("I").size))
                print(fields[0])
                exit()

    def kmall_index_mrz(self):
        kmall = KMALL.kmall(self.in_file)
        kmall.index_file()
        # print(kmall.Index)

        # Keep track of total number of MRZ datagrams
        mrz_count = 0
        # Keep track of number of datagrams that are corrupt and where corruption is located/found
        num_bytes_cmn_part_corrupt = 0
        num_bytes_ping_info_corrupt = 0
        num_bytes_ping_info = []
        # Keep track of partition numbers of bad (corrupt) datagrams and good datagrams
        partition_ping_info_bad = []
        partition_ping_info_good = []
        # Keep track of how many datagrams are dropped by comparing timestamps
        dgm_count_expected = None
        timestamp = None
        timestamp_count = 0
        unique_timestamp_count = 0
        total_dropped_dgms = 0

        # Get MWC offsets:
        mrz_offsets = [x for x, y in zip(kmall.msgoffset, kmall.msgtype) if y == "b'#MRZ'"]
        for offset in mrz_offsets:
            mrz_count += 1
            kmall.FID.seek(offset, 0)

            header = kmall.read_EMdgmHeader()
            partition = kmall.read_EMdgmMpartition()

            if timestamp is None:
                timestamp = header['dgdatetime']  # For determining how many datagrams are dropped
                dgm_count_expected = partition["numOfDgms"]
                timestamp_count = 1
                unique_timestamp_count = 1
            elif timestamp == header['dgdatetime']:
                timestamp_count += 1
            else:  # timestamp != header['dgdatetime']
                total_dropped_dgms += (dgm_count_expected - timestamp_count)
                timestamp = header['dgdatetime']
                dgm_count_expected = partition["numOfDgms"]
                timestamp_count = 1
                unique_timestamp_count += 1

            print("dgm {} of {}, {}".format(partition["dgmNum"], partition["numOfDgms"], header['dgdatetime']))

            if header['dgmVersion'] == 0 or 1 or 2:
                if partition['dgmNum'] == 1:
                    cmn_part = kmall.read_EMdgmMbody()

                    if cmn_part["numBytesCmnPart"] != 12:
                        num_bytes_cmn_part_corrupt += 1
                        num_bytes_tx_info.append(cmn_part["numBytesCmnPart"])
                        partition_tx_info_bad.append(partition['dgmNum'])

                        kmall.FID.seek(offset, 0)
                        kmall.FID.read(header['numBytesDgm'])

                        continue

                # Read pingInfo:
                ping_info = kmall.read_EMdgmMRZ_pingInfo()
                print("ping_info: ", ping_info)
                if ping_info['numBytesInfoData'] != 152:
                    num_bytes_ping_info_corrupt += 1
                    num_bytes_ping_info.append(tx_info['numBytesTxInfo'])
                    partition_ping_info_bad.append(partition['dgmNum'])
                else:
                    partition_ping_info_good.append(partition['dgmNum'])

                kmall.FID.seek(offset, 0)
                kmall.FID.read(header['numBytesDgm'])

        print("mwc_count: ", mrz_count)
        print("num_bytes_cmn_part_corrupt: ", num_bytes_cmn_part_corrupt)
        print("num_bytes_tx_info_corrupt: ", num_bytes_ping_info_corrupt)
        print(num_bytes_ping_info)
        print("Bad partitions: ", partition_ping_info_bad)
        print("Bad 1s: ", partition_ping_info_bad.count(1))
        print("Bad 2s: ", partition_ping_info_bad.count(2))
        print("Bad 3s: ", partition_ping_info_bad.count(3))
        print("Bad 4s: ", partition_ping_info_bad.count(4))
        print("Bad 5s: ", partition_ping_info_bad.count(5))
        print("Good partitions: ", partition_ping_info_good)
        print("Good 1s: ", partition_ping_info_good.count(1))
        print("total_dropped_dgms: ", total_dropped_dgms)
        print("unique_timestamp_count:", unique_timestamp_count)
        print(header)

    def listofdicts2dictoflists(self, listofdicts):
        """ A utility  to convert a list of dicts to a dict of lists."""
        if listofdicts:
            needs_flattening = [k for (k,v) in listofdicts[0].items() if isinstance(v, list)]
            d_of_l = {k: [dic[k] for dic in listofdicts] for k in listofdicts[0]}
            if needs_flattening:
                # print('flattening {}'.format(needs_flattening))
                for nf in needs_flattening:
                    d_of_l[nf] = [item for sublist in d_of_l[nf] for item in sublist]
            return d_of_l
        else:
            return None

class MObject:
    def __init__(self, dgm_type):
        self.dgm_type = dgm_type
        self.num_of_dgms = None
        self.dgm_count = 0
        self.pingCnt = None

        self.common_bytes = None
        self.beam_data = []

        self.is_complete = False

    def set_num_of_dgms(self, num_of_dgms):
        self.num_of_dgms = num_of_dgms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("in_file", help="File to read. Binary file containing Kongsberg M datagrams.")

    args = parser.parse_args()

    read_test_log = ReadTestLog(args.in_file)
    # read_test_log.read_file()
    # read_test_log.read_headers_only()
    read_test_log.kmall_index_mwc()
