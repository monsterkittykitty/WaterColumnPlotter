#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A python class to replay Kongsberg .kmall and .kmwcd files over unicast/multicast.
# Adapted from Giuseppe Masetti's HydrOffice hyo2_kng code.
#
# Lynette Davis, CCOM

import argparse
import datetime
import KMALL
import logging
import numpy as np
import os
import socket
import struct
import sys
import threading
import math

# __name__ is module's name
logger = logging.getLogger(__name__)


class KmallPlayer:

    def __init__(self, path=None, replay_timing=None, tx_ip="224.1.20.40", tx_port=26103, connection=False):

        self.path = path
        # List of .kmall / .kmwcd files found at given path:
        self.file_list = self.__create_file_list()

        # If None, replay_timing defaults to real-time replaying of datagrams
        self.replay_timing = replay_timing

        self.tx_ip = tx_ip
        self.tx_port = tx_port
        self.connection = connection

        self.SOCKET_TIMEOUT = 60  # Seconds
        self.sock_out = self.__init_sockets()

        # Counter for number of sent datagrams:
        self.dg_counter = 0

        self.MAX_DATAGRAM_SIZE = 64000

        # As of REV I (April 2021), it appears data fields contained in header, partition, and cmnPart
        # structs have not changed and are consistent across all revisions (REV B - REV I).
        self.HEADER_STRUCT_FORMAT = '1I4s2B1H2I'
        self.HEADER_STRUCT_SIZE = struct.calcsize(self.HEADER_STRUCT_FORMAT)
        self.PARTITION_STRUCT_FORMAT = '2H'
        self.PARTITION_STRUCT_SIZE = struct.calcsize(self.PARTITION_STRUCT_FORMAT)
        self.CMNPART_STRUCT_FORMAT = '2H8B'
        self.CMNPART_STRUCT_SIZE = struct.calcsize(self.CMNPART_STRUCT_FORMAT)

        # All 'M' datagrams (MRZ, MWC) contain header and partition structs. As of REV I (April 2021), all 'M' datagrams
        # also contain cmnPart structs. Previous to REV I (REV B - REV H), only partition 1 contains cmnPart struct.
        # *** Note: Final four bytes of datagram repeat datagram size. ***

        # A datagram is made of a header, a partition structure and the data, ended with a 4bytes
        # integer which repeats the message size. The data part to split shall have a max length of:
        # self.MAX_DATA_SIZE = self.MAX_DATAGRAM_SIZE - self.HEADER_AND_PART_SIZE - 4

    def __create_file_list(self):
        """
        Return a list containing file paths to all .kmall / .kmwcd files at specified path.
        :param path: Path to single .kmall / .kmwcd file or to a folder containing .kmall / .kmwcd file.
        :return: A list containing file paths to all .kmall / .kmwcd files at specified path.
        """
        if os.path.isfile(self.path):
            if self.path.lower().endswith(('.kmall', '.kmwcd')):
                return [self.path]
            else:
                logger.warning("Invalid file path: {}".format(self.path))
                sys.exit(1)
        elif os.path.isdir(self.path):
            file_list = []
            for root, dirs, files in os.walk(self.path):
                for filename in files:
                    print(filename)
                    if filename.lower().endswith(('.kmall', '.kmwcd')):
                        file_list.append(os.path.join(root, filename))
            return file_list
        else:
            logger.warning("Invalid file path: {}".format(self.path))
            sys.exit(1)

    def __init_sockets(self):
        if self.connection == "TCP":
            # temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.warning("Only UDP and Multicast connections supported at this time.")
            sys.exit(1)
        elif self.connection == "UDP":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif self.connection == "Multicast":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # TTL (time-to-live) controls how many networks will receive the packet;
            # Set TTL to 1 to avoid forwarding beyond current network segment.
            ttl = struct.pack('b', 1)
            temp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
            # Set socket send buffer to size of UDP packet
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 ** 16)
        else:
            logger.warning("Connection type must be 'TCP', 'UDP', or 'Multicast'.")
            sys.exit(1)

        return temp_sock

    # def valid_file_ext(self, fp):
    #     # Error checking for appropriate file types:
    #     fp_ext = os.path.splitext(fp)[-1].lower()
    #     if fp_ext not in [".kmall", ".kmwcd"]:
    #         logger.info("SIS 5 mode -> Skipping unsupported file extension: %s" % fp)
    #         return False
    #     else:
    #         return True


    def calculate_dgm_schedule(self, df):
        """
        Inserts 'ScheduledDelay' and 'ScheduledPlay' fields into dataframe (df);
        the values in these fields are determined based on datagram timestamps (dataframe index field).
        :param df: Dataframe obtained from kmall.index_file() function.
        """
        # Find #IIP and #IOP datagrams; capture timestamps (index).
        # We will want to send #IIP and #IOP datagrams first.
        IIP_index = None
        IOP_index = None
        for index, row in df.iterrows():
            if IIP_index is None:
                if '#IIP' in row['MessageType']:
                    IIP_index = index
            if IOP_index is None:
                if '#IOP' in row['MessageType']:
                    IOP_index = index
            if IIP_index is not None and IOP_index is not None:
                break

        # Sort k.Index by timestamp
        df.sort_index(inplace=True)

        # Calculate delay:
        if self.replay_timing is None:  # Play datagrams in 'real-time'...
            # Calculate scheduled delay (earliest time (k.Index.index[0]) is reference, with delay of zero).
            sched_delay = [x - df.index[0] for x in df.index]
        else:  # Play datagrams at some fixed interval...
            # Calculate scheduled delay at some fixed interval.
            sched_delay = np.linspace(0, (len(df) * self.replay_timing), len(df), endpoint=False)

        df['ScheduledDelay'] = sched_delay
        mean_delay = df['ScheduledDelay'].mean()

        # Reset scheduled delay for #IIP and #IOP datagrams (these will play first):
        df.at[IIP_index, 'ScheduledDelay'] = df['ScheduledDelay'].iloc[0] - (2 * mean_delay)
        df.at[IOP_index, 'ScheduledDelay'] = df['ScheduledDelay'].iloc[0] - (1 * mean_delay)
        # Sort k.Index by scheduled delay (this ensures that #IIP and #IOP are moved to the front of the list
        df.sort_values(by=['ScheduledDelay'], inplace=True)

        # Calculate scheduled time to play data datagram based on scheduled delay, current time, plus a small delay:
        now = datetime.datetime.now()
        delay = 3  # Seconds
        sched_play = [now + datetime.timedelta(seconds=(x + delay)) for x in df['ScheduledDelay']]
        df['ScheduledPlay'] = sched_play

    @staticmethod
    def read_header_raw(data) -> dict:
        header = {}
        format_to_unpack = "1I4s2B1H2I"
        fields = struct.unpack(format_to_unpack, data[0:struct.calcsize(format_to_unpack)])
        # Datagram length in bytes. The length field at the start (4 bytes) and end
        # of the datagram (4 bytes) are included in the length count.
        header['numBytesDgm'] = fields[0]
        # Array of length 4. Multibeam datagram type definition, e.g. #AAA
        header['dgmType'] = fields[1]
        # Datagram version.
        header['dgmVersion'] = fields[2]
        # System ID. Parameter used for separating datagrams from different echosounders
        # if more than one system is connected to SIS/K-Controller.
        header['systemID'] = fields[3]
        # Echo sounder identity, e.g. 122, 302, 710, 712, 2040, 2045, 850.
        header['echoSounderID'] = fields[4]
        # UTC time in seconds + Nano seconds remainder. Epoch 1970-01-01.
        header['time_sec'] = fields[5]
        header['time_nanosec'] = fields[6]
        return header

    @staticmethod
    def update_header_with_dgm_size(header, new_size) -> bytes:
        header['numBytesDgm'] = new_size
        format_to_pack = "1I4s2B1H2I"
        header_in_bytes = struct.pack(format_to_pack, header['numBytesDgm'], header['dgmType'],
                                      header['dgmVersion'], header['systemID'],
                                      header['echoSounderID'], header['time_sec'], header['time_nanosec'])
        return header_in_bytes

    def partition_msg(self, msg_to_split: bytes) -> []:
        message_size = len(msg_to_split)
        if message_size <= self.MAX_DATAGRAM_SIZE:
            # No partitionning needed
            return msg_to_split
        else:
            # Header from original message
            header_dict = self.read_header_raw(msg_to_split[:self.HEADER_STRUCT_SIZE])

            rev_i_plus = False
            if header_dict['dgmType'] == b'#MRZ':
                if header_dict['dgmVersion'] >= 3:
                    rev_i_plus = True
            if header_dict['dgmType'] == b'#MWC':
                if header_dict['dgmVersion'] >= 2:
                    rev_i_plus = True

            # size_appended_to_all is size of structs that must be included in every partition
            if rev_i_plus:  # cmnPart should be included in all partitions
                size_appended_to_all = self.HEADER_STRUCT_SIZE + self.PARTITION_STRUCT_SIZE + \
                                       self.CMNPART_STRUCT_SIZE + 4
            else:  # cmnPart should only be included in partition 1
                size_appended_to_all = self.HEADER_STRUCT_SIZE + self.PARTITION_STRUCT_SIZE + 4

            max_data_size = self.MAX_DATAGRAM_SIZE - size_appended_to_all
            # Data to be split is only a subset of the datagram:
            data_size = message_size - size_appended_to_all
            # Each partition must have header and partition fields appended to it.
            numOfDgms = math.ceil(data_size / float(max_data_size))
            # Get the data content in the datagram and split it into smaller packs
            data_to_split = msg_to_split[(size_appended_to_all - 4):-4]

            messages = []
            # Partitions created in this loop will all have the max packet size of 64000
            for i in range(numOfDgms - 1):
                header = self.update_header_with_dgm_size(header_dict, self.MAX_DATAGRAM_SIZE)
                # Partition index changes
                part_struct = struct.pack("2H", numOfDgms, i+1)
                split = data_to_split[i*max_data_size:(i+1)*max_data_size]

                # Header + partition + data + message size repeated
                m = bytearray(header) + bytearray(part_struct) + bytearray(split) \
                    + bytearray(struct.pack('I', max_data_size))
                messages.append(m)

            # Last partition  must contain the rest
            rest_size = data_size % max_data_size
            header = self.update_header_with_dgm_size(header_dict, rest_size + size_appended_to_all)
            part_struct = struct.pack("2H", numOfDgms, numOfDgms)
            split = data_to_split[(numOfDgms - 1) * max_data_size:]
            m = header + part_struct + split + struct.pack('I', max_data_size)
            messages.append(m)

            return messages

    def test_message_partition(self, messages):
        pass

    def send_all_datagrams(self, fp, df):
        """
        Sends all UDP datagrams extracted from kmall file. Will send at scheduled time
        (real time) or as soon as possible after previous message is sent.
        :param fp: Binary file (.kmall / .kmwcd) to be opened and read.
        :param df: Dataframe containing datagram offsets, message sizes, and scheduled times.
        :param final_byteOffset: Close file when current byte offset equals final_byteOffset.
        """
        first_tx_time = None  # For testing
        mwc_counter = 0  # For testing

        # Open file:
        with open(fp, 'rb') as file:
            # Iterate through rows of sorted dataframe:
            for index, row in df.iterrows():
                sent = False

                # TODO: Busy waiting... Not ideal.
                # Wait for scheduled time:
                while row['ScheduledPlay'] > datetime.datetime.now():
                    pass

                if self.dg_counter == 0:  # For testing
                    first_tx_time = datetime.datetime.now()

                # Seek to position in file:
                file.seek(row['ByteOffset'], 0)
                if row['MessageSize'] <= self.MAX_DATAGRAM_SIZE:
                    # Send datagram:
                    try:
                        sent = self.sock_out.sendto(file.read(row['MessageSize']), (self.tx_ip, self.tx_port))
                    except OSError as e:
                        logger.warning("Send datagram error: %s" % e)

                    if sent:
                        self.dg_counter += 1
                else:
                    messages = self.partition_msg(file.read(row['MessageSize']))
                    for m in messages:
                        # Send datagram:
                        try:
                            sent = self.sock_out.sendto(m, (self.tx_ip, self.tx_port))
                        except OSError as e:
                            logger.warning("Send datagram error: %s" % e)
                        if sent:
                            self.dg_counter += 1

                    if row['MessageType'] == "b'#MWC'":  # For testing
                        mwc_counter += 1
                    logging.warning("Split message : size %s of type %s", str(row['MessageSize']), row['MessageType'])

            last_tx_time = datetime.datetime.now()  # For testing

            print("KMALLPLAYER, Sent: ", self.dg_counter)
            print("KMALLPLAYER, Sent MWCs: ", mwc_counter)
            print("KMALLPLAYER, First transmit: {}; Final transmit: {}; Total time: {}".format(first_tx_time, last_tx_time,
                                                                                  (last_tx_time - first_tx_time).total_seconds()))


    def play_datagrams(self, fp, df):
        if self.replay_timing is None:  # Real-time replay:
            # Replay all datagrams in single new thread:
            #threading.Timer(-1, self.send_all_datagrams(fp, df)).start()
            self.send_all_datagrams(fp, df)
        else:  # Fixed-interval reply:
            # TODO: Untested.
            # Replay all datagrams in single new thread:
            threading.Timer(-1, self.send_all_datagrams(fp, df)).start()

    def run(self):
        self.dg_counter = 0

        for file in self.file_list:
            # Index file (find offsets and sizes of each datagram):
            # Function index_file() creates a dataframe ("k.Index") containing fields for
            # "Time" (index), "ByteOffset","MessageSize", and "MessageType".
            k = KMALL.kmall(file)
            k.index_file()

            # Calculate scheduled delay and play time for each datagram:
            self.calculate_dgm_schedule(k.Index)

            self.play_datagrams(file, k.Index)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("path", help="Path to file or folder containing .kmall or .kmwcd file extensions.")
    parser.add_argument("--replay_timing", default=None, help="Interval at which to send Kongsberg datagrams.")
    parser.add_argument("tx_ip", help="IP address to send Kongsberg datagrams.")
    parser.add_argument("tx_port", help="Port to send Kongsberg datagrams.", type=int)
    parser.add_argument("--connection", default="Multicast", help="Connection type: TCP, UDP, or Multicast.",
                        choices={"TCP", "UDP", "Multicast"})

    args = parser.parse_args()

    # TODO: TCP connection not currently supported.
    if args.connection == "TCP":
        logger.warning("Only UDP and Multicast connections supported at this time.")
        sys.exit(1)

    kmall_player = KmallPlayer(path=args.path, replay_timing=args.replay_timing, tx_ip=args.tx_ip,
                               tx_port=args.tx_port, connection=args.connection)
    kmall_player.run()