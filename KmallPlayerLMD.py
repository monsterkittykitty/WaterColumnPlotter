#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A python class to replay Kongsberg .kmall files over unicast/multicast.
# Adapted from Giuseppe Masetti's HydrOffice hyo2_kng code.
#
# Lynette Davis, CCOM

import datetime
import getopt
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

    # Typing available Python 3.5+...
    # def __init__(self, files: list, replay_timing: float = None, ip_out: str = "224.1.20.40",
    #              port_out: int = 26103, port_in: int = 4001, unicast=False):
    def __init__(self, files=None, replay_timing=None, ip_out="224.1.20.40", port_out=26103, unicast=False):
        self.files = files
        self._replay_timing = replay_timing
        self.ip_out = ip_out
        self.port_out = port_out
        self.unicast = unicast

        self.sock_out = None
        self.sock_in = None

        self.dg_counter = None

        self.MAX_DATAGRAM_SIZE = 64000
        self.HEADER_STRUCT_FORMAT = '1I4s2B1H2I'
        self.HEADER_STRUCT_SIZE = struct.calcsize(self.HEADER_STRUCT_FORMAT)
        self.PART_STRUCT_SIZE = struct.calcsize("2H")
        self.HEADER_AND_PART_SIZE =self.HEADER_STRUCT_SIZE + self.PART_STRUCT_SIZE
        # A datagram is made of a header, a partition structure and the data, ended with a 4bytes
        # integer which repeats the message size. The data part to split shall have a max length of:
        self.MAX_DATA_SIZE = self.MAX_DATAGRAM_SIZE - self.HEADER_AND_PART_SIZE - 4

    @property
    def replay_timing(self):
        return self._replay_timing

    @replay_timing.setter
    def replay_timing(self, value):
        self._replay_timing = value

    def _close_sockets(self):
        if self.sock_in:
            self.sock_in.close()
            self.sock_in = None
        if self.sock_out:
            self.sock_out.close()
            self.sock_out = None

    def init_sockets(self):
        """Initialize UDP sockets"""

        # TODO: I'm not sure if this is set up correctly for unicast vs multicast...
        # Unicast / Multicast:
        self.sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Multicast only:
        if not self.unicast:
            # Allow reuse of addresses
            self.sock_out.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # TODO: I think setting TTL to 1 makes this be unicast?
            # Set messages time-to-live to 1 to avoid forwarding beyond current network segment
            ttl = struct.pack('b', 1) # TODO: How does K-Controller do this?
            self.sock_out.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
            # Set socket send buffer to size of UDP packet
            self.sock_out.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 ** 16)

            logger.debug("sock_out > buffer %sKB" %
                         (self.sock_out.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) / 1024))

    def send_single_datagram(self, f, row, final_byteOffset):
        """
        Sends single UDP datagrams extracted from kmall file.
        :param f: Opened, binary file to be read.
        :param row: Row of dataframe from indexed kmall file.
        :param final_byteOffset: Close file when current byte offset equals final_byteOffset.
        """
        f.seek(row['ByteOffset'], 0)
        sent = False
        try:
            sent = self.sock_out.sendto(f.read(row['MessageSize']), (self.ip_out, self.port_out))
        except OSError as e:
            logger.warning("%s" % e)

        if sent:
            self.dg_counter += 1

        # TODO: TESTING
        # if self.dg_counter == 5:
        #     f.close()
        #     exit()
        # if self.dg_counter in (85, 86, 87, 88, 89, 90):
        #     print(self.dg_counter, ": ", row['MessageSize'], ", ", row['MessageType'])

        if row['ByteOffset'] == final_byteOffset:
            print("Datagrams transmitted: ", self.dg_counter)
            print("Closing file.")
            f.close()

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
            # Data to be split is only a subset of the datagram:
            data_size = message_size - self.HEADER_AND_PART_SIZE - 4
            numOfDgms = math.ceil(data_size / float(self.MAX_DATA_SIZE))
            # Header from original message
            header_dict = self.read_header_raw(msg_to_split[:self.HEADER_STRUCT_SIZE])
            # Get the data content in the datagram and split it into smaller packs
            data_to_split = msg_to_split[self.HEADER_AND_PART_SIZE:-4]

            messages = []
            # Partitions created in this loop will all have the max packet size of 64000
            for i in range(numOfDgms - 1):
                header = self.update_header_with_dgm_size(header_dict, self.MAX_DATAGRAM_SIZE)
                # Partition index changes
                part_struct = struct.pack("2H", numOfDgms, i+1)
                split = data_to_split[i*self.MAX_DATA_SIZE:(i+1)*self.MAX_DATA_SIZE]
                # Header + partition + data + message size repeated
                m = bytearray(header) + bytearray(part_struct) + bytearray(split) \
                    + bytearray(struct.pack('I', self.MAX_DATA_SIZE))
                messages.append(m)

            # Last partition  must contain the rest
            rest_size = data_size % self.MAX_DATA_SIZE
            header = self.update_header_with_dgm_size(header_dict, rest_size + self.HEADER_AND_PART_SIZE + 4)
            part_struct = struct.pack("2H", numOfDgms, numOfDgms)
            split = data_to_split[(numOfDgms - 1) * self.MAX_DATA_SIZE:]
            m = header + part_struct + split + struct.pack('I', self.MAX_DATA_SIZE)
            messages.append(m)

            return messages

    def send_all_datagrams_rt(self, fp, df):
        """
        Sends all UDP datagrams extracted from kmall file. Will send at scheduled time
        (real time) or as soon as possible after previous message is sent.
        :param fp: Binary file (.kmall) to be opened and read.
        :param df: Dataframe containing datagram offsets, message sizes, and scheduled times.
        :param final_byteOffset: Close file when current byte offset equals final_byteOffset.
        """
        # self.init_sockets()
        f = open(fp, 'rb')
        # Iterate through rows of sorted dataframe:
        for index, row in df.iterrows():
            sent = False

            # if '#MRZ' in row['MessageType'] and row['MessageSize'] > 64000:
            #     ## Write it using timestamp
            #     file_name = "/tmp/MRZ_" + str(row['ByteOffset']) + ".kmall"
            #     with open(file_name, "wb") as file:
            #         logging.warning("Writing new MRZ data to %s", file_name)
            #         f.seek(row['ByteOffset'], 0)
            #         file.write(f.read(row['MessageSize']))


            # TODO: Deal with large MRZs and MWCs..
            #  For now, skip messages that are too big.

            # TODO: Busy waiting... Not ideal.
            # Wait for scheduled time:
            while row['ScheduledPlay'] > datetime.datetime.now():
                pass

            # Seek to position in file:
            f.seek(row['ByteOffset'], 0)
            if row['MessageSize'] <= self.MAX_DATAGRAM_SIZE:
                # Send datagram:
                try:
                    sent = self.sock_out.sendto(f.read(row['MessageSize']), (self.ip_out, self.port_out))
                except OSError as e:
                    logger.warning("%s" % e)

                if sent:
                    self.dg_counter += 1
            else:
                messages = self.partition_msg(f.read(row['MessageSize']))
                for m in messages:
                    # Send datagram:
                    try:
                        sent = self.sock_out.sendto(m, (self.ip_out, self.port_out))
                    except OSError as e:
                        logger.warning("%s" % e)
                    if sent:
                        self.dg_counter += 1
                logging.warning("Split message : size %s of type %s", str(row['MessageSize']), row['MessageType'])

        print("Sent: ", self.dg_counter)
        f.close()

    def create_file_list(self):
        if os.path.isfile(self.files):
            self.files = [self.files]
        elif os.path.isdir(self.files):
            tempList = []
            for root, dirs, files in os.walk(self.files):
                for filename in files:
                    print(filename)
                    if filename.lower().endswith(('.kmall', '.kmwcd')):
                        tempList.append(os.path.join(root, filename))
            self.files = tempList
        else:
            logger.warning("Invalid file path: %s" % self.files)
            sys.exit(1)

    def valid_file_ext(self, fp):
        print("File: ", fp)
        # Error checking for appropriate file types:
        fp_ext = os.path.splitext(fp)[-1].lower()
        if fp_ext not in [".kmall", ".kmwcd"]:
            logger.info("SIS 5 mode -> Skipping unsupported file extension: %s" % fp)
            return False
        else:
            return True

    def calculate_dgm_schedule(self, df):
        """
        Inserts 'ScheduledDelay' and 'ScheduledPlay' fields into dataframe;
        the values in these fields are determined based on datagram timestamps (datagram index field).
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
        if self._replay_timing is None:  # Play datagrams in 'real-time'...
            # Calculate scheduled delay (earliest time (k.Index.index[0]) is reference, with delay of zero).
            sched_delay = [x - df.index[0] for x in df.index]

        else:  # Play datagrams at some fixed interval...
            # Calculate scheduled delay at some fixed interval.
            sched_delay = np.linspace(0, (len(df) * self._replay_timing), len(df), endpoint=False)

        df['ScheduledDelay'] = sched_delay
        # Reset scheduled delay for #IIP and #IOP datagrams (these will play first):
        df.at[IIP_index, 'ScheduledDelay'] = -2
        df.at[IOP_index, 'ScheduledDelay'] = -1

        # Sort k.Index by scheduled delay
        df.sort_values(by=['ScheduledDelay'], inplace=True)

        # Calculate scheduled time to play data datagram based on delay and current time:
        # TODO: This may only be needed for real-time...
        now = datetime.datetime.now()
        sched_play = [now + datetime.timedelta(seconds=(x + 3)) for x in df['ScheduledDelay']]
        df['ScheduledPlay'] = sched_play

    def play_datagrams(self, fp, df):
        if self._replay_timing is None:  # Real-time replay:
            # TODO: Last few messages are being counted as sent, but are not being written to file at rx side.
            # Replay all datagrams in single new thread:
            threading.Timer(-1, self.send_all_datagrams_rt(fp, df)).start()
        else:  # Fixed-interval reply:
            # Schedule each datagram in its own new thread to avoid busy waiting for extended periods of time.
            # TODO: This could still have problems with overlapping messages if interval is too small.
            # TODO: Main thread could close (along with socket) before all messages are sent?
            nonMWCdgms = 0

            f = open(fp, 'rb')

            final_byteOffset = df['ByteOffset'].iloc[-1]
            now = datetime.datetime.now()

            # Iterate through rows of sorted dataframe:
            for index, row in df.iterrows():

                # Send negative and zero delay datagrams immediately (#IIP, #IOP)
                # TODO: Handle MWC datagrams.
                if row['ScheduledDelay'] <= 0 and "#MWC" not in row['MessageType']:
                    # TODO: Testing:
                    nonMWCdgms += 1
                    self.send_single_datagram(f, row, final_byteOffset)
                    if row['ScheduledDelay'] == 0:
                        now = datetime.datetime.now()

                # Schedule positive delay datagrams
                else:
                    # TODO: Handle MWC datagrams.
                    if "#MWC" not in row['MessageType']:
                        # TODO: Testing:
                        nonMWCdgms += 1
                        run_at = now + datetime.timedelta(seconds=row['ScheduledDelay'])
                        delay = (run_at - now).total_seconds()
                        threading.Timer(delay, self.send_single_datagram, [f, row, final_byteOffset]).start()

            print("Sent dgms: ", nonMWCdgms)

    def interaction(self):
        """ Read and transmit datagrams """
        self.dg_counter = 0

        self.create_file_list()

        # Iterate over list of files:
        for fp in self.files:

            if self.valid_file_ext(fp):

                # (From GM's code:)
                # try:
                #     f = open(fp, 'rb')
                #     f_sz = os.path.getsize(fp)
                # except (OSError, IOError):
                #     raise RuntimeError("Unable to open %s" % fp)

                # Index file (find offsets and sizes of each datagram):
                # Function index_file() creates a dataframe ("k.Index") containing fields for
                # "Time" (index), "ByteOffset","MessageSize", and "MessageType".
                k = KMALL.kmall(fp)
                k.index_file()

                # Calculate scheduled delay and play time for each datagram:
                self.calculate_dgm_schedule(k.Index)

                # TODO: LMD Debug:
                print("LMD Debug: ")
                #print(k.Index['MessageType'])
                #print(type(k.Index['MessageType'].iloc(1)))
                temp_df = k.Index[k.Index['MessageType'].isin(["b'#MWC'"])]
                temp_df.drop("ByteOffset", axis=1, inplace=True)
                temp_df.drop("ScheduledPlay", axis=1, inplace=True)
                print(temp_df)

                #print(k.Index['MessageType'])
                #print(k.Index['ScheduledPlay'])

                self.play_datagrams(fp, k.Index)

    def run(self):
        logger.debug("kmall_player started -> out: %s:%s, timing: %s"
                      % (self.ip_out, self.port_out, self._replay_timing))

        self.init_sockets()
        self.interaction()

        logger.debug("kmall_player ended")

    def count_datagrams(self, df):
        # TODO: For testing.
        svt = 0
        cpo = 0
        spo = 0

        for index, row in df.iterrows():
            if '#SVT' in row['MessageType']:
                svt += 1
            elif '#CPO' in row['MessageType']:
                cpo += 1
            elif '#SPO' in row['MessageType']:
                spo += 1

        print("SVT: ", svt)
        print("CPO: ", cpo)
        print("SPO: ", spo)

if __name__ == '__main__':

    # TODO: I'm not sure what all the default values should be:
    # Default values:
    # No default file name or directory
    file_m = None;
    # When replay_timing is set to None, file will replay at real-time speed
    replay_timing_m = None;
    # Default port_in, port_out, and ip_out based on G. Masseti's code
    #port_in_m = 4001
    port_out_m = 26103
    ip_out_m = "224.1.20.40"
    # Multicast by default; set unicast to True for unicast
    unicast_m = False

    # Testing:
    # ip_out = "127.0.0.1" # For testing

    # # 2019 Thunder Bay - With Water Column
    # This file has small enough MRZs to transfer over UDP:
    # file = 'data/0019_20190511_204630_ASVBEN.kmall' # For testing

    # Read command line arguments for file/directory, replay_timing, ip_address, port_out, multicast/unicast
    try:
        opts, args = getopt.getopt(sys.argv[1:], "humi:p:f:t:", ["ip=", "port=", "file=", "timing="])
    except getopt.GetoptError:
        print("kmall_player.py")
        print("-f   <file_or_directory>")
        print("-t   <replay_timing_sec>")
        print("-i   <ip_address>")
        print("-p   <port>")
        print("-m   multicast")
        print("-u   unicast")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("kmall_player.py")
            print("-f   <file_or_directory>")
            print("-t   <replay_timing_sec>")
            print("-i   <ip_address>")
            print("-p   <port>")
            print("-m   multicast")
            print("-u   unicast")
            sys.exit()
        elif opt in ('-f', '--file'):
            file_m = arg
        elif opt in ('-t', '--timing'):
            replay_timing_m = float(arg)
        elif opt in ('-i', '--ip'):
            ip_out_m = arg
        elif opt in ('-p', '--port'):
            port_out_m = int(arg)
        elif opt in ('-m', '--multicast'):
            unicast_m = False
        elif opt in ('-u', '--unicast'):
            unicast_m = True

    if file_m is None:
        print("Must enter file or directory: kmall_player.py -f <file_or_directory>")
        sys.exit()

    # Create/initialize new instance of KmallPlayer:
    player = KmallPlayer(file_m, replay_timing_m, ip_out_m, port_out_m, unicast_m)
    player.run()