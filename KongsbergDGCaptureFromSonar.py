# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# May 2021

# Description: Capture UDP datagrams from Kongsberg sonar;
# insert datagram into shared queue (multiprocessing.Queue), or write to file.

# Note: Can configure in SIS which datagrams are sent to this program along with IP and port.
# 13 May 2021: When operating on RVGS, program run with following arguments:
# "0.0.0.0" "8080" "testlog.txt" --connection "UDP"

# TODO: Break out buffering mechanism into its own method, so it can be used to reconstruct and write to a file.

import argparse
import ctypes
import datetime
import io
from KmallReaderForMDatagrams import KmallReaderForMDatagrams as k
import logging
from multiprocessing import Process
import multiprocessing as mp
import socket
import struct
import sys

logger = logging.getLogger(__name__)


class KongsbergDGCaptureFromSonar(Process):

    def __init__(self, rx_ip, rx_port, ip_protocol="UDP", socket_buffer_multiplier=4, queue_datagram=None,
                 full_ping_count=None, discard_ping_count=None, process_flag=None, out_file=None):
        super().__init__()

        print("New instance of KongsbergDGCapture.")

        self.rx_ip = rx_ip
        self.rx_port = rx_port
        self.ip_protocol = ip_protocol
        self.socket_buffer_multiplier = socket_buffer_multiplier

        # When run as main, out_file is required;
        # when run with multiprocessing, queue is required (multiprocessing.Queue)
        self.queue_datagram = queue_datagram  # multiprocessing.Queue
        self.out_file = out_file

        # TODO: Not sure if this is the best way to do it?
        if full_ping_count:
            self.full_ping_count = full_ping_count
        else:
            self.full_ping_count = mp.Value(ctypes.c_uint32, 0)

        if discard_ping_count:
            self.discard_ping_count = discard_ping_count
        else:
            self.discard_ping_count = mp.Value(ctypes.c_uint32, 0)

        # Boolean shared across processes (multiprocessing.Value)
        if process_flag:
            self.process_flag = process_flag
        else:
            self.process_flag = mp.Value(ctypes.c_bool, True)

        self.SOCKET_TIMEOUT = 10  # Seconds
        self.MAX_DATAGRAM_SIZE = 2 ** 16
        self.sock_in = self._init_socket()

        # TODO: Make this a configurable setting. When it is very large and ping rates are slow,
        #  it can cause delays in sending datagrams.
        self.MAX_NUM_PINGS_TO_BUFFER = 20

        # self.REQUIRED_DATAGRAMS = [b'#MRZ', b'#MWC', b'#SKM', b'#SPO']
        self.REQUIRED_DATAGRAMS = [b'#MWC']

        # self.buffer = {'dgmType': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
        #                'dgmVersion': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
        #                'dgTime': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
        #                'pingCnt': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
        #                'numOfDgms': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
        #                'dgmsRxed': [0] * self.MAX_NUM_PINGS_TO_BUFFER,
        #                'complete': [False] * self.MAX_NUM_PINGS_TO_BUFFER,
        #                'data': [None] * self.MAX_NUM_PINGS_TO_BUFFER}

        self.buffer = self._init_buffer()

        # TODO: FOR TESTING
        self.dgms_rxed = 0
        # self.all_data_rxed = 0
        # self.data_overwrite = 0

    def _init_socket(self):
        if self.ip_protocol == "TCP":
            # temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.warning("Only UDP and Multicast connections supported at this time.")
            sys.exit(1)

        elif self.ip_protocol == "UDP":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # TODO: Change buffer size if packets are being lost:
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                 self.MAX_DATAGRAM_SIZE * self.socket_buffer_multiplier)
            temp_sock.bind((self.rx_ip, self.rx_port))

        elif self.ip_protocol == "Multicast":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # TODO: Change buffer size if packets are being lost:
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                 self.MAX_DATAGRAM_SIZE * self.socket_buffer_multiplier)
            temp_sock.bind(('', self.rx_port))
            # Tell the operating system to add the socket to the multicast group on all interfaces.
            # (From: https://pymotw.com/2/socket/multicast.html)
            group = socket.inet_aton(self.rx_ip)
            mreq = struct.pack('4sL', group, socket.INADDR_ANY)
            temp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        else:
            raise RuntimeError("Connection type must be 'TCP', 'UDP', or 'Multicast'.")

        temp_sock.settimeout(self.SOCKET_TIMEOUT)
        return temp_sock

    def _init_buffer(self):
        buffer = {'dgmType': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'dgmVersion': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'dgTime': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'pingCnt': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'numOfDgms': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'dgmsRxed': [0] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'complete': [False] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'data': [None] * self.MAX_NUM_PINGS_TO_BUFFER}

        return buffer

    def print_settings(self):
        print("Receive (IP:Port, Connection): ", self.rx_ip, ":", self.rx_port, ",", self.ip_protocol)

    def print_packet_details(self, data):
        pass

    def receive_dg_and_write_raw(self):
        """
        Receives data at specified socket; writes binary data to specified file.
        *** Note, this does NOT reconstruct partitioned datagrams. ***
        """
        # print("Writing.")
        file_io = open(self.out_file, 'wb')

        # while self.process_flag.value:
        while True:
            with self.process_flag.get_lock():
                if not self.process_flag.value:
                    break
            self.print_settings()
            try:
                data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)

                # To write only #MWC data:
                # bytes_io = io.BytesIO(data)
                # header = k.read_EMdgmHeader(bytes_io)
                #
                # if header['dgmType'] == b'#MWC':
                #     file_io.write(data)

                # To write all data:
                file_io.write(data)

            except socket.timeout:
                logger.exception("Socket timeout exception.")
                self.sock_in.close()
                file_io.close()
                break

    def receive_dg_and_queue(self):
        """
        Receives data at specified socket; places data in specified queue (multiprocessing.Queue).
        """

        print("DGCapture: receive_dg_and_queue")  # For debugging

        mwc_counter = 0  # For testing

        timestamp_index = 0  # Index of oldest timestamp in buffer
        next_index = 0  # Index of next position in buffer to be filled

        while True:

            with self.process_flag.get_lock():
                local_process_flag_value = self.process_flag.value

            if local_process_flag_value == 1:  # Play pressed
                try:
                    data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)
                except BlockingIOError:
                    continue
                except socket.timeout:
                    logger.exception("Socket timeout exception.")
                    break

                # TODO: FOR TESTING:
                self.dgms_rxed += 1

                bytes_io = io.BytesIO(data)

                header = k.read_EMdgmHeader(bytes_io)

                #print("header[numBytesDgm]: ", header['numBytesDgm'], type(header['dgmType']))

                if header['dgmType'] in self.REQUIRED_DATAGRAMS:
                    # print("header['dgmType']", header['dgmType'])
                    # print("header['dgTime']", header['dgTime'])
                    if header['dgmType'] == b'#MRZ' or header['dgmType'] == b'#MWC':  # Datagrams may be partitioned

                        # For testing:
                        if header['dgmType'] == b'#MWC':
                            mwc_counter += 1
                            #print("dgm_timestamp: ", header['dgdatetime'], "mwc_counter: ", mwc_counter)

                        partition = k.read_EMdgmMpartition(bytes_io, header['dgmType'], header['dgmVersion'])

                        if partition['numOfDgms'] == 1:  # Only one datagram; no need to reconstruct
                            # print("Num {} partitions is 1.".format(header['dgmType']))
                            self.queue_datagram.put(data)
                            with self.full_ping_count.get_lock():
                                self.full_ping_count.value += 1

                        else:  # Greater than one datagram; needs to be reconstructed
                            # Check for timestamp in buffer:
                            if header['dgTime'] in self.buffer['dgTime']:  # Timestamp in buffer

                                # For testing:
                                #print("Timestamp in buffer: {}, {}".format(header['dgmType'], header['dgTime']))

                                index = self.buffer['dgTime'].index(header['dgTime'])

                                # Append new data to existing data in buffer:
                                self.buffer['dgmsRxed'][index] += 1
                                self.buffer['data'][index][partition['dgmNum'] - 1] = data

                                # For debugging:
                                # print("Inserting existing datagram {}, {} into index {}. Part {} of {}."
                                #       .format(header['dgmType'], header['dgTime'], index,
                                #               partition['dgmNum'], partition['numOfDgms']))
                                # print("datagrams rxed: ", self.buffer['dgmsRxed'][index])
                                # print("Existing: self.buffer['dgmsRxed'][index]:", self.buffer['dgmsRxed'][index])
                                # print("Existing: self.buffer['numOfDgms'][index]:", self.buffer['numOfDgms'][index])

                                # Check if all data received:
                                if self.buffer['dgmsRxed'][index] == self.buffer['numOfDgms'][index]:

                                    # self.all_data_rxed += 1

                                    # For testing:
                                    # print("All data received: {}, {}, ping: ".format(self.buffer['dgmType'],
                                    #                                                  self.buffer['dgTime'],
                                    #                                                  self.buffer['pingCnt']))

                                    self.buffer['complete'][index] = True

                                    # Check if current index equals (earliest) timestamp_index:
                                    if index == timestamp_index:

                                        # For testing:
                                        # print("Reconstructing datagram.")

                                        # Earliest timestamp index is complete! Reconstruct data and place in queue
                                        data_reconstruct, data_size = self.reconstruct_data(
                                            self.buffer['dgmType'][timestamp_index],
                                            self.buffer['dgmVersion'][timestamp_index],
                                            self.buffer['data'][timestamp_index])

                                        self.queue_datagram.put(data_reconstruct)
                                        with self.full_ping_count.get_lock():
                                            self.full_ping_count.value += 1

                                        # For debugging:
                                        # print("Complete datablock: {}, {}, {} bytes"
                                        #       .format(self.buffer['dgmType'][timestamp_index],
                                        #               self.buffer['dgTime'][timestamp_index], data_size))

                                        # Clear entry
                                        # TODO: Practically, do I need to clear any more than this?
                                        self.buffer['dgTime'][timestamp_index] = None
                                        self.buffer['complete'][timestamp_index] = False  # Probably not necessary?

                                        # Advance timestamp_index to oldest timestamp in buffer
                                        self.advance_timestamp_index()

                                    # else:  # index != timestamp_index
                                    #     print("Not reconstructing datagram. Index: {}, Timestamp index: {}"
                                    #           .format(index, timestamp_index))

                            else:  # Timestamp not in buffer

                                # For testing:
                                # print("Timestamp not in buffer: {}, {}".format(header['dgmType'], header['dgTime']))

                                # Check whether next_index currently points to incomplete data.
                                # If so, log and discard / overwrite data.
                                if self.buffer['dgTime'][next_index] is not None:  # Index contains data

                                    if next_index == timestamp_index:  # This should always be True
                                        if self.buffer['complete'][next_index]:
                                            # Earliest timestamp index is complete! Reconstruct data and place in queue
                                            data_reconstruct, data_size = self.reconstruct_data(
                                                self.buffer['dgmType'][timestamp_index],
                                                self.buffer['dgmVersion'][timestamp_index],
                                                self.buffer['data'][timestamp_index])

                                            self.queue_datagram.put(data_reconstruct)
                                            with self.full_ping_count.get_lock():
                                                self.full_ping_count.value += 1

                                            # Clear entry
                                            # TODO: Practically, do I need to clear any more than this?
                                            self.buffer['dgTime'][timestamp_index] = None
                                            self.buffer['complete'][timestamp_index] = False  # Probably not necessary?

                                            # Advance timestamp_index to oldest timestamp in buffer
                                            self.advance_timestamp_index()

                                        else:  # Earliest timestamp is not complete; overwrite
                                            # # For debugging:
                                            # print("Overwriting data at index {}. This timestamp: {}. All timestamps: {}"
                                            #       .format(next_index, self.buffer['dgTime'][next_index], self.buffer['dgTime']))

                                            empty_data_reconstruct, data_size = self.reconstruct_empty_data(
                                                self.buffer['dgmType'][next_index],
                                                self.buffer['dgmVersion'][next_index],
                                                self.buffer['data'][next_index])

                                            self.queue_datagram.put(empty_data_reconstruct)
                                            with self.discard_ping_count.get_lock():
                                                self.discard_ping_count.value += 1
                                                # print("discard value: ", self.discard_ping_count.value)

                                            logger.warning("Data block incomplete. Discarding {}, {}. (Ping {}, {} of {} datagrams.) "
                                                           "\nConsidering increasing size of buffer. (Current buffer size: {}.)"
                                                           .format(self.buffer['dgmType'][next_index],
                                                                   self.buffer['dgTime'][next_index],
                                                                   self.buffer['pingCnt'][next_index],
                                                                   self.buffer['dgmsRxed'][next_index],
                                                                   self.buffer['numOfDgms'][next_index],
                                                                   self.MAX_NUM_PINGS_TO_BUFFER))

                                            # self.data_overwrite += 1
                                            # print("All data rx to data overwrite: {}:{}".format(self.all_data_rxed, self.data_overwrite))

                                    # For testing:
                                    # print("Next index: {}. Timestamp index and timestamp: {}, {}. All timestamps: {}"
                                    #       .format(next_index, timestamp_index,
                                    #               self.buffer['dgTime'][timestamp_index], self.buffer['dgTime']))

                                    # Error checking: If we are overwriting data, next_index must equal timestamp_index or
                                    # something is wrong! This should never print. If it does, there's an error in code.
                                    else:  # next_index != timestamp_index:
                                        logger.error("Error indexing incoming data buffer. Next: {}, Timestamp: {}. "
                                                     "\nThis should never print; if it does, there's an error in the code."
                                                     .format(next_index, timestamp_index))

                                    # Advance timestamp_index to oldest timestamp in buffer
                                    #timestamp_index = self.buffer['dgTime'].index(min(self.buffer['dgTime']))

                                # Insert new data into self.buffer, overwriting existing data if present:
                                cmnPart = k.read_EMdgmMbody(bytes_io, header['dgmType'], header['dgmVersion'])

                                # For debugging:
                                # print("Inserting new datagram {}, {} into index {}. Part {} of {}."
                                #       .format(header['dgmType'], header['dgTime'], next_index,
                                #               partition['dgmNum'], partition['numOfDgms']))
                                # print("Current time: {}; dg_timestamp: {}; difference: {}".format(
                                #     datetime.datetime.now(), datetime.datetime.utcfromtimestamp(header['dgTime']),
                                #     (datetime.datetime.now() - datetime.datetime.utcfromtimestamp(header['dgTime'])).total_seconds()))

                                self.buffer['dgmType'][next_index] = header['dgmType']
                                self.buffer['dgmVersion'][next_index] = header['dgmVersion']
                                self.buffer['dgTime'][next_index] = header['dgTime']
                                self.buffer['pingCnt'][next_index] = cmnPart['pingCnt']
                                self.buffer['numOfDgms'][next_index] = partition['numOfDgms']
                                self.buffer['dgmsRxed'][next_index] = 1
                                self.buffer['complete'][next_index] = False
                                # Initialize a data array at this position with length equal to numOfDgms
                                self.buffer['data'][next_index] = [None] * partition['numOfDgms']
                                # Insert data at appropriate position
                                self.buffer['data'][next_index][partition['dgmNum'] - 1] = data

                                # For debugging:
                                # print("dgmVersion:", header['dgmVersion'])
                                # print("numBytesCmnPart:", cmnPart['numBytesCmnPart'])
                                # print("Inserting new datagram {}, {} into index {}. Part {} of {}."
                                #       .format(header['dgmType'], header['dgTime'], next_index,
                                #               partition['dgmNum'], partition['numOfDgms']))
                                # print("New: self.buffer['dgmsRxed'][index]:", self.buffer['dgmsRxed'][next_index])
                                # print("New: self.buffer['numOfDgms'][index]:", self.buffer['numOfDgms'][next_index])

                                # Advance timestamp_index to oldest timestamp in buffer
                                # This should be OK. We don't need recursive method here because timestamp_index should
                                # already be pointing to the earliest timestamp. This newly added datagram is the only
                                # timestamp that could be earlier than those already existing in the buffer, and a newly
                                # added datagram is certainly incomplete.
                                timestamp_index = self.buffer['dgTime'] \
                                    .index(min(time for time in self.buffer['dgTime'] if time is not None))

                                # Advance next_index
                                # If unused slots exits in self.buffer, select first unused slot
                                if None in self.buffer['dgTime']:
                                    next_index = self.buffer['dgTime'].index(None)
                                # Otherwise, select position with oldest timestamp
                                else:
                                    next_index = timestamp_index

            elif local_process_flag_value == 2:  # Pause pressed
                print("flushing buffer")
                # Flush completed datagrams in buffer into queue_datagram
                self.flush_buffer()
                # Poison pill
                self.queue_datagram.put(None)

                break  # Exit loop

            elif local_process_flag_value == 3:  # Stop pressed
                print("discarding buffer contents")
                # Discard all datagrams in buffer
                self.buffer = self._init_buffer()
                # Poison pill
                self.queue_datagram.put(None)
                break  # Exit loop

            else:
                logger.error("Error in KongsbergDGCaptureFromSonar. Invalid process_flag value: {}."
                             .format(local_process_flag_value))
                break  # Exit loop

        print("closing socket")
        self.sock_in.close()

    def flush_buffer(self):
        """
        Recursive method to flush all complete records from buffer after pause or stop command or socket timeout.
        """
        # For debugging:
        print("Flushing buffer. Timestamps: ", self.buffer['dgTime'])
        temp_index = 0

        # If all items in self.buffer['dgTime'] are None:
        if self.buffer['dgTime'] == [None] * len(self.buffer['dgTime']):
            return
        else:
            # Advance temp_index to oldest timestamp in buffer
            temp_index = self.buffer['dgTime'] \
                .index(min(time for time in self.buffer['dgTime'] if time is not None))

            # When advancing index to existing entry, check if complete:
            if self.buffer['complete'][temp_index]:  # Data at temp_index is complete!
                data_reconstruct, data_size = self.reconstruct_data(
                    self.buffer['dgmType'][temp_index],
                    self.buffer['dgmVersion'][temp_index],
                    self.buffer['data'][temp_index])

                # Add reconstructed data to queue
                self.queue_datagram.put(data_reconstruct)
                with self.full_ping_count.get_lock():
                    self.full_ping_count.value += 1

                # For debugging:
                print("Flushing buffer; complete datablock: {}, {}, {} bytes"
                      .format(self.buffer['dgmType'][temp_index],
                              self.buffer['dgTime'][temp_index], data_size))

            # Whether or not data block is complete, clear entry:
            self.buffer['dgTime'][temp_index] = None
            self.buffer['complete'][temp_index] = False

            # Advance timestamp again to continue flushing buffer
            self.flush_buffer()

    def advance_timestamp_index(self):
        """
        Recursive method to advance timestamp_index (index of earliest active timestamp in self.buffer) and check if
        data at new timestamp_index is complete. If data is complete, reconstruct data, add to queue, and repeat.
        :return: New timestamp_index (index of earliest active timestamp in self.buffer).
        If no active timestamps, return 0.
        """

        # If all items in self.buffer['dgTime'] are None:
        if self.buffer['dgTime'] == [None] * len(self.buffer['dgTime']):
            timestamp_index = 0
        else:
            # Advance timestamp_index to oldest timestamp in buffer
            timestamp_index = self.buffer['dgTime'] \
                .index(min(time for time in self.buffer['dgTime'] if time is not None))

            # When advancing index to existing entry, check if complete:
            if self.buffer['complete'][timestamp_index]:  # Data at timestamp_index is complete!
                data_reconstruct, data_size = self.reconstruct_data(
                    self.buffer['dgmType'][timestamp_index],
                    self.buffer['dgmVersion'][timestamp_index],
                    self.buffer['data'][timestamp_index])

                # Add reconstructed data to queue
                self.queue_datagram.put(data_reconstruct)
                with self.full_ping_count.get_lock():
                    self.full_ping_count.value += 1

                # For debugging:
                # print("Advancing timestamp; complete datablock: {}, {}, {} bytes"
                #       .format(self.buffer['dgmType'][timestamp_index],
                #               self.buffer['dgTime'][timestamp_index], data_size))

                # Clear entry
                self.buffer['dgTime'][timestamp_index] = None
                self.buffer['complete'][timestamp_index] = False

                # Advance timestamp again
                self.advance_timestamp_index()

        return timestamp_index

    def reconstruct_empty_data(self, dgmType, dgmVersion, data):
        temp_buffer = []
        numBytesDgm = 0

        header_struct_format = k.read_EMdgmHeader(None, return_format=True)
        partition_struct_format = k.read_EMdgmMpartition(None, dgmType, dgmVersion, return_format=True)
        # cmnPart_struct_format = k.read_EMdgmMbody(data, dgmType, dgmVersion, return_format=True)

        length_to_strip = struct.calcsize(header_struct_format) + \
                          struct.calcsize(partition_struct_format)

        index = 0
        temp_datagram = data[index]

        while not temp_datagram:
            index += 1
            if index < len(data):
                temp_datagram = data[index]
            else:
                # This should never occur. If this method is being called, there should always be *something* present.
                return None

        # Add header and partition data to temp_buffer
        temp_buffer.append(bytearray(temp_datagram[:length_to_strip]))
        # temp_buffer.append(bytearray(b''))
        numBytesDgm += len(temp_datagram[:length_to_strip])

        # Adjust header values
        numBytesDgm_packed = struct.pack("I", numBytesDgm)  # Returns type <class 'bytes'>
        temp_buffer[0][:struct.calcsize("I")] = numBytesDgm_packed

        # Adjust partition values
        partition_packed = struct.pack("2H", 1, 1)  # Returns type <class 'bytes'>
        temp_buffer[0][struct.calcsize(header_struct_format):
                       (struct.calcsize(header_struct_format) +
                        struct.calcsize(partition_struct_format))] = partition_packed

        # Flatten buffer
        flat_buffer = self.flatten_buffer(temp_buffer)

        return flat_buffer, numBytesDgm

    def reconstruct_data(self, dgmType, dgmVersion, data):
        """

        :param data: A sorted list containing all raw, partitioned Kongsberg datagrams from a single ping.
        Example: [<ping 1 - datagram 1 of 3>, <ping 1 - datagram 2 of 3>, <ping 1 - datagram 3 of 3>].
        :return: A single reconstructed (non-partitioned) datagram.
        """
        temp_buffer = []
        numBytesDgm = 0

        header_struct_format = k.read_EMdgmHeader(None, return_format=True)
        partition_struct_format = k.read_EMdgmMpartition(None, dgmType, dgmVersion, return_format=True)
        cmnPart_struct_format = k.read_EMdgmMbody(None, dgmType, dgmVersion, return_format=True)

        # Length to strip for Kongsberg *.kmall datagram format revisions A - H.
        length_to_strip = struct.calcsize(header_struct_format) + struct.calcsize(partition_struct_format)

        # Determine Kongsberg *.kmall datagram format revision version.
        # Revision A - H contain cmnPart only in partition 1; revisions I+ contain cmnPart in all partitions.
        # Revision I updated #MRZ datagram to version 3 and #MWC datagram to version 2.
        if (dgmType == b'#MRZ' and dgmVersion >= 3) or (dgmType == b'#MWC' and dgmVersion >= 2):
            # Length to strip for Kongsberg *.kmall datagram format revisions I+.
            length_to_strip += struct.calcsize(cmnPart_struct_format)

        for i in range(len(data)):
            if i == 0:  # First dgm must have last 4 bytes removed
                temp_buffer.append(bytearray(data[i][:-4]))
                numBytesDgm += len(data[i][:-4])
            else:  # Final dgm(s) must have leading fields and last 4 bytes removed
                temp_buffer.append(bytearray(data[i][length_to_strip:-4]))
                numBytesDgm += len(data[i][length_to_strip:-4])

        # Add 4 to numBytesDgm to account for 4-byte size field to be appended to end of datagram
        numBytesDgm += 4

        # Adjust header values
        numBytesDgm_packed = struct.pack("I", numBytesDgm)  # Returns type <class 'bytes'>
        temp_buffer[0][:struct.calcsize("I")] = numBytesDgm_packed

        # Adjust partition values
        partition_packed = struct.pack("2H", 1, 1)  # Returns type <class 'bytes'>
        temp_buffer[0][struct.calcsize(header_struct_format):
                       (struct.calcsize(header_struct_format) +
                        struct.calcsize(partition_struct_format))] = partition_packed

        # Flatten buffer
        flat_buffer = self.flatten_buffer(temp_buffer)

        # Add final 4-byte size field:
        flat_buffer += struct.pack("I", numBytesDgm)

        return flat_buffer, numBytesDgm

    def flatten_buffer(self, buffer):
        flat_buffer = b''
        for item in buffer:
            flat_buffer += item
        return flat_buffer

    def run(self):
        #print("Running KongsbergDGCapture process.")
        if self.queue_datagram:
            self.receive_dg_and_queue()

        else:
            self.receive_dg_and_write_raw()


if __name__ == "__main__":
    """
    If run as main, program will capture datagrams from Kongsberg sonar and write to file specified ("out_file").
    Note: No file rotation has been implemented; all datagrams will be written to a single file.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("rx_ip", help="IP address to receive Kongsberg datagrams.")
    parser.add_argument("rx_port", help="Port to receive Kongsberg datagrams.", type=int)
    parser.add_argument("--connection", default="Multicast", help="Connection type: TCP, UDP, or Multicast.",
                        choices={"TCP", "UDP", "Multicast"})
    parser.add_argument("out_file", help="Path to write file.")

    args = parser.parse_args()

    # TODO: TCP connection not currently supported.
    if args.connection == "TCP":
        logger.warning("Only UDP and Multicast connections supported at this time.")
        sys.exit(1)

    kongsberg_dg_capture_from_sonar = KongsbergDGCaptureFromSonar(args.rx_ip, args.rx_port,
                                                                  args.connection, out_file=args.out_file)
    kongsberg_dg_capture_from_sonar.run()
