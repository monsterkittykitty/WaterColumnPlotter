# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# May 2021

# Description: Captures UDP datagrams directly from Kongsberg sonar system or SIS; reconstructs partitioned 'M'
# datagrams; inserts datagrams into shared queue (multiprocessing.Queue), or writes to file.

# Note: Can receive datagrams directly from Kongsberg sonar system (recommended) by listening for multicast UDP packets
# in the same way that SIS does (generally at multicast address: 225.255.255.255; and multicast port: 6020).
# Alternatively, can receive datagrams forwarded by SIS using SIS's Data Distribution Table;
# here, one can configure specify IP and port for datagram forwarding and which datagrams are sent.

# TODO: Break out buffering mechanism for reconstructing 'M' datagrams
#  into its own method, so it can be used when writing to a file.

import argparse
import ctypes
import io
from WaterColumnPlotter.Kongsberg.KmallReaderForMDatagrams import KmallReaderForMDatagrams as k
import logging
from multiprocessing import Process
import multiprocessing as mp
import socket
import struct
import sys

logger = logging.getLogger(__name__)


class KongsbergDGCaptureFromSonar(Process):

    def __init__(self, ip, port, protocol, socket_buffer_multiplier, settings_edited, queue_datagram,
                 full_ping_count=None, discard_ping_count=None, process_flag=None, out_file=None):
        super().__init__()

        self.ip = ip  # multiprocessing.Array
        self.port = port  # multiprocessing.Value
        self.protocol = protocol  # multiprocessing.Value
        self.socket_buffer_multiplier = socket_buffer_multiplier  # multiprocessing.Value

        # A boolean flag to indicate when settings have been edited
        self.settings_edited = settings_edited  # multiprocessing.Value
        # Local boolean flag indicating whether IP settings have been edited
        # self.ip_settings_edited = False

        # Local copies of above multiprocessing.Array and multiprocessing.Values (to avoid frequent accessing of locks)
        self.ip_local = None
        self.port_local = None
        self.protocol_local = None
        self.socket_buffer_multiplier_local = None
        # Initialize above local copies
        self.update_local_settings()

        # When run as main, out_file is required;
        # when run with multiprocessing, queue is required (multiprocessing.Queue)
        self.queue_datagram = queue_datagram  # multiprocessing.Queue
        self.out_file = out_file  # Path to file for writing data

        # A count to track the number of full #MWC records (pings) received and reconstructed
        if full_ping_count:
            self.full_ping_count = full_ping_count  # multiprocessing.Value
        else:
            self.full_ping_count = mp.Value(ctypes.c_uint32, 0)

        # A count to track the number of #MWC records (pings) that could not be reconstructed
        if discard_ping_count:
            self.discard_ping_count = discard_ping_count  # multiprocessing.Value
        else:
            self.discard_ping_count = mp.Value(ctypes.c_uint32, 0)

        # A boolean flag to indicate status of process. When true, this process is able to listen for and receive
        # incoming datagrams; when false, this process is unable to listen for or receive incoming datagrams.
        if process_flag:
            self.process_flag = process_flag  # multiprocessing.Value
        else:
            self.process_flag = mp.Value(ctypes.c_bool, True)

        # TODO: Do we need / want a socket timeout?
        self.SOCKET_TIMEOUT = 10  # Seconds
        self.MAX_DATAGRAM_SIZE = 2 ** 16  # Maximum size of UDP packet
        self.sock_in = self._init_socket()

        # self.REQUIRED_DATAGRAMS = [b'#MRZ', b'#MWC', b'#SKM', b'#SPO']
        self.REQUIRED_DATAGRAMS = [b'#MWC']

        # The number of pings with partial data that can be accomodated in the buffer before discarding / overwriting
        # old data. Note that when this number becomes large, there are likely to be greater delays in sending
        # reconstructed data to the next process.
        self.MAX_NUM_PINGS_TO_BUFFER = 20

        # Buffer to accomodate pings with partial data prior to reconstruction
        self.buffer = self._init_buffer()

    def update_local_settings(self):
        """
        At object initialization, this method initializes local copies of shared variables;
        after initialization, this method updates local copies of shared variables when settings are changed.
        """
        # Outer lock to ensure atomicity of updates; this lock must be held when updating settings.
        with self.settings_edited.get_lock():
            with self.ip.get_lock():
                # if self.ip_local and self.ip_local != self.editIP(self.ip[:], append=False):
                #     self.ip_settings_edited = True
                self.ip_local = self.editIP(self.ip[:], append=False)
            with self.port.get_lock():
                # if self.port_local and self.port_local != self.port.value:
                #     self.ip_settings_edited = True
                self.port_local = self.port.value
            with self.protocol.get_lock():
                # if self.protocol_local and self.protocol_local != self.protocol.value:
                #     self.ip_settings_edited = True
                self.protocol_local = self.protocol.value
            with self.socket_buffer_multiplier.get_lock():
                # if self.socket_buffer_multiplier_local and \
                #         self.socket_buffer_multiplier_local != self.socket_buffer_multiplier.value:
                #     self.ip_settings_edited = True
                self.socket_buffer_multiplier_local = self.socket_buffer_multiplier.value

    def _init_socket(self):
        """
        Initializes UDP or Multicast socket; TCP sockets not supported.
        """
        if self.protocol_local == "T":  # TCP
            # temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.warning("Only UDP and Multicast connections supported at this time.")
            sys.exit(1)

        elif self.protocol_local == "U":  # UDP
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Note: If packets are being lost, try increasing size of self.socket_buffer_multiplier_local?
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                 self.MAX_DATAGRAM_SIZE * self.socket_buffer_multiplier_local)
            temp_sock.bind((self.ip_local, self.port_local))

        elif self.protocol_local == "M":  # Multicast
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Note: If packets are being lost, try increasing size of self.socket_buffer_multiplier_local?
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                 self.MAX_DATAGRAM_SIZE * self.socket_buffer_multiplier_local)
            temp_sock.bind(('', self.port_local))
            # Tell the operating system to add the socket to the multicast group on all interfaces.
            # (From: https://pymotw.com/2/socket/multicast.html)
            group = socket.inet_aton(self.ip_local)
            mreq = struct.pack('4sL', group, socket.INADDR_ANY)
            temp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        else:
            raise RuntimeError("Connection type must be 'TCP', 'UDP', or 'Multicast'.")

        # TODO: Do we need / want a socket timeout?
        temp_sock.settimeout(self.SOCKET_TIMEOUT)
        return temp_sock

    def _init_buffer(self):
        """
        Initializes buffer to accommodate incomplete #MWC records (pings). Records are either: 1) completed,
        reconstructed, and sent; or, 2) discarded and overwritten if incomplete and buffer is full.
        """
        buffer = {'dgmType': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'dgmVersion': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'dgTime': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'pingCnt': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'numOfDgms': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'dgmsRxed': [0] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'complete': [False] * self.MAX_NUM_PINGS_TO_BUFFER,
                  'data': [None] * self.MAX_NUM_PINGS_TO_BUFFER}

        return buffer

    def editIP(self, ip, append=True):
        """
        IP addresses shared between processes must be 15 characters in length when stored as a multiprocessing.Array.
        When append is true, this method appends characters ("_") to the left side of the IP address string to meet the
        15-character requirement; when append is false, those characters ("_") are stripped from the left side of the
        IP address string so that the string can be interpreted as a valid ip address.
        :param ip: IP address, with or without special characters ("_") appended
        :param append: When true, special characters are appended to ip;
        when false, special characters are stripped from ip
        """
        if append:  # Append
            while len(ip) < 15:
                ip = "_" + ip
        else:  # Strip
            ip = ip.lstrip("_")

        return ip

    def print_settings(self):
        """
        Prints IP settings.
        """
        print("Receive (IP:Port, Connection): ", self.ip_local, ":", self.port_local, ",", self.protocol_local)

    def receive_dg_and_write_raw(self):
        """
        Receives data at specified socket; writes binary data to specified file.
        This is meant to only be used when KongsbergDGCaptureFromSonar is run as main.
        *** Note, this does NOT reconstruct partitioned datagrams. ***
        """
        file_io = open(self.out_file, 'wb')

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
        Receives data at specified socket; buffers incomplete #MWC records; reconstructs #MWC records when all
        partitions received; places complete data records in specified shared queue (multiprocessing.Queue).
        """
        # mwc_counter = 0  # For debugging

        timestamp_index = 0  # Index of oldest timestamp in buffer
        next_index = 0  # Index of next position in buffer to be filled

        while True:

            with self.process_flag.get_lock():
                local_process_flag_value = self.process_flag.value

            if local_process_flag_value == 1:  # Play pressed
                # Check for signal to update settings:
                with self.settings_edited.get_lock():
                    if self.settings_edited.value:
                        # Note that all local settings in this process are IP-related settings.
                        # If these settings are updated, the current socket must closed and reinitialized.
                        self.update_local_settings()
                        # Flush buffer here? Probably not totally necessary.
                        self.sock_in.close()
                        self.sock_in = self._init_socket()
                        self.settings_edited.value = False

                try:
                    data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)
                except BlockingIOError:
                    continue
                except socket.timeout:
                    logger.exception("Socket timeout exception.")
                    break

                bytes_io = io.BytesIO(data)

                header = k.read_EMdgmHeader(bytes_io)

                if header['dgmType'] in self.REQUIRED_DATAGRAMS:
                    if header['dgmType'] == b'#MRZ' or header['dgmType'] == b'#MWC':  # Datagrams may be partitioned

                        # For debugging:
                        # if header['dgmType'] == b'#MWC':
                        #     mwc_counter += 1
                        #     print("dgm_timestamp: ", header['dgdatetime'], "mwc_counter: ", mwc_counter)

                        partition = k.read_EMdgmMpartition(bytes_io, header['dgmType'], header['dgmVersion'])

                        if partition['numOfDgms'] == 1:  # Only one datagram; no need to reconstruct
                            self.queue_datagram.put(data)
                            with self.full_ping_count.get_lock():
                                self.full_ping_count.value += 1

                        else:  # Greater than one datagram; needs to be reconstructed
                            # Check for timestamp in buffer:
                            if header['dgTime'] in self.buffer['dgTime']:  # Timestamp in buffer

                                index = self.buffer['dgTime'].index(header['dgTime'])

                                # Though not strictly necessary, adding an accurate ping count to each
                                # record can help with debugging.
                                # NOTE: Ping count is included in the 'cmnPart' field of #MWC datagrams.
                                # Kongsberg's datagram revisions B - H include the 'cmnPart' field of a
                                # partitioned datagram in only partition #1. (This policy is reflected in versions
                                # 0 - 1 of the #MWC datagram.) Revision I+ includes the 'cmnPart' field of a
                                # partitioned datagram in all partitions. (This change is reflected in version 2+
                                # of the #MWC datagram.)
                                # If 'cmnPart' field was present in first partition to be added to buffer,
                                # the 'pingCnt' field of the buffer will have already been populated; if not, the
                                # 'pingCnt' field of the buffer will be None.
                                if self.buffer['pingCnt'][index] is None:
                                    if header['dgmVersion'] == 2 or partition['dgmNum'] == 1:
                                        cmnPart = k.read_EMdgmMbody(bytes_io, header['dgmType'], header['dgmVersion'])

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

                                    # For debugging:
                                    # print("All data received: {}, {}, ping: ".format(self.buffer['dgmType'],
                                    #                                                  self.buffer['dgTime'],
                                    #                                                  self.buffer['pingCnt']))

                                    self.buffer['complete'][index] = True

                                    # Check if current index equals (earliest) timestamp_index:
                                    if index == timestamp_index:

                                        # For debugging:
                                        # print("Reconstructing datagram.")

                                        # Earliest timestamp index is complete; reconstruct data and place in queue
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

                                        # Clear entry; practically, we don't need to clear any more than this
                                        self.buffer['dgTime'][timestamp_index] = None
                                        self.buffer['complete'][timestamp_index] = False

                                        # Advance timestamp_index to oldest timestamp in buffer
                                        self.advance_timestamp_index()

                                    # For debugging:
                                    # else:  # index != timestamp_index
                                    #     print("Not reconstructing datagram. Index: {}, Timestamp index: {}"
                                    #           .format(index, timestamp_index))

                            else:  # Timestamp not in buffer

                                # For debugging:
                                # print("Timestamp not in buffer: {}, {}".format(header['dgmType'], header['dgTime']))

                                # Check whether next_index currently points to incomplete data.
                                # If so, log and discard / overwrite data.
                                if self.buffer['dgTime'][next_index] is not None:  # Index contains data

                                    if next_index == timestamp_index:  # This should always be True
                                        if self.buffer['complete'][next_index]:
                                            # Earliest timestamp index is complete; reconstruct data and place in queue
                                            data_reconstruct, data_size = self.reconstruct_data(
                                                self.buffer['dgmType'][timestamp_index],
                                                self.buffer['dgmVersion'][timestamp_index],
                                                self.buffer['data'][timestamp_index])

                                            self.queue_datagram.put(data_reconstruct)
                                            with self.full_ping_count.get_lock():
                                                self.full_ping_count.value += 1

                                            # Clear entry; practically, we don't need to clear any more than this
                                            self.buffer['dgTime'][timestamp_index] = None
                                            self.buffer['complete'][timestamp_index] = False

                                            # Advance timestamp_index to oldest timestamp in buffer
                                            self.advance_timestamp_index()

                                        else:  # Earliest timestamp is not complete; overwrite
                                            # For debugging:
                                            # print("Overwriting data at index {}. This timestamp: {}. "
                                            #       "All timestamps: {}".format(next_index,
                                            #                                   self.buffer['dgTime'][next_index],
                                            #                                   self.buffer['dgTime']))

                                            empty_data_reconstruct, data_size = self.reconstruct_empty_data(
                                                self.buffer['dgmType'][next_index],
                                                self.buffer['dgmVersion'][next_index],
                                                self.buffer['data'][next_index])

                                            self.queue_datagram.put(empty_data_reconstruct)
                                            with self.discard_ping_count.get_lock():
                                                self.discard_ping_count.value += 1

                                            logger.warning("Data block incomplete. Discarding {}, {}. (Ping {}, {} of {} datagrams.) "
                                                           "\nConsidering increasing size of buffer. (Current buffer size: {}.)"
                                                           .format(self.buffer['dgmType'][next_index],
                                                                   self.buffer['dgTime'][next_index],
                                                                   self.buffer['pingCnt'][next_index],
                                                                   self.buffer['dgmsRxed'][next_index],
                                                                   self.buffer['numOfDgms'][next_index],
                                                                   self.MAX_NUM_PINGS_TO_BUFFER))

                                    # For debugging:
                                    # print("Next index: {}. Timestamp index and timestamp: {}, {}. All timestamps: {}"
                                    #       .format(next_index, timestamp_index, self.buffer['dgTime'][timestamp_index],
                                    #               self.buffer['dgTime']))

                                    # Error checking: When overwriting data, next_index must equal timestamp_index or
                                    # something is wrong! This should never print. If it does, there's an error in code.
                                    else:  # next_index != timestamp_index:
                                        logger.error("Error indexing incoming data buffer. Next: {}, Timestamp: {}. "
                                                     "\nThis should never print; if it does, there's an error in the "
                                                     "code.".format(next_index, timestamp_index))

                                # Insert new data into self.buffer, overwriting existing data if present:
                                # NOTE: Kongsberg's datagram revisions B - H include the 'cmnPart' field of a
                                # partitioned datagram in only partition #1. (This policy is reflected in versions
                                # 0 - 1 of the #MWC datagram.) Revision I+ includes the 'cmnPart' field of a
                                # partitioned datagram in all partitions. (This change is reflected in version 2+
                                # of the #MWC datagram.)
                                if header['dgmVersion'] == 2 or partition['dgmNum'] == 1:
                                    cmnPart = k.read_EMdgmMbody(bytes_io, header['dgmType'], header['dgmVersion'])
                                else:
                                    cmnPart = None

                                # For debugging:
                                # print("Inserting new datagram {}, {} into index {}. Part {} of {}."
                                #       .format(header['dgmType'], header['dgTime'], next_index,
                                #               partition['dgmNum'], partition['numOfDgms']))

                                self.buffer['dgmType'][next_index] = header['dgmType']
                                self.buffer['dgmVersion'][next_index] = header['dgmVersion']
                                self.buffer['dgTime'][next_index] = header['dgTime']
                                if cmnPart:
                                    self.buffer['pingCnt'][next_index] = cmnPart['pingCnt']
                                else:
                                    self.buffer['pingCnt'][next_index] = None
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
                # print("Local process flag is 2. Flushing buffer.")  # For debugging
                # Flush completed datagrams in buffer into queue_datagram
                self.flush_buffer()
                # Poison pill to signal next process
                self.queue_datagram.put(None)
                break  # Exit loop

            elif local_process_flag_value == 3:  # Stop pressed
                # print("Local process flag is 3. Discarding buffer contents.")  # For debugging
                # Discard all datagrams in buffer
                self.buffer = self._init_buffer()
                # Poison pill to signal next process
                self.queue_datagram.put(None)
                break  # Exit loop

            else:
                logger.error("Error in KongsbergDGCaptureFromSonar. Invalid process_flag value: {}."
                             .format(local_process_flag_value))
                break  # Exit loop

        # print("Closing socket.")  # For debugging
        self.sock_in.close()

    def flush_buffer(self):
        """
        Recursive method to flush all complete records from buffer after pause or stop command or socket timeout.
        """
        # print("Flushing buffer. Timestamps: ", self.buffer['dgTime'])  # For debugging
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
        """
        When an incomplete #MWC record cannot be completed and is discarded,
        an 'empty' datagram is reconstructed via this method.
        :param dgmType: Byte string indicating datagram type: b'#MRZ' or b'#MWC'
        :param dgmVersion: Version of datagram
        :param data: An incomplete sorted list containing raw, partitioned Kongsberg datagrams from a single ping.
        Example: [<ping 1 - datagram 1 of 3>, _X_, <ping 1 - datagram 3 of 3>].
        :return: A single reconstructed 'empty' #MWC record and the number of bytes contained in it.
        """
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
        When all #MWC partitions are received, it is reconstructed via this method.
        :param dgmType: Byte string indicating datagram type: b'#MRZ' or b'#MWC'
        :param dgmVersion: Version of datagram
        :param data: A sorted list containing all raw, partitioned Kongsberg datagrams from a single ping.
        Example: [<ping 1 - datagram 1 of 3>, <ping 1 - datagram 2 of 3>, <ping 1 - datagram 3 of 3>].
        :return: A single reconstructed (non-partitioned) #MWC record, and the number of bytes contained in it.
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
        """
        Appends data fields of buffered data such that they are a contiguous
        byte string rather than discrete entries in a list.
        :param buffer: Data buffered as discrete entries in a list.
        :return: A contiguous bytes string of data representing a single #MWC record.
        """
        flat_buffer = b''
        for item in buffer:
            flat_buffer += item
        return flat_buffer

    def run(self):
        """
        Runs process. Process queues data in multiprocessing.Queue if provided;
        otherwise, writes raw binary data to file.
        """
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
