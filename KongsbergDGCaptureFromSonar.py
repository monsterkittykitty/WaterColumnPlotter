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
from multiprocessing import Process, Value
import socket
import struct
import sys

logger = logging.getLogger(__name__)

class KongsbergDGCaptureFromSonar(Process):
    #class KongsbergDGCaptureFromSonar:
    def __init__(self, rx_ip, rx_port, connection="Multicast", queue_datagram=None, process_flag=None, out_file=None):
        super().__init__()

        print("New instance of KongsbergDGCapture.")

        self.rx_ip = rx_ip
        self.rx_port = rx_port
        self.connection = connection

        # When run as main, out_file is required;
        # when run with multiprocessing, queue is required (multiprocessing.Queue)
        self.queue_datagram = queue_datagram
        self.out_file = out_file

        # Boolean shared across processes (multiprocessing.Value)
        if process_flag:
            self.process_flag = process_flag
        else:
            self.process_flag = mp.Value(ctypes.c_bool, True)

        self.SOCKET_TIMEOUT = 5  # Seconds
        self.MAX_DATAGRAM_SIZE = 2 ** 16
        self.sock_in = self.__init_socket()
        #self.print_settings()

        self.MAX_NUM_PINGS_TO_BUFFER = 5

        self.REQUIRED_DATAGRAMS = [b'#MRZ', b'#MWC', b'#SKM', b'#SPO']

        self.buffer = {'dgmType': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                       'dgmVersion': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                       'dgTime': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                       'pingCnt': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                       'numOfDgms': [None] * self.MAX_NUM_PINGS_TO_BUFFER,
                       'dgmsRxed': [0] * self.MAX_NUM_PINGS_TO_BUFFER,
                       'complete': [False] * self.MAX_NUM_PINGS_TO_BUFFER,
                       'data': [None] * self.MAX_NUM_PINGS_TO_BUFFER}

        # TODO: FOR TESTING
        self.dgms_rxed = 0

    def __init_socket(self):
        if self.connection == "TCP":
            # temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.warning("Only UDP and Multicast connections supported at this time.")
            sys.exit(1)

        elif self.connection == "UDP":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # TODO: Change buffer size if packets are being lost:
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.MAX_DATAGRAM_SIZE * 2 * 2)
            temp_sock.bind((self.rx_ip, self.rx_port))

        elif self.connection == "Multicast":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # TODO: Change buffer size if packets are being lost:
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.MAX_DATAGRAM_SIZE * 2 * 2)
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

    def print_settings(self):
        print("Receive (IP:Port, Connection): ", self.rx_ip, ":", self.rx_port, ",", self.connection)

    def print_packet_details(self, data):
        pass

    def receive_dg_and_write_raw(self):
        """
        Receives data at specified socket; writes binary data to specified file.
        *** Note, this does NOT reconstruct partitioned datagrams. ***
        """
        file_io = open(self.out_file, 'wb')

        #while True:
        while self.process_flag.value:
            try:
                data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)
                #print(data)  # For debugging
                file_io.write(data)
            except socket.timeout:
                logger.exception("Socket timeout exception.")
                self.sock_in.close()
                file_io.close()
                break

    # TODO: Delete this
    def printCapture(self):
        print("CAPTURE")

    def receive_dg_and_queue(self):
        """
        Receives data at specified socket; places data in specified queue (multiprocessing.Queue).
        """

        print("DGCapture: receive_dg_and_queue")  # For debugging

        first_tx_time = None  # For testing
        dg_counter = 0  # For testing
        mwc_counter = 0  # For testing

        timestamp_index = 0  # Index of oldest timestamp in buffer
        next_index = 0  # Index of next position in buffer to be filled

        # while True:
        while self.process_flag.value:
            # print("Capture, self.receive_data_and_queue: ", self.process_flag.value)
            try:
                data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)

                # For testing:
                if dg_counter == 0:
                    first_tx_time = datetime.datetime.now()
                dg_counter += 1

            except BlockingIOError:
                continue
            except socket.timeout:
                logger.exception("Socket timeout exception.")
                self.sock_in.close()
                break

            # TODO: FOR TESTING:
            self.dgms_rxed += 1

            bytes_io = io.BytesIO(data)

            header = k.read_EMdgmHeader(bytes_io)

            #print("header[numBytesDgm]: ", header['numBytesDgm'], type(header['dgmType']))

            if header['dgmType'] in self.REQUIRED_DATAGRAMS:
                if header['dgmType'] == b'#MRZ' or header['dgmType'] == b'#MWC':  # Datagrams may be partitioned

                    # For testing:
                    # if header['dgmType'] == b'#MWC':
                    #     mwc_counter += 1
                    #     print("dgm_timestamp: ", header['dgdatetime'], "mwc_counter: ", mwc_counter)

                    partition = k.read_EMdgmMpartition(bytes_io, header['dgmType'], header['dgmVersion'])

                    if partition['numOfDgms'] == 1:  # Only one datagram; no need to reconstruct
                        self.queue_datagram.put(data)

                    else:  # Greater than one datagram; needs to be reconstructed
                        # Check for timestamp in buffer:
                        if header['dgTime'] in self.buffer['dgTime']:  # Timestamp in buffer

                            # For testing:
                            # print("Timestamp in buffer: {}, {}".format(header['dgmType'], header['dgTime']))

                            index = self.buffer['dgTime'].index(header['dgTime'])

                            # Append new data to existing data in buffer:
                            self.buffer['dgmsRxed'][index] += 1
                            self.buffer['data'][index][partition['dgmNum'] - 1] = data

                            # For debugging:
                            # print("Inserting existing datagram {}, {} into index {}. Part {} of {}."
                            #       .format(header['dgmType'], header['dgTime'], index,
                            #               partition['dgmNum'], partition['numOfDgms']))
                            # print("Existing: self.buffer['dgmsRxed'][index]:", self.buffer['dgmsRxed'][index])
                            # print("Existing: self.buffer['numOfDgms'][index]:", self.buffer['numOfDgms'][index])

                            # Check if all data received:
                            if self.buffer['dgmsRxed'][index] == self.buffer['numOfDgms'][index]:

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

                                        logger.warning("Data block incomplete. Discarding {}, {}. (Ping {}, {} of {} datagrams.) "
                                                       "\nConsidering increasing size of buffer. (Current buffer size: {}.)"
                                                       .format(self.buffer['dgmType'][next_index],
                                                               self.buffer['dgTime'][next_index],
                                                               self.buffer['pingCnt'][next_index],
                                                               self.buffer['dgmsRxed'][next_index],
                                                               self.buffer['numOfDgms'][next_index],
                                                               self.MAX_NUM_PINGS_TO_BUFFER))

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



            if dg_counter == 8709:  # For testing
                last_tx_time = datetime.datetime.now()
                print("DGCAPTURE, Received: ", dg_counter)
                print("DGCAPTURE, Received MWCs: ", mwc_counter)
                print("DGCAPTURE, First transmit: {}; Final transmit: {}; Total time: {}"
                      .format(first_tx_time, last_tx_time, (last_tx_time - first_tx_time).total_seconds()))

        print("BOOLEAN STOPPED.")
        self.sock_in.close()

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

            # When advancing timestamp to existing entry, check if complete:
            if self.buffer['complete'][timestamp_index]:  # Data at timestamp_index is complete!
                data_reconstruct, data_size = self.reconstruct_data(
                    self.buffer['dgmType'][timestamp_index],
                    self.buffer['dgmVersion'][timestamp_index],
                    self.buffer['data'][timestamp_index])

                # Add reconstructed data to queue
                self.queue_datagram.put(data_reconstruct)
                # Clear entry
                self.buffer['dgTime'][timestamp_index] = None
                self.buffer['complete'][timestamp_index] = False

                # For debugging:
                print("Complete datablock: {}, {}, {} bytes"
                      .format(self.buffer['dgmType'][timestamp_index],
                              self.buffer['dgTime'][timestamp_index], data_size))

                # Advance timestamp again
                self.advance_timestamp_index()

        return timestamp_index

    def reconstruct_data(self, dgmType, dgmVersion, data):
        """

        :param data: A sorted list containing all raw, partitioned Kongsberg datagrams from a single ping.
        Example: [<ping 1 - datagram 1 of 3>, <ping 1 - datagram 2 of 3>, <ping 1 - datagram 3 of 3>].
        :return: A single reconstructed (non-partitioned) datagram.
        """
        temp_buffer = []
        numBytesDgm = 0

        header_struct_format = k.read_EMdgmHeader(data, return_format=True)
        partition_struct_format = k.read_EMdgmMpartition(data, dgmType, dgmVersion, return_format=True)
        cmnPart_struct_format = k.read_EMdgmMbody(data, dgmType, dgmVersion, return_format=True)

        # Length to strip for Kongsberg *.kmall datagram format revisions A - H.
        length_to_strip = struct.calcsize(header_struct_format) + struct.calcsize(partition_struct_format)

        # Determine Kongsberg *.kmall datagram format revision version.
        # Revision A - H contain cmnPart only in partition 1; revisions I+ contain cmnPart in all partitions.
        # Revision I updated #MRZ datagram to version 3 and #MWC datagram to version 2.
        if (dgmType == b'#MRZ' and dgmVersion >= 3) or (dgmType == b'#MWC' and dgmVersion >= 2):
            # Length to strip for Kongsberg *.kmall datagram format revisions I+.
            length_to_strip += cmnPart_struct_format

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
        print("Running KongsbergDGCapture process.")
        if self.queue_datagram:
            # print(self.queue_datagram)
            # print(type(self.queue_datagram))
            self.receive_dg_and_queue()

        else:
            self.receive_dg_and_write_raw()
        #self.join()


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
