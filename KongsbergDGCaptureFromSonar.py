# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# May 2021

# Description: Capture UDP datagrams from Kongsberg sonar;
# insert datagram into shared queue (multiprocessing.Queue), or write to file.

# Note: Can configure in SIS which datagrams are sent to this program along with IP and port.
# 13 May 2021: When operating on RVGS, program run with following arguments:
# "0.0.0.0" "8080" "testlog.txt" --connection "UDP"

import argparse
import datetime
import io
from KmallReaderForMDatagrams import KmallReaderForMDatagrams as k
import logging
import socket
import struct
import sys

logger = logging.getLogger(__name__)

class KongsbergDGCaptureFromSonar:
    def __init__(self, rx_ip, rx_port, connection="Multicast", queue_data=None, out_file=None):

        print("dgcapture: init")
        self.rx_ip = rx_ip
        self.rx_port = rx_port
        self.connection = connection

        # When run as main, out_file is required;
        # when run with multiprocessing, queue is required (multiprocessing.Queue)
        self.queue_tx_data = queue_data
        self.out_file = out_file

        self.SOCKET_TIMEOUT = 5  # Seconds
        self.MAX_DATAGRAM_SIZE = 2 ** 16
        self.sock_in = self.__init_socket()
        self.print_settings()

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

    def receive_dg_and_write(self):
        """
        Receives data at specified socket; writes binary data to specified file.
        *** Note, this does NOT reconstruct partitioned datagrams. ***
        """
        file_io = open(self.out_file, 'wb')

        while True:
            try:
                data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)
                #print(data)  # For debugging
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

        first_tx_time = None  # For testing
        dg_counter = 0  # For testing
        mwc_counter = 0  # For testing

        timestamp_index = 0  # Index of oldest timestamp in buffer
        next_index = 0  # Index of next position in buffer to be filled

        data_size = 0
        data_count = 0

        while True:
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

            # print("header[numBytesDgm]: ", header['numBytesDgm'], type(header['dgmType']))

            if header['dgmType'] in self.REQUIRED_DATAGRAMS:
                if header['dgmType'] == b'#MRZ' or header['dgmType'] == b'#MWC':  # Datagrams may be partitioned

                    # For testing:
                    if header['dgmType'] == b'#MWC':
                        mwc_counter += 1
                        print("dgm_timestamp: ", header['dgdatetime'], "mwc_counter: ", mwc_counter)

                    partition = k.read_EMdgmMpartition(bytes_io, header['dgmType'], header['dgmVersion'])

                    if partition['numOfDgms'] == 1:  # Only one datagram; no need to reconstruct
                        self.queue_tx_data.put(data)

                    else:  # Greater than one datagram; needs to be reconstructed
                        # Check for timestamp in buffer:
                        if header['dgTime'] in self.buffer['dgTime']:  # Timestamp in buffer

                            # For testing:
                            print("Timestamp in buffer: {}, {}".format(header['dgmType'], header['dgTime']))

                            index = self.buffer['dgTime'].index(header['dgTime'])

                            # Append new data to existing data in buffer:
                            self.buffer['dgmsRxed'][index] += 1
                            self.buffer['data'][index][partition['dgmNum'] - 1] = data

                            # Check if all data received:
                            if self.buffer['dgmsRxed'][index] == self.buffer['numOfDgms'][index]:

                                # For testing:
                                print("All data received: {}, {}, ping: ".format(self.buffer['dgmType'],
                                                                                 self.buffer['dgTime'],
                                                                                 self.buffer['pingCnt']))
                                self.buffer['complete'][index] = True

                                # Check if current index equals (earliest) timestamp_index:
                                if index == timestamp_index:

                                    # For testing:
                                    print("Reconstructing datagram.")

                                    # Earliest timestamp index is complete! Reconstruct data and place in queue
                                    data_reconstruct, data_size = self.reconstruct_data(
                                        self.buffer['dgmType'][timestamp_index],
                                        self.buffer['dgmVersion'][timestamp_index],
                                        self.buffer['data'][timestamp_index])

                                    self.queue_tx_data.put(data_reconstruct)

                                    # For debugging:
                                    print("Complete datablock: {}, {}, {} bytes"
                                          .format(self.buffer['dgmType'][timestamp_index],
                                                  self.buffer['dgTime'][timestamp_index], data_size))

                                    # Clear entry
                                    # TODO: Practically, do I need to clear any more than this?
                                    self.buffer['dgTime'][timestamp_index] = None
                                    self.buffer['complete'][timestamp_index] = False  # Probably not necessary?

                                    # Advance timestamp_index to oldest timestamp in buffer
                                    self.advance_timestamp_index()

                                else:  # index != timestamp_index
                                    print("Not reconstructing datagram. Index: {}, Timestamp index: {}"
                                          .format(index, timestamp_index))

                        else:  # Timestamp not in buffer

                            # For testing:
                            print("Timestamp not in buffer: {}, {}".format(header['dgmType'], header['dgTime']))

                            # Check whether next_index currently points to incomplete data.
                            # If so, log and discard / overwrite data.
                            if self.buffer['dgTime'][next_index] is not None:  # Index contains incomplete data
                                logger.warning("Data block incomplete. Discarding {}, {} (ping {}, {} of {} datagrams)."
                                               .format(self.buffer['dgmType'][next_index],
                                                       self.buffer['dgTime'][next_index],
                                                       self.buffer['pingCnt'][next_index],
                                                       self.buffer['dgmsRxed'][next_index],
                                                       self.buffer['numOfDgms'][next_index]))

                                # For testing:
                                print("Next index: {}. Timestamp index and timestamp: {}, {}. All timestamps: {}"
                                      .format(next_index, timestamp_index,
                                              self.buffer['dgTime'][timestamp_index], self.buffer['dgTime']))

                                # Error checking: If we are overwriting data, next_index must equal timestamp_index or
                                # something is wrong! This should never print. If it does, there's an error in code.
                                if next_index != timestamp_index:
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
                self.queue_tx_data.put(data_reconstruct)
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
                print("len(data[i][:-4]): ", len(data[i][:-4]))
            else:  # Final dgm(s) must have leading fields and last 4 bytes removed
                temp_buffer.append(bytearray(data[i][length_to_strip:-4]))
                numBytesDgm += len(data[i][length_to_strip:-4])
                print("len(data[i][length_to_strip:-4]): ", len(data[i][length_to_strip:-4]))

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

    # def receive_dg_and_queue(self):
    #     """
    #     Receives data at specified socket; places data in specified queue (multiprocessing.Queue).
    #
    #     OLD IMPLEMENTATION. NO INCOMING DATA BUFFER.
    #     DISCARDS ALL DATA THAT IS IMCOMPLETE AND OLDER THAN THE CURRENT PING!
    #     """
    #
    #     print("DGCapture: receive_dg_and_queue")  # For debugging
    #     first_tx_time = None  # For testing
    #     dg_counter = 0  # For testing
    #     mwc_counter = 0  # For testing
    #
    #     data_type = ""
    #     data_timestamp = 0.0
    #     data_buffer = []
    #     data_size = 0
    #     data_count = 0
    #
    #     while True:
    #         try:
    #             data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)
    #             # print(data)  # For debugging
    #
    #             if dg_counter == 0:  # For testing
    #                 first_tx_time = datetime.datetime.now()
    #             dg_counter += 1
    #
    #         except BlockingIOError:
    #             continue
    #         except socket.timeout:
    #             logger.exception("Socket timeout exception.")
    #             self.sock_in.close()
    #             break
    #
    #         # TODO: FOR TESTING:
    #         self.dgms_rxed += 1
    #
    #         bytes_io = io.BytesIO(data)
    #
    #         header = struct.unpack(self.HEADER_STRUCT_FORMAT,
    #                                bytes_io.read(struct.Struct(self.HEADER_STRUCT_FORMAT).size))
    #
    #         num_bytes_dgm = header[0]
    #         dgm_type = header[1]
    #         dgm_version = header[2]
    #         dgm_timestamp = datetime.datetime.utcfromtimestamp((header[5] + header[6] / 1.0E9))
    #
    #         if dgm_type in self.REQUIRED_DATAGRAMS:
    #             # print(dgm_type)  # For debugging
    #             if dgm_type == b'#MRZ' or dgm_type == b'#MWC':  # Datagrams may be partitioned
    #
    #                 if dgm_type == b'#MWC':  # For testing
    #                     mwc_counter += 1
    #                     print("dgm_timestamp: ", dgm_timestamp, "mwc_counter: ", mwc_counter)
    #
    #                 # Skip over remainder of header
    #                 # bytes_io.seek(struct.Struct(self.HEADER_STRUCT_FORMAT).size, 0)
    #                 # Read partition; dg[0] is numOfDgms; dg[1] is dgmNum
    #                 partition = struct.unpack(self.M_PARTITION_STRUCT_FORMAT,
    #                                           bytes_io.read(struct.Struct(self.M_PARTITION_STRUCT_FORMAT).size))
    #                 num_of_dgms = partition[0]
    #                 if num_of_dgms == 1:  # There is only one part to the datagram; no need to reconstruct
    #                     self.queue_tx_data.put(data)
    #                 else:  # There is more than one part to the datagram; datagram needs to be reconstructed
    #
    #                     if data_timestamp != dgm_timestamp:
    #                         if data_count != 0:  # Previous data block is incomplete
    #                             # TODO: If many of these are reported and datagrams arrive out-of-order,
    #                             #  consider adding multiple buffers.
    #                             print("data_timestamp: ", data_timestamp, ", dgm_timestamp: ", dgm_timestamp)
    #                             logger.warning("Data block incomplete: {}, {}".format(data_type, data_timestamp))
    #                             data_count = 0
    #                             data_size = 0
    #                         # Reset data_type, data_timestamp, and data_buffer
    #                         data_type = dgm_type
    #                         data_timestamp = dgm_timestamp
    #                         data_buffer = [None] * num_of_dgms
    #
    #                     dgm_num = partition[1]
    #
    #                     if dgm_num == 1:  # dgmNum (partition) 1 can be added in its entirety to data_buffer
    #                         # data_buffer[dgm_num - 1] = data[:-4]
    #                         data_buffer[dgm_num - 1] = bytearray(data[:-4])
    #                         data_count += 1
    #                         data_size += len(data[:-4])
    #                     else:  # dgmNum (partition) 2+ must have leading fields removed prior to addition to data_buffer
    #                         rev_i_plus = False
    #                         length_to_strip = struct.calcsize(self.HEADER_STRUCT_FORMAT) + \
    #                                           struct.calcsize(self.M_PARTITION_STRUCT_FORMAT)
    #
    #                         if dgm_type == b'#MRZ':
    #                             if dgm_version in [3]:
    #                                 length_to_strip += struct.calcsize(self.M_BODY_STRUCT_FORMAT)
    #                         elif dgm_type == b'#MWC':
    #                             if dgm_version in [2]:
    #                                 length_to_strip += struct.calcsize(self.M_BODY_STRUCT_FORMAT)
    #
    #                         # Note: Final 4 bytes of partitioned datagram is a repeated size field. Remove this.
    #                         data_buffer[dgm_num - 1] = bytearray(data[length_to_strip:-4])
    #                         data_count += 1
    #                         data_size += len(data[length_to_strip:-4])
    #
    #                         if data_count == num_of_dgms:
    #                             print("full reconstruction")
    #                             # Add 4 to data_size to account for 4-byte 'length' field to be appended to end of dgm
    #                             data_size += 4
    #
    #                             # Adjust header values:
    #                             new_num_bytes_dgm = struct.pack("I", data_size)  # Returns type <class 'bytes'>
    #                             data_buffer[0][:struct.calcsize("I")] = new_num_bytes_dgm
    #
    #                             # Adjust partition values:
    #                             new_partition = struct.pack("2H", 1, 1)  # Returns type <class 'bytes'>
    #                             data_buffer[0][struct.calcsize(self.HEADER_STRUCT_FORMAT):
    #                                            (struct.calcsize(self.HEADER_STRUCT_FORMAT) +
    #                                             struct.calcsize(self.M_PARTITION_STRUCT_FORMAT))] = new_partition
    #
    #                             # Flatten data_buffer
    #                             flat_data_buffer = self.flatten_data_buffer(data_buffer)
    #                             # Add final 4-byte size field:
    #                             flat_data_buffer += struct.pack("I", data_size)
    #
    #                             self.queue_tx_data.put(flat_data_buffer)
    #
    #                             # print("Complete datablock: {}; {}; {}".format(data_type, data_size, data_timestamp))
    #
    #                             # Reset data_count, data_size
    #                             data_count = 0
    #                             data_size = 0
    #
    #             else:  # Datagrams are not partitioned
    #                 self.queue_tx_data.put(data)
    #
    #         if dg_counter == 8709:  # For testing
    #             last_tx_time = datetime.datetime.now()
    #             print("DGCAPTURE, Received: ", dg_counter)
    #             print("DGCAPTURE, Received MWCs: ", mwc_counter)
    #             print("DGCAPTURE, First transmit: {}; Final transmit: {}; Total time: {}"
    #                   .format(first_tx_time, last_tx_time, (last_tx_time - first_tx_time).total_seconds()))

    def flatten_buffer(self, buffer):
        flat_buffer = b''
        for item in buffer:
            flat_buffer += item
        return flat_buffer

    def run(self):
        if self.queue_tx_data is not None:
            print(self.queue_tx_data)
            print(type(self.queue_tx_data))
            self.receive_dg_and_queue()

        else:
            self.receive_dg_and_write()


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
