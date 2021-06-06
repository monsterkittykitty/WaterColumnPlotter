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
import KMALL
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

        self.REQUIRED_DATAGRAMS = [b'#MRZ', b'#MWC', b'#SKM', b'#SPO']

        # Note: Structs current as of *.kmall format REV. I.
        self.HEADER_STRUCT_FORMAT = '1I4s2B1H2I'
        self.M_PARTITION_STRUCT_FORMAT = '2H'
        self.M_BODY_STRUCT_FORMAT = '2H8B'

        # TODO: FOR TESTING
        self.dgms_rxed = 0

    def __init_socket(self):
        if self.connection == "TCP":
            # temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.warning("Only UDP and Multicast connections supported at this time.")
            sys.exit(1)

        elif self.connection == "UDP":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # TODO: Change buffer size of packets are being lost:
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.MAX_DATAGRAM_SIZE * 2 * 2)
            temp_sock.bind((self.rx_ip, self.rx_port))

        elif self.connection == "Multicast":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # TODO: Change buffer size of packets are being lost:
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
        print("Recieve (IP:Port, Connection): ", self.rx_ip, ":", self.rx_port, ",", self.connection)

    def print_packet_details(self, data):
        pass

    def receive_dg_and_write(self):
        """
        Receives data at specified socket; writes binary data to specified file.
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

        data_type = ""
        data_timestamp = 0.0
        data_buffer = []
        data_size = 0
        data_count = 0

        while True:
            try:
                data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)
                # print(data)  # For debugging
            except BlockingIOError:
                continue
            except socket.timeout:
                logger.exception("Socket timeout exception.")
                self.sock_in.close()
                break

            # TODO: FOR TESTING:
            self.dgms_rxed += 1

            bytes_io = io.BytesIO(data)

            header = struct.unpack(self.HEADER_STRUCT_FORMAT,
                                       bytes_io.read(struct.Struct(self.HEADER_STRUCT_FORMAT).size))
            num_bytes_dgm = header[0]
            dgm_type = header[1]
            dgm_version = header[2]
            dgm_timestamp = datetime.datetime.utcfromtimestamp((header[5] + header[6] / 1.0E9))

            if dgm_type in self.REQUIRED_DATAGRAMS:
                # print(dgm_type)  # For debugging
                if dgm_type == b'#MRZ' or dgm_type == b'#MWC':  # Datagrams may be partitioned
                    # Skip over remainder of header
                    # bytes_io.seek(struct.Struct(self.HEADER_STRUCT_FORMAT).size, 0)
                    # Read partition; dg[0] is numOfDgms; dg[1] is dgmNum
                    partition = struct.unpack(self.M_PARTITION_STRUCT_FORMAT,
                                       bytes_io.read(struct.Struct(self.M_PARTITION_STRUCT_FORMAT).size))
                    num_of_dgms = partition[0]
                    if num_of_dgms == 1:  # There is only one part to the datagram; no need to reconstruct
                        self.queue_tx_data.put(data)
                    else:  # There is more than one part to the datagram; datagram needs to be reconstructed
                        if data_timestamp != dgm_timestamp:
                            if data_count != 0:  # Previous data block is incomplete
                                # TODO: If many of these are reported and datagrams arrive out-of-order,
                                #  consider adding multiple buffers.
                                logger.warning("Data block incomplete: {}, {}".format(data_type, data_timestamp))
                                data_count = 0
                                data_size = 0
                            # Reset data_type, data_timestamp, and data_buffer
                            data_type = dgm_type
                            data_timestamp = dgm_timestamp
                            data_buffer = [None] * num_of_dgms

                        dgm_num = partition[1]

                        if dgm_num == 1:  # dgmNum (partition) 1 can be added in its entirety to data_buffer
                            # data_buffer[dgm_num - 1] = data[:-4]
                            data_buffer[dgm_num - 1] = bytearray(data[:-4])
                            data_count += 1
                            data_size += len(data[:-4])
                        else:  # dgmNum (partition) 2+ must have leading fields removed prior to addition to data_buffer
                            rev_i_plus = False
                            length_to_strip = struct.calcsize(self.HEADER_STRUCT_FORMAT) + \
                                              struct.calcsize(self.M_PARTITION_STRUCT_FORMAT)

                            if dgm_type == b'#MRZ':
                                if dgm_version in [3]:
                                    rev_i_plus = True
                                    length_to_strip += struct.calcsize(self.M_BODY_STRUCT_FORMAT)
                            elif dgm_type == b'#MWC':
                                if dgm_version in [2]:
                                    rev_i_plus = True
                                    length_to_strip += struct.calcsize(self.M_BODY_STRUCT_FORMAT)

                            # Note: Final 4 bytes of partitioned datagram is a repeated size field. Remove this.
                            # data_buffer[dgm_num - 1] = data[length_to_strip:-4]
                            data_buffer[dgm_num - 1] = bytearray(data[length_to_strip:-4])
                            data_count += 1
                            data_size += len(data[length_to_strip:-4])

                            if data_count == num_of_dgms:

                                # Add 4 to data_size to account for 4-byte 'length' field to be appended to end of dgm
                                data_size += 4

                                # Adjust header values:
                                new_num_bytes_dgm = struct.pack("I", data_size)  # Returns type <class 'bytes'>
                                data_buffer[0][:struct.calcsize("I")] = new_num_bytes_dgm

                                # Adjust partition values:
                                new_partition = struct.pack("2H", 1, 1)  # Returns type <class 'bytes'>
                                data_buffer[0][struct.calcsize(self.HEADER_STRUCT_FORMAT):
                                               (struct.calcsize(self.HEADER_STRUCT_FORMAT) +
                                               struct.calcsize(self.M_PARTITION_STRUCT_FORMAT))] = new_partition

                                # Flatten data_buffer
                                flat_data_buffer = self.flatten_data_buffer(data_buffer)
                                # Add final 4-byte size field:
                                flat_data_buffer += struct.pack("I", data_size)

                                self.queue_tx_data.put(flat_data_buffer)

                                print("Complete datablock: {}; {}; {}".format(data_type, data_size, data_timestamp))

                                # Reset data_count, data_size
                                data_count = 0
                                data_size = 0

                else:  # Datagrams are not partitioned
                    self.queue_tx_data.put(data)

            #print("self.dgms_rxed: ", self.dgms_rxed)




                # FOR TESTING:
                # if num_bytes_dgm < self.MAX_DATAGRAM_SIZE:
                #     print("datagram small enough to queue: ", dgm_type, num_bytes_dgm)
                # else:
                #     print("datagram must be split: ", dgm_type)


            # if data_buffer is None:
            #     k.FID = io.BytesIO(data)
            # else:
            #     k.FID = io.BytesIO(data_buffer + data)
            #
            # k.FID.seek(0, 0)
            # # dg_header = k.read_EMdgmHeader()
            # # print(dg_header)
            # num_bytes_dg = k.FID.read(4)
            # dg_type = k.FID.read(4)
            # print(dg_type)
            # print(type(dg_type))
            # if dg_type in self.REQUIRED_DATAGRAMS:
            #     print('req dg: ', dg_type)
            #     exit()

    def flatten_data_buffer(self, data_buffer):
        flat_data_buffer = b''
        # flat_data_buffer = bytearray()
        for item in data_buffer:
            flat_data_buffer += item
        return flat_data_buffer

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
