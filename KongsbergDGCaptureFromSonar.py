# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# May 2021

# Description: Capture UDP datagrams from Kongsberg sonar;
# insert datagram into shared queue (multiprocessing.Queue), or write to file.

import argparse
import io
import KMALL
import logging
import socket
import struct
import sys

logger = logging.getLogger(__name__)

class KongsbergDGCaptureFromSonar:
    def __init__(self, rx_ip, rx_port, connection="Multicast", queue=None, out_file=None):
        self.rx_ip = rx_ip
        self.rx_port = rx_port
        self.connection = connection

        # When run as main, out_file is required;
        # when run with multiprocessing, queue is required (multiprocessing.Queue)
        self.queue = queue
        self.out_file = out_file

        self.SOCKET_TIMEOUT = 60  # Seconds
        self.sock_in = self.__init_socket()

        self.MAX_DATAGRAM_SIZE = 2 ** 16
        self.REQUIRED_DATAGRAMS = [b'#MRZ', b'#MWC', b'#SKM']

        # Note: Structs current as of *.kmall format REV. I.
        self.HEADER_STRUCT_FORMAT = '1I4s2B1H2I'
        self.M_PARTITION_STRUCT_FORMAT = '2H'
        self.M_BODY_STRUCT_FORMAT = '2H8B'


    def __init_socket(self):
        if self.connection == "TCP":
            # temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.warning("Only UDP and Multicast connections supported at this time.")
            sys.exit(1)
        elif self.connection == "UDP":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.bind((self.rx_ip, self.rx_port))
        elif self.connection == "Multicast":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            # Allow reuse of addresses
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

        # TODO: We cannot guarantee that split 'M' datagrams will be received in order or that all partitions
        #  will be received at all (dropped packets). Track ping number, number of datagrams, and datagram number.

        # From Val's code:
        k = KMALL.kmall(filename=None)
        data_buffer = None

        while True:
            try:
                data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)
                # print(data)  # For debugging
            except socket.timeout:
                logger.exception("Socket timeout exception.")
                self.sock_in.close()
                break

            # TODO: Fix this to account for non-consecutive datagrams!
            bytes_io = io.BytesIO(data)
            num_bytes_dg = int.from_bytes(bytes_io.read(4), byteorder="little")
            dg_type = bytes_io.read(4)

            if dg_type in self.REQUIRED_DATAGRAMS:
                if dg_type == b'#MRZ' or dg_type == b'MWC':
                    # Skip over remainder of header
                    bytes_io.seek(struct.Struct(self.HEADER_STRUCT_FORMAT).size, 0)
                    # Read partition; dg[0] is numOfDgms; dg[1] is dgmNum
                    dg = struct.unpack(self.M_PARTITION_STRUCT_FORMAT,
                                       bytes_io.read(struct.Struct(self.M_PARTITION_STRUCT_FORMAT).size))
                    if dg[0] == 1:  # There is only one part to the datagram; no need to reconstruct
                        self.queue.put(data)
                    else:  # There is more than one part to the datagram
                        num_of_dgms = dg[0]
                        dgm_num = dg[1]
                        # Read cmnPart; dg[1] is pingCnt
                        dg = struct.unpack(self.M_BODY_STRUCT_FORMAT,
                                           bytes_io.read(struct.Struct(self.M_BODY_STRUCT_FORMAT).size))
                        pnt_cnt = dg[1]

                        # TODO: Complete reconstruction of datagram. This will depend on how Kongsberg splits datagrams.
                        #  Reassemble full datagram. Is it better to do this on this (put) end or the receiving (get) end?

                else:  # Datagrams are not split and do not need reconstruction
                    self.queue.put(data)

                # FOR TESTING:
                if num_bytes_dg < self.MAX_DATAGRAM_SIZE:
                    print("datagram small enough to queue: ", dg_type, num_bytes_dg)
                else:
                    print("datagram must be split: ", dg_type)

                    pass

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

    def run(self):
        if self.queue is not None:
            print(self.queue)
            print(type(self.queue))
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
    parser.add_argument("--connection", default="Multicast", help="Connection type: TCP or UDP.",
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
