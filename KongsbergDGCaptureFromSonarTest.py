# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# May 2021

# Description: Quick test of KongsbergDGCaptureFromSonar.

import argparse
import logging
import socket
import struct

logger = logging.getLogger(__name__)

class KongsbergDGCaptureFromSonarTest:
    def __init__(self, tx_ip, tx_port, connection):
        self.tx_ip = tx_ip
        self.tx_port = tx_port
        self.connection = connection

        self.sock_out = self.__init_socket()

    def __init_socket(self):
        if self.connection == "TCP":
            logger.warning("TCP connections unsupported at this time.")
            pass
        elif self.connection == "UDP":
            logger.warning("UDP connections unsupported at this time.")
            pass
        elif self.connection == "Multicast":
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            temp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        else:
            raise RuntimeError("Connection type must be 'TCP', 'UDP', or 'Multicast'.")

        return temp_sock

    def run(self):

        while True:
            try:
                self.sock_out.sendto(b'Hello!', (self.tx_ip, self.tx_port))
            except socket.timeout:
                logger.exception("Socket timeout exception.")
                self.sock_out.close()
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("rx_ip", help="IP address to receive Kongsberg datagrams.")
    parser.add_argument("rx_port", help="Port to receive Kongsberg datagrams.", type=int)
    parser.add_argument("--connection", default="Multicast", help="Connection type: TCP or UDP.", choices={"TCP", "UDP", "Multicast"})

    args = parser.parse_args()

    if args.connection == "TCP" or args.connection == "UDP":
        logger.warning("Only Multicast connections supported at this time.")
        args.connection = "Multicast"

    kongsberg_dg_capture_from_sonar_test = KongsbergDGCaptureFromSonarTest(args.rx_ip, args.rx_port, args.connection)
    kongsberg_dg_capture_from_sonar_test.run()
