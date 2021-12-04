import argparse
import ctypes
import datetime
import io
from KmallReaderForMDatagrams import KmallReaderForMDatagrams as k
import logging
#from multiprocessing import Process, Value
import multiprocessing as mp
mp.allow_connection_pickling()
import socket
import struct
import sys

logger = logging.getLogger(__name__)

class Snippet:
    #class KongsbergDGCaptureFromSonar:
    def __init__(self, rx_ip, rx_port):
        #super().__init__()

        #self.name = "Snippet"

        print("New instance of ForTestingSnippets4.")

        self.rx_ip = rx_ip
        self.rx_port = rx_port

        self.SOCKET_TIMEOUT = 5  # Seconds
        self.MAX_DATAGRAM_SIZE = 2 ** 16
        self.sock_in = self.__init_socket()

        self.REQUIRED_DATAGRAMS = [b'#MRZ', b'#MWC', b'#SKM', b'#SPO']


    def __init_socket(self):
        temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow reuse of addresses
        temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # TODO: Change buffer size if packets are being lost:
        temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.MAX_DATAGRAM_SIZE * 2 * 2)
        temp_sock.bind((self.rx_ip, self.rx_port))
        #temp_sock.settimeout(self.SOCKET_TIMEOUT)
        return temp_sock

    def run(self):
        while True:
            print("Listening")
            try:
                data, address = self.sock_in.recvfrom(self.MAX_DATAGRAM_SIZE)
            except BlockingIOError:
                print('error')
                continue
            except socket.timeout:
                logger.exception("Socket timeout exception.")
                self.sock_in.close()
                break

            bytes_io = io.BytesIO(data)

            header = k.read_EMdgmHeader(bytes_io)
            print("hi")
            print("header[numBytesDgm]: ", header['numBytesDgm'], type(header['dgmType']))


class Main:
    def __init__(self):
        self.snippet = None

    def run(self):
        self.snippet = Snippet("127.0.0.1", 8080)
        #self.snippet.daemon = True
        #self.snippet.start()
        #self.snippet.join()
        self.snippet.run()

if __name__ == "__main__":
    main = Main()
    main.run()