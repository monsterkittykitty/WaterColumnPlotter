# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

import argparse
import io
import KMALL
import logging
import socket

logger = logging.getLogger(__name__)


class KongsbergDGCapture:
    def __init__(self, rx_ip, rx_port, connection="UDP", queue=None):
        # self.q = multiprocessing.Pipe()
        # self.q = multiprocessing.Queue()

        self.connection = connection
        self.rx_ip = rx_ip
        self.rx_port = rx_port
        self.queue = queue

        self.rx_socket = self.__init_socket()

        self.REQUIRED_DATAGRAMS = [b'#MRZ', b'#MWC', b'#SKM']

    def __init_socket(self):
        if self.connection == "TCP":
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif self.connection == "UDP":
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise RuntimeError("Connection type must be 'TCP' or 'UDP'.")

    #def rx_dg(self, test_q):
    def receive_and_put_dg(self):
        self.rx_socket.bind((self.rx_ip, self.rx_port))
        # TODO: Choose more appropriate timeout number. Currently 5 sec only for testing.
        self.rx_socket.settimeout(5.0)

        #self.queue = test_q

        # From Val's code:
        k = KMALL.kmall(filename=None)
        data_buffer = None

        while True:
            try:
                data, address = self.rx_socket.recvfrom(2 ** 16)
            except socket.timeout:
                logger.exception("Socket timeout exception.")
                break

            if data_buffer is None:
                k.FID = io.BytesIO(data)
            else:
                k.FID = io.BytesIO(data_buffer + data)

            k.FID.seek(0, 0)
            #dg_header = k.read_EMdgmHeader()
            # print(dg_header)
            num_bytes_dg = k.FID.read(4)
            dg_type = k.FID.read(4)
            print(dg_type)
            print(type(dg_type))
            if dg_type in self.REQUIRED_DATAGRAMS:
                print('req dg: ', dg_type)
                exit()




            # # TODO: Not sure this line is necessary?
            # k.decode_datagram()
            # # Try to read the datagram. If it fails, we assume we've not got
            # # it all in one packet. So we seek to the beginning, read the whole
            # # into a buffer, and then loop.
            # try:
            #     k.read_datagram()
            #     data_buffer = None
            #
            #     # TODO: For testing. Note there is some error where first print statement prints long hex string.
            #     #  Error in Val's code? May results in missed data/packets that I need!
            #     # print(k.datagram_data['header']['dgmType'])
            #     # print(type(k.datagram_data['header']['dgmType']))
            #
            #     # TODO: Ensure this is always run with a queue, so this check is not needed?
            #     if self.queue is not None:
            #         if k.datagram_data['header']['dgmType'] in [b'#MRZ', b'#MWC', b'#SKM']:
            #             if k.datagram_data['header']['dgmType'] in [b'#MWC']:
            #                 print("capture: ", k.datagram_data['beamData']['beamTxSectorNum'])
            #             print("putting data")
            #             # self.queue.put(data)  # Pass binary data, this can be a problem when datagram split in two.
            #             self.queue.put(k.datagram_data)  # Pass Python dictionary
            #             print("capture q size: ", self.queue.qsize())
            # except:
            #     k.FID.seek(0, 0)
            #     # TODO: data_buffer = data instead of k.FID.read()???
            #     data_buffer = k.FID.read()
            #     #data_buffer = data
            #     continue
            #
            # print(k.datagram_data['header']['dgmType'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("rx_ip", help="IP address to receive Kongsberg datagrams.")
    parser.add_argument("rx_port", help="Port to receive Kongsberg datagrams.", type=int)
    parser.add_argument("--connection", default="UDP", help="Connection type: TCP or UDP.", choices={"TCP", "UDP"})

    args = parser.parse_args()

    if args.connection is None:
        args.connection = "UDP"

    dg_capture = KongsbergDGCapture(args.rx_ip, args.rx_port, connection=args.connection)
    dg_capture.receive_and_put_dg()
