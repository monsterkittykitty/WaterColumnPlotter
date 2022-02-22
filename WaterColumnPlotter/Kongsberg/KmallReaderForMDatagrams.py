# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# May 2021

# Description: A modified version of select methods from KMALL.kmall for
# reading / parsing Kongsberg kmall 'M' datagramsreceived directly from SIS.

import datetime
import logging
import struct
import sys

logger = logging.getLogger(__name__)


class KmallReaderForMDatagrams:
    def __init__(self):
        pass

    # ##### ----- METHODS FOR READING ALL M DATAGRAMS ----- ##### #
    @staticmethod
    def read_EMdgmHeader(file_io, return_format=False, return_fields=False):
        """
        Read general datagram header.
        :return: A list containing EMdgmHeader ('header') fields: [0] = numBytesDgm; [1] = dgmType; [2] = dgmVersion;
        [3] = systemID; [4] = echoSounderID; [5] = time_sec; [6] = time_nanosec.
        """

        format_to_unpack = "1I4s2B1H2I"

        if return_format:
            return format_to_unpack

        fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

        if return_fields:
            return fields

        dg = {}

        # Datagram length in bytes. The length field at the start (4 bytes) and end
        # of the datagram (4 bytes) are included in the length count.
        dg['numBytesDgm'] = fields[0]
        # Array of length 4. Multibeam datagram type definition, e.g. #AAA
        dg['dgmType'] = fields[1]
        # Datagram version.
        dg['dgmVersion'] = fields[2]
        # System ID. Parameter used for separating datagrams from different echosounders
        # if more than one system is connected to SIS/K-Controller.
        dg['systemID'] = fields[3]
        # Echo sounder identity, e.g. 122, 302, 710, 712, 2040, 2045, 850.
        dg['echoSounderID'] = fields[4]
        # UTC time in seconds. Epoch 1970-01-01. time_nanosec part to be added for more exact time.
        dg['time_sec']  = fields[5]
        # Nano seconds remainder. time_nanosec part to be added to time_sec for more exact time.
        dg['time_nanosec'] = fields[6]
        # UTC time in seconds + Nano seconds remainder. Epoch 1970-01-01.
        dg['dgTime'] = fields[5] + fields[6] / 1.0E9
        dg['dgdatetime'] = datetime.datetime.utcfromtimestamp(dg['dgTime'])

        return dg

    @staticmethod
    def read_EMdgmMpartition(file_io, dgm_type, dgm_version, return_format=False, return_fields=False):
        """
        Read multibeam (M) datagrams - data partition info. General for all M datagrams.
        Kongsberg documentation: "If a multibeam depth datagram (or any other large datagram) exceeds the limit of a
        UDP package (64 kB), the datagram is split into several datagrams =< 64 kB before sending from the PU.
        The parameters in this struct will give information of the partitioning of datagrams. K-Controller/SIS merges
        all UDP packets/datagram parts to one datagram, and store it as one datagram in the .kmall files. Datagrams
        stored in .kmall files will therefore always have numOfDgm = 1 and dgmNum = 1, and may have size > 64 kB.
        The maximum number of partitions from PU is given by MAX_NUM_MWC_DGMS and MAX_NUM_MRZ_DGMS."
        :return: A list containing EMdgmMpartition ('partition') fields:
            MRZ, MWC dgmVersion 0: [0] = numOfDgms; [1] = dgmNum.
            MRZ, MWC dgmVersion 1 (REV G): (See dgmVersion 0.)
            MRZ, MWC dgmVersion 2 (REV H, REV I): (See dgmVersion 0.)
            MRZ dgmVersion 3 (REV I): (See dgmVersion 0.)

        """

        if dgm_type == b'#MRZ' and dgm_version in [0, 1, 2, 3] or dgm_type == b'#MWC' and dgm_version in [0, 1, 2]:
            format_to_unpack = "2H"
        else:
            logger.warning("Datagram {} version {} unsupported.".format(dgm_type, dgm_version))
            sys.exit(1)

        if return_format:
            return format_to_unpack

        fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

        if return_fields:
            return fields

        dg = {}

        # Number of datagram parts to re-join to get one Multibeam datagram. E.g. 3.
        dg['numOfDgms'] = fields[0]
        # Datagram part number, e.g. 2 (of 3).
        dg['dgmNum'] = fields[1]

        return dg

    @staticmethod
    def read_EMdgmMbody(file_io, dgm_type, dgm_version, return_format=False, return_fields=False):
        """
        Read multibeam (M) datagrams - body part. Start of body of all M datagrams.
        Contains information of transmitter and receiver used to find data in datagram.
        :return: A list containing EMdgmMbody ('cmnPart') fields:
            MRZ, MWC dgmVersion 0: [0] = numBytesCmnPart; [1] = pingCnt; [2] = rxFansPerPing; [3] = rxFanIndex;
                [4] = swathsPerPing; [5] = swathAlongPosition; [6] = txTransducerInd; [7] = rxTransducerInd;
                [8] = numRxTransducers; [9] = algorithmType.
            MRZ, MWC dgmVersion 1 (REV G): (See dgmVersion 0.)
            MRZ, MWC dgmVersion 2 (REV H, REV I): (See dgmVersion 0.)***
                *** Kongsberg: "Major change from Revision I: Every partition contains the datafield EMdgmMbody_def.
                Before Revision I, EMdgmMbody_def was only in the first partition."
            MRZ dgmVersion 3 (REV I): (See dgmVersion 0.)***
                *** Kongsberg: "Major change from Revision I: Every partition contains the datafield EMdgmMbody_def.
                Before Revision I, EMdgmMbody_def was only in the first partition."
        """

        if dgm_type == b'#MRZ' and dgm_version in [0, 1, 2, 3] or dgm_type == b'#MWC' and dgm_version in [0, 1, 2]:
            format_to_unpack = "2H8B"
        else:
            logger.warning("Datagram {} version {} unsupported.".format(dgm_type, dgm_version))
            sys.exit(1)

        if return_format:
            return format_to_unpack

        fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

        if return_fields:
            return fields

        dg = {}

        # Used for denoting size of current struct.
        dg['numBytesCmnPart'] = fields[0]
        # A ping is made of one or more RX fans and one or more TX pulses transmitted at approximately the same time.
        # Ping counter is incremented at every set of TX pulses
        # (one or more pulses transmitted at approximately the same time).
        dg['pingCnt'] = fields[1]
        # Number of rx fans per ping gives information of how many #MRZ datagrams are generated per ping.
        # Combined with swathsPerPing, number of datagrams to join for a complete swath can be found.
        dg['rxFansPerPing'] = fields[2]
        # Index 0 is the aft swath, port side.
        dg['rxFanIndex'] = fields[3]
        # Number of swaths per ping. A swath is a complete set of across track data.
        # A swath may contain several transmit sectors and RX fans.
        dg['swathsPerPing'] = fields[4]
        # Alongship index for the location of the swath in multi swath mode. Index 0 is the aftmost swath.
        dg['swathAlongPosition'] = fields[5]
        # Transducer used in this tx fan. Index: 0 = TRAI_TX1; 1 = TRAI_TX2 etc.
        dg['txTransducerInd'] = fields[6]
        # Transducer used in this rx fan. Index: 0 = TRAI_RX1; 1 = TRAI_RX2 etc.
        dg['rxTransducerInd'] = fields[7]
        # Total number of receiving units.
        dg['numRxTransducers'] = fields[8]
        # For future use. 0 - current algorithm, >0 - future algorithms.
        dg['algorithmType'] = fields[9]

        # Skip unknown fields.
        file_io.seek(dg['numBytesCmnPart'] - struct.Struct(format_to_unpack).size, 1)

        return dg

    # ##### ----- METHODS FOR READING MRZ DATAGRAMS ----- ##### #

    @staticmethod
    def read_EMdgmMRZ_pingInfo(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Information on vessel/system level, i.e. information common to all beams in the current ping
        :param file_io: File or Bytes_IO object to be read.
        :param dgm_version: Kongsberg MRZ datagram version.
        :param return_format: Optional boolean parameter. When true, returns struct format string. Default is false.
        :return: A list containing EMdgmMRZ_pingInfo fields:
            MRZ dgmVersion 0:
            MRZ dgmVersion 1:
            MRZ dgmVersion 2:
            MRZ dgmVersion 3:
        """
        if dgm_version in [0, 1, 2, 3]:
            # For some reason, reading this all in one step does not work.
            format_to_unpack_a = "2H1f6B1H11f2h2B1H1I3f2H1f2H6f4B"
            format_to_unpack_b = "2d1f"
            # Fields common to dgm_versions 0, 1, 2, 3:
            fields_a = struct.unpack(format_to_unpack_a, file_io.read(struct.Struct(format_to_unpack_a).size))
            fields_b = struct.unpack(format_to_unpack_b, file_io.read(struct.Struct(format_to_unpack_b).size))

            if dgm_version == 0:
                if return_format:
                    return format_to_unpack_a + format_to_unpack_b
                if return_fields:
                    return fields_a + fields_b
            else:  # dgm_version in [1, 2, 3]:
                format_to_unpack_c = "1f2B2H"
                if return_format:
                    return format_to_unpack_a + format_to_unpack_b + format_to_unpack_c
                fields_c = struct.unpack(format_to_unpack_c, file_io.read(struct.Struct(format_to_unpack_c).size))
                if return_fields:
                    return fields_a + fields_b + fields_c

            dg = {}

            # Number of bytes in current struct.
            dg['numBytesInfoData'] = fields_a[0]
            # Byte alignment.
            dg['padding0'] = fields_a[1]

            # # # # # Ping Info # # # # #
            # Ping rate. Filtered/averaged.
            dg['pingRate_Hz'] = fields_a[2]
            # 0 = Eqidistance; 1 = Equiangle; 2 = High density
            dg['beamSpacing'] = fields_a[3]
            # Depth mode. Describes setting of depth in K-Controller. Depth mode influences the PUs choice of pulse length
            # and pulse type. If operator has manually chosen the depth mode to use, this is flagged by adding 100 to the
            # mode index. 0 = Very Shallow; 1 = Shallow; 2 = Medium; 3 = Deep; 4 = Deeper; 5 = Very Deep; 6 = Extra Deep;
            # 7 = Extreme Deep
            dg['depthMode'] = fields_a[4]
            # For advanced use when depth mode is set manually. 0 = Sub depth mode is not used (when depth mode is auto).
            dg['subDepthMode'] = fields_a[5]
            # Achieved distance between swaths, in percent relative to required swath distance.
            # 0 = function is not used; 100 = achieved swath distance equals required swath distance.
            dg['distanceBtwSwath'] = fields_a[6]
            # Detection mode. Bottom detection algorithm used. 0 = normal; 1 = waterway; 2 = tracking;
            # 3 = minimum depth; If system running in simulation mode: detectionmode + 100 = simulator.
            dg['detectionMode'] = fields_a[7]
            # Pulse forms used for current swath. 0 = CW; 1 = mix; 2 = FM
            dg['pulseForm'] = fields_a[8]
            # Kongsberg documentation lists padding1 as "Ping rate. Filtered/averaged." This appears to be incorrect.
            # In testing, padding1 prints all zeros. I'm assuming this is for byte alignment, as with other 'padding' cases.
            # Byte alignment.
            dg['padding1'] = fields_a[9]
            # Ping frequency in hertz. E.g. for EM 2040: 200 000 Hz, 300 000 Hz or 400 000 Hz.
            # If values is less than 100, it refers to a code defined below:
            # -1 = Not used; 0 = 40 - 100 kHz, EM 710, EM 712; 1 = 50 - 100 kHz, EM 710, EM 712;
            # 2 = 70 - 100 kHz, EM 710, EM 712; 3 = 50 kHz, EM 710, EM 712; 4 = 40 kHz, EM 710, EM 712;
            # 180 000 - 400 000 = 180-400 kHz, EM 2040C (10 kHz steps)
            # 200 000 = 200 kHz, EM 2040; 300 000 = 300 kHz, EM 2040; 400 000 = 400 kHz, EM 2040
            dg['frequencyMode_Hz'] = fields_a[10]
            # Lowest centre frequency of all sectors in this swath. Unit hertz. E.g. for EM 2040: 260 000 Hz.
            dg['freqRangeLowLim_Hz'] = fields_a[11]
            # Highest centre frequency of all sectors in this swath. Unit hertz. E.g. for EM 2040: 320 000 Hz.
            dg['freqRangeHighLim_Hz'] = fields_a[12]
            # Total signal length of the sector with longest tx pulse. Unit second.
            dg['maxTotalTxPulseLength_sec'] = fields_a[13]
            # Effective signal length (-3dB envelope) of the sector with longest effective tx pulse. Unit second.
            dg['maxEffTxPulseLength_sec'] = fields_a[14]
            # Effective bandwidth (-3dB envelope) of the sector with highest bandwidth.
            dg['maxEffTxBandWidth_Hz'] = fields_a[15]
            # Average absorption coefficient, in dB/km, for vertical beam at current depth. Not currently in use.
            dg['absCoeff_dBPerkm'] = fields_a[16]
            # Port sector edge, used by beamformer, Coverage is refered to z of SCS.. Unit degree.
            dg['portSectorEdge_deg'] = fields_a[17]
            # Starboard sector edge, used by beamformer. Coverage is referred to z of SCS. Unit degree.
            dg['starbSectorEdge_deg'] = fields_a[18]
            # Coverage achieved, corrected for raybending. Coverage is referred to z of SCS. Unit degree.
            dg['portMeanCov_deg'] = fields_a[19]
            # Coverage achieved, corrected for raybending. Coverage is referred to z of SCS. Unit degree.
            dg['stbdMeanCov_deg'] = fields_a[20]
            # Coverage achieved, corrected for raybending. Coverage is referred to z of SCS. Unit meter.
            dg['portMeanCov_m'] = fields_a[21]
            # Coverage achieved, corrected for raybending. Unit meter.
            dg['starbMeanCov_m'] = fields_a[22]
            # Modes and stabilisation settings as chosen by operator. Each bit refers to one setting in K-Controller.
            # Unless otherwise stated, default: 0 = off, 1 = on/auto.
            # Bit: 1 = Pitch stabilisation; 2  = Yaw stabilisation; 3 = Sonar mode; 4 = Angular coverage mode;
            # 5 = Sector mode; 6 = Swath along position (0 = fixed, 1 = dynamic); 7-8 = Future use
            dg['modeAndStabilisation'] = fields_a[23]
            # Filter settings as chosen by operator. Refers to settings in runtime display of K-Controller.
            # Each bit refers to one filter setting. 0 = off, 1 = on/auto.
            # Bit: 1 = Slope filter; 2 = Aeration filter; 3 = Sector filter;
            # 4 = Interference filter; 5 = Special amplitude detect; 6-8 = Future use
            dg['runtimeFilter1'] = fields_a[24]
            # Filter settings as chosen by operator. Refers to settings in runtime display of K-Controller. 4 bits used per filter.
            # Bits: 1-4 = Range gate size: 0 = small, 1 = normal, 2 = large
            # 5-8 = Spike filter strength: 0 = off, 1= weak, 2 = medium, 3 = strong
            # 9-12 = Penetration filter: 0 = off, 1 = weak, 2 = medium, 3 = strong
            # 13-16 = Phase ramp: 0 = short, 1 = normal, 2 = long
            dg['runtimeFilter2'] = fields_a[25]
            # Pipe tracking status. Describes how angle and range of top of pipe is determined.
            # 0 = for future use; 1 = PU uses guidance from SIS.
            dg['pipeTrackingStatus'] = fields_a[26]
            # Transmit array size used. Direction along ship. Unit degree.
            dg['transmitArraySizeUsed_deg'] = fields_a[27]
            # Receiver array size used. Direction across ship. Unit degree.
            dg['receiveArraySizeUsed_deg'] = fields_a[28]
            # Operator selected tx power level re maximum. Unit dB. E.g. 0 dB, -10 dB, -20 dB.
            dg['transmitPower_dB'] = fields_a[29]
            # For marine mammal protection. The parameters describes time remaining until max source level (SL) is achieved.
            # Unit %.
            dg['SLrampUpTimeRemaining'] = fields_a[30]
            # Byte alignment.
            dg['padding2'] = fields_a[31]
            # Yaw correction angle applied. Unit degree.
            dg['yawAngle_deg'] = fields_a[32]

            # # # # # Info of Tx Sector Data Block # # # # #
            # Number of transmit sectors. Also called Ntx in documentation. Denotes how
            # many times the struct EMdgmMRZ_txSectorInfo is repeated in the datagram.
            dg['numTxSectors'] = fields_a[33]
            # Number of bytes in the struct EMdgmMRZ_txSectorInfo, containing tx sector
            # specific information. The struct is repeated numTxSectors times.
            dg['numBytesPerTxSector'] = fields_a[34]

            # # # # # Info at Time of Midpoint of First Tx Pulse # # # # #
            # Heading of vessel at time of midpoint of first tx pulse. From active heading sensor.
            dg['headingVessel_deg'] = fields_a[35]
            # At time of midpoint of first tx pulse. Value as used in depth calculations.
            # Source of sound speed defined by user in K-Controller.
            dg['soundSpeedAtTxDepth_mPerSec'] = fields_a[36]
            # Tx transducer depth in meters below waterline, at time of midpoint of first tx pulse.
            # For the tx array (head) used by this RX-fan. Use depth of TX1 to move depth point (XYZ)
            # from water line to transducer (reference point of old datagram format).
            dg['txTransducerDepth_m'] = fields_a[37]
            # Distance between water line and vessel reference point in meters. At time of midpoint of first tx pulse.
            # Measured in the surface coordinate system (SCS).See Coordinate systems 'Coordinate systems' for definition.
            # Used this to move depth point (XYZ) from vessel reference point to waterline.
            dg['z_waterLevelReRefPoint_m'] = fields_a[38]
            # Distance between *.all reference point and *.kmall reference point (vessel referenece point) in meters,
            # in the surface coordinate system, at time of midpoint of first tx pulse. Used this to move depth point (XYZ)
            # from vessel reference point to the horisontal location (X,Y) of the active position sensor's reference point
            # (old datagram format).
            dg['x_kmallToall_m'] = fields_a[39]
            # Distance between *.all reference point and *.kmall reference point (vessel referenece point) in meters,
            # in the surface coordinate system, at time of midpoint of first tx pulse. Used this to move depth point (XYZ)
            # from vessel reference point to the horisontal location (X,Y) of the active position sensor's reference point
            # (old datagram format).
            dg['y_kmallToall_m'] = fields_a[40]
            # Method of position determination from position sensor data:
            # 0 = last position received; 1 = interpolated; 2 = processed.
            dg['latLongInfo'] = fields_a[41]
            # Status/quality for data from active position sensor. 0 = valid data, 1 = invalid data, 2 = reduced performance
            dg['posSensorStatus'] = fields_a[42]
            # Status/quality for data from active attitude sensor. 0 = valid data, 1 = invalid data, 2 = reduced performance
            dg['attitudeSensorStatus'] = fields_a[43]
            # Padding for byte alignment.
            dg['padding3'] = fields_a[44]

            # Latitude (decimal degrees) of vessel reference point at time of midpoint of first tx pulse.
            # Negative on southern hemisphere. Parameter is set to define UNAVAILABLE_LATITUDE if not available.
            dg['latitude_deg'] = fields_b[0]
            # Longitude (decimal degrees) of vessel reference point at time of midpoint of first tx pulse.
            # Negative on western hemisphere. Parameter is set to define UNAVAILABLE_LONGITUDE if not available.
            dg['longitude_deg'] = fields_b[1]
            # Height of vessel reference point above the ellipsoid, derived from active GGA sensor.
            # ellipsoidHeightReRefPoint_m is GGA height corrected for motion and installation offsets
            # of the position sensor.
            dg['ellipsoidHeightReRefPoint_m'] = fields_b[2]

            if dgm_version == 0:
                # Skip unknown fields.
                file_io.seek(dg['numBytesInfoData'] - struct.Struct(format_to_unpack_a).size -
                             struct.Struct(format_to_unpack_b).size, 1)
                return dg
            else:  # dgm_version in [1, 2, 3]:
                # Backscatter offset set in the installation menu
                dg['bsCorrectionOffset_dB'] = fields_c[0]
                # Beam intensity data corrected as seabed image data (Lambert and normal incidence corrections)
                dg['lambertsLawApplied'] = fields_c[1]
                # Ice window installed
                dg['iceWindow'] = fields_c[2]
                # Sets status for active modes.
                # |_Bit__|_________Modes__________|________Setting________|
                # |__1___|_EM_Multifrequency_Mode_|_0=Not_Active;_1=Active|
                # |_2-16_|_Not_In_Use_____________|_NA____________________|
                dg['activeModes'] = fields_c[3]

                # Skip unknown fields.
                file_io.seek(dg['numBytesInfoData'] - struct.Struct(format_to_unpack_a).size -
                             struct.Struct(format_to_unpack_b).size - struct.Struct(format_to_unpack_c).size, 1)
                return dg

        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMRZ_txSectorInfo(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Information specific to each transmitting sector. sectorInfo is repeated numTxSectors (Ntx)- times in datagram.
        :param file_io: File or Bytes_IO object to be read.
        :param dgm_version: Kongsberg MRZ datagram version.
        :param return_format: Optional boolean parameter. When true, returns struct format string. Default is false.
        :return: A list containing EMdgmMRZ_txSectorInfo fields:
            MRZ dgmVersion 0:
            MRZ dgmVersion 1:
            MRZ dgmVersion 2:
            MRZ dgmVersion 3:
        """
        if dgm_version in [0, 1, 2, 3]:
            format_to_unpack_a = "4B7f2B1H"
            # Fields common to dgm_versions 0, 1, 2, 3:
            fields_a = struct.unpack(format_to_unpack_a, file_io.read(struct.Struct(format_to_unpack_a).size))

            if dgm_version == 0:
                if return_format:
                    return format_to_unpack_a
                if return_fields:
                    return fields_a
            else:  # dgm_version in [1, 2, 3]:
                format_to_unpack_b = "3f"
                if return_format:
                    return format_to_unpack_a + format_to_unpack_b
                fields_b = struct.unpack(format_to_unpack_b, file_io.read(struct.Struct(format_to_unpack_b).size))
                if return_fields:
                    return fields_a + fields_b

            dg = {}

            # TX sector index number, used in the sounding section. Starts at 0.
            dg['txSectorNumb'] = fields_a[0]
            # TX array number. Single TX, txArrNumber = 0.
            dg['txArrNumber'] = fields_a[1]
            # Default = 0. E.g. for EM2040, the transmitted pulse consists of three sectors, each transmitted from
            # separate txSubArrays. Orientation and numbers are relative the array coordinate system. Sub array
            # installation offsets can be found in the installation datagram, #IIP. 0 = Port subarray; 1 = middle
            # subarray; 2 = starboard subarray
            dg['txSubArray'] = fields_a[2]
            # Byte alignment.
            dg['padding0'] = fields_a[3]
            # Transmit delay of the current sector/subarray. Delay is the time from the midpoint of the current
            # transmission to midpoint of the first transmitted pulse of the ping, i.e. relative to the time used in
            # the datagram header.
            dg['sectorTransmitDelay_sec'] = fields_a[4]
            # Along ship steering angle of the TX beam (main lobe of transmitted pulse),
            # angle referred to transducer array coordinate system. Unit degree.
            dg['tiltAngleReTx_deg'] = fields_a[5]
            # Unit dB re 1 microPascal.
            dg['txNominalSourceLevel_dB'] = fields_a[6]
            # 0 = no focusing applied.
            dg['txFocusRange_m'] = fields_a[7]
            # Centre frequency. Unit hertz.
            dg['centreFreq_Hz'] = fields_a[8]
            # FM mode: effective bandwidth; CW mode: 1/(effective TX pulse length)
            dg['signalBandWidth_Hz'] = fields_a[9]
            # Also called pulse length. Unit second.
            dg['totalSignalLength_sec'] = fields_a[10]
            # Transmit pulse is shaded in time (tapering). Amplitude shading in %.
            # cos2- function used for shading the TX pulse in time.
            dg['pulseShading'] = fields_a[11]
            # Transmit signal wave form. 0 = CW; 1 = FM upsweep; 2 = FM downsweep.
            dg['signalWaveForm'] = fields_a[12]
            # Byte alignment.
            dg['padding1'] = fields_a[13]

            if dgm_version == 0:
                return dg
            else:  # dgm_version in [1, 2, 3]
                # 20 log(Measured high voltage power level at TX pulse / Nominal high voltage power level). This
                # parameter will also include the effect of user selected transmit power reduction (transmitPower_dB)
                # and mammal protection. Actual SL = txNominalSourceLevel_dB + highVoltageLevel_dB. Unit dB.
                dg['highVoltageLeveldB'] = fields_b[0]
                # Backscatter correction added in sector tracking mode. Unit dB.
                dg['sectorTrackingCorr_dB'] = fields_b[1]
                # Signal length used for backscatter footprint calculation. This compensates for the TX pulse
                # tapering and the RX filter bandwidths. Unit second.
                dg['effectiveSignalLength'] = fields_b[2]
                return dg

        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMRZ_rxInfo(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Receiver specific information. Information specific to the receiver unit used in this swath.
        :param file_io: File or Bytes_IO object to be read.
        :param dgm_version: Kongsberg MRZ datagram version.
        :param return_format: Optional boolean parameter. When true, returns struct format string. Default is false.
        :return: A list containing EMdgmMRZ_rxInfo fields:
            MRZ dgmVersion 0:
            MRZ dgmVersion 1:
            MRZ dgmVersion 2:
            MRZ dgmVersion 3:
        """
        if dgm_version in [0, 1, 2, 3]:
            format_to_unpack = "4H4f4H"

            if return_format:
                return format_to_unpack

            fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

            if return_fields:
                return fields

            dg = {}

            # Bytes in current struct.
            dg['numBytesRxInfo'] = fields[0]
            # Maximum number of main soundings (bottom soundings) in this datagram, extra detections
            # (soundings in water column) excluded. Also referred to as Nrx. Denotes how many bottom points
            # (or loops) given in the struct EMdgmMRZ_sounding_def.
            dg['numSoundingsMaxMain'] = fields[1]
            # Number of main soundings of valid quality. Extra detections not included.
            dg['numSoundingsValidMain'] = fields[2]
            # Bytes per loop of sounding (per depth point), i.e. bytes per loops of the struct EMdgmMRZ_sounding_def.
            dg['numBytesPerSounding'] = fields[3]
            # Sample frequency divided by water column decimation factor. Unit hertz.
            dg['WCSampleRate'] = fields[4]
            # Sample frequency divided by seabed image decimation factor. Unit hertz.
            dg['seabedImageSampleRate'] = fields[5]
            # Backscatter level, normal incidence. Unit dB.
            dg['BSnormal_dB'] = fields[6]
            # Backscatter level, oblique incidence. Unit dB.
            dg['BSoblique_dB'] = fields[7]
            # extraDetectionAlarmFlag = sum of alarm flags. Range 0-10.
            dg['extraDetectionAlarmFlag'] = fields[8]
            # Sum of extradetection from all classes. Also refered to as Nd.
            dg['numExtraDetections'] = fields[9]
            # Range 0-10.
            dg['numExtraDetectionClasses'] = fields[10]
            # Number of bytes in the struct EMdgmMRZ_extraDetClassInfo_def.
            dg['numBytesPerClass'] = fields[11]

            # Skip unknown fields.
            file_io(dg['numBytesRxInfo'] - struct.Struct(format_to_unpack).size, 1)

            return dg

        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMRZ_extraDetClassInfo(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Extra detection class information. To be entered in loop numExtraDetectionClasses times.
        :param file_io: File or Bytes_IO object to be read.
        :param dgm_version: Kongsberg MRZ datagram version.
        :param return_format: Optional boolean parameter. When true, returns struct format string. Default is false.
        :return: A list containing EMdgmMRZ_rxInfo fields:
            MRZ dgmVersion 0:
            MRZ dgmVersion 1:
            MRZ dgmVersion 2:
            MRZ dgmVersion 3:
        """
        if dgm_version in [0, 1, 2, 3]:
            format_to_unpack = "1H1b1B"
            if return_format:
                return format_to_unpack

            fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

            if return_fields:
                return fields

            dg = {}

            # Number of extra detection in this class.
            dg['numExtraDetInClass'] = fields[0]
            # Byte alignment.
            dg['padding'] = fields[1]
            # 0 = no alarm; 1 = alarm.
            dg['alarmFlag'] = fields[2]

            return dg

        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMRZ_sounding(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Data for each sounding, e.g. XYZ, reflectivity, two way travel time etc. Also contains information necessary
        to read seabed image following this datablock (number of samples in SI etc.).
        To be entered in loop (numSoundingsMaxMain + numExtraDetections) times.
        :param file_io: File or Bytes_IO object to be read.
        :param dgm_version: Kongsberg MRZ datagram version.
        :param return_format: Optional boolean parameter. When true, returns struct format string. Default is false.
        :return: A list containing EMdgmMRZ_rxInfo fields:
            MRZ dgmVersion 0:
            MRZ dgmVersion 1:
            MRZ dgmVersion 2:
            MRZ dgmVersion 3:
        """
        if dgm_version in [0, 1, 2, 3]:
            format_to_unpack = "1H8B1H6f2H18f4H"
            if return_format:
                return format_to_unpack

            fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

            if return_fields:
                return fields

            dg = {}

            # Sounding index. Cross reference for seabed image.
            # Valid range: 0 to (numSoundingsMaxMain+numExtraDetections)-1, i.e. 0 - (Nrx+Nd)-1.
            dg['soundingIndex'] = fields[0]
            # Transmitting sector number. Valid range: 0-(Ntx-1), where Ntx is numTxSectors.
            dg['txSectorNumb'] = fields[1]

            # # # # # D E T E C T I O N   I N F O # # # # #
            # Bottom detection type. Normal bottom detection, extra detection, or rejected.
            # 0 = normal detection; 1 = extra detection; 2 = rejected detection
            # In case 2, the estimated range has been used to fill in amplitude samples in the seabed image datagram.
            dg['detectionType'] = fields[2]
            # Method for determining bottom detection, e.g. amplitude or phase.
            # 0 = no valid detection; 1 = amplitude detection; 2 = phase detection; 3-15 for future use.
            dg['detectionMethod'] = fields[3]
            # For Kongsberg use.
            dg['rejectionInfo1'] = fields[4]
            # For Kongsberg use.
            dg['rejectionInfo2'] = fields[5]
            # For Kongsberg use.
            dg['postProcessingInfo'] = fields[6]
            # Only used by extra detections. Detection class based on detected range. Detection class 1 to 7
            # corresponds to value 0 to 6. If the value is between 100 and 106, the class is disabled by the
            # operator. If the value is 107, the detections are outside the treshhold limits.
            dg['detectionClass'] = fields[7]
            # Detection confidence level.
            dg['detectionConfidenceLevel'] = fields[8]
            # Byte alignment.
            dg['padding'] = fields[9]
            # Unit %. rangeFactor = 100 if main detection.
            dg['rangeFactor'] = fields[10]
            # Estimated standard deviation as % of the detected depth. Quality Factor (QF) is
            # calculated from IFREMER Quality Factor (IFQ): QF=Est(dz)/z=100*10^-IQF
            dg['qualityFactor'] = fields[11]
            # Vertical uncertainty, based on quality factor (QF, qualityFactor).
            dg['detectionUncertaintyVer_m'] = fields[12]
            # Horizontal uncertainty, based on quality factor (QF, qualityFactor).
            dg['detectionUncertaintyHor_m'] = fields[13]
            # Detection window length. Unit second. Sample data range used in final detection.
            dg['detectionWindowLength_sec'] = fields[14]
            # Measured echo length. Unit second.
            dg['echoLength_sec'] = fields[15]

            # # # # # W A T E R   C O L U M N   P A R A M E T E R S # # # # #
            # Water column beam number. Info for plotting soundings together with water column data.
            dg['WCBeamNumb'] = fields[16]
            # Water column range. Range of bottom detection, in samples.
            dg['WCrange_samples'] = fields[17]
            # Water column nominal beam angle across. Re vertical.
            dg['WCNomBeamAngleAcross_deg'] = fields[18]

            # # # # # REFLECTIVITY DATA (BACKSCATTER (BS) DATA) # # # # #
            # Mean absorption coefficient, alfa. Used for TVG calculations. Value as used. Unit dB/km.
            dg['meanAbsCoeff_dbPerkm'] = fields[19]
            # Beam intensity, using the traditional KM special TVG.
            dg['reflectivity1_dB'] = fields[20]
            # Beam intensity (BS), using TVG = X log(R) + 2 alpha R. X (operator selected) is common to all beams in
            # datagram. Alpha (variabel meanAbsCoeff_dBPerkm) is given for each beam (current struct).
            # BS = EL - SL - M + TVG + BScorr, where EL= detected echo level (not recorded in datagram),
            # and the rest of the parameters are found below.
            dg['reflectivity2_dB'] = fields[21]
            # Receiver sensitivity (M), in dB, compensated for RX beampattern
            # at actual transmit frequency at current vessel attitude.
            dg['receiverSensitivityApplied_dB'] = fields[22]
            # Source level (SL) applied (dB): SL = SLnom + SLcorr, where SLnom = Nominal maximum SL,
            # recorded per TX sector (variable txNominalSourceLevel_dB in struct EMdgmMRZ_txSectorInfo_def) and
            # SLcorr = SL correction relative to nominal TX power based on measured high voltage power level and
            # any use of digital power control. SL is corrected for TX beampattern along and across at actual transmit
            # frequency at current vessel attitude.
            dg['sourceLevelApplied_dB'] = fields[23]
            # Backscatter (BScorr) calibration offset applied (default = 0 dB).
            dg['BScalibration_dB'] = fields[24]
            # Time Varying Gain (TVG) used when correcting reflectivity.
            dg['TVG_dB'] = fields[25]

            # # # # # R A N G E   A N D   A N G L E   D A T A # # # # #
            # Angle relative to the RX transducer array, except for ME70,
            # where the angles are relative to the horizontal plane.
            dg['beamAngleReRx_deg'] = fields[26]
            # Applied beam pointing angle correction.
            dg['beamAngleCorrection_deg'] = fields[27]
            # Two way travel time (also called range). Unit second.
            dg['twoWayTravelTime_sec'] = fields[28]
            # Applied two way travel time correction. Unit second.
            dg['twoWayTravelTimeCorrection_sec'] = fields[29]

            # # # # # G E O R E F E R E N C E D   D E P T H   P O I N T S # # # # # Distance from vessel reference
            # point at time of first tx pulse in ping, to depth point. Measured in the surface coordinate system (
            # SCS), see Coordinate systems for definition. Unit decimal degrees.
            dg['deltaLatitude_deg'] = fields[30]
            # Distance from vessel reference point at time of first tx pulse in ping, to depth point. Measured in the
            # surface coordinate system (SCS), see Coordinate systems for definition. Unit decimal degrees.
            dg['deltaLongitude_deg'] = fields[31]
            # Vertical distance z. Distance from vessel reference point at time of first tx pulse in ping,
            # to depth point. Measured in the surface coordinate system (SCS), see Coordinate systems for definition.
            dg['z_reRefPoint_m'] = fields[32]
            # Horizontal distance y. Distance from vessel reference point at time of first tx pulse in ping,
            # to depth point. Measured in the surface coordinate system (SCS), see Coordinate systems for definition.
            dg['y_reRefPoint_m'] = fields[33]
            # Horizontal distance x. Distance from vessel reference point at time of first tx pulse in ping,
            # to depth point. Measured in the surface coordinate system (SCS), see Coordinate systems for definition.
            dg['x_reRefPoint_m'] = fields[34]
            # Beam incidence angle adjustment (IBA) unit degree.
            dg['beamIncAngleAdj_deg'] = fields[35]
            # For future use.
            dg['realTimeCleanInfo'] = fields[36]

            # # # # # S E A B E D   I M A G E # # # # #
            # Seabed image start range, in sample number from transducer. Valid only for the current beam.
            dg['SIstartRange_samples'] = fields[37]
            # Seabed image. Number of the centre seabed image sample for the current beam.
            dg['SIcentreSample'] = fields[38]
            # Seabed image. Number of range samples from the current beam, used to form the seabed image.
            dg['SInumSamples'] = fields[39]

            return dg

        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @classmethod
    def read_EMdgmMRZ(cls, file_io, return_format=False, return_fields=False):
        # TODO: Test!
        file_io.seek(0, 0)

        dg = {}
        dg['header'] = cls.read_EMdgmHeader(file_io)
        dg['partition'] = cls.read_EMdgmMpartition(file_io, dgm_type=dg['header']['dgmType'],
                                                   dgm_version=dg['header']['dgmVersion'])
        dg['cmnPart'] = cls.read_EMdgmMbody(file_io, dgm_type=dg['header']['dgmType'],
                                            dgm_version=dg['header']['dgmVersion'])
        dg['pingInfo'] = cls.read_EMdgmMRZ_pingInfo(file_io, dgm_version=dg['header']['dgmVersion'])

        # Read TX sector info for each sector
        txSectorInfo = []
        for sector in range(dg['pingInfo']['numTxSectors']):
            txSectorInfo.append(cls.read_EMdgmMRZ_txSectorInfo(file_io, dgm_version=dg['header']['dgmVersion']))
        dg['txSectorInfo'] = cls.listofdicts2dictoflists(txSectorInfo)

        dg['rxInfo'] = cls.read_EMdgmMRZ_rxInfo(file_io, dgm_version=dg['header']['dgmVersion'])

        # Read extra detect metadata if they exist.
        extraDetClassInfo = []
        for detclass in range(dg['rxInfo']['numExtraDetectionClasses']):
            extraDetClassInfo.append(cls.read_EMdgmMRZ_extraDetClassInfo(file_io,
                                                                         dgm_version=dg['header']['dgmVersion']))
        dg['extraDetClassInfo'] = cls.listofdicts2dictoflists(extraDetClassInfo)

        # Read the sounding data.
        soundings = []
        Nseabedimage_samples = 0
        for record in range(dg['rxInfo']['numExtraDetections'] + dg['rxInfo']['numSoundingsMaxMain']):
            soundings.append(cls.read_EMdgmMRZ_sounding(file_io, dgm_version=dg['header']['dgmVersion']))
            Nseabedimage_samples += soundings[record]['SInumSamples']
        dg['sounding'] = cls.listofdicts2dictoflists(soundings)

        # Read seabed image sample.
        format_to_unpack = str(Nseabedimage_samples) + "h"
        dg['SIsample_desidB'] = struct.unpack(format_to_unpack, cls.FID.read(struct.Struct(format_to_unpack).size))

    # ##### ----- METHODS FOR READING MWC DATAGRAMS ----- ##### #

    @staticmethod
    def read_EMdgmMWC_txInfo(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Read #MWC - data block 1: transmit sectors, general info for all sectors.
        :return: A list containing EMdgmMWCtxInfo fields:
            MWC dgmVersion 0: [0] = numBytesTxInfo; [1] = numTxSectors; [2] = numBytesPerTxSector;
                [3] = padding; [4] = heave_m.
            MWC dgmVersion 1 (REV G): (See dgmVersion 0.)
            MWC dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version in [0, 1, 2]:
            format_to_unpack = "3H1h1f"

            if return_format:
                return format_to_unpack

            fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

            if return_fields:
                return fields

            dg = {}

            # Number of bytes in current struct.
            dg['numBytesTxInfo'] = fields[0]
            # Number of transmitting sectors (Ntx). Denotes the number of times
            # the struct EMdgmMWCtxSectorData is repeated in the datagram.
            dg['numTxSectors'] = fields[1]
            # Number of bytes in EMdgmMWCtxSectorData.
            dg['numBytesPerTxSector'] = fields[2]
            # Byte alignment.
            dg['padding'] = fields[3]
            # Heave at vessel reference point, at time of ping, i.e. at midpoint of first tx pulse in rxfan.
            dg['heave_m'] = fields[4]

            # Skip unknown fields.
            file_io.seek(dg['numBytesTxInfo'] - struct.Struct(format_to_unpack).size, 1)

            return dg

        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMWC_txSectorData(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Read #MWC - data block 1: transmit sector data, loop for all i = numTxSectors.
        :return: A list containing EMdgmMWCtxSectorData fields:
            MWC dgmVersion 0: [0] = tiltAngleReTx_deg; [1] = centreFreq_Hz; [2] = txBeamWidthAlong_deg;
                [3] = txSectorNum; [4] = padding.
            MWC dgmVersion 1 (REV G): (See dgmVersion 0.)
            MWC dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version in [0, 1, 2]:
            format_to_unpack = "3f1H1h"

            if return_format:
                return format_to_unpack

            fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

            if return_fields:
                return fields

            dg = {}

            # Along ship steering angle of the TX beam (main lobe of transmitted pulse), angle referred to transducer face.
            # Angle as used by beamformer (includes stabilisation). Unit degree.
            dg['tiltAngleReTx_deg'] = fields[0]
            # Centre frequency of current sector. Unit hertz.
            dg['centreFreq_Hz'] = fields[1]
            # Corrected for frequency, sound velocity and tilt angle. Unit degree.
            dg['txBeamWidthAlong_deg'] = fields[2]
            # Transmitting sector number.
            dg['txSectorNum'] = fields[3]
            # Byte alignment.
            dg['padding'] = fields[4]

            return dg

        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMWC_rxInfo(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Read #MWC - data block 2: receiver, general info.
        :return: A list containing EMdgmMWCrxInfo fields:
            MWC dgmVersion 0: [0] = numBytesRxInfo; [1] = numBeams; [2] = numBytesPerBeamEntry; [3] = phaseFlag;
                [4] = TVGfunctionApplied; [5] = TVGoffset_dB; [6] = sampleFreq_Hz; [7] = soundVelocity_mPerSec.
            MWC dgmVersion 1 (REV G): (See dgmVersion 0.)
            MWC dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version in [0, 1, 2]:
            format_to_unpack = "2H3B1b2f"

            if return_format:
                return format_to_unpack

            fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

            if return_fields:
                return fields

            dg = {}

            # Number of bytes in current struct.
            dg['numBytesRxInfo'] = fields[0]
            # Number of beams in this datagram (Nrx).
            dg['numBeams'] = fields[1]
            # Bytes in EMdgmMWCrxBeamData struct, excluding sample amplitudes (which have varying lengths).
            dg['numBytesPerBeamEntry'] = fields[2]
            # 0 = off; 1 = low resolution; 2 = high resolution.
            dg['phaseFlag'] = fields[3]
            # Time Varying Gain function applied (X). X log R + 2 Alpha R + OFS + C, where X and C is documented
            # in #MWC datagram. OFS is gain offset to compensate for TX source level, receiver sensitivity etc.
            dg['TVGfunctionApplied'] = fields[4]
            # Time Varying Gain offset used (OFS), unit dB. X log R + 2 Alpha R + OFS + C, where X and C is documented
            # in #MWC datagram. OFS is gain offset to compensate for TX source level, receiver sensitivity etc.
            dg['TVGoffset_dB'] = fields[5]
            # The sample rate is normally decimated to be approximately the same as the bandwidth of the transmitted
            # pulse. Unit hertz.
            dg['sampleFreq_Hz'] = fields[6]
            # Sound speed at transducer, unit m/s.
            dg['soundVelocity_mPerSec'] = fields[7]

            # Skip unknown fields.
            file_io.seek(dg['numBytesRxInfo'] - struct.Struct(format_to_unpack).size, 1)

            return dg

        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMWC_rxBeamData(file_io, dgm_version, return_format=False, return_fields=False):
        """
        Read #MWC - data block 2: receiver, specific info for each beam.
        :return: A list containing EMdgmMWCrxBeamData fields:
            MWC dgmVersion 0: [0] = beamPointAngReVertical_deg; [1] = startRangeSampleNum;
                [2] = detectedRangeInSamples; [3] = beamTxSectorNum; [4] = numSampleData; [5] = sampleAmplitude05dB_p.
                (If phase_flag > 0: [6] = rxBeamPhase.)
            MWC dgmVersion 1 (REV G): [0] = beamPointAngReVertical_deg; [1] = startRangeSampleNum;
                [2] = detectedRangeInSamples; [3] = beamTxSectorNum; [4] = numSampleData;
                [5] = detectedRangeInSamplesHighResolution; [6] = sampleAmplitude05dB_p.
                (If phase_flag > 0: [7] = rxBeamPhase.)
            MWC dgmVersion 2 (REV I): (See dgmVersion 1 (REV G).)
        """
        if dgm_version == 0:
            format_to_unpack_a = "1f4H"
        elif dgm_version in [1, 2]:
            format_to_unpack_a = "1f4H1f"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        fields_a = struct.unpack(format_to_unpack_a, file_io.read(struct.Struct(format_to_unpack_a).size))

        format_to_unpack_b = str(fields_a[4]) + "b"

        if return_format:
            return format_to_unpack_a + format_to_unpack_b

        # Pointer to start of array with Water Column data. Length of array = numSampleData.
        # Sample amplitudes in 0.5 dB resolution. Size of array is numSampleData * int8_t.
        # Amplitude array is followed by phase information if phaseFlag >0.
        # Use (numSampleData * int8_t) to jump to next beam, or to start of phase info for this beam, if phase flag > 0.
        fields_b = struct.unpack(format_to_unpack_b, file_io.read(struct.Struct(format_to_unpack_b).size))

        if return_fields:
            return fields_a + fields_b

        dg = {}

        dg['beamPointAngReVertical_deg'] = fields_a[0]
        dg['startRangeSampleNum'] = fields_a[1]
        # Two way range in samples. Approximation to calculated distance from tx to bottom detection
        # [meters] = soundVelocity_mPerSec * detectedRangeInSamples / (sampleFreq_Hz * 2).
        # The detected range is set to zero when the beam has no bottom detection.
        dg['detectedRangeInSamples'] = fields_a[2]
        dg['beamTxSectorNum'] = fields_a[3]
        # Number of sample data for current beam. Also denoted Ns.
        dg['numSampleData'] = fields_a[4]

        if dgm_version in [1, 2]:

            dg['detectedRangeInSamplesHighResolution'] = fields_a[5]

        # Pointer to start of array with Water Column data. Length of array = numSampleData.
        # Sample amplitudes in 0.5 dB resolution. Size of array is numSampleData * int8_t.
        # Amplitude array is followed by phase information if phaseFlag >0. Use (numSampleData * int8_t)
        # to jump to next beam, or to start of phase info for this beam, if phase flag > 0.

        dg['sampleAmplitude05dB_p'] = fields_b

        return dg

    @staticmethod
    def read_EMdgmMWC_rxBeamPhase1(file_io, dgm_version, num_sample_data,
                                   return_format=False, return_fields=False):

        if dgm_version in [0, 1, 2]:
            format_to_unpack = str(num_sample_data) + "b"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        if return_format:
            return format_to_unpack

        fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

        if return_fields:
            return fields

        dg = {}
        # Rx beam phase in 180/128 degree resolution.
        dg['rxBeamPhase'] = fields
        return dg

    @staticmethod
    def read_EMdgmMWC_rxBeamPhase2(file_io, dgm_version, num_sample_data,
                                   return_format=False, return_fields=False):

        if dgm_version in [0, 1, 2]:
            format_to_unpack = str(num_sample_data) + "h"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        if return_format:
            return format_to_unpack

        fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

        if return_fields:
            return fields

        dg = {}
        # Rx beam phase in 0.01 degree resolution.
        dg['rxBeamPhase'] = fields
        return dg

    @classmethod
    def read_EMdgmMWC(cls, file_io, return_format=False, return_fields=False):
        file_io.seek(0, 0)

        dg = {}
        dg['header'] = cls.read_EMdgmHeader(file_io)
        dg['partition'] = cls.read_EMdgmMpartition(file_io, dgm_type=dg['header']['dgmType'],
                                                    dgm_version=dg['header']['dgmVersion'])
        dg['cmnPart'] = cls.read_EMdgmMbody(file_io, dgm_type=dg['header']['dgmType'],
                                                    dgm_version=dg['header']['dgmVersion'])
        dg['txInfo'] = cls.read_EMdgmMWC_txInfo(file_io, dgm_version=dg['header']['dgmVersion'])

        # Read TX sector info for each sector
        txSectorData = []
        for sector in range(dg['txInfo']['numTxSectors']):
            txSectorData.append(cls.read_EMdgmMWC_txSectorData(file_io, dgm_version=dg['header']['dgmVersion']))
        dg['sectorData'] = cls.listofdicts2dictoflists(txSectorData)

        dg['rxInfo'] = cls.read_EMdgmMWC_rxInfo(file_io, dgm_version=dg['header']['dgmVersion'])
        # Pointer to beam related information. Struct defines information about data for a beam. Beam information is
        # followed by sample amplitudes in 0.5 dB resolution . Amplitude array is followed by phase information if
        # phaseFlag >0. These data defined by struct EMdgmMWCrxBeamPhase1_def (int8_t) or struct
        # EMdgmMWCrxBeamPhase2_def (int16_t) if indicated in the field phaseFlag in struct EMdgmMWCrxInfo_def.
        # Length of data block for each beam depends on the operators choice of phase information (see table):
        '''
                phaseFlag:      Beam Block Size: 
                0               numBytesPerBeamEntry + numSampleData * size(sampleAmplitude05dB_p)
                1               numBytesPerBeamEntry + numSampleData * size(sampleAmplitude05dB_p)
                                    + numSampleData * size(EMdgmMWCrxBeamPhase1_def)
                2               numBytesPerBeamEntry + numSampleData * size(sampleAmplitude05dB_p)
                                    + numSampleData * size(EMdgmMWCrxBeamPhase2_def)
        '''

        rxBeamData = []
        rxPhaseInfo = []
        for idx in range(dg['rxInfo']['numBeams']):
            rxBeamData.append(cls.read_EMdgmMWC_rxBeamData(file_io, dgm_version=dg['header']['dgmVersion']))

            if dg['rxInfo']['phaseFlag'] == 0:
                pass

            elif dg['rxInfo']['phaseFlag'] == 1:
                # TODO: Test with water column data, phaseFlag = 1 to complete/test this function.
                rxPhaseInfo.append(cls.read_EMdgmMWC_rxBeamPhase1(file_io, dgm_version=dg['header']['dgmVersion'],
                                                                   num_sample_data=rxBeamData[idx]['numSampleData']))

            elif dg['rxInfo']['phaseFlag'] == 2:
                # TODO: Test with water column data, phaseFlag = 2 to complete/test this function.
                rxPhaseInfo.append(cls.read_EMdgmMWC_rxBeamPhase2(file_io, dgm_version=dg['header']['dgmVersion'],
                                                                   num_sample_data=rxBeamData[idx]['numSampleData']))
            else:
                logger.warning("Phase flag {} unsupported.".format(dg['rxInfo']['phaseFlag']))
                sys.exit(1)

        dg['beamData'] = cls.listofdicts2dictoflists(rxBeamData)

        # TODO: Should this be handled in a different way? By this method, number of fields in dg is variable.
        if dg['rxInfo']['phaseFlag'] == 1 or dg['rxInfo']['phaseFlag'] == 2:
            dg['phaseInfo'] = cls.listofdicts2dictoflists(rxPhaseInfo)

        return dg

    @staticmethod
    def read_format(file_io, format_to_unpack):
        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    @staticmethod
    def listofdicts2dictoflists(listofdicts):
        """ A utility  to convert a list of dicts to a dict of lists."""
        if listofdicts:
            needs_flattening = [k for (k, v) in listofdicts[0].items() if isinstance(v, list)]
            d_of_l = {k: [dic[k] for dic in listofdicts] for k in listofdicts[0]}
            if needs_flattening:
                # print('flattening {}'.format(needs_flattening))
                for nf in needs_flattening:
                    d_of_l[nf] = [item for sublist in d_of_l[nf] for item in sublist]
            return d_of_l
        else:
            return None
