# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# May 2021

# Description: A modified version of select methods from KMALL.kmall for reading / parsing Kongsberg kmall 'M' datagrams
# received directly from SIS.

import logging
import struct
import sys

logger = logging.getLogger(__name__)


class KmallReaderForWaterColumn:
    def __init__(self):
        pass

    # ##### ----- METHODS FOR READING ALL M DATAGRAMS ----- ##### #
    @staticmethod
    def read_EMdgmHeader(file_io):
        """
        Read general datagram header.
        :return: A list containing EMdgmHeader ('header') fields: [0] = numBytesDgm; [1] = dgmType; [2] = dgmVersion;
        [3] = systemID; [4] = echoSounderID; [5] = time_sec; [6] = time_nanosec.
        """

        format_to_unpack = "1I4s2B1H2I"
        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMpartition(file_io, dgm_version):
        """
        Read multibeam (M) datagrams - data partition info. General for all M datagrams.
        Kongsberg documentation: "If a multibeam depth datagram (or any other large datagram) exceeds the limit of a
        UDP package (64 kB), the datagram is split into several datagrams =< 64 kB before sending from the PU.
        The parameters in this struct will give information of the partitioning of datagrams. K-Controller/SIS merges
        all UDP packets/datagram parts to one datagram, and store it as one datagram in the .kmall files. Datagrams
        stored in .kmall files will therefore always have numOfDgm = 1 and dgmNum = 1, and may have size > 64 kB.
        The maximum number of partitions from PU is given by MAX_NUM_MWC_DGMS and MAX_NUM_MRZ_DGMS."
        :return: A list containing EMdgmMpartition ('partition') fields:
            dgmVersion 0: [0] = numOfDgms; [1] = dgmNum.
            dgmVersion 1 (REV G): (See dgmVersion 0.)
            dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version == 0 or dgm_version == 1 or dgm_version == 2:
            format_to_unpack = "2H"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMbody(file_io, dgm_version):
        """
        Read multibeam (M) datagrams - body part. Start of body of all M datagrams.
        Contains information of transmitter and receiver used to find data in datagram.
        :return: A list containing EMdgmMbody ('cmnPart') fields:
            dgmVersion 0: [0] = numBytesCmnPart; [1] = pingCnt; [2] = rxFansPerPing; [3] = rxFanIndex;
                [4] = swathsPerPing; [5] = swathAlongPosition; [6] = txTransducerInd; [7] = rxTransducerInd;
                [8] = numRxTransducers; [9] = algorithmType.
            dgmVersion 1 (REV G): (See dgmVersion 0.)
            dgmVersion 2 (REV I): (See dgmVersion 0.)***
                *** Kongsberg: "Major change from Revision I: Every partition contains the datafield EMdgmMbody_def.
                Before Revision I, EMdgmMbody_def was only in the first partition."
        """

        if dgm_version == 0 or dgm_version == 1 or dgm_version == 2:
            format_to_unpack = "2H8B"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    # ##### ----- METHODS FOR READING MRZ DATAGRAMS ----- ##### #

    # ##### ----- METHODS FOR READING MWC DATAGRAMS ----- ##### #

    def read_EMdgmMWCtxInfo(file_io, dgm_version):
        """
        Read #MWC - data block 1: transmit sectors, general info for all sectors.
        :return: A list containing EMdgmMWCtxInfo fields:
            dgmVersion 0: [0] = numBytesTxInfo; [1] = numTxSectors; [2] = numBytesPerTxSector;
                [3] = padding; [4] = heave_m.
            dgmVersion 1 (REV G): (See dgmVersion 0.)
            dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version == 0 or dgm_version == 1 or dgm_version == 2:
            format_to_unpack = "3H1h1f"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMWCtxSectorData(file_io, dgm_version):
        """
        Read #MWC - data block 1: transmit sector data, loop for all i = numTxSectors.
        :return: A list containing EMdgmMWCtxSectorData fields:
            dgmVersion 0: [0] = tiltAngleReTx_deg; [1] = centreFreq_Hz; [2] = txBeamWidthAlong_deg;
                [3] = txSectorNum; [4] = padding.
            dgmVersion 1 (REV G): (See dgmVersion 0.)
            dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version == 0 or dgm_version == 1 or dgm_version == 2:
            format_to_unpack = "3f1H1h"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMWCrxInfo(file_io, dgm_version):
        """
        Read #MWC - data block 2: receiver, general info.
        :return: A list containing EMdgmMWCrxInfo fields:
            dgmVersion 0: [0] = numBytesRxInfo; [1] = numBeams; [2] = numBytesPerBeamEntry; [3] = phaseFlag;
                [4] = TVGfunctionApplied; [5] = TVGoffset_dB; [6] = sampleFreq_Hz; [7] = soundVelocity_mPerSec.
            dgmVersion 1 (REV G): (See dgmVersion 0.)
            dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version == 0 or dgm_version == 1 or dgm_version == 2:
            format_to_unpack = "2H3B1b2f"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMWCrxBeamData(file_io, dgm_version, phase_flag):
        """
        Read #MWC - data block 2: receiver, specific info for each beam.
        :return: A list containing EMdgmMWCrxBeamData fields:
            dgmVersion 0: [0] = beamPointAngReVertical_deg; [1] = startRangeSampleNum;
                [2] = detectedRangeInSamples; [3] = beamTxSectorNum; [4] = numSampleData; [5] = sampleAmplitude05dB_p.
                (If phase_flag > 0: [6] = rxBeamPhase.)
            dgmVersion 1 (REV G): [0] = beamPointAngReVertical_deg; [1] = startRangeSampleNum;
                [2] = detectedRangeInSamples; [3] = beamTxSectorNum; [4] = numSampleData;
                [5] = detectedRangeInSamplesHighResolution; [6] = sampleAmplitude05dB_p.
                (If phase_flag > 0: [7] = rxBeamPhase.)
            dgmVersion 2 (REV I): (See dgmVersion 1 (REV G).)
        """

        if dgm_version == 0:
            format_to_unpack = "1f4H"
        elif dgm_version == 1 or dgm_version == 2:
            format_to_unpack = "1f4H1f"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))
        fields_list = list(fields)


        # Pointer to start of array with Water Column data. Length of array = numSampleData.
        # Sample amplitudes in 0.5 dB resolution. Size of array is numSampleData * int8_t.
        # Amplitude array is followed by phase information if phaseFlag >0.
        # Use (numSampleData * int8_t) to jump to next beam, or to start of phase info for this beam, if phase flag > 0.
        format_to_unpack = str(fields[4]) + "b"
        sample_amplitude_fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))
        sample_amplitude_fields_list = list(sample_amplitude_fields)
        fields_list.append(sample_amplitude_fields_list)

        if phase_flag == 0:
            return fields_list
        # TODO: phase_flag > 0 untested!
        elif phase_flag == 1:
            # format_to_unpack = str(num_beams * fields[4]) + "b"
            format_to_unpack = str(fields[4]) + "b"
            fields.append(struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size)))
        elif phase_flag == 2:
            # format_to_unpack = str(num_beams * fields[4]) + "h"
            format_to_unpack = str(fields[4]) + "h"
            fields.append(struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size)))
        else:
            logger.warning("Invalid phase_flag value {}.".format(phase_flag))
            sys.exit(1)

    def read_format(file_io, format_to_unpack):
        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))