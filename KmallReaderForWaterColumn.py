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
    def read_EMdgmHeader(file_io, return_format=False):
        """
        Read general datagram header.
        :return: A list containing EMdgmHeader ('header') fields: [0] = numBytesDgm; [1] = dgmType; [2] = dgmVersion;
        [3] = systemID; [4] = echoSounderID; [5] = time_sec; [6] = time_nanosec.
        """

        format_to_unpack = "1I4s2B1H2I"

        if return_format:
            return format_to_unpack

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMpartition(file_io, dgm_type, dgm_version, return_format=False):
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

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMbody(file_io, dgm_type, dgm_version, return_format=False):
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

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    # ##### ----- METHODS FOR READING MRZ DATAGRAMS ----- ##### #

    @staticmethod
    def read_EMdgmMRZ_pingInfo(file_io, dgm_version, return_format=False):
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
        if dgm_version == 0 or 1 or 2 or 3:
            # For some reason, reading this all in one step does not work.
            format_to_unpack_a = "2H1f6B1H11f2h2B1H1I3f2H1f2H6f4B"
            format_to_unpack_b = "2d1f"
            # Fields common to dgm_versions 0, 1, 2, 3:
            fields_a = struct.unpack(format_to_unpack_a, file_io.read(struct.Struct(format_to_unpack_a).size))
            fields_b = struct.unpack(format_to_unpack_b, file_io.read(struct.Struct(format_to_unpack_b).size))

            if dgm_version == 0:
                if return_format:
                    return format_to_unpack_a + format_to_unpack_b
                return fields_a + fields_b
            else:  # dgm_version == 1 or 2 or 3:
                format_to_unpack_c = "1f2B2H"
                if return_format:
                    return format_to_unpack_a + format_to_unpack_b + format_to_unpack_c
                fields_c = struct.unpack(format_to_unpack_c, file_io.read(struct.Struct(format_to_unpack_c).size))
                return fields_a + fields_b + fields_c
        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMRZ_txSectorInfo(file_io, dgm_version, return_format=False):
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
        if dgm_version == 0 or 1 or 2 or 3:
            format_to_unpack_a = "4B7f2B1H"
            # Fields common to dgm_versions 0, 1, 2, 3:
            fields_a = struct.unpack(format_to_unpack_a, file_io.read(struct.Struct(format_to_unpack_a).size))

            if dgm_version == 0:
                if return_format:
                    return format_to_unpack_a
                return fields_a
            else:  # dgm_version == 1 or 2 or 3:
                format_to_unpack_b = "3f"
                if return_format:
                    return format_to_unpack_a + format_to_unpack_b
                fields_b = struct.unpack(format_to_unpack_b, file_io.read(struct.Struct(format_to_unpack_b).size))
                return fields_a + fields_b
        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMRZ_rxInfo(file_io, dgm_version, return_format=False):
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
        if dgm_version == 0 or 1 or 2 or 3:
            format_to_unpack = "4H4f4H"
            if return_format:
                return format_to_unpack
            return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))
        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMRZ_extraDetClassInfo(file_io, dgm_version, return_format=False):
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
        if dgm_version == 0 or 1 or 2 or 3:
            format_to_unpack = "1H1b1B"
            if return_format:
                return format_to_unpack
            return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))
        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    @staticmethod
    def read_EMdgmMRZ_sounding(file_io, dgm_version, return_format=False):
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
        if dgm_version == 0 or 1 or 2 or 3:
            format_to_unpack = "1H8B1H6f2H18f4H"
            if return_format:
                return format_to_unpack
            return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))
        else:
            logger.warning("Datagram #MWC version {} unsupported.".format(dgm_version))
            sys.exit(1)

    # ##### ----- METHODS FOR READING MWC DATAGRAMS ----- ##### #

    def read_EMdgmMWC_txInfo(file_io, dgm_version, return_format=False):
        """
        Read #MWC - data block 1: transmit sectors, general info for all sectors.
        :return: A list containing EMdgmMWCtxInfo fields:
            MWC dgmVersion 0: [0] = numBytesTxInfo; [1] = numTxSectors; [2] = numBytesPerTxSector;
                [3] = padding; [4] = heave_m.
            MWC dgmVersion 1 (REV G): (See dgmVersion 0.)
            MWC dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version == 0 or 1 or 2:
            format_to_unpack = "3H1h1f"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        if return_format:
            return format_to_unpack

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMWC_txSectorData(file_io, dgm_version, return_format=False):
        """
        Read #MWC - data block 1: transmit sector data, loop for all i = numTxSectors.
        :return: A list containing EMdgmMWCtxSectorData fields:
            MWC dgmVersion 0: [0] = tiltAngleReTx_deg; [1] = centreFreq_Hz; [2] = txBeamWidthAlong_deg;
                [3] = txSectorNum; [4] = padding.
            MWC dgmVersion 1 (REV G): (See dgmVersion 0.)
            MWC dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version == 0 or 1 or 2:
            format_to_unpack = "3f1H1h"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        if return_format:
            return format_to_unpack

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMWC_rxInfo(file_io, dgm_version, return_format=False):
        """
        Read #MWC - data block 2: receiver, general info.
        :return: A list containing EMdgmMWCrxInfo fields:
            MWC dgmVersion 0: [0] = numBytesRxInfo; [1] = numBeams; [2] = numBytesPerBeamEntry; [3] = phaseFlag;
                [4] = TVGfunctionApplied; [5] = TVGoffset_dB; [6] = sampleFreq_Hz; [7] = soundVelocity_mPerSec.
            MWC dgmVersion 1 (REV G): (See dgmVersion 0.)
            MWC dgmVersion 2 (REV I): (See dgmVersion 0.)
        """

        if dgm_version == 0 or 1 or 2:
            format_to_unpack = "2H3B1b2f"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        if return_format:
            return format_to_unpack

        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))

    def read_EMdgmMWC_rxBeamData(file_io, dgm_version, phase_flag, return_format=False):
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
            format_to_unpack = "1f4H"
        elif dgm_version == 1 or 2:
            format_to_unpack = "1f4H1f"
        else:
            logger.warning("Datagram version {} unsupported.".format(dgm_version))
            sys.exit(1)

        fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))
        fields_list = list(fields)

        format_to_return = format_to_unpack + str(fields[4]) + "b"

        # Pointer to start of array with Water Column data. Length of array = numSampleData.
        # Sample amplitudes in 0.5 dB resolution. Size of array is numSampleData * int8_t.
        # Amplitude array is followed by phase information if phaseFlag >0.
        # Use (numSampleData * int8_t) to jump to next beam, or to start of phase info for this beam, if phase flag > 0.
        format_to_unpack = str(fields[4]) + "b"
        sample_amplitude_fields = struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))
        sample_amplitude_fields_list = list(sample_amplitude_fields)
        fields_list.append(sample_amplitude_fields_list)

        if phase_flag == 0:
            pass
        # TODO: phase_flag > 0 untested!
        elif phase_flag == 1:
            # format_to_unpack = str(num_beams * fields[4]) + "b"
            format_to_unpack = str(fields[4]) + "b"
            format_to_return += format_to_unpack
            fields_list.append(struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size)))
        elif phase_flag == 2:
            # format_to_unpack = str(num_beams * fields[4]) + "h"
            format_to_unpack = str(fields[4]) + "h"
            format_to_return += format_to_unpack
            fields_list.append(struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size)))
        else:
            logger.warning("Invalid phase_flag value {}.".format(phase_flag))
            sys.exit(1)

        if return_format:
            return format_to_return

        return fields_list

    def read_format(file_io, format_to_unpack):
        return struct.unpack(format_to_unpack, file_io.read(struct.Struct(format_to_unpack).size))