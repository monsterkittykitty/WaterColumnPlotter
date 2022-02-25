# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description: A 'standard' format to contain necessary water column data for plotting functions;
# meant to standardize data from any sonar system for use with Water Column Plotter.

class PieStandardFormat:
    def __init__(self, bin_size, max_heave, pie_chart_amplitudes,
                 pie_chart_counts, timestamp, latitude=None, longitude=None):

        self.bin_size = bin_size
        self.max_heave = max_heave

        self.pie_chart_amplitudes = pie_chart_amplitudes  # Numpy matrix containing sums of amplitudes in each bin
        self.pie_chart_counts = pie_chart_counts  # Numpy matrix containing count of values in each bin
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
