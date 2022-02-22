# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

class KongsbergDGPie:
    def __init__(self, bin_size, max_heave, pie_chart_values,
                 pie_chart_count, timestamp, latitude=None, longitude=None):

        self.bin_size = bin_size
        self.max_heave = max_heave

        self.pie_chart_values = pie_chart_values  # Numpy matrix containing sums of values in each bin
        self.pie_chart_count = pie_chart_count  # Numpy matrix containing count of values in each bin
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
