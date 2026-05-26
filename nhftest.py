from nanosurf.lib.util import nhf_reader#, gwy_export, nid_reader
import numpy as np
from matplotlib import pyplot as plt

source_file = 'test.nhf'
verbose = False

# create file-reader
nhf_file = nhf_reader.NHFFileReader(verbose)
if nhf_file.read(source_file) == False:
    print("Could not read file")
else:
    print(f"Found {nhf_file.measurement_count()} measurements in file:")

# Choose measurement - there is only one - at "0" index
measurement_name = nhf_file.measurement_name(0)
measurement = nhf_file.measurement[measurement_name]

# Choose segment and read channels
segment_name = 'Forward'
segment = measurement.segment[segment_name]
ch_topography = segment.read_channel('Topography')
ch_deflection = segment.read_channel('Deflection')

NHF2 = True # NHF version

# read the size and number of points and lines
if not NHF2:
    image_size_x_m = measurement.attribute['image_size_x']
    image_size_y_m = measurement.attribute['image_size_y']
    image_points_per_line = measurement.attribute['image_points_per_line']
    image_number_of_lines  = measurement.attribute['image_number_of_lines']
else:
    image_size_x_m = measurement.attribute['rect_axis_range'][0]
    image_size_y_m = measurement.attribute['rect_axis_range'][1]
    image_points_per_line = segment.attribute['rect_axis_size'][0]
    image_number_of_lines  = segment.attribute['rect_axis_size'][1]

# Reshape the data from 1D array to 2D array according to the number of points and lines
ch_topography.dataset = np.reshape(np.array(ch_topography.dataset), (image_number_of_lines, image_points_per_line))
ch_deflection.dataset = np.reshape(np.array(ch_deflection.dataset), (image_number_of_lines, image_points_per_line))




