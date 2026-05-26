import numpy as np
import os

ref_file_name = 'reference file.png' 
ref_file_data = []
ref_file_path = None
original_compfiles_list = []
compfiles_list = []
num_compfile_list = []
num_lines_px = 0
num_points_px = 0

###### FOR .NID FILES #######
signal_type = None
fit_type = None
ref_file_flag = False
img_sizeX = 0
img_sizeY = 0 
num_lines = 0
num_points = 0
 
###### .PNG parameter #######


##### RESULT STORAGE VARIABLEs ########
px_rel_drift_results = []
um_rel_drift_results = []
px_abs_drift_results = []
um_abs_drift_results = []


##### DIRECTORIES ######
crop_dir = os.getcwd()+'//cropped png'
frames_dir = os.getcwd()+'//png frames'
