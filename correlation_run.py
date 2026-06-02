import cv2
import numpy as np
import matplotlib.pyplot as plt
from files_handler import files_handler
from nid_importer import nid_importer
import global_var as glb
import os

flh = files_handler()
nidimp = nid_importer()

class worker_template_matching:

    def run_template_matching(self):

        flh.dir_empty(glb.crop_dir)
        
        ### SELECT TARGET TO CROP OUT FROM REFERENCE IMAGE ###
        [xi,yi,w,h], crop_size = flh.crop_png('reference file.png')
        print('Coordinates target crop top-left corner: ' + str([xi,yi]))
        print("Width and height of crop portion: " + str([w,h]))
        print('Pixels size target crop: '+ str(crop_size))

        ### CONVERT THE REFERENCE .PNG IMAGE IN GRAYSCALE ###
        ref_image = cv2.imread('reference file.png', cv2.IMREAD_GRAYSCALE)
        template = ref_image[yi:yi+h, xi:xi+w]

        ### CHECK SIZE OF THE LIST OF COMPARISON FILES ###
        num_comp = np.shape(glb.compfiles_list)[0]
        print('Number of file in the list to compare: ' + str(num_comp) + '\n')

        drift_list_px = []
        for idx, file_name in enumerate(glb.compfiles_list):

            comp_image = cv2.imread(glb.frames_dir + '\\' + file_name, cv2.IMREAD_GRAYSCALE)

            result = cv2.matchTemplate(comp_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Finds the integer which are the coordinates of the peak
            max_x, max_y = max_loc
            
            ########## Sub-pixel Interpolation (last update for small drifts, e.g. thermal) ########################################
            # Default subpixel shift is 0 unless is possible to interpolate
            sub_x, sub_y = float(max_x), float(max_y)
            
            # Makes sure that the peak is not on the extreme edge of the resulting map to avoid indexing errors (p.s. check if gives error or set to 0 when I have time)
            if 0 < max_x < result.shape[1] - 1 and 0 < max_y < result.shape[0] - 1:
                # Gets a matrix 3x3 around the peak point p.s. max and min ref to peak
                alpha = result[max_y, max_x - 1] # Left
                beta  = result[max_y, max_x]     # Center (max_val)
                gamma = result[max_y, max_x + 1] # Right
                
                # Parabolich fit (1D) this is for X
                denom_x = (alpha - 2 * beta + gamma)
                if denom_x != 0:
                    dx = 0.5 * (alpha - gamma) / denom_x
                    sub_x += dx

                alpha_y = result[max_y - 1, max_x] # Top
                gamma_y = result[max_y + 1, max_x] # Bottom
                
                # Parabolich fit (1D) this is for Y
                denom_y = (alpha_y - 2 * beta + gamma_y)
                if denom_y != 0:
                    dy = 0.5 * (alpha_y - gamma_y) / denom_y
                    sub_y += dy
            ####################################################################################

            drift_x = sub_x - xi
            drift_y = sub_y - yi

            # The rectangle in the pic is still using the integer position so may be slightly offsetted respect the results for very small drifts
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            comp_image_marked = cv2.cvtColor(comp_image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(comp_image_marked, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imwrite(glb.crop_dir + '\\' + file_name, comp_image_marked)

            drift_list_px.append([drift_x, drift_y])
        
        drift_list_um = []
        for idx, driftval in enumerate(drift_list_px): 
            drift_list_um.append([driftval[0] / glb.num_points_px * glb.img_sizeX,
                                  driftval[1] / glb.num_lines_px * glb.img_sizeY])
        
        for idx, coord in enumerate(drift_list_px):
            print(f'Comparison File {idx} absolute pixel drift: [{coord[0]:.3f}, {coord[1]:.3f}]')
        
        for idx, coord in enumerate(drift_list_um):
            print(f'Comparison File {idx} absolute µm drift: [{coord[0]:.4f}, {coord[1]:.4f}]')
        
        with open('drift_µm.txt', 'w') as file:
            for idx, line in enumerate(drift_list_um):
                file.write(f'[{line[0]:.4f}, {line[1]:.4f}]\n')

        return drift_list_um