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

        drift_list_px =[]
        for idx,file_name in enumerate(glb.compfiles_list):

            comp_image = cv2.imread(glb.frames_dir+'\\'+file_name, cv2.IMREAD_GRAYSCALE)

            result = cv2.matchTemplate(comp_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            drift_x = top_left[0] - xi
            drift_y = top_left[1] - yi

            comp_image_marked = cv2.cvtColor(comp_image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(comp_image_marked, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imwrite(glb.crop_dir + '\\' + file_name,comp_image_marked)

            drift_list_px.append([drift_x,drift_y])
        
        drift_list_um = []
        # print(glb.num_points_px,glb.img_sizeX,glb.num_lines_px,glb.img_sizeY)
        for idx,driftval in enumerate(drift_list_px): 
            drift_list_um.append([driftval[0]/glb.num_points_px*glb.img_sizeX,
                                 driftval[1]/glb.num_lines_px*glb.img_sizeY])
        
        for idx, coord in enumerate(drift_list_px):
            print('Cmparison File ' + str(idx) + ' absolute pixel drift: ' + str(coord))
        
        for idx, coord in enumerate(drift_list_um):
            print('Cmparison File ' + str(idx) + ' absolute µm drift: ' + str(coord))
        

        with open('drift_µm.txt','w') as file:
            for idx,line in enumerate(drift_list_um):
                file.write(str(drift_list_um[idx])+'\n')

        return drift_list_um
