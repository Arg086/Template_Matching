from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from scipy.fft import fft, fft2, fftshift, ifft2
import matplotlib.patches as patches
from tkinter import Tk, filedialog
import glob
from PIL import Image
# import globalvar
import os
import pandas as pd
from NSFopen.read import read as afmreader




class files_handler:
    def __init__(self, *args, **kwargs):
        pass

    
    # MODIFY THE AMPLITUDE CHANNEL TO SOLVE THE ZERO AMPLITUDE ISSUE
    def rename_empty_amplitude_channels(self, file):
        empty_amplitude_forward = "[DataSet-0:2]".encode()
        empty_amplitude_backward = "[DataSet-1:2]".encode()
        expected_amplitude_name = "Dim2Name=Amplitude".encode()
        new_amplitude_name = "Dim2Name=AmplitudeEmpty".encode()    
        
        with open(file, 'rb') as f:
            data = f.read()
        pos_empty_amplitude_forward = data.find(empty_amplitude_forward)
        pos_empty_amplitude_backward = data.find(empty_amplitude_backward)

        if pos_empty_amplitude_forward != -1:
            data = data[:pos_empty_amplitude_forward] + data[pos_empty_amplitude_forward:].replace(expected_amplitude_name, new_amplitude_name, 1)
        if pos_empty_amplitude_backward != -1:
            data = data[:pos_empty_amplitude_backward] + data[pos_empty_amplitude_backward:].replace(expected_amplitude_name, new_amplitude_name, 1)

        with open(file, 'wb') as f:
            f.write(data)

    
    #SELECT FILES TO CROSS CORRELATE
    def file_select(self, filenum):
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        if filenum == 'single':
            open_file = filedialog.askopenfilename()
            print("Reference File: ", open_file)
        elif filenum == 'multiple':
            open_file = filedialog.askopenfilenames()
            print("Comparison Files:\n")
            for file in open_file:
                print(file, "\n")
        else:
            print('Error: wrong request submitted!')
        num_files = np.shape(open_file)
        return(open_file, num_files)


    
    #SEARCH IN THE .INI FILE THE STRING WITH SCANSIZE AND REPORT THE VALUE IN µm
    def scan_size(self, file):
        searchline = "Image size=".encode()   
        
        with open(file, 'rb') as f:
            data = f.read()
        idx = data.index(searchline)
        crop = data[idx:idx+20]
        crop = str(crop).encode()
        start = crop.find('='.encode())
        end = crop.find('\\'.encode())
        scanrange = int(crop[start+1:end])*1e-6
        return(scanrange)



    #SEARCH IN THE .INI FILE/S THE STRING WITH SCAN DIRECTION AND FILTER THE MAPS
    def scan_direction(self, open_file, ref_map, ref_type=None):
        searchline = "Scan direction=".encode()
        new_list = []
        b = []
        
        for file in (open_file):
            with open(file, 'rb') as f:
                data = f.read()
            idx = data.index(searchline)
            crop = data[idx:idx+17]
            crop = str(crop).encode()
            start = crop.find('='.encode())
            end = crop.find('\\'.encode())
            
            if ref_map == True:
                if crop[start+1:end] == b'Up':
                    return b'Up'
                elif crop[start+1:end] == b'Do':
                    return b'Do'
                else:
                    print('The info regarding scan direction is not provided by the .ini file')
            else:
                if crop[start+1:end] == ref_type:
                    new_list.append(file)
                else:
                    pass
        return new_list
    


    #SAVE THE DATA AS .PNG FILE
    def from_data_to_png(self, data, filename, folder_name=None):
        try:
            png_path = os.getcwd()+folder_name+'\\'+filename
        except:
            png_path = filename
        fig1 = plt.figure(frameon=False)
        fig1.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(np.float64(np.asarray(data)))
        plt.savefig(png_path, bbox_inches='tight', pad_inches=000)
        mapSize = Image.open(png_path).size
        plt.close()
        # globalvar.cc_px_size = mapSize
        return mapSize
    

    #SAVE THE PNG FILE WITH DIFFERENT NAME  
    def rename_png(self,file_path,newname):
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print('Error: could not load image!')
        else:
            cv2.imwrite(newname, img)
        return 0
    

    def is_a_png(self,file):
        return file.lower().endswith('.png')



    #SAVE THE DATA AS .PNG FILE FOR USING AS FRAME OF TARGET TRAKING
    def from_data_to_frame(self, data, filename, path, shiftX=None, shiftY=None, deltaX=None, deltaY=None):
        png_path = path+'\\'+filename
        fig2 = plt.figure(frameon=False)
        ax = plt.subplot(1,1,1)
        ax.axis('off')
        ax.imshow(np.float64(np.asarray(data)))
        try:
            ax.add_patch(patches.Rectangle((shiftX, shiftY), deltaX, deltaY, linewidth=2, edgecolor='r', facecolor='none'))
        except:
            pass
        fig2.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return png_path



    #GENERATE GIF FROM PNG FRAMES
    def gif_generator(self, png_path):
        images = []
        for filename in sorted(glob.glob(png_path+'\*.png')):
            im = Image.open(filename)
            im_rescale = im.resize((1000,1000), resample=0)
            images.append(im_rescale)
        # last_frame = len(images)
        # for x in range(0,9):
        #     im = images[last_frame-1]
        #     images.append(im)
        images[0].save(png_path+'tracking_gif.gif', save_all=True, append_images=images[1:], optimize=False, duration=250, loop=0)

    

    #CROP THE ORIGINAL MATRIX AS .PNG FORMAT IN A SMALLER PORTION
    # WITH THE INTERESTING FEATURE INSIDE
    def crop_png(self, filename):
        img_to_crop = cv2.imread(filename)
        roi = cv2.selectROI("ROI selector", img_to_crop)    #n.b. roi = [x_topleft, y toplevt, Width, Height]
        img_cropped = img_to_crop[int(roi[1]):int(roi[1]+roi[3]) , int(roi[0]):int(roi[0]+roi[2])] 
        cv2.imwrite('cropped target.png', img_cropped)
        croppedSize = Image.open('cropped target.png').size
        cv2.destroyWindow("ROI selector")
        cv2.waitKey(1)
        return(roi, croppedSize)
    


    #MAKE PROPORTIONS BETWEEN NUMBER OF PIXELS AND ORIGINAL MATRIX SIZE TO OBTAIN 
    #THE COORDINATE AND DIMENSION OF CROPPED AREA
    def px_to_idx(self, coordinates_crop, size_fig, size_matrix):
        px_to_idx_X = np.int64(coordinates_crop[0]/size_fig[0]*size_matrix[0])
        px_to_idx_dX = np.int64(coordinates_crop[2]/size_fig[0]*size_matrix[0])
        px_to_idx_Y = np.int64(coordinates_crop[1]/size_fig[0]*size_matrix[0])
        px_to_idx_dY = np.int64(coordinates_crop[3]/size_fig[0]*size_matrix[0])
        idx_X_dX = [px_to_idx_X, px_to_idx_dX]
        idx_Y_dY = [px_to_idx_Y, px_to_idx_dY]
        return(idx_X_dX, idx_Y_dY)
    


    #CREATE FOLDER IN THE WORKING DIRECTORY
    def dir_generation(self, dir_name, parent_dir):
        path = os.path.join(parent_dir,dir_name)
        if  os.path.exists(path) == True:
            pass
        else:
            os.mkdir(path)
        return path



    #ELIMINATE ALL PNG IN THE PNG FOLDER
    def dir_empty(self, png_path):
        list_file = os.listdir(png_path)
        for images in list_file:
            if images.endswith(".png"):
                os.remove(os.path.join(png_path, images))