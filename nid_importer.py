from NSFopen.read import read as afmreader
from tkinter import Tk, filedialog
import numpy as np
from nanosurf.lib.util import nhf_reader, gwy_export, fileutil




class nid_importer:

    #EXTRACT AFM Z SIGNAL FROM FILE
    def open_Z_data(self, Z_file):
        signal_to_plot = 'Z-Axis'
        scan_direction = 'Backward'
        scan_to_plot = 'Image'

        Z_data = afmreader(Z_file, verbose=False).data[scan_to_plot][scan_direction][signal_to_plot]
        Z_data = np.flipud(np.array(Z_data))
        Z_data_shape = np.shape(Z_data)
        return(Z_data,Z_data_shape)

    
    #EXTRACT AFM AMPLITUDE SIGNAL FROM FILE
    def open_AMP_data(self, AMP_file):
        signal_to_plot = 'Amplitude'
        scan_direction = 'Backward'
        scan_to_plot = 'Image'

        AMP_data = afmreader(AMP_file, verbose=False).data[scan_to_plot][scan_direction][signal_to_plot]
        AMP_data = np.flipud(np.array(AMP_data))
        AMP_data_shape = np.shape(AMP_data)
        return(AMP_data,AMP_data_shape)


    #EXTRACT AFM PHASE SIGNAL FROM FILE
    def open_PHI_data(self, PHI_file):
        signal_to_plot = 'Phase'
        scan_direction = 'Backward'
        scan_to_plot = 'Image'

        PHI_data = afmreader(PHI_file, verbose=False).data[scan_to_plot][scan_direction][signal_to_plot]
        PHI_data = np.flipud(np.array(PHI_data))
        PHI_data_shape = np.shape(PHI_data)
        return(PHI_data,PHI_data_shape)