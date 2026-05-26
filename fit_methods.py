import numpy as np
from numpy.fft import fft,fft2,fftshift,ifft
import scipy as spy

class fit_fun():

    # LINEAR FIT
    def line_fit(self,data,col):
        try:
            size_data = np.shape(data)
            filtered_data_row = []
            for idx, line in enumerate(data):
                x = np.linspace(0,size_data[0],size_data[1])
                lincoeff_A,lincoeff_B = np.polyfit(x,np.array(line,dtype = float),1)
                fitline = lincoeff_A*x + lincoeff_B
                filterrow = line - fitline
                filtered_data_row.append(filterrow)

            if col:
                data_column = np.transpose(filtered_data_row)
                filtered_data_col = []
                for idx, line in enumerate(data_column):
                    x = np.linspace(0,size_data[0],size_data[1])
                    lincoeff_A,lincoeff_B = np.polyfit(x,np.array(line,dtype = float),1)
                    fitline = lincoeff_A*x + lincoeff_B
                    filtercolumn = line - fitline
                    filtered_data_col.append(filtercolumn)
                
                filtered_data = np.transpose(filtered_data_col)
            else:
                filtered_data = filtered_data_row
            
            return filtered_data 
        except:
            print("Line Fit did not work!")
    
    
    # QUADRATIC FIT
    def quadratic_fit(self,data,col):
        size_data = np.shape(data)
        filtered_data_row = []
        for idx, line in enumerate(data):
            x = np.linspace(0,size_data[0],size_data[1])
            lincoeff_A,lincoeff_B,lincoeff_C = np.polyfit(x,np.array(line,dtype = float),2)
            fitline = lincoeff_A*x**2 + lincoeff_B*x + lincoeff_C
            filterrow = line - fitline
            filtered_data_row.append(filterrow)

        if col:
            data_column = np.transpose(filtered_data_row)
            filtered_data_col = []
            for idx, row in enumerate(data_column):
                x = np.linspace(0,size_data[0],size_data[1])
                lincoeff_A,lincoeff_B,lincoeff_C = np.polyfit(x,np.array(line,dtype = float),2)
                fitline = lincoeff_A*x**2 + lincoeff_B*x + lincoeff_C
                filtercolumn = line - fitline
                filtered_data_col.append(filtercolumn)
            
            filtered_data = np.transpose(filtered_data_col)
        else:
            filtered_data = filtered_data_row
        
        return filtered_data 
    
    

    def filtering_fun(self, data, filtertype, parameter=None):
        # Explicitly check for None so that 0.0 is allowed to pass through
        if parameter is None:
            parameter = 1
            
        try:
            match filtertype:
                case 'median':
                    # A median filter size must be an odd integer >= 1
                    size_param = int(parameter)
                    if size_param < 1:
                        raise ValueError("Median filter size must be at least 1")
                    newdata = spy.ndimage.median_filter(data, size=size_param)
                case 'gaussian':
                    newdata = spy.ndimage.gaussian_filter(data, sigma=parameter)
                case _:
                    return data 
            return newdata
            
        except Exception as e:
            print(f"Error: Not able to apply filters! Details: {e}")
            return data  