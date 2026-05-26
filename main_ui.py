from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform,QDoubleValidator)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QSpacerItem, QStatusBar, QTextBrowser, QVBoxLayout,
    QWidget)
import numpy as np
import global_var as glb
from nid_importer import nid_importer
from files_handler import files_handler
from correlation_run import worker_template_matching
from fit_methods import fit_fun
import os
import cv2
from drift_plotter_window import DriftPlotterWindow
import imageio.v3 as iio
from gif_viewer import GifViewerWindow

nidimp = nid_importer()
flh = files_handler()
workertm = worker_template_matching()
ffun = fit_fun()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(694, 595)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.parameters_frm = QFrame(self.centralwidget)
        self.parameters_frm.setObjectName(u"parameters_frm")
        self.parameters_frm.setFrameShape(QFrame.StyledPanel)
        self.parameters_frm.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.parameters_frm)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.fileselect_frm = QFrame(self.parameters_frm)
        self.fileselect_frm.setObjectName(u"fileselect_frm")
        self.fileselect_frm.setFrameShape(QFrame.StyledPanel)
        self.fileselect_frm.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.fileselect_frm)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.reffileselect_btn = QPushButton(self.fileselect_frm)
        self.reffileselect_btn.setObjectName(u"reffileselect_btn")
        font = QFont()
        font.setBold(True)
        self.reffileselect_btn.setFont(font)

        self.verticalLayout_3.addWidget(self.reffileselect_btn)

        self.compfileselect_btn = QPushButton(self.fileselect_frm)
        self.compfileselect_btn.setObjectName(u"compfileselect_btn")
        self.compfileselect_btn.setFont(font)

        self.verticalLayout_3.addWidget(self.compfileselect_btn)

        self.runcalc_btn = QPushButton(self.fileselect_frm)
        self.runcalc_btn.setObjectName(u"calculationrun_btn")
        self.runcalc_btn.setFont(font)

        self.verticalLayout_3.addWidget(self.runcalc_btn)


        #  Create GIF Generation & Playback Trigger Button
        self.playgif_btn = QPushButton(self.fileselect_frm)
        self.playgif_btn.setObjectName(u"playgif_btn")
        self.playgif_btn.setFont(font)
        self.playgif_btn.setText("Generate & View GIF")  # Default placeholder text
        self.verticalLayout_3.addWidget(self.playgif_btn)
        # --- exception end ---


        self.verticalLayout.addWidget(self.fileselect_frm)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_4)

        self.titlenidparam_lbl = QLabel(self.parameters_frm)
        self.titlenidparam_lbl.setObjectName(u"titlenidparam_lbl")

        self.verticalLayout.addWidget(self.titlenidparam_lbl)

        self.nidparam_frm = QFrame(self.parameters_frm)
        self.nidparam_frm.setObjectName(u"nidparam_frm")
        self.nidparam_frm.setFrameShape(QFrame.StyledPanel)
        self.nidparam_frm.setFrameShadow(QFrame.Raised)
        self.gridLayout = QGridLayout(self.nidparam_frm)
        self.gridLayout.setObjectName(u"gridLayout")
        self.directionfilt_chkbx = QCheckBox(self.nidparam_frm)
        self.directionfilt_chkbx.setObjectName(u"directionfilt_chkbx")

        self.gridLayout.addWidget(self.directionfilt_chkbx, 5, 0, 1, 1, Qt.AlignLeft)

        self.signtype_lbl = QLabel(self.nidparam_frm)
        self.signtype_lbl.setObjectName(u"signtype_lbl")

        self.gridLayout.addWidget(self.signtype_lbl, 4, 0, 1, 1)

        self.signl_cmbox = QComboBox(self.nidparam_frm)
        self.signl_cmbox.addItem("")
        self.signl_cmbox.addItem("")
        self.signl_cmbox.addItem("")
        self.signl_cmbox.setObjectName(u"signl_cmbox")

        self.gridLayout.addWidget(self.signl_cmbox, 4, 1, 1, 1)


        self.verticalLayout.addWidget(self.nidparam_frm)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_3)

        self.label = QLabel(self.parameters_frm)
        self.label.setObjectName(u"label")

        self.verticalLayout.addWidget(self.label)

        self.pngparam_frm = QFrame(self.parameters_frm)
        self.pngparam_frm.setObjectName(u"pngparam_frm")
        self.pngparam_frm.setFrameShape(QFrame.StyledPanel)
        self.pngparam_frm.setFrameShadow(QFrame.Raised)
        self.gridLayout_3 = QGridLayout(self.pngparam_frm)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.numpoints_lbl = QLabel(self.pngparam_frm)
        self.numpoints_lbl.setObjectName(u"numpoints_lbl")

        self.gridLayout_3.addWidget(self.numpoints_lbl, 0, 0, 1, 1)

        self.imgsizeX_edit = QLineEdit(self.pngparam_frm)
        self.imgsizeX_edit.setObjectName(u"img_sizeX_edit")
        self.imgsizeX_edit.setMaximumSize(QSize(50, 16777215))

        self.gridLayout_3.addWidget(self.imgsizeX_edit, 0, 1, 1, 1)

        self.numlines_lbl = QLabel(self.pngparam_frm)
        self.numlines_lbl.setObjectName(u"numlines_lbl")

        self.gridLayout_3.addWidget(self.numlines_lbl, 1, 0, 1, 1)

        self.imgsizeY_edit = QLineEdit(self.pngparam_frm)
        self.imgsizeY_edit.setObjectName(u"img_sizeY_edit")
        self.imgsizeY_edit.setMaximumSize(QSize(50, 16777215))

        self.gridLayout_3.addWidget(self.imgsizeY_edit, 1, 1, 1, 1)


        self.verticalLayout.addWidget(self.pngparam_frm, 0, Qt.AlignLeft)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_5)

        self.titleimgproc_lbl = QLabel(self.parameters_frm)
        self.titleimgproc_lbl.setObjectName(u"titleimgproc_lbl")

        self.verticalLayout.addWidget(self.titleimgproc_lbl)

        self.imgprocess_frm = QFrame(self.parameters_frm)
        self.imgprocess_frm.setObjectName(u"imgprocess_frm")
        self.imgprocess_frm.setFrameShape(QFrame.StyledPanel)
        self.imgprocess_frm.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.imgprocess_frm)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gaussian_chkbx = QCheckBox(self.imgprocess_frm)
        self.gaussian_chkbx.setObjectName(u"gaussian_chkbx")
        self.gaussian_chkbx.setLayoutDirection(Qt.LeftToRight)

        self.gridLayout_2.addWidget(self.gaussian_chkbx, 2, 1, 1, 1)

        self.median_lbl = QLabel(self.imgprocess_frm)
        self.median_lbl.setObjectName(u"median_lbl")

        self.gridLayout_2.addWidget(self.median_lbl, 7, 1, 1, 1)

        self.median_editlbl = QLineEdit(self.imgprocess_frm)
        self.median_editlbl.setObjectName(u"median_editlbl")
        self.median_editlbl.setMaximumSize(QSize(50, 16777215))

        self.gridLayout_2.addWidget(self.median_editlbl, 7, 2, 1, 1)

        self.median_chkbx = QCheckBox(self.imgprocess_frm)
        self.median_chkbx.setObjectName(u"median_chkbx")

        self.gridLayout_2.addWidget(self.median_chkbx, 6, 1, 1, 1)

        self.gauss_edit = QLineEdit(self.imgprocess_frm)
        self.gauss_edit.setObjectName(u"gauss_edit")
        self.gauss_edit.setMaximumSize(QSize(50, 16777215))

        self.gridLayout_2.addWidget(self.gauss_edit, 4, 2, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 1, 1, 1, 1)

        self.fittype_cmbox = QComboBox(self.imgprocess_frm)
        self.fittype_cmbox.addItem("")
        self.fittype_cmbox.addItem("")
        self.fittype_cmbox.addItem("")
        self.fittype_cmbox.setObjectName(u"fittype_cmbox")

        self.gridLayout_2.addWidget(self.fittype_cmbox, 0, 2, 1, 1)

        self.gauss_lbl = QLabel(self.imgprocess_frm)
        self.gauss_lbl.setObjectName(u"gauss_lbl")

        self.gridLayout_2.addWidget(self.gauss_lbl, 4, 1, 1, 1)

        self.fittype_lbl = QLabel(self.imgprocess_frm)
        self.fittype_lbl.setObjectName(u"fittype_lbl")

        self.gridLayout_2.addWidget(self.fittype_lbl, 0, 1, 1, 1)

        self.updatefilt_btn = QPushButton(self.imgprocess_frm)
        self.updatefilt_btn.setObjectName(u"filterupdate_btn")

        self.gridLayout_2.addWidget(self.updatefilt_btn, 8, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_2, 5, 1, 1, 1)


        self.verticalLayout.addWidget(self.imgprocess_frm, 0, Qt.AlignLeft)


        self.horizontalLayout.addWidget(self.parameters_frm, 0, Qt.AlignTop)

        self.imagelog_frm = QFrame(self.centralwidget)
        self.imagelog_frm.setObjectName(u"imagelog_frm")
        self.imagelog_frm.setFrameShape(QFrame.StyledPanel)
        self.imagelog_frm.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.imagelog_frm)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.titleimg_lbl = QLabel(self.imagelog_frm)
        self.titleimg_lbl.setObjectName(u"titleimg_lbl")
        self.titleimg_lbl.setFont(font)

        self.verticalLayout_2.addWidget(self.titleimg_lbl, 0, Qt.AlignHCenter|Qt.AlignTop)

        self.image_lbl = QLabel(self.imagelog_frm)
        self.image_lbl.setObjectName(u"image_lbl")

        self.verticalLayout_2.addWidget(self.image_lbl, 0, Qt.AlignHCenter|Qt.AlignTop)

        self.frame_6 = QFrame(self.imagelog_frm)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_6)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.logtitle_lbl = QLabel(self.frame_6)
        self.logtitle_lbl.setObjectName(u"logtitle_lbl")
        self.logtitle_lbl.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_4.addWidget(self.logtitle_lbl)

        self.log_txtbrs = QTextBrowser(self.frame_6)
        self.log_txtbrs.setObjectName(u"log_txtbrs")
        self.log_txtbrs.setMaximumSize(QSize(16777215, 100))

        self.verticalLayout_4.addWidget(self.log_txtbrs)


        self.verticalLayout_2.addWidget(self.frame_6, 0, Qt.AlignBottom)


        self.horizontalLayout.addWidget(self.imagelog_frm)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 694, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

        ###############################################################################################
        ### VALIDATOR/S ####
        self.double_validator = QDoubleValidator(0,9999.99, 2)
        self.imgsizeY_edit.setValidator(self.double_validator)
        self.imgsizeX_edit.setValidator(self.double_validator)

        self.update_viewer()
        self.reffileselect_btn.clicked.connect(lambda:self.file_opening('reference_file'))
        self.compfileselect_btn.clicked.connect(lambda:self.file_opening('comparison_list'))
        self.runcalc_btn.clicked.connect(self.run_template_matching_process)
        self.signl_cmbox.activated.connect(self.on_signaltype_change)
        self.fittype_cmbox.activated.connect(self.update_ref_png)
        self.imgsizeY_edit.textChanged.connect(lambda:self.on_editing_linedit_double_numbers(self.imgsizeY_edit.text(),'height'))
        self.imgsizeX_edit.textChanged.connect(lambda:self.on_editing_linedit_double_numbers(self.imgsizeX_edit.text(),'width'))
        self.updatefilt_btn.clicked.connect(self.update_ref_png)
        self.playgif_btn.clicked.connect(self.generate_and_play_gif)
        

    def update_ref_png(self):
        glb.ref_file_data = self.nid_data_extraction(glb.ref_file_path) # extract the signal data accordingly to the signal type     
        glb.fit_type = self.fittype_cmbox.currentText()
        match glb.fit_type:
            case 'None':
                data = glb.ref_file_data
                # print("No Fit")
            case 'Linear':
                data = ffun.line_fit(glb.ref_file_data, False)
                # print("Linear")
            case 'Quadratic':
                data = ffun.quadratic_fit(glb.ref_file_data, False)
                # print("Quadratic")
            case _:  # Default fallback if fit_type is unexpected
                data = glb.ref_file_data
                print("Fallback Reference: Using raw data due to unknown fit_type")

        if self.gaussian_chkbx.isChecked():
            # Fallback to 0.0 if the line edit is completely empty
            g_text = self.gauss_edit.text()
            g_param = np.float64(g_text) if g_text.strip() else np.float64(0.0)
            
            data = ffun.filtering_fun(data, 'gaussian', g_param)
            print("Gaussian Filter Applied!")

        if self.median_chkbx.isChecked():
            m_text = self.median_editlbl.text()
            m_param = np.float64(m_text) if m_text.strip() else np.float64(1.0)
            
            data = ffun.filtering_fun(data, 'median', m_param)
            print("Median Filter Applied!")

        flh.from_data_to_png(data, glb.ref_file_name) # create the .png file from the amf data (nid)
        self.update_viewer()
        return  None
    

    def run_template_matching_process(self):
        self.update_comparison_file_list()
        print("Comparison Files processes as Reference File!")
        drift_results = workertm.run_template_matching()
        if drift_results and len(drift_results) > 0:
            self.plot_viewer = DriftPlotterWindow(drift_results)
            self.plot_viewer.show()
        else:
            self.log_txtbrs.append("Calculation Warning: No valid drift data generated to plot.")
        return None



    def update_comparison_file_list(self):
        flh.dir_empty(glb.frames_dir) # erase the previous selection to host the new selection
        for idx,file in enumerate(glb.original_compfiles_list):
            filename = 'comparison file '+str(idx)+'.png'
            if not flh.is_a_png(file):
                flh.rename_empty_amplitude_channels(file)
                data = self.nid_data_extraction(file)
                match glb.fit_type:
                    case 'None':
                        pass
                    case 'Linear':
                        data = ffun.line_fit(data,False)
                    case 'Quadratic':
                        data = ffun.quadratic_fit(data,False)
                    case _:  # Default fallback if fit_type is unexpected
                        data = glb.ref_file_data
                        print("Fallback Comparison: Using raw data due to unknown fit_type")

                if self.gaussian_chkbx.isChecked():
                    # Fallback to 0.0 if the line edit is completely empty
                    g_text = self.gauss_edit.text()
                    g_param = np.float64(g_text) if g_text.strip() else np.float64(0.0)
                    
                    data = ffun.filtering_fun(data, 'gaussian', g_param)
                    # print("Gaussian Filter Applied!")

                if self.median_chkbx.isChecked():
                    m_text = self.median_editlbl.text()
                    m_param = np.float64(m_text) if m_text.strip() else np.float64(1.0)
                    
                    data = ffun.filtering_fun(data, 'median', m_param)
                    # print("Median Filter Applied!")

                flh.from_data_to_png(data,filename,'\\png frames')
                comp_image = cv2.imread(os.getcwd()+'\\png frames\\'+filename)
                [glb.num_lines_px,glb.num_points_px] = comp_image.shape[:2]                        
            else:
                comp_image = cv2.imread(file)
                cv2.imwrite(glb.frames_dir+'\\'+filename,comp_image)
                [glb.num_lines_px,glb.num_points_px] = comp_image.shape[:2]
            print('Comparison Files Updated!')
        return None




    def file_opening(self,type):
        match type:
            case 'reference_file':
                glb.ref_file_path, pxsize = flh.file_select('single')
                if not flh.is_a_png(glb.ref_file_path):
                    glb.img_sizeX = glb.img_sizeY = np.round(flh.scan_size(glb.ref_file_path)*1e6,3)
                    self.update_image_size(False)
                    glb.ref_file_flag = True    # this flag allows to dynamically change the type of signal in case user works with .nid files
                    flh.rename_empty_amplitude_channels(glb.ref_file_path)  # fix the bug in the file with amplitude signal name
                    self.update_ref_png()             
                else:
                    self.update_image_size(True)
                    flh.rename_png(glb.ref_file_path, glb.ref_file_name) #take the path of the png file and save with reference file tag .png
                    self.update_viewer()
                print(glb.ref_file_name)
            case 'comparison_list':
                glb.compfiles_list = []
                provvisory_list,glb.num_compfile_list = flh.file_select('multiple')
                glb.original_compfiles_list = provvisory_list
                flh.dir_empty(glb.frames_dir) # erase the previous selection to host the new selection
                for idx,file in enumerate(provvisory_list):
                    filename = 'comparison file '+str(idx)+'.png'
                    if not flh.is_a_png(file):
                        flh.rename_empty_amplitude_channels(file)
                        AFMdata = self.nid_data_extraction(file)
                        match glb.fit_type:
                            case 'None':
                                pass
                            case 'Linear':
                                AFMdata = ffun.line_fit(AFMdata,False)
                            case 'Quadratic':
                                AFMdata = ffun.quadratic_fit(AFMdata,False)

                        flh.from_data_to_png(AFMdata,filename,'\\png frames')
                        comp_image = cv2.imread(os.getcwd()+'\\png frames\\'+filename)
                        [glb.num_lines_px,glb.num_points_px] = comp_image.shape[:2]                        
                    else:
                        comp_image = cv2.imread(file)
                        cv2.imwrite(glb.frames_dir+'\\'+filename,comp_image)
                        [glb.num_lines_px,glb.num_points_px] = comp_image.shape[:2]
                    glb.compfiles_list.append(filename)
        return 0

                    


    def nid_data_extraction(self,data):
        match glb.signal_type:
            case 'Z':
                AFMdata,[glb.num_lines,glb.num_points] = nidimp.open_Z_data(data)
            case 'Amplitude':
                AFMdata,[glb.num_lines,glb.num_points] = nidimp.open_AMP_data(data)
            case 'Phase':
                AFMdata,[glb.num_lines,glb.num_points] = nidimp.open_PHI_data(data)
        return AFMdata



    def set_starting_template(self):
        # Create the directories to store the different type of png
        glb.frames_dir = flh.dir_generation('png frames', os.getcwd())
        glb.crop_dir = flh.dir_generation('cropped png', os.getcwd())
        # Check signal type for .nid files 
        glb.signal_type = self.signl_cmbox.currentText()
        # if the reference file png is in the folder it shows it on viewer
        self.update_viewer()
        # check if reference file exists
        if glb.ref_file_name:
            print('ref file is selected')
        else:
            pass
        return 0


    def on_signaltype_change(self):
        glb.signal_type = self.signl_cmbox.currentText()
        print(glb.signal_type)
        if glb.ref_file_flag:    
            self.update_ref_png()              
        return 0
    

    def on_editing_linedit_double_numbers(self,text,subject):
        match subject:
            case 'height': 
                glb.img_sizeY = np.float64(text)
            case 'width':
                glb.img_sizeX = np.float64(text)
        # print(glb.img_sizeX,glb.img_sizeY)
        return 0



    def update_image_size(self, state):
        if state:
            # Editable Mode: Sleek dark gray background with clear white text
            dark_editable = "background-color: #25252b; color: #ffffff; border: 1px solid #3a3a45; padding: 4px 8px; border-radius: 4px;"
            self.imgsizeY_edit.setStyleSheet(dark_editable)
            self.imgsizeY_edit.setReadOnly(False)
            self.imgsizeX_edit.setStyleSheet(dark_editable)
            self.imgsizeX_edit.setReadOnly(False)
        else:
            # Locked/Read-Only Mode: Darker matte background with muted gray text
            dark_readonly = "background-color: #1a1a1f; color: #7a7a85; border: 1px solid #2d2d35; padding: 4px 8px; border-radius: 4px;"
            self.imgsizeY_edit.setStyleSheet(dark_readonly)
            self.imgsizeY_edit.setReadOnly(True)
            self.imgsizeY_edit.setText(str(glb.img_sizeY))
            self.imgsizeX_edit.setStyleSheet(dark_readonly)
            self.imgsizeX_edit.setReadOnly(True)
            self.imgsizeX_edit.setText(str(glb.img_sizeX))
        return 0
    
   
    
    def update_viewer(self):
        self.image_ref = QPixmap('reference file.png')
        self.image_lbl.setPixmap(self.image_ref)
        return 0
    


    def generate_and_play_gif(self):
        crop_dir = glb.crop_dir if hasattr(glb, 'crop_dir') else os.path.join(os.getcwd(), 'cropped png')
        
        if not os.path.exists(crop_dir):
            self.log_txtbrs.append(f"Error: Directory '{crop_dir}' does not exist.")
            return

        png_files = [f for f in os.listdir(crop_dir) if f.lower().endswith('.png')]
        
        if not png_files:
            self.log_txtbrs.append("Animation Error: No PNG images found inside 'cropped png' directory.")
            return
            
        png_files.sort()
        
        self.log_txtbrs.append(f"Compiling {len(png_files)} frames into an animation loop...")
        QApplication.processEvents() # Refresh GUI log window

        try:
            frames = []
            for file_name in png_files:
                file_path = os.path.join(crop_dir, file_name)
                # Read image frame data array matrix
                frames.append(iio.imread(file_path))
            
            # Save the file as a compiled GIF loop inside your main workspace directory
            output_gif_path = "drift_timelapse.gif"
            
            # duration = 200 means 200 milliseconds per frame (5 frames per second)
            iio.imwrite(output_gif_path, frames, duration=600, loop=0)
            
            self.log_txtbrs.append("GIF generated successfully! Opening viewer...")
            
            # Spawn and launch the native player layout container
            self.gif_window = GifViewerWindow(output_gif_path, parent=self.centralwidget)
            self.gif_window.exec() # Spawns as modal overlay window layer
            
        except Exception as e:
            self.log_txtbrs.append(f"Processing Failure during GIF building step: {str(e)}")

    ##################################################################################################
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.reffileselect_btn.setText(QCoreApplication.translate("MainWindow", u"Select Reference File", None))
        self.compfileselect_btn.setText(QCoreApplication.translate("MainWindow", u"Select Comparison Files", None))
        self.runcalc_btn.setText(QCoreApplication.translate("MainWindow", u"Run Cross Correlation", None))
        self.playgif_btn.setText(QCoreApplication.translate("MainWindow", u"Generate & View GIF", None))
        self.titlenidparam_lbl.setText(QCoreApplication.translate("MainWindow", u"If format loaded is .nid fill the following:", None))
        self.directionfilt_chkbx.setText(QCoreApplication.translate("MainWindow", u"Scan Direction filter", None))
        self.signtype_lbl.setText(QCoreApplication.translate("MainWindow", u"Map Signal Type:", None))
        self.signl_cmbox.setItemText(0, QCoreApplication.translate("MainWindow", u"Z", None))
        self.signl_cmbox.setItemText(1, QCoreApplication.translate("MainWindow", u"Amplitude", None))
        self.signl_cmbox.setItemText(2, QCoreApplication.translate("MainWindow", u"Phase", None))

        self.label.setText(QCoreApplication.translate("MainWindow", u"If format loaded is .png or .jpg set the following:", None))
        self.numpoints_lbl.setText(QCoreApplication.translate("MainWindow", u"Image Width [µm]:", None))
        self.numlines_lbl.setText(QCoreApplication.translate("MainWindow", u"Image Height [µm]:", None))
        self.titleimgproc_lbl.setText(QCoreApplication.translate("MainWindow", u"Image processing:", None))
        self.gaussian_chkbx.setText(QCoreApplication.translate("MainWindow", u"Gaussian Filter", None))
        self.median_lbl.setText(QCoreApplication.translate("MainWindow", u"Pxl Size Median Filter:", None))
        self.median_editlbl.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.median_chkbx.setText(QCoreApplication.translate("MainWindow", u"Median Filter:", None))
        self.gauss_edit.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.fittype_cmbox.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))
        self.fittype_cmbox.setItemText(1, QCoreApplication.translate("MainWindow", u"Linear", None))
        self.fittype_cmbox.setItemText(2, QCoreApplication.translate("MainWindow", u"Quadratic", None))

        self.gauss_lbl.setText(QCoreApplication.translate("MainWindow", u"Sigma Gauss Filter:", None))
        self.fittype_lbl.setText(QCoreApplication.translate("MainWindow", u"Fit type:", None))
        self.updatefilt_btn.setText(QCoreApplication.translate("MainWindow", u"Filters Update", None))
        self.titleimg_lbl.setText(QCoreApplication.translate("MainWindow", u"Reference Image Selected", None))
        self.image_lbl.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.logtitle_lbl.setText(QCoreApplication.translate("MainWindow", u"Log Panel:", None))
    # retranslateUi

