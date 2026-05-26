import os
import sys
import ctypes
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QObject, Signal
from main_ui import Ui_MainWindow

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class SignalingStream(QObject):
    text_written = Signal(str)

    def write(self, text):
        if text.strip():  
            self.text_written.emit(text)

    def flush(self):
        pass


class main_window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(main_window, self).__init__()
        self.setupUi(self)
        
        # LOG REDIRECTION SETUP
        self.stdout_redirector = SignalingStream()
        self.stdout_redirector.text_written.connect(self.append_to_log)
        sys.stdout = self.stdout_redirector
        
        # Initialize helpers
        if hasattr(self, "on_signaltype_change"):
            self.on_signaltype_change()
        if hasattr(self, "set_starting_template"):
            self.set_starting_template()

    def append_to_log(self, text):
        self.log_txtbrs.append(text)
        self.log_txtbrs.ensureCursorVisible()

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


if __name__ == '__main__':
    # WINDOWS TASKBAR ICON FIX
    # Assign a unique App User Model ID so Windows links the process to your icon
    try:
        myappid = "mycompany.afmprocessingapp.template_matching.1" 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        print(f"Could not set AppUserModelID: {e}")

    app = QApplication(sys.argv)

    # STYLE SHEET LOADER (EXE COMPATIBLE)
    # Use get_resource_path so it works inside the PyInstaller temporary context
    qss_path = get_resource_path("style.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    window = main_window()
    window.setObjectName('Main Window')
    window.setWindowTitle('Template matching')
    
    # SET WINDOW WINDOW ICON
    # Replace "icon.png" with your asset's exact file name
    icon_filename = "icon.png" 
    window.setWindowIcon(QIcon(get_resource_path(icon_filename)))
    
    # FIXED DIMENSIONS BLOCK
    window.resize(694, 694) 
    window.setMinimumSize(694, 694)

    window.show()
    sys.exit(app.exec())