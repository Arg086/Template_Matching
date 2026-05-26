from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel
from PySide6.QtGui import QMovie
from PySide6.QtCore import Qt

class GifViewerWindow(QDialog):
    def __init__(self, gif_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cropped Area Animation Loop")
        self.resize(500, 500)
        
        layout = QVBoxLayout(self)
        
        # Display label for the animation
        self.movie_lbl = QLabel(self)
        self.movie_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.movie_lbl)
        
        # Initialize and play the GIF loop with QMovie
        self.movie = QMovie(gif_path)
        self.movie_lbl.setMovie(self.movie)
        self.movie.start()

    def closeEvent(self, event):
        # Explicitly stop the animation tracking loop on window close to save CPU memory
        self.movie.stop()
        super().closeEvent(event)