import numpy as np
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DriftPlotterWindow(QMainWindow):
    def __init__(self, drift_list_um, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drift Analysis Results")
        self.resize(800, 600)

        # Convert raw list data to numpy arrays
        self.drift_abs = np.array(drift_list_um)
        self.frames = np.arange(len(self.drift_abs))
        self.drift_rel = self._calculate_relative_drift()

        # Main Layout Setup using a Tab Widget to separate components and 2D tracking
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Build tabs
        self._init_trends_tab()
        self._init_trajectory_tab()

    def _calculate_relative_drift(self):
        """Calculates step-by-step frame drift jumps."""
        if len(self.drift_abs) <= 1:
            return np.zeros_like(self.drift_abs)
        rel_drift = np.zeros_like(self.drift_abs)
        rel_drift[1:] = np.diff(self.drift_abs, axis=0)
        return rel_drift

    def _init_trends_tab(self):
        """Creates the absolute vs relative trend chart tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvas(fig)
        axs = fig.subplots(2, 1, sharex=True)

        # 1. Absolute Drift Plot
        axs[0].plot(self.frames, self.drift_abs[:, 0], 'o-', label='X Drift (Abs)', color='royalblue')
        axs[0].plot(self.frames, self.drift_abs[:, 1], 'o-', label='Y Drift (Abs)', color='darkorange')
        axs[0].set_ylabel('Absolute Drift ($\mu m$)')
        axs[0].set_title('Image Drift Tracking')
        axs[0].grid(True, linestyle='--', alpha=0.5)
        axs[0].legend()

        # 2. Relative Drift Plot
        axs[1].plot(self.frames, self.drift_rel[:, 0], 's--', label='X Drift (Rel)', color='cornflowerblue')
        axs[1].plot(self.frames, self.drift_rel[:, 1], 's--', label='Y Drift (Rel)', color='orange')
        axs[1].set_xlabel('Comparison File Index')
        axs[1].set_ylabel('Relative Drift ($\mu m$)')
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[1].legend()

        fig.tight_layout()
        layout.addWidget(canvas)
        self.tabs.addTab(widget, "Trend Components")

    def _init_trajectory_tab(self):
        """Creates the spatial 2D trajectory tracking tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig = Figure(figsize=(6, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Plot structural pathway line
        ax.plot(self.drift_abs[:, 0], self.drift_abs[:, 1], color='purple', linestyle='-', alpha=0.4)
        
        # Sequentially color-coded spatial drift scatter points
        scatter = ax.scatter(self.drift_abs[:, 0], self.drift_abs[:, 1], 
                             c=self.frames, cmap='viridis', s=45, edgecolors='k', zorder=3)
        
        # Directional markers
        ax.scatter(self.drift_abs[0, 0], self.drift_abs[0, 1], color='green', marker='^', s=120, label='Start')
        ax.scatter(self.drift_abs[-1, 0], self.drift_abs[-1, 1], color='red', marker='v', s=120, label='End')

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Comparison File Sequence')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('X Drift ($\mu m$)')
        ax.set_ylabel('Y Drift ($\mu m$)')
        ax.set_title('Spatial 2D Drift Trajectory')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()
        layout.addWidget(canvas)
        self.tabs.addTab(widget, "2D Spatial Map")