import numpy as np
import matplotlib.pyplot as plt

class DriftPlotter:
    def __init__(self, drift_list_um):
        """
        Initializes the plotter with absolute drift values in micrometers.
        :param drift_list_um: List of lists or tuples, e.g., [[x1, y1], [x2, y2], ...]
        """
        self.drift_abs = np.array(drift_list_um)
        self.frames = np.arange(len(self.drift_abs))
        self.drift_rel = self._calculate_relative_drift()

    def _calculate_relative_drift(self):
        """Calculates step-by-step drift between consecutive frames."""
        if len(self.drift_abs) <= 1:
            return np.zeros_like(self.drift_abs)
        
        # Relative drift starts at 0 for the first frame
        rel_drift = np.zeros_like(self.drift_abs)
        # Difference between frame(i) and frame(i-1)
        rel_drift[1:] = np.diff(self.drift_abs, axis=0)
        return rel_drift

    def plot_components(self, save_path=None):
        """Plots X and Y drift over time for both absolute and relative metrics."""
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Absolute Drift Plot
        axs[0].plot(self.frames, self.drift_abs[:, 0], 'o-', label='X Drift (Abs)', color='royalblue')
        axs[0].plot(self.frames, self.drift_abs[:, 1], 'o-', label='Y Drift (Abs)', color='darkorange')
        axs[0].set_ylabel('Absolute Drift ($\mu m$)')
        axs[0].set_title('Image Drift Analysis over Frames')
        axs[0].grid(True, linestyle='--', alpha=0.6)
        axs[0].legend()

        # Relative Drift Plot
        axs[1].plot(self.frames, self.drift_rel[:, 0], 's--', label='X Drift (Rel)', color='cornflowerblue')
        axs[1].plot(self.frames, self.drift_rel[:, 1], 's--', label='Y Drift (Rel)', color='orange')
        axs[1].set_xlabel('Comparison File Index (Time/Frames)')
        axs[1].set_ylabel('Relative Drift ($\mu m$)')
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved successfully to: {save_path}")
        
        plt.show()

    def plot_2d_trajectory(self, save_path=None):
        """Plots the actual 2D path of the drift (X vs Y)."""
        plt.figure(figsize=(7, 7))
        
        # Plot the path line
        plt.plot(self.drift_abs[:, 0], self.drift_abs[:, 1], color='purple', linestyle='-', alpha=0.5)
        
        # Scatter points colored by frame sequence to see the direction of drift
        scatter = plt.scatter(self.drift_abs[:, 0], self.drift_abs[:, 1], 
                              c=self.frames, cmap='viridis', s=50, edgecolors='k', zorder=3)
        
        # Marks start and end points
        plt.scatter(self.drift_abs[0, 0], self.drift_abs[0, 1], color='green', marker='^', s=150, label='Start', zorder=4)
        plt.scatter(self.drift_abs[-1, 0], self.drift_abs[-1, 1], color='red', marker='v', s=150, label='End', zorder=4)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Frame / File Index')
        
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        
        plt.xlabel('X Drift ($\mu m$)')
        plt.ylabel('Y Drift ($\mu m$)')
        plt.title('2D Drift Trajectory')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.axis('equal') # Keeps the aspect ratio uniform
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()