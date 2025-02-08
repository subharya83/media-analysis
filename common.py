import cv2
import numpy as np
import os
import psutil
import GPUtil
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation


class Visualizer:
    def __init__(self, out_dir=None):
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 125, 0)]
        self.header_txt_y = 40
        self.header_circ_rad = 20
        self.bar_length = 512
        self.bar_thickness = 16
        self.pad_width = 16
        self.dbg_dir = out_dir
        self.labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        self.face_size = 256
        os.makedirs(self.dbg_dir, exist_ok=True)

    def generate_header(self, width=None):
        """
        Generate a header banner showing all emotion labels
        Args:
            width: width of video frame

        Returns:
            header (width of the video frame, height is determined)
        """
        cell_size = int(width / len(self.labels))
        _ct = 0
        _txt_y = self.header_txt_y
        _rad = self.header_circ_rad

        # Generate static header for every frame
        header = np.zeros((_txt_y + _rad, width, 3), dtype=np.uint8)
        for _itm in self.labels:
            # Adding Legend color
            _ctr = (_ct * cell_size + _rad, _txt_y - int(_rad / 2))
            cv2.circle(header, _ctr, _rad, self.colors[_ct], thickness=-1, lineType=8, shift=0)
            # Adding Text
            _ctr = (_ct * cell_size + 2 * _rad, _txt_y)
            cv2.putText(header, _itm, _ctr, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _ct += 1

        return header

    def generate_body(self, entries=None, frame=None):
        """
        Render confidence scores for all emotion score given the entries
        Args:
            entries: entries is a list of tuples frame number, bounding box coordinates, and confidence scores
            frame: numpy array containing frame data

        Returns:
            body (width of the video frame, height is determined)
        """
        body = np.zeros((self.face_size, frame.shape[1], 3), dtype=np.uint8)
        if entries:
            cell_size = int(frame.shape[1] / len(entries))
            _ctr = 0
            for _line in entries:
                _vals = _line.split(",")
                # Obtain bounding box coordinates and extract image

                _face = frame[int(_vals[1]):int(_vals[3]), int(_vals[0]):int(_vals[2]), :]
                _face = cv2.resize(_face, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA)
                # Populate img data into body
                body[:, _ctr*cell_size:_ctr*cell_size + self.face_size, :] = _face

                # Populate bars
                _itc = 0
                for _itm in self.labels:
                    _col = self.colors[_itc]
                    _y = _itc*(self.bar_thickness + self.pad_width) + self.pad_width

                    _x1 = _ctr*cell_size + self.face_size + self.pad_width
                    _x2 = _x1 + int(self.bar_length*float(_vals[4 + _itc]))

                    cv2.line(body, (_x1, _y), (_x2, _y), _col, thickness=self.bar_thickness)
                    _itc += 1

        return body


class ResourceUsageViewer:
    def __init__(self):
        # Initialize lists to store data
        self.times = []
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_usage = []
        self.gpu_usage = []

        # Set up the plot
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 12))
        self.fig.suptitle('System Resource Usage')
        self.start_time = time.time()

    # Function to update the plot
    def update(self, frame):
        current_time = time.time() - self.start_time
        self.times.append(current_time)

        # CPU usage
        self.cpu_usage.append(psutil.cpu_percent())
        self.ax1.clear()
        self.ax1.plot(self.times, self.cpu_usage)
        self.ax1.set_ylabel('CPU Usage (%)')
        self.ax1.set_ylim(0, 100)

        # Memory usage
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.ax2.clear()
        self.ax2.plot(self.times, self.memory_usage)
        self.ax2.set_ylabel('Memory Usage (%)')
        self.ax2.set_ylim(0, 100)

        # Disk usage
        self.disk_usage.append(psutil.disk_usage('/').percent)
        self.ax3.clear()
        self.ax3.plot(self.times, self.disk_usage)
        self.ax3.set_ylabel('Disk Usage (%)')
        self.ax3.set_ylim(0, 100)

        # GPU usage (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_usage.append(gpus[0].load * 100)
            else:
                self.gpu_usage.append(0)
        except:
            self.gpu_usage.append(0)
        self.ax4.clear()
        self.ax4.plot(self.times, self.gpu_usage)
        self.ax4.set_ylabel('GPU Usage (%)')
        self.ax4.set_ylim(0, 100)
        self.ax4.set_xlabel('Time (seconds)')

    def view_usage(self):
        # Create animation
        ani = FuncAnimation(self.fig, self.update, interval=1000)

        # Show the plot
        plt.tight_layout()
        plt.show()
