# main.py
import argparse
import numpy as np
import cv2
import os
import subprocess
from facial_emotion import FacialEmotionFIP
import matplotlib.pyplot as plt
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
        cell_size = int(width / len(self.labels))
        _ct = 0
        _txt_y = self.header_txt_y
        _rad = self.header_circ_rad

        header = np.zeros((_txt_y + _rad, width, 3), dtype=np.uint8)
        for _itm in self.labels:
            _ctr = (_ct * cell_size + _rad, _txt_y - int(_rad / 2))
            cv2.circle(header, _ctr, _rad, self.colors[_ct], thickness=-1, lineType=8, shift=0)
            _ctr = (_ct * cell_size + 2 * _rad, _txt_y)
            cv2.putText(header, _itm, _ctr, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _ct += 1

        return header

    def generate_body(self, entries=None, frame=None):
        body = np.zeros((self.face_size, frame.shape[1], 3), dtype=np.uint8)
        if entries:
            cell_size = int(frame.shape[1] / len(entries))
            _ctr = 0
            for _line in entries:
                _vals = _line.split(",")
                _face = frame[int(_vals[1]):int(_vals[3]), int(_vals[0]):int(_vals[2]), :]
                _face = cv2.resize(_face, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA)
                body[:, _ctr*cell_size:_ctr*cell_size + self.face_size, :] = _face

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
        self.times = []
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_usage = []
        self.gpu_usage = []

        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 12))
        self.fig.suptitle('System Resource Usage')
        self.start_time = time.time()

    def update(self, frame):
        current_time = time.time() - self.start_time
        self.times.append(current_time)

        self.cpu_usage.append(psutil.cpu_percent())
        self.ax1.clear()
        self.ax1.plot(self.times, self.cpu_usage)
        self.ax1.set_ylabel('CPU Usage (%)')
        self.ax1.set_ylim(0, 100)

        self.memory_usage.append(psutil.virtual_memory().percent)
        self.ax2.clear()
        self.ax2.plot(self.times, self.memory_usage)
        self.ax2.set_ylabel('Memory Usage (%)')
        self.ax2.set_ylim(0, 100)

        self.disk_usage.append(psutil.disk_usage('/').percent)
        self.ax3.clear()
        self.ax3.plot(self.times, self.disk_usage)
        self.ax3.set_ylabel('Disk Usage (%)')
        self.ax3.set_ylim(0, 100)

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
        ani = FuncAnimation(self.fig, self.update, interval=1000)
        plt.tight_layout()
        plt.show()

class FaceExpAnalyzer:
    def __init__(self):
        pass

    def process_image(self, fileName=None):
        if os.path.exists(fileName):
            fed = FacialEmotionFIP()
            img = cv2.imread(fileName)
            info = fed.get_facial_emotions(img)
            return 0
        else:
            return 1

    def process_video(self, infile=None, outfile=None, dbg_dir=None):
        if os.path.exists(infile):
            cap = cv2.VideoCapture(infile)
            _vis = None
            if dbg_dir:
                _vis = Visualizer(dbg_dir)

            chunk_size = 16
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fed = FacialEmotionFIP()
            frame_number = 0
            f = open(outfile, 'w')
            if dbg_dir:
                hdr = _vis.generate_header(width=wd)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_number += 1
                _info = fed.get_facial_emotions(frame)

                if dbg_dir:
                    body = _vis.generate_body(entries=_info, frame=frame)
                    canvas = np.vstack((np.vstack((hdr, body)), frame))
                    cv2.imwrite(os.path.join(_vis.dbg_dir, "%05d.png" % frame_number), canvas)
                    _rsz = cv2.resize(canvas, (640, 480))
                    cv2.imshow('Output', _rsz)
                    cv2.waitKey(1)
                for _ln in _info:
                    _line = '%d,%s\n' % (frame_number, _ln)
                    f.write(_line)

                if frame_number % chunk_size == 0:
                    print('Chunk processed. [%04d] frames remaining.' % (length-frame_number + 1))
                    print('Predicted audio-visual emotion vector for chunk')
                    print(_info)
            cap.release()
            f.close()

            if dbg_dir:
                cv2.destroyAllWindows()
                _copts = " -map 0:v -map 1:a -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "
                _cmd = f"ffmpeg -hide_banner -r {fps} -i {dbg_dir}/%05d.png -i {infile} {_copts} {dbg_dir}_output.mp4"
                print('Running %s' % _cmd)
                subprocess.run(_cmd, shell=True, check=True)
                _cmd = f"rm -rf {dbg_dir}"
                print('Running %s' % _cmd)
                subprocess.run(_cmd, shell=True, check=True)

            return 0
        else:
            print('Input file non-existent')
            return 1

def process_image(fileName=None):
    if os.path.exists(fileName):
        fed = FacialEmotionFIP()
        img = cv2.imread(fileName)
        info = fed.get_facial_emotions(img)
        return 0
    else:
        return 1


def process_video(infile=None, outfile=None, dbg_dir=None):
    if os.path.exists(infile):
        cap = cv2.VideoCapture(infile)
        _vis = None
        if dbg_dir:
            _vis = Visualizer(dbg_dir)

        chunk_size = 16
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fed = FacialEmotionFIP()
        frame_number = 0
        f = open(outfile, 'w')
        if dbg_dir:
            hdr = _vis.generate_header(width=wd)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            _info = fed.get_facial_emotions(frame)

            if dbg_dir:
                body = _vis.generate_body(entries=_info, frame=frame)
                # Concatenate and write to disk
                canvas = np.vstack((np.vstack((hdr, body)), frame))
                cv2.imwrite(os.path.join(_vis.dbg_dir, "%05d.png" % frame_number), canvas)
                _rsz = cv2.resize(canvas, (640, 480))
                cv2.imshow('Output', _rsz)
                cv2.waitKey(1)
            for _ln in _info:
                _line = '%d,%s\n' % (frame_number, _ln)
                f.write(_line)

            if frame_number % chunk_size == 0:
                print('Chunk processed. [%04d] frames remaining.' % (length-frame_number + 1))
                print('Predicted audio-visual emotion vector for chunk')
                print(_info)
        cap.release()
        f.close()

        # Making output video generation optional
        if dbg_dir:
            cv2.destroyAllWindows()
            _copts = " -map 0:v -map 1:a -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "
            _cmd = f"ffmpeg -hide_banner -r {fps} -i {dbg_dir}/%05d.png -i {infile} {_copts} {dbg_dir}_output.mp4"
            print('Running %s' % _cmd)
            subprocess.run(_cmd, shell=True, check=True)
            _cmd = f"rm -rf {dbg_dir}"
            print('Running %s' % _cmd)
            subprocess.run(_cmd, shell=True, check=True)

        return 0
    else:
        print('Input file non-existent')
        return 1


def main():
    parser = argparse.ArgumentParser(description='Pov shot based emotion recognition')
    parser.add_argument('-i', required=True, type=str, help='Input Image or Video')
    parser.add_argument('-o', required=True, type=str, help='Output Csv file path')
    parser.add_argument('-d', required=False, type=str, help='Output debug dir')
    args = parser.parse_args()
    process_video(infile=args.i, outfile=args.o, dbg_dir=args.d)


if __name__ == '__main__':
    main()
