import argparse
import numpy as np
import cv2
import os
import subprocess

from models.Fip import FacialEmotionFIP
from models.common import Visualizer


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
