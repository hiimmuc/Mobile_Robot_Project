# TODO: provide path to video, auto run extract image and export dataset

import argparse
import os
from pathlib import Path

import cv2
from tqdm.auto import tqdm

from yolov5 import *

allowed_extensions = ['mp4', 'avi', 'mpg']


def check_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def extract_frame(video_path, interval=15, save_path='backup/dataset/'):
    '''
    Extract frame from video by provided fps
    '''
    if not check_extension(video_path):
        raise Exception('Invalid file extension')
    os.makedirs(save_path, exist_ok=True)

    print(f'Extracting frame from {video_path}...')

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if ret:
            if count == interval:
                cv2.imwrite(str(Path(f'{Path(save_path)}', f'frame{frame_count}.jpg'.format(count))), frame)
                print(f'Frame {frame_count} extracted')
                count = 0
                frame_count += 1
            count += 1
        else:
            break
    cap.release()


def extract_bbox(dataset_dir='backup/dataset', img_size=640, weights_path=None, conf=0.6):
    print('Extracting bounding box...')
    imgs = Path(dataset_dir).glob('*.jpg')
    for img in imgs:
        cmd = f'python yolov5/detect.py --weights {weights_path} --source {img} --img-size {img_size} --conf {conf} --save-txt'
        os.system(cmd)


if __name__ == '__main__':

    video_path = r"backup\test_vid.mp4"
    extract_frame(video_path, interval=1)
    extract_bbox(weights_path=r'backup\weights\best.pt')
