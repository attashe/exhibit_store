import os
import shutil
import subprocess

from distutils import copy_tree
from loguru import logger


command_yolo = [
    'python',
    '../yolov5/inference_folder.py',
    '--source', 'imgs/',
    '--weights', '../yolov5/runs/exp80/weights/best.pt',
    '--conf', '0.45',
    '--output', 'inference/',
    '--augment',
    '--agnostic-nms',
    '--save-result',
]

command_email = 'python get_emails.py -c config.yaml'.split(' ')


def job():
    subprocess.run(command_email)
    logger.info('Email downloader was run')
    subprocess.run(command_yolo)
    logger.info('Predictor was run')

    shutil.rmtree('imgs')

    if os.path.exists('inference'):
        copy_tree('./inference/img/', './predict/img/')
        shutil.copy('./inference/objects.csv', './predict/')
    logger.info('Job was run')