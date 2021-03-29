import cv2
import time
import random
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger

from PIL import ImageFont

import torch
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


class DotDict(dict):
    def __getattribute__(self, name):
        try:
            dict.__getattribute__(self, name)
        except AttributeError:
            if name in self:
                return self[name]
            raise

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __setitem__(self, key, value):
        try:
            class test(object): __slots__ = key
        except TypeError:
            raise TypeError("invalid identifier: '%s'" % key)

        dict.__setitem__(self, key, value)


@st.cache
def load_model(imgsz):
    weights = '../yolov5/runs/exp80/weights/best.pt'

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    return model, imgsz


# @st.cache
# def load_cfg():
#     cfg = {
#         "imgsz": 720,
#         "conf_thresh": 0.4,
#         "iou_thresh": 0.5,
#         "agnostic_nms": False,
#         "colors": [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
#     }

#     return DotDict(cfg)
@st.cache
def get_colors():
    return [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


# Inittalize font file
fontpath = "./19894.ttf"
font = ImageFont.truetype(fontpath, 32)

# Init names and colors
names = ['лось', 'медведь', 'кабан', 'рысь', 'барсук', 'олень',
         'человек', 'техника', 'лиса', 'заяц', 'волк', 'куница',
         'енотовидная собака', 'глухарь', 'собака']
eng_names = ['Moose', 'Bear', 'Boar', 'Lynx', 'Badger', 'Deer',
             'People', 'Technic', 'fox', 'hare', 'Wolf', 'Marten',
             'Racoon Dog', 'Capercaillie', 'Dog']

# CONFIG
imgsz = 720
conf_thres = 0.4
iou_thres = 0.5
classes = None
agnostic_nms = False
colors = get_colors()
# cfg = load_cfg()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half = device.type != 'cpu'  # half precision only supported on CUDA
augment = st.sidebar.checkbox('Augment')
rgb = st.sidebar.checkbox('RGB')
model, imgsz = load_model(imgsz)


path_input = st.sidebar.text_input('Путь к папке: ', value='')
if path_input != '':
    path = Path(path_input)
    if path.exists():
        selected_path = st.sidebar.radio("What's your favorite movie genre",
                list(path.glob('*'))[:50])


def load_image(path, img_size=720, augment=False):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
        nw0, nh0 = int(w0 * r), int(h0 * r)
        nw0, nh0 = nw0 - nw0 % 32, nh0 - nh0 % 32
        img = cv2.resize(img, (nw0, nh0), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


# @st.cache
def inference(img_path: Path):
    img, size_orig, size = load_image(str(img_path), img_size=imgsz, augment=augment)
    # img = cv2.imread(str(img_path))
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im0s = img.copy()

    img = torch.from_numpy(img).to(device)
    img = img.permute(2, 0, 1)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()

    torch.cuda.synchronize()
    # INFO: Apply Classifier deleted

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = img_path, '', im0s

        s += '%gx%g ' % img.shape[2:]  # print string
        h, w = im0s.shape[:2]
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3, font=font)

        # Print time (inference + NMS)
        logger.info('%sDone. (%.3fs)' % (s, t2 - t1))

    return im0


if selected_path:
    st.image(inference(Path(selected_path)), caption='Image', use_column_width=True)

# display = ("male", "female")

# options = list(range(len(display)))

# value = st.selectbox("gender", options, format_func=lambda x: display[x])

# st.write(value)


# options = st.multiselect(
#             'What are your favorite colors',
#             ['Green', 'Yellow', 'Red', 'Blue'],
#             ['Yellow', 'Red'])

# st.write('You selected:', options)

# genre = st.radio(
#         "What's your favorite movie genre",
#         ('Comedy', 'Drama', 'Documentary'))
