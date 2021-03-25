import os, sys
import json
import time
import shelve
from pathlib import Path
from typing import List, Dict
from collections import namedtuple, OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
import random as r

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, send, emit

from loguru import logger

import shutil
import schedule
import time
import subprocess
import distutils

from multiprocessing import Process


r.seed(125)
logger.add(sys.stdout)  # TODO: configure loguru

app = Flask(__name__, static_folder='./static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


command_yolo = [
    'python3',
    '../yolov5/inference_folder.py',
    '--source', 'imgs/',
    '--weights', '../yolov5/runs/exp80/weights/best.pt',
    '--conf', '0.45',
    '--output', 'inference/'
    '--augment',
    '--agnostic-nms',
    '--save-result',
]

command_email = 'python3 get_emails.py -c config.yaml'.split(' ')


def job():
    subprocess.run(command_email)

    subprocess.run(command_yolo)

    shutil.rmtree('imgs')
    distutils.dir_util.copy_tree('./inference/img', './predict/')
    distutils.dir_util.copy_tree('./inference/json', './predict/')
    print('job was run')

#schedule.every().hour.do(job)
#schedule.every().day.at("10:30").do(job)


def f():
    schedule.every(1).minutes.do(job)
    while 1:
        schedule.run_pending()
        time.sleep(1)


def start_process():
    p = Process(target=f, args=[], daemon=True)
    p.start()


@dataclass
class User:
    user_id: str
    main_id: str


users_dict: Dict[str, User] = {}


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


@app.route("/")
def hello():
    # return render_template('index.html')
    with open('./templates/index.html', mode='r') as f:
        s = f.readlines()
    html_string = ''.join(s)
    return html_string


@app.route('/imgs/<int:img_id>/cat.jpg', methods=['GET'])
def get_img_by_id(img_id):
    # path = './emails/RAM:Cam001_20200928_1327.jpg/Cam001_20200928_1327.jpg'
    path = Path('./predict/img/')
    path = path / imgs[img_id]
    
    _, res = cv2.imencode('.jpg', cv2.imread(str(path)))

    return res.tostring()


@app.route('/imgs/cat.jpg', methods=['GET'])
def new_value():
    path = './emails/RAM:Cam001_20200928_1327.jpg/Cam001_20200928_1327.jpg'
    
    _, res = cv2.imencode('.jpg', cv2.imread(str(path)))

    return res.tostring()


@app.route('/imgs/<int:number>/preview.jpg', methods=['GET'])
def get_preview(number):
    path = Path('./predict/img/')

    img = cv2.imread(str(path / imgs[number]))
    h, w = img.shape[:2]
    new_h = 200
    _, res = cv2.imencode('.jpg', cv2.resize(img, (int(w * new_h / h), new_h)))

    return res.tostring()

emails_list = OrderedDict()


imgs = []
def load_imgs_from_db():
    length = 9
    count = 0
    # with shelve.open('emails_store.shelve') as store:
    #     keys = sorted(map(int, store.keys()), reverse=True)
    #     for key in keys:
    #         key = str(key)
    #         print(f'Key: {key}\tValue: {store[key]}')
    #         if len(store[key]) == 0:
    #             continue
    #         filename = store[key][0]['attach'][0]
    #         imgs.append(filename)
    #         count += 1
    #         if count == length:
    #             break
    path = 'predict/img'
    if os.path.exists(path):
        imgs = list(sorted(os.listdir(path)))
        imgs = imgs[:length]


def update_emails():
    global emails_list

    for email in emails_list:
        if email not in emails_list:
            emails_list[email]
    pass


@socketio.on('login')
def login(data):
    username = data['username']
    user = User(username, None)
    users_dict[username] = user
    emit('message', json.dumps({'message': 'I was logged'}))


@socketio.on('set_main')
def tick(data):
    """
    data = {
        username: str,
        main: str,
    }
    """
    user = data['username']
    logger.info(f'{user} set main to {data["id"]}')
    users_dict[user].main_id = data['id']
    logger.debug(users_dict[user])


@app.route('/js/<path:path>', methods=['GET'])
def send_js(path):
    return send_from_directory('js', path ,mimetype='text/javascript')


@app.route('/sourcemaps/<path:path>', methods=['GET'])
def send_sourcemaps(path):
    return send_from_directory('sourcemaps', path ,mimetype='text/javascript')



@app.route('/css/<path:path>', methods=['GET'])
def send_css(path):
    return send_from_directory('css', path ,mimetype='text/css')


@app.route('/update_db/', methods=['POST'])
def update_db():
    load_imgs_from_db()
    return ""


if __name__ == '__main__':
    start_process()
    load_imgs_from_db()
    socketio.run(app, host='127.0.0.1', port=5001, debug=True)
