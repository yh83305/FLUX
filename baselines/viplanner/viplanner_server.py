from PIL import Image
from flask import Flask, request, jsonify
from viplanner_agent import VIPlannerAgent
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os
from PIL import Image, ImageDraw, ImageFont
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--config",type=str,default="./configs/viplanner.yaml")
parser.add_argument("--checkpoint",type=str,default="./checkpoints/viplanner.pt")
parser.add_argument("--m2f_config",type=str,default="~/miniconda3/envs/habitat/lib/python3.9/site-packages/mmdet/.mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py")
parser.add_argument("--m2f_checkpoint",type=str,default="./checkpoints/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--record-input-video", action="store_true",
                    help="Record input RGB only; this is not an evaluation video")
args = parser.parse_known_args()[0]

app = Flask(__name__)
viplanner_navigator = None
viplanner_fps_writer = None

@app.route("/health")
def health():
    return jsonify({"algo": "viplanner", "loaded": viplanner_navigator is not None,
                    "device": args.device, "checkpoint": os.path.abspath(args.checkpoint),
                    "segmentation_checkpoint": os.path.abspath(args.m2f_checkpoint)})

@app.route("/navigator_reset",methods=['POST'])
def iplanner_reset():
    global viplanner_navigator,viplanner_fps_writer
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))
    if viplanner_navigator is None:
        viplanner_navigator = VIPlannerAgent(intrinsic,
                                            m2f_path=args.m2f_checkpoint,
                                            m2f_config_path=args.m2f_config,
                                            model_path=args.checkpoint,
                                            model_config_path=args.config,
                                            device=args.device)
    if viplanner_fps_writer is not None:
        viplanner_fps_writer.close()
        viplanner_fps_writer = None
    if args.record_input_video:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        viplanner_fps_writer = imageio.get_writer(f"{stamp}_input_rgb_debug.mp4",
                                               fps=7, macro_block_size=1)
    return jsonify({"algo":"viplanner"})

@app.route("/navigator_reset_env",methods=['POST'])
def viplanner_reset_env():
    return jsonify({"algo":"viplanner"})

def process_goal(goal,range=10.0):
    return_goal = np.clip(goal,-range,range)
    return return_goal

@app.route("/pointgoal_step",methods=['POST'])
def viplanner_step_pointgoal():
    global viplanner_navigator,viplanner_fps_writer
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    goal = process_goal(goal)
    batch_size = goal.shape[0]

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    _,trajectory,fear = viplanner_navigator.step_pointgoal(image,depth,goal)
    if viplanner_fps_writer is not None:
        viplanner_fps_writer.append_data(image.reshape(-1,image.shape[2],3))

    return jsonify({'trajectory': trajectory.cpu().numpy().tolist(),
                    'all_trajectory': trajectory.cpu().numpy()[:,None,:,:].tolist(),
                    'all_values': fear.cpu().numpy().tolist()})

if __name__ == "__main__":
    app.run(host=args.host,port=args.port)

