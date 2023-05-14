from vibe import ViBe
import os
import cv2
import warnings
import os.path as osp
import time
import numpy as np
from loguru import logger
import argparse
warnings.filterwarnings("ignore")

def make_parser():
    parser = argparse.ArgumentParser("Vibe demo!")
    parser.add_argument("--video_path", type=str, default="training098.mp4")
    parser.add_argument("--save_path", type=str, default="data/results")
    parser.add_argument("--N", default=20, type=int, help="N")
    parser.add_argument("--min", default=2, type=int, help="min")
    parser.add_argument("--R", default=20, type=int, help="R")
    parser.add_argument("--phi", default=1, type=int, help="phi")
    parser.add_argument("--scale", default=2, type=int, help="scale image by some value")
    parser.add_argument("--numba", dest="numba", default=True, action="store_true", help="test mot20.")
    return parser

def main(args):
    vibe=ViBe(N=args.N, _min=args.min, R=args.R, phi=args.phi)

    #Initialize background with frame 50
    vis_folder=args.save_path
    #video_path=input("Video path: ")
    video_path=args.video_path
    #video_path="training092.mp4"
    assert os.path.isfile(video_path), "Wrong path"
    vs = cv2.VideoCapture(video_path)
    vs.set(cv2.CAP_PROP_POS_FRAMES, 50)
    #scale=int(input("Scale video by:"))
    scale=args.scale
    while True:
        ret, bg_frame = vs.read()
        if ret:
            break
    if scale!=1:
        bg_frame=cv2.resize(bg_frame, (int(bg_frame.shape[1] /scale), int(bg_frame.shape[0] /scale)), interpolation=cv2.INTER_LINEAR)
    bg_frame=cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
    bg_frame_gray=cv2.cvtColor(bg_frame, cv2.COLOR_RGB2GRAY)

    samples = vibe.set_background(bg_frame_gray)
    

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, "background_"+video_path.split("/")[-1])
    save_path_rgb = osp.join(save_folder, "rgb_"+video_path.split("/")[-1])


    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width/scale), int(height/scale))
    )
    seg_images=[]
    masks=[]
    rgb_images=[]
    frame_id=0
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id % 10 == 0:
                logger.info('Processing frame {} '.format(frame_id))

            frame=cv2.resize(frame, (int(frame.shape[1] /scale), int(frame.shape[0] /scale)), interpolation=cv2.INTER_LINEAR)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            segMap, samples = vibe.update(frame_gray, samples)
            segMap=cv2.cvtColor(cv2.medianBlur(segMap, 5),cv2.COLOR_GRAY2RGB)
            seg_img=cv2.bitwise_and(np.array(frame), segMap)
            seg_images.append(seg_img)
            masks.append(segMap)
            vid_writer.write(segMap)
            rgb_images.append(cv2.hconcat([seg_img, frame]))
            frame_id+=1
            
        else:
            break
    vid_writer.release()

    vid_writer_rgb = cv2.VideoWriter(
        save_path_rgb, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2*(int(width/scale)), int(height/scale)))
    for frame in rgb_images:
        vid_writer_rgb.write(frame[:, :, ::-1])
    vid_writer_rgb.release()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)