import cv2
from vibe import ViBe
import argparse
import warnings
import os.path as osp
import os
import time
import numpy as np
warnings.filterwarnings("ignore")
def make_parser():
    parser = argparse.ArgumentParser("Vibe demo!")
    parser.add_argument("--save_path", type=str, default="data/results")
    parser.add_argument("--N", default=20, type=int, help="N")
    parser.add_argument("--min", default=2, type=int, help="min")
    parser.add_argument("--R", default=20, type=int, help="R")
    parser.add_argument("--phi", default=1, type=int, help="phi")
    parser.add_argument("--save_video", dest="save_video", default=False, action="store_true", help="save recorded video")
    parser.add_argument("--numba", dest="numba", default=True, action="store_true", help="use numba")
    return parser

def main(args):
    vibe=ViBe(N=args.N, _min=args.min, R=args.R, phi=args.phi)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if args.save_video:
        vis_folder=args.save_path
        width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = vc.get(cv2.CAP_PROP_FPS)/2

        current_time = time.localtime()
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        save_path_rgb = osp.join(save_folder, "webcam_rgb.mp4")
        vid_writer = cv2.VideoWriter(
            save_path_rgb, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2*int(width), int(height))
        )

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        samples = vibe.set_background(frame_gray)
    else:
        rval = False
    #time.sleep(1)
    while rval:
        rval, frame = vc.read()
        frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        segMap, samples = vibe.update(frame_gray, samples)
        segMap=cv2.cvtColor(cv2.medianBlur(segMap, 5),cv2.COLOR_GRAY2RGB)
        seg_img=cv2.bitwise_and(np.array(frame), segMap)
        seg_img=cv2.hconcat([frame, seg_img])
        if args.save_video:
            vid_writer.write(seg_img)
        cv2.imshow("preview", seg_img)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    if args.save_video:
        vid_writer.release()
    vc.release()
    cv2.destroyWindow("preview")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)