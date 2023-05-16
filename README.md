# ViBe-python
Python implementation of ViBe-gray, A universal background subtraction algorithm for video sequences. Based on: http://orbi.ulg.ac.be/bitstream/2268/145853/1/Barnich2011ViBe.pdf

# Examples
## HexBug videos background subtraction

By using phi=5 and a region of just 1 pixel to define the background model we get some ghosts that are absorbed into the background model very slow. The masks were found using grayscale frames and then applied to the original rgb frames.
```
python video_test.py --video_path /data/training098.mp4 --phi 5 --scale 2
```
<img src="https://github.com/maavilapa/ViBe-python/blob/main/data/results/rgb_training098.gif" width="640">

By reducing N from 20 to 5 and R to 10 we get more ghosts and False positives, due to the higher sensitivity of the model to shadows and small light changes.
```
python video_test.py --video_path /data/training098.mp4 --phi 4 --scale 2 --N 5 --R 10
```
<img src="https://github.com/maavilapa/ViBe-python/blob/main/data/results/rgb_training098_n5.gif" width="640">

Use of phi=1 and a region of 8 pixels improve the results for objects that are moving very fast and in a controlled setup with no camera motion.
```
python video_test.py --video_path /data/training098.mp4 --phi 1 --scale 2
```
<img src="https://github.com/maavilapa/ViBe-python/blob/main/data/results/rgb_training098_best.gif" width="640">
<img src="https://github.com/maavilapa/ViBe-python/blob/main/data/results/background_training098_best.gif" width="320">

## Real time background subtraction
ViBe was tested in real time with default values N = 20, R = 20 and min=2 for a camera resolution of 1280x480.   
```
python webcam_test.py --video_path /data/training098.mp4 --phi 5 --save_video
```
<img src="https://github.com/maavilapa/ViBe-python/blob/main/data/results/webcam_rgb.gif" width="800">

## MOTChallenge data background subtraction

ViBe was also tested in one video of the MOT tracking challenge, which present changes in Brightness, camera motion and more foreground instances than background areas.
```
python video_test.py --video_path /data/MOT20-01-raw.mp4 --phi 16 --scale 2
```

<img src="https://github.com/maavilapa/ViBe-python/blob/main/data/results/rgb_MOT20-01-raw_phi1.gif" width="800">
```
python video_test.py --video_path /data/MOT20-01-raw.mp4 --phi 1 --scale 2
```

<img src="https://github.com/maavilapa/ViBe-python/blob/main/data/results/rgb_MOT20-01-raw_phi16.gif" width="800">

