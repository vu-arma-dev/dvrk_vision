# DvRK Vision

# Dependencies

UVC Camera:

`sudo apt-get install ros-<distro>-uvc-camera  `

Where \<distro\> is probably kinetic
 
# New Camera Setup
just_cams.launch is the base launch file that will allow you to launch a camera. In order to not have to change that file every time you have a new camera, we use a wrapper launch file like just_cams_vu.launch which starts the EX8029 camera from sparkfun/eys3D. 

Get info on your camera:
`v4l2-ctl - -list-formats-ext`

Make a copy of 
```
roscd dvrk_vision/launch
cp just_cams_vu.launch just_cams_new_camera.launch
```

Things you may have to change in your new launch file:
* width
* height
* fps

In other cameras, you may want to add options for:
* format
* image_format

A somewhat confusing parameter is "single_image_from_camera". This depends on the type of stereo camera you use. Some give one video stream with the left and right eyes smashed together that's double the width of the video feed from one of the cameras alone (e.g. for a 640x480 image, you'll get a stream of data thats 1280x480). Other cameras return two separate image streams like you would get from plugging in two separate cameras. If the camera is the former type, "single_image_from_camera" should be true, otherwise false.

After running calibration (below), change:
* camera_info_url_right
* camera_info_url_left

# Camera Calibration
Make sure you have a calibration checkerboard: 
https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf 

Start the camera
`roslaunch dvrk_vision just_cams_irep_oldcamera.launch`
Run the ros calibration script:
`rosrun camera_calibration cameracalibrator.py --size 7x9 --square 0.02 right:=/stereo/right/image_raw left:=/stereo/left/image_raw left_camera:=/stereo/left right_camera:=/stereo/right`

A window will pop up - show the cameras the checkerboard in a variety of poses. There's a gray'ed out set of buttons on the right of the window (may be hard to see if the window isn't maximized). Once you collect enough poses, the "calibration" button will turn green. I usually recommend taking a few more images then clicking "calibrate". The window will usually hang for a few seconds, don't worry about that, just be patient. Then click "save", you'll get a message on the terminal like: 
`('Wrote calibration data to', '/tmp/calibrationdata.tar.gz')`

So close the window and cd to /tmp/ and unzip those files, and rename the new files accordingly:
```
cd /tmp
mkdir output
tar -xvf calibrationdata.tar -C ./output
cd output
mv left.yaml new_camera_left.yaml
mv right.yaml new_camera_right.yaml
```

Now go back to your just_cams_new_camera.launch and edit the files to point to wherever you want to put new_camera_left.yaml and new_camera_right.yaml. The smart thing to do is put those into the /defaults folder of this repository. We often label them with resolutions so that you can have different yaml files for different resolutions of the cameras.
