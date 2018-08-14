#!/usr/bin/env python
import sys
import yaml
import rospy
import cv2
import numpy as np
from dvrk import psm
from collections import deque
from rigid_transform_3d import rigidTransform3D, calculateRMSE
import PyKDL
from dvrk_vision.vtk_stereo_viewer import StereoCameras
from image_geometry import StereoCameraModel
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose
from tf_conversions import posemath
from tf_sync import CameraSync

_WINDOW_NAME = "Registration"

def combineImages(imageL, imageR):
    (rows,cols) = imageL.shape[0:2]
    if rows > 640:
        imageL = cv2.resize(imageL, (640, int(640.0 / rows * cols)))
        imageR = cv2.resize(imageR, (640, int(640.0 / rows * cols)))
    if len(imageL.shape) == 2:
        shape = (rows, cols*2)
    elif len(imageL.shape) == 3:
        shape = (rows, cols*2, imageL.shape[2])
    doubleImage = np.zeros(shape,np.uint8)
    doubleImage[0:rows,0:cols] = imageL
    doubleImage[0:rows,cols:cols*2] = imageR
    return doubleImage

def nothingCB(data):
    pass

def displayRegistration(cams, camModel, toolOffset, camTransform, tfSync):
    rate = rospy.Rate(15) # 15hz
    while not rospy.is_shutdown():
        # Get last images
        imageL = cams.camL.image
        imageR = cams.camR.image

        # Wait for images to exist
        if type(imageR) == type(None) or type(imageL) == type(None):
            continue

        # Check we have valid images
        (rows,cols,channels) = imageL.shape
        if cols < 60 or rows < 60 or imageL.shape != imageR.shape:
            rate.sleep()
            continue

        t = cams.camL.info.header.stamp
        try:
            poseMsg = tfSync.synchedMessages[0].pose
        except:
            rate.sleep()
            continue
        robotPosition = posemath.fromMsg(poseMsg)


        # Find 3D position of end effector
        zVector = robotPosition.M.UnitZ()
        pVector = robotPosition.p
        offset = np.array([zVector.x(), zVector.y(), zVector.z(), 0])
        offset = offset * toolOffset
        pos = np.matrix([pVector.x(), pVector.y(), pVector.z(), 1]) + offset;
        pos = pos.transpose()
        pos = np.linalg.inv(camTransform) * pos
        
        # Project position into 2d coordinates
        posL = camModel.left.project3dToPixel(pos)
        posL = [int(l) for l in posL]
        posR = camModel.right.project3dToPixel(pos)
        posR = [int(l) for l in posR]

        (rows,cols) = imageL.shape[0:2]
        posR = (posR[0] + cols, posR[1])

        transforms = tfSync.getTransforms()
        posEnd = posL
        for i in range(0,len(transforms)-1):

            start = [transforms[i].transform.translation.x,
                     transforms[i].transform.translation.y,
                     transforms[i].transform.translation.z]

            end = [transforms[i+1].transform.translation.x,
                   transforms[i+1].transform.translation.y,
                   transforms[i+1].transform.translation.z]

            # Project position into 2d coordinates
            posStartL = camModel.left.project3dToPixel(start)
            posEndL = camModel.left.project3dToPixel(end)
            posStartR = camModel.right.project3dToPixel(start)
            posEndR = camModel.right.project3dToPixel(end)
            # Draw on left and right images
            if not np.isnan(posStartL + posEndL + posStartR + posEndR).any(): 
                posStartL = [int(l) for l in posStartL]
                posEndL = [int(l) for l in posEndL]
                cv2.line(imageL, tuple(posStartL), tuple(posEndL), (0, 255, 0), 1)
                posStartR = [int(l) for l in posStartR]
                posEndR = [int(l) for l in posEndR]
                cv2.line(imageR, tuple(posStartR), tuple(posEndR), (0, 255, 0), 1)

        cv2.line(imageL, tuple(posEnd), tuple(posL), (0,255,0),1)

        image = combineImages(imageL, imageR)
        
        cv2.imshow(_WINDOW_NAME, image)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows() 
            quit()  # esc to quit

        rate.sleep()

def main(psmName):
    rospy.init_node('dvrk_registration', anonymous=True)
    robot = psm(psmName)
   
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras( "left/image_rect",
                          "right/image_rect",
                          "left/camera_info",
                          "right/camera_info",
                          slop = slop)
    
    tfSync = CameraSync('/stereo/left/camera_info',
                        topics = ['/dvrk/' + psmName + '/position_cartesian_current'],
                        frames = [psmName + '_psm_base_link',
                                  psmName + '_tool_wrist_link',
                                  psmName + '_tool_wrist_caudier_link_shaft'])

    camModel = StereoCameraModel()
    topicLeft = rospy.resolve_name("left/camera_info")
    msgL = rospy.wait_for_message(topicLeft,CameraInfo, 10);
    topicRight = rospy.resolve_name("right/camera_info")
    msgR = rospy.wait_for_message(topicRight,CameraInfo, 10);
    camModel.fromCameraInfo(msgL,msgR)

    # Set up GUI
    filePath = rospy.get_param('~registration_yaml')
    print(filePath)
    with open(filePath, 'r') as f:
        data = yaml.load(f)
    if any (k not in data for k in ['H', 'minS', 'minV', 'maxV', 'transform', 'points']):

        rospy.logfatal('dVRK Registration: ' + filePath +
                       ' empty or malformed.')
        quit()

    cv2.namedWindow(_WINDOW_NAME)

    transformOld = np.array(data['transform'])

    toolOffset = data['toolOffset'] # distance from pinching axle to center of orange nub
    points = np.array(data['points']) # Set of points in robots frame to register against

    # Wait for registration to start
    displayRegistration(cams, camModel, toolOffset, transformOld, tfSync)
    
    print('Done')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('PSM2')
