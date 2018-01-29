#!/usr/bin/env python
import sys
import os.path
import yaml
import rospy
import cv2
import numpy as np
from collections import deque
from rigid_transform_3d import rigidTransform3D, calculateRMSE
import PyKDL
from dvrk import psm
from IPython import embed
from dvrk_vision.vtk_stereo_viewer import StereoCameras
from image_geometry import StereoCameraModel
from sensor_msgs.msg import CameraInfo

_WINDOW_NAME = "Registration"

def combineImages(imageL, imageR):
    (rows,cols) = imageL.shape[0:2]
    if len(imageL.shape) == 2:
        shape = (rows, cols*2)
    elif len(imageL.shape) == 3:
        shape = (rows, cols*2, imageL.shape[2])
    doubleImage = np.zeros(shape,np.uint8)
    doubleImage[0:rows,0:cols] = imageL
    doubleImage[0:rows,cols:cols*2] = imageR
    return doubleImage

def calculate3DPoint(imageL, imageR, camModel):
    point3d = None
    # Process left image if it exists
    (rows,cols,channels) = imageL.shape
    if cols > 60 and rows > 60 :
        maskImageL = mask(imageL)
        centerL, radiusL = getCentroid(maskImageL)

    # if it doesn't exist, don't do anything
    else:
        return None, None

    (rows,cols,channels) = imageR.shape
    if cols > 60 and rows > 60 :
        maskImageR = mask(imageR)
        centerR, radiusR = getCentroid(maskImageR)
    else:
        return None, combineImages(imageL, imageR)

    if(centerL != None and centerR != None):
        # disparity = abs(centerL[0] - centerR[0])
        disparity = centerL[0] - centerR[0]
        point3d = camModel.projectPixelTo3d(centerL,disparity)
        cv2.circle(imageL, centerL, 2,(0, 255, 0), -1)
        cv2.circle(imageR, centerR, 2,(0, 255, 0), -1)
        cv2.circle(imageL, centerL, radiusL,(0, 255, 0), 1)
        cv2.circle(imageR, centerR, radiusR,(0, 255, 0), 1)

    if cv2.getTrackbarPos('masked',_WINDOW_NAME) == 0:
        return point3d, combineImages(imageL, imageR)
    else:
        return point3d, combineImages(maskImageL, maskImageR)

def mask(img):
    # Convert to HSV and mask colors
    h = cv2.getTrackbarPos('H',_WINDOW_NAME)
    sMin = cv2.getTrackbarPos('min S',_WINDOW_NAME)
    vMin = cv2.getTrackbarPos('min V',_WINDOW_NAME)
    vMax = cv2.getTrackbarPos('max V',_WINDOW_NAME)
    colorLower = (np.max((h-15,0)), sMin, vMin)
    colorUpper = (np.min((h+15,180)), 255, vMax)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, colorLower, colorUpper )
    # Refine mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

def getCentroid(maskImage):
    # With help from http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(maskImage.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 3:
            return center, int(radius)
    # Otherwise return nonsense
    return None, None

def nothingCB(data):
    pass

def generateRandomPoint(xMinMax, yMinMax, zMinMax):
    randomPoint = [0, 0, 0]
    for idx, minMax in enumerate([xMinMax, yMinMax, zMinMax]):
        r = np.random.random_sample()
        randomPoint[idx] = r * (minMax[1] - minMax[0]) + minMax[0]
    return randomPoint

def main(psmName):
    rospy.init_node('dvrk_registration', anonymous=True)
    toolOffset = .012 # distance from pinching axle to center of orange nub

    robot = psm(psmName)
    rate = rospy.Rate(15) # 30hz
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(scriptDirectory,'..','..','defaults','registration_params.yaml')
    print(filePath)
    with open(filePath, 'r') as f:
        data = yaml.load(f)
    if 'H' not in data:
        rospy.logwarn('dVRK Registration: defaults/registration_params.yaml empty or malformed. Using defaults for orange tip')
        data = {'H': 23, 'minS': 173, 'minV': 68, 'maxV': 255, 'transform':np.eye(4).tolist()}
    
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras( "left/image_rect",
                          "right/image_rect",
                          "left/camera_info",
                          "right/camera_info",
                          slop = slop)

    camModel = StereoCameraModel()
    topicLeft = rospy.resolve_name("left/camera_info")
    msgL = rospy.wait_for_message(topicLeft,CameraInfo,3);
    topicRight = rospy.resolve_name("right/camera_info")
    msgR = rospy.wait_for_message(topicRight,CameraInfo,3);
    camModel.fromCameraInfo(msgL,msgR)

    # Set up GUI
    cv2.namedWindow(_WINDOW_NAME)
    cv2.createTrackbar('H', _WINDOW_NAME, data['H'], 180, nothingCB)
    cv2.createTrackbar('min S', _WINDOW_NAME, data['minS'], 255, nothingCB)
    cv2.createTrackbar('min V', _WINDOW_NAME, data['minV'], 255, nothingCB)
    cv2.createTrackbar('max V', _WINDOW_NAME, data['maxV'], 255, nothingCB)
    cv2.createTrackbar('masked', _WINDOW_NAME, 0, 1, nothingCB)

    # Wait for registration to begin
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
            continue

        point3d, image = calculate3DPoint(imageL, imageR, camModel)
        message = "Press s to start registration. Robot will move to its joint limits."
        cv2.putText(image, message, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, [0, 0, 255])
        message = "MAKE SURE AREA IS CLEAR"
        cv2.putText(image, message, (50,100), cv2.FONT_HERSHEY_DUPLEX, 1, [0, 0, 255])
        cv2.imshow(_WINDOW_NAME, image)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows() 
            quit()  # esc to quit
        elif chr(key%256) == 's' or chr(key%256) == 'S':
            break # s to continue
        rate.sleep()
    np.random.seed(1)
    
    # Main registration
    points = np.array([[ 0.10, 0.00,-0.15],
                       [ 0.05, 0.10,-0.18], 
                       [-0.04, 0.13,-0.15], 
                       [ 0.05, 0.05,-0.18], 
                       [-0.05,-0.02,-0.15]])
    print points
    pointsCam = np.empty(points.shape)
    for i, point in enumerate(points):
        if rospy.is_shutdown():
            quit()
        if not robot.move(PyKDL.Vector(point[0], point[1], point[2])):
            rospy.logfatal("dVRK Registration: Unable to move robot")
            quit()
        rospy.sleep(.1)
        pBuffer = deque([], 50)
        rBuffer = deque([], 50)
        zVector = robot.get_current_position().M.UnitZ()
        offset = np.array([zVector.x(), zVector.y(), zVector.z()])
        offset = offset * toolOffset
        startTime = rospy.get_time()

        while rospy.get_time() - startTime < 1:
            # Get last images
            imageL = cams.camL.image
            imageR = cams.camR.image

            point3d, image = calculate3DPoint(imageL, imageR, camModel)
            if type(image) != type(None):
                cv2.imshow(_WINDOW_NAME, image)
                key = cv2.waitKey(1)
                if key == 27: 
                    cv2.destroyAllWindows()
                    quit()
            rate.sleep()
            if point3d != None:
                pBuffer.append(point3d)
                pVector = robot.get_current_position().p
                rBuffer.append(np.array([pVector.x(), pVector.y(), pVector.z()]) + offset)
                
        pointsCam[i,:] = np.median(pBuffer,0)
        points[i,:] = np.median(rBuffer,0)
        print("Using median of %d values: (%f, %f, %f)" % (len(pBuffer),
                                                          pointsCam[i,0],
                                                          pointsCam[i,1],
                                                          pointsCam[i,2]))
    
    (rot, pos) = rigidTransform3D(np.mat(pointsCam), np.mat(points))
    calculateRMSE(np.mat(pointsCam), np.mat(points), rot, pos)

    # retval, out, inliers = cv2.estimateAffine3D(pointsCam, points)
    out = np.hstack((rot,pos))
    transform = np.matrix(np.vstack((out,[0, 0, 0, 1])))

    # Save all parameters to YAML file
    data['transform'] = transform.tolist()
    data['H'] = cv2.getTrackbarPos('H',_WINDOW_NAME)
    data['minS'] = cv2.getTrackbarPos('min S',_WINDOW_NAME)
    data['minV'] = cv2.getTrackbarPos('min V',_WINDOW_NAME)
    data['maxV'] = cv2.getTrackbarPos('max V',_WINDOW_NAME)
    with open(filePath, 'w') as f:
        yaml.dump(data,f)

    # Evaluate registration
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
            continue

        # Find 3D position of end effector
        zVector = robot.get_current_position().M.UnitZ()
        pVector = robot.get_current_position().p
        offset = np.array([zVector.x(), zVector.y(), zVector.z(), 0])
        offset = offset * toolOffset
        pos = np.matrix([pVector.x(), pVector.y(), pVector.z(), 1]) + offset;
        pos = pos.transpose()
        pos = np.linalg.inv(transform) * pos

        # Project position into 2d coordinates
        posL = camModel.left.project3dToPixel(pos)
        posL = [int(l) for l in posL]
        posR = camModel.right.project3dToPixel(pos)
        posR = [int(l) for l in posR]

        # Draw images and display them
        cv2.circle(imageL, tuple(posL), 2,(255, 255, 0), -1)
        cv2.circle(imageR, tuple(posR), 2,(255, 255, 0), -1)
        cv2.circle(imageL, tuple(posL), 7,(255, 255, 0), 2)
        cv2.circle(imageR, tuple(posR), 7,(255, 255, 0), 2)
        image = combineImages(imageL, imageR)
        cv2.imshow(_WINDOW_NAME, image)
        key = cv2.waitKey(1)
        if key == 27: 
            cv2.destroyAllWindows()
            quit()
        rate.sleep()

    print('Done')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('PSM2')
