#!/usr/bin/env python
import sys
import os.path
import yaml
import rospy
import cv2
import numpy as np
from collections import deque
from image_geometry import StereoCameraModel
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import PyKDL
from dvrk import psm

_WINDOW_NAME = "Registration"

def combineImages(imageL, imageR):
    (rows,cols,channels) = imageL.shape
    doubleImage = np.zeros((rows,cols*2,channels),np.uint8)
    doubleImage[0:rows,0:cols] = imageL
    doubleImage[0:rows,cols:cols*2] = imageR
    return doubleImage

def calculate3DPoint(imageL, imageR, camModel):
    point3d = None
    # Process left image if it exists
    (rows,cols,channels) = imageL.shape
    if cols > 60 and rows > 60 :
        maskImageL = mask(imageL)
        centerL = getCentroid(maskImageL)

    # if it doesn't exist, don't do anything
    else:
        return None, None

    (rows,cols,channels) = imageR.shape
    if cols > 60 and rows > 60 :
        maskImageR = mask(imageR)
        centerR = getCentroid(maskImageR)
    else:
        return None, combineImages(imageL, imageR)
    if(centerL != None and centerR != None):
        point3d = camModel.projectPixelTo3d(centerL,centerL[0] - centerR[0])
        cv2.circle(imageL, centerL, 2,(0, 255, 0), -1)
        cv2.circle(imageR, centerR, 2,(0, 255, 0), -1)

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
            return center
    # Otherwise return nonsense
    return None

class StereoCameras:
    def __init__(self, imageTopic="image_rect"):
        ns = rospy.get_namespace()
        print(ns)
        if ns == "/":
            rospy.logwarn("Node started in default namespace.\n\t"+
                          "This is probably a mistake.\n\t" +
                          "This node's namespace should look something like /stereo/");
        self.bridge = CvBridge()
        # Create camera model for calculating 3d position
        self.camModel = StereoCameraModel()
        topicLeft = rospy.resolve_name("left/camera_info")
        msgL = rospy.wait_for_message(topicLeft,CameraInfo,3);
        topicRight = rospy.resolve_name("right/camera_info")
        msgR = rospy.wait_for_message(topicRight,CameraInfo,3);
        self.camModel.fromCameraInfo(msgL,msgR)
        # Set up subscribers for camera images
        topicLeft = rospy.resolve_name("left/" + imageTopic)
        self.imageSubR = rospy.Subscriber(topicLeft, Image, self.imageCallbackR)
        topicRight = rospy.resolve_name("right/" + imageTopic)
        self.imageSubL = rospy.Subscriber(topicRight, Image, self.imageCallbackL)
        # Set up blank image for left camera to update
        self.imageL = None
        self.imageR = None

    def imageCallbackL(self,data):
        try:
            self.imageL = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e

    def imageCallbackR(self,data):
        try:
            self.imageR = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e


def nothingCB(data):
    pass

def main(psmName):
    rospy.init_node('dvrk_registration', anonymous=True)
    toolOffset = .012 # distance from pinching axle to center of orange nub

    robot = psm(psmName)
    rate = rospy.Rate(15) # 30hz

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(scriptDirectory,'..','defaults','registration_params.yaml')
    print(filePath)
    with open(filePath, 'r') as f:
        data = yaml.load(f)
    if 'H' not in data:
        rospy.logwarn('dVRK Registration: defaults/registration_params.yaml empty or malformed. Using defaults for orange tip')
        data = {'H': 23, 'minS': 173, 'minV': 68, 'maxV': 255, 'transform':np.eye(4).tolist()}
    cams = StereoCameras()

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
        imageR = cams.imageR
        imageL = cams.imageL

        # Wait for images to exist
        if type(imageR) == type(None) or type(imageL) == type(None):
            continue

        # Check we have valid images
        (rows,cols,channels) = imageL.shape
        if cols < 60 or rows < 60 or imageL.shape != imageR.shape:
            continue

        point3d, image = calculate3DPoint(imageL, imageR, cams.camModel)
        message = "Press s to start registration. Robot will move to its joint limits."
        cv2.putText(image, message, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, [0, 0, 255])
        message = "MAKE SURE AREA IS CLEAR"
        cv2.putText(image, message, (50,100), cv2.FONT_HERSHEY_DUPLEX, 1, [0, 0, 255])
        cv2.imshow(_WINDOW_NAME, image)
        key = cv2.waitKey(1)
        if key == 27 or key == -1:
            cv2.destroyAllWindows() 
            quit()  # esc to quit
        elif chr(key%256) == 's' or chr(key%256) == 'S':
            break # s to continue
        rate.sleep()

    # Main registration
    points = np.array([[ 0.00, 0.00,-0.10],
                       [ 0.08, 0.08,-0.15], 
                       [-0.08, 0.08,-0.10], 
                       [ 0.00, 0.00,-0.15], 
                       [ 0.00,-0.05,-0.10]])
    pointsCam = np.empty(points.shape)
    for i, point in enumerate(points):
        if rospy.is_shutdown():
            quit()
        if not robot.move(PyKDL.Vector(point[0], point[1], point[2])):
            rospy.logfatal("dVRK Registration: Unable to move robot")
            quit()
        rospy.sleep(.1)
        pBuffer = deque([], 50)
        zVector = robot.get_current_position().M.UnitZ()
        pVector = robot.get_current_position().p
        offset = np.array([zVector.x(), zVector.y(), zVector.z()])
        offset = offset * toolOffset
        points[i,:] = np.array([pVector.x(), pVector.y(), pVector.z()]) + offset
        startTime = rospy.get_time()

        while rospy.get_time() - startTime < 1:
            # Get last images
            imageR = cams.imageR
            imageL = cams.imageL
            point3d, image = calculate3DPoint(imageL, imageR, cams.camModel)
            if type(image) != type(None):
                cv2.imshow(_WINDOW_NAME, image)
                key = cv2.waitKey(1)
                if key == 27 or key == -1: 
                    cv2.destroyAllWindows()
                    quit()
            rate.sleep()
            if point3d != None:
                pBuffer.append(point3d)
        pointsCam[i,:] = np.median(pBuffer,0)
        print("Using median of %d values: (%f, %f, %f)" % (len(pBuffer),
                                                          pointsCam[i,0],
                                                          pointsCam[i,1],
                                                          pointsCam[i,2]))

    retval, out, inliers = cv2.estimateAffine3D(pointsCam, points)

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
        imageR = cams.imageR
        imageL = cams.imageL

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
        posL = cams.camModel.left.project3dToPixel(pos)
        posL = [int(l) for l in posL]
        posR = cams.camModel.right.project3dToPixel(pos)
        posR = [int(l) for l in posR]

        # Draw images and display them
        cv2.circle(imageL, tuple(posL), 2,(255, 255, 0), -1)
        cv2.circle(imageR, tuple(posR), 2,(255, 255, 0), -1)
        image = combineImages(imageL, imageR)
        cv2.imshow(_WINDOW_NAME, image)
        key = cv2.waitKey(1)
        if key == 27 or key == -1: 
            cv2.destroyAllWindows()
            quit()
        rate.sleep()

    print('Done')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('PSM2')
