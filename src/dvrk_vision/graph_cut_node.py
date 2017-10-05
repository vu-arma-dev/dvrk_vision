#!/usr/bin/env python
import numpy as np
import rospy
import cv2
import tf2_ros
from tf import transformations
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError
from time import time

class SegmentedImage:
    def __init__(self):
        # Set up blank image for left camera to update
        self.image = np.zeros((1,1,3),np.uint8)
        self.mask = np.ones((1,1),np.uint8)
        self.paint = np.zeros((1,1),np.uint8)
        self.mode = "PAINT"
        self.crop = [0,0,0,0]
        self.drawing = False
        # Double click variables
        self.numberOfClicks = 0
        self.previousPos = (-100,-100)
        self.doublClickTolerance = 5

    def onMouse(self, event, x, y, flags, params):
        t = time()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True

            # Check for double-click to switch to "paint" mode
            self.numberOfClicks += 1
            dist = int(np.linalg.norm(np.array((x,y))-self.previousPos))
            self.previousPos = (x,y)
            if dist > self.doublClickTolerance:
                self.numberOfClicks = 1
            if self.numberOfClicks >= 2:
                self.numberOfClicks = 0
                self.mode="PAINT"
                cv2.circle(self.paint, (x,y),20,1,-1)
            # Otherwise just do a rectancular mask
            else:
                self.crop[0] = x
                self.crop[1] = y
                self.crop[2] = x
                self.crop[3] = y
                mask = np.zeros(self.mask.shape, np.uint8)
                self.mode="RECT"

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.mode=="PAINT":
                cv2.circle(self.paint, (x,y),20,1,-1)
            elif self.drawing and self.mode=="RECT":
                self.crop[2] = x
                self.crop[3] = y

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            if self.mode == "PAINT":
                mask = self.paint.copy()
                mask = mask*self.mask
                cv2.grabCut(self.image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
                self.mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                self.paint = np.ones(self.paint.shape, np.uint8)*2

            if self.mode == "RECT":
                dist = int(np.linalg.norm(np.array((x,y))-self.previousPos))
                if(dist<self.doublClickTolerance):
                    return
                cropA = (self.crop[0],self.crop[1])
                cropB = (self.crop[2],self.crop[3])
                self.paint = np.ones(self.paint.shape, np.uint8)*2
                self.mask = np.zeros(self.mask.shape, np.uint8)
                cv2.rectangle(self.mask, cropA,cropB, 1, -1)

    def setImage(self, image):
        self.image = image
        shape = (self.image.shape[0],self.image.shape[1])
        if self.mask.shape != shape:
            self.mask = np.ones(shape, np.uint8)
            self.paint = np.ones(shape, np.uint8)*2

    def getMaskedImage(self):
        img = self.image.copy()
        # img = cv2.blur(img, (3,3))
        # img[:,:,0] = img[:,:,0]*self.mask
        # img[:,:,1] = img[:,:,1]*self.mask
        img[:,:,0] = img[:,:,0]*self.mask[:,:]
        img[:,:,1] = img[:,:,1]*self.mask[:,:]
        color = np.zeros(img.shape,np.uint8)
        paintMask = np.zeros(self.paint.shape,np.uint8)
        paintMask = np.where(self.paint == 2, paintMask,1)
        cv2.bitwise_not(color,img,paintMask)
        if self.drawing == True and self.mode=="RECT":
            cropA = (self.crop[0],self.crop[1])
            cropB = (self.crop[2],self.crop[3])
            cv2.rectangle(img, cropA,cropB, (0,255,0), 1, 8, 0 )
        return img

class ImageSegmenter:
    def __init__(self, camImageTopicL, camImageTopicR=None, scale=0.5):

        self.bridge = CvBridge()
        self.L = SegmentedImage()
        self.scale = scale
        self.width = 0
        self.twoImages = camImageTopicR != None

        msg = rospy.wait_for_message(camImageTopicL, Image, timeout=10)
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.L.setImage(image)
        # Set up subscribers for camera images
        imageSubL = rospy.Subscriber(camImageTopicL,Image,self.imageCallbackL)
        pubTopic = camImageTopicL + "_mask"
        self.maskPubL = rospy.Publisher(pubTopic, Image, queue_size=10)

        if self.twoImages:
            self.R = SegmentedImage()
            msg = rospy.wait_for_message(camImageTopicR, Image, timeout=2)
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.R.setImage(image)
            # Set up subscribers for camera images
            imageSubR = rospy.Subscriber(camImageTopicR,Image,self.imageCallbackR)
            pubTopic = camImageTopicR + "_mask"
            self.maskPubR = rospy.Publisher(pubTopic, Image, queue_size=10)

        #msgL = rospy.wait_for_message(camImageTopicL, Image, timeout=2)
        #self.imageCallbackL(msgL)

    def update(self):
        shape = (self.L.image.shape[0],self.L.image.shape[1])
        self.width = shape[1]
        shape = (int(shape[1]*self.scale),int(shape[0]*self.scale))
        # Update and publish mask image (L)
        imgLarge = self.L.getMaskedImage()
        img =  cv2.resize(imgLarge,shape)
        self.maskPubL.publish(self.bridge.cv2_to_imgmsg(self.L.mask*255, "mono8"))
        if(self.twoImages):
            # Update and publish mask image (R)
            img2 = cv2.resize(self.R.getMaskedImage(),shape)
            self.maskPubR.publish(self.bridge.cv2_to_imgmsg(self.R.mask*255, "mono8"))
            img = np.concatenate((img, img2), axis=1)
        cv2.imshow('Segmentation', img)
        (cv_major, cv_minor, _) = cv2.__version__.split(".")
        if cv_major < 3:
            cv2.cv.SetMouseCallback('Segmentation', self.onMouse, 0)
        else:
            cv2.SetMouseCallback('Segmentation', self.onMouse, 0)
        cv2.waitKey(30)

    def imageCallbackL(self,data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.L.setImage(image)

    def imageCallbackR(self,data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.R.setImage(image)

    def onMouse(self, event, x, y, flags, params):
        x = int(x/self.scale)
        y = int(y/self.scale)
        if self.twoImages:
            if(x > self.width):
                 self.R.onMouse(event,x-self.width,y,flags,params)
            else:
                 self.L.onMouse(event, x, y, flags, params)
        else:
            self.L.onMouse(event, x, y, flags, params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = 'Register a point cloud to an STL using dual quaternions')
    args, unknown = parser.parse_known_args()
    # Initialize the node
    rospy.init_node("image_segmenter")
    rate = rospy.Rate(10) # 5hz
    try:
        camImageTopicL = rospy.get_param('~image_topic_l')
    except KeyError:
        raise KeyError("Parameter image_topic_l is not set. Killing node")
    try:
        camImageTopicR = rospy.get_param('~image_topic_r')
    except KeyError:
        camImageTopicR = None

    reg = ImageSegmenter(camImageTopicL, camImageTopicR)
    while not rospy.is_shutdown():
        reg.update()
        rate.sleep()