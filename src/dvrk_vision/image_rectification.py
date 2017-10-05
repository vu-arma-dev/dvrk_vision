#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

def makeRectificationMaps(camInfo):
    matrix = np.reshape(np.array(camInfo.K),(3,3))
    matrixNew = np.reshape(np.array(camInfo.P),(3,4))
    distort = np.array(camInfo.D)
    rotation = np.reshape(np.array(camInfo.R),(3,3))
    shape = (camInfo.width, camInfo.height)
    mapx,mapy = cv2.initUndistortRectifyMap(matrix,distort,rotation,matrixNew,shape,cv2.CV_16SC2)
    return mapx,mapy

class ImageRectification:
    def __init__(self,imageTopic):
    	namespace = imageTopic.rsplit('/', 1)[0]
        self.bridge = CvBridge()
    	self.imgPub = rospy.Publisher(namespace + "/image_rect", Image, queue_size=10)
        camInfo = rospy.wait_for_message(namespace + "/camera_info", CameraInfo, timeout=2)
        # Build rectification mapping from camera info
        self.xMap, self.yMap = makeRectificationMaps(camInfo)
        # Set up subscribers for capturing images
        self.imgSub = rospy.Subscriber(imageTopic, Image, self.imageCb)

    def imageCb(self, data):
    	img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        imgRect = cv2.remap(img,self.xMap,self.yMap,cv2.INTER_LINEAR).astype(np.uint8)
        newMsg = self.bridge.cv2_to_imgmsg(imgRect, "bgr8")
        newMsg.header = data.header
        self.imgPub.publish(newMsg)

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("image_rectification")
    rate = rospy.Rate(30) # 5hz
    try:
        imageTopic = rospy.get_param('~image_topic')
    except KeyError:
        raise KeyError("Parameter image_topic is not set. Killing node")

    reg = ImageRectification(imageTopic)
    while not rospy.is_shutdown():
        rate.sleep()