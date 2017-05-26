#!/usr/bin/env python
import rospy
import tf2_ros
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import geometry_msgs.msg
from tf import transformations

def drawAxes(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    
def axisFromPoints(points,center):
    axis = np.zeros(3)
    for i in range(1,len(points)):
        vec = points[i] - center
        vec = normalized(vec)
        axis += vec
    axis = normalized(axis)
    return axis

def makeTransformMsg(translation, quaternion, name, parent):
    msg = geometry_msgs.msg.TransformStamped()
    msg.header.frame_id = parent
    msg.child_frame_id = name
    msg.transform.translation.x = translation[0]
    msg.transform.translation.y = translation[1]
    msg.transform.translation.z = translation[2]
    msg.transform.rotation.x = quaternion[0]
    msg.transform.rotation.y = quaternion[1]
    msg.transform.rotation.z = quaternion[2]
    msg.transform.rotation.w = quaternion[3]
    return msg

def sendTF(name, broadcaster, rotation, translation, parent):    
    tChessboard = makeTransformMsg(translation, rotation, name, parent)
    # while not rospy.is_shutdown():
    timeStamp = rospy.Time.now()
    tChessboard.header.stamp = timeStamp
    broadcaster.sendTransform(tChessboard)

class DVRKRegistration:
	def __init__(self):
		camInfo = rospy.wait_for_message(camInfoTopic,CameraInfo,2);
		self.frameID = camInfo.header.frame_id
		self.camMatrix = np.reshape(camInfo.K, (3,3))
		self.camDistort = np.array(camInfo.D)
		print camDistort



