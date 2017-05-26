#!/usr/bin/env python
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

# dependencies can be any iterable with strings, 
# e.g. file line-by-line iterator
dependencies = [
  'numpy>=1.1',
  'rospy>=1.11'
]

# here, if a dependency is not met, a DistributionNotFound or VersionConflict
# exception is thrown. 
pkg_resources.require(dependencies)

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2
import struct

def pack_bytes(array):
    # Converts an array of length 3 representing color as [r g b]
    #   and returns the three values packed into a single float
    x = (array[0] << 16) + (array[1] << 8) + array[2]
    return np.float32(struct.unpack('f', struct.pack('I', x))[0])

def get_reprojection_matrix(projectionMatrixL, projectionMatrixR, downsample):
    # Projection/camera matrix from ROS camera_info message
    #     [fx   0  cx  Tx] cx, cy are principal points of left camera in pixels
    # P = [ 0  fy  cy  Ty] fx, fy are the focal lengths in pixels. We assume they are equal
    #     [ 0   0   1   0] Tx, Ty are the baseline distances in meters multiplied by fx, fy respectively
    #
    # Q Matrix as defined by opencv
    #     [ 1  0    0           -cx       ] cx, cy are principal points of left camera
    # Q = [ 0  1    0           -cy       ] cxR is the principal x point of the right camera
    #     [ 0  0    1             f       ] f is the focal length in pixels
    #     [ 0  0  -1/tx   (cx - cxR) / tx ] tx is the baseline length in meters
    #
    # http://stackoverflow.com/questions/27374970/q-matrix-for-the-reprojectimageto3d-function-in-opencv

    Q = np.identity(4)
    Q[2,2] = 0
    fx = projectionMatrixL[0] / 2**downsample
    tx = projectionMatrixR[3] / 2**downsample
    cx = projectionMatrixL[2] / 2**downsample
    cy = projectionMatrixL[6] / 2**downsample
    cxR = projectionMatrixR[2] / 2**downsample
    Q[0,3] = -cx
    Q[1,3] = -cy
    Q[2,3] = fx
    Q[3,2] = -1 / (tx / fx)
    Q[3,3] = (cx - cxR) / tx

    return Q

class StereoProcessing:
    def __init__(self,namespace,maskTopic=None):
        # Set up empty variables for image capture
        self.imgL = np.zeros((1,1,3))
        self.imgR = np.zeros((1,1,3))
        self.mask = np.zeros((1,1,3))
        self.bridge = CvBridge()
        self.downsample = 2

        namespaceL = namespace+"/left"
        namespaceR = namespace+"/right"
        camInfoL = rospy.wait_for_message(namespaceL + "/camera_info", CameraInfo, timeout=2)
        camInfoR = rospy.wait_for_message(namespaceR + "/camera_info", CameraInfo, timeout=2)

        # Set up variables for publishing stereo points
        self.pcl_pub = rospy.Publisher(namespace+"/point_cloud", PointCloud2, queue_size=1)
        self.pcl_header = Header()
        self.pcl_header.frame_id = camInfoL.header.frame_id

        # disparity range is hardcoded for now
        window_size = 3
        min_disp = 16*int(5 / 2**self.downsample)
        num_disp = 16*int(10 / 2**self.downsample) - min_disp
        self.stereo = cv2.StereoSGBM(minDisparity = min_disp,
                                     numDisparities = num_disp,
                                     SADWindowSize = window_size,
                                     uniquenessRatio = 45,
                                     speckleWindowSize = 200,
                                     speckleRange = 10, 
                                     disp12MaxDiff = 1,
                                     P1 = 8*3*window_size**2,
                                     P2 = 32*3*window_size**2,
                                     fullDP = False
        )

        self.points = np.zeros((1,3))

        self.Q = get_reprojection_matrix(camInfoL.P,camInfoR.P, self.downsample)

        # # Build reprojection matrix from camera info http://stackoverflow.com/questions/27374970/q-matrix-for-the-reprojectimageto3d-function-in-opencv
        # self.Q = np.identity(4)
        # self.Q[2,2] = 0
        # # self.Q[0:3,0:3] = np.linalg.inv([camInfoL.R[0:3],camInfoL.R[3:6],camInfoL.R[6:9]])
        # self.Q[3,2] = -1 / (camInfoR.P[3] / camInfoR.P[0])
        # self.Q[3,3] = (camInfoL.P[2] - camInfoR.P[2]) / camInfoR.P[3]
        # self.Q[0,3] = -camInfoL.P[2] / 2**self.downsample
        # self.Q[1,3] = -camInfoL.P[6] / 2**self.downsample
        # self.Q[2,3] = camInfoL.P[0] / 2**self.downsample

        # # Build reprojection matrix from camera info
        # self.Q = np.identity(4)
        # self.Q[2,2] = 0
        # # self.Q[0:3,0:3] = np.linalg.inv([camInfoL.R[0:3],camInfoL.R[3:6],camInfoL.R[6:9]])
        # self.Q[3,2] = -camInfoR.P[3] / 2 # Get baseline
        # self.Q[0,3] = -camInfoL.P[2] / 2**self.downsample
        # self.Q[1,3] = -camInfoL.P[6] / 2**self.downsample
        # self.Q[2,3] = camInfoL.P[0] / 2**self.downsample
        
        # print self.Q


        # Set up subscribers for capturing images
        self.imgSubL = rospy.Subscriber(namespaceL + "/image_rect", Image, self.imageCbL)
        self.imgSubR = rospy.Subscriber(namespaceR + "/image_rect", Image, self.imageCbR)

        if maskTopic:
            self.maskSub = rospy.Subscriber(maskTopic, Image, self.maskCb)

    def imageCbL(self, data):
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.imgL = img.astype(np.uint8)

    def imageCbR(self, data):
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.imgR = img.astype(np.uint8)

    def update(self):
        # Make copies of images so they can be resized and not be overridden by callbacks
        imgL = self.imgL.copy()
        imgR = self.imgR.copy()

        if imgL.shape != imgR.shape or imgL.shape == (1,1,3):
            return

        if self.mask.shape[0:2] == imgL.shape[0:2]:
            maskImg = self.mask.copy()

            maskImg = maskImg.reshape(self.mask.shape[0:2])
        else:
            if self.mask.shape != (1,1,3): 
                print "Shape mismatch in mask", self.mask.shape, imgL.shape[0:2]
            maskImg = np.ones(imgL.shape[0:2],np.uint8)*255

        for i in range(0,self.downsample):
            imgL = cv2.pyrDown(imgL)  # downscale images for faster processing
            imgR = cv2.pyrDown(imgR)  # downscale images for faster processing
            maskImg = cv2.pyrDown(maskImg)


        disp = self.stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        points = cv2.reprojectImageTo3D(disp, self.Q)

        mask = np.logical_and(disp > disp.min() , maskImg > 0)
        points = points[mask]
        index = np.random.permutation(points.size/3)
        # offset = np.mean(points,0)
        self.points = points[index] # - offset

        # Don't publish anything if too few points are found
        if len(self.points) < 5:
            return

        # Get color data
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        out_colors = np.apply_along_axis(pack_bytes, 1, colors[mask][index])
        out_colors = np.reshape(out_colors, (out_colors.shape[0],1))

        # Combine colors with xyz into pointcloud message
        out_points = np.hstack((self.points,out_colors))
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb',12, PointField.FLOAT32,1)]
        self.pcl_header.stamp = rospy.Time.now()
        scaled_polygon_pcl = pcl2.create_cloud(self.pcl_header, fields, out_points)

        self.pcl_pub.publish(scaled_polygon_pcl)

    def maskCb(self,data):
        self.mask = self.bridge.imgmsg_to_cv2(data, "mono8")


if __name__ == '__main__':

    # Initialize the node
    rospy.init_node("stereo_processing")
    rate = rospy.Rate(30) # 30hz
    
    stereo = StereoProcessing("/stereo")

    while not rospy.is_shutdown():
        rate.sleep()