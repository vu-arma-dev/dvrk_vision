#!/usr/bin/env python
import yaml
import numpy as np
import rospy
import cv2
import tf2_ros
import geometry_msgs.msg
from tf import transformations
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError

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

def getChessboardTransform(img, camMatrix, camDistort, board_shape, board_size, showImage = False):
    pattern = np.zeros( (np.prod(board_shape),1, 3), np.float32 )
    pattern[:,:,:2] = np.indices(board_shape).T.reshape(-1, 1, 2)
    pattern *= board_size
    k = 0
    boards_found = 0
    started = False
    # Find chessboard
    k = cv2.waitKey(1)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(img, board_shape)

    if found:
        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), term)
        cv2.drawChessboardCorners(img, board_shape, corners, found)
        #if k & 0xFF == ord(' '):
        # Find the rotation and translation vectors.
        if(cv2.__version__[0] == '2'):
            rots, trans, inliers = cv2.solvePnPRansac(pattern, corners, camMatrix, camDistort)
        else:
            retval,rots, trans, inliers = cv2.solvePnPRansac(pattern, corners, camMatrix, camDistort)
        rotation = rots.ravel()
        translation = trans.ravel()

        # Create a 3D axis for visualizing chessboard transform
        axis = np.float32([[board_size,0,0], [0,board_size,0], [0,0,board_size]])
        axis = axis.reshape(-1,3)
        imgpts, jac = cv2.projectPoints(axis, rotation, translation, camMatrix, camDistort)
        drawAxes(img,corners,imgpts)
        if showImage:
            cv2.imshow('image', img)
        return rotation, translation, corners

    else: 
        if showImage:
            cv2.imshow('image', img)
        return None, None, None

def normalized(vec):
    ''' Normalizes a vector'''
    mag = np.linalg.norm(vec)
    return vec/mag

class ChessboardPublisher:
    def __init__(self, boardW, boardH, boardSize, camInfoTopic, camImageTopic, camFrame, camCalibration):
        self.boardW = boardW
        self.boardH = boardH
        self.boardSize = boardSize
        self.camFrame = camFrame

        with open(camCalibration, 'r') as stream:
            try:
                camParams=yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        camMatrix = np.matrix(camParams['camera_matrix']['data'])
        shape = (camParams['camera_matrix']['rows'],camParams['camera_matrix']['cols'])
        self.camMatrix = np.reshape(camMatrix, shape)
        camDistort = np.matrix(camParams['distortion_coefficients']['data'])
        shape = (camParams['distortion_coefficients']['rows'],camParams['distortion_coefficients']['cols'])
        self.camDistort = np .reshape(camDistort, shape)

        # Initialize TF listener
        self.broadcaster = tf2_ros.TransformBroadcaster()

        self.camInfo = rospy.wait_for_message(camInfoTopic,CameraInfo,2);

        # Set up blank image for left camera to update
        self.image = np.zeros((1,1,3),np.uint8)
        self.bridge = CvBridge()

        # Set up subscribers for camera images
        image_sub = rospy.Subscriber(camImageTopic,Image,self.imageCallback)

        topic = 'chessboard_marker_array'

        # # Set up publisher for rectified image
        # self.image_pub = rospy.Publisher("stereo/left/img_rect",Image, queue_size=10)
        # self.cam_info_pub = rospy.Publisher("stereo/left/img_rect/camera_info",CameraInfo, queue_size=10)

        # Set up chessboard markers
        self.markerPub = rospy.Publisher(topic, MarkerArray, queue_size=10)

        self.markerArray = MarkerArray()

        marker_id = 0
        for i in range(0, boardW+1):
          for j in range(0, boardH+1):
            if j % 2 != i % 2:
              continue
            marker = Marker()
            marker.header.frame_id = "/chessboard"
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.scale.x = boardSize
            marker.scale.y = boardSize
            marker.scale.z = 0.0001
            marker.color.a = 1.0
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 0
            marker.pose.position.y = boardSize * (j+.5) - boardSize
            marker.pose.position.x = boardSize * (i+.5) - boardSize
            marker.id = marker_id
            self.markerArray.markers.append(marker)
            marker_id +=1

    def imageCallback(self,data):
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Find chessboard and publish TF and marker array
        rot, trans, corners = getChessboardTransform(img, self.camMatrix,
                                                     self.camDistort,
                                                     (self.boardW, self.boardH),
                                                     self.boardSize, True)
        if(rot != None):
            quat = transformations.quaternion_from_euler(rot[0],rot[1], rot[2])
            sendTF("chessboard", self.broadcaster, quat, trans, self.camFrame)
            self.markerPub.publish(self.markerArray)

        # # Rectify image and publish it
        # newCamMatrix, roi = cv2.getOptimalNewCameraMatrix(self.camMatrix,
        #                                                   self.camDistort,
        #                                                   (800,600), 1)

        # map1, map2 = cv2.initUndistortRectifyMap(self.camMatrix,self.camDistort,
        #                                          np.identity(3),newCamMatrix,
        #                                          (800,600),cv2.CV_32FC2)

        # img = cv2.remap(img, map1,map2, cv2.INTER_LINEAR)
        # try:
        #     msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        #     msg.header = data.header
        #     msg.header.stamp = rospy.Time.now()
        #     self.image_pub.publish(msg)
        #     self.camInfo.header.stamp = msg.header.stamp
        #     self.camInfo.K = np.ravel(newCamMatrix)
        #     P = np.array(self.camInfo.P)
        #     P = np.reshape(P,(3,4))
        #     P[0:3,0:3] = newCamMatrix
        #     self.camInfo.P = np.ravel(P)
        #     self.camInfo.header.frame_id = "stereo_left_frame"
        #     self.cam_info_pub.publish(self.camInfo)
        # except CvBridgeError as e:
        #     print(e)



if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser(description = 'Publish a detected chessboard transform to TF')
    parser.add_argument("-s","--size", help="Size of one square on the board. Default is .025",
                        type = float, default = 0.03048)
    parser.add_argument("--width", help="Number of inner corners along the width of board. Default is 8",
                        type = int, default=8)
    parser.add_argument("--height", help="Number of inner corners along the height of board. Default is 6",
                        type = int, default=6)
    parser.add_argument("-c","--calibration_file", 
                        help="YAML file containing camera calibration data. Default is 'stereo_cam0_calibration.yaml' found in package's 'defaults' directory",
                        type = str, default=None)
    parser.add_argument("-p", "--parent_frame", help="The name of the TF frame representing the camera from which the chessboard is detected. The default is '/camera'",
                        type = str, default="/stereo_left_frame")
    args, unknown = parser.parse_known_args()
    boardW = args.width
    boardH = args.height
    boardSize = args.size
    yamlFile = args.calibration_file
    frame = args.parent_frame

    if yamlFile == None:
        functionPath = os.path.dirname(os.path.realpath(__file__))
        yamlFile = os.path.join(functionPath,"..","defaults","stereo_cam0_calibration.yaml")

    # Initialize the node
    rospy.init_node("chessboard_publisher")
    rate = rospy.Rate(10) # 5hz
    camInfoTopic = "/stereo/left/camera_info"
    camImageTopic = "/stereo/left/image_raw/"
    chessboardPub = ChessboardPublisher(boardW, boardH, boardSize, camInfoTopic, camImageTopic, frame, yamlFile)
    rospy.spin()
    #publishChessboardTF)
    
        
    