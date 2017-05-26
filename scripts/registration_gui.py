#!/usr/bin/env python
import sys
import os
from PyQt4 import QtGui, uic
from PyQt4.QtCore import QThread
import vtk
import numpy as np
import rospy
import cv2
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtktools
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from graph_cut_node import SegmentedImage
from cv_bridge import CvBridge, CvBridgeError
from tf import transformations

functionPath = os.path.dirname(os.path.realpath(__file__))

class RosThread(QThread):
    # Qt thread designed to allow ROS to run in the background
    def __init__(self, parent = None):
        QThread.__init__(self, parent)
        # Initialize the node
        if rospy.get_node_uri() == None:
            rospy.init_node("vtk_test")
        self.rate = rospy.Rate(30) # 30hz
    def run(self):
        while not rospy.is_shutdown():
            self.update()
            self.rate.sleep()
    def update(self):
        pass

class vtkTimerCallback():
   # Callback that renders on a timer
   def __init__(self):
       self.timer_count = 0
   def execute(self,obj,event):
       obj.GetRenderWindow().Render()

def vtkCameraFromCamInfo(camInfo):
    # Get intrinsic matrix of rectified image
    intrinsicMatrix = np.reshape(camInfo.P,(3,4))[0:3,0:3]
    # Get camera extrinsic matrix
    extrinsicMatrix = np.identity(4)
    # Get extrinsic rotation
    extrinsicMatrix [0:3,0:3] = np.reshape(camInfo.R,(3,3))
    # Get baseline translation (will usually be zero as this is for left camera)
    xBaseline = camInfo.P[3] / -camInfo.P[0]
    yBaseline = camInfo.P[7] / -camInfo.P[5]
    extrinsicMatrix [0:3,3] = [xBaseline, yBaseline, 0]

    return intrinsicMatrix, extrinsicMatrix

class RegistrationInteractorStyle(vtk.vtkInteractorStyle):
    # Interactor style for masking
    def __init__(self, segmentedImage, parent=None):
        self.segmentedImage = segmentedImage
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPressEvent)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonReleaseEvent)
        self.AddObserver("LeftButtonPressEvent",self.leftButtonPressEvent)
        self.AddObserver("LeftButtonReleaseEvent",self.leftButtonReleaseEvent)
        self.AddObserver("MouseMoveEvent",self.mouseMoveEvent)

    def middleButtonPressEvent(self,obj,event):
        return
 
    def middleButtonReleaseEvent(self,obj,event):
        return

    def leftButtonPressEvent(self,obj,event):
        event = cv2.EVENT_LBUTTONDOWN
        (x,y) = self.getMousePos()
        self.segmentedImage.onMouse(event,x,y,0,0)

    def leftButtonReleaseEvent(self,obj,event):
        event = cv2.EVENT_LBUTTONUP
        (x,y) = self.getMousePos()
        self.segmentedImage.onMouse(event,x,y,0,0)

    def mouseMoveEvent(self,obj,event):
        event = cv2.EVENT_MOUSEMOVE
        (x,y) = self.getMousePos()
        self.segmentedImage.onMouse(event,x,y,0,0)

    def getMousePos(self):
        iren = self.GetInteractor ()
        pos = iren.GetEventPosition()
        windowShape =  iren.GetSize()
        imageShape = self.segmentedImage.image.shape
        ratio = imageShape[0] / float(windowShape[1])
        offset = (windowShape[0] * ratio - imageShape[1]) / 2.0
        x = int(pos[0] * ratio - offset)
        y = imageShape[0] - int(pos[1] * ratio)
        return (x,y)

class RegistrationWindow(QtGui.QMainWindow):
    def __init__(self, stlPath, scale=0.001, namespace="/stereo", parentWindow=None):

        super(RegistrationWindow, self).__init__()
        uic.loadUi(functionPath + "/registration_gui.ui", self)

        # Check whether this is the left (primary) or the right (secondary) window
        self.isPrimaryWindow = parentWindow == None
        side = "left" if self.isPrimaryWindow else "right"
        if self.isPrimaryWindow:
            # Connect buttons to functions
            self.registerButton.clicked.connect(self.register)

        # RosThread.update = self.update
        self.rosThread = RosThread()

        # Set up subscriber for camera image
        self.bridge = CvBridge()
        imgSubTopic = namespace + "/"+side+"/image_rect"
        imageSub = rospy.Subscriber(imgSubTopic, Image, self.imageCallback)

        # Set up vtk background image
        msg = rospy.wait_for_message(imgSubTopic, Image, timeout=2)
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.imgDims = image.shape
        self.segmentation = SegmentedImage()
        self.segmentation.setImage(image)

        # Set up subscriber for registered organ position
        poseSubTopic = namespace + "/organMarker"
        poseSub = rospy.Subscriber(poseSubTopic, Marker, self.poseCallback)

        # Set up publisher for masking
        pubTopic = namespace + "/" + side + "/image_rect_mask"
        self.maskPub = rospy.Publisher(pubTopic, Image, queue_size=10)

        # Add vtk widget
        self.vl = QtGui.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.vtkFrame)
        self.vl.addWidget(self.vtkWidget)
        self.vtkFrame.setLayout(self.vl)

        # Set up vtk camera using camera info
        self.bgImage = vtktools.makeVtkImage(self.imgDims[0:2])
        self.renWin = self.vtkWidget.GetRenderWindow()
        camInfo = rospy.wait_for_message(namespace + "/" + side + "/camera_info", CameraInfo, timeout=2)
        intrinsicMatrix, extrinsicMatrix = vtkCameraFromCamInfo(camInfo)
        self.ren, self.bgRen = vtktools.setupRenWinForRegistration(self.renWin, self.bgImage,intrinsicMatrix)
        pos = extrinsicMatrix[0:3,3]
        self.ren.GetActiveCamera().SetPosition(pos)
        pos[2] = 1
        self.ren.GetActiveCamera().SetFocalPoint(pos)
        self.zBuff = vtktools.zBuff(self.renWin)

        # Set up 3D actor for organ
        self.stlReader = vtk.vtkSTLReader()
        self.stlReader.SetFileName(stlPath)
        self.stlReader.Update()
        transform = vtk.vtkTransform()
        transform.Scale(scale,scale,scale)
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(self.stlReader.GetOutputPort())
        self.actor_moving = vtk.vtkActor()
        self.actor_moving.GetProperty().SetOpacity(0.35)
        self._updateActorPolydata(self.actor_moving, transformFilter.GetOutput(), (0,1, 0))
        self.ren.AddActor(self.actor_moving)

        # Setup interactor
        self.iren = self.renWin.GetInteractor()
        if self.isPrimaryWindow:
            self.iren.SetInteractorStyle(RegistrationInteractorStyle(self.segmentation))
        else:
            self.iren.SetInteractorStyle(RegistrationInteractorStyle(parentWindow.segmentation))
        self.show()
        self.iren.Initialize()
        # Set up timer to refresh render
        cb = vtkTimerCallback()
        self.iren.AddObserver('TimerEvent', cb.execute)
        timerId = self.iren.CreateRepeatingTimer(30);

        self.rosThread.start()
        self.iren.Start()

    def imageCallback(self, data):
        # TODO no try catch
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.isPrimaryWindow:
                self.segmentation.setImage(image)
                vtktools.numpyToVtkImage(self.segmentation.getMaskedImage(),self.bgImage)
                self.maskPub.publish(self.bridge.cv2_to_imgmsg(self.segmentation.mask*255, "mono8"))
            else:
                vtktools.numpyToVtkImage(image,self.bgImage)
        except:
            pass

    def poseCallback(self, data):
        pos = data.pose.position
        rot = data.pose.orientation
        mat = transformations.quaternion_matrix([rot.x,rot.y,rot.z,rot.w])
        mat[0:3,3] = [pos.x,pos.y,pos.z]
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.SetMatrix(mat.ravel())
        self.actor_moving.SetUserTransform(transform)        

    def _updateActorPolydata(self,actor,polydata,color):
        # Modifies an actor with new polydata
        bounds = polydata.GetBounds()

        # Visualization
        mapper = actor.GetMapper()
        if mapper == None:
            mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(polydata)
        else:
            mapper.SetInputData(polydata)
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])

    def register(self):
        # TODO
        pass

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    stlPath = functionPath+"/../defaults/femur.stl"
    windowL = RegistrationWindow(stlPath)
    windowR = RegistrationWindow(stlPath, parentWindow=windowL)
    sys.exit(app.exec_())