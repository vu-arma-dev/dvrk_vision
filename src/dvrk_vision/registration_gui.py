#!/usr/bin/env python
import sys
import os
import vtk
import numpy as np
import rospy
import rospkg
import cv2
# Which PyQt we use depends on our vtk version. QT4 causes segfaults with vtk > 6
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
    from PyQt5 import uic
    from PyQt5.QtCore import QThread
    _QT_VERSION = 5
    from QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
else:
    from PyQt4.QtGui import QWidget, QVBoxLayout, QApplication
    from PyQt4 import uic
    from PyQt4.QtCore import QThread
    _QT_VERSION = 4
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtktools
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from graph_cut_node import SegmentedImage
from cv_bridge import CvBridge, CvBridgeError
from tf import transformations

from IPython import embed

def cleanResourcePath(path):
    newPath = path
    if path.find("package://") == 0:
        newPath = newPath[len("package://"):]
        pos = newPath.find("/")
        if pos == -1:
            rospy.logfatal("%s Could not parse package:// format", path)

        package = newPath[0:pos]
        newPath = newPath[pos:]
        package_path = rospkg.RosPack().get_path(package)

        if package_path == "":
            rospy.logfatal("%s Package [%s] does not exist",
                           path.c_str(),
                           package.c_str());

        newPath = package_path + newPath;
    elif path.find("file://") == 0:
        newPath = newPath[len("file://"):]

    if not os.path.isfile(newPath):
        rospy.logfatal("%s file does not exist", newPath)
    return newPath;

class RegistrationWidget(QWidget):

    def __init__(self, meshPath, scale=1, namespace="/stereo", parentWindow=None):

        super(RegistrationWidget, self).__init__()
        functionPath = os.path.dirname(os.path.realpath(__file__))
        uic.loadUi(functionPath + "/registration_gui.ui", self)

        # Check whether this is the left (primary) or the right (secondary) window
        self.isPrimaryWindow = parentWindow == None
        side = "left" if self.isPrimaryWindow else "right"

        if self.isPrimaryWindow:
            # Set up publisher for masking
            pubTopic = namespace + "/disparity_mask"
            self.maskPub = rospy.Publisher(pubTopic, Image, queue_size=1)
        
        # Set up vtk background image
        self.bridge = CvBridge()
        imgSubTopic = namespace + "/"+side+"/image_rect"
        msg = rospy.wait_for_message(imgSubTopic, Image)
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.isPrimaryWindow:
            self.segmentation = SegmentedImage()
            self.segmentation.setImage(image)
        else:
            self.segmentation = parentWindow.segmentation

        # Add vtk widget
        self.vl = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.vtkFrame)
        self.vl.addWidget(self.vtkWidget)
        self.vtkFrame.setLayout(self.vl)

        # Set up vtk camera using camera info
        self.bgImage = vtktools.makeVtkImage(image.shape[0:2])
        self.renWin = self.vtkWidget.GetRenderWindow()
        camInfo = rospy.wait_for_message(namespace + "/" + side + "/camera_info",
                                         CameraInfo, timeout=2)
        intrinsicMatrix, extrinsicMatrix = vtktools.matrixFromCamInfo(camInfo)
        self.ren, self.bgRen = vtktools.setupRenWinForRegistration(self.renWin,
                                                                   self.bgImage,
                                                                   intrinsicMatrix)
        pos = extrinsicMatrix[0:3,3]
        self.ren.GetActiveCamera().SetPosition(pos)
        pos[2] = 1
        self.ren.GetActiveCamera().SetFocalPoint(pos)

        # Set up 3D actor for organ
        meshPath = cleanResourcePath(meshPath)
        extension = os.path.splitext(meshPath)[1]
        if extension == ".stl" or extension == ".STL":
            meshReader = vtk.vtkSTLReader()
        elif extension == ".obj" or extension == ".OBJ":
            meshReader = vtk.vtkOBJReader()
        else:
            ROS_FATAL("Mesh file has invalid extension (" + extension + ")")
        meshReader.SetFileName(meshPath)
        # Scale STL
        transform = vtk.vtkTransform()
        transform.Scale(scale,scale,scale)
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(meshReader.GetOutputPort())
        transformFilter.Update()
        self.actor_moving = vtk.vtkActor()
        self.actor_moving.GetProperty().SetOpacity(1)
        self._updateActorPolydata(self.actor_moving,
                                  polydata=transformFilter.GetOutput(),
                                  color=(0,1, 0))
        # Hide actor
        self.actor_moving.VisibilityOff()
        self.ren.AddActor(self.actor_moving)

        # Set up subscriber for registered organ position
        poseSubTopic = namespace + "/registration_marker"
        poseSub = rospy.Subscriber(poseSubTopic, Marker, self.poseCallback)

        # Set up registration button
        pubTopic = namespace + "/registration/reset"
        self.resetPub = rospy.Publisher(pubTopic, Bool, queue_size=1)
        pubTopic = namespace + "/registration/toggle"
        self.active = False
        self.activePub = rospy.Publisher(pubTopic, Bool, queue_size=1)
        self.registerButton.clicked.connect(self.register)
        self.stopButton.clicked.connect(self.stop)

        self.otherWindows = []
        if not self.isPrimaryWindow:
            parentWindow.otherWindows.append(self)
            self.otherWindows.append(parentWindow) 

        # Set up checkbox
        self.renderMaskCheckBox.stateChanged.connect(self.checkBoxChanged)

        # Setup interactor
        self.iren = self.renWin.GetInteractor()
        self.iren.RemoveObservers('LeftButtonPressEvent')
        self.iren.AddObserver('LeftButtonPressEvent', self.leftButtonPressEvent, 1.0)
        self.iren.RemoveObservers('LeftButtonReleaseEvent')
        self.iren.AddObserver('LeftButtonReleaseEvent', self.leftButtonReleaseEvent, 1.0)
        self.iren.RemoveObservers('MouseMoveEvent')
        self.iren.AddObserver('MouseMoveEvent', self.mouseMoveEvent, 1.0)
        self.iren.RemoveObservers('MiddleButtonPressEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')
        self.iren.RemoveObservers('MouseWheelForwardEvent')
        self.iren.RemoveObservers('MouseWheelBackwardEvent')
        self.show()
        self.iren.Initialize()
        if self.isPrimaryWindow:
            # Set up timer to refresh render if this is primary window
            self.cb = vtktools.vtkTimerCallback(self.renWin)
            self.iren.AddObserver('TimerEvent', self.cb.execute)
            timerId = self.iren.CreateRepeatingTimer(30);
        else:
            # Otherwise piggyback off of primary window callback
            parentWindow.cb.addRenWin(self.renWin) 

        self.zRen = vtk.vtkRenderer()
        self.zRenWin = vtk.vtkRenderWindow()
        self.zRenWin.AddRenderer(self.zRen)
        self.zRenWin.SetOffScreenRendering(1)

        imgDims = self.bgImage.GetDimensions()
        self.zRenWin.SetSize(imgDims[0],imgDims[1])

        self.zRen.SetActiveCamera(self.ren.GetActiveCamera())
        self.zRen.AddActor(self.actor_moving)

        self.zBuff = vtktools.zBuff(self.zRenWin)

        # Set up publisher for rendered image
        renWinTopic = namespace + '/registration_render'

        self.zRenWin.Render()

        self.renWinFilter = vtk.vtkWindowToImageFilter()
        self.renWinFilter.SetInput(self.zRenWin)
        self.renWinFilter.SetMagnification(1)
        self.renWinFilter.Update()
        self.renWinPub = rospy.Publisher(renWinTopic, Image, queue_size = 1)

        # Set up subscriber for camera image
        imageSub = rospy.Subscriber(imgSubTopic, Image, self.imageCallback)
        
        self.iren.Start()

    def checkBoxChanged(self):
        self.renderMaskCheckBox.isChecked()
        for window in self.otherWindows:
            window.renderMaskCheckBox.setChecked(self.renderMaskCheckBox.isChecked())

    def leftButtonPressEvent(self,obj,event):
        if self.renderMaskCheckBox.isChecked():
            return
        event = cv2.EVENT_LBUTTONDOWN
        (x,y) = self.getMousePos()
        self.segmentation.onMouse(event,x,y,0,0)

    def leftButtonReleaseEvent(self,obj,event):
        if self.renderMaskCheckBox.isChecked():
            return
        event = cv2.EVENT_LBUTTONUP
        (x,y) = self.getMousePos()
        self.segmentation.onMouse(event,x,y,0,0)

    def mouseMoveEvent(self,obj,event):
        event = cv2.EVENT_MOUSEMOVE
        (x,y) = self.getMousePos()
        self.segmentation.onMouse(event,x,y,0,0)

    def getMousePos(self):
        pos = self.iren.GetEventPosition()
        windowShape =  self.iren.GetSize()
        imageShape = self.segmentation.image.shape
        ratio = imageShape[0] / float(windowShape[1])
        offset = (windowShape[0] * ratio - imageShape[1]) / 2.0
        x = int(pos[0] * ratio - offset)
        y = imageShape[0] - int(pos[1] * ratio)
        return (x,y)

    def imageCallback(self, data):
        # TODO no try catch
        # try:
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if self.isPrimaryWindow:
            self.segmentation.setImage(image)
            if self.renderMaskCheckBox.isChecked():
                self.zRen.ResetCameraClippingRange()
                mask = vtktools.vtkImageToNumpy(self.zBuff.GetOutput())
                mask = np.where(mask<5,0,1).astype('uint8')
                self.segmentation.mask = mask[:,:,0]
            vtktools.numpyToVtkImage(self.segmentation.getMaskedImage(),self.bgImage)
            self.maskPub.publish(self.bridge.cv2_to_imgmsg(self.segmentation.mask*255, "mono8"))
            self.renWinFilter.Modified()
            self.renWinFilter.Update()
            render = vtktools.vtkImageToNumpy(self.renWinFilter.GetOutput())
            self.renWinPub.publish(self.bridge.cv2_to_imgmsg(render, "rgb8"))
        else:
            vtktools.numpyToVtkImage(image,self.bgImage)
        # except Exception as e:
        #     print("Error in registration window image callback:", e)

    def poseCallback(self, data):
        pos = data.pose.position
        rot = data.pose.orientation
        mat = transformations.quaternion_matrix([rot.x,rot.y,rot.z,rot.w])
        mat[0:3,3] = [pos.x,pos.y,pos.z]
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.SetMatrix(mat.ravel())
        self.actor_moving.SetUserTransform(transform)
        self.actor_moving.VisibilityOn()             

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
        self.active = True
        self.activePub.publish(self.active)
        self.resetPub.publish(True)

    def stop(self):
        self.active = not self.active
        self.activePub.publish(self.active)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # RosThread.update = self.update
    rosThread = vtktools.RosQThread()
    meshPath = rospy.get_param("~mesh_path")
    stlScale = rospy.get_param("~mesh_scale")
    windowL = RegistrationWidget(meshPath, scale=stlScale)
    windowR = RegistrationWidget(meshPath, scale=stlScale, parentWindow=windowL)
    rosThread.start()
    sys.exit(app.exec_())