#!/usr/bin/env python
import sys
import os
import vtk
import numpy as np
import rospy
import rospkg
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from graph_cut_node import SegmentedImage
from tf import transformations
from IPython import embed
from vtk_stereo_viewer import StereoCameras, QVTKStereoViewer
import vtktools
# Which PyQt we use depends on our vtk version. QT4 causes segfaults with vtk > 6
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
    from PyQt5 import uic
    _QT_VERSION = 5
else:
    from PyQt4.QtGui import QWidget, QVBoxLayout, QApplication
    from PyQt4 import uic
    _QT_VERSION = 4

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
    bridge = CvBridge()
    def __init__(self, camera, meshPath, scale=1, masterWidget=None, parent=None):

        super(RegistrationWidget, self).__init__()
        functionPath = os.path.dirname(os.path.realpath(__file__))
        uic.loadUi(functionPath + "/registration_gui.ui", self)

        self.meshPath = meshPath
        self.scale = scale

        # Check whether this is the left (primary) or the right (secondary) window
        self.isPrimaryWindow = masterWidget == None
        side = "left" if self.isPrimaryWindow else "right"

        self.vtkWidget = QVTKStereoViewer(camera, parent=self)

        self.vtkWidget.renderSetup = self.renderSetup

        # Set up segmentation
        if self.isPrimaryWindow:
            self.vtkWidget.imageProc = self.imageProc
            self.segmentation = SegmentedImage()
            # Set up publisher for masking
            pubTopic = "disparity_mask"
            self.maskPub = rospy.Publisher(pubTopic, Image, queue_size=1)

        else:
            self.segmentation = masterWidget.segmentation

        # Add vtk widget
        self.vl = QVBoxLayout()
        self.vl.addWidget(self.vtkWidget)
        self.vtkFrame.setLayout(self.vl)

        # Set up subscriber for registered organ position
        poseSubTopic = "registration_marker"
        poseSub = rospy.Subscriber(poseSubTopic, Marker, self.poseCallback)

        # Set up registration button
        pubTopic = "registration_reset"
        self.resetPub = rospy.Publisher(pubTopic, Bool, queue_size=1)
        pubTopic = "registration_toggle"
        self.active = False
        self.activePub = rospy.Publisher(pubTopic, Bool, queue_size=1)
        self.registerButton.clicked.connect(self.register)
        self.stopButton.clicked.connect(self.stop)

        self.otherWindows = []
        if not self.isPrimaryWindow:
            masterWidget.otherWindows.append(self)
            self.otherWindows.append(masterWidget) 

        # Set up checkbox
        self.renderMaskCheckBox.stateChanged.connect(self.checkBoxChanged)

        self.vtkWidget.Initialize()
        self.vtkWidget.start()

    def renderSetup(self):
        # Set up 3D actor for organ
        meshPath = cleanResourcePath(self.meshPath)
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
        transform.Scale(self.scale,self.scale,self.scale)
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
        # Add actor
        self.vtkWidget.ren.AddActor(self.actor_moving)
        # Setup interactor
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
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
        # Setup z-buffer
        self.zRen = vtk.vtkRenderer()
        self.zRenWin = vtk.vtkRenderWindow()
        self.zRenWin.AddRenderer(self.zRen)
        self.zRenWin.SetOffScreenRendering(1)
        imgDims = self.vtkWidget.cam.image.shape
        self.zRenWin.SetSize(imgDims[1],imgDims[0])
        self.zRen.SetActiveCamera(self.vtkWidget.ren.GetActiveCamera())
        self.zRen.AddActor(self.actor_moving)
        self.zBuff = vtktools.zBuff(self.zRenWin)
        self.zRenWin.Render()

        # Set up publisher for rendered image
        # renWinTopic = "registration_render"
        # self.renWinFilter = vtk.vtkWindowToImageFilter()
        # self.renWinFilter.SetInput(self.zRenWin)
        # self.renWinFilter.SetMagnification(1)
        # self.renWinFilter.Update()
        # self.renWinPub = rospy.Publisher(renWinTopic, Image, queue_size = 1)
        
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

    def imageProc(self, image):
        self.segmentation.setImage(image)
        if self.renderMaskCheckBox.isChecked():
            self.zRen.ResetCameraClippingRange()
            mask = vtktools.vtkImageToNumpy(self.zBuff.GetOutput())
            mask = np.where(mask>1,1,0).astype('uint8')
            kernel = np.ones((5,5),np.uint8)
            self.segmentation.mask = cv2.dilate(mask[:,:,0], kernel)
        self.maskPub.publish(self.bridge.cv2_to_imgmsg(self.segmentation.mask*255, "mono8"))
        return self.segmentation.getMaskedImage()

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
    rosThread = vtktools.QRosThread()
    meshPath = rospy.get_param("~mesh_path")
    stlScale = rospy.get_param("~mesh_scale")
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("/stereo/left/image_rect",
                         "/stereo/right/image_rect",
                         "/stereo/left/camera_info",
                         "/stereo/right/camera_info",
                         slop = slop)
    windowL = RegistrationWidget(cams.camL, meshPath, scale=stlScale)
    windowL.show()
    windowR = RegistrationWidget(cams.camR, meshPath, scale=stlScale, masterWidget=windowL)
    windowR.show()
    rosThread.start()
    sys.exit(app.exec_())