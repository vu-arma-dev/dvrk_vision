#!/usr/bin/env python
import vtk
# Which PyQt we use depends on our vtk version. QT4 causes segfaults with vtk > 6
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QCheckBox, QPushButton
    from PyQt5 import uic
    _QT_VERSION = 5
else:
    from PyQt4.QtGui import QWidget, QVBoxLayout, QApplication, QCheckBox, QPushButton
    from PyQt4 import uic
    _QT_VERSION = 4
# General imports
import numpy as np
import cv2
import os
# Ros specific
import rospy
import message_filters
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import tf2_ros
from tf import transformations
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

# DVRK vision stuff
from dvrk_vision.vtk_stereo_viewer import QVTKStereoViewer
from dvrk_vision.clean_resource_path import cleanResourcePath
import vtktools

#For Force bar
from geometry_msgs.msg import WrenchStamped, PoseStamped
import PyKDL
from tf_conversions import posemath
import colorsys

# Registration file parsing
import yaml

# Make a 3D text actor
def makeTextActor3D(text="Uninitialized"):
    vecText = vtk.vtkVectorText()
    vecText.SetText(text)
    textMapper = vtk.vtkPolyDataMapper()
    textMapper.SetInputConnection(vecText.GetOutputPort())
    textActor = vtk.vtkFollower()
    textActor.SetMapper(textMapper)
    textActor.SetPosition(-.022,-0.038,0.15)
    textActor.SetScale(.0044,.0044,.0044)
    textActor.SetOrientation(0,180,180)
    txtProp = textActor.GetProperty()
    txtProp.SetColor(1,0,0)
    return textActor,vecText

# Set a VTK actor's transformation based on a 4x4 homogenous transform
def setActorMatrix(actor, npMatrix):
    transform = vtk.vtkTransform()
    transform.Identity()
    transform.SetMatrix(npMatrix.ravel())
    actor.SetPosition(transform.GetPosition())
    actor.SetOrientation(transform.GetOrientation())
    actor.SetScale(transform.GetScale())


class IrepWidget(QWidget):
    bridge = CvBridge()
    def __init__(self, camera, markerTopic, robotFrame, tipFrame, cameraFrame, masterWidget=None, parent=None):
        super(IrepWidget, self).__init__(parent=parent)

        # Set up UI
        uiPath = cleanResourcePath("package://dvrk_vision/src/dvrk_vision/irep_widget.ui")
        uic.loadUi(uiPath, self)

        # Load in variables
        self.forcePos=[-0.01,.03,0.15]
        self.b_use_GT = False
        self.b_useForceY = False
        self.robotFrame = robotFrame
        self.tipFrame = tipFrame
        self.markerTopic = markerTopic
        self.cameraFrame = cameraFrame
        self.masterWidget = masterWidget
        # Set Defaults
        self.otherWindows = []
        if self.masterWidget is not None:
            self.masterWidget.otherWindows.append(self)
            self.otherWindows.append(self.masterWidget)

        # Set up VTK widget
        self.vtkWidget = QVTKStereoViewer(camera, parent=self)
        self.vtkWidget.renderSetup = self.renderSetup
        self.vtkWidget.imageProc = self.imageProc
        # Add VTK widget to window
        self.vl = QVBoxLayout()
        self.vl.addWidget(self.vtkWidget)
        self.vtkFrame.setLayout(self.vl)
        # Set up interactor
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
        # Set up VTK to publish render
        self.winToImage = vtk.vtkWindowToImageFilter()
        self.winToImage.SetInput(self.vtkWidget._RenderWindow)
        self.winToImage.ReadFrontBufferOff()
        # Start the widget
        self.vtkWidget.Initialize()
        self.vtkWidget.start()

        # Get defaults from parameter server
        self.filePath = rospy.get_param('~camera_registration')
        with open(self.filePath, 'r') as f:
            data = yaml.load(f)
        camTransform = data['transform']
        self.cameraTransform = arrayToPyKDLFrame(camTransform)

        # Set up ros publishers and subscribers
        self.psmName = 'PSM1'
        pubTopic = self.vtkWidget.cam.topic[:-len(self.vtkWidget.cam.topic.split('/')[-1])] + "image_rendered"
        self.imagePub = rospy.Publisher(pubTopic, Image, queue_size=1)

    def renderSetup(self):
        if self.masterWidget is not None:
            self.textActor = self.masterWidget.textActor

            # Add force bar
            self.bar = self.masterWidget.bar
            self.forceBar = self.masterWidget.forceBar
            self.greenLine = self.masterWidget.greenLine
            self.vtkWidget.ren.AddActor(self.bar)
            self.vtkWidget.ren.AddActor(self.forceBar)
            self.vtkWidget.ren.AddActor(self.greenLine)

            # Add text actor
            self.textActor= self.masterWidget.textActor
            self.vecText= self.masterWidget.vecText
            self.vtkWidget.ren.AddActor(self.textActor)
            return

        [self.textActor,self.vecText] = makeTextActor3D()

        # Add actors
        self.vtkWidget.ren.AddActor(self.textActor)

        # Make two color bars to show current force
        source = vtk.vtkCubeSource()
        source.SetBounds((-.002, .002, 0, .05, 0, .001))
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(source.GetOutput())
        else:
            mapper.SetInputConnection(source.GetOutputPort())
        self.bar = vtk.vtkActor()
        self.bar.SetMapper(mapper)
        self.bar.GetProperty().SetColor(.2,.2,.2)
        self.bar.GetProperty().LightingOff()
        self.forceBar = vtk.vtkActor()
        self.forceBar.SetMapper(mapper)
        self.forceBar.GetProperty().LightingOff()
        self.vtkWidget.ren.AddActor(self.bar)
        self.vtkWidget.ren.AddActor(self.forceBar)
        # Make a green line to show target force
        source2 = vtk.vtkCubeSource()
        source2.SetBounds((-.002, .002, .0245, .0255, 0, .001001))
        mapper2 = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper2.SetInput(source2.GetOutput())
        else:
            mapper2.SetInputConnection(source2.GetOutputPort())
        self.greenLine = vtk.vtkActor()
        self.greenLine.SetMapper(mapper2)
        self.greenLine.GetProperty().SetColor(.9,.9,.9)
        self.greenLine.GetProperty().LightingOff()
        self.vtkWidget.ren.AddActor(self.greenLine)

    # Bar processing
    def imageProc(self,image):
        if self.masterWidget is not None:
            return image
        self.forceUpdate()
        return image

    def forceUpdate(self):
        # Get current force
        try:
            fmsg = rospy.wait_for_message(self.forceTopic,WrenchStamped,1.0)
        except:
            fmsg=WrenchStamped()        
        force = [fmsg.wrench.force.x, fmsg.wrench.force.y, fmsg.wrench.force.z]
        forceMag = np.linalg.norm(force)
        forceY = -force[1]
        try:
            targetF = rospy.get_param('/f_desired') # Newtons
        except:
            targetF = 0.75
        targetMax = targetF*2 # Newtons
        
        # Calculate color
        xp = [0, targetF, targetMax]
        fp = [0, 1, 0]

        if self.b_useForceY:
            colorPos = np.interp(forceY, xp, fp)
        else:
            colorPos = np.interp(forceMag, xp, fp)
        color = colorsys.hsv_to_rgb(colorPos**3 / 3, .8,1)
        posMsg = PoseStamped()
        pos = posemath.fromMsg(posMsg.pose)

        self.forceBar.GetProperty().SetColor(color[0], color[1], color[2])
        # Move background bar
        pos = self.cameraTransform.Inverse() * pos
        pos2 = PyKDL.Frame(PyKDL.Rotation.Identity(), pos.p)
        pos2.M.DoRotZ(np.pi)
        pos2.p = pos2.p + pos2.M.UnitX() * -.025 - pos2.M.UnitY() * 0.02

        pos2.p = PyKDL.Vector(self.forcePos[0],self.forcePos[1],self.forcePos[2])

        posMat = posemath.toMatrix(pos2)
        setActorMatrix(self.bar, posMat)
        setActorMatrix(self.greenLine, posMat)
        
        # Scale color bar
        fp2 = [0, .5, 1]
        if self.b_useForceY:
            scalePos = np.interp(forceY, xp, fp2)
        else:
            scalePos = np.interp(forceMag, xp, fp2)
        posMat[1,0:3] = posMat[1,0:3] * scalePos
        setActorMatrix(self.forceBar, posMat)
        return

    def setUseForceY(self,b_input):
        self.b_useForceY = b_input

    def setForcePos(self,moveInput):
        self.forcePos=[moveInput[0],moveInput[1],moveInput[2]]

    def setText(self,textInput):
        self.vecText.SetText(textInput)

    def setTextPos(self,moveInput):
        self.textActor.SetPosition(moveInput[0],moveInput[1],moveInput[2])

    def setTextScale(self,scaleInput):
        self.textActor.SetScale(scaleInput,scaleInput,scaleInput)

    def setBarSrc(self,barsrc):
        if barsrc==1:
            self.forceTopic = '/dvrk/' + self.psmName + '/wrench_current_gt'
        else:
            self.forceTopic = '/dvrk/' + self.psmName + '/wrench_current'

    def setBarVisibility(self,b_input=True):
        self.bar.SetVisibility(b_input)
        self.forceBar.SetVisibility(b_input)
        self.greenLine.SetVisibility(b_input)

def arrayToPyKDLRotation(array):
    x = PyKDL.Vector(array[0][0], array[1][0], array[2][0])
    y = PyKDL.Vector(array[0][1], array[1][1], array[2][1])
    z = PyKDL.Vector(array[0][2], array[1][2], array[2][2])
    return PyKDL.Rotation(x,y,z)

def arrayToPyKDLFrame(array):
    rot = arrayToPyKDLRotation(array)
    pos = PyKDL.Vector(array[0][3],array[1][3],array[2][3])
    return PyKDL.Frame(rot,pos)

if __name__ == '__main__':
    import sys
    from dvrk_vision import vtktools
    from dvrk_vision.vtk_stereo_viewer import StereoCameras
    app = QApplication(sys.argv)
    rosThread = vtktools.QRosThread()
    rosThread.start()
    rosThread.update = app.processEvents
    
    markerTopic = "/stereo/registration_marker"
    robotFrame = "PSM2_SIM_psm_base_link"
    cameraFrame = "stereo_camera_frame"
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("/stereo/left/image_rect",
                         "/stereo/right/image_rect",
                         "/stereo/left/camera_info",
                         "/stereo/right/camera_info",
                         slop = slop)

    windowL = IrepWidget(cams.camL, markerTopic, robotFrame, cameraFrame)
    windowL.show()
    sys.exit(app.exec_())



