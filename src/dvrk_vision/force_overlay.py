#!/usr/bin/env python
import vtk
import numpy as np
import rospy
import cv2
# Which PyQt we use depends on our vtk version. QT4 causes segfaults with vtk > 6
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
    _QT_VERSION = 5
else:
    from PyQt4.QtGui import QWidget, QVBoxLayout, QApplication
    _QT_VERSION = 4
from geometry_msgs.msg import WrenchStamped, PoseStamped
from dvrk_vision.vtk_stereo_viewer import StereoCameras, QVTKStereoViewer
from dvrk_vision.clean_resource_path import cleanResourcePath
from dvrk import psm
import PyKDL
from tf_conversions import posemath
import colorsys

def makeArrowActor(coneRadius = .1, shaftRadius = 0.03, tipLength = 0.35):
    """ Creates an arrow actor with given properties
    """
    arrowSource = vtk.vtkArrowSource()
    arrowSource.SetShaftRadius (shaftRadius)
    arrowSource.SetTipRadius (coneRadius)
    arrowSource.SetTipLength (tipLength)
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(arrowSource.GetOutput())
    else:
        mapper.SetInputConnection(arrowSource.GetOutputPort())
    arrowActor = vtk.vtkActor()
    arrowActor.SetMapper(mapper)
    return arrowActor

# Make a 2D text actor that you can place as a particular 2D location (not as useful)
def makeTextActor(text="Move around and do some stuff"):
    textActor = vtk.vtkTextActor()
    textActor.SetInput(text)
    txtprop=textActor.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(25)
    txtprop.SetColor(1,0,0)
    txtprop.SetBackgroundColor(1,1,1)
    txtprop.SetBackgroundOpacity(1.0)
    txtprop.SetJustificationToCentered()
    textActor.SetDisplayPosition(300,50)

    return textActor

# Make a 3D text actor
def makeTextActor3D(text="Uninitialized"):
    vecText = vtk.vtkVectorText()
    vecText.SetText(text)
    textMapper = vtk.vtkPolyDataMapper()
    textMapper.SetInputConnection(vecText.GetOutputPort())
    textActor = vtk.vtkFollower()
    textActor.SetMapper(textMapper)
    textActor.SetPosition(-.065,0.03,0.25)
    textActor.SetScale(.008,.008,.008)
    textActor.SetOrientation(0,180,180)
    txtProp = textActor.GetProperty()
    txtProp.SetColor(1,0,0)
    return textActor,vecText

def setActorMatrix(actor, npMatrix):
    """ Set a VTK actor's transformation based on a 4x4 homogenous transform
    """
    transform = vtk.vtkTransform()
    transform.Identity()
    transform.SetMatrix(npMatrix.ravel())
    actor.SetPosition(transform.GetPosition())
    actor.SetOrientation(transform.GetOrientation())
    actor.SetScale(transform.GetScale())

class ForceOverlayWidget(QVTKStereoViewer):
    def __init__(self, cam, camSync, camTransform, dvrkTopic, forceTopic, 
                 draw="bar", masterWidget=None, parent=None):
        """ Widget for relaying force information to a user
    
        Args:
            cams (dvrk_vision.vtk_stereo_viewer import StereoCameras): Camera
                object which contains images for background and utilities for
                synching transforms.

            camSync (dvrk_vision.tf_sync.CameraSync): Object that allows
                us to synchronize camera images with robot topics

            camTransform (enumerable[float]): 4x4 homogenous transformation of
                the camera in the robot's frame

            dvrkTopic (string): ROS topic representing the current position of
                the robot (geometry_msgs.msg.PoseStamped)

            forceTopic (string): ROS topic representing the current force
                exerted at the end effector (geometry_msgs.msg.WrenchStamped)

            draw (string): Either "arrow or "bar" depending on how force should
                be represented on screen

            masterWidget (ForceOverlayWidget): The widget showing the left image
                that controls both widgets. If None, this IS the control widget

            parent (PyQtX.QWidget): Qt Parent
        """

        super(ForceOverlayWidget, self).__init__(cam, parent=parent)
        self.masterWidget = masterWidget
        self.cameraTransform = arrayToPyKDLFrame(camTransform)
        self.drawType = draw
        self.camSync = camSync
        self.dvrkTopic = dvrkTopic
        self.forceTopic = forceTopic
        self.camSync.addTopics([self.dvrkTopic, self.forceTopic])
        # if self.masterWidget is None:
        #     self.robot = psm(dvrkName)
        #     rospy.Subscriber(forceTopic, WrenchStamped, self.forceCB)

    def renderSetup(self):
        # Setup interactor
        self.iren = self.GetRenderWindow().GetInteractor()
        self.iren.RemoveObservers('LeftButtonPressEvent')
        self.iren.RemoveObservers('LeftButtonReleaseEvent')
        self.iren.RemoveObservers('MouseMoveEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')
        self.currentForce = [0,0,0]

        # Change camera location/orientation to get text facing upright
        # This completely breaks the tracking of the force bar with the robot, but does get the text looking ok (as long as you switch the left and right...)
        # cam = vtk.vtkCamera()
        # cam.SetFocalPoint(0,0,-1)
        # cam.SetViewUp(0,-1,0)
        # self.ren.SetActiveCamera(cam)

        if self.masterWidget is not None:
            self.textActor = self.masterWidget.textActor
            self.vecText = self.masterWidget.vecText
        else:
            [self.textActor,self.vecText] = makeTextActor3D()
            # self.textActor.SetPosition(-0.04,0.04,0.75)
            # self.textActor.SetOrientation(0,0,0)
            # self.vecText.SetText("TEST")
        self.ren.AddActor(self.textActor)

        if self.drawType == "arrow":
            if self.masterWidget is not None:
                self.arrowActor = self.masterWidget.arrowActor
                self.targetActor = self.masterWidget.targetActor
                self.ren.AddActor(self.arrowActor)
                self.ren.AddActor(self.targetActor)
                return
            arrowSource = vtk.vtkArrowSource()
            mapper = vtk.vtkPolyDataMapper()
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(arrowSource.GetOutput())
            else:
                mapper.SetInputConnection(arrowSource.GetOutputPort())
            # Create actor that we will position according to dVRK
            self.arrowActor = makeArrowActor()
            self.targetActor = makeArrowActor(coneRadius = .07,
                                              shaftRadius = .02)
            self.targetActor.GetProperty().SetOpacity(.2)
            self.ren.AddActor(self.arrowActor)
            self.ren.AddActor(self.targetActor)

        elif self.drawType == "bar":
            if self.masterWidget is not None:
                self.bar = self.masterWidget.bar
                self.forceBar = self.masterWidget.forceBar
                self.greenLine = self.masterWidget.greenLine
                self.ren.AddActor(self.bar)
                self.ren.AddActor(self.forceBar)
                self.ren.AddActor(self.greenLine)
                return
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
            self.ren.AddActor(self.bar)
            self.ren.AddActor(self.forceBar)
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
            self.ren.AddActor(self.greenLine)

    def imageProc(self,image):
        if self.masterWidget is not None:
            return image
        # Get current force
        fmsg = self.camSync.getMsg(self.forceTopic)
        if type(fmsg) is not WrenchStamped:
            return image
        force = [fmsg.wrench.force.x, fmsg.wrench.force.y, fmsg.wrench.force.z]
        force = np.linalg.norm(force)
        targetF = 3 # Newtons
        targetMax = 6 # Newtons
        # Calculate color
        xp = [0, targetF, targetMax]
        fp = [0, 1, 0]
        colorPos = np.interp(force, xp, fp)
        color = colorsys.hsv_to_rgb(colorPos**3 / 3, .8,1)

        # Get robot pose
        posMsg = self.camSync.getMsg(self.dvrkTopic)
        if type(posMsg) is not PoseStamped:
            return image
        pos = posemath.fromMsg(posMsg.pose)

        if self.drawType == "arrow":
            self.arrowActor.GetProperty().SetColor(color[0], color[1], color[2])
            # Calculate pose of arrows
            initialRot = PyKDL.Frame(PyKDL.Rotation.RotY(np.pi / 2),
                                     PyKDL.Vector(0, 0, 0))
            pos = pos * initialRot
            pos = self.cameraTransform.Inverse() * pos
            posMat = posemath.toMatrix(pos)
            posMatTarget = posMat.copy()
            # Scale arrows
            posMat[0:3,0:3] = posMat[0:3,0:3] * .025 * targetF
            setActorMatrix(self.targetActor, posMat)
            posMat[0:3,0:3] = posMat[0:3,0:3] * force / targetF
            setActorMatrix(self.arrowActor, posMat)

        elif self.drawType == "bar":
            self.forceBar.GetProperty().SetColor(color[0], color[1], color[2])
            # Move background bar
            pos = self.cameraTransform.Inverse() * pos
            pos2 = PyKDL.Frame(PyKDL.Rotation.Identity(), pos.p)
            pos2.M.DoRotZ(np.pi)
            pos2.p = pos2.p + pos2.M.UnitX() * -.015# + pos2.M.UnitZ() * 0.01
            posMat = posemath.toMatrix(pos2)
            setActorMatrix(self.bar, posMat)
            setActorMatrix(self.greenLine, posMat)
            # setActorMatrix(self.textActor,posMat) #I don't know why this doesn't work...

            # Scale color bar
            fp2 = [0, .5, 1]
            scalePos = np.interp(force, xp, fp2)
            posMat[1,0:3] = posMat[1,0:3] * scalePos
            setActorMatrix(self.forceBar, posMat)
            

        return image

    def setBarVisibility(self,b_input=True):
        self.bar.SetVisibility(b_input)
        self.forceBar.SetVisibility(b_input)
        self.greenLine.SetVisibility(b_input)

    def setText(self,textInput):
        self.vecText.SetText(textInput)

def arrayToPyKDLRotation(array):
    x = PyKDL.Vector(array[0][0], array[1][0], array[2][0])
    y = PyKDL.Vector(array[0][1], array[1][1], array[2][1])
    z = PyKDL.Vector(array[0][2], array[1][2], array[2][2])
    return PyKDL.Rotation(x,y,z)

def arrayToPyKDLFrame(array):
    rot = arrayToPyKDLRotation(array)
    pos = PyKDL.Vector(array[0][3],array[1][3],array[2][3])
    return PyKDL.Frame(rot,pos)

if __name__ == "__main__":
    import sys
    import yaml
    import dvrk_vision.vtktools as vtktools
    """A simple example that uses the ForceOverlayWidget class."""

    # every QT app needs an app
    app = QApplication(['Force Overlay'])
    yamlFile = cleanResourcePath("package://dvrk_vision/defaults/registration_params.yaml")
    with open(yamlFile, 'r') as stream:
        data = yaml.load(stream)
    cameraTransform = data['transform']

    rosThread = vtktools.QRosThread()
    rosThread.start()
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("stereo/left/image_rect",
                         "stereo/right/image_rect",
                         "stereo/left/camera_info",
                         "stereo/right/camera_info",
                         slop = slop)

    windowL = ForceOverlayWidget(cam = cams.camL,
                                 camTransform = cameraTransform,
                                 dvrkName = 'PSM2',
                                 forceTopic = '/atinetft/wrench')
    windowL.Initialize()
    windowL.start()
    windowL.show()
    sys.exit(app.exec_())