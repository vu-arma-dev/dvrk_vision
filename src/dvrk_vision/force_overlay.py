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
from geometry_msgs.msg import WrenchStamped
from dvrk_vision.vtk_stereo_viewer import StereoCameras, QVTKStereoViewer
from dvrk_vision.clean_resource_path import cleanResourcePath
from dvrk import psm
import PyKDL
from tf_conversions import posemath
import colorsys

def makeArrowActor(coneRadius = .1, shaftRadius = 0.03, tipLength = 0.35):
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

def setActorMatrix(actor, npMatrix):
    transform = vtk.vtkTransform()
    transform.Identity()
    transform.SetMatrix(npMatrix.ravel())
    actor.SetPosition(transform.GetPosition())
    actor.SetOrientation(transform.GetOrientation())
    actor.SetScale(transform.GetScale())

class ForceOverlayWidget(QVTKStereoViewer):
    def __init__(self, cam, camTransform, dvrkName, forceTopic, draw="bar", masterWidget=None, parent=None):
        super(ForceOverlayWidget, self).__init__(cam, parent=parent)
        self.masterWidget = masterWidget
        self.cameraTransform = arrayToPyKDLFrame(camTransform)
        self.drawType = draw
        if self.masterWidget is None:
            self.robot = psm(dvrkName)
            rospy.Subscriber(forceTopic, WrenchStamped, self.forceCB)

        

    def renderSetup(self):
        # Setup interactor
        self.iren = self.GetRenderWindow().GetInteractor()
        self.iren.RemoveObservers('LeftButtonPressEvent')
        self.iren.RemoveObservers('LeftButtonReleaseEvent')
        self.iren.RemoveObservers('MouseMoveEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')
        self.currentForce = [0,0,0]

        if self.drawType == "arrow":
            if self.masterWidget is not None:
                print("HELLO MASTER")
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
    
    def forceCB(self, data):
        self.currentForce = [data.wrench.force.x, data.wrench.force.y, data.wrench.force.z]

    def imageProc(self,image):
        if self.masterWidget is not None:
            return image
        # Get current force
        force = self.currentForce
        force = np.linalg.norm(force)
        targetF = 2 # Newtons
        targetR = 2 # Newtons
        # Calculate color
        xp = [targetF-targetR, targetF, targetF+targetR]
        fp = [0, 1, 0]
        colorPos = np.interp(force, xp, fp)
        color = colorsys.hsv_to_rgb(colorPos**3 / 3, .8,1)

        if self.drawType == "arrow":
            self.arrowActor.GetProperty().SetColor(color[0], color[1], color[2])
            # Calculate pose of arrows
            initialRot = PyKDL.Frame(PyKDL.Rotation.RotY(np.pi / 2),
                                     PyKDL.Vector(0, 0, 0))
            pos = self.robot.get_current_position() * initialRot
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
            pos = self.robot.get_current_position()
            pos = self.cameraTransform.Inverse() * pos
            pos2 = PyKDL.Frame(PyKDL.Rotation.Identity(), pos.p)
            pos2.M.DoRotZ(np.pi)
            pos2.p = pos2.p + pos2.M.UnitX() * -.015
            posMat = posemath.toMatrix(pos2)
            setActorMatrix(self.bar, posMat)
            setActorMatrix(self.greenLine, posMat)
            # Scale color bar
            fp2 = [0, .5, 1]
            scalePos = np.interp(force, xp, fp2)
            posMat[1,0:3] = posMat[1,0:3] * scalePos
            setActorMatrix(self.forceBar, posMat)

        return image

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