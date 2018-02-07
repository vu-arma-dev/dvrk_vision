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
    from PyQt5.QtWidgets import QApplication
else:
    from PyQt4.QtGui import QApplication

import dvrk_vision.vtktools as vtktools
from geometry_msgs.msg import PoseStamped
from dvrk_vision.vtk_stereo_viewer import StereoCameras, QVTKStereoViewer
from QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from force_sensor_gateway.msg import ForceSensorData
import yaml
import PyKDL
from tf_conversions import posemath
import colorsys


def cleanResourcePath(path):
    newPath = path
    if path.find("package://") == 0:
        newPath = newPath[len("package://"):]
        pos = newPath.find("/")
        if pos == -1:
            rospy.logfatal("%s Could not parse package:// format", path)
            quit(1)

        package = newPath[0:pos]
        newPath = newPath[pos:]
        package_path = rospkg.RosPack().get_path(package)

        if package_path == "":
            rospy.logfatal("%s Package [%s] does not exist",
                           path.c_str(),
                           package.c_str());
            quit(1)

        newPath = package_path + newPath;
    elif path.find("file://") == 0:
        newPath = newPath[len("file://"):]

    if not os.path.isfile(newPath):
        rospy.logfatal("%s file does not exist", newPath)
        quit(1)
    return newPath;

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
    actor.SetUserTransform(transform)

class OverlayWidget(QVTKStereoViewer):
    def __init__(self, camera, dvrkName, cameraTransform, masterWidget=None, parent=None):
        super(OverlayWidget, self).__init__(camera, parent=parent)
        self.masterWidget = masterWidget
        self.cameraTransform = cameraTransform
        self.drawType = "arrow"

        self.force = None
        self.pose = None
        self.forceSub = rospy.Subscriber('/force_sensor_topic',
                                     ForceSensorData, self.forceCb)
        self.pose = None
        self.poseSub = rospy.Subscriber('/dvrk/PSM2/position_cartesian_current',
                                        PoseStamped, self.poseCb)

        if self.masterWidget == None:
            pass
        else:
            pass

    def forceCb(self, data):
        force = np.array([data.data1, data.data2, data.data3, data.data4], np.float32) / 30.0
        force[force < 0] = 0
        self.force = force

    def poseCb(self, data):
        self.pose = posemath.fromMsg(data.pose)

    def renderSetup(self):
        if self.drawType == "arrow":
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

        # Setup interactor
        self.iren = self.GetRenderWindow().GetInteractor()
        self.iren.RemoveObservers('LeftButtonPressEvent')
        self.iren.RemoveObservers('LeftButtonReleaseEvent')
        self.iren.RemoveObservers('MouseMoveEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')
    
    def imageProc(self,image):
        if type(self.force) != type(None):
            # Get current force
            force = self.force # self.robot.get_current_wrench_body()[0:3]
        else :
            force = [0,0,0,0];
        force = np.linalg.norm(force)
        targetF = .5 # Target force
        targetR = .5 # Force Range
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
            pos = self.pose * initialRot
            pos = self.cameraTransform.Inverse() * pos
            posMat = posemath.toMatrix(pos)
            posMatTarget = posMat.copy()
            # Scale arrows
            posMat[0:3,0:3] = posMat[0:3,0:3] * targetR / 5 * targetF
            setActorMatrix(self.targetActor, posMat)
            posMat[0:3,0:3] = posMat[0:3,0:3] * force / targetF
            setActorMatrix(self.arrowActor, posMat)

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
    """A simple example that uses the QVTKRenderWindowInteractor class."""

    # every QT app needs an app
    app = QApplication(['QVTKRenderWindowInteractor'])
    yamlFile = cleanResourcePath("package://dvrk_vision/defaults/registration_params.yaml")
    with open(yamlFile, 'r') as stream:
        data = yaml.load(stream)
    cameraTransform = arrayToPyKDLFrame(data['transform'])

    rosThread = vtktools.QRosThread()
    rosThread.start()
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("stereo/left/image_rect",
                         "stereo/right/image_rect",
                         "stereo/left/camera_info",
                         "stereo/right/camera_info",
                         slop = slop)
    # windowL = QVTKStereoViewer(cams.camL)
    windowL = OverlayWidget(cams.camL, 'PSM2', cameraTransform)
    windowL.Initialize()
    windowL.start()
    windowL.show()
    windowR = OverlayWidget(cams.camR, 'PSM2', cameraTransform, masterWidget = windowL)
    windowR.Initialize()
    windowR.start()
    windowR.show()
    sys.exit(app.exec_())