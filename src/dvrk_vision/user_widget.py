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
import scipy.interpolate
import cv2
import os
# Ros specific
import rospy
import message_filters
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import tf2_ros
from tf import transformations
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from std_msgs.msg import Empty
from std_msgs.msg import Bool

# DVRK vision stuff
from dvrk_vision.overlay_gui import vtkRosTextureActor
from dvrk_vision.vtk_stereo_viewer import QVTKStereoViewer
from dvrk_vision.clean_resource_path import cleanResourcePath
from dvrk_vision import uvtoworld
from force_overlay import makeTextActor3D

#For Force bar
from geometry_msgs.msg import WrenchStamped, PoseStamped
from dvrk_vision.vtk_stereo_viewer import StereoCameras
from dvrk_vision.tf_sync import CameraSync
import PyKDL
from tf_conversions import posemath
import colorsys

def setActorMatrix(actor, npMatrix):
    """ Set a VTK actor's transformation based on a 4x4 homogenous transform
    """
    transform = vtk.vtkTransform()
    transform.Identity()
    transform.SetMatrix(npMatrix.ravel())
    actor.SetPosition(transform.GetPosition())
    actor.SetOrientation(transform.GetOrientation())
    actor.SetScale(transform.GetScale())

class vtkTimerCallback(object):
    def __init__(self, renWin):
        self.rate = rospy.Rate(30)
        self.renWin = renWin

    def execute(self, obj, event):
        self.renWin.Render()
        self.rate.sleep()
        self.update()

    def update(self):
        pass

## Take a Float64 MultiArray message, convert it into a numpyMatrix
def multiArrayToMatrixList(ma_msg):
    dim = len(ma_msg.layout.dim)
    offset = ma_msg.layout.data_offset

    if (ma_msg.layout.dim[0].label != "rows"):
        print "Error: dim[0] should be the rows"
    rows = ma_msg.layout.dim[0].size
    
    if (ma_msg.layout.dim[1].label != "cols"):
        print "Error: dim[1] should be the columns"
    columns = ma_msg.layout.dim[1].size

    data = np.array(ma_msg.data, dtype=np.float64)
    if(len(data) != rows*columns):
        rospy.logwarn_throttle(5, "Data in Float64MultiArray message does not match stated size")
        return np.empty([])

    return data.reshape((rows, columns))

def matrixListToMultiarray(matrix):
    rows, columns = matrix.shape
    msg = Float64MultiArray()
    # Set up layout
    msg.layout.data_offset = 0
    
    row_dim = MultiArrayDimension()
    row_dim.label = "rows"
    row_dim.size = rows
    row_dim.stride = columns * rows
    
    col_dim = MultiArrayDimension()
    col_dim.label = "cols"
    col_dim.size = columns
    col_dim.stride = columns

    msg.layout.dim = [row_dim, col_dim]
    msg.data = matrix.flatten().tolist()
    
    return msg

def loadMesh(path, scale):
    # Read in STL
    meshPath = cleanResourcePath(path)
    extension = os.path.splitext(path)[1]
    if extension == ".stl" or extension == ".STL":
        meshInput = vtk.vtkSTLReader()
        meshInput.SetFileName(path)
        meshReader = vtk.vtkTextureMapToPlane()
        meshReader.SetInputConnection(meshInput.GetOutputPort())
    elif extension == ".obj" or extension == ".OBJ":
        meshReader = vtk.vtkOBJReader()
        meshReader.SetFileName(path)
    else:
        ROS_FATAL("Mesh file has invalid extension (" + extension + ")")

    # Scale STL
    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, scale)
    # transform.RotateWXYZ(rot[1], rot[2], rot[3], rot[0])
    # transform.Translate(pos[0],pos[1], pos[2])
    transformFilter = vtk.vtkTransformFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputConnection(meshReader.GetOutputPort())
    transformFilter.Update()
    return transformFilter.GetOutput()

def generateGrid(xmin, xmax, ymin, ymax, res):
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    Xg,Yg = np.meshgrid(x,y)
    grid = np.array([Xg.flatten(), Yg.flatten()]).T

    return grid

def makeSphere(radius):
    # create source
    source = vtk.vtkSphereSource()
    source.SetCenter(0,0,0)
    source.SetRadius(radius)
     
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(source.GetOutput())
    else:
        mapper.SetInputConnection(source.GetOutputPort())
     
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


class UserWidget(QWidget):
    bridge = CvBridge()
    def __init__(self, camera, tfBuffer, markerTopic, robotFrame, tipFrame, cameraFrame, masterWidget=None, parent=None):
        super(UserWidget, self).__init__(parent=parent)
        # Load in variables
        self.tfBuffer = tfBuffer
        self.robotFrame = robotFrame
        self.tipFrame = tipFrame
        self.markerTopic = markerTopic
        self.cameraFrame = cameraFrame
        self.masterWidget = masterWidget

        uiPath = cleanResourcePath("package://dvrk_vision/src/dvrk_vision/overlay_widget.ui")
        # Get CV image from path
        uic.loadUi(uiPath, self)
        self.vtkWidget = QVTKStereoViewer(camera, parent=self)
        self.vtkWidget.renderSetup = self.renderSetup
        self.vtkWidget.imageProc = self.imageProc
        self.pausePOI = True

        # Add vtk widget
        self.vl = QVBoxLayout()
        self.vl.addWidget(self.vtkWidget)
        self.vtkFrame.setLayout(self.vl)

        self.otherWindows = []
        if self.masterWidget is not None:
            self.masterWidget.otherWindows.append(self)
            self.otherWindows.append(self.masterWidget)

        # Set up QT slider for opacity
        self.opacitySlider.valueChanged.connect(self.sliderChanged) 
        self.textureCheckBox.stateChanged.connect(self.checkBoxChanged)

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())

        self.organFrame = None

        self.textureCheckBox.setText("Show GP mesh")

        self.POICheckBox = QCheckBox("Show markers", self)
        self.POICheckBox.setChecked(True)
        self.POICheckBox.stateChanged.connect(self.checkBoxChanged)
        self.horizontalLayout.addWidget(self.POICheckBox)
        self.POI = []

        
        self.clearButton = QPushButton("Clear markers", self)
        self.clearButton.pressed.connect(self.clearPOI)
        self.horizontalLayout.addWidget(self.clearButton)

        self.vtkWidget.Initialize()
        self.vtkWidget.start()

        self.pointSelectPub = rospy.Publisher('/dvrk_vision/user_POI', Point, latch = False, queue_size = 1)
        self.clearPOIPub    = rospy.Publisher('/dvrk_vision/clear_POI', Empty, latch = False, queue_size = 1)
        self.clearPOISub = rospy.Subscriber('/control/clearPOI',Empty,self.clearCB,queue_size=1)
        self.pausePOISub = rospy.Subscriber('/control/pausePOI',Bool,self.pauseCB,queue_size=1)

        # Force bar TEMP TODO: REMOVE THIS FROM BEING HARDCODED
        psmName = rospy.get_param('~psm_name')
        filePath = rospy.get_param('~camera_registration')

        import yaml
        print(filePath)
        with open(filePath, 'r') as f:
            data = yaml.load(f)
        camTransform = data['transform']

        frameRate = 15
        slop = 1.0 / frameRate
        cams = StereoCameras("stereo/left/image_rect",
                         "stereo/right/image_rect",
                         "stereo/left/camera_info",
                         "stereo/right/camera_info",
                         slop = slop)
        self.cameraTransform = arrayToPyKDLFrame(camTransform)
        self.camSync = CameraSync("left/camera_info")
        self.dvrkTopic = '/dvrk/' + psmName + "/position_cartesian_current"
        self.forceTopic = '/dvrk/' + psmName + '_FT/raw_wrench'
        self.camSync.addTopics([self.dvrkTopic, self.forceTopic])

    def sliderChanged(self):
        self.actorOrgan.GetProperty().SetOpacity(self.opacitySlider.value() / 255.0)
        self.gpActor.GetProperty().SetOpacity(self.opacitySlider.value() / 255.0)
        for window in self.otherWindows:
            window.opacitySlider.setValue(self.opacitySlider.value())

    def checkBoxChanged(self):
        if(self.textureCheckBox.isChecked()):
            self.gpActor.VisibilityOn()
        else:
            self.gpActor.VisibilityOff()
        for window in self.otherWindows:
            window.textureCheckBox.setChecked(self.textureCheckBox.isChecked())

    def clearCB(self,emptyData):
        self.clearPOI()

    def pauseCB(self,boolData):
        self.pausePOI = boolData.data

    def clearPOI(self):        
        for actor in self.POI:
            self.actorGroup.RemovePart(actor)
            del actor
        self.POI = []
        self.clearPOIPub.publish(Empty())

    def checkBoxPOIChanged(self):
        if(self.POICheckBox.isChecked()):
            for actor in self.POI:
                actor.VisibilityOn()
        else:
            for actor in self.POI:
                actor.VisibilityOff()
        # self.actorOrgan.textureOnOff(self.textureCheckBox.isChecked())
        # self.actorOrgan.GetProperty().LightingOn()
        for window in self.otherWindows:
            window.POICheckBox.setChecked(self.POICheckBox.isChecked())

    def renderSetup(self):
        if self.masterWidget is not None:
            self.actorGroup = self.masterWidget.actorGroup
            self.sphere = self.masterWidget.sphere
            self.POI = self.masterWidget.POI
            self.gpActor = self.masterWidget.gpActor
            self.actorOrgan = self.masterWidget.actorOrgan
            self.vtkWidget.ren.AddActor(self.actorGroup)
            self.textActor = self.masterWidget.textActor

            # Add force bar
            self.bar = self.masterWidget.bar
            self.forceBar = self.masterWidget.forceBar
            self.greenLine = self.masterWidget.greenLine
            self.vtkWidget.ren.AddActor(self.bar)
            self.vtkWidget.ren.AddActor(self.forceBar)
            self.vtkWidget.ren.AddActor(self.greenLine)
            self.bar.VisibilityOn()
            self.forceBar.VisibilityOn()
            self.greenLine.VisibilityOn()

            # Add text actor
            self.textActor= self.masterWidget.textActor
            self.vecText= self.masterWidget.vecText
            self.vtkWidget.ren.AddActor(self.textActor)
            return

        self.actorGroup = vtk.vtkAssembly()
        [self.textActor,self.vecText] = makeTextActor3D()

        self.sphere = makeSphere(.003)
        self.POI = []

        # Empty variables for organ data
        color = (0,0,1)
        self.actorOrgan = vtkRosTextureActor("stiffness_texture", color = color)
        self.actorOrgan.GetProperty().BackfaceCullingOn()
        self.organPolydata = vtk.vtkPolyData()
        self.converter = None

        # Empty variables for point actor
        self.gpPolyData = vtk.vtkPolyData()
        delaunay = vtk.vtkDelaunay2D()
        if vtk.VTK_MAJOR_VERSION <= 5:
            delaunay.SetInput(self.gpPolyData)
        else:
            delaunay.SetInputData(self.gpPolyData)
        gpMapper = vtk.vtkDataSetMapper()
        gpMapper.SetInputConnection(delaunay.GetOutputPort())

        # Create gp Mapper actor
        self.gpActor = vtk.vtkActor()
        self.gpActor.SetMapper(gpMapper)
        self.gpActor.GetProperty().SetDiffuse(0)
        self.gpActor.GetProperty().SetSpecular(0)
        self.gpActor.GetProperty().SetAmbient(1)

        # Set up subscriber for marker
        self.meshPath = ""
        self.organFrame = None
        markerSub = message_filters.Subscriber(self.markerTopic, Marker)
        markerSub.registerCallback(self.markerCB)

        # Set up subscribers for GP
        self.points = np.empty((0,3))
        self.oldPoints = np.empty((0,3))
        self.stiffness = np.empty(0)
        self.points2D = np.empty((0,2))
        stiffSub = message_filters.Subscriber('/dvrk/GP/get_stiffness', Float64MultiArray)
        pointsSub = message_filters.Subscriber('/dvrk/GP/get_surface_points', Float64MultiArray)
        stiffSub.registerCallback(self.stiffnessCB)
        pointsSub.registerCallback(self.pointsCB)

        # Set texture to default
        self.resolution = 512
        self.texture = np.ones((self.resolution, self.resolution, 3), np.uint8) * 255
        self.actorOrgan.setTexture(self.texture.copy())
        self.actorOrgan.textureOnOff(True)
        self.actorOrgan.GetProperty().LightingOn()

        # Add actors
        self.actorGroup.AddPart(self.actorOrgan)
        self.actorGroup.AddPart(self.gpActor)
        self.vtkWidget.ren.AddActor(self.actorGroup)
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

        # Set up timer callback
        cb = vtkTimerCallback(self.vtkWidget._RenderWindow)
        cb.update = self.update
        self.iren.AddObserver('TimerEvent', cb.execute)
        self.iren.CreateRepeatingTimer(15)

    def setText(self,textInput):
        self.vecText.SetText(textInput)

    def _updateActorPolydata(self,actor,polydata,color=None):
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
        if type(color) !=  type(None):
            actor.GetProperty().SetColor(color[0], color[1], color[2])
        else:
            actor.GetProperty().SetColor(1, 0, 0)
        self.sliderChanged()

    def addPOI(self):
        if self.organFrame is None or self.pausePOI:
            return
        poseRobot = self.tfBuffer.lookup_transform(self.cameraFrame, self.tipFrame, rospy.Time())
        posRobot = poseRobot.transform.translation
        rotRobot = poseRobot.transform.rotation
        matRobot = transformations.quaternion_matrix([rotRobot.x, rotRobot.y, rotRobot.z, rotRobot.w])

        actor = vtk.vtkActor()
        actor.ShallowCopy(self.sphere)
        posTip = np.array([posRobot.x, posRobot.y, posRobot.z])# + np.array(matRobot[3,0:3].tolist())
        actor.SetPosition(posTip[0], posTip[1], posTip[2])

        self.pointSelectPub.publish(Point(posTip[0],posTip[1],posTip[2]))

        self.POI.append(actor)
        self.actorGroup.AddPart(actor);

    def markerCB(self, data):
        meshPath = cleanResourcePath(data.mesh_resource)
        if meshPath != self.meshPath:
            self.meshPath = meshPath
            try:
                self.organFrame = data.header.frame_id
                if self.organFrame[0] == "/":
                    self.organFrame = self.organFrame[1:]
                poseCamera = self.tfBuffer.lookup_transform(self.cameraFrame, self.organFrame, rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(e)
                return

            posCam = poseCamera.transform.translation
            rotCam = poseCamera.transform.rotation
            matCam = transformations.quaternion_matrix([rotCam.x, rotCam.y, rotCam.z, rotCam.w])
            matCam[0:3,3] = [posCam.x, posCam.y, posCam.z]

            transformCam = vtk.vtkTransform()
            transformCam.SetMatrix(matCam.ravel())

            polydata = loadMesh(meshPath, 1)

            # Scale STL
            transform = vtk.vtkTransform()
            transformFilter = vtk.vtkTransformFilter()
            transformFilter.SetTransform(transformCam)
            transformFilter.SetInputData(polydata)
            transformFilter.Update()

            organPolyData = transformFilter.GetOutput()
            self._updateActorPolydata(self.actorOrgan, organPolyData)
            if self.converter != None:
                del(self.converter)
            self.converter = uvtoworld.UVToWorldConverter(organPolyData)

    def pointsCB(self, points):
        # Preetham's stuff is in millimeters
        self.points = multiArrayToMatrixList(points) * 0.001

    def stiffnessCB(self, stiffness):
        self.stiffness = multiArrayToMatrixList(stiffness).transpose()

    def update(self):
        if not self.isVisible():
            return
        if len(self.points) < 2:
            return
        if len(self.points) != len(self.stiffness) or self.meshPath == "":
            return
        if np.all(self.points == self.oldPoints):
            return
        self.oldPoints = self.points

        poseCamera = self.tfBuffer.lookup_transform(self.cameraFrame, self.robotFrame, rospy.Time())
        posCam = poseCamera.transform.translation
        rotCam = poseCamera.transform.rotation
        matCam = transformations.quaternion_matrix([rotCam.x, rotCam.y, rotCam.z, rotCam.w])
        matCam[0:3,3] = [posCam.x, posCam.y, posCam.z]
        points = np.matrix(matCam) * np.hstack((self.points, np.ones((self.points.shape[0],1)))).transpose()
        points = np.array(points.transpose()[:,0:3])
        scalars = self.stiffness

        self.gpPolyData.Reset()
        vtkPoints = vtk.vtkPoints()
        vtkCells = vtk.vtkCellArray()
        colored = True
        if colored:
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            minZ = np.min(scalars)
            maxZ = np.max(scalars)
            stiffness = (scalars - minZ) / (maxZ - minZ)
            r = np.clip(stiffness * 3, 0, 1) * 255
            g = np.clip(stiffness * 3 - 1, 0, 1) * 255
            b = np.clip(stiffness * 3 - 2, 0, 1) * 255
            for i, point in enumerate(np.hstack((points, r, g, b))):
                colors.InsertNextTuple3(point[3], point[4], point[5])
                pointId = vtkPoints.InsertNextPoint(point[0:3])
                vtkCells.InsertNextCell(1)
                vtkCells.InsertCellPoint(pointId)
        else:
            for i, point in enumerate(points):
                pointId = vtkPoints.InsertNextPoint(point)
                vtkCells.InsertNextCell(1)
                vtkCells.InsertCellPoint(pointId)
        self.gpPolyData.SetPoints(vtkPoints)
        self.gpPolyData.SetVerts(vtkCells)
        self.gpPolyData.Modified()
        if colored:
            self.gpPolyData.GetPointData().SetScalars(colors)

        if self.converter == None:
            return   

        # Project points into UV space        
        texCoords = self.converter.toUVSpace(points)[:, 0:2]
        # Flip y coordinates to match image space
        texCoords[:,1] *= -1
        texCoords[:,1] +=  1

        resolution = 100
        grid = generateGrid(0, 1, 0, 1, resolution)
        stiffMap = scipy.interpolate.griddata(texCoords, scalars, grid, method="linear", fill_value=-1)
        stiffMap = stiffMap.reshape(resolution, resolution)
        stiffMap[stiffMap == -1] = np.min(stiffMap[stiffMap != -1])
        
        # Normalize
        stiffMap -= np.min(stiffMap)
        stiffMap /= np.max(stiffMap)
        scale = 255 * 0.3
        r = np.clip(stiffMap * 3, 0, 1) * scale
        g = np.clip(stiffMap * 3 - 1, 0, 1) * scale
        b = np.clip(stiffMap * 3 - 2, 0, 1) * scale
        stiffImg = np.dstack((b, g, r)).astype(np.uint8)
        shape = self.texture.shape
        stiffImg = cv2.resize(stiffImg, (shape[1], shape[0]))
        img = self.texture.copy()
        img = np.subtract(img, stiffImg.astype(int))
        img = np.clip(img, 0, 255).astype(np.uint8)
        self.actorOrgan.setTexture(img)
        self.actorOrgan.textureOnOff(True)
        self.actorOrgan.GetProperty().LightingOn()


    # Bar processing
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

if __name__ == '__main__':
    import sys
    from dvrk_vision import vtktools
    from dvrk_vision.vtk_stereo_viewer import StereoCameras
    app = QApplication(sys.argv)
    rosThread = vtktools.QRosThread()
    rosThread.start()
    # markerTopic = rospy.get_param("~marker_topic")
    # robotFrame = rospy.get_param("~robot_frame")
    # cameraFrame = rospy.get_param("~robot_frame")
    # markerTopic = "/dvrk/MTMR_PSM2/proxy_slave_phantom"
    markerTopic = "/stereo/registration_marker"
    robotFrame = "PSM2_SIM_psm_base_link"
    cameraFrame = "stereo_camera_frame"
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("stereo/left/image_rect",
                         "stereo/right/image_rect",
                         "stereo/left/camera_info",
                         "stereo/right/camera_info",
                         slop = slop)
    windowL = UserWidget(cams.camL, markerTopic, robotFrame, cameraFrame)
    windowL.show()
    # windowR = OverlayWidget(cams.camR, meshPath, scale=stlScale, masterWidget=windowL)
    sys.exit(app.exec_())