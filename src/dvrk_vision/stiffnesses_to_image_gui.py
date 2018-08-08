#!/usr/bin/env python
import vtk
# Which PyQt we use depends on our vtk version. QT4 causes segfaults with vtk > 6
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
    from PyQt5 import uic
    _QT_VERSION = 5
else:
    from PyQt4.QtGui import QWidget, QVBoxLayout, QApplication
    from PyQt4 import uic
    _QT_VERSION = 4
# General imports
import numpy as np
import os
# Ros specific
import rospy
import message_filters
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import tf2_ros
from tf import transformations
from cv_bridge import CvBridge, CvBridgeError
# DVRK vision stuff
from dvrk_vision.overlay_gui import vtkRosTextureActor
from dvrk_vision.vtk_stereo_viewer import QVTKStereoViewer
from dvrk_vision.clean_resource_path import cleanResourcePath

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

    return np.array(ma_msg.data, dtype=np.float64).reshape((rows, columns))

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

class GpOverlayWidget(QWidget):
    bridge = CvBridge()
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    def __init__(self, camera, markerTopic, robotFrame, cameraFrame, masterWidget=None, parent=None):
        super(GpOverlayWidget, self).__init__(parent=parent)
        # Load in variables
        self.robotFrame = robotFrame
        self.markerTopic = markerTopic
        self.cameraFrame = cameraFrame
        self.masterWidget = masterWidget

        uiPath = cleanResourcePath("package://dvrk_vision/src/dvrk_vision/overlay_widget.ui")
        # Get CV image from path
        uic.loadUi(uiPath, self)
        self.vtkWidget = QVTKStereoViewer(camera, parent=self)
        self.vtkWidget.renderSetup = self.renderSetup

        # Add vtk widget
        self.vl = QVBoxLayout()
        self.vl.addWidget(self.vtkWidget)
        self.vtkFrame.setLayout(self.vl)

        self.otherWindows = []
        if type(self.masterWidget) != type(None):
            self.masterWidget.otherWindows.append(self)
            self.otherWindows.append(self.masterWidget)

        # Set up QT slider for opacity
        self.opacitySlider.valueChanged.connect(self.sliderChanged) 
        self.textureCheckBox.stateChanged.connect(self.checkBoxChanged)

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())


        self.vtkWidget.Initialize()
        self.vtkWidget.start()



    def sliderChanged(self):
        self.actorOrgan.GetProperty().SetOpacity(self.opacitySlider.value() / 255.0)
        for window in self.otherWindows:
            window.opacitySlider.setValue(self.opacitySlider.value())

    def checkBoxChanged(self):
        self.actorOrgan.textureOnOff(self.textureCheckBox.isChecked())
        self.actorOrgan.GetProperty().LightingOn()
        for window in self.otherWindows:
            window.textureCheckBox.setChecked(self.textureCheckBox.isChecked())

    def renderSetup(self):
        if type(self.masterWidget) != type(None):
            self.actorOrgan = self.masterWidget.actorOrgan
            return

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
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetScalarVisibility(1)
        # if vtk.VTK_MAJOR_VERSION <= 5:
        #     mapper.SetInput(self.polyData)
        # else:
        #     mapper.SetInputData(self.polyData)
        gpActor = vtk.vtkActor()
        gpActor.SetMapper(gpMapper)
        gpActor.GetProperty().SetDiffuse(0)
        gpActor.GetProperty().SetSpecular(0)
        gpActor.GetProperty().SetAmbient(1)

        # Set up subscriber for marker
        self.meshPath = ""
        markerSub = message_filters.Subscriber(markerTopic, Marker)
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
        self.vtkWidget.ren.AddActor(self.actorOrgan)
        self.vtkWidget.ren.AddActor(gpActor)

        # Set up timer callback
        cb = vtkTimerCallback(self.vtkWidget._RenderWindow)
        cb.update = self.update
        self.iren.AddObserver('TimerEvent', cb.execute)
        self.iren.CreateRepeatingTimer(15)

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

    def markerCB(self, data):
        meshPath = cleanResourcePath(data.mesh_resource)
        if meshPath != self.meshPath:
            self.meshPath = meshPath
            try:
                organFrame = data.header.frame_id
                if organFrame[0] == "/":
                    organFrame = organFrame[1:]
                poseCamera = self.tfBuffer.lookup_transform(self.cameraFrame, organFrame, rospy.Time())
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
            # transform.Scale(scale, scale, scale)
            # transform.RotateWXYZ(rot[1], rot[2], rot[3], rot[0])
            # transform.Translate(pos[0],pos[1], pos[2])
            transformFilter = vtk.vtkTransformFilter()
            transformFilter.SetTransform(transformCam)
            transformFilter.SetInputData(polydata)
            transformFilter.Update()

            self._updateActorPolydata(self.actorOrgan, transformFilter.GetOutput())

            # self.actorOrgan.SetPosition(transformCam.GetPosition())
            # self.actorOrgan.SetOrientation(transformCam.GetOrientation())
            # self.actorOrgan.VisibilityOn()

    def pointsCB(self, points):
        # Preetham's stuff is in millimeters
        self.points = multiArrayToMatrixList(points) * 0.001

    def stiffnessCB(self, stiffness):
        self.stiffness = multiArrayToMatrixList(stiffness).transpose()

    def update(self):
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
    markerTopic = "/dvrk/MTMR_PSM2/proxy_slave_phantom"
    robotFrame = "PSM2_SIM_psm_base_link"
    cameraFrame = "stereo_camera_frame"
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("stereo/left/image_rect",
                         "stereo/right/image_rect",
                         "stereo/left/camera_info",
                         "stereo/right/camera_info",
                         slop = slop)
    windowL = GpOverlayWidget(cams.camL, markerTopic, robotFrame, cameraFrame)
    windowL.show()
    # windowR = OverlayWidget(cams.camR, meshPath, scale=stlScale, masterWidget=windowL)
    sys.exit(app.exec_())