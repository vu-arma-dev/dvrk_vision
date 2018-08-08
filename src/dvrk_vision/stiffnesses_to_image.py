#!/usr/bin/env python
import os
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import message_filters
import matplotlib.pyplot as plt
import scipy.interpolate
import vtk
import cv2
from dvrk_vision.clean_resource_path import cleanResourcePath
import dvrk_vision.uvtoworld as uvtoworld
from cv_bridge import CvBridge, CvBridgeError
import yaml
import tf2_ros
from tf import transformations
from dvrk_vision.overlay_gui import vtkRosTextureActor

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

def generateGrid(xmin, xmax, ymin, ymax, res):
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    Xg,Yg = np.meshgrid(x,y)
    grid = np.array([Xg.flatten(), Yg.flatten()]).T

    return grid

class StiffnessToImageNode(object):
    bridge = CvBridge()
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    def __init__(self, visualize=True):
        rospy.init_node('stiffness_to_image_converter', anonymous=True)
        self.domain = [0, 1, 0, 1]
        self.resolution = 100
        # self.gp = self.gp_init()
        # Publishers and subscribers
        self.imagePub = rospy.Publisher('/stereo/stiffness_image', Image, queue_size = 1)
        stiffSub = message_filters.Subscriber('/dvrk/GP/get_stiffness', Float64MultiArray)
        pointsSub = message_filters.Subscriber('/dvrk/GP/get_surface_points', Float64MultiArray)
        stiffSub.registerCallback(self.stiffnessCB)
        pointsSub.registerCallback(self.pointsCB)
        self.points = np.empty((0,2))
        self.oldPoints = np.empty((0,3))
        self.stiffness = np.empty(0)
        self.points2D = np.empty((0,3))

        # Get Rospy params
        markerTopic = rospy.get_param("~marker_topic")
        self.scale = rospy.get_param("~marker_scale")
        self.robotFrame = rospy.get_param("~robot_frame")
        if self.robotFrame[0] == "/":
            self.robotFrame = self.robotFrame[1:]
        stiffSub = message_filters.Subscriber(markerTopic, Marker)

        # Empty variables for organ data
        color = (0,0,1)
        self.actor_organ = vtkRosTextureActor("stiffness_texture", color = color)
        self.organPolydata = vtk.vtkPolyData()
        self.converter = None

        # Set up subscriber for marker
        self.meshPath = ""
        markerSub = message_filters.Subscriber(markerTopic, Marker)
        markerSub.registerCallback(self.markerCB)

        # filePath = rospy.get_param('~registration_yaml')
        # print(filePath)
        # with open(filePath, 'r') as f:
        #     data = yaml.load(f)
        # pos = data['position']
        # rot = data['quaternion']

       

        # Set texture to default
        self.resolution = 512
        self.texture = np.ones((self.resolution, self.resolution, 3), np.uint8) * 255
        # self.texture = cv2.imread(cleanResourcePath(texturePath))
        # self.converter = uvtoworld.UVToWorldConverter(self.organPolydata)

        if visualize:
            self.visualize = visualize
            self.ren = vtk.vtkRenderer()

            self.actor_organ.GetProperty().BackfaceCullingOn()
            self._updateActorPolydata(self.actor_organ,
                                      polydata= self.organPolydata,
                                      color = color)
            self.actor_organ.setTexture(self.texture.copy())
            self.actor_organ.textureOnOff(True)
            self.actor_organ.GetProperty().LightingOn()
            self.ren.AddActor(self.actor_organ)

            self.polyData = vtk.vtkPolyData()
            delaunay = vtk.vtkDelaunay2D()
            if vtk.VTK_MAJOR_VERSION <= 5:
                delaunay.SetInput(self.polyData)
            else:
                delaunay.SetInputData(self.polyData)
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(delaunay.GetOutputPort())
            # mapper = vtk.vtkPolyDataMapper()
            # mapper.SetScalarVisibility(1)
            # if vtk.VTK_MAJOR_VERSION <= 5:
            #     mapper.SetInput(self.polyData)
            # else:
            #     mapper.SetInputData(self.polyData)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetDiffuse(0)
            actor.GetProperty().SetSpecular(0)
            actor.GetProperty().SetAmbient(1)
            self.ren.AddActor(actor)

            self.renWin = vtk.vtkRenderWindow()
            self.renWin.AddRenderer(self.ren)
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetRenderWindow(self.renWin)
            self.renWin.Render()
            cb = vtkTimerCallback(self.renWin)
            cb.update = self.update
            self.iren.AddObserver('TimerEvent', cb.execute)
            self.iren.CreateRepeatingTimer(15)
            self.iren.Start()

        else:
            rate = rospy.Rate(100)
            while not rospy.is_shutdown():
                self.update()
                rate.sleep()

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

    def pointsCB(self, points):
        # Preetham's stuff is in millimeters
        self.points = multiArrayToMatrixList(points) * 0.001

    def stiffnessCB(self, stiffness):
        self.stiffness = multiArrayToMatrixList(stiffness).transpose()

    def getNewPoints(self):
        # Turn to binary for easy comparison
        pointsNewB = [a.tobytes() for a in self.points]
        pointsOldB = [a.tobytes() for a in self.oldPoints]
        if len(pointsOldB) < len(pointsNewB):
            pointsOldB.append([np.nan] * (len(pointsNewB) - len(pointsOldB)))
        else:
            pointsOldB = pointsOldB[:len(pointsNewB)]
        pointsNew = np.setxor1d(pointsNewB, pointsOldB)
        pointsNew = np.array([np.frombuffer(a, count=3) for a in pointsNew])
        pointsOld = np.setand1d(pointsNewB, pointsOldB)
        pointsOld = np.array([np.frombuffer(a, count=3) for a in pointsOld])
        return pointsNew, pointsOld

    def loadMesh(self, path, scale):
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
        transform.Scale(self.scale, self.scale, self.scale)
        # transform.RotateWXYZ(rot[1], rot[2], rot[3], rot[0])
        # transform.Translate(pos[0],pos[1], pos[2])
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(meshReader.GetOutputPort())
        transformFilter.Update()
        self.organPolydata = transformFilter.GetOutput()
        self._updateActorPolydata(self.actor_organ, self.organPolydata)

    def markerCB(self, data):
        meshPath = cleanResourcePath(data.mesh_resource)
        if meshPath != self.meshPath:
            self.loadMesh(meshPath, 1)
            self.meshPath = meshPath
        try:
            organFrame = data.header.frame_id
            if organFrame[0] == "/":
                organFrame = organFrame[1:]
            pose = self.tfBuffer.lookup_transform(self.robotFrame, organFrame, rospy.Time()).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(e)
            return

        pos = pose.translation
        rot = pose.rotation
        mat = transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        mat[0:3,3] = [pos.x, pos.y, pos.z]

        transform = vtk.vtkTransform()
        transform.SetMatrix(mat.ravel())

        self.actor_organ.SetPosition(transform.GetPosition())
        self.actor_organ.SetOrientation(transform.GetOrientation())
        self.actor_organ.VisibilityOn()

    def update(self):
        if len(self.points) != len(self.stiffness):
            return
        points = self.points
        scalars = self.stiffness
        if np.all(self.points == self.oldPoints):
            return

        self.oldPoints = self.points

        if self.visualize:
            self.polyData.Reset()
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
            self.polyData.SetPoints(vtkPoints)
            self.polyData.SetVerts(vtkCells)
            self.polyData.Modified()
            if colored:
                self.polyData.GetPointData().SetScalars(colors)

        

        # # self.getNewPoints()
        # # Project points into UV space        
        # texCoords = self.converter.toUVSpace(points)[:, 0:2]
        # # Flip y coordinates to match image space
        # texCoords[:,1] *= -1
        # texCoords[:,1] +=  1

        # resolution = 100
        # grid = generateGrid(0, 1, 0, 1, resolution)
        # stiffMap = scipy.interpolate.griddata(texCoords, scalars, grid, method="linear", fill_value=-1)
        # stiffMap = stiffMap.reshape(resolution, resolution)
        # stiffMap[stiffMap == -1] = np.min(stiffMap[stiffMap != -1])
        # # print(np.min(stiffMap), np.max(stiffMap), np.min(stiffMat), np.max(stiffMat))
        # # Normalize
        # # stiffMap[stiffMap < np.mean(stiffMap)] = np.mean(stiffMap)
        # stiffMap -= np.min(stiffMap)
        # stiffMap /= np.max(stiffMap)
        # scale = 255 * 0.3
        # r = np.clip(stiffMap * 3, 0, 1) * scale
        # g = np.clip(stiffMap * 3 - 1, 0, 1) * scale
        # b = np.clip(stiffMap * 3 - 2, 0, 1) * scale
        # stiffImg = np.dstack((b, g, r)).astype(np.uint8)
        # shape = self.texture.shape
        # stiffImg = cv2.resize(stiffImg, (shape[1], shape[0]))
        # img = self.texture.copy()
        # img = np.subtract(img, stiffImg.astype(int))
        # img = np.clip(img, 0, 255).astype(np.uint8)

        # msg = self.bridge.cv2_to_imgmsg(image, 'rgb8')
        # self.imagePub.publish(msg)
        
    
    def gp_init(self):
        # kernel = C(1.0, (1e-3, 1e3))*RBF(6, (1e-2, 1e2))
        # gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, n_restarts_optimizer=9)
        kernel = RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=9)
        return gp

if __name__ == '__main__':
    node = StiffnessToImageNode()

