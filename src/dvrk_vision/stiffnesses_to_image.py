#!/usr/bin/env python
import os
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import message_filters
import matplotlib.pyplot as plt
import scipy
import vtk
import cv2
from dvrk_vision.clean_resource_path import cleanResourcePath
import dvrk_vision.uvtoworld as uvtoworld
from cv_bridge import CvBridge, CvBridgeError
import yaml

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

        meshPath = rospy.get_param("~mesh_path")
        scale = rospy.get_param("~mesh_scale")
        texturePath = rospy.get_param("~texture_path")

        # Read in STL
        meshPath = cleanResourcePath(meshPath)
        extension = os.path.splitext(meshPath)[1]
        if extension == ".stl" or extension == ".STL":
            meshReader = vtk.vtkSTLReader()
        elif extension == ".obj" or extension == ".OBJ":
            meshReader = vtk.vtkOBJReader()
        else:
            ROS_FATAL("Mesh file has invalid extension (" + extension + ")")
        meshReader.SetFileName(meshPath)

        filePath = rospy.get_param('~registration_yaml')
        print(filePath)
        with open(filePath, 'r') as f:
            data = yaml.load(f)
        pos = data['position']
        rot = data['quaternion']

        # Scale STL
        transform = vtk.vtkTransform()
        transform.Scale(scale, scale, scale)
        transform.RotateWXYZ(rot[1], rot[2], rot[3], rot[0])
        transform.Translate(pos[0],pos[1], pos[2])
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(meshReader.GetOutputPort())
        transformFilter.Update()
        self.organPolydata = transformFilter.GetOutput()
        # Set texture to default
        self.texture = cv2.imread(cleanResourcePath(texturePath))
        self.converter = uvtoworld.UVToWorldConverter(self.organPolydata)

        if visualize:
            from dvrk_vision.overlay_gui import vtkRosTextureActor
            self.visualize = visualize
            self.ren = vtk.vtkRenderer()

            color = (0,0,1)
            self.actor_organ = vtkRosTextureActor("stiffness_texture", color = color)
            self.actor_organ.GetProperty().BackfaceCullingOn()
            self._updateActorPolydata(self.actor_organ,
                                      polydata=  self.organPolydata,
                                      color = color)
            self.actor_organ.setTexture(self.texture.copy())
            self.actor_organ.textureOnOff(True)
            self.ren.AddActor(self.actor_organ)

            self.polyData = vtk.vtkPolyData()
            # delaunay = vtk.vtkDelaunay2D()
            # if vtk.VTK_MAJOR_VERSION <= 5:
            #     delaunay.SetInput(self.polyData)
            # else:
            #     delaunay.SetInputData(self.polyData)
            # mapper = vtk.vtkDataSetMapper()
            # mapper.SetInputConnection(delaunay.GetOutputPort())
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetScalarVisibility(1)
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(self.polyData)
            else:
                mapper.SetInputData(self.polyData)
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
            # cb = vtkTimerCallback(self.renWin)
            # cb.update = self.update
            # self.iren.AddObserver('TimerEvent', cb.execute)
            # self.iren.CreateRepeatingTimer(15)
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
        self.points = multiArrayToMatrixList(points)

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

    def poseCallback(self, data):
        pos = data.pose.position
        rot = data.pose.orientation
        mat = transformations.quaternion_matrix([rot.x,rot.y,rot.z,rot.w])
        mat[0:3,3] = [pos.x,pos.y,pos.z]
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat.ravel())
        # self.actor_moving.SetPosition(transform.GetPosition())
        # self.actor_moving.SetOrientation(transform.GetOrientation())
        # self.actor_moving.VisibilityOn()

    def update(self):
        if len(self.points) != len(self.stiffness):
            return
        points = self.points
        scalars = self.stiffness
        if np.all(self.points == self.oldPoints):
            return

        self.getNewPoints()
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
        # print(np.min(stiffMap), np.max(stiffMap), np.min(stiffMat), np.max(stiffMat))
        # Normalize
        # stiffMap[stiffMap < np.mean(stiffMap)] = np.mean(stiffMap)
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

        msg = self.bridge.cv2_to_imgmsg(image, 'rgb8')
        self.imagePub.publish(msg)

        self.oldPoints = self.points

        if self.visualize:
            self.polyData.Reset()
            vtkPoints = vtk.vtkPoints()
            vtkCells = vtk.vtkCellArray()
            # colors = vtk.vtkUnsignedCharArray()
            # colors.SetNumberOfComponents(3)
            # colors.SetName("Colors")
            # minZ = np.min(scalars)
            # maxZ = np.max(scalars)
            # stiffness = (scalars - minZ) / (maxZ - minZ)
            # r = np.clip(stiffness * 3, 0, 1) * 255
            # g = np.clip(stiffness * 3 - 1, 0, 1) * 255
            # b = np.clip(stiffness * 3 - 2, 0, 1) * 255
            # for i, point in enumerate(np.hstack((points, r, g, b))):
            for i, point in enumerate(points):
                pointId = vtkPoints.InsertNextPoint(point)
                vtkCells.InsertNextCell(1)
                vtkCells.InsertCellPoint(pointId)
            self.polyData.SetPoints(vtkPoints)
            self.polyData.SetVerts(vtkCells)
            self.polyData.Modified()
            # self.polyData.GetPointData().SetScalars(colors)

            self.actor_organ.setTexture(img)
        
    
    def gp_init(self):
        # kernel = C(1.0, (1e-3, 1e3))*RBF(6, (1e-2, 1e2))
        # gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, n_restarts_optimizer=9)
        kernel = RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=9)
        return gp

if __name__ == '__main__':
    node = StiffnessToImageNode()

