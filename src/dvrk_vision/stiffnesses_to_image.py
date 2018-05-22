#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import message_filters
import matplotlib.pyplot as plt
import scipy

# Visualization stuff
import vtk
import dvrk_vision.vtktools as vtktools
from dvrk_vision.clean_resource_path import cleanResourcePath
from dvrk_vision.overlay_gui import vtkRosTextureActor
import cv2
import os
import ctypes

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

class vtkTimerCallback():
    def __init__(self, renWin):
        self.rate = rospy.Rate(5)
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

class StiffnessToImageNode:
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
        self.points = None
        self.oldPoints = None
        self.stiffness = None

        self.visualize = visualize

        if self.visualize:
            # Organ stuff
            meshPath = "package://oct_15_demo/resources/largeProstate.obj"
            texturePath = "package://oct_15_demo/resources/largeProstate.png"
            scale = 1000.06
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
            # Scale STL
            transform = vtk.vtkTransform()
            transform.Scale(scale, scale, scale)
            transform.Translate(0,.015,-.142)
            transform.RotateX(110)
            transformFilter = vtk.vtkTransformFilter()
            transformFilter.SetTransform(transform)
            transformFilter.SetInputConnection(meshReader.GetOutputPort())
            transformFilter.Update()
            self.organPolydata = transformFilter.GetOutput()
            color = (0,0,1)
            self.actor_organ = vtkRosTextureActor("stiffness_texture", color = color)
            self.actor_organ.GetProperty().BackfaceCullingOn()
            self._updateActorPolydata(self.actor_organ,
                                      polydata=  self.organPolydata,
                                      color = color)

            # Set texture to default
            self.texture = cv2.imread(cleanResourcePath(texturePath))
            self.actor_organ.setTexture(self.texture.copy())
            self.actor_organ.textureOnOff(True)
            
            # Build cell search structure
            self.cellLocator = vtk.vtkCellLocator()
            self.cellLocator.SetDataSet(transformFilter.GetOutput())
            self.cellLocator.BuildLocator()

            # Make 2D Coordinates for lookup
            tCoords = self.organPolydata.GetPointData().GetTCoords()
            nTuples = tCoords.GetNumberOfTuples()
            tCoordPoints = vtk.vtkFloatArray()
            tCoordPoints.SetNumberOfComponents(3)
            # tCoordPoints.SetNumberOfTuples(3)
            tCoordPoints.Allocate(nTuples*3)
            tCoordPoints.SetNumberOfTuples(nTuples)
            tCoordPoints.CopyComponent(0, tCoords, 0)
            tCoordPoints.CopyComponent(1, tCoords, 1)
            tCoordPoints.FillComponent(2,0)
            pts = vtk.vtkPoints()
            pts.SetData(tCoordPoints)
            self.polyData2D = vtk.vtkPolyData()
            self.polyData2D.SetPoints(pts)
            self.polyData2D.SetPolys(self.organPolydata.GetPolys())
            self.polyData2D.BuildCells()

            self.ren = vtk.vtkRenderer()
            self.polyData = vtk.vtkPolyData()
            # delaunay = vtk.vtkDelaunay2D()
            # if vtk.VTK_MAJOR_VERSION <= 5:
            #     delaunay.SetInput(self.polyData)
            # else:
            #     delaunay.SetInputData(self.polyData)
            # mapper = vtk.vtkDataSetMapper()
            # mapper.SetInputConnection(delaunay.GetOutputPort())
            self.ren.AddActor(self.actor_organ)
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
            cb = vtkTimerCallback(self.renWin)
            cb.update = self.update
            self.iren.AddObserver('TimerEvent', cb.execute)
            self.iren.CreateRepeatingTimer(100)
            self.iren.Start()

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

    def updatePoints(self, points, scalars):
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
        cell = vtk.vtkGenericCell()
        cell.SetCellTypeToTriangle ()

        tolerance = .01
        t = vtk.mutable(0)
        subId = vtk.mutable(0)
        cellId = vtk.mutable(0)
        dist = vtk.mutable(0)
        pos = [0.0, 0.0, 0.0]
        pos2 = [0.0,0.0,0.0]
        pcoords = [0.0, 0.0, 0.0]
        tCoords = [0.0 ,0., 0.0]
        weights = [0.0, 0.0, 0.0]
        shape = self.texture.shape

        img = self.texture.copy()

        projectedPoints = np.empty((len(points), 2))

        for idx, point in enumerate(points):
            start = point + [0,0, 5]
            end = point + [0,0,-5]
            # self.cellLocator.IntersectWithLine(start, end, tolerance, t, pos, pcoords, subId, cellId, cell)
            self.cellLocator.FindClosestPoint(point, pos, cell, cellId, subId, dist)
            cell.EvaluatePosition(point, pos2, subId, pcoords, dist, weights)
            self.polyData2D.GetCell(cellId).EvaluateLocation(subId, pcoords, tCoords, weights)
            texCoords = [tCoords[0] * shape[0], (1 - tCoords[1]) * self.texture.shape[1]]
            projectedPoints[idx, :] = texCoords
            texCoords = tuple(int(a) for a in texCoords)
            # img = cv2.circle(img, texCoords, 5, (0,255,0), -1)

        resolution = 100
        grid = generateGrid(0, shape[1], 0, shape[0], resolution)
        stiffMap = scipy.interpolate.griddata(projectedPoints, scalars, grid, method="linear", fill_value=-1)
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
        stiffImg = cv2.resize(stiffImg, (shape[1], shape[0]))
        img = np.subtract(img, stiffImg.astype(int))
        img = np.clip(img, 0, 255).astype(np.uint8)
        self.actor_organ.setTexture(img)
        # self.renWin.Render()

    def update(self):
        if self.points is None:
            return
        if np.all(self.points == self.oldPoints):
            return
        self.oldPoints = self.points
        pointsMat = self.points
        stiffMat = self.stiffness
        if len(pointsMat) != len(stiffMat):
            return
        assert pointsMat.shape[1] == 3
        # self.gp.fit(pointsMat[:,0:2], stiffMat)
        # stiffMap = self.gp.predict(self.grid)
        # stiffMap = stiffMap.reshape(self.resolution, self.resolution)
        # stiffMap = scipy.interpolate.griddata(pointsMat[:,0:2], stiffMat, self.grid, method="linear", fill_value=-1).reshape(self.resolution, self.resolution)
        # stiffMap[stiffMap == -1] = np.min(stiffMap[stiffMap != -1])
        # # print(np.min(stiffMap), np.max(stiffMap), np.min(stiffMat), np.max(stiffMat))
        # # Normalize
        # stiffMap[stiffMap < np.mean(stiffMap)] = np.mean(stiffMap)
        # stiffMap -= np.min(stiffMap)
        # stiffMap /= np.max(stiffMap)
        # stiffMap *= 255
        self.updatePoints(pointsMat, stiffMat)
        # plt.figure(1)
        # plt.clf()
        # plt.imshow(stiffMap, origin='lower', cmap="hot", extent=self.domain)
        # plt.colorbar()
        # plt.scatter(pointsMat[:,0],pointsMat[:,1])
        # plt.pause(0.05)
    
    def gp_init(self):
        # kernel = C(1.0, (1e-3, 1e3))*RBF(6, (1e-2, 1e2))
        # gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, n_restarts_optimizer=9)
        kernel = RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=9)
        return gp

if __name__ == '__main__':
    node = StiffnessToImageNode()

