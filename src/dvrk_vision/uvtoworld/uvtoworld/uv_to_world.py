""" UV to world conversion tools

This module contains tools to convert 2D UV coordinates to 3D world
coordinates using VTK

    TODO:
        * Make ROS node
"""
import vtk
import numpy as np
from vtk.util import numpy_support
import time

from IPython import embed

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

__all__ = ["makeTexturedObjData","CleanTexturedPolyData","pointToBarycentric", "UVToWorldConverter"]

def makeTexturedObjData(objPath, scale=1):
    """ Loads .obj into VTK polyData optimized for searching texture space. 
    
    Args:
        objPath (string): File path to .obj file to load

    Returns:
        polyData (vtk.vtkPolyData): VTK polyData object optimized for finding
            mapping from 2D texture coordinates to 3D world coordinates
    """
    meshReader = vtk.vtkOBJReader()
    meshReader.SetFileName(objPath)
    triFilter = vtk.vtkTriangleFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        triFilter.SetInput(meshReader.GetOutput())
    else:
        triFilter.SetInputConnection(meshReader.GetOutputPort())
    transform = vtk.vtkTransform()
    transform.Scale(scale,scale,scale)
    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    if vtk.VTK_MAJOR_VERSION <= 5:
        triFilter.SetInput(meshReader.GetOutput())
    else:
        transformFilter.SetInputConnection(triFilter.GetOutputPort())
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ComputeCellNormalsOn()
    if vtk.VTK_MAJOR_VERSION <= 5:
        normalGenerator.SetInput(transformFilter.GetOutput())
    else:
        normalGenerator.SetInputConnection(transformFilter.GetOutputPort())
    normalGenerator.Update()
    polyData = normalGenerator.GetOutput()
    return polyData

class UVToWorldConverter:
    """ This class is used to convert 2D texture coordinates to 3D object space.

    Args:
        data (vtk.vtkPolyData): VTK object which contains geometry and
            texture coordinates
    """
    def __init__(self, data):
        self.polyData = data
        self.polyData.BuildCells()
        self.polyData.BuildLinks()
        self.tCoords = data.GetPointData().GetTCoords()
        self.points = data.GetPoints()

        self.npTCoords = np.empty((data.GetPolys().GetNumberOfCells(),6))
        self.npPolys = np.empty((data.GetPolys().GetNumberOfCells(),3), dtype=np.uint16)

        nTuples = self.tCoords.GetNumberOfTuples()
        tCoordPoints = vtk.vtkFloatArray()
        tCoordPoints.SetNumberOfComponents(3)
        # tCoordPoints.SetNumberOfTuples(3)
        tCoordPoints.Allocate(nTuples*3)
        tCoordPoints.SetNumberOfTuples(nTuples)
        tCoordPoints.CopyComponent(0, self.tCoords, 0)
        tCoordPoints.CopyComponent(1, self.tCoords, 1)
        tCoordPoints.FillComponent(2,0)
        self.polyData2D = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(tCoordPoints)
        self.polyData2D.SetPoints(points)
        self.polyData2D.SetPolys(data.GetPolys())
        self.polyData2D.BuildCells()
        self.polyData2D.BuildLinks()

        self.pointLocator = vtk.vtkCellLocator()
        self.pointLocator.SetDataSet(data)
        self.pointLocator.BuildLocator()

        self.pointLocator2D = vtk.vtkCellLocator()
        self.pointLocator2D.SetDataSet(self.polyData2D)
        self.pointLocator2D.BuildLocator()

        # Reused variables
        self._cell = vtk.vtkGenericCell()
        self._cell.SetCellTypeToTriangle ()

    # @timing
    def toWorldSpace(self,p):
        """ This function converts a 2D texture coordinate to 3D object space.
        
        Args:
            p (iterable of type float): 2D texture coordinate in (x,y) format 
                with x and y values between 0 and 1 representing a relative
                position on the image texure.

        Returns:
            worldPoint(iterable of type float): 3D euclidian coordinate of the
                point corresponding to the 2D texture coordinate 'p' in the VTK
                polyData object's frame.
            normalVector(iterable of type float): 3D vector (x,y,z) representing
                the normal vector on the current face.

        TODO:
            * Return color data
        """

        # Add dimensions if necessary
        points2D = np.atleast_2d(p)
        if points2D.shape[-1] == 2:
            points2D = np.hstack((points2D, np.zeros((1,len(points2D)))))
        if points2D.ndim != 2 or points2D.shape[1] != 3:
            raise TypeError('toWorldSpace: input point must be 2xN or 3xN')

        subId = vtk.mutable(0)
        cellId = vtk.mutable(0)
        dist = vtk.mutable(0)
        closest = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        tCoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0]

        points3D = np.empty(points2D.shape)
        normals = np.zeros(points2D.shape)

        for idx, point2D in enumerate(points2D):
            self.pointLocator2D.FindClosestPoint(point2D, closest, self._cell, cellId, subId, dist)
            self._cell.EvaluatePosition(point2D, closest, subId, pcoords, dist, weights)
            self.polyData.GetCell(cellId).EvaluateLocation(subId, pcoords, tCoords, weights)
            points3D[idx,:] = tCoords
            pointIds = self.polyData.GetCell(cellId).GetPointIds()
            normals[idx,:] = np.array(self.polyData.GetCellData().GetNormals().GetTuple(cellId))
        # If only queried one point, only return one point, no need for 2D array
        if len(points3D) == 1:
            return points3D[0], normals[0]
        return points3D, normals

    # @timing
    def toUVSpace(self,p):
        """ This function converts a 3D world space to 2D texture coordinates.
        
        Args:
            p (iterable of type float): 2D texture coordinate in (x,y) format 
                with x and y values between 0 and 1 representing a relative
                position on the image texure.

        Returns:
            worldPoint(iterable of type float): 3D euclidian coordinate of the
                point corresponding to the 2D texture coordinate 'p' in the VTK
                polyData object's frame.
            normalVector(iterable of type float): 3D vector (x,y,z) representing
                the normal vector on the current face.

        TODO:
            * Return color data
        """

        # Add dimensions if necessary
        points3D = np.atleast_2d(p)
        if points3D.ndim != 2 or points3D.shape[1] != 3:
            raise TypeError('toWorldSpace: input point must be 3xN')

        subId = vtk.mutable(0)
        cellId = vtk.mutable(0)
        dist = vtk.mutable(0)
        closest = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        tCoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0]

        points2D = np.empty(points2D.shape)

        for idx, point3D in enumerate(points3D):
            self.pointLocator.FindClosestPoint(point3D, closest, self._cell, cellId, subId, dist)
            self._cell.EvaluatePosition(point3D, closest, subId, pcoords, dist, weights)
            if type(pcoords) == str:
                raise TypeError('VTK function EvaluatePosition returned a null pointer. please use either ' +
                                'a newer version of VTK or build the dvrk_vision package with vtkCell.h')
            self.polyData2D.GetCell(cellId).EvaluateLocation(subId, pcoords, tCoords, weights)
            points2D[idx,:] = tCoords

        # If only queried one point, only return one point, no need for 2D array
        if len(points2D) == 1:
            return points2D[0]
        return points2D

if __name__ == '__main__':
    import sys, getopt
    objFile = ''
    texFile = ''
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        elif o in ("-o", "--obj"):
            inputfile = arg
        elif o in ("-i", "--image"):
            inputfile = arg

    polyData = makeTexturedObjData('Jesus_Unity.obj')
    uvConverter = UVToWorldConverter(polyData)
    print uvConverter.toWorldSpace((.3,.8))
    print uvConverter.toWorldSpace((-1,0))
