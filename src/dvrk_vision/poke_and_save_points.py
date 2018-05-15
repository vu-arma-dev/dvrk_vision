#!/usr/bin/env python
import rospy
from dvrk import psm
from geometry_msgs.msg import PoseStamped
from tf_conversions import posemath
from tf import transformations
import numpy as np
from stl import mesh
import os
import yaml
import PyKDL
from IPython import embed
import vtk
import sys
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
import time
from QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from dvrk_vision.vtktools import QRosThread
from IPython import embed


class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal()

    def __init__(self, robot, parent=None, **kw):
        super(Worker, self).__init__(parent, **kw)
        self.robot = robot
        self.probedPoints = []
        self.probedPoints = np.loadtxt('./results/03/probed_points.txt').tolist()

        self.pos = [0,0,0]

    @pyqtSlot()
    def procCounter(self): # A slot takes no params


        scriptDirectory = os.path.dirname(os.path.abspath(__file__))
        filePath = os.path.join(scriptDirectory, '..', '..', 'defaults', 
                                'registration_params.yaml')
        with open(filePath, 'r') as f:
            data = yaml.load(f)


        camTransform = np.array(data['transform'])
        thresh = 3
        poked = False
        rate = rospy.Rate(15) # 15hz
        while not rospy.is_shutdown():
            pos = self.robot.get_current_position()
            zVector = pos.M.UnitZ()
            offset = np.array([zVector.x(), zVector.y(), zVector.z()])
            offset = offset * 0.008
            self.pos = np.array([pos.p.x(), pos.p.y(), pos.p.z()]) + offset

            norm = np.array([zVector.x(), zVector.y(), zVector.z()]) * -1
            force = self.robot.get_current_wrench_body()[0:3]
            force = np.dot(force, norm)
            if force > thresh and poked == False:
                poked = True
            elif force < thresh/2 and poked == True:
                self.probedPoints.append(self.pos.tolist())
                print("POKED", str(len(self.probedPoints)).zfill(4), self.pos)
                poked = False


            rate.sleep()
            self.intReady.emit()

        self.finished.emit()

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.GetProperty().SetPointSize(5);
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

class MainWindow(QtWidgets.QMainWindow):
 
    def __init__(self, stlScale = 0.001, parent = None):
        QtWidgets.QMainWindow.__init__(self, parent)
 
        self.frame = QtWidgets.QFrame()
 
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)
 
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
 
        # Create sphere
        source = vtk.vtkSphereSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(0.003)
        sphereMapper = vtk.vtkPolyDataMapper()
        sphereMapper.SetInputConnection(source.GetOutputPort())
        self.sphereActor = vtk.vtkActor()
        self.sphereActor.SetMapper(sphereMapper)
        self.sphereActor.GetProperty().SetColor(1, 0, 0)
        self.ren.AddActor(self.sphereActor)
 
        # Read in STL
        reader = vtk.vtkSTLReader()
        reader.SetFileName('/home/biomed/october_15_ws/src/dvrk_vision/defaults/femur.stl')
        scaler = vtk.vtkTransformFilter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            scaler.SetInputConnection(reader.GetOutput())
        else:
            scaler.SetInputConnection(reader.GetOutputPort())
        scaleTransform = vtk.vtkTransform()
        scaleTransform.Identity()
        scaleTransform.Scale(stlScale, stlScale, stlScale)
        scaler.SetTransform(scaleTransform)
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(scaler.GetOutput())
        else:
            mapper.SetInputConnection(scaler.GetOutputPort())
        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)

        # Read in camera registration
        scriptDirectory = os.path.dirname(os.path.abspath(__file__))
        filePath = os.path.join(scriptDirectory, '..', '..', 'defaults', 
                                'registration_params.yaml')
        with open(filePath, 'r') as f:
            data = yaml.load(f)

        self.camTransform = np.array(data['transform'])
 
        # Add point cloud
        self.pointCloud = VtkPointCloud()
        tf = np.linalg.inv(self.camTransform)
        transform = vtk.vtkTransform()
        transform.SetMatrix(tf.ravel())
        self.pointCloud.vtkActor.SetPosition(transform.GetPosition())
        self.pointCloud.vtkActor.SetOrientation(transform.GetOrientation())
        self.ren.AddActor(self.pointCloud.vtkActor)
 
        self.ren.AddActor(self.actor)

        poseSub = rospy.Subscriber("/stereo/registration_pose", PoseStamped, self.poseCallback)
 
        self.ren.ResetCamera()
 
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.vtkThread = QThread()
        
        self.actorMat = np.eye(4)

        self.thread = QThread()
        robot = psm('PSM2')
        self.worker = Worker(robot)
        self.worker.intReady.connect(self.onIntReady)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.worker.procCounter)
        self.thread.start()

        self.started = False
 
        self.show()
        self.iren.Initialize()

    def saveData(self):
        print("Saving data in")
        np.savetxt("probed_points.txt", self.worker.probedPoints)
        np.savetxt("bone_transform.txt", self.actorMat)
        np.savetxt("camera_transform.txt", np.linalg.inv(self.camTransform))


    def poseCallback(self, data):
        pos = data.pose.position
        rot = data.pose.orientation
        self.actorMat = transformations.quaternion_matrix([rot.x,rot.y,rot.z,rot.w])
        self.actorMat[0:3,3] = [pos.x,pos.y,pos.z]
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.SetMatrix(self.actorMat.ravel())
        self.actor.SetPosition(transform.GetPosition())
        self.actor.SetOrientation(transform.GetOrientation())
        self.actor.VisibilityOn()

        if not self.started:
            self.ren.ResetCamera()
            self.started = True

    def onIntReady(self):
        mat = np.eye(4)
        mat[0:3,3] = self.worker.pos
        tf = np.linalg.inv(self.camTransform)
        mat = np.dot(tf, mat)
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.SetMatrix(mat.ravel())
        self.sphereActor.SetPosition(transform.GetPosition())
        self.sphereActor.SetOrientation(transform.GetOrientation())
        # Modify probedPoints
        self.pointCloud.clearPoints()
        for point in self.worker.probedPoints:
            self.pointCloud.addPoint(point)
        self.vtkWidget.GetRenderWindow().Render()



    
if __name__ == '__main__':
 
    app = QtWidgets.QApplication(sys.argv)
 
    window = MainWindow()
    app.aboutToQuit.connect(window.saveData)
    sys.exit(app.exec_())
