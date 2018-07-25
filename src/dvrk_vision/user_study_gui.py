#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
import rospkg
from PyQt5 import QtWidgets, QtGui, QtCore
from dvrk_vision.registration_gui import RegistrationWidget
from dvrk_vision.manual_registration_gui import ManualRegistrationWidget
from dvrk_vision.overlay_gui import OverlayWidget
import dvrk_vision.vtktools as vtktools
from dvrk_vision.force_overlay import ForceOverlayWidget
from dvrk_vision.vtk_stereo_viewer import StereoCameras

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, camera, camTransform, masterWidget = None):

        super(MainWindow, self).__init__()
        self.tabWidget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabWidget)
        
        meshPath = rospy.get_param("~mesh_path")
        try:
            secondaryMeshPath = rospy.get_param("~secondary_mesh_path")
        except:
            rospy.logwarn("No secondary_mesh_path found. Using mesh_path for overlay")
            secondaryMeshPath = meshPath

        stlScale = rospy.get_param("~mesh_scale")
        texturePath = rospy.get_param("~texture_path")

        # Set up parents
        # regParent = None if masterWidget == None else masterWidget.reg
        overlayParent = None if masterWidget == None else masterWidget.overlay
        manualParent = None if masterWidget == None else masterWidget.manual

        # self.reg = RegistrationWidget(camera,
        #                               meshPath,
        #                               scale=stlScale,
        #                               masterWidget = regParent,
        #                               parent = self)
        # self.tabWidget.addTab(self.reg, "Organ Registration")
        
        self.manual = ManualRegistrationWidget(camera,
                                               texturePath,
                                               secondaryMeshPath,
                                               scale=stlScale,
                                               masterWidget = manualParent,
                                               parent = self)
        self.tabWidget.addTab(self.manual, "Manual Registration")
        
        self.overlay = OverlayWidget(camera,
                                     texturePath,
                                     secondaryMeshPath,
                                     scale=stlScale,
                                     masterWidget = overlayParent,
                                     parent = self)
        self.tabWidget.addTab(self.overlay, "Stiffness Overlay")

        self.forceOverlay = ForceOverlayWidget(camera,
                                               camTransform,
                                               'PSM2',
                                               '/dvrk/PSM2_FT/raw_wrench')

        self.tabWidget.addTab(self.forceOverlay, "Force Bar Overlay")

        self.otherWindows = []
        if masterWidget != None:
            masterWidget.otherWindows.append(self)
            self.otherWindows.append(masterWidget)

        self.tabWidget.currentChanged.connect(self.tabChanged)

    def tabChanged(self):
        idx = self.tabWidget.currentIndex()
        for window in self.otherWindows:
            window.tabWidget.setCurrentIndex(idx)

    def closeEvent(self, qCloseEvent):
        for window in self.otherWindows:
            window.close()
        self.close()

if __name__ == "__main__":
    from tf import transformations
    import yaml
    app = QtWidgets.QApplication(sys.argv)
    rosThread = vtktools.QRosThread()
    rosThread.start()
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("left/image_rect",
                      "right/image_rect",
                      "left/camera_info",
                      "right/camera_info",
                      slop = slop)


    filePath = rospy.get_param('~camera_registration')
    print(filePath)
    with open(filePath, 'r') as f:
        data = yaml.load(f)
    camTransform = data['transform']

    mainWin = MainWindow(cams.camL, camTransform)
    secondWin = MainWindow(cams.camR, camTransform, masterWidget = mainWin)
    mainWin.show()
    secondWin.show()
    sys.exit(app.exec_())
