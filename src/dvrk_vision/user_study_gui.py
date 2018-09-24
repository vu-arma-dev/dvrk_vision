#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
import rospkg
from PyQt5 import QtWidgets, QtGui, QtCore
from dvrk_vision.registration_gui import RegistrationWidget
import dvrk_vision.vtktools as vtktools
from dvrk_vision.tf_sync import CameraSync
from dvrk_vision.force_overlay import ForceOverlayWidget
from dvrk_vision.gp_overlay_gui import GpOverlayWidget
from dvrk_vision.vtk_stereo_viewer import StereoCameras
from dvrk_vision.mark_roi_gui import MarkRoiWidget

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, camera, cameraSync, camTransform, psmName, masterWidget = None):

        super(MainWindow, self).__init__()
        self.tabWidget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabWidget)

        # Set up parents
        # regParent = None if masterWidget == None else masterWidget.reg
        forceParent = None if masterWidget is None else masterWidget.forceOverlay
        gpParent = None if masterWidget is None else masterWidget.gpWidget
        # markerParent = None if masterWidget is None else masterWidget.markerWidget

        # self.reg = RegistrationWidget(camera,
        #                               meshPath,
        #                               scale=stlScale,
        #                               masterWidget = regParent,
        #                               parent = self)
        # self.tabWidget.addTab(self.reg, "Organ Registration")
        
       
        self.forceOverlay = ForceOverlayWidget(camera,
                                               cameraSync,
                                               camTransform,
                                               '/dvrk/' + psmName + "/position_cartesian_current",
                                               '/dvrk/' + psmName + '_FT/raw_wrench',
                                               masterWidget = forceParent,
                                               parent = self)

        self.tabWidget.addTab(self.forceOverlay, "Force Bar Overlay")

        markerTopic = rospy.get_param('~marker_topic')
        robotFrame = rospy.get_param('~robot_frame')
        tipFrame = rospy.get_param('~end_effector_frame')
        cameraFrame = rospy.get_param('~camera_frame')

        # self.gpWidget = GpOverlayWidget(camera,
        #                                 robotFrame,
        #                                 cameraFrame,
        #                                 markerTopic,
        #                                 masterWidget = gpParent,
        #                                 parent = self)
    
        # self.tabWidget.addTab(self.gpWidget, "Stiffness Overlay")

        # self.markerWidget = MarkRoiWidget(camera,
        #                                   cameraSync._tfBuffer,
        #                                   markerTopic,
        #                                   robotFrame,
        #                                   cameraFrame,
        #                                   masterWidget = markerParent,
        #                                   parent = self)

        self.gpWidget = GpOverlayWidget(camera,
                                        cameraSync._tfBuffer,
                                        markerTopic,
                                        robotFrame,
                                        tipFrame,
                                        cameraFrame,
                                        masterWidget = gpParent,
                                        parent = self)

        self.tabWidget.addTab(self.gpWidget, "Stiffness Overlay")

        # self.tabWidget.addTab(self.markerWidget, "Stiffness Overlay")

        self.forceOverlay.Initialize()
        self.forceOverlay.start()

        self.otherWindows = []
        if masterWidget != None:
            masterWidget.otherWindows.append(self)
            self.otherWindows.append(masterWidget)

        self.tabWidget.currentChanged.connect(self.tabChanged)

        self.widgets = {"Force Bar Overlay": self.forceOverlay,
                        "Stiffness Overlay": self.gpWidget}

        # self.widgets = {"Force Bar Overlay": self.forceOverlay}


    def tabChanged(self):
        idx = self.tabWidget.currentIndex()
        for window in self.otherWindows:
            window.tabWidget.setCurrentIndex(idx)

    def closeEvent(self, qCloseEvent):
        for window in self.otherWindows:
            window.close()
        self.close()

    def hideButtons(self):
        self.tabWidget.tabBar().hide()
        self.gpWidget.opacitySlider.hide()
        self.gpWidget.textureCheckBox.hide()

    def showButtons(self):
        self.tabWidget.tabBar().show()
        self.gpWidget.opacitySlider.show()
        self.gpWidget.textureCheckBox.show()

    def changeTab(self, idx):
        self.tabWidget.setCurrentIndex(idx)
        for window in self.otherWindows:
            window.tabWidget.setCurrentIndex(idx)

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

    camSync = CameraSync("left/camera_info")

    psmName = rospy.get_param('~psm_name')
    filePath = rospy.get_param('~camera_registration')
    
    print(filePath)
    with open(filePath, 'r') as f:
        data = yaml.load(f)
    camTransform = data['transform']

    mainWin = MainWindow(cams.camL, camSync, camTransform, psmName)
    secondWin = MainWindow(cams.camR, camSync, camTransform, psmName, masterWidget = mainWin)
    mainWin.show()
    secondWin.show()
    sys.exit(app.exec_())
