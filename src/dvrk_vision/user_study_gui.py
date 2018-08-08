#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
import rospkg
from PyQt5 import QtWidgets, QtGui, QtCore
from dvrk_vision.registration_gui import RegistrationWidget
import dvrk_vision.vtktools as vtktools
from dvrk_vision.force_overlay import ForceOverlayWidget
from dvrk_vision.stiffnesses_to_image_gui import GpOverlayWidget
from dvrk_vision.vtk_stereo_viewer import StereoCameras

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, camera, camTransform, masterWidget = None):

        super(MainWindow, self).__init__()
        self.tabWidget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabWidget)

        # Set up parents
        # regParent = None if masterWidget == None else masterWidget.reg
        forceParent = None if masterWidget is None else masterWidget.forceOverlay
        gpParent = None if masterWidget is None else masterWidget.gpWidget

        # self.reg = RegistrationWidget(camera,
        #                               meshPath,
        #                               scale=stlScale,
        #                               masterWidget = regParent,
        #                               parent = self)
        # self.tabWidget.addTab(self.reg, "Organ Registration")
        
       
        self.forceOverlay = ForceOverlayWidget(camera,
                                               camTransform,
                                               'PSM2',
                                               '/dvrk/PSM2_FT/raw_wrench',
                                               masterWidget = forceParent,
                                               parent = self)

        self.tabWidget.addTab(self.forceOverlay, "Force Bar Overlay")

        markerTopic = rospy.get_param('~marker_topic')
        robotFrame = rospy.get_param('~robot_frame')
        cameraFrame = rospy.get_param('~camera_frame')

        self.gpWidget = GpOverlayWidget(camera,
                                        markerTopic,
                                        robotFrame,
                                        cameraFrame,
                                        masterWidget = gpParent,
                                        parent = self)
    
        self.tabWidget.addTab(self.gpWidget, "Force Bar Overlay")

        self.forceOverlay.Initialize()
        self.forceOverlay.start()

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
