#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import Empty
import rospkg
from PyQt5 import QtWidgets, QtGui, QtCore
from dvrk_vision.registration_gui import RegistrationWidget
import dvrk_vision.vtktools as vtktools
from dvrk_vision.tf_sync import CameraSync
from dvrk_vision.force_overlay import ForceOverlayWidget
from dvrk_vision.gp_overlay_gui import GpOverlayWidget
from dvrk_vision.vtk_stereo_viewer import StereoCameras

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
        self.selectSub = rospy.Subscriber(name='/control/windowSelect', 
                                        data_class=Int32,
                                        callback=self.windowSelectCB,
                                        queue_size=1)

        self.forceSub = rospy.Subscriber(name='/control/forceDisplay', 
                                        data_class=Bool,
                                        callback=self.forceBarCB,
                                        queue_size=1)

        self.textSub = rospy.Subscriber(name='/control/textDisplay', 
                                        data_class=String,
                                        callback=self.textCB,
                                        queue_size=1)
        # self.pinchSub = rospy.Subscriber(name='/dvrk/MTMR/gripper_pinch_event', 
        #                                 data_class=Empty,
        #                                 callback=self.pinchCB,
        #                                 queue_size=1)

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

    def changeTab(self, idxmsg):
        self.tabWidget.setCurrentIndex(idxmsg.data)
        for window in self.otherWindows:
            window.tabWidget.setCurrentIndex(idxmsg.data)

    def windowSelectCB(self,windowNum):
        print('SELECTING WINDOW')
        self.changeTab(windowNum)

    def forceBarCB(self,b_input):
        print('FORCE VISIBILITY')
        self.forceOverlay.setBarVisibility(b_input=b_input.data)

    def textCB(self,textInput):
        self.forceOverlay.setText(textInput.data)
        self.gpWidget.setText(textInput.data)

    def pinchCB(self,emptyInput):
        self.gpWidget.addPOI()

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
