#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from std_msgs.msg import Empty
import rospkg
from PyQt5 import QtWidgets, QtGui, QtCore
from dvrk_vision.registration_gui import RegistrationWidget
import dvrk_vision.vtktools as vtktools

from dvrk_vision.force_overlay import ForceOverlayWidget
from dvrk_vision.gp_overlay_gui import GpOverlayWidget
from dvrk_vision.vtk_stereo_viewer import StereoCameras

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, camera, cameraSync, camTransform, psmName, masterWidget = None):

        super(MainWindow, self).__init__()

        # Set up parents
        # regParent = None if masterWidget == None else masterWidget.reg
        # forceParent = None if masterWidget is None else masterWidget.forceOverlay
        gpParent = None if masterWidget is None else masterWidget.gpWidget
        # markerParent = None if masterWidget is None else masterWidget.markerWidget

        # self.forceOverlay = ForceOverlayWidget(camera,
        #                                        cameraSync,
        #                                        camTransform,
        #                                        '/dvrk/' + psmName + "/position_cartesian_current",
        #                                        '/dvrk/' + psmName + '_FT/raw_wrench',
        #                                        masterWidget = forceParent,
        #                                        parent = self)

        markerTopic = rospy.get_param('~marker_topic')
        robotFrame = rospy.get_param('~robot_frame')
        tipFrame = rospy.get_param('~end_effector_frame')
        cameraFrame = rospy.get_param('~camera_frame')

        self.gpWidget = GpOverlayWidget(camera,
                                        cameraSync._tfBuffer,
                                        markerTopic,
                                        robotFrame,
                                        tipFrame,
                                        cameraFrame,
                                        masterWidget = gpParent,
                                        parent = self)

        self.setCentralWidget(self.gpWidget)

        # self.forceOverlay.Initialize()
        # self.forceOverlay.start()

        self.otherWindows = []
        if masterWidget != None:
            masterWidget.otherWindows.append(self)
            self.otherWindows.append(masterWidget)

        self.forceSub = rospy.Subscriber(name='/control/forceDisplay', 
                                        data_class=Bool,
                                        callback=self.forceBarCB,
                                        queue_size=1)
        self.pinchSub = rospy.Subscriber(name='/dvrk/MTMR/gripper_pinch_event', 
                                        data_class=Empty,
                                        callback=self.pinchCB,
                                        queue_size=1)

    def closeEvent(self, qCloseEvent):
        for window in self.otherWindows:
            window.close()
        self.close()

    def hideButtons(self):
        self.gpWidget.opacitySlider.hide()
        self.gpWidget.textureCheckBox.hide()

    def showButtons(self):
        self.gpWidget.opacitySlider.show()
        self.gpWidget.textureCheckBox.show()

    def forceBarCB(self,b_input):
        print('FORCE VISIBILITY')
        # self.forceOverlay.setBarVisibility(b_input=b_input.data)

    def pinchCB(self,emptyInput):
        self.gpWidget.addPOI()

if __name__ == "__main__":
    rospy.init_node('test')
    from tf import transformations
    import yaml
    from dvrk_vision.tf_sync import CameraSync

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
