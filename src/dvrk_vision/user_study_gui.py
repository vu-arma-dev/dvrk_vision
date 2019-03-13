#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import rospkg
from PyQt5 import QtWidgets, QtGui, QtCore
from dvrk_vision.registration_gui import RegistrationWidget
import dvrk_vision.vtktools as vtktools
from dvrk_vision.user_widget import UserWidget

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, camera, cameraSync, camTransform, psmName, masterWidget = None):

        super(MainWindow, self).__init__()

        # Set up parent widget
        widgetParent = None if masterWidget is None else masterWidget.userWidget
        
        markerTopic = rospy.get_param('~marker_topic')
        robotFrame = rospy.get_param('~robot_frame')
        tipFrame = rospy.get_param('~end_effector_frame')
        cameraFrame = rospy.get_param('~camera_frame')

        self.userWidget = UserWidget(camera,
                                        camSync,
                                        markerTopic,
                                        robotFrame,
                                        tipFrame,
                                        cameraFrame,
                                        masterWidget = widgetParent,
                                        parent = self)

        self.setCentralWidget(self.userWidget)

        self.otherWindows = []
        if masterWidget != None:
            masterWidget.otherWindows.append(self)
            self.otherWindows.append(masterWidget)

        self.forceSub = rospy.Subscriber(name='/control/forceDisplay', 
                                        data_class=Bool,
                                        callback=self.forceBarCB,
                                        queue_size=1)

        self.textSub = rospy.Subscriber(name='/control/textDisplay', 
                                        data_class=String,
                                        callback=self.textCB,
                                        queue_size=1)

        self.camSub = rospy.Subscriber(name='/dvrk/footpedals/camera', 
                                         data_class=Joy,
                                         callback=self.camCB,
                                         queue_size=1)

        self.camMinusSub = rospy.Subscriber(name='/dvrk/footpedals/cam_plus', 
                                         data_class=Joy,
                                         callback=self.camPlusCB,
                                         queue_size=1)

        self.camMinusSub = rospy.Subscriber(name='/dvrk/footpedals/cam_minus', 
                                         data_class=Joy,
                                         callback=self.camMinusCB,
                                         queue_size=1)

        self.opacitySub = rospy.Subscriber(name='/control/meshOpacity', 
                                        data_class=Int32,
                                        callback=self.opacityCB,
                                        queue_size=1)
        # self.hideButtons()

    def closeEvent(self, qCloseEvent):
        for window in self.otherWindows:
            window.close()
        self.close()

    def hideButtons(self):
        self.userWidget.opacitySlider.hide()
        self.userWidget.textureCheckBox.hide()
        self.userWidget.clearButton.hide()
        self.userWidget.POICheckBox.hide()

    def showButtons(self):
        self.userWidget.opacitySlider.show()
        self.userWidget.textureCheckBox.show()

    def forceBarCB(self,b_input):
        self.userWidget.setBarVisibility(b_input=b_input.data)
        print "Force Vision"
        print b_input

    def textCB(self,textInput):
        self.userWidget.setText(textInput.data)

    def opacityCB(self,newValue):
        self.userWidget.opacitySlider.setValue(newValue.data)
        self.userWidget.textureCheckBox.setChecked(False)

    def camCB(self,dataInput):
        if dataInput.buttons[0]:
            self.userWidget.clearPOI()

    def camPlusCB(self,dataInput):
        if dataInput.buttons[0]:
            self.userWidget.addPOI()

    def camMinusCB(self,dataInput):
        if dataInput.buttons[0]:
            self.userWidget.removePOI()
            

if __name__ == "__main__":
    from tf import transformations
    import yaml
    from dvrk_vision.vtk_stereo_viewer import StereoCameras
    rospy.init_node('user_study_gui')
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
    mainWin.move(QtWidgets.QApplication.desktop().screenGeometry(1).bottomLeft())
    
    # TODO uncomment for proper use, add ros interfaces for default screen choice
    mainWin.showMaximized()
    secondWin.show()
    secondWin.move(QtWidgets.QApplication.desktop().screenGeometry(2).bottomLeft())
    secondWin.showMaximized()

    sys.exit(app.exec_())
