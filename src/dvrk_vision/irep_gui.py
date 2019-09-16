#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import Header
from std_msgs.msg import Empty
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Vector3
import rospkg
from PyQt5 import QtWidgets, QtGui, QtCore
from dvrk_vision.registration_gui import RegistrationWidget
import dvrk_vision.vtktools as vtktools
from dvrk_vision.irep_widget import IrepWidget
import Queue

def get_PSM_Position():
    # Often can get an old message, reading twice will clear that out
    slave_name = rospy.get_param('/slave')
    rospy.wait_for_message('/dvrk/'+slave_name+'/position_cartesian_current',PoseStamped,1.0)
    return rospy.wait_for_message('/dvrk/'+slave_name+'/position_cartesian_current',PoseStamped,1.0)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, camera, camTransform, psmName, masterWidget = None, idNum=0):
        self.idNum=idNum
        super(MainWindow, self).__init__()

        # Set up parent widget
        widgetParent = None if masterWidget is None else masterWidget.userWidget
        
        markerTopic = rospy.get_param('~marker_topic')
        robotFrame = rospy.get_param('~robot_frame')
        tipFrame = rospy.get_param('~end_effector_frame')
        cameraFrame = rospy.get_param('~camera_frame')

        self.userWidget = IrepWidget(camera,
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

        self.forceYSub = rospy.Subscriber(name='/control/forceUseY', 
                                        data_class=Bool,
                                        callback=self.forceYCB,
                                        queue_size=1)

        self.forceSrcSub = rospy.Subscriber(name='/control/forceDisplaySrc', 
                                        data_class=Int32,
                                        callback=self.forceBarSrcCB,
                                        queue_size=1)

        self.forceMoveSub = rospy.Subscriber(name='/control/forceMove', 
                                        data_class=Vector3,
                                        callback=self.moveBarCB,
                                        queue_size=1)

        self.textSub = rospy.Subscriber(name='/control/textDisplay', 
                                        data_class=String,
                                        callback=self.textCB,
                                        queue_size=1)

        self.textMoveSub = rospy.Subscriber(name='/control/textMove', 
                                        data_class=Vector3,
                                        callback=self.moveTextCB,
                                        queue_size=1)

        self.textScaleSub = rospy.Subscriber(name='/control/textScale', 
                                        data_class=Float32,
                                        callback=self.scaleTextCB,
                                        queue_size=1)

        self.maximiseSub = rospy.Subscriber(name='/control/moveFront', 
                                        data_class=Empty,
                                        callback=self.moveCB,
                                        queue_size=1)

        self.displayList=[]

    def closeEvent(self, qCloseEvent):
        for window in self.otherWindows:
            window.close()
        self.close()

    def forceYCB(self,b_input):
        self.userWidget.setUseForceY(b_input=b_input.data)

    def forceBarCB(self,b_input):
        self.userWidget.setBarVisibility(b_input=b_input.data)

    def forceBarSrcCB(self,myinput):
        self.userWidget.setBarSrc(myinput.data)

    def textCB(self,textInput):
        self.userWidget.setText(textInput.data)

    def moveBarCB(self,moveInput):
        self.userWidget.setForcePos([moveInput.x,moveInput.y,moveInput.z])

    def moveTextCB(self,moveInput):
        self.userWidget.setTextPos([moveInput.x,moveInput.y,moveInput.z])

    def scaleTextCB(self,scaleInput):
        self.userWidget.setTextScale(scaleInput.data)

    def moveCB(self,emptyInput):
        screenOne = rospy.get_param('~screen_one')
        screenTwo = rospy.get_param('~screen_two')

        for window in self.otherWindows:
            window.move(QtWidgets.QApplication.desktop().screenGeometry(screenOne).bottomLeft())
            window.showNormal()
            # window.showFullScreen()
            window.showMaximized()
            window.raise_()
        self.move(QtWidgets.QApplication.desktop().screenGeometry(screenTwo).bottomLeft())
        self.showNormal()
        # self.showFullScreen()
        self.showMaximized()
        self.raise_()

if __name__ == "__main__":
    from tf import transformations
    import yaml
    from dvrk_vision.vtk_stereo_viewer import StereoCameras
    rospy.init_node('user_study_gui')

    app = QtWidgets.QApplication(sys.argv)
    rosThread = vtktools.QRosThread()
    rosThread.start()
    rosThread.update = app.processEvents

    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("left/image_rect",
                      "right/image_rect",
                      "left/camera_info",
                      "right/camera_info",
                      slop = slop)
    
    psmName = rospy.get_param('~psm_name')
    filePath = rospy.get_param('~camera_registration')

    screenOne = rospy.get_param('~screen_one')
    screenTwo = rospy.get_param('~screen_two')
    
    print(filePath)
    with open(filePath, 'r') as f:
        data = yaml.load(f)
    camTransform = data['transform']

    mainWin = MainWindow(cams.camL, camTransform, psmName)
    secondWin = MainWindow(cams.camR, camTransform, psmName, masterWidget = mainWin,idNum=1)
    mainWin.show()
    mainWin.move(QtWidgets.QApplication.desktop().screenGeometry(screenOne).bottomLeft())
    # mainWin.showFullScreen()
    mainWin.showMaximized()

    secondWin.show()
    secondWin.move(QtWidgets.QApplication.desktop().screenGeometry(screenTwo).bottomLeft())
    # secondWin.showFullScreen()
    secondWin.showMaximized()

    sys.exit(app.exec_())
