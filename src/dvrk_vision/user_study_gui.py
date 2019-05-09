#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import Header
from std_msgs.msg import Empty
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
import rospkg
from PyQt5 import QtWidgets, QtGui, QtCore
from dvrk_vision.registration_gui import RegistrationWidget
import dvrk_vision.vtktools as vtktools
from dvrk_vision.user_widget import UserWidget
import Queue

def get_PSM_Position():
    # Often can get an old message, reading twice will clear that out
    slave_name = rospy.get_param('/slave')
    rospy.wait_for_message('/dvrk/'+slave_name+'/position_cartesian_current',PoseStamped,1.0)
    return rospy.wait_for_message('/dvrk/'+slave_name+'/position_cartesian_current',PoseStamped,1.0)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, camera, cameraSync, camTransform, psmName, masterWidget = None, idNum=0):
        self.idNum=idNum
        super(MainWindow, self).__init__()

        # Set up parent widget
        widgetParent = None if masterWidget is None else masterWidget.userWidget
        
        markerTopic = rospy.get_param('~marker_topic')
        robotFrame = rospy.get_param('~robot_frame')
        tipFrame = rospy.get_param('~end_effector_frame')
        cameraFrame = rospy.get_param('~camera_frame')

        self.allowPoints=False
        self.userWidget = UserWidget(camera,
                                        cameraSync,
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

        self.allowPointsSub = rospy.Subscriber(name='/control/allowPoints', 
                                        data_class=Bool,
                                        callback=self.allowPointsCB,
                                        queue_size=1)

        self.forceSub = rospy.Subscriber(name='/control/forceDisplay', 
                                        data_class=Bool,
                                        callback=self.forceBarCB,
                                        queue_size=1)

        self.textSub = rospy.Subscriber(name='/control/textDisplay', 
                                        data_class=String,
                                        callback=self.textCB,
                                        queue_size=1)

        self.clearSub = rospy.Subscriber(name='/dvrk_vision/clear_POI',
                                        data_class=Empty,
                                        callback=self.clearPOICB,
                                        queue_size=1)

        self.camPlusSub = rospy.Subscriber(name='/dvrk/footpedals/cam_plus', 
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

        self.maximiseSub = rospy.Subscriber(name='/control/moveFront', 
                                        data_class=Empty,
                                        callback=self.moveCB,
                                        queue_size=1)

        self.displayPub = rospy.Publisher('/control/Vision_Point_List',PoseArray,latch=False,queue_size=1)

        self.displayList=[]

        self.hideButtons()

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

    def textCB(self,textInput):
        self.userWidget.setText(textInput.data)

    def opacityCB(self,newValue):
        self.userWidget.opacitySlider.setValue(newValue.data)
        self.userWidget.textureCheckBox.setChecked(False)

    def allowPointsCB(self,b_input):
        self.allowPoints=b_input.data

    def camPlusCB(self,dataInput):
        if dataInput.buttons[0] and self.allowPoints:
            self.userWidget.addPOI()

            if self.idNum==1:
                curPoseMsg = get_PSM_Position()
                self.displayList.append(curPoseMsg.pose)
                self.publishDisplayList()
                
    def camMinusCB(self,dataInput):
        if dataInput.buttons[0] and self.allowPoints:
            self.userWidget.removePOI()
            if len(self.displayList)>0:
                self.displayList.pop()
                self.publishDisplayList()

    def clearPOICB(self,emptyInput):
        self.displayList=[]

    def moveCB(self,emptyInput):
        self.raise_()

    def publishDisplayList(self):
        header=Header()
        header.stamp=rospy.Time.now()
        self.displayPub.publish(PoseArray(header,self.displayList))
        rospy.sleep(0.2)

if __name__ == "__main__":
    from tf import transformations
    import yaml
    from dvrk_vision.vtk_stereo_viewer import StereoCameras
    rospy.init_node('user_study_gui')
    from dvrk_vision.tf_sync import CameraSync

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

    camSync = CameraSync("left/camera_info")

    psmName = rospy.get_param('~psm_name')
    filePath = rospy.get_param('~camera_registration')

    screenOne = rospy.get_param('~screen_one')
    screenTwo = rospy.get_param('~screen_two')
    
    print(filePath)
    with open(filePath, 'r') as f:
        data = yaml.load(f)
    camTransform = data['transform']

    mainWin = MainWindow(cams.camL, camSync, camTransform, psmName)
    secondWin = MainWindow(cams.camR, camSync, camTransform, psmName, masterWidget = mainWin,idNum=1)
    mainWin.show()
    mainWin.move(QtWidgets.QApplication.desktop().screenGeometry(screenOne).bottomLeft())
    
    # TODO uncomment for proper use, add ros interfaces for default screen choice
    mainWin.showMaximized()
    secondWin.show()
    secondWin.move(QtWidgets.QApplication.desktop().screenGeometry(screenTwo).bottomLeft())
    secondWin.showMaximized()

    sys.exit(app.exec_())
