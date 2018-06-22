import vtk
# Which PyQt we use depends on our vtk version. QT4 causes segfaults with vtk > 6
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    from PyQt5.QtWidgets import QApplication
    _QT_VERSION = 5
else:
    from PyQt4.QtGui import QApplication
    _QT_VERSION = 4
from dvrk_vision.overlay_gui import OverlayWidget
import dvrk_vision.vtktools as vtktools
from visualization_msgs.msg import Marker
import numpy as np
import rospy
import os

def makeMarker(path, scale):
    markerMsg = Marker()
    # HACK BECAUSE RVIZ DOESN'T USE OBJ
    filename, extension = os.path.splitext(path)
    markerMsg.mesh_resource = filename + '.stl'
    markerMsg.header.frame_id = "/stereo_camera_frame"
    markerMsg.header.stamp    = rospy.Time.now()
    markerMsg.id = 0
    markerMsg.type = 10 # mesh resource
    markerMsg.action = 0
    markerMsg.color.r = 0
    markerMsg.color.g = 1.0
    markerMsg.color.b = 0
    markerMsg.color.a = 1.0
    markerMsg.scale.x = scale
    markerMsg.scale.y = scale
    markerMsg.scale.z = scale
    return markerMsg

class ManualRegistrationWidget(OverlayWidget):
    def __init__(self, camera, texturePath, meshPath, scale=1, masterWidget=None, parent=None):
        super(ManualRegistrationWidget, self).__init__(camera, texturePath, meshPath, scale, masterWidget, parent)
        self.interacting = False
        self.dolly = False
        # Set up subscriber for registered organ position
        poseSubTopic = "/stereo/registration_marker"
        self.poseSub = rospy.Subscriber(poseSubTopic, Marker, self.poseCallback)

        self.posePub = rospy.Publisher("/stereo/registration_marker", Marker, queue_size=10)
        self.marker = makeMarker(meshPath, scale)


    def renderSetup(self):
        super(ManualRegistrationWidget, self).renderSetup()
        self.actor_moving.VisibilityOn()
        self.actor_moving.SetPosition(0,0,.1)
        self.vtkWidget.ren.ResetCameraClippingRange()
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
        self.iren.AddObserver("StartInteractionEvent", self.interactionChange)
        self.iren.AddObserver("EndInteractionEvent", self.interactionChange)
        self.iren.RemoveObservers("RightButtonPressEvent")#, self.buttonEvent)
        self.iren.RemoveObservers("RightButtonReleaseEvent")#, self.buttonEvent)
        self.iren.AddObserver("RightButtonPressEvent", self.interactionChange)
        self.iren.AddObserver("RightButtonReleaseEvent", self.interactionChange)
        self.iren.AddObserver("InteractionEvent", self.interactionEvent)
        self.iren.AddObserver("MouseMoveEvent", self.interactionEvent)
        # self.iren.Disable()

    def publishPose(self):
        pos = self.actor_moving.GetPosition()
        rot = self.actor_moving.GetOrientationWXYZ()
        w = (rot[0] / 180) * np.pi
        sinW = np.sin(w / 2)
        cosW = np.cos(w / 2)
        self.marker.pose.position.x = pos[0]
        self.marker.pose.position.y = pos[1]
        self.marker.pose.position.z = pos[2]
        self.marker.pose.orientation.x = rot[1] * sinW
        self.marker.pose.orientation.y = rot[2] * sinW
        self.marker.pose.orientation.z = rot[3] * sinW
        self.marker.pose.orientation.w = cosW
        print("Publishing: ", self)
        self.posePub.publish(self.marker)

    # Handle the mouse button events.
    def interactionChange(self, obj, event):
        print(event)
        if event == "RightButtonPressEvent":
            self.dolly = True
        elif event == "RightButtonReleaseEvent":
            self.dolly = False
        if event == "StartInteractionEvent":
            self.interacting = True
        elif event == "EndInteractionEvent":
            self.interacting = False
            self.dolly = False
        self.publishPose()

    def interactionEvent(self, obj, event):
        if self.dolly:
            lastX, lastY = self.iren.GetLastEventPosition()
            x, y = self.iren.GetEventPosition()
            h, w = self.iren.GetSize()
            dollyFactor = 0.5 * (y-lastY) / h
            actorPos = self.actor_moving.GetPosition()
            camPos = self.vtkWidget.ren.GetActiveCamera().GetPosition()
            direction = np.subtract(actorPos, camPos)
            direction = direction / np.linalg.norm(direction) * dollyFactor
            self.actor_moving.AddPosition(direction)
            if self.isVisible():
                self.vtkWidget.ren.ResetCameraClippingRange()
                
    def poseCallback(self, data):
        super(ManualRegistrationWidget, self).poseCallback(data)
        if self.isVisible():
            self.vtkWidget.ren.ResetCameraClippingRange()

if __name__ == "__main__":
    # App specific imports
    import sys
    from dvrk_vision.vtk_stereo_viewer import StereoCameras

    app = QApplication(sys.argv)
    rosThread = vtktools.QRosThread()
    rosThread.start()
    meshPath = "package://oct_15_demo/resources/largeProstate.obj"
    texturePath = "package://oct_15_demo/resources/largeProstate.png"
    stlScale = 1.06
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("stereo/left/image_rect",
                         "stereo/right/image_rect",
                         "stereo/left/camera_info",
                         "stereo/right/camera_info",
                         slop = slop)
    windowL = ManualRegistrationWidget(cams.camL, texturePath, meshPath, scale=stlScale)
    windowL.show()
    windowR = ManualRegistrationWidget(cams.camR, texturePath, meshPath, masterWidget=windowL)
    windowR.show()
    sys.exit(app.exec_())