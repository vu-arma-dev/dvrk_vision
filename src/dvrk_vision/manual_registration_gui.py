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

class ManualRegistrationWidget(OverlayWidget):
    def __init__(self, camera, texturePath, meshPath, scale=1, masterWidget=None, parent=None):
        super(ManualRegistrationWidget, self).__init__(camera, texturePath, meshPath, scale, masterWidget, parent)
        self.interacting = False
        self.dolly = False
        # Set up subscriber for registered organ position
        poseSubTopic = "/stereo/registration_marker"
        self.poseSub = rospy.Subscriber(poseSubTopic, Marker, self.poseCallback)

        self.posePub = rospy.Publisher("/stereo/registration_marker", Marker, queue_size=10)


    def renderSetup(self):
        super(ManualRegistrationWidget, self).renderSetup()
        self.actor_moving.VisibilityOn()
        self.actor_moving.SetPosition(0,0,.1)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
        self.iren.AddObserver("StartInteractionEvent", self.interactionChange)
        self.iren.AddObserver("EndInteractionEvent", self.interactionChange)
        # self.iren.AddObserver("LeftButtonPressEvent", self.buttonEvent)
        # self.iren.AddObserver("LeftButtonReleaseEvent", self.buttonEvent)
        # self.iren.AddObserver("MiddleButtonPressEvent", self.buttonEvent)
        # self.iren.AddObserver("MiddleButtonReleaseEvent", self.buttonEvent)
        self.iren.RemoveObservers("RightButtonPressEvent")#, self.buttonEvent)
        self.iren.RemoveObservers("RightButtonReleaseEvent")#, self.buttonEvent)
        self.iren.AddObserver("RightButtonPressEvent", self.interactionChange)
        self.iren.AddObserver("RightButtonReleaseEvent", self.interactionChange)
        self.iren.AddObserver("InteractionEvent", self.interactionEvent)
        self.iren.AddObserver("MouseMoveEvent", self.interactionEvent)
    #     self.iren.AddObserver("KeyPressEvent", self.keypress)


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

    # General high-level logic
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
            self.vtkWidget.ren.ResetCameraClippingRange()
            self.vtkWidget.GetRenderWindow().Render()

    def poseCallback(self, data):
        super(ManualRegistrationWidget, self).poseCallback(data)
        self.vtkWidget.ren.ResetCameraClippingRange()

            # mat = vtktools.vtkMatrixtoNpMatrix(self.actor_moving.GetMatrix())
            # print mat
    #     ren = self.vtkWidget.ren
    #     renWin = self.vtkWidget.GetRenderWindow()
    #     iren = self.iren

    #     lastXYpos = iren.GetLastEventPosition()
    #     lastX = lastXYpos[0]
    #     lastY = lastXYpos[1]

    #     xypos = iren.GetEventPosition()
    #     x = xypos[0]
    #     y = xypos[1]

    #     center = renWin.GetSize()
    #     centerX = center[0]/2.0
    #     centerY = center[1]/2.0

    #     if self.rotating:
    #         self.rotate(ren, ren.GetActiveCamera(), x, y, lastX, lastY,
    #                centerX, centerY)
    #     elif self.panning:
    #         self.pan(ren, ren.GetActiveCamera(), x, y, lastX, lastY, centerX,
    #             centerY)
    #     elif self.zooming:
    #         self.dolly(ren, ren.GetActiveCamera(), x, y, lastX, lastY,
    #               centerX, centerY)

    # def keypress(self, obj, event):
    #     key = obj.GetKeySym()
    #     if key == "e":
    #         obj.InvokeEvent("DeleteAllObjects")
    #         sys.exit()
    #     elif key == "w":
    #         self.wireframe()
    #     elif key =="s":
    #         self.surface()


    # # Routines that translate the events into camera motions.

    # # This one is associated with the left mouse button. It translates x
    # # and y relative motions into camera azimuth and elevation commands.
    # def rotate(self, renderer, camera, x, y, lastX, lastY, centerX, centerY):
    #     camera.Azimuth(lastX-x)
    #     camera.Elevation(lastY-y)
    #     camera.OrthogonalizeViewUp()
    #     renWin = self.vtkWidget.GetRenderWindow()
    #     renWin.Render()


    # # Pan translates x-y motion into translation of the focal point and
    # # position.
    # def pan(self, renderer, camera, x, y, lastX, lastY, centerX, centerY):
    #     FPoint = camera.GetFocalPoint()
    #     FPoint0 = FPoint[0]
    #     FPoint1 = FPoint[1]
    #     FPoint2 = FPoint[2]

    #     PPoint = camera.GetPosition()
    #     PPoint0 = PPoint[0]
    #     PPoint1 = PPoint[1]
    #     PPoint2 = PPoint[2]

    #     renderer.SetWorldPoint(FPoint0, FPoint1, FPoint2, 1.0)
    #     renderer.WorldToDisplay()
    #     DPoint = renderer.GetDisplayPoint()
    #     focalDepth = DPoint[2]

    #     APoint0 = centerX+(x-lastX)
    #     APoint1 = centerY+(y-lastY)

    #     renderer.SetDisplayPoint(APoint0, APoint1, focalDepth)
    #     renderer.DisplayToWorld()
    #     RPoint = renderer.GetWorldPoint()
    #     RPoint0 = RPoint[0]
    #     RPoint1 = RPoint[1]
    #     RPoint2 = RPoint[2]
    #     RPoint3 = RPoint[3]

    #     if RPoint3 != 0.0:
    #         RPoint0 = RPoint0/RPoint3
    #         RPoint1 = RPoint1/RPoint3
    #         RPoint2 = RPoint2/RPoint3

    #     camera.SetFocalPoint( (FPoint0-RPoint0)/2.0 + FPoint0,
    #                           (FPoint1-RPoint1)/2.0 + FPoint1,
    #                           (FPoint2-RPoint2)/2.0 + FPoint2)
    #     camera.SetPosition( (FPoint0-RPoint0)/2.0 + PPoint0,
    #                         (FPoint1-RPoint1)/2.0 + PPoint1,
    #                         (FPoint2-RPoint2)/2.0 + PPoint2)

    #     renWin = self.vtkWidget.GetRenderWindow()
    #     renWin.Render()


    # # Dolly converts y-motion into a camera dolly commands.
    # def dolly(self, renderer, camera, x, y, lastX, lastY, centerX, centerY):
    #     dollyFactor = pow(1.02,(0.5*(y-lastY)))
    #     if camera.GetParallelProjection():
    #         parallelScale = camera.GetParallelScale()*dollyFactor
    #         camera.SetParallelScale(parallelScale)
    #     else:
    #         camera.Dolly(dollyFactor)
    #         renderer.ResetCameraClippingRange()

    #     renWin = self.vtkWidget.GetRenderWindow()
    #     renWin.Render()

    # # Wireframe sets the representation of all actors to wireframe.
    # def wireframe(self):
    #     actors = self.ren.GetActors()
    #     actors.InitTraversal()
    #     actor = actors.GetNextItem()
    #     while actor:
    #         actor.GetProperty().SetRepresentationToWireframe()
    #         actor = actors.GetNextItem()

    #     renWin = self.vtkWidget.GetRenderWindow()
    #     renWin.Render()

    # # Surface sets the representation of all actors to surface.
    # def surface(self):
    #     actors = ren.GetActors()
    #     actors.InitTraversal()
    #     actor = actors.GetNextItem()
    #     while actor:
    #         actor.GetProperty().SetRepresentationToSurface()
    #         actor = actors.GetNextItem()
    #     renWin = self.vtkWidget.GetRenderWindow()
    #     renWin.Render()

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
    # windowR = OverlayWidget(cams.camR, meshPath, scale=stlScale, masterWidget=windowL)
    sys.exit(app.exec_())