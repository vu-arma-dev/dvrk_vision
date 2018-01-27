#!/usr/bin/env python
import sys
import cv2
import vtk
import vtktools
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    _QT_VERSION = 5
    from PyQt5.QtCore import QObject, pyqtSignal
    from PyQt5.QtWidgets import QWidget
    from QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from PyQt5.QtWidgets import QApplication
else:
    _QT_VERSION = 4
    from PyQt4.QtCore import QObject, pyqtSignal
    from PyQt4.QtWidgets import QWidget
    from PyQt4.QtWidgets import QApplication
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import message_filters

class Camera(QObject):
    trigger = pyqtSignal()
    def __init__(self, parent):
        super(Camera, self).__init__(parent)
        self.info = None
        self.image = None

class StereoCameras(QObject):
    bridge = CvBridge()

    def __init__(self,imageTopicL, imageTopicR, infoTopicL, infoTopicR, slop=0, parent=None):
        super(StereoCameras, self).__init__(parent)
        # Set up internal variables
        self.camR = Camera(parent)
        self.camL = Camera(parent)

        # Create subscribers
        subs = [message_filters.Subscriber(imageTopicL, Image),
                message_filters.Subscriber(imageTopicR, Image),
                message_filters.Subscriber(infoTopicL, CameraInfo),
                message_filters.Subscriber(infoTopicR, CameraInfo)]
        if(slop == 0):
            ts = message_filters.TimeSynchronizer(subs,1)
        else:
            ts = message_filters.ApproximateTimeSynchronizer(subs, 1, slop)
        
        ts.registerCallback(self.cb)
        self.started = False

    def cb(self, imageMsgL, imageMsgR, camInfoL, camInfoR):
        self.camL.image = self.bridge.imgmsg_to_cv2(imageMsgL, "bgr8")
        self.camR.image = self.bridge.imgmsg_to_cv2(imageMsgR, "bgr8")
        self.camL.info = camInfoL
        self.camR.info = camInfoR
        # Emit a trigger for repaint
        self.camL.trigger.emit()
        self.camR.trigger.emit()

class QVTKStereoViewer(QVTKRenderWindowInteractor):
    def __init__(self, camera, parent=None, **kw):
        super(QVTKStereoViewer, self).__init__(parent, **kw)
        self.cam = camera
        self.setUp = False
        self.aspectRatio = 1
        self.numResizes = 0

    def start(self):
        self.cam.trigger.connect(self.imageCb)

    def renderSetup(self):
        pass

    def imageCb(self):
        if not self.setUp:
            image = self.cam.image
            self.aspectRatio = image.shape[1] / float(image.shape[0])
            # self.resize(self.width(), self.height())
            # Set up vtk camera using camera info
            self.bgImage = vtktools.makeVtkImage(image.shape[0:2])
            renWin = self.GetRenderWindow()
            intrinsicMat, extrinsicMat = vtktools.matrixFromCamInfo(self.cam.info)
            self.ren, bgRen = vtktools.setupRenWinForRegistration(renWin,
                                                                  self.bgImage,
                                                                  intrinsicMat)
            size = self._RenderWindow.GetSize()
            self._Iren.SetSize(size)
            self._Iren.ConfigureEvent()
            super(QVTKRenderWindowInteractor,self).resize(size[0], size[1])
            pos = extrinsicMat[0:3,3]
            self.ren.GetActiveCamera().SetPosition(pos)
            pos[2] = 1
            self.ren.GetActiveCamera().SetFocalPoint(pos)
            self.renderSetup()
            self.setUp = True
        if self.isVisible():
            image = self.imageProc(self.cam.image)
            self.aspectRatio = image.shape[1] / float(image.shape[0])
            vtktools.numpyToVtkImage(image,self.bgImage)
            self.Render()

    def imageProc(self, image):
        return image

    def resizeEvent(self, ev):
        self.numResizes = self.numResizes + 1
        print self.width(), self.height(), self.numResizes
        newHeight = int(ev.size().height() * self.aspectRatio)
        size = min(newHeight, ev.size().width())
        offset = ((ev.size().width() - size) / 2,
                  (ev.size().height() - int(size / self.aspectRatio)) / 2)
        width, height = (size,int(size / self.aspectRatio))
        QWidget.resize(self, width, height)
        self._RenderWindow.SetSize(width, height)
        self._Iren.SetSize(width, height)
        self._Iren.ConfigureEvent()
        self.move(offset[0], offset[1])
        self.update()

if __name__ == "__main__":

    app = QApplication(sys.argv)
    rosThread = vtktools.QRosThread()
    rosThread.start()
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("stereo/left/image_rect",
                         "stereo/right/image_rect",
                         "stereo/left/camera_info",
                         "stereo/right/camera_info",
                         slop = slop)
    windowL = QVTKStereoViewer(cams.camL)
    windowL.Initialize()
    windowL.start()
    windowL.show()
    sys.exit(app.exec_())