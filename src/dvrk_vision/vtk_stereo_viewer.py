#!/usr/bin/env python
import sys
import cv2
import vtk
import vtktools
import numpy as np
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    _QT_VERSION = 5
    from PyQt5.QtCore import QObject, pyqtSignal
    from PyQt5.QtWidgets import QWidget, QVBoxLayout
    from QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from PyQt5.QtWidgets import QApplication
else:
    _QT_VERSION = 4
    from PyQt4.QtCore import QObject, pyqtSignal
    from PyQt4.QtWidgets import QWidget
    from PyQt4.QtGUI import QVBoxLayout
    from PyQt4.QtWidgets import QApplication
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import message_filters

class Camera(QObject):
    trigger = pyqtSignal()
    def __init__(self, parent, topic):
        super(Camera, self).__init__(parent)
        self.info = None
        self.image = None
        self.topic = topic

class StereoCameras(QObject):
    bridge = CvBridge()

    def __init__(self,imageTopicL, imageTopicR, infoTopicL, infoTopicR, slop=0, parent=None):
        super(StereoCameras, self).__init__(parent)
        # Set up internal variables
        self.camL = Camera(self, imageTopicL)
        self.camR = Camera(self, imageTopicR)

        # Create subscribers
        subs = [message_filters.Subscriber(self.camL.topic, Image),
                message_filters.Subscriber(self.camR.topic, Image),
                message_filters.Subscriber(infoTopicL, CameraInfo),
                message_filters.Subscriber(infoTopicR, CameraInfo)]
        if(slop == 0):
            self.ts = message_filters.TimeSynchronizer(subs,1)
        else:
            self.ts = message_filters.ApproximateTimeSynchronizer(subs, 1, slop)
        
        self.ts.registerCallback(self.cb)
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
        self.shown = False
        self.aspectRatio = 1

        # cb = vtktools.vtkTimerCallback(self._Iren, self)
        # # cb.update = self.update
        # self._Iren.AddObserver('TimerEvent', cb.execute)
        # self._Iren.CreateRepeatingTimer(15)


    def start(self):
        self.cam.trigger.connect(self.imageCb)

    def renderSetup(self):
        pass

    def imageCb(self):
        if not self.setUp:
            image = self.cam.image
            fgImage = np.dstack((image, np.zeros(image.shape[0:2], np.uint8)))
            self.aspectRatio = image.shape[1] / float(image.shape[0])
            # self.resize(self.width(), self.height())
            # Set up vtk camera using camera info
            self.image = vtktools.makeVtkImage(image.shape)
            self.fgImage = vtktools.makeVtkImage(fgImage.shape)
            renWin = self.GetRenderWindow()
            intrinsicMat, extrinsicMat = vtktools.matrixFromCamInfo(self.cam.info)
            self.ren, bgRen, fgRen = vtktools.setupRenWinForRegistration(renWin,
                                                                         self.image,
                                                                         self.fgImage,
                                                                         intrinsicMat)

            pos = extrinsicMat[0:3,3]
            self.ren.GetActiveCamera().SetPosition(pos)
            pos[2] = 1
            self.ren.GetActiveCamera().SetFocalPoint(pos)
            self.renderSetup()
            self.setUp = True

        if self.isVisible():
            bgImage = self.imageProc(self.cam.image)
            fgImage = np.dstack((bgImage, np.zeros(bgImage.shape[0:2], np.uint8)))
            self.aspectRatio = bgImage.shape[1] / float(bgImage.shape[0])
            vtktools.numpyToVtkImage(bgImage,self.image)
            vtktools.numpyToVtkImage(fgImage,self.fgImage)
            self._Iren.Render()

    def showEvent(self, event):
        if not self.shown:
            size = self._RenderWindow.GetSize()
            self._Iren.SetSize(size)
            self._Iren.ConfigureEvent()
            super(QVTKRenderWindowInteractor,self).resize(size[0], size[1])
            self.shown = True
        super(QVTKStereoViewer, self).showEvent(event)

    def imageProc(self, image):
        return image

    def resizeEvent(self, ev):
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
    if _QT_VERSION == 5:
        from PyQt5.QtWidgets import QMainWindow
    elif _QT_VERSION == 4:
        from PyQt4.QtGUI import QMainWindow
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
    winL = QMainWindow()
    winR = QMainWindow()
    windowL = QVTKStereoViewer(cams.camL)

    windowR = QVTKStereoViewer(cams.camR)

    layoutL = QVBoxLayout()
    winL.setCentralWidget(windowL)
    winR.setCentralWidget(windowR)

    windowL.Initialize()
    windowL.start()

    windowR.Initialize()
    windowR.start()

    winL.show()
    winR.show()
    sys.exit(app.exec_())