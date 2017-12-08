import cv2
import vtk
import vtktools
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    _QT_VERSION = 5
    from PyQt5.QtCore import QObject, pyqtSignal
    from QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
else:
    _QT_VERSION = 4
    from PyQt4.QtCore import QObject, pyqtSignal
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

    def start(self):
        self.cam.trigger.connect(self.setupWindow)

    def setupWindow(self):
        image = self.cam.image
        # Set up vtk camera using camera info
        self.bgImage = vtktools.makeVtkImage(image.shape[0:2])
        renWin = self.GetRenderWindow()
        intrinsicMat, extrinsicMat = vtktools.matrixFromCamInfo(self.cam.info)
        self.ren, bgRen = vtktools.setupRenWinForRegistration(renWin,
                                                              self.bgImage,
                                                              intrinsicMat)
        renWin.SetSize(self.width(),self.height())
        pos = extrinsicMat[0:3,3]
        self.ren.GetActiveCamera().SetPosition(pos)
        pos[2] = 1
        self.ren.GetActiveCamera().SetFocalPoint(pos)
        self.renderSetup()
        self.cam.trigger.disconnect()
        self.cam.trigger.connect(self.imageCb)

    def renderSetup(self):
        pass

    def imageCb(self):
        image = self.imageProc(self.cam.image)
        vtktools.numpyToVtkImage(image,self.bgImage)
        self.Render()

    def imageProc(self, image):
        return image