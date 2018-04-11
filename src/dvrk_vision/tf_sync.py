import sys
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo
from collections import namedtuple, deque
import rostopic
import message_filters

class CameraSync:
    # Variables to be shared across all TransformCameraSync objects
    _tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(1))
    _listener = tf2_ros.TransformListener(_tfBuffer)

    def __init__(self, cameraTopic, topics, frames, slop = 1/15.0):
        self.camTime = None
        self.baseFrame = 'world'
        self._camSub = None
        self.subs = [None] * (len(topics) + 1) 
        self.synchedMessages = [None] * (len(topics) + 1)
        self.syncher = None
        self.setCameraTopic(cameraTopic)
        self.topics = topics
        self.slop = slop
        self.registerTopics()
        self.frames = frames

    def unregister(self):
        if self._camSub != None:
            self._camSub.unregister()

    def registerTopics(self):
        for idx, topic in enumerate(self.topics):
            try:
                msg_class, _, _ = rostopic.get_topic_class(topic)
                self.subs[idx] = message_filters.Subscriber(topic, msg_class)
            except ValueError:
                rospy.logfatal("Could not subscribe to " + topic + ". Make sure topic exists")
                sys.exit()
        self.subs[len(self.topics)] = self._camSub
        self.syncher = message_filters.ApproximateTimeSynchronizer(self.subs,
                                                                   queue_size = 10,
                                                                   slop = self.slop)
        self.syncher.registerCallback(self._topicCB)

    def _topicCB(self, *args):
        for idx, msg in enumerate(args):
            self.synchedMessages[idx] = msg

    def setCameraTopic(self, topic):
        self.unregister()
        # Unregister and add 'camera_info' to topic if necessary
        infoTopic = topic
        if topic[-12:] != '/camera_info':
            infoTopic = topic + '/camera_info'
        print infoTopic
        self._camSub = message_filters.Subscriber(infoTopic, CameraInfo)
        self._camSub.registerCallback(self._camCB)

    def _camCB(self, msg):
        self.baseFrame = msg.header.frame_id
        self.camTime = msg.header.stamp

    def getTransforms(self):
        # Make local variables so they won't change during _camCB
        camTime = self.camTime
        baseFrame = self.baseFrame
        transforms = []
        if camTime == None:
            rospy.loginfo("TransformCameraSync: camera_info topic not found. \
                           Returning [].")
            return []

        for name in self.frames:
            # Check for transformations between all frames and base
            try:
                transform = self._tfBuffer.lookup_transform(baseFrame,
                                                            name,
                                                            camTime,
                                                            rospy.Duration(0.01))
                transforms.append(transform)
            except tf2_ros.TransformException as e:
                pass
        return transforms
        

if __name__ == '__main__':
    from dvrk_vision.vtk_stereo_viewer import StereoCameras
    from image_geometry import StereoCameraModel
    import yaml
    import os
    import numpy as np
    import cv2
    from tf.transformations import quaternion_matrix
    from geometry_msgs.msg import TransformStamped, Transform


    # Get camera registration
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(scriptDirectory, '..', '..', 'defaults', 
                            'registration_params.yaml')
    with open(filePath, 'r') as f:
        data = yaml.load(f)
    camTransform = np.array(data['transform'])

    # create node
    if not rospy.get_node_uri():
        rospy.init_node('tf_synch_test', anonymous = True, log_level = rospy.WARN)
    else:
        rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras( "/stereo/left/image_rect",
                          "/stereo/right/image_rect",
                          "/stereo/left/camera_info",
                          "/stereo/right/camera_info",
                          slop = slop)

    camModel = StereoCameraModel()
    topicLeft = rospy.resolve_name("/stereo/left/camera_info")
    msgL = rospy.wait_for_message(topicLeft,CameraInfo,3);
    topicRight = rospy.resolve_name("/stereo/right/camera_info")
    msgR = rospy.wait_for_message(topicRight,CameraInfo,3);
    camModel.fromCameraInfo(msgL,msgR)

    tfSynch = CameraSync('/stereo/left/camera_info',
                         topics = [],
                         frames = ['PSM2_psm_base_link',
                                   'PSM2_tool_wrist_link',
                                   'PSM2_tool_wrist_caudier_link_shaft'])
    rate = rospy.Rate(15) # 15hz
    while not rospy.is_shutdown():
        # Get last images
        image = cams.camL.image

        # Wait for images to exist
        if type(image) == type(None):
            continue

        # Check we have valid images
        (rows,cols,channels) = image.shape
        if cols < 60 or rows < 60:
            continue

        transforms = tfSynch.getTransforms()

        # transform = Transform(pose.position, pose.orientation)
        # transforms.append(TransformStamped(msgs[0].header, '', transform))
        for i in range(0,len(transforms)-1):

            start = [transforms[i].transform.translation.x,
                     transforms[i].transform.translation.y,
                     transforms[i].transform.translation.z]

            end = [transforms[i+1].transform.translation.x,
                   transforms[i+1].transform.translation.y,
                   transforms[i+1].transform.translation.z]

            # Project position into 2d coordinates
            posStart = camModel.left.project3dToPixel(start)
            posEnd = camModel.left.project3dToPixel(end)
            if not np.isnan(posStart + posEnd).any(): 
                posStart = [int(l) for l in posStart]
                posEnd = [int(l) for l in posEnd]
                cv2.line(image, tuple(posStart), tuple(posEnd), (255, 255, 0), 2)

        
        # Draw images and display them
        cv2.imshow('tf_synch_test', image)
        key = cv2.waitKey(1)
        rate.sleep()