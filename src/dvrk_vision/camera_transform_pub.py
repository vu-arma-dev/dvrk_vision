#!/usr/bin/env python
import rospy
from registration_gui import cleanResourcePath
from tf_conversions import posemath
import yaml
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import PyKDL

def arrayToPyKDLFrame(array):
    rot = arrayToPyKDLRotation(array)
    pos = PyKDL.Vector(array[0][3],array[1][3],array[2][3])
    return PyKDL.Frame(rot,pos)

def arrayToPyKDLRotation(array):
    x = PyKDL.Vector(array[0][0], array[1][0], array[2][0])
    y = PyKDL.Vector(array[0][1], array[1][1], array[2][1])
    z = PyKDL.Vector(array[0][2], array[1][2], array[2][2])
    return PyKDL.Rotation(x,y,z)

def pubTF(pose, parentName, childName):
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parentName
    t.child_frame_id = childName

    transform = posemath.toTf(pose)

    t.transform.translation.x = transform[0][0]
    t.transform.translation.y = transform[0][1]
    t.transform.translation.z = transform[0][2]

    t.transform.rotation.x = transform[1][0]
    t.transform.rotation.y = transform[1][1]
    t.transform.rotation.z = transform[1][2]
    t.transform.rotation.w = transform[1][3]

    br.sendTransform(t)

if __name__=="__main__":
    rospy.init_node('camera_transform_publisher')
    yamlPath = rospy.get_param("~transform_yaml")
    worldFrame = rospy.get_param("~world_frame")
    childFrame = rospy.get_param("~child_frame")
        

    if yamlPath != "":
        yamlFile = cleanResourcePath(yamlPath)
        
        with open(yamlFile, 'r') as stream:
            data = yaml.load(stream)
        cameraTransform = arrayToPyKDLFrame(data['transform'])
    else:
        cameraTransform = PyKDL.Frame()

    print(cameraTransform)

    def poseCB(data):
        global cameraTransform
        cameraTransform = posemath.fromMsg(data)

    rospy.Subscriber('/stereo/set_camera_transform', geometry_msgs.msg.Pose, poseCB)

    rate = rospy.Rate(100) # 60hz
    while not rospy.is_shutdown():
        pubTF(cameraTransform, worldFrame, childFrame)
        rate.sleep()