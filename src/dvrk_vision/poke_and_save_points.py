#!/usr/bin/env python
import rospy
from dvrk import psm
from geometry_msgs.msg import PoseStamped
from tf_conversions import posemath
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import yaml
import PyKDL


def plot(stlPoints, probePoints, camTransform, stlTransform, stlScale):
    global fig, axis
    plt.cla()
    ax.auto_scale_xyz(.5,.5,.5)
    bonePts = np.hstack((stlPoints,
                         np.ones((stlPoints.shape[0],1))))
    tf = stlTransform
    tf[0:3,0:3] *= stlScale
    for idx, point in enumerate(bonePts):
        bonePts[idx,:] = np.dot(tf,  point)
    ax.scatter(bonePts[0::4,0],
               bonePts[0::4,1],
               zs = bonePts[0::4,2])
    ptsTransformed = np.hstack((probePoints,
                                np.ones((probePoints.shape[0],1))))
    tf = np.linalg.inv(camTransform)
    for idx, point in enumerate(ptsTransformed):
        ptsTransformed[idx,:] = np.dot(tf, point)
    ax.scatter(ptsTransformed[:,0],
               ptsTransformed[:,1],
               ptsTransformed[:,2], c = 'r')

    ax.scatter([0],[0],[0],c='g')
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-0.5, 0.5)
    # print(ptsTransformed)
    ax.set_aspect('equal')

    plt.pause(0.001)

def main():
    robot = psm('PSM2')
    poseSub = rospy.Subscriber("/stereo/registration_pose", PoseStamped, poseCB)
    stl = mesh.Mesh.from_file('/home/biomed/october_15_ws/src/dvrk_vision/defaults/bunny.stl') 


    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(scriptDirectory, '..', '..', 'defaults', 
                            'registration_params.yaml')
    with open(filePath, 'r') as f:
        data = yaml.load(f)


    camTransform = np.array(data['transform'])
    stlTransform = np.eye(4)
    stlScale = 0.102

    rate = rospy.Rate(15) # 15hz
    while not rospy.is_shutdown():
        pos = robot.get_current_position()
        zVector = pos.M.UnitZ()
        pos = np.array([[pos.p.x(), pos.p.y(), pos.p.z()]])
        offset = np.array([zVector.x(), zVector.y(), zVector.z()])
        offset = offset * 0.008
        pos += offset
        stlTransform = posemath.toMatrix(pose)
        plot(stl.points[:,0:3], pos, camTransform, stlTransform, stlScale)
        rate.sleep()
    quit()
    poked = False
    pokePoints = []
    thresh = 2.5
    while not rospy.is_shutdown():
        pos = robot.get_current_position()
        norm = pos.M.UnitZ()
        offset = np.array([norm.x(), norm.y(), norm.z()]) * 0.008
        pos += offset
        norm = np.array([norm.x(), norm.y(), norm.z()]) * -1
        force = robot.get_current_wrench_body()[0:3]
        force = np.dot(force, norm)
        if force > thresh and poked == False:
            pokePoints.append([pos.p.x(), pos.p.y(), pos.p.z()])
            print("POKED", len(pokePoints))
            poked = True
        elif force < thresh/2 and poked == True:
            poked = False
        # print("%2.3f, %2.3f, %2.3f" % tuple(force[0:3].tolist()))
        rate.sleep()
    print("Camera transform:")
    print(np.linalg.inv(camTransform))
    print("Stl Transform")

    print("Probed Points")
    print(pokePoints)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    plt.show()
    def poseCB(data):
        global pose
        pose = posemath.fromMsg(data.pose)

    pose = PyKDL.Frame()
    main()
    plt.close(fig)