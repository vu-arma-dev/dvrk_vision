''' 
TODO:

Initial rotaition problem
1) Return a quality value from the cpp code
2) Use that quality to decide on which of many random initial rotations

Jitteryness problem
1) cpp code should accept a covariance scalar
2) find some way to scale that uncertainty over time

General bugs
1) Window size larger than number of points breaks code

'''

#!/usr/bin/env python
__all__ = ["PointCloudRegistration"]

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
# dependencies can be any iterable with strings, 
# e.g. file line-by-line iterator
dependencies = [
  'numpy>=1.0',
  'numpy-stl>=0.9',
  'rospy>=1.11'
]
# here, if a dependency is not met, a DistributionNotFound or VersionConflict
# exception is thrown. 
pkg_resources.require(dependencies)

import os
import numpy as np
import rospy
from tf import transformations
from visualization_msgs.msg import Marker
from dual_quaternion_registration.qf_register import qf_register, reg_params_to_transformation_matrix
from stl import mesh

def write_txt(filename, points):
    # Writes an n x 3 matrix into a text file 
    points = points.reshape(-1, 3)
    with open(filename, 'w') as f:
        f.seek(0)
        np.savetxt(f,points,'%f %f %f')
        f.truncate()

def make_marker(id, rgba):
    # Set up marker to send to rViz
    marker = Marker()
    marker.header.frame_id = "/world"
    marker.header.stamp    = rospy.get_rostime()
    #marker.ns = "robot"
    marker.id = id
    marker.type = 10 # mesh resource
    marker.action = 0
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]
    marker.color.a = rgba[3]
    marker.mesh_resource = "package://dvrk_vision/scripts/fixed.stl"
    return marker

def delete_all_markers(pub):
    marker = Marker()
    marker.header.frame_id="/world"
    marker.header.stamp = rospy.get_rostime()
    marker.id = -1
    marker.action = 3
    pub.publish(marker)

def set_marker_matrix(marker, tfMatrix):
    # Set the transformation of a visualization marker
    position = tfMatrix[0:3,3]
    rotation = transformations.quaternion_from_matrix(tfMatrix)
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = rotation[0]
    marker.pose.orientation.y = rotation[1]
    marker.pose.orientation.z = rotation[2]
    marker.pose.orientation.w = rotation[3]

class PointCloudRegistration:
    def __init__(self, stlPath, scale):
        # Initialize paths which we will send to the registration function
        functionPath = os.path.dirname(os.path.realpath(__file__))
        self.fileNameMoving = os.path.join(functionPath,'moving.txt')
        self.fileNameFixed = os.path.join(functionPath,'fixed.txt')

        # Load STL from file and save the scaled / rotated version to a local folder
        stlMesh = mesh.Mesh.from_file(stlPath)
        stlMesh.points *= scale
        # stlMesh.rotate([1,0,0], -np.pi/2)
        # stlMesh.rotate([0,1,0], np.pi)
        stlMesh.save(os.path.join(functionPath,'fixed.stl'))

        # Get a set of unique vertices from the STL
        points = stlMesh.points.reshape((stlMesh.points.shape[0]*3,3))
        pointsSliced = np.ascontiguousarray(points)
        pointsSliced = pointsSliced.view(np.dtype((np.void, points.dtype.itemsize * points.shape[1])))
        _, uniqueIdx = np.unique(pointsSliced, return_index=True)

        # Initialize variables for registration
        self.pointsFixed = points[uniqueIdx]
        self.pointsMoving = np.zeros((1,3))
        self.finalTransform = np.identity(4)

        self.stlMarker = make_marker(0, [0,1,0,1])

        self.markerPub = rospy.Publisher('organMarker', Marker, queue_size=10)
        self.markerPub.publish(self.stlMarker)

        self.lastError = 1
        self.initialized = False

    def update(self, pointsMoving, maxIter=100, inlierRatio=.8):
        # the update function takes a n x 3 numpy array representing a new set of registration points
        # and returns a 4 x 4 numpy array representing the transformation matrix that best fits
        # that point cloud to the STL used to initialize this object

        # write moving points to file for registration
        self.pointsMoving = pointsMoving # - offset
        write_txt(self.fileNameMoving, self.pointsMoving)

        # transform fixed pointcloud from STL based on initial guess and save to file
        newRow = np.ones((1,self.pointsFixed.shape[0]))
        pointsTransformed = np.vstack([self.pointsFixed.transpose(), newRow])
        #print(pointsTransformed[0])
        pointsTransformed = np.dot(self.finalTransform, pointsTransformed)[0:3,:].transpose()
        write_txt(self.fileNameFixed, pointsTransformed)

        # Perform bingham registration
        # TODO: Right now parameters are hard-coded
        inlierRatio = 1
        windowSize = 20
        transTolerance = .00001
        rotTolerance = .00001

        # Add uncertainty and randomize rotation based on error
        uncertainty = 10*min(1,self.lastError)

        regParams, self.lastError = qf_register(self.fileNameMoving, self.fileNameFixed,
                                                inlierRatio,maxIter,windowSize,
                                                transTolerance,rotTolerance, uncertainty)
        # Turn registration into a rotation matrix for our next inital guess
        tf = np.linalg.inv(reg_params_to_transformation_matrix(regParams))
        # Calculate actual transformation by adding current guess to previous transformations
        self.finalTransform = np.dot(tf, self.finalTransform)
        set_marker_matrix(self.stlMarker, self.finalTransform)
        self.stlMarker.header.stamp = rospy.get_rostime()
        self.markerPub.publish(self.stlMarker)

        return self.finalTransform

    def reset(self):
        pass
        self.lastError = 1
        self.finalTransform = np.identity(4)
        
        if len(self.pointsMoving) < 50:
            return

        # Find a suitable initial rotation
        sqrt = 1/np.sqrt(2)

        quats =[[ sqrt,  0.00,  0.00, sqrt],
                [-sqrt,  0.00,  0.00, sqrt],
                [ 0.00,  sqrt,  0.00, sqrt],
                [ 0.00, -sqrt,  0.00, sqrt],
                [ 0.00,  0.00,  sqrt, sqrt],
                [ 0.00,  0.00, -sqrt, sqrt],
                [ 0, 0, 0, 1],
                [ 1, 0, 0, 0],
                [ 0, 1, 0, 0],
                [ 0, 0, 1, 0],
                [-0.5,-0.5,-0.5, 0.5],
                [-0.5,-0.5, 0.5, 0.5],
                [-0.5, 0.5,-0.5, 0.5],
                [-0.5, 0.5, 0.5, 0.5],
                [ 0.5,-0.5,-0.5, 0.5],
                [ 0.5,-0.5, 0.5, 0.5],
                [ 0.5, 0.5,-0.5, 0.5],
                [ 0.5, 0.5, 0.5, 0.5],
                [0,sqrt,sqrt,0],
                [0,sqrt,-sqrt,0],
                [sqrt,0,sqrt,0],
                [sqrt,0,-sqrt,0],
                [sqrt,sqrt,0,0],
                [sqrt,-sqrt,0,0]]

        error = self.lastError
        bestTransforms = [np.identity(4),np.identity(4),np.identity(4)]
        numTransforms = 0
        maxTransforms = 3
        bestErrors = [1, 1, 1]

        pointsFixedCopy = self.pointsFixed.copy()
        self.pointsFixed = self.pointsFixed[range(0,len(self.pointsFixed),max(1,len(self.pointsFixed)/1000))]
        pointsMovingCopy = self.pointsMoving.copy()
        self.pointsMoving = self.pointsMoving[range(0,len(self.pointsMoving),max(1,len(self.pointsMoving)/1000))]


        for i in range(0,len(quats)):
            self.finalTransform = transformations.quaternion_matrix(quats[i])
            quat = quats[i]
            self.lastError = 1
            self.finalTransform[0:3,3] = [0, 0, 1]
            self.update(self.pointsMoving, maxIter=100, inlierRatio=1)
            # print 'x: %+.2f, y: %+.2f, z: %+.2f' % transformations.euler_from_quaternion(quat), self.lastError,
            if max(bestErrors) > self.lastError:
                idx = bestErrors.index(max(bestErrors))
                bestErrors[idx] = self.lastError
                bestTransforms[idx] = self.finalTransform.copy()
                # print "Best errors: ", bestErrors,
            # raw_input("Press Enter to continue...")
            # print ""
        
        self.pointsFixed = pointsFixedCopy
        self.pointsMoving = pointsMovingCopy

        # for quat in quats:
        #     quat = np.array(quat) / np.linalg.norm(quat)
        #     self.finalTransform = transformations.quaternion_matrix(quat)
        #     self.lastError = 1
        #     self.finalTransform[0:3,3] = [0, 0, 1]
        #     self.update(self.pointsMoving, maxIter=100)
        #     print error, self.lastError,
        #     if(self.lastError < error):
        #         error = self.lastError
        #         bestQuat = quat
        #         print "BESTQUAT",
        #     # raw_input("Press Enter to continue...")
        #     self.markerPub.publish(self.stlMarker)
        #     print ""
        
        # print bestQuat
        # print ""
        # position = self.finalTransform[0:3,3].copy()
        # self.finalTransform = transformations.quaternion_matrix(bestQuat)

        bestTransform = bestTransforms[0]
        bestError = bestErrors[0]
        for i in range(0,len(bestErrors)):
            self.lastError = 1
            self.finalTransform = bestTransforms[i]
            self.update(self.pointsMoving, maxIter=100, inlierRatio=1)
            if bestError > self.lastError:
                bestTransform = self.finalTransform.copy()
                bestError = self.lastError

        self.finalTransform = bestTransform

        # for i in range(0,len(bestErrors)):
        #     rgba = [0,0,0,.5]
        #     rgba[i] = 1
        #     stlMarker = make_marker(i, rgba)
        #     set_marker_matrix(stlMarker, bestTransforms[i])
        #     stlMarker.header.stamp = rospy.get_rostime()
        #     self.markerPub.publish(stlMarker)

        # guess = raw_input("Input r g or b to pick a registration...")
        # if(guess == "r"):
        #     self.finalTransform = bestTransforms[0]
        # elif(guess == "g"):
        #     self.finalTransform = bestTransforms[1]
        # elif(guess == "b"):
        #     self.finalTransform = bestTransforms[2]
        # delete_all_markers(self.markerPub)
        self.lastError = 1


