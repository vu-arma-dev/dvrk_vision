#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
import message_filters
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

## Take a Float64 MultiArray message, convert it into a numpyMatrix
def multiArrayToMatrixList(ma_msg):
    dim = len(ma_msg.layout.dim)
    offset = ma_msg.layout.data_offset

    if (ma_msg.layout.dim[0].label != "rows"):
        print "Error: dim[0] should be the rows"
    rows = ma_msg.layout.dim[0].size
    
    if (ma_msg.layout.dim[1].label != "cols"):
        print "Error: dim[1] should be the columns"
    columns = ma_msg.layout.dim[1].size

    return np.array(ma_msg.data, dtype=np.float64).reshape((rows, columns))

def matrixListToMultiarray(matrix):
    rows, columns = matrix.shape
    msg = Float64MultiArray()
    # Set up layout
    msg.layout.data_offset = 0
    
    row_dim = MultiArrayDimension()
    row_dim.label = "rows"
    row_dim.size = rows
    row_dim.stride = columns * rows
    
    col_dim = MultiArrayDimension()
    col_dim.label = "cols"
    col_dim.size = columns
    col_dim.stride = columns

    msg.layout.dim = [row_dim, col_dim]
    msg.data = matrix.flatten().tolist()
    
    return msg

def generateGrid(xmin, xmax, ymin, ymax, res):
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    Xg,Yg = np.meshgrid(x,y)
    grid = np.array([Xg.flatten(), Yg.flatten()]).T
    return grid

class StiffnessToImageNode:
    def __init__(self):
        rospy.init_node('stiffness_to_image_converter', anonymous=True)
        self.domain = [-15, 10, -15, 30]
        self.resolution = 100
        self.grid = generateGrid(self.domain[0], self.domain[1], self.domain[2], self.domain[3], self.resolution)
        # Publishers and subscribers
        self.imagePub = rospy.Publisher('/stereo/stiffness_image', Image, queue_size = 1)
        stiffSub = message_filters.Subscriber('/dvrk/GP/get_stiffness', Float64MultiArray)
        pointsSub = message_filters.Subscriber('/dvrk/GP/get_surface_points', Float64MultiArray)
        stiffSub.registerCallback(self.stiffnessCB)
        pointsSub.registerCallback(self.pointsCB)
        self.points = None
        self.stiffness = None
        # ts = message_filters.TimeSynchronizer([stiffSub, pointsSub], 1)
        # ts.registerCallback(self.stiffnessCB)



    def pointsCB(self, points):
        self.points = multiArrayToMatrixList(points)

    def stiffnessCB(self, stiffness):
        self.stiffness = multiArrayToMatrixList(stiffness).transpose()

    def update(self):
        if self.points is None:
            return
        pointsMat = self.points
        stiffMat = self.stiffness
        if len(pointsMat) != len(stiffMat):
            return
        assert pointsMat.shape[1] == 3
        stiffMap = griddata(pointsMat[:,0:2], stiffMat, self.grid, method="linear", fill_value=-1).reshape(self.resolution, self.resolution)
        stiffMap[stiffMap == -1] = np.min(stiffMap[stiffMap != -1])
        print(np.min(stiffMap), np.max(stiffMap), np.min(stiffMat), np.max(stiffMat))
        # Normalize
        stiffMap[stiffMap < np.mean(stiffMap)] = np.mean(stiffMap)
        stiffMap -= np.min(stiffMap)
        stiffMap /= np.max(stiffMap)
        stiffMap *= 255
        plt.figure(1)
        plt.clf()
        plt.imshow(stiffMap, origin='lower', cmap="hot", extent=self.domain)
        plt.colorbar()
        plt.scatter(pointsMat[:,0],pointsMat[:,1])
        plt.pause(0.05)

if __name__ == '__main__':
    node = StiffnessToImageNode()
    plt.axis([0, node.resolution, 0, node.resolution])
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        node.update()
        rate.sleep()
    rospy.spin()

