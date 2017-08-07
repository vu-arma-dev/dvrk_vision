#!/usr/bin/env python
# from point_cloud_registration import PointCloudRegistration
from stereo_processing import StereoProcessing
# from dual_quaternion_registration.qf_register import qf_register
import numpy as np
import rospy
import os


if __name__ == '__main__':

    # Initialize the node
    rospy.init_node("stereo_processing_registration")
    rate = rospy.Rate(1) # 30hz

    # Set up stereo processing
    stereo = StereoProcessing("/stereo", maskTopic="/stereo/left/image_rect_mask")

    # Get txt file of fixed organ
    functionPath = os.path.dirname(os.path.realpath(__file__))
    stlFileName = os.path.join(functionPath,'..','defaults','femur.stl')
    # registration = PointCloudRegistration(stlFileName, .001)

    mask = stereo.mask

    while not rospy.is_shutdown():
        stereo.update()
        # registration.update(stereo.points)
        # # Reset registration if mask changes
        # if not np.allclose(mask, stereo.mask) and 0 in stereo.mask:
        #     registration.reset()
        #     mask = stereo.mask
        rate.sleep()