#include <nodelet/nodelet.h>
#include <nodelet_topic_tools/nodelet_lazy.h>
#include <ros/ros.h>
#include <camera_info_manager/camera_info_manager.h>
#include "sensor_msgs/CameraInfo.h"

namespace dvrk_vision {

class StereoCameraInfo{
  public:
    StereoCameraInfo(ros::NodeHandle comm_nh, ros::NodeHandle param_nh);
    void onInit();
	void infoCB(const sensor_msgs::CameraInfoConstPtr& info_msg);
    ~StereoCameraInfo();

  private:
    ros::NodeHandle node, pnode;
	// ROS communication
	ros::Publisher left_info_pub, right_info_pub;
	ros::Subscriber info_sub;

	camera_info_manager::CameraInfoManager left_info_mgr;
	camera_info_manager::CameraInfoManager right_info_mgr;
};

StereoCameraInfo::StereoCameraInfo(ros::NodeHandle comm_nh, ros::NodeHandle param_nh) :
  node(comm_nh), pnode(param_nh),
  left_info_mgr(ros::NodeHandle(comm_nh, "left"), "left_camera"),
  right_info_mgr(ros::NodeHandle(comm_nh, "right"), "right_camera") {

	std::string left_url, right_url;
	pnode.getParam("left_camera_info", left_url);
	pnode.getParam("right_camera_info", right_url);

	const std::string left_name = "left_camera";

	pnode.getParam("left/camera_info_url", left_url);
	pnode.getParam("right/camera_info_url", right_url);

	left_info_mgr.loadCameraInfo(left_url);
	right_info_mgr.loadCameraInfo(right_url);

	left_info_pub = node.advertise<sensor_msgs::CameraInfo>("left/camera_info", 1);
	right_info_pub = node.advertise<sensor_msgs::CameraInfo>("right/camera_info", 1);

	info_sub = node.subscribe("camera_info", 1, &StereoCameraInfo::infoCB, this);
}

void StereoCameraInfo::infoCB(const sensor_msgs::CameraInfoConstPtr& info_msg) {
  sensor_msgs::CameraInfoPtr info_left(new sensor_msgs::CameraInfo(left_info_mgr.getCameraInfo()));
  sensor_msgs::CameraInfoPtr info_right(new sensor_msgs::CameraInfo(right_info_mgr.getCameraInfo()));

  info_left->header.stamp = info_msg->header.stamp;
  info_right->header.stamp = info_msg->header.stamp;
  info_left->header.frame_id = info_msg->header.frame_id;
  info_right->header.frame_id = info_msg->header.frame_id;

  left_info_pub.publish(info_left);
  right_info_pub.publish(info_right);
}

class StereoCameraInfoNodelet : public nodelet::Nodelet {
  public:
    StereoCameraInfoNodelet() {}

    void onInit() {
      ros::NodeHandle node = getNodeHandle();
      ros::NodeHandle pnode = getPrivateNodeHandle();

      stereo = new StereoCameraInfo(node, pnode);
    }

    ~StereoCameraInfoNodelet() {
      if (stereo) delete stereo;
    }

  private:
    StereoCameraInfo *stereo;
};

};
// Export nodelet
#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(dvrk_vision, StereoCameraInfoNodelet, dvrk_vision::StereoCameraInfoNodelet, nodelet::Nodelet);