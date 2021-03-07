#include "utility.h"
ros::Subscriber subLaserOdometry;

void odomGtHandler(const geometry_msgs::PoseStamped::ConstPtr& odomGt) {
    ofstream foutC("/home/zhikaizhu/output/ss_01_gt.csv", ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(10);
    foutC << odomGt->header.stamp.toSec()<< " ";
    foutC.precision(5);
    foutC << odomGt->pose.position.x << " "
            << odomGt->pose.position.y << " "
            << odomGt->pose.position.z << " "
            << odomGt->pose.orientation.w << " "
            << odomGt->pose.orientation.x << " "
            << odomGt->pose.orientation.y << " "
            << odomGt->pose.orientation.z << endl;
    foutC.close();
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "publish_gt");
    ros::NodeHandle nh("~");

    ROS_INFO("Publishing groundtruth----");
    subLaserOdometry = nh.subscribe<geometry_msgs::PoseStamped>(
        "/lips_sim/truepose_lidar", 5, odomGtHandler);
    ros::spin();
    return 0;
}