#! /usr/bin/env python3
from threading import local
import rospy
import sys
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pcl
import pcl.pcl_visualization
import numpy as np
import ros_numpy
import math
import signal
import cv2
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
#fig= plt.figure()
#angle_v=np.cos(np.dot(np.arange(0,256,1),0.3515625))
#print(math.cos(np.arange(0,255,1)*np.array(0.3515625)))
#des_ext_angle = np.array([])
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_15 = np.ones((15, 15), np.uint8)
# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)
fx=480
fy=240


class Collision_Avoidance:
    frames=[]
    desired_position = np.array([[10],[0],[5]])
    local_position = np.zeros((3,1))
    ranges = np.zeros((120))
    #local_position = PoseStamped()

def state_callback(data):
    current_state = data
    
def local_pose_callback(data):
    
    Collision_Avoidance.local_position[0]=data.pose.position.x 
    Collision_Avoidance.local_position[1]=data.pose.position.y 
    Collision_Avoidance.local_position[2]=data.pose.position.z 
    
def lidar_callback(data):
    point_map_x = []
    point_map_y = []    
    depth_map=np.zeros((860,860))
    depth_map2=np.zeros((860,860))
    for p in pc2.read_points(data,skip_nans=True,field_names=("x","y","z")):
        point_angle = np.math.atan2(p[0],-p[1])*180.0/np.math.pi
        
        #print(p[0],p[1],p[2])
        if(point_angle>30.0 and point_angle<150.0 ):#and abs(p[2])<0.05):#and abs(p[2])<0.05
            w=int(p[1]/p[0]*fy+380)
            h=int(p[2]/p[0]*fx+380)
            #print(w,h,p[0]*255/6)
            depth_map2[h][w]=p[0]*255/5
            #depth_map2[h][w]=p[0]*100/5
            
            
            
            #print(p[0],p[1],p[2])
            #print(point_angle)
            #stream=cv2.VideoCapture(depth_map)
            Collision_Avoidance.ranges[int(abs(point_angle-30))] = np.math.sqrt((np.math.pow(p[0],2)+np.math.pow(p[1],2)))
    depth_map=np.array(depth_map2).astype('uint8')
    depth_map2=depth_map
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = 255 - depth_map[valid_pixels]
    depth_map=cv2.dilate(depth_map,DIAMOND_KERNEL_5)

    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    #depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_7)

    #depth_map = cv2.erode(depth_map,FULL_KERNEL_5)

    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]
    
    depth_map = cv2.medianBlur(depth_map, 5)

    #empty_pixels = depth_map < 0.1
    #dilated = cv2.dilate(depth_map, FULL_KERNEL_9)
    #depth_map[empty_pixels] = dilated[empty_pixels]

    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_15)
    depth_map[empty_pixels] = dilated[empty_pixels]

    #depth_map=cv2.dilate(depth_map,DIAMOND_KERNEL_5)

    depth_map = cv2.medianBlur(depth_map, 5)
    
    #depth_map = cv2.bilateralFilter(depth_map,5,1.0,1.5)
    
    valid_pixels = (depth_map > 0.1)
    blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
    depth_map[valid_pixels] = blurred[valid_pixels]
    
    #depth_map = cv2.medianBlur(depth_map, 5)
    cv2.imshow('cam',depth_map2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

            #point_map_x.append(int(p[0]*1000))
            #point_map_y.append(int(p[1]*1000))   
    #print(len(Collision_Avoidance.ranges),Collision_Avoidance.ranges)
    #frames.append((point_map_y,point_map_x))
    
            #print(depth_map.shape)
            #point_map_y.append(p[1])
    
    #print('------------------------------------------------')
    #print('------------------------------------------------')
    #print('------------------------------------------------')
    
    #pc = ros_numpy.numpify(data)
    #print(pc.shape)
    #points=np.zeros((pc.shape[0],3))
    
    
    #print(angle_v.shape,pc[15][0:256])
    #print('x: ', np.dot(pc[15][0:256],angle_v))#0.3515625 1개 당 deg #x좌표계에 대해서만
    #plt.imshow(pc[15][:256])
    #plt.plot()
    #p = pcl.PointCloud(np.array(points, dtype=np.float32))
def Velocity_Pub(des_position,local_position,lidar_range):
    cmd_vel = Twist()

    distance = np.linalg.norm((des_position[:1]-local_position[:1]))
    K_att = 0.5
    K_rep = 0.01
    F_rep_x=0.0
    F_rep_y=0.0
#외란이 있을법한 확률 장애물이 있을법한
    pos_error_x = des_position[0]-local_position[0]
    pos_error_y = des_position[1]-local_position[1]
    pos_error_z = des_position[2]-local_position[2] 

    F=np.zeros((3,1))
    R=np.zeros((3,1))
    F[0]=K_att*(pos_error_x)/distance
    F[1]=K_att*(pos_error_y)/distance
    F[2]=K_att*(pos_error_z)

    ####추후 numpy이용해서 한번에 연산하기 연산속도++
    for n in range(0,120):
        #print(n,lidar_range[n])
        obs_pos_x = lidar_range[n]*np.math.sin((n+30)*np.math.pi/180.0)
        obs_pos_y = lidar_range[n]*np.math.cos((n+30)*np.math.pi/180.0)
        #print(n,lidar_range[n],obs_pos_x,obs_pos_y)
        obs_distance = lidar_range[n]
        if(obs_distance<3.0 and obs_distance>0.5):
            R[0]=R[0]+(K_rep*(1/obs_distance - 1/6.0))/np.math.pow(obs_distance,2)*(obs_pos_x/obs_distance)
            R[1]=R[1]+(K_rep*(1/obs_distance - 1/6.0))/np.math.pow(obs_distance,2)*(obs_pos_y/obs_distance)
        else:
            R[0]=R[0]
            R[1]=R[1]
    if (abs(pos_error_z)<0.2):
        cmd_vel.linear.x = F[0] - R[0]
        cmd_vel.linear.y = F[1] - R[1]
    if(abs(pos_error_x)<0.2 and abs(pos_error_y)<0.2):
        cmd_vel.linear.x = 0
        cmd_vel.linear.y = 0
    cmd_vel.linear.z=F[2]
    print(F[0],F[1],R[0],R[1])
    local_vel_pub.publish(cmd_vel)
    
#def func(each_frame):
    #plt.clf()
    #x,y = each_frame
    #plt.scatter(x,y)
    
def main():
#publish node init
    
    
    #10hz frequency
    rate = rospy.Rate(20) 
    count=0
    
    while not rospy.is_shutdown():
        #ani = FuncAnimation(fig,func,frames=frames)
        #plt.show()
        
        Velocity_Pub(Collision_Avoidance.desired_position,Collision_Avoidance.local_position,Collision_Avoidance.ranges)
        #print(Collision_Avoidance.ranges)
        #Collision_Avoidance.ranges=np.zeros((120))
        rate.sleep()
        count+=1

    rospy.spin()
if __name__=='__main__':
    try:
    #init
        rospy.init_node('test_off',anonymous=False)
    #publisher name
        local_vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
        lidar_sub = rospy.Subscriber('/os_cloud_node/points',PointCloud2,lidar_callback)
        pose_sub = rospy.Subscriber('mavros/local_position/pose',PoseStamped,local_pose_callback)
        state_sub = rospy.Subscriber('mavros/state',State,state_callback)
        main()
    except rospy.ROSInterruptException:
        pass