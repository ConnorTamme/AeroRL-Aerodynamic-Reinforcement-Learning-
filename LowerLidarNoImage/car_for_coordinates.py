import airsim
import time
import numpy as np
from Heap import Heap

def get_lidar(client):
    lidarData = client.getLidarData()
    points = np.zeros((10, 3)) #3*15 gives room for num_rows * 15 lidar data points
    bundledLidar = []
    for i in range(0,len(lidarData.point_cloud),3):
        bundledLidar.append([lidarData.point_cloud[i],lidarData.point_cloud[i+1],lidarData.point_cloud[i+2]])
    lidarPos = np.array([lidarData.pose.position.x_val, lidarData.pose.position.y_val, lidarData.pose.position.z_val])
    heap = Heap(bundledLidar, compare_lidar_points)
    done = 0
    for i in range(0,10):
        if ((i*3) >= len(bundledLidar)):
            done = 1
            break
        pt = heap.Pop()
        points[i][0] = pt[0]
        points[i][1] = pt[1]
        points[i][2] = pt[2]

        if done == 1:
            break
    return points
def compare_lidar_points(p1, p2):
    distP1 = np.linalg.norm(np.zeros(3) - np.array(p1))
    distP2 = np.linalg.norm(np.zeros(3) - np.array(p2))
    return distP2 < distP1 
class XYZ_data():
    def __init__(self, x_val, y_val, z_val):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
    def toString(self):
        return f"X_val: {self.x_val}, Y_val: {self.y_val}, Z_val: {self.z_val}"
def interpret_action(action):
        """Interprete action"""
        scaling_factor = 3

        if action == "w":
            quad_offset = (scaling_factor, 0, 0)
        elif action == "s":
            quad_offset = (-scaling_factor, 0, 0)
        elif action == "d":
            quad_offset = (0, scaling_factor, 0)
        elif action == "a":
            quad_offset = (0, -scaling_factor, 0)
        elif action == "c":
            quad_offset = (0, 0, scaling_factor)
        elif action == " ":
            quad_offset = (0,0, -scaling_factor)
        else:
            quad_offset = (0,0,0)
        return quad_offset
loops = 0
client = airsim.MultirotorClient()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
while(True):
    time.sleep(0.1)
    if (loops % 50 == 0):
        gps_data = client.getMultirotorState().gps_location
        hp = get_lidar(client)
        print(f"Shortest Lidar Dist : {np.linalg.norm(np.zeros(3) - hp[0])}")
        quad_state = XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude)
        print(quad_state.toString())
    loops += 1

    file = open("input.txt", "r")
    cmd = file.readline().strip('\n')
    quad_offset = interpret_action(cmd)
    quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    client.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            1
        )



