#! /usr/bin/env python
"""Environment for Microsoft AirSim Unity Quadrotor using AirSim python API

- Author: Subin Yang
- Contact: subinlab.yang@gmail.com
- Date: 2019.06.20.
"""
import csv
import math
import pprint
import time
import random as r
import torch
from Heap import Heap
from PIL import Image
import time
import numpy as np

import airsim
#import setup_path

MOVEMENT_INTERVAL = 3
DESTS = [[47.64146798797463, -122.14188548770431, 135],[47.64257772583905, -122.14188416031806, 135],[47.64263376582024, -122.14016499591769, 135],[47.64258777435291, -122.13852585284403, 135],[47.64145763763342, -122.13847971184445, 135],[47.640335494691755, -122.14176225282225, 135],[47.64034533624314, -122.13848987494362, 135],[47.64030617319528, -122.14013820145897, 135]]

class XYZ_data():
    def __init__(self, x_val, y_val, z_val):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
    def toString(self):
        return f"X_val: {self.x_val}, Y_val: {self.y_val}, Z_val: {self.z_val}"
    def toList(self):
        return [self.x_val, self.y_val, self.z_val]

class DroneEnv(object):
    """Drone environment class using AirSim python API"""

    def __init__(self, useDepth=False, useLidar=False):
        self.startTime = time.time()
        self.client = airsim.MultirotorClient() #gets the airsim client
        self.dest = DESTS[r.randrange(0, len(DESTS))]
        gps_data = self.client.getMultirotorState().gps_location
        self.last_dist = self.get_distance(XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude))
    #The multirotor state looks like this:
    #can_arm = False
    #collision = <CollisionInfo> {   }
    #gps_location = <GeoPoint> {   }
    #kinematics_estimated = <KinematicsState> {   }
    #landed_state = 0
    #rc_data = <RCData> {   'is_initialized': False,     'is_valid': False,     'pitch': 0.0,     'roll': 0.0,     'switch1': 0,     'switch2': 0,     'switch3': 0,     'switch4': 0,     'switch5': 0,     'switch6': 0,     'switch7': 0,     'switch8': 0,     'throttle': 0.0,     'timestamp': 0,     'yaw': 0.0}
    #ready = False
    #ready_message = ''
    #timestamp = 0
        self.running_reward = 0
        self.last_vel = np.zeros(3)
        self.quad_offset = (0, 0, 0)
        self.useDepth = useDepth
        self.pastDist = np.zeros(50)
        self.last_pos = np.zeros(3)
        self.useLidar = useLidar
        
        self.image_height = 84
        self.image_width = 84
        self.image_channels = 3
        self.image_size = self.image_height * self.image_width * self.image_channels
        self.checkpoints = 0
    def step(self, action):
        """Step"""
        print("\n\nnew step ------------------------------")

        self.quad_offset = self.interpret_action(action) #interpret_action gives 3 values back based on what the action was
        
        quad_vel = self.client.getImuData().angular_velocity
        self.client.moveByVelocityAsync(
            quad_vel.x_val + self.quad_offset[0],
            quad_vel.y_val + self.quad_offset[1],
            quad_vel.z_val + self.quad_offset[2],
            MOVEMENT_INTERVAL #move in this way for MOVEMENT_INTERVAL seconds
        )
        collision = (self.client.simGetCollisionInfo().has_collided and (time.time() - self.startTime) > 5.0)

        time.sleep(0.1)
        gps_data = self.client.getMultirotorState().gps_location
        quad_state = XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude)
        quad_vel = self.client.getImuData().angular_velocity

        if quad_state.z_val < - 7.3:
            self.client.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 1).join()
            #if drone is too low after step it climbs
        
        result, done = self.compute_reward(quad_state, quad_vel, collision, 0,self.dest, 0)
        state, image = self.get_obs(quad_state)

        return state, result, done, image

    def reset(self):
        self.client.reset() #moves vehicle to default position
        self.startTime = time.time()
        gps_data = self.client.getMultirotorState().gps_location
        self.last_dist = 0
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        gps_data = self.client.getMultirotorState().gps_location
        quad_state = XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude)
        print(quad_state.toString())
        newDest = self.dest
        while newDest == self.dest:
            newDest = DESTS[r.randrange(0,len(DESTS))]
        self.dest = newDest
        self.client.moveByVelocityAsync(0, 0, -7, 2).join()
        self.pastDist = np.zeros(50)
        self.running_reward = 0
        self.checkpoints = 0
        obs, image = self.get_obs(quad_state)
        
        return obs, image

    def get_obs(self, quad_state):
        size = 0
        while size != self.image_size: # Sometimes simGetImages() return an unexpected resonpse. 
                                       # If so, try it again.
            response = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])[0]
            obs = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            size = obs.size



        obs = obs.reshape(self.image_height, self.image_width, self.image_channels)
        image = obs.reshape(self.image_height, self.image_width, self.image_channels)
        coordinate_part = np.zeros((1,84,3))
        cur_loc = quad_state.toList()
        coordinate_part[0][0][0]=cur_loc[0]
        coordinate_part[0][0][1]=cur_loc[1]
        coordinate_part[0][0][2]=cur_loc[2]
        cur_dest = self.dest
        coordinate_part[0][1][0]=cur_loc[0]
        coordinate_part[0][1][1]=cur_loc[1]
        coordinate_part[0][1][2]=cur_loc[2]

        obs = np.concatenate((obs, coordinate_part), axis=0)
        if self.useLidar:
            obs = np.concatenate((obs, self.get_lidar(np.size(obs,1))), axis=0)
        return obs, image

    def get_lidar(self, num_rows):
        lidarData = self.client.getLidarData()
        points = np.zeros((5,num_rows,3)) #3*15 gives room for num_rows * 15 lidar data points
        bundledLidar = []
        for i in range(0,len(lidarData.point_cloud),3):
            bundledLidar.append([lidarData.point_cloud[i],lidarData.point_cloud[i+1],lidarData.point_cloud[i+2]])
        self.lidarPos = np.array([lidarData.pose.position.x_val, lidarData.pose.position.y_val, lidarData.pose.position.z_val])
        heap = Heap(bundledLidar, self.compare_lidar_points)
        
        done = 0
        for i in range(0,5):
            for j in range(0,num_rows):
                if ((i*84)+j >= len(bundledLidar)):
                    done = 1
                    break
                pt = heap.Pop()
                points[i][j][0] = pt[0]
                points[i][j][1] = pt[1]
                points[i][j][2] = pt[2]

            if done == 1:
                break
        return points
    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        pts = np.array(self.dest)
        pts[0] *= 30000
        pts[1] *= 30000
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        print(f"Going to [{self.dest}]\nCurrently at [{quad_pt}]")
        quad_pt[0] *= 30000
        quad_pt[1] *= 30000
        dist = np.linalg.norm(quad_pt - pts)
        print("Distance is " + str(dist))
        return dist

    def compute_reward(self, quad_state, quad_vel, collision, obstacles, goal, power_usage):
        """Compute reward"""
        done = 0
        reward = -1

        if collision:
            reward -= 1000  # Penalty for collision
            done = 1
        else:
            # Calculate 3D Euclidean distance to the goal
            dist = self.get_distance(quad_state)
            
            # Reward for reaching the goal
            if dist < 1:
                reward += 1000  
                self.checkpoints += 1
                newDest = self.dest
                while newDest == self.dest:
                    newDest = DESTS[r.randrange(0,len(DESTS))]
                self.dest = newDest
                self.last_dist = 0
            else:
                self.last_dist = dist
            
        if (((self.last_pos - np.array(quad_state.toList())) == np.zeros(3)).all()):
            done = 1
            reward -= 1000
        self.running_reward += reward
        if self.running_reward < -1000:
            done = 1
        if done == 1:
            reward = 100 * self.checkpoints
        self.last_pos = np.array(quad_state.toList())
        print(f"Reward: {reward}")
        print(f"Reward so Far: {self.running_reward}")
        return reward, done

    def interpret_action(self, action):
        """Interprete action"""
        scaling_factor = 3

        if action == 0:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action == 1:
            self.quad_offset = (-scaling_factor, 0, 0)
        elif action == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            self.quad_offset = (0, -scaling_factor, 0)
        elif action == 4:
            self.quad_offset = (0,0,scaling_factor)
        elif action == 5:
            self.quad_offset = (0,0,-scaling_factor)
        return self.quad_offset
    def compare_lidar_points(self, p1, p2):
        distP1 = np.linalg.norm(self.lidarPos - np.array(p1))
        distP2 = np.linalg.norm(self.lidarPos - np.array(p2))
        return distP2 < distP1 
        
