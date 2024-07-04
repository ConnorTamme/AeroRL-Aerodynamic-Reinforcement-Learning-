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
DESTS = [[47.641467981817705,-122.14221072362143,135], [47.64228995061023,-122.1420177906823,135], [47.641467985202176,-122.13831772992684,135], [47.64258919126217,-122.13870886337459,135],[47.642480522863224, -122.14016500005329, 135]]

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

    def step(self, action):
        """Step"""
        #print("new step ------------------------------")

        self.quad_offset = self.interpret_action(action) #interpret_action gives 3 values back based on what the action was
        #print("quad_offset: ", self.quad_offset)
        
        quad_vel = self.client.getImuData().angular_velocity
        #quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(
            quad_vel.x_val + self.quad_offset[0],
            quad_vel.y_val + self.quad_offset[1],
            quad_vel.z_val + self.quad_offset[2],
            MOVEMENT_INTERVAL #move in this way for MOVEMENT_INTERVAL seconds
        )
        print(f"Time Stamp: {self.client.simGetCollisionInfo().time_stamp}")
        collision = (self.client.simGetCollisionInfo().has_collided and (time.time() - self.startTime) > 5.0)

        time.sleep(0.1)
        gps_data = self.client.getMultirotorState().gps_location
        #print(self.client.getMultirotorState().kinematics_estimated.position)
        quad_state = XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude)
        #print(quad_state.toString())
        quad_vel = self.client.getImuData().angular_velocity
        #quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

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
        self.last_dist = self.get_distance(XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude))
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        gps_data = self.client.getMultirotorState().gps_location
        quad_state = XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude)
        print(quad_state.toString())
        #quad_state = self.client.getMultirotorState().kinematics_estimated.position
        newDest = self.dest
        while newDest == self.dest:
            newDest = DESTS[r.randrange(0,len(DESTS))]
        self.dest = newDest
        self.client.moveByVelocityAsync(0, 0, -7, 2).join()
        self.pastDist = np.zeros(50)
        self.running_reward = 0
        obs, image = self.get_obs(quad_state)
        
        return obs, image

    def get_obs(self, quad_state):
        if self.useDepth:
            # get depth image
            responses = self.client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float64)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            image = np.reshape(img1d, (responses[0].height, responses[0].width))
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")
        else:
            # Get rgb image
            responses = self.client.simGetImages(
                [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
            )
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, 3)
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")

        obs = np.array(image_array)
        cur_loc = np.array(quad_state.toList())
        cur_loc = cur_loc.view().reshape(3,-1)
        cur_loc = np.pad(cur_loc, [(0, np.size(obs,0) - np.size(cur_loc, 0)),(0,0)])
        cur_dest = np.array(self.dest)
        cur_dest = cur_dest.view().reshape(3,-1)
        cur_dest = np.pad(cur_dest, [(0,np.size(obs,0) - np.size(cur_dest,0)), (0,0)])
        obs = np.concatenate((obs, cur_loc, cur_dest), axis=1)
        if self.useLidar:
            obs = np.concatenate((obs, self.get_lidar(np.size(obs,0))), axis=1)
        return obs, image

    def get_lidar(self, num_rows):
        lidarData = self.client.getLidarData()
        points = np.zeros((num_rows,3*15)) #3*15 gives room for num_rows * 15 lidar data points
        bundledLidar = []
        for i in range(0,len(lidarData.point_cloud),3):
            bundledLidar.append([lidarData.point_cloud[i],lidarData.point_cloud[i+1],lidarData.point_cloud[i+2]])
        self.lidarPos = np.array([lidarData.pose.position.x_val, lidarData.pose.position.y_val, lidarData.pose.position.z_val])
        heap = Heap(bundledLidar, self.compare_lidar_points)
        
        done = 0
        for i in range(0,num_rows):
            for j in range(0,45,3):
                if ((i*15)+j >= len(bundledLidar)):
                    done = 1
                    break
                pt = heap.Pop()
                points[i][j] = pt[0]
                points[i][j+1] = pt[1]
                points[i][j+2] = pt[2]

            if done == 1:
                break
        return points
    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        print(f"Quad State in get_distance {quad_state.x_val} {quad_state.y_val} {quad_state.z_val}")
        pts = np.array(self.dest)
        pts[0] *= 30000
        pts[1] *= 30000
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        print(f"Going to [{self.dest}] currently at [{quad_pt}]")
        quad_pt[0] *= 30000
        quad_pt[1] *= 30000
        dist = np.linalg.norm(quad_pt - pts)
        print("Distance is " + str(dist))
        return dist

    def compute_reward(self, quad_state, quad_vel, collision, obstacles, goal, power_usage):
        """Compute reward"""
        done = 0
        reward = 0

        if collision:
            reward -= 100  # Penalty for collision
            done = 1
        else:
            # Calculate 3D Euclidean distance to the goal
            dist = self.get_distance(quad_state)
            #dist = np.linalg.norm(np.array(quad_state.toList()) - np.array(goal))
            diff = self.last_dist - dist

            # Dynamic scaling of rewards and penalties based on proximity to goal
            scale_factor = 10 if dist > 5 else 50 if dist > 1 else 100
        
        
            if self.last_dist != 0:
                reward += diff * scale_factor  # Reward for getting closer

        
            if diff < 0:
                reward += diff * scale_factor / 2  # Penalize for increasing distance

            # Reward for being very close to the destination
            if dist < 5:
                reward += 500  

            # Reward for reaching the goal
            if dist < 1:
                reward += 1000  
                newDest = self.dest
                while newDest == self.dest:
                    newDest = DESTS[r.randrange(0,len(DESTS))]
                self.dest = newDest

            # Penalize proximity to obstacles
            #min_dist_to_obstacles = min(np.linalg.norm(np.array(quad_state) - np.array(obs)) for obs in obstacles)
            #if min_dist_to_obstacles < 2:  
            #    reward -= (2 - min_dist_to_obstacles) * 50  
            vel_array = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
             # Penalizing high speeds and accelerations
            speed = np.linalg.norm(vel_array)
            reward -= speed * 0.1  
        
            # Penalize for sudden changes in speed
            acceleration = np.linalg.norm(self.last_vel - vel_array)
            reward -= acceleration * 0.05  
    
            # Penalize frequent changes in speed or direction
            if np.linalg.norm(self.last_vel - vel_array) > 1:
                reward -= 10 
            self.last_dist = dist
            self.last_vel = vel_array
        if (((self.last_pos - np.array(quad_state.toList())) == np.zeros(3)).all()):
            done = 1
            reward -= 100
        self.running_reward += reward
        if self.running_reward < -200:
            done = 1
        self.last_pos = np.array(quad_state.toList())
        print(f"Reward: {reward}")
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
        
