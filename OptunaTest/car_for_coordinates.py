import airsim
import time

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

        if action == "d":
            quad_offset = (scaling_factor, 0, 0)
        elif action == "a":
            quad_offset = (-scaling_factor, 0, 0)
        elif action == "w":
            quad_offset = (0, scaling_factor, 0)
        elif action == "s":
            quad_offset = (0, -scaling_factor, 0)
        elif action == " ":
            quad_offset = (0, 0, scaling_factor)
        elif action == "c":
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



