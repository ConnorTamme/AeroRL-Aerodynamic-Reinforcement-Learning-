import airsim
import time
#import matlab as ml
import math
 
dest =[47.64148158030679, -122.14186713106668, 135]
def angle_to_dest(client):
    gps_data = client.getMultirotorState().gps_location
    quad_state = XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude)
    angle = math.degrees(math.atan(abs(dest[1] - quad_state.y_val)/ abs(dest[0] - quad_state.x_val)))
    if dest[0] >= quad_state.x_val:
        if dest[1] >= quad_state.y_val:
            return angle
        else:
            return 360 - angle
    else:
        if dest[1] >= quad_state.y_val:
            return 180 - angle
        else:
            return angle + 180

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in degrees (counterclockwise)
        pitch is rotation around y in degrees (counterclockwise)
        yaw is rotation around z in degrees (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [math.degrees(roll_x), math.degrees(pitch_y), math.degrees(yaw_z)] # in radians
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
            client.rotateByYawRateAsync(10,2)
        elif action == "a":
            client.rotateByYawRateAsync(-10,2)
loops = 0
client = airsim.MultirotorClient()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.hoverAsync().join()
while(True):
    time.sleep(0.1)
    if (loops % 30 == 0):
        gps_data = client.getMultirotorState().gps_location
        ori = client.getImuData().orientation
        
        ori = euler_from_quaternion(ori.x_val, ori.y_val, ori.z_val, ori.w_val)
        for i in range(0, len(ori)):
            if ori[i] < 0:
                ori[i] += 360
        print(ori)
        print(angle_to_dest(client))
        quad_state = XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude)
        print(quad_state.toString())
    loops += 1

    file = open("input.txt", "r")
    cmd = file.readline().strip('\n')
    interpret_action(cmd)
   
