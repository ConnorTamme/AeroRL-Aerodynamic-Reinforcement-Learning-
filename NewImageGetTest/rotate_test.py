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
    if (loops % 50 == 0):
        gps_data = client.getMultirotorState().gps_location
        quad_state = XYZ_data(gps_data.latitude, gps_data.longitude, gps_data.altitude)
        print(quad_state.toString())
    loops += 1

    file = open("input.txt", "r")
    cmd = file.readline().strip('\n')
    interpret_action(cmd)
   
