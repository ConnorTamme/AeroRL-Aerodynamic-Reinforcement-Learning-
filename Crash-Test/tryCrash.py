import airsim
import random as r
client = airsim.MultirotorClient()
loops = 0
def interpret_action(action):
    """Interprete action"""
    scaling_factor = 3

    if action == 0:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 1:
        quad_offset = (-scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 4:
        quad_offset = (0,0,scaling_factor)
    elif action == 5:
        quad_offset = (0,0,-scaling_factor)
    return quad_offset
while True:
    client.reset()
    quad_offset = interpret_action(r.randrange(0,6))
    quad_vel = client.getImuData().angular_velocity
    client.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            3 #move in this way for MOVEMENT_INTERVAL seconds
        )
    client.getLidarData()
    client.getMultirotorState()
    client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True)])
    print(f"Finished Loop: {loops}")
    loops += 1
