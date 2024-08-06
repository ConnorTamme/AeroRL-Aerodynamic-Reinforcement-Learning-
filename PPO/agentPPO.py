from gymEnv import DroneEnv
from stable_baselines3 import PPO
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

ROTARY_BUF_SIZE = 400
n_episodes=2000
n_steps = 1024
save_interval = 20


optunaTestNumb = 1
path="./runs/optunaTrial_"
while (os.path.isdir(f"{path}{optunaTestNumb}/")):
    optunaTestNumb += 1
path=f"{path}{optunaTestNumb}/"
writer = SummaryWriter(log_dir=path)

env = DroneEnv()

model=PPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=0.000170,
    n_steps=n_steps,
    tensorboard_log=path,
    batch_size=128,
    gamma=0.8,
)

env = model.get_env()
obs = env.reset()
rotary_reward_buffer = np.zeros(ROTARY_BUF_SIZE)

model.learn(total_timesteps=300000)

model.save("./models/test1")

#for ep in range(n_episodes):
#    score = 0
 #   if (ep % save_interval == 0):
  #      model.save(f"./models/EPISODE_{ep}")
   # done = False
    #print(f"\n\nEpisode : {ep}")
#    step = 1
 #   while not done and step <= n_steps:
  #      print(f"----------- Step : {step} -----------")
   #     action, _states = model.predict(obs)
    #    obs, reward, done, _ = env.step(action)
     #   score += reward
      #  rotary_reward_buffer[0] = reward
#        rotary_reward_buffer = np.roll(rotary_reward_buffer, 1)
 #       writer.add_scalar('score_history', score, ep)
  #      writer.add_scalar('reward_history', reward, ep)
   #     writer.add_scalar('Average Reward', np.average(rotary_reward_buffer), ep)
#        step += 1
 #   print(f"Final Average : {np.average(rotary_reward_buffer)}")
