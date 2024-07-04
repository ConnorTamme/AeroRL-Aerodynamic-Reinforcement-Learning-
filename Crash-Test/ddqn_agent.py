import math
import random
from collections import deque
import airsim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from setuptools import glob
from env import DroneEnv
from torch.utils.tensorboard import SummaryWriter
import time
from prioritized_memory import Memory

writer = SummaryWriter()

torch.manual_seed(0)#all 3 are to set up rng
random.seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Define the attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attention_scores = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(self.context_vector(attention_scores), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=6, hidden_dim=128):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 84, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(84, 42, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(42, 21, kernel_size=2, stride=2)
        self.fc4 = nn.Linear(21 * 4 * 4, 168)
        self.lstm = nn.LSTM(168, hidden_dim, batch_first=True)
        self.hidden_size = hidden_dim
        self.attention = Attention(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        batch_size, seq_len, c, h = x.size() 
       # x = x.view(batch_size * seq_len, c, h, w) 
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        # Skip connections (Couldn't get them to work due to differing dimensions)
      #  x3 += x2
      #  x2 += x1
        x3 = x3.view(batch_size, seq_len, -1)
        
        x3 = F.relu(self.fc4(x3))
        h_0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        c_0 = torch.zeros(1, batch_size, self.hidden_size).cuda()

        lstm_out, hidden_state = self.lstm(x3, (h_0, c_0))
        
        context_vector, attention_weights = self.attention(lstm_out)
        
        return self.fc5(context_vector)


class DDQN_Agent:
    def __init__(self, useDepth=False):
        self.useDepth = useDepth
        self.eps_start = 0.9#epsilon is the probability of taking a random action.
                             #epsilon decays overtime and bottoms out at 0.05 or 5%
        self.eps_end = 0.05
        self.eps_decay = 30000
        self.gamma = 0.8#gamma is how much future rewards matter. Low gamma is impulsive
                         #high gamma (closer to 1) will delay rewards now for bigger rewards later
        self.learning_rate = 0.001
        self.batch_size = 128
        self.memory = Memory(10000)
        self.max_episodes = 10000
        self.save_interval = 20
        self.test_interval = 100
        self.network_update_interval = 100
        self.episode = -1
        self.steps_done = 0
        self.max_steps = 400

        self.policy = DQN()#DQN is reinforcement NN
        self.target = DQN()
        self.test_network = DQN()
        self.target.eval()
        self.test_network.eval()
        self.updateNetworks()

        self.env = DroneEnv(useDepth)
        self.optimizer = optim.Adam(self.policy.parameters(), self.learning_rate)

        if torch.cuda.is_available():
            print('Using device:', device)
            print(torch.cuda.get_device_name(0))
        else:
            print("Using CPU")

        # LOGGING
        cwd = os.getcwd()
        self.save_dir = os.path.join(cwd, "saved models")
        if not os.path.exists(self.save_dir):
            os.mkdir("saved models")
        if not os.path.exists(os.path.join(cwd, "videos")):
            os.mkdir("videos")

        if torch.cuda.is_available():
            self.policy = self.policy.to(device)  # to use GPU
            self.target = self.target.to(device)  # to use GPU
            self.test_network = self.test_network.to(device)  # to use GPU

        # model backup
        files = glob.glob(self.save_dir + '\\*.pt')
        if len(files) > 0:
            files.sort(key=os.path.getmtime)
            file = files[-1]
            checkpoint = torch.load(file)
            self.policy.load_state_dict(checkpoint['state_dict'])
            self.episode = checkpoint['episode']
            self.steps_done = checkpoint['steps_done']
            self.updateNetworks()
            print("Saved parameters loaded"
                  "\nModel: ", file,
                  "\nSteps done: ", self.steps_done,
                  "\nEpisode: ", self.episode)


        else:
            if os.path.exists("log.txt"):
                open('log.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('last_episode.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('saved_model_params.txt', 'w').close()

        self.optimizer = optim.Adam(self.policy.parameters(), self.learning_rate)
        obs, _ = self.env.reset()
        tensor = self.transformToTensor(obs)
        writer.add_graph(self.policy, tensor)

    def updateNetworks(self):
        self.target.load_state_dict(self.policy.state_dict())

    def transformToTensor(self, img):
        tensor = torch.FloatTensor(img).to(device)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        return tensor

    def convert_size(self, size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    def act(self, state):
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if random.random() > self.eps_threshold:#randomly either do a random action or ask the model what to do
            # print("greedy")
            if torch.cuda.is_available():
                action = np.argmax(self.policy(state).cpu().data.squeeze().numpy())
            else:
                action = np.argmax(self.policy(state).data.squeeze().numpy())
        else:
            action = random.randrange(0, 6)
        return int(action)

    def append_sample(self, state, action, reward, next_state):
        next_state = self.transformToTensor(next_state)

        current_q = self.policy(state).squeeze().cpu().detach().numpy()[action]
        next_q = self.target(next_state).squeeze().cpu().detach().numpy()[action]
        expected_q = reward + (self.gamma * next_q)

        error = abs(current_q - expected_q),

        self.memory.add(error, state, action, reward, next_state)

    def learn(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        states, actions, rewards, next_states, idxs, is_weights = self.memory.sample(self.batch_size)

        states = tuple(states)
        next_states = tuple(next_states)

        states = torch.cat(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = torch.cat(next_states)

        current_q = self.policy(states)[[range(0, self.batch_size)], [actions]]
        next_q =self.target(next_states).cpu().detach().numpy()[[range(0, self.batch_size)], [actions]]
        expected_q = torch.FloatTensor(rewards + (self.gamma * next_q)).to(device)

        errors = torch.abs(current_q.squeeze() - expected_q.squeeze()).cpu().detach().numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
        self.test_network.load_state_dict(self.target.state_dict())
        self.memory.update(idx, errors[i])

        loss = F.smooth_l1_loss(current_q.squeeze(), expected_q.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        print("Starting...")
        #self.test()
        #print("\n\n\n\n\nDone test\n\n\n\n\n\n\n\n")
        score_history = []
        reward_history = []

        if self.episode == -1:
            self.episode = 1

        for e in range(1, self.max_episodes + 1):
            start = time.time()
            state, _ = self.env.reset()
            steps = 0
            score = 0
            while True:
                state = self.transformToTensor(state)

                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)

                if steps == self.max_steps:
                    done = 1

                #self.memorize(state, action, reward, next_state)
                self.append_sample(state, action, reward, next_state)
                self.learn()

                state = next_state
                steps += 1
                score += reward
                if done:
                    print("----------------------------------------------------------------------------------------")
                    if self.memory.tree.n_entries < self.batch_size:
                        print("Training will start after ", self.batch_size - self.memory.tree.n_entries, " steps.")
                        break

                    print(
                        "episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}".format(
                            self.episode, reward, round(score / steps, 2), score, self.eps_threshold, self.steps_done))
                    score_history.append(score)
                    reward_history.append(reward)
                    with open('log.txt', 'a') as file:
                        file.write(
                            "episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}\n".format(
                                self.episode, reward, round(score / steps, 2), score, self.eps_threshold,
                                self.steps_done))

                    if torch.cuda.is_available():
                        print('Total Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory))
                        print('Allocated Memory:', self.convert_size(torch.cuda.memory_allocated(0)))
                        print('Cached Memory:', self.convert_size(torch.cuda.memory_reserved(0)))
                        print('Free Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory - (
                                torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved())))

                        # tensorboard --logdir=runs
                        memory_usage_allocated = np.float64(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
                        memory_usage_cached = np.float64(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))

                        writer.add_scalar("memory_usage_allocated", memory_usage_allocated, self.episode)
                        writer.add_scalar("memory_usage_cached", memory_usage_cached, self.episode)

                    writer.add_scalar('epsilon_value', self.eps_threshold, self.episode)
                    writer.add_scalar('score_history', score, self.episode)
                    writer.add_scalar('reward_history', reward, self.episode)
                    writer.add_scalar('Total steps', self.steps_done, self.episode)
                    writer.add_scalars('General Look', {'score_history': score,
                                                        'reward_history': reward}, self.episode)

                    # save checkpoint
                    if self.episode % self.save_interval == 0:
                        checkpoint = {
                            'episode': self.episode,
                            'steps_done': self.steps_done,
                            'state_dict': self.policy.state_dict()
                        }
                        torch.save(checkpoint, self.save_dir + '/EPISODE{}.pt'.format(self.episode))

                    if self.episode % self.network_update_interval == 0:
                        self.updateNetworks()

                    self.episode += 1
                    end = time.time()
                    stopWatch = end - start
                    print("Episode is done, episode time: ", stopWatch)

                    if self.episode % self.test_interval == 0:
                        self.test()

                    break
        writer.close()

    def test(self):
        self.test_network.load_state_dict(self.target.state_dict())

        #self.test_network.load_state_dict(torch.load(self.save_dir + '/EPISODE9982.pt')['state_dict'])
        start = time.time()
        steps = 0
        score = 0
        image_array = []
        state, next_state_image = self.env.reset()
        image_array.append(next_state_image)

        while True:
            state = self.transformToTensor(state)

            action = int(np.argmax(self.test_network(state).cpu().data.squeeze().numpy()))
            next_state, reward, done, next_state_image = self.env.step(action)
            image_array.append(next_state_image)

            if steps == self.max_steps:
                done = 1

            state = next_state
            steps += 1
            score += reward

            if done:
                print("----------------------------------------------------------------------------------------")
                print("TEST, reward: {}, score: {}, total steps: {}".format(
                    reward, score, self.steps_done))

                with open('tests.txt', 'a') as file:
                    file.write("TEST, reward: {}, score: {}, total steps: {}\n".format(
                        reward, score, self.steps_done))

                writer.add_scalars('Test', {'score': score, 'reward': reward}, self.episode)

                end = time.time()
                stopWatch = end - start
                print("Test is done, test time: ", stopWatch)

                break
