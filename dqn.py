import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import config
from gameAPI.game import GamePacmanAgent


FloatTensor = torch.cuda.FloatTensor if config.use_cuda else torch.FloatTensor


def convert_idx_to_2_dim_tensor(action):
    if action == 0:
        return [-1, 0]
    elif action == 1:
        return [1, 0]
    elif action == 2:
        return [0, -1]
    elif action == 3:
        return [0, 1]


def convert_2_dim_tensor_to_4_dim_tensor(action):
    if action == [-1, 0]:
        return [1, 0, 0, 0]
    elif action == [1, 0]:
        return [0, 1, 0, 0]
    elif action == [0, -1]:
        return [0, 0, 1, 0]
    elif action == [0, 1]:
        return [0, 0, 0, 1]


def frames_to_tensor(frames):
    images_input = []
    for frame in frames:
        image_input = frame.astype(np.float32) / 255.
        image_input.resize((1, *image_input.shape))
        images_input.append(image_input)

    return torch.from_numpy(np.concatenate(images_input, 0)).permute(0, 3, 1, 2).type(FloatTensor)


class ReplayMemory:

    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, is_gameover, next_state):
        self.memory.append((state, action, reward, is_gameover, next_state))

    def sample(self, batch_size):
        rand_ind = random.sample(range(0, len(self.memory)), min(batch_size, len(self.memory)))
        state_lst = [self.memory[i][0] for i in rand_ind]
        action_lst = [self.memory[i][1] for i in rand_ind]
        reward_lst = [self.memory[i][2] for i in rand_ind]
        is_gameover_lst = [self.memory[i][3] for i in rand_ind]
        next_state_lst = [self.memory[i][4] for i in rand_ind]
        return state_lst, action_lst, reward_lst, is_gameover_lst, next_state_lst


class DQN(torch.nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        #self.model = torchvision.models.alexnet(pretrained=True)
        #self.model.classifier[6] = nn.Linear(4096, num_actions)

        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=4)

    def forward(self, x):
        #x = self.model(x)
        x = self.resnet18(x)
        return x


class Agent:

    def __init__(self, net: torch.nn.Module):
        self.game_agent = GamePacmanAgent(config)
        self.net = net
        if config.use_cuda:
            self.net = self.net.cuda()

    def train(self):
        replay_memory = ReplayMemory()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        mse_loss = nn.MSELoss(reduction='elementwise_mean')
        for episode in range(1, config.M+1):
            is_train = len(replay_memory.memory) >= config.start_training_threshold
            # init
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=None)
            if episode % config.save_model_threshold == 0:
                torch.save(self.net.state_dict(), './models/model' + str(episode) + '.pkl')
            while is_gameover:
                self.game_agent.reset()
                frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=None)

            while not is_gameover:
                if not is_train or random.random() <= config.epsilon:
                    action = None
                else:
                    temp_frames = [frame]
                    actions_values = self.net(frames_to_tensor(temp_frames))
                    max_value, action = torch.max(actions_values, 1)
                    action = convert_idx_to_2_dim_tensor(action[0])

                next_frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=action)
                if is_gameover:
                    self.game_agent.reset()

                replay_memory.add(frame, convert_2_dim_tensor_to_4_dim_tensor(action), reward, is_gameover, next_frame)

                if is_train:
                    # sampling
                    (state_lst, action_lst, reward_lst, is_gameover_lst, next_state_lst) = replay_memory.sample(config.sample_size)
                    q_t = self.net(frames_to_tensor(next_state_lst))
                    q_t = torch.max(q_t, dim=1)[0]
                    loss = mse_loss(
                        FloatTensor(reward_lst) + (1 - FloatTensor(is_gameover_lst)) * (config.gamma * q_t),
                        (self.net(frames_to_tensor(state_lst)) * FloatTensor(action_lst)).sum(1)
                    )
                    loss.backward()
                    print("episode: %d,  loss: %.4f, " % (episode, loss))
                    optimizer.step()

                frame = next_frame
