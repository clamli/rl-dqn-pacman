import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import config
from gameAPI.game import GamePacmanAgent
from collections import deque

# torch.manual_seed(9001)
# random.seed(9001)


FloatTensor = torch.cuda.FloatTensor if config.use_cuda else torch.FloatTensor


def convert_idx_to_2_dim_tensor(action):
    # idx = action.index(max(action))
    idx = action
    if idx == 0:
        return [-1, 0]
    elif idx == 1:
        return [1, 0]
    elif idx == 2:
        return [0, -1]
    elif idx == 3:
        return [0, 1]
    else:
        raise RuntimeError('something wrong in DQNAgent.formatAction')


def convert_2_dim_tensor_to_4_dim_tensor(action):
    if action == [-1, 0]:
        return [1, 0, 0, 0]
    elif action == [1, 0]:
        return [0, 1, 0, 0]
    elif action == [0, -1]:
        return [0, 0, 1, 0]
    elif action == [0, 1]:
        return [0, 0, 0, 1]
    else:
        raise RuntimeError('something wrong in DQNAgent.formatAction')


def frames_to_tensor(frames):
    images_input = []
    for frame in frames:
        image = np.concatenate([frame], -1)
        image_input = image.astype(np.float32) / 255.
        image_input.resize((1, *image_input.shape))
        images_input.append(image_input)

    return torch.from_numpy(np.concatenate(images_input, 0)).permute(0, 3, 1, 2).type(FloatTensor)


def single_frame_to_tensor(frame):
    image = np.concatenate([frame], -1)
    image_input = image.astype(np.float32) / 255.
    image_input.resize((1, *image_input.shape))
    image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(
        FloatTensor)

    return image_input_torch


class ReplayMemory:

    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, is_gameover, next_state):
        if len(self.memory) >= config.max_memory_size:
            self.memory.pop(0)
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

    def train2(self):
        action_pred = None
        image = None
        image_prev = None
        frames = []
        game_memories = deque()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        mse_loss = nn.MSELoss(reduction='elementwise_mean')
        num_iter = 0
        while True:
            if len(game_memories) > config.max_memory_size:
                game_memories.popleft()
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=action_pred)

            if is_gameover:
                self.game_agent.reset()
            frames.append(frame)
            if len(frames) == 1:
                image_prev = image
                image = np.concatenate(frames, -1)
                frames.pop(0)
                if image_prev is not None:
                    game_memories.append((image, image_prev, reward, convert_2_dim_tensor_to_4_dim_tensor(action),
                                      is_gameover))

            # explore
            if len(game_memories) < config.start_training_threshold:
                print('[STATE]: explore, [MEMORYLEN]: %d' % len(game_memories))

            # train
            else:
                # --get data
                num_iter += 1
                images_input = []
                images_prev_input = []
                reward_lst = []
                action_lst = []
                is_gameover_lst = []
                for each in random.sample(game_memories, config.sample_size):
                    image_input = each[0].astype(np.float32) / 255.
                    image_input.resize((1, *image_input.shape))
                    images_input.append(image_input)
                    image_prev_input = each[1].astype(np.float32) / 255.
                    image_prev_input.resize((1, *image_prev_input.shape))
                    images_prev_input.append(image_prev_input)
                    reward_lst.append(each[2])
                    action_lst.append(each[3])
                    is_gameover_lst.append(each[4])

                images_input_torch = torch.from_numpy(np.concatenate(images_input, 0)).permute(0, 3, 1, 2).type(
                    FloatTensor)
                images_prev_input_torch = torch.from_numpy(np.concatenate(images_prev_input, 0)).permute(0, 3, 1,
                                                                                                         2).type(
                    FloatTensor)

                # --compute loss
                optimizer.zero_grad()
                q_t = self.net(images_input_torch)
                q_t = torch.max(q_t, dim=1)[0]
                loss = mse_loss(
                    torch.Tensor(reward_lst).type(FloatTensor) + (1 - torch.Tensor(is_gameover_lst).type(FloatTensor)) * (
                                0.95 * q_t),
                    (self.net(images_prev_input_torch) * torch.Tensor(action_lst).type(FloatTensor)).sum(1))
                loss.backward()
                optimizer.step()
                # --make decision
                prob = max(config.eps_start - (
                            config.eps_start - config.eps_end) / config.eps_num_steps * num_iter,
                           config.eps_end)
                rand_value = random.random()
                if rand_value > prob:
                    with torch.no_grad():
                        image_input = image.astype(np.float32) / 255.
                        image_input.resize((1, *image_input.shape))
                        image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(FloatTensor)
                        action_pred = self.net(image_input_torch)
                        max_value, action = torch.max(action_pred, dim=1)
                        action_pred = convert_idx_to_2_dim_tensor(action[0])
                        #actions_values = self.net(single_frame_to_tensor(frame))
                        #max_value, action = torch.max(actions_values, dim=1)
                        #action_pred = convert_idx_to_2_dim_tensor(action[0])
                else:
                    action_pred = None

                print('[STATE]: training, [ITER]: %d, [LOSS]: %.3f, [ACTION]: %s' % (num_iter, loss.item(), str(action_pred)))
                if num_iter % config.save_model_threshold == 0:
                    torch.save(self.net.state_dict(), str(num_iter) + '.pkl')



    def train(self):
        replay_memory = ReplayMemory()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        mse_loss = nn.MSELoss(reduction='elementwise_mean')
        count = 0
        for episode in range(1, config.M+1):
            # init
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=None)
            while is_gameover:
                self.game_agent.reset()
                frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=None)

            while True:
                state_lst = []
                action_lst = []
                reward_lst = []
                is_gameover_lst = []
                next_state_lst = []
                is_train = len(replay_memory.memory) >= config.start_training_threshold
                if not is_train:
                    action = None
                else:
                    count += 1
                    (state_lst, action_lst, reward_lst, is_gameover_lst, next_state_lst) = replay_memory.sample(
                        config.sample_size)
                    prob = max(config.eps_start - (config.eps_start - config.eps_end) / config.eps_num_steps * count, config.eps_end)
                    random_value = random.random()
                    if random_value <= prob:
                        action = None
                    else:
                        with torch.no_grad():
                            actions_values = self.net(single_frame_to_tensor(frame))
                            max_value, action = torch.max(actions_values, dim=1)
                            action = convert_idx_to_2_dim_tensor(action[0])

                next_frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=action)
                if is_gameover:
                    print("reset")
                    self.game_agent.reset()

                replay_memory.add(frame, convert_2_dim_tensor_to_4_dim_tensor(action), reward, is_gameover, next_frame)

                if is_train:
                    # sampling
                    optimizer.zero_grad()
                    q_t = self.net(frames_to_tensor(next_state_lst))
                    q_t = torch.max(q_t, dim=1)[0]
                    loss = mse_loss(
                        FloatTensor(reward_lst) + (1 - FloatTensor(is_gameover_lst)) * (config.gamma * q_t),
                        (self.net(frames_to_tensor(state_lst)) * FloatTensor(action_lst)).sum(1)
                    )
                    loss.backward()
                    print("episode: %d, iteration: %d, loss: %.4f, action: %s" % (episode, count, loss, str(action)))
                    optimizer.step()
                    if count % config.save_model_threshold == 0:
                        torch.save(self.net.state_dict(), './models/model' + str(count) + '.pkl')

                frame = next_frame


    def test(self):
        action = None
        while True:
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=action)
            if is_gameover:
                self.game_agent.reset()
            if random.random() <= config.eps_end:
                action = None
            else:
                with torch.no_grad():
                    actions_values = self.net(single_frame_to_tensor(frame))
                    print(actions_values)
                    max_value, action = torch.max(actions_values, dim=1)
                    print(action)
                    action = convert_idx_to_2_dim_tensor(action)
            print('[ACTION]: %s' % str(action))
